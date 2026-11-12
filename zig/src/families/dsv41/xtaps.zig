//! Extra taps of a marked prompt pass, for looking inside the model (TF_DS_XTAPS, TF_DS_XTAPS_DIR; off when unset: no
//! buffer, no launch, the served pass as it is). A request that names a file ("taps": "name") has its prompt pass
//! write, for each listed layer, the mean of the streams entering it (h.to(fp32).mean(1).to(bf16), what a DSpark tap
//! reads; layer id `layers`: the streams after the last layer) for every row the pass runs through that layer, to
//! DIR/name.xtaps. The taps read the streams without changing them: a pending MoE post goes to a scratch copy, so a
//! marked pass launches the same kernels on the same buffers as an unmarked one, plus these. Rank 0 alone taps.
//!
//! The file (little endian): "XTAP", u32 version, u32 hidden, u32 n, n u32 layer ids; then records, each a u32 kind:
//! 1: u64 len, len u32 (the prompt's ids, an image position the image token's);
//! 2: u32 layer, u64 start, u32 rows, rows * hidden bf16 (the streams' mean entering the layer);
//! 3: u32 vocab, vocab f32: the prompt's logits (its last position's; the first token is drawn from them);
//! 4: u32 layer, u32 mode, u32 vocab, vocab f32: a lens row (xedit.zig; "lens": 2), the head on the last position's
//!    streams entering the layer; mode 0 with the layer's own pre, mode 1 + s on stream s alone;
//! 5: u32 layer, u32 mode, f32 logsumexp, u32 k, k (u32 id, f32 logit): the same row's k largest ("lens": 1);
//! 6: u32 layer, u32 sub (0 attention, 1 FFN), u64 start, u32 rows, u32 hc, then fp32 pre [rows, hc] (the weights the
//!    sublayer reads the streams with), post [rows, hc] and comb [rows, hc, hc] (how its output goes back into them):
//!    every layer of a marked pass;
//! 7: u32 layer, u64 start, u32 rows, u32 hc, rows * hc * hidden bf16: the streams themselves entering a listed layer,
//!    the prompt's last "streams": K positions.
//! It is written as name.xtaps.part and renamed when the pass ends with its first token drawn.
const std = @import("std");
const cuda = @import("cuda");

extern "c" fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*anyopaque;
extern "c" fn fwrite(ptr: [*]const u8, size: usize, n: usize, f: *anyopaque) usize;
extern "c" fn fclose(f: *anyopaque) c_int;
extern "c" fn rename(old: [*:0]const u8, new: [*:0]const u8) c_int;
extern "c" fn unlink(path: [*:0]const u8) c_int;

pub const max_layers = 48;
pub const version: u32 = 2;
/// The largest logits kept of a lens row ("lens": 1).
pub const lens_top = 256;
/// The last positions whose streams a request may ask for ("streams": K).
pub const max_stream_rows = 64;
/// A request's name for its file: 1 to 64 of [A-Za-z0-9_.-], not starting with a dot.
pub const max_name = 64;
/// A name that ends so keeps the served bounded replay (the decoder layers on the prompt's last window alone).
pub const replay_suffix = ".replay";

/// A zero-terminated path in `buf`.
fn pathZ(buf: []u8, comptime fmt: []const u8, args: anytype) ![:0]const u8 {
    const s = try std.fmt.bufPrint(buf, fmt ++ "\x00", args);
    return s[0 .. s.len - 1 :0];
}

pub const Mark = struct { start: usize = 0, n: usize = 0 };

/// "0,8,20,40" -> the layer ids, ascending and distinct, each at most `layers`.
pub fn parseLayers(text: []const u8, layers: usize, out: *[max_layers]u16) !usize {
    var n: usize = 0;
    var it = std.mem.tokenizeScalar(u8, text, ',');
    while (it.next()) |t| {
        const v = std.fmt.parseInt(u16, std.mem.trim(u8, t, " "), 10) catch return error.BadXTaps;
        if (v > layers or n == max_layers or (n > 0 and out[n - 1] >= v)) return error.BadXTaps;
        out[n] = v;
        n += 1;
    }
    if (n == 0) return error.BadXTaps;
    return n;
}

pub fn validName(name: []const u8) bool {
    if (name.len == 0 or name.len > max_name or name[0] == '.') return false;
    for (name) |ch| if (!(std.ascii.isAlphanumeric(ch) or ch == '_' or ch == '.' or ch == '-')) return false;
    return true;
}

/// Whether a marked request's pass runs every layer on every row (the default) or keeps the served replay.
pub fn fullPass(name: []const u8) bool {
    return !std.mem.endsWith(u8, name, replay_suffix);
}

pub const XTaps = struct {
    layers: [max_layers]u16 = undefined,
    n: usize = 0,
    dir: []const u8,
    d: usize, // hidden
    cap: usize, // a marked pass's chunk rows at most (TF_DS_XTAPS_ROWS; a chunk's when unset)
    hc: usize = 0,
    dev: cuda.DeviceBuffer, // bf16 [n, cap, d]: each listed layer's rows of the chunk
    scratch: cuda.DeviceBuffer, // bf16 [a chunk's rows, hc, d]: the streams with a pending MoE post applied
    mix: cuda.DeviceBuffer = undefined, // fp32 [max_layers, 2, mixRow() * cap]: every layer's two sublayers' pre, post and comb
    host: []u8, // a layer's rows on their way to the file
    shost: []u8 = &.{}, // bf16 [max_stream_rows, hc, d]: a layer's streams on their way to the file
    mhost: []u8 = &.{}, // a sublayer's mixes on their way
    at: [max_layers]Mark = @splat(.{}), // the rows each listed layer's tap holds of the chunk being run
    mix_at: [max_layers][2]Mark = @splat(@splat(.{})), // the rows each layer's sublayers' mixes hold of it

    /// `cap`: the rows of a marked pass's chunk the buffers hold; `chunk`: a chunk's rows (the scratch's).
    pub fn init(a: std.mem.Allocator, drv: *const cuda.Driver, text: []const u8, dir: []const u8, layers: usize, hidden: usize, hc: usize, cap: usize, chunk: usize) !XTaps {
        if (cap == 0 or cap > chunk or layers > max_layers) return error.BadXTaps;
        var x: XTaps = .{ .dir = try a.dupe(u8, dir), .d = hidden, .cap = cap, .hc = hc, .dev = undefined, .scratch = undefined, .host = undefined };
        x.n = try parseLayers(text, layers, &x.layers);
        x.host = try a.alloc(u8, cap * hidden * 2);
        x.shost = try a.alloc(u8, max_stream_rows * hc * hidden * 2);
        x.mhost = try a.alloc(u8, cap * x.mixRow());
        x.dev = try cuda.DeviceBuffer.alloc(drv, x.n * cap * hidden * 2);
        errdefer x.dev.free();
        x.scratch = try cuda.DeviceBuffer.alloc(drv, chunk * hc * hidden * 2);
        errdefer x.scratch.free();
        x.mix = try cuda.DeviceBuffer.alloc(drv, max_layers * 2 * cap * x.mixRow());
        return x;
    }

    pub fn deinit(x: *XTaps) void {
        x.dev.free();
        x.scratch.free();
        x.mix.free();
    }

    /// A row's mixes in bytes: pre [hc], post [hc], comb [hc, hc] fp32.
    pub fn mixRow(x: *const XTaps) usize {
        return x.hc * (2 + x.hc) * 4;
    }

    /// Layer `layer`'s sublayer `sub`'s mixes on the device: pre [cap, hc], then post [cap, hc], then comb [cap, hc, hc].
    pub fn mixes(x: *const XTaps, layer: usize, sub: usize) u64 {
        return x.mix.ptr + (layer * 2 + sub) * x.cap * x.mixRow();
    }

    pub fn index(x: *const XTaps, layer: usize) ?usize {
        for (x.layers[0..x.n], 0..) |l, j| if (l == layer) return j;
        return null;
    }

    /// Tap j's rows on the device (bf16 [cap, d]).
    pub fn rows(x: *const XTaps, j: usize) u64 {
        return x.dev.ptr + j * x.cap * x.d * 2;
    }

    pub fn bytes(x: *const XTaps) usize {
        return x.dev.len + x.scratch.len + x.mix.len;
    }
};

/// A marked pass's file while the pass runs (its pieces may be rounds apart; another pass may run between them).
pub const Sink = struct {
    f: *anyopaque,
    path: [600]u8 = undefined, // DIR/name.xtaps, zero terminated
    len: usize = 0,
    failed: bool = false, // a short write: the file is dropped at the end

    pub fn open(gpa: std.mem.Allocator, x: *const XTaps, name: []const u8, ids: []const u32) !*Sink {
        if (!validName(name)) return error.BadXTapsName;
        const s = try gpa.create(Sink);
        errdefer gpa.destroy(s);
        s.* = .{ .f = undefined };
        const final = pathZ(&s.path, "{s}/{s}.xtaps", .{ x.dir, name }) catch return error.BadXTapsName;
        s.len = final.len;
        var part: [640]u8 = undefined;
        const pz = pathZ(&part, "{s}.part", .{final}) catch return error.BadXTapsName;
        s.f = fopen(pz.ptr, "wb") orelse return error.XTapsNotWritable;
        s.put("XTAP");
        s.int(u32, version);
        s.int(u32, @intCast(x.d));
        s.int(u32, @intCast(x.n));
        for (x.layers[0..x.n]) |l| s.int(u32, l);
        s.int(u32, 1);
        s.int(u64, ids.len);
        s.put(std.mem.sliceAsBytes(ids));
        return s;
    }

    fn put(s: *Sink, b: []const u8) void {
        if (b.len == 0 or s.failed) return;
        if (fwrite(b.ptr, 1, b.len, s.f) != b.len) s.failed = true;
    }

    fn int(s: *Sink, comptime T: type, v: T) void {
        var b: [@sizeOf(T)]u8 = undefined;
        std.mem.writeInt(T, &b, v, .little);
        s.put(&b);
    }

    /// A layer's rows of positions start .. start + n - 1 (bf16 [n, hidden]).
    pub fn rows(s: *Sink, layer: u16, start: usize, n: usize, b: []const u8) void {
        s.int(u32, 2);
        s.int(u32, layer);
        s.int(u64, start);
        s.int(u32, @intCast(n));
        s.put(b);
    }

    /// The prompt's logits (the vocabulary of its last position).
    pub fn logits(s: *Sink, row: []const f32) void {
        s.int(u32, 3);
        s.int(u32, @intCast(row.len));
        s.put(std.mem.sliceAsBytes(row));
    }

    /// A lens row whole (the vocabulary).
    pub fn lensFull(s: *Sink, layer: u32, mode: u32, row: []const f32) void {
        s.int(u32, 4);
        s.int(u32, layer);
        s.int(u32, mode);
        s.int(u32, @intCast(row.len));
        s.put(std.mem.sliceAsBytes(row));
    }

    /// A lens row's lens_top largest logits (largest first; of equal ones the lower id) and its logsumexp.
    pub fn lensTop(s: *Sink, layer: u32, mode: u32, row: []const f32) void {
        var ids: [lens_top]u32 = undefined;
        const k = topK(row, &ids);
        s.int(u32, 5);
        s.int(u32, layer);
        s.int(u32, mode);
        s.int(u32, @bitCast(logSumExp(row)));
        s.int(u32, @intCast(k));
        for (ids[0..k]) |id| {
            s.int(u32, id);
            s.int(u32, @bitCast(row[id]));
        }
    }

    /// A sublayer's mixes of positions start .. start + n - 1: `b` holds pre, post and comb one after another.
    pub fn mix(s: *Sink, layer: u32, sub: u32, start: usize, n: usize, hc: usize, b: []const u8) void {
        s.int(u32, 6);
        s.int(u32, layer);
        s.int(u32, sub);
        s.int(u64, start);
        s.int(u32, @intCast(n));
        s.int(u32, @intCast(hc));
        s.put(b);
    }

    /// The streams of positions start .. start + n - 1 entering a layer (bf16 [n, hc, hidden]).
    pub fn streams(s: *Sink, layer: u32, start: usize, n: usize, hc: usize, b: []const u8) void {
        s.int(u32, 7);
        s.int(u32, layer);
        s.int(u64, start);
        s.int(u32, @intCast(n));
        s.int(u32, @intCast(hc));
        s.put(b);
    }

    /// The pass ended: with `ok` (its first token drawn) the file takes its name, else it goes.
    pub fn close(s: *Sink, gpa: std.mem.Allocator, ok: bool) void {
        const closed = fclose(s.f) == 0;
        var part: [640]u8 = undefined;
        const final: [*:0]const u8 = @ptrCast(&s.path);
        if (pathZ(&part, "{s}.part", .{s.path[0..s.len]})) |pz| {
            if (ok and closed and !s.failed) {
                if (rename(pz.ptr, final) != 0) std.log.err("dsv41: the taps file {s} could not take its name", .{s.path[0..s.len]});
            } else {
                _ = unlink(pz.ptr);
                std.log.warn("dsv41: the taps file {s} was dropped (the pass did not end, or a write failed)", .{s.path[0..s.len]});
            }
        } else |_| {}
        gpa.destroy(s);
    }
};

/// The ids of row's min(out.len, row.len) largest values, largest first, of equal ones the lower id; a NaN ranks
/// below every number.
pub fn topK(row: []const f32, out: []u32) usize {
    const k = @min(out.len, row.len);
    if (k == 0) return 0;
    const less = struct {
        // whether id a ranks below id b
        fn f(r: []const f32, a: u32, b: u32) bool {
            const x = r[a];
            const y = r[b];
            if (x != x) return y == y or a > b;
            if (y != y) return false;
            return x < y or (x == y and a > b);
        }
    }.f;
    for (0..k) |i| out[i] = @intCast(i);
    var low: usize = 0; // the kept id that ranks lowest
    for (1..k) |i| if (less(row, out[i], out[low])) {
        low = i;
    };
    for (k..row.len) |i| {
        const id: u32 = @intCast(i);
        if (!less(row, out[low], id)) continue;
        out[low] = id;
        low = 0;
        for (1..k) |j| if (less(row, out[j], out[low])) {
            low = j;
        };
    }
    const Ctx = struct { r: []const f32 };
    std.mem.sort(u32, out[0..k], Ctx{ .r = row }, struct {
        fn f(c: Ctx, a: u32, b: u32) bool {
            return less(c.r, b, a);
        }
    }.f);
    return k;
}

/// ln of the sum of exp over the row (fp64 sums about its largest value), as fp32.
pub fn logSumExp(row: []const f32) f32 {
    var top: f32 = -std.math.inf(f32);
    for (row) |v| if (v > top) {
        top = v;
    };
    if (!std.math.isFinite(top)) return top;
    var sum: f64 = 0;
    for (row) |v| sum += @exp(@as(f64, v) - @as(f64, top));
    return @floatCast(@as(f64, top) + @log(sum));
}

test "the largest logits come out in order, ties to the lower id, and their logsumexp" {
    const row = [_]f32{ 1, 5, -2, 5, 0.5, std.math.nan(f32), 3 };
    var ids: [3]u32 = undefined;
    try std.testing.expectEqual(@as(usize, 3), topK(&row, &ids));
    try std.testing.expectEqualSlices(u32, &.{ 1, 3, 6 }, &ids);
    var all: [16]u32 = undefined;
    try std.testing.expectEqual(@as(usize, 7), topK(&row, &all));
    try std.testing.expectEqualSlices(u32, &.{ 1, 3, 6, 0, 4, 2, 5 }, all[0..7]);
    const two = [_]f32{ 0, 0 };
    try std.testing.expectApproxEqAbs(@as(f32, @log(2.0)), logSumExp(&two), 1e-6);
    const big = [_]f32{ 1000, 1000, -1000 };
    try std.testing.expectApproxEqAbs(@as(f32, 1000 + @log(2.0)), logSumExp(&big), 1e-3);
}

test "the layer list is ascending, distinct and within the model" {
    var out: [max_layers]u16 = undefined;
    try std.testing.expectEqual(@as(usize, 4), try parseLayers("0, 8,20,40", 40, &out));
    try std.testing.expectEqualSlices(u16, &.{ 0, 8, 20, 40 }, out[0..4]);
    try std.testing.expectError(error.BadXTaps, parseLayers("", 40, &out));
    try std.testing.expectError(error.BadXTaps, parseLayers("8,8", 40, &out));
    try std.testing.expectError(error.BadXTaps, parseLayers("9,8", 40, &out));
    try std.testing.expectError(error.BadXTaps, parseLayers("41", 40, &out));
    try std.testing.expectError(error.BadXTaps, parseLayers("x", 40, &out));
}

test "a name stays inside the directory, and its suffix picks the pass" {
    try std.testing.expect(validName("news-001_a.b"));
    try std.testing.expect(!validName(""));
    try std.testing.expect(!validName("../x"));
    try std.testing.expect(!validName(".hidden"));
    try std.testing.expect(!validName("a/b"));
    const long: [max_name + 1]u8 = @splat('a');
    try std.testing.expect(validName(long[0..max_name]));
    try std.testing.expect(!validName(&long));
    try std.testing.expect(fullPass("news-001"));
    try std.testing.expect(!fullPass("news-001.replay"));
}

test "a sink writes the header, the ids and the rows, and takes its name only when the pass ended" {
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var dir_buf: [512]u8 = undefined;
    const dir = try std.fmt.bufPrint(&dir_buf, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    var x: XTaps = .{ .dir = dir, .d = 2, .cap = 4, .dev = undefined, .scratch = undefined, .host = undefined };
    x.n = try parseLayers("0,3", 3, &x.layers);
    const ids = [_]u32{ 5, 0xffffffff, 7 };
    const row = [_]u8{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var s = try Sink.open(gpa, &x, "t1", &ids);
    s.rows(3, 1, 2, &row);
    s.close(gpa, true);
    var path: [600]u8 = undefined;
    const got = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, try std.fmt.bufPrint(&path, "{s}/t1.xtaps", .{dir}), gpa, .limited(1 << 20));
    defer gpa.free(got);
    const want = "XTAP" ++ [_]u8{ 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0 } ++
        [_]u8{ 1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0xff, 0xff, 0xff, 0xff, 7, 0, 0, 0 } ++
        [_]u8{ 2, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0 } ++ row;
    try std.testing.expectEqualSlices(u8, want, got);
    // a pass that did not end leaves no file under either name
    s = try Sink.open(gpa, &x, "t2", &ids);
    s.close(gpa, false);
    try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().access(std.testing.io, try std.fmt.bufPrint(&path, "{s}/t2.xtaps", .{dir}), .{}));
    try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().access(std.testing.io, try std.fmt.bufPrint(&path, "{s}/t2.xtaps.part", .{dir}), .{}));
    try std.testing.expectError(error.BadXTapsName, Sink.open(gpa, &x, "../t3", &ids));
}
