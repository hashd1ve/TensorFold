//! Edits of a marked prompt pass's streams, and its logit lens (TF_DS_XEDIT=MiB on every rank; off when unset: no
//! buffer, no launch, no frame, the served pass as it is). With xtaps.zig's taps these are the write side and the
//! per-layer read-out of the same marked pass.
//!
//! A request that names an edits file ("edits": "name", DIR/name.xedit in TF_DS_XTAPS_DIR) has its prompt pass change
//! the streams as they enter the listed layers (layer id `layers`: the streams after the last layer, before the head),
//! at the listed prompt positions, before that layer's tap:
//!   add     h_s += scale * v                      every stream s, one vector for every position (steering)
//!   patch   h_s += scale * (t_p - mean_s h_s)     position p's row t_p: at scale 1 the streams' mean becomes t_p, which
//!                                                 is what a tap of another pass holds (activation patching)
//!   ablate  h_s -= scale * (h_s.v / v.v) v        each stream's own component along v (directional ablation)
//!   set     h_s += scale * (t_p - h_s)            position p's row t_p: at scale 1 the stream becomes t_p (a stream of
//!                                                 another pass, or zeros)
//! each on the streams its mask names (bit s: stream s; 0: every stream; patch's mean is over all of them).
//! The streams are the residual a layer reads; a pending MoE post is applied to them first, as a DSpark tap does. Both
//! ranks hold the same streams, so both apply the same edits: rank 0 sends the plan and its rows to the followers
//! (lanes.zig's fill_mark and fill_xdata frames) before the pass's first chunk.
//!
//! A request with "lens": true (and "taps") also gets, for each tapped layer, the head's logits of the prompt's last
//! position had the model ended there: collapse_norm and the head on the streams entering that layer, with that
//! layer's own pre. It is a side computation (a pending post goes to a scratch copy; the head's buffers are free until
//! the pass's own head), run on both ranks because the head's columns are split between them.
//!
//! Such a pass runs every layer on every row from the prompt's first token, neither resumes nor keeps a prompt state,
//! and nothing cuts into it. Only the prompt pass is edited: the tokens generated after it are decoded as ever, on
//! the edited prompt's caches.
//!
//! The file (little endian): "XEDT", u32 version, u32 hidden, u32 n; then n edits, each u32 op (1 add, 2 patch,
//! 3 ablate, 4 set), u32 layer, i64 start, i64 end, f32 scale, u32 rows, u32 mask, u32 0, rows * hidden f32. Positions are [start, end) of the
//! prompt; a negative one counts from its end, and end = maxInt(i64) is its end. rows is 1 for add and ablate, and
//! end - start for patch and set. Every value is finite.
const std = @import("std");

pub const version: u32 = 1;
pub const max_edits = 32;
pub const max_lens = 48;
/// The data bytes a fill_xdata frame carries at most (a follower's frame buffer holds 64 KiB past a pool's ids).
pub const piece = 32 << 10;
/// "XEDT", version, hidden, n.
pub const header_bytes = 16;
/// op, layer, start, end, scale, rows, mask, a zero word.
pub const edit_bytes = 40;
pub const to_end = std.math.maxInt(i64);

pub const Op = enum(u32) {
    add = 1,
    patch = 2,
    ablate = 3,
    set = 4,

    pub fn of(v: u32) ?Op {
        return switch (v) {
            1 => .add,
            2 => .patch,
            3 => .ablate,
            4 => .set,
            else => null,
        };
    }

    /// Whether the op takes a row a position (else one row for every position).
    pub fn perRow(op: Op) bool {
        return op == .patch or op == .set;
    }
};

/// An edit with its positions resolved against the prompt: [start, end), its data rows from `row` of the plan's.
pub const Edit = struct { op: Op, layer: u32, start: u64, end: u64, scale: f32, row: u64, rows: u32, mask: u32 = 0 };

pub const Plan = struct {
    edits: [max_edits]Edit = undefined,
    n: usize = 0,
    rows: u64 = 0, // the edits' data rows together (fp32 [rows, hidden])
    lens: [max_lens]u16 = undefined, // the layers whose entering streams the head reads, ascending
    n_lens: usize = 0,

    pub fn lensIndex(p: *const Plan, layer: usize) ?usize {
        for (p.lens[0..p.n_lens], 0..) |l, j| if (l == layer) return j;
        return null;
    }

    pub fn empty(p: *const Plan) bool {
        return p.n == 0 and p.n_lens == 0;
    }

    /// Every field within a model of `layers` layers and `hc` streams, a prompt of `len` tokens and a data buffer of
    /// `cap_rows` rows: what a rank checks before it runs a plan, whoever made it.
    pub fn check(p: *const Plan, layers: usize, hc: usize, len: usize, cap_rows: usize) !void {
        if (p.n > max_edits or p.n_lens > max_lens) return error.BadXEdit;
        var at: u64 = 0;
        for (p.edits[0..p.n]) |ed| {
            if (ed.layer > layers or ed.start >= ed.end or ed.end > len) return error.BadXEdit;
            if (!std.math.isFinite(ed.scale)) return error.BadXEdit;
            const want: u64 = if (ed.op.perRow()) ed.end - ed.start else 1;
            if (ed.rows != want or ed.row != at) return error.BadXEdit;
            if (hc >= 32 or ed.mask >> @intCast(hc) != 0) return error.BadXEdit;
            at += ed.rows;
        }
        if (at != p.rows) return error.BadXEdit;
        if (p.rows > cap_rows) return error.XEditTooLarge;
        for (p.lens[0..p.n_lens], 0..) |l, j| if (l > layers or (j > 0 and p.lens[j - 1] >= l)) return error.BadXEdit;
    }

    /// The plan into a frame (`w`: lanes.Wire's `int`).
    pub fn put(p: *const Plan, gpa: std.mem.Allocator, w: anytype) !void {
        try w.int(gpa, u32, @intCast(p.n_lens));
        for (p.lens[0..p.n_lens]) |l| try w.int(gpa, u16, l);
        try w.int(gpa, u32, @intCast(p.n));
        for (p.edits[0..p.n]) |ed| {
            try w.int(gpa, u32, @intFromEnum(ed.op));
            try w.int(gpa, u32, ed.layer);
            try w.int(gpa, u64, ed.start);
            try w.int(gpa, u64, ed.end);
            try w.int(gpa, u32, @bitCast(ed.scale));
            try w.int(gpa, u64, ed.row);
            try w.int(gpa, u32, ed.rows);
            try w.int(gpa, u32, ed.mask);
        }
        try w.int(gpa, u64, p.rows);
    }

    /// A plan out of a frame (`r`: lanes.Reader's `int`); `check` it before it runs.
    pub fn get(r: anytype) !Plan {
        var p: Plan = .{};
        const nl = try r.int(u32);
        if (nl > max_lens) return error.BadFrame;
        p.n_lens = nl;
        for (p.lens[0..nl]) |*l| l.* = try r.int(u16);
        const n = try r.int(u32);
        if (n > max_edits) return error.BadFrame;
        p.n = n;
        for (p.edits[0..n]) |*ed| {
            const op = Op.of(try r.int(u32)) orelse return error.BadFrame;
            ed.* = .{ .op = op, .layer = try r.int(u32), .start = try r.int(u64), .end = try r.int(u64), .scale = @bitCast(try r.int(u32)), .row = try r.int(u64), .rows = try r.int(u32), .mask = try r.int(u32) };
        }
        p.rows = try r.int(u64);
        return p;
    }
};

fn rd(comptime T: type, b: []const u8, at: usize) T {
    return std.mem.readInt(T, b[at..][0..@sizeOf(T)], .little);
}

/// A position of the file against a prompt of `len` tokens: a negative one counts from the end.
fn resolve(v: i64, len: usize) !u64 {
    const n: i64 = @intCast(len);
    const at = if (v < 0) v + n else v;
    if (at < 0 or at > n) return error.BadXEditPosition;
    return @intCast(at);
}

/// An edits file's edits into `p` (its lens list left as it is), each one's positions resolved against a prompt of
/// `len` tokens; at[i]: where edit i's data starts in `bytes`. The file must be this model's (`hidden`) and whole,
/// every value finite.
pub fn parse(bytes: []const u8, hidden: usize, len: usize, p: *Plan, at: *[max_edits]usize) !void {
    if (bytes.len < header_bytes or !std.mem.eql(u8, bytes[0..4], "XEDT")) return error.NotAnXEditFile;
    if (rd(u32, bytes, 4) != version) return error.BadXEditVersion;
    if (rd(u32, bytes, 8) != hidden) return error.BadXEditHidden;
    const n = rd(u32, bytes, 12);
    if (n == 0 or n > max_edits) return error.BadXEdit;
    var o: usize = header_bytes;
    var row: u64 = 0;
    for (0..n) |i| {
        if (bytes.len - o < edit_bytes) return error.ShortXEditFile;
        const op = Op.of(rd(u32, bytes, o)) orelse return error.BadXEdit;
        const layer = rd(u32, bytes, o + 4);
        const start = try resolve(rd(i64, bytes, o + 8), len);
        const end_raw = rd(i64, bytes, o + 16);
        const end = if (end_raw == to_end) @as(u64, len) else try resolve(end_raw, len);
        const scale: f32 = @bitCast(rd(u32, bytes, o + 24));
        const rows = rd(u32, bytes, o + 28);
        const mask = rd(u32, bytes, o + 32);
        o += edit_bytes;
        if (start >= end) return error.BadXEditPosition;
        if (!std.math.isFinite(scale)) return error.BadXEditValue;
        if (rows != (if (op.perRow()) end - start else 1)) return error.BadXEditRows;
        const data = @as(usize, rows) * hidden * 4;
        if (bytes.len - o < data) return error.ShortXEditFile;
        var k: usize = 0;
        while (k < data) : (k += 4) {
            if (rd(u32, bytes, o + k) & 0x7f80_0000 == 0x7f80_0000) return error.BadXEditValue; // an infinity or a NaN
        }
        at[i] = o;
        o += data;
        p.edits[i] = .{ .op = op, .layer = layer, .start = start, .end = end, .scale = scale, .row = row, .rows = rows, .mask = mask };
        row += rows;
    }
    if (o != bytes.len) return error.BadXEdit;
    p.n = n;
    p.rows = row;
}

/// The size of the largest file a buffer of `cap_rows` rows can take.
pub fn fileLimit(hidden: usize, cap_rows: usize) usize {
    return header_bytes + max_edits * edit_bytes + cap_rows * hidden * 4;
}

// -- tests ---------------------------------------------------------------------------------------------------------

const TestWire = struct {
    buf: std.ArrayList(u8) = .empty,
    fn int(w: *TestWire, gpa: std.mem.Allocator, comptime T: type, v: T) !void {
        var b: [@sizeOf(T)]u8 = undefined;
        std.mem.writeInt(T, &b, v, .little);
        try w.buf.appendSlice(gpa, &b);
    }
};

const TestReader = struct {
    b: []const u8,
    at: usize = 0,
    fn int(r: *TestReader, comptime T: type) !T {
        const n = @sizeOf(T);
        if (r.at + n > r.b.len) return error.ShortFrame;
        const v = std.mem.readInt(T, r.b[r.at..][0..n], .little);
        r.at += n;
        return v;
    }
};

fn testFile(gpa: std.mem.Allocator, hidden: u32, edits: []const struct { op: u32, layer: u32, start: i64, end: i64, scale: f32, rows: u32, fill: f32, mask: u32 = 0 }) ![]u8 {
    var w: TestWire = .{};
    errdefer w.buf.deinit(gpa);
    try w.buf.appendSlice(gpa, "XEDT");
    try w.int(gpa, u32, version);
    try w.int(gpa, u32, hidden);
    try w.int(gpa, u32, @intCast(edits.len));
    for (edits) |e| {
        try w.int(gpa, u32, e.op);
        try w.int(gpa, u32, e.layer);
        try w.int(gpa, i64, e.start);
        try w.int(gpa, i64, e.end);
        try w.int(gpa, u32, @bitCast(e.scale));
        try w.int(gpa, u32, e.rows);
        try w.int(gpa, u32, e.mask);
        try w.int(gpa, u32, 0);
        for (0..e.rows * hidden) |_| try w.int(gpa, u32, @bitCast(e.fill));
    }
    return w.buf.toOwnedSlice(gpa);
}

test "a file's edits resolve against the prompt, and their rows follow one another" {
    const gpa = std.testing.allocator;
    const f = try testFile(gpa, 4, &.{
        .{ .op = 1, .layer = 20, .start = 0, .end = to_end, .scale = 4, .rows = 1, .fill = 0.5 },
        .{ .op = 2, .layer = 12, .start = -3, .end = -1, .scale = 1, .rows = 2, .fill = 1 },
        .{ .op = 3, .layer = 40, .start = 9, .end = 10, .scale = 1, .rows = 1, .fill = -2, .mask = 5 },
    });
    defer gpa.free(f);
    var p: Plan = .{};
    var at: [max_edits]usize = undefined;
    try parse(f, 4, 10, &p, &at);
    try std.testing.expectEqual(@as(usize, 3), p.n);
    try std.testing.expectEqual(@as(u64, 4), p.rows);
    try std.testing.expectEqual(Edit{ .op = .add, .layer = 20, .start = 0, .end = 10, .scale = 4, .row = 0, .rows = 1 }, p.edits[0]);
    try std.testing.expectEqual(Edit{ .op = .patch, .layer = 12, .start = 7, .end = 9, .scale = 1, .row = 1, .rows = 2 }, p.edits[1]);
    try std.testing.expectEqual(Edit{ .op = .ablate, .layer = 40, .start = 9, .end = 10, .scale = 1, .row = 3, .rows = 1, .mask = 5 }, p.edits[2]);
    try std.testing.expectEqual(@as(usize, header_bytes + edit_bytes), at[0]);
    try std.testing.expectEqual(@as(usize, header_bytes + 2 * edit_bytes + 16), at[1]);
    try std.testing.expectEqual(@as(usize, header_bytes + 3 * edit_bytes + 48), at[2]);
    try p.check(40, 4, 10, 4);
    try std.testing.expectError(error.XEditTooLarge, p.check(40, 4, 10, 3));
    try std.testing.expectError(error.BadXEdit, p.check(39, 4, 10, 4)); // layer 40 of a 39-layer model
    try std.testing.expectError(error.BadXEdit, p.check(40, 4, 9, 4)); // a position past a shorter prompt
    try std.testing.expectError(error.BadXEdit, p.check(40, 2, 10, 4)); // stream 2 of a two-stream model
}

test "a file that is not this model's, not whole, or not finite is refused" {
    const gpa = std.testing.allocator;
    var p: Plan = .{};
    var at: [max_edits]usize = undefined;
    const ok = try testFile(gpa, 4, &.{.{ .op = 1, .layer = 0, .start = 0, .end = 1, .scale = 1, .rows = 1, .fill = 1 }});
    defer gpa.free(ok);
    try parse(ok, 4, 1, &p, &at);
    try std.testing.expectError(error.BadXEditHidden, parse(ok, 8, 1, &p, &at));
    try std.testing.expectError(error.ShortXEditFile, parse(ok[0 .. ok.len - 1], 4, 1, &p, &at));
    const other = try gpa.dupe(u8, ok);
    defer gpa.free(other);
    other[3] = 'P';
    try std.testing.expectError(error.NotAnXEditFile, parse(other, 4, 1, &p, &at));
    try std.testing.expectError(error.NotAnXEditFile, parse(ok[0..8], 4, 1, &p, &at));
    try std.testing.expectError(error.BadXEditPosition, parse(ok, 4, 0, &p, &at)); // an empty range of an empty prompt
    const long = try gpa.alloc(u8, ok.len + 4);
    defer gpa.free(long);
    @memcpy(long[0..ok.len], ok);
    @memset(long[ok.len..], 0);
    try std.testing.expectError(error.BadXEdit, parse(long, 4, 1, &p, &at)); // bytes after the last edit
    inline for (.{
        .{ @as(u32, 1), @as(i64, 0), @as(i64, 1), @as(f32, 1), @as(u32, 1), std.math.nan(f32), error.BadXEditValue },
        .{ @as(u32, 1), @as(i64, 0), @as(i64, 1), @as(f32, 1), @as(u32, 1), std.math.inf(f32), error.BadXEditValue },
        .{ @as(u32, 1), @as(i64, 0), @as(i64, 1), std.math.inf(f32), @as(u32, 1), @as(f32, 1), error.BadXEditValue },
        .{ @as(u32, 5), @as(i64, 0), @as(i64, 1), @as(f32, 1), @as(u32, 1), @as(f32, 1), error.BadXEdit }, // no such op
        .{ @as(u32, 1), @as(i64, 0), @as(i64, 2), @as(f32, 1), @as(u32, 2), @as(f32, 1), error.BadXEditRows }, // add takes one row
        .{ @as(u32, 2), @as(i64, 0), @as(i64, 2), @as(f32, 1), @as(u32, 1), @as(f32, 1), error.BadXEditRows }, // patch a row a position
        .{ @as(u32, 1), @as(i64, 2), @as(i64, 1), @as(f32, 1), @as(u32, 1), @as(f32, 1), error.BadXEditPosition },
        .{ @as(u32, 1), @as(i64, 0), @as(i64, 4), @as(f32, 1), @as(u32, 1), @as(f32, 1), error.BadXEditPosition }, // past the prompt
        .{ @as(u32, 1), @as(i64, -4), @as(i64, 1), @as(f32, 1), @as(u32, 1), @as(f32, 1), error.BadXEditPosition }, // before it
    }) |c| {
        const f = try testFile(gpa, 4, &.{.{ .op = c[0], .layer = 0, .start = c[1], .end = c[2], .scale = c[3], .rows = c[4], .fill = c[5] }});
        defer gpa.free(f);
        try std.testing.expectError(c[6], parse(f, 4, 3, &p, &at));
    }
    const none = try testFile(gpa, 4, &.{});
    defer gpa.free(none);
    try std.testing.expectError(error.BadXEdit, parse(none, 4, 3, &p, &at));
}

test "a plan crosses the ranks' link as it is, and a follower checks what it is sent" {
    const gpa = std.testing.allocator;
    var p: Plan = .{};
    p.n = 2;
    p.edits[0] = .{ .op = .ablate, .layer = 3, .start = 1, .end = 5, .scale = -0.75, .row = 0, .rows = 1, .mask = 9 };
    p.edits[1] = .{ .op = .set, .layer = 9, .start = 2, .end = 4, .scale = 1, .row = 1, .rows = 2 };
    p.rows = 3;
    p.n_lens = 3;
    p.lens[0..3].* = .{ 0, 8, 40 };
    var w: TestWire = .{};
    defer w.buf.deinit(gpa);
    try p.put(gpa, &w);
    var r: TestReader = .{ .b = w.buf.items };
    const q = try Plan.get(&r);
    try std.testing.expectEqual(w.buf.items.len, r.at);
    try std.testing.expectEqual(p.n, q.n);
    try std.testing.expectEqual(p.rows, q.rows);
    try std.testing.expectEqualSlices(Edit, p.edits[0..p.n], q.edits[0..q.n]);
    try std.testing.expectEqualSlices(u16, p.lens[0..p.n_lens], q.lens[0..q.n_lens]);
    try q.check(40, 4, 5, 3);
    try std.testing.expectEqual(@as(?usize, 1), q.lensIndex(8));
    try std.testing.expectEqual(@as(?usize, null), q.lensIndex(9));
    var short: TestReader = .{ .b = w.buf.items[0 .. w.buf.items.len - 1] };
    try std.testing.expectError(error.ShortFrame, Plan.get(&short));
    // what a frame could carry and a rank must not run
    var bad = q;
    bad.edits[1].row = 0; // rows that do not follow one another
    try std.testing.expectError(error.BadXEdit, bad.check(40, 4, 5, 3));
    bad = q;
    bad.rows = 9;
    try std.testing.expectError(error.BadXEdit, bad.check(40, 4, 5, 9));
    bad = q;
    bad.edits[0].scale = std.math.nan(f32);
    try std.testing.expectError(error.BadXEdit, bad.check(40, 4, 5, 3));
    bad = q;
    bad.lens[1] = 0; // not ascending
    try std.testing.expectError(error.BadXEdit, bad.check(40, 4, 5, 3));
    bad = q;
    bad.edits[0].mask = 16; // a fifth stream
    try std.testing.expectError(error.BadXEdit, bad.check(40, 4, 5, 3));
    bad = q;
    bad.edits[1].end = 6; // past the prompt, and a row count that no longer matches
    try std.testing.expectError(error.BadXEdit, bad.check(40, 4, 5, 3));
}
