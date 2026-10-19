//! Tool calls while they are written: XML and DeepSeek-V4.1 DSML calls as argument deltas, and the one-call policy's text filter.
const std = @import("std");
const json = @import("json.zig");
const ids = @import("ids.zig");
const reply_text = @import("reply_text.zig");
const tool_params = @import("tool_params.zig");
const tool_specs = @import("tool_specs.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

const space = reply_text.isSpaceByte;

const dsml41_open = tool_parse.dsml41_open;
const dsml41_close = tool_parse.dsml41_close;
const invoke41_open = tool_parse.invoke41_open;
const invoke41_close = tool_parse.invoke41_close;
const param41_open = tool_parse.param41_open;
const param41_close = tool_parse.param41_close;
const string_true = " string=\"true\"";
const string_false = " string=\"false\"";

/// ``\s*<TAG=([^>\n]+)>\n?`` at the start of ``rest``: the name and the bytes matched.
fn opening(rest: []const u8, tag: []const u8) ?struct { name: []const u8, len: usize, newline: bool } {
    var i: usize = 0;
    while (i < rest.len and space(rest[i])) i += 1;
    if (!std.mem.startsWith(u8, rest[i..], tag)) return null;
    i += tag.len;
    const start = i;
    while (i < rest.len and rest[i] != '>' and rest[i] != '\n') i += 1;
    if (i == start or i >= rest.len or rest[i] != '>') return null;
    const name = rest[start..i];
    i += 1;
    const newline = i < rest.len and rest[i] == '\n';
    return .{ .name = name, .len = i + @intFromBool(newline), .newline = newline };
}

/// ``\s*TEXT`` at the start of ``rest``: bytes matched.
fn closing(rest: []const u8, text: []const u8) ?usize {
    var i: usize = 0;
    while (i < rest.len and space(rest[i])) i += 1;
    return if (std.mem.startsWith(u8, rest[i..], text)) i + text.len else null;
}

/// Streams Qwen's XML calls as JSON argument deltas matching the final parser.
pub const Streamer = struct {
    a: Allocator,
    schemas: tool_params.Schemas,
    known: std.StringHashMapUnmanaged([]const u8) = .empty,
    current: []const u8 = "",
    value_schema: Value = .null,
    typed: bool = false,
    buffer: std.ArrayList(u8) = .empty,
    pos: usize = 0,
    state: enum { outside, call, function, value, closing, abort } = .outside,
    index: i64 = -1,
    args_open: bool = false,
    lead: bool = false,
    held: []const u8 = "",
    streamed: bool = false,
    searched: usize = 0, // the answer is only ever extended, so prose already searched holds no opener
    dsml: Dsml = .{},

    /// Where ``feedDsml`` is in the answer, apart from the XML calls' state.
    const Dsml = struct {
        pos: usize = 0,
        searched: usize = 0,
        state: enum { outside, block, invoke, string_value, json_value, skip } = .outside,
        args_open: bool = false,
    };

    pub fn init(a: Allocator, tools: []const Value) Allocator.Error!Streamer {
        var s: Streamer = .{ .a = a, .schemas = try tool_params.schemas(a, tools) };
        for (tools) |tool| {
            if (tool != .object) continue;
            const spec = if (tool.get("function")) |f| (if (f == .object) f else tool) else tool;
            const name = if (spec.get("name")) |n| (if (n.truthy()) try tool_specs.pyStr(a, n) else "") else "";
            if (name.len == 0) continue;
            try s.known.put(a, try std.ascii.allocLowerString(a, name), name);
        }
        return s;
    }

    fn args(s: *Streamer, fragment: []const u8) Allocator.Error!Value {
        const function = try json.newObject(s.a);
        try function.put(s.a, "arguments", .{ .string = fragment });
        const one = try json.newObject(s.a);
        try one.put(s.a, "index", try json.intValue(s.a, s.index));
        try one.put(s.a, "function", .{ .object = function });
        return wrap(s.a, .{ .object = one });
    }

    fn esc(s: *Streamer, text: []const u8) Allocator.Error![]const u8 {
        const q = try json.quote(s.a, text, .{ .ascii = false });
        return q[1 .. q.len - 1];
    }

    /// A new call's opening delta: its index, id and name, no arguments yet.
    fn opened(s: *Streamer, name: []const u8, out: *std.ArrayList(Value)) Allocator.Error!void {
        s.index += 1;
        s.streamed = true;
        const function = try json.newObject(s.a);
        try function.put(s.a, "name", .{ .string = name });
        try function.put(s.a, "arguments", .{ .string = "" });
        const one = try json.newObject(s.a);
        try one.put(s.a, "index", try json.intValue(s.a, s.index));
        try one.put(s.a, "id", .{ .string = try ids.make(s.a, "call_", 24) });
        try one.put(s.a, "type", .{ .string = "function" });
        try one.put(s.a, "function", .{ .object = function });
        try out.append(s.a, try wrap(s.a, .{ .object = one }));
    }

    /// The deltas DeepSeek-V4.1's calls blocks in ``text`` (the whole answer so far) add, read as ``tool_parse.dsml41Calls`` reads them:
    /// an invoke's opening delta once its name is whole, a string value's text as it is written, a JSON value (``string="false"``) at its closer.
    /// A call the reply's length cuts off stays open, as one ``feed`` streamed does.
    pub fn feedDsml(s: *Streamer, text: []const u8, out: *std.ArrayList(Value)) Allocator.Error!void {
        const d = &s.dsml;
        while (d.pos <= text.len) {
            const rest = text[d.pos..];
            switch (d.state) {
                .outside => {
                    const from = @max(d.pos, d.searched -| (dsml41_open.len - 1));
                    const j = reply_text.findTag(text, from, dsml41_open) orelse {
                        d.searched = text.len;
                        return;
                    };
                    d.pos = j + dsml41_open.len;
                    d.state = .block;
                },
                .block => {
                    const at = reply_text.findTag(rest, 0, invoke41_open);
                    if (reply_text.findTag(rest, 0, dsml41_close)) |c| if (at == null or c < at.?) {
                        d.pos += c + dsml41_close.len;
                        d.state = .outside;
                        continue;
                    };
                    const start = (at orelse return) + invoke41_open.len;
                    const q = std.mem.indexOfScalarPos(u8, rest, start, '"') orelse return;
                    if (q + 1 == rest.len) return;
                    if (q == start or rest[q + 1] != '>') {
                        d.pos += at.? + 1;
                        continue;
                    }
                    const name = reply_text.pyStrip(rest[start..q]);
                    d.pos += q + 2;
                    if (name.len == 0) {
                        d.state = .skip; // the parse drops a nameless invoke
                        continue;
                    }
                    const lower = try std.ascii.allocLowerString(s.a, name);
                    try s.opened(s.known.get(lower) orelse name, out); // an unoffered tool is a call under the name it was given
                    d.args_open = false;
                    d.state = .invoke;
                },
                .invoke => {
                    const p = reply_text.findTag(rest, 0, param41_open);
                    if (reply_text.findTag(rest, 0, invoke41_close)) |e| if (p == null or e < p.?) {
                        try out.append(s.a, try s.args(if (d.args_open) "}" else "{}"));
                        d.pos += e + invoke41_close.len;
                        d.state = .block;
                        continue;
                    };
                    const ps = p orelse return;
                    const ns = ps + param41_open.len;
                    const nq = std.mem.indexOfScalarPos(u8, rest, ns, '"') orelse return;
                    if (nq == ns) {
                        d.pos += ps + 1;
                        continue;
                    }
                    var v = nq + 1;
                    var json_value = false;
                    const flag = rest[v..];
                    if (std.mem.startsWith(u8, flag, string_true)) {
                        v += string_true.len;
                    } else if (std.mem.startsWith(u8, flag, string_false)) {
                        v += string_false.len;
                        json_value = true;
                    } else if (std.mem.startsWith(u8, string_true, flag) or std.mem.startsWith(u8, string_false, flag)) return; // either flag yet
                    if (v == rest.len) return;
                    if (rest[v] != '>') {
                        d.pos += ps + 1;
                        continue;
                    }
                    const key = try json.quote(s.a, rest[ns..nq], .{ .ascii = false });
                    try out.append(s.a, try s.args(try std.mem.concat(s.a, u8, &.{ if (d.args_open) "," else "{", key, if (json_value) ":" else ":\"" })));
                    d.args_open = true;
                    d.pos += v + 1;
                    d.state = if (json_value) .json_value else .string_value;
                },
                .string_value, .json_value => {
                    const value_end = reply_text.findTag(rest, 0, param41_close);
                    // the invoke's closer before the value's: the parse drops the value, and the stream closes the call there
                    const invoke_end = if (reply_text.findTag(rest, 0, invoke41_close)) |e| (if (value_end == null or e < value_end.?) e else null) else null;
                    const end = invoke_end orelse value_end;
                    var piece: []const u8 = undefined;
                    if (d.state == .string_value) {
                        const upto = end orelse rest.len - @max(reply_text.partialTag(rest, param41_close), reply_text.partialTag(rest, invoke41_close));
                        piece = try s.esc(rest[0..upto]);
                        if (end == null) {
                            if (piece.len > 0) try out.append(s.a, try s.args(piece));
                            d.pos += upto;
                            return;
                        }
                        piece = try std.mem.concat(s.a, u8, &.{ piece, "\"" });
                    } else {
                        const raw = rest[0 .. end orelse return];
                        const value: Value = switch (try json.parseText(s.a, raw)) {
                            .ok => |parsed| parsed,
                            .err => .{ .string = raw },
                        };
                        piece = try json.stringify(s.a, value, .{ .ascii = false, .compact = true });
                    }
                    if (invoke_end) |e| {
                        try out.append(s.a, try s.args(try std.mem.concat(s.a, u8, &.{ piece, "}" })));
                        d.pos += e + invoke41_close.len;
                        d.state = .block;
                        continue;
                    }
                    try out.append(s.a, try s.args(piece));
                    d.pos += value_end.? + param41_close.len;
                    d.state = .invoke;
                },
                .skip => {
                    const e = reply_text.findTag(rest, 0, invoke41_close) orelse return;
                    d.pos += e + invoke41_close.len;
                    d.state = .block;
                },
            }
        }
    }

    /// The deltas the reply's ``text`` (its whole answer so far) adds.
    pub fn feed(s: *Streamer, text: []const u8, out: *std.ArrayList(Value)) Allocator.Error!void {
        while (s.state != .abort) {
            const rest = text[@min(s.pos, text.len)..];
            switch (s.state) {
                .outside => {
                    if (text.len < s.searched) s.searched = s.pos;
                    const from = @min(text.len, @max(s.pos, s.searched -| ("<tool_call>".len - 1)));
                    const j = reply_text.findTag(text, from, "<tool_call>") orelse {
                        s.searched = text.len;
                        return;
                    };
                    s.pos = j + "<tool_call>".len;
                    s.state = .call;
                },
                .call => {
                    const m = opening(rest, "<function=") orelse {
                        const trimmed = reply_text.pyStrip(rest);
                        if (reply_text.charCount(rest) > 256 or (trimmed.len > 0 and trimmed[0] != '<')) s.state = .abort;
                        return;
                    };
                    const lower = try std.ascii.allocLowerString(s.a, reply_text.pyStrip(m.name));
                    const name = s.known.get(lower) orelse {
                        s.state = .abort;
                        return;
                    };
                    s.current = name;
                    try s.opened(name, out);
                    s.args_open = false;
                    s.pos += m.len;
                    s.state = .function;
                },
                .function => {
                    if (opening(rest, "<parameter=")) |m| {
                        const key = reply_text.pyStrip(m.name);
                        s.value_schema = try tool_params.schemaOf(&s.schemas, s.current, key, s.a);
                        s.typed = typedSchema(s.value_schema);
                        s.buffer.clearRetainingCapacity();
                        const quoted = try json.quote(s.a, key, .{ .ascii = false });
                        try out.append(s.a, try s.args(try std.mem.concat(s.a, u8, &.{ if (s.args_open) "," else "{", quoted, if (s.typed) ":" else ":\"" })));
                        s.args_open = true;
                        s.pos += m.len;
                        s.state = .value;
                        s.lead = !m.newline;
                        s.held = "";
                        continue;
                    }
                    if (closing(rest, "</function>")) |n| {
                        try out.append(s.a, try s.args(if (s.args_open) "}" else "{}"));
                        s.pos += n;
                        s.state = .closing;
                        continue;
                    }
                    return;
                },
                .value => {
                    const tail = "</parameter>";
                    const end = reply_text.findTag(rest, 0, tail);
                    const chunk = if (end) |e| rest[0..e] else rest[0..trimTail(rest, tail.len)];
                    s.pos += chunk.len;
                    var piece = try std.mem.concat(s.a, u8, &.{ s.held, chunk });
                    if (s.lead and piece.len > 0) {
                        if (piece[0] == '\n') piece = piece[1..];
                        s.lead = false;
                    }
                    var keep: []const u8 = undefined;
                    if (end != null) {
                        keep = if (std.mem.endsWith(u8, piece, "\n")) piece[0 .. piece.len - 1] else piece; // the framing newline
                        s.held = "";
                    } else {
                        keep = reply_text.pyRstrip(piece); // its last newline may be the framing one
                        s.held = piece[keep.len..];
                    }
                    if (s.typed) {
                        try s.buffer.appendSlice(s.a, keep);
                    } else if (keep.len > 0) try out.append(s.a, try s.args(try s.esc(keep)));
                    if (end == null) return;
                    if (s.typed) {
                        const v = try tool_params.decode(s.a, s.buffer.items, s.value_schema, true);
                        try out.append(s.a, try s.args(try json.stringify(s.a, v, .{ .ascii = false })));
                    } else try out.append(s.a, try s.args("\""));
                    s.pos += tail.len;
                    s.state = .function;
                },
                .closing => {
                    const n = closing(rest, "</tool_call>") orelse return;
                    s.pos += n;
                    s.state = .outside;
                },
                .abort => return,
            }
        }
    }
};

/// Bytes of ``rest`` that cannot be the start of a ``tail``-byte closer (Python slices characters).
fn trimTail(rest: []const u8, tail: usize) usize {
    const chars = reply_text.charCount(rest);
    if (chars <= tail) return 0;
    return rest.len - reply_text.afterChars(rest, chars - tail).len;
}

fn typedSchema(schema: Value) bool {
    const t = schema.get("type") orelse return false;
    if (t != .string) return false;
    for ([_][]const u8{ "array", "object", "boolean", "integer", "number", "null" }) |k| if (std.mem.eql(u8, k, t.string)) return true;
    return false;
}

const wrap = @import("tool_parse.zig").wrapCalls;

fn nameLike(s: []const u8) bool {
    if (s.len == 0 or !(std.ascii.isAlphabetic(s[0]) or s[0] == '_')) return false;
    for (s[1..]) |ch| if (!(std.ascii.isAlphanumeric(ch) or ch == '_' or ch == '.' or ch == '-' or ch >= 0x80)) return false;
    return true;
}

/// Hides tool envelopes from streamed prose, holding only a partial tag between deltas.
pub const TextFilter = struct {
    pending: std.ArrayList(u8) = .empty,
    hidden: ?[]const u8 = null,

    pub fn feed(f: *TextFilter, a: Allocator, text: []const u8) Allocator.Error![]const u8 {
        try f.pending.appendSlice(a, text);
        var out: std.ArrayList(u8) = .empty;
        while (f.pending.items.len > 0) {
            const p = f.pending.items;
            if (f.hidden) |name| {
                const close = try std.mem.concat(a, u8, &.{ "</", name, ">" });
                const end = std.ascii.findIgnoreCase(p, close) orelse {
                    const keep = reply_text.charCount(close) - 1;
                    const chars = reply_text.charCount(p);
                    const tail = if (chars > keep) reply_text.afterChars(p, chars - keep) else p;
                    try f.reset(a, tail);
                    break;
                };
                try f.reset(a, p[end + close.len ..]);
                f.hidden = null;
                continue;
            }
            const start = std.mem.indexOfScalar(u8, p, '<') orelse {
                try out.appendSlice(a, p);
                f.pending.clearRetainingCapacity();
                break;
            };
            try out.appendSlice(a, p[0..start]);
            const rest = p[start..];
            const end = std.mem.indexOfScalar(u8, rest, '>') orelse {
                const name = rest[1..];
                const colon = std.mem.indexOfScalar(u8, name, ':');
                const possible = name.len == 0 or nameLike(name) or (colon != null and nameLike(name[0..colon.?]) and
                    std.ascii.startsWithIgnoreCase("tool_call", name[colon.? + 1 ..]));
                if (possible) {
                    try f.reset(a, rest);
                    break;
                }
                try out.append(a, '<');
                try f.reset(a, rest[1..]);
                continue;
            };
            const name = rest[1..end];
            if (callTag(name)) {
                f.hidden = try std.ascii.allocLowerString(a, name);
            } else try out.appendSlice(a, rest[0 .. end + 1]);
            try f.reset(a, rest[end + 1 ..]);
        }
        return out.items;
    }

    fn reset(f: *TextFilter, a: Allocator, rest: []const u8) Allocator.Error!void {
        const copy = try a.dupe(u8, rest);
        f.pending.clearRetainingCapacity();
        try f.pending.appendSlice(a, copy);
    }

    pub fn finish(f: *TextFilter) []const u8 {
        const tail = if (f.hidden != null) "" else f.pending.items;
        f.pending = .empty;
        return tail;
    }
};

/// ``(?:NAME:)?tool_call``, any case.
fn callTag(name: []const u8) bool {
    if (std.ascii.eqlIgnoreCase(name, "tool_call")) return true;
    const colon = std.mem.indexOfScalar(u8, name, ':') orelse return false;
    return nameLike(name[0..colon]) and std.ascii.eqlIgnoreCase(name[colon + 1 ..], "tool_call");
}

/// ``parallel_tool_calls: false`` keeps one call and hides call markup from streamed prose.
pub const Policy = struct {
    single: bool = false,
    filter: TextFilter = .{},

    pub fn maxCalls(p: *const Policy) ?usize {
        return if (p.single) 1 else null;
    }

    /// Filters a str delta's text (dict deltas lose tool_calls and filter content), or passes it through.
    pub fn text(p: *Policy, a: Allocator, s: []const u8) Allocator.Error![]const u8 {
        return if (p.single) p.filter.feed(a, s) else s;
    }

    pub fn flush(p: *Policy) []const u8 {
        return if (p.single) p.filter.finish() else "";
    }

    pub fn content(p: *const Policy, a: Allocator, s: []const u8) Allocator.Error![]const u8 {
        if (!p.single) return s;
        var f: TextFilter = .{};
        const head = try f.feed(a, s);
        return std.mem.concat(a, u8, &.{ head, f.finish() });
    }

    /// ``parsed_content``: a parsed reply's content, whole envelopes the parser left as text kept and the rest filtered, so an unclosed envelope stays hidden.
    pub fn parsedContent(p: *const Policy, a: Allocator, s: []const u8) Allocator.Error![]const u8 {
        if (!p.single) return s;
        var out: std.ArrayList(u8) = .empty;
        var cursor: usize = 0;
        while (try wholeEnvelope(a, s, cursor)) |e| : (cursor = e.end) {
            try out.appendSlice(a, try p.content(a, s[cursor..e.start]));
            try out.appendSlice(a, s[e.start..e.end]);
        }
        try out.appendSlice(a, try p.content(a, s[cursor..]));
        return out.items;
    }

    /// What the filter held back that the reply keeps as text (a malformed call): ``final`` past the prose ``sent``.
    pub fn kept(p: *const Policy, final: []const u8, sent: []const u8) ?[]const u8 {
        if (!p.single) return null;
        const shown = reply_text.pyLstrip(sent); // the parsed content starts stripped
        if (!std.mem.startsWith(u8, final, shown)) return null;
        const rest = final[shown.len..];
        return if (reply_text.pyStrip(rest).len > 0) rest else null;
    }
};

/// The first whole envelope from ``from`` (``<((?:NAME:)?tool_call)>.*?</\1>``, any case).
fn wholeEnvelope(a: Allocator, s: []const u8, from: usize) Allocator.Error!?struct { start: usize, end: usize } {
    var at = from;
    while (std.mem.indexOfScalarPos(u8, s, at, '<')) |start| : (at = start + 1) {
        const close = std.mem.indexOfScalarPos(u8, s, start, '>') orelse return null;
        if (!callTag(s[start + 1 .. close])) continue;
        const closer = try std.mem.concat(a, u8, &.{ "</", s[start + 1 .. close], ">" });
        const end = std.ascii.findIgnoreCasePos(s, close + 1, closer) orelse continue;
        return .{ .start = start, .end = end + closer.len };
    }
    return null;
}

test "a streamed call its end token left open completes as its closing markup completes it" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"properties\":{\"n\":{\"type\":\"integer\"}}}}}]")).ok.array;
    const open = "<tool_call>\n<function=lookup>\n<parameter=q>\nfast cars\n</parameter>\n<parameter=n>\n5\n</parameter>\n";
    const sent = try sentArguments(a, try streamed(a, tools, &.{open}));
    const ended = try streamed(a, tools, &.{ open, try std.mem.concat(a, u8, &.{ open, try tool_parse.closeCall(a, open, tools) }) });
    const marked = try streamed(a, tools, &.{ open, open ++ "</function>\n</tool_call>" });
    // the arguments sent before the end, closed by closedJson, are the ones the client ends with
    try std.testing.expectEqualStrings((try tool_params.closedJson(a, sent)).?, try sentArguments(a, ended));
    try std.testing.expectEqual(marked.len, ended.len);
    for (marked[1..], ended[1..]) |x, y| try std.testing.expectEqualStrings(try json.stringify(a, x, .{}), try json.stringify(a, y, .{}));
}

test "a streamed call cut inside a value stays open" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}]")).ok.array;
    const open = "<tool_call>\n<function=lookup>\n<parameter=q>\nfast cars and more";
    try std.testing.expectEqualStrings("", try tool_parse.closeCall(a, open, tools));
    // its opening delta went out and cannot be taken back; what it sent does not close under closedJson
    const sent = try sentArguments(a, try streamed(a, tools, &.{open}));
    try std.testing.expectEqualStrings("{\"q\":\"fast c", sent);
    try std.testing.expect((try tool_params.closedJson(a, sent)) == null);
}

test "a one-call reply drops an unoffered call and still hides an unclosed one" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}]")).ok.array;
    const single: Policy = .{ .single = true };
    const text = "Launching. <tool_call><function=launch><parameter=when>now</function></tool_call>";
    // what attachCalls makes of the reply: the parse, then the policy's content
    const r = try tool_parse.parse(a, text, tools, single.maxCalls());
    try std.testing.expect(r.calls == null);
    try std.testing.expectEqualStrings("Launching.", r.content);
    try std.testing.expectEqualStrings("Launching.", try single.parsedContent(a, r.content));
    try std.testing.expectEqualStrings("Launching. ", try single.parsedContent(a, "Launching. <tool_call>{\"name\""));
    // a stream that sent the prose before the block ends with the block
    try std.testing.expectEqualStrings(text["Launching. ".len..], single.kept(text, "\nLaunching. ").?);
    try std.testing.expect(single.kept("Launching.", "Launching. ") == null);
    const parallel: Policy = .{};
    try std.testing.expect(parallel.kept(text, "Launching. ") == null);
}

const tool_parse = @import("tool_parse.zig");

/// The deltas one streamer sends while the reply grows through ``texts``.
fn streamed(a: Allocator, tools: []const Value, texts: []const []const u8) Allocator.Error![]Value {
    var s = try Streamer.init(a, tools);
    var out: std.ArrayList(Value) = .empty;
    for (texts) |t| try s.feed(t, &out);
    return out.items;
}

/// The arguments a client joins from streamed deltas.
fn sentArguments(a: Allocator, deltas: []const Value) Allocator.Error![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (deltas) |d| for (d.get("tool_calls").?.array) |one| {
        try out.appendSlice(a, one.get("function").?.get("arguments").?.string);
    };
    return out.items;
}

/// The deltas a reply's calls send while its answer grows through ``text`` a character at a time, both forms fed as chat feeds them.
fn answerStreamed(a: Allocator, tools: []const Value, text: []const u8) Allocator.Error![]Value {
    var s = try Streamer.init(a, tools);
    var out: std.ArrayList(Value) = .empty;
    for (0..text.len + 1) |i| {
        if (i < text.len and text[i] & 0xc0 == 0x80) continue; // whole characters, as a reply decodes
        try s.feed(tool_parse.beforeDsml(text[0..i]), &out);
        try s.feedDsml(text[0..i], &out);
    }
    return out.items;
}

/// Each streamed call as a client joins it: its name and arguments, by index.
fn joinedCalls(a: Allocator, deltas: []const Value) Allocator.Error![][2][]const u8 {
    var calls: std.ArrayList([2][]const u8) = .empty;
    for (deltas) |d| for (d.get("tool_calls").?.array) |one| {
        const i: usize = @intCast(one.get("index").?.int64().?);
        const function = one.get("function").?;
        if (i == calls.items.len) try calls.append(a, .{ function.get("name").?.string, "" });
        calls.items[i][1] = try std.mem.concat(a, u8, &.{ calls.items[i][1], function.get("arguments").?.string });
    };
    return calls.items;
}

const dsml_tools = "[{\"type\":\"function\",\"function\":{\"name\":\"write\"}},{\"type\":\"function\",\"function\":{\"name\":\"Lookup\"}}]";
const o41 = "<\u{ff5c}DSML\u{ff5c} ";
const c41 = "</\u{ff5c}DSML\u{ff5c} ";
const dsml_reply = "<tool_call>\n<function=lookup>\n<parameter=q>\nfirst\n</parameter>\n</function>\n</tool_call>\nThen the file.\n\n" ++
    o41 ++ "calls>\n" ++
    o41 ++ "invoke name=\"write\">\n" ++
    o41 ++ "parameter name=\"filePath\" string=\"true\">src/a.py" ++ c41 ++ "parameter>\n" ++
    o41 ++ "parameter name=\"content\" string=\"true\">def f(x):\n    return \"<b>\" + x  # \u{f1}, \u{6f22}\u{5b57}, \\ and </ tags\n" ++ c41 ++ "parameter>\n" ++
    o41 ++ "parameter name=\"mode\" string=\"false\">{\"append\": false, \"n\": [1, 2.5]}" ++ c41 ++ "parameter>\n" ++
    o41 ++ "parameter name=\"note\">plain" ++ c41 ++ "parameter>\n" ++
    o41 ++ "parameter name=\"bad\" string=\"false\">not json" ++ c41 ++ "parameter>\n" ++
    c41 ++ "invoke>\n" ++
    o41 ++ "invoke name=\"lookup\">\n" ++ c41 ++ "invoke>\n" ++
    o41 ++ "invoke name=\"launch\">\n" ++ o41 ++ "parameter name=\"when\" string=\"true\">now" ++ c41 ++ "parameter>\n" ++ c41 ++ "invoke>\n" ++
    c41 ++ "calls>";

test "DeepSeek-V4.1 calls stream their arguments as they are written and join to the parsed calls" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, dsml_tools)).ok.array;
    const parsed = (try tool_parse.parse(a, dsml_reply, tools, null)).calls.?;
    const got = try joinedCalls(a, try answerStreamed(a, tools, dsml_reply));
    try std.testing.expectEqual(@as(usize, 4), parsed.len);
    try std.testing.expectEqual(parsed.len, got.len);
    for (parsed, got) |want, have| {
        try std.testing.expectEqualStrings(want.get("function").?.get("name").?.string, have[0]);
        try std.testing.expectEqualStrings(want.get("function").?.get("arguments").?.string, have[1]);
    }
    // the content goes out before its closer is written
    const cut = std.mem.indexOf(u8, dsml_reply, "return").? + "return".len;
    const early = try joinedCalls(a, try answerStreamed(a, tools, dsml_reply[0..cut]));
    try std.testing.expectEqual(@as(usize, 2), early.len);
    try std.testing.expectEqualStrings("write", early[1][0]);
    try std.testing.expectEqualStrings("{\"filePath\":\"src/a.py\",\"content\":\"def f(x):\\n    return", early[1][1]);
}

test "a DeepSeek-V4.1 call the reply's length cuts off stays open" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, dsml_tools)).ok.array;
    const cut = dsml_reply[0 .. std.mem.indexOf(u8, dsml_reply, "+ x").? + 1];
    try std.testing.expect((try tool_parse.parse(a, cut, tools, null)).calls.?.len == 1); // the XML call alone
    const got = try joinedCalls(a, try answerStreamed(a, tools, cut));
    try std.testing.expectEqual(@as(usize, 2), got.len);
    try std.testing.expect(try json.parseText(a, got[1][1]) == .err);
    // a closer written partway is held, not sent as text
    const partial = try std.mem.concat(a, u8, &.{ cut, "\n", c41[0..5] });
    const held = try joinedCalls(a, try answerStreamed(a, tools, partial));
    try std.testing.expectEqualStrings(try std.mem.concat(a, u8, &.{ got[1][1], "\\n" }), held[1][1]);
}

test "a DeepSeek-V4.1 value its invoke closes ends that call, and the next call streams" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, dsml_tools)).ok.array;
    const text = o41 ++ "calls>\n" ++ o41 ++ "invoke name=\"write\">\n" ++ o41 ++ "parameter name=\"c\" string=\"true\">a" ++ c41 ++ "invoke>\n" ++
        o41 ++ "invoke name=\"Lookup\">\n" ++ o41 ++ "parameter name=\"n\" string=\"false\">5" ++ c41 ++ "parameter>\n" ++ c41 ++ "invoke>\n" ++ c41 ++ "calls>";
    const parsed = (try tool_parse.parse(a, text, tools, null)).calls.?;
    const got = try joinedCalls(a, try answerStreamed(a, tools, text));
    try std.testing.expectEqual(@as(usize, 2), got.len);
    // the parse drops the unclosed value; the stream had sent it, and closes the call around it
    try std.testing.expectEqualStrings("{}", parsed[0].get("function").?.get("arguments").?.string);
    try std.testing.expectEqualStrings("{\"c\":\"a\"}", got[0][1]);
    try std.testing.expectEqualStrings(parsed[1].get("function").?.get("arguments").?.string, got[1][1]);
}
