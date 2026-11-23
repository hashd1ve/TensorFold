//! The tool_parse tests, as their own test binary: whole-reply parses and the close-call helpers.
const std = @import("std");
const json = @import("json.zig");
const tool_parse = @import("tool_parse.zig");
const parse = tool_parse.parse;
const closeCall = tool_parse.closeCall;
const reply_text = @import("reply_text.zig");
const tool_stream = @import("tool_stream.zig");

test "families" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"properties\":{\"n\":{\"type\":\"integer\"}}}}}]")).ok.array;
    const cases = [_][2][]const u8{
        .{ "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"x\"}}</tool_call>", "{\"q\":\"x\"}" },
        .{ "<tool_call>\n<function=lookup>\n<parameter=n>\n5\n</parameter>\n</function>\n</tool_call>", "{\"n\":5}" },
        .{ "<tool_call>lookup<arg_key>n</arg_key><arg_value>7</arg_value></tool_call>", "{\"n\":7}" },
        .{ "<|tool_call>call:lookup{q:<|\"|>hi, there<|\"|>,n:3}<tool_call|>", "{\"q\":\"hi, there\",\"n\":3}" },
        .{ "{\"tool\":\"lookup\",\"query\":\"ping\"}", "{\"query\":\"ping\"}" },
    };
    for (cases) |c| {
        const r = try parse(a, c[0], tools, null);
        try std.testing.expectEqualStrings(c[1], r.calls.?[0].get("function").?.get("arguments").?.string);
    }
    const kept = try parse(a, "hi <tool_call>{\"name\":\"other tool\"}</tool_call>", tools, null);
    try std.testing.expect(kept.calls == null);
    try std.testing.expectEqualStrings("hi <tool_call>{\"name\":\"other tool\"}</tool_call>", kept.content);
}

test "a call the end token left open closes when it parses whole" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"properties\":{\"n\":{\"type\":\"integer\"}}}}}]")).ok.array;
    const cases = [_][4][]const u8{ // the reply, what closes it, the call's arguments, the content left
        .{ "Looking.<tool_call>\n{\"name\":\"lookup\",\"arguments\":{\"q\":\"x\"}}\n", "</tool_call>", "{\"q\":\"x\"}", "Looking." },
        .{ "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":[\"x\"", "]}}</tool_call>", "{\"q\":[\"x\"]}", "" },
        .{ "<tool_call>\n<function=lookup>\n<parameter=n>\n5\n</parameter>\n", "</function></tool_call>", "{\"n\":5}", "" },
        .{ "<tool_call>\n<function=lookup>\n", "</function></tool_call>", "{}", "" },
        .{ "<tool_call>\n<function=lookup>\n<parameter=n>\n5\n</parameter>\n</function>\n", "</tool_call>", "{\"n\":5}", "" },
        .{ "<tool_call>lookup<arg_key>n</arg_key><arg_value>7</arg_value>", "</tool_call>", "{\"n\":7}", "" },
        .{ "<tool_call>{\"name\":\"lookup\"}</tool_call>\n<tool_call>{\"name\":\"lookup\",\"arguments\":{\"n\":2}", "}</tool_call>", "{\"n\":2}", "" },
        .{ "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"<tool_call>\"}", "}</tool_call>", "{\"q\":\"<tool_call>\"}", "" },
    };
    for (cases) |c| {
        const close = try closeCall(a, c[0], tools);
        try std.testing.expectEqualStrings(c[1], close);
        const r = try parse(a, try std.mem.concat(a, u8, &.{ c[0], close }), tools, null);
        try std.testing.expectEqualStrings(c[2], r.calls.?[r.calls.?.len - 1].get("function").?.get("arguments").?.string);
        try std.testing.expectEqualStrings(c[3], r.content);
    }
}

test "a call left open that would not parse whole stays the reply's text" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"properties\":{\"n\":{\"type\":\"integer\"}}}}}]")).ok.array;
    const replies = [_][]const u8{
        "Trying.<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"fast ca", // inside a string
        "Trying.<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":", // a key without its value
        "Trying.<tool_call>\n<function=lookup>\n<parameter=q>\nfast ca", // inside a parameter
        "Trying.<tool_call>\n<function=lookup>\n<parameter=n>\n5\n", // before its </parameter>
        "Trying.<tool_call>{\"name\":\"launch rocket!\",\"arguments\":{}}", // no function name
        "Trying.<tool_call>lookup<arg_key>n</arg_key><arg_value>7", // a GLM value left open
        "Trying.<tool_call>", // nothing written
        "Trying <tool_call> tags.<tool_call>{\"name\":\"lookup\"}", // an earlier opener would take the closer
    };
    for (replies) |text| {
        try std.testing.expectEqualStrings("", try closeCall(a, text, tools));
        const r = try parse(a, text, tools, null);
        try std.testing.expect(r.calls == null);
        try std.testing.expectEqualStrings(text, r.content);
    }
}

test "replies that end outside a call close nothing" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}]")).ok.array;
    const replies = [_][]const u8{
        "Done.",
        "<tool_call>{\"name\":\"lookup\"}</tool_call>",
        "<tool_call>\n<function=lookup>\n</function>\n</tool_call> Done.",
    };
    for (replies) |text| try std.testing.expectEqualStrings("", try closeCall(a, text, tools));
}

test "a call to a tool the request did not declare stays text" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}]")).ok.array;
    const whole = [_][]const u8{
        "<tool_call><function=launch><parameter=when>now</parameter></function></tool_call>",
        "<tool_call>{\"name\": \"launch rocket!\", \"arguments\": {\"when\": \"now\"}}</tool_call>",
        "<tool_call>{\"name\": \"launch\", \"arguments\": {\"when\": NaN}}</tool_call>",
    };
    for (whole) |text| {
        const parallel = try parse(a, text, tools, null);
        try std.testing.expect(parallel.calls == null);
        try std.testing.expectEqualStrings(text, parallel.content);
        const single = try parse(a, text, tools, 1);
        try std.testing.expect(single.calls == null);
        try std.testing.expectEqualStrings("", single.content);
    }
    // an unclosed envelope is not a call in either mode, so the reply keeps it
    const open = "<tool_call>{\"name\":\"launch\",\"arguments\":{\"when\":\"now\"}";
    for ([_]?usize{ null, 1 }) |max_calls| {
        const k = try parse(a, open, tools, max_calls);
        try std.testing.expect(k.calls == null);
        try std.testing.expectEqualStrings(open, k.content);
    }
    const prose = "Launching. <tool_call><function=launch><parameter=when>now</function></tool_call>";
    try std.testing.expectEqualStrings("Launching.", (try parse(a, prose, tools, 1)).content);
    // an offered tool keeps its offered spelling; bare JSON naming one not offered stays content
    const offered = try parse(a, "<tool_call>{\"name\":\"LOOKUP\"}</tool_call>", tools, null);
    try std.testing.expectEqualStrings("lookup", offered.calls.?[0].get("function").?.get("name").?.string);
    try std.testing.expect((try parse(a, "{\"name\":\"launch\",\"arguments\":{}}", tools, null)).calls == null);
}

test "DeepSeek-V4.1's DSML calls, read as the served lane reads them" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"weather\"}},{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}]")).ok.array;
    const D = "\u{ff5c}DSML\u{ff5c}";
    // two invokes, string and JSON values, prose before the block
    const two = "Checking.\n\n<" ++ D ++ " calls>\n<" ++ D ++ " invoke name=\"weather\">\n<" ++ D ++ " parameter name=\"city\" string=\"true\">Oslo</" ++ D ++ " parameter>\n<" ++ D ++ " parameter name=\"days\" string=\"false\">3</" ++ D ++ " parameter>\n</" ++ D ++ " invoke>\n<" ++ D ++ " invoke name=\"LOOKUP\">\n<" ++ D ++ " parameter name=\"q\" string=\"false\">{\"a\": [1, 2.5, null]}</" ++ D ++ " parameter>\n</" ++ D ++ " invoke>\n</" ++ D ++ " calls>";
    const r = try parse(a, two, tools, null);
    try std.testing.expectEqualStrings("Checking.", r.content);
    try std.testing.expectEqual(@as(usize, 2), r.calls.?.len);
    try std.testing.expectEqualStrings("weather", r.calls.?[0].get("function").?.get("name").?.string);
    try std.testing.expectEqualStrings("{\"city\":\"Oslo\",\"days\":3}", r.calls.?[0].get("function").?.get("arguments").?.string);
    try std.testing.expectEqualStrings("lookup", r.calls.?[1].get("function").?.get("name").?.string);
    try std.testing.expectEqualStrings("{\"q\":{\"a\":[1,2.5,null]}}", r.calls.?[1].get("function").?.get("arguments").?.string);
    // one call a reply: the second invoke drops out
    const single = try parse(a, two, tools, 1);
    try std.testing.expectEqual(@as(usize, 1), single.calls.?.len);
    // JSON that does not parse is the string; no string attribute is a string
    const loose = "<" ++ D ++ " calls><" ++ D ++ " invoke name=\"weather\"><" ++ D ++ " parameter name=\"x\" string=\"false\">not json</" ++ D ++ " parameter><" ++ D ++ " parameter name=\"y\">7</" ++ D ++ " parameter></" ++ D ++ " invoke></" ++ D ++ " calls>";
    try std.testing.expectEqualStrings("{\"x\":\"not json\",\"y\":\"7\"}", (try parse(a, loose, tools, null)).calls.?[0].get("function").?.get("arguments").?.string);
    // a tool the request did not offer is a call under the model's name, for the client to answer; one call a reply drops it
    const other = "<" ++ D ++ " calls>\n<" ++ D ++ " invoke name=\"launch\">\n<" ++ D ++ " parameter name=\"when\" string=\"true\">now</" ++ D ++ " parameter>\n</" ++ D ++ " invoke>\n</" ++ D ++ " calls>";
    const kept = try parse(a, other, tools, null);
    try std.testing.expectEqualStrings("", kept.content);
    try std.testing.expectEqual(@as(usize, 1), kept.calls.?.len);
    try std.testing.expectEqualStrings("launch", kept.calls.?[0].get("function").?.get("name").?.string);
    try std.testing.expectEqualStrings("{\"when\":\"now\"}", kept.calls.?[0].get("function").?.get("arguments").?.string);
    const one = try parse(a, other, tools, 1);
    try std.testing.expectEqualStrings("", one.content);
    try std.testing.expect(one.calls == null);
    // a block with no invoke goes; an unclosed block stays text
    try std.testing.expectEqualStrings("Hi.", (try parse(a, "Hi.<" ++ D ++ " calls>\n</" ++ D ++ " calls>", tools, null)).content);
    const open = "Hi.<" ++ D ++ " calls>\n<" ++ D ++ " invoke name=\"weather\">";
    try std.testing.expectEqualStrings(open, (try parse(a, open, tools, null)).content);
}

test "a DSML block is read from its own tags: <tool_call> text around it or in an argument moves nothing" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"weather\"}},{\"type\":\"function\",\"function\":{\"name\":\"write\"}}]")).ok.array;
    const D = "\u{ff5c}DSML\u{ff5c}";
    const quoted = "a </tool_call> and <tool_call>{\"name\": \"weather\", \"arguments\": {}}</tool_call>";
    const block = "<" ++ D ++ " calls>\n<" ++ D ++ " invoke name=\"write\">\n<" ++ D ++ " parameter name=\"content\" string=\"true\">" ++ quoted ++ "</" ++ D ++ " parameter>\n</" ++ D ++ " invoke>\n</" ++ D ++ " calls>";
    // a <tool_call> in the prose before the block, and a </tool_call> and a whole call in its argument
    const r = try parse(a, "I will not use <tool_call> here.\n\n" ++ block, tools, null);
    try std.testing.expectEqualStrings("I will not use <tool_call> here.", r.content);
    try std.testing.expectEqual(@as(usize, 1), r.calls.?.len);
    try std.testing.expectEqualStrings("write", r.calls.?[0].get("function").?.get("name").?.string);
    const args = (try json.parse(a, r.calls.?[0].get("function").?.get("arguments").?.string)).ok;
    try std.testing.expectEqualStrings(quoted, args.get("content").?.string);
    // calls keep the reply's order: a <tool_call> call before the block comes first
    const both = try parse(a, "<tool_call>{\"name\": \"weather\", \"arguments\": {\"city\": \"Oslo\"}}</tool_call>\n" ++ block, tools, null);
    try std.testing.expectEqual(@as(usize, 2), both.calls.?.len);
    try std.testing.expectEqualStrings("weather", both.calls.?[0].get("function").?.get("name").?.string);
    try std.testing.expectEqualStrings("write", both.calls.?[1].get("function").?.get("name").?.string);
    // the call streamer reads only the text before the first block
    try std.testing.expectEqualStrings("Hi. ", tool_parse.beforeDsml("Hi. " ++ block ++ " more"));
    try std.testing.expectEqualStrings("no block", tool_parse.beforeDsml("no block"));
}

test "a tool-call block too short to hold the DSML block it opens with is no DSML block: no call, text in a whole reply, dropped in a one-call reply, no stream sends a call" {
    const D = "\u{ff5c}DSML\u{ff5c}";
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"write\"}}]")).ok.array;
    const short = [_][]const u8{
        "<tool_call><" ++ D ++ " calls></tool_call>",
        "<tool_call><" ++ D ++ "tool_calls></tool_call>",
        "Sure: <tool_call> <" ++ D ++ " calls> x </tool_call>",
        "<TOOL_CALL><" ++ D ++ " calls></TOOL_CALL>",
        "<|tool_call><" ++ D ++ " calls><tool_call|>",
        "<ns:tool_call><" ++ D ++ "tool_calls></ns:tool_call>",
        "<tool_call><" ++ D ++ " calls></" ++ D ++ " calls</tool_call>", // one byte short
        "<tool_call><" ++ D ++ "tool_calls></" ++ D ++ "tool_calls</tool_call>", // one byte short, V4
    };
    // the stream filter's text at the end of each reply: it hides an unclosed DSML block from its opener on
    const streamed_text = [_][]const u8{ "", "", "Sure:", "<TOOL_CALL>", "", "<ns:tool_call>", "", "" };
    for (short, streamed_text) |text, shown| {
        // with no limit the block stays the reply's text; a one-call reply drops it and keeps the prose before it
        const r = try parse(a, text, tools, null);
        try std.testing.expect(r.calls == null);
        try std.testing.expectEqualStrings(text, r.content);
        const one = try parse(a, text, tools, 1);
        try std.testing.expect(one.calls == null);
        try std.testing.expectEqualStrings(if (std.mem.startsWith(u8, text, "Sure:")) "Sure:" else "", one.content);
        try std.testing.expectEqualStrings(shown, std.mem.trim(u8, try reply_text.hideToolCalls(a, text, true), " \t\r\n"));
        // the live streamers, fed byte by byte, send no call: the <tool_call> one and the DSML one
        var st = try tool_stream.Streamer.init(a, tools);
        var out: std.ArrayList(json.Value) = .empty;
        for (1..text.len + 1) |n| {
            try st.feed(text[0..n], &out);
            try st.feedDsml(text[0..n], &out);
        }
        try std.testing.expect(!st.streamed);
        try std.testing.expectEqual(@as(usize, 0), out.items.len);
    }
    // after an offered call, either spelling, each limit: a V4 block is read after the limit is checked, a V4.1 block
    // before it, so before the guard a V4.1 block crashed the parser at any limit and a V4 block only with no limit
    const call = "<tool_call>{\"name\": \"write\", \"arguments\": {}}</tool_call>";
    for ([_][]const u8{ "<tool_call><" ++ D ++ "tool_calls></tool_call>", "<tool_call><" ++ D ++ " calls></tool_call>" }) |block| {
        const text = try std.mem.concat(a, u8, &.{ call, "\n", block });
        const unlimited = try parse(a, text, tools, null);
        try std.testing.expectEqual(@as(usize, 1), unlimited.calls.?.len);
        try std.testing.expectEqualStrings(block, unlimited.content);
        const none = try parse(a, text, tools, 0);
        try std.testing.expect(none.calls == null);
        try std.testing.expectEqualStrings("", none.content);
        const one = try parse(a, text, tools, 1);
        try std.testing.expectEqual(@as(usize, 1), one.calls.?.len);
        try std.testing.expectEqualStrings("", one.content);
    }
    // long enough, a payload reads exactly as it always did, with or without a closer at its end: these are the parser's
    // results before the guard, unchanged (content with no limit, then with a one-call limit; no call in any of them)
    const long_enough = [_][3][]const u8{
        .{ "<tool_call><" ++ D ++ " calls></" ++ D ++ " calls></tool_call>", "<tool_call></tool_call>", "<tool_call></tool_call>" },
        .{ "<tool_call><" ++ D ++ "tool_calls></" ++ D ++ "tool_calls></tool_call>", "<tool_call><" ++ D ++ "tool_calls></" ++ D ++ "tool_calls></tool_call>", "" },
        .{ "<tool_call><" ++ D ++ " calls>0123456789012345678901</tool_call>", "", "" },
        .{ "<tool_call><" ++ D ++ "tool_calls>01234567890123456789012345</tool_call>", "<tool_call><" ++ D ++ "tool_calls>01234567890123456789012345</tool_call>", "" },
    };
    for (long_enough) |c| {
        const r = try parse(a, c[0], tools, null);
        try std.testing.expect(r.calls == null);
        try std.testing.expectEqualStrings(c[1], r.content);
        const one = try parse(a, c[0], tools, 1);
        try std.testing.expect(one.calls == null);
        try std.testing.expectEqualStrings(c[2], one.content);
    }
}
