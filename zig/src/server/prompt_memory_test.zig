//! A request keeps its prompt's ids, not the scratch of rendering, encoding and preparing its images: the reply can run
//! for minutes, and a long conversation's renders (two of them) or a few dozen screenshots' pixels held all that time
//! once took a GiB a request.
const std = @import("std");
const api = @import("engine_api");
const server = @import("root.zig");
const json = server.json;
const model_text = @import("model_text.zig");
const prompt = @import("prompt.zig");
const chat = @import("chat.zig");

/// Each byte its id, the image placeholder (\x01) included; rendering and encoding leave 64 bytes of scratch a byte.
const Wasteful = struct {
    fn text(t: *@This()) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource } };
    }

    fn encode(_: *anyopaque, a: std.mem.Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        _ = try a.alloc(u8, 64 * input.len);
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
    }

    fn decode(_: *anyopaque, a: std.mem.Allocator, ids: []const u32) std.mem.Allocator.Error![]u8 {
        const decoded = try a.alloc(u8, ids.len);
        for (ids, decoded) |id, *byte| byte.* = @intCast(id);
        return decoded;
    }

    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }

    fn tokenString(_: *anyopaque, a: std.mem.Allocator, id: u32) std.mem.Allocator.Error![]u8 {
        return std.fmt.allocPrint(a, "{d}", .{id});
    }

    fn vocabSize(_: *anyopaque) u32 {
        return 256;
    }

    fn eosIds(_: *anyopaque) []const u32 {
        return &.{};
    }

    fn render(_: *anyopaque, a: std.mem.Allocator, messages: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        if (messages != .array) return out.items;
        for (messages.array) |m| {
            const content = textOf(m, "content");
            _ = try a.alloc(u8, 64 * content.len);
            try out.print(a, "[{s}]{s}\n", .{ textOf(m, "role"), content });
        }
        return out.items;
    }

    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

fn textOf(v: json.Value, key: []const u8) []const u8 {
    const x = v.get(key) orelse return "";
    return if (x == .string) x.string else "";
}

/// An engine with vision whose prepare leaves 4 MiB of pixels behind for each image and gives a 64-byte payload.
const Seeing = struct {
    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn info(ctx: *anyopaque) api.Info {
        return .{ .vision = .{ .ctx = ctx, .placeholder = "\x01", .token = 1, .prepare = prepare } };
    }

    fn prepare(_: *anyopaque, a: std.mem.Allocator, bytes: []const u8, _: *[]const u8) api.Vision.PrepareError!api.Vision.Prepared {
        _ = try a.alloc(u8, 4 << 20);
        const payload = try a.alloc(u8, 64);
        @memset(payload, bytes[0]);
        return .{ .tokens = 4, .payload = payload };
    }

    fn submit(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {
        return error.Closed;
    }

    fn cancel(_: *anyopaque, _: api.Id) void {}

    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }

    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

test "a prepared prompt keeps its ids and image payloads, not its renders' or its images' scratch" {
    const gpa = std.testing.allocator;
    var text: Wasteful = .{};
    var engine: Seeing = .{};
    var srv = try server.Server.init(gpa, std.testing.io, engine.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
    defer srv.deinit();
    var body: std.ArrayList(u8) = .empty;
    defer body.deinit(gpa);
    try body.appendSlice(gpa, "[");
    const images = 12;
    for (0..200) |i| {
        if (i > 0) try body.appendSlice(gpa, ",");
        try body.print(gpa, "{{\"role\": \"{s}\", \"content\": \"", .{if (i % 2 == 0) "user" else "assistant"});
        for (0..1000) |_| try body.append(gpa, 'x');
        if (i % 2 == 0 and i / 2 < images) try body.appendSlice(gpa, "\\u0001");
        try body.appendSlice(gpa, "\"}");
    }
    try body.appendSlice(gpa, "]");
    var request: std.heap.ArenaAllocator = .init(gpa); // a request's allocator: nothing comes back before it ends
    defer request.deinit();
    const messages = (try json.parse(request.allocator(), body.items)).ok;
    var bytes: [images][]const u8 = undefined;
    for (&bytes, 0..) |*b, i| b.* = try std.fmt.allocPrint(request.allocator(), "{c}", .{@as(u8, @intCast('a' + i))});
    const before = request.queryCapacity();
    var cx: server.errors.Cx = .{ .a = request.allocator() };
    const rendered = try prompt.prepare(srv, &cx, .{ .messages = messages, .fields = .null, .images = &bytes }, false, null);
    try std.testing.expectEqual(@as(usize, images), rendered.images.len);
    try std.testing.expectEqual(@as(u8, 'c'), rendered.images[2].bytes[0]);
    try std.testing.expect(rendered.ids.len > 200_000);
    // the ids, the cache key and the payloads, a few bytes a token: two renders' and twelve pictures' scratch, kept, are
    // 64 bytes a token and 48 MiB
    try std.testing.expect(request.queryCapacity() - before < 24 * rendered.ids.len);
}
