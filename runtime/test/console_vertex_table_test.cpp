// leonida::console_vertex_table (src/console_vertex_table.h): the table of vertex-buffer V#s a console-compiled vertex
// program reads through s[0:1], and the user words such a draw gets. No GPU: the V#s are decoded back and checked
// against what the composite's vertex shader (fspx_gfx 54edf226) reads from them (docs/d3d12_vulkan.md section 3.5).
#include "../src/console_vertex_table.h"

#include <cstdint>
#include <cstdio>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

int main() {
    using namespace leonida;
    std::array<VertexBufferView, 32> views{};
    views[0] = {0x15232badf10ull, 4080, 24};  // title memory: a 24-byte vertex (position, colour, uv)
    views[2] = {0x3000000ull, 256, 16};
    const auto shader_va = [](std::uint64_t va, std::uint64_t) {
        return va == 0x15232badf10ull ? 0x4123f10ull : va;  // the arena copy of the title's page
    };
    std::array<std::uint32_t, 4 * 32> words{};

    // The input layout uses slot 0 only: one V#.
    EXPECT(console_vertex_table(views, 0x1u, shader_va, words) == 1);
    gfx10::BufferWords v0{words[0], words[1], words[2], words[3]};
    gfx10::Buffer b{};
    EXPECT(gfx10::decode(v0, b));
    EXPECT(b.address == 0x4123f10ull);  // the copy's VA, not the title's
    EXPECT(b.stride == 24 && b.num_records == 4080 / 24);
    // What 54edf226 reads: base = s8 | (s9 & 0xffff) << 32, stride = s9 bits 16-29, NUM_RECORDS = s10, FORMAT (gates
    // the colour fetch) = s11 bits 12-18 non-zero, OOB_SELECT = s11 bits 28-29.
    EXPECT((words[0] | (std::uint64_t(words[1] & 0xFFFFu) << 32)) == 0x4123f10ull);
    EXPECT(((words[1] >> 16) & 0x3FFFu) == 24);
    EXPECT(words[2] == 170);
    EXPECT(((words[3] >> 12) & 0x7Fu) != 0);
    EXPECT(((words[3] >> 28) & 3u) == std::uint32_t(gfx10::OobSelect::structured));
    EXPECT(words[4] == 0 && words[7] == 0);  // nothing past the used slots

    // Slots 0 and 2 used, slot 1 unbound: a null V# between them.
    EXPECT(console_vertex_table(views, 0x5u, shader_va, words) == 3);
    EXPECT(words[4] == 0 && words[5] == 0 && words[6] == 0 && words[7] == 0);
    EXPECT(words[8] == 0x3000000u && ((words[9] >> 16) & 0x3FFFu) == 16 && words[10] == 16);

    // No input layout recorded: up to the last bound view.
    EXPECT(console_vertex_table(views, 0, shader_va, words) == 3);

    // The user words: s[0:1] the table (s1 keeps bits 8-31), s2 the vertex offset, s3 the first instance.
    std::uint32_t s[32] = {};
    s[1] = 0xABCD0000u;
    console_vertex_user_words(s, 0x12345678ull, false, 7, -3, 2);
    EXPECT(s[0] == 0x12345678u && s[1] == 0xABCD0000u && s[2] == 7 && s[3] == 2);
    console_vertex_user_words(s, 0x1'0000'0040ull, true, 7, 5, 0);
    EXPECT(s[0] == 0x40u && s[1] == 0xABCD0001u && s[2] == 5 && s[3] == 0);

    std::printf("console_vertex_table: V#s per used slot at the arena copy, null V# for an unbound slot, user words\n");
    return 0;
}
