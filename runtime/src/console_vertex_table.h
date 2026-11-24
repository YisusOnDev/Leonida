#pragma once
// The console driver's vertex-buffer table (docs/d3d12_vulkan.md section 3.5, "The console's vertex-buffer table").
//
// A vertex shader compiled for the console and shipped precompiled (slot 57's pipeline library, fspx_gfx) does not
// call a fetch shader: it reads one 16-byte buffer V# per input slot from the table the 40-bit pointer s[0:1] names
// (s0, and bits 0-7 of s1), adds s2 to its vertex index and fetches each attribute itself (measured on fspx_gfx
// 54edf226..., the composite into the back buffer: V# at s[0:1] + 0, stride from word 1 bits 29:16, NUM_RECORDS
// checked, attribute offsets 0, 12 and 16 of a 24-byte vertex). The layer builds that table from the bound
// IASetVertexBuffers views, as the console's IASetVertexBuffers would.
#include "command_list.h"
#include "gfx10_descriptors.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace leonida {

// Bytes per table: one V# for each of the 32 input slots D3D12 has.
inline constexpr std::uint32_t kConsoleVertexTableBytes = 32 * 16;

// Writes the V#s of `views` to `out` (four words per slot), from slot 0 up to the last slot `used` names (bit per
// slot; 0: up to the last bound view). `shader_va` turns a view's VA into the console VA a shader reads it by (the
// arena copy of the title's memory, title_va_alias.h). An unbound slot gets a null V# (NUM_RECORDS 0: loads read 0).
// Returns the slots written.
template <class ShaderVa>
std::uint32_t console_vertex_table(const std::array<VertexBufferView, 32>& views, std::uint32_t used,
                                   ShaderVa&& shader_va, std::array<std::uint32_t, 4 * 32>& out) {
    out.fill(0);
    std::uint32_t slots = 0;
    for (std::uint32_t slot = 0; slot < 32; ++slot) {
        const bool wanted = used ? ((used >> slot) & 1u) != 0 : views[slot].va != 0;
        if (wanted) slots = slot + 1;
    }
    for (std::uint32_t slot = 0; slot < slots; ++slot) {
        const VertexBufferView& v = views[slot];
        if (!v.va || !v.size) continue;
        gfx10::Buffer buffer{};
        buffer.address = shader_va(v.va, v.size);
        buffer.stride = v.stride & 0x3FFFu;
        buffer.num_records = buffer.stride ? v.size / buffer.stride : v.size;
        // Any valid format: the fetches are typed by the instruction (tbuffer) or raw dwords; FORMAT 0 (INVALID)
        // would suppress the typed ones.
        buffer.format = gfx10::format_from_vulkan(VK_FORMAT_R32_UINT);
        buffer.oob_select = buffer.stride ? gfx10::OobSelect::structured : gfx10::OobSelect::raw;
        const gfx10::BufferWords words = gfx10::encode(buffer);
        for (unsigned k = 0; k < 4; ++k) out[slot * 4 + k] = words[k];
    }
    return slots;
}

// A rectangle list (topology 17) drawn by such a program: its pipeline is a triangle list (slot 57 does not carry the
// rectangle topology type), so the vertices are expanded before the draw, six per rectangle: the three given, then the
// two that share the corner opposite the right angle and the fourth corner. The right angle is found from the
// position (the first two dwords, floats); the fourth corner is a + b - corner for every dword, except one with the
// same bits at all three vertices (a packed colour, say), which is copied. `in` holds 3 * rects vertices of `stride`
// bytes (a multiple of 4, at least 8); `out` gets 6 * rects. The composite into the back buffer is one such rectangle:
// (0,0), (0,2160), (3840,2160), right angle at the second vertex, fourth corner (3840,0).
// Reference: VIctory (GPL-3) crates/victory-gpu/src/primitive.rs expands rectangles the same way on the CPU, from the
// vertex inputs, with the fourth corner v0 + v2 - v1 (GTA VI's full-screen rectangles come top left, bottom left,
// bottom right); here the right-angle corner is found instead of assumed. No code is taken from it.
inline bool expand_rect_list(const std::uint8_t* in, std::uint32_t stride, std::uint32_t rects, std::uint8_t* out) {
    if (stride < 8 || stride % 4) return false;
    const std::uint32_t words = stride / 4;
    const auto word = [&](std::uint32_t vertex, std::uint32_t k) {
        std::uint32_t w;
        std::memcpy(&w, in + std::size_t(vertex) * stride + 4 * k, 4);
        return w;
    };
    const auto real = [](std::uint32_t w) {
        float f;
        std::memcpy(&f, &w, 4);
        return f;
    };
    for (std::uint32_t r = 0; r < rects; ++r) {
        const std::uint32_t v = 3 * r;
        // The corner whose two edges are perpendicular (the smallest |dot|).
        std::uint32_t corner = 1;
        float best = -1.0f;
        for (std::uint32_t c = 0; c < 3; ++c) {
            const std::uint32_t a = (c + 1) % 3, b = (c + 2) % 3;
            const float ax = real(word(v + a, 0)) - real(word(v + c, 0)), ay = real(word(v + a, 1)) - real(word(v + c, 1));
            const float bx = real(word(v + b, 0)) - real(word(v + c, 0)), by = real(word(v + b, 1)) - real(word(v + c, 1));
            const float dot = std::fabs(ax * bx + ay * by);
            if (!(dot == dot)) continue;  // NaN
            if (best < 0.0f || dot < best) best = dot, corner = c;
        }
        const std::uint32_t a = (corner + 1) % 3, b = (corner + 2) % 3;
        std::uint8_t* to = out + std::size_t(6 * r) * stride;
        std::memcpy(to, in + std::size_t(v) * stride, std::size_t(3) * stride);
        std::memcpy(to + 3 * std::size_t(stride), in + std::size_t(v + a) * stride, stride);
        std::memcpy(to + 4 * std::size_t(stride), in + std::size_t(v + b) * stride, stride);
        for (std::uint32_t k = 0; k < words; ++k) {
            const std::uint32_t wc = word(v + corner, k), wa = word(v + a, k), wb = word(v + b, k);
            std::uint32_t w4 = wc;
            if (!(wc == wa && wa == wb)) {
                const float f = real(wa) + real(wb) - real(wc);
                if (f == f && std::fabs(f) <= 3.0e38f) std::memcpy(&w4, &f, 4);
            }
            std::memcpy(to + 5 * std::size_t(stride) + 4 * k, &w4, 4);
        }
    }
    return true;
}

// The user words a draw with such a vertex shader gets: s[0:1] the table's console VA (s1 keeps its upper bits), s2
// the vertex offset (first vertex, or the base vertex of an indexed draw), s3 the first instance.
inline void console_vertex_user_words(std::uint32_t* s, std::uint64_t table_va, bool indexed, std::uint32_t first,
                                      std::int32_t base_vertex, std::uint32_t first_instance) {
    s[0] = std::uint32_t(table_va);
    s[1] = (s[1] & ~0xFFu) | std::uint32_t((table_va >> 32) & 0xFFu);
    s[2] = indexed ? std::uint32_t(base_vertex) : first;
    s[3] = first_instance;
}

}  // namespace leonida
