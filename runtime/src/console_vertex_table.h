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
#include <cstdint>

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
