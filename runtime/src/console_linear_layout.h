// The console's linear texture layout as the runtime hands it to the title: rows of whole blocks at a row pitch
// rounded up to 256 bytes, slices of whole rows, subresources one after another. Shared by xg_xs.dll's texture
// computer (slots 2, 8 and 9) and the Vulkan layer's texture Map, which must agree byte for byte: the title fills a
// texture through the Map pointer at the offsets and pitches slot 2 told it.
//
// Why 256 and not tight rows: the title's own code assumes it. game_scarlett_master.exe+0x532F140 writes back (clwb)
// the CPU-filled planes of a texture set before the GPU reads them, each over align256(width) * height bytes
// (+0x5EC160: `lea eax,[rcx+0FFh]; and eax,0FFFFFF00h; imul eax,edx`). With tight rows, a 1920x1080 one-byte plane is
// 0x1FA400 bytes and the write-back runs 0x21C00 bytes past it: the read access violation at +0x532F1F0 (boot
// frontier4-7, 105.7 s, the first page after a Map copy of exactly that size). 256 bytes is also D3D12's
// D3D12_TEXTURE_DATA_PITCH_ALIGNMENT, the pitch the PC asks of linear texture data.
#pragma once

#include <cstdint>

namespace leonida {

constexpr std::uint64_t kConsoleLinearPitchAlignment = 256;

// The row pitch of a row of `row_bytes` bytes of whole blocks.
constexpr std::uint64_t console_linear_pitch(std::uint64_t row_bytes) {
    return (row_bytes + kConsoleLinearPitchAlignment - 1) & ~(kConsoleLinearPitchAlignment - 1);
}

// The bytes one subresource takes: `slices` slices of `rows` rows of blocks.
constexpr std::uint64_t console_linear_size(std::uint64_t row_bytes, std::uint64_t rows, std::uint64_t slices) {
    return console_linear_pitch(row_bytes) * rows * slices;
}

}  // namespace leonida
