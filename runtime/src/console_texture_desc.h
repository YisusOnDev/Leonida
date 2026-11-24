// A console D3D12_RESOURCE_DESC as the console lays it out: the swizzle mode its Layout field names (or the one the
// console's library picks), the texture layout of src/console_texture_layout.h, and the allocation size and alignment
// the console's device answers for it. xg_xs.dll's texture computer (runtime/shims/xg_xs.cpp) and the device proxy's
// GetResourceAllocationInfo (slot 25, runtime/shims/d3d12_xs_device.cpp) share it, so the two cannot disagree.
//
// Origin: crates/victory-hle/src/modules/xg.rs (the suggested mode) and crates/victory-gpu/src/resource.rs
// (texture_layout and footprint) of VIctory (github.com/Marvisak/VIctory, upstream 204bdcc, GPL-3.0-or-later).
//
// Why the allocation size matters: the title builds its per-texture streaming classes (exe 0x149C3E1D0, 0x38-byte
// entries, built at +0x51FB680) from one of two answers. Tiled textures go through XGCreateTextureComputer and the
// computer's GetResourceTiling (+0x3441F0); the others through ID3D12Device::GetResourceAllocationInfo (+0x343E80, the
// device vtable's +0xC8), whose size becomes the class's page count ((size + 0xFFF) >> 12 at +0x20/+0x22, the size at
// +0x2C). The archive read builder (+0x4FFD520) then reads that many pages and asserts (int3 +0x4ffd879) when a
// compressed chunk is longer. Measured (boot frontier5-3): a 128x4 R16G16B16A16 4KB_S texture's class held 1 page
// (the PC answer, 0x1000 bytes) where its archive chunk holds 0x4000, its 4KB_S size (32x16-element blocks, 4 of them).
#pragma once

#include <windows.h>
#include <d3d12.h>

#include <algorithm>
#include <cstdint>
#include <optional>

#include "console_texture_layout.h"

namespace leonida::console_layout {

constexpr uint32_t kDescSwizzled = 0x100, kDescRowMajor = 1;
constexpr uint32_t kDescRenderTarget = 1, kDescDepthStencil = 2;  // D3D12_RESOURCE_FLAGS, same bits on the console

// The mode the console's library picks for Layout 0: linear for 1D, the XOR depth and render modes for targets, else
// 4 KiB standard (VIctory's rule, from the library's older optimal-mode function).
inline uint32_t suggested_swizzle(const D3D12_RESOURCE_DESC& d) {
    if (d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D) return kLinear;
    if (static_cast<uint32_t>(d.Flags) & kDescDepthStencil) return kSw64kbZX;
    if (static_cast<uint32_t>(d.Flags) & kDescRenderTarget) return kSw64kbRX;
    return kSw4kbS;
}

// The console layout of a texture description, or nothing (a buffer, an unknown format or Layout value).
inline std::optional<Texture> console_texture(const D3D12_RESOURCE_DESC& d) {
    TextureDesc t;
    switch (d.Dimension) {
    case D3D12_RESOURCE_DIMENSION_TEXTURE1D: t.dimension = Dimension::tex1d; break;
    case D3D12_RESOURCE_DIMENSION_TEXTURE2D: t.dimension = Dimension::tex2d; break;
    case D3D12_RESOURCE_DIMENSION_TEXTURE3D: t.dimension = Dimension::tex3d; break;
    default: return std::nullopt;
    }
    const uint32_t layout = static_cast<uint32_t>(d.Layout);
    if (layout == 0) t.swizzle = suggested_swizzle(d);
    else if (layout == kDescRowMajor) t.swizzle = kLinear;
    else if ((layout & ~0xffu) == kDescSwizzled) t.swizzle = layout & 0xff;
    else return std::nullopt;
    if (d.Width > 0xffffffffull) return std::nullopt;
    t.width = static_cast<uint32_t>(d.Width);
    t.height = d.Height;
    t.depth_or_array = d.DepthOrArraySize;
    t.mips = d.MipLevels;
    t.format = static_cast<uint32_t>(d.Format);
    t.samples = std::max<uint32_t>(d.SampleDesc.Count, 1);
    return Texture::make(t);
}

// The console's allocation answer for one texture (VIctory's footprint): its layout's size rounded up to the larger of
// the requested and the layout's own alignment, at that alignment (4 KiB for the 4 KiB modes, where a PC device says
// 64 KiB). False when the description has no console layout or the alignment is not a power of two.
inline bool console_allocation(const D3D12_RESOURCE_DESC& d, uint64_t& size, uint64_t& alignment) {
    const auto t = console_texture(d);
    if (!t || !t->size) return false;
    const uint64_t a = std::max<uint64_t>(d.Alignment, t->alignment);
    if (!a || (a & (a - 1))) return false;
    if (t->size > UINT64_MAX - (a - 1)) return false;
    size = (t->size + a - 1) / a * a;
    alignment = a;
    return true;
}

// GetResourceAllocationInfo for a list of sampled textures (no render-target or depth flag: those keep the device
// proxy's own rule, d3d12_xs_alloc_size.h): each placed after the previous at its own alignment, the total at the
// largest alignment (VIctory's place). False when any description is something else or has no console layout.
inline bool console_texture_placement(const D3D12_RESOURCE_DESC* descs, uint32_t count, uint64_t& size,
                                      uint64_t& alignment) {
    if (!descs || !count) return false;
    uint64_t total = 0, largest = 0;
    for (uint32_t i = 0; i < count; ++i) {
        if (static_cast<uint32_t>(descs[i].Flags) & (kDescRenderTarget | kDescDepthStencil)) return false;
        uint64_t one = 0, a = 0;
        if (!console_allocation(descs[i], one, a)) return false;
        const uint64_t offset = (total + a - 1) / a * a;
        if (offset < total || offset > UINT64_MAX - one) return false;
        total = offset + one;
        largest = std::max(largest, a);
    }
    size = total;
    alignment = largest;
    return true;
}

}  // namespace leonida::console_layout
