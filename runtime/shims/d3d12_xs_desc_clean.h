// The PC form of a resource description the retail title builds for the console, for the Xbox-ABI device proxy.
//
// The title's D3D12_RESOURCE_DESC carries console values the PC device cannot take: the layout word is 0x101 (tiled)
// or 0x1 (row-major buffer), and the flags have bits above the PC ones set (docs/retail_boot_chain.md, #6982). Slot 76
// replaces the layout word the same way before it forwards to CreateCommittedResource; slot 25 and the slot 127
// stand-in need the same cleaning. The title's own description is never written: a caller gets a copy.
#pragma once

#include <windows.h>
#include <d3d12.h>

#include <cstdint>

namespace rt::d3d12_xs {

// The D3D12_RESOURCE_FLAGS bits the PC runtime defines; every other bit is a console one.
constexpr std::uint32_t kPcResourceFlagsMask =
    D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL |
    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE |
    D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER | D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS |
    D3D12_RESOURCE_FLAG_VIDEO_DECODE_REFERENCE_ONLY | D3D12_RESOURCE_FLAG_VIDEO_ENCODE_REFERENCE_ONLY |
    D3D12_RESOURCE_FLAG_RAYTRACING_ACCELERATION_STRUCTURE;

// Not constexpr: 0x101 is outside the unfixed enum's value range, which clang-cl rejects in a constant expression.
inline const auto kXboxTiledResourceLayout = static_cast<D3D12_TEXTURE_LAYOUT>(0x101);

// The console allows a depth-stencil texture to allow unordered access too (measured locally at a title's call sites:
// a 2D R32G8X24_TYPELESS texture with ALLOW_DEPTH_STENCIL | ALLOW_UNORDERED_ACCESS, sized through slot 25), but the PC
// runtime refuses that pair, and Vulkan has no depth format with storage support to make it with. Stand-in: such a
// texture is a depth-stencil texture without unordered access. Its unordered-access views write no descriptor (the PC
// layer refuses a UAV of a texture without the flag), so what a shader writes through them goes nowhere and their
// reads see whatever the descriptor held.
inline std::uint32_t without_depth_stencil_uav(D3D12_RESOURCE_DIMENSION dimension, std::uint32_t flags) {
    if (dimension != D3D12_RESOURCE_DIMENSION_BUFFER && (flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL))
        flags &= ~static_cast<std::uint32_t>(D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    return flags;
}

// The texture flag combinations the PC refuses (the Vulkan layer's committed_resource_refusal mirrors the host:
// RT+DS, DS+UAV, and DENY_SHADER_RESOURCE without DS), made acceptable. The console makes 3D textures with such a
// combination (three refusals in a retail boot past 124,000 D3D12 calls, then a null-resource crash): DS+UAV loses the
// UAV (without_depth_stencil_uav); DENY_SHADER_RESOURCE without DS is only a promise not to sample the resource, so
// dropping it loses nothing; RT+DS keeps the depth-stencil and loses the render target, as the depth-stencil texture
// is the one the PC knows how to size. Buffers are left alone: none of the three applies to them.
inline std::uint32_t without_refused_combinations(D3D12_RESOURCE_DIMENSION dimension, std::uint32_t flags) {
    if (dimension == D3D12_RESOURCE_DIMENSION_BUFFER) return flags;
    flags = without_depth_stencil_uav(dimension, flags);
    if (flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)
        flags &= ~static_cast<std::uint32_t>(D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    else
        flags &= ~static_cast<std::uint32_t>(D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE);
    return flags;
}

// The initial barrier layout of a texture whose flags without_refused_combinations changed: a UNORDERED_ACCESS (3)
// layout without the UAV flag, or a RENDER_TARGET (2) one without the RT flag, is refused by the PC, so the texture
// starts in COMMON (0) instead. Measured on the retail boot: slot 133 places 2D R16_TYPELESS textures with DS+UAV in the
// UNORDERED_ACCESS layout. Every other layout goes on unchanged.
inline std::uint32_t pc_initial_layout(std::uint32_t pc_flags, std::uint32_t layout) {
    if (layout == 3 && !(pc_flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)) return 0;
    if (layout == 2 && !(pc_flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET)) return 0;
    return layout;
}

// The same for a legacy initial state (slot 76): UNORDERED_ACCESS or RENDER_TARGET without the flag becomes COMMON.
inline std::uint32_t pc_initial_state(std::uint32_t pc_flags, std::uint32_t state) {
    if (state == D3D12_RESOURCE_STATE_UNORDERED_ACCESS && !(pc_flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS))
        return D3D12_RESOURCE_STATE_COMMON;
    if (state == D3D12_RESOURCE_STATE_RENDER_TARGET && !(pc_flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET))
        return D3D12_RESOURCE_STATE_COMMON;
    return state;
}

// The PC flags of a console description: the PC bits only, in a combination the PC accepts.
inline std::uint32_t pc_resource_flags(D3D12_RESOURCE_DIMENSION dimension, std::uint32_t xbox_flags) {
    return without_refused_combinations(dimension, xbox_flags & kPcResourceFlagsMask);
}

// Strip Xbox-only flags and translate the Xbox tiled layout. Preserve all other layouts so the host can validate them.
inline D3D12_RESOURCE_DESC pc_resource_desc(const D3D12_RESOURCE_DESC& xbox) {
    D3D12_RESOURCE_DESC pc = xbox;
    if (pc.Layout == kXboxTiledResourceLayout) {
        pc.Layout = pc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER ? D3D12_TEXTURE_LAYOUT_ROW_MAJOR
                                                                   : D3D12_TEXTURE_LAYOUT_UNKNOWN;
    }
    pc.Flags = static_cast<D3D12_RESOURCE_FLAGS>(pc_resource_flags(xbox.Dimension, static_cast<std::uint32_t>(xbox.Flags)));
    return pc;
}

}  // namespace rt::d3d12_xs
