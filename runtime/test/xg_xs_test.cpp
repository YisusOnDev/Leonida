// Checks the registered xg_xs.dll shim against the call shapes measured in the game executable (runtime/shims/xg_xs.cpp).
//
// Both exports are resolved through rt::shims::find, so this is the table the runtime installs. The checks: both
// return E_NOTIMPL, XGSuggestSwizzleMode leaves the caller's field alone, XGCreateTextureComputer still hands out a
// computer (two measured callers use it without checking the HRESULT), slots 2 and 8 (and slot 10 on a description it
// cannot tile) return without writing what their arguments point to, and slot 10 answers the PC tiling of a
// description it can tile (GetResourceTiling, compared with src/resource_tiling.h).
#include <windows.h>
#include <d3d12.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

#include "../src/resource_tiling.h"
#include "shims.h"

namespace {

using CreateFn = HRESULT(__stdcall*)(const void*, void**);
using SuggestFn = HRESULT(__stdcall*)(const void*, uint32_t, uint32_t*);
using ReleaseFn = ULONG(__stdcall*)(void*);
using Slot2Fn = HRESULT(__stdcall*)(void*, void*);
using Slot8Fn = HRESULT(__stdcall*)(void*, void*, uint32_t, uint32_t, const void*, uint32_t, uint32_t);
using Slot10Fn = HRESULT(__stdcall*)(void*, uint32_t*, void*, void*, uint32_t*, uint32_t, void*);
constexpr HRESULT kNotImpl = static_cast<HRESULT>(rt::shims::kENotImpl);

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

bool all_bytes(const uint8_t* p, size_t n, uint8_t value) {
    for (size_t i = 0; i < n; ++i)
        if (p[i] != value) return false;
    return true;
}

}  // namespace

int main() {
    auto create = reinterpret_cast<CreateFn>(rt::shims::find("xg_xs.dll", "XGCreateTextureComputer"));
    auto suggest = reinterpret_cast<SuggestFn>(rt::shims::find("xg_xs.dll", "XGSuggestSwizzleMode"));
    check(create != nullptr, "XGCreateTextureComputer is registered");
    check(suggest != nullptr, "XGSuggestSwizzleMode is registered");
    check(rt::shims::owns_dll("xg_xs.dll"), "the registry owns xg_xs.dll");
    if (!create || !suggest) return 1;

    uint8_t desc[0x38] = {};
    uint32_t mode = 0x1234;
    check(suggest(desc, 0, &mode) == kNotImpl, "XGSuggestSwizzleMode returns E_NOTIMPL");
    check(mode == 0x1234, "XGSuggestSwizzleMode leaves the mode field as the caller set it");

    check(create(desc, nullptr) == E_POINTER, "XGCreateTextureComputer(desc, nullptr) is E_POINTER");
    void* computer = nullptr;
    check(create(desc, &computer) == kNotImpl, "XGCreateTextureComputer returns E_NOTIMPL");
    check(computer != nullptr, "XGCreateTextureComputer still stores a computer for the callers that do not check");
    if (!computer) return 1;

    void* const* vtable = *static_cast<void* const* const*>(computer);
    uint8_t layout[0x800];
    std::memset(layout, 0xcd, sizeof(layout));
    check(reinterpret_cast<Slot2Fn>(vtable[2])(computer, layout) == kNotImpl, "slot 2 returns E_NOTIMPL");
    check(all_bytes(layout, sizeof(layout), 0xcd), "slot 2 writes nothing to its output");

    uint8_t dst[64];
    std::memset(dst, 0xcd, sizeof(dst));
    const uint8_t src[64] = {};
    check(reinterpret_cast<Slot8Fn>(vtable[8])(computer, dst, 0, 0, src, 16, 64) == kNotImpl,
          "slot 8 returns E_NOTIMPL");
    check(all_bytes(dst, sizeof(dst), 0xcd), "slot 8 writes nothing to its destination");

    uint32_t count = 0, inout = 0x10;
    uint8_t a[12] = {}, b[12] = {}, list[0xc0] = {};
    check(reinterpret_cast<Slot10Fn>(vtable[10])(computer, &count, a, b, &inout, 0, list) == kNotImpl,
          "slot 10 returns E_NOTIMPL");
    check(count == 0 && inout == 0x10 && all_bytes(a, 12, 0) && all_bytes(b, 12, 0) &&
              all_bytes(list, sizeof(list), 0),
          "slot 10 leaves the caller's zero-filled outputs as they were");
    check(reinterpret_cast<Slot2Fn>(vtable[31])(computer, layout) == kNotImpl, "the last slot (31) is traced too");

    // Slot 10 on a description it can tile: the first one measured at the game's call sites, 256 x 128 R8_UNORM, one
    // mip, 4 KiB alignment, with the console layout word and console flag bits set. An 8-bit texel's 64 KiB tile is
    // 256 x 256, so the texture is one tile and its one mip is standard (wider than half a tile).
    D3D12_RESOURCE_DESC r8{};
    r8.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    r8.Alignment = 0x1000;
    r8.Width = 256;
    r8.Height = 128;
    r8.DepthOrArraySize = 1;
    r8.MipLevels = 1;
    r8.Format = DXGI_FORMAT_R8_UNORM;
    r8.SampleDesc.Count = 1;
    r8.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x105);
    r8.Flags = static_cast<D3D12_RESOURCE_FLAGS>(0x300000);
    uint8_t console_desc[0x58];
    std::memset(console_desc, 0x5a, sizeof(console_desc));  // the console's own fields after the PC description
    std::memcpy(console_desc, &r8, sizeof(r8));
    void* r8_computer = nullptr;
    create(console_desc, &r8_computer);
    void* const* r8_vtable = r8_computer ? *static_cast<void* const* const*>(r8_computer) : nullptr;
    UINT total = 0, n = 16;
    D3D12_PACKED_MIP_INFO packed{};
    D3D12_TILE_SHAPE shape{};
    D3D12_SUBRESOURCE_TILING tilings[16] = {};
    check(r8_vtable && reinterpret_cast<Slot10Fn>(r8_vtable[10])(r8_computer, &total, &packed, &shape, &n, 0, tilings) == S_OK,
          "slot 10 tiles a 256 x 128 R8_UNORM description");
    check(total == 1 && packed.NumStandardMips == 1 && packed.NumPackedMips == 0 && packed.NumTilesForPackedMips == 0,
          "one tile, one standard mip, nothing packed");
    check(shape.WidthInTexels == 256 && shape.HeightInTexels == 256 && shape.DepthInTexels == 1, "the 8-bit tile shape");
    check(n == 1 && tilings[0].WidthInTiles == 1 && tilings[0].HeightInTiles == 1 && tilings[0].DepthInTiles == 1 &&
              tilings[0].StartTileIndexInOverallResource == 0,
          "one subresource tiling, starting at tile 0");

    // A full mip chain (MipLevels 0), from the second subresource on: the same numbers as src/resource_tiling.h gives.
    D3D12_RESOURCE_DESC chain = r8;
    chain.Width = 256;
    chain.Height = 256;
    chain.MipLevels = 0;
    chain.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    void* chain_computer = nullptr;
    create(&chain, &chain_computer);
    leonida::TilingDesc expect_desc{};
    expect_desc.width = 256;
    expect_desc.height = 256;
    expect_desc.mips = 9;
    expect_desc.element_bytes = 4;
    leonida::ResourceTiling expect;
    check(leonida::resource_tiling(expect_desc, expect), "the reference tiling of 256 x 256 RGBA8 with 9 mips");
    n = 16;
    std::memset(tilings, 0, sizeof(tilings));
    check(reinterpret_cast<Slot10Fn>(r8_vtable[10])(chain_computer, &total, &packed, &shape, &n, 2, tilings) == S_OK &&
              total == expect.total_tiles && packed.NumStandardMips == expect.standard_mips &&
              packed.NumPackedMips == expect.packed_mips && packed.NumTilesForPackedMips == expect.packed_tiles &&
              packed.StartTileIndexInOverallResource == expect.packed_start && shape.WidthInTexels == expect.shape.width,
          "a full chain (MipLevels 0) answers the reference tiling");
    check(n == 7 && tilings[0].WidthInTiles == expect.subresources[2].width &&
              tilings[0].StartTileIndexInOverallResource == expect.subresources[2].start,
          "the subresource tilings start at the first one asked for");

    void* again = nullptr;
    create(console_desc, &again);
    check(again == r8_computer && chain_computer != r8_computer, "one computer per description, kept for the run");

    auto release = reinterpret_cast<ReleaseFn>(vtable[1]);
    release(computer);
    release(computer);
    check(reinterpret_cast<Slot2Fn>(vtable[2])(computer, layout) == kNotImpl,
          "the computer stays usable after more Releases than it was handed out");

    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
