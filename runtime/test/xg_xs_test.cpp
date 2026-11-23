// Checks the registered xg_xs.dll shim against the call shapes measured in the game executable (runtime/shims/xg_xs.cpp).
//
// Both exports are resolved through rt::shims::find, so this is the table the runtime installs. The checks: both
// return E_NOTIMPL for a description they cannot handle, XGSuggestSwizzleMode leaves the caller's field alone,
// XGCreateTextureComputer still hands out a computer (two measured callers use it without checking the HRESULT), slots
// 2 and 8 (and slot 10 on a description it cannot tile) return without writing what their arguments point to, slot 10
// answers the PC tiling of a description it can tile (GetResourceTiling, compared with src/resource_tiling.h), and a
// description it can lay out linearly gets S_OK, slot 2's whole layout structure and slots 8 and 9's linear copies.
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

    // A description it can lay out linearly: S_OK, and slot 2 fills the whole 0x1B40-byte layout structure.
    D3D12_RESOURCE_DESC rgba = r8;
    rgba.Width = 8;
    rgba.Height = 4;
    rgba.MipLevels = 2;
    rgba.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    void* rgba_computer = nullptr;
    check(create(&rgba, &rgba_computer) == S_OK && rgba_computer, "a description it can lay out gives S_OK");
    void* const* sv = rgba_computer ? *static_cast<void* const* const*>(rgba_computer) : vtable;
    static uint8_t big[0x1C00];
    std::memset(big, 0xcd, sizeof(big));
    check(reinterpret_cast<Slot2Fn>(sv[2])(rgba_computer, big) == S_OK, "slot 2 answers the linear layout");
    auto u64 = [&](size_t at) { uint64_t v = 0; std::memcpy(&v, big + at, 8); return v; };
    auto u32 = [&](size_t at) { uint32_t v = 0; std::memcpy(&v, big + at, 4); return v; };
    constexpr size_t kMip0 = 0x18 + 0x38, kMip1 = kMip0 + 0x70;
    check(u64(kMip0) == 128 && u64(kMip0 + 0x10) == 128 && u32(kMip0 + 0x1C) == 32 && u64(kMip1) == 32 &&
              u32(kMip1 + 0x1C) == 16,
          "mip 0 is 8 x 4 RGBA8 (32-byte rows, 128 bytes), mip 1 4 x 2 (16-byte rows, 32 bytes) at the measured offsets");
    check(u64(kMip0 + 0x08) == 0 && u64(kMip1 + 0x08) == 128, "mip 1 starts after mip 0");
    check(u32(kMip0 + 0x18) == 8 && u32(kMip0 + 0x38) == 8 && u32(kMip0 + 0x3C) == 4 && u32(kMip1 + 0x38) == 4 &&
              u32(kMip1 + 0x3C) == 2 && u32(kMip0 + 0x40) == 1 && u32(kMip0 + 0x44) == 1 && u32(kMip0 + 0x48) == 0,
          "pitch in texels, sizes in texels, array size, one sample, linear mode");
    check(u64(0x00) == 160 && u64(0x08) == 0x1000 && u32(0x10) == 2 && u32(0x14) == 1 && u32(0x1B38) == 3,
          "the total size (both mips), the alignment, the mip and plane counts and the dimension");
    check(u32(0x18) == 1 && u64(0x18 + 0x08) == 160 && u64(0x18 + 0x10) == 160 && u64(0x18 + 0x18) == 0 &&
              u32(0x18 + 0x28) == 4 && u32(0x18 + 0x2C) == 2 && u32(0x18 + 0x30) == 0,
          "plane 0: data, its size, array-slice size, offset, element bytes, two standard mips and no tail");
    check(all_bytes(big + kMip0 + 2 * 0x70, 0x1B38 - (kMip0 + 2 * 0x70), 0) && all_bytes(big + 0x1B3C, 4, 0),
          "the rest of the structure (other mips, planes 1-3) is zeroed, not left as the caller's stack held it");
    check(all_bytes(big + 0x1B40, sizeof(big) - 0x1B40, 0xcd), "nothing past the 0x1B40-byte structure is written");

    // A 2-slice array: the plane holds both slices, each slice's mips at the same offsets.
    D3D12_RESOURCE_DESC arr = rgba;
    arr.DepthOrArraySize = 2;
    void* arr_computer = nullptr;
    check(create(&arr, &arr_computer) == S_OK && arr_computer, "a 2-slice array can be laid out");
    std::memset(big, 0xcd, sizeof(big));
    check(arr_computer && reinterpret_cast<Slot2Fn>((*static_cast<void* const* const*>(arr_computer))[2])(arr_computer, big) == S_OK &&
              u64(0x00) == 320 && u64(0x18 + 0x08) == 320 && u64(0x18 + 0x10) == 160 && u32(kMip0 + 0x40) == 2,
          "the array's total is two slices of 160 bytes and the mips report the array size");

    // Slot 8 writes mip 1 from a wider source into the texture memory, after mip 0; slot 9 reads it back.
    uint8_t memory[160];
    std::memset(memory, 0xcd, sizeof(memory));
    uint8_t source[48];
    for (int i = 0; i < 48; ++i) source[i] = static_cast<uint8_t>(i);
    check(reinterpret_cast<Slot8Fn>(sv[8])(rgba_computer, memory, 0, 1, source, 24, 48) == S_OK,
          "slot 8 copies a subresource in");
    check(all_bytes(memory, 128, 0xcd) && memory[128] == 0 && memory[143] == 15 && memory[144] == 24 &&
              memory[159] == 39,
          "mip 1's two rows land tightly after mip 0, read at the caller's row pitch");
    check(reinterpret_cast<Slot8Fn>(sv[8])(rgba_computer, memory, 0, 2, source, 24, 48) == kNotImpl &&
              reinterpret_cast<Slot8Fn>(sv[8])(rgba_computer, memory, 1, 0, source, 24, 48) == kNotImpl,
          "slot 8 refuses a subresource past the description or another plane");

    // Slot 9 copies mip 1 (the bytes after mip 0) into a wider destination with the caller's row pitch.
    using Slot9Fn = HRESULT(__stdcall*)(void*, void*, UINT, UINT, const void*, const void*, UINT, UINT);
    uint8_t texels[160];
    for (int i = 0; i < 160; ++i) texels[i] = static_cast<uint8_t>(i);
    uint8_t out[64];
    std::memset(out, 0xcd, sizeof(out));
    check(reinterpret_cast<Slot9Fn>(sv[9])(rgba_computer, out, 0, 1, nullptr, texels, 24, 48) == S_OK,
          "slot 9 copies a subresource out");
    check(out[0] == 128 && out[15] == 143 && out[16] == 0xcd && out[24] == 144 && out[39] == 159 && out[40] == 0xcd,
          "mip 1's two rows land at the caller's row pitch, nothing between them");
    check(reinterpret_cast<Slot9Fn>(sv[9])(rgba_computer, out, 0, 2, nullptr, texels, 24, 48) == kNotImpl,
          "slot 9 refuses a subresource past the description");

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
