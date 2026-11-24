// Checks the registered xg_xs.dll shim (runtime/shims/xg_xs.cpp) and the console layouts behind it
// (runtime/src/console_texture_layout.h).
//
// Both exports are resolved through rt::shims::find, so this is the table the runtime installs. Expected numbers:
// the suggestion, layout, copy and tiling cases are the ones VIctory's xg.rs and texture.rs tests check (VIctory,
// github.com/Marvisak/VIctory, GPL-3.0; addrlib-derived); the sizes of the three BC7 chains are the ones VIctory's
// xg module reported for the same descriptions in the oracle run (S\oracle\capture_102517, "texture computer ...
// size="), which this repository does not ship.
#include <windows.h>
#include <d3d12.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "../src/console_texture_layout.h"
#include "shims.h"

namespace {

using CreateFn = HRESULT(__stdcall*)(const void*, void**);
using SuggestFn = HRESULT(__stdcall*)(const void*, uint32_t, uint32_t*);
using Slot2Fn = HRESULT(__stdcall*)(void*, void*);
using Slot6Fn = uint64_t(__stdcall*)(void*, uint32_t, uint32_t, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t*);
using Slot8Fn = HRESULT(__stdcall*)(void*, void*, uint32_t, uint32_t, const void*, uint32_t, uint32_t);
using Slot9Fn = HRESULT(__stdcall*)(void*, void*, uint32_t, uint32_t, uint32_t, const void*, uint32_t, uint32_t);
using Slot10Fn = HRESULT(__stdcall*)(void*, UINT*, D3D12_PACKED_MIP_INFO*, D3D12_TILE_SHAPE*, UINT*, UINT,
                                     D3D12_SUBRESOURCE_TILING*);

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

D3D12_RESOURCE_DESC desc(uint64_t w, uint32_t h, uint16_t mips, uint32_t format, uint32_t layout, uint16_t array = 1) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Width = w;
    d.Height = h;
    d.DepthOrArraySize = array;
    d.MipLevels = mips;
    d.Format = static_cast<DXGI_FORMAT>(format);
    d.SampleDesc.Count = 1;
    d.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(layout);
    return d;
}

template <typename Fn>
Fn slot(void* computer, size_t index) {
    return reinterpret_cast<Fn>((*reinterpret_cast<void* const* const*>(computer))[index]);
}

// Every element of every mip and slice lands on its own address inside the texture, and tiling then detiling the
// rows gives them back (texture.rs addresses_are_a_bijection_and_copies_round_trip, a subset).
bool round_trip(const leonida::console_layout::Desc& d) {
    namespace cl = leonida::console_layout;
    auto l = cl::Layout::make(d);
    if (!l) return false;
    std::vector<uint8_t> seen(static_cast<size_t>(l->size), 0), tiled(static_cast<size_t>(l->size), 0);
    const size_t n = d.element_bytes;
    const uint32_t slices = d.dimension == cl::Dimension::tex3d ? 1 : d.depth;
    for (uint32_t slice = 0; slice < slices; ++slice)
        for (uint32_t mip = 0; mip < d.mips; ++mip) {
            const cl::Mip& m = l->mips[mip];
            const size_t active = size_t{m.width} * n, row = active + 13, plane = row * m.height + 11;
            std::vector<uint8_t> linear(plane * m.depth);
            for (size_t i = 0; i < linear.size(); ++i) linear[i] = static_cast<uint8_t>(i * 7 + mip * 13 + slice);
            std::vector<cl::Run> runs;
            uint32_t first = 0;
            if (cl::plan_copy(*l, tiled.size(), mip, slice, linear.size(), row, plane, first, runs) != cl::Error::none)
                return false;
            for (uint32_t z = 0; z < m.depth; ++z)
                for (uint32_t y = 0; y < m.height; ++y) {
                    const uint64_t base = cl::row_base(*l, m, first, z, y);
                    for (const cl::Run& r : runs)
                        std::memcpy(&tiled[base + r.tiled], &linear[z * plane + y * row + r.linear], r.length);
                    for (uint32_t x = 0; x < m.width; ++x) {
                        uint64_t at = 0;
                        if (l->offset(mip, x, y, slice + z, at) != cl::Error::none) return false;
                        if (at + n > seen.size() || at % n || seen[at]) return false;
                        if (std::memcmp(&tiled[at], &linear[z * plane + y * row + x * n], n)) return false;
                        seen[at] = 1;
                    }
                }
        }
    return true;
}

}  // namespace

int main() {
    namespace cl = leonida::console_layout;
    auto create = reinterpret_cast<CreateFn>(rt::shims::find("xg_xs.dll", "XGCreateTextureComputer"));
    auto suggest = reinterpret_cast<SuggestFn>(rt::shims::find("xg_xs.dll", "XGSuggestSwizzleMode"));
    check(create != nullptr, "XGCreateTextureComputer is registered");
    check(suggest != nullptr, "XGSuggestSwizzleMode is registered");
    check(rt::shims::owns_dll("xg_xs.dll"), "the registry owns xg_xs.dll");
    if (!create || !suggest) return 1;

    // Suggestions follow the use: 4 KiB standard, render targets 64 KiB R_X, depth 64 KiB Z_X.
    D3D12_RESOURCE_DESC d = desc(256, 256, 1, 98, 0);
    uint32_t mode = 0;
    check(suggest(&d, 0, &mode) == S_OK && mode == 0x105, "XGSuggestSwizzleMode: 0x105 for a plain texture");
    d.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    check(suggest(&d, 0, &mode) == S_OK && mode == 0x11b, "XGSuggestSwizzleMode: 0x11b for a render target");
    d.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    check(suggest(&d, 0, &mode) == S_OK && mode == 0x118, "XGSuggestSwizzleMode: 0x118 for depth");

    // A description without a layout: E_INVALIDARG, but *out still holds a computer whose methods fail.
    check(create(&d, nullptr) == E_POINTER, "XGCreateTextureComputer(desc, nullptr) is E_POINTER");
    void* computer = nullptr;
    D3D12_RESOURCE_DESC bad = desc(256, 256, 1, 0, 0x10a);
    check(create(&bad, &computer) == E_INVALIDARG && computer, "unknown format: E_INVALIDARG with a computer");
    bad = desc(256, 256, 1, 28, 0x103);
    check(create(&bad, &computer) == E_INVALIDARG && computer, "a mode GFX10 lacks: E_INVALIDARG");
    std::vector<uint8_t> scratch(0x1B40, 0xAB);
    check(slot<Slot2Fn>(computer, 2)(computer, scratch.data()) == E_INVALIDARG && scratch[0] == 0xAB,
          "slot 2 of a failed computer writes nothing");

    // 256x256 BC7, full chain, 64 KiB display mode: one tail block holding mips 1-8, then mip 0.
    d = desc(256, 256, 0, 98, 0x10a);
    check(create(&d, &computer) == S_OK, "256x256 BC7 0x10a: S_OK");
    std::vector<uint8_t> layout(0x1B40, 0xCD);
    check(slot<Slot2Fn>(computer, 2)(computer, layout.data()) == S_OK, "slot 2 answers the layout");
    auto u64 = [&](size_t at) { uint64_t v; std::memcpy(&v, &layout[at], 8); return v; };
    auto u32 = [&](size_t at) { uint32_t v; std::memcpy(&v, &layout[at], 4); return v; };
    check(u64(0) == 131072 && u64(8) == 65536 && u32(0x10) == 9 && u32(0x14) == 1,
          "layout: 128 KiB, 64 KiB aligned, 9 mips, 1 plane");
    const size_t plane = 0x18, mips = plane + 0x38;
    check(u32(plane) == 1 && u32(plane + 0x28) == 16 && u32(plane + 0x2C) == 1 && u32(plane + 0x30) == 8,
          "plane: data, 16-byte elements, 1 standard mip, 8 in the tail");
    check(u64(mips + 0x08) == 65536 && u32(mips + 0x1C) == 1024, "mip 0 at 64 KiB, 1024-byte rows");
    check(u64(mips + 0x70 + 0x08) == 32768 && u32(mips + 0x70 + 0x64) == 1, "mip 1 in the tail at 32 KiB");
    check(u32(0x1B38) == 3, "dimension at +0x1B38");

    std::vector<uint8_t> tiled(131072, 0);
    bool copies = true;
    for (uint32_t mip = 0; mip < 9; ++mip) {
        const uint32_t elements = u32(mips + 0x70 * mip + 0x30), rows_n = u32(mips + 0x70 * mip + 0x34);
        const uint32_t row = elements * 16;
        std::vector<uint8_t> rows(size_t{row} * rows_n), back(rows.size(), 0);
        for (size_t i = 0; i < rows.size(); ++i) rows[i] = static_cast<uint8_t>(i ^ mip);
        copies &= slot<Slot8Fn>(computer, 8)(computer, tiled.data(), 0, mip, rows.data(), row, 0) == S_OK;
        copies &= slot<Slot9Fn>(computer, 9)(computer, back.data(), 0, mip, 0, tiled.data(), row, 0) == S_OK;
        copies &= back == rows;
    }
    check(copies, "slots 8 and 9: every mip copies in and back out unchanged");

    UINT tiles = 0, n = 4;
    D3D12_PACKED_MIP_INFO packed{};
    D3D12_TILE_SHAPE shape{};
    D3D12_SUBRESOURCE_TILING subs[4]{};
    check(slot<Slot10Fn>(computer, 10)(computer, &tiles, &packed, &shape, &n, 0, subs) == S_OK, "slot 10: S_OK");
    check(tiles == 2 && packed.NumStandardMips == 1 && packed.NumPackedMips == 8 && packed.NumTilesForPackedMips == 1,
          "slot 10: 2 tiles, 1 standard mip, 8 packed in 1 tile");
    check(shape.WidthInTexels == 256 && shape.HeightInTexels == 256 && shape.DepthInTexels == 1 && n == 1 &&
              subs[0].StartTileIndexInOverallResource == 1,
          "slot 10: 256x256 tile shape, mip 0 starts at tile 1");

    d = desc(256, 256, 0, 98, 0x105);
    check(create(&d, &computer) == S_OK, "256x256 BC7 0x105: S_OK");
    check(slot<Slot10Fn>(computer, 10)(computer, &tiles, nullptr, nullptr, nullptr, 0, nullptr) == E_FAIL,
          "slot 10: a 4 KiB mode has no 64 KiB tiling (E_FAIL)");

    // Slot 6: element offsets of a linear 128x4 R16G16B16A16 (rows of 1 KiB), 0 outside.
    d = desc(128, 4, 1, 11, 0x100);
    check(create(&d, &computer) == S_OK, "128x4 linear: S_OK");
    uint32_t bit = 7;
    check(slot<Slot6Fn>(computer, 6)(computer, 0, 0, 3, 1, 0, 0, &bit) == (128 + 3) * 8 && bit == 0,
          "slot 6: (3, 1) is 131 elements in");
    check(slot<Slot6Fn>(computer, 6)(computer, 0, 0, 128, 0, 0, 0, nullptr) == 0, "slot 6: outside is 0");

    // Sizes VIctory's library answered for full BC7 chains in 0x10a (oracle run).
    struct { uint32_t w, h; uint64_t size; } chains[] = {{1024, 1024, 1441792}, {2048, 2048, 5636096},
                                                         {4096, 4096, 22413312}};
    for (const auto& c : chains) {
        d = desc(c.w, c.h, 0, 98, 0x10a);
        create(&d, &computer);
        slot<Slot2Fn>(computer, 2)(computer, layout.data());
        char what[96];
        std::snprintf(what, sizeof(what), "%ux%u BC7 full chain 0x10a is %llu bytes", c.w, c.h,
                      static_cast<unsigned long long>(c.size));
        check(u64(0) == c.size, what);
    }

    // Layout-level round trips over the addressable modes (texture.rs cases).
    auto ld = [](uint32_t w, uint32_t h, uint32_t m, uint32_t bpe, uint32_t s) {
        cl::Desc x;
        x.width = w, x.height = h, x.mips = m, x.element_bytes = bpe, x.swizzle = s;
        return x;
    };
    cl::Desc cases[] = {ld(64, 64, 7, 16, cl::kSw64kbD), ld(100, 60, 7, 4, cl::kSw64kbD), ld(250, 175, 8, 8, cl::kSw4kbS),
                        ld(33, 17, 6, 1, cl::kSw4kbD),   ld(40, 9, 3, 2, cl::kSw64kbS),   ld(30, 20, 5, 4, cl::kLinear),
                        ld(83, 31, 6, 1, cl::kSw256bS),  ld(109, 71, 6, 16, cl::kSw256bD), ld(70, 70, 4, 16, cl::kSw4kbS),
                        ld(64, 32, 7, 4, cl::kSw4kbS)};
    cases[7].block = 4;
    cases[8].depth = 3;
    cases[9].dimension = cl::Dimension::tex3d;
    cases[9].depth = 16;
    bool all = true;
    for (const cl::Desc& c : cases) all &= round_trip(c);
    check(all, "addresses are a bijection and copies round-trip (10 layouts)");

    // Hand-checked addresses: 64 KiB D, 4-byte elements; 4 KiB S, 16-byte elements is 16x16 with y first.
    auto l = cl::Layout::make(ld(128, 128, 1, 4, cl::kSw64kbD));
    uint64_t at = 0;
    bool hand = l && l->offset(0, 1, 0, 0, at) == cl::Error::none && at == 4 && l->offset(0, 4, 0, 0, at) == cl::Error::none &&
                at == 64 && l->offset(0, 0, 4, 0, at) == cl::Error::none && at == 128 &&
                l->offset(0, 64, 64, 0, at) == cl::Error::none && at == ((1u << 15) | (1u << 14));
    l = cl::Layout::make(ld(32, 32, 1, 16, cl::kSw4kbS));
    hand = hand && l && l->block[0] == 16 && l->block[1] == 16 && l->offset(0, 0, 1, 0, at) == cl::Error::none &&
           at == 16 && l->offset(0, 16, 0, 0, at) == cl::Error::none && at == 4096 &&
           l->offset(0, 0, 16, 0, at) == cl::Error::none && at == 8192;
    check(hand, "hand-checked element addresses");
    l = cl::Layout::make(ld(64, 64, 9, 16, cl::kSw4kbS));
    check(l && l->first_tail_mip == 3 && l->mips[2].offset == 4096 && l->mips[1].offset == 8192 &&
              l->mips[0].offset == 8192 + 16384,
          "4 KiB blocks hold fewer tail mips");

    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
