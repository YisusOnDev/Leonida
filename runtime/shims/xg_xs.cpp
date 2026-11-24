// xg_xs.dll: the console's texture layout library. The game executable (1.0.2.1) imports two functions from it.
//
// Origin: the texture computer follows crates/victory-hle/src/modules/xg.rs of VIctory (github.com/Marvisak/VIctory,
// upstream 204bdcc, GPL-3.0-or-later): the slot assignments, the layout structure, the suggested swizzle modes and the
// 64 KiB tiling. The layouts themselves come from src/console_texture_layout.h (VIctory's addrlib-based texture.rs).
//
// Measured from the call sites in the executable, read statically (shapes and checks only):
//   HRESULT XGSuggestSwizzleMode(const desc*, 0, uint32_t* mode)
//     2 call sites, both in the game's helper that fills a texture description before XGCreateTextureComputer. The
//     out pointer is the description's Layout field; the console library writes 0x100 | swizzle mode there.
//   HRESULT XGCreateTextureComputer(const desc*, computer** out)
//     4+ call sites. Two test the HRESULT; two ignore it and call through *out at once, so *out is always set (to a
//     computer whose methods fail when the description has no layout). Measured slots: 1 Release, 2 the layout,
//     6 an element's byte offset, 8 a copy into a subresource, 9 a copy out of one, 10 the 64 KiB tiling.
//
// Why the console layouts and not PC ones: the title sizes its streaming reads from these answers (boot frontier5-3:
// the per-texture-class page counts at exe 0x149C3E1D0 are built from XGCreateTextureComputer + slot 10 at +0x344283),
// and the archives were packed for the console sizes; PC-sized (linear / PC 64 KiB tiling) answers made those reads
// shorter than the archive's chunks and the title's archive asserts (+0x4ffd868/879) fire. VIctory answers every call
// with the console layout and its title reaches the press-A screen.
//
// Description: a D3D12_RESOURCE_DESC (0x38 bytes) whose Layout field carries 0x100 | swizzle mode on the console
// (1 row major, 0 "pick one"). Computers are kept per distinct description for the run (Release counts nothing: one
// site stores the computer for later, and the title asks for the same descriptions again and again).
#include <windows.h>
#include <d3d12.h>
#include <intrin.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../src/console_texture_desc.h"
#include "shims.h"

namespace {

namespace cl = leonida::console_layout;

constexpr HRESULT kNotImpl = static_cast<HRESULT>(rt::shims::kENotImpl);
constexpr size_t kSlots = 32;
constexpr uint32_t kSwizzled = cl::kDescSwizzled;
constexpr uint32_t kAllowDepthStencil = cl::kDescDepthStencil;
constexpr uint32_t kTexture3d = 4;
constexpr size_t kLayoutMips = 15;

// Prints once per entry point; `slot` is the vtable index for a computer method, or -1 for an export.
void warn_once(std::atomic<bool>& warned, const char* what, int slot, const void* caller) {
    if (warned.exchange(true, std::memory_order_relaxed)) return;
    char name[64];
    if (slot < 0) std::snprintf(name, sizeof(name), "%s", what);
    else std::snprintf(name, sizeof(name), "%s%d", what, slot);
    std::fprintf(stderr, "[rt] xg_xs: %s: not implemented; E_NOTIMPL (caller %s, logged once)\n", name,
                 rt::shims::module_offset(reinterpret_cast<uint64_t>(caller)).c_str());
}

// The mode the console's library picks (src/console_texture_desc.h, shared with the device's GetResourceAllocationInfo).
uint32_t suggested(const D3D12_RESOURCE_DESC& d) { return cl::suggested_swizzle(d); }

struct Computer {
    const void* const* vtable;
    D3D12_RESOURCE_DESC desc;
    bool valid;
    uint32_t mips;
    cl::Texture texture;
};

std::optional<cl::Texture> texture_for(const D3D12_RESOURCE_DESC& d) {
    auto texture = cl::console_texture(d);
    if (!texture || texture->planes[0].layout.mips.size() > kLayoutMips) return std::nullopt;
    return texture;
}

// ---- slot 2: the layout structure (0x1B40 bytes) ------------------------------------------------------------------
// Field offsets as VIctory documents them; GTA VI reads mip +0x00, +0x08, +0x10, +0x1C, +0x50+0x70*m, +0x60, +0x6C.
#pragma pack(push, 1)
struct MipOut {
    uint64_t size, offset, slice_size;
    uint32_t pitch_texels, pitch_bytes, alignment;
    uint32_t padded[3];
    uint32_t elements[2];
    uint32_t texels[2];
    uint32_t depth_or_array, samples, swizzle;
    uint32_t pad0;
    uint64_t bank_rotation[2];
    uint32_t slice_depth, in_tail, tail_position;
    uint32_t pad1;
};
struct PlaneOut {
    uint32_t usage;  // 1 data, 7 depth, 8 stencil
    uint32_t pad0;
    uint64_t size, slice_size, offset, alignment;
    uint32_t element_bytes, standard_mips, tail_mips;
    uint32_t pad1;
    MipOut mips[kLayoutMips];
};
struct LayoutOut {
    uint64_t size, alignment;
    uint32_t mips, planes;
    PlaneOut plane[4];
    uint32_t dimension;
    uint32_t pad;
};
#pragma pack(pop)
static_assert(sizeof(MipOut) == 0x70 && offsetof(MipOut, swizzle) == 0x48 && offsetof(MipOut, slice_depth) == 0x60);
static_assert(sizeof(PlaneOut) == 0x6C8 && offsetof(PlaneOut, mips) == 0x38);
static_assert(sizeof(LayoutOut) == 0x1B40 && offsetof(LayoutOut, plane) == 0x18 && offsetof(LayoutOut, dimension) == 0x1B38);

// The mode a tail mip is reported in: the 256-byte one of its kind (3 for the render kind), the depth mode itself.
uint32_t tail_swizzle(uint32_t s) {
    switch (s) {
    case 2: case 6: case 10: case 18: case 22: case 26: return cl::kSw256bD;
    case 23: case 27: return 3;
    case 24: return s;
    default: return cl::kSw256bS;
    }
}

MipOut mip_out(const D3D12_RESOURCE_DESC& desc, const cl::Layout& l, size_t i) {
    const cl::Mip& m = l.mips[i];
    const uint64_t bpe = l.desc.element_bytes;
    const uint32_t block = l.desc.block;
    const bool three_d = static_cast<uint32_t>(desc.Dimension) == kTexture3d;
    const uint32_t array = std::max<uint32_t>(desc.DepthOrArraySize, 1);
    MipOut out{};
    out.pitch_texels = m.pitch * block;
    out.pitch_bytes = m.pitch * l.desc.element_bytes;
    out.padded[0] = m.pitch;
    out.padded[1] = m.rows;
    out.padded[2] = three_d ? m.depth_pitch : array;
    out.elements[0] = m.width;
    out.elements[1] = m.height;
    out.texels[0] = std::max<uint32_t>(static_cast<uint32_t>(desc.Width >> i), 1);
    out.texels[1] = std::max<uint32_t>(desc.Height >> i, 1);
    out.depth_or_array = three_d ? m.depth : array;
    out.samples = l.desc.samples;
    out.slice_depth = l.block[2];
    const uint64_t slice = uint64_t{m.pitch} * m.rows * bpe;
    const uint64_t depth = three_d ? m.depth_pitch : 1;
    if (!m.in_tail) {
        out.offset = m.offset;
        out.slice_size = slice * l.desc.samples;
        out.size = out.slice_size * depth;
        out.alignment = static_cast<uint32_t>(l.alignment);
        out.swizzle = l.desc.swizzle;
    } else {
        out.in_tail = 1;
        out.alignment = 256;
        out.swizzle = tail_swizzle(l.desc.swizzle);
        if (l.thick) {  // a 3D tail mip is placed by position, not by offset
            out.slice_size = l.alignment / l.block[2];
            out.tail_position = m.tail_x | m.tail_y << 11;
        } else {
            out.offset = m.tail_offset;
            out.slice_size = slice;
        }
        out.size = out.slice_size * depth;
    }
    return out;
}

ULONG __stdcall computer_add_ref(Computer*) { return 1; }
ULONG __stdcall computer_release(Computer*) { return 1; }

template <size_t Slot>
HRESULT __stdcall computer_slot(Computer*) {
    static std::atomic<bool> warned{false};
    warn_once(warned, "texture computer slot ", static_cast<int>(Slot), _ReturnAddress());
    return kNotImpl;
}

HRESULT __stdcall computer_get_resource_layout(Computer* c, LayoutOut* out) {
    if (!c || !out || !c->valid) return E_INVALIDARG;
    std::memset(out, 0, sizeof(*out));
    out->size = c->texture.size;
    out->alignment = c->texture.alignment;
    out->mips = c->mips;
    out->planes = static_cast<uint32_t>(c->texture.planes.size());
    out->dimension = static_cast<uint32_t>(c->desc.Dimension);
    // The library calls a depth plane "depth" only beside stencil or without mips.
    const bool depth = (static_cast<uint32_t>(c->desc.Flags) & kAllowDepthStencil) &&
                       (c->texture.planes.size() > 1 || c->mips == 1);
    for (size_t p = 0; p < c->texture.planes.size() && p < 4; ++p) {
        const cl::Plane& src = c->texture.planes[p];
        const cl::Layout& l = src.layout;
        PlaneOut& plane = out->plane[p];
        plane.usage = src.stencil ? 8 : depth ? 7 : 1;
        plane.size = l.size;
        plane.slice_size = l.slice_size;
        plane.offset = src.offset;
        plane.alignment = l.alignment;
        plane.element_bytes = l.desc.element_bytes;
        plane.standard_mips = l.first_tail_mip;
        plane.tail_mips = c->mips - l.first_tail_mip;
        for (size_t i = 0; i < l.mips.size() && i < kLayoutMips; ++i) plane.mips[i] = mip_out(c->desc, l, i);
    }
    return S_OK;
}

// ---- slots 8 and 9: copies --------------------------------------------------------------------------------------
// The row loop runs under __try (no objects with destructors here) so a bad pointer from the title is a failure, not
// a crash of the runtime.
bool guarded_copy(const cl::Layout* l, const cl::Mip* m, uint32_t first, const cl::Run* runs, size_t run_count,
                  uint8_t* tiled, uint8_t* linear, size_t row_pitch, size_t depth_pitch, bool into) {
    __try {
        for (uint32_t z = 0; z < m->depth; ++z)
            for (uint32_t y = 0; y < m->height; ++y) {
                uint8_t* const t = tiled + cl::row_base(*l, *m, first, z, y);
                uint8_t* const r = linear + z * depth_pitch + y * row_pitch;
                for (size_t i = 0; i < run_count; ++i) {
                    if (into) std::memcpy(t + runs[i].tiled, r + runs[i].linear, runs[i].length);
                    else std::memcpy(r + runs[i].linear, t + runs[i].tiled, runs[i].length);
                }
            }
        return true;
    } __except (GetExceptionCode() == EXCEPTION_ACCESS_VIOLATION ? EXCEPTION_EXECUTE_HANDLER
                                                                  : EXCEPTION_CONTINUE_SEARCH) {
        return false;
    }
}

HRESULT copy(Computer* c, uint8_t* tiled, uint32_t plane, uint32_t subresource, uint8_t* linear, uint32_t row,
             uint32_t depth, bool into) {
    if (!c || !c->valid || !tiled || !linear || plane >= c->texture.planes.size()) return E_INVALIDARG;
    const cl::Plane& p = c->texture.planes[plane];
    const cl::Layout& l = p.layout;
    const uint32_t mip = subresource % c->mips, slice = subresource / c->mips;
    if (mip >= l.mips.size()) return E_INVALIDARG;
    const cl::Mip& m = l.mips[mip];
    const size_t linear_len = size_t{m.depth - 1} * depth + size_t{m.height - 1} * row + size_t{m.width} * l.desc.element_bytes;
    std::vector<cl::Run> runs;
    uint32_t first = 0;
    const cl::Error e = cl::plan_copy(l, static_cast<size_t>(l.size), mip, slice, linear_len, row, depth, first, runs);
    if (e == cl::Error::unsupported) return kNotImpl;
    if (e != cl::Error::none) return E_INVALIDARG;
    return guarded_copy(&l, &m, first, runs.data(), runs.size(), tiled + p.offset, linear, row, depth, into) ? S_OK
                                                                                                           : E_POINTER;
}

// Slot 8: (tiled base, plane, subresource, linear, row pitch, depth pitch).
HRESULT __stdcall computer_copy_into(Computer* c, uint8_t* tiled, uint32_t plane, uint32_t subresource,
                                     const uint8_t* linear, uint32_t row, uint32_t depth) {
    return copy(c, tiled, plane, subresource, const_cast<uint8_t*>(linear), row, depth, true);
}

// Slot 9: (linear, plane, subresource, sample, tiled base, row pitch, depth pitch).
HRESULT __stdcall computer_copy_from(Computer* c, uint8_t* linear, uint32_t plane, uint32_t subresource,
                                     uint32_t /*sample*/, const uint8_t* tiled, uint32_t row, uint32_t depth) {
    return copy(c, const_cast<uint8_t*>(tiled), plane, subresource, linear, row, depth, false);
}

// ---- slot 10: 64 KiB tiling ---------------------------------------------------------------------------------------
// (total tiles, packed mips, tile shape in texels, in: room / out: count, first, subresource tilings), in 64 KiB tiles
// of plane 0, for the 2D 64 KiB modes only (others fail, as on the console). The mip tail is the packed part, tile 0
// of each slice; with a tail the console's library lists two slices at most and zeroes all but the first.
constexpr uint64_t kTile = 64 << 10;

HRESULT __stdcall computer_get_resource_tiling(Computer* c, UINT* total, D3D12_PACKED_MIP_INFO* packed,
                                               D3D12_TILE_SHAPE* shape, UINT* count, UINT first,
                                               D3D12_SUBRESOURCE_TILING* tilings) {
    if (!c || !c->valid) return E_INVALIDARG;
    const cl::Layout& l = c->texture.planes[0].layout;
    if (l.alignment != kTile) return E_FAIL;
    if (l.thick) return kNotImpl;
    const uint32_t standard = l.first_tail_mip;
    const bool tail = standard < c->mips;
    uint64_t tail_tiles = 0;
    if (standard == 0) tail_tiles = l.slice_size / kTile;
    else if (standard < c->mips) tail_tiles = l.mips[standard - 1].offset / kTile;
    if (total) *total = static_cast<UINT>(l.size / kTile);
    if (packed)
        *packed = D3D12_PACKED_MIP_INFO{static_cast<UINT8>(standard), static_cast<UINT8>(c->mips - standard),
                                        static_cast<UINT>(tail_tiles), 0};
    if (shape) *shape = D3D12_TILE_SHAPE{l.block[0] * l.desc.block, l.block[1] * l.desc.block, l.block[2]};
    if (count) {
        const uint32_t slices = tail ? std::min<uint32_t>(l.slices, 2) : l.slices;
        const uint64_t all = uint64_t{standard} * slices;
        const uint32_t wanted = static_cast<uint32_t>(std::min<uint64_t>(all > first ? all - first : 0, *count));
        for (uint32_t i = 0; i < wanted && tilings; ++i) {
            const uint32_t slice = (first + i) / standard, mip = (first + i) % standard;
            const cl::Mip& m = l.mips[mip];
            if (tail && slice > 0) tilings[i] = D3D12_SUBRESOURCE_TILING{0, 0, 0, 0};
            else
                tilings[i] = D3D12_SUBRESOURCE_TILING{
                    m.pitch / l.block[0], static_cast<UINT16>(m.rows / l.block[1]), 1,
                    static_cast<UINT>((uint64_t{slice} * l.slice_size + m.offset) / kTile)};
        }
        *count = tilings ? wanted : 0;
    }
    return S_OK;
}

// ---- slot 6: an element's byte offset ------------------------------------------------------------------------------
// (plane, mip, x, y, z or slice, sample, bit position out) -> byte offset from the texture's start; 0 when outside.
uint64_t __stdcall computer_element_offset(Computer* c, uint32_t plane, uint32_t mip, uint64_t x, uint32_t y,
                                           uint32_t z, uint32_t /*sample*/, uint32_t* bit) {
    if (bit) *bit = 0;
    if (!c || !c->valid || plane >= c->texture.planes.size()) return 0;
    const cl::Plane& p = c->texture.planes[plane];
    uint64_t at = 0;
    const uint32_t x32 = x > 0xffffffffull ? 0xffffffffu : static_cast<uint32_t>(x);
    if (p.layout.offset(mip, x32, y, z, at) != cl::Error::none) return 0;
    return p.offset + at;
}

template <size_t... Slot>
constexpr auto make_vtable(std::index_sequence<Slot...>) {
    return std::array<void*, sizeof...(Slot)>{
        (Slot == 0    ? reinterpret_cast<void*>(&computer_add_ref)
         : Slot == 1  ? reinterpret_cast<void*>(&computer_release)
         : Slot == 2  ? reinterpret_cast<void*>(&computer_get_resource_layout)
         : Slot == 6  ? reinterpret_cast<void*>(&computer_element_offset)
         : Slot == 8  ? reinterpret_cast<void*>(&computer_copy_into)
         : Slot == 9  ? reinterpret_cast<void*>(&computer_copy_from)
         : Slot == 10 ? reinterpret_cast<void*>(&computer_get_resource_tiling)
                      : reinterpret_cast<void*>(&computer_slot<Slot>))...};
}

const auto kVtable = make_vtable(std::make_index_sequence<kSlots>{});
Computer g_invalid{kVtable.data(), {}, false, 0, {}};  // for a call without a description or out of memory

Computer* computer_for(const void* desc) {
    if (!desc) return &g_invalid;
    D3D12_RESOURCE_DESC key{};
    std::memcpy(&key, desc, sizeof(key));
    std::string bytes(reinterpret_cast<const char*>(&key), sizeof(key));
    static std::mutex mutex;
    static std::unordered_map<std::string, std::unique_ptr<Computer>> computers;
    const std::lock_guard lock(mutex);
    std::unique_ptr<Computer>& slot = computers[bytes];
    if (!slot) {
        auto texture = texture_for(key);
        slot.reset(new (std::nothrow) Computer{kVtable.data(), key, texture.has_value(),
                                               texture ? static_cast<uint32_t>(texture->planes[0].layout.mips.size()) : 0,
                                               texture ? std::move(*texture) : cl::Texture{}});
    }
    return slot ? slot.get() : &g_invalid;
}

HRESULT __stdcall XGCreateTextureComputer(const void* desc, void** out) {
    if (!out) return E_POINTER;
    Computer* computer = computer_for(desc);
    *out = computer;  // two measured callers use *out without checking the HRESULT
    static const bool io_log = std::getenv("LEONIDA_IO_LOG") != nullptr;
    static std::atomic<int> n{0}, failed{0};
    const int k = ++n;
    if ((io_log && (k <= 300 || k % 500 == 0)) || (!computer->valid && failed.fetch_add(1) < 16))
        std::fprintf(stderr,
                     "[rt] xg_xs: XGCreateTextureComputer #%d dim %d %llux%u d/a %u mips %u fmt %d layout 0x%x flags "
                     "0x%x -> %s size 0x%llx (caller %s)\n",
                     k, static_cast<int>(computer->desc.Dimension),
                     static_cast<unsigned long long>(computer->desc.Width), computer->desc.Height,
                     computer->desc.DepthOrArraySize, computer->desc.MipLevels, static_cast<int>(computer->desc.Format),
                     static_cast<unsigned>(computer->desc.Layout), static_cast<unsigned>(computer->desc.Flags),
                     computer->valid ? "ok" : "NO LAYOUT", static_cast<unsigned long long>(computer->texture.size),
                     rt::shims::module_offset(reinterpret_cast<uint64_t>(_ReturnAddress())).c_str());
    return computer->valid ? S_OK : E_INVALIDARG;
}

// (desc, flags, out layout): the title passes flags 0 and the description's Layout field.
HRESULT __stdcall XGSuggestSwizzleMode(const void* desc, uint32_t /*flags*/, uint32_t* mode) {
    if (!desc || !mode) return E_INVALIDARG;
    D3D12_RESOURCE_DESC d{};
    std::memcpy(&d, desc, sizeof(d));
    *mode = kSwizzled | suggested(d);
    return S_OK;
}

}  // namespace

SHIM_TABLE(xg_xs,
    {"xg_xs.dll", "XGCreateTextureComputer", reinterpret_cast<void*>(&XGCreateTextureComputer)},
    {"xg_xs.dll", "XGSuggestSwizzleMode", reinterpret_cast<void*>(&XGSuggestSwizzleMode)},
);
