// xg_xs.dll: the console's texture layout library. The game executable (1.0.2.1) imports two functions from it.
//
// Measured from the call sites in the executable, read statically (shapes and checks only):
//   HRESULT XGSuggestSwizzleMode(const desc*, 0, uint32_t* mode)
//     2 call sites, both in the game's helper that fills a texture description before XGCreateTextureComputer. The
//     out pointer is a 32-bit field of that description, which the helper first sets to 0 and, when the mode is
//     known, writes itself as 0x100 | mode. The HRESULT is only passed back up; no caller branches on it.
//   HRESULT XGCreateTextureComputer(const desc*, computer** out)
//     4 call sites. Two test the HRESULT and take their failure path when it is negative; two ignore it and call
//     through *out at once. On the computer the callers use vtable slot 1 after they are done (3 sites, read as
//     Release; no site calls slot 0, read as AddRef, nor QueryInterface), slot 2 with one out pointer to a large
//     stack structure (2 sites), slot 8 with six arguments in a loop over subresources (1 site) and slot 10 with
//     seven arguments (1 site). No site reads the return value of slots 2, 8 or 10.
//
// Slot 10 is GetResourceTiling (its section below): it answers the PC device's 64 KiB tiling of the description, the
// numbers ID3D12Device::GetResourceTiling gives for it, so the title's mip counts are the description's.
//
// The runtime has no model of the console's resource layout (runtime/src/gfx10_tiling.h covers two swizzle modes of
// one 2D level, without mips or the layout structure slot 2 fills), so nothing else here computes a layout:
// - XGSuggestSwizzleMode returns E_NOTIMPL and writes nothing, so the field keeps the 0 the caller put there.
// - XGCreateTextureComputer stores a computer in *out whatever it returns. For a description it can lay out linearly
//   (2D or 3D, one sample, a format element_size knows) it returns S_OK, and slots 2, 8 and 9 answer that linear layout
//   (their sections below). Otherwise it returns E_NOTIMPL: the two checking callers then fail cleanly, and the two
//   that do not check would otherwise call through a null pointer. COM says *out is null on failure; this breaks that
//   on purpose, because the measured callers need it.
// - Slot 0 and slot 1 count nothing (a computer lives for the run). Every other slot but 2, 8, 9 and 10, up to kSlots, prints its
//   index and caller once and returns E_NOTIMPL without writing its arguments, so a caller's own zero-filled outputs
//   stay zero and a stack structure the caller did not fill stays as it was.
// Hypothesis, not measured: slots 0 and 1 being AddRef and Release; that callers
// reach no slot above kSlots - 1 (the computer is also stored in an object at one site and may be called later).
#include <windows.h>
#include <d3d12.h>
#include <intrin.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>
#include <utility>

#include "../src/console_linear_layout.h"
#include "../src/resource_tiling.h"
#include "shims.h"

namespace {

constexpr HRESULT kNotImpl = static_cast<HRESULT>(rt::shims::kENotImpl);
constexpr size_t kSlots = 32;

// Prints once per entry point; `slot` is the vtable index for a computer method, or -1 for an export.
void warn_once(std::atomic<bool>& warned, const char* what, int slot, const void* caller) {
    if (warned.exchange(true, std::memory_order_relaxed)) return;
    char name[64];
    if (slot < 0) std::snprintf(name, sizeof(name), "%s", what);
    else std::snprintf(name, sizeof(name), "%s%d", what, slot);
    std::fprintf(stderr, "[rt] xg_xs: %s: no console texture layout on the PC; E_NOTIMPL (caller %s, logged once)\n",
                 name, rt::shims::module_offset(reinterpret_cast<uint64_t>(caller)).c_str());
}

ULONG __stdcall computer_add_ref(void*) { return 1; }
ULONG __stdcall computer_release(void*) { return 1; }

template <size_t Slot>
HRESULT __stdcall computer_slot(void*) {
    static std::atomic<bool> warned{false};
    warn_once(warned, "texture computer slot ", static_cast<int>(Slot), _ReturnAddress());
    return kNotImpl;
}

// A computer keeps the description it was made from. The description starts with a D3D12_RESOURCE_DESC: measured at
// the game's XGCreateTextureComputer call sites under gta6rt --run, the first 52 bytes are a dimension (3 for 2D,
// 4 for 3D), an alignment, width, height, depth or array size, mip levels, a DXGI format, a sample count, the
// console layout word in Layout and console bits in Flags; the console's own fields follow. Computers are never
// freed (Release counts nothing, as before: no site calls AddRef and one stores the computer for later), so one is
// kept per distinct description and reused.
struct Computer {
    const void* const* vtable;
    D3D12_RESOURCE_DESC desc;
    bool has_desc;
};

// Bytes per element and texels per block side for the formats D3D12 tiles with the standard 64 KiB tile shapes
// (public DXGI values). Formats without a measured tile shape (12-byte texels, the depth-stencil formats with a
// stencil plane, the packed 4:2:2 and 1-bit formats) return false, as the Vulkan layer's GetResourceTiling does.
bool element_size(DXGI_FORMAT format, uint32_t& bytes, uint32_t& block) {
    block = 1;
    const unsigned f = static_cast<unsigned>(format);
    if (f >= 1 && f <= 4) bytes = 16;                                                  // R32G32B32A32
    else if (f >= 9 && f <= 18) bytes = 8;                                             // R16G16B16A16, R32G32
    else if ((f >= 23 && f <= 43) || f == 67 || (f >= 87 && f <= 93)) bytes = 4;       // 32-bit, R9G9B9E5, B8G8R8A8/X8
    else if ((f >= 48 && f <= 59) || f == 85 || f == 86 || f == 115) bytes = 2;        // R8G8, R16, B5G6R5, B5G5R5A1, B4G4R4A4
    else if (f >= 60 && f <= 65) bytes = 1;                                            // R8, A8
    else if ((f >= 70 && f <= 72) || (f >= 79 && f <= 81)) bytes = 8, block = 4;       // BC1, BC4
    else if ((f >= 73 && f <= 78) || (f >= 82 && f <= 84) || (f >= 94 && f <= 99)) bytes = 16, block = 4;  // BC2, BC3, BC5, BC6H, BC7
    else return false;
    return true;
}

bool computer_tiling(const Computer& c, leonida::ResourceTiling& out) {
    if (!c.has_desc) return false;
    const D3D12_RESOURCE_DESC& d = c.desc;
    if (d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D && d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE3D) return false;
    leonida::TilingDesc t{};
    t.kind = d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D ? leonida::TiledKind::texture3d : leonida::TiledKind::texture2d;
    t.width = d.Width;
    t.height = d.Height;
    t.depth_or_layers = d.DepthOrArraySize;
    t.samples = d.SampleDesc.Count ? d.SampleDesc.Count : 1;
    if (!d.Width || !d.Height || !d.DepthOrArraySize || !element_size(d.Format, t.element_bytes, t.block)) return false;
    t.mips = d.MipLevels;
    if (!t.mips) {  // 0 is the full chain, as in D3D12
        uint64_t largest = std::max<uint64_t>(d.Width, d.Height);
        if (t.kind == leonida::TiledKind::texture3d) largest = std::max<uint64_t>(largest, d.DepthOrArraySize);
        while (largest >> t.mips) ++t.mips;
    }
    return leonida::resource_tiling(t, out);
}

// Slot 10: GetResourceTiling(total tiles, packed mip info, tile shape, subresource tiling count in/out, first,
// subresource tilings), the arguments of ID3D12Device::GetResourceTiling without the resource. Measured at the one call
// site: four out pointers to zeroed 4-byte and 12-byte locals, a count of 16, first 0 and an array; after the call the
// title reads the packed mip info's two counts (standard and packed mips) and adds them. The answer is the PC
// device's tiling of the same description (src/resource_tiling.h, as the Vulkan layer's GetResourceTiling gives it),
// not the console's: on the PC the texture is a PC resource. A description it cannot tile keeps the old answer:
// E_NOTIMPL with nothing written.
HRESULT __stdcall computer_get_resource_tiling(Computer* self, UINT* total, D3D12_PACKED_MIP_INFO* packed,
                                               D3D12_TILE_SHAPE* shape, UINT* count, UINT first,
                                               D3D12_SUBRESOURCE_TILING* tilings) {
    leonida::ResourceTiling tiling;
    if (!self || !computer_tiling(*self, tiling)) {
        static std::atomic<bool> warned{false};
        warn_once(warned, "texture computer slot ", 10, _ReturnAddress());
        return kNotImpl;
    }
    if (total) *total = tiling.total_tiles;
    if (packed)
        *packed = D3D12_PACKED_MIP_INFO{static_cast<UINT8>(tiling.standard_mips), static_cast<UINT8>(tiling.packed_mips),
                                        tiling.packed_tiles, tiling.packed_start};
    if (shape) *shape = D3D12_TILE_SHAPE{tiling.shape.width, tiling.shape.height, tiling.shape.depth};
    if (count) {
        const size_t available = first < tiling.subresources.size() ? tiling.subresources.size() - first : 0;
        const UINT n = tilings ? static_cast<UINT>(std::min<size_t>(*count, available)) : 0;
        for (UINT i = 0; i < n; ++i) {
            const leonida::SubresourceTiling& s = tiling.subresources[first + i];
            tilings[i] = D3D12_SUBRESOURCE_TILING{s.width, static_cast<UINT16>(s.height), static_cast<UINT16>(s.depth), s.start};
        }
        *count = n;
    }
    return S_OK;
}

// The linear layout a computer answers for its description: subresources in D3D12 order (array slice by array slice,
// mip by mip), rows of whole blocks at a pitch rounded up to 256 bytes, slices of whole rows (src/console_linear_layout.h:
// the title assumes that pitch). Not the console's tiled layouts: on the PC the texture is a PC resource, and the memory
// the title hands slot 9 is only read back through this same layout.
struct MipLayout {
    uint64_t size;   // one subresource of this mip (all its depth slices)
    uint64_t slice;  // one depth slice
    uint32_t row;    // the bytes of one row of blocks
    uint32_t pitch;  // from one row to the next: row rounded up to 256 bytes
    uint32_t rows;   // rows of blocks in a slice
    uint32_t depth;  // depth slices (1 for 2D)
};
constexpr uint32_t kMaxLayoutMips = 15;

bool linear_layout(const Computer& c, MipLayout (&mips)[kMaxLayoutMips], uint32_t& mip_count, uint32_t& arrays) {
    if (!c.has_desc) return false;
    const D3D12_RESOURCE_DESC& d = c.desc;
    const bool volume = d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    if (d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D && !volume) return false;
    if (d.SampleDesc.Count > 1 || !d.Width || !d.Height || !d.DepthOrArraySize || d.Width > 0x10000) return false;
    uint32_t bytes = 0, block = 1;
    if (!element_size(d.Format, bytes, block)) return false;
    mip_count = d.MipLevels;
    if (!mip_count) {
        uint64_t largest = std::max<uint64_t>(d.Width, d.Height);
        if (volume) largest = std::max<uint64_t>(largest, d.DepthOrArraySize);
        while (largest >> mip_count) ++mip_count;
    }
    if (mip_count > kMaxLayoutMips) return false;
    arrays = volume ? 1 : d.DepthOrArraySize;
    for (uint32_t m = 0; m < mip_count; ++m) {
        const uint64_t w = std::max<uint64_t>(1, d.Width >> m), h = std::max<uint64_t>(1, d.Height >> m);
        MipLayout& l = mips[m];
        l.row = static_cast<uint32_t>((w + block - 1) / block * bytes);
        l.pitch = static_cast<uint32_t>(leonida::console_linear_pitch(l.row));
        l.rows = static_cast<uint32_t>((h + block - 1) / block);
        l.depth = volume ? std::max<uint32_t>(1, d.DepthOrArraySize >> m) : 1;
        l.slice = uint64_t{l.pitch} * l.rows;
        l.size = l.slice * l.depth;
    }
    return true;
}

bool can_lay_out(const Computer& c) {
    MipLayout mips[kMaxLayoutMips];
    uint32_t count = 0, arrays = 0;
    return linear_layout(c, mips, count, arrays);
}

// Slot 2: GetResourceLayout(layout*). The structure is 0x1B40 bytes: measured at +0x5121a84, which hands slot 2 a stack
// structure and then copies one 0x6C8-byte plane record out of it (from +0x18 + 0x6C8 * plane) into a cache the title
// sizes its texture memory from; and at the checking creation site (+0x979c40), which for mip m reads +0x50 + 0x70*m
// (the subresource's size: it allocates that much for slot 9's output), +0x60 + 0x70*m (the depth-slice pitch) and
// +0x6C + 0x70*m (the row pitch). Field offsets as documented by VIctory (github.com/Marvisak/VIctory, GPL-3.0,
// crates/victory-hle/src/modules/xg.rs), which agree with both sites:
//   +0x00 u64 size, +0x08 u64 alignment, +0x10 u32 mips, +0x14 u32 planes, +0x18 plane[4] (0x6C8 each),
//   +0x1B38 u32 dimension;
//   plane: +0x00 u32 usage (1 data), +0x08 u64 size, +0x10 u64 slice (one array slice) size, +0x18 u64 offset,
//   +0x20 u64 alignment, +0x28 u32 element bytes, +0x2C u32 standard mips, +0x30 u32 tail mips, +0x38 mip[15]
//   (0x70 each);
//   mip: +0x00 u64 size, +0x08 u64 offset in the array slice, +0x10 u64 depth-slice size, +0x18 u32 pitch in texels,
//   +0x1C u32 pitch in bytes, +0x20 u32 alignment, +0x24 u32[3] padded width/height (elements) and depth or array
//   size, +0x30 u32[2] width/height in elements, +0x38 u32[2] width/height in texels, +0x40 u32 depth or array size,
//   +0x44 u32 samples, +0x48 u32 swizzle mode (0 linear), +0x60 u32 slice depth, +0x64 u32 in mip tail.
// The whole structure is written (zeros first): before, everything but the three fields above was whatever the
// caller's stack held, and +0x5121a84 copied that into its plane cache. (Not the cause of the AV at +0x13360a3:
// boot frontier3-2 still had it with this change.) The layout is the linear one slots 8 and 9 use, one data plane,
// no mip tail.
constexpr size_t kLayoutBytes = 0x1B40, kPlaneAt = 0x18, kPlaneBytes = 0x6C8, kMipAt = 0x38, kMipBytes = 0x70;
constexpr uint64_t kLinearAlignment = 0x100, kResourceAlignment = 0x1000;

template <typename T>
void put(uint8_t* at, T value) {
    std::memcpy(at, &value, sizeof(value));
}

HRESULT __stdcall computer_get_resource_layout(Computer* self, uint8_t* layout) {
    MipLayout mips[kMaxLayoutMips];
    uint32_t count = 0, arrays = 0;
    if (!self || !layout || !linear_layout(*self, mips, count, arrays)) {
        static std::atomic<bool> warned{false};
        warn_once(warned, "texture computer slot ", 2, _ReturnAddress());
        return kNotImpl;
    }
    const D3D12_RESOURCE_DESC& d = self->desc;
    const bool volume = d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    uint32_t bytes = 0, block = 1;
    element_size(d.Format, bytes, block);
    std::memset(layout, 0, kLayoutBytes);
    uint64_t per_array = 0;
    uint8_t* const plane = layout + kPlaneAt;
    for (uint32_t m = 0; m < count; ++m) {
        const MipLayout& l = mips[m];
        const uint32_t width = static_cast<uint32_t>(std::max<uint64_t>(1, d.Width >> m));
        const uint32_t height = std::max<uint32_t>(1, d.Height >> m);
        const uint32_t elements_wide = l.row / bytes, pitch_elements = l.pitch / bytes;
        uint8_t* entry = plane + kMipAt + kMipBytes * m;
        put<uint64_t>(entry + 0x00, l.size);
        put<uint64_t>(entry + 0x08, per_array);
        put<uint64_t>(entry + 0x10, l.slice);
        put<uint32_t>(entry + 0x18, pitch_elements * block);
        put<uint32_t>(entry + 0x1C, l.pitch);
        put<uint32_t>(entry + 0x20, static_cast<uint32_t>(kLinearAlignment));
        put<uint32_t>(entry + 0x24, pitch_elements);
        put<uint32_t>(entry + 0x28, l.rows);
        put<uint32_t>(entry + 0x2C, volume ? l.depth : arrays);
        put<uint32_t>(entry + 0x30, elements_wide);
        put<uint32_t>(entry + 0x34, l.rows);
        put<uint32_t>(entry + 0x38, width);
        put<uint32_t>(entry + 0x3C, height);
        put<uint32_t>(entry + 0x40, volume ? l.depth : arrays);
        put<uint32_t>(entry + 0x44, 1);
        put<uint32_t>(entry + 0x60, 1);
        per_array += l.size;
    }
    const uint64_t total = per_array * arrays;
    put<uint32_t>(plane + 0x00, 1);
    put<uint64_t>(plane + 0x08, total);
    put<uint64_t>(plane + 0x10, per_array);
    put<uint64_t>(plane + 0x20, kResourceAlignment);
    put<uint32_t>(plane + 0x28, bytes);
    put<uint32_t>(plane + 0x2C, count);
    put<uint64_t>(layout + 0x00, total);
    put<uint64_t>(layout + 0x08, kResourceAlignment);
    put<uint32_t>(layout + 0x10, count);
    put<uint32_t>(layout + 0x14, 1);
    put<uint32_t>(layout + kPlaneAt + 4 * kPlaneBytes, static_cast<uint32_t>(d.Dimension));
    return S_OK;
}

// Row copies that turn an access violation into a failure (no C++ objects here: the function has __try).
// `into` copies the caller's rows into the texture's memory (slot 8) instead of out of it (slot 9).
bool guarded_rows(uint8_t* caller, uint64_t caller_row, uint64_t caller_slice, uint8_t* texels, const MipLayout& l,
                  bool into) {
    __try {
        for (uint32_t z = 0; z < l.depth; ++z)
            for (uint32_t y = 0; y < l.rows; ++y) {
                uint8_t* const c = caller + z * caller_slice + y * caller_row;
                uint8_t* const t = texels + z * l.slice + uint64_t{y} * l.pitch;
                if (into) std::memcpy(t, c, l.row);
                else std::memcpy(c, t, l.row);
            }
        return true;
    } __except (GetExceptionCode() == EXCEPTION_ACCESS_VIOLATION ? EXCEPTION_EXECUTE_HANDLER
                                                                  : EXCEPTION_CONTINUE_SEARCH) {
        return false;
    }
}

// The byte offset of a subresource in the linear layout, or false for one past the description.
bool subresource_at(const Computer& c, UINT subresource, MipLayout& mip, uint64_t& offset) {
    MipLayout mips[kMaxLayoutMips];
    uint32_t count = 0, arrays = 0;
    if (!linear_layout(c, mips, count, arrays) || subresource >= count * arrays) return false;
    uint64_t per_array = 0;
    for (uint32_t m = 0; m < count; ++m) per_array += mips[m].size;
    const uint32_t m = subresource % count;
    offset = per_array * (subresource / count);
    for (uint32_t i = 0; i < m; ++i) offset += mips[i].size;
    mip = mips[m];
    return true;
}

// Slot 9: copies one subresource out of the texture's memory: (dst, plane, subresource, box, texel base, dst row
// pitch, dst slice pitch). Measured at +0x979e30: dst is the title's buffer (or a temporary of slot 2's size), plane 0,
// box null, the texel base the texture's memory; the title reads dst afterwards. The texel base is read in the linear
// layout slot 2 answers. A box, another plane or a subresource past the description is refused with nothing written.
HRESULT __stdcall computer_copy_from_subresource(Computer* self, uint8_t* dst, UINT plane, UINT subresource,
                                                 const void* box, const uint8_t* texels, UINT dst_row, UINT dst_slice) {
    MipLayout l{};
    uint64_t offset = 0;
    if (!self || !dst || !texels || plane || box || !subresource_at(*self, subresource, l, offset)) {
        static std::atomic<bool> warned{false};
        warn_once(warned, "texture computer slot ", 9, _ReturnAddress());
        return kNotImpl;
    }
    if (dst_row < l.row || (l.depth > 1 && dst_slice < uint64_t{dst_row} * l.rows)) return E_INVALIDARG;
    return guarded_rows(dst, dst_row, dst_slice, const_cast<uint8_t*>(texels) + offset, l, false) ? S_OK : E_POINTER;
}

// Slot 8: copies one subresource into the texture's memory: (texel base, plane, subresource, src, src row pitch, src
// slice pitch). Measured at +0x16f71fb, in a loop over the subresources of an upload: the texel base is the title's
// texture memory, plane 0, the subresource index, the source the title's data at its running offset and its tight row
// and slice pitches. Written in the linear layout slot 2 answers; another plane or a subresource past the description
// is refused with nothing written.
HRESULT __stdcall computer_copy_into_subresource(Computer* self, uint8_t* texels, UINT plane, UINT subresource,
                                                 const uint8_t* src, UINT src_row, UINT src_slice) {
    MipLayout l{};
    uint64_t offset = 0;
    if (!self || !texels || !src || plane || !subresource_at(*self, subresource, l, offset)) {
        static std::atomic<bool> warned{false};
        warn_once(warned, "texture computer slot ", 8, _ReturnAddress());
        return kNotImpl;
    }
    if (src_row < l.row || (l.depth > 1 && src_slice < uint64_t{src_row} * l.rows)) return E_INVALIDARG;
    return guarded_rows(const_cast<uint8_t*>(src), src_row, src_slice, texels + offset, l, true) ? S_OK : E_POINTER;
}

template <size_t... Slot>
constexpr auto make_vtable(std::index_sequence<Slot...>) {
    return std::array<void*, sizeof...(Slot)>{
        (Slot == 0    ? reinterpret_cast<void*>(&computer_add_ref)
         : Slot == 1  ? reinterpret_cast<void*>(&computer_release)
         : Slot == 2  ? reinterpret_cast<void*>(&computer_get_resource_layout)
         : Slot == 8  ? reinterpret_cast<void*>(&computer_copy_into_subresource)
         : Slot == 9  ? reinterpret_cast<void*>(&computer_copy_from_subresource)
         : Slot == 10 ? reinterpret_cast<void*>(&computer_get_resource_tiling)
                      : reinterpret_cast<void*>(&computer_slot<Slot>))...};
}

const auto kVtable = make_vtable(std::make_index_sequence<kSlots>{});
Computer g_computer{kVtable.data(), {}, false};  // for a call without a description

// One computer per distinct description, kept for the run (see Computer).
Computer* computer_for(const void* desc) {
    if (!desc) return &g_computer;
    D3D12_RESOURCE_DESC key{};
    std::memcpy(&key, desc, sizeof(key));
    std::string bytes(reinterpret_cast<const char*>(&key), sizeof(key));
    static std::mutex mutex;
    static std::unordered_map<std::string, std::unique_ptr<Computer>> computers;
    const std::lock_guard lock(mutex);
    std::unique_ptr<Computer>& slot = computers[bytes];
    if (!slot) slot.reset(new (std::nothrow) Computer{kVtable.data(), key, true});
    return slot ? slot.get() : &g_computer;
}

HRESULT __stdcall XGCreateTextureComputer(const void* desc, void** out) {
    static std::atomic<bool> warned{false};
    warn_once(warned, "XGCreateTextureComputer", -1, _ReturnAddress());
    if (!out) return E_POINTER;
    Computer* computer = computer_for(desc);
    *out = computer;  // see the top of the file: two measured callers use *out without checking the HRESULT
    if (!can_lay_out(*computer)) return kNotImpl;
    static std::atomic<int> logged{0};
    if (logged.fetch_add(1, std::memory_order_relaxed) < 6)
        std::fprintf(stderr,
                     "[rt] xg_xs: XGCreateTextureComputer: linear layout for dimension %d %llux%u depth/array %u, "
                     "%u mip(s), format %d (caller %s)\n",
                     static_cast<int>(computer->desc.Dimension), static_cast<unsigned long long>(computer->desc.Width),
                     computer->desc.Height, computer->desc.DepthOrArraySize, computer->desc.MipLevels,
                     static_cast<int>(computer->desc.Format),
                     rt::shims::module_offset(reinterpret_cast<uint64_t>(_ReturnAddress())).c_str());
    return S_OK;
}

HRESULT __stdcall XGSuggestSwizzleMode(const void* /*desc*/, uint32_t /*flags*/, uint32_t* /*mode*/) {
    static std::atomic<bool> warned{false};
    warn_once(warned, "XGSuggestSwizzleMode", -1, _ReturnAddress());
    return kNotImpl;
}

}  // namespace

SHIM_TABLE(xg_xs,
    {"xg_xs.dll", "XGCreateTextureComputer", reinterpret_cast<void*>(&XGCreateTextureComputer)},
    {"xg_xs.dll", "XGSuggestSwizzleMode", reinterpret_cast<void*>(&XGSuggestSwizzleMode)},
);
