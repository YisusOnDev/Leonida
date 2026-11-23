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
// - XGCreateTextureComputer returns E_NOTIMPL, but still stores a computer in *out. The two checking callers then
//   fail cleanly; the two that do not check would otherwise call through a null pointer. COM says *out is null on
//   failure; this breaks that on purpose, because the measured callers need it.
// - Slot 0 and slot 1 count nothing (a computer lives for the run). Every other slot but 10, up to kSlots, prints its
//   index and caller once and returns E_NOTIMPL without writing its arguments, so a caller's own zero-filled outputs
//   stay zero and a stack structure the caller did not fill stays as it was.
// Hypothesis, not measured: slots 0 and 1 being AddRef and Release; the meaning of slots 2 and 8; that callers
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

template <size_t... Slot>
constexpr auto make_vtable(std::index_sequence<Slot...>) {
    return std::array<void*, sizeof...(Slot)>{
        (Slot == 0    ? reinterpret_cast<void*>(&computer_add_ref)
         : Slot == 1  ? reinterpret_cast<void*>(&computer_release)
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
    *out = computer_for(desc);  // see the top of the file: two measured callers use *out without checking the HRESULT
    return kNotImpl;
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
