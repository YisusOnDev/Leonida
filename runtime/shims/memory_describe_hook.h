#pragma once
// A one-line account of a title address for a diagnostic: the physical-page window it is in, the page mapped there (or
// none), the last map or unmap of that 64 KiB page with its caller and time, and the host's view of it. xmem sets the
// hook once it has physical pages (rt::xmem::describe_physical_address); the D3D12 layer's CopyMemoryToMemoryX (console
// list slot 62) names its source and destination with it when a copy faults.
//
// Header-only: d3d12_xs_device.cpp is built into executables that don't link xmem.cpp.
#include <atomic>
#include <cstddef>

namespace rt::memory_describe {

// Writes at most `size` bytes (terminated) to `out` and returns the length.
using Hook = std::size_t (*)(const void* address, char* out, std::size_t size);

inline std::atomic<Hook> hook{nullptr};

inline void set_hook(Hook h) { hook.store(h, std::memory_order_release); }

// "" when no describer is set.
inline std::size_t describe(const void* address, char* out, std::size_t size) {
    if (!out || !size) return 0;
    out[0] = '\0';
    const Hook h = hook.load(std::memory_order_acquire);
    return h ? h(address, out, size) : 0;
}

// A code address as module+RVA, including the title's own image, which the PE loader maps outside the host's module
// list (xmem sets it with the loader's own naming). "" when no hook is set.
inline std::atomic<Hook> code_hook{nullptr};

inline void set_code_hook(Hook h) { code_hook.store(h, std::memory_order_release); }

inline std::size_t code_address(const void* address, char* out, std::size_t size) {
    if (!out || !size) return 0;
    out[0] = '\0';
    const Hook h = code_hook.load(std::memory_order_acquire);
    return h ? h(address, out, size) : 0;
}

}  // namespace rt::memory_describe
