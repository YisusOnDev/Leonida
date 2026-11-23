#pragma once
// A write by the kernel into the title's memory (ReadFile and the rest of the family) gets no first-touch fault.
// xmem commits a physical page when the title first touches it, from a vectored handler; the kernel's write into a
// page not committed yet raises no user-mode fault, so ReadFile fails with ERROR_NOACCESS instead. The file shims
// (file_system.cpp) call before() on a read's destination; xmem sets the hook to commit the physical pages there
// (rt::xmem::commit_physical_range) once it has physical pages.
//
// Header-only: file_system.cpp is built into executables that don't link xmem.cpp.
#include <atomic>
#include <cstddef>

namespace rt::kernel_write {

using Hook = void (*)(const void* address, std::size_t size);

inline std::atomic<Hook> hook{nullptr};

inline void set_hook(Hook h) { hook.store(h, std::memory_order_release); }

// Before the kernel writes [address, address + size).
inline void before(const void* address, std::size_t size) {
    const Hook h = hook.load(std::memory_order_acquire);
    if (h && address && size) h(address, size);
}

}  // namespace rt::kernel_write
