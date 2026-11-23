#pragma once

#include <windows.h>

inline bool xmem_placeholder_exact(const MEMORY_BASIC_INFORMATION& info, const void* address, SIZE_T size) {
    return info.BaseAddress == address && info.RegionSize == size;
}
