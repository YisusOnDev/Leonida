// Window placement for test runs on a multi-monitor desktop (window_placement.h): the helpers. The title's
// CreateWindowExA/W and ShowWindow imports are in window_placement_shims.cpp.
#include "window_placement.h"

#include <cwchar>
#include <cwctype>
#include <vector>


namespace rt::window {
namespace {

bool env_flag(const wchar_t* name) {
    wchar_t value[8] = {};
    const DWORD n = GetEnvironmentVariableW(name, value, 8);
    return n > 0 && n < 8 && value[0] != L'0';
}

struct Monitors {
    std::vector<MONITORINFOEXW> list;
};

BOOL CALLBACK collect_monitor(HMONITOR monitor, HDC, LPRECT, LPARAM data) {
    MONITORINFOEXW info{};
    info.cbSize = sizeof(info);
    if (GetMonitorInfoW(monitor, &info)) reinterpret_cast<Monitors*>(data)->list.push_back(info);
    return TRUE;
}

bool same_device(const wchar_t* device, const wchar_t* wanted) {
    // "\\.\DISPLAY2" or "DISPLAY2", case-insensitive.
    const wchar_t* d = device;
    if (std::wcsncmp(d, L"\\\\.\\", 4) == 0) d += 4;
    const wchar_t* w = wanted;
    if (std::wcsncmp(w, L"\\\\.\\", 4) == 0) w += 4;
    for (; *d && *w; ++d, ++w)
        if (std::towupper(*d) != std::towupper(*w)) return false;
    return *d == 0 && *w == 0;
}

}  // namespace

bool target_work_area(RECT* out) {
    wchar_t wanted[64] = {};
    const DWORD n = GetEnvironmentVariableW(L"LEONIDA_WINDOW_MONITOR", wanted, 64);
    if (n == 0 || n >= 64) return false;
    Monitors monitors;
    EnumDisplayMonitors(nullptr, nullptr, &collect_monitor, reinterpret_cast<LPARAM>(&monitors));
    wchar_t* end = nullptr;
    const unsigned long index = std::wcstoul(wanted, &end, 10);
    const MONITORINFOEXW* chosen = nullptr;
    if (end && *end == 0 && index >= 1 && index <= monitors.list.size()) {
        chosen = &monitors.list[index - 1];
    } else {
        for (const auto& info : monitors.list)
            if (same_device(info.szDevice, wanted)) chosen = &info;
    }
    if (!chosen) return false;
    if (out) *out = chosen->rcWork;
    return true;
}

// Read on every call, not cached in a function-local static: these run on the title's thread, where a static's
// thread-safe-init guard reads TLS the title owns (#7359). A window is created or shown rarely.
bool no_activate() { return env_flag(L"LEONIDA_WINDOW_NO_ACTIVATE"); }

void place(HWND window) {
    if (!window || (GetWindowLongPtrW(window, GWL_STYLE) & WS_CHILD)) return;
    RECT work{};
    if (!target_work_area(&work)) return;
    RECT frame{};
    if (!GetWindowRect(window, &frame)) return;
    const LONG work_w = work.right - work.left, work_h = work.bottom - work.top;
    LONG w = frame.right - frame.left, h = frame.bottom - frame.top;
    if (w <= 0 || w > work_w) w = work_w;
    if (h <= 0 || h > work_h) h = work_h;
    const LONG x = work.left + (work_w - w) / 2, y = work.top + (work_h - h) / 2;
    SetWindowPos(window, nullptr, x, y, w, h, SWP_NOZORDER | SWP_NOACTIVATE | SWP_NOOWNERZORDER);
}

int show_command(int command) {
    if (!no_activate()) return command;
    switch (command) {
    case SW_SHOWNORMAL:
    case SW_SHOWDEFAULT:
    case SW_RESTORE: return SW_SHOWNOACTIVATE;
    case SW_SHOW:
    case SW_SHOWMAXIMIZED: return SW_SHOWNA;
    case SW_SHOWMINIMIZED: return SW_SHOWMINNOACTIVE;
    default: return command;
    }
}
}  // namespace rt::window
