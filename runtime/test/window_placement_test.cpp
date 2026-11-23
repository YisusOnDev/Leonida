// LEONIDA_WINDOW_MONITOR and LEONIDA_WINDOW_NO_ACTIVATE (shims/window_placement.h), through the registry entries a
// title's CreateWindowEx/ShowWindow imports bind to. Windows are created on this desktop and destroyed again; no
// game files. On a single-monitor machine the second-monitor checks are skipped and monitor 1 is used instead.
#include <windows.h>

#include <cstdio>
#include <string>

#include "shims.h"
#include "window_placement.h"

namespace {

int g_failures = 0;
void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_failures;
}

using CreateW = HWND(WINAPI*)(DWORD, LPCWSTR, LPCWSTR, DWORD, int, int, int, int, HWND, HMENU, HINSTANCE, LPVOID);
using Show = BOOL(WINAPI*)(HWND, int);

struct MonitorList {
    int count = 0;
    RECT work[16]{};
};
BOOL CALLBACK collect(HMONITOR monitor, HDC, LPRECT, LPARAM data) {
    auto* list = reinterpret_cast<MonitorList*>(data);
    MONITORINFO info{sizeof(info)};
    if (list->count < 16 && GetMonitorInfoW(monitor, &info)) list->work[list->count++] = info.rcWork;
    return TRUE;
}

bool inside(const RECT& r, const RECT& area) {
    return r.left >= area.left && r.top >= area.top && r.right <= area.right && r.bottom <= area.bottom;
}

HWND create(CreateW fn, DWORD style, int w, int h) {
    return fn(0, L"STATIC", L"window_placement_test", style, CW_USEDEFAULT, CW_USEDEFAULT, w, h, nullptr, nullptr,
              GetModuleHandleW(nullptr), nullptr);
}

}  // namespace

int main() {
    const auto create_w = reinterpret_cast<CreateW>(rt::shims::find("user32.dll", "CreateWindowExW"));
    const auto create_set = reinterpret_cast<CreateW>(
        rt::shims::find("ext-ms-win-rtcore-ntuser-window-l1-1-0.dll", "CreateWindowExW"));
    const auto show = reinterpret_cast<Show>(rt::shims::find("ext-ms-win-rtcore-ntuser-window-l1-1-0.dll", "ShowWindow"));
    check(create_w && create_set && show, "CreateWindowExW and ShowWindow are served for user32 and the API set");
    check(rt::shims::find("ext-ms-win-rtcore-ntuser-window-ansi-l1-1-0.dll", "CreateWindowExA") != nullptr,
          "CreateWindowExA is served for the ANSI API set the retail executable imports it from");
    if (!create_w || !create_set || !show) return 1;

    MonitorList monitors;
    EnumDisplayMonitors(nullptr, nullptr, &collect, reinterpret_cast<LPARAM>(&monitors));
    check(monitors.count >= 1, "at least one monitor");
    const int target = monitors.count >= 2 ? 2 : 1;
    if (monitors.count < 2) std::printf("SKIP second-monitor placement: one monitor, monitor 1 is used instead\n");
    const RECT area = monitors.work[target - 1];

    // Unset: nothing changes, and the mapping leaves commands alone.
    SetEnvironmentVariableW(L"LEONIDA_WINDOW_MONITOR", nullptr);
    SetEnvironmentVariableW(L"LEONIDA_WINDOW_NO_ACTIVATE", nullptr);
    RECT none{};
    check(!rt::window::target_work_area(&none), "no target without LEONIDA_WINDOW_MONITOR");
    check(rt::window::show_command(SW_SHOWNORMAL) == SW_SHOWNORMAL, "commands unchanged without no-activate");

    // By number: a top-level window lands inside the chosen monitor's work area, centred.
    SetEnvironmentVariableW(L"LEONIDA_WINDOW_MONITOR", std::to_wstring(target).c_str());
    RECT got{};
    check(rt::window::target_work_area(&got) && EqualRect(&got, &area), "LEONIDA_WINDOW_MONITOR=<n> names monitor n");
    HWND a = create(create_set, WS_OVERLAPPEDWINDOW, 640, 360);
    RECT frame{};
    check(a && GetWindowRect(a, &frame) && inside(frame, area), "a top-level window is moved onto the target monitor");
    const LONG cx = (frame.left + frame.right) / 2, ax = (area.left + area.right) / 2;
    check(cx >= ax - 2 && cx <= ax + 2, "and centred in its work area");
    DestroyWindow(a);

    // A window bigger than the work area is shrunk to fit.
    HWND big = create(create_w, WS_OVERLAPPEDWINDOW, 100000, 100000);
    check(big && GetWindowRect(big, &frame) && inside(frame, area), "a window larger than the monitor is fitted");
    DestroyWindow(big);

    // By device name, with or without the \\.\ prefix; an unknown name changes nothing.
    MONITORINFOEXW info{};
    info.cbSize = sizeof(info);
    POINT inside_point{area.left + 1, area.top + 1};
    GetMonitorInfoW(MonitorFromPoint(inside_point, MONITOR_DEFAULTTONEAREST), &info);
    const std::wstring device = info.szDevice;                                    // "\\.\DISPLAYn"
    SetEnvironmentVariableW(L"LEONIDA_WINDOW_MONITOR", device.c_str());
    check(rt::window::target_work_area(&got) && EqualRect(&got, &area), "a device name with \\\\.\\ selects it");
    SetEnvironmentVariableW(L"LEONIDA_WINDOW_MONITOR", device.substr(4).c_str());
    check(rt::window::target_work_area(&got) && EqualRect(&got, &area), "and without the prefix");
    SetEnvironmentVariableW(L"LEONIDA_WINDOW_MONITOR", L"DISPLAY999");
    check(!rt::window::target_work_area(&got), "an unknown monitor is no target");
    SetEnvironmentVariableW(L"LEONIDA_WINDOW_MONITOR", L"99");
    check(!rt::window::target_work_area(&got), "nor is a number past the last monitor");

    // A child window is never moved.
    SetEnvironmentVariableW(L"LEONIDA_WINDOW_MONITOR", std::to_wstring(target).c_str());
    HWND parent = CreateWindowExW(0, L"STATIC", L"parent", WS_OVERLAPPEDWINDOW, 0, 0, 300, 200, nullptr, nullptr,
                                  GetModuleHandleW(nullptr), nullptr);
    HWND child = create_w(0, L"STATIC", L"child", WS_CHILD, 10, 20, 50, 40, parent, nullptr, GetModuleHandleW(nullptr),
                          nullptr);
    RECT child_rect{};
    GetWindowRect(child, &child_rect);
    RECT parent_rect{};
    GetWindowRect(parent, &parent_rect);
    MapWindowPoints(HWND_DESKTOP, parent, reinterpret_cast<POINT*>(&child_rect), 2);
    check(child && child_rect.left == 10 && child_rect.top == 20, "a child window keeps its position");
    DestroyWindow(parent);

    // No-activate: the command mapping, and a window created visible is shown but never becomes the foreground window.
    SetEnvironmentVariableW(L"LEONIDA_WINDOW_NO_ACTIVATE", L"1");
    check(rt::window::show_command(SW_SHOWNORMAL) == SW_SHOWNOACTIVATE &&
              rt::window::show_command(SW_SHOWDEFAULT) == SW_SHOWNOACTIVATE &&
              rt::window::show_command(SW_SHOW) == SW_SHOWNA && rt::window::show_command(SW_SHOWMAXIMIZED) == SW_SHOWNA &&
              rt::window::show_command(SW_SHOWMINIMIZED) == SW_SHOWMINNOACTIVE &&
              rt::window::show_command(SW_HIDE) == SW_HIDE,
          "no-activate maps every showing command to its non-activating form");
    HWND v = create(create_set, WS_OVERLAPPEDWINDOW | WS_VISIBLE, 400, 300);
    check(v && IsWindowVisible(v) && GetForegroundWindow() != v, "a window created visible is shown, not activated");
    check(GetWindowRect(v, &frame) && inside(frame, area), "and it is on the target monitor");
    HWND s = create(create_set, WS_OVERLAPPEDWINDOW, 400, 300);
    show(s, SW_SHOWNORMAL);
    check(IsWindowVisible(s) && GetForegroundWindow() != s, "ShowWindow(SW_SHOWNORMAL) does not activate either");
    show(s, SW_SHOWMAXIMIZED);
    check(GetWindowRect(s, &frame) && inside(frame, area) && GetForegroundWindow() != s,
          "a 'maximized' window fills the target's work area without activation");
    DestroyWindow(v);
    DestroyWindow(s);
    SetEnvironmentVariableW(L"LEONIDA_WINDOW_MONITOR", nullptr);
    SetEnvironmentVariableW(L"LEONIDA_WINDOW_NO_ACTIVATE", nullptr);

    std::printf("window_placement_test: %s (%d failure%s)\n", g_failures ? "FAILED" : "OK", g_failures,
                g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}
