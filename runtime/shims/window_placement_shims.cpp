// The title's CreateWindowExA/W and ShowWindow imports, placed and shown per window_placement.h. Only gta6rt and
// window_placement_test link this table; the helpers in window_placement.cpp link anywhere.
#include "window_placement.h"

#include "shims.h"

namespace rt::window {
namespace {

// A window created visible is shown (and activated) by CreateWindowEx itself, before place() could move it: it is
// created hidden instead, placed, then shown without activation. A child window or one with a parent is passed on.
template <class CreateFn, class Char>
HWND create_placed(CreateFn create, DWORD ex_style, const Char* class_name, const Char* window_name, DWORD style,
                   int x, int y, int width, int height, HWND parent, HMENU menu, HINSTANCE instance, void* param) {
    const bool top_level = !(style & WS_CHILD) && !parent;
    const bool managed = top_level && (no_activate() || target_work_area(nullptr));
    const bool visible = (style & WS_VISIBLE) != 0;
    HWND window = create(ex_style, class_name, window_name, managed ? (style & ~WS_VISIBLE) : style, x, y, width,
                         height, parent, menu, instance, param);
    if (!window || !managed) return window;
    place(window);
    if (visible) ShowWindow(window, no_activate() ? SW_SHOWNOACTIVATE : SW_SHOWNORMAL);
    return window;
}

HWND WINAPI create_window_a(DWORD ex_style, LPCSTR class_name, LPCSTR window_name, DWORD style, int x, int y,
                            int width, int height, HWND parent, HMENU menu, HINSTANCE instance, LPVOID param) {
    return create_placed(&CreateWindowExA, ex_style, class_name, window_name, style, x, y, width, height, parent,
                         menu, instance, param);
}

HWND WINAPI create_window_w(DWORD ex_style, LPCWSTR class_name, LPCWSTR window_name, DWORD style, int x, int y,
                            int width, int height, HWND parent, HMENU menu, HINSTANCE instance, LPVOID param) {
    return create_placed(&CreateWindowExW, ex_style, class_name, window_name, style, x, y, width, height, parent,
                         menu, instance, param);
}

BOOL WINAPI show_window(HWND window, int command) {
    const int adjusted = show_command(command);
    if (adjusted != command && command == SW_SHOWMAXIMIZED) place(window);  // "maximized" as the work area
    return ShowWindow(window, adjusted);
}

}  // namespace
}  // namespace rt::window

// The retail executable imports these from the API sets below; a title linked against user32 directly gets the same.
SHIM_TABLE(window_placement,
    {"ext-ms-win-rtcore-ntuser-window-ansi-l1-1-0.dll", "CreateWindowExA",
     reinterpret_cast<void*>(&rt::window::create_window_a)},
    {"ext-ms-win-rtcore-ntuser-window-l1-1-0.dll", "CreateWindowExW",
     reinterpret_cast<void*>(&rt::window::create_window_w)},
    {"ext-ms-win-rtcore-ntuser-window-l1-1-0.dll", "ShowWindow", reinterpret_cast<void*>(&rt::window::show_window)},
    {"user32.dll", "CreateWindowExA", reinterpret_cast<void*>(&rt::window::create_window_a)},
    {"user32.dll", "CreateWindowExW", reinterpret_cast<void*>(&rt::window::create_window_w)},
    {"user32.dll", "ShowWindow", reinterpret_cast<void*>(&rt::window::show_window)},
);
