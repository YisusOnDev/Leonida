#pragma once
// Where a run's windows open, so a title can be tested on one monitor of a multi-monitor desktop while the person
// keeps using the others. Two environment variables, read once:
//
//   LEONIDA_WINDOW_MONITOR=<n>    the monitor: its 1-based position in EnumDisplayMonitors order, or its device name
//                                 ("DISPLAY2" or "\\.\DISPLAY2"). A top-level window is moved into that monitor's work
//                                 area, centred, and shrunk to fit. A name or number no monitor has changes nothing.
//   LEONIDA_WINDOW_NO_ACTIVATE=1  windows are shown without being activated: the run never takes the keyboard focus
//                                 (and with it the keyboard and mouse input) from whatever the person is using.
//
// Unset, nothing changes. The title's CreateWindowExA/W and ShowWindow imports go through window_placement_shims.cpp;
// the runtime's own windows call place() and show_command() (window_placement.cpp) themselves.
#include <windows.h>

namespace rt::window {

// The target monitor's work area, or false when LEONIDA_WINDOW_MONITOR is unset or names no monitor.
bool target_work_area(RECT* out);

// Whether LEONIDA_WINDOW_NO_ACTIVATE is set.
bool no_activate();

// Moves a top-level window into the target monitor's work area (no-op without a target or for a child window).
// Never activates or reorders it.
void place(HWND window);

// `command` for ShowWindow, with activation taken out when no_activate(): SW_SHOWNORMAL and SW_SHOWDEFAULT become
// SW_SHOWNOACTIVATE, SW_SHOW and SW_SHOWMAXIMIZED become SW_SHOWNA (a maximized window is sized to the work area by
// place() instead), SW_RESTORE becomes SW_SHOWNOACTIVATE. Hiding and minimizing commands are unchanged.
int show_command(int command);

}  // namespace rt::window
