// The game's command line, exit and stack, for `gta6rt --run` (shims/process.cpp).
#pragma once
#include <cstddef>
#include <string>
#include <vector>

namespace rt::process {

// Set GetCommandLine to `program` then `args`, quoted where needed, before process-attach callbacks. The wide text
// holds the arguments exactly; the narrow text uses the ANSI code page (RT-84). Reset CRT argv to argc=0/null:
// its startup configuration selects no arguments, the original arguments, or wildcard expansion for each width.
void set_command_line(const std::wstring& program, const std::vector<std::wstring>& args);

// The mapped main image's OptionalHeader.SizeOfStackReserve, read by the loader before any image code runs. Windows
// gives a size-0 thread the main image's reserve; under gta6rt the host EXE's reserve would be used, so the
// CreateThread/_beginthreadex shims and run_entry use the larger of this and kDefaultStackReserve instead (0 means
// the image named none). The 8 MB floor is BUILD-1457's guarantee for every thread that asks for no size.
void set_thread_stack(std::size_t reserve_bytes);

inline constexpr std::size_t kDefaultStackReserve = 8u << 20;  // gta6rt's /STACK default: a floor, not just a fallback

// Runs a mapped EXE's entry point on a new thread with `stack_bytes` of reserved stack (game_stack_reserve() when
// this is 0) and returns the game's exit code: the entry point's return value, or what it
// passed to exit/ExitProcess (3 for abort). Returns when the first of those happens; other game threads are left
// running (the host process ends them when it exits).
int run_entry(void* entry, std::size_t stack_bytes = 0);

}  // namespace rt::process
