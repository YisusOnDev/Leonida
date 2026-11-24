// LEONIDA_EXEC_BP=<hex address>[,...] (diagnostic, up to four): hardware execute breakpoints in debug registers
// DR0-DR3 of every thread (re-applied every 20 ms for threads created later). The first three hits of each address
// go to stderr with the thread, rcx/rdx and the stack (module+RVA); later hits are only counted, and the counts are
// printed at every LEONIDA_THREAD_SAMPLE. Tells "this code never runs" from "runs and does not finish". It uses the
// same debug registers as LEONIDA_WATCH_XVA / LEONIDA_WATCH_ADDR: use one or the other.
#pragma once

#include <windows.h>
#include <tlhelp32.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <set>

#include "startup_trace.h"

namespace rt::exec_watch {

namespace detail {
inline uint64_t at[4]{};
inline int count = 0;
inline std::atomic<uint64_t> hits[4]{};

inline LONG CALLBACK handler(EXCEPTION_POINTERS* info) {
    if (!info || info->ExceptionRecord->ExceptionCode != EXCEPTION_SINGLE_STEP) return EXCEPTION_CONTINUE_SEARCH;
    CONTEXT& c = *info->ContextRecord;
    int slot = -1;
    for (int i = 0; i < count; ++i)
        if ((c.Dr6 & (uint64_t{1} << i)) && c.Rip == at[i]) slot = i;
    if (slot < 0) return EXCEPTION_CONTINUE_SEARCH;
    c.Dr6 = 0;
    c.EFlags |= 0x10000;  // RF: the instruction runs once without faulting again
    const uint64_t n = ++hits[slot];
    if (n <= 3) {
        std::fprintf(stderr, "[rt] exec-bp: %s hit %llu thread %lu rcx=%llx rdx=%llx r8=%llx stack %s\n",
                     rt::startup_trace::where(c.Rip).c_str(), static_cast<unsigned long long>(n), GetCurrentThreadId(),
                     c.Rcx, c.Rdx, c.R8, rt::startup_trace::detail::stack_from(c).c_str());
        std::fflush(stderr);
    }
    return EXCEPTION_CONTINUE_EXECUTION;
}

inline void arm(HANDLE thread) {
    CONTEXT c{};
    c.ContextFlags = CONTEXT_DEBUG_REGISTERS;
    if (SuspendThread(thread) == static_cast<DWORD>(-1)) return;
    if (GetThreadContext(thread, &c)) {
        DWORD64* dr[4] = {&c.Dr0, &c.Dr1, &c.Dr2, &c.Dr3};
        uint64_t dr7 = c.Dr7;
        for (int i = 0; i < count; ++i) {
            *dr[i] = at[i];
            dr7 &= ~((uint64_t{0xf} << (16 + 4 * i)) | (uint64_t{3} << (2 * i)));  // R/W = 00 (execute), LEN = 00
            dr7 |= uint64_t{1} << (2 * i);                                           // local enable
        }
        c.Dr7 = dr7;
        SetThreadContext(thread, &c);
    }
    ResumeThread(thread);
}

inline DWORD WINAPI armer(void*) {
    std::set<DWORD> armed;
    const DWORD self = GetCurrentThreadId();
    for (;;) {
        const HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
        if (snap != INVALID_HANDLE_VALUE) {
            THREADENTRY32 e{};
            e.dwSize = sizeof(e);
            for (BOOL ok = Thread32First(snap, &e); ok; ok = Thread32Next(snap, &e)) {
                if (e.th32OwnerProcessID != GetCurrentProcessId() || e.th32ThreadID == self) continue;
                if (!armed.insert(e.th32ThreadID).second) continue;
                const HANDLE t = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_SET_CONTEXT, FALSE,
                                            e.th32ThreadID);
                if (t) {
                    arm(t);
                    CloseHandle(t);
                }
            }
            CloseHandle(snap);
        }
        Sleep(20);
    }
}
}  // namespace detail

inline void start() {
    char buf[256]{};
    if (!GetEnvironmentVariableA("LEONIDA_EXEC_BP", buf, sizeof(buf))) return;
    for (char* p = buf; *p && detail::count < 4;) {
        char* end = nullptr;
        const uint64_t v = std::strtoull(p, &end, 16);
        if (end == p) break;
        detail::at[detail::count++] = v;
        p = *end == ',' ? end + 1 : end;
    }
    if (!detail::count) return;
    AddVectoredExceptionHandler(1, &detail::handler);
    for (int i = 0; i < detail::count; ++i)
        std::fprintf(stderr, "[rt] exec-bp: armed %llx\n", static_cast<unsigned long long>(detail::at[i]));
    const HANDLE t = CreateThread(nullptr, 0, &detail::armer, nullptr, 0, nullptr);
    if (t) CloseHandle(t);
}

inline void report(double t) {
    for (int i = 0; i < detail::count; ++i)
        std::fprintf(stderr, "[rt] exec-bp: t=%d %s hits %llu\n", static_cast<int>(t),
                     rt::startup_trace::where(detail::at[i]).c_str(),
                     static_cast<unsigned long long>(detail::hits[i].load()));
}

}  // namespace rt::exec_watch
