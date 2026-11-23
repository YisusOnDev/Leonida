// LEONIDA_WATCH_XVA=<call>:<offset> (diagnostic): a hardware write watchpoint on the 8 bytes at <offset> (hex) into
// the region the <call>-th XMemVirtualAlloc (0-based, counting every call) returns. Every thread gets debug register
// DR0, re-applied every 20 ms for threads created later, and each hit goes to stderr with the thread, the value now
// there and the writer's stack (module+RVA). Hits are logged and execution goes on, so the title's own writes (its
// constructor, its destructor) show up next to the one that corrupts it. Writes by the kernel (ReadFile into the
// buffer) or the GPU raise no hit: a corrupted value with no hit for it came from one of those.
#pragma once

#include <windows.h>
#include <tlhelp32.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cwchar>
#include <mutex>
#include <set>
#include <string>

#include "startup_trace.h"

namespace rt::write_watch {

namespace detail {
inline std::atomic<uint64_t> address{0};
inline std::atomic<int> hits{0};

inline LONG CALLBACK handler(EXCEPTION_POINTERS* info) {
    if (!info || info->ExceptionRecord->ExceptionCode != EXCEPTION_SINGLE_STEP) return EXCEPTION_CONTINUE_SEARCH;
    CONTEXT& c = *info->ContextRecord;
    if (!(c.Dr6 & 1)) return EXCEPTION_CONTINUE_SEARCH;
    c.Dr6 = 0;
    const uint64_t at = address.load();
    const int n = ++hits;
    if (n <= 400) {
        const uint64_t value = at ? *reinterpret_cast<volatile uint64_t*>(at) : 0;
        std::fprintf(stderr, "[rt] watch: hit %d thread %lu value %016llx after %s stack %s\n", n, GetCurrentThreadId(),
                     static_cast<unsigned long long>(value), rt::startup_trace::where(c.Rip).c_str(),
                     rt::startup_trace::stack_from(c).c_str());
        std::fflush(stderr);
    }
    return EXCEPTION_CONTINUE_EXECUTION;
}

inline void arm(HANDLE thread, uint64_t at) {
    CONTEXT c{};
    c.ContextFlags = CONTEXT_DEBUG_REGISTERS;
    if (SuspendThread(thread) == static_cast<DWORD>(-1)) return;
    if (GetThreadContext(thread, &c)) {
        c.Dr0 = at;
        // L0 enable, R/W0 = 01 (write), LEN0 = 10 (8 bytes).
        c.Dr7 = (c.Dr7 & ~((uint64_t{0xf} << 16) | 3)) | 1 | (uint64_t{1} << 16) | (uint64_t{2} << 18);
        SetThreadContext(thread, &c);
    }
    ResumeThread(thread);
}

inline DWORD WINAPI armer(void*) {
    std::set<DWORD> armed;
    const DWORD self = GetCurrentThreadId();
    for (;;) {
        const uint64_t at = address.load();
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
                    arm(t, at);
                    CloseHandle(t);
                }
            }
            CloseHandle(snap);
        }
        Sleep(20);
    }
}
}  // namespace detail

// Called with every XMemVirtualAlloc result, in call order.
inline void note_region(void* result) {
    static std::atomic<int> calls{0};
    static const auto config = [] {
        struct Config {
            int call = -1;
            uint64_t offset = 0;
        } c;
        wchar_t buf[64]{};
        if (GetEnvironmentVariableW(L"LEONIDA_WATCH_XVA", buf, 64)) {
            wchar_t* end = nullptr;
            c.call = static_cast<int>(std::wcstol(buf, &end, 10));
            if (end && *end == L':') c.offset = std::wcstoull(end + 1, nullptr, 16);
        }
        return c;
    }();
    const int index = calls++;
    if (config.call < 0 || index != config.call || !result) return;
    const uint64_t at = reinterpret_cast<uint64_t>(result) + config.offset;
    detail::address.store(at);
    AddVectoredExceptionHandler(1, &detail::handler);
    std::fprintf(stderr, "[rt] watch: XMemVirtualAlloc call %d returned %p; watching writes to %llx\n", index, result,
                 static_cast<unsigned long long>(at));
    const HANDLE t = CreateThread(nullptr, 0, &detail::armer, nullptr, 0, nullptr);
    if (t) CloseHandle(t);
}

}  // namespace rt::write_watch
