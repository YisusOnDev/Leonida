// LEONIDA_WATCH_XVA=<call>:<offset> (diagnostic): a hardware write watchpoint on the 8 bytes at <offset> (hex) into
// the region the <call>-th XMemVirtualAlloc (0-based, counting every call) returns. Every thread gets debug register
// DR0, re-applied every 20 ms for threads created later, and each hit goes to stderr with the thread, the value now
// there and the writer's stack (module+RVA) and general registers. Hits are logged and execution goes on, so the title's own writes (its
// constructor, its destructor) show up next to the one that corrupts it. Writes by the kernel (ReadFile into the
// buffer) or the GPU raise no hit: a corrupted value with no hit for it came from one of those.
#pragma once

#include <windows.h>
#include <tlhelp32.h>

#include <algorithm>
#include <atomic>
#include <cctype>
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

// LEONIDA_WATCH_DUMP="<expr>:<hex length>,...": memory each hit also prints. An expression is a register name
// followed by steps: +hex or -hex add to the address, @ replaces it with the qword stored there. Example:
// "r14:40,r15+1070@:80" dumps 0x40 bytes at r14 and 0x80 bytes at the pointer stored at r15+0x1070. Reads go through
// ReadProcessMemory, so an unmapped address prints "unreadable" instead of faulting inside the handler.
inline bool read_memory(uint64_t at, void* out, size_t n) {
    SIZE_T got = 0;
    return ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(at), out, n, &got) && got == n;
}

inline bool register_value(const CONTEXT& c, const std::string& name, uint64_t& out) {
    static const struct {
        const char* name;
        DWORD64 CONTEXT::*field;
    } regs[] = {{"rax", &CONTEXT::Rax}, {"rbx", &CONTEXT::Rbx}, {"rcx", &CONTEXT::Rcx}, {"rdx", &CONTEXT::Rdx},
                {"rsi", &CONTEXT::Rsi}, {"rdi", &CONTEXT::Rdi}, {"rbp", &CONTEXT::Rbp}, {"rsp", &CONTEXT::Rsp},
                {"r8", &CONTEXT::R8},   {"r9", &CONTEXT::R9},   {"r10", &CONTEXT::R10}, {"r11", &CONTEXT::R11},
                {"r12", &CONTEXT::R12}, {"r13", &CONTEXT::R13}, {"r14", &CONTEXT::R14}, {"r15", &CONTEXT::R15}};
    for (const auto& r : regs)
        if (name == r.name) return out = c.*r.field, true;
    return false;
}

inline void dump(int n, const CONTEXT& c) {
    static const std::string spec = [] {
        char buf[512]{};
        return GetEnvironmentVariableA("LEONIDA_WATCH_DUMP", buf, sizeof(buf)) ? std::string(buf) : std::string();
    }();
    size_t pos = 0;
    while (pos < spec.size()) {
        size_t end = spec.find(',', pos);
        if (end == std::string::npos) end = spec.size();
        const std::string item = spec.substr(pos, end - pos);
        pos = end + 1;
        const size_t colon = item.rfind(':');
        if (colon == std::string::npos) continue;
        const size_t length = std::min<size_t>(std::strtoull(item.c_str() + colon + 1, nullptr, 16), 0x400);
        size_t i = 0;
        while (i < colon && std::isalnum(static_cast<unsigned char>(item[i]))) ++i;
        uint64_t at = 0;
        bool ok = register_value(c, item.substr(0, i), at);
        while (ok && i < colon) {
            if (item[i] == '@') {
                ok = read_memory(at, &at, sizeof(at));
                ++i;
            } else if (item[i] == '+' || item[i] == '-') {
                char* stop = nullptr;
                const uint64_t delta = std::strtoull(item.c_str() + i + 1, &stop, 16);
                at = item[i] == '+' ? at + delta : at - delta;
                i = static_cast<size_t>(stop - item.c_str());
            } else {
                ok = false;
            }
        }
        std::string line = "[rt] watch: hit " + std::to_string(n) + " dump " + item.substr(0, colon) + " = ";
        char hex[24];
        std::snprintf(hex, sizeof(hex), "%llx:", static_cast<unsigned long long>(at));
        line += hex;
        uint8_t bytes[0x400];
        if (!ok || !read_memory(at, bytes, length)) {
            line += " unreadable";
        } else {
            for (size_t b = 0; b < length; ++b) {
                std::snprintf(hex, sizeof(hex), "%s%02x", b % 8 ? "" : " ", bytes[b]);
                line += hex;
            }
        }
        std::fprintf(stderr, "%s\n", line.c_str());
    }
}

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
                     rt::startup_trace::detail::stack_from(c).c_str());
        std::fprintf(stderr,
                     "[rt] watch: hit %d regs rax=%llx rbx=%llx rcx=%llx rdx=%llx rsi=%llx rdi=%llx rbp=%llx rsp=%llx "
                     "r8=%llx r9=%llx r10=%llx r11=%llx r12=%llx r13=%llx r14=%llx r15=%llx\n",
                     n, c.Rax, c.Rbx, c.Rcx, c.Rdx, c.Rsi, c.Rdi, c.Rbp, c.Rsp, c.R8, c.R9, c.R10, c.R11, c.R12, c.R13,
                     c.R14, c.R15);
        dump(n, c);
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
