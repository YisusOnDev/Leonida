// LEONIDA_THREAD_SAMPLE=<s>[,<s>...] (diagnostic, with --trace): at each of those seconds after the trace opened,
// every other thread of the process is suspended in turn, its stack (module+RVA, at most 24 frames) captured into a
// fixed array while it is stopped, and printed to stderr after it resumes. Two samples some seconds apart tell a
// thread blocked in a wait (same stack) from one spinning (different stacks). Every 5 s a "[rt] tick" line also goes
// to stderr, so the d3d12 trace lines between two ticks can be placed in time.
//
// Nothing is allocated while a thread is suspended: the suspended thread may hold the heap lock.
#pragma once

#include <windows.h>
#include <tlhelp32.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <string>
#include <thread>
#include <vector>

#include "startup_trace.h"
#include "write_watch.h"

namespace rt::thread_sample {

namespace detail {
constexpr int kFrames = 24;

inline int capture(HANDLE thread, uint64_t (&frames)[kFrames]) {
    CONTEXT c{};
    c.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER;
    if (SuspendThread(thread) == static_cast<DWORD>(-1)) return -1;
    int n = 0;
    if (GetThreadContext(thread, &c)) {
        for (; n < kFrames && c.Rip; ++n) {
            frames[n] = c.Rip;
            DWORD64 image_base = 0;
            PRUNTIME_FUNCTION fn = RtlLookupFunctionEntry(c.Rip, &image_base, nullptr);
            if (!fn) {
                MEMORY_BASIC_INFORMATION m{};
                if (!c.Rsp || !VirtualQuery(reinterpret_cast<void*>(c.Rsp), &m, sizeof(m)) || m.State != MEM_COMMIT ||
                    (m.Protect & (PAGE_READWRITE | PAGE_READONLY)) == 0)
                    { ++n; break; }
                c.Rip = *reinterpret_cast<DWORD64*>(c.Rsp);
                c.Rsp += 8;
                continue;
            }
            void* handler_data = nullptr;
            DWORD64 frame = 0;
            RtlVirtualUnwind(UNW_FLAG_NHANDLER, image_base, c.Rip, fn, &c, &handler_data, &frame, nullptr);
        }
    }
    ResumeThread(thread);
    return n;
}

// LEONIDA_PEEK="<hex addr>[@]:<hex len>,...": memory printed at every sample; a trailing @ reads the qword at the
// address and dumps the memory it points to instead.
inline void peek(double t) {
    static const std::string spec = [] {
        char buf[512]{};
        return GetEnvironmentVariableA("LEONIDA_PEEK", buf, sizeof(buf)) ? std::string(buf) : std::string();
    }();
    size_t pos = 0;
    while (pos < spec.size()) {
        size_t end = spec.find(',', pos);
        if (end == std::string::npos) end = spec.size();
        const std::string item = spec.substr(pos, end - pos);
        pos = end + 1;
        const size_t colon = item.find(':');
        if (colon == std::string::npos) continue;
        std::string a = item.substr(0, colon);
        const bool deref = !a.empty() && a.back() == '@';
        if (deref) a.pop_back();
        uint64_t at = std::strtoull(a.c_str(), nullptr, 16);
        const size_t len = std::min<size_t>(std::strtoull(item.c_str() + colon + 1, nullptr, 16), 0x400);
        SIZE_T got = 0;
        if (deref && !ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(at), &at, 8, &got)) at = 0;
        std::vector<uint8_t> bytes(len);
        const bool ok = at && ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(at), bytes.data(),
                                                len, &got) && got == len;
        std::string line = "[rt] peek t=" + std::to_string(static_cast<int>(t)) + " " + item + " -> " +
                           startup_trace::hex(at) + ":";
        if (!ok) line += " unreadable";
        for (size_t i = 0; ok && i < len; i += 8) {
            uint64_t q = 0;
            std::memcpy(&q, bytes.data() + i, std::min<size_t>(8, len - i));
            char b[24];
            std::snprintf(b, sizeof(b), "%s%016llx", i % 64 ? " " : "\n   ", static_cast<unsigned long long>(q));
            line += b;
        }
        std::fprintf(stderr, "%s\n", line.c_str());
    }
}

inline void sample(double t) {
    const DWORD self = GetCurrentThreadId(), pid = GetCurrentProcessId();
    std::vector<DWORD> ids;
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snap == INVALID_HANDLE_VALUE) return;
    THREADENTRY32 te{sizeof(te)};
    for (BOOL ok = Thread32First(snap, &te); ok; ok = Thread32Next(snap, &te))
        if (te.th32OwnerProcessID == pid && te.th32ThreadID != self) ids.push_back(te.th32ThreadID);
    CloseHandle(snap);
    std::fprintf(stderr, "[rt] thread-sample t=%.1f s: %zu threads\n", t, ids.size());
    for (DWORD id : ids) {
        HANDLE h = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_LIMITED_INFORMATION, FALSE, id);
        if (!h) continue;
        uint64_t frames[kFrames]{};
        const int n = capture(h, frames);
        std::string name;
        PWSTR desc = nullptr;
        if (SUCCEEDED(GetThreadDescription(h, &desc)) && desc) {
            name = startup_trace::detail::utf8_from_wide(desc, static_cast<int>(wcslen(desc)));
            LocalFree(desc);
        }
        CloseHandle(h);
        std::string line = "[rt] thread-sample t=" + std::to_string(static_cast<int>(t)) + " tid " + std::to_string(id) +
                           " \"" + name + "\":";
        for (int i = 0; i < n; ++i) line += " " + startup_trace::where(frames[i]);
        std::fprintf(stderr, "%s\n", line.c_str());
    }
    std::fflush(stderr);
}
}  // namespace detail

// Called once the trace is open.
inline void start() {
    char buf[256]{};
    std::vector<double> at;
    if (GetEnvironmentVariableA("LEONIDA_THREAD_SAMPLE", buf, sizeof(buf))) {
        for (char* p = buf; *p;) {
            char* end = nullptr;
            const double v = std::strtod(p, &end);
            if (end == p) break;
            at.push_back(v);
            p = *end == ',' ? end + 1 : end;
        }
    }
    if (GetEnvironmentVariableA("LEONIDA_WATCH_ADDR", buf, sizeof(buf)))
        write_watch::watch_absolute(std::strtoull(buf, nullptr, 16));
    std::thread([at] {
        const auto t0 = std::chrono::steady_clock::now();
        size_t next = 0;
        for (int tick = 1;; ++tick) {
            std::this_thread::sleep_until(t0 + std::chrono::seconds(tick));
            const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (tick % 5 == 0) std::fprintf(stderr, "[rt] tick t=%d s\n", tick);
            while (next < at.size() && t >= at[next]) detail::peek(t), detail::sample(t), ++next;
        }
    }).detach();
}

}  // namespace rt::thread_sample
