// RT-119: the stack shims in shims/process.cpp are served under the DLL names a title imports them by, substitute
// the mapped image's reserve for a size of 0 (and for a nonzero commit without the reservation flag), and leave a
// caller's explicit reservation alone. No game files, no GPU.
#include <windows.h>

#include <cstdint>
#include <cstdio>

#include "process_shim.h"
#include "shims.h"

namespace {

constexpr unsigned long long kReserveMib = 16;
int g_failures = 0;

void check(bool ok, const char* what) {
    std::printf("stack_shim: %s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_failures;
}

struct Result {
    bool ok = false;
    unsigned long long mib = 0;
};

void measure(Result& r) {
    ULONG_PTR low = 0, high = 0;
    GetCurrentThreadStackLimits(&low, &high);
    r.mib = high > low ? (high - low) >> 20 : 0;
}

DWORD WINAPI large_thread(void* p) {
    auto* r = static_cast<Result*>(p);
    measure(*r);
    r->ok = r->mib >= kReserveMib;
    return 0;
}

DWORD WINAPI small_thread(void* p) {
    auto* r = static_cast<Result*>(p);
    measure(*r);
    r->ok = r->mib >= kReserveMib;  // a nonzero commit still grows the reserve to the image's
    return 0;
}

DWORD WINAPI flagged_thread(void* p) {
    auto* r = static_cast<Result*>(p);
    measure(*r);
    r->ok = r->mib > 0 && r->mib < kReserveMib;  // explicit reservation: left alone
    return 0;
}

unsigned __stdcall measure_crt_thread(void* p) {
    auto* r = static_cast<Result*>(p);
    measure(*r);
    return 0;
}

}  // namespace

int main() {
    // Every name the loader can see must hand out the same shim (the registry matches DLL names exactly).
    void* kernel_create = rt::shims::find("kernel32.dll", "CreateThread");
    check(kernel_create != nullptr, "kernel32.dll!CreateThread is shimmed");
    for (const char* dll : {"kernel32.dll", "kernelbase.dll", "api-ms-win-core-processthreads-l1-1-0.dll",
                            "api-ms-win-core-processthreads-l1-1-1.dll", "api-ms-win-core-processthreads-l1-1-2.dll",
                            "api-ms-win-core-processthreads-l1-1-3.dll"})
        check(rt::shims::find(dll, "CreateThread") == kernel_create, "CreateThread is shimmed under every name");

    void* ucrt_begin = rt::shims::find("api-ms-win-crt-runtime-l1-1-0.dll", "_beginthreadex");
    check(ucrt_begin != nullptr, "api-ms-win-crt-runtime-l1-1-0.dll!_beginthreadex is shimmed");
    for (const char* dll : {"api-ms-win-crt-runtime-l1-1-0.dll", "ucrtbase.dll", "ucrtbased.dll", "msvcrt.dll"})
        check(rt::shims::find(dll, "_beginthreadex") == ucrt_begin, "_beginthreadex is shimmed under every name");

    for (const char* dll : {"api-ms-win-crt-runtime-l1-1-0.dll", "ucrtbase.dll", "ucrtbased.dll", "msvcrt.dll"})
        check(rt::shims::find(dll, "_beginthread") == nullptr, "_beginthread stays in the native CRT");

    rt::process::set_thread_stack(static_cast<std::size_t>(kReserveMib << 20));
    if (!kernel_create || !ucrt_begin) return 1;

    using CreateThreadFn = HANDLE(WINAPI*)(LPSECURITY_ATTRIBUTES, SIZE_T, LPTHREAD_START_ROUTINE, LPVOID, DWORD,
                                           LPDWORD);
    auto create = reinterpret_cast<CreateThreadFn>(kernel_create);

    Result large;
    if (HANDLE h = create(nullptr, 0, large_thread, &large, 0, nullptr)) {
        WaitForSingleObject(h, INFINITE);
        CloseHandle(h);
    }
    std::printf("stack_shim: size-0 CreateThread stack = %llu MiB\n", large.mib);
    check(large.ok, "size 0 gets the game image's reserve");

    Result small;
    if (HANDLE h = create(nullptr, 64u << 10, small_thread, &small, 0, nullptr)) {
        WaitForSingleObject(h, INFINITE);
        CloseHandle(h);
    }
    std::printf("stack_shim: nonzero CreateThread stack = %llu MiB\n", small.mib);
    check(small.ok, "a nonzero commit still grows the reserve to the image's");

    Result flagged;
    if (HANDLE h = create(nullptr, 1u << 20, flagged_thread, &flagged, STACK_SIZE_PARAM_IS_A_RESERVATION, nullptr)) {
        WaitForSingleObject(h, INFINITE);
        CloseHandle(h);
    }
    std::printf("stack_shim: flagged CreateThread stack = %llu MiB\n", flagged.mib);
    check(flagged.ok, "an explicit reservation is left alone");

    using BeginThreadExFn = uintptr_t(__cdecl*)(void*, unsigned, unsigned(__stdcall*)(void*), void*, unsigned,
                                               unsigned*);
    auto begin_ex = reinterpret_cast<BeginThreadExFn>(ucrt_begin);
    const auto check_crt = [begin_ex](unsigned stack, unsigned flags, unsigned long long expected_mib,
                                     const char* label) {
        Result result;
        unsigned thread_id = 0;
        if (const uintptr_t thread = begin_ex(nullptr, stack, measure_crt_thread, &result, flags, &thread_id)) {
            auto* handle = reinterpret_cast<HANDLE>(thread);
            check(thread_id != 0 && thread_id == GetThreadId(handle), "_beginthreadex reports the thread ID");
            if (flags & CREATE_SUSPENDED)
                check(ResumeThread(handle) == 1, "_beginthreadex preserves CREATE_SUSPENDED");
            WaitForSingleObject(handle, INFINITE);
            CloseHandle(handle);
        }
        std::printf("stack_shim: _beginthreadex stack = %llu MiB\n", result.mib);
        check(result.mib == expected_mib, label);
    };
    check_crt(0, 0, kReserveMib, "size 0 through _beginthreadex gets the game image's reserve");
    check_crt(64u << 10, CREATE_SUSPENDED, kReserveMib, "a small _beginthreadex commit keeps the image reserve");
    check_crt((static_cast<unsigned>(kReserveMib) << 20) + 1u, 0, kReserveMib + 1,
              "a larger _beginthreadex commit rounds up to 1 MiB");
    check_crt(1u << 20, STACK_SIZE_PARAM_IS_A_RESERVATION, 1, "an explicit _beginthreadex reserve is left alone");

    std::printf("stack_shim: %s\n", g_failures ? "FAILED" : "OK");
    return g_failures ? 1 : 0;
}
