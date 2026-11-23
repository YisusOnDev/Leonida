// A stand-in game EXE for RT-119, linked with a large /STACK (64 MiB, see runtime/CMakeLists.txt). Under
// `gta6rt --run`, its entry point runs on the thread run_entry creates, and the threads it creates go through the
// CreateThread and _beginthreadex shims or the native _beginthread. On Windows a size-0 thread inherits the main
// image's SizeOfStackReserve; a nonzero commit keeps that reserve too. The native path keeps the host policy.
// Test code; no game data.
#include <windows.h>

#include <process.h>

#include <cstdint>
#include <cstdio>

namespace {

constexpr ULONG_PTR kExpected = 64ull << 20;
int g_failures = 0;

void check(bool ok, const char* what) {
    std::printf("stack_fixture: %s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_failures;
}

void check_stack(const char* what, bool expect_image_reserve = true) {
    ULONG_PTR low = 0, high = 0;
    GetCurrentThreadStackLimits(&low, &high);
    const ULONG_PTR bytes = high > low ? high - low : 0;
    std::printf("stack_fixture: %s limits=[%p,%p] size=%llu MiB\n", what, reinterpret_cast<void*>(low),
                reinterpret_cast<void*>(high), static_cast<unsigned long long>(bytes >> 20));
    check(bytes > 0 && (bytes >= kExpected) == expect_image_reserve, what);
}

DWORD WINAPI thread_fn(void*) {
    check_stack("a size-0 CreateThread inherits the game image's stack reserve");
    return 0;
}

DWORD WINAPI commit_thread_fn(void*) {
    check_stack("a nonzero CreateThread commit keeps the game image's stack reserve");
    return 0;
}

unsigned __stdcall crt_thread_fn(void* arg) {
    check_stack(static_cast<const char*>(arg));
    return 0;
}

void check_beginthreadex(unsigned stack, char* label) {
    if (const uintptr_t h = _beginthreadex(nullptr, stack, crt_thread_fn, label, 0, nullptr)) {
        auto* handle = reinterpret_cast<HANDLE>(h);
        WaitForSingleObject(handle, INFINITE);
        CloseHandle(handle);
    } else {
        check(false, label);
    }
}

void __cdecl begin_thread_fn(void* release) {
    WaitForSingleObject(release, INFINITE);
    check_stack("a nonzero _beginthread commit uses the host stack policy", false);
    _endthread();
}

bool check_native_beginthread() {
    const HANDLE release = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (!release) return false;

    const uintptr_t thread = _beginthread(begin_thread_fn, 64u << 10, release);
    if (thread == static_cast<uintptr_t>(-1)) {
        CloseHandle(release);
        return false;
    }
    const auto handle = reinterpret_cast<HANDLE>(thread);
    const HANDLE process = GetCurrentProcess();
    HANDLE wait_handle = nullptr;
    // Duplicate the handle while the worker waits, then let _endthread close the original.
    if (!DuplicateHandle(process, handle, process, &wait_handle, 0, FALSE, DUPLICATE_SAME_ACCESS)) return false;
    SetEvent(release);
    const bool exited = WaitForSingleObject(wait_handle, 5000) == WAIT_OBJECT_0;
    DWORD flags = 0;
    const bool auto_closed = !GetHandleInformation(handle, &flags) && GetLastError() == ERROR_INVALID_HANDLE;
    CloseHandle(wait_handle);
    CloseHandle(release);
    return exited && auto_closed;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    check_stack("the entry point runs with the game image's stack reserve");

    HANDLE t = CreateThread(nullptr, 0, thread_fn, nullptr, 0, nullptr);
    if (!t) {
        check(false, "CreateThread with size 0 succeeded");
    } else {
        WaitForSingleObject(t, INFINITE);
        CloseHandle(t);
    }

    HANDLE t2 = CreateThread(nullptr, 64u << 10, commit_thread_fn, nullptr, 0, nullptr);
    if (!t2) {
        check(false, "CreateThread with a nonzero commit succeeded");
    } else {
        WaitForSingleObject(t2, INFINITE);
        CloseHandle(t2);
    }

    char crt_default[] = "a size-0 _beginthreadex inherits the game image's stack reserve";
    char crt_commit[] = "a nonzero _beginthreadex commit keeps the game image's stack reserve";
    check_beginthreadex(0, crt_default);
    check_beginthreadex(64u << 10, crt_commit);

    if (!check_native_beginthread()) {
        std::puts("stack_fixture: FAIL native _endthread terminated and closed its original handle");
        return 1;
    }
    check(true, "native _endthread terminated and closed its original handle");

    std::printf("stack_fixture: %s\n", g_failures ? "FAILED" : "OK");
    std::fflush(stdout);
    return g_failures ? 1 : 0;
}
