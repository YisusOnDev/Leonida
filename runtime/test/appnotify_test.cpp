// Tests for runtime/shims/appnotify.cpp through rt::shims::find.
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

#include "shims.h"

namespace {

using NotifyCallback = void(WINAPI*)(BOOLEAN, PVOID);
using RegisterNotifyFn = DWORD(WINAPI*)(NotifyCallback, PVOID, PVOID*);
using UnregisterNotifyFn = void(WINAPI*)(PVOID);
using WerRegisterFileFn = HRESULT(WINAPI*)(PCWSTR, int, DWORD);
using WerUnregisterFileFn = HRESULT(WINAPI*)(PCWSTR);

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

template <typename Fn>
Fn resolve(const char* dll, const char* name) {
    return reinterpret_cast<Fn>(rt::shims::find(dll, name));
}

void WINAPI dummy_notify(BOOLEAN, PVOID) {}

void test_registry_coverage() {
    for (const char* dll : {"kernel32.dll", "kernelbase.dll", "twinapi.appcore.dll",
                            "api-ms-win-core-psm-appnotify-l1-1-0.dll",
                            "api-ms-win-core-psm-appnotify-l1-1-1.dll"}) {
        check(resolve<RegisterNotifyFn>(dll, "RegisterAppStateChangeNotification") != nullptr,
              "RegisterAppStateChangeNotification resolves");
        check(resolve<UnregisterNotifyFn>(dll, "UnregisterAppStateChangeNotification") != nullptr,
              "UnregisterAppStateChangeNotification resolves");
        check(resolve<RegisterNotifyFn>(dll, "RegisterAppConstrainedChangeNotification") != nullptr,
              "RegisterAppConstrainedChangeNotification resolves");
        check(resolve<UnregisterNotifyFn>(dll, "UnregisterAppConstrainedChangeNotification") != nullptr,
              "UnregisterAppConstrainedChangeNotification resolves");
    }

    for (const char* dll : {"kernel32.dll", "kernelbase.dll", "wer.dll",
                            "api-ms-win-core-windowserrorreporting-l1-1-0.dll",
                            "api-ms-win-core-windowserrorreporting-l1-1-1.dll"}) {
        check(resolve<WerRegisterFileFn>(dll, "WerRegisterFile") != nullptr, "WerRegisterFile resolves");
        check(resolve<WerUnregisterFileFn>(dll, "WerUnregisterFile") != nullptr, "WerUnregisterFile resolves");
    }
}

void test_appnotify_registrations() {
    auto reg_state = resolve<RegisterNotifyFn>("api-ms-win-core-psm-appnotify-l1-1-0.dll",
                                               "RegisterAppStateChangeNotification");
    auto unreg_state = resolve<UnregisterNotifyFn>("api-ms-win-core-psm-appnotify-l1-1-0.dll",
                                                   "UnregisterAppStateChangeNotification");
    auto reg_constrained = resolve<RegisterNotifyFn>("api-ms-win-core-psm-appnotify-l1-1-1.dll",
                                                     "RegisterAppConstrainedChangeNotification");
    auto unreg_constrained = resolve<UnregisterNotifyFn>("api-ms-win-core-psm-appnotify-l1-1-1.dll",
                                                         "UnregisterAppConstrainedChangeNotification");

    check(reg_state(dummy_notify, nullptr, nullptr) == ERROR_INVALID_PARAMETER,
          "RegisterAppStateChangeNotification rejects null registration pointer");
    check(reg_constrained(dummy_notify, nullptr, nullptr) == ERROR_INVALID_PARAMETER,
          "RegisterAppConstrainedChangeNotification rejects null registration pointer");

    void* sentinel = reinterpret_cast<void*>(static_cast<uintptr_t>(0xdeadbeef));
    void* out = sentinel;
    check(reg_state(nullptr, nullptr, &out) == ERROR_INVALID_PARAMETER && out == nullptr,
          "RegisterAppStateChangeNotification rejects null callback and zeroes out handle");

    out = sentinel;
    check(reg_constrained(nullptr, nullptr, &out) == ERROR_INVALID_PARAMETER && out == nullptr,
          "RegisterAppConstrainedChangeNotification rejects null callback and zeroes out handle");

    void* state_handle = sentinel;
    void* constrained_handle = sentinel;
    int ctx = 42;
    check(reg_state(dummy_notify, &ctx, &state_handle) == ERROR_SUCCESS &&
              state_handle != nullptr && state_handle != sentinel,
          "RegisterAppStateChangeNotification returns non-null handle");
    check(reg_constrained(dummy_notify, &ctx, &constrained_handle) == ERROR_SUCCESS &&
              constrained_handle != nullptr && constrained_handle != sentinel &&
              constrained_handle != state_handle,
          "RegisterAppConstrainedChangeNotification returns distinct non-null handle");

    // Unregister nullptr, foreign pointer, and cross-kind handles must not fault.
    unreg_state(nullptr);
    unreg_constrained(nullptr);
    unreg_state(reinterpret_cast<void*>(static_cast<uintptr_t>(0x1234)));
    unreg_constrained(reinterpret_cast<void*>(static_cast<uintptr_t>(0x1234)));
    unreg_state(constrained_handle);
    unreg_constrained(state_handle);

    // Real unregister and double-unregister are safe.
    unreg_state(state_handle);
    unreg_state(state_handle);
    unreg_constrained(constrained_handle);
    unreg_constrained(constrained_handle);
    check(true, "unregister handles null, foreign, cross-kind, and double-unregister cleanly");

    // Concurrent multi-thread registration/unregistration.
    std::vector<std::thread> workers;
    bool thread_ok[4] = {false, false, false, false};
    for (int t = 0; t < 4; ++t) {
        workers.emplace_back([&, t]() {
            bool ok = true;
            for (int i = 0; i < 64; ++i) {
                void* h1 = nullptr;
                void* h2 = nullptr;
                if (reg_state(dummy_notify, nullptr, &h1) != ERROR_SUCCESS || !h1) ok = false;
                if (reg_constrained(dummy_notify, nullptr, &h2) != ERROR_SUCCESS || !h2 || h1 == h2) ok = false;
                unreg_state(h1);
                unreg_constrained(h2);
            }
            thread_ok[t] = ok;
        });
    }
    for (auto& w : workers) w.join();
    check(thread_ok[0] && thread_ok[1] && thread_ok[2] && thread_ok[3],
          "concurrent register/unregister across threads succeeds");
}

void test_wer_file_registration() {
    auto wer_reg = resolve<WerRegisterFileFn>("api-ms-win-core-windowserrorreporting-l1-1-0.dll", "WerRegisterFile");
    auto wer_unreg =
        resolve<WerUnregisterFileFn>("api-ms-win-core-windowserrorreporting-l1-1-0.dll", "WerUnregisterFile");

    check(wer_reg(nullptr, 1, 2) == E_INVALIDARG, "WerRegisterFile rejects nullptr path");
    check(wer_reg(L"GAME:\\log.txt", 0, 2) == E_INVALIDARG, "WerRegisterFile rejects file type 0");
    check(wer_reg(L"GAME:\\log.txt", 3, 2) == E_INVALIDARG, "WerRegisterFile rejects file type 3");
    check(wer_reg(L"GAME:\\log.txt", 1, 0x10) == E_INVALIDARG, "WerRegisterFile rejects unknown flags");

    const HRESULT invalid_name = HRESULT_FROM_WIN32(ERROR_INVALID_NAME);
    const HRESULT not_found = HRESULT_FROM_WIN32(ERROR_NOT_FOUND);

    check(wer_reg(L"GAME:\\..\\escape.txt", 2, 2) == invalid_name,
          "WerRegisterFile refuses GAME: traversal past root");
    check(wer_reg(L"G:/../escape.txt", 2, 2) == invalid_name, "WerRegisterFile refuses G: traversal past root");
    check(wer_reg(L"T:\\sub\\..\\..\\escape.txt", 2, 2) == invalid_name,
          "WerRegisterFile refuses T: nested traversal past root");
    check(wer_unreg(nullptr) == E_INVALIDARG, "WerUnregisterFile rejects nullptr");
    check(wer_unreg(L"GAME:\\..\\escape.txt") == invalid_name,
          "WerUnregisterFile refuses traversal past root");
    check(wer_unreg(L"GAME:\\missing.txt") == not_found,
          "WerUnregisterFile returns ERROR_NOT_FOUND for unregistered file");

    check(wer_reg(L"GAME:\\logs\\boot.txt", 2, 2) == S_OK, "WerRegisterFile accepts GAME: path");
    check(wer_reg(L"game:/LOGS/boot.txt", 1, 1) == S_OK, "WerRegisterFile updates matching normalized entry");
    check(wer_unreg(L"GAME:\\logs\\BOOT.TXT") == S_OK, "WerUnregisterFile removes normalized entry");
    check(wer_unreg(L"GAME:\\logs\\boot.txt") == not_found,
          "WerUnregisterFile reports ERROR_NOT_FOUND after removal");

    check(wer_reg(L"T:\\sub\\..\\crash.dmp", 2, 3) == S_OK, "WerRegisterFile accepts T: dot-dot within root");
    check(wer_unreg(L"t:/crash.dmp") == S_OK, "WerUnregisterFile matches normalized T: path");

    check(wer_reg(L"GAME:\\", 2, 2) == S_OK, "WerRegisterFile accepts console drive root");
    check(wer_unreg(L"game:\\sub\\..") == S_OK, "WerUnregisterFile normalizes trailing dot-dot to root");

    check(wer_reg(L"C:\\host\\trace.log", 2, 2) == S_OK, "WerRegisterFile accepts host path");
    check(wer_unreg(L"c:/HOST/trace.log") == S_OK, "WerUnregisterFile matches normalized host path");
}

}  // namespace

int main() {
    test_registry_coverage();
    test_appnotify_registrations();
    test_wer_file_registration();
    std::printf("%d check(s), %d failure(s)\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
