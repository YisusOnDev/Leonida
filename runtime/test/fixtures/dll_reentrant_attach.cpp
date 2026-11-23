// Two DLLs that load each other during process attach.
#include <windows.h>

namespace {
#ifdef ATTACH_DLL_A
constexpr auto peer_name = L"dll_attach_b.dll";
constexpr int detach_id = 1;
#else
constexpr auto peer_name = L"dll_attach_a.dll";
constexpr int detach_id = 2;
#endif
int process_attaches;
HMODULE peer;
int* detach_log;
}

extern "C" BOOL WINAPI dll_attach_main(HINSTANCE, DWORD reason, LPVOID) {
    // Stop a broken loader's recursion so the test can report duplicate attaches.
    if (reason == DLL_PROCESS_ATTACH && ++process_attaches == 1) {
        peer = LoadLibraryW(peer_name);
        return peer != nullptr;
    }
    if (reason == DLL_PROCESS_DETACH && detach_log) *detach_log = *detach_log * 10 + detach_id;
    return TRUE;
}

extern "C" __declspec(dllexport) int WINAPI attach_count() { return process_attaches; }
extern "C" __declspec(dllexport) HMODULE WINAPI peer_handle() { return peer; }
extern "C" __declspec(dllexport) void WINAPI set_detach_log(int* log) { detach_log = log; }
