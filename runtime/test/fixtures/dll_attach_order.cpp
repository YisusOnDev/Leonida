// Two DLLs that import each other, so the loader must break the static cycle. With ATTACH_A_FAILS, A's DllMain
// returns FALSE after B has already attached.
#include <windows.h>

#ifdef ATTACH_CYCLE_A
#define OWN_ATTACHED cycle_a_attached
#define PEER_ATTACHED cycle_b_attached
#else
#define OWN_ATTACHED cycle_b_attached
#define PEER_ATTACHED cycle_a_attached
#endif

#ifdef ATTACH_CYCLE_SEED

// Export the same symbol as A so B can be linked before the real A exists.
extern "C" __declspec(dllexport) BOOL WINAPI cycle_a_attached() { return FALSE; }
extern "C" BOOL WINAPI dll_attach_main(HINSTANCE, DWORD, LPVOID) { return TRUE; }

#else

extern "C" __declspec(dllimport) BOOL WINAPI PEER_ATTACHED();
namespace {
BOOL attached;
BOOL observed_peer;
}
extern "C" BOOL WINAPI dll_attach_main(HINSTANCE, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) {
        observed_peer = PEER_ATTACHED();
        attached = TRUE;
#if defined(ATTACH_CYCLE_A) && defined(ATTACH_A_FAILS)
        return FALSE;
#endif
    }
    return TRUE;
}
extern "C" __declspec(dllexport) BOOL WINAPI OWN_ATTACHED() { return attached; }
extern "C" __declspec(dllexport) BOOL WINAPI observed_peer_attach() { return observed_peer; }
// Calls through the import, as an importer does after startup.
extern "C" __declspec(dllexport) BOOL WINAPI call_peer() { return PEER_ATTACHED(); }

#endif
