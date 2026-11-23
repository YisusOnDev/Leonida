// PSM app-notification and Windows Error Reporting registration shims.
//
// Xbox GDK / UWP-linked titles import RegisterAppStateChangeNotification and
// RegisterAppConstrainedChangeNotification (api-ms-win-core-psm-appnotify-l1-1-0/1) to track suspend/resume and
// constrained transitions, and WerRegisterFile (api-ms-win-core-windowserrorreporting-l1-1-0) to attach logs/dumps.
// On an unpackaged desktop Win32 host (gta6rt):
//   - RegisterAppConstrainedChangeNotification returns ERROR_SUCCESS without writing *Registration, leaving caller
//     stack state uninitialized or null;
//   - UnregisterAppStateChangeNotification(nullptr) faults with an access violation;
//   - host WerRegisterFile registers paths into gta6rt's own WER block and has no console drive mapping for GAME:/G:/T:.
//
// Each registration returns a distinct tracked handle in-process, never touches freed or foreign handles on unregister,
// and normalizes console/host WER paths offline without writing to disk or mutating host WER state.
#include <windows.h>

#include <cstdint>
#include <cstring>
#include <cwchar>
#include <new>
#include <string>

#include "shims.h"

namespace rt::appnotify {
namespace {

using NotifyCallback = void(WINAPI*)(BOOLEAN state, PVOID context);

enum class NotifyKind { AppState, AppConstrained };

struct Registration {
    NotifyKind kind;
    NotifyCallback callback;
    void* context;
    Registration* next;
};

struct WerEntry {
    std::wstring key;
    int file_type;
    DWORD flags;
    WerEntry* next;
};

// Constant-initialized synchronization and raw linked lists avoid MSVC function-local static TLS init guards
// (_Init_thread_header) and C++ destructor ordering hazards during ExitProcess teardown.
SRWLOCK g_lock = SRWLOCK_INIT;
Registration* g_registrations = nullptr;
WerEntry* g_wer_entries = nullptr;

DWORD register_notification(NotifyKind kind, NotifyCallback callback, void* context, PVOID* registration) {
    if (!registration) return ERROR_INVALID_PARAMETER;
    *registration = nullptr;
    if (!callback) return ERROR_INVALID_PARAMETER;

    auto* entry = new (std::nothrow) Registration{kind, callback, context, nullptr};
    if (!entry) return ERROR_OUTOFMEMORY;

    AcquireSRWLockExclusive(&g_lock);
    entry->next = g_registrations;
    g_registrations = entry;
    ReleaseSRWLockExclusive(&g_lock);

    *registration = entry;
    return ERROR_SUCCESS;
}

void unregister_notification(NotifyKind kind, PVOID registration) {
    if (!registration) return;

    Registration* removed = nullptr;
    AcquireSRWLockExclusive(&g_lock);
    Registration** cur = &g_registrations;
    while (*cur) {
        if (*cur == registration && (*cur)->kind == kind) {
            removed = *cur;
            *cur = removed->next;
            break;
        }
        cur = &((*cur)->next);
    }
    ReleaseSRWLockExclusive(&g_lock);
    delete removed;
}

DWORD WINAPI register_app_state(NotifyCallback callback, void* context, PVOID* registration) {
    return register_notification(NotifyKind::AppState, callback, context, registration);
}

void WINAPI unregister_app_state(PVOID registration) {
    unregister_notification(NotifyKind::AppState, registration);
}

DWORD WINAPI register_app_constrained(NotifyCallback callback, void* context, PVOID* registration) {
    return register_notification(NotifyKind::AppConstrained, callback, context, registration);
}

void WINAPI unregister_app_constrained(PVOID registration) {
    unregister_notification(NotifyKind::AppConstrained, registration);
}

bool ascii_iequals(wchar_t a, wchar_t b) {
    if (a >= L'A' && a <= L'Z') a = static_cast<wchar_t>(a + 32);
    if (b >= L'A' && b <= L'Z') b = static_cast<wchar_t>(b + 32);
    return a == b;
}

// Normalizes a WER file path for in-process tracking and rejects ".." segments that climb above GAME:, G:, or T:.
bool normalize_wer_key(PCWSTR raw_path, std::wstring& out_key, HRESULT& out_error) {
    out_error = S_OK;
    if (!raw_path) {
        out_error = E_INVALIDARG;
        return false;
    }

    std::wstring path(raw_path);
    size_t console_prefix = 0;
    std::wstring drive_tag;
    if (path.size() >= 5 && ascii_iequals(path[0], L'g') && ascii_iequals(path[1], L'a') &&
        ascii_iequals(path[2], L'm') && ascii_iequals(path[3], L'e') && path[4] == L':') {
        console_prefix = 5;
        drive_tag = L"game:";
    } else if (path.size() >= 2 && (ascii_iequals(path[0], L'g') || ascii_iequals(path[0], L't')) && path[1] == L':') {
        console_prefix = 2;
        drive_tag = ascii_iequals(path[0], L'g') ? L"g:" : L"t:";
    }

    if (console_prefix != 0) {
        size_t pos = console_prefix;
        if (pos < path.size() && (path[pos] == L'\\' || path[pos] == L'/')) ++pos;
        size_t depth = 0;
        std::wstring normalized = drive_tag + L"\\";
        bool first_part = true;
        while (pos <= path.size()) {
            size_t next = path.find_first_of(L"\\/", pos);
            if (next == std::wstring::npos) next = path.size();
            const std::wstring seg = path.substr(pos, next - pos);
            if (seg == L"..") {
                if (depth == 0) {
                    out_error = HRESULT_FROM_WIN32(ERROR_INVALID_NAME);
                    return false;
                }
                --depth;
                const size_t slash = normalized.find_last_of(L'\\');
                if (slash == drive_tag.size()) {
                    normalized.erase(slash + 1);
                    first_part = true;
                } else if (slash != std::wstring::npos && slash > drive_tag.size()) {
                    normalized.erase(slash);
                }
            } else if (!seg.empty() && seg != L".") {
                if (!first_part) normalized += L'\\';
                first_part = false;
                normalized += seg;
                ++depth;
            }
            if (next == path.size()) break;
            pos = next + 1;
        }
        out_key = normalized;
        return true;
    }

    for (wchar_t& c : path) {
        if (c == L'/') c = L'\\';
    }
    out_key = path;
    return true;
}

constexpr int kWerRegFileTypeUserDocument = 1;
constexpr int kWerRegFileTypeOther = 2;
constexpr DWORD kWerKnownFlagsMask = 0x3u;  // WER_FILE_DELETE_WHEN_DONE (1) | WER_FILE_ANONYMOUS_DATA (2)

HRESULT WINAPI wer_register_file(PCWSTR file, int reg_file_type, DWORD flags) {
    if (!file || (reg_file_type != kWerRegFileTypeUserDocument && reg_file_type != kWerRegFileTypeOther) ||
        (flags & ~kWerKnownFlagsMask) != 0) {
        return E_INVALIDARG;
    }

    std::wstring key;
    HRESULT parse_hr = S_OK;
    if (!normalize_wer_key(file, key, parse_hr)) return parse_hr;

    AcquireSRWLockExclusive(&g_lock);
    for (WerEntry* cur = g_wer_entries; cur; cur = cur->next) {
        if (_wcsicmp(cur->key.c_str(), key.c_str()) == 0) {
            cur->file_type = reg_file_type;
            cur->flags = flags;
            ReleaseSRWLockExclusive(&g_lock);
            return S_OK;
        }
    }
    auto* entry = new (std::nothrow) WerEntry{key, reg_file_type, flags, g_wer_entries};
    if (!entry) {
        ReleaseSRWLockExclusive(&g_lock);
        return E_OUTOFMEMORY;
    }
    g_wer_entries = entry;
    ReleaseSRWLockExclusive(&g_lock);
    return S_OK;
}

HRESULT WINAPI wer_unregister_file(PCWSTR file) {
    std::wstring key;
    HRESULT parse_hr = S_OK;
    if (!normalize_wer_key(file, key, parse_hr)) return parse_hr;

    WerEntry* removed = nullptr;
    AcquireSRWLockExclusive(&g_lock);
    WerEntry** cur = &g_wer_entries;
    while (*cur) {
        if (_wcsicmp((*cur)->key.c_str(), key.c_str()) == 0) {
            removed = *cur;
            *cur = removed->next;
            break;
        }
        cur = &((*cur)->next);
    }
    ReleaseSRWLockExclusive(&g_lock);
    delete removed;
    return removed ? S_OK : HRESULT_FROM_WIN32(ERROR_NOT_FOUND);
}

}  // namespace
}  // namespace rt::appnotify

namespace {

using namespace rt::appnotify;

#define APPNOTIFY_API(dll)                                                                                   \
    {dll, "RegisterAppStateChangeNotification", reinterpret_cast<void*>(&register_app_state)},               \
        {dll, "UnregisterAppStateChangeNotification", reinterpret_cast<void*>(&unregister_app_state)},       \
        {dll, "RegisterAppConstrainedChangeNotification", reinterpret_cast<void*>(&register_app_constrained)}, \
        {dll, "UnregisterAppConstrainedChangeNotification", reinterpret_cast<void*>(&unregister_app_constrained)}

#define WER_API(dll)                                                                                         \
    {dll, "WerRegisterFile", reinterpret_cast<void*>(&wer_register_file)},                                   \
        {dll, "WerUnregisterFile", reinterpret_cast<void*>(&wer_unregister_file)}

}  // namespace

SHIM_TABLE(appnotify,
    APPNOTIFY_API("kernel32.dll"),
    APPNOTIFY_API("kernelbase.dll"),
    APPNOTIFY_API("twinapi.appcore.dll"),
    APPNOTIFY_API("api-ms-win-core-psm-appnotify-l1-1-0.dll"),
    APPNOTIFY_API("api-ms-win-core-psm-appnotify-l1-1-1.dll"),
    WER_API("kernel32.dll"),
    WER_API("kernelbase.dll"),
    WER_API("wer.dll"),
    WER_API("api-ms-win-core-windowserrorreporting-l1-1-0.dll"),
    WER_API("api-ms-win-core-windowserrorreporting-l1-1-1.dll"),
);
