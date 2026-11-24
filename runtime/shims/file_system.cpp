// file_system: the file API of a mapped title, seen through the console's drives.
//
// A GDK title names its files on the console's drives, "GAME:\\data\\x.sps" being one in the installed game folder, so
// the imports it makes of CreateFile*, FindFirstFile*/FindNextFile*/FindClose, GetFileAttributes*/GetFileAttributesEx*,
// DeleteFile*, CreateDirectory*/RemoveDirectory*, MoveFile*, CopyFile*/CopyFile2, SetFileAttributes* and
// SetFileInformationByHandle's delete requests (std::filesystem::remove deletes through a handle), and the
// Universal CRT's fopen/_open families and its remove/rename/_stat/_access/_mkdir/_rmdir, are answered here (docs/runtime_imports.md, "Console paths and the write overlay"; the prefixes come from ivall/xcl,
// GPL-2.0-only, as facts). The rules:
//   - "GAME:" (any case, "\\" or "/") maps under --game-dir, "G:\\" is the same folder and "T:\\" a scratch folder in
//     the overlay (RT-218); any other path is a host path and goes to the host;
//   - an open that can write, and DeleteFile, go to the overlay (--overlay, default <game-dir>\\overlay), so the
//     package stays exactly as installed. A write-open of a package file copies it into the overlay first, so
//     OPEN_EXISTING keeps its bytes; a delete of a package file leaves an empty marker under
//     <overlay>\\.gta6rt-deleted, and reads and listings treat the name as gone. The marker is never removed and a
//     deleted name is never copied up: a re-created file is its overlay copy, which reads and listings already
//     prefer, while an open of a deleted name that fails leaves the name deleted;
//   - FILE_FLAG_DELETE_ON_CLOSE is a delete whatever access the open asks for: the open uses the overlay copy and a
//     package name gets its marker once the open has succeeded;
//   - PLS: names only the private persistent storage folder in the overlay;
//   - GAME: reads prefer the overlay copy and then the package;
//   - FindFirstFile merges the two folders, the overlay winning on a name, and hides deleted names and the marker
//     folder;
//   - folders are made in the overlay; a removed package folder is hidden by a marker, a moved package file is
//     copied into the overlay and its old name deleted, and attributes are set on the overlay copy;
//   - a path that climbs above the root with ".." or names another drive is refused with ERROR_INVALID_NAME, so a
//     title cannot reach outside the package and the overlay.
// Only images mapped by the loader see these: they are bound through the registry, the runtime's own code calls the
// host directly.
#include <windows.h>
#include <intrin.h>

#include <algorithm>
#include <atomic>

#include <fcntl.h>
#include <share.h>
#include <sys/stat.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <cwctype>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>

#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "file_system.h"
#include "shims.h"
#include "startup_trace.h"
#include "kernel_write_hook.h"

namespace rt::fs {
namespace {

constexpr wchar_t kGamePrefix[] = L"game:";
constexpr size_t kGamePrefixLength = 5;
constexpr wchar_t kDeletedDir[] = L".gta6rt-deleted";
constexpr wchar_t kScratchDir[] = L".gta6rt-temp";  // what "T:\" names, inside the overlay
// A removed package folder is marked by this file inside its marker folder: the marker folder itself can already
// exist because files below it were deleted, and that alone does not hide the folder.
constexpr wchar_t kRemovedDirMarker[] = L".gta6rt-dir";
// Copy-up writes here first and then renames into place, so a half-copied file never sits under the title's name.
constexpr wchar_t kCopyDir[] = L".gta6rt-copy";

// The overlay root's own folders, which a title never sees in a listing.
bool internal_dir(const wchar_t* name) {
    return _wcsicmp(name, kDeletedDir) == 0 || _wcsicmp(name, kCopyDir) == 0 || _wcsicmp(name, kScratchDir) == 0 ||
           _wcsicmp(name, kPersistentStorageDir) == 0;
}

std::mutex g_mutex;
std::filesystem::path g_game_dir;
std::filesystem::path g_overlay_dir;

struct Roots {
    std::filesystem::path game, overlay;
};

Roots roots() {
    std::lock_guard lock(g_mutex);
    return {g_game_dir, g_overlay_dir};
}

std::wstring lower(std::wstring s) {
    for (auto& c : s) c = static_cast<wchar_t>(std::towlower(c));
    return s;
}

bool same_name(const wchar_t* a, const wchar_t* b) { return _wcsicmp(a, b) == 0; }

// The narrow API (CreateFileA and friends) takes ANSI code page text, as the CRT does; the loader's own paths are
// handled as wide strings everywhere else.
std::wstring widen_ansi(const char* s) {
    if (!s) return {};
    const int n = MultiByteToWideChar(CP_ACP, 0, s, -1, nullptr, 0);
    if (n <= 1) return {};
    std::wstring w(static_cast<size_t>(n - 1), L'\0');
    MultiByteToWideChar(CP_ACP, 0, s, -1, w.data(), n);
    return w;
}

void narrow_ansi(const std::wstring& wide, char* out, size_t capacity) {
    if (capacity == 0) return;
    const int n = WideCharToMultiByte(CP_ACP, 0, wide.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (n <= 1 || static_cast<size_t>(n) > capacity) {
        out[0] = '\0';
        return;
    }
    WideCharToMultiByte(CP_ACP, 0, wide.c_str(), -1, out, static_cast<int>(capacity), nullptr, nullptr);
}

enum class Parsed { Host, Console, Refused };

// Splits a console path into the components below the root. A component that names a drive or a stream ("C:", "x:y")
// and a ".." that would leave the root are refused: the package and the overlay are the only folders a title may name.
bool is_separator(wchar_t c) { return c == L'\\' || c == L'/'; }

// True when `path` is `root` or lies below it, compared by name (case-insensitive, either separator).
bool lies_under(const std::wstring& path, const std::filesystem::path& root) {
    std::wstring base = lower(root.native());
    for (auto& c : base) if (c == L'/') c = L'\\';
    while (!base.empty() && base.back() == L'\\') base.pop_back();
    if (base.empty()) return false;
    std::wstring p = lower(path);
    for (auto& c : p) if (c == L'/') c = L'\\';
    return p.compare(0, base.size(), base) == 0 && (p.size() == base.size() || p[base.size()] == L'\\');
}

// The console shim may resolve even a short guest name to an extended-length host path. GetFullPathNameW
// applies the ordinary DOS path rules (including trailing dots and spaces) before the extended prefix disables
// that normalization. The headroom is for CreateDirectoryW's shorter limit on an unprefixed directory name.
std::filesystem::path win32_io_path(const std::filesystem::path& path) {
    const auto& native = path.native();
    constexpr size_t kDirectoryPathLimit = MAX_PATH - 12;
    if (native.size() < kDirectoryPathLimit || !path.is_absolute() ||
        native.starts_with(L"\\\\?\\") || native.starts_with(L"\\\\.\\"))
        return path;

    const DWORD capacity = ::GetFullPathNameW(native.c_str(), 0, nullptr, nullptr);
    if (capacity == 0) return path;
    std::wstring full(capacity, L'\0');
    const DWORD size = ::GetFullPathNameW(native.c_str(), capacity, full.data(), nullptr);
    if (size == 0 || size >= capacity) return path;
    full.resize(size);

    // DOS device names such as NUL can normalize to a device path. Leave those to the native API as before.
    if (full.starts_with(L"\\\\.\\")) return path;
    if (full.starts_with(L"\\\\")) return std::filesystem::path(L"\\\\?\\UNC\\" + full.substr(2));
    return std::filesystem::path(L"\\\\?\\" + full);
}

std::filesystem::path win32_find_path(const std::filesystem::path& path, const std::wstring& leaf) {
    if (path.native().size() < MAX_PATH) return path;
    const auto search = win32_io_path(path);
    if (search == path || !leaf.ends_with(L'.') || leaf.find_first_of(L"*?") == std::wstring::npos) return search;

    // GetFullPathNameW strips a terminal dot, but that dot changes a native Win32 search (e.g. "mesh*.").
    // Preserve the DOS wildcard expression for the extended path and let the filesystem match it, including
    // short names. The DOS_STAR, DOS_QM and DOS_DOT tokens are defined by MS-FSA 2.1.4.3-4.
    std::wstring mask = search.filename().native();
    if (mask.ends_with(L'*')) mask.back() = L'<';
    else mask.push_back(L'"');
    for (size_t i = 0; i + 1 < mask.size(); ++i) {
        const wchar_t next = mask[i + 1];
        if (mask[i] == L'?') mask[i] = L'>';
        else if (mask[i] == L'*' && next == L'.') mask[i] = L'<';
        else if (mask[i] == L'.' && (next == L'*' || next == L'?' || next == L'<')) mask[i] = L'"';
    }
    return search.parent_path() / mask;
}

// "G:\x" is the installed package, like "GAME:\x", and "T:\x" a scratch folder the title may write to, the way the
// other console layers map those two drives (RT-218). Only the executable can say which spelling the title uses, so
// a host drive of the same letter must keep working: a path under --game-dir or the overlay (a package on drive G:)
// is not read as a console drive, and every G:/T: path stays a host path while no roots are configured. A path under
// --game-dir is the package by its host name instead (game_dir_rest below). The scratch folder lives in the overlay,
// as `.gta6rt-temp`.
enum class Drive { None, Game, Scratch };

// A host path whose normalised form lies inside --game-dir (and not inside the overlay) names a package file by its
// host name: a title builds one from GetModuleFileName or from the folder it was started in. It gets the same rules
// as its GAME: spelling, so a read sees the overlay copy and a write, delete, move or attribute change goes to the
// overlay instead of the package. Returns the part below the game folder ("\data\x", or "" for the folder itself);
// false for any other path. ".." is resolved first, so a path that climbs out of the game folder is a host path.
bool game_dir_rest(const std::wstring& path, std::wstring& rest) {
    const bool absolute = (path.size() >= 3 && path[1] == L':' && is_separator(path[2])) ||
                          (path.size() >= 2 && is_separator(path[0]) && is_separator(path[1]));
    if (!absolute) return false;
    const Roots r = roots();
    if (r.game.empty()) return false;
    const std::wstring normal = std::filesystem::path(path).lexically_normal().native();
    if (!lies_under(normal, r.game) || (!r.overlay.empty() && lies_under(normal, r.overlay))) return false;
    std::wstring base = r.game.lexically_normal().native();
    while (!base.empty() && is_separator(base.back())) base.pop_back();
    rest = normal.substr(std::min(base.size(), normal.size()));
    return true;
}

Drive console_drive(const std::wstring& path, size_t& prefix_length) {
    if (path.size() >= kGamePrefixLength && lower(path.substr(0, kGamePrefixLength)) == kGamePrefix) {
        prefix_length = kGamePrefixLength;
        return Drive::Game;
    }
    if (path.size() < 2 || path[1] != L':' || (path.size() > 2 && !is_separator(path[2]))) return Drive::None;
    const wchar_t letter = static_cast<wchar_t>(std::towlower(path[0]));
    if (letter != L'g' && letter != L't') return Drive::None;
    const Roots r = roots();
    if (r.game.empty() && r.overlay.empty()) return Drive::None;
    if (lies_under(path, r.game) || lies_under(path, r.overlay)) return Drive::None;
    prefix_length = 2;
    return letter == L'g' ? Drive::Game : Drive::Scratch;
}

bool parse_relative(std::wstring_view path, size_t prefix_length, std::vector<std::wstring>& parts) {
    const size_t floor = parts.size();  // ".." may not climb out of the drive's own root
    std::wstring part;
    auto flush = [&]() -> bool {
        if (part.empty() || part == L".") {
            part.clear();
            return true;
        }
        if (part == L"..") {
            if (parts.size() <= floor) return false;
            parts.pop_back();
        } else if (part.find(L':') != std::wstring::npos) {
            return false;
        } else {
            parts.push_back(part);
        }
        part.clear();
        return true;
    };
    for (size_t i = prefix_length; i < path.size(); ++i) {
        const wchar_t c = path[i];
        if (is_separator(c)) {
            if (!flush()) return false;
        } else {
            part.push_back(c);
        }
    }
    if (!flush()) return false;
    return true;
}

Parsed parse_console_parts(const std::wstring& path, std::vector<std::wstring>& parts, Drive* out_drive = nullptr,
                           bool create_scratch = true) {
    if (std::wstring rest; game_dir_rest(path, rest))
        return parse_console_parts(std::wstring(kGamePrefix) + rest, parts, out_drive, create_scratch);
    size_t prefix_length = 0;
    const Drive drive = console_drive(path, prefix_length);
    if (out_drive) *out_drive = drive;
    if (drive == Drive::None) return Parsed::Host;
    parts.clear();
    if (drive == Drive::Scratch) {
        parts.push_back(kScratchDir);
        if (create_scratch) {
            const std::filesystem::path overlay = roots().overlay;
            const DWORD last_error = ::GetLastError();
            std::error_code ec;
            if (!overlay.empty()) std::filesystem::create_directories(win32_io_path(overlay / kScratchDir), ec);
            ::SetLastError(last_error);
        }
    }
    if (!parse_relative(path, prefix_length, parts)) return Parsed::Refused;
    // The overlay's own folders are not part of the title's GAME: tree: hidden from its listings (internal_dir) and
    // refused as an explicit path too, in every spelling the loop above normalises (case, slashes, "." and "..").
    // T: still names the scratch folder: its first part is pushed above, not parsed from the title's path.
    if (drive == Drive::Game && !parts.empty() && internal_dir(parts.front().c_str())) return Parsed::Refused;
    return Parsed::Console;
}

Parsed parse_console(const std::wstring& path, std::vector<std::wstring>& parts, bool create_scratch = true) {
    return parse_console_parts(path, parts, nullptr, create_scratch);
}

std::filesystem::path join(const std::filesystem::path& root, const std::vector<std::wstring>& parts) {
    std::filesystem::path p = root;
    for (const auto& part : parts) p /= part;
    return p;
}

std::filesystem::path disk_path(const std::filesystem::path& root, const std::vector<std::wstring>& parts) {
    return win32_io_path(join(root, parts));
}

bool drive_root(const std::wstring& path, const std::vector<std::wstring>& parts) {
    size_t prefix_length = 0;
    return parts.empty() || (parts.size() == 1 && console_drive(path, prefix_length) == Drive::Scratch);
}

// A folder's entries change one step at a time, as on the host: RemoveDirectory holds the folder's lock from its
// emptiness check to its marker, and a GAME: write-open or CreateDirectory holds the lock of the folder it creates in
// from its "does the folder exist" check, through mirroring that folder into the overlay, to the host call that makes
// the entry. One of the two goes first: the create fails with ERROR_PATH_NOT_FOUND, or the remove with
// ERROR_DIR_NOT_EMPTY. Without it both succeeded and the removed folder was back, the new entry in it. The locks are
// striped by the folder's name, so calls in different folders rarely wait on each other; reads never take one.
// Namespace scope, not function-local statics: these run on the title's threads. Taken after roots(); nothing holding
// one takes another of them or calls back into the title (a copy-up's lock may be taken inside one, never around one).
constexpr size_t kFolderLocks = 64;
std::mutex g_folder_locks[kFolderLocks];

std::mutex& folder_lock(const std::vector<std::wstring>& folder) {
    // FNV-1a over the upper-cased components: names compare without case, so every spelling of a folder gets its lock.
    uint64_t hash = 14695981039346656037ull;
    const auto mix = [&hash](wchar_t c) {
        hash ^= static_cast<uint64_t>(c);
        hash *= 1099511628211ull;
    };
    for (const auto& part : folder) {
        std::wstring upper(part.size(), L'\0');
        if (part.empty() || LCMapStringEx(LOCALE_NAME_INVARIANT, LCMAP_UPPERCASE, part.data(),
                                          static_cast<int>(part.size()), upper.data(), static_cast<int>(upper.size()),
                                          nullptr, nullptr, 0) != static_cast<int>(part.size()))
            upper = part;
        for (const wchar_t c : upper) mix(c);
        mix(L'\\');
    }
    return g_folder_locks[hash % kFolderLocks];
}

struct Target {
    std::filesystem::path path;
    DWORD error = 0;  // non-zero: the caller fails with this error and does not touch the host
    // A write-open's hold on the folder it opens in, until the Target goes after the host open (g_folder_locks).
    std::unique_lock<std::mutex> folder;
    bool replaces_package_file = false;  // CREATE_ALWAYS on a file only the package has: see open_target
};

Target fail(DWORD error) {
    Target t;
    t.error = error;
    return t;
}

// PLS is its own guest namespace. Its backing directory stays outside the package view and file operations
// use the host's ordinary semantics on that directory, without package copy-up or deletion markers.
constexpr std::wstring_view kPersistentPrefix = L"PLS:";

bool persistent_guest_path(std::wstring_view path) {
    return path.size() >= kPersistentPrefix.size() &&
           _wcsnicmp(path.data(), kPersistentPrefix.data(), kPersistentPrefix.size()) == 0;
}

std::filesystem::path extended_path(const std::filesystem::path& path) {
    std::error_code ec;
    const auto absolute = std::filesystem::absolute(path, ec).lexically_normal().make_preferred();
    if (ec) return {};
    const auto& wide = absolute.native();
    if (wide.starts_with(L"\\\\?\\") || wide.starts_with(L"\\\\.\\")) return absolute;
    if (wide.starts_with(L"\\\\")) return std::filesystem::path(L"\\\\?\\UNC\\" + wide.substr(2));
    return std::filesystem::path(L"\\\\?\\" + wide);
}

// Check every existing component in the storage subtree. Never follow a preexisting directory junction
// or symlink from the guest alias, including one in place of the storage directory itself.
DWORD check_persistent_component(const std::filesystem::path& path) {
    const DWORD saved_error = ::GetLastError();
    const DWORD attributes = ::GetFileAttributesW(path.c_str());
    const DWORD error = ::GetLastError();
    ::SetLastError(saved_error);
    if (attributes != INVALID_FILE_ATTRIBUTES)
        return (attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0 ? ERROR_ACCESS_DENIED : ERROR_SUCCESS;
    return error == ERROR_FILE_NOT_FOUND || error == ERROR_PATH_NOT_FOUND ? ERROR_SUCCESS : error;
}

std::optional<Target> persistent_target(std::wstring_view name, bool allow_root = true) {
    if (!persistent_guest_path(name)) return std::nullopt;
    const size_t prefix_length = kPersistentPrefix.size();
    if (name.size() > prefix_length && !is_separator(name[prefix_length]))
        return fail(ERROR_INVALID_NAME);

    std::vector<std::wstring> parts;
    if (!parse_relative(name, prefix_length, parts)) return fail(ERROR_INVALID_NAME);
    if (parts.empty() && !allow_root) return fail(ERROR_ACCESS_DENIED);

    const auto overlay = roots().overlay;
    if (overlay.empty()) return fail(ERROR_PATH_NOT_FOUND);
    auto path = extended_path(overlay / kPersistentStorageDir);
    if (path.empty()) return fail(ERROR_PATH_NOT_FOUND);
    if (const DWORD error = check_persistent_component(path)) return fail(error);
    for (size_t i = 0; i < parts.size(); ++i) {
        const bool wildcard = parts[i].find_first_of(L"*?") != std::wstring::npos;
        if (wildcard && i + 1 != parts.size()) return fail(ERROR_INVALID_NAME);
        path /= parts[i];
        // FindFirstFile accepts a wildcard in the last component; its parent was already checked.
        if (!wildcard) {
            if (const DWORD error = check_persistent_component(path)) return fail(error);
        }
    }
    return Target{std::move(path)};
}

// The overlay keeps one empty marker per deleted package path, outside the mirrored tree so it cannot collide with a
// file the title creates.
std::filesystem::path marker(const std::filesystem::path& overlay, const std::vector<std::wstring>& parts) {
    return join(overlay / kDeletedDir, parts);
}

// A deleted file leaves an empty file at its marker path and a removed folder leaves kRemovedDirMarker inside its
// marker folder.
bool deleted(const std::filesystem::path& overlay, const std::vector<std::wstring>& parts) {
    if (overlay.empty() || parts.empty()) return false;
    std::error_code ec;
    // Only regular files are deletion markers; parent directories inside .gta6rt-deleted/ exist only to hold markers
    // for nested deleted files and must not hide the corresponding ancestor directories in the package.
    const std::filesystem::path m = marker(overlay, parts);
    return std::filesystem::is_regular_file(win32_io_path(m), ec) ||
           std::filesystem::is_regular_file(win32_io_path(m / kRemovedDirMarker), ec);
}

// Writes an empty hidden marker file. On failure `error` holds the reason.
bool write_marker(const std::filesystem::path& path, DWORD& error) {
    const std::filesystem::path mark = win32_io_path(path);
    std::error_code ec;
    std::filesystem::create_directories(mark.parent_path(), ec);
    if (ec) {
        error = static_cast<DWORD>(ec.value());
        return false;
    }
    const HANDLE h = ::CreateFileW(mark.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                   nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_HIDDEN, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        error = ::GetLastError();
        return false;
    }
    ::CloseHandle(h);
    return true;
}

// Flushes a file the shim wrote on the title's behalf, for MOVEFILE_WRITE_THROUGH. On failure `error` holds the reason.
bool flush_file(const std::filesystem::path& path, DWORD& error) {
    const HANDLE h = ::CreateFileW(win32_io_path(path).c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                   nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        error = ::GetLastError();
        return false;
    }
    const BOOL ok = ::FlushFileBuffers(h);
    if (!ok) error = ::GetLastError();
    ::CloseHandle(h);
    return ok != FALSE;
}

// What a console path is in the title's view: the overlay copy if there is one, else the package's unless deleted.
enum class Kind { None, File, Directory };
struct View {
    Kind kind = Kind::None;
    bool in_overlay = false;  // the overlay holds the name
    bool in_package = false;  // the package holds the name and it is not deleted
};

View view(const Roots& r, const std::vector<std::wstring>& parts) {
    View v;
    std::error_code ec;
    if (parts.empty()) {  // "GAME:" itself
        v.kind = Kind::Directory;
        v.in_package = !r.game.empty();
        return v;
    }
    if (!r.overlay.empty()) {
        const std::filesystem::path p = disk_path(r.overlay, parts);
        if (std::filesystem::exists(p, ec)) {
            v.in_overlay = true;
            v.kind = std::filesystem::is_directory(p, ec) ? Kind::Directory : Kind::File;
        }
    }
    if (!r.game.empty() && !deleted(r.overlay, parts)) {
        const std::filesystem::path p = disk_path(r.game, parts);
        if (std::filesystem::exists(p, ec)) {
            v.in_package = true;
            if (v.kind == Kind::None) v.kind = std::filesystem::is_directory(p, ec) ? Kind::Directory : Kind::File;
        }
    }
    return v;
}

// Walks up `p` without creating directories to find the nearest existing directory on its volume.
std::filesystem::path existing_ancestor_dir(std::filesystem::path p, DWORD missing_error, DWORD& error) {
    std::error_code ec;
    while (!p.empty()) {
        const auto st = std::filesystem::status(p, ec);
        if (st.type() == std::filesystem::file_type::directory) return p;
        if (st.type() != std::filesystem::file_type::not_found && !ec) {
            error = ERROR_DIRECTORY;
            return {};
        }
        const std::filesystem::path parent = p.parent_path();
        if (parent == p) break;
        p = parent;
    }
    error = missing_error;
    return {};
}

// Resolves a console path for read-only disk-space or drive-type queries without creating `overlay` or `.gta6rt-temp`.
// Sets `handled` true when `path` is refused or console-owned (`out_dir` valid on success, `error` set on failure).
bool resolve_console_readonly_dir(const wchar_t* path, DWORD missing_error, bool& handled,
                                  std::filesystem::path& out_dir, bool& is_root, DWORD& error) {
    handled = false;
    is_root = false;
    error = 0;
    std::vector<std::wstring> parts;
    Drive drive = Drive::None;
    const Parsed parsed = parse_console_parts(path, parts, &drive, /*create_scratch=*/false);
    if (parsed == Parsed::Host) return false;
    handled = true;
    if (parsed == Parsed::Refused) {
        error = ERROR_INVALID_NAME;
        return false;
    }
    const Roots r = roots();
    std::error_code ec;
    if (drive == Drive::Scratch) {
        if (r.overlay.empty()) {
            error = ERROR_PATH_NOT_FOUND;
            return false;
        }
        is_root = (parts.size() == 1);
        if (is_root) {
            out_dir = existing_ancestor_dir(r.overlay / kScratchDir, missing_error, error);
            return !out_dir.empty();
        }
        const std::filesystem::path p = join(r.overlay, parts);
        const auto st = std::filesystem::status(p, ec);
        if (st.type() == std::filesystem::file_type::directory) {
            out_dir = p;
            return true;
        }
        error = (st.type() != std::filesystem::file_type::not_found && !ec) ? ERROR_DIRECTORY : missing_error;
        return false;
    }
    if (r.game.empty()) {
        error = ERROR_PATH_NOT_FOUND;
        return false;
    }
    is_root = parts.empty();
    if (is_root) {
        const auto st = std::filesystem::status(r.game, ec);
        if (st.type() == std::filesystem::file_type::directory) {
            out_dir = r.game;
            return true;
        }
        error = (st.type() != std::filesystem::file_type::not_found && !ec) ? ERROR_DIRECTORY : missing_error;
        return false;
    }
    const View v = view(r, parts);
    if (v.kind == Kind::None) {
        error = missing_error;
        return false;
    }
    if (v.kind == Kind::File) {
        error = ERROR_DIRECTORY;
        return false;
    }
    out_dir = v.in_overlay ? join(r.overlay, parts) : join(r.game, parts);
    return true;
}

bool is_folder(const Roots& r, const std::vector<std::wstring>& parts) { return view(r, parts).kind == Kind::Directory; }

// Whether two console paths name the same entry: names compare without case, as the lookups do.
bool same_entry(const std::vector<std::wstring>& a, const std::vector<std::wstring>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (!same_name(a[i].c_str(), b[i].c_str())) return false;
    return true;
}

std::vector<std::wstring> parent_of(const std::vector<std::wstring>& parts) {
    return parts.empty() ? parts : std::vector<std::wstring>(parts.begin(), parts.end() - 1);
}

// Whether a folder has nothing in the title's view: no overlay entry and no package entry that is not deleted.
bool empty_in_view(const Roots& r, const std::vector<std::wstring>& parts) {
    std::error_code ec;
    if (!r.overlay.empty()) {
        const std::filesystem::path p = disk_path(r.overlay, parts);
        for (auto it = std::filesystem::directory_iterator(p, ec); !ec && it != std::filesystem::directory_iterator();
             it.increment(ec)) {
            if (parts.empty() && internal_dir(it->path().filename().c_str())) continue;
            return false;
        }
    }
    ec.clear();
    if (!r.game.empty()) {
        const std::filesystem::path p = disk_path(r.game, parts);
        for (auto it = std::filesystem::directory_iterator(p, ec); !ec && it != std::filesystem::directory_iterator();
             it.increment(ec)) {
            std::vector<std::wstring> child = parts;
            child.push_back(it->path().filename().wstring());
            if (!deleted(r.overlay, child)) return false;
        }
    }
    return true;
}

BOOL fail_bool(DWORD error) {
    SetLastError(error);
    return FALSE;
}

Target read_target(const std::wstring& path) {
    if (auto target = persistent_target(path)) return std::move(*target);
    std::vector<std::wstring> parts;
    const Parsed parsed = parse_console(path, parts);
    if (parsed == Parsed::Host) {
        Target t;
        t.path = std::filesystem::path(path);
        return t;
    }
    if (parsed == Parsed::Refused) return fail(ERROR_INVALID_NAME);

    const Roots r = roots();
    if (r.game.empty()) return fail(ERROR_PATH_NOT_FOUND);

    std::error_code ec;
    if (!r.overlay.empty()) {
        const std::filesystem::path overlay_path = disk_path(r.overlay, parts);
        if (std::filesystem::exists(overlay_path, ec)) {
            Target t;
            t.path = overlay_path;
            return t;
        }
    }
    // A package file the title deleted is gone even though the package still holds it.
    if (deleted(r.overlay, parts)) return fail(ERROR_FILE_NOT_FOUND);

    Target t;
    t.path = disk_path(r.game, parts);
    return t;
}

Target write_target(const std::wstring& path, bool copy_up) {
    if (auto target = persistent_target(path, false)) return std::move(*target);
    std::vector<std::wstring> parts;
    const Parsed parsed = parse_console(path, parts);
    if (parsed == Parsed::Host) {
        Target t;
        t.path = std::filesystem::path(path);
        return t;
    }
    if (parsed == Parsed::Refused) return fail(ERROR_INVALID_NAME);

    const Roots r = roots();
    if (r.overlay.empty()) return fail(ERROR_PATH_NOT_FOUND);
    if (parts.empty()) return fail(ERROR_INVALID_NAME);  // "GAME:" is a folder, not a file

    // The folder has to exist in the title's view, as CreateDirectory, MoveFile and CopyFile require: without the
    // check a failed open left the missing folders behind, and a create succeeded in a folder that was never made,
    // below a package file (which then read as a folder) and inside a folder the title had removed. The overlay may
    // still lack a folder only the package has, so it is mirrored here, which the title cannot see.
    // T:'s scratch folder is the exception: it creates its parent folders, as merged with RT-218.
    // The folder's lock is held from this check to the caller's host open, so a RemoveDirectory of the folder comes
    // wholly before or after (g_folder_locks).
    std::unique_lock folder(folder_lock(parent_of(parts)));
    const bool scratch = !parts.empty() && same_name(parts.front().c_str(), kScratchDir);
    if (!scratch && !is_folder(r, parent_of(parts))) return fail(ERROR_PATH_NOT_FOUND);

    Target t;
    t.folder = std::move(folder);
    t.path = disk_path(r.overlay, parts);

    std::error_code ec;
    std::filesystem::create_directories(t.path.parent_path(), ec);
    if (ec) {
        t.error = static_cast<DWORD>(ec.value());
        return t;
    }

    // A deleted package file is not there to copy: copying it up would bring the name back, and an open that keeps
    // the contents (OPEN_EXISTING, TRUNCATE_EXISTING) has to fail on it instead. CREATE_ALWAYS and CREATE_NEW still
    // make a new overlay file, which reads prefer over the package and so over the marker.
    const std::filesystem::path package_path = disk_path(r.game, parts);
    // One copy-up at a time: a second thread writing the same file waits here and then finds the first one's copy, so
    // it neither fails on the taken name nor collides with that file's open through a rename of its own.
    static SRWLOCK s_copy_lock = SRWLOCK_INIT;
    struct DeferredSrwLock {
        SRWLOCK* lock;
        bool held = false;
        explicit DeferredSrwLock(SRWLOCK& l) noexcept : lock(&l) {}
        void acquire() noexcept {
            AcquireSRWLockExclusive(lock);
            held = true;
        }
        ~DeferredSrwLock() {
            if (held) ReleaseSRWLockExclusive(lock);
        }
        DeferredSrwLock(const DeferredSrwLock&) = delete;
        DeferredSrwLock& operator=(const DeferredSrwLock&) = delete;
    } copy_lock(s_copy_lock);
    if (copy_up) copy_lock.acquire();
    if (copy_up && !deleted(r.overlay, parts) && !std::filesystem::exists(t.path, ec) &&
        std::filesystem::is_regular_file(package_path, ec)) {
        // Copied to a private name, then renamed into place: a process killed mid-copy leaves only a hidden leftover,
        // not a truncated file that reads would prefer over the package.
        static unsigned s_copies = 0;  // under s_copy_mutex
        const std::filesystem::path staging = r.overlay / kCopyDir;
        std::filesystem::create_directories(win32_io_path(staging), ec);
        const std::filesystem::path tmp =
            win32_io_path(staging / (std::to_wstring(::GetCurrentProcessId()) + L"-" + std::to_wstring(++s_copies)));
        // Use CopyFileW for staging because MSVC's std::filesystem::copy_file calls CopyFile2, which Wine 9 lacks.
        if (ec) {
            t.error = static_cast<DWORD>(ec.value());
        } else if (!::CopyFileW(package_path.c_str(), tmp.c_str(), FALSE)) {
            t.error = ::GetLastError();
        }
        if (t.error) {
            std::filesystem::remove(tmp, ec);
            return t;
        }
        if (!::MoveFileExW(tmp.c_str(), t.path.c_str(), 0)) {
            // The name taken means the title created the file itself meanwhile (CREATE_NEW and CREATE_ALWAYS don't
            // copy up, so they don't take the lock); that file is what it wrote, so keep it.
            const DWORD error = ::GetLastError();
            std::filesystem::remove(tmp, ec);
            if (!std::filesystem::exists(t.path, ec)) t.error = error;
        }
    } else if (copy_up && !deleted(r.overlay, parts) && std::filesystem::is_directory(package_path, ec)) {
        // A package folder is mirrored, as SetFileAttributes does, so a handle opened on it for DELETE or attribute
        // writes (std::filesystem::remove) names the overlay folder and never the package's.
        std::filesystem::create_directories(t.path, ec);
        if (ec) t.error = static_cast<DWORD>(ec.value());
    }
    return t;
}

// FILE_FLAG_DELETE_ON_CLOSE makes any open a delete: the host adds DELETE to the access it was asked for, so a
// read-only open with the flag removes the file when its last handle is closed.
// MAXIMUM_ALLOWED can grant write access, so it also requires the overlay.
bool writes(DWORD access, DWORD disposition, DWORD flags) {
    constexpr DWORD kWriteAccess = GENERIC_WRITE | GENERIC_ALL | MAXIMUM_ALLOWED | FILE_WRITE_DATA | FILE_APPEND_DATA | FILE_WRITE_EA |
                                   FILE_WRITE_ATTRIBUTES | DELETE | WRITE_DAC | WRITE_OWNER;
    if ((access & kWriteAccess) != 0 || (flags & FILE_FLAG_DELETE_ON_CLOSE) != 0) return true;
    return disposition == CREATE_NEW || disposition == CREATE_ALWAYS || disposition == OPEN_ALWAYS ||
           disposition == TRUNCATE_EXISTING;
}

Target open_target(const std::wstring& path, DWORD access, DWORD disposition, DWORD flags = 0) {
    if (!writes(access, disposition, flags)) return read_target(path);
    // OPEN_ALWAYS and TRUNCATE_EXISTING read what is already there, and a write-open must not lose a package file's
    // bytes; CREATE_ALWAYS and CREATE_NEW replace it.
    const bool keeps_contents =
        disposition == OPEN_EXISTING || disposition == OPEN_ALWAYS || disposition == TRUNCATE_EXISTING;
    // The host is asked about the overlay name, which a name only the package has does not have yet, so what a
    // creating open answers for a name that exists is decided here: a folder's name cannot be created as a file,
    // CREATE_NEW fails on a file, and CREATE_ALWAYS replaces it and says so (OPEN_ALWAYS does through its copy-up).
    // FILE_FLAG_BACKUP_SEMANTICS is how a folder is opened: with it the host refuses CREATE_NEW and CREATE_ALWAYS
    // on a folder with ERROR_FILE_EXISTS, and OPEN_ALWAYS opens the folder and reports ERROR_ALREADY_EXISTS itself.
    // That open goes to the package folder when it asks for no write access, as OPEN_EXISTING does, and otherwise
    // to a mirror of the folder in the overlay, as SetFileAttributes makes one, so the package is not written.
    bool replaces_package_file = false;
    std::vector<std::wstring> parts;
    if (disposition != OPEN_EXISTING && disposition != TRUNCATE_EXISTING &&
        parse_console(path, parts) == Parsed::Console && !parts.empty()) {  // the root is write_target's to refuse
        const View v = view(roots(), parts);
        if (v.in_package && !v.in_overlay) {
            if (v.kind == Kind::Directory) {
                if ((flags & FILE_FLAG_BACKUP_SEMANTICS) == 0) return fail(ERROR_ACCESS_DENIED);
                if (disposition != OPEN_ALWAYS) return fail(ERROR_FILE_EXISTS);
                if (!writes(access, OPEN_EXISTING, flags)) return read_target(path);
                Target folder = write_target(path, false);
                std::error_code ec;
                if (!folder.error) std::filesystem::create_directories(folder.path, ec);
                if (ec) folder.error = static_cast<DWORD>(ec.value());
                return folder;
            }
            if (disposition == CREATE_NEW) return fail(ERROR_FILE_EXISTS);
            replaces_package_file = disposition == CREATE_ALWAYS;
        }
    }
    Target t = write_target(path, keeps_contents);
    t.replaces_package_file = replaces_package_file;
    return t;
}

// A delete-on-close open that succeeded is on the overlay copy, which the host removes at the last close. A name the
// package has gets the marker DeleteFile leaves, so the package file does not come back into view then; until the
// close, reads still find the overlay copy. A marker that cannot be written is not reported: the open succeeded and
// the delete it carries cannot be taken back, so the package file is what the title sees again after the close.
void mark_deleted_on_close(const std::wstring& path) {
    std::vector<std::wstring> parts;
    if (parse_console(path, parts) != Parsed::Console || parts.empty()) return;
    const Roots r = roots();
    if (r.game.empty() || r.overlay.empty() || deleted(r.overlay, parts)) return;
    std::error_code ec;
    if (!std::filesystem::is_regular_file(join(r.game, parts), ec)) return;
    DWORD error = 0;
    write_marker(marker(r.overlay, parts), error);
}

// The marker for a delete-on-close open the host has just granted. The title may read the open's last error
// (ERROR_ALREADY_EXISTS or 0), which the file queries behind the marker would replace, so it is put back. The caller
// reads it as the first thing after the host's open: building `path` allocates, and that may change it too.
void opened_delete_on_close(const std::wstring& path, DWORD last_error) {
    mark_deleted_on_close(path);
    SetLastError(last_error);
}

// ---- FindFirstFile: the overlay and the package merged into one listing
//
// The host enumerates each folder; the results are merged here, overlay first, so a name that exists in both appears
// once with the overlay's times and size, a name the title deleted does not appear at all, and the overlay's marker
// folder never does.
struct Enumeration {
    std::vector<WIN32_FIND_DATAW> entries;
    size_t next = 0;
};

std::mutex g_enum_mutex;
std::map<HANDLE, std::unique_ptr<Enumeration>> g_enumerations;  // keyed by the pseudo-handle handed to the title

HANDLE register_enumeration(std::unique_ptr<Enumeration> e) {
    std::lock_guard lock(g_enum_mutex);
    const HANDLE h = reinterpret_cast<HANDLE>(e.get());
    g_enumerations.emplace(h, std::move(e));
    return h;
}

bool unregister_enumeration(HANDLE h) {
    std::lock_guard lock(g_enum_mutex);
    return g_enumerations.erase(h) != 0;
}

bool collect(const std::wstring& pattern, const std::vector<std::wstring>& parts, std::vector<WIN32_FIND_DATAW>& out,
             DWORD& error) {
    // "GAME:" and "GAME:\\dir\\" name the folder itself: list everything in it.
    std::vector<std::wstring> dirs;
    std::wstring leaf;
    const bool trailing_separator = !pattern.empty() && (pattern.back() == L'\\' || pattern.back() == L'/');
    if (parts.empty() || trailing_separator) {
        dirs = parts;
        leaf = L"*";
    } else {
        dirs.assign(parts.begin(), parts.end() - 1);
        leaf = parts.back();
    }

    const Roots r = roots();
    if (r.game.empty()) {
        error = ERROR_PATH_NOT_FOUND;
        return false;
    }
    // The folder being listed has to exist in the title's view. The host would still enumerate a package folder the
    // title removed (its "." and ".."), and a folder that is missing is a missing path, not a missing file. A listing
    // below a file is a third answer on the host, ERROR_DIRECTORY.
    const Kind listed = view(r, dirs).kind;
    if (listed != Kind::Directory) {
        error = listed == Kind::File ? ERROR_DIRECTORY : ERROR_PATH_NOT_FOUND;
        return false;
    }

    // Every deletion marker for an entry of this folder lives inside its marker folder, so that folder is read once
    // instead of probing it twice per package entry (deleted()). An entry is deleted when its marker is a file, or a
    // folder holding kRemovedDirMarker: the rule deleted() applies. The lookup folds ASCII letters only, which matches
    // the file system's case-insensitive compare exactly when both names are ASCII. Any other name, a non-ASCII entry
    // or a marker that is not plain ASCII or is a reparse point, goes through deleted() and the file system's rule.
    auto ascii = [](const wchar_t* s) {
        for (; *s; ++s)
            if (*s > 0x7f) return false;
        return true;
    };
    auto ascii_key = [](const wchar_t* s) {
        std::wstring k = s;
        for (auto& c : k)
            if (c >= L'A' && c <= L'Z') c = static_cast<wchar_t>(c - L'A' + L'a');
        return k;
    };
    bool has_markers = false;
    bool markers_exact = true;
    std::unordered_set<std::wstring> deleted_here;
    if (!r.overlay.empty()) {
        const std::filesystem::path marks = marker(r.overlay, dirs);
        WIN32_FIND_DATAW m{};
        const HANDLE mh = ::FindFirstFileW(win32_io_path(marks / L"*").c_str(), &m);
        if (mh != INVALID_HANDLE_VALUE) {
            has_markers = true;
            do {
                if (same_name(m.cFileName, L".") || same_name(m.cFileName, L"..")) continue;
                if (!ascii(m.cFileName) || (m.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT)) {
                    markers_exact = false;
                    continue;
                }
                if (!(m.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                    deleted_here.insert(ascii_key(m.cFileName));
                    continue;
                }
                // A marker folder is a removed folder only with kRemovedDirMarker in it; without one it just holds the
                // markers of a nested delete. A missing sentinel is that ordinary answer (status() reports not_found
                // with an error code set); any other error leaves the decision to deleted() per entry.
                std::error_code ec;
                const auto st = std::filesystem::status(win32_io_path(marks / m.cFileName / kRemovedDirMarker), ec);
                if (st.type() == std::filesystem::file_type::regular) deleted_here.insert(ascii_key(m.cFileName));
                else if (st.type() != std::filesystem::file_type::not_found && ec) markers_exact = false;
            } while (::FindNextFileW(mh, &m));
            if (::GetLastError() != ERROR_NO_MORE_FILES) markers_exact = false;  // an incomplete scan
            ::FindClose(mh);
        } else {
            // Only a missing marker folder means "nothing deleted here". Any other failure (a folder that cannot be
            // listed although its markers can be queried by path, for one) leaves the decision to deleted().
            const DWORD e = ::GetLastError();
            if (e != ERROR_FILE_NOT_FOUND && e != ERROR_PATH_NOT_FOUND) {
                has_markers = true;
                markers_exact = false;
            }
        }
    }
    auto entry_deleted = [&](const wchar_t* name) {
        if (!has_markers) return false;
        if (markers_exact && ascii(name)) return deleted_here.count(ascii_key(name)) != 0;
        std::vector<std::wstring> entry_parts = dirs;
        entry_parts.push_back(name);
        return deleted(r.overlay, entry_parts);
    };
    // Names already listed, ordered by _wcsicmp itself, so two names are one entry exactly when same_name says so, in
    // whatever locale the title set (a lower-cased key is not: under .UTF-8 _wcslwr_s folds É and é, _wcsicmp not).
    struct NoCase {
        bool operator()(const std::wstring& a, const std::wstring& b) const { return _wcsicmp(a.c_str(), b.c_str()) < 0; }
    };
    std::set<std::wstring, NoCase> seen;
    for (const auto& e : out) seen.insert(e.cFileName);

    auto add_root = [&](const std::filesystem::path& root, bool overlay_root) {
        if (root.empty()) return;
        WIN32_FIND_DATAW found{};
        const HANDLE h = ::FindFirstFileW(win32_find_path(join(root, dirs) / leaf, leaf).c_str(), &found);
        if (h == INVALID_HANDLE_VALUE) return;
        do {
            if (overlay_root && internal_dir(found.cFileName)) continue;
            const bool dot = same_name(found.cFileName, L".") || same_name(found.cFileName, L"..");
            if (!overlay_root && !dot && entry_deleted(found.cFileName)) continue;
            if (seen.insert(found.cFileName).second) out.push_back(found);
        } while (::FindNextFileW(h, &found));
        ::FindClose(h);
    };
    add_root(r.overlay, true);
    add_root(r.game, false);

    if (out.empty()) {
        error = ERROR_FILE_NOT_FOUND;
        return false;
    }
    return true;
}

void to_ansi(const WIN32_FIND_DATAW& wide, WIN32_FIND_DATAA* ansi) {
    *ansi = WIN32_FIND_DATAA{};
    ansi->dwFileAttributes = wide.dwFileAttributes;
    ansi->ftCreationTime = wide.ftCreationTime;
    ansi->ftLastAccessTime = wide.ftLastAccessTime;
    ansi->ftLastWriteTime = wide.ftLastWriteTime;
    ansi->nFileSizeHigh = wide.nFileSizeHigh;
    ansi->nFileSizeLow = wide.nFileSizeLow;
    ansi->dwReserved0 = wide.dwReserved0;
    ansi->dwReserved1 = wide.dwReserved1;
    narrow_ansi(wide.cFileName, ansi->cFileName, MAX_PATH);
    narrow_ansi(wide.cAlternateFileName, ansi->cAlternateFileName, 14);
}

}  // namespace

void configure(const std::filesystem::path& game_dir, const std::filesystem::path& overlay_dir) {
    std::lock_guard lock(g_mutex);
    g_game_dir = game_dir;
    g_overlay_dir = overlay_dir;
}

std::filesystem::path game_dir() {
    std::lock_guard lock(g_mutex);
    return g_game_dir;
}

std::filesystem::path overlay_dir() {
    std::lock_guard lock(g_mutex);
    return g_overlay_dir;
}

std::filesystem::path ensure_persistent_storage(std::filesystem::path* host_path) {
    const auto overlay = overlay_dir();
    if (overlay.empty()) return {};
    const auto configured = overlay / kPersistentStorageDir;
    const auto path = extended_path(configured);
    if (path.empty()) return {};
    std::error_code ec;
    std::filesystem::create_directories(path, ec);
    if (ec) return {};
    if (host_path) *host_path = configured;
    return path;
}

// gta6rt --trace FILE: a failed open of a guest path goes to the startup trace as a "file_open_failed" event (the
// path the title passed, the Win32 error, the caller as module+RVA), so a boot that stops on a missing or refused
// file names it. Failures only, and at most 256 of them, so a title probing many optional files can't fill the file.
void note_open_failure(const char* api, LPCWSTR name, DWORD error, const void* caller) {
    if (!rt::startup_trace::enabled()) return;
    static std::atomic<int> left{256};
    if (left.fetch_sub(1, std::memory_order_relaxed) <= 0) return;
    std::string utf8;
    if (const int n = WideCharToMultiByte(CP_UTF8, 0, name, -1, nullptr, 0, nullptr, nullptr); n > 1) {
        utf8.resize(static_cast<size_t>(n) - 1);
        WideCharToMultiByte(CP_UTF8, 0, name, -1, utf8.data(), n, nullptr, nullptr);
    }
    rt::startup_trace::event(
        "file_open_failed",
        std::string("\"api\":\"") + api + "\",\"path\":" + rt::startup_trace::quote(utf8) +
            ",\"error\":" + std::to_string(error) + ",\"caller\":" +
            rt::startup_trace::quote(rt::startup_trace::where(reinterpret_cast<uint64_t>(caller))));
}

HANDLE open_file(const char* api, const void* caller, LPCWSTR name, DWORD access, DWORD share,
                 LPSECURITY_ATTRIBUTES attributes, DWORD disposition, DWORD flags, HANDLE template_file) {
    if (!name) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return INVALID_HANDLE_VALUE;
    }
    const Target t = open_target(name, access, disposition, flags);
    if (t.error) {
        note_open_failure(api, name, t.error, caller);
        SetLastError(t.error);
        return INVALID_HANDLE_VALUE;
    }
    const HANDLE h = ::CreateFileW(t.path.c_str(), access, share, attributes, disposition, flags, template_file);
    const DWORD last_error = GetLastError();
    if (h == INVALID_HANDLE_VALUE) {
        note_open_failure(api, name, last_error, caller);
        SetLastError(last_error);
    } else if ((flags & FILE_FLAG_DELETE_ON_CLOSE) != 0) {
        opened_delete_on_close(name, last_error);
    }
    // Diagnostic: LEONIDA_IO_LOG=1 prints the first 3000 opens (guest path, result, caller).
    static const bool io_log = std::getenv("LEONIDA_IO_LOG") != nullptr;
    if (io_log) {
        static std::atomic<int> left{3000};
        if (left.fetch_sub(1, std::memory_order_relaxed) > 0)
            std::fprintf(stderr, "[rt] io-log: %s %ls -> %s (err %lu, caller %s)\n", api, name,
                         h == INVALID_HANDLE_VALUE ? "FAIL" : "ok", h == INVALID_HANDLE_VALUE ? last_error : 0ul,
                         rt::startup_trace::where(reinterpret_cast<uint64_t>(caller)).c_str());
    }
    if (h != INVALID_HANDLE_VALUE && t.replaces_package_file) SetLastError(ERROR_ALREADY_EXISTS);
    return h;
}

HANDLE WINAPI create_file_w(LPCWSTR name, DWORD access, DWORD share, LPSECURITY_ATTRIBUTES attributes,
                            DWORD disposition, DWORD flags, HANDLE template_file) {
    return open_file("CreateFileW", _ReturnAddress(), name, access, share, attributes, disposition, flags, template_file);
}

HANDLE WINAPI create_file_a(LPCSTR name, DWORD access, DWORD share, LPSECURITY_ATTRIBUTES attributes,
                            DWORD disposition, DWORD flags, HANDLE template_file) {
    if (!name) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return INVALID_HANDLE_VALUE;
    }
    const std::wstring wide = widen_ansi(name);
    return open_file("CreateFileA", _ReturnAddress(), wide.c_str(), access, share, attributes, disposition, flags, template_file);
}

HANDLE WINAPI create_file2(LPCWSTR name, DWORD access, DWORD share, DWORD disposition,
                           LPCREATEFILE2_EXTENDED_PARAMETERS params) {
    if (!name) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return INVALID_HANDLE_VALUE;
    }
    const DWORD flags = params ? params->dwFileFlags : 0;
    const Target t = open_target(name, access, disposition, flags);
    if (t.error) {
        note_open_failure("CreateFile2", name, t.error, _ReturnAddress());
        SetLastError(t.error);
        return INVALID_HANDLE_VALUE;
    }
    const HANDLE h = ::CreateFile2(t.path.c_str(), access, share, disposition, params);
    const DWORD last_error = GetLastError();
    if (h == INVALID_HANDLE_VALUE) {
        note_open_failure("CreateFile2", name, last_error, _ReturnAddress());
        SetLastError(last_error);
    } else if ((flags & FILE_FLAG_DELETE_ON_CLOSE) != 0) {
        opened_delete_on_close(name, last_error);
    }
    // Diagnostic: LEONIDA_IO_LOG=1 prints the first 3000 opens (guest path, result, caller).
    static const bool io_log = std::getenv("LEONIDA_IO_LOG") != nullptr;
    if (io_log) {
        static std::atomic<int> left{3000};
        if (left.fetch_sub(1, std::memory_order_relaxed) > 0)
            std::fprintf(stderr, "[rt] io-log: CreateFile2 %ls -> %s (err %lu, caller %s)\n", name,
                         h == INVALID_HANDLE_VALUE ? "FAIL" : "ok", h == INVALID_HANDLE_VALUE ? last_error : 0ul,
                         rt::startup_trace::where(reinterpret_cast<uint64_t>(_ReturnAddress())).c_str());
    }
    if (h != INVALID_HANDLE_VALUE && t.replaces_package_file) SetLastError(ERROR_ALREADY_EXISTS);
    return h;
}

DWORD WINAPI get_file_attributes_w(LPCWSTR name) {
    if (!name) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return INVALID_FILE_ATTRIBUTES;
    }
    const Target t = read_target(name);
    if (t.error) {
        SetLastError(t.error);
        return INVALID_FILE_ATTRIBUTES;
    }
    return ::GetFileAttributesW(t.path.c_str());
}

DWORD WINAPI get_file_attributes_a(LPCSTR name) {
    if (!name) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return INVALID_FILE_ATTRIBUTES;
    }
    const std::wstring wide = widen_ansi(name);
    return get_file_attributes_w(wide.c_str());
}

// GetFullPathName on a "GAME:" path. The host function reads "GAME:\x" as a relative name and answers
// "<current folder>\GAME:\x", which no longer names the console drive, so a caller that makes its path absolute
// before opening it (Rust's std::fs, std::filesystem::absolute) lost the game folder: every "GAME:" open failed
// with ERROR_PATH_NOT_FOUND while the same name through CreateFileW worked. A "GAME:" path is already absolute: it
// comes back with one separator style and no "." or ".." parts, and a trailing separator stays, as Windows keeps it
// for "C:\foo\" (then there is no file part). A path that climbs above GAME:\ is refused with ERROR_INVALID_NAME,
// as CreateFileW refuses the same name, and never reaches the host. Every other path goes to the host.
enum class FullPath { NotGame, Refused, Answered };

FullPath full_game_path(const wchar_t* name, std::wstring& full, bool& has_file_part) {
    const std::wstring path = name;
    size_t prefix_length = 0;
    if (console_drive(path, prefix_length) != Drive::Game || prefix_length != kGamePrefixLength) return FullPath::NotGame;
    std::vector<std::wstring> parts;
    if (parse_console(path, parts) != Parsed::Console) return FullPath::Refused;
    full = L"GAME:";
    for (const auto& part : parts) full += L'\\' + part;
    const bool trailing_separator = !path.empty() && is_separator(path.back());
    if (parts.empty() || trailing_separator) full += L'\\';
    has_file_part = !parts.empty() && !trailing_separator;
    return FullPath::Answered;
}

DWORD WINAPI get_full_path_name_w(LPCWSTR name, DWORD length, LPWSTR buffer, LPWSTR* file_part) {
    std::wstring full;
    bool has_file_part = false;
    const FullPath kind = name ? full_game_path(name, full, has_file_part) : FullPath::NotGame;
    if (kind == FullPath::NotGame) return ::GetFullPathNameW(name, length, buffer, file_part);
    if (file_part) *file_part = nullptr;  // as when the host fails or the buffer is too small: no file part
    if (kind == FullPath::Refused) {
        SetLastError(ERROR_INVALID_NAME);
        return 0;
    }
    if (!buffer || full.size() + 1 > length) return static_cast<DWORD>(full.size() + 1);  // size query or too small: the size with the NUL
    std::wmemcpy(buffer, full.c_str(), full.size() + 1);
    if (file_part) *file_part = has_file_part ? buffer + full.rfind(L'\\') + 1 : nullptr;
    return static_cast<DWORD>(full.size());
}

DWORD WINAPI get_full_path_name_a(LPCSTR name, DWORD length, LPSTR buffer, LPSTR* file_part) {
    std::wstring full;
    bool has_file_part = false;
    const FullPath kind = name ? full_game_path(widen_ansi(name).c_str(), full, has_file_part) : FullPath::NotGame;
    if (kind == FullPath::NotGame) return ::GetFullPathNameA(name, length, buffer, file_part);
    if (file_part) *file_part = nullptr;
    if (kind == FullPath::Refused) {
        SetLastError(ERROR_INVALID_NAME);
        return 0;
    }
    const int needed = WideCharToMultiByte(CP_ACP, 0, full.c_str(), -1, nullptr, 0, nullptr, nullptr);  // with the NUL
    if (needed <= 0) {
        if (GetLastError() == ERROR_SUCCESS) SetLastError(ERROR_NO_UNICODE_TRANSLATION);  // never fail with a stale or zero error
        return 0;
    }
    if (!buffer || static_cast<DWORD>(needed) > length) return static_cast<DWORD>(needed);  // size query or too small
    WideCharToMultiByte(CP_ACP, 0, full.c_str(), -1, buffer, needed, nullptr, nullptr);
    if (file_part) *file_part = has_file_part ? std::strrchr(buffer, '\\') + 1 : nullptr;
    return static_cast<DWORD>(needed - 1);
}

BOOL WINAPI get_file_attributes_ex_w(LPCWSTR name, GET_FILEEX_INFO_LEVELS level, LPVOID info) {
    if (!name || !info) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
    const Target t = read_target(name);
    if (t.error) {
        SetLastError(t.error);
        return FALSE;
    }
    return ::GetFileAttributesExW(t.path.c_str(), level, info);
}

BOOL WINAPI get_file_attributes_ex_a(LPCSTR name, GET_FILEEX_INFO_LEVELS level, LPVOID info) {
    if (!name || !info) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
    const std::wstring wide = widen_ansi(name);
    return get_file_attributes_ex_w(wide.c_str(), level, info);
}

HANDLE WINAPI find_first_file_w(LPCWSTR pattern, LPWIN32_FIND_DATAW data) {
    static const bool io_log = std::getenv("LEONIDA_IO_LOG") != nullptr;
    if (io_log && pattern) std::fprintf(stderr, "[rt] io-log: FindFirstFileW %ls\n", pattern);
    if (!pattern || !data) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return INVALID_HANDLE_VALUE;
    }
    if (auto target = persistent_target(pattern)) {
        if (target->error) {
            SetLastError(target->error);
            return INVALID_HANDLE_VALUE;
        }
        return ::FindFirstFileW(target->path.c_str(), data);
    }
    std::vector<std::wstring> parts;
    const Parsed parsed = parse_console(pattern, parts);
    if (parsed == Parsed::Host) return ::FindFirstFileW(pattern, data);
    if (parsed == Parsed::Refused) {
        SetLastError(ERROR_INVALID_NAME);
        return INVALID_HANDLE_VALUE;
    }

    std::vector<WIN32_FIND_DATAW> entries;
    DWORD error = 0;
    if (!collect(pattern, parts, entries, error)) {
        SetLastError(error);
        return INVALID_HANDLE_VALUE;
    }
    auto e = std::make_unique<Enumeration>();
    e->entries = std::move(entries);
    e->next = 1;
    *data = e->entries[0];
    return register_enumeration(std::move(e));
}

HANDLE WINAPI find_first_file_ex_w(LPCWSTR pattern, FINDEX_INFO_LEVELS info_level, LPVOID data,
                                   FINDEX_SEARCH_OPS search_op, LPVOID search_filter, DWORD additional_flags) {
    static const bool io_log = std::getenv("LEONIDA_IO_LOG") != nullptr;
    if (io_log && pattern) std::fprintf(stderr, "[rt] io-log: FindFirstFileExW %ls\n", pattern);
    if (!pattern || !data) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return INVALID_HANDLE_VALUE;
    }
    if (auto target = persistent_target(pattern)) {
        if (target->error) {
            SetLastError(target->error);
            return INVALID_HANDLE_VALUE;
        }
        return ::FindFirstFileExW(target->path.c_str(), info_level, data, search_op, search_filter, additional_flags);
    }
    std::vector<std::wstring> parts;
    const Parsed parsed = parse_console(pattern, parts);
    if (parsed == Parsed::Host)
        return ::FindFirstFileExW(pattern, info_level, data, search_op, search_filter, additional_flags);
    if (parsed == Parsed::Refused) {
        SetLastError(ERROR_INVALID_NAME);
        return INVALID_HANDLE_VALUE;
    }
    if (info_level != FindExInfoStandard && info_level != FindExInfoBasic) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return INVALID_HANDLE_VALUE;
    }
    if (search_op != FindExSearchNameMatch && search_op != FindExSearchLimitToDirectories) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return INVALID_HANDLE_VALUE;
    }

    std::vector<WIN32_FIND_DATAW> entries;
    DWORD error = 0;
    if (!collect(pattern, parts, entries, error)) {
        SetLastError(error);
        return INVALID_HANDLE_VALUE;
    }
    if (search_op == FindExSearchLimitToDirectories) {
        std::vector<WIN32_FIND_DATAW> dirs_only;
        for (auto& entry : entries)
            if ((entry.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0) dirs_only.push_back(entry);
        entries = std::move(dirs_only);
        if (entries.empty()) {
            SetLastError(ERROR_FILE_NOT_FOUND);
            return INVALID_HANDLE_VALUE;
        }
    }
    if (info_level == FindExInfoBasic) {
        for (auto& entry : entries) entry.cAlternateFileName[0] = L'\0';
    }
    auto e = std::make_unique<Enumeration>();
    e->entries = std::move(entries);
    e->next = 1;
    *static_cast<LPWIN32_FIND_DATAW>(data) = e->entries[0];
    return register_enumeration(std::move(e));
}

HANDLE WINAPI find_first_file_ex_a(LPCSTR pattern, FINDEX_INFO_LEVELS info_level, LPVOID data,
                                   FINDEX_SEARCH_OPS search_op, LPVOID search_filter, DWORD additional_flags) {
    if (!pattern || !data) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return INVALID_HANDLE_VALUE;
    }
    const std::wstring wide = widen_ansi(pattern);
    std::vector<std::wstring> parts;
    if (!persistent_guest_path(wide) && parse_console(wide, parts) == Parsed::Host)
        return ::FindFirstFileExA(pattern, info_level, data, search_op, search_filter, additional_flags);
    WIN32_FIND_DATAW wide_data{};
    const HANDLE h = find_first_file_ex_w(wide.c_str(), info_level, &wide_data, search_op, search_filter, additional_flags);
    if (h != INVALID_HANDLE_VALUE) to_ansi(wide_data, static_cast<LPWIN32_FIND_DATAA>(data));
    return h;
}

HANDLE WINAPI find_first_file_a(LPCSTR pattern, LPWIN32_FIND_DATAA data) {
    if (!pattern || !data) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return INVALID_HANDLE_VALUE;
    }
    const std::wstring wide = widen_ansi(pattern);
    WIN32_FIND_DATAW found{};
    const HANDLE h = find_first_file_w(wide.c_str(), &found);
    if (h != INVALID_HANDLE_VALUE) to_ansi(found, data);
    return h;
}

BOOL WINAPI find_next_file_w(HANDLE find, LPWIN32_FIND_DATAW data) {
    if (!find || !data) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
    {
        std::lock_guard lock(g_enum_mutex);
        const auto it = g_enumerations.find(find);
        if (it != g_enumerations.end()) {
            Enumeration& e = *it->second;
            if (e.next >= e.entries.size()) {
                SetLastError(ERROR_NO_MORE_FILES);
                return FALSE;
            }
            *data = e.entries[e.next++];
            SetLastError(ERROR_SUCCESS);
            return TRUE;
        }
    }
    return ::FindNextFileW(find, data);  // a host listing
}

BOOL WINAPI find_next_file_a(HANDLE find, LPWIN32_FIND_DATAA data) {
    if (!find || !data) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
    WIN32_FIND_DATAW found{};
    const BOOL ok = find_next_file_w(find, &found);
    if (ok) to_ansi(found, data);
    return ok;
}

BOOL WINAPI find_close(HANDLE find) {
    if (!find) {
        SetLastError(ERROR_INVALID_HANDLE);
        return FALSE;
    }
    if (unregister_enumeration(find)) return TRUE;
    return ::FindClose(find);
}

BOOL WINAPI delete_file_w(LPCWSTR name) {
    if (!name) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
    if (auto target = persistent_target(name, false))
        return target->error ? fail_bool(target->error) : ::DeleteFileW(target->path.c_str());
    std::vector<std::wstring> parts;
    const Parsed parsed = parse_console(name, parts);
    if (parsed == Parsed::Host) return ::DeleteFileW(name);
    if (parsed == Parsed::Refused) {
        SetLastError(ERROR_INVALID_NAME);
        return FALSE;
    }

    const Roots r = roots();
    if (r.overlay.empty() || parts.empty()) {
        SetLastError(ERROR_INVALID_NAME);
        return FALSE;
    }

    bool removed = false;
    std::error_code ec;
    const std::filesystem::path overlay_path = disk_path(r.overlay, parts);
    if (std::filesystem::exists(overlay_path, ec)) {
        // Refuse a package directory before changing a same-named overlay file.
        if (!r.game.empty() && !deleted(r.overlay, parts) &&
            std::filesystem::is_directory(join(r.game, parts), ec))
            return fail_bool(ERROR_ACCESS_DENIED);
        if (!::DeleteFileW(overlay_path.c_str())) return FALSE;
        removed = true;
    }
    if (!r.game.empty() && !deleted(r.overlay, parts)) {
        const std::filesystem::path package_path = disk_path(r.game, parts);
        if (std::filesystem::exists(package_path, ec)) {
            if (std::filesystem::is_directory(package_path, ec)) {
                SetLastError(ERROR_ACCESS_DENIED);
                return FALSE;
            }
            // The overlay delete already checked its own attributes. Without an overlay, the package file's
            // read-only bit still applies even though deletion only writes a marker instead of deleting that file.
            if (!removed) {
                const DWORD attributes = ::GetFileAttributesW(package_path.c_str());
                if (attributes != INVALID_FILE_ATTRIBUTES && (attributes & FILE_ATTRIBUTE_READONLY) != 0)
                    return fail_bool(ERROR_ACCESS_DENIED);
            }
            // The package file stays where it is; the marker is what hides it.
            DWORD error = 0;
            if (!write_marker(marker(r.overlay, parts), error)) return fail_bool(error);
            removed = true;
        }
    }
    if (!removed) {
        SetLastError(ERROR_FILE_NOT_FOUND);
        return FALSE;
    }
    return TRUE;
}

BOOL WINAPI delete_file_a(LPCSTR name) {
    if (!name) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
    const std::wstring wide = widen_ansi(name);
    return delete_file_w(wide.c_str());
}

// ---- Delete through a handle (RT-369)
//
// std::filesystem::remove, which msvcprt.lib links into the title itself (filesystem.obj), deletes through a handle:
// CreateFileW with DELETE, then SetFileInformationByHandle with FileDispositionInfoEx or FileDispositionInfo. DELETE
// makes the open a write, so the handle names the overlay copy of a package file; deleting only that copy would bring
// the package file back into view. A delete request on a handle under the overlay therefore also leaves the marker
// DeleteFile and RemoveDirectory leave.

std::wstring final_path(HANDLE h) {
    const DWORD n = ::GetFinalPathNameByHandleW(h, nullptr, 0, FILE_NAME_NORMALIZED);
    if (n == 0) return {};
    std::wstring path(n, L'\0');
    const DWORD got = ::GetFinalPathNameByHandleW(h, path.data(), n, FILE_NAME_NORMALIZED);
    path.resize(got < n ? got : 0);
    return path;
}

// The overlay-relative components of the file a handle names. False when it is not below the overlay root or is in
// one of the overlay's own folders: a host file, or a package file opened for reading.
bool overlay_parts(HANDLE file, const std::filesystem::path& overlay, std::vector<std::wstring>& parts) {
    const HANDLE root_handle =
        ::CreateFileW(overlay.c_str(), 0, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                      OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr);
    if (root_handle == INVALID_HANDLE_VALUE) return false;
    const std::wstring root = final_path(root_handle);
    ::CloseHandle(root_handle);
    const std::wstring path = final_path(file);
    if (root.empty() || path.size() <= root.size() + 1 || path[root.size()] != L'\\' ||
        ::CompareStringOrdinal(path.c_str(), static_cast<int>(root.size()), root.c_str(),
                               static_cast<int>(root.size()), TRUE) != CSTR_EQUAL)
        return false;
    for (size_t start = root.size() + 1; start < path.size();) {
        const size_t end = std::min(path.find(L'\\', start), path.size());
        parts.push_back(path.substr(start, end - start));
        start = end + 1;
    }
    return !parts.empty() && !internal_dir(parts.front().c_str());
}

bool sets_delete(FILE_INFO_BY_HANDLE_CLASS info_class, const void* info, DWORD size) {
    if (!info) return false;
    if (info_class == FileDispositionInfo && size >= sizeof(FILE_DISPOSITION_INFO))
        return static_cast<const FILE_DISPOSITION_INFO*>(info)->DeleteFile != FALSE;
    if (info_class == FileDispositionInfoEx && size >= sizeof(FILE_DISPOSITION_INFO_EX))
        return (static_cast<const FILE_DISPOSITION_INFO_EX*>(info)->Flags & FILE_DISPOSITION_FLAG_DELETE) != 0;
    return false;
}

BOOL WINAPI set_file_information_by_handle(HANDLE file, FILE_INFO_BY_HANDLE_CLASS info_class, LPVOID info,
                                           DWORD size) {
    const Roots r = roots();
    std::vector<std::wstring> parts;
    if (!sets_delete(info_class, info, size) || r.overlay.empty() || !overlay_parts(file, r.overlay, parts))
        return ::SetFileInformationByHandle(file, info_class, info, size);
    // A package folder goes only once nothing is left in it, as RemoveDirectory requires.
    const View v = view(r, parts);
    if (v.in_package && v.kind == Kind::Directory && !empty_in_view(r, parts)) return fail_bool(ERROR_DIR_NOT_EMPTY);
    if (!::SetFileInformationByHandle(file, info_class, info, size)) return FALSE;
    if (!v.in_package) return TRUE;
    std::filesystem::path mark = marker(r.overlay, parts);
    if (v.kind == Kind::Directory) mark /= kRemovedDirMarker;
    DWORD error = 0;
    return write_marker(mark, error) ? TRUE : fail_bool(error);
}

// ---- Folders: created in the overlay, removed by a marker when the package has them

namespace {

BOOL create_directory(LPCWSTR template_name, LPCWSTR name, LPSECURITY_ATTRIBUTES security) {
    if (!name) return fail_bool(ERROR_INVALID_PARAMETER);
    std::wstring template_path;
    if (template_name) {
        const Target t = read_target(template_name);
        if (t.error) return fail_bool(t.error);
        template_path = t.path.wstring();
    }
    if (auto target = persistent_target(name)) {
        if (target->error) return fail_bool(target->error);
        return template_name ? ::CreateDirectoryExW(template_path.c_str(), target->path.c_str(), security)
                             : ::CreateDirectoryW(target->path.c_str(), security);
    }
    std::vector<std::wstring> parts;
    const Parsed parsed = parse_console(name, parts);
    if (parsed == Parsed::Host)
        return template_name ? ::CreateDirectoryExW(template_path.c_str(), name, security)
                             : ::CreateDirectoryW(name, security);
    if (parsed == Parsed::Refused) return fail_bool(ERROR_INVALID_NAME);

    const Roots r = roots();
    if (r.game.empty() || r.overlay.empty()) return fail_bool(ERROR_PATH_NOT_FOUND);
    if (view(r, parts).kind != Kind::None) return fail_bool(ERROR_ALREADY_EXISTS);  // "GAME:" included
    // From the parent's check to the new folder: a RemoveDirectory of the parent comes wholly before or after.
    std::lock_guard folder(folder_lock(parent_of(parts)));
    if (!is_folder(r, parent_of(parts))) return fail_bool(ERROR_PATH_NOT_FOUND);

    const std::filesystem::path target = disk_path(r.overlay, parts);
    std::error_code ec;
    std::filesystem::create_directories(target.parent_path(), ec);
    if (ec) return fail_bool(static_cast<DWORD>(ec.value()));
    const BOOL created = template_name ? ::CreateDirectoryExW(template_path.c_str(), target.c_str(), security)
                                       : ::CreateDirectoryW(target.c_str(), security);
    if (created) {
        // Keep markers for deleted children, which must remain hidden after the folder is recreated.
        std::filesystem::remove(marker(r.overlay, parts) / kRemovedDirMarker, ec);
        if (ec) {
            const DWORD error = static_cast<DWORD>(ec.value());
            ::RemoveDirectoryW(target.c_str());
            return fail_bool(error);
        }
    }
    return created;
}

}  // namespace

BOOL WINAPI create_directory_w(LPCWSTR name, LPSECURITY_ATTRIBUTES security) {
    return create_directory(nullptr, name, security);
}

BOOL WINAPI create_directory_a(LPCSTR name, LPSECURITY_ATTRIBUTES security) {
    if (!name) return fail_bool(ERROR_INVALID_PARAMETER);
    const std::wstring wide = widen_ansi(name);
    return create_directory(nullptr, wide.c_str(), security);
}

BOOL WINAPI create_directory_ex_w(LPCWSTR template_name, LPCWSTR name, LPSECURITY_ATTRIBUTES security) {
    if (!template_name) return fail_bool(ERROR_INVALID_PARAMETER);
    return create_directory(template_name, name, security);
}

BOOL WINAPI create_directory_ex_a(LPCSTR template_name, LPCSTR name, LPSECURITY_ATTRIBUTES security) {
    if (!template_name || !name) return fail_bool(ERROR_INVALID_PARAMETER);
    const std::wstring wide_template = widen_ansi(template_name), wide = widen_ansi(name);
    return create_directory(wide_template.c_str(), wide.c_str(), security);
}

BOOL WINAPI remove_directory_w(LPCWSTR name) {
    if (!name) return fail_bool(ERROR_INVALID_PARAMETER);
    if (auto target = persistent_target(name, false))
        return target->error ? fail_bool(target->error) : ::RemoveDirectoryW(target->path.c_str());
    std::vector<std::wstring> parts;
    const Parsed parsed = parse_console(name, parts);
    if (parsed == Parsed::Host) return ::RemoveDirectoryW(name);
    if (parsed == Parsed::Refused) return fail_bool(ERROR_INVALID_NAME);

    const Roots r = roots();
    if (r.game.empty() || r.overlay.empty()) return fail_bool(ERROR_PATH_NOT_FOUND);
    if (drive_root(name, parts)) return fail_bool(ERROR_ACCESS_DENIED);  // "GAME:" and "T:" themselves stay
    // From the emptiness check to the marker: a create in the folder comes wholly before or after (g_folder_locks).
    std::lock_guard folder(folder_lock(parts));
    const View v = view(r, parts);
    if (v.kind == Kind::None) return fail_bool(ERROR_FILE_NOT_FOUND);
    if (v.kind == Kind::File) return fail_bool(ERROR_DIRECTORY);
    if (!empty_in_view(r, parts)) return fail_bool(ERROR_DIR_NOT_EMPTY);

    if (v.in_overlay && !::RemoveDirectoryW(disk_path(r.overlay, parts).c_str())) return FALSE;  // its error
    if (v.in_package) {
        // The package folder stays where it is; the marker hides it, and its files are already hidden one by one.
        DWORD error = 0;
        if (!write_marker(marker(r.overlay, parts) / kRemovedDirMarker, error)) return fail_bool(error);
    }
    return TRUE;
}

BOOL WINAPI remove_directory_a(LPCSTR name) {
    if (!name) return fail_bool(ERROR_INVALID_PARAMETER);
    const std::wstring wide = widen_ansi(name);
    return remove_directory_w(wide.c_str());
}

// ---- Move, copy and attributes: the result lands in the overlay, a moved package name is deleted by a marker

BOOL WINAPI move_file_ex_w(LPCWSTR from, LPCWSTR to, DWORD flags) {
    if (!from) return fail_bool(ERROR_INVALID_PARAMETER);
    const auto persistent_from = persistent_target(from, false);
    const auto persistent_to = to ? persistent_target(to, false) : std::nullopt;
    if (persistent_from && persistent_from->error) return fail_bool(persistent_from->error);
    if (persistent_to && persistent_to->error) return fail_bool(persistent_to->error);
    if (persistent_from || persistent_to) {
        if (!to || (flags & MOVEFILE_DELAY_UNTIL_REBOOT) != 0) return fail_bool(ERROR_INVALID_PARAMETER);
        if (persistent_from) from = persistent_from->path.c_str();
        if (persistent_to) to = persistent_to->path.c_str();
    }
    std::vector<std::wstring> src, dst;
    const Parsed from_kind = parse_console(from, src);
    const Parsed to_kind = to ? parse_console(to, dst) : Parsed::Host;
    if (from_kind == Parsed::Host && to_kind == Parsed::Host) return ::MoveFileExW(from, to, flags);
    if (from_kind == Parsed::Refused || to_kind == Parsed::Refused) return fail_bool(ERROR_INVALID_NAME);
    // A boot-time move or delete is a host feature a title cannot use on the console's drives.
    if (!to || (flags & MOVEFILE_DELAY_UNTIL_REBOOT) != 0) return fail_bool(ERROR_INVALID_PARAMETER);
    // GAME: is its own volume: a move between it and a host path is refused the way a cross-volume move is.
    if (from_kind != to_kind) return fail_bool(ERROR_NOT_SAME_DEVICE);

    const Roots r = roots();
    if (r.game.empty() || r.overlay.empty()) return fail_bool(ERROR_PATH_NOT_FOUND);
    if (drive_root(from, src) || drive_root(to, dst)) return fail_bool(ERROR_ACCESS_DENIED);
    const View from_view = view(r, src);
    if (from_view.kind == Kind::None) return fail_bool(ERROR_FILE_NOT_FOUND);
    const std::filesystem::path overlay_from = disk_path(r.overlay, src), overlay_to = disk_path(r.overlay, dst);
    if (same_entry(src, dst)) {
        // The destination is the source, so it is not "already there". The host succeeds at a move onto the same
        // name and changes nothing, and renames when only the case differs. Here the case of a name is the overlay's
        // when the overlay holds it, so a package file gets an overlay copy under the new spelling (no marker: the
        // listing takes the overlay's entry and skips the package's as the same name); a package folder is left
        // where it is, as for any other move of one.
        if (src == dst) return TRUE;
        if (from_view.kind == Kind::Directory && from_view.in_package) return fail_bool(ERROR_ACCESS_DENIED);
        if (from_view.in_overlay)
            return ::MoveFileExW(overlay_from.c_str(), overlay_to.c_str(), flags & MOVEFILE_WRITE_THROUGH);
        std::error_code same_ec;
        std::filesystem::create_directories(overlay_to.parent_path(), same_ec);
        if (same_ec) return fail_bool(static_cast<DWORD>(same_ec.value()));
        if (!::CopyFileW(disk_path(r.game, src).c_str(), overlay_to.c_str(), TRUE)) return FALSE;
        DWORD flush_error = 0;
        if ((flags & MOVEFILE_WRITE_THROUGH) != 0 && !flush_file(overlay_to, flush_error)) return fail_bool(flush_error);
        return TRUE;
    }

    const View to_view = view(r, dst);
    if (to_view.kind == Kind::Directory) return fail_bool(ERROR_ACCESS_DENIED);
    if (to_view.kind == Kind::File && (flags & MOVEFILE_REPLACE_EXISTING) == 0) return fail_bool(ERROR_ALREADY_EXISTS);
    if (!is_folder(r, parent_of(dst))) return fail_bool(ERROR_PATH_NOT_FOUND);

    std::error_code ec;
    std::filesystem::create_directories(overlay_to.parent_path(), ec);
    if (ec) return fail_bool(static_cast<DWORD>(ec.value()));

    if (from_view.kind == Kind::Directory) {
        // A package folder is left where it is (as DeleteFile does); an overlay-only folder is renamed in place.
        if (from_view.in_package) return fail_bool(ERROR_ACCESS_DENIED);
        if (to_view.kind != Kind::None) return fail_bool(ERROR_ACCESS_DENIED);
        return ::MoveFileExW(overlay_from.c_str(), overlay_to.c_str(), flags & MOVEFILE_WRITE_THROUGH);
    }

    const bool write_through = (flags & MOVEFILE_WRITE_THROUGH) != 0;
    const DWORD keep = flags & (MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH);
    DWORD error = 0;
    if (from_view.in_overlay) {
        if (!::MoveFileExW(overlay_from.c_str(), overlay_to.c_str(), keep | MOVEFILE_COPY_ALLOWED)) return FALSE;
    } else {
        // The host never sees a move here, only a copy and a marker, so MOVEFILE_WRITE_THROUGH is honoured by
        // flushing both before returning, as MoveFileEx does for its own copy and delete.
        const BOOL fail_if_exists = (flags & MOVEFILE_REPLACE_EXISTING) == 0;
        if (!::CopyFileW(disk_path(r.game, src).c_str(), overlay_to.c_str(), fail_if_exists)) return FALSE;
        if (write_through && !flush_file(overlay_to, error)) return fail_bool(error);
    }
    if (from_view.in_package) {
        const std::filesystem::path mark = marker(r.overlay, src);
        if (!write_marker(mark, error)) return fail_bool(error);
        if (write_through && !flush_file(mark, error)) return fail_bool(error);
    }
    return TRUE;
}

BOOL WINAPI move_file_w(LPCWSTR from, LPCWSTR to) {
    if (!from || !to) return fail_bool(ERROR_INVALID_PARAMETER);
    return move_file_ex_w(from, to, MOVEFILE_COPY_ALLOWED);  // MoveFile moves files across volumes
}

BOOL WINAPI move_file_ex_a(LPCSTR from, LPCSTR to, DWORD flags) {
    if (!from) return fail_bool(ERROR_INVALID_PARAMETER);
    const std::wstring wide_from = widen_ansi(from), wide_to = to ? widen_ansi(to) : std::wstring();
    return move_file_ex_w(wide_from.c_str(), to ? wide_to.c_str() : nullptr, flags);
}

BOOL WINAPI move_file_a(LPCSTR from, LPCSTR to) {
    if (!from || !to) return fail_bool(ERROR_INVALID_PARAMETER);
    return move_file_ex_a(from, to, MOVEFILE_COPY_ALLOWED);
}

namespace {

// Where a copy writes: a host path as given; a GAME: path is checked against the title's view (an existing name
// counts even when only the package has it) and then means its overlay copy.
bool copy_destination(LPCWSTR to, bool fail_if_exists, std::filesystem::path& out, DWORD& error) {
    if (auto target = persistent_target(to, false)) {
        error = target->error;
        if (error) return false;
        out = std::move(target->path);
        return true;
    }
    std::vector<std::wstring> parts;
    const Parsed parsed = parse_console(to, parts);
    if (parsed == Parsed::Host) {
        out = std::filesystem::path(to);
        return true;
    }
    error = ERROR_INVALID_NAME;
    if (parsed == Parsed::Refused || parts.empty()) return false;
    const Roots r = roots();
    error = ERROR_PATH_NOT_FOUND;
    if (r.game.empty() || r.overlay.empty()) return false;
    const View v = view(r, parts);
    error = ERROR_ACCESS_DENIED;
    if (v.kind == Kind::Directory) return false;
    error = ERROR_FILE_EXISTS;
    if (v.kind == Kind::File && fail_if_exists) return false;
    error = ERROR_PATH_NOT_FOUND;
    if (!is_folder(r, parent_of(parts))) return false;
    out = disk_path(r.overlay, parts);
    std::error_code ec;
    std::filesystem::create_directories(out.parent_path(), ec);
    if (ec) {
        error = static_cast<DWORD>(ec.value());
        return false;
    }
    error = 0;
    return true;
}

// The source and destination a copy opens, or the error it fails with.
bool copy_paths(LPCWSTR from, LPCWSTR to, bool fail_if_exists, std::filesystem::path& source,
                std::filesystem::path& destination, DWORD& error) {
    if (!from || !to) {
        error = ERROR_INVALID_PARAMETER;
        return false;
    }
    const Target t = read_target(from);
    if (t.error) {
        error = t.error;
        return false;
    }
    source = t.path;
    return copy_destination(to, fail_if_exists, destination, error);
}

}  // namespace

BOOL WINAPI copy_file_w(LPCWSTR from, LPCWSTR to, BOOL fail_if_exists) {
    std::filesystem::path source, destination;
    DWORD error = 0;
    if (!copy_paths(from, to, fail_if_exists != FALSE, source, destination, error)) return fail_bool(error);
    return ::CopyFileW(source.c_str(), destination.c_str(), fail_if_exists);
}

BOOL WINAPI copy_file_a(LPCSTR from, LPCSTR to, BOOL fail_if_exists) {
    if (!from || !to) return fail_bool(ERROR_INVALID_PARAMETER);
    const std::wstring wide_from = widen_ansi(from), wide_to = widen_ansi(to);
    return copy_file_w(wide_from.c_str(), wide_to.c_str(), fail_if_exists);
}

BOOL WINAPI copy_file_ex_w(LPCWSTR from, LPCWSTR to, LPPROGRESS_ROUTINE progress, LPVOID data, LPBOOL cancel,
                           DWORD flags) {
    std::filesystem::path source, destination;
    DWORD error = 0;
    if (!copy_paths(from, to, (flags & COPY_FILE_FAIL_IF_EXISTS) != 0, source, destination, error))
        return fail_bool(error);
    return ::CopyFileExW(source.c_str(), destination.c_str(), progress, data, cancel, flags);
}

BOOL WINAPI copy_file_ex_a(LPCSTR from, LPCSTR to, LPPROGRESS_ROUTINE progress, LPVOID data, LPBOOL cancel,
                           DWORD flags) {
    if (!from || !to) return fail_bool(ERROR_INVALID_PARAMETER);
    const std::wstring wide_from = widen_ansi(from), wide_to = widen_ansi(to);
    return copy_file_ex_w(wide_from.c_str(), wide_to.c_str(), progress, data, cancel, flags);
}

namespace {

// The host's CopyFile2, or null where the host has none. Wine 9's kernel32 and kernelbase don't export it: Wine's
// loader binds a static import of it to a stub that aborts the process when called ("unimplemented function
// KERNEL32.dll.CopyFile2, aborting"), so a direct ::CopyFile2 call (or MSVC's std::filesystem::copy_file, which uses
// it) kills a title under Wine 9 or a Proton built on it; pe_loader.cpp and loader_tls_test.cpp avoid it for the same
// reason. GetProcAddress returns null there instead, and the fallback below runs.
using CopyFile2Fn = HRESULT(WINAPI*)(PCWSTR, PCWSTR, COPYFILE2_EXTENDED_PARAMETERS*);
std::atomic<bool> g_force_copy_file2_fallback{false};
INIT_ONCE g_host_copy_file2_once = INIT_ONCE_STATIC_INIT;
CopyFile2Fn g_host_copy_file2_fn = nullptr;

BOOL CALLBACK resolve_copy_file2_once(PINIT_ONCE, PVOID, PVOID*) noexcept {
    for (const wchar_t* dll : {L"kernelbase.dll", L"kernel32.dll"}) {
        if (const HMODULE module = GetModuleHandleW(dll))
            if (const FARPROC p = GetProcAddress(module, "CopyFile2")) {
                g_host_copy_file2_fn = reinterpret_cast<CopyFile2Fn>(p);
                return TRUE;
            }
    }
    return TRUE;
}

CopyFile2Fn host_copy_file2() {
    InitOnceExecuteOnce(&g_host_copy_file2_once, &resolve_copy_file2_once, nullptr, nullptr);
    return g_force_copy_file2_fallback.load(std::memory_order_relaxed) ? CopyFile2Fn{} : g_host_copy_file2_fn;
}

// The fallback's flags, by what CopyFileExW can do with them (values from the SDK's WinBase.h, spelled out because
// some of the names are only declared for newer NTDDI_VERSION targets):
// - passed on: the flags both calls define, with the same meaning;
// - dropped: CopyFile2-only requests about how the copy runs (privileges, resuming a paused copy, no offload, EDP,
//   destination DACL, compressed SMB traffic, pre-allocation, low-free-space mode, sparse copy) that leave the
//   destination's contents as CopyFileExW writes them;
// - everything else changes what the copy produces (0x80 DIRECTORY, 0x8000 SKIP_ALTERNATE_STREAMS, 0x200000
//   OPEN_AND_COPY_REPARSE_POINT, 0x800000 IGNORE_SOURCE_ENCRYPTION) or isn't defined, so the fallback refuses the
//   call with ERROR_NOT_SUPPORTED instead of copying something else.
constexpr DWORD kFallbackPassedFlags = COPY_FILE_FAIL_IF_EXISTS | COPY_FILE_RESTARTABLE |
                                       COPY_FILE_OPEN_SOURCE_FOR_WRITE | COPY_FILE_ALLOW_DECRYPTED_DESTINATION |
                                       COPY_FILE_COPY_SYMLINK | COPY_FILE_NO_BUFFERING;
constexpr DWORD kFallbackDroppedFlags = 0x00002000 /* REQUEST_SECURITY_PRIVILEGES */ |
                                        0x00004000 /* RESUME_FROM_PAUSE */ | 0x00040000 /* NO_OFFLOAD */ |
                                        0x00400000 /* IGNORE_EDP_BLOCK */ |
                                        0x02000000 /* DONT_REQUEST_DEST_WRITE_DAC */ |
                                        0x04000000 /* DISABLE_PRE_ALLOCATION */ |
                                        0x08000000 /* ENABLE_LOW_FREE_SPACE_MODE */ |
                                        0x10000000 /* REQUEST_COMPRESSED_TRAFFIC */ |
                                        0x20000000 /* ENABLE_SPARSE_COPY */ | 0x80000000 /* DISABLE_SPARSE_COPY */;

// Whether the fallback can run this call as CopyFile2 would: a COPYFILE2_EXTENDED_PARAMETERS block (the V2 block's
// extra fields have no CopyFileExW equivalent) and no flag outside the two sets above.
bool fallback_supports(const COPYFILE2_EXTENDED_PARAMETERS* parameters) {
    if (!parameters) return true;
    if (parameters->dwSize != sizeof(COPYFILE2_EXTENDED_PARAMETERS)) return false;
    return (parameters->dwCopyFlags & ~(kFallbackPassedFlags | kFallbackDroppedFlags)) == 0;
}

// CopyFile2 through CopyFileExW, for hosts without it (only for calls fallback_supports). The cancel flag has the
// same type in both (one already set is answered by copy_file2 before this runs). A COPYFILE2 progress routine has
// another signature and message set, so it isn't called here.
HRESULT copy_file2_fallback(const wchar_t* source, const wchar_t* destination,
                            const COPYFILE2_EXTENDED_PARAMETERS* parameters) {
    const DWORD flags = parameters ? parameters->dwCopyFlags & kFallbackPassedFlags : 0;
    BOOL* const cancel = parameters ? parameters->pfCancel : nullptr;
    if (::CopyFileExW(source, destination, nullptr, nullptr, cancel, flags)) return S_OK;
    const DWORD error = GetLastError();
    return HRESULT_FROM_WIN32(error ? error : ERROR_GEN_FAILURE);
}

}  // namespace

void force_copy_file2_fallback(bool force) { g_force_copy_file2_fallback.store(force, std::memory_order_relaxed); }

// The parameter block is checked before the destination is prepared, so a refused call leaves nothing behind: a
// block shorter than COPYFILE2_EXTENDED_PARAMETERS is E_INVALIDARG, as the host's CopyFile2 answers (measured on
// Windows 11 for dwSize 0, 4 and 31, which also left no destination), and a call the fallback can't run as asked is
// ERROR_NOT_SUPPORTED. A longer block goes to the host's CopyFile2, which checks it itself.
HRESULT WINAPI copy_file2(PCWSTR from, PCWSTR to, COPYFILE2_EXTENDED_PARAMETERS* parameters) {
    if (parameters && parameters->dwSize < sizeof(COPYFILE2_EXTENDED_PARAMETERS)) return E_INVALIDARG;
    const CopyFile2Fn host = host_copy_file2();
    if (!host && !fallback_supports(parameters)) return HRESULT_FROM_WIN32(ERROR_NOT_SUPPORTED);
    // A cancel flag already set is answered here, for the host's CopyFile2 as well as the fallback: Windows returns
    // ERROR_REQUEST_ABORTED and writes nothing, while Wine reads the flag in neither call (Wine 11.20's CopyFile2 logs
    // "pfCancel is not supported" and copies, dlls/kernelbase/file.c copy_file; Wine 9 has only CopyFileExW).
    if (parameters && parameters->pfCancel && *parameters->pfCancel) return HRESULT_FROM_WIN32(ERROR_REQUEST_ABORTED);
    const bool fail_if_exists = parameters && (parameters->dwCopyFlags & COPY_FILE_FAIL_IF_EXISTS) != 0;
    std::filesystem::path source, destination;
    DWORD error = 0;
    if (!copy_paths(from, to, fail_if_exists, source, destination, error)) return HRESULT_FROM_WIN32(error);
    if (host) return host(source.c_str(), destination.c_str(), parameters);
    return copy_file2_fallback(source.c_str(), destination.c_str(), parameters);
}

BOOL WINAPI set_file_attributes_w(LPCWSTR name, DWORD attributes) {
    if (!name) return fail_bool(ERROR_INVALID_PARAMETER);
    if (auto target = persistent_target(name, false))
        return target->error ? fail_bool(target->error) : ::SetFileAttributesW(target->path.c_str(), attributes);
    std::vector<std::wstring> parts;
    const Parsed parsed = parse_console(name, parts);
    if (parsed == Parsed::Host) return ::SetFileAttributesW(name, attributes);
    if (parsed == Parsed::Refused) return fail_bool(ERROR_INVALID_NAME);

    const Roots r = roots();
    if (r.game.empty() || r.overlay.empty()) return fail_bool(ERROR_PATH_NOT_FOUND);
    if (parts.empty()) return fail_bool(ERROR_ACCESS_DENIED);
    const View v = view(r, parts);
    if (v.kind == Kind::None) return fail_bool(ERROR_FILE_NOT_FOUND);

    // The attributes belong to the overlay copy: a package file is copied up and a package folder mirrored first,
    // so the package keeps its own attributes and reads, which prefer the overlay, see the new ones.
    const std::filesystem::path target = disk_path(r.overlay, parts);
    if (!v.in_overlay) {
        std::error_code ec;
        if (v.kind == Kind::Directory) {
            std::filesystem::create_directories(target, ec);
            if (ec) return fail_bool(static_cast<DWORD>(ec.value()));
        } else {
            const Target t = write_target(name, true);
            if (t.error) return fail_bool(t.error);
        }
    }
    return ::SetFileAttributesW(target.c_str(), attributes);
}

BOOL WINAPI set_file_attributes_a(LPCSTR name, DWORD attributes) {
    if (!name) return fail_bool(ERROR_INVALID_PARAMETER);
    const std::wstring wide = widen_ansi(name);
    return set_file_attributes_w(wide.c_str(), attributes);
}

// ---- CRT opens (RT-354)
//
// A package DLL linked against the dynamic CRT opens files through ucrtbase, whose own CreateFileW is the host's, so a
// GAME: path reached it as a relative name. These shims map a console path with open_target, the rules CreateFile
// uses, and hand the host path to ucrtbase's wide function, so the FILE* or descriptor belongs to the CRT the caller
// reads it with (gta6rt links the static CRT). A host path, and an open the CRT rejects itself (a null name, a bad
// mode), go to ucrtbase unchanged.
namespace {

struct OpenIntent {
    DWORD access = 0;
    DWORD disposition = 0;
};

// Not a function-local static: those take a guard through the TLS slot, which on a title thread is the exe's.
std::atomic<HMODULE> g_ucrtbase{nullptr};

template <typename Fn>
Fn ucrt(const char* name) {
    HMODULE module = g_ucrtbase.load(std::memory_order_acquire);
    if (!module) {
        module = ::LoadLibraryW(L"ucrtbase.dll");
        g_ucrtbase.store(module, std::memory_order_release);
    }
    return reinterpret_cast<Fn>(::GetProcAddress(module, name));
}

// The errors the file shims give, mapped as the CRT's own table maps them (anything else is EINVAL there too).
void set_crt_error(DWORD error) {
    int e = EINVAL;
    if (error == ERROR_FILE_NOT_FOUND || error == ERROR_PATH_NOT_FOUND) e = ENOENT;
    if (error == ERROR_ACCESS_DENIED) e = EACCES;
    if (error >= ERROR_WRITE_PROTECT && error <= ERROR_SHARING_BUFFER_EXCEEDED) e = EACCES;  // sharing, locks
    if (error == ERROR_FILE_EXISTS || error == ERROR_ALREADY_EXISTS) e = EEXIST;
    if (error == ERROR_DISK_FULL) e = ENOSPC;
    if (error == ERROR_DIR_NOT_EMPTY) e = ENOTEMPTY;
    if (error == ERROR_NOT_SAME_DEVICE) e = EXDEV;
    ucrt<errno_t(__cdecl*)(unsigned long)>("_set_doserrno")(error);
    ucrt<errno_t(__cdecl*)(int)>("_set_errno")(e);
    ::SetLastError(error);
}

// fopen's mode as the CRT reads it: leading spaces skipped, r opens an existing file, w creates or truncates (wx fails
// on an existing one), a creates or appends, '+' adds the other direction; ",ccs=" and the rest only change how the
// stream is read. Any other first letter is the CRT's to reject.
std::optional<OpenIntent> from_mode(const wchar_t* mode) {
    if (!mode) return std::nullopt;
    while (*mode == L' ') ++mode;
    bool plus = false, exclusive = false;
    for (const wchar_t* p = mode; *p && *p != L','; ++p) {
        if (*p == L'+') plus = true;
        if (*p == L'x') exclusive = true;
    }
    const DWORD other = plus ? GENERIC_READ | GENERIC_WRITE : 0u;
    switch (*mode) {
        case L'r': return OpenIntent{GENERIC_READ | other, OPEN_EXISTING};
        case L'w': return OpenIntent{GENERIC_WRITE | other, static_cast<DWORD>(exclusive ? CREATE_NEW : CREATE_ALWAYS)};
        case L'a': return OpenIntent{GENERIC_WRITE | other, OPEN_ALWAYS};
        default: return std::nullopt;
    }
}

// The _O_ flags as the CRT turns them into a CreateFile disposition.
OpenIntent from_oflag(int oflag) {
    OpenIntent o;
    o.access = (oflag & (_O_WRONLY | _O_RDWR)) ? GENERIC_WRITE : GENERIC_READ;
    switch (oflag & (_O_CREAT | _O_EXCL | _O_TRUNC)) {
        case _O_CREAT: o.disposition = OPEN_ALWAYS; break;
        case _O_CREAT | _O_EXCL:
        case _O_CREAT | _O_EXCL | _O_TRUNC: o.disposition = CREATE_NEW; break;
        case _O_CREAT | _O_TRUNC: o.disposition = CREATE_ALWAYS; break;
        case _O_TRUNC:
        case _O_TRUNC | _O_EXCL: o.disposition = TRUNCATE_EXISTING; break;
        default: o.disposition = OPEN_EXISTING; break;
    }
    return o;
}

// False, with errno set, when a console path is refused. `mapped` is left empty when the caller should pass its own
// arguments through: a host path, or an open the CRT rejects itself.
bool map_open(const wchar_t* name, const std::optional<OpenIntent>& intent, std::wstring& mapped) {
    if (!name || !intent) return true;
    const Target t = open_target(name, intent->access, intent->disposition);
    if (t.error) {
        set_crt_error(t.error);
        return false;
    }
    if (t.path.native() != name) mapped = t.path.native();
    return true;
}

bool map_open_a(const char* name, const std::optional<OpenIntent>& intent, std::wstring& mapped) {
    return !name || map_open(widen_ansi(name).c_str(), intent, mapped);
}

std::optional<OpenIntent> from_mode_a(const char* mode) {
    return mode ? from_mode(widen_ansi(mode).c_str()) : std::nullopt;
}

using WfsopenFn = FILE*(__cdecl*)(const wchar_t*, const wchar_t*, int);
using WfopenSFn = errno_t(__cdecl*)(FILE**, const wchar_t*, const wchar_t*);
using WfreopenFn = FILE*(__cdecl*)(const wchar_t*, const wchar_t*, FILE*);
using WfreopenSFn = errno_t(__cdecl*)(FILE**, const wchar_t*, const wchar_t*, FILE*);
using WsopenDispatchFn = errno_t(__cdecl*)(const wchar_t*, int, int, int, int*, int);

// The errno set_crt_error left, as the _s functions return it.
errno_t crt_errno() { return *ucrt<int*(__cdecl*)()>("_errno")(); }

// ---- The CRT's other path calls (RT-359)
//
// remove, rename, _mkdir and _rmdir change names, which on GAME: can mean a marker over a package file rather than a
// host call, so a console path goes to the Win32 shim the CRT function wraps (DeleteFileW, MoveFileExW with
// MOVEFILE_COPY_ALLOWED, CreateDirectoryW, RemoveDirectoryW) and its error is mapped as the CRT maps it. _stat* and
// _access* only read, so a console path is mapped with read_target and ucrtbase's wide function looks at the copy a
// read would open. A host path, null included, goes to ucrtbase unchanged.

bool is_console(const wchar_t* name) {
    std::vector<std::wstring> parts;
    return name && parse_console(name, parts) != Parsed::Host;
}

bool is_console_a(const char* name) { return name && is_console(widen_ansi(name).c_str()); }

int crt_result(BOOL ok) {
    if (ok) return 0;
    set_crt_error(::GetLastError());
    return -1;
}

// As map_open, for a call that only reads the name.
bool map_read(const wchar_t* name, std::wstring& mapped) {
    if (!name) return true;
    const Target t = read_target(name);
    if (t.error) {
        set_crt_error(t.error);
        return false;
    }
    if (t.path.native() != name) mapped = t.path.native();
    return true;
}

template <typename Stat>
int stat_w(const char* wide_fn, const wchar_t* name, Stat* out) {
    std::wstring mapped;
    if (!map_read(name, mapped)) return -1;
    return ucrt<int(__cdecl*)(const wchar_t*, Stat*)>(wide_fn)(mapped.empty() ? name : mapped.c_str(), out);
}

template <typename Stat>
int stat_a(const char* narrow_fn, const char* wide_fn, const char* name, Stat* out) {
    if (!is_console_a(name)) return ucrt<int(__cdecl*)(const char*, Stat*)>(narrow_fn)(name, out);
    return stat_w(wide_fn, widen_ansi(name).c_str(), out);
}


}  // namespace

FILE* __cdecl crt_wfopen(const wchar_t* name, const wchar_t* mode) {
    std::wstring mapped;
    if (!map_open(name, from_mode(mode), mapped)) return nullptr;
    return ucrt<FILE*(__cdecl*)(const wchar_t*, const wchar_t*)>("_wfopen")(mapped.empty() ? name : mapped.c_str(),
                                                                          mode);
}

FILE* __cdecl crt_fopen(const char* name, const char* mode) {
    std::wstring mapped;
    if (!map_open_a(name, from_mode_a(mode), mapped)) return nullptr;
    if (mapped.empty()) return ucrt<FILE*(__cdecl*)(const char*, const char*)>("fopen")(name, mode);
    return ucrt<FILE*(__cdecl*)(const wchar_t*, const wchar_t*)>("_wfopen")(mapped.c_str(), widen_ansi(mode).c_str());
}

FILE* __cdecl crt_wfsopen(const wchar_t* name, const wchar_t* mode, int share) {
    std::wstring mapped;
    if (!map_open(name, from_mode(mode), mapped)) return nullptr;
    return ucrt<WfsopenFn>("_wfsopen")(mapped.empty() ? name : mapped.c_str(), mode, share);
}

FILE* __cdecl crt_fsopen(const char* name, const char* mode, int share) {
    std::wstring mapped;
    if (!map_open_a(name, from_mode_a(mode), mapped)) return nullptr;
    if (mapped.empty()) return ucrt<FILE*(__cdecl*)(const char*, const char*, int)>("_fsopen")(name, mode, share);
    return ucrt<WfsopenFn>("_wfsopen")(mapped.c_str(), widen_ansi(mode).c_str(), share);
}

errno_t __cdecl crt_wfopen_s(FILE** stream, const wchar_t* name, const wchar_t* mode) {
    std::wstring mapped;
    if (!map_open(name, from_mode(mode), mapped)) {
        if (stream) *stream = nullptr;
        return crt_errno();
    }
    return ucrt<WfopenSFn>("_wfopen_s")(stream, mapped.empty() ? name : mapped.c_str(), mode);
}

errno_t __cdecl crt_fopen_s(FILE** stream, const char* name, const char* mode) {
    std::wstring mapped;
    if (!map_open_a(name, from_mode_a(mode), mapped)) {
        if (stream) *stream = nullptr;
        return crt_errno();
    }
    if (mapped.empty()) {
        return ucrt<errno_t(__cdecl*)(FILE**, const char*, const char*)>("fopen_s")(stream, name, mode);
    }
    return ucrt<WfopenSFn>("_wfopen_s")(stream, mapped.c_str(), widen_ansi(mode).c_str());
}

FILE* __cdecl crt_wfreopen(const wchar_t* name, const wchar_t* mode, FILE* stream) {
    std::wstring mapped;
    if (!map_open(name, from_mode(mode), mapped)) return nullptr;
    return ucrt<WfreopenFn>("_wfreopen")(mapped.empty() ? name : mapped.c_str(), mode, stream);
}

FILE* __cdecl crt_freopen(const char* name, const char* mode, FILE* stream) {
    std::wstring mapped;
    if (!map_open_a(name, from_mode_a(mode), mapped)) return nullptr;
    if (mapped.empty()) {
        return ucrt<FILE*(__cdecl*)(const char*, const char*, FILE*)>("freopen")(name, mode, stream);
    }
    return ucrt<WfreopenFn>("_wfreopen")(mapped.c_str(), widen_ansi(mode).c_str(), stream);
}

errno_t __cdecl crt_wfreopen_s(FILE** out, const wchar_t* name, const wchar_t* mode, FILE* stream) {
    std::wstring mapped;
    if (!map_open(name, from_mode(mode), mapped)) {
        if (out) *out = nullptr;
        return crt_errno();
    }
    return ucrt<WfreopenSFn>("_wfreopen_s")(out, mapped.empty() ? name : mapped.c_str(), mode, stream);
}

errno_t __cdecl crt_freopen_s(FILE** out, const char* name, const char* mode, FILE* stream) {
    std::wstring mapped;
    if (!map_open_a(name, from_mode_a(mode), mapped)) {
        if (out) *out = nullptr;
        return crt_errno();
    }
    if (mapped.empty()) {
        return ucrt<errno_t(__cdecl*)(FILE**, const char*, const char*, FILE*)>("freopen_s")(out, name, mode, stream);
    }
    return ucrt<WfreopenSFn>("_wfreopen_s")(out, mapped.c_str(), widen_ansi(mode).c_str(), stream);
}

// The descriptor opens all end in _wsopen_dispatch, as they do inside the CRT; `secure` is _sopen_s's pmode check.
// _open, _wopen, _sopen and _wsopen are variadic in C: their pmode is the next int argument, which the x64 convention
// passes in the same register a fixed parameter would use.
errno_t __cdecl crt_wsopen_dispatch(const wchar_t* name, int oflag, int share, int pmode, int* fd, int secure) {
    std::wstring mapped;
    if (!map_open(name, from_oflag(oflag), mapped)) {
        if (fd) *fd = -1;
        return crt_errno();
    }
    return ucrt<WsopenDispatchFn>("_wsopen_dispatch")(mapped.empty() ? name : mapped.c_str(), oflag, share, pmode, fd,
                                                      secure);
}

errno_t __cdecl crt_sopen_dispatch(const char* name, int oflag, int share, int pmode, int* fd, int secure) {
    std::wstring mapped;
    if (!map_open_a(name, from_oflag(oflag), mapped)) {
        if (fd) *fd = -1;
        return crt_errno();
    }
    if (mapped.empty()) {
        return ucrt<errno_t(__cdecl*)(const char*, int, int, int, int*, int)>("_sopen_dispatch")(name, oflag, share,
                                                                                               pmode, fd, secure);
    }
    return ucrt<WsopenDispatchFn>("_wsopen_dispatch")(mapped.c_str(), oflag, share, pmode, fd, secure);
}

int __cdecl crt_wopen(const wchar_t* name, int oflag, int pmode) {
    int fd = -1;
    return crt_wsopen_dispatch(name, oflag, _SH_DENYNO, pmode, &fd, 0) == 0 ? fd : -1;
}

int __cdecl crt_open(const char* name, int oflag, int pmode) {
    int fd = -1;
    return crt_sopen_dispatch(name, oflag, _SH_DENYNO, pmode, &fd, 0) == 0 ? fd : -1;
}

int __cdecl crt_wsopen(const wchar_t* name, int oflag, int share, int pmode) {
    int fd = -1;
    return crt_wsopen_dispatch(name, oflag, share, pmode, &fd, 0) == 0 ? fd : -1;
}

int __cdecl crt_sopen(const char* name, int oflag, int share, int pmode) {
    int fd = -1;
    return crt_sopen_dispatch(name, oflag, share, pmode, &fd, 0) == 0 ? fd : -1;
}

errno_t __cdecl crt_wsopen_s(int* fd, const wchar_t* name, int oflag, int share, int pmode) {
    return crt_wsopen_dispatch(name, oflag, share, pmode, fd, 1);
}

errno_t __cdecl crt_sopen_s(int* fd, const char* name, int oflag, int share, int pmode) {
    return crt_sopen_dispatch(name, oflag, share, pmode, fd, 1);
}

// remove and _unlink are the same call in the CRT (_wunlink calls _wremove).
int __cdecl crt_wremove(const wchar_t* name) {
    if (!is_console(name)) return ucrt<int(__cdecl*)(const wchar_t*)>("_wremove")(name);
    return crt_result(delete_file_w(name));
}

int __cdecl crt_remove(const char* name) {
    if (!is_console_a(name)) return ucrt<int(__cdecl*)(const char*)>("remove")(name);
    return crt_result(delete_file_a(name));
}

int __cdecl crt_wrename(const wchar_t* from, const wchar_t* to) {
    if (!is_console(from) && !is_console(to))
        return ucrt<int(__cdecl*)(const wchar_t*, const wchar_t*)>("_wrename")(from, to);
    return crt_result(move_file_ex_w(from, to, MOVEFILE_COPY_ALLOWED));
}

int __cdecl crt_rename(const char* from, const char* to) {
    if (!is_console_a(from) && !is_console_a(to))
        return ucrt<int(__cdecl*)(const char*, const char*)>("rename")(from, to);
    return crt_result(move_file_ex_a(from, to, MOVEFILE_COPY_ALLOWED));
}

int __cdecl crt_wmkdir(const wchar_t* name) {
    if (!is_console(name)) return ucrt<int(__cdecl*)(const wchar_t*)>("_wmkdir")(name);
    return crt_result(create_directory_w(name, nullptr));
}

int __cdecl crt_mkdir(const char* name) {
    if (!is_console_a(name)) return ucrt<int(__cdecl*)(const char*)>("_mkdir")(name);
    return crt_result(create_directory_a(name, nullptr));
}

int __cdecl crt_wrmdir(const wchar_t* name) {
    if (!is_console(name)) return ucrt<int(__cdecl*)(const wchar_t*)>("_wrmdir")(name);
    return crt_result(remove_directory_w(name));
}

int __cdecl crt_rmdir(const char* name) {
    if (!is_console_a(name)) return ucrt<int(__cdecl*)(const char*)>("_rmdir")(name);
    return crt_result(remove_directory_a(name));
}

int __cdecl crt_waccess(const wchar_t* name, int mode) {
    std::wstring mapped;
    if (!map_read(name, mapped)) return -1;
    return ucrt<int(__cdecl*)(const wchar_t*, int)>("_waccess")(mapped.empty() ? name : mapped.c_str(), mode);
}

int __cdecl crt_access(const char* name, int mode) {
    if (!is_console_a(name)) return ucrt<int(__cdecl*)(const char*, int)>("_access")(name, mode);
    return crt_waccess(widen_ansi(name).c_str(), mode);
}

errno_t __cdecl crt_waccess_s(const wchar_t* name, int mode) {
    std::wstring mapped;
    if (!map_read(name, mapped)) return crt_errno();
    return ucrt<errno_t(__cdecl*)(const wchar_t*, int)>("_waccess_s")(mapped.empty() ? name : mapped.c_str(), mode);
}

errno_t __cdecl crt_access_s(const char* name, int mode) {
    if (!is_console_a(name)) return ucrt<errno_t(__cdecl*)(const char*, int)>("_access_s")(name, mode);
    return crt_waccess_s(widen_ansi(name).c_str(), mode);
}

int __cdecl crt_wstat32(const wchar_t* n, struct _stat32* s) { return stat_w("_wstat32", n, s); }
int __cdecl crt_wstat32i64(const wchar_t* n, struct _stat32i64* s) { return stat_w("_wstat32i64", n, s); }
int __cdecl crt_wstat64(const wchar_t* n, struct _stat64* s) { return stat_w("_wstat64", n, s); }
int __cdecl crt_wstat64i32(const wchar_t* n, struct _stat64i32* s) { return stat_w("_wstat64i32", n, s); }
int __cdecl crt_stat32(const char* n, struct _stat32* s) { return stat_a("_stat32", "_wstat32", n, s); }
int __cdecl crt_stat32i64(const char* n, struct _stat32i64* s) { return stat_a("_stat32i64", "_wstat32i64", n, s); }
int __cdecl crt_stat64(const char* n, struct _stat64* s) { return stat_a("_stat64", "_wstat64", n, s); }
int __cdecl crt_stat64i32(const char* n, struct _stat64i32* s) { return stat_a("_stat64i32", "_wstat64i32", n, s); }

BOOL WINAPI get_disk_free_space_ex_w(LPCWSTR directory, PULARGE_INTEGER available, PULARGE_INTEGER total,
                                     PULARGE_INTEGER free_bytes) {
    if (!directory || !*directory) return ::GetDiskFreeSpaceExW(directory, available, total, free_bytes);
    bool handled = false;
    bool is_root = false;
    DWORD error = 0;
    std::filesystem::path resolved;
    if (!resolve_console_readonly_dir(directory, ERROR_PATH_NOT_FOUND, handled, resolved, is_root, error)) {
        if (!handled) return ::GetDiskFreeSpaceExW(directory, available, total, free_bytes);
        return fail_bool(error);
    }
    return ::GetDiskFreeSpaceExW(resolved.c_str(), available, total, free_bytes);
}

BOOL WINAPI get_disk_free_space_ex_a(LPCSTR directory, PULARGE_INTEGER available, PULARGE_INTEGER total,
                                     PULARGE_INTEGER free_bytes) {
    if (!directory) return ::GetDiskFreeSpaceExW(nullptr, available, total, free_bytes);
    const std::wstring wide = widen_ansi(directory);
    return get_disk_free_space_ex_w(wide.c_str(), available, total, free_bytes);
}

BOOL WINAPI get_disk_free_space_w(LPCWSTR root_path, LPDWORD sectors_per_cluster, LPDWORD bytes_per_sector,
                                  LPDWORD free_clusters, LPDWORD total_clusters) {
    if (!root_path || !*root_path)
        return ::GetDiskFreeSpaceW(root_path, sectors_per_cluster, bytes_per_sector, free_clusters, total_clusters);
    bool handled = false;
    bool is_root = false;
    DWORD error = 0;
    std::filesystem::path resolved;
    if (!resolve_console_readonly_dir(root_path, ERROR_FILE_NOT_FOUND, handled, resolved, is_root, error)) {
        if (!handled)
            return ::GetDiskFreeSpaceW(root_path, sectors_per_cluster, bytes_per_sector, free_clusters, total_clusters);
        return fail_bool(error);
    }
    return ::GetDiskFreeSpaceW(resolved.c_str(), sectors_per_cluster, bytes_per_sector, free_clusters, total_clusters);
}

BOOL WINAPI get_disk_free_space_a(LPCSTR root_path, LPDWORD sectors_per_cluster, LPDWORD bytes_per_sector,
                                  LPDWORD free_clusters, LPDWORD total_clusters) {
    if (!root_path)
        return ::GetDiskFreeSpaceW(nullptr, sectors_per_cluster, bytes_per_sector, free_clusters, total_clusters);
    const std::wstring wide = widen_ansi(root_path);
    return get_disk_free_space_w(wide.c_str(), sectors_per_cluster, bytes_per_sector, free_clusters, total_clusters);
}

UINT WINAPI get_drive_type_w(LPCWSTR root_path) {
    if (!root_path || !*root_path) return ::GetDriveTypeW(root_path);
    bool handled = false;
    bool is_root = false;
    DWORD error = 0;
    std::filesystem::path resolved;
    if (!resolve_console_readonly_dir(root_path, ERROR_PATH_NOT_FOUND, handled, resolved, is_root, error)) {
        return handled ? DRIVE_NO_ROOT_DIR : ::GetDriveTypeW(root_path);
    }
    if (!is_root) return DRIVE_NO_ROOT_DIR;
    std::vector<wchar_t> volume_root((std::max<size_t>)(MAX_PATH, resolved.native().size() + 16), L'\0');
    if (!::GetVolumePathNameW(resolved.c_str(), volume_root.data(), static_cast<DWORD>(volume_root.size())))
        return DRIVE_NO_ROOT_DIR;
    return ::GetDriveTypeW(volume_root.data());
}

UINT WINAPI get_drive_type_a(LPCSTR root_path) {
    if (!root_path) return ::GetDriveTypeW(nullptr);
    const std::wstring wide = widen_ansi(root_path);
    return get_drive_type_w(wide.c_str());
}

// The volume shims answer for a console spelling (GAME:, G:\, T:\) only. A host path inside --game-dir is the package
// to the file shims (game_dir_rest), but the title named the host volume, so the host answers for it.
bool console_spelling(const std::wstring& name) {
    size_t prefix = 0;
    return console_drive(name, prefix) != Drive::None;
}

// Resolve the backing volume without exposing its host root to the title.
bool volume_host_root(const std::wstring& name, std::wstring& result) {
    const Roots r = roots();
    size_t prefix = 0;
    const Drive drive = console_drive(name, prefix);
    const bool package_volume = drive == Drive::Game && prefix == 2;
    auto dir = package_volume ? r.game : r.overlay;
    const DWORD attributes = dir.empty() ? INVALID_FILE_ATTRIBUTES : ::GetFileAttributesW(dir.c_str());
    if (attributes == INVALID_FILE_ATTRIBUTES) {
        if (package_volume || r.game.empty() || ::GetFileAttributesW(r.game.c_str()) == INVALID_FILE_ATTRIBUTES)
            return fail_bool(ERROR_PATH_NOT_FOUND);
        if (dir.empty()) dir = r.game;
        while (!dir.empty() && ::GetFileAttributesW(dir.c_str()) == INVALID_FILE_ATTRIBUTES) {
            const auto parent = dir.parent_path();
            if (parent == dir) break;
            dir = parent;
        }
    }
    if (dir.empty() || ::GetFileAttributesW(dir.c_str()) == INVALID_FILE_ATTRIBUTES)
        return fail_bool(ERROR_PATH_NOT_FOUND);
    wchar_t root[MAX_PATH]{};
    if (!::GetVolumePathNameW(dir.c_str(), root, MAX_PATH)) return false;
    result = root;
    return true;
}

BOOL WINAPI get_volume_information_w(LPCWSTR name, LPWSTR label, DWORD label_size, LPDWORD serial,
                                     LPDWORD max_component, LPDWORD flags, LPWSTR filesystem, DWORD filesystem_size) {
    if (!name) return ::GetVolumeInformationW(name, label, label_size, serial, max_component, flags, filesystem, filesystem_size);
    std::vector<std::wstring> parts;
    const Parsed parsed = parse_console(name, parts, false);
    if (parsed == Parsed::Host || !console_spelling(name)) return ::GetVolumeInformationW(name, label, label_size, serial, max_component, flags, filesystem, filesystem_size);
    if (parsed == Parsed::Refused || !drive_root(name, parts)) return fail_bool(ERROR_INVALID_NAME);
    std::wstring root;
    if (!volume_host_root(name, root)) return FALSE;
    return ::GetVolumeInformationW(root.c_str(), label, label_size, serial, max_component, flags, filesystem, filesystem_size);
}

BOOL WINAPI get_volume_information_a(LPCSTR name, LPSTR label, DWORD label_size, LPDWORD serial,
                                     LPDWORD max_component, LPDWORD flags, LPSTR filesystem, DWORD filesystem_size) {
    if (!name) return ::GetVolumeInformationA(name, label, label_size, serial, max_component, flags, filesystem, filesystem_size);
    const std::wstring wide = widen_ansi(name);
    std::vector<std::wstring> parts;
    const Parsed parsed = parse_console(wide, parts, false);
    if (parsed == Parsed::Host || !console_spelling(wide)) return ::GetVolumeInformationA(name, label, label_size, serial, max_component, flags, filesystem, filesystem_size);
    if (parsed == Parsed::Refused || !drive_root(wide, parts)) return fail_bool(ERROR_INVALID_NAME);
    std::wstring root;
    if (!volume_host_root(wide, root)) return FALSE;
    // Keep the mount path Unicode; only the two returned strings use the caller's ANSI code page.
    wchar_t wide_label[MAX_PATH]{}, wide_filesystem[MAX_PATH]{};
    if (!::GetVolumeInformationW(root.c_str(), label && label_size ? wide_label : nullptr,
                                 label && label_size ? MAX_PATH : 0, serial, max_component, flags,
                                 filesystem && filesystem_size ? wide_filesystem : nullptr,
                                 filesystem && filesystem_size ? MAX_PATH : 0)) return FALSE;
    if (label && label_size && !::WideCharToMultiByte(CP_ACP, 0, wide_label, -1, label,
                                                       static_cast<int>(label_size > 0x7fffffffU ? 0x7fffffffU : label_size), nullptr, nullptr)) return FALSE;
    if (filesystem && filesystem_size && !::WideCharToMultiByte(CP_ACP, 0, wide_filesystem, -1, filesystem,
                                                                static_cast<int>(filesystem_size > 0x7fffffffU ? 0x7fffffffU : filesystem_size), nullptr, nullptr)) return FALSE;
    return TRUE;
}

BOOL WINAPI get_volume_path_name_w(LPCWSTR name, LPWSTR output, DWORD size) {
    if (!name) return ::GetVolumePathNameW(name, output, size);
    std::vector<std::wstring> parts;
    const Parsed parsed = parse_console(name, parts, false);
    if (parsed == Parsed::Host || !console_spelling(name)) return ::GetVolumePathNameW(name, output, size);
    if (parsed == Parsed::Refused) return fail_bool(ERROR_INVALID_NAME);
    size_t prefix = 0;
    console_drive(name, prefix);
    const std::wstring root = std::wstring(name, prefix) + L"\\";
    if (!output || size <= root.size()) return fail_bool(ERROR_FILENAME_EXCED_RANGE);
    std::wmemcpy(output, root.c_str(), root.size() + 1);
    return TRUE;
}

BOOL WINAPI get_volume_path_name_a(LPCSTR name, LPSTR output, DWORD size) {
    if (!name) return ::GetVolumePathNameA(name, output, size);
    const std::wstring wide = widen_ansi(name);
    std::vector<std::wstring> parts;
    const Parsed parsed = parse_console(wide, parts, false);
    if (parsed == Parsed::Host || !console_spelling(wide)) return ::GetVolumePathNameA(name, output, size);
    if (parsed == Parsed::Refused) return fail_bool(ERROR_INVALID_NAME);
    size_t prefix = 0;
    console_drive(wide, prefix);
    if (!output || size <= prefix + 1) return fail_bool(ERROR_FILENAME_EXCED_RANGE);
    std::memcpy(output, name, prefix);
    output[prefix] = '\\';
    output[prefix + 1] = '\0';
    return TRUE;
}

// Console temporary directories always live in the writable overlay, independent of the package tree.
static Target temp_directory(const std::wstring& path, bool& console) {
    size_t prefix_length = 0;
    const Drive drive = console_drive(path, prefix_length);
    console = drive != Drive::None;
    if (!console) return {};
    std::vector<std::wstring> parts;
    if (parse_console(path, parts) == Parsed::Refused) return fail(ERROR_INVALID_NAME);
    const auto overlay = roots().overlay;
    if (overlay.empty()) return fail(ERROR_PATH_NOT_FOUND);
    std::error_code ec;
    const auto scratch = overlay / kScratchDir;
    std::filesystem::create_directories(scratch, ec);
    if (ec) return fail(static_cast<DWORD>(ec.value()));
    Target target;
    if (drive == Drive::Scratch) parts.erase(parts.begin());
    target.path = join(scratch, parts);
    return target;
}

static std::wstring temp_environment() {
    for (const wchar_t* key : {L"TMP", L"TEMP"}) {
        const DWORD size = ::GetEnvironmentVariableW(key, nullptr, 0);
        if (!size) continue;
        std::vector<wchar_t> value(size);
        const DWORD length = ::GetEnvironmentVariableW(key, value.data(), size);
        if (length && length < size) return std::wstring(value.data(), length);
    }
    return {};
}

static void temp_name_error(const std::filesystem::path& directory) {
    const DWORD error = ::GetLastError();
    if (::GetFileAttributesW(directory.c_str()) == INVALID_FILE_ATTRIBUTES) {
        const DWORD attributes_error = ::GetLastError();
        if (attributes_error == ERROR_FILE_NOT_FOUND || attributes_error == ERROR_PATH_NOT_FOUND) {
            ::SetLastError(ERROR_PATH_NOT_FOUND);
            return;
        }
    }
    ::SetLastError(error);
}

DWORD WINAPI get_temp_path_w(DWORD size, LPWSTR buffer) {
    bool console = false;
    const Target target = temp_directory(temp_environment(), console);
    if (!console) return ::GetTempPathW(size, buffer);
    if (target.error) return fail_bool(target.error);
    std::wstring path = target.path.wstring();
    if (path.empty() || !is_separator(path.back())) path += static_cast<wchar_t>(92);
    const DWORD required = static_cast<DWORD>(path.size() + 1);
    if (size < required) return required;
    std::wmemcpy(buffer, path.c_str(), required);
    return required - 1;
}

DWORD WINAPI get_temp_path_a(DWORD size, LPSTR buffer) {
    bool console = false;
    const Target target = temp_directory(temp_environment(), console);
    if (!console) return ::GetTempPathA(size, buffer);
    if (target.error) return fail_bool(target.error);
    std::wstring path = target.path.wstring();
    if (path.empty() || !is_separator(path.back())) path += static_cast<wchar_t>(92);
    const bool utf8 = ::GetACP() == CP_UTF8;
    const DWORD flags = utf8 ? 0 : WC_NO_BEST_FIT_CHARS;
    BOOL used_default = FALSE;
    BOOL* const used_default_ptr = utf8 ? nullptr : &used_default;
    const int required = ::WideCharToMultiByte(CP_ACP, flags, path.c_str(), -1, nullptr, 0, nullptr, used_default_ptr);
    if (!required) return 0;
    if (used_default) {
        ::SetLastError(ERROR_NO_UNICODE_TRANSLATION);
        return 0;
    }
    if (size < static_cast<DWORD>(required))
        return size && size == static_cast<DWORD>(required - 1) ? size : static_cast<DWORD>(required);
    if (!::WideCharToMultiByte(CP_ACP, flags, path.c_str(), -1, buffer, required, nullptr, used_default_ptr)) return 0;
    if (used_default) {
        ::SetLastError(ERROR_NO_UNICODE_TRANSLATION);
        return 0;
    }
    return static_cast<DWORD>(required - 1);
}

UINT WINAPI get_temp_file_name_w(LPCWSTR path, LPCWSTR prefix, UINT unique, LPWSTR name) {
    if (!path) return ::GetTempFileNameW(path, prefix, unique, name);
    bool console = false;
    const Target target = temp_directory(path, console);
    if (!console) return ::GetTempFileNameW(path, prefix, unique, name);
    if (target.error) return fail_bool(target.error);
    const UINT result = ::GetTempFileNameW(target.path.c_str(), prefix, unique, name);
    if (!result) temp_name_error(target.path);
    return result;
}

UINT WINAPI get_temp_file_name_a(LPCSTR path, LPCSTR prefix, UINT unique, LPSTR name) {
    if (!path) return ::GetTempFileNameA(path, prefix, unique, name);
    bool console = false;
    const Target target = temp_directory(widen_ansi(path), console);
    if (!console) return ::GetTempFileNameA(path, prefix, unique, name);
    if (target.error) return fail_bool(target.error);
    const std::wstring wide_prefix = widen_ansi(prefix);
    wchar_t wide_name[MAX_PATH]{};
    const UINT result = ::GetTempFileNameW(target.path.c_str(), prefix ? wide_prefix.c_str() : nullptr, unique, wide_name);
    if (!result) {
        temp_name_error(target.path);
        return 0;
    }
    char converted[MAX_PATH]{};
    const int length = ::WideCharToMultiByte(CP_ACP, 0, wide_name, -1, converted, MAX_PATH, nullptr, nullptr);
    if (!length || widen_ansi(converted) != wide_name) {
        if (!unique) ::DeleteFileW(wide_name);
        ::SetLastError(ERROR_NO_UNICODE_TRANSLATION);
        return 0;
    }
    std::memcpy(name, converted, static_cast<size_t>(length));
    return result;
}

}  // namespace rt::fs

namespace {
using namespace rt::fs;

// The title's reads into its own memory. The kernel writes the destination, which raises no fault in it: xmem's
// physical pages, committed on the title's first touch, would fail the read with ERROR_NOACCESS while not committed
// yet. So the destination is prepared first (rt::kernel_write::before; xmem commits its pages there).
BOOL WINAPI read_file(HANDLE file, LPVOID buffer, DWORD bytes, LPDWORD read, LPOVERLAPPED overlapped) {
    rt::kernel_write::before(buffer, bytes);
    return ::ReadFile(file, buffer, bytes, read, overlapped);
}

BOOL WINAPI read_file_ex(HANDLE file, LPVOID buffer, DWORD bytes, LPOVERLAPPED overlapped,
                         LPOVERLAPPED_COMPLETION_ROUTINE routine) {
    rt::kernel_write::before(buffer, bytes);
    return ::ReadFileEx(file, buffer, bytes, overlapped, routine);
}

// Each segment is one system page.
BOOL WINAPI read_file_scatter(HANDLE file, FILE_SEGMENT_ELEMENT segments[], DWORD bytes, LPDWORD reserved,
                              LPOVERLAPPED overlapped) {
    if (segments) {
        SYSTEM_INFO info{};
        GetSystemInfo(&info);
        for (DWORD i = 0, n = (bytes + info.dwPageSize - 1) / info.dwPageSize; i < n; ++i)
            rt::kernel_write::before(segments[i].Buffer, info.dwPageSize);
    }
    return ::ReadFileScatter(file, segments, bytes, reserved, overlapped);
}

// CreateFile, the FindFirstFile family, the attribute queries and DeleteFile arrive through kernel32/kernelbase
// imports and through the api sets a GDK title is linked against (file-l1-1-0 for the find and attribute calls,
// file-l1-2-0 and file-l2-1-0 for CreateFile; ReadFile and its variants are in file-l1-1-0).
#define FILE_API(dll)                                                                                    \
    {dll, "GetVolumeInformationA", reinterpret_cast<void*>(&get_volume_information_a)},                   \
        {dll, "GetVolumeInformationW", reinterpret_cast<void*>(&get_volume_information_w)},             \
        {dll, "GetVolumePathNameA", reinterpret_cast<void*>(&get_volume_path_name_a)},                   \
        {dll, "GetVolumePathNameW", reinterpret_cast<void*>(&get_volume_path_name_w)},                   \
    {dll, "GetTempPathA", reinterpret_cast<void*>(&get_temp_path_a)},                                  \
    {dll, "GetTempPathW", reinterpret_cast<void*>(&get_temp_path_w)},                                  \
    {dll, "GetTempFileNameA", reinterpret_cast<void*>(&get_temp_file_name_a)},                          \
    {dll, "GetTempFileNameW", reinterpret_cast<void*>(&get_temp_file_name_w)},                          \
    {dll, "CreateFileA", reinterpret_cast<void*>(&create_file_a)},                                       \
        {dll, "CreateFileW", reinterpret_cast<void*>(&create_file_w)},                                   \
        {dll, "CreateFile2", reinterpret_cast<void*>(&create_file2)},                                    \
        {dll, "FindFirstFileA", reinterpret_cast<void*>(&find_first_file_a)},                            \
        {dll, "FindFirstFileW", reinterpret_cast<void*>(&find_first_file_w)},                            \
        {dll, "FindFirstFileExA", reinterpret_cast<void*>(&find_first_file_ex_a)},                       \
        {dll, "FindFirstFileExW", reinterpret_cast<void*>(&find_first_file_ex_w)},                       \
        {dll, "FindNextFileA", reinterpret_cast<void*>(&find_next_file_a)},                              \
        {dll, "FindNextFileW", reinterpret_cast<void*>(&find_next_file_w)},                              \
        {dll, "FindClose", reinterpret_cast<void*>(&find_close)},                                        \
        {dll, "GetFileAttributesA", reinterpret_cast<void*>(&get_file_attributes_a)},                    \
        {dll, "GetFullPathNameA", reinterpret_cast<void*>(&get_full_path_name_a)},                       \
        {dll, "GetFullPathNameW", reinterpret_cast<void*>(&get_full_path_name_w)},                       \
        {dll, "GetFileAttributesW", reinterpret_cast<void*>(&get_file_attributes_w)},                    \
        {dll, "GetFileAttributesExA", reinterpret_cast<void*>(&get_file_attributes_ex_a)},               \
        {dll, "GetFileAttributesExW", reinterpret_cast<void*>(&get_file_attributes_ex_w)},               \
        {dll, "DeleteFileA", reinterpret_cast<void*>(&delete_file_a)},                                   \
        {dll, "DeleteFileW", reinterpret_cast<void*>(&delete_file_w)},                                   \
        {dll, "CreateDirectoryA", reinterpret_cast<void*>(&create_directory_a)},                         \
        {dll, "CreateDirectoryW", reinterpret_cast<void*>(&create_directory_w)},                         \
        {dll, "CreateDirectoryExA", reinterpret_cast<void*>(&create_directory_ex_a)},                    \
        {dll, "CreateDirectoryExW", reinterpret_cast<void*>(&create_directory_ex_w)},                    \
        {dll, "RemoveDirectoryA", reinterpret_cast<void*>(&remove_directory_a)},                         \
        {dll, "RemoveDirectoryW", reinterpret_cast<void*>(&remove_directory_w)},                         \
        {dll, "MoveFileA", reinterpret_cast<void*>(&move_file_a)},                                       \
        {dll, "MoveFileW", reinterpret_cast<void*>(&move_file_w)},                                       \
        {dll, "MoveFileExA", reinterpret_cast<void*>(&move_file_ex_a)},                                  \
        {dll, "MoveFileExW", reinterpret_cast<void*>(&move_file_ex_w)},                                  \
        {dll, "CopyFileA", reinterpret_cast<void*>(&copy_file_a)},                                       \
        {dll, "CopyFileW", reinterpret_cast<void*>(&copy_file_w)},                                       \
        {dll, "CopyFileExA", reinterpret_cast<void*>(&copy_file_ex_a)},                                  \
        {dll, "CopyFileExW", reinterpret_cast<void*>(&copy_file_ex_w)},                                  \
        {dll, "CopyFile2", reinterpret_cast<void*>(&copy_file2)},                                        \
        {dll, "SetFileAttributesA", reinterpret_cast<void*>(&set_file_attributes_a)},                    \
        {dll, "SetFileAttributesW", reinterpret_cast<void*>(&set_file_attributes_w)},                    \
        {dll, "ReadFile", reinterpret_cast<void*>(&read_file)},                                          \
        {dll, "ReadFileEx", reinterpret_cast<void*>(&read_file_ex)},                                     \
        {dll, "ReadFileScatter", reinterpret_cast<void*>(&read_file_scatter)},                           \
        {dll, "GetDiskFreeSpaceExA", reinterpret_cast<void*>(&get_disk_free_space_ex_a)},                \
        {dll, "GetDiskFreeSpaceExW", reinterpret_cast<void*>(&get_disk_free_space_ex_w)},                \
        {dll, "GetDiskFreeSpaceA", reinterpret_cast<void*>(&get_disk_free_space_a)},                     \
        {dll, "GetDiskFreeSpaceW", reinterpret_cast<void*>(&get_disk_free_space_w)},                     \
        {dll, "GetDriveTypeA", reinterpret_cast<void*>(&get_drive_type_a)},                              \
        {dll, "GetDriveTypeW", reinterpret_cast<void*>(&get_drive_type_w)},                              \
        {dll, "SetFileInformationByHandle", reinterpret_cast<void*>(&set_file_information_by_handle)}

// The Universal CRT's opens, under the api set a package DLL links (api-ms-win-crt-stdio-l1-1-0) and under ucrtbase
// itself, which that set forwards to.
#define CRT_OPEN_API(dll)                                                                                \
    {dll, "fopen", reinterpret_cast<void*>(&crt_fopen)},                                                 \
        {dll, "_wfopen", reinterpret_cast<void*>(&crt_wfopen)},                                          \
        {dll, "_fsopen", reinterpret_cast<void*>(&crt_fsopen)},                                          \
        {dll, "_wfsopen", reinterpret_cast<void*>(&crt_wfsopen)},                                        \
        {dll, "fopen_s", reinterpret_cast<void*>(&crt_fopen_s)},                                         \
        {dll, "_wfopen_s", reinterpret_cast<void*>(&crt_wfopen_s)},                                      \
        {dll, "freopen", reinterpret_cast<void*>(&crt_freopen)},                                         \
        {dll, "_wfreopen", reinterpret_cast<void*>(&crt_wfreopen)},                                      \
        {dll, "freopen_s", reinterpret_cast<void*>(&crt_freopen_s)},                                     \
        {dll, "_wfreopen_s", reinterpret_cast<void*>(&crt_wfreopen_s)},                                  \
        {dll, "_open", reinterpret_cast<void*>(&crt_open)},                                              \
        {dll, "_wopen", reinterpret_cast<void*>(&crt_wopen)},                                            \
        {dll, "_sopen", reinterpret_cast<void*>(&crt_sopen)},                                            \
        {dll, "_wsopen", reinterpret_cast<void*>(&crt_wsopen)},                                          \
        {dll, "_sopen_s", reinterpret_cast<void*>(&crt_sopen_s)},                                        \
        {dll, "_wsopen_s", reinterpret_cast<void*>(&crt_wsopen_s)},                                      \
        {dll, "_sopen_dispatch", reinterpret_cast<void*>(&crt_sopen_dispatch)},                          \
        {dll, "_wsopen_dispatch", reinterpret_cast<void*>(&crt_wsopen_dispatch)}

// The CRT's other path calls (RT-359), all exported by api-ms-win-crt-filesystem-l1-1-0 and ucrtbase.
#define CRT_PATH_API(dll)                                                                                \
    {dll, "remove", reinterpret_cast<void*>(&crt_remove)},                                               \
        {dll, "_wremove", reinterpret_cast<void*>(&crt_wremove)},                                        \
        {dll, "_unlink", reinterpret_cast<void*>(&crt_remove)},                                          \
        {dll, "_wunlink", reinterpret_cast<void*>(&crt_wremove)},                                        \
        {dll, "rename", reinterpret_cast<void*>(&crt_rename)},                                           \
        {dll, "_wrename", reinterpret_cast<void*>(&crt_wrename)},                                        \
        {dll, "_mkdir", reinterpret_cast<void*>(&crt_mkdir)},                                            \
        {dll, "_wmkdir", reinterpret_cast<void*>(&crt_wmkdir)},                                          \
        {dll, "_rmdir", reinterpret_cast<void*>(&crt_rmdir)},                                            \
        {dll, "_wrmdir", reinterpret_cast<void*>(&crt_wrmdir)},                                          \
        {dll, "_access", reinterpret_cast<void*>(&crt_access)},                                          \
        {dll, "_waccess", reinterpret_cast<void*>(&crt_waccess)},                                        \
        {dll, "_access_s", reinterpret_cast<void*>(&crt_access_s)},                                      \
        {dll, "_waccess_s", reinterpret_cast<void*>(&crt_waccess_s)},                                    \
        {dll, "_stat32", reinterpret_cast<void*>(&crt_stat32)},                                          \
        {dll, "_stat32i64", reinterpret_cast<void*>(&crt_stat32i64)},                                    \
        {dll, "_stat64", reinterpret_cast<void*>(&crt_stat64)},                                          \
        {dll, "_stat64i32", reinterpret_cast<void*>(&crt_stat64i32)},                                    \
        {dll, "_wstat32", reinterpret_cast<void*>(&crt_wstat32)},                                        \
        {dll, "_wstat32i64", reinterpret_cast<void*>(&crt_wstat32i64)},                                  \
        {dll, "_wstat64", reinterpret_cast<void*>(&crt_wstat64)},                                        \
        {dll, "_wstat64i32", reinterpret_cast<void*>(&crt_wstat64i32)}
}  // namespace

SHIM_TABLE(file_system,
    FILE_API("kernel32.dll"),
    FILE_API("kernelbase.dll"),
    FILE_API("api-ms-win-core-file-l1-1-0.dll"),
    FILE_API("api-ms-win-core-file-l1-2-0.dll"),
    FILE_API("api-ms-win-core-file-l2-1-0.dll"),
    CRT_OPEN_API("api-ms-win-crt-stdio-l1-1-0.dll"),
    CRT_OPEN_API("ucrtbase.dll"),
    CRT_PATH_API("api-ms-win-crt-filesystem-l1-1-0.dll"),
    CRT_PATH_API("ucrtbase.dll"),
);
