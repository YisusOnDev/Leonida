// DisableThreadLibraryCalls from a mapped image's DllMain (shims/library_loader.cpp).
//
// The real function only accepts a module the Windows loader knows, so for a manually mapped image it returned FALSE
// and a DllMain that returns its result made Loader::initialise() throw "DllMain returned FALSE". The shim answers
// TRUE for a mapped module and for a module the shims own (a pseudo-handle), and leaves every other handle to the
// real function.
//
// Maps dll_thread_calls.dll (test/fixtures/dll_thread_calls.cpp). xmem.cpp is linked only so the shims own a module
// (xmem.dll) to hand out a pseudo-handle for. No game files.
#include <windows.h>

#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "pe_loader.h"
#include "shims.h"

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    std::fflush(stdout);
    if (!ok) ++g_failures;
}

int fake_host_named_shim() { return 11; }
int fake_host_ordinal_shim() { return 22; }

struct BundledProbe {
    std::vector<std::string> seen;

    static void* find_loaded(void* ctx, const std::string& dll_lower) {
        static_cast<BundledProbe*>(ctx)->seen.push_back(dll_lower);
        return nullptr;
    }
    static void* load_bundled(void* ctx, const std::string& dll_lower, uint32_t, bool& handled) {
        handled = false;
        static_cast<BundledProbe*>(ctx)->seen.push_back(dll_lower);
        return nullptr;
    }
    static void* get_proc(void*, void*, const std::string&, bool& owns_handle) {
        owns_handle = false;
        return nullptr;
    }
    static bool owns_handle(void*, void*) { return false; }
};

struct BundledGuard {
    BundledProbe* probe;
    rt::shims::BundledModuleOps ops{
        &BundledProbe::find_loaded,
        &BundledProbe::load_bundled,
        &BundledProbe::get_proc,
        &BundledProbe::owns_handle,
    };
    explicit BundledGuard(BundledProbe* p) : probe(p) { rt::shims::add_bundled_loader(probe, &ops); }
    ~BundledGuard() { rt::shims::remove_bundled_loader(probe); }
    BundledGuard(const BundledGuard&) = delete;
    BundledGuard& operator=(const BundledGuard&) = delete;
};

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: dll_thread_calls_test <dll_thread_calls.dll>\n");
        return 2;
    }
    const std::filesystem::path dll = argv[1];
    try {
        rt::Loader loader(dll.parent_path());
        rt::Module& m = loader.load(dll);

        bool shimmed = false;
        for (const auto& r : m.imports)
            if (r.name == "DisableThreadLibraryCalls") shimmed = r.binding == rt::Binding::Shim;
        check(shimmed, "the DisableThreadLibraryCalls import is bound to the shim");

        bool initialised = true;
        try {
            loader.initialise();
        } catch (const std::exception& e) {
            initialised = false;
            std::printf("     initialise() threw: %s\n", e.what());
        }
        check(initialised, "initialise() succeeds when DllMain returns DisableThreadLibraryCalls(hinstance)");

        const auto bogus =
            reinterpret_cast<BOOL(__cdecl*)()>(m.exports.at("dll_thread_calls_bogus_handle"));
        SetLastError(0);
        check(bogus() == FALSE, "a handle that names no module still gets the real answer (FALSE)");

        // A shim pseudo-handle is no module the Windows loader knows, so the real function would fail on it.
        using GetModuleHandleAFn = HMODULE(WINAPI*)(LPCSTR);
        using DisableFn = BOOL(WINAPI*)(HMODULE);
        const auto get_module =
            reinterpret_cast<GetModuleHandleAFn>(rt::shims::find("kernel32.dll", "GetModuleHandleA"));
        const auto disable =
            reinterpret_cast<DisableFn>(rt::shims::find("kernel32.dll", "DisableThreadLibraryCalls"));
        check(get_module != nullptr && disable != nullptr,
              "the registry serves GetModuleHandleA and DisableThreadLibraryCalls");
        const HMODULE owned = get_module ? get_module("xmem.dll") : nullptr;
        check(owned != nullptr, "the shims hand out a pseudo-handle for the module they own (xmem.dll)");
        check(disable != nullptr && owned != nullptr && disable(owned) == TRUE,
              "DisableThreadLibraryCalls on a shim pseudo-handle answers TRUE");

        {
            using GetModAFn = HMODULE(WINAPI*)(LPCSTR);
            using GetModWFn = HMODULE(WINAPI*)(LPCWSTR);
            using LoadAFn = HMODULE(WINAPI*)(LPCSTR);
            using LoadWFn = HMODULE(WINAPI*)(LPCWSTR);
            using LoadExAFn = HMODULE(WINAPI*)(LPCSTR, HANDLE, DWORD);
            using LoadExWFn = HMODULE(WINAPI*)(LPCWSTR, HANDLE, DWORD);
            const auto get_mod_a =
                reinterpret_cast<GetModAFn>(rt::shims::find("kernel32.dll", "GetModuleHandleA"));
            const auto get_mod_w =
                reinterpret_cast<GetModWFn>(rt::shims::find("kernel32.dll", "GetModuleHandleW"));
            const auto load_a = reinterpret_cast<LoadAFn>(rt::shims::find("kernel32.dll", "LoadLibraryA"));
            const auto load_w = reinterpret_cast<LoadWFn>(rt::shims::find("kernel32.dll", "LoadLibraryW"));
            const auto load_ex_a =
                reinterpret_cast<LoadExAFn>(rt::shims::find("kernel32.dll", "LoadLibraryExA"));
            const auto load_ex_w =
                reinterpret_cast<LoadExWFn>(rt::shims::find("kernel32.dll", "LoadLibraryExW"));
            check(get_mod_a && get_mod_w && load_a && load_w && load_ex_a && load_ex_w,
                  "the registry serves all LoadLibrary and GetModuleHandle variants");
            if (get_mod_a && get_mod_w && load_a && load_w && load_ex_a && load_ex_w) {
                BundledProbe probe;
                BundledGuard guard(&probe);
                get_mod_a("");
                get_mod_w(L"");
                get_mod_a("C:\\some\\dir\\");
                get_mod_w(L"C:/some/dir/");
                get_mod_a("C:");
                load_a("");
                load_w(L"");
                load_a("C:\\some\\dir\\");
                load_w(L"C:/some/dir/");
                load_ex_a("C:\\some\\dir\\", nullptr, 0);
                load_ex_w(L"C:/some/dir/", nullptr, 0);
                check(probe.seen.empty(),
                      "empty and trailing-slash paths do not synthesize '.dll' for bundled lookup");
                check(get_mod_a("C:xmem.dll") == owned && get_mod_w(L"C:XMEM") == owned,
                      "drive-relative paths resolve owned modules");
            }
        }

        {
            const rt::shims::Entry host_entries[] = {
                {"dll_thread_calls.dll", "test_host_cache_eviction", reinterpret_cast<void*>(&fake_host_named_shim)},
                {"dll_thread_calls.dll", "#1", reinterpret_cast<void*>(&fake_host_ordinal_shim)},
            };
            const rt::shims::Table host_table(host_entries, 2);
            using GetProcFn = FARPROC(WINAPI*)(HMODULE, LPCSTR);
            using FreeLibFn = BOOL(WINAPI*)(HMODULE);
            const auto get_proc =
                reinterpret_cast<GetProcFn>(rt::shims::find("kernel32.dll", "GetProcAddress"));
            const auto free_lib =
                reinterpret_cast<FreeLibFn>(rt::shims::find("kernel32.dll", "FreeLibrary"));
            check(get_proc != nullptr && free_lib != nullptr,
                  "the registry serves GetProcAddress and FreeLibrary");

            HMODULE host_mod = ::LoadLibraryW(dll.c_str());
            check(host_mod != nullptr, "host LoadLibraryW loads dll_thread_calls.dll");
            if (host_mod && get_proc && free_lib) {
                check(get_proc(host_mod, "test_host_cache_eviction") ==
                          reinterpret_cast<FARPROC>(&fake_host_named_shim),
                      "GetProcAddress resolves named shim on live host module");
                check(get_proc(host_mod, MAKEINTRESOURCEA(1)) ==
                          reinterpret_cast<FARPROC>(&fake_host_ordinal_shim),
                      "GetProcAddress resolves ordinal shim on live host module");
                check(free_lib(host_mod) == TRUE, "FreeLibrary unloads host module");
                check(::GetModuleHandleW(dll.c_str()) == nullptr,
                      "host module is actually unloaded after FreeLibrary");
                check(get_proc(host_mod, "test_host_cache_eviction") == nullptr,
                      "GetProcAddress does not return stale cached shim after FreeLibrary");
                check(get_proc(host_mod, MAKEINTRESOURCEA(1)) == nullptr,
                      "GetProcAddress does not return stale ordinal shim after FreeLibrary");
            }
        }
    } catch (const std::exception& e) {
        std::printf("FAIL unexpected exception: %s\n", e.what());
        ++g_failures;
    }
    std::printf("dll_thread_calls_test: %s (%d failures)\n", g_failures ? "FAILED" : "OK", g_failures);
    return g_failures ? 1 : 0;
}
