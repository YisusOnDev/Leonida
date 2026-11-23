// A stand-in title for `gta6rt --run` that ends the way a crashing title does, with its own handlers in place.
//   tiny_crash.exe filter      installs a top-level exception filter, checks that installing a second one returns
//                              the first, then reads through a null pointer: the filter must run (exit code 10)
//   tiny_crash.exe terminate   installs a terminate handler and throws a C++ exception nobody catches: the handler
//                              must run (exit code 12). It gets there through the filter the MSVC start-up code
//                              installs with SetUnhandledExceptionFilter, which calls std::terminate for a C++
//                              exception.
// Natively both modes end with those codes. Any other exit code (0xC0000005, 0xE06D7363) means the handler did not
// run. Ordinary test code; no game data.
#include <windows.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>

namespace {

bool g_round_trip_ok = false;

LONG WINAPI first_filter(EXCEPTION_POINTERS*) { return EXCEPTION_CONTINUE_SEARCH; }

LONG WINAPI crash_filter(EXCEPTION_POINTERS* info) {
    const bool fault = info && info->ExceptionRecord &&
                       info->ExceptionRecord->ExceptionCode == static_cast<DWORD>(EXCEPTION_ACCESS_VIOLATION);
    std::printf("tiny_crash: the top-level filter ran (%s)\n", fault ? "access violation" : "another exception");
    ExitProcess(fault && g_round_trip_ok ? 10 : 11);
}

void on_terminate() {
    std::printf("tiny_crash: the terminate handler ran\n");
    ExitProcess(12);
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc == 2 && std::strcmp(argv[1], "filter") == 0) {
        SetUnhandledExceptionFilter(&first_filter);
        const LPTOP_LEVEL_EXCEPTION_FILTER previous = SetUnhandledExceptionFilter(&crash_filter);
        g_round_trip_ok = previous == &first_filter;
        std::printf("tiny_crash: %s installing a filter returns the one installed before\n",
                    g_round_trip_ok ? "PASS" : "FAIL");
        volatile int* null = nullptr;
        return *null;  // faults
    }
    if (argc == 2 && std::strcmp(argv[1], "terminate") == 0) {
        std::set_terminate(&on_terminate);
        throw 7;  // nobody catches it
    }
    std::printf("usage: tiny_crash filter|terminate\n");
    return 2;
}
