// The shim registry reports a name that two SHIM_TABLEs register with different functions (shims.h,
// duplicate_registrations). Two files serving one import is a merge mistake: across translation units the static tables
// initialise in link order, so which function the title gets is not decided by the code. The later table still wins.
// No game files, no GPU.
#include <cstdio>
#include <string>
#include <vector>

#include "../shims/shims.h"

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

int first_factory() { return 1; }
int second_factory() { return 2; }
int same_function() { return 3; }
int other_name() { return 4; }

// Within one file, tables initialise in the order they are written: `later` registers after `earlier`.
SHIM_TABLE(earlier,
    {"Test_Dup.dll", "DStorageGetFactory", reinterpret_cast<void*>(&first_factory)},
    {"test_dup.dll", "SameFunction", reinterpret_cast<void*>(&same_function)},
    {"test_dup.dll", "OnlyHere", reinterpret_cast<void*>(&other_name)});
SHIM_TABLE(later,
    {"test_dup.dll", "DStorageGetFactory", reinterpret_cast<void*>(&second_factory)},
    {"test_dup.dll", "SameFunction", reinterpret_cast<void*>(&same_function)});

}  // namespace

int main() {
    const std::vector<std::string> duplicates = rt::shims::duplicate_registrations();
    check(duplicates.size() == 1, "exactly one name is reported");
    check(!duplicates.empty() && duplicates.front() == "test_dup.dll!DStorageGetFactory",
          "the name two tables registered with different functions is reported, its DLL lower-cased");
    check(rt::shims::find("test_dup.dll", "DStorageGetFactory") == reinterpret_cast<void*>(&second_factory),
          "the later table still serves the name, as before");
    check(rt::shims::find("test_dup.dll", "SameFunction") == reinterpret_cast<void*>(&same_function),
          "a name registered twice with the same function is served and not reported");
    check(rt::shims::find("test_dup.dll", "OnlyHere") == reinterpret_cast<void*>(&other_name),
          "a name registered once is served");
    std::printf("RESULT: %s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
