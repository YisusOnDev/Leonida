// d3d12_xs.dll!D3D12ConfigureVirtualMemoryX through the shim registry, in the shape the retail executable calls it
// (1.0.2.1, RVA 0x1e1cfe0; see the comment above configure_virtual_memory in shims/d3d12_xs.cpp): one pointer to a
// zeroed 32-byte block with the 32-bit field at +4 set to 3 or 4, junk left in RDX/R8/R9, and the result tested as an
// HRESULT by its sign bit. Also checks that the block comes back unchanged and that a --trace run records the first
// call once. No game files and no GPU are used.
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

#include "shims.h"
#include "startup_trace.h"

namespace {

int g_failures = 0;

void check(bool condition, const char* what) {
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

using ConfigureFn = HRESULT(WINAPI*)(const void*);
// The measured call loads only RCX; the other argument registers hold whatever an earlier call left.
using ConfigureWithJunkFn = HRESULT(WINAPI*)(const void*, uint64_t, uint64_t, uint64_t);

struct Block {
    uint32_t words[8];
};
static_assert(sizeof(Block) == 32, "the caller zeroes a 32-byte block");

Block measured_block(uint32_t field4) {
    Block b{};
    b.words[1] = field4;
    return b;
}

// The caller's test: `test eax, eax` then a branch on the sign flag. A one-byte bool or a nonzero-means-failure
// BOOL reading would not match an HRESULT, so check the 32-bit sign explicitly.
bool caller_accepts(HRESULT hr) { return static_cast<int32_t>(hr) >= 0; }

void test_measured_calls(ConfigureFn configure) {
    for (uint32_t field4 : {4u, 3u}) {
        Block block = measured_block(field4);
        const Block before = block;
        const HRESULT hr = configure(&block);
        char what[160];
        std::snprintf(what, sizeof(what), "a zeroed 32-byte block with +4 = %u returns S_OK (0x%08lx)", field4,
                      static_cast<unsigned long>(hr));
        check(hr == S_OK && caller_accepts(hr), what);
        check(std::memcmp(&block, &before, sizeof(block)) == 0, "the block comes back unchanged");
    }

    // The 32-bit field is read from a block whose other words could be anything; nothing past 32 bytes is touched.
    unsigned char guarded[48];
    std::memset(guarded, 0xcd, sizeof(guarded));
    Block block = measured_block(4);
    std::memcpy(guarded, &block, sizeof(block));
    check(configure(guarded) == S_OK, "a block followed by other stack data returns S_OK");
    bool tail_intact = true;
    for (size_t i = sizeof(Block); i < sizeof(guarded); ++i) tail_intact = tail_intact && guarded[i] == 0xcd;
    check(tail_intact, "nothing after the 32-byte block is written");
}

void test_junk_registers(void* raw) {
    const auto configure = reinterpret_cast<ConfigureWithJunkFn>(raw);
    Block block = measured_block(4);
    const Block before = block;
    // The values a boot under LEONIDA_UNIMPLEMENTED=stop showed in RDX/R8/R9 were a heap pointer, a stack pointer
    // and zero. Use addresses that would fault if they were dereferenced.
    const HRESULT hr = configure(&block, 0x00000270deadbeefull, 0x00000072deadbeefull, 0);
    check(hr == S_OK, "junk in RDX, R8 and R9 is ignored (one-argument function)");
    check(std::memcmp(&block, &before, sizeof(block)) == 0, "the block is unchanged with junk registers");
}

void test_null(ConfigureFn configure) {
    const HRESULT hr = configure(nullptr);
    check(hr == E_INVALIDARG && !caller_accepts(hr), "a null block returns E_INVALIDARG (the title never passes one)");
}

void test_trace(ConfigureFn configure) {
    wchar_t temp[MAX_PATH];
    GetTempPathW(MAX_PATH, temp);
    const std::wstring path = std::wstring(temp) + L"d3d12_xs_virtual_memory_trace_test_" +
                              std::to_wstring(GetCurrentProcessId()) + L".jsonl";
    DeleteFileW(path.c_str());

    // The calls above ran with the trace closed: they must not have used up the one traced call.
    check(rt::startup_trace::open(path.c_str()), "open the startup trace file");
    Block first = measured_block(4);
    first.words[7] = 0x1234abcd;
    Block second = measured_block(3);
    check(configure(&first) == S_OK, "traced call returns S_OK");
    check(configure(&second) == S_OK, "second traced call returns S_OK");
    rt::startup_trace::close();

    std::ifstream in(path);
    std::stringstream text;
    text << in.rdbuf();
    in.close();
    const std::string log = text.str();
    auto count = [&](const std::string& needle) {
        int n = 0;
        for (size_t at = log.find(needle); at != std::string::npos; at = log.find(needle, at + 1)) ++n;
        return n;
    };
    check(count("\"kind\":\"d3d12_configure_virtual_memory\"") == 1, "one event: only the first traced call");
    check(count("\"api\":\"D3D12ConfigureVirtualMemoryX\",\"words\":"
                "\"00000000 00000004 00000000 00000000 00000000 00000000 00000000 1234abcd\",\"field4\":4") == 1,
          "the event records the first call's eight words and its +4 field");
    check(count("\"field4\":3") == 0, "the second call is not recorded");
    check(count("\"caller\":\"") == 1, "the event names its caller");
    DeleteFileW(path.c_str());
}

}  // namespace

int main() {
    std::printf("d3d12_xs_virtual_memory_test: D3D12ConfigureVirtualMemoryX through the shim registry\n");
    void* raw = rt::shims::find("d3d12_xs.dll", "D3D12ConfigureVirtualMemoryX");
    check(raw != nullptr, "shim registry exposes d3d12_xs.dll!D3D12ConfigureVirtualMemoryX");
    if (raw) {
        const auto configure = reinterpret_cast<ConfigureFn>(raw);
        test_measured_calls(configure);
        test_junk_registers(raw);
        test_null(configure);
        test_trace(configure);
    }
    std::printf("d3d12_xs_virtual_memory_test: %s (%d failure%s)\n", g_failures ? "FAILED" : "OK", g_failures,
                g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}
