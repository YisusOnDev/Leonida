// A recorded call the layer refuses marks its list invalid, so Close returns E_INVALIDARG; each refusing call site is
// logged once, with its method and line, so a failing Close names its cause. Here ClearRenderTargetView with a null
// view and RSSetViewports with one viewport too many are recorded three times each on a direct list: every Close
// returns E_INVALIDARG, and stderr gets exactly one line per method. RSSetViewports records nothing in the D3D12 trace
// before its check, so the method comes from the call site, not from the trace. A valid list closes with S_OK and logs
// nothing. Needs a Vulkan 1.3 device (SKIP otherwise). No game data.
#include "test_temp_dir.h"
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"

#include <io.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

#define REQUIRE(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using namespace d3d12test;

namespace {

int count_of(const std::string& text, const char* needle) {
    int count = 0;
    for (std::size_t at = text.find(needle); at != std::string::npos; at = text.find(needle, at + 1)) ++count;
    return count;
}

// Sends stderr to a file while it lives (the trace stays off: the line under test is an ordinary [rt] line).
struct StderrCapture {
    std::filesystem::path path;
    int saved = -1;
    explicit StderrCapture(const std::filesystem::path& file) : path(file) {
        std::fflush(stderr);
        std::FILE* out = std::fopen(path.string().c_str(), "wb");
        saved = _dup(2);
        if (out && saved >= 0) _dup2(_fileno(out), 2);
        if (out) std::fclose(out);
    }
    std::string finish() {
        std::fflush(stderr);
        if (saved >= 0) {
            _dup2(saved, 2);
            _close(saved);
            saved = -1;
        }
        std::ifstream in(path, std::ios::binary);
        return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    }
    ~StderrCapture() { finish(); }
};

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    vktest::Device vk;
    const auto device_status = vk.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) {
        std::fprintf(stderr, "SKIP: no usable Vulkan 1.3 device\n");
        return 77;
    }
    REQUIRE(device_status == vktest::DeviceStatus::ready);
    const testfiles::ScratchDir scratch("leonida_refused_call_log_test");  // removed on every way out of main

    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = vk.instance_proc;
    info.instance = vk.instance;
    info.physical = vk.physical;
    info.device = vk.device;
    info.queue = vk.queue;
    info.queue_family = vk.family;
    info.buffer_device_address = true;
    Ref<ID3D12Device> device;
    REQUIRE(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))));
    Ref<ID3D12CommandAllocator> allocator;
    REQUIRE(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    Ref<ID3D12GraphicsCommandList> list;
    REQUIRE(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                                IID_PPV_ARGS(&list.p))));

    const float color[4] = {};
    D3D12_VIEWPORT viewports[D3D12_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE + 1] = {};
    for (D3D12_VIEWPORT& v : viewports) v = {0.0f, 0.0f, 64.0f, 64.0f, 0.0f, 1.0f};
    HRESULT refused[3] = {S_OK, S_OK, S_OK};
    HRESULT valid = E_FAIL;
    std::string log;
    {
        StderrCapture capture(scratch.path() / "stderr.txt");
        for (HRESULT& close : refused) {
            if (&close != refused) {
                if (FAILED(allocator->Reset()) || FAILED(list->Reset(allocator.p, nullptr))) break;
            }
            list->ClearRenderTargetView(D3D12_CPU_DESCRIPTOR_HANDLE{0}, color, 0, nullptr);
            list->RSSetViewports(UINT(std::size(viewports)), viewports);
            close = list->Close();
        }
        if (SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, nullptr))) {
            list->RSSetViewports(1, viewports);
            valid = list->Close();
        }
        log = capture.finish();
    }
    std::fputs(log.c_str(), stdout);

    for (const HRESULT close : refused) REQUIRE(close == E_INVALIDARG);
    REQUIRE(valid == S_OK);
    REQUIRE(count_of(log, "ID3D12GraphicsCommandList::ClearRenderTargetView refused a recorded call") == 1);
    REQUIRE(count_of(log, "ID3D12GraphicsCommandList::RSSetViewports refused a recorded call") == 1);
    REQUIRE(count_of(log, "refused a recorded call") == 2);
    REQUIRE(count_of(log, "(d3d12_com.cpp:") == 2);
    std::printf("d3d12_refused_call_log: three refused lists close with E_INVALIDARG and name their two refusing calls "
                "once each; a valid list closes with S_OK and logs nothing: PASS\n");
    return 0;
}
