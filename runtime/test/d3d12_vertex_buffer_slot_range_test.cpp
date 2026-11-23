// D3D12 IASetVertexBuffers slot overflow parity against native WARP. Synthetic COM test, no game data.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include <dxgi1_4.h>
#include <cstdio>
#include <cstdint>

namespace {
template<class T> struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
};
struct Step { UINT start, count; bool null_views; const char* label; };
constexpr Step cases[] = {
    {30, 4, false, "30+4 overflows two slots"},
    {31, 2, false, "31+2 overflows one slot"},
    {30, 2, false, "30+2 fits exactly"},
    {30, 4, true,  "30+4 unbinds via nullptr"},
    {32, 1, false, "32+1 starts past end"},
    {0, 0, true,   "zero count"},
};

int run(ID3D12Device* device, const char* name, bool warp, bool results[sizeof(cases)/sizeof(cases[0])]) {
    Ref<ID3D12CommandQueue> queue;
    D3D12_COMMAND_QUEUE_DESC q{};
    q.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (FAILED(device->CreateCommandQueue(&q, IID_PPV_ARGS(&queue.p)))) return 1;
    Ref<ID3D12Fence> fence;
    if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p)))) return 1;
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event) return 1;
    for (unsigned i = 0; i < sizeof(cases)/sizeof(cases[0]); ++i) {
        Ref<ID3D12CommandAllocator> allocator;
        Ref<ID3D12GraphicsCommandList> list;
        if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))) ||
            FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                             IID_PPV_ARGS(&list.p)))) { CloseHandle(event); return 1; }
        const D3D12_VERTEX_BUFFER_VIEW views[4] = {};
        const Step& step = cases[i];
        list->IASetVertexBuffers(step.start, step.count, step.null_views ? nullptr : views);
        const HRESULT hr = list->Close();
        results[i] = SUCCEEDED(hr);
        std::printf("%s %s: Close %08x\n", name, step.label, unsigned(hr));
        if (!warp && !results[i]) continue;
        if (results[i]) {
            ID3D12CommandList* lists[] = {list.p};
            queue->ExecuteCommandLists(1, lists);
            if (FAILED(queue->Signal(fence.p, i + 1)) ||
                FAILED(fence->SetEventOnCompletion(i + 1, event)) ||
                WaitForSingleObject(event, 5000) != WAIT_OBJECT_0) {
                std::fprintf(stderr, "%s %s: GPU completion failed\n", name, step.label);
                CloseHandle(event); return 1;
            }
        }
    }
    CloseHandle(event);
    return 0;
}

int run_bundle(ID3D12Device* device, const char* name) {
    Ref<ID3D12CommandAllocator> bundle_allocator, direct_allocator;
    Ref<ID3D12GraphicsCommandList> bundle, direct;
    if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_BUNDLE, IID_PPV_ARGS(&bundle_allocator.p))) ||
        FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&direct_allocator.p))) ||
        FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_BUNDLE, bundle_allocator.p,
                                         nullptr, IID_PPV_ARGS(&bundle.p))) ||
        FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, direct_allocator.p,
                                         nullptr, IID_PPV_ARGS(&direct.p)))) return 1;
    const D3D12_VERTEX_BUFFER_VIEW views[4] = {};
    bundle->IASetVertexBuffers(30, 4, views);
    bundle->IASetVertexBuffers(31, 2, nullptr);
    bundle->IASetVertexBuffers(32, 1, views);
    if (FAILED(bundle->Close())) {
        std::fprintf(stderr, "%s: bundle overflow Close failed\n", name);
        return 1;
    }
    direct->ExecuteBundle(bundle.p);
    if (FAILED(direct->Close())) {
        std::fprintf(stderr, "%s: bundle parent Close failed\n", name);
        return 1;
    }
    Ref<ID3D12CommandQueue> queue;
    D3D12_COMMAND_QUEUE_DESC desc{};
    desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (FAILED(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&queue.p)))) return 1;
    ID3D12CommandList* lists[] = {direct.p};
    queue->ExecuteCommandLists(1, lists);
    Ref<ID3D12Fence> fence;
    if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))) ||
        FAILED(queue->Signal(fence.p, 1))) return 1;
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event) return 1;
    const bool completed = SUCCEEDED(fence->SetEventOnCompletion(1, event)) &&
                           WaitForSingleObject(event, 5000) == WAIT_OBJECT_0;
    CloseHandle(event);
    std::printf("%s: overflow bundle recording, parent Close and GPU execution %s\n",
                name, completed ? "passed" : "failed");
    return completed ? 0 : 1;
}
}

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);  // Preserve the last successful step on a failing driver path.
    Ref<IDXGIFactory4> factory;
    Ref<IDXGIAdapter> adapter;
    Ref<ID3D12Device> warp, layer;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p))) ||
        FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter.p))) ||
        FAILED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&warp.p)))) return 77;
    bool host_results[sizeof(cases)/sizeof(cases[0])]{};
    bool layer_results[sizeof(cases)/sizeof(cases[0])]{};
    if (run(warp.p, "native WARP", true, host_results)) return 1;
    if (run_bundle(warp.p, "native WARP")) return 1;
    vktest::Device vk;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;
    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = vk.instance_proc;
    info.instance = vk.instance;
    info.physical = vk.physical;
    info.device = vk.device;
    info.queue = vk.queue;
    info.queue_family = vk.family;
    if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&layer.p)))) return 1;
    int layer_failure = run(layer.p, "Leonida Vulkan", false, layer_results);
    if (!layer_failure) layer_failure = run_bundle(layer.p, "Leonida Vulkan");
    int mismatches = 0;
    for (unsigned i = 0; i < sizeof(cases)/sizeof(cases[0]); ++i)
        if (host_results[i] != layer_results[i]) {
            std::fprintf(stderr, "parity mismatch: %s WARP=%d Leonida=%d\n",
                         cases[i].label, int(host_results[i]), int(layer_results[i]));
            ++mismatches;
        }
    const unsigned errors = vk.validation_errors();
    std::printf("Vulkan validation errors: %u\n", errors);
    // The externally owned Vulkan device must outlive its ID3D12Device wrapper and queues.
    // `layer` was declared before `vk`, so release it explicitly before vk's destructor.
    layer.p->Release();
    layer.p = nullptr;
    return (layer_failure || mismatches || errors) ? 1 : 0;
}
