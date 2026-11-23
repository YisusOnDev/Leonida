// A resource released while no submission is in flight gives its Vulkan memory back at once, as the host D3D12
// runtime does (docs/d3d12_vulkan.md section 4.4). Released objects go through the device's deletion queue, which
// waits for the submission after the latest one; with the queues idle that submission may never come, so 2000
// create/release cycles of a 512x512 render target with nothing submitted grew local video memory by 2.5 GB on an
// RX 6750 XT (native D3D12: +0). Counted here through vkAllocateMemory/vkFreeMemory hooks, so it runs on any GPU:
//   1. nothing submitted yet: a committed texture and an upload buffer, released, are freed before Release returns;
//   2. after a submission that used it has completed (CPU fence wait): an upload buffer is freed on release;
//   3. released while a GPU-gated copy still reads it (queue A waits on the GPU for a fence queue C signals after
//      a long run of copies): not freed while the gate is closed, the copy reads its bytes, and it is freed by the
//      first release after the work has completed.
// No shaders or game data. LEONIDA_VK_VALIDATION=1 fails the test on any validation error.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"

#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <utility>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)
// A finding that does not stop the run, so every case reports.
#define CHECK(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); ++failures; } \
} while (0)

namespace {
template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
};

// Live device memory, each with the serial of its allocation (a freed handle value may come back for a new one).
struct Hooks {
    PFN_vkGetInstanceProcAddr instance_proc = nullptr;
    PFN_vkGetDeviceProcAddr device_proc = nullptr;
    PFN_vkAllocateMemory allocate = nullptr;
    PFN_vkFreeMemory free = nullptr;
    std::mutex mutex;
    std::map<VkDeviceMemory, std::uint64_t> live;
    std::uint64_t serial = 0;
    bool capturing = false;
    std::vector<std::pair<VkDeviceMemory, std::uint64_t>> captured;
} g;

VKAPI_ATTR VkResult VKAPI_CALL allocate_memory(VkDevice device, const VkMemoryAllocateInfo* info,
                                               const VkAllocationCallbacks* callbacks, VkDeviceMemory* memory) {
    const VkResult result = g.allocate(device, info, callbacks, memory);
    if (result == VK_SUCCESS) {
        std::lock_guard lock(g.mutex);
        g.live[*memory] = ++g.serial;
        if (g.capturing) g.captured.emplace_back(*memory, g.serial);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL free_memory(VkDevice device, VkDeviceMemory memory, const VkAllocationCallbacks* callbacks) {
    {
        std::lock_guard lock(g.mutex);
        g.live.erase(memory);
    }
    g.free(device, memory, callbacks);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL device_proc(VkDevice device, const char* name) {
    if (!std::strcmp(name, "vkAllocateMemory")) {
        g.allocate = reinterpret_cast<PFN_vkAllocateMemory>(g.device_proc(device, name));
        return g.allocate ? reinterpret_cast<PFN_vkVoidFunction>(&allocate_memory) : nullptr;
    }
    if (!std::strcmp(name, "vkFreeMemory")) {
        g.free = reinterpret_cast<PFN_vkFreeMemory>(g.device_proc(device, name));
        return g.free ? reinterpret_cast<PFN_vkVoidFunction>(&free_memory) : nullptr;
    }
    return g.device_proc(device, name);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL instance_proc(VkInstance instance, const char* name) {
    if (!std::strcmp(name, "vkGetDeviceProcAddr")) return reinterpret_cast<PFN_vkVoidFunction>(&device_proc);
    return g.instance_proc(instance, name);
}

using Memory = std::vector<std::pair<VkDeviceMemory, std::uint64_t>>;

void begin_capture() {
    std::lock_guard lock(g.mutex);
    g.captured.clear();
    g.capturing = true;
}

Memory end_capture() {
    std::lock_guard lock(g.mutex);
    g.capturing = false;
    return std::move(g.captured);
}

// How many of `memory` are still allocated.
unsigned live_count(const Memory& memory) {
    std::lock_guard lock(g.mutex);
    unsigned n = 0;
    for (const auto& [handle, serial] : memory) {
        const auto it = g.live.find(handle);
        n += it != g.live.end() && it->second == serial;
    }
    return n;
}

D3D12_RESOURCE_DESC buffer_desc(UINT64 bytes) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return desc;
}

HRESULT make_buffer(ID3D12Device* device, D3D12_HEAP_TYPE type, UINT64 bytes, ID3D12Resource** out) {
    const D3D12_HEAP_PROPERTIES heap{type};
    const D3D12_RESOURCE_STATES state = type == D3D12_HEAP_TYPE_UPLOAD     ? D3D12_RESOURCE_STATE_GENERIC_READ
                                                                           : D3D12_RESOURCE_STATE_COPY_DEST;
    const D3D12_RESOURCE_DESC desc = buffer_desc(bytes);
    return device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(out));
}

void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource, D3D12_RESOURCE_STATES before,
                D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after};
    list->ResourceBarrier(1, &barrier);
}

// An upload buffer of `bytes` filled with `seed`-dependent words, and the memory its creation allocated.
HRESULT make_filled_upload(ID3D12Device* device, UINT64 bytes, std::uint32_t seed, ID3D12Resource** out,
                           Memory& memory) {
    begin_capture();
    const HRESULT hr = make_buffer(device, D3D12_HEAP_TYPE_UPLOAD, bytes, out);
    memory = end_capture();
    if (FAILED(hr)) return hr;
    void* data = nullptr;
    if (FAILED((*out)->Map(0, nullptr, &data))) return E_FAIL;
    auto* words = static_cast<std::uint32_t*>(data);
    for (UINT64 i = 0; i < bytes / 4; ++i) words[i] = std::uint32_t(i * 2654435761u) ^ seed;
    (*out)->Unmap(0, nullptr);
    return S_OK;
}

bool wait_fence(ID3D12Fence* fence, UINT64 value, DWORD ms) {
    if (fence->GetCompletedValue() >= value) return true;
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    const bool ok = SUCCEEDED(fence->SetEventOnCompletion(value, event)) && WaitForSingleObject(event, ms) == WAIT_OBJECT_0;
    CloseHandle(event);
    return ok;
}

struct Queue {
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    bool create(ID3D12Device* device) {
        const D3D12_COMMAND_QUEUE_DESC desc{D3D12_COMMAND_LIST_TYPE_DIRECT};
        return SUCCEEDED(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&queue.p))) &&
               SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))) &&
               SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                                   IID_PPV_ARGS(&list.p))) &&
               SUCCEEDED(list->Close());
    }
    bool reset() { return SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, nullptr)); }
    void execute() {
        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
    }
};

// Words of `readback` that differ from what make_filled_upload(seed) wrote.
UINT64 mismatches(ID3D12Resource* readback, UINT64 bytes, std::uint32_t seed) {
    void* data = nullptr;
    const D3D12_RANGE range{0, SIZE_T(bytes)};
    if (FAILED(readback->Map(0, &range, &data))) return bytes / 4;
    const auto* words = static_cast<const std::uint32_t*>(data);
    UINT64 bad = 0;
    for (UINT64 i = 0; i < bytes / 4; ++i) bad += words[i] != (std::uint32_t(i * 2654435761u) ^ seed);
    readback->Unmap(0, nullptr);
    return bad;
}

int run(ID3D12Device* device) {
    int failures = 0;
    // 1. Nothing submitted yet.
    {
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Width = 512;
        desc.Height = 512;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.SampleDesc.Count = 1;
        desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
        ID3D12Resource* texture = nullptr;
        begin_capture();
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                         D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr,
                                                         IID_PPV_ARGS(&texture))));
        const Memory texture_memory = end_capture();
        ID3D12Resource* upload = nullptr;
        Memory upload_memory;
        EXPECT(SUCCEEDED(make_filled_upload(device, 1 << 20, 1, &upload, upload_memory)));
        EXPECT(!texture_memory.empty() && !upload_memory.empty());
        texture->Release();
        upload->Release();
        const unsigned texture_live = live_count(texture_memory), upload_live = live_count(upload_memory);
        std::printf("nothing submitted: released texture keeps %u of %zu allocations, upload buffer %u of %zu\n",
                    texture_live, texture_memory.size(), upload_live, upload_memory.size());
        CHECK(texture_live == 0 && upload_live == 0);
    }

    constexpr UINT64 kSmall = 64 << 10;
    Queue a, c;
    EXPECT(a.create(device) && c.create(device));
    Ref<ID3D12Fence> gate, done;
    EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&gate.p))));
    EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&done.p))));
    UINT64 gate_value = 0, done_value = 0;
    Ref<ID3D12Resource> readback;
    EXPECT(SUCCEEDED(make_buffer(device, D3D12_HEAP_TYPE_READBACK, kSmall, &readback.p)));

    // 2. After a submission that used it has completed.
    {
        ID3D12Resource* upload = nullptr;
        Memory memory;
        EXPECT(SUCCEEDED(make_filled_upload(device, kSmall, 2, &upload, memory)));
        EXPECT(a.reset());
        a.list->CopyBufferRegion(readback.p, 0, upload, 0, kSmall);
        EXPECT(SUCCEEDED(a.list->Close()));
        a.execute();
        EXPECT(SUCCEEDED(a.queue->Signal(done.p, ++done_value)));
        EXPECT(wait_fence(done.p, done_value, 10000));
        EXPECT(mismatches(readback.p, kSmall, 2) == 0);
        upload->Release();
        const unsigned live = live_count(memory);
        std::printf("after a completed submission: released upload buffer keeps %u of %zu allocations\n", live,
                    memory.size());
        CHECK(live == 0);
    }

    // 3. Released while a GPU-gated copy still reads it. Queue C's copies run long enough that the gate is still
    // closed when the buffer is released; with a GPU fast enough to finish them first the run doubles them.
    constexpr UINT64 kBig = 64ull << 20;
    Ref<ID3D12Resource> big_src, big_dst;
    EXPECT(SUCCEEDED(make_buffer(device, D3D12_HEAP_TYPE_UPLOAD, kBig, &big_src.p)));
    EXPECT(SUCCEEDED(make_buffer(device, D3D12_HEAP_TYPE_DEFAULT, kBig, &big_dst.p)));
    bool gated = false;
    for (UINT copies = 16; copies <= 1024 && !gated; copies *= 2) {
        EXPECT(c.reset());
        // Each copy is ordered after the last (COPY_DEST, COPY_SOURCE and back): copies to one buffer with no
        // barrier between them are a write-after-write hazard.
        for (UINT i = 0; i < copies; ++i) {
            c.list->CopyBufferRegion(big_dst.p, 0, big_src.p, 0, kBig);
            transition(c.list.p, big_dst.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
            transition(c.list.p, big_dst.p, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
        }
        EXPECT(SUCCEEDED(c.list->Close()));
        EXPECT(a.reset());
        ID3D12Resource* upload = nullptr;
        Memory memory;
        const std::uint32_t seed = 3 + copies;
        EXPECT(SUCCEEDED(make_filled_upload(device, kSmall, seed, &upload, memory)));
        a.list->CopyBufferRegion(readback.p, 0, upload, 0, kSmall);
        EXPECT(SUCCEEDED(a.list->Close()));

        c.execute();
        EXPECT(SUCCEEDED(c.queue->Signal(gate.p, ++gate_value)));
        EXPECT(SUCCEEDED(a.queue->Wait(gate.p, gate_value)));
        a.execute();
        EXPECT(SUCCEEDED(a.queue->Signal(done.p, ++done_value)));
        upload->Release();
        const unsigned live_gated = live_count(memory);
        gated = gate->GetCompletedValue() < gate_value;
        EXPECT(wait_fence(done.p, done_value, 60000));
        const UINT64 bad = mismatches(readback.p, kSmall, seed);
        // The first release after the work has completed takes the gated buffer's memory with it.
        ID3D12Resource* other = nullptr;
        EXPECT(SUCCEEDED(make_buffer(device, D3D12_HEAP_TYPE_UPLOAD, kSmall, &other)));
        other->Release();
        const unsigned live_after = live_count(memory);
        std::printf("gated read, %u copies of 64 MiB: gate %s at release; released buffer keeps %u of %zu allocations "
                    "while the read is pending, %u after it completed; %llu of %llu words wrong\n",
                    copies, gated ? "closed" : "already open", live_gated, memory.size(), live_after,
                    static_cast<unsigned long long>(bad), static_cast<unsigned long long>(kSmall / 4));
        if (gated) CHECK(live_gated == memory.size());
        CHECK(bad == 0);
        CHECK(live_after == 0);
    }
    if (!gated) std::printf("the gate was open before every release: the in-flight case was not exercised\n");
    return failures;
}
}  // namespace

int main() {
    vktest::Device vk;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;
    g.instance_proc = vk.instance_proc;
    g.device_proc = reinterpret_cast<PFN_vkGetDeviceProcAddr>(vk.instance_proc(vk.instance, "vkGetDeviceProcAddr"));
    if (!g.device_proc) return 1;
    int failures = 0;
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = instance_proc;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        // Distinct VkQueues for the two D3D12 queues where the family has them, so the gate is a cross-queue wait.
        if (vk.queues.size() >= 2) info.command_queues = vk.queues;
        Ref<ID3D12Device> device;
        if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p)))) return 1;
        if (!g.allocate || !g.free) {
            std::fprintf(stderr, "the device did not load vkAllocateMemory/vkFreeMemory through the hook\n");
            return 1;
        }
        failures += run(device.p);
    }
    if (vk.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", vk.validation_errors());
        return 1;
    }
    return failures ? 1 : 0;
}
