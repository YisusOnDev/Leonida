// WriteToSubresource / ReadFromSubresource via staged_copy() must leave the subresource in VK_IMAGE_LAYOUT_GENERAL
// (for a fresh COMMON texture, matching VK_EXT_host_image_copy) or restore its prior Vulkan layout (for textures
// already transitioned to RENDER_TARGET, PIXEL_SHADER_RESOURCE, DEPTH_WRITE, COPY_DEST, etc.).
//
// In D3D12, CPU subresource copies do not change a resource's state. Leaving a subresource in
// TRANSFER_DST_OPTIMAL / TRANSFER_SRC_OPTIMAL breaks AMD/NVIDIA parity (AMD GPUs without VK_EXT_host_image_copy
// always take staged_copy()) and leaves non-GENERAL textures in the wrong Vulkan layout for subsequent GPU work.
//
// Intercepts vkCreateImage, vkCmdPipelineBarrier, vkCmdPipelineBarrier2, and vkTransitionImageLayoutEXT on a real
// Vulkan 1.3 device to verify both pixel data and the resulting VkImageLayout across staging and host-copy paths.
// No shaders or game data. Skips with exit code 77 when no Vulkan 1.3 device is available.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string_view>
#include <unordered_map>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { reset(); }
    T* operator->() const { return p; }
    void reset() {
        if (p) p->Release();
        p = nullptr;
    }
};

namespace spy {

PFN_vkGetInstanceProcAddr real_get_instance = nullptr;
PFN_vkGetDeviceProcAddr real_get_device = nullptr;
PFN_vkCreateImage real_create_image = nullptr;
PFN_vkDestroyImage real_destroy_image = nullptr;
PFN_vkCmdPipelineBarrier real_barrier = nullptr;
PFN_vkCmdPipelineBarrier2 real_barrier2 = nullptr;
PFN_vkTransitionImageLayoutEXT real_host_transition = nullptr;

std::mutex mutex;
std::unordered_map<VkImage, VkImageLayout> layouts;
VkImage last_created_image = VK_NULL_HANDLE;

void reset() {
    std::lock_guard lock(mutex);
    layouts.clear();
    last_created_image = VK_NULL_HANDLE;
}

VkImage take_last_image() {
    std::lock_guard lock(mutex);
    return last_created_image;
}

VkImageLayout layout_of(VkImage img) {
    std::lock_guard lock(mutex);
    const auto it = layouts.find(img);
    return it == layouts.end() ? VK_IMAGE_LAYOUT_UNDEFINED : it->second;
}

VKAPI_ATTR VkResult VKAPI_CALL create_image(VkDevice dev, const VkImageCreateInfo* info,
                                            const VkAllocationCallbacks* alloc, VkImage* out) {
    const VkResult r = real_create_image(dev, info, alloc, out);
    if (r == VK_SUCCESS && out && *out) {
        std::lock_guard lock(mutex);
        layouts[*out] = info ? info->initialLayout : VK_IMAGE_LAYOUT_UNDEFINED;
        last_created_image = *out;
    }
    return r;
}

VKAPI_ATTR void VKAPI_CALL destroy_image(VkDevice dev, VkImage img, const VkAllocationCallbacks* alloc) {
    if (img) {
        std::lock_guard lock(mutex);
        layouts.erase(img);
    }
    real_destroy_image(dev, img, alloc);
}

VKAPI_ATTR void VKAPI_CALL cmd_barrier(VkCommandBuffer cb, VkPipelineStageFlags src_stage,
                                       VkPipelineStageFlags dst_stage, VkDependencyFlags dep,
                                       std::uint32_t mem_count, const VkMemoryBarrier* mems,
                                       std::uint32_t buf_count, const VkBufferMemoryBarrier* bufs,
                                       std::uint32_t img_count, const VkImageMemoryBarrier* imgs) {
    {
        std::lock_guard lock(mutex);
        for (std::uint32_t i = 0; i < img_count; ++i)
            layouts[imgs[i].image] = imgs[i].newLayout;
    }
    real_barrier(cb, src_stage, dst_stage, dep, mem_count, mems, buf_count, bufs, img_count, imgs);
}

VKAPI_ATTR void VKAPI_CALL cmd_barrier2(VkCommandBuffer cb, const VkDependencyInfo* dep) {
    if (dep) {
        std::lock_guard lock(mutex);
        for (std::uint32_t i = 0; i < dep->imageMemoryBarrierCount; ++i)
            layouts[dep->pImageMemoryBarriers[i].image] = dep->pImageMemoryBarriers[i].newLayout;
    }
    real_barrier2(cb, dep);
}

VKAPI_ATTR VkResult VKAPI_CALL host_transition(VkDevice dev, std::uint32_t count,
                                               const VkHostImageLayoutTransitionInfoEXT* infos) {
    const VkResult r = real_host_transition(dev, count, infos);
    if (r == VK_SUCCESS && infos) {
        std::lock_guard lock(mutex);
        for (std::uint32_t i = 0; i < count; ++i)
            layouts[infos[i].image] = infos[i].newLayout;
    }
    return r;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_device(VkDevice dev, const char* name) {
    const std::string_view n = name;
    if (n == "vkCreateImage") {
        if (!real_create_image) real_create_image = reinterpret_cast<PFN_vkCreateImage>(real_get_device(dev, name));
        return reinterpret_cast<PFN_vkVoidFunction>(create_image);
    }
    if (n == "vkDestroyImage") {
        if (!real_destroy_image) real_destroy_image = reinterpret_cast<PFN_vkDestroyImage>(real_get_device(dev, name));
        return reinterpret_cast<PFN_vkVoidFunction>(destroy_image);
    }
    if (n == "vkCmdPipelineBarrier") {
        if (!real_barrier) real_barrier = reinterpret_cast<PFN_vkCmdPipelineBarrier>(real_get_device(dev, name));
        return reinterpret_cast<PFN_vkVoidFunction>(cmd_barrier);
    }
    if (n == "vkCmdPipelineBarrier2") {
        if (!real_barrier2) real_barrier2 = reinterpret_cast<PFN_vkCmdPipelineBarrier2>(real_get_device(dev, name));
        return real_barrier2 ? reinterpret_cast<PFN_vkVoidFunction>(cmd_barrier2) : nullptr;
    }
    if (n == "vkTransitionImageLayoutEXT") {
        if (!real_host_transition)
            real_host_transition = reinterpret_cast<PFN_vkTransitionImageLayoutEXT>(real_get_device(dev, name));
        return real_host_transition ? reinterpret_cast<PFN_vkVoidFunction>(host_transition) : nullptr;
    }
    return real_get_device(dev, name);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_instance(VkInstance inst, const char* name) {
    const std::string_view n = name;
    if (n == "vkGetDeviceProcAddr") return reinterpret_cast<PFN_vkVoidFunction>(get_device);
    return real_get_instance(inst, name);
}

void install(PFN_vkGetInstanceProcAddr gipa, PFN_vkGetDeviceProcAddr gdpa) {
    real_get_instance = gipa;
    real_get_device = gdpa;
    real_create_image = nullptr;
    real_destroy_image = nullptr;
    real_barrier = nullptr;
    real_barrier2 = nullptr;
    real_host_transition = nullptr;
    reset();
}

}  // namespace spy

const char* layout_name(VkImageLayout l) {
    switch (l) {
    case VK_IMAGE_LAYOUT_UNDEFINED: return "UNDEFINED";
    case VK_IMAGE_LAYOUT_GENERAL: return "GENERAL";
    case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL: return "COLOR_ATTACHMENT_OPTIMAL";
    case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL: return "DEPTH_STENCIL_ATTACHMENT_OPTIMAL";
    case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL: return "DEPTH_STENCIL_READ_ONLY_OPTIMAL";
    case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL: return "SHADER_READ_ONLY_OPTIMAL";
    case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL: return "TRANSFER_SRC_OPTIMAL";
    case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL: return "TRANSFER_DST_OPTIMAL";
    default: return "OTHER";
    }
}

bool submit_and_wait(ID3D12CommandQueue* queue, ID3D12Fence* fence, UINT64& fence_val,
                     ID3D12GraphicsCommandList* list) {
    if (FAILED(list->Close())) return false;
    ID3D12CommandList* lists[] = {list};
    queue->ExecuteCommandLists(1, lists);
    const UINT64 target = ++fence_val;
    if (FAILED(queue->Signal(fence, target))) return false;
    if (fence->GetCompletedValue() < target) {
        HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!ev) return false;
        const HRESULT hr = fence->SetEventOnCompletion(target, ev);
        if (SUCCEEDED(hr)) WaitForSingleObject(ev, 10000);
        CloseHandle(ev);
        if (FAILED(hr) || fence->GetCompletedValue() < target) return false;
    }
    return true;
}

D3D12_HEAP_PROPERTIES custom_wb_heap() {
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_CUSTOM;
    hp.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
    hp.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
    return hp;
}

D3D12_RESOURCE_DESC tex2d_desc(DXGI_FORMAT fmt, D3D12_RESOURCE_FLAGS flags) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Width = 4;
    d.Height = 4;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.Format = fmt;
    d.SampleDesc.Count = 1;
    d.Flags = flags;
    return d;
}

unsigned check_layout(const char* path, const char* step, VkImage img, VkImageLayout expected) {
    const VkImageLayout actual = spy::layout_of(img);
    if (actual != expected) {
        std::fprintf(stderr, "FAIL [%s] %s: layout is %s (%d), expected %s (%d)\n",
                     path, step, layout_name(actual), int(actual), layout_name(expected), int(expected));
        return 1;
    }
    return 0;
}

unsigned run_path(vktest::Device& d, bool host_copy) {
    const char* path = host_copy ? "host_image_copy" : "staged_copy";
    spy::reset();

    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = spy::get_instance;
    info.instance = d.instance;
    info.physical = d.physical;
    info.device = d.device;
    info.queue = d.queue;
    info.queue_family = d.family;
    info.texture_compression_bc = d.bc;
    info.host_image_copy = host_copy;

    Ref<ID3D12Device> dev;
    if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&dev.p))) || !dev.p) {
        std::fprintf(stderr, "FAIL [%s]: create_vulkan_device failed\n", path);
        return 1;
    }

    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12Fence> fence;
    Ref<ID3D12CommandAllocator> alloc;
    Ref<ID3D12GraphicsCommandList> list;
    if (FAILED(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue.p))) ||
        FAILED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))) ||
        FAILED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc.p))) ||
        FAILED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr, IID_PPV_ARGS(&list.p))) ||
        FAILED(list->Close())) {
        std::fprintf(stderr, "FAIL [%s]: queue/fence/list creation failed\n", path);
        return 1;
    }
    UINT64 fence_val = 0;
    unsigned failures = 0;

    const D3D12_HEAP_PROPERTIES hp = custom_wb_heap();

    auto transition = [&](ID3D12Resource* res, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) -> bool {
        if (FAILED(alloc->Reset()) || FAILED(list->Reset(alloc.p, nullptr))) return false;
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.pResource = res;
        b.Transition.Subresource = 0;
        b.Transition.StateBefore = before;
        b.Transition.StateAfter = after;
        list->ResourceBarrier(1, &b);
        return submit_and_wait(queue.p, fence.p, fence_val, list.p);
    };

    // 1. Fresh COMMON texture: WriteToSubresource must leave it in GENERAL (matching host_image_copy).
    {
        Ref<ID3D12Resource> tex;
        const D3D12_RESOURCE_DESC td = tex2d_desc(DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
        if (FAILED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &td, D3D12_RESOURCE_STATE_COMMON,
                                                nullptr, IID_PPV_ARGS(&tex.p)))) {
            std::fprintf(stderr, "FAIL [%s]: create tex1 failed\n", path);
            return 1;
        }
        const VkImage img = spy::take_last_image();
        std::array<std::uint32_t, 16> src{};
        src.fill(0x11223344u);
        std::array<std::uint32_t, 16> dst{};

        if (FAILED(tex->Map(0, nullptr, nullptr)) ||
            FAILED(tex->WriteToSubresource(0, nullptr, src.data(), 16, 64))) {
            std::fprintf(stderr, "FAIL [%s]: fresh WriteToSubresource failed\n", path);
            ++failures;
        }
        failures += check_layout(path, "fresh COMMON after WriteToSubresource", img, VK_IMAGE_LAYOUT_GENERAL);

        if (FAILED(tex->ReadFromSubresource(dst.data(), 16, 64, 0, nullptr)) || dst != src) {
            std::fprintf(stderr, "FAIL [%s]: fresh ReadFromSubresource data mismatch\n", path);
            ++failures;
        }
        failures += check_layout(path, "COMMON after ReadFromSubresource", img, VK_IMAGE_LAYOUT_GENERAL);

        // 2. RENDER_TARGET (COLOR_ATTACHMENT_OPTIMAL): ReadFromSubresource & WriteToSubresource preserve layout.
        if (!transition(tex.p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET)) {
            std::fprintf(stderr, "FAIL [%s]: transition to RENDER_TARGET failed\n", path);
            ++failures;
        }
        failures += check_layout(path, "after transition to RENDER_TARGET", img, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

        // Clear via RTV so GPU writes a new colour (1, 0, 0, 1) -> 0xFF0000FF in RGBA8_UNORM.
        D3D12_DESCRIPTOR_HEAP_DESC rtv_hd{};
        rtv_hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        rtv_hd.NumDescriptors = 1;
        Ref<ID3D12DescriptorHeap> rtv_heap;
        if (SUCCEEDED(dev->CreateDescriptorHeap(&rtv_hd, IID_PPV_ARGS(&rtv_heap.p)))) {
            const D3D12_CPU_DESCRIPTOR_HANDLE rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
            dev->CreateRenderTargetView(tex.p, nullptr, rtv);
            if (SUCCEEDED(alloc->Reset()) && SUCCEEDED(list->Reset(alloc.p, nullptr))) {
                const float red[4] = {1.0f, 0.0f, 0.0f, 1.0f};
                list->ClearRenderTargetView(rtv, red, 0, nullptr);
                submit_and_wait(queue.p, fence.p, fence_val, list.p);
            }
        }
        dst.fill(0);
        if (FAILED(tex->ReadFromSubresource(dst.data(), 16, 64, 0, nullptr)) || dst[0] != 0xFF0000FFu) {
            std::fprintf(stderr, "FAIL [%s]: ReadFromSubresource in RENDER_TARGET got 0x%08x, expected 0xFF0000FF\n",
                         path, dst[0]);
            ++failures;
        }
        failures += check_layout(path, "RENDER_TARGET after ReadFromSubresource", img,
                                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

        src.fill(0x55667788u);
        if (FAILED(tex->WriteToSubresource(0, nullptr, src.data(), 16, 64))) {
            std::fprintf(stderr, "FAIL [%s]: WriteToSubresource in RENDER_TARGET failed\n", path);
            ++failures;
        }
        failures += check_layout(path, "RENDER_TARGET after WriteToSubresource", img,
                                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

        // 3. PIXEL_SHADER_RESOURCE (SHADER_READ_ONLY_OPTIMAL when bindless == false): preserved across CPU copies.
        if (!transition(tex.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)) {
            std::fprintf(stderr, "FAIL [%s]: transition to PIXEL_SHADER_RESOURCE failed\n", path);
            ++failures;
        }
        failures += check_layout(path, "after transition to PIXEL_SHADER_RESOURCE", img,
                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        dst.fill(0);
        if (FAILED(tex->ReadFromSubresource(dst.data(), 16, 64, 0, nullptr)) || dst != src) {
            std::fprintf(stderr, "FAIL [%s]: ReadFromSubresource in PIXEL_SHADER_RESOURCE mismatch\n", path);
            ++failures;
        }
        failures += check_layout(path, "PIXEL_SHADER_RESOURCE after ReadFromSubresource", img,
                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        // 4. COPY_DEST (TRANSFER_DST_OPTIMAL): ReadFromSubresource must restore TRANSFER_DST_OPTIMAL.
        if (!transition(tex.p, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST)) {
            std::fprintf(stderr, "FAIL [%s]: transition to COPY_DEST failed\n", path);
            ++failures;
        }
        failures += check_layout(path, "after transition to COPY_DEST", img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        dst.fill(0);
        if (FAILED(tex->ReadFromSubresource(dst.data(), 16, 64, 0, nullptr)) || dst != src) {
            std::fprintf(stderr, "FAIL [%s]: ReadFromSubresource in COPY_DEST mismatch\n", path);
            ++failures;
        }
        failures += check_layout(path, "COPY_DEST after ReadFromSubresource", img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        tex->Unmap(0, nullptr);
    }

    // 5. Fresh COMMON texture read-first (ReadFromSubresource before any write): must end in GENERAL with zeros.
    {
        Ref<ID3D12Resource> tex;
        const D3D12_RESOURCE_DESC td = tex2d_desc(DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE);
        if (FAILED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &td, D3D12_RESOURCE_STATE_COMMON,
                                                nullptr, IID_PPV_ARGS(&tex.p)))) {
            std::fprintf(stderr, "FAIL [%s]: create tex2 failed\n", path);
            return 1;
        }
        const VkImage img = spy::take_last_image();
        std::array<std::uint32_t, 16> dst{};
        dst.fill(0xDEADBEEFu);
        if (FAILED(tex->Map(0, nullptr, nullptr)) ||
            FAILED(tex->ReadFromSubresource(dst.data(), 16, 64, 0, nullptr))) {
            std::fprintf(stderr, "FAIL [%s]: fresh ReadFromSubresource failed\n", path);
            ++failures;
        }
        for (std::uint32_t px : dst) {
            if (px != 0u) {
                std::fprintf(stderr, "FAIL [%s]: fresh texture not zero-initialized (got 0x%08x)\n", path, px);
                ++failures;
                break;
            }
        }
        failures += check_layout(path, "fresh COMMON after ReadFromSubresource", img, VK_IMAGE_LAYOUT_GENERAL);
        tex->Unmap(0, nullptr);
    }

    // 6. DEPTH_WRITE (DEPTH_STENCIL_ATTACHMENT_OPTIMAL, D32_FLOAT): ReadFromSubresource preserves layout.
    {
        Ref<ID3D12Resource> tex;
        const D3D12_RESOURCE_DESC td = tex2d_desc(DXGI_FORMAT_D32_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
        if (FAILED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &td, D3D12_RESOURCE_STATE_COMMON,
                                                nullptr, IID_PPV_ARGS(&tex.p)))) {
            std::fprintf(stderr, "FAIL [%s]: create depth tex failed\n", path);
            return 1;
        }
        const VkImage img = spy::take_last_image();
        if (FAILED(tex->Map(0, nullptr, nullptr))) {
            std::fprintf(stderr, "FAIL [%s]: depth Map failed\n", path);
            ++failures;
        }
        if (!transition(tex.p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_DEPTH_WRITE)) {
            std::fprintf(stderr, "FAIL [%s]: transition to DEPTH_WRITE failed\n", path);
            ++failures;
        }
        failures += check_layout(path, "after transition to DEPTH_WRITE", img,
                                 VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        std::array<float, 16> depth_in{};
        depth_in.fill(0.25f);
        std::array<float, 16> depth_out{};
        if (FAILED(tex->WriteToSubresource(0, nullptr, depth_in.data(), 16, 64))) {
            std::fprintf(stderr, "FAIL [%s]: depth WriteToSubresource failed\n", path);
            ++failures;
        }
        failures += check_layout(path, "DEPTH_WRITE after WriteToSubresource", img,
                                 VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        if (FAILED(tex->ReadFromSubresource(depth_out.data(), 16, 64, 0, nullptr)) || depth_out[0] != 0.25f) {
            std::fprintf(stderr, "FAIL [%s]: depth ReadFromSubresource got %f, expected 0.25\n", path, depth_out[0]);
            ++failures;
        }
        failures += check_layout(path, "DEPTH_WRITE after ReadFromSubresource", img,
                                 VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        tex->Unmap(0, nullptr);
    }

    std::printf("subresource copy layout [%s]: %u failure(s)\n", path, failures);
    return failures;
}

}  // namespace

int main() {
    vktest::Device d;
    const auto status = d.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;
    spy::install(d.instance_proc, d.device_proc);

    unsigned failures = 0;
    failures += run_path(d, false);
    if (d.host_image_copy) {
        failures += run_path(d, true);
    } else {
        std::printf("subresource copy layout [host_image_copy]: no VK_EXT_host_image_copy on this GPU, skipped\n");
    }

    if (failures) {
        std::fprintf(stderr, "d3d12_subresource_copy_layout_test: FAIL (%u failures)\n", failures);
        return 1;
    }
    std::printf("d3d12_subresource_copy_layout_test: PASS\n");
    return 0;
}
