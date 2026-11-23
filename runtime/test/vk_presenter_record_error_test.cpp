// Cross-platform Vulkan regression test for VulkanPresenter command-buffer recording error handling
// (runtime/src/vulkan_present.h):
//   1. record(), record_planes(), and record_blank() must check vkResetCommandBuffer, vkBeginCommandBuffer,
//      and vkEndCommandBuffer, propagating the VkResult without recording commands into an unbegun buffer or
//      submitting an invalid command buffer.
//   2. When recording fails after detail::initialize() has transitioned UNDEFINED textures in LayoutTracker
//      (e.g. vkEndCommandBuffer fails), submit() must roll back LayoutTracker before unlocking tracker_mutex_
//      and calling give_back(), and record_planes() must drop its plane VkImageViews.
//   3. tracker_mutex_ must not stay held across give_back()'s vkQueueSubmit on any recording failure.
//   4. After recovery via give_back(), subsequent present(), present_blank(), and present_planes() succeed cleanly.
// Uses a synthetic WSI dispatch backed by real VkImages and real semaphore signal/wait submissions so it runs
// headless on Linux (llvmpipe), Windows, and macOS under VK_LAYER_KHRONOS_validation. Exits 77 without Vulkan 1.3.
// No game data.
#include "vk_test_device.h"
#include "../src/vulkan_present.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <mutex>
#include <string_view>
#include <thread>
#include <vector>

#define EXPECT(cond)                                                                                   \
    do {                                                                                               \
        if (!(cond)) {                                                                                 \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #cond);                                    \
            return 1;                                                                                  \
        }                                                                                              \
    } while (0)

namespace {

constexpr std::uint32_t kWidth = 64;
constexpr std::uint32_t kHeight = 64;

struct FakeSwapchain {
    std::array<VkImage, 2> images{};
    std::array<VkDeviceMemory, 2> memories{};
    std::uint32_t next_image = 0;
};

namespace driver {
vktest::Device* dev = nullptr;
PFN_vkGetInstanceProcAddr real_get_instance = nullptr;
PFN_vkGetDeviceProcAddr real_get_device = nullptr;
PFN_vkResetCommandBuffer real_reset_cb = nullptr;
PFN_vkBeginCommandBuffer real_begin_cb = nullptr;
PFN_vkEndCommandBuffer real_end_cb = nullptr;
PFN_vkQueueSubmit real_queue_submit = nullptr;
PFN_vkCmdPipelineBarrier real_barrier = nullptr;
PFN_vkCmdBlitImage real_blit = nullptr;
PFN_vkCmdClearColorImage real_clear_color = nullptr;
PFN_vkCmdDispatch real_dispatch = nullptr;
PFN_vkCreateImageView real_create_view = nullptr;
PFN_vkDestroyImageView real_destroy_view = nullptr;

std::vector<VkImage> swapchain_images;
std::mutex swapchain_mutex;

std::atomic<VkResult> fail_next_reset{VK_SUCCESS};
std::atomic<VkResult> fail_next_begin{VK_SUCCESS};
std::atomic<VkResult> fail_next_end{VK_SUCCESS};
std::atomic<int> fail_view_countdown{0};  // 1 => fail next vkCreateImageView; 2 => fail the one after, etc.
std::atomic<VkResult> fail_view_error{VK_SUCCESS};

std::atomic<std::uint32_t> recorded_cmds{0};
std::atomic<std::uint32_t> cb_submits{0};
std::atomic<std::uint32_t> empty_submits{0};
std::atomic<std::uint32_t> present_layout_barriers{0};
std::atomic<std::int64_t> live_views{0};

std::function<void()> on_next_empty_submit;

bool is_swapchain_image(VkImage img) {
    std::lock_guard lk(swapchain_mutex);
    return std::find(swapchain_images.begin(), swapchain_images.end(), img) != swapchain_images.end();
}

VKAPI_ATTR VkResult VKAPI_CALL fake_surface_support(VkPhysicalDevice, std::uint32_t, VkSurfaceKHR, VkBool32* out) {
    *out = VK_TRUE;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL fake_surface_caps(VkPhysicalDevice, VkSurfaceKHR, VkSurfaceCapabilitiesKHR* caps) {
    *caps = {};
    caps->minImageCount = 2;
    caps->maxImageCount = 2;
    caps->currentExtent = {kWidth, kHeight};
    caps->minImageExtent = {kWidth, kHeight};
    caps->maxImageExtent = {kWidth, kHeight};
    caps->maxImageArrayLayers = 1;
    caps->supportedTransforms = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    caps->currentTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    caps->supportedCompositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    caps->supportedUsageFlags = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL fake_surface_formats(VkPhysicalDevice, VkSurfaceKHR, std::uint32_t* count,
                                                    VkSurfaceFormatKHR* formats) {
    if (!formats) {
        *count = 1;
        return VK_SUCCESS;
    }
    if (*count >= 1) {
        formats[0] = {VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
        *count = 1;
    }
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL fake_present_modes(VkPhysicalDevice, VkSurfaceKHR, std::uint32_t* count,
                                                  VkPresentModeKHR* modes) {
    if (!modes) {
        *count = 1;
        return VK_SUCCESS;
    }
    if (*count >= 1) {
        modes[0] = VK_PRESENT_MODE_FIFO_KHR;
        *count = 1;
    }
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL fake_destroy_surface(VkInstance, VkSurfaceKHR, const VkAllocationCallbacks*) {}

VKAPI_ATTR VkResult VKAPI_CALL fake_create_swapchain(VkDevice device, const VkSwapchainCreateInfoKHR* info,
                                                     const VkAllocationCallbacks*, VkSwapchainKHR* out) {
    auto* sc = new FakeSwapchain{};
    const auto create_image = dev->fn<PFN_vkCreateImage>("vkCreateImage");
    const auto get_reqs = dev->fn<PFN_vkGetImageMemoryRequirements>("vkGetImageMemoryRequirements");
    const auto alloc_mem = dev->fn<PFN_vkAllocateMemory>("vkAllocateMemory");
    const auto bind_mem = dev->fn<PFN_vkBindImageMemory>("vkBindImageMemory");
    const auto destroy_image = dev->fn<PFN_vkDestroyImage>("vkDestroyImage");
    const auto free_mem = dev->fn<PFN_vkFreeMemory>("vkFreeMemory");

    VkPhysicalDeviceMemoryProperties mem_props{};
    dev->instance_fn<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties")(dev->physical,
                                                                                                     &mem_props);
    for (std::size_t i = 0; i < sc->images.size(); ++i) {
        VkImageCreateInfo img{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        img.imageType = VK_IMAGE_TYPE_2D;
        img.format = info->imageFormat;
        img.extent = {info->imageExtent.width, info->imageExtent.height, 1};
        img.mipLevels = 1;
        img.arrayLayers = 1;
        img.samples = VK_SAMPLE_COUNT_1_BIT;
        img.tiling = VK_IMAGE_TILING_OPTIMAL;
        img.usage = info->imageUsage;
        img.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        img.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (create_image(device, &img, nullptr, &sc->images[i]) != VK_SUCCESS) {
            for (std::size_t j = 0; j < i; ++j) {
                destroy_image(device, sc->images[j], nullptr);
                free_mem(device, sc->memories[j], nullptr);
            }
            delete sc;
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        }
        VkMemoryRequirements reqs{};
        get_reqs(device, sc->images[i], &reqs);
        std::uint32_t type = UINT32_MAX;
        for (std::uint32_t t = 0; t < mem_props.memoryTypeCount; ++t) {
            if (!(reqs.memoryTypeBits & (1u << t))) continue;
            if (type == UINT32_MAX) type = t;
            if (mem_props.memoryTypes[t].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
                type = t;
                break;
            }
        }
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = reqs.size;
        ai.memoryTypeIndex = type;
        if (type == UINT32_MAX || alloc_mem(device, &ai, nullptr, &sc->memories[i]) != VK_SUCCESS ||
            bind_mem(device, sc->images[i], sc->memories[i], 0) != VK_SUCCESS) {
            for (std::size_t j = 0; j <= i; ++j) {
                if (sc->images[j]) destroy_image(device, sc->images[j], nullptr);
                if (sc->memories[j]) free_mem(device, sc->memories[j], nullptr);
            }
            delete sc;
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        }
    }
    {
        std::lock_guard lk(swapchain_mutex);
        for (VkImage img : sc->images) swapchain_images.push_back(img);
    }
    *out = reinterpret_cast<VkSwapchainKHR>(sc);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL fake_destroy_swapchain(VkDevice device, VkSwapchainKHR swapchain,
                                                  const VkAllocationCallbacks*) {
    if (!swapchain) return;
    auto* sc = reinterpret_cast<FakeSwapchain*>(swapchain);
    const auto destroy_image = dev->fn<PFN_vkDestroyImage>("vkDestroyImage");
    const auto free_mem = dev->fn<PFN_vkFreeMemory>("vkFreeMemory");
    {
        std::lock_guard lk(swapchain_mutex);
        for (VkImage img : sc->images) {
            swapchain_images.erase(std::remove(swapchain_images.begin(), swapchain_images.end(), img),
                                   swapchain_images.end());
        }
    }
    for (std::size_t i = 0; i < sc->images.size(); ++i) {
        if (sc->images[i]) destroy_image(device, sc->images[i], nullptr);
        if (sc->memories[i]) free_mem(device, sc->memories[i], nullptr);
    }
    delete sc;
}

VKAPI_ATTR VkResult VKAPI_CALL fake_get_swapchain_images(VkDevice, VkSwapchainKHR swapchain, std::uint32_t* count,
                                                         VkImage* images) {
    auto* sc = reinterpret_cast<FakeSwapchain*>(swapchain);
    if (!images) {
        *count = std::uint32_t(sc->images.size());
        return VK_SUCCESS;
    }
    const std::uint32_t n = std::min<std::uint32_t>(*count, std::uint32_t(sc->images.size()));
    for (std::uint32_t i = 0; i < n; ++i) images[i] = sc->images[i];
    *count = n;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL fake_acquire_next_image(VkDevice, VkSwapchainKHR swapchain, std::uint64_t,
                                                       VkSemaphore semaphore, VkFence, std::uint32_t* image_index) {
    auto* sc = reinterpret_cast<FakeSwapchain*>(swapchain);
    *image_index = sc->next_image++ % std::uint32_t(sc->images.size());
    if (semaphore != VK_NULL_HANDLE) {
        // Genuinely signal the binary acquire semaphore on the queue so VVL sees balanced signal/wait pairs.
        VkSubmitInfo sig{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        sig.signalSemaphoreCount = 1;
        sig.pSignalSemaphores = &semaphore;
        const VkResult r = real_queue_submit(dev->queue, 1, &sig, VK_NULL_HANDLE);
        if (r != VK_SUCCESS) return r;
    }
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL fake_queue_present(VkQueue queue, const VkPresentInfoKHR* info) {
    if (info && info->waitSemaphoreCount > 0 && info->pWaitSemaphores) {
        // Consume the binary ready semaphore on the queue so it can be safely reused on later frames under VVL.
        std::vector<VkPipelineStageFlags> stages(info->waitSemaphoreCount, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
        VkSubmitInfo wait{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        wait.waitSemaphoreCount = info->waitSemaphoreCount;
        wait.pWaitSemaphores = info->pWaitSemaphores;
        wait.pWaitDstStageMask = stages.data();
        const VkResult r = real_queue_submit(queue, 1, &wait, VK_NULL_HANDLE);
        if (r != VK_SUCCESS) return r;
    }
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL reset_cb(VkCommandBuffer cb, VkCommandBufferResetFlags flags) {
    const VkResult armed = fail_next_reset.exchange(VK_SUCCESS, std::memory_order_acq_rel);
    if (armed != VK_SUCCESS) return armed;
    return real_reset_cb(cb, flags);
}

VKAPI_ATTR VkResult VKAPI_CALL begin_cb(VkCommandBuffer cb, const VkCommandBufferBeginInfo* info) {
    const VkResult armed = fail_next_begin.exchange(VK_SUCCESS, std::memory_order_acq_rel);
    if (armed != VK_SUCCESS) return armed;
    return real_begin_cb(cb, info);
}

VKAPI_ATTR VkResult VKAPI_CALL end_cb(VkCommandBuffer cb) {
    const VkResult armed = fail_next_end.exchange(VK_SUCCESS, std::memory_order_acq_rel);
    if (armed != VK_SUCCESS) return armed;
    return real_end_cb(cb);
}

VKAPI_ATTR VkResult VKAPI_CALL queue_submit(VkQueue queue, std::uint32_t count, const VkSubmitInfo* submits,
                                            VkFence fence) {
    bool has_cb = false;
    for (std::uint32_t i = 0; i < count; ++i)
        if (submits[i].commandBufferCount > 0) has_cb = true;
    if (has_cb) {
        cb_submits.fetch_add(1, std::memory_order_relaxed);
    } else {
        empty_submits.fetch_add(1, std::memory_order_relaxed);
        if (on_next_empty_submit) {
            auto cb = std::move(on_next_empty_submit);
            on_next_empty_submit = {};
            cb();
        }
    }
    return real_queue_submit(queue, count, submits, fence);
}

VKAPI_ATTR void VKAPI_CALL cmd_pipeline_barrier(VkCommandBuffer cb, VkPipelineStageFlags src_stage,
                                                VkPipelineStageFlags dst_stage, VkDependencyFlags flags,
                                                std::uint32_t mem_count, const VkMemoryBarrier* mem,
                                                std::uint32_t buf_count, const VkBufferMemoryBarrier* buf,
                                                std::uint32_t img_count, const VkImageMemoryBarrier* img) {
    recorded_cmds.fetch_add(1, std::memory_order_relaxed);
    // Translate VK_IMAGE_LAYOUT_PRESENT_SRC_KHR on synthetic headless swapchain images so devices initialized
    // without VK_KHR_swapchain pass VK_LAYER_KHRONOS_validation cleanly.
    std::vector<VkImageMemoryBarrier> patched;
    const VkImageMemoryBarrier* img_arg = img;
    for (std::uint32_t i = 0; i < img_count; ++i) {
        if (img[i].newLayout == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR && is_swapchain_image(img[i].image)) {
            present_layout_barriers.fetch_add(1, std::memory_order_relaxed);
            if (patched.empty()) patched.assign(img, img + img_count);
            patched[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
            img_arg = patched.data();
        }
    }
    real_barrier(cb, src_stage, dst_stage, flags, mem_count, mem, buf_count, buf, img_count, img_arg);
}

VKAPI_ATTR void VKAPI_CALL cmd_blit(VkCommandBuffer cb, VkImage src, VkImageLayout src_layout, VkImage dst,
                                    VkImageLayout dst_layout, std::uint32_t region_count, const VkImageBlit* regions,
                                    VkFilter filter) {
    recorded_cmds.fetch_add(1, std::memory_order_relaxed);
    real_blit(cb, src, src_layout, dst, dst_layout, region_count, regions, filter);
}

VKAPI_ATTR void VKAPI_CALL cmd_clear_color(VkCommandBuffer cb, VkImage img, VkImageLayout layout,
                                           const VkClearColorValue* color, std::uint32_t range_count,
                                           const VkImageSubresourceRange* ranges) {
    recorded_cmds.fetch_add(1, std::memory_order_relaxed);
    real_clear_color(cb, img, layout, color, range_count, ranges);
}

VKAPI_ATTR void VKAPI_CALL cmd_dispatch(VkCommandBuffer cb, std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    recorded_cmds.fetch_add(1, std::memory_order_relaxed);
    real_dispatch(cb, x, y, z);
}

VKAPI_ATTR VkResult VKAPI_CALL create_view(VkDevice device, const VkImageViewCreateInfo* info,
                                           const VkAllocationCallbacks* alloc, VkImageView* out) {
    int prev = fail_view_countdown.load(std::memory_order_acquire);
    while (prev > 0) {
        if (fail_view_countdown.compare_exchange_weak(prev, prev - 1, std::memory_order_acq_rel)) {
            if (prev == 1) return fail_view_error.exchange(VK_SUCCESS, std::memory_order_acq_rel);
            break;
        }
    }
    const VkResult res = real_create_view(device, info, alloc, out);
    if (res == VK_SUCCESS && out && *out != VK_NULL_HANDLE)
        live_views.fetch_add(1, std::memory_order_relaxed);
    return res;
}

VKAPI_ATTR void VKAPI_CALL destroy_view(VkDevice device, VkImageView view, const VkAllocationCallbacks* alloc) {
    if (view != VK_NULL_HANDLE) live_views.fetch_sub(1, std::memory_order_relaxed);
    real_destroy_view(device, view, alloc);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_device(VkDevice device, const char* name) {
    const std::string_view n = name;
    if (n == "vkCreateSwapchainKHR") return reinterpret_cast<PFN_vkVoidFunction>(fake_create_swapchain);
    if (n == "vkDestroySwapchainKHR") return reinterpret_cast<PFN_vkVoidFunction>(fake_destroy_swapchain);
    if (n == "vkGetSwapchainImagesKHR") return reinterpret_cast<PFN_vkVoidFunction>(fake_get_swapchain_images);
    if (n == "vkAcquireNextImageKHR") return reinterpret_cast<PFN_vkVoidFunction>(fake_acquire_next_image);
    if (n == "vkQueuePresentKHR") return reinterpret_cast<PFN_vkVoidFunction>(fake_queue_present);
    if (n == "vkResetCommandBuffer") return reinterpret_cast<PFN_vkVoidFunction>(reset_cb);
    if (n == "vkBeginCommandBuffer") return reinterpret_cast<PFN_vkVoidFunction>(begin_cb);
    if (n == "vkEndCommandBuffer") return reinterpret_cast<PFN_vkVoidFunction>(end_cb);
    if (n == "vkQueueSubmit") return reinterpret_cast<PFN_vkVoidFunction>(queue_submit);
    if (n == "vkCmdPipelineBarrier") return reinterpret_cast<PFN_vkVoidFunction>(cmd_pipeline_barrier);
    if (n == "vkCmdBlitImage") return reinterpret_cast<PFN_vkVoidFunction>(cmd_blit);
    if (n == "vkCmdClearColorImage") return reinterpret_cast<PFN_vkVoidFunction>(cmd_clear_color);
    if (n == "vkCmdDispatch") return reinterpret_cast<PFN_vkVoidFunction>(cmd_dispatch);
    if (n == "vkCreateImageView") return reinterpret_cast<PFN_vkVoidFunction>(create_view);
    if (n == "vkDestroyImageView") return reinterpret_cast<PFN_vkVoidFunction>(destroy_view);
    return real_get_device(device, name);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_instance(VkInstance instance, const char* name) {
    const std::string_view n = name;
    if (n == "vkGetPhysicalDeviceSurfaceSupportKHR") return reinterpret_cast<PFN_vkVoidFunction>(fake_surface_support);
    if (n == "vkGetPhysicalDeviceSurfaceCapabilitiesKHR") return reinterpret_cast<PFN_vkVoidFunction>(fake_surface_caps);
    if (n == "vkGetPhysicalDeviceSurfaceFormatsKHR") return reinterpret_cast<PFN_vkVoidFunction>(fake_surface_formats);
    if (n == "vkGetPhysicalDeviceSurfacePresentModesKHR") return reinterpret_cast<PFN_vkVoidFunction>(fake_present_modes);
    if (n == "vkDestroySurfaceKHR") return reinterpret_cast<PFN_vkVoidFunction>(fake_destroy_surface);
    if (n == "vkGetDeviceProcAddr") return reinterpret_cast<PFN_vkVoidFunction>(get_device);
    return real_get_instance(instance, name);
}

void install(vktest::Device& device) {
    dev = &device;
    real_get_instance = device.instance_proc;
    real_get_device = device.device_proc;
    real_reset_cb = reinterpret_cast<PFN_vkResetCommandBuffer>(real_get_device(device.device, "vkResetCommandBuffer"));
    real_begin_cb = reinterpret_cast<PFN_vkBeginCommandBuffer>(real_get_device(device.device, "vkBeginCommandBuffer"));
    real_end_cb = reinterpret_cast<PFN_vkEndCommandBuffer>(real_get_device(device.device, "vkEndCommandBuffer"));
    real_queue_submit = reinterpret_cast<PFN_vkQueueSubmit>(real_get_device(device.device, "vkQueueSubmit"));
    real_barrier = reinterpret_cast<PFN_vkCmdPipelineBarrier>(real_get_device(device.device, "vkCmdPipelineBarrier"));
    real_blit = reinterpret_cast<PFN_vkCmdBlitImage>(real_get_device(device.device, "vkCmdBlitImage"));
    real_clear_color =
        reinterpret_cast<PFN_vkCmdClearColorImage>(real_get_device(device.device, "vkCmdClearColorImage"));
    real_dispatch = reinterpret_cast<PFN_vkCmdDispatch>(real_get_device(device.device, "vkCmdDispatch"));
    real_create_view = reinterpret_cast<PFN_vkCreateImageView>(real_get_device(device.device, "vkCreateImageView"));
    real_destroy_view = reinterpret_cast<PFN_vkDestroyImageView>(real_get_device(device.device, "vkDestroyImageView"));
    live_views.store(0);
}

void reset_counters() {
    recorded_cmds.store(0, std::memory_order_relaxed);
    cb_submits.store(0, std::memory_order_relaxed);
    empty_submits.store(0, std::memory_order_relaxed);
    present_layout_barriers.store(0, std::memory_order_relaxed);
}
}  // namespace driver

// Probes context.layout(id, 0) from a worker thread while running inside give_back()'s vkQueueSubmit hook to verify
// that tracker_mutex_ is unlocked before give_back() and that rollback already restored UNDEFINED.
struct GiveBackProbe {
    std::mutex mutex;
    std::condition_variable cv;
    bool started = false;
    bool done = false;
    bool unblocked = false;
    VkImageLayout observed = VK_IMAGE_LAYOUT_MAX_ENUM;
    std::thread worker;

    void arm(leonida::VulkanContext& context, leonida::ResourceId id) {
        driver::on_next_empty_submit = [this, &context, id] {
            worker = std::thread([this, &context, id] {
                {
                    std::lock_guard lk(mutex);
                    started = true;
                }
                cv.notify_all();
                const VkImageLayout layout = context.layout(id, 0);
                {
                    std::lock_guard lk(mutex);
                    observed = layout;
                    done = true;
                }
                cv.notify_all();
            });
            std::unique_lock lk(mutex);
            const bool did_start = cv.wait_for(lk, std::chrono::seconds(2), [this] { return started; });
            unblocked = did_start && cv.wait_for(lk, std::chrono::milliseconds(500), [this] { return done; });
        };
    }

    ~GiveBackProbe() {
        driver::on_next_empty_submit = {};
        if (worker.joinable()) worker.join();
    }
};

}  // namespace

int main() {
    using namespace leonida;

    vktest::Device d;
    const auto status = d.initialize();
    if (status == vktest::DeviceStatus::unavailable) {
        std::printf("SKIP: vk_presenter_record_error_test: no Vulkan 1.3 device\n");
        return 77;
    }
    if (status != vktest::DeviceStatus::ready) return 1;

    driver::install(d);

    const VkImage back_img = d.image(VK_FORMAT_R8G8B8A8_UNORM, kWidth, kHeight, 1,
                                     VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    const VkImage plane0_img = d.image(VK_FORMAT_R8G8B8A8_UNORM, kWidth, kHeight, 1,
                                       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    const VkImage plane1_img = d.image(VK_FORMAT_R8G8B8A8_UNORM, kWidth, kHeight, 1,
                                       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    EXPECT(back_img && plane0_img && plane1_img);

    {
        VulkanContext context(driver::get_device, d.device, true);
        EXPECT(context.loaded());

        auto make_binding = [](VkImage img) {
            ResourceBinding rb{};
            rb.image = img;
            rb.format = VK_FORMAT_R8G8B8A8_UNORM;
            rb.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
            rb.width = kWidth;
            rb.height = kHeight;
            rb.mip_levels = 1;
            rb.array_layers = 1;
            rb.depth = 1;
            rb.samples = 1;
            rb.color_attachment = true;
            rb.sampled = true;
            return rb;
        };
        context.add_resource(1, make_binding(back_img), VK_IMAGE_LAYOUT_UNDEFINED);
        context.add_resource(2, make_binding(plane0_img), VK_IMAGE_LAYOUT_UNDEFINED);
        context.add_resource(3, make_binding(plane1_img), VK_IMAGE_LAYOUT_UNDEFINED);

        const auto fake_surface = reinterpret_cast<VkSurfaceKHR>(std::uintptr_t(0x1));
        VulkanPresenter presenter(context, driver::get_instance, d.instance, d.physical, d.queue, d.family, fake_surface);
        EXPECT(presenter.ok());

        // 1. present(): failed vkResetCommandBuffer must not record commands, touch tracker, or submit the CB.
        driver::reset_counters();
        driver::fail_next_reset.store(VK_ERROR_OUT_OF_HOST_MEMORY, std::memory_order_release);
        EXPECT(presenter.present(1, true) == VK_ERROR_OUT_OF_HOST_MEMORY);
        EXPECT(driver::recorded_cmds.load() == 0);
        EXPECT(driver::cb_submits.load() == 0);
        EXPECT(driver::empty_submits.load() == 1);
        EXPECT(context.layout(1, 0) == VK_IMAGE_LAYOUT_UNDEFINED);
        EXPECT(presenter.ok());

        // 2. present(): failed vkBeginCommandBuffer must not record commands, touch tracker, or submit the CB.
        driver::reset_counters();
        driver::fail_next_begin.store(VK_ERROR_OUT_OF_DEVICE_MEMORY, std::memory_order_release);
        EXPECT(presenter.present(1, true) == VK_ERROR_OUT_OF_DEVICE_MEMORY);
        EXPECT(driver::recorded_cmds.load() == 0);
        EXPECT(driver::cb_submits.load() == 0);
        EXPECT(driver::empty_submits.load() == 1);
        EXPECT(context.layout(1, 0) == VK_IMAGE_LAYOUT_UNDEFINED);
        EXPECT(presenter.ok());

        // 3. present(): failed vkEndCommandBuffer after detail::initialize() must roll back LayoutTracker,
        //    unlock tracker_mutex_ before give_back(), avoid submitting the CB, and recover on the next present.
        {
            GiveBackProbe probe;
            probe.arm(context, 1);
            driver::reset_counters();
            driver::fail_next_end.store(VK_ERROR_OUT_OF_HOST_MEMORY, std::memory_order_release);
            EXPECT(presenter.present(1, true) == VK_ERROR_OUT_OF_HOST_MEMORY);
            EXPECT(driver::cb_submits.load() == 0);
            EXPECT(driver::empty_submits.load() == 1);
            EXPECT(probe.unblocked);
            EXPECT(probe.observed == VK_IMAGE_LAYOUT_UNDEFINED);
            EXPECT(context.layout(1, 0) == VK_IMAGE_LAYOUT_UNDEFINED);
            EXPECT(presenter.ok());
        }
        driver::reset_counters();
        EXPECT(presenter.present(1, true) == VK_SUCCESS);
        EXPECT(driver::cb_submits.load() == 1);
        EXPECT(driver::present_layout_barriers.load() == 1);
        EXPECT(context.layout(1, 0) == VK_IMAGE_LAYOUT_GENERAL);
        EXPECT(presenter.presented() == 1);

        // 4. present_blank(): failed reset, begin, and end must propagate errors, skip CB submit, and recover.
        driver::reset_counters();
        driver::fail_next_reset.store(VK_ERROR_OUT_OF_HOST_MEMORY, std::memory_order_release);
        EXPECT(presenter.present_blank(true) == VK_ERROR_OUT_OF_HOST_MEMORY);
        EXPECT(driver::recorded_cmds.load() == 0);
        EXPECT(driver::cb_submits.load() == 0);
        EXPECT(driver::empty_submits.load() == 1);
        EXPECT(presenter.ok());

        driver::reset_counters();
        driver::fail_next_begin.store(VK_ERROR_OUT_OF_DEVICE_MEMORY, std::memory_order_release);
        EXPECT(presenter.present_blank(true) == VK_ERROR_OUT_OF_DEVICE_MEMORY);
        EXPECT(driver::recorded_cmds.load() == 0);
        EXPECT(driver::cb_submits.load() == 0);
        EXPECT(driver::empty_submits.load() == 1);
        EXPECT(presenter.ok());

        driver::reset_counters();
        driver::fail_next_end.store(VK_ERROR_OUT_OF_HOST_MEMORY, std::memory_order_release);
        EXPECT(presenter.present_blank(true) == VK_ERROR_OUT_OF_HOST_MEMORY);
        EXPECT(driver::cb_submits.load() == 0);
        EXPECT(driver::empty_submits.load() == 1);
        EXPECT(presenter.ok());

        driver::reset_counters();
        EXPECT(presenter.present_blank(true) == VK_SUCCESS);
        EXPECT(driver::cb_submits.load() == 1);
        EXPECT(driver::present_layout_barriers.load() == 1);
        EXPECT(presenter.presented() == 2);

        // 5. present_planes() (2 UNDEFINED planes): check view-creation failure, reset/begin/end CB failures,
        //    plane view cleanup (0 leaked views), LayoutTracker rollback for both planes, tracker_mutex_ unlocked
        //    during give_back(), and clean recovery.
        PresentPlanes planes{};
        planes.count = 2;
        planes.plane[0].source = 2;
        planes.plane[1].source = 3;

        // First call initializes PresentComposer (creating its persistent target image view before record_planes).
        // Trigger a failure on the 2nd plane view (3rd view overall: target + plane0 + failing plane1) and verify
        // plane0's view is dropped while PresentComposer's 1 target view stays alive.
        {
            GiveBackProbe probe;
            probe.arm(context, 2);
            driver::reset_counters();
            driver::fail_view_error.store(VK_ERROR_OUT_OF_HOST_MEMORY, std::memory_order_release);
            driver::fail_view_countdown.store(3, std::memory_order_release);
            EXPECT(presenter.present_planes(planes, true) == VK_ERROR_OUT_OF_HOST_MEMORY);
            EXPECT(driver::recorded_cmds.load() == 0);
            EXPECT(driver::cb_submits.load() == 0);
            EXPECT(driver::empty_submits.load() == 1);
            EXPECT(driver::live_views.load() == 1);  // only PresentComposer's persistent target view
            EXPECT(probe.unblocked);
            EXPECT(probe.observed == VK_IMAGE_LAYOUT_UNDEFINED);
            EXPECT(context.layout(2, 0) == VK_IMAGE_LAYOUT_UNDEFINED);
            EXPECT(context.layout(3, 0) == VK_IMAGE_LAYOUT_UNDEFINED);
            EXPECT(presenter.ok());
        }

        // present_planes(): failed vkResetCommandBuffer drops both plane views and records nothing.
        driver::reset_counters();
        driver::fail_next_reset.store(VK_ERROR_OUT_OF_HOST_MEMORY, std::memory_order_release);
        EXPECT(presenter.present_planes(planes, true) == VK_ERROR_OUT_OF_HOST_MEMORY);
        EXPECT(driver::recorded_cmds.load() == 0);
        EXPECT(driver::cb_submits.load() == 0);
        EXPECT(driver::empty_submits.load() == 1);
        EXPECT(driver::live_views.load() == 1);
        EXPECT(context.layout(2, 0) == VK_IMAGE_LAYOUT_UNDEFINED);
        EXPECT(context.layout(3, 0) == VK_IMAGE_LAYOUT_UNDEFINED);
        EXPECT(presenter.ok());

        // present_planes(): failed vkBeginCommandBuffer drops both plane views and records nothing.
        driver::reset_counters();
        driver::fail_next_begin.store(VK_ERROR_OUT_OF_DEVICE_MEMORY, std::memory_order_release);
        EXPECT(presenter.present_planes(planes, true) == VK_ERROR_OUT_OF_DEVICE_MEMORY);
        EXPECT(driver::recorded_cmds.load() == 0);
        EXPECT(driver::cb_submits.load() == 0);
        EXPECT(driver::empty_submits.load() == 1);
        EXPECT(driver::live_views.load() == 1);
        EXPECT(context.layout(2, 0) == VK_IMAGE_LAYOUT_UNDEFINED);
        EXPECT(context.layout(3, 0) == VK_IMAGE_LAYOUT_UNDEFINED);
        EXPECT(presenter.ok());

        // present_planes(): failed vkEndCommandBuffer after both planes were initialized rolls back BOTH planes,
        // drops both plane views, unlocks tracker_mutex_ before give_back(), and does not submit the invalid CB.
        {
            GiveBackProbe probe;
            probe.arm(context, 3);
            driver::reset_counters();
            driver::fail_next_end.store(VK_ERROR_OUT_OF_HOST_MEMORY, std::memory_order_release);
            EXPECT(presenter.present_planes(planes, true) == VK_ERROR_OUT_OF_HOST_MEMORY);
            EXPECT(driver::cb_submits.load() == 0);
            EXPECT(driver::empty_submits.load() == 1);
            EXPECT(driver::live_views.load() == 1);
            EXPECT(probe.unblocked);
            EXPECT(probe.observed == VK_IMAGE_LAYOUT_UNDEFINED);
            EXPECT(context.layout(2, 0) == VK_IMAGE_LAYOUT_UNDEFINED);
            EXPECT(context.layout(3, 0) == VK_IMAGE_LAYOUT_UNDEFINED);
            EXPECT(presenter.ok());
        }

        // Recovery: normal present_planes() succeeds, composes both planes, and commits GENERAL for both.
        driver::reset_counters();
        EXPECT(presenter.present_planes(planes, true) == VK_SUCCESS);
        EXPECT(driver::cb_submits.load() == 1);
        EXPECT(driver::present_layout_barriers.load() == 1);
        EXPECT(context.layout(2, 0) == VK_IMAGE_LAYOUT_GENERAL);
        EXPECT(context.layout(3, 0) == VK_IMAGE_LAYOUT_GENERAL);
        EXPECT(presenter.composed_presents() == 1);
        EXPECT(presenter.presented() == 3);
    }

    EXPECT(driver::live_views.load() == 0);
    if (vktest::validation_error_count() != 0) {
        std::fprintf(stderr, "vk_presenter_record_error_test: %u validation errors\n",
                     vktest::validation_error_count());
        return 1;
    }
    std::printf("vk_presenter_record_error_test: all present/blank/planes recording-error cases passed (validation errors: 0)\n");
    return 0;
}
