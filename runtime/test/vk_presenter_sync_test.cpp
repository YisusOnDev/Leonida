// VulkanPresenter synchronization regressions (runtime/src/vulkan_present.h):
//   1. A failed vkQueueSubmit rolls back LayoutTracker changes (an UNDEFINED back buffer stays UNDEFINED until a
//      present actually submits its zero-fill).
//   2. With enable_capture(), every frame's vkCmdCopyImageToBuffer into the shared capture buffer is preceded by a
//      TRANSFER_WRITE -> TRANSFER_WRITE barrier to prevent WRITE_AFTER_WRITE hazards across the two in-flight frames.
//      A failed capture fence wait also leaves the caller's outputs untouched.
//   3. Concurrent present() + VulkanQueue::execute() + add_resource()/remove_resource() follows VulkanContext's
//      resource_mutex_ -> tracker_mutex_ -> submit_mutex_ lock hierarchy without data races or deadlocks.
//   4. A swapchain-destroy WSI layer which calls vkDeviceWaitIdle cannot overlap vkQueueSubmit on another thread.
// Skips with exit code 77 when no Vulkan 1.3 device with Win32 presentation is available. No game data.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define VK_USE_PLATFORM_WIN32_KHR
#include <windows.h>

#include "vk_presenter_capture_colour.h"
#include "vk_test_device.h"
#include "../src/vulkan_present.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <string_view>
#include <thread>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

LRESULT CALLBACK window_proc(HWND window, UINT message, WPARAM w, LPARAM l) {
    return DefWindowProcW(window, message, w, l);
}

void pump() {
    MSG message;
    while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) {
        TranslateMessage(&message);
        DispatchMessageW(&message);
    }
}

namespace driver {
PFN_vkGetInstanceProcAddr real_get_instance = nullptr;
PFN_vkGetDeviceProcAddr real_get_device = nullptr;
PFN_vkQueueSubmit real_submit = nullptr;
PFN_vkQueuePresentKHR real_present = nullptr;
PFN_vkWaitForFences real_wait = nullptr;
PFN_vkCmdPipelineBarrier real_barrier = nullptr;
PFN_vkCmdCopyImageToBuffer real_copy_i2b = nullptr;
PFN_vkDestroySwapchainKHR real_destroy_swapchain = nullptr;
PFN_vkDeviceWaitIdle real_device_wait_idle = nullptr;

std::atomic<VkResult> fail_next_submit{VK_SUCCESS};
std::atomic<VkResult> fail_next_wait{VK_SUCCESS};
std::function<void()> on_next_present;
std::atomic<bool> observe_capture{false};
std::atomic<bool> saw_capture_waw{false};
std::atomic<std::uint32_t> capture_copies{0};
std::atomic<std::uint32_t> capture_waw_barriers{0};
std::atomic<bool> check_swapchain_destroy{false};
std::atomic<bool> destroy_entered{false};
std::atomic<bool> submit_entered{false};
std::atomic<bool> overlap_detected{false};
std::atomic<bool> unexpected_submit_error{false};
std::mutex destroy_mutex;
std::condition_variable destroy_cv;
bool allow_destroy_finish = false;

// Synthetic WSI layer: a destroy hook initiates vkDeviceWaitIdle, as a capture layer may do internally.
// Holding the hook behind a condition variable makes the concurrency window deterministic rather than dependent
// on the installed capture software or validation logging overhead. Only the test hook waits on this condition.
VKAPI_ATTR void VKAPI_CALL destroy_swapchain(VkDevice device, VkSwapchainKHR swapchain,
                                              const VkAllocationCallbacks* alloc) {
    if (check_swapchain_destroy.load(std::memory_order_relaxed)) {
        destroy_entered.store(true, std::memory_order_release);
        destroy_cv.notify_all();
        {
            std::unique_lock lock(destroy_mutex);
            destroy_cv.wait(lock, [] { return allow_destroy_finish; });
        }
        overlap_detected.store(submit_entered.load(std::memory_order_acquire), std::memory_order_release);
        if (real_device_wait_idle(device) != VK_SUCCESS) unexpected_submit_error.store(true);
    }
    real_destroy_swapchain(device, swapchain, alloc);
}

VKAPI_ATTR VkResult VKAPI_CALL queue_submit(VkQueue queue, std::uint32_t count, const VkSubmitInfo* submits,
                                            VkFence fence) {
    submit_entered.store(true, std::memory_order_release);
    const VkResult armed = fail_next_submit.exchange(VK_SUCCESS);
    if (armed != VK_SUCCESS) return armed;
    return real_submit(queue, count, submits, fence);
}

VKAPI_ATTR VkResult VKAPI_CALL queue_present(VkQueue queue, const VkPresentInfoKHR* info) {
    if (on_next_present) {
        auto cb = std::move(on_next_present);
        on_next_present = {};
        cb();
    }
    return real_present(queue, info);
}

VKAPI_ATTR VkResult VKAPI_CALL wait_for_fences(VkDevice device, std::uint32_t count, const VkFence* fences,
                                               VkBool32 wait_all, std::uint64_t timeout) {
    const VkResult armed = fail_next_wait.exchange(VK_SUCCESS);
    if (armed != VK_SUCCESS) return armed;
    return real_wait(device, count, fences, wait_all, timeout);
}

VKAPI_ATTR void VKAPI_CALL cmd_pipeline_barrier(VkCommandBuffer cb, VkPipelineStageFlags src_stage,
                                                VkPipelineStageFlags dst_stage, VkDependencyFlags flags,
                                                std::uint32_t mem_count, const VkMemoryBarrier* mem,
                                                std::uint32_t buf_count, const VkBufferMemoryBarrier* buf,
                                                std::uint32_t img_count, const VkImageMemoryBarrier* img) {
    if (observe_capture.load(std::memory_order_relaxed) && (src_stage & VK_PIPELINE_STAGE_TRANSFER_BIT) &&
        (dst_stage & VK_PIPELINE_STAGE_TRANSFER_BIT)) {
        for (std::uint32_t i = 0; i < buf_count; ++i) {
            if ((buf[i].srcAccessMask & VK_ACCESS_TRANSFER_WRITE_BIT) &&
                (buf[i].dstAccessMask & VK_ACCESS_TRANSFER_WRITE_BIT))
                saw_capture_waw.store(true, std::memory_order_relaxed);
        }
        for (std::uint32_t i = 0; i < mem_count; ++i) {
            if ((mem[i].srcAccessMask & VK_ACCESS_TRANSFER_WRITE_BIT) &&
                (mem[i].dstAccessMask & VK_ACCESS_TRANSFER_WRITE_BIT))
                saw_capture_waw.store(true, std::memory_order_relaxed);
        }
    }
    real_barrier(cb, src_stage, dst_stage, flags, mem_count, mem, buf_count, buf, img_count, img);
}

VKAPI_ATTR void VKAPI_CALL cmd_copy_image_to_buffer(VkCommandBuffer cb, VkImage src, VkImageLayout layout,
                                                    VkBuffer dst, std::uint32_t count,
                                                    const VkBufferImageCopy* regions) {
    if (observe_capture.load(std::memory_order_relaxed)) {
        capture_copies.fetch_add(1, std::memory_order_relaxed);
        if (saw_capture_waw.exchange(false, std::memory_order_relaxed))
            capture_waw_barriers.fetch_add(1, std::memory_order_relaxed);
    }
    real_copy_i2b(cb, src, layout, dst, count, regions);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_device(VkDevice device, const char* name) {
    const std::string_view n = name;
    if (n == "vkDestroySwapchainKHR") return reinterpret_cast<PFN_vkVoidFunction>(destroy_swapchain);
    if (n == "vkQueueSubmit") return reinterpret_cast<PFN_vkVoidFunction>(queue_submit);
    if (n == "vkQueuePresentKHR") return reinterpret_cast<PFN_vkVoidFunction>(queue_present);
    if (n == "vkWaitForFences") return reinterpret_cast<PFN_vkVoidFunction>(wait_for_fences);
    if (n == "vkCmdPipelineBarrier") return reinterpret_cast<PFN_vkVoidFunction>(cmd_pipeline_barrier);
    if (n == "vkCmdCopyImageToBuffer") return reinterpret_cast<PFN_vkVoidFunction>(cmd_copy_image_to_buffer);
    return real_get_device(device, name);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_instance(VkInstance instance, const char* name) {
    if (std::string_view(name) == "vkGetDeviceProcAddr") return reinterpret_cast<PFN_vkVoidFunction>(get_device);
    return real_get_instance(instance, name);
}

void install(PFN_vkGetInstanceProcAddr instance_proc, VkInstance instance, VkDevice device) {
    real_get_instance = instance_proc;
    real_get_device = reinterpret_cast<PFN_vkGetDeviceProcAddr>(instance_proc(instance, "vkGetDeviceProcAddr"));
    real_destroy_swapchain = reinterpret_cast<PFN_vkDestroySwapchainKHR>(real_get_device(device, "vkDestroySwapchainKHR"));
    real_device_wait_idle = reinterpret_cast<PFN_vkDeviceWaitIdle>(real_get_device(device, "vkDeviceWaitIdle"));
    real_submit = reinterpret_cast<PFN_vkQueueSubmit>(real_get_device(device, "vkQueueSubmit"));
    real_present = reinterpret_cast<PFN_vkQueuePresentKHR>(real_get_device(device, "vkQueuePresentKHR"));
    real_wait = reinterpret_cast<PFN_vkWaitForFences>(real_get_device(device, "vkWaitForFences"));
    real_barrier = reinterpret_cast<PFN_vkCmdPipelineBarrier>(real_get_device(device, "vkCmdPipelineBarrier"));
    real_copy_i2b = reinterpret_cast<PFN_vkCmdCopyImageToBuffer>(real_get_device(device, "vkCmdCopyImageToBuffer"));
}
}  // namespace driver

}  // namespace

int main() {
    using namespace leonida;

    vktest::Device d;
    if (!d.create(true)) {
        std::fprintf(stderr, "vk_presenter_sync_test: no Vulkan device with presentation, skipped\n");
        return 77;
    }

    WNDCLASSW wc{};
    wc.lpfnWndProc = window_proc;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = L"LeonidaPresenterSyncTest";
    wc.hCursor = LoadCursorW(nullptr, reinterpret_cast<LPCWSTR>(IDC_ARROW));
    EXPECT(RegisterClassW(&wc));
    constexpr std::uint32_t width = 128, height = 96;
    RECT fr{0, 0, LONG(width), LONG(height)};
    AdjustWindowRect(&fr, WS_OVERLAPPEDWINDOW, FALSE);
    HWND window = CreateWindowExW(0, wc.lpszClassName, L"Leonida presenter sync test", WS_OVERLAPPEDWINDOW, 80, 80,
                                  fr.right - fr.left, fr.bottom - fr.top, nullptr, nullptr, wc.hInstance, nullptr);
    EXPECT(window);
    ShowWindow(window, SW_SHOWNOACTIVATE);
    UpdateWindow(window);
    pump();

    driver::install(d.instance_proc, d.instance, d.device);
    {
        VulkanContext context(driver::get_device, d.device, true);
        EXPECT(context.loaded());
        VulkanQueue queue(context, d.queue, d.family);
        VulkanQueue worker_queue(context, d.queue, d.family);
        EXPECT(queue.ok() && worker_queue.ok());

        const VkImage back_img = d.image(VK_FORMAT_R8G8B8A8_UNORM, width, height, 1,
                                         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                             VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        const VkImage side_img = d.image(VK_FORMAT_R8G8B8A8_UNORM, 32, 32, 1,
                                         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                             VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        EXPECT(back_img != VK_NULL_HANDLE && side_img != VK_NULL_HANDLE);

        ResourceBinding back_rb{};
        back_rb.image = back_img;
        back_rb.format = VK_FORMAT_R8G8B8A8_UNORM;
        back_rb.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
        back_rb.width = width;
        back_rb.height = height;
        back_rb.mip_levels = 1;
        back_rb.array_layers = 1;
        back_rb.depth = 1;
        back_rb.samples = 1;
        back_rb.color_attachment = true;
        context.add_resource(1, back_rb, VK_IMAGE_LAYOUT_UNDEFINED);

        ResourceBinding side_rb = back_rb;
        side_rb.image = side_img;
        side_rb.width = 32;
        side_rb.height = 32;

        const auto create_surface = reinterpret_cast<PFN_vkCreateWin32SurfaceKHR>(
            d.instance_proc(d.instance, "vkCreateWin32SurfaceKHR"));
        EXPECT(create_surface);
        VkWin32SurfaceCreateInfoKHR surface_info{VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
        surface_info.hinstance = GetModuleHandleW(nullptr);
        surface_info.hwnd = window;
        VkSurfaceKHR surface = VK_NULL_HANDLE;
        EXPECT(create_surface(d.instance, &surface_info, nullptr, &surface) == VK_SUCCESS);

        VulkanPresenter presenter(context, driver::get_instance, d.instance, d.physical, d.queue, d.family, surface);
        EXPECT(presenter.ok());
        EXPECT(presenter.enable_capture());

        // 1. Failed vkQueueSubmit must NOT commit the UNDEFINED -> GENERAL transition in LayoutTracker (O(1) rollback
        //    even with many live tracked textures), and tracker_mutex_ must be released before vkQueuePresentKHR.
        for (ResourceId extra = 1000; extra < 1512; ++extra)
            context.add_resource(extra, side_rb, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        EXPECT(context.layout(1, 0) == VK_IMAGE_LAYOUT_UNDEFINED);
        driver::fail_next_submit.store(VK_ERROR_OUT_OF_HOST_MEMORY);
        EXPECT(presenter.present(1, true) == VK_ERROR_OUT_OF_HOST_MEMORY);
        EXPECT(context.layout(1, 0) == VK_IMAGE_LAYOUT_UNDEFINED);
        EXPECT(context.layout(1000, 0) == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        // Next present succeeds, runs the deferred zero-fill, commits GENERAL, and unlocks tracker_mutex_ before
        // vkQueuePresentKHR so other threads' emit() / layout() calls are not blocked by presentation.
        std::mutex sync_mutex;
        std::condition_variable sync_cv;
        bool reader_started = false;
        bool read_done = false;
        VkImageLayout seen_during_present = VK_IMAGE_LAYOUT_UNDEFINED;
        bool unblocked_during_present = false;
        std::thread reader;
        driver::on_next_present = [&] {
            reader = std::thread([&] {
                {
                    std::lock_guard lk(sync_mutex);
                    reader_started = true;
                }
                sync_cv.notify_all();
                const VkImageLayout layout = context.layout(1, 0);
                {
                    std::lock_guard lk(sync_mutex);
                    seen_during_present = layout;
                    read_done = true;
                }
                sync_cv.notify_all();
            });
            std::unique_lock lk(sync_mutex);
            const bool started = sync_cv.wait_for(lk, std::chrono::seconds(2), [&] { return reader_started; });
            unblocked_during_present =
                started && sync_cv.wait_for(lk, std::chrono::milliseconds(500), [&] { return read_done; });
        };
        const VkResult present_rc = presenter.present(1, true);
        if (reader.joinable()) reader.join();
        EXPECT(present_rc == VK_SUCCESS);
        EXPECT(unblocked_during_present);
        EXPECT(seen_during_present == VK_IMAGE_LAYOUT_GENERAL);
        EXPECT(context.layout(1, 0) == VK_IMAGE_LAYOUT_GENERAL);
        for (ResourceId extra = 1000; extra < 1512; ++extra) context.remove_resource(extra);
        std::vector<std::uint8_t> pixels{0x7f};
        VkExtent2D ext{1, 2};
        VkFormat fmt = VK_FORMAT_R8_UNORM;
        driver::fail_next_wait = VK_ERROR_DEVICE_LOST;
        EXPECT(!presenter.read_back(pixels, &ext, &fmt));
        EXPECT(pixels.size() == 1 && pixels[0] == 0x7f);
        EXPECT(ext.width == 1 && ext.height == 2 && fmt == VK_FORMAT_R8_UNORM);
        EXPECT(presenter.read_back(pixels, &ext, &fmt));
        EXPECT(pixels.size() == std::size_t(ext.width) * ext.height * 4);
        const std::size_t mid = (std::size_t(ext.height / 2) * ext.width + ext.width / 2) * 4;
        EXPECT(pixels[mid + 0] == 0 && pixels[mid + 1] == 0 && pixels[mid + 2] == 0);
        std::printf("presenter sync: failed-submit LayoutTracker rollback ok\n");

        // 2. Capture-buffer WAW barrier before vkCmdCopyImageToBuffer on both present() and present_blank().
        {
            CommandList clear_list;
            EXPECT(clear_list.transition({1, 0, d3d12_state::common, d3d12_state::render_target}));
            EXPECT(clear_list.clear_render_target({1, 0, 0, 1}, {0.2f, 0.6f, 0.9f, 1.0f}));
            EXPECT(clear_list.transition({1, 0, d3d12_state::render_target, d3d12_state::common}));
            EXPECT(clear_list.close());
            const CommandList* ptr = &clear_list;
            EXPECT(queue.execute({&ptr, 1}) == ReplayResult::success);
        }
        driver::saw_capture_waw.store(false);
        driver::capture_copies.store(0);
        driver::capture_waw_barriers.store(0);
        driver::observe_capture.store(true);
        EXPECT(presenter.present(1, true) == VK_SUCCESS);
        EXPECT(presenter.present(1, true) == VK_SUCCESS);
        EXPECT(presenter.present_blank(true) == VK_SUCCESS);
        EXPECT(presenter.present(1, true) == VK_SUCCESS);
        driver::observe_capture.store(false);
        EXPECT(driver::capture_copies.load() == 4);
        EXPECT(driver::capture_waw_barriers.load() == 4);

        EXPECT(presenter.read_back(pixels, &ext, &fmt));
        const bool bgra = (fmt == VK_FORMAT_B8G8R8A8_UNORM || fmt == VK_FORMAT_B8G8R8A8_SRGB);
        EXPECT(presenter_capture_matches_clear(pixels, ext.width, ext.height, bgra));
        std::printf("presenter sync: capture-buffer WAW barriers (4/4) and full-frame read_back ok\n");

        // 2b. A Present behind held work shows what that work draws (queue_holds.h). The render waits for a gate with
        //     no submitted Signal, so the queue holds it; the presents are queued behind it instead of blitting the
        //     back buffers as they were. Three presents queue up and keep their order: a blit, a blank one (reads
        //     nothing) and a composed one (present_planes), whose frame is the last one shown, so a reordering
        //     would end on the black frame.
        {
            VulkanFence gate(context, 0);
            EXPECT(gate.semaphore() != VK_NULL_HANDLE);
            const VkImage plane_img = d.image(VK_FORMAT_R8G8B8A8_UNORM, 32, 32, 1,
                                              VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
            EXPECT(plane_img != VK_NULL_HANDLE);
            ResourceBinding plane_rb = side_rb;
            plane_rb.image = plane_img;
            plane_rb.sampled = true;
            context.add_resource(2, plane_rb, VK_IMAGE_LAYOUT_UNDEFINED);
            CommandList red;
            EXPECT(red.transition({1, 0, d3d12_state::common, d3d12_state::render_target}));
            EXPECT(red.clear_render_target({1, 0, 0, 1}, {0.9f, 0.1f, 0.2f, 1.0f}));
            EXPECT(red.transition({1, 0, d3d12_state::render_target, d3d12_state::common}));
            EXPECT(red.transition({2, 0, d3d12_state::common, d3d12_state::render_target}));
            EXPECT(red.clear_render_target({2, 0, 0, 1}, {0.2f, 0.8f, 0.4f, 1.0f}));
            EXPECT(red.transition({2, 0, d3d12_state::render_target, d3d12_state::pixel_shader_resource}));
            EXPECT(red.close());
            const CommandList* ptr = &red;
            queue.wait(gate.semaphore(), 1);
            EXPECT(queue.execute({&ptr, 1}) == ReplayResult::success);   // held
            const auto capture_matches = [&](const std::array<int, 3>& rgb, int tolerance) {
                std::vector<std::uint8_t> shot;
                VkExtent2D e{};
                VkFormat f{};
                if (!presenter.read_back(shot, &e, &f)) return false;
                const VkExtent2D current = presenter.extent();
                if (e.width != current.width || e.height != current.height) return false;
                const bool bgr = (f == VK_FORMAT_B8G8R8A8_UNORM || f == VK_FORMAT_B8G8R8A8_SRGB);
                return presenter_capture_matches_rgb(shot, e.width, e.height, bgr, rgb, tolerance);
            };
            const std::uint64_t shown = presenter.presented();
            PresentPlanes planes;
            planes.count = 1;
            planes.plane[0].source = 2;
            EXPECT(presenter.present(1, true) == VK_SUCCESS);            // queued: returns at once
            EXPECT(presenter.present_blank(true) == VK_SUCCESS);         // queued behind the one before it
            EXPECT(presenter.present_planes(planes, true) == VK_SUCCESS);
            EXPECT(presenter.presented() == shown && !presenter.last_present_tore());
            EXPECT(capture_matches({51, 153, 230}, 2));
            EXPECT(gate.signal(1));   // releases the render, then the presents, in order, on this thread
            EXPECT(queue.idle());
            EXPECT(presenter.presented() == shown + 3 && presenter.composed_presents() >= 1);
            // The composed frame (0.2, 0.8, 0.4 in the plane) is the last one shown; a blank last would be black.
            EXPECT(capture_matches({51, 204, 102}, 12));
            // The blit alone, with nothing held, is the plain path again and shows the red the render drew.
            EXPECT(presenter.present(1, true) == VK_SUCCESS);
            EXPECT(capture_matches({230, 26, 51}, 2));
            std::printf("presenter sync: three Presents queued behind held work ran in order\n");
            context.remove_resource(2);
        }

        // 2c. A queue destroyed with held work lets the Presents queued behind it run (the work is dropped).
        {
            VulkanFence never(context, 0);
            EXPECT(never.semaphore() != VK_NULL_HANDLE);
            CommandList noop;
            EXPECT(noop.transition({1, 0, d3d12_state::common, d3d12_state::render_target}));
            EXPECT(noop.clear_render_target({1, 0, 0, 1}, {0.0f, 0.0f, 0.0f, 1.0f}));  // never replayed
            EXPECT(noop.transition({1, 0, d3d12_state::render_target, d3d12_state::common}));
            EXPECT(noop.close());
            const CommandList* ptr = &noop;
            const std::uint64_t shown = presenter.presented();
            {
                VulkanQueue doomed(context, d.queue, d.family);
                EXPECT(doomed.ok());
                doomed.wait(never.semaphore(), 1);
                EXPECT(doomed.execute({&ptr, 1}) == ReplayResult::success);   // held for good
                EXPECT(presenter.present(1, true) == VK_SUCCESS);             // queued behind it
                EXPECT(presenter.presented() == shown);
            }
            EXPECT(presenter.presented() == shown + 1);
            std::printf("presenter sync: Present queued behind a destroyed queue's held work ran\n");
        }

        // 3. Concurrent present() + VulkanQueue::execute() + add/remove_resource() hammering LayoutTracker.
        std::atomic<bool> stop{false};
        std::atomic<bool> worker_failed{false};
        // The worker is paced: VulkanQueue::execute() has no back-pressure (it allocates a new command buffer whenever
        // none has finished), so an unthrottled loop queues thousands of submissions on the shared VkQueue and
        // present()'s frame fence is signalled only after all of them drain, while the worker keeps contending for
        // submit_mutex_ and tracker_mutex_ at full CPU. A slow GPU (an iGPU, or several processes sharing one) then
        // leaves present() stuck for tens of seconds. Draining every few rounds keeps the interleaving, not the backlog.
        constexpr std::uint64_t drain_every = 8;
        std::thread worker([&] {
            std::uint64_t round = 0;
            while (!stop.load(std::memory_order_relaxed)) {
                const ResourceId base = 100 + (round % 8) * 32;
                for (ResourceId i = 0; i < 32; ++i)
                    context.add_resource(base + i, side_rb, VK_IMAGE_LAYOUT_UNDEFINED);
                CommandList list;
                if (!list.clear_render_target({base, 0, 0, 1}, {0.1f, 0.2f, 0.3f, 1.0f}) || !list.close()) {
                    worker_failed.store(true);
                    break;
                }
                const CommandList* ptr = &list;
                if (worker_queue.execute({&ptr, 1}) != ReplayResult::success) {
                    worker_failed.store(true);
                    break;
                }
                (void)context.layout(base, 0);
                for (ResourceId i = 0; i < 32; ++i) context.remove_resource(base + i);
                if (++round % drain_every == 0 && !worker_queue.idle()) {
                    worker_failed.store(true);
                    break;
                }
            }
        });

        VkResult present_err = VK_SUCCESS;
        for (int i = 0; i < 40; ++i) {
            pump();
            present_err = presenter.present(1, false);
            if (present_err != VK_SUCCESS) break;
        }
        stop.store(true, std::memory_order_relaxed);
        worker.join();
        EXPECT(present_err == VK_SUCCESS);
        EXPECT(!worker_failed.load());
        EXPECT(queue.idle() && worker_queue.idle());
        std::printf("presenter sync: concurrent present + queue.execute + tracker mutations ok\n");

        // 4. Emulate a WSI capture layer that performs device-idle inside vkDestroySwapchainKHR.
        // The presenter first enters the destroy hook; a second thread then attempts a real queue submission.
        // Once the hook is released, that submission must not have entered the Vulkan driver while the hook ran.
        driver::destroy_entered.store(false);
        driver::submit_entered.store(false);
        driver::overlap_detected.store(false);
        driver::unexpected_submit_error.store(false);
        {
            std::lock_guard lock(driver::destroy_mutex);
            driver::allow_destroy_finish = false;
        }
        driver::check_swapchain_destroy.store(true);
        VkResult recreated = VK_ERROR_UNKNOWN;
        std::thread destroy_thread([&] { recreated = presenter.present(1, true); });
        bool entered = false;
        {
            std::unique_lock lock(driver::destroy_mutex);
            entered = driver::destroy_cv.wait_for(lock, std::chrono::seconds(5),
                                                  [] { return driver::destroy_entered.load(); });
        }
        std::atomic<bool> submit_started{false};
        ReplayResult submit_result = ReplayResult::unsupported;
        std::thread submit_thread;
        if (entered) {
            submit_thread = std::thread([&] {
                submit_started.store(true);
                CommandList list;
                if (!list.clear_render_target({1, 0, 0, 1}, {0.3f, 0.2f, 0.1f, 1.0f}) || !list.close()) return;
                const CommandList* ptr = &list;
                submit_result = worker_queue.execute({&ptr, 1});
            });
            for (int attempt = 0; attempt < 100 && !submit_started.load(); ++attempt)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            // Give the competing thread time to reach the submit point if no mutex is protecting destruction.
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        {
            std::lock_guard lock(driver::destroy_mutex);
            driver::allow_destroy_finish = true;
        }
        driver::destroy_cv.notify_all();
        destroy_thread.join();
        if (submit_thread.joinable()) submit_thread.join();
        driver::check_swapchain_destroy.store(false);
        EXPECT(entered);
        EXPECT(recreated == VK_SUCCESS);
        EXPECT(submit_started.load());
        EXPECT(submit_result == ReplayResult::success);
        EXPECT(!driver::overlap_detected.load());
        EXPECT(!driver::unexpected_submit_error.load());
        EXPECT(queue.idle() && worker_queue.idle());
        std::printf("presenter sync: synthetic WSI device-idle during swapchain destruction serialized ok\n");
    }

    DestroyWindow(window);
    UnregisterClassW(wc.lpszClassName, wc.hInstance);

    EXPECT(d.validation_errors() == 0);

    // 4. PresentComposer (runtime/src/vulkan_present_compose.h) MockVk regressions:
    //    missing function pointers, null device, partial-init cleanup on ready()/ensure_target() errors,
    //    failed driver output handles, zero extents, memoryTypeCount clamp, record() guards, and dispatches.
    {
        struct ComposeMock {
            std::uint64_t next_handle = 1;
            unsigned create_set_layout_calls = 0;
            unsigned created_set_layouts = 0;
            unsigned destroyed_set_layouts = 0;
            unsigned created_pools = 0;
            unsigned destroyed_pools = 0;
            unsigned created_pipeline_layouts = 0;
            unsigned destroyed_pipeline_layouts = 0;
            unsigned created_modules = 0;
            unsigned destroyed_modules = 0;
            unsigned create_pipeline_calls = 0;
            unsigned created_pipelines = 0;
            unsigned destroyed_pipelines = 0;
            unsigned created_images = 0;
            unsigned destroyed_images = 0;
            unsigned allocated_memories = 0;
            unsigned destroyed_memories = 0;
            unsigned created_views = 0;
            unsigned destroyed_views = 0;
            unsigned update_sets_calls = 0;
            unsigned barrier_calls = 0;
            unsigned dispatch_calls = 0;
            std::uint32_t last_dispatch[3]{};

            bool fail_create_set_layout = false;
            bool fail_create_pool = false;
            bool fail_alloc_sets = false;
            bool fail_create_pipe_layout = false;
            bool fail_create_module = false;
            bool fail_create_pipeline = false;
            bool null_pipeline_on_success = false;
            bool fail_create_image = false;
            bool fail_alloc_mem = false;
            bool fail_bind_mem = false;
            bool fail_create_view = false;
            std::uint32_t mem_type_bits = 1u;

            VkPipeline garbage_pipeline = reinterpret_cast<VkPipeline>(std::uintptr_t(0xdead));
            VkDeviceMemory garbage_memory = reinterpret_cast<VkDeviceMemory>(std::uintptr_t(0xbeef));
            VkImageView garbage_view = reinterpret_cast<VkImageView>(std::uintptr_t(0xcafe));
            bool destroyed_garbage_pipeline = false;
            bool destroyed_garbage_memory = false;
            bool destroyed_garbage_view = false;

            static ComposeMock*& state() {
                static ComposeMock* s = nullptr;
                return s;
            }
            static VKAPI_ATTR VkResult VKAPI_CALL create_set_layout(VkDevice, const VkDescriptorSetLayoutCreateInfo*,
                                                                    const VkAllocationCallbacks*,
                                                                    VkDescriptorSetLayout* out) {
                ++state()->create_set_layout_calls;
                if (state()->fail_create_set_layout) return VK_ERROR_OUT_OF_HOST_MEMORY;
                *out = reinterpret_cast<VkDescriptorSetLayout>(state()->next_handle++);
                ++state()->created_set_layouts;
                return VK_SUCCESS;
            }
            static VKAPI_ATTR void VKAPI_CALL destroy_set_layout(VkDevice, VkDescriptorSetLayout l,
                                                                 const VkAllocationCallbacks*) {
                if (l) ++state()->destroyed_set_layouts;
            }
            static VKAPI_ATTR VkResult VKAPI_CALL create_pool(VkDevice, const VkDescriptorPoolCreateInfo*,
                                                              const VkAllocationCallbacks*, VkDescriptorPool* out) {
                if (state()->fail_create_pool) return VK_ERROR_OUT_OF_HOST_MEMORY;
                *out = reinterpret_cast<VkDescriptorPool>(state()->next_handle++);
                ++state()->created_pools;
                return VK_SUCCESS;
            }
            static VKAPI_ATTR void VKAPI_CALL destroy_pool(VkDevice, VkDescriptorPool p,
                                                           const VkAllocationCallbacks*) {
                if (p) ++state()->destroyed_pools;
            }
            static VKAPI_ATTR VkResult VKAPI_CALL alloc_sets(VkDevice, const VkDescriptorSetAllocateInfo* info,
                                                             VkDescriptorSet* out) {
                if (state()->fail_alloc_sets) return VK_ERROR_OUT_OF_POOL_MEMORY;
                for (std::uint32_t i = 0; i < info->descriptorSetCount; ++i)
                    out[i] = reinterpret_cast<VkDescriptorSet>(state()->next_handle++);
                return VK_SUCCESS;
            }
            static VKAPI_ATTR void VKAPI_CALL update_sets(VkDevice, std::uint32_t, const VkWriteDescriptorSet*,
                                                          std::uint32_t, const VkCopyDescriptorSet*) {
                ++state()->update_sets_calls;
            }
            static VKAPI_ATTR VkResult VKAPI_CALL create_pipe_layout(VkDevice, const VkPipelineLayoutCreateInfo*,
                                                                     const VkAllocationCallbacks*,
                                                                     VkPipelineLayout* out) {
                if (state()->fail_create_pipe_layout) return VK_ERROR_OUT_OF_HOST_MEMORY;
                *out = reinterpret_cast<VkPipelineLayout>(state()->next_handle++);
                ++state()->created_pipeline_layouts;
                return VK_SUCCESS;
            }
            static VKAPI_ATTR void VKAPI_CALL destroy_pipe_layout(VkDevice, VkPipelineLayout l,
                                                                  const VkAllocationCallbacks*) {
                if (l) ++state()->destroyed_pipeline_layouts;
            }
            static VKAPI_ATTR VkResult VKAPI_CALL create_module(VkDevice, const VkShaderModuleCreateInfo*,
                                                                const VkAllocationCallbacks*, VkShaderModule* out) {
                if (state()->fail_create_module) return VK_ERROR_OUT_OF_HOST_MEMORY;
                *out = reinterpret_cast<VkShaderModule>(state()->next_handle++);
                ++state()->created_modules;
                return VK_SUCCESS;
            }
            static VKAPI_ATTR void VKAPI_CALL destroy_module(VkDevice, VkShaderModule m,
                                                             const VkAllocationCallbacks*) {
                if (m) ++state()->destroyed_modules;
            }
            static VKAPI_ATTR VkResult VKAPI_CALL create_pipelines(VkDevice, VkPipelineCache, std::uint32_t count,
                                                                   const VkComputePipelineCreateInfo*,
                                                                   const VkAllocationCallbacks*, VkPipeline* out) {
                ++state()->create_pipeline_calls;
                if (state()->fail_create_pipeline) {
                    if (count > 0 && out) out[0] = state()->garbage_pipeline;
                    return VK_ERROR_OUT_OF_HOST_MEMORY;
                }
                if (state()->null_pipeline_on_success) {
                    if (count > 0 && out) out[0] = VK_NULL_HANDLE;
                    return VK_SUCCESS;
                }
                for (std::uint32_t i = 0; i < count; ++i)
                    out[i] = reinterpret_cast<VkPipeline>(state()->next_handle++);
                state()->created_pipelines += count;
                return VK_SUCCESS;
            }
            static VKAPI_ATTR void VKAPI_CALL destroy_pipeline(VkDevice, VkPipeline p,
                                                               const VkAllocationCallbacks*) {
                if (p == state()->garbage_pipeline) state()->destroyed_garbage_pipeline = true;
                if (p) ++state()->destroyed_pipelines;
            }
            static VKAPI_ATTR void VKAPI_CALL cmd_bind_pipe(VkCommandBuffer, VkPipelineBindPoint, VkPipeline) {}
            static VKAPI_ATTR void VKAPI_CALL cmd_bind_sets(VkCommandBuffer, VkPipelineBindPoint, VkPipelineLayout,
                                                            std::uint32_t, std::uint32_t, const VkDescriptorSet*,
                                                            std::uint32_t, const std::uint32_t*) {}
            static VKAPI_ATTR void VKAPI_CALL cmd_push(VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags,
                                                       std::uint32_t, std::uint32_t, const void*) {}
            static VKAPI_ATTR void VKAPI_CALL cmd_dispatch(VkCommandBuffer, std::uint32_t x, std::uint32_t y,
                                                           std::uint32_t z) {
                ++state()->dispatch_calls;
                state()->last_dispatch[0] = x;
                state()->last_dispatch[1] = y;
                state()->last_dispatch[2] = z;
            }
            static VKAPI_ATTR void VKAPI_CALL cmd_barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags,
                                                          VkDependencyFlags, std::uint32_t, const VkMemoryBarrier*,
                                                          std::uint32_t, const VkBufferMemoryBarrier*, std::uint32_t,
                                                          const VkImageMemoryBarrier*) {
                ++state()->barrier_calls;
            }
            static VKAPI_ATTR VkResult VKAPI_CALL create_image(VkDevice, const VkImageCreateInfo*,
                                                               const VkAllocationCallbacks*, VkImage* out) {
                if (state()->fail_create_image) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
                *out = reinterpret_cast<VkImage>(state()->next_handle++);
                ++state()->created_images;
                return VK_SUCCESS;
            }
            static VKAPI_ATTR void VKAPI_CALL destroy_image(VkDevice, VkImage img, const VkAllocationCallbacks*) {
                if (img) ++state()->destroyed_images;
            }
            static VKAPI_ATTR void VKAPI_CALL img_mem_req(VkDevice, VkImage, VkMemoryRequirements* req) {
                *req = {};
                req->size = 4096;
                req->alignment = 256;
                req->memoryTypeBits = state()->mem_type_bits;
            }
            static VKAPI_ATTR VkResult VKAPI_CALL alloc_mem(VkDevice, const VkMemoryAllocateInfo*,
                                                            const VkAllocationCallbacks*, VkDeviceMemory* out) {
                if (state()->fail_alloc_mem) {
                    if (out) *out = state()->garbage_memory;
                    return VK_ERROR_OUT_OF_DEVICE_MEMORY;
                }
                *out = reinterpret_cast<VkDeviceMemory>(state()->next_handle++);
                ++state()->allocated_memories;
                return VK_SUCCESS;
            }
            static VKAPI_ATTR void VKAPI_CALL free_mem(VkDevice, VkDeviceMemory mem, const VkAllocationCallbacks*) {
                if (mem == state()->garbage_memory) state()->destroyed_garbage_memory = true;
                if (mem) ++state()->destroyed_memories;
            }
            static VKAPI_ATTR VkResult VKAPI_CALL bind_img_mem(VkDevice, VkImage, VkDeviceMemory, VkDeviceSize) {
                return state()->fail_bind_mem ? VK_ERROR_OUT_OF_DEVICE_MEMORY : VK_SUCCESS;
            }
            static VKAPI_ATTR VkResult VKAPI_CALL create_view(VkDevice, const VkImageViewCreateInfo*,
                                                              const VkAllocationCallbacks*, VkImageView* out) {
                if (state()->fail_create_view) {
                    if (out) *out = state()->garbage_view;
                    return VK_ERROR_OUT_OF_HOST_MEMORY;
                }
                *out = reinterpret_cast<VkImageView>(state()->next_handle++);
                ++state()->created_views;
                return VK_SUCCESS;
            }
            static VKAPI_ATTR void VKAPI_CALL destroy_view(VkDevice, VkImageView v, const VkAllocationCallbacks*) {
                if (v == state()->garbage_view) state()->destroyed_garbage_view = true;
                if (v) ++state()->destroyed_views;
            }
            static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL gdpa(VkDevice, const char* name) {
                const std::string_view n = name;
                if (n == "vkCreateDescriptorSetLayout") return reinterpret_cast<PFN_vkVoidFunction>(create_set_layout);
                if (n == "vkDestroyDescriptorSetLayout") return reinterpret_cast<PFN_vkVoidFunction>(destroy_set_layout);
                if (n == "vkCreateDescriptorPool") return reinterpret_cast<PFN_vkVoidFunction>(create_pool);
                if (n == "vkDestroyDescriptorPool") return reinterpret_cast<PFN_vkVoidFunction>(destroy_pool);
                if (n == "vkAllocateDescriptorSets") return reinterpret_cast<PFN_vkVoidFunction>(alloc_sets);
                if (n == "vkUpdateDescriptorSets") return reinterpret_cast<PFN_vkVoidFunction>(update_sets);
                if (n == "vkCreatePipelineLayout") return reinterpret_cast<PFN_vkVoidFunction>(create_pipe_layout);
                if (n == "vkDestroyPipelineLayout") return reinterpret_cast<PFN_vkVoidFunction>(destroy_pipe_layout);
                if (n == "vkCreateShaderModule") return reinterpret_cast<PFN_vkVoidFunction>(create_module);
                if (n == "vkDestroyShaderModule") return reinterpret_cast<PFN_vkVoidFunction>(destroy_module);
                if (n == "vkCreateComputePipelines") return reinterpret_cast<PFN_vkVoidFunction>(create_pipelines);
                if (n == "vkDestroyPipeline") return reinterpret_cast<PFN_vkVoidFunction>(destroy_pipeline);
                if (n == "vkCmdBindPipeline") return reinterpret_cast<PFN_vkVoidFunction>(cmd_bind_pipe);
                if (n == "vkCmdBindDescriptorSets") return reinterpret_cast<PFN_vkVoidFunction>(cmd_bind_sets);
                if (n == "vkCmdPushConstants") return reinterpret_cast<PFN_vkVoidFunction>(cmd_push);
                if (n == "vkCmdDispatch") return reinterpret_cast<PFN_vkVoidFunction>(cmd_dispatch);
                if (n == "vkCmdPipelineBarrier") return reinterpret_cast<PFN_vkVoidFunction>(cmd_barrier);
                if (n == "vkCreateImage") return reinterpret_cast<PFN_vkVoidFunction>(create_image);
                if (n == "vkDestroyImage") return reinterpret_cast<PFN_vkVoidFunction>(destroy_image);
                if (n == "vkGetImageMemoryRequirements") return reinterpret_cast<PFN_vkVoidFunction>(img_mem_req);
                if (n == "vkAllocateMemory") return reinterpret_cast<PFN_vkVoidFunction>(alloc_mem);
                if (n == "vkFreeMemory") return reinterpret_cast<PFN_vkVoidFunction>(free_mem);
                if (n == "vkBindImageMemory") return reinterpret_cast<PFN_vkVoidFunction>(bind_img_mem);
                if (n == "vkCreateImageView") return reinterpret_cast<PFN_vkVoidFunction>(create_view);
                if (n == "vkDestroyImageView") return reinterpret_cast<PFN_vkVoidFunction>(destroy_view);
                return nullptr;
            }
        };

        const VkDevice fake_dev = reinterpret_cast<VkDevice>(std::uintptr_t(1));
        const VkCommandBuffer fake_cb = reinterpret_cast<VkCommandBuffer>(std::uintptr_t(0x400));
        const VkImageView fake_plane = reinterpret_cast<VkImageView>(std::uintptr_t(0x500));
        VkPhysicalDeviceMemoryProperties mem{};
        mem.memoryTypeCount = 1;
        mem.memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

        // 4a: missing any of the 11 previously unchecked function pointers or null device must fail ready().
        {
            ComposeMock mock{};
            ComposeMock::state() = &mock;
            {
                VulkanContext ctx(&ComposeMock::gdpa, fake_dev, false);
                ctx.vk.vkGetImageMemoryRequirements = nullptr;
                PresentComposer c(ctx, mem);
                EXPECT(!c.ready());
                EXPECT(mock.create_set_layout_calls == 0);
            }
            for (int which = 0; which < 11; ++which) {
                VulkanContext ctx(&ComposeMock::gdpa, fake_dev, false);
                switch (which) {
                    case 0: ctx.vk.vkGetImageMemoryRequirements = nullptr; break;
                    case 1: ctx.vk.vkBindImageMemory = nullptr; break;
                    case 2: ctx.vk.vkCmdPipelineBarrier = nullptr; break;
                    case 3: ctx.vk.vkDestroyImageView = nullptr; break;
                    case 4: ctx.vk.vkDestroyImage = nullptr; break;
                    case 5: ctx.vk.vkFreeMemory = nullptr; break;
                    case 6: ctx.vk.vkDestroyPipeline = nullptr; break;
                    case 7: ctx.vk.vkDestroyPipelineLayout = nullptr; break;
                    case 8: ctx.vk.vkDestroyDescriptorPool = nullptr; break;
                    case 9: ctx.vk.vkDestroyDescriptorSetLayout = nullptr; break;
                    case 10: ctx.vk.vkDestroyShaderModule = nullptr; break;
                }
                PresentComposer c(ctx, mem);
                EXPECT(!c.ready());
                bool changed = true;
                EXPECT(!c.ensure_target({64, 64}, changed) && !changed);
                EXPECT(mock.create_set_layout_calls == 0 && mock.created_images == 0);
            }
            {
                VulkanContext null_ctx(&ComposeMock::gdpa, VK_NULL_HANDLE, false);
                PresentComposer c(null_ctx, mem);
                EXPECT(!c.ready());
                EXPECT(mock.create_set_layout_calls == 0);
            }
            ComposeMock::state() = nullptr;
        }

        // 4b: partial-init cleanup on ready() failure + failed vkCreateComputePipelines garbage handle guard.
        {
            ComposeMock mock{};
            ComposeMock::state() = &mock;
            VulkanContext ctx(&ComposeMock::gdpa, fake_dev, false);
            for (int mode = 0; mode < 6; ++mode) {
                mock.fail_create_pool = (mode == 0);
                mock.fail_alloc_sets = (mode == 1);
                mock.fail_create_pipe_layout = (mode == 2);
                mock.fail_create_module = (mode == 3);
                mock.fail_create_pipeline = (mode == 4);
                mock.null_pipeline_on_success = (mode == 5);
                {
                    PresentComposer c(ctx, mem);
                    EXPECT(!c.ready());
                    EXPECT(!c.ready());  // stays State::failed without re-creating
                    // Cleanup must happen immediately inside ready(), not wait for ~PresentComposer().
                    EXPECT(mock.destroyed_set_layouts == mock.created_set_layouts);
                    EXPECT(mock.destroyed_pools == mock.created_pools);
                    EXPECT(mock.destroyed_pipeline_layouts == mock.created_pipeline_layouts);
                    EXPECT(mock.destroyed_modules == mock.created_modules);
                    EXPECT(mock.created_pipelines == 0 && !mock.destroyed_garbage_pipeline);
                }
                EXPECT(!mock.destroyed_garbage_pipeline);
            }
            ComposeMock::state() = nullptr;
        }

        // 4c & 4d: ensure_target() guards, memoryTypeCount clamp, failure cleanup, and record() dispatches.
        {
            ComposeMock mock{};
            ComposeMock::state() = &mock;
            {
                VulkanContext ctx(&ComposeMock::gdpa, fake_dev, false);
                PresentComposer c(ctx, mem);

                // Untried composer cannot ensure_target or record.
                bool changed = true;
                EXPECT(!c.ensure_target({64, 64}, changed) && !changed && mock.created_images == 0);
                c.record(fake_cb, 0, fake_plane, VK_IMAGE_LAYOUT_GENERAL, fake_plane, VK_IMAGE_LAYOUT_GENERAL,
                         PresentComposePush{});
                EXPECT(mock.update_sets_calls == 0 && mock.barrier_calls == 0 && mock.dispatch_calls == 0);

                EXPECT(c.ready());
                // Ready composer with no target yet cannot record.
                c.record(fake_cb, 0, fake_plane, VK_IMAGE_LAYOUT_GENERAL, fake_plane, VK_IMAGE_LAYOUT_GENERAL,
                         PresentComposePush{});
                EXPECT(mock.update_sets_calls == 0 && mock.barrier_calls == 0 && mock.dispatch_calls == 0);

                // Active 64x64 target; zero-dimension ensure_target() calls must fail cleanly and keep it.
                EXPECT(c.ensure_target({64, 64}, changed) && changed);
                const unsigned imgs_before = mock.created_images;
                EXPECT(!c.ensure_target({0, 64}, changed) && !changed);
                EXPECT(!c.ensure_target({64, 0}, changed) && !changed);
                EXPECT(!c.ensure_target({0, 0}, changed) && !changed);
                EXPECT(mock.created_images == imgs_before && c.target() != VK_NULL_HANDLE &&
                       c.extent().width == 64 && c.extent().height == 64);

                // Calling ensure_target with the same extent does not recreate.
                EXPECT(c.ensure_target({64, 64}, changed) && !changed && mock.created_images == imgs_before);

                // Allocation failures release prior target cleanly and never free/destroy driver garbage handles.
                mock.fail_alloc_mem = true;
                EXPECT(!c.ensure_target({32, 32}, changed));
                EXPECT(c.target() == VK_NULL_HANDLE && mock.destroyed_images == mock.created_images &&
                       !mock.destroyed_garbage_memory);
                mock.fail_alloc_mem = false;

                mock.fail_bind_mem = true;
                EXPECT(!c.ensure_target({32, 32}, changed));
                EXPECT(c.target() == VK_NULL_HANDLE && mock.destroyed_images == mock.created_images &&
                       mock.destroyed_memories == mock.allocated_memories);
                mock.fail_bind_mem = false;

                mock.fail_create_view = true;
                EXPECT(!c.ensure_target({32, 32}, changed));
                EXPECT(c.target() == VK_NULL_HANDLE && mock.destroyed_images == mock.created_images &&
                       mock.destroyed_memories == mock.allocated_memories && !mock.destroyed_garbage_view);
                mock.fail_create_view = false;

                // memoryTypeCount > VK_MAX_MEMORY_TYPES (64) must clamp safely without OOB reads/UB.
                VkPhysicalDeviceMemoryProperties large_mem{};
                large_mem.memoryTypeCount = 64;
                large_mem.memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
                PresentComposer clamped(ctx, large_mem);
                EXPECT(clamped.ready());
                mock.mem_type_bits = 0;
                EXPECT(!clamped.ensure_target({16, 16}, changed));
                EXPECT(mock.destroyed_images == mock.created_images);
                mock.mem_type_bits = 1u;
                EXPECT(clamped.ensure_target({16, 16}, changed) && changed);

                // Recover original composer to 128x96 and verify record() guards & dispatch counts.
                EXPECT(c.ensure_target({128, 96}, changed) && changed);
                c.record(VK_NULL_HANDLE, 0, fake_plane, VK_IMAGE_LAYOUT_GENERAL, fake_plane,
                         VK_IMAGE_LAYOUT_GENERAL, PresentComposePush{});
                EXPECT(mock.update_sets_calls == 0 && mock.barrier_calls == 0 && mock.dispatch_calls == 0);

                c.record(fake_cb, 0, fake_plane, VK_IMAGE_LAYOUT_GENERAL, fake_plane, VK_IMAGE_LAYOUT_GENERAL,
                         PresentComposePush{});
                EXPECT(mock.update_sets_calls == 1 && mock.barrier_calls == 2 && mock.dispatch_calls == 1);
                EXPECT(mock.last_dispatch[0] == 16 && mock.last_dispatch[1] == 12 && mock.last_dispatch[2] == 1);

                EXPECT(c.ensure_target({13, 9}, changed) && changed);
                c.record(fake_cb, 1, fake_plane, VK_IMAGE_LAYOUT_GENERAL, fake_plane, VK_IMAGE_LAYOUT_GENERAL,
                         PresentComposePush{});
                EXPECT(mock.dispatch_calls == 2);
                EXPECT(mock.last_dispatch[0] == 2 && mock.last_dispatch[1] == 2 && mock.last_dispatch[2] == 1);
            }
            EXPECT(!mock.destroyed_garbage_pipeline && !mock.destroyed_garbage_memory && !mock.destroyed_garbage_view);
            EXPECT(mock.destroyed_set_layouts == mock.created_set_layouts);
            EXPECT(mock.destroyed_pools == mock.created_pools);
            EXPECT(mock.destroyed_pipeline_layouts == mock.created_pipeline_layouts);
            EXPECT(mock.destroyed_modules == mock.created_modules);
            EXPECT(mock.destroyed_pipelines == mock.created_pipelines);
            EXPECT(mock.destroyed_images == mock.created_images);
            EXPECT(mock.destroyed_memories == mock.allocated_memories);
            EXPECT(mock.destroyed_views == mock.created_views);
            ComposeMock::state() = nullptr;
        }
        std::printf("presenter sync: PresentComposer hardening ok\n");
    }

    std::printf("vk_presenter_sync_test: ok\n");
    return 0;
}
