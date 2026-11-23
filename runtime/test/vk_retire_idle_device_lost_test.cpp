// The device-wide retirement and VkQueueWaitIdle paths must notice VK_ERROR_DEVICE_LOST.
// Synthetic one-shot error injection, never delivered to the physical GPU driver.
#include "vk_test_device.h"
#include "../src/vulkan_backend.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <string_view>
#include <thread>

namespace {
namespace driver {
PFN_vkGetDeviceProcAddr real_device = nullptr;
PFN_vkGetSemaphoreCounterValue real_counter = nullptr;
PFN_vkQueueWaitIdle real_idle = nullptr;
std::thread::id caller;
std::atomic<VkResult> fail_counter{VK_SUCCESS};
std::atomic<VkResult> fail_idle{VK_SUCCESS};
std::atomic<unsigned> intercepted_counter{0};
std::atomic<unsigned> intercepted_idle{0};

VKAPI_ATTR VkResult VKAPI_CALL counter(VkDevice device, VkSemaphore semaphore, std::uint64_t* value) {
    if (std::this_thread::get_id() == caller) {
        const VkResult injected = fail_counter.exchange(VK_SUCCESS);
        if (injected != VK_SUCCESS) {
            *value = 999;  // unsuccessful query outputs are undefined
            ++intercepted_counter;
            return injected;
        }
    }
    return real_counter(device, semaphore, value);
}
VKAPI_ATTR VkResult VKAPI_CALL idle(VkQueue queue) {
    if (std::this_thread::get_id() == caller) {
        const VkResult injected = fail_idle.exchange(VK_SUCCESS);
        if (injected != VK_SUCCESS) {
            ++intercepted_idle;
            return injected;
        }
    }
    return real_idle(queue);
}
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_device(VkDevice device, const char* name) {
    const std::string_view n(name);
    if (n == "vkGetSemaphoreCounterValue") return reinterpret_cast<PFN_vkVoidFunction>(counter);
    if (n == "vkQueueWaitIdle") return reinterpret_cast<PFN_vkVoidFunction>(idle);
    return real_device(device, name);
}
}
}

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    vktest::Device d;
    if (!d.create(false)) return 77;
    driver::real_device = d.device_proc;
    driver::real_counter = reinterpret_cast<PFN_vkGetSemaphoreCounterValue>(
        d.device_proc(d.device, "vkGetSemaphoreCounterValue"));
    driver::real_idle = reinterpret_cast<PFN_vkQueueWaitIdle>(d.device_proc(d.device, "vkQueueWaitIdle"));
    if (!driver::real_counter || !driver::real_idle) return 1;
    driver::caller = std::this_thread::get_id();
    unsigned failed = 0;
    {
        leonida::VulkanContext context(driver::get_device, d.device, true);
        leonida::VulkanQueue queue(context, d.queue, d.family);
        if (!context.loaded() || !queue.ok()) return 1;
        leonida::CommandList empty;
        if (!empty.close()) return 1;
        const leonida::CommandList* list = &empty;
        if (queue.execute({&list, 1}) != leonida::ReplayResult::success) return 1;
        driver::fail_counter.store(VK_ERROR_OUT_OF_HOST_MEMORY);
        (void)context.completed_retire_ticket();
        const bool nonfatal_ok = !context.is_device_lost() && driver::intercepted_counter.load() == 1;
        // This query belongs to RetireTickets::completed, not VulkanQueue::completed.
        driver::fail_counter.store(VK_ERROR_DEVICE_LOST);
        (void)context.completed_retire_ticket();
        const bool lost = context.is_device_lost();
        const auto executed = queue.execute({&list, 1});
        std::printf("global retirement query: nonfatal_ok=%d, context_lost=%d, later_execute=%d, interceptions=%u\n",
                    int(nonfatal_ok), int(lost), int(executed), driver::intercepted_counter.load());
        if (!nonfatal_ok || !lost || executed != leonida::ReplayResult::unsupported ||
            driver::intercepted_counter.load() != 2) ++failed;
    }
    {
        leonida::VulkanContext context(driver::get_device, d.device, true);
        leonida::VulkanQueue queue(context, d.queue, d.family);
        if (!context.loaded() || !queue.ok()) return 1;
        driver::fail_idle.store(VK_ERROR_OUT_OF_HOST_MEMORY);
        const bool nonfatal_idle = queue.idle();
        const bool nonfatal_ok = !nonfatal_idle && !context.is_device_lost() &&
                                 driver::intercepted_idle.load() == 1;
        driver::fail_idle.store(VK_ERROR_DEVICE_LOST);
        const bool idle = queue.idle();
        const bool lost = context.is_device_lost();
        std::printf("queue idle query: nonfatal_ok=%d, idle=%d, context_lost=%d, interceptions=%u\n",
                    int(nonfatal_ok), int(idle), int(lost), driver::intercepted_idle.load());
        if (!nonfatal_ok || idle || !lost || driver::intercepted_idle.load() != 2) ++failed;
    }
    std::printf("Vulkan validation errors: %u\n", d.validation_errors());
    return failed || d.validation_errors() ? 1 : 0;
}
