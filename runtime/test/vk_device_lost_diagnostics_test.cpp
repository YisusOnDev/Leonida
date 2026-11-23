// Always-on device-lost host-cause and VK_EXT_device_fault diagnostics tests (runtime/src/vulkan_backend.h):
//   1. Originating host site and VkResult are captured and logged once on first device loss without duplicate spam.
//   2. FenceWaitDispatcher::run() records its originating site on vkWaitSemaphores and vkGetSemaphoreCounterValue loss.
//   3. VK_EXT_device_fault is queried once on VK_ERROR_DEVICE_LOST when enabled (with bounded counts, vendorBinarySize=0,
//      control-char sanitization, and VK_INCOMPLETE/error handling), skipped when disabled, and skipped for non-DEVICE_LOST.
//   4. Concurrent multi-threaded device-loss transitions log once, query fault info once, and preserve a consistent summary.
// Skips with exit code 77 when no Vulkan 1.3 device is available. No game data.
#include "vk_test_device.h"
#include "../src/vulkan_backend.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#else
#include <unistd.h>
#endif

#define EXPECT(condition) do { \
    if (!(condition)) { \
        std::printf("FAIL line %d: %s\n", __LINE__, #condition); \
        std::fflush(stdout); \
        std::_Exit(1); \
    } \
} while (0)

namespace {

// Captures stderr inside a sub-test so we can assert exact bounded log lines without leaking test noise.
class StderrCapture {
public:
    StderrCapture() {
        std::fflush(stderr);
        int fds[2]{-1, -1};
#ifdef _WIN32
        if (_pipe(fds, 65536, _O_BINARY) != 0) return;
        const int err_fd = _fileno(stderr);
        saved_fd_ = _dup(err_fd);
        if (saved_fd_ < 0 || _dup2(fds[1], err_fd) != 0) {
            if (saved_fd_ >= 0) _close(saved_fd_);
            saved_fd_ = -1;
            _close(fds[0]);
            _close(fds[1]);
            return;
        }
        _close(fds[1]);
#else
        if (pipe(fds) != 0) return;
        const int err_fd = fileno(stderr);
        saved_fd_ = dup(err_fd);
        if (saved_fd_ < 0 || dup2(fds[1], err_fd) < 0) {
            if (saved_fd_ >= 0) close(saved_fd_);
            saved_fd_ = -1;
            close(fds[0]);
            close(fds[1]);
            return;
        }
        close(fds[1]);
#endif
        read_fd_ = fds[0];
        reader_ = std::thread([this] {
            char buf[512];
            for (;;) {
#ifdef _WIN32
                const int n = _read(read_fd_, buf, sizeof(buf));
#else
                const ssize_t n = read(read_fd_, buf, sizeof(buf));
#endif
                if (n <= 0) break;
                captured_.append(buf, static_cast<std::size_t>(n));
            }
        });
    }

    ~StderrCapture() { finish(); }

    std::string stop() {
        finish();
        return captured_;
    }

    bool active() const { return read_fd_ >= 0; }

private:
    void finish() {
        if (saved_fd_ >= 0) {
            std::fflush(stderr);
#ifdef _WIN32
            _dup2(saved_fd_, _fileno(stderr));
            _close(saved_fd_);
#else
            dup2(saved_fd_, fileno(stderr));
            close(saved_fd_);
#endif
            saved_fd_ = -1;
        }
        if (reader_.joinable()) reader_.join();
        if (read_fd_ >= 0) {
#ifdef _WIN32
            _close(read_fd_);
#else
            close(read_fd_);
#endif
            read_fd_ = -1;
        }
    }

    int saved_fd_ = -1;
    int read_fd_ = -1;
    std::string captured_;
    std::thread reader_;
};

std::size_t count_occurrences(const std::string& haystack, std::string_view needle) {
    if (needle.empty()) return 0;
    std::size_t count = 0;
    std::size_t pos = 0;
    while ((pos = haystack.find(needle, pos)) != std::string::npos) {
        ++count;
        pos += needle.size();
    }
    return count;
}

namespace driver {

PFN_vkGetDeviceProcAddr real_get_device = nullptr;
PFN_vkQueueSubmit real_queue_submit = nullptr;
PFN_vkWaitSemaphores real_wait_semaphores = nullptr;
PFN_vkGetSemaphoreCounterValue real_get_counter = nullptr;

std::atomic<VkResult> next_submit_result{VK_SUCCESS};
std::atomic<VkResult> next_wait_result{VK_SUCCESS};
std::atomic<VkResult> dispatcher_wait_result{VK_SUCCESS};
std::atomic<bool> in_dispatcher_wait{false};
std::atomic<bool> fail_counter_lost{false};

enum class FaultMode {
    rich,
    oversized_incomplete,
    query_error,
};

std::mutex fault_mutex;
FaultMode fault_mode = FaultMode::rich;
std::atomic<std::uint32_t> fault_calls{0};
bool saw_zero_vendor_binary = false;
bool saw_null_vendor_binary_ptr = false;
std::uint32_t max_seen_addr_capacity = 0;
std::uint32_t max_seen_vendor_capacity = 0;

void reset_mocks() {
    next_submit_result.store(VK_SUCCESS, std::memory_order_release);
    next_wait_result.store(VK_SUCCESS, std::memory_order_release);
    dispatcher_wait_result.store(VK_SUCCESS, std::memory_order_release);
    in_dispatcher_wait.store(false, std::memory_order_release);
    fail_counter_lost.store(false, std::memory_order_release);
    std::lock_guard lock(fault_mutex);
    fault_mode = FaultMode::rich;
    fault_calls.store(0, std::memory_order_release);
    saw_zero_vendor_binary = false;
    saw_null_vendor_binary_ptr = false;
    max_seen_addr_capacity = 0;
    max_seen_vendor_capacity = 0;
}

VKAPI_ATTR VkResult VKAPI_CALL queue_submit(VkQueue queue, std::uint32_t count, const VkSubmitInfo* submits,
                                            VkFence fence) {
    const VkResult injected = next_submit_result.exchange(VK_SUCCESS, std::memory_order_acq_rel);
    if (injected != VK_SUCCESS) return injected;
    return real_queue_submit(queue, count, submits, fence);
}

VKAPI_ATTR VkResult VKAPI_CALL wait_semaphores(VkDevice device, const VkSemaphoreWaitInfo* info, std::uint64_t timeout) {
    const VkResult single = next_wait_result.exchange(VK_SUCCESS, std::memory_order_acq_rel);
    if (single != VK_SUCCESS) return single;
    if (info && info->semaphoreCount >= 2) {
        in_dispatcher_wait.store(true, std::memory_order_release);
        const VkResult disp = dispatcher_wait_result.exchange(VK_SUCCESS, std::memory_order_acq_rel);
        if (disp != VK_SUCCESS) {
            in_dispatcher_wait.store(false, std::memory_order_release);
            return disp;
        }
    }
    const VkResult res = real_wait_semaphores(device, info, timeout);
    if (info && info->semaphoreCount >= 2) in_dispatcher_wait.store(false, std::memory_order_release);
    return res;
}

VKAPI_ATTR VkResult VKAPI_CALL get_counter(VkDevice device, VkSemaphore sem, std::uint64_t* value) {
    if (fail_counter_lost.load(std::memory_order_acquire)) return VK_ERROR_DEVICE_LOST;
    return real_get_counter(device, sem, value);
}

#ifdef VK_EXT_DEVICE_FAULT_EXTENSION_NAME
VKAPI_ATTR VkResult VKAPI_CALL get_device_fault_info(VkDevice, VkDeviceFaultCountsEXT* pFaultCounts,
                                                     VkDeviceFaultInfoEXT* pFaultInfo) {
    fault_calls.fetch_add(1, std::memory_order_acq_rel);
    std::lock_guard lock(fault_mutex);
    if (!pFaultCounts) return VK_ERROR_INITIALIZATION_FAILED;
    if (fault_mode == FaultMode::query_error) return VK_ERROR_OUT_OF_HOST_MEMORY;

    if (!pFaultInfo) {
        if (fault_mode == FaultMode::rich) {
            pFaultCounts->addressInfoCount = 2;
            pFaultCounts->vendorInfoCount = 1;
            pFaultCounts->vendorBinarySize = 0;
        } else {
            pFaultCounts->addressInfoCount = 32;
            pFaultCounts->vendorInfoCount = 24;
            pFaultCounts->vendorBinarySize = 4096;
        }
        return VK_SUCCESS;
    }

    saw_zero_vendor_binary = (pFaultCounts->vendorBinarySize == 0);
    saw_null_vendor_binary_ptr = (pFaultInfo->pVendorBinaryData == nullptr);
    max_seen_addr_capacity = std::max(max_seen_addr_capacity, pFaultCounts->addressInfoCount);
    max_seen_vendor_capacity = std::max(max_seen_vendor_capacity, pFaultCounts->vendorInfoCount);

    if (fault_mode == FaultMode::rich) {
        std::snprintf(pFaultInfo->description, VK_MAX_DESCRIPTION_SIZE, "shader page fault\non\tread");
        const std::uint32_t n_addr = std::min<std::uint32_t>(pFaultCounts->addressInfoCount, 2u);
        pFaultCounts->addressInfoCount = n_addr;
        if (n_addr >= 1 && pFaultInfo->pAddressInfos) {
            pFaultInfo->pAddressInfos[0] = {VK_DEVICE_FAULT_ADDRESS_TYPE_READ_INVALID_EXT, 0x10002000ULL, 0x1000ULL};
        }
        if (n_addr >= 2 && pFaultInfo->pAddressInfos) {
            pFaultInfo->pAddressInfos[1] = {VK_DEVICE_FAULT_ADDRESS_TYPE_WRITE_INVALID_EXT, 0x20004000ULL, 0x2000ULL};
        }
        const std::uint32_t n_vendor = std::min<std::uint32_t>(pFaultCounts->vendorInfoCount, 1u);
        pFaultCounts->vendorInfoCount = n_vendor;
        if (n_vendor >= 1 && pFaultInfo->pVendorInfos) {
            std::snprintf(pFaultInfo->pVendorInfos[0].description, VK_MAX_DESCRIPTION_SIZE, "MMU fault\ndetail");
            pFaultInfo->pVendorInfos[0].vendorFaultCode = 0xdeadULL;
            pFaultInfo->pVendorInfos[0].vendorFaultData = 0xbeefULL;
        }
        return VK_SUCCESS;
    }

    std::snprintf(pFaultInfo->description, VK_MAX_DESCRIPTION_SIZE, "oversized fault table");
    const std::uint32_t n_addr = std::min<std::uint32_t>(pFaultCounts->addressInfoCount, 32u);
    pFaultCounts->addressInfoCount = n_addr;
    for (std::uint32_t i = 0; i < n_addr && pFaultInfo->pAddressInfos; ++i) {
        pFaultInfo->pAddressInfos[i] = {VK_DEVICE_FAULT_ADDRESS_TYPE_EXECUTE_INVALID_EXT, 0x50000000ULL + i * 0x1000ULL,
                                        0x1000ULL};
    }
    const std::uint32_t n_vendor = std::min<std::uint32_t>(pFaultCounts->vendorInfoCount, 24u);
    pFaultCounts->vendorInfoCount = n_vendor;
    for (std::uint32_t i = 0; i < n_vendor && pFaultInfo->pVendorInfos; ++i) {
        std::snprintf(pFaultInfo->pVendorInfos[i].description, VK_MAX_DESCRIPTION_SIZE, "vendor_%u", i);
        pFaultInfo->pVendorInfos[i].vendorFaultCode = 0x100ULL + i;
        pFaultInfo->pVendorInfos[i].vendorFaultData = 0x200ULL + i;
    }
    return VK_INCOMPLETE;
}
#endif

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_device(VkDevice device, const char* name) {
    const std::string_view n = name;
    if (n == "vkQueueSubmit") return reinterpret_cast<PFN_vkVoidFunction>(queue_submit);
    if (n == "vkWaitSemaphores") return reinterpret_cast<PFN_vkVoidFunction>(wait_semaphores);
    if (n == "vkGetSemaphoreCounterValue") return reinterpret_cast<PFN_vkVoidFunction>(get_counter);
#ifdef VK_EXT_DEVICE_FAULT_EXTENSION_NAME
    if (n == "vkGetDeviceFaultInfoEXT") return reinterpret_cast<PFN_vkVoidFunction>(get_device_fault_info);
#endif
    return real_get_device(device, name);
}

void install(PFN_vkGetDeviceProcAddr device_proc, VkDevice device) {
    real_get_device = device_proc;
    real_queue_submit = reinterpret_cast<PFN_vkQueueSubmit>(device_proc(device, "vkQueueSubmit"));
    real_wait_semaphores = reinterpret_cast<PFN_vkWaitSemaphores>(device_proc(device, "vkWaitSemaphores"));
    real_get_counter =
        reinterpret_cast<PFN_vkGetSemaphoreCounterValue>(device_proc(device, "vkGetSemaphoreCounterValue"));
}

}  // namespace driver

bool wait_until_dispatcher_in_wait(int timeout_ms = 2000) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (!driver::in_dispatcher_wait.load(std::memory_order_acquire)) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

}  // namespace

int main() {
    using namespace leonida;

    vktest::Device d;
    if (!d.create(false)) {
        std::fprintf(stderr, "vk_device_lost_diagnostics_test: no Vulkan 1.3 device, skipped\n");
        return 77;
    }
    driver::install(d.device_proc, d.device);

    // 1. Originating host site + VkResult on vkQueueSubmit loss; no duplicate spam or fault query when disabled.
    {
        driver::reset_mocks();
        std::string captured;
        {
            StderrCapture cap;
            EXPECT(cap.active());
            VulkanContext context(driver::get_device, d.device, true, false, false, false, false);
            EXPECT(context.loaded());
            EXPECT(!context.device_fault_enabled());
            VulkanQueue queue(context, d.queue, d.family);
            EXPECT(queue.ok());
            VulkanFence fence(context, 0);
            EXPECT(fence.semaphore() != VK_NULL_HANDLE);

            CommandList empty;
            EXPECT(empty.close());
            const CommandList* ptr = &empty;
            driver::next_submit_result.store(VK_ERROR_DEVICE_LOST, std::memory_order_release);
            EXPECT(queue.execute({&ptr, 1}) == ReplayResult::unsupported);
            EXPECT(context.is_device_lost());

            // Subsequent calls must not overwrite the first cause or emit duplicate device-lost logs.
            EXPECT(fence.completed() == UINT64_MAX);
            EXPECT(!fence.signal(1));
            EXPECT(queue.execute({&ptr, 1}) == ReplayResult::unsupported);
            context.mark_device_lost("follow_on_site", VK_ERROR_DEVICE_LOST);

            const DeviceLostDiagnostics diag = context.device_lost_diagnostics();
            EXPECT(diag.lost);
            EXPECT(diag.site && std::string_view(diag.site) == "VulkanQueue::submit (vkQueueSubmit)");
            EXPECT(diag.result == VK_ERROR_DEVICE_LOST);
            EXPECT(!diag.fault_queried);
            EXPECT(driver::fault_calls.load(std::memory_order_acquire) == 0);
            captured = cap.stop();
        }
        EXPECT(count_occurrences(captured, "[rt] d3d12: device lost (") == 1);
        EXPECT(captured.find("VulkanQueue::submit (vkQueueSubmit)") != std::string::npos);
        EXPECT(captured.find("VK_ERROR_DEVICE_LOST (-4)") != std::string::npos);
        EXPECT(captured.find("VK_EXT_device_fault unavailable") != std::string::npos);
        EXPECT(captured.find("follow_on_site") == std::string::npos);
        std::printf("diagnostics: vkQueueSubmit originating site & single-log guard ok\n");
    }

    // 2a. FenceWaitDispatcher::run() discovering VK_ERROR_DEVICE_LOST on vkWaitSemaphores.
    {
        driver::reset_mocks();
        std::string captured;
        {
            StderrCapture cap;
            EXPECT(cap.active());
            VulkanContext context(driver::get_device, d.device, true);
            EXPECT(context.loaded());
            VulkanFence fence(context, 0);
            EXPECT(fence.semaphore() != VK_NULL_HANDLE);

            std::mutex mu;
            std::condition_variable cv;
            bool drained = false;
            driver::dispatcher_wait_result.store(VK_ERROR_DEVICE_LOST, std::memory_order_release);
            context.enqueue_fence_callback(fence.semaphore(), 1, [&] {
                std::lock_guard lk(mu);
                drained = true;
                cv.notify_all();
            });
            {
                std::unique_lock lk(mu);
                EXPECT(cv.wait_for(lk, std::chrono::seconds(2), [&] { return drained; }));
            }
            EXPECT(context.is_device_lost());
            const DeviceLostDiagnostics diag = context.device_lost_diagnostics();
            EXPECT(diag.site && std::string_view(diag.site) == "FenceWaitDispatcher::run (vkWaitSemaphores)");
            EXPECT(diag.result == VK_ERROR_DEVICE_LOST);
            context.cancel_fence_waits(fence.semaphore());
            captured = cap.stop();
        }
        EXPECT(count_occurrences(captured, "[rt] d3d12: device lost (") == 1);
        EXPECT(captured.find("FenceWaitDispatcher::run (vkWaitSemaphores)") != std::string::npos);
        std::printf("diagnostics: FenceWaitDispatcher::run (vkWaitSemaphores) ok\n");
    }

    // 2b. FenceWaitDispatcher::run() discovering VK_ERROR_DEVICE_LOST on vkGetSemaphoreCounterValue.
    {
        driver::reset_mocks();
        std::string captured;
        {
            StderrCapture cap;
            EXPECT(cap.active());
            VulkanContext context(driver::get_device, d.device, true);
            EXPECT(context.loaded());
            VulkanFence fence(context, 0);
            EXPECT(fence.semaphore() != VK_NULL_HANDLE);

            std::mutex mu;
            std::condition_variable cv;
            bool drained = false;
            context.enqueue_fence_callback(fence.semaphore(), 1, [&] {
                std::lock_guard lk(mu);
                drained = true;
                cv.notify_all();
            });
            EXPECT(wait_until_dispatcher_in_wait());
            driver::fail_counter_lost.store(true, std::memory_order_release);
            EXPECT(fence.signal(1));
            {
                std::unique_lock lk(mu);
                EXPECT(cv.wait_for(lk, std::chrono::seconds(2), [&] { return drained; }));
            }
            driver::fail_counter_lost.store(false, std::memory_order_release);
            EXPECT(context.is_device_lost());
            const DeviceLostDiagnostics diag = context.device_lost_diagnostics();
            EXPECT(diag.site && std::string_view(diag.site) == "FenceWaitDispatcher::run (vkGetSemaphoreCounterValue)");
            EXPECT(diag.result == VK_ERROR_DEVICE_LOST);
            context.cancel_fence_waits(fence.semaphore());
            captured = cap.stop();
        }
        EXPECT(count_occurrences(captured, "[rt] d3d12: device lost (") == 1);
        EXPECT(captured.find("FenceWaitDispatcher::run (vkGetSemaphoreCounterValue)") != std::string::npos);
        std::printf("diagnostics: FenceWaitDispatcher::run (vkGetSemaphoreCounterValue) ok\n");
    }

#ifdef VK_EXT_DEVICE_FAULT_EXTENSION_NAME
    // 3a. VK_EXT_device_fault rich description, address infos, vendor infos, and control-char sanitization.
    {
        driver::reset_mocks();
        std::string captured;
        {
            StderrCapture cap;
            EXPECT(cap.active());
            VulkanContext context(driver::get_device, d.device, true, false, false, false, true);
            EXPECT(context.loaded() && context.device_fault_enabled());
            VulkanFence fence(context, 0);
            EXPECT(fence.semaphore() != VK_NULL_HANDLE);

            driver::next_wait_result.store(VK_ERROR_DEVICE_LOST, std::memory_order_release);
            EXPECT(!fence.wait(1, 1000));
            EXPECT(context.is_device_lost());

            const DeviceLostDiagnostics diag = context.device_lost_diagnostics();
            EXPECT(diag.site && std::string_view(diag.site) == "VulkanFence::wait (vkWaitSemaphores)");
            EXPECT(diag.result == VK_ERROR_DEVICE_LOST);
            EXPECT(diag.fault_queried && diag.fault_query_result == VK_SUCCESS);
            EXPECT(diag.fault_description == "shader page fault on read");
            EXPECT(diag.total_address_infos == 2 && diag.address_infos.size() == 2);
            EXPECT(diag.address_infos[0].address_type == VK_DEVICE_FAULT_ADDRESS_TYPE_READ_INVALID_EXT);
            EXPECT(diag.address_infos[0].address == 0x10002000ULL && diag.address_infos[0].precision == 0x1000ULL);
            EXPECT(diag.address_infos[1].address_type == VK_DEVICE_FAULT_ADDRESS_TYPE_WRITE_INVALID_EXT);
            EXPECT(diag.total_vendor_infos == 1 && diag.vendor_infos.size() == 1);
            EXPECT(diag.vendor_infos[0].description == "MMU fault detail");
            EXPECT(diag.vendor_infos[0].code == 0xdeadULL && diag.vendor_infos[0].data == 0xbeefULL);
            EXPECT(driver::fault_calls.load(std::memory_order_acquire) == 2);
            captured = cap.stop();
        }
        EXPECT(count_occurrences(captured, "[rt] d3d12: device lost (") == 1);
        EXPECT(captured.find("shader page fault on read") != std::string::npos);
        EXPECT(captured.find("READ_INVALID(1) addr=0x10002000 precision=0x1000") != std::string::npos);
        EXPECT(captured.find("\"MMU fault detail\" code=0xdead data=0xbeef") != std::string::npos);
        std::printf("diagnostics: VK_EXT_device_fault rich report ok\n");
    }

    // 3b. VK_EXT_device_fault boundedness, vendorBinarySize=0 guard, and VK_INCOMPLETE handling.
    {
        driver::reset_mocks();
        {
            std::lock_guard lock(driver::fault_mutex);
            driver::fault_mode = driver::FaultMode::oversized_incomplete;
        }
        std::string captured;
        {
            StderrCapture cap;
            EXPECT(cap.active());
            VulkanContext context(driver::get_device, d.device, true, false, false, false, true);
            context.mark_device_lost("VulkanQueue::submit (vkQueueSubmit)", VK_ERROR_DEVICE_LOST);

            const DeviceLostDiagnostics diag = context.device_lost_diagnostics();
            EXPECT(diag.fault_queried && diag.fault_query_result == VK_INCOMPLETE);
            EXPECT(diag.total_address_infos == 32 && diag.address_infos.size() == DeviceLostDiagnostics::kMaxFaultEntries);
            EXPECT(diag.total_vendor_infos == 24 && diag.vendor_infos.size() == DeviceLostDiagnostics::kMaxFaultEntries);
            EXPECT(diag.vendor_binary_size == 4096);
            EXPECT(driver::fault_calls.load(std::memory_order_acquire) == 2);
            EXPECT(driver::saw_zero_vendor_binary && driver::saw_null_vendor_binary_ptr);
            EXPECT(driver::max_seen_addr_capacity == DeviceLostDiagnostics::kMaxFaultEntries);
            EXPECT(driver::max_seen_vendor_capacity == DeviceLostDiagnostics::kMaxFaultEntries);
            captured = cap.stop();
        }
        EXPECT(count_occurrences(captured, "[rt] d3d12: device lost (") == 1);
        EXPECT(captured.find("addresses=16/32") != std::string::npos);
        EXPECT(captured.find("vendor=16/24") != std::string::npos);
        EXPECT(captured.find("vendor_binary=4096B(omitted)") != std::string::npos);
        EXPECT(captured.find("(VK_INCOMPLETE)") != std::string::npos);
        std::printf("diagnostics: VK_EXT_device_fault bounded truncation & vendorBinarySize guard ok\n");
    }

    // 3c. VK_EXT_device_fault query failure still logs host-cause cleanly.
    {
        driver::reset_mocks();
        {
            std::lock_guard lock(driver::fault_mutex);
            driver::fault_mode = driver::FaultMode::query_error;
        }
        std::string captured;
        {
            StderrCapture cap;
            EXPECT(cap.active());
            VulkanContext context(driver::get_device, d.device, true, false, false, false, true);
            context.mark_device_lost("VulkanFence::signal (vkSignalSemaphore)", VK_ERROR_DEVICE_LOST);
            const DeviceLostDiagnostics diag = context.device_lost_diagnostics();
            EXPECT(diag.fault_queried && diag.fault_query_result == VK_ERROR_OUT_OF_HOST_MEMORY);
            captured = cap.stop();
        }
        EXPECT(count_occurrences(captured, "[rt] d3d12: device lost (") == 1);
        EXPECT(captured.find("VulkanFence::signal (vkSignalSemaphore)") != std::string::npos);
        EXPECT(captured.find("VK_EXT_device_fault query failed: VK_ERROR_OUT_OF_HOST_MEMORY (-1)") != std::string::npos);
        std::printf("diagnostics: VK_EXT_device_fault query-error fallback ok\n");
    }

    // 3d. Non-VK_ERROR_DEVICE_LOST losses (and host-only mark_device_lost()) must NOT call vkGetDeviceFaultInfoEXT.
    {
        driver::reset_mocks();
        std::string captured;
        {
            StderrCapture cap;
            EXPECT(cap.active());
            VulkanContext ctx_oom(driver::get_device, d.device, true, false, false, false, true);
            ctx_oom.mark_device_lost("VulkanQueue::~VulkanQueue (vkQueueWaitIdle)", VK_ERROR_OUT_OF_HOST_MEMORY);
            EXPECT(ctx_oom.is_device_lost());
            EXPECT(!ctx_oom.device_lost_diagnostics().fault_queried);

            VulkanContext ctx_host(driver::get_device, d.device, true, false, false, false, true);
            ctx_host.mark_device_lost();
            EXPECT(ctx_host.is_device_lost());
            EXPECT(!ctx_host.device_lost_diagnostics().fault_queried);
            EXPECT(driver::fault_calls.load(std::memory_order_acquire) == 0);
            captured = cap.stop();
        }
        EXPECT(count_occurrences(captured, "[rt] d3d12: device lost (") == 2);
        EXPECT(captured.find("VK_ERROR_OUT_OF_HOST_MEMORY (-1)") != std::string::npos);
        EXPECT(captured.find("reason=host-triggered") != std::string::npos);
        std::printf("diagnostics: VUID-07336 non-DEVICE_LOST guard ok\n");
    }
#endif

    // 4. Multi-threaded concurrent device loss: single log, single fault query, and consistent summary for readers.
    {
        driver::reset_mocks();
        std::string captured;
        {
            StderrCapture cap;
            EXPECT(cap.active());
            VulkanContext context(driver::get_device, d.device, true, false, false, false, true);
            static constexpr const char* kSites[8] = {
                "worker_0", "worker_1", "worker_2", "worker_3", "worker_4", "worker_5", "worker_6", "worker_7",
            };
            std::atomic<bool> go{false};
            std::vector<std::thread> writers;
            std::vector<std::thread> readers;
            std::atomic<bool> reader_saw_inconsistent{false};

            for (int i = 0; i < 8; ++i) {
                writers.emplace_back([&, i] {
                    while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
                    context.mark_device_lost(kSites[i], VK_ERROR_DEVICE_LOST);
                });
            }
            for (int i = 0; i < 4; ++i) {
                readers.emplace_back([&] {
                    while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
                    while (!context.is_device_lost()) std::this_thread::yield();
                    const DeviceLostDiagnostics diag = context.device_lost_diagnostics();
                    const std::string summary = context.device_lost_summary();
                    if (!diag.lost || !diag.site || summary.empty() || diag.summary != summary) {
                        reader_saw_inconsistent.store(true, std::memory_order_release);
                    }
                });
            }
            go.store(true, std::memory_order_release);
            for (auto& t : writers) t.join();
            for (auto& t : readers) t.join();

            EXPECT(!reader_saw_inconsistent.load(std::memory_order_acquire));
            EXPECT(context.is_device_lost());
#ifdef VK_EXT_DEVICE_FAULT_EXTENSION_NAME
            EXPECT(driver::fault_calls.load(std::memory_order_acquire) == 2);
#endif
            captured = cap.stop();
        }
        EXPECT(count_occurrences(captured, "[rt] d3d12: device lost (") == 1);
        std::printf("diagnostics: concurrent multi-thread device-loss stress ok\n");
    }

    EXPECT(d.validation_errors() == 0);
    std::printf("vk_device_lost_diagnostics_test: ok\n");
    return 0;
}
