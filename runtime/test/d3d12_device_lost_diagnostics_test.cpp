// D3D12 device-lost diagnostics end-to-end tests:
//   1. create_vulkan_device_standalone probes and enables VK_EXT_device_fault when supported by the GPU.
//   2. ID3D12Device5::RemoveDevice() + repeated GetDeviceRemovedReason() logs once with host cause and no spam.
//   3. Queue submit VK_ERROR_DEVICE_LOST marks the device removed, queries VK_EXT_device_fault once, and
//      GetDeviceRemovedReason() logs once with the recorded originating site and fault info.
//   4. WindowPresenter present hitting VK_ERROR_DEVICE_LOST marks the device removed, logs the presenter host site once,
//      and surfaces DXGI_ERROR_DEVICE_REMOVED from both present() and GetDeviceRemovedReason().
// Skips with exit code 77 when no usable Vulkan device is available. No game data.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"

#include <dxgi1_4.h>
#include <fcntl.h>
#include <io.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { \
        std::printf("FAIL line %d: %s\n", __LINE__, #condition); \
        std::fflush(stdout); \
        std::_Exit(1); \
    } \
} while (0)

namespace {

template<class T> struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
};

class StderrCapture {
public:
    StderrCapture() {
        std::fflush(stderr);
        int fds[2]{-1, -1};
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
        read_fd_ = fds[0];
        reader_ = std::thread([this] {
            char buf[512];
            for (;;) {
                const int n = _read(read_fd_, buf, sizeof(buf));
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
            _dup2(saved_fd_, _fileno(stderr));
            _close(saved_fd_);
            saved_fd_ = -1;
        }
        if (reader_.joinable()) reader_.join();
        if (read_fd_ >= 0) {
            _close(read_fd_);
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

LRESULT CALLBACK window_proc(HWND window, UINT message, WPARAM w, LPARAM l) {
    return DefWindowProcW(window, message, w, l);
}

bool has_interactive_window_station() {
    const HWINSTA winsta = GetProcessWindowStation();
    if (!winsta) return false;
    USEROBJECTFLAGS flags{};
    return GetUserObjectInformationW(winsta, UOI_FLAGS, &flags, sizeof(flags), nullptr) &&
           (flags.dwFlags & WSF_VISIBLE) != 0;
}

namespace driver {

PFN_vkGetInstanceProcAddr real_get_instance = nullptr;
PFN_vkGetDeviceProcAddr real_get_device = nullptr;
PFN_vkQueueSubmit real_queue_submit = nullptr;
PFN_vkQueuePresentKHR real_queue_present = nullptr;

std::atomic<VkResult> next_submit_result{VK_SUCCESS};
std::atomic<VkResult> next_present_result{VK_SUCCESS};
std::atomic<std::uint32_t> fault_calls{0};
const char* fault_description = "synthetic D3D12 fault";

void reset_mocks(const char* desc = "synthetic D3D12 fault") {
    next_submit_result.store(VK_SUCCESS, std::memory_order_release);
    next_present_result.store(VK_SUCCESS, std::memory_order_release);
    fault_calls.store(0, std::memory_order_release);
    fault_description = desc;
}

VKAPI_ATTR VkResult VKAPI_CALL queue_submit(VkQueue queue, std::uint32_t count, const VkSubmitInfo* submits,
                                            VkFence fence) {
    const VkResult injected = next_submit_result.exchange(VK_SUCCESS, std::memory_order_acq_rel);
    if (injected != VK_SUCCESS) return injected;
    return real_queue_submit(queue, count, submits, fence);
}

VKAPI_ATTR VkResult VKAPI_CALL queue_present(VkQueue queue, const VkPresentInfoKHR* info) {
    const VkResult injected = next_present_result.exchange(VK_SUCCESS, std::memory_order_acq_rel);
    if (injected != VK_SUCCESS) return injected;
    return real_queue_present(queue, info);
}

#ifdef VK_EXT_DEVICE_FAULT_EXTENSION_NAME
VKAPI_ATTR VkResult VKAPI_CALL get_device_fault_info(VkDevice, VkDeviceFaultCountsEXT* pFaultCounts,
                                                     VkDeviceFaultInfoEXT* pFaultInfo) {
    fault_calls.fetch_add(1, std::memory_order_acq_rel);
    if (!pFaultCounts) return VK_ERROR_INITIALIZATION_FAILED;
    if (!pFaultInfo) {
        pFaultCounts->addressInfoCount = 1;
        pFaultCounts->vendorInfoCount = 0;
        pFaultCounts->vendorBinarySize = 0;
        return VK_SUCCESS;
    }
    std::snprintf(pFaultInfo->description, VK_MAX_DESCRIPTION_SIZE, "%s", fault_description);
    pFaultCounts->addressInfoCount = std::min<std::uint32_t>(pFaultCounts->addressInfoCount, 1u);
    if (pFaultCounts->addressInfoCount >= 1 && pFaultInfo->pAddressInfos) {
        pFaultInfo->pAddressInfos[0] = {VK_DEVICE_FAULT_ADDRESS_TYPE_READ_INVALID_EXT, 0xfeedface0000ULL, 0x1000ULL};
    }
    pFaultCounts->vendorInfoCount = 0;
    return VK_SUCCESS;
}
#endif

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_device(VkDevice device, const char* name) {
    const std::string_view n = name;
    if (n == "vkQueueSubmit") return reinterpret_cast<PFN_vkVoidFunction>(queue_submit);
    if (n == "vkQueuePresentKHR") return reinterpret_cast<PFN_vkVoidFunction>(queue_present);
#ifdef VK_EXT_DEVICE_FAULT_EXTENSION_NAME
    if (n == "vkGetDeviceFaultInfoEXT") return reinterpret_cast<PFN_vkVoidFunction>(get_device_fault_info);
#endif
    return real_get_device(device, name);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_instance(VkInstance instance, const char* name) {
    const std::string_view n = name;
    if (n == "vkGetDeviceProcAddr") return reinterpret_cast<PFN_vkVoidFunction>(get_device);
    return real_get_instance(instance, name);
}

void install(PFN_vkGetInstanceProcAddr gipa, VkInstance instance, VkDevice device) {
    real_get_instance = gipa;
    real_get_device = reinterpret_cast<PFN_vkGetDeviceProcAddr>(gipa(instance, "vkGetDeviceProcAddr"));
    real_queue_submit = reinterpret_cast<PFN_vkQueueSubmit>(real_get_device(device, "vkQueueSubmit"));
    real_queue_present = reinterpret_cast<PFN_vkQueuePresentKHR>(real_get_device(device, "vkQueuePresentKHR"));
}

}  // namespace driver

bool physical_supports_device_fault(const rt::d3d12::VulkanDeviceInfo& made) {
#ifdef VK_EXT_DEVICE_FAULT_EXTENSION_NAME
    if (!made.get_instance_proc || !made.instance || !made.physical) return false;
    const auto enum_exts = reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(
        made.get_instance_proc(made.instance, "vkEnumerateDeviceExtensionProperties"));
    const auto get_features2 = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(
        made.get_instance_proc(made.instance, "vkGetPhysicalDeviceFeatures2"));
    if (!enum_exts || !get_features2) return false;
    std::uint32_t count = 0;
    if (enum_exts(made.physical, nullptr, &count, nullptr) != VK_SUCCESS) return false;
    std::vector<VkExtensionProperties> exts(count);
    if (enum_exts(made.physical, nullptr, &count, exts.data()) != VK_SUCCESS) return false;
    bool listed = false;
    for (const auto& e : exts)
        listed = listed || std::strcmp(e.extensionName, VK_EXT_DEVICE_FAULT_EXTENSION_NAME) == 0;
    if (!listed) return false;
    VkPhysicalDeviceFaultFeaturesEXT fault{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT};
    VkPhysicalDeviceFeatures2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &fault};
    get_features2(made.physical, &query);
    return fault.deviceFault == VK_TRUE;
#else
    (void)made;
    return false;
#endif
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    // 1. Standalone device probes/enables VK_EXT_device_fault cleanly and GetDeviceRemovedReason() is silent on S_OK.
    {
        Ref<ID3D12Device> standalone;
        if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&standalone.p))) || !standalone.p) {
            std::fprintf(stderr, "d3d12_device_lost_diagnostics_test: no standalone Vulkan device, skipped\n");
            return 77;
        }
        const rt::d3d12::VulkanDeviceInfo* made = rt::d3d12::vulkan_device_info(standalone.p);
        EXPECT(made != nullptr);
        const bool expected_fault = physical_supports_device_fault(*made);
        EXPECT(made->device_fault == expected_fault);
        std::printf("standalone: VK_EXT_device_fault supported=%d enabled=%d\n",
                    int(expected_fault), int(made->device_fault));

        std::string captured;
        {
            StderrCapture cap;
            EXPECT(cap.active());
            EXPECT(standalone->GetDeviceRemovedReason() == S_OK);
            EXPECT(standalone->GetDeviceRemovedReason() == S_OK);
            captured = cap.stop();
        }
        EXPECT(captured.find("GetDeviceRemovedReason") == std::string::npos);
        EXPECT(captured.find("device lost") == std::string::npos);

        // 2. RemoveDevice() + repeated GetDeviceRemovedReason() logs once without duplicate spam.
        Ref<ID3D12Device5> dev5;
        EXPECT(SUCCEEDED(standalone->QueryInterface(IID_PPV_ARGS(&dev5.p))));
        {
            StderrCapture cap;
            EXPECT(cap.active());
            dev5->RemoveDevice();
            dev5->RemoveDevice();
            for (int i = 0; i < 16; ++i) EXPECT(dev5->GetDeviceRemovedReason() == DXGI_ERROR_DEVICE_REMOVED);
            captured = cap.stop();
        }
        EXPECT(count_occurrences(captured, "[rt] d3d12: device removed via ID3D12Device5::RemoveDevice") == 1);
        EXPECT(count_occurrences(captured, "[rt] d3d12: GetDeviceRemovedReason -> DXGI_ERROR_DEVICE_REMOVED") == 1);
        EXPECT(captured.find("ID3D12Device5::RemoveDevice") != std::string::npos);
        std::printf("d3d12: RemoveDevice + repeated GetDeviceRemovedReason single-log ok\n");
    }

    vktest::Device vk;
    const auto status = vk.initialize(true);
    if (status == vktest::DeviceStatus::unavailable) {
        std::fprintf(stderr, "d3d12_device_lost_diagnostics_test: Vulkan presentation device unavailable, skipped\n");
        return 77;
    }
    EXPECT(status == vktest::DeviceStatus::ready);
    driver::install(vk.instance_proc, vk.instance, vk.device);

    // 3. Queue submit VK_ERROR_DEVICE_LOST on ID3D12Device + GetDeviceRemovedReason() reports host site + fault info.
    {
        driver::reset_mocks("compute shader timeout");
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = driver::get_instance;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        info.device_fault = true;

        std::string captured;
        {
            StderrCapture cap;
            EXPECT(cap.active());
            Ref<ID3D12Device> device;
            EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))));
            Ref<ID3D12CommandQueue> queue;
            D3D12_COMMAND_QUEUE_DESC qdesc{};
            qdesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
            EXPECT(SUCCEEDED(device->CreateCommandQueue(&qdesc, IID_PPV_ARGS(&queue.p))));
            Ref<ID3D12CommandAllocator> alloc;
            EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc.p))));
            Ref<ID3D12GraphicsCommandList> list;
            EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr,
                                                       IID_PPV_ARGS(&list.p))));
            EXPECT(SUCCEEDED(list->Close()));

            driver::next_submit_result.store(VK_ERROR_DEVICE_LOST, std::memory_order_release);
            ID3D12CommandList* lists[] = {list.p};
            queue->ExecuteCommandLists(1, lists);
            for (int i = 0; i < 8; ++i) EXPECT(device->GetDeviceRemovedReason() == DXGI_ERROR_DEVICE_REMOVED);
            captured = cap.stop();
        }
        EXPECT(count_occurrences(captured, "[rt] d3d12: device lost (") == 1);
        EXPECT(count_occurrences(captured, "[rt] d3d12: GetDeviceRemovedReason -> DXGI_ERROR_DEVICE_REMOVED") == 1);
        EXPECT(captured.find("VulkanQueue::submit (vkQueueSubmit)") != std::string::npos);
        EXPECT(captured.find("VK_ERROR_DEVICE_LOST (-4)") != std::string::npos);
#ifdef VK_EXT_DEVICE_FAULT_EXTENSION_NAME
        EXPECT(driver::fault_calls.load(std::memory_order_acquire) == 2);
        EXPECT(captured.find("compute shader timeout") != std::string::npos);
        EXPECT(captured.find("0xfeedface0000") != std::string::npos);
#endif
        std::printf("d3d12: ExecuteCommandLists VK_ERROR_DEVICE_LOST + GetDeviceRemovedReason ok\n");
    }

    // 4. WindowPresenter present hitting VK_ERROR_DEVICE_LOST marks context lost and exposes cause in GetDeviceRemovedReason.
    if (has_interactive_window_station()) {
        WNDCLASSW wc{};
        wc.lpfnWndProc = window_proc;
        wc.hInstance = GetModuleHandleW(nullptr);
        wc.lpszClassName = L"LeonidaDeviceLostPresentTest";
        if (RegisterClassW(&wc)) {
            RECT fr{0, 0, 128, 96};
            AdjustWindowRect(&fr, WS_OVERLAPPEDWINDOW, FALSE);
            HWND window = CreateWindowExW(0, wc.lpszClassName, L"Leonida device lost test", WS_OVERLAPPEDWINDOW, 40, 40,
                                          fr.right - fr.left, fr.bottom - fr.top, nullptr, nullptr, wc.hInstance,
                                          nullptr);
            if (window) {
                driver::reset_mocks("present queue hang");
                rt::d3d12::VulkanDeviceInfo info{};
                info.get_instance_proc = driver::get_instance;
                info.instance = vk.instance;
                info.physical = vk.physical;
                info.device = vk.device;
                info.queue = vk.queue;
                info.queue_family = vk.family;
                info.presentation = true;
                info.device_fault = true;

                std::string captured;
                {
                    StderrCapture cap;
                    EXPECT(cap.active());
                    Ref<ID3D12Device> device;
                    EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))));
                    auto presenter = rt::d3d12::create_window_presenter(device.p, window);
                    if (presenter) {
                        driver::next_present_result.store(VK_ERROR_DEVICE_LOST, std::memory_order_release);
                        EXPECT(presenter->present_blank(0) == DXGI_ERROR_DEVICE_REMOVED);
                        EXPECT(presenter->present_blank(0) == DXGI_ERROR_DEVICE_REMOVED);
                        EXPECT(presenter->present_blank(0) == DXGI_ERROR_DEVICE_REMOVED);
                        EXPECT(device->GetDeviceRemovedReason() == DXGI_ERROR_DEVICE_REMOVED);
                        EXPECT(device->GetDeviceRemovedReason() == DXGI_ERROR_DEVICE_REMOVED);
                    }
                    captured = cap.stop();
                }
                if (captured.find("cannot present to this window") == std::string::npos) {
                    EXPECT(count_occurrences(captured, "[rt] d3d12: device lost (") == 1);
                    EXPECT(count_occurrences(captured, "[rt] d3d12: present failed (VkResult -4)") == 1);
                    EXPECT(count_occurrences(captured,
                                             "[rt] d3d12: GetDeviceRemovedReason -> DXGI_ERROR_DEVICE_REMOVED") == 1);
                    EXPECT(captured.find("VulkanPresenter::submit_now (vkQueuePresentKHR)") != std::string::npos);
#ifdef VK_EXT_DEVICE_FAULT_EXTENSION_NAME
                    EXPECT(driver::fault_calls.load(std::memory_order_acquire) == 2);
                    EXPECT(captured.find("present queue hang") != std::string::npos);
#endif
                    std::printf("d3d12: WindowPresenter VK_ERROR_DEVICE_LOST + GetDeviceRemovedReason ok\n");
                }
                DestroyWindow(window);
            }
            UnregisterClassW(wc.lpszClassName, wc.hInstance);
        }
    }

    EXPECT(vk.validation_errors() == 0);
    std::printf("d3d12_device_lost_diagnostics_test: ok\n");
    return 0;
}
