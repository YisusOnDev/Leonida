// Regression test for VulkanQueue::acquire() reclaiming completed CommandSlot GPU batches
// (runtime/src/vulkan_backend.h):
//   1. After a multi-submission burst that populates PredicateReducer::Batch across multiple slots,
//      a single subsequent submission (reusing only slot 0) must free the batches on all completed
//      slots, leaving 0 live VkBuffer / VkDeviceMemory / VkDescriptorPool handles.
//   2. Same for PredicatedBufferCopier::Batch (VkDescriptorPool handles).
//   3. Same for IndirectDispatchCounter::Batch (VkBuffer / VkDeviceMemory handles).
//   4. Even if vkResetCommandBuffer fails on the reused slot, all completed slots (including the
//      reused slot) must already have released their GPU batches.
// Exits 77 when no Vulkan 1.3 device (or none of the batch features) is available. No game data.
#include "vk_test_device.h"
#include "../src/vulkan_backend.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string_view>

#define EXPECT(cond)                                                                                   \
    do {                                                                                               \
        if (!(cond)) {                                                                                 \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #cond);                                    \
            return 1;                                                                                  \
        }                                                                                              \
    } while (0)

namespace {

// Minimal synthetic SPIR-V 1.0 no-op compute shader: local_size(1, 1, 1), entry point "main".
constexpr std::uint32_t kNoopComputeSpirv[] = {
    0x07230203u, 0x00010000u, 0x00000000u, 0x00000006u, 0x00000000u,
    0x00020011u, 1u,                                          // OpCapability Shader
    0x0003000eu, 0u, 1u,                                      // OpMemoryModel Logical GLSL450
    0x0005000fu, 5u, 4u, 0x6e69616du, 0x00000000u,            // OpEntryPoint GLCompute %4 "main"
    0x00060010u, 4u, 17u, 1u, 1u, 1u,                         // OpExecutionMode %4 LocalSize 1 1 1
    0x00020013u, 1u,                                          // %1 = OpTypeVoid
    0x00030021u, 2u, 1u,                                      // %2 = OpTypeFunction %1
    0x00050036u, 1u, 4u, 0u, 2u,                              // %4 = OpFunction %1 None %2
    0x000200f8u, 5u,                                          // %5 = OpLabel
    0x000100fdu,                                              // OpReturn
    0x00010038u,                                              // OpFunctionEnd
};

namespace batch_tracker {
PFN_vkGetDeviceProcAddr real_get_device = nullptr;
PFN_vkCreateBuffer real_create_buffer = nullptr;
PFN_vkDestroyBuffer real_destroy_buffer = nullptr;
PFN_vkAllocateMemory real_allocate_memory = nullptr;
PFN_vkFreeMemory real_free_memory = nullptr;
PFN_vkCreateDescriptorPool real_create_pool = nullptr;
PFN_vkDestroyDescriptorPool real_destroy_pool = nullptr;

std::atomic<std::int64_t> live_buffers{0};
std::atomic<std::int64_t> live_memories{0};
std::atomic<std::int64_t> live_pools{0};

VKAPI_ATTR VkResult VKAPI_CALL create_buffer(VkDevice device, const VkBufferCreateInfo* info,
                                             const VkAllocationCallbacks* alloc, VkBuffer* out) {
    const VkResult res = real_create_buffer(device, info, alloc, out);
    if (res == VK_SUCCESS && out && *out != VK_NULL_HANDLE)
        live_buffers.fetch_add(1, std::memory_order_relaxed);
    return res;
}

VKAPI_ATTR void VKAPI_CALL destroy_buffer(VkDevice device, VkBuffer buffer, const VkAllocationCallbacks* alloc) {
    if (buffer != VK_NULL_HANDLE) live_buffers.fetch_sub(1, std::memory_order_relaxed);
    real_destroy_buffer(device, buffer, alloc);
}

VKAPI_ATTR VkResult VKAPI_CALL allocate_memory(VkDevice device, const VkMemoryAllocateInfo* info,
                                               const VkAllocationCallbacks* alloc, VkDeviceMemory* out) {
    const VkResult res = real_allocate_memory(device, info, alloc, out);
    if (res == VK_SUCCESS && out && *out != VK_NULL_HANDLE)
        live_memories.fetch_add(1, std::memory_order_relaxed);
    return res;
}

VKAPI_ATTR void VKAPI_CALL free_memory(VkDevice device, VkDeviceMemory mem, const VkAllocationCallbacks* alloc) {
    if (mem != VK_NULL_HANDLE) live_memories.fetch_sub(1, std::memory_order_relaxed);
    real_free_memory(device, mem, alloc);
}

VKAPI_ATTR VkResult VKAPI_CALL create_pool(VkDevice device, const VkDescriptorPoolCreateInfo* info,
                                           const VkAllocationCallbacks* alloc, VkDescriptorPool* out) {
    const VkResult res = real_create_pool(device, info, alloc, out);
    if (res == VK_SUCCESS && out && *out != VK_NULL_HANDLE)
        live_pools.fetch_add(1, std::memory_order_relaxed);
    return res;
}

VKAPI_ATTR void VKAPI_CALL destroy_pool(VkDevice device, VkDescriptorPool pool, const VkAllocationCallbacks* alloc) {
    if (pool != VK_NULL_HANDLE) live_pools.fetch_sub(1, std::memory_order_relaxed);
    real_destroy_pool(device, pool, alloc);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_device(VkDevice device, const char* name) {
    const std::string_view n = name;
    if (n == "vkCreateBuffer") return reinterpret_cast<PFN_vkVoidFunction>(create_buffer);
    if (n == "vkDestroyBuffer") return reinterpret_cast<PFN_vkVoidFunction>(destroy_buffer);
    if (n == "vkAllocateMemory") return reinterpret_cast<PFN_vkVoidFunction>(allocate_memory);
    if (n == "vkFreeMemory") return reinterpret_cast<PFN_vkVoidFunction>(free_memory);
    if (n == "vkCreateDescriptorPool") return reinterpret_cast<PFN_vkVoidFunction>(create_pool);
    if (n == "vkDestroyDescriptorPool") return reinterpret_cast<PFN_vkVoidFunction>(destroy_pool);
    return real_get_device(device, name);
}

void install(PFN_vkGetDeviceProcAddr device_proc, VkDevice device) {
    real_get_device = device_proc;
    real_create_buffer = reinterpret_cast<PFN_vkCreateBuffer>(device_proc(device, "vkCreateBuffer"));
    real_destroy_buffer = reinterpret_cast<PFN_vkDestroyBuffer>(device_proc(device, "vkDestroyBuffer"));
    real_allocate_memory = reinterpret_cast<PFN_vkAllocateMemory>(device_proc(device, "vkAllocateMemory"));
    real_free_memory = reinterpret_cast<PFN_vkFreeMemory>(device_proc(device, "vkFreeMemory"));
    real_create_pool = reinterpret_cast<PFN_vkCreateDescriptorPool>(device_proc(device, "vkCreateDescriptorPool"));
    real_destroy_pool = reinterpret_cast<PFN_vkDestroyDescriptorPool>(device_proc(device, "vkDestroyDescriptorPool"));
    live_buffers.store(0);
    live_memories.store(0);
    live_pools.store(0);
}
}  // namespace batch_tracker

namespace queue_hook {
PFN_vkGetDeviceProcAddr real_get_device = nullptr;
PFN_vkResetCommandBuffer real_reset_cb = nullptr;
std::atomic<VkResult> fail_next_reset{VK_SUCCESS};

VKAPI_ATTR VkResult VKAPI_CALL reset_cb(VkCommandBuffer cb, VkCommandBufferResetFlags flags) {
    const VkResult armed = fail_next_reset.exchange(VK_SUCCESS, std::memory_order_acq_rel);
    if (armed != VK_SUCCESS) return armed;
    return real_reset_cb(cb, flags);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_device(VkDevice device, const char* name) {
    if (std::string_view(name) == "vkResetCommandBuffer")
        return reinterpret_cast<PFN_vkVoidFunction>(reset_cb);
    return real_get_device(device, name);
}

void install(PFN_vkGetDeviceProcAddr device_proc, VkDevice device) {
    real_get_device = device_proc;
    real_reset_cb = reinterpret_cast<PFN_vkResetCommandBuffer>(device_proc(device, "vkResetCommandBuffer"));
    fail_next_reset.store(VK_SUCCESS);
}
}  // namespace queue_hook

}  // namespace

int main() {
    using namespace leonida;

    vktest::Device d;
    d.request_buffer_device_address = true;
    const auto status = d.initialize();
    if (status == vktest::DeviceStatus::unavailable) {
        std::printf("SKIP: vk_queue_slot_batch_reclaim_test: no Vulkan 1.3 device\n");
        return 77;
    }
    if (status != vktest::DeviceStatus::ready) return 1;
    if (!d.conditional_rendering && !d.buffer_device_address) {
        std::printf("SKIP: vk_queue_slot_batch_reclaim_test: neither conditionalRendering nor bufferDeviceAddress\n");
        return 77;
    }

    batch_tracker::install(d.device_proc, d.device);
    queue_hook::install(d.device_proc, d.device);

    VkPhysicalDeviceMemoryProperties memory{};
    d.instance_fn<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties")(d.physical, &memory);
    VkPhysicalDeviceProperties properties{};
    d.instance_fn<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties")(d.physical, &properties);

    VkPipelineLayout noop_layout = VK_NULL_HANDLE;
    VkPipeline noop_pipeline = VK_NULL_HANDLE;
    const auto destroy_pipeline_objects = [&] {
        if (noop_pipeline) d.fn<PFN_vkDestroyPipeline>("vkDestroyPipeline")(d.device, noop_pipeline, nullptr);
        if (noop_layout) d.fn<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(d.device, noop_layout, nullptr);
    };

    if (d.buffer_device_address) {
        VkShaderModule module = VK_NULL_HANDLE;
        VkShaderModuleCreateInfo mod_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        mod_info.codeSize = sizeof(kNoopComputeSpirv);
        mod_info.pCode = kNoopComputeSpirv;
        EXPECT(d.fn<PFN_vkCreateShaderModule>("vkCreateShaderModule")(d.device, &mod_info, nullptr, &module) == VK_SUCCESS);

        const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ShaderUserData)};
        VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout_info.pushConstantRangeCount = 1;
        layout_info.pPushConstantRanges = &push;
        const VkResult lr = d.fn<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(
            d.device, &layout_info, nullptr, &noop_layout);

        VkComputePipelineCreateInfo pipe_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipe_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipe_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipe_info.stage.module = module;
        pipe_info.stage.pName = "main";
        pipe_info.layout = noop_layout;
        const VkResult pr = (lr == VK_SUCCESS)
                                ? d.fn<PFN_vkCreateComputePipelines>("vkCreateComputePipelines")(
                                      d.device, VK_NULL_HANDLE, 1, &pipe_info, nullptr, &noop_pipeline)
                                : lr;
        d.fn<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(d.device, module, nullptr);
        if (pr != VK_SUCCESS) {
            destroy_pipeline_objects();
            std::fprintf(stderr, "failed to create noop compute pipeline: %d\n", int(pr));
            return 1;
        }
    }

    int rc = 0;
    {
        VulkanContext context(queue_hook::get_device, d.device, true, d.conditional_rendering);
        if (!context.loaded()) {
            destroy_pipeline_objects();
            return 1;
        }
        if (d.conditional_rendering) {
            context.configure_predication(batch_tracker::get_device, memory,
                                          properties.limits.maxStorageBufferRange, d.family);
            if (d.storage_buffer_8bit)
                context.configure_predicated_copy(batch_tracker::get_device, properties.limits);
        }
        if (d.buffer_device_address) {
            context.configure_counted_dispatch(batch_tracker::get_device, memory, properties.limits);
            context.add_pipeline(1, ComputePipeline{noop_pipeline, noop_layout, 0, VK_NULL_HANDLE});
        }

        VulkanQueue queue(context, d.queue, d.family);
        VulkanFence gate(context, 0);
        if (!queue.ok() || !gate.semaphore()) {
            destroy_pipeline_objects();
            return 1;
        }
        // Bypass CPU Wait-before-Signal deferral so all burst submissions enter VkQueue immediately and occupy
        // distinct CommandSlots simultaneously while blocked on the real Vulkan timeline wait.
        context.holds().signaled(gate.semaphore(), 100);

        constexpr VkBufferUsageFlags buf_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
        void* pred_ptr = nullptr;
        void* src_ptr = nullptr;
        void* dst_ptr = nullptr;
        void* args_ptr = nullptr;
        void* count_ptr = nullptr;
        const VkBuffer pred_buf = d.buffer(16, buf_usage, &pred_ptr);
        const VkBuffer src_buf = d.buffer(64, buf_usage, &src_ptr);
        const VkBuffer dst_buf = d.buffer(64, buf_usage, &dst_ptr);
        const VkBuffer args_buf = d.buffer(16, buf_usage, &args_ptr);
        const VkBuffer count_buf = d.buffer(16, buf_usage, &count_ptr);
        if (!pred_buf || !src_buf || !dst_buf || !args_buf || !count_buf) {
            destroy_pipeline_objects();
            return 1;
        }
        const std::uint64_t pred_val = 1;
        std::memcpy(pred_ptr, &pred_val, sizeof(pred_val));
        std::memset(src_ptr, 0x5a, 64);
        std::memset(dst_ptr, 0x00, 64);
        const std::uint32_t dispatch_args[3] = {1, 1, 1};
        std::memcpy(args_ptr, dispatch_args, sizeof(dispatch_args));
        const std::uint32_t dispatch_count = 1;
        std::memcpy(count_ptr, &dispatch_count, sizeof(dispatch_count));

        ResourceBinding pred_rb{};
        pred_rb.buffer = pred_buf;
        pred_rb.size = 16;
        ResourceBinding src_rb{};
        src_rb.buffer = src_buf;
        src_rb.size = 64;
        ResourceBinding dst_rb{};
        dst_rb.buffer = dst_buf;
        dst_rb.size = 64;
        ResourceBinding args_rb{};
        args_rb.buffer = args_buf;
        args_rb.size = 16;
        ResourceBinding count_rb{};
        count_rb.buffer = count_buf;
        count_rb.size = 16;
        context.add_resource(1, pred_rb);
        context.add_resource(2, src_rb);
        context.add_resource(3, dst_rb);
        context.add_resource(4, args_rb);
        context.add_resource(5, count_rb);

        CommandList plain;
        EXPECT(plain.close());
        const CommandList* plain_ptr = &plain;

        constexpr std::uint32_t kBurst = 4;
        std::uint64_t gate_val = 0;

        auto run_burst_and_check = [&](const char* label, const CommandList& burst_list,
                                       std::int64_t min_buffers, std::int64_t min_memories,
                                       std::int64_t min_pools) -> bool {
            const CommandList* list_ptr = &burst_list;
            ++gate_val;
            queue.wait(gate.semaphore(), gate_val);
            bool submitted = true;
            for (std::uint32_t i = 0; i < kBurst; ++i) {
                if (queue.execute({&list_ptr, 1}) != ReplayResult::success) {
                    submitted = false;
                    break;
                }
            }
            const auto peak_buffers = batch_tracker::live_buffers.load(std::memory_order_relaxed);
            const auto peak_memories = batch_tracker::live_memories.load(std::memory_order_relaxed);
            const auto peak_pools = batch_tracker::live_pools.load(std::memory_order_relaxed);
            const bool signaled = gate.signal(gate_val);
            const bool idled = queue.idle();
            // vkQueueWaitIdle can return before the queue's progress timeline reads the last ticket (AMD Windows
            // 32.0.32029 can read 3 of 4 right after idle), so wait on the retire timeline before acquire() reads it.
            const std::uint64_t target_ticket = context.submitted_retire_ticket();
            const bool retired = idled && context.wait_retire_ticket(target_ticket) >= target_ticket;
            if (!submitted || !signaled || !retired) {
                std::fprintf(stderr, "%s: burst submit/signal/idle failed\n", label);
                return false;
            }
            if (peak_buffers < min_buffers || peak_memories < min_memories || peak_pools < min_pools) {
                std::fprintf(stderr, "%s: unexpected peak handles (buf=%lld mem=%lld pool=%lld)\n", label,
                             static_cast<long long>(peak_buffers), static_cast<long long>(peak_memories),
                             static_cast<long long>(peak_pools));
                return false;
            }
            // One plain submission reuses only slot 0; acquire() must still release batches on slots 1..kBurst-1.
            if (queue.execute({&plain_ptr, 1}) != ReplayResult::success || !queue.idle()) {
                std::fprintf(stderr, "%s: steady-state submission failed\n", label);
                return false;
            }
            const auto after_buffers = batch_tracker::live_buffers.load(std::memory_order_relaxed);
            const auto after_memories = batch_tracker::live_memories.load(std::memory_order_relaxed);
            const auto after_pools = batch_tracker::live_pools.load(std::memory_order_relaxed);
            if (after_buffers != 0 || after_memories != 0 || after_pools != 0) {
                std::fprintf(stderr, "%s: leaked GPU batch handles on non-reused slots (buf=%lld mem=%lld pool=%lld)\n",
                             label, static_cast<long long>(after_buffers), static_cast<long long>(after_memories),
                             static_cast<long long>(after_pools));
                return false;
            }
            return true;
        };

        CommandList pred_list;
        EXPECT(pred_list.set_predication({1, 0, d3d12_predication::equal_zero}) &&
               pred_list.set_predication({0, 0, 0}) && pred_list.close());

        CommandList pred_copy_list;
        EXPECT(pred_copy_list.set_predication({1, 0, d3d12_predication::equal_zero}) &&
               pred_copy_list.copy_buffer({3, 0, 2, 0, 32}) &&
               pred_copy_list.set_predication({0, 0, 0}) && pred_copy_list.close());

        CommandList counted_list;
        ExecuteIndirect indirect{};
        indirect.kind = IndirectKind::dispatch;
        indirect.stride = 12;
        indirect.max_count = 1;
        indirect.argument_buffer = 4;
        indirect.count_buffer = 5;
        indirect.pipeline = 1;
        EXPECT(counted_list.execute_indirect(indirect) && counted_list.close());

        unsigned cases_run = 0;
        if (d.conditional_rendering) {
            if (!run_burst_and_check("predicate", pred_list, kBurst, kBurst, kBurst)) rc = 1;
            ++cases_run;
        }
        if (d.conditional_rendering && d.storage_buffer_8bit) {
            if (!run_burst_and_check("predicated_copy", pred_copy_list, kBurst, kBurst, kBurst * 2)) rc = 1;
            ++cases_run;
        }
        if (d.buffer_device_address) {
            if (!run_burst_and_check("counted_dispatch", counted_list, kBurst, kBurst, 0)) rc = 1;
            ++cases_run;
        }

        // Case 4: Even when vkResetCommandBuffer fails on the reused slot, all completed slots (including slot 0)
        // must already have released their GPU batches during the completed-slot sweep.
        if (rc == 0) {
            const CommandList& reset_case_list = d.conditional_rendering ? pred_list : counted_list;
            const CommandList* list_ptr = &reset_case_list;
            ++gate_val;
            queue.wait(gate.semaphore(), gate_val);
            const bool ok0 = queue.execute({&list_ptr, 1}) == ReplayResult::success;
            const bool ok1 = queue.execute({&list_ptr, 1}) == ReplayResult::success;
            const bool sig = gate.signal(gate_val);
            const bool idl = queue.idle();
            const std::uint64_t reset_ticket = context.submitted_retire_ticket();
            const bool ret = idl && context.wait_retire_ticket(reset_ticket) >= reset_ticket;
            if (!ok0 || !ok1 || !sig || !ret || batch_tracker::live_buffers.load() < 2) {
                std::fprintf(stderr, "reset_failure: setup failed\n");
                rc = 1;
            } else {
                queue_hook::fail_next_reset.store(VK_ERROR_OUT_OF_HOST_MEMORY, std::memory_order_release);
                if (queue.execute({&plain_ptr, 1}) != ReplayResult::unsupported) {
                    std::fprintf(stderr, "reset_failure: expected unsupported on failed vkResetCommandBuffer\n");
                    rc = 1;
                } else if (batch_tracker::live_buffers.load() != 0 || batch_tracker::live_memories.load() != 0 ||
                           batch_tracker::live_pools.load() != 0) {
                    std::fprintf(stderr,
                                 "reset_failure: leaked handles after failed vkResetCommandBuffer "
                                 "(buf=%lld mem=%lld pool=%lld)\n",
                                 static_cast<long long>(batch_tracker::live_buffers.load()),
                                 static_cast<long long>(batch_tracker::live_memories.load()),
                                 static_cast<long long>(batch_tracker::live_pools.load()));
                    rc = 1;
                } else if (queue.execute({&plain_ptr, 1}) != ReplayResult::success || !queue.idle()) {
                    std::fprintf(stderr, "reset_failure: recovery submission failed\n");
                    rc = 1;
                }
            }
            ++cases_run;
        }

        if (rc == 0) {
            std::printf("vk_queue_slot_batch_reclaim_test: %u cases passed (validation errors: %u)\n",
                        cases_run, vktest::validation_error_count());
        }
    }

    destroy_pipeline_objects();
    if (vktest::validation_error_count() != 0) {
        std::fprintf(stderr, "vk_queue_slot_batch_reclaim_test: %u validation errors\n",
                     vktest::validation_error_count());
        return 1;
    }
    return rc;
}
