// Storage-image initialization and cross-queue init-fence coverage (docs/d3d12_vulkan.md section 6, M4):
//   0. CPU replay contract (no GPU required; deterministic negative control vs. main):
//      - Replaying CopyTextureRegion (upload) and CopyResource (multi-subresource: mip 0 GENERAL, mip 1 UNDEFINED) on
//        a storage_image=true ResourceBinding leaves LayoutTracker in VK_IMAGE_LAYOUT_GENERAL for every touched
//        subresource and passes GENERAL to the Vulkan copy commands (whereas non-storage textures still use
//        TRANSFER_DST_OPTIMAL).
//      - detail::initialize() returns true on UNDEFINED -> GENERAL and false when already GENERAL (preventing spurious
//        init-fence publication on VulkanQueue).
//   1. Copy-queue upload before first Dispatch: ALLOW_UNORDERED_ACCESS texture created in COMMON, uploaded via
//      CopyTextureRegion on a COPY queue, then read by cs_m4_indirect on a COMPUTE queue with no barrier (implicit
//      COMMON -> UNORDERED_ACCESS promotion). Verifies GENERAL layout and exact uploaded pattern.
//   2. Barrier + copy readback before any Dispatch: ALLOW_UNORDERED_ACCESS texture created in UNORDERED_ACCESS,
//      read back on a DIRECT queue via shader-less UNORDERED_ACCESS -> COPY_SOURCE -> CopyTextureRegion ->
//      UNORDERED_ACCESS (verifies zero-initialized), then filled by cs_m4_fill on COMPUTE and read back on COPY.
//   3. Cross-queue independent initializers: COMPUTE dispatches on ImgA before DIRECT ever runs a shader; DIRECT
//      dispatches on ImgB; COPY waits on both fences and reads back both ImgA and ImgB (verifies per-queue init
//      fences do not overwrite each other).
//   4. Partial subresource upload (2 mips) before Dispatch: 16x16 MipLevels=2 ALLOW_UNORDERED_ACCESS texture in
//      COMMON; COPY uploads pattern to mip 0 only; COMPUTE reads mip 0 and mip 1 via cs_m4_indirect and COPY reads
//      both mips back (verifies mip 0 keeps uploaded data and mip 1 is zero-initialized in GENERAL).
// Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation (0 errors required).
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../src/shader_library.h"
#include "../src/vulkan_replay.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#ifndef LEONIDA_TEST_M4_DIR
#error "LEONIDA_TEST_M4_DIR must name the folder build.py wrote (spirv/index.json, fssm/cs_*.fssm)"
#endif

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using namespace d3d12test;
using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void* heap);  // heap slots 9/10: handle in RAX

namespace {

std::vector<std::uint8_t> read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

constexpr UINT records = 256;       // 4 groups of 64
constexpr UINT image_dim = 16;      // 16 x 16 R32_UINT
constexpr UINT mip1_dim = 8;        // 8 x 8 R32_UINT
constexpr UINT image_pitch = D3D12_TEXTURE_DATA_PITCH_ALIGNMENT;
constexpr UINT64 data_bytes = records * 16;
constexpr UINT64 args_bytes = 256;
constexpr UINT64 in_bytes = records * 4;
constexpr UINT64 out_bytes = records * 8;
constexpr UINT64 image_copy_bytes = UINT64(image_dim) * image_pitch;
constexpr UINT64 mip1_copy_bytes = UINT64(mip1_dim) * image_pitch;

std::uint32_t pattern_word(UINT g, std::uint32_t salt) {
    return ((g + 1u) * 0x9e3779b1u) ^ (g << 7) ^ salt;
}

std::uint32_t fill_image_word(std::uint32_t in_word, UINT g) {
    const UINT x = g & 15u, y = g >> 4;
    return in_word * 3u + x * 0x10001u + y;
}

D3D12_RESOURCE_DESC tex2d_uav_desc(UINT width, UINT height, UINT16 mips = 1) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = mips;
    desc.Format = DXGI_FORMAT_R32_UINT;
    desc.SampleDesc.Count = 1;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    return desc;
}

D3D12_UNORDERED_ACCESS_VIEW_DESC buffer_uav_desc(DXGI_FORMAT format, UINT count, UINT stride,
                                                 D3D12_BUFFER_UAV_FLAGS flags) {
    D3D12_UNORDERED_ACCESS_VIEW_DESC view{};
    view.Format = format;
    view.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    view.Buffer = {0, count, stride, 0, flags};
    return view;
}

D3D12_UNORDERED_ACCESS_VIEW_DESC tex2d_uav_view(UINT mip_slice = 0) {
    D3D12_UNORDERED_ACCESS_VIEW_DESC view{};
    view.Format = DXGI_FORMAT_R32_UINT;
    view.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    view.Texture2D.MipSlice = mip_slice;
    return view;
}

bool wait_fence(ID3D12Fence* fence, UINT64 value) {
    const HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event) return false;
    const HRESULT hr = fence->SetEventOnCompletion(value, event);
    const DWORD waited = SUCCEEDED(hr) ? WaitForSingleObject(event, 20000) : WAIT_FAILED;
    CloseHandle(event);
    return SUCCEEDED(hr) && waited == WAIT_OBJECT_0 && fence->GetCompletedValue() >= value;
}

bool check_image_rows(const char* what, const std::uint8_t* base, UINT width, UINT height,
                      const std::vector<std::uint32_t>& expected) {
    unsigned bad = 0;
    for (UINT y = 0; y < height; ++y) {
        for (UINT x = 0; x < width; ++x) {
            std::uint32_t got = 0;
            std::memcpy(&got, base + UINT64(y) * image_pitch + UINT64(x) * 4, 4);
            const std::uint32_t want = expected[y * width + x];
            if (got != want) {
                if (bad++ < 4)
                    std::fprintf(stderr, "%s (%u,%u): got %08x want %08x\n", what, x, y, got, want);
            }
        }
    }
    if (bad) std::fprintf(stderr, "%s: %u of %u texels differ\n", what, bad, width * height);
    return bad == 0;
}

bool check_indirect_out_matches_image(const char* what, const std::uint8_t* base,
                                      const std::vector<std::uint32_t>& expected) {
    unsigned bad = 0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        std::uint32_t w0 = 0, w1 = 0;
        std::memcpy(&w0, base + i * 8 + 0, 4);
        std::memcpy(&w1, base + i * 8 + 4, 4);
        if (w0 != expected[i] || w1 != expected[i]) {
            if (bad++ < 4)
                std::fprintf(stderr, "%s[%zu]: got (%08x, %08x) want (%08x, %08x)\n",
                             what, i, w0, w1, expected[i], expected[i]);
        }
    }
    if (bad) std::fprintf(stderr, "%s: %u of %zu pairs differ\n", what, bad, expected.size());
    return bad == 0;
}

struct CpuReplayRecorder {
    VkImageLayout last_buf_to_img_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout last_copy_img_src_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout last_copy_img_dst_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    std::uint32_t last_copy_img_regions = 0;
    std::uint32_t buf_to_img_calls = 0;
    std::uint32_t copy_img_calls = 0;
    std::uint32_t clear_calls = 0;
};

CpuReplayRecorder* g_cpu_recorder = nullptr;

void VKAPI_CALL stub_pipeline_barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
                                      std::uint32_t, const VkMemoryBarrier*, std::uint32_t,
                                      const VkBufferMemoryBarrier*, std::uint32_t, const VkImageMemoryBarrier*) {}

void VKAPI_CALL stub_clear_color_image(VkCommandBuffer, VkImage, VkImageLayout, const VkClearColorValue*,
                                       std::uint32_t, const VkImageSubresourceRange*) {
    if (g_cpu_recorder) ++g_cpu_recorder->clear_calls;
}

void VKAPI_CALL stub_copy_buffer(VkCommandBuffer, VkBuffer, VkBuffer, std::uint32_t, const VkBufferCopy*) {}

void VKAPI_CALL stub_copy_buffer_to_image(VkCommandBuffer, VkBuffer, VkImage, VkImageLayout dst_layout,
                                          std::uint32_t, const VkBufferImageCopy*) {
    if (g_cpu_recorder) {
        g_cpu_recorder->last_buf_to_img_layout = dst_layout;
        ++g_cpu_recorder->buf_to_img_calls;
    }
}

void VKAPI_CALL stub_copy_image_to_buffer(VkCommandBuffer, VkImage, VkImageLayout, VkBuffer, std::uint32_t,
                                          const VkBufferImageCopy*) {}

void VKAPI_CALL stub_copy_image(VkCommandBuffer, VkImage, VkImageLayout src_layout, VkImage, VkImageLayout dst_layout,
                                std::uint32_t region_count, const VkImageCopy*) {
    if (g_cpu_recorder) {
        g_cpu_recorder->last_copy_img_src_layout = src_layout;
        g_cpu_recorder->last_copy_img_dst_layout = dst_layout;
        g_cpu_recorder->last_copy_img_regions = region_count;
        ++g_cpu_recorder->copy_img_calls;
    }
}

// Deterministic CPU-only replay contract test (no VkDevice required). Pins:
//   (a) CopyTextureRegion upload and multi-subresource CopyResource (mip 0 GENERAL, mip 1 UNDEFINED) on a
//       storage_image=true ResourceBinding keep every touched subresource in VK_IMAGE_LAYOUT_GENERAL in
//       LayoutTracker and pass GENERAL to the Vulkan copy commands.
//   (b) Non-storage textures (storage_image=false) still transition to VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL.
//   (c) detail::initialize() returns true on UNDEFINED -> GENERAL and false when already GENERAL.
int run_cpu_replay_contract_test() {
    using namespace leonida;
    CpuReplayRecorder rec{};
    g_cpu_recorder = &rec;

    ReplayFunctions vk{};
    vk.pipeline_barrier = &stub_pipeline_barrier;
    vk.clear_color_image = &stub_clear_color_image;
    vk.copy_buffer = &stub_copy_buffer;
    vk.copy_buffer_to_image = &stub_copy_buffer_to_image;
    vk.copy_image_to_buffer = &stub_copy_image_to_buffer;
    vk.copy_image = &stub_copy_image;

    const auto dummy_cmd = reinterpret_cast<VkCommandBuffer>(std::uintptr_t(0x1));
    constexpr ResourceId kUploadBuf = 1;
    constexpr ResourceId kStorageDst = 2;
    constexpr ResourceId kStorageSrc = 3;
    constexpr ResourceId kRegularDst = 4;

    ResourceBinding upload_buf{};
    upload_buf.buffer = reinterpret_cast<VkBuffer>(std::uintptr_t(0x1001));
    upload_buf.size = image_copy_bytes;

    ResourceBinding storage_dst{};
    storage_dst.image = reinterpret_cast<VkImage>(std::uintptr_t(0x2001));
    storage_dst.format = VK_FORMAT_R32_UINT;
    storage_dst.width = image_dim;
    storage_dst.height = image_dim;
    storage_dst.mip_levels = 2;
    storage_dst.array_layers = 1;
    storage_dst.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
    storage_dst.storage_image = true;

    ResourceBinding storage_src = storage_dst;
    storage_src.image = reinterpret_cast<VkImage>(std::uintptr_t(0x2002));

    ResourceBinding regular_dst = storage_dst;
    regular_dst.image = reinterpret_cast<VkImage>(std::uintptr_t(0x2003));
    regular_dst.storage_image = false;

    const auto resolve = [&](ResourceId id) -> const ResourceBinding* {
        if (id == kUploadBuf) return &upload_buf;
        if (id == kStorageDst) return &storage_dst;
        if (id == kStorageSrc) return &storage_src;
        if (id == kRegularDst) return &regular_dst;
        return nullptr;
    };

    LayoutTracker tracker;
    tracker.track(kStorageDst, storage_dst, VK_IMAGE_LAYOUT_UNDEFINED);
    tracker.track(kStorageSrc, storage_src, VK_IMAGE_LAYOUT_GENERAL);
    tracker.track(kRegularDst, regular_dst, VK_IMAGE_LAYOUT_UNDEFINED);

    // (a1) CopyTextureRegion upload into mip 0 of a storage_image=true texture.
    {
        CommandList list;
        CopyTexture copy{};
        copy.dst = {kStorageDst, false, 0, 0, 0, 0, 0, 0};
        copy.src = {kUploadBuf, true, 0, 0, image_dim, image_dim, 1, image_pitch};
        EXPECT(list.copy_texture(copy));
        EXPECT(list.close());
        EXPECT(replay(list, dummy_cmd, resolve, vk, &tracker) == ReplayResult::success);
        EXPECT(rec.buf_to_img_calls == 1);
        EXPECT(rec.last_buf_to_img_layout == VK_IMAGE_LAYOUT_GENERAL);
        EXPECT(tracker.layout(kStorageDst, 0) == VK_IMAGE_LAYOUT_GENERAL);
        EXPECT(tracker.layout(kStorageDst, 1) == VK_IMAGE_LAYOUT_UNDEFINED);
    }

    // (a2) CopyResource (all 2 mips) into storage_dst where mip 0 is GENERAL and mip 1 is UNDEFINED.
    {
        CommandList list;
        EXPECT(list.copy_resource({kStorageDst, kStorageSrc}));
        EXPECT(list.close());
        EXPECT(replay(list, dummy_cmd, resolve, vk, &tracker) == ReplayResult::success);
        EXPECT(rec.copy_img_calls == 1);
        EXPECT(rec.last_copy_img_regions == 2);
        EXPECT(rec.last_copy_img_src_layout == VK_IMAGE_LAYOUT_GENERAL);
        EXPECT(rec.last_copy_img_dst_layout == VK_IMAGE_LAYOUT_GENERAL);
        EXPECT(tracker.layout(kStorageDst, 0) == VK_IMAGE_LAYOUT_GENERAL);
        EXPECT(tracker.layout(kStorageDst, 1) == VK_IMAGE_LAYOUT_GENERAL);
    }

    // (b) Positive control: non-storage texture (storage_image=false) uses TRANSFER_DST_OPTIMAL.
    {
        CommandList list;
        CopyTexture copy{};
        copy.dst = {kRegularDst, false, 0, 0, 0, 0, 0, 0};
        copy.src = {kUploadBuf, true, 0, 0, image_dim, image_dim, 1, image_pitch};
        EXPECT(list.copy_texture(copy));
        EXPECT(list.close());
        EXPECT(replay(list, dummy_cmd, resolve, vk, &tracker) == ReplayResult::success);
        EXPECT(rec.buf_to_img_calls == 2);
        EXPECT(rec.last_buf_to_img_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        EXPECT(tracker.layout(kRegularDst, 0) == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        EXPECT(tracker.layout(kRegularDst, 1) == VK_IMAGE_LAYOUT_UNDEFINED);
    }

    // (c) detail::initialize() returns true when transitioning from UNDEFINED and false when already GENERAL.
    {
        constexpr ResourceId kInitTex = 5;
        ResourceBinding init_tex = storage_dst;
        init_tex.image = reinterpret_cast<VkImage>(std::uintptr_t(0x2004));
        init_tex.mip_levels = 1;
        tracker.track(kInitTex, init_tex, VK_IMAGE_LAYOUT_UNDEFINED);
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        LayoutPieces pieces;
        EXPECT(tracker.pieces(kInitTex, range, pieces));
        const bool first_init = detail::initialize(vk, dummy_cmd, init_tex, kInitTex, pieces, VK_IMAGE_LAYOUT_GENERAL,
                                                   &tracker);
        EXPECT(first_init);
        EXPECT(tracker.layout(kInitTex, 0) == VK_IMAGE_LAYOUT_GENERAL);

        pieces.clear();
        EXPECT(tracker.pieces(kInitTex, range, pieces));
        const bool second_init = detail::initialize(vk, dummy_cmd, init_tex, kInitTex, pieces, VK_IMAGE_LAYOUT_GENERAL,
                                                    &tracker);
        EXPECT(!second_init);
        EXPECT(tracker.layout(kInitTex, 0) == VK_IMAGE_LAYOUT_GENERAL);
    }

    g_cpu_recorder = nullptr;
    return 0;
}

}  // namespace

int main() {
    if (run_cpu_replay_contract_test() != 0) return 1;

    vktest::Device d;
    d.queue_count_wanted = 3;  // copy, compute and direct each get a VkQueue where the family has three
    const auto device_status = d.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) return 77;
    if (device_status != vktest::DeviceStatus::ready) return 1;
    if (!d.bindless) {
        std::printf("SKIP: the device lacks the descriptor indexing features of the bindless set 0\n");
        return 77;
    }

    VkPhysicalDeviceProperties phys_props{};
    if (auto get_props = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(
            d.instance_proc(d.instance, "vkGetPhysicalDeviceProperties"))) {
        get_props(d.physical, &phys_props);
    }

    const std::filesystem::path root_dir = LEONIDA_TEST_M4_DIR;
    leonida::ShaderLibrary library;
    EXPECT(library.load((root_dir / "spirv").string()));
    const auto program = [&](const char* name) { return read_file(root_dir / "fssm" / (std::string(name) + ".fssm")); };
    const auto fill = program("cs_m4_fill"), indirect = program("cs_m4_indirect");
    EXPECT(fill.size() >= 0x150 && indirect.size() >= 0x150 && fill != indirect);
    EXPECT(library.find("cs", leonida::sha256_hex(fill)) && library.find("cs", leonida::sha256_hex(indirect)));
    const auto fill_container = shader_container(fill), indirect_container = shader_container(indirect);

    bool skipped_for_push_constants = false;
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        if (d.queues.size() >= 2) info.command_queues = d.queues;
        info.buffer_device_address = true;
        info.bindless = d.bindless;
        info.bindless_partially_bound = d.partially_bound;
        info.image_view_min_lod = d.image_view_min_lod;
        info.shader_dir = (root_dir / "spirv").string();

        ID3D12Device* inner = nullptr;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                           reinterpret_cast<void**>(&device.p))));
        rt::d3d12::set_device_outer(inner, device.p);
        EXPECT(rt::d3d12::bindless_set(device.p) != VK_NULL_HANDLE);

        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
        const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
        constexpr auto uav_flag = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        constexpr auto uav_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        constexpr auto common = D3D12_RESOURCE_STATE_COMMON;

        const auto create_buffer = [&](Ref<ID3D12Resource>& out, const D3D12_HEAP_PROPERTIES& heap, UINT64 bytes,
                                       D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state) {
            const auto desc = buffer_desc(bytes, flags);
            return SUCCEEDED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                                             IID_PPV_ARGS(&out.p)));
        };
        const auto create_tex = [&](Ref<ID3D12Resource>& out, D3D12_RESOURCE_STATES state, UINT16 mips = 1) {
            const auto desc = tex2d_uav_desc(image_dim, image_dim, mips);
            return SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                                             IID_PPV_ARGS(&out.p)));
        };

        // 30 slots = 6 tables of 5 UAVs each.
        const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 30,
                                                   D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        Ref<ID3D12DescriptorHeap> heap;
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap.p))));
        const std::uint64_t cpu_base = reinterpret_cast<XboxHeapHandleFn>(slot(heap.p, 9))(heap.p);
        const std::uint64_t gpu_base = reinterpret_cast<XboxHeapHandleFn>(slot(heap.p, 10))(heap.p);
        const UINT step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        EXPECT(cpu_base && gpu_base >= 0x10000 && gpu_base < (1ull << 32) && step == 32);
        const auto cpu_handle = [&](UINT i) {
            return D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(cpu_base + std::uint64_t(i) * step)};
        };
        const auto gpu_handle = [&](UINT i) {
            return D3D12_GPU_DESCRIPTOR_HANDLE{gpu_base + std::uint64_t(i) * step};
        };

        D3D12_COMMAND_QUEUE_DESC qdesc{};
        Ref<ID3D12CommandQueue> copy_queue, compute_queue, direct_queue;
        qdesc.Type = D3D12_COMMAND_LIST_TYPE_COPY;
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&qdesc, IID_PPV_ARGS(&copy_queue.p))));
        qdesc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&qdesc, IID_PPV_ARGS(&compute_queue.p))));
        qdesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&qdesc, IID_PPV_ARGS(&direct_queue.p))));

        const auto make_list = [&](D3D12_COMMAND_LIST_TYPE type, Ref<ID3D12CommandAllocator>& alloc,
                                   Ref<ID3D12GraphicsCommandList>& list) {
            return SUCCEEDED(device->CreateCommandAllocator(type, IID_PPV_ARGS(&alloc.p))) &&
                   SUCCEEDED(device->CreateCommandList(0, type, alloc.p, nullptr, IID_PPV_ARGS(&list.p)));
        };
        const auto execute = [&](ID3D12CommandQueue* queue, ID3D12GraphicsCommandList* list) {
            ID3D12CommandList* lists[] = {list};
            queue->ExecuteCommandLists(1, lists);
            return rt::d3d12::last_execute_result(queue) == leonida::ReplayResult::success;
        };

        Ref<ID3D12Fence> direct_fence, compute_fence, copy_fence;
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&direct_fence.p))));
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&compute_fence.p))));
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&copy_fence.p))));

        // =========================================================================
        // Subtest 2 (part 1, non-shader): barrier + copy readback before any Dispatch.
        // Runs before compute PSO creation so drivers with <144B push constants still verify it.
        // =========================================================================
        Ref<ID3D12Resource> img2, rb2_zero;
        EXPECT(create_tex(img2, uav_state));
        EXPECT(create_buffer(rb2_zero, readback_heap, image_copy_bytes, D3D12_RESOURCE_FLAG_NONE,
                             D3D12_RESOURCE_STATE_COPY_DEST));
        // Create UAV for img2 (table 1, slot 2 = index 7) before any dispatch exists.
        {
            const auto tv = tex2d_uav_view(0);
            device->CreateUnorderedAccessView(img2.p, nullptr, &tv, cpu_handle(7));
        }
        {
            Ref<ID3D12CommandAllocator> alloc;
            Ref<ID3D12GraphicsCommandList> list;
            EXPECT(make_list(D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, list));
            D3D12_RESOURCE_BARRIER to_src = transition(img2.p, uav_state, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->ResourceBarrier(1, &to_src);
            D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
            dst.pResource = rb2_zero.p;
            dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            dst.PlacedFootprint = {0, {DXGI_FORMAT_R32_UINT, image_dim, image_dim, 1, image_pitch}};
            src.pResource = img2.p;
            src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            D3D12_RESOURCE_BARRIER to_uav = transition(img2.p, D3D12_RESOURCE_STATE_COPY_SOURCE, uav_state);
            list->ResourceBarrier(1, &to_uav);
            EXPECT(list->Close() == S_OK);
            EXPECT(execute(direct_queue.p, list.p));
            EXPECT(SUCCEEDED(direct_queue->Signal(direct_fence.p, 1)));
            EXPECT(wait_fence(direct_fence.p, 1));
        }
        {
            void* mapped = nullptr;
            EXPECT(SUCCEEDED(rb2_zero->Map(0, nullptr, &mapped)));
            const std::vector<std::uint32_t> zeros(records, 0u);
            EXPECT(check_image_rows("subtest2a zero-init readback", static_cast<const std::uint8_t*>(mapped),
                                    image_dim, image_dim, zeros));
            rb2_zero->Unmap(0, nullptr);
        }
        rb2_zero.reset();

        // Root signature + compute PSOs for the shader subtests.
        D3D12_DESCRIPTOR_RANGE uav_range{D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 5, 0, 0, 0};
        D3D12_ROOT_PARAMETER params[2] = {};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[0].Constants = {0, 0, 4};
        params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[1].DescriptorTable = {1, &uav_range};
        params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
        root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
        root_desc.Desc_1_0 = {2, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        const auto root_blob = serialize_root_signature(root_desc);
        EXPECT(!root_blob.empty());
        Ref<ID3D12RootSignature> root;
        EXPECT(SUCCEEDED(device->CreateRootSignature(0, root_blob.data(), root_blob.size(), IID_PPV_ARGS(&root.p))));

        D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc{};
        pso_desc.pRootSignature = root.p;
        pso_desc.CS = {fill_container.data(), fill_container.size()};
        Ref<ID3D12PipelineState> pso_fill, pso_indirect;
        const HRESULT hr_fill = device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&pso_fill.p));
        pso_desc.CS = {indirect_container.data(), indirect_container.size()};
        const HRESULT hr_indirect = SUCCEEDED(hr_fill)
                                        ? device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&pso_indirect.p))
                                        : hr_fill;
        if (FAILED(hr_fill) || FAILED(hr_indirect)) {
            if (phys_props.limits.maxPushConstantsSize < 144) {
                std::printf("SKIP: non-shader storage-image barrier+copy init passed; skipping compute shader "
                            "subtests because device maxPushConstantsSize=%u < 144\n",
                            phys_props.limits.maxPushConstantsSize);
                skipped_for_push_constants = true;
            } else {
                std::fprintf(stderr, "CreateComputePipelineState failed (hr_fill=0x%08lx hr_indirect=0x%08lx)\n",
                             static_cast<unsigned long>(hr_fill), static_cast<unsigned long>(hr_indirect));
                return 1;
            }
        }

        if (!skipped_for_push_constants) {
            // Shared helper buffers for cs_m4_fill / cs_m4_indirect. Sized to match each UAV slot's raw view so
            // unused slots never trigger rejected-UAV diagnostics from the runtime.
            Ref<ID3D12Resource> zero_data, scratch_data, scratch_args, scratch_in, scratch_out;
            EXPECT(create_buffer(zero_data, default_heap, data_bytes, uav_flag, uav_state));
            EXPECT(create_buffer(scratch_data, default_heap, data_bytes, uav_flag, uav_state));
            EXPECT(create_buffer(scratch_args, default_heap, args_bytes, uav_flag, uav_state));
            EXPECT(create_buffer(scratch_in, default_heap, in_bytes, uav_flag, uav_state));
            EXPECT(create_buffer(scratch_out, default_heap, out_bytes, uav_flag, uav_state));

            const auto fill_table = [&](UINT base_slot, ID3D12Resource* data_buf, ID3D12Resource* args_buf,
                                        ID3D12Resource* img, UINT mip_slice, ID3D12Resource* in_buf,
                                        ID3D12Resource* out_buf) {
                auto v0 = buffer_uav_desc(DXGI_FORMAT_UNKNOWN, records, 16, D3D12_BUFFER_UAV_FLAG_NONE);
                device->CreateUnorderedAccessView(data_buf, nullptr, &v0, cpu_handle(base_slot + 0));
                auto v1 = buffer_uav_desc(DXGI_FORMAT_R32_TYPELESS, 4, 0, D3D12_BUFFER_UAV_FLAG_RAW);
                device->CreateUnorderedAccessView(args_buf, nullptr, &v1, cpu_handle(base_slot + 1));
                auto v2 = tex2d_uav_view(mip_slice);
                device->CreateUnorderedAccessView(img, nullptr, &v2, cpu_handle(base_slot + 2));
                auto v3 = buffer_uav_desc(DXGI_FORMAT_R32_TYPELESS, records, 0, D3D12_BUFFER_UAV_FLAG_RAW);
                device->CreateUnorderedAccessView(in_buf, nullptr, &v3, cpu_handle(base_slot + 3));
                auto v4 = buffer_uav_desc(DXGI_FORMAT_R32_TYPELESS, UINT(out_bytes / 4), 0, D3D12_BUFFER_UAV_FLAG_RAW);
                device->CreateUnorderedAccessView(out_buf, nullptr, &v4, cpu_handle(base_slot + 4));
            };

            // =========================================================================
            // Subtest 1: Copy-queue upload before first Dispatch (implicit COMMON -> UAV).
            // =========================================================================
            {
                Ref<ID3D12Resource> img1, up1, out1, rb1;
                EXPECT(create_tex(img1, common));
                EXPECT(create_buffer(up1, upload_heap, image_copy_bytes, D3D12_RESOURCE_FLAG_NONE,
                                     D3D12_RESOURCE_STATE_GENERIC_READ));
                EXPECT(create_buffer(out1, default_heap, out_bytes, uav_flag, uav_state));
                EXPECT(create_buffer(rb1, readback_heap, out_bytes + image_copy_bytes, D3D12_RESOURCE_FLAG_NONE,
                                     D3D12_RESOURCE_STATE_COPY_DEST));

                std::vector<std::uint32_t> pat1(records);
                {
                    std::uint8_t* mapped = nullptr;
                    EXPECT(SUCCEEDED(up1->Map(0, nullptr, reinterpret_cast<void**>(&mapped))));
                    std::memset(mapped, 0, image_copy_bytes);
                    for (UINT g = 0; g < records; ++g) {
                        pat1[g] = pattern_word(g, 0x13579bdfu);
                        const UINT x = g & 15u, y = g >> 4;
                        std::memcpy(mapped + UINT64(y) * image_pitch + UINT64(x) * 4, &pat1[g], 4);
                    }
                    up1->Unmap(0, nullptr);
                }
                // Table 0: slot 0 = zero_data, slot 2 = img1, slot 4 = out1
                fill_table(0, zero_data.p, scratch_args.p, img1.p, 0, scratch_in.p, out1.p);

                Ref<ID3D12CommandAllocator> up_alloc, cs_alloc, rb_alloc;
                Ref<ID3D12GraphicsCommandList> up_list, cs_list, rb_list;
                EXPECT(make_list(D3D12_COMMAND_LIST_TYPE_COPY, up_alloc, up_list));
                EXPECT(make_list(D3D12_COMMAND_LIST_TYPE_COMPUTE, cs_alloc, cs_list));
                EXPECT(make_list(D3D12_COMMAND_LIST_TYPE_COPY, rb_alloc, rb_list));

                // Copy queue uploads pat1 into img1 (created in COMMON).
                D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
                dst.pResource = img1.p;
                dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                src.pResource = up1.p;
                src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                src.PlacedFootprint = {0, {DXGI_FORMAT_R32_UINT, image_dim, image_dim, 1, image_pitch}};
                up_list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
                EXPECT(up_list->Close() == S_OK);

                // Compute queue reads img1 via cs_m4_indirect with NO barrier (implicit COMMON -> UAV).
                ID3D12DescriptorHeap* heaps[] = {heap.p};
                cs_list->SetDescriptorHeaps(1, heaps);
                cs_list->SetComputeRootSignature(root.p);
                cs_list->SetPipelineState(pso_indirect.p);
                const std::uint32_t c0[4] = {0, 0, 0, 0};
                cs_list->SetComputeRoot32BitConstants(0, 4, c0, 0);
                cs_list->SetComputeRootDescriptorTable(1, gpu_handle(0));
                cs_list->Dispatch(records / 64, 1, 1);
                D3D12_RESOURCE_BARRIER rel[2] = {transition(out1.p, uav_state, common),
                                                 transition(img1.p, uav_state, common)};
                cs_list->ResourceBarrier(2, rel);
                EXPECT(cs_list->Close() == S_OK);

                // Copy queue reads back out1 and img1.
                rb_list->CopyBufferRegion(rb1.p, 0, out1.p, 0, out_bytes);
                D3D12_TEXTURE_COPY_LOCATION rdst{}, rsrc{};
                rdst.pResource = rb1.p;
                rdst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                rdst.PlacedFootprint = {out_bytes, {DXGI_FORMAT_R32_UINT, image_dim, image_dim, 1, image_pitch}};
                rsrc.pResource = img1.p;
                rsrc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                rb_list->CopyTextureRegion(&rdst, 0, 0, 0, &rsrc, nullptr);
                EXPECT(rb_list->Close() == S_OK);

                EXPECT(execute(copy_queue.p, up_list.p) && SUCCEEDED(copy_queue->Signal(copy_fence.p, 1)));
                EXPECT(SUCCEEDED(compute_queue->Wait(copy_fence.p, 1)) && execute(compute_queue.p, cs_list.p) &&
                       SUCCEEDED(compute_queue->Signal(compute_fence.p, 1)));
                EXPECT(SUCCEEDED(copy_queue->Wait(compute_fence.p, 1)) && execute(copy_queue.p, rb_list.p) &&
                       SUCCEEDED(copy_queue->Signal(copy_fence.p, 2)));
                EXPECT(wait_fence(copy_fence.p, 2));

                void* mapped = nullptr;
                EXPECT(SUCCEEDED(rb1->Map(0, nullptr, &mapped)));
                const auto* bytes = static_cast<const std::uint8_t*>(mapped);
                EXPECT(check_indirect_out_matches_image("subtest1 cs_m4_indirect output", bytes, pat1));
                EXPECT(check_image_rows("subtest1 image readback", bytes + out_bytes, image_dim, image_dim, pat1));
                rb1->Unmap(0, nullptr);
            }

            // =========================================================================
            // Subtest 2 (part 2): Compute dispatch on img2 after shader-less barrier+copy readback.
            // =========================================================================
            {
                Ref<ID3D12Resource> up_in2, in2, rb2;
                EXPECT(create_buffer(up_in2, upload_heap, in_bytes, D3D12_RESOURCE_FLAG_NONE,
                                     D3D12_RESOURCE_STATE_GENERIC_READ));
                EXPECT(create_buffer(in2, default_heap, in_bytes, uav_flag, common));
                EXPECT(create_buffer(rb2, readback_heap, image_copy_bytes, D3D12_RESOURCE_FLAG_NONE,
                                     D3D12_RESOURCE_STATE_COPY_DEST));

                std::vector<std::uint32_t> want2(records);
                {
                    std::uint32_t* mapped = nullptr;
                    EXPECT(SUCCEEDED(up_in2->Map(0, nullptr, reinterpret_cast<void**>(&mapped))));
                    for (UINT g = 0; g < records; ++g) {
                        mapped[g] = pattern_word(g, 0x2468ace0u);
                        want2[g] = fill_image_word(mapped[g], g);
                    }
                    up_in2->Unmap(0, nullptr);
                }
                // Table 1 (slots 5..9): slot 2 (index 7) already holds img2's UAV; bind the rest.
                fill_table(5, scratch_data.p, scratch_args.p, img2.p, 0, in2.p, scratch_out.p);

                Ref<ID3D12CommandAllocator> up_alloc, cs_alloc, rb_alloc;
                Ref<ID3D12GraphicsCommandList> up_list, cs_list, rb_list;
                EXPECT(make_list(D3D12_COMMAND_LIST_TYPE_COPY, up_alloc, up_list));
                EXPECT(make_list(D3D12_COMMAND_LIST_TYPE_COMPUTE, cs_alloc, cs_list));
                EXPECT(make_list(D3D12_COMMAND_LIST_TYPE_COPY, rb_alloc, rb_list));

                up_list->CopyBufferRegion(in2.p, 0, up_in2.p, 0, in_bytes);
                EXPECT(up_list->Close() == S_OK);

                ID3D12DescriptorHeap* heaps[] = {heap.p};
                D3D12_RESOURCE_BARRIER acq = transition(in2.p, common, uav_state);
                cs_list->ResourceBarrier(1, &acq);
                cs_list->SetDescriptorHeaps(1, heaps);
                cs_list->SetComputeRootSignature(root.p);
                cs_list->SetPipelineState(pso_fill.p);
                const std::uint32_t c2[4] = {0x1111u, 0x2222u, 1u, 0u};
                cs_list->SetComputeRoot32BitConstants(0, 4, c2, 0);
                cs_list->SetComputeRootDescriptorTable(1, gpu_handle(5));
                cs_list->Dispatch(records / 64, 1, 1);
                D3D12_RESOURCE_BARRIER rel = transition(img2.p, uav_state, common);
                cs_list->ResourceBarrier(1, &rel);
                EXPECT(cs_list->Close() == S_OK);

                D3D12_TEXTURE_COPY_LOCATION rdst{}, rsrc{};
                rdst.pResource = rb2.p;
                rdst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                rdst.PlacedFootprint = {0, {DXGI_FORMAT_R32_UINT, image_dim, image_dim, 1, image_pitch}};
                rsrc.pResource = img2.p;
                rsrc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                rb_list->CopyTextureRegion(&rdst, 0, 0, 0, &rsrc, nullptr);
                EXPECT(rb_list->Close() == S_OK);

                EXPECT(execute(copy_queue.p, up_list.p) && SUCCEEDED(copy_queue->Signal(copy_fence.p, 3)));
                EXPECT(SUCCEEDED(compute_queue->Wait(copy_fence.p, 3)) && execute(compute_queue.p, cs_list.p) &&
                       SUCCEEDED(compute_queue->Signal(compute_fence.p, 2)));
                EXPECT(SUCCEEDED(copy_queue->Wait(compute_fence.p, 2)) && execute(copy_queue.p, rb_list.p) &&
                       SUCCEEDED(copy_queue->Signal(copy_fence.p, 4)));
                EXPECT(wait_fence(copy_fence.p, 4));

                void* mapped = nullptr;
                EXPECT(SUCCEEDED(rb2->Map(0, nullptr, &mapped)));
                EXPECT(check_image_rows("subtest2b compute fill readback", static_cast<const std::uint8_t*>(mapped),
                                        image_dim, image_dim, want2));
                rb2->Unmap(0, nullptr);
            }

            // =========================================================================
            // Subtest 3: Cross-queue independent initializers (COMPUTE -> ImgA, DIRECT -> ImgB).
            // =========================================================================
            {
                Ref<ID3D12Resource> imgA, imgB, up_inA, up_inB, inA, inB, dataA, dataB, argsA, argsB, rb3;
                EXPECT(create_tex(imgA, uav_state));
                EXPECT(create_tex(imgB, uav_state));
                EXPECT(create_buffer(up_inA, upload_heap, in_bytes, D3D12_RESOURCE_FLAG_NONE,
                                     D3D12_RESOURCE_STATE_GENERIC_READ));
                EXPECT(create_buffer(up_inB, upload_heap, in_bytes, D3D12_RESOURCE_FLAG_NONE,
                                     D3D12_RESOURCE_STATE_GENERIC_READ));
                EXPECT(create_buffer(inA, default_heap, in_bytes, uav_flag, common));
                EXPECT(create_buffer(inB, default_heap, in_bytes, uav_flag, common));
                EXPECT(create_buffer(dataA, default_heap, data_bytes, uav_flag, uav_state));
                EXPECT(create_buffer(dataB, default_heap, data_bytes, uav_flag, uav_state));
                EXPECT(create_buffer(argsA, default_heap, args_bytes, uav_flag, uav_state));
                EXPECT(create_buffer(argsB, default_heap, args_bytes, uav_flag, uav_state));
                EXPECT(create_buffer(rb3, readback_heap, 2 * image_copy_bytes, D3D12_RESOURCE_FLAG_NONE,
                                     D3D12_RESOURCE_STATE_COPY_DEST));

                std::vector<std::uint32_t> wantA(records), wantB(records);
                {
                    std::uint32_t* mA = nullptr;
                    std::uint32_t* mB = nullptr;
                    EXPECT(SUCCEEDED(up_inA->Map(0, nullptr, reinterpret_cast<void**>(&mA))));
                    EXPECT(SUCCEEDED(up_inB->Map(0, nullptr, reinterpret_cast<void**>(&mB))));
                    for (UINT g = 0; g < records; ++g) {
                        mA[g] = pattern_word(g, 0xaaaa1111u);
                        mB[g] = pattern_word(g, 0xbbbb2222u);
                        wantA[g] = fill_image_word(mA[g], g);
                        wantB[g] = fill_image_word(mB[g], g);
                    }
                    up_inA->Unmap(0, nullptr);
                    up_inB->Unmap(0, nullptr);
                }

                // Upload both input buffers on COPY queue first.
                {
                    Ref<ID3D12CommandAllocator> up_alloc;
                    Ref<ID3D12GraphicsCommandList> up_list;
                    EXPECT(make_list(D3D12_COMMAND_LIST_TYPE_COPY, up_alloc, up_list));
                    up_list->CopyBufferRegion(inA.p, 0, up_inA.p, 0, in_bytes);
                    up_list->CopyBufferRegion(inB.p, 0, up_inB.p, 0, in_bytes);
                    EXPECT(up_list->Close() == S_OK);
                    EXPECT(execute(copy_queue.p, up_list.p) && SUCCEEDED(copy_queue->Signal(copy_fence.p, 5)));
                }

                // Create UAVs for ImgA (table 2: slots 10..14) and dispatch on COMPUTE queue.
                fill_table(10, dataA.p, argsA.p, imgA.p, 0, inA.p, scratch_out.p);
                {
                    Ref<ID3D12CommandAllocator> cs_alloc;
                    Ref<ID3D12GraphicsCommandList> cs_list;
                    EXPECT(make_list(D3D12_COMMAND_LIST_TYPE_COMPUTE, cs_alloc, cs_list));
                    ID3D12DescriptorHeap* heaps[] = {heap.p};
                    D3D12_RESOURCE_BARRIER acq = transition(inA.p, common, uav_state);
                    cs_list->ResourceBarrier(1, &acq);
                    cs_list->SetDescriptorHeaps(1, heaps);
                    cs_list->SetComputeRootSignature(root.p);
                    cs_list->SetPipelineState(pso_fill.p);
                    const std::uint32_t cA[4] = {0x10u, 0x20u, 1u, 0u};
                    cs_list->SetComputeRoot32BitConstants(0, 4, cA, 0);
                    cs_list->SetComputeRootDescriptorTable(1, gpu_handle(10));
                    cs_list->Dispatch(records / 64, 1, 1);
                    D3D12_RESOURCE_BARRIER rel = transition(imgA.p, uav_state, common);
                    cs_list->ResourceBarrier(1, &rel);
                    EXPECT(cs_list->Close() == S_OK);
                    EXPECT(SUCCEEDED(compute_queue->Wait(copy_fence.p, 5)) && execute(compute_queue.p, cs_list.p) &&
                           SUCCEEDED(compute_queue->Signal(compute_fence.p, 3)));
                }

                // Create UAVs for ImgB (table 3: slots 15..19) and dispatch on DIRECT queue (first shader on DIRECT).
                fill_table(15, dataB.p, argsB.p, imgB.p, 0, inB.p, scratch_out.p);
                {
                    Ref<ID3D12CommandAllocator> dir_alloc;
                    Ref<ID3D12GraphicsCommandList> dir_list;
                    EXPECT(make_list(D3D12_COMMAND_LIST_TYPE_DIRECT, dir_alloc, dir_list));
                    ID3D12DescriptorHeap* heaps[] = {heap.p};
                    D3D12_RESOURCE_BARRIER acq = transition(inB.p, common, uav_state);
                    dir_list->ResourceBarrier(1, &acq);
                    dir_list->SetDescriptorHeaps(1, heaps);
                    dir_list->SetComputeRootSignature(root.p);
                    dir_list->SetPipelineState(pso_fill.p);
                    const std::uint32_t cB[4] = {0x30u, 0x40u, 1u, 0u};
                    dir_list->SetComputeRoot32BitConstants(0, 4, cB, 0);
                    dir_list->SetComputeRootDescriptorTable(1, gpu_handle(15));
                    dir_list->Dispatch(records / 64, 1, 1);
                    D3D12_RESOURCE_BARRIER rel = transition(imgB.p, uav_state, common);
                    dir_list->ResourceBarrier(1, &rel);
                    EXPECT(dir_list->Close() == S_OK);
                    EXPECT(SUCCEEDED(direct_queue->Wait(copy_fence.p, 5)) && execute(direct_queue.p, dir_list.p) &&
                           SUCCEEDED(direct_queue->Signal(direct_fence.p, 2)));
                }

                // Copy queue waits on both COMPUTE and DIRECT fences and reads back both ImgA and ImgB.
                {
                    Ref<ID3D12CommandAllocator> rb_alloc;
                    Ref<ID3D12GraphicsCommandList> rb_list;
                    EXPECT(make_list(D3D12_COMMAND_LIST_TYPE_COPY, rb_alloc, rb_list));
                    D3D12_TEXTURE_COPY_LOCATION dA{}, sA{}, dB{}, sB{};
                    dA.pResource = rb3.p;
                    dA.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                    dA.PlacedFootprint = {0, {DXGI_FORMAT_R32_UINT, image_dim, image_dim, 1, image_pitch}};
                    sA.pResource = imgA.p;
                    sA.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                    rb_list->CopyTextureRegion(&dA, 0, 0, 0, &sA, nullptr);

                    dB.pResource = rb3.p;
                    dB.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                    dB.PlacedFootprint = {image_copy_bytes, {DXGI_FORMAT_R32_UINT, image_dim, image_dim, 1, image_pitch}};
                    sB.pResource = imgB.p;
                    sB.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                    rb_list->CopyTextureRegion(&dB, 0, 0, 0, &sB, nullptr);
                    EXPECT(rb_list->Close() == S_OK);

                    EXPECT(SUCCEEDED(copy_queue->Wait(compute_fence.p, 3)) &&
                           SUCCEEDED(copy_queue->Wait(direct_fence.p, 2)) &&
                           execute(copy_queue.p, rb_list.p) &&
                           SUCCEEDED(copy_queue->Signal(copy_fence.p, 6)));
                    EXPECT(wait_fence(copy_fence.p, 6));
                }

                void* mapped = nullptr;
                EXPECT(SUCCEEDED(rb3->Map(0, nullptr, &mapped)));
                const auto* bytes = static_cast<const std::uint8_t*>(mapped);
                EXPECT(check_image_rows("subtest3 ImgA readback", bytes, image_dim, image_dim, wantA));
                EXPECT(check_image_rows("subtest3 ImgB readback", bytes + image_copy_bytes, image_dim, image_dim, wantB));
                rb3->Unmap(0, nullptr);
            }

            // =========================================================================
            // Subtest 4: Partial subresource upload (2 mips) before Dispatch.
            // =========================================================================
            {
                Ref<ID3D12Resource> img4, up4, out4_m0, out4_m1, rb4;
                EXPECT(create_tex(img4, common, 2));
                EXPECT(create_buffer(up4, upload_heap, image_copy_bytes, D3D12_RESOURCE_FLAG_NONE,
                                     D3D12_RESOURCE_STATE_GENERIC_READ));
                EXPECT(create_buffer(out4_m0, default_heap, out_bytes, uav_flag, uav_state));
                EXPECT(create_buffer(out4_m1, default_heap, out_bytes, uav_flag, uav_state));
                const UINT64 off_m0 = out_bytes;
                const UINT64 off_m1 = off_m0 + image_copy_bytes;
                EXPECT(create_buffer(rb4, readback_heap, off_m1 + mip1_copy_bytes, D3D12_RESOURCE_FLAG_NONE,
                                     D3D12_RESOURCE_STATE_COPY_DEST));

                std::vector<std::uint32_t> pat4(records);
                {
                    std::uint8_t* mapped = nullptr;
                    EXPECT(SUCCEEDED(up4->Map(0, nullptr, reinterpret_cast<void**>(&mapped))));
                    std::memset(mapped, 0, image_copy_bytes);
                    for (UINT g = 0; g < records; ++g) {
                        pat4[g] = pattern_word(g, 0x55aa33ccu);
                        const UINT x = g & 15u, y = g >> 4;
                        std::memcpy(mapped + UINT64(y) * image_pitch + UINT64(x) * 4, &pat4[g], 4);
                    }
                    up4->Unmap(0, nullptr);
                }

                // Table 4 (slots 20..24): mip 0 UAV; Table 5 (slots 25..29): mip 1 UAV.
                fill_table(20, zero_data.p, scratch_args.p, img4.p, 0, scratch_in.p, out4_m0.p);
                fill_table(25, zero_data.p, scratch_args.p, img4.p, 1, scratch_in.p, out4_m1.p);

                Ref<ID3D12CommandAllocator> up_alloc, cs_alloc, rb_alloc;
                Ref<ID3D12GraphicsCommandList> up_list, cs_list, rb_list;
                EXPECT(make_list(D3D12_COMMAND_LIST_TYPE_COPY, up_alloc, up_list));
                EXPECT(make_list(D3D12_COMMAND_LIST_TYPE_COMPUTE, cs_alloc, cs_list));
                EXPECT(make_list(D3D12_COMMAND_LIST_TYPE_COPY, rb_alloc, rb_list));

                // Upload mip 0 ONLY on COPY queue.
                D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
                dst.pResource = img4.p;
                dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                dst.SubresourceIndex = 0;
                src.pResource = up4.p;
                src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                src.PlacedFootprint = {0, {DXGI_FORMAT_R32_UINT, image_dim, image_dim, 1, image_pitch}};
                up_list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
                EXPECT(up_list->Close() == S_OK);

                // Compute queue reads mip 0 (already in GENERAL with uploaded pattern) and mip 1 (zero-initialized
                // into GENERAL without clobbering mip 0).
                ID3D12DescriptorHeap* heaps[] = {heap.p};
                cs_list->SetDescriptorHeaps(1, heaps);
                cs_list->SetComputeRootSignature(root.p);
                cs_list->SetPipelineState(pso_indirect.p);
                const std::uint32_t c4[4] = {0, 0, 0, 0};
                cs_list->SetComputeRoot32BitConstants(0, 4, c4, 0);
                cs_list->SetComputeRootDescriptorTable(1, gpu_handle(20));
                cs_list->Dispatch(records / 64, 1, 1);
                cs_list->SetComputeRootDescriptorTable(1, gpu_handle(25));
                cs_list->Dispatch(1, 1, 1);
                D3D12_RESOURCE_BARRIER rel[2] = {transition(out4_m0.p, uav_state, common),
                                                 transition(img4.p, uav_state, common)};
                cs_list->ResourceBarrier(2, rel);
                EXPECT(cs_list->Close() == S_OK);

                // Read back out4_m0, mip 0, and mip 1 on COPY queue.
                rb_list->CopyBufferRegion(rb4.p, 0, out4_m0.p, 0, out_bytes);
                D3D12_TEXTURE_COPY_LOCATION d0{}, s0{}, d1{}, s1{};
                d0.pResource = rb4.p;
                d0.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                d0.PlacedFootprint = {off_m0, {DXGI_FORMAT_R32_UINT, image_dim, image_dim, 1, image_pitch}};
                s0.pResource = img4.p;
                s0.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                s0.SubresourceIndex = 0;
                rb_list->CopyTextureRegion(&d0, 0, 0, 0, &s0, nullptr);

                d1.pResource = rb4.p;
                d1.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                d1.PlacedFootprint = {off_m1, {DXGI_FORMAT_R32_UINT, mip1_dim, mip1_dim, 1, image_pitch}};
                s1.pResource = img4.p;
                s1.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                s1.SubresourceIndex = 1;
                rb_list->CopyTextureRegion(&d1, 0, 0, 0, &s1, nullptr);
                EXPECT(rb_list->Close() == S_OK);

                EXPECT(execute(copy_queue.p, up_list.p) && SUCCEEDED(copy_queue->Signal(copy_fence.p, 7)));
                EXPECT(SUCCEEDED(compute_queue->Wait(copy_fence.p, 7)) && execute(compute_queue.p, cs_list.p) &&
                       SUCCEEDED(compute_queue->Signal(compute_fence.p, 4)));
                EXPECT(SUCCEEDED(copy_queue->Wait(compute_fence.p, 4)) && execute(copy_queue.p, rb_list.p) &&
                       SUCCEEDED(copy_queue->Signal(copy_fence.p, 8)));
                EXPECT(wait_fence(copy_fence.p, 8));

                void* mapped = nullptr;
                EXPECT(SUCCEEDED(rb4->Map(0, nullptr, &mapped)));
                const auto* bytes = static_cast<const std::uint8_t*>(mapped);
                EXPECT(check_indirect_out_matches_image("subtest4 mip0 cs_m4_indirect output", bytes, pat4));
                EXPECT(check_image_rows("subtest4 mip0 readback", bytes + off_m0, image_dim, image_dim, pat4));
                const std::vector<std::uint32_t> mip1_zeros(mip1_dim * mip1_dim, 0u);
                EXPECT(check_image_rows("subtest4 mip1 zero-init readback", bytes + off_m1, mip1_dim, mip1_dim,
                                        mip1_zeros));
                rb4->Unmap(0, nullptr);
            }
        }

        img2.reset();
        pso_fill.reset();
        pso_indirect.reset();
        root.reset();
        heap.reset();
        direct_fence.reset();
        compute_fence.reset();
        copy_fence.reset();
        direct_queue.reset();
        compute_queue.reset();
        copy_queue.reset();
        EXPECT(rt::d3d12_xs::device_proxy_ref_count(device.p) == 1);
    }

    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    if (skipped_for_push_constants) return 77;
    std::printf("d3d12_storage_init: CPU replay contract + all 4 GPU subtests passed (%zu VkQueues, 0 validation errors)\n",
                d.queues.size());
    return 0;
}
