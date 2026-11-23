// Typeless and castable 3D volume SRV descriptor and GPU readback test.
// Exercises:
//   - Typeless 3D volume families (R8G8B8A8, R16G16B16A16, R32G32B32A32, R32, R16, R8, B8G8R8A8)
//     viewed through typed SRVs in each family.
//   - Relaxed castable-format 3D textures created via ID3D12Device10::CreateCommittedResource3.
//   - Negative controls (DXGI_FORMAT_UNKNOWN, typeless view formats, unlisted cross-family casts)
//     producing all-zero null descriptors.
//   - GPU upload, bindless set 0 flush, and texelFetch readback across float/unorm/srgb/snorm,
//     uint, sint, BGRA swizzle, and cross-family relaxed castable 3D views (when glslangValidator
//     compiled cs_volume_srv.comp).
// Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation.
#include "shader_fixture_path.h"
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../src/gfx10_descriptors.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <span>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { reset(); }
    void reset() {
        if (p) p->Release();
        p = nullptr;
    }
    T* operator->() const { return p; }
};

D3D12_RESOURCE_DESC volume_desc(DXGI_FORMAT format, UINT64 width = 4, UINT height = 4, UINT16 depth = 4,
                                UINT16 mips = 2) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = depth;
    desc.MipLevels = mips;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    return desc;
}

D3D12_RESOURCE_DESC1 volume_desc1(DXGI_FORMAT format, UINT64 width = 4, UINT height = 4, UINT16 depth = 4,
                                  UINT16 mips = 1) {
    D3D12_RESOURCE_DESC1 desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = depth;
    desc.MipLevels = mips;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    return desc;
}

D3D12_SHADER_RESOURCE_VIEW_DESC volume_srv_desc(DXGI_FORMAT format, UINT first_mip = 0, UINT mip_levels = 1) {
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = format;
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Texture3D = {first_mip, mip_levels, 0.0f};
    return srv;
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

D3D12_RESOURCE_BARRIER transition(ID3D12Resource* resource, D3D12_RESOURCE_STATES before,
                                  D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    return barrier;
}

float srgb_to_linear(std::uint8_t byte) {
    const float s = float(byte) / 255.0f;
    return s <= 0.04045f ? s / 12.92f : std::pow((s + 0.055f) / 1.055f, 2.4f);
}

bool near_f(float got, float expected, float eps = 1e-2f) {
    return std::fabs(got - expected) <= eps;
}

#ifdef LEONIDA_TEST_CS_VOLUME_SRV_SPV
struct FetchParams {
    std::uint32_t tsharp[8];
    std::uint32_t out_index;
    std::uint32_t mode;  // 0: float, 1: uint, 2: sint
    std::uint32_t x;
    std::uint32_t y;
    std::uint32_t z;
};

VkShaderModule load_shader(vktest::Device& d, const char* path) {
    std::ifstream file(vktest::shader_fixture_path(path), std::ios::binary | std::ios::ate);
    if (!file) return VK_NULL_HANDLE;
    std::vector<std::uint32_t> code(std::size_t(file.tellg()) / 4);
    file.seekg(0);
    if (code.empty() || !file.read(reinterpret_cast<char*>(code.data()), std::streamsize(code.size() * 4)))
        return VK_NULL_HANDLE;
    VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    info.codeSize = code.size() * 4;
    info.pCode = code.data();
    VkShaderModule module{};
    return d.fn<PFN_vkCreateShaderModule>("vkCreateShaderModule")(d.device, &info, nullptr, &module) == VK_SUCCESS
               ? module
               : VK_NULL_HANDLE;
}

bool fetch_volumes(vktest::Device& d, VkDescriptorSet set0, VkDescriptorSetLayout set0_layout,
                   std::span<const FetchParams> queries, std::vector<std::array<std::uint32_t, 4>>& out) {
    const VkDevice dev = d.device;
    void* mapped = nullptr;
    const VkDeviceSize byte_size = queries.size() * 16;
    const VkBuffer output = d.buffer(byte_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &mapped);
    if (!output) return false;

    const VkShaderModule module = load_shader(d, LEONIDA_TEST_CS_VOLUME_SRV_SPV);
    VkDescriptorSetLayout set1_layout = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorSet set1 = VK_NULL_HANDLE;

    const VkDescriptorSetLayoutBinding binding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
                                               nullptr};
    VkDescriptorSetLayoutCreateInfo set1_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    set1_info.bindingCount = 1;
    set1_info.pBindings = &binding;
    const VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;

    bool ok = module != VK_NULL_HANDLE &&
              d.fn<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout")(dev, &set1_info, nullptr,
                                                                                   &set1_layout) == VK_SUCCESS &&
              d.fn<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool")(dev, &pool_info, nullptr, &pool) == VK_SUCCESS;
    if (ok) {
        VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        alloc.descriptorPool = pool;
        alloc.descriptorSetCount = 1;
        alloc.pSetLayouts = &set1_layout;
        ok = d.fn<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets")(dev, &alloc, &set1) == VK_SUCCESS;
    }
    if (ok) {
        const VkDescriptorBufferInfo buf{output, 0, byte_size};
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = set1;
        write.dstBinding = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &buf;
        d.fn<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets")(dev, 1, &write, 0, nullptr);

        const VkDescriptorSetLayout layouts[2] = {set0_layout, set1_layout};
        const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(FetchParams)};
        VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout_info.setLayoutCount = 2;
        layout_info.pSetLayouts = layouts;
        layout_info.pushConstantRangeCount = 1;
        layout_info.pPushConstantRanges = &push;
        ok = d.fn<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(dev, &layout_info, nullptr, &layout) ==
             VK_SUCCESS;
    }
    if (ok) {
        VkComputePipelineCreateInfo pipe_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipe_info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT,
                           module, "main"};
        pipe_info.layout = layout;
        ok = d.fn<PFN_vkCreateComputePipelines>("vkCreateComputePipelines")(dev, VK_NULL_HANDLE, 1, &pipe_info, nullptr,
                                                                            &pipeline) == VK_SUCCESS;
    }
    if (ok) {
        const auto bind_sets = d.fn<PFN_vkCmdBindDescriptorSets>("vkCmdBindDescriptorSets");
        const auto push_constants = d.fn<PFN_vkCmdPushConstants>("vkCmdPushConstants");
        const auto dispatch = d.fn<PFN_vkCmdDispatch>("vkCmdDispatch");
        ok = d.run([&](VkCommandBuffer cb) {
            d.fn<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            const VkDescriptorSet sets[2] = {set0, set1};
            bind_sets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 2, sets, 0, nullptr);
            for (const FetchParams& q : queries) {
                push_constants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(q), &q);
                dispatch(cb, 1, 1, 1);
            }
            return true;
        });
    }
    if (ok) {
        out.resize(queries.size());
        std::memcpy(out.data(), mapped, byte_size);
    }
    if (pipeline) d.fn<PFN_vkDestroyPipeline>("vkDestroyPipeline")(dev, pipeline, nullptr);
    if (layout) d.fn<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(dev, layout, nullptr);
    if (pool) d.fn<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool")(dev, pool, nullptr);
    if (set1_layout) d.fn<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(dev, set1_layout, nullptr);
    if (module) d.fn<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(dev, module, nullptr);
    return ok;
}
#endif

}  // namespace

int main() {
    namespace g = leonida::gfx10;

    vktest::Device d;
    const auto status = d.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;

    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        info.texture_compression_bc = d.bc;
        info.bindless = d.bindless;
        info.bindless_partially_bound = d.partially_bound;
        info.image_view_min_lod = d.image_view_min_lod;
        info.sampler_filter_minmax = d.sampler_filter_minmax;
        info.custom_border_color = d.custom_border_color;
        info.custom_border_color_without_format = d.custom_border_color_without_format;

        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))));
        Ref<ID3D12Device10> device10;
        EXPECT(SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&device10.p))));

        const UINT step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        EXPECT(step >= sizeof(g::ImageWords));

        constexpr UINT heap_slots = 64;
        const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, heap_slots,
                                                   D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        Ref<ID3D12DescriptorHeap> heap;
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap.p))));
        const D3D12_CPU_DESCRIPTOR_HANDLE heap_start = heap->GetCPUDescriptorHandleForHeapStart();

        const auto slot_handle = [&](UINT i) {
            return D3D12_CPU_DESCRIPTOR_HANDLE{heap_start.ptr + SIZE_T(i) * step};
        };
        const auto slot_bytes = [&](UINT i) {
            return reinterpret_cast<std::uint8_t*>(heap_start.ptr + SIZE_T(i) * step);
        };
        const auto is_null_slot = [&](UINT i) {
            const std::uint8_t* p = slot_bytes(i);
            return std::all_of(p, p + step, [](std::uint8_t b) { return b == 0; });
        };
        const auto decode_slot = [&](UINT i, g::Image& out) {
            g::ImageWords words{};
            std::memcpy(words.data(), slot_bytes(i), sizeof(words));
            return g::decode(words, out);
        };
        const auto expect_null = [&](ID3D12Resource* res, const D3D12_SHADER_RESOURCE_VIEW_DESC* desc) {
            constexpr UINT scratch = heap_slots - 1;
            std::memset(slot_bytes(scratch), 0xA5, step);
            device->CreateShaderResourceView(res, desc, slot_handle(scratch));
            return is_null_slot(scratch);
        };

        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};

        // 1. Typeless 3D volume families and valid/invalid views.
        struct ViewExpect {
            DXGI_FORMAT dxgi;
            VkFormat vk;
            g::Swizzle swizzle;
        };
        struct FamilyCase {
            const char* name;
            DXGI_FORMAT typeless;
            std::vector<ViewExpect> valid;
            std::vector<DXGI_FORMAT> invalid;
        };
        const std::vector<FamilyCase> families = {
            {"R8G8B8A8_TYPELESS",
             DXGI_FORMAT_R8G8B8A8_TYPELESS,
             {{DXGI_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM, g::channels_rgba},
              {DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, VK_FORMAT_R8G8B8A8_SRGB, g::channels_rgba},
              {DXGI_FORMAT_R8G8B8A8_UINT, VK_FORMAT_R8G8B8A8_UINT, g::channels_rgba},
              {DXGI_FORMAT_R8G8B8A8_SINT, VK_FORMAT_R8G8B8A8_SINT, g::channels_rgba},
              {DXGI_FORMAT_R8G8B8A8_SNORM, VK_FORMAT_R8G8B8A8_SNORM, g::channels_rgba}},
             {DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_R8G8B8A8_TYPELESS, DXGI_FORMAT_R32_UINT,
              DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_B8G8R8A8_UNORM}},
            {"R16G16B16A16_TYPELESS",
             DXGI_FORMAT_R16G16B16A16_TYPELESS,
             {{DXGI_FORMAT_R16G16B16A16_FLOAT, VK_FORMAT_R16G16B16A16_SFLOAT, g::channels_rgba},
              {DXGI_FORMAT_R16G16B16A16_UNORM, VK_FORMAT_R16G16B16A16_UNORM, g::channels_rgba},
              {DXGI_FORMAT_R16G16B16A16_UINT, VK_FORMAT_R16G16B16A16_UINT, g::channels_rgba},
              {DXGI_FORMAT_R16G16B16A16_SINT, VK_FORMAT_R16G16B16A16_SINT, g::channels_rgba},
              {DXGI_FORMAT_R16G16B16A16_SNORM, VK_FORMAT_R16G16B16A16_SNORM, g::channels_rgba}},
             {DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_R16G16B16A16_TYPELESS, DXGI_FORMAT_R32G32_UINT,
              DXGI_FORMAT_R8G8B8A8_UNORM}},
            {"R32G32B32A32_TYPELESS",
             DXGI_FORMAT_R32G32B32A32_TYPELESS,
             {{DXGI_FORMAT_R32G32B32A32_FLOAT, VK_FORMAT_R32G32B32A32_SFLOAT, g::channels_rgba},
              {DXGI_FORMAT_R32G32B32A32_UINT, VK_FORMAT_R32G32B32A32_UINT, g::channels_rgba},
              {DXGI_FORMAT_R32G32B32A32_SINT, VK_FORMAT_R32G32B32A32_SINT, g::channels_rgba}},
             {DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_R32G32B32A32_TYPELESS, DXGI_FORMAT_R32_FLOAT}},
            {"R32_TYPELESS",
             DXGI_FORMAT_R32_TYPELESS,
             {{DXGI_FORMAT_R32_FLOAT, VK_FORMAT_R32_SFLOAT, g::channels_r},
              {DXGI_FORMAT_R32_UINT, VK_FORMAT_R32_UINT, g::channels_r},
              {DXGI_FORMAT_R32_SINT, VK_FORMAT_R32_SINT, g::channels_r}},
             {DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_R8G8B8A8_UINT}},
            {"R16_TYPELESS",
             DXGI_FORMAT_R16_TYPELESS,
             {{DXGI_FORMAT_R16_FLOAT, VK_FORMAT_R16_SFLOAT, g::channels_r},
              {DXGI_FORMAT_R16_UNORM, VK_FORMAT_R16_UNORM, g::channels_r},
              {DXGI_FORMAT_R16_UINT, VK_FORMAT_R16_UINT, g::channels_r}},
             {DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_R16_TYPELESS, DXGI_FORMAT_D16_UNORM, DXGI_FORMAT_R8G8_UINT}},
            {"R8_TYPELESS",
             DXGI_FORMAT_R8_TYPELESS,
             {{DXGI_FORMAT_R8_UNORM, VK_FORMAT_R8_UNORM, g::channels_r},
              {DXGI_FORMAT_R8_UINT, VK_FORMAT_R8_UINT, g::channels_r}},
             {DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_R8_TYPELESS, DXGI_FORMAT_R16_UINT}},
            {"B8G8R8A8_TYPELESS",
             DXGI_FORMAT_B8G8R8A8_TYPELESS,
             {{DXGI_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_B8G8R8A8_UNORM, g::channels_bgra},
              {DXGI_FORMAT_B8G8R8A8_UNORM_SRGB, VK_FORMAT_B8G8R8A8_SRGB, g::channels_bgra}},
             {DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_B8G8R8A8_TYPELESS, DXGI_FORMAT_B8G8R8X8_UNORM,
              DXGI_FORMAT_R8G8B8A8_UNORM}},
        };

        UINT total_valid_descriptors = 0;
        UINT total_null_descriptors = 0;
        for (const FamilyCase& fam : families) {
            const D3D12_RESOURCE_DESC desc = volume_desc(fam.typeless, 4, 4, 4, 2);
            Ref<ID3D12Resource> vol;
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                             D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                             IID_PPV_ARGS(&vol.p))));

            // Default SRV (nullptr) must produce a null descriptor on a typeless volume.
            EXPECT(expect_null(vol.p, nullptr));
            ++total_null_descriptors;

            for (const ViewExpect& v : fam.valid) {
                const auto srv = volume_srv_desc(v.dxgi, 0, 2);
                device->CreateShaderResourceView(vol.p, &srv, slot_handle(0));
                g::Image img{};
                EXPECT(decode_slot(0, img));
                EXPECT(img.type == g::ImageType::image_3d);
                EXPECT(img.width == 4 && img.height == 4 && img.depth == 4 && img.layer_count == 1);
                EXPECT(img.first_mip == 0 && img.mip_count == 2 && img.resource_mips == 2);
                EXPECT(img.format == g::format_from_vulkan(v.vk));
                EXPECT(img.swizzle == v.swizzle);
                ++total_valid_descriptors;
            }

            for (const DXGI_FORMAT bad : fam.invalid) {
                const auto srv = volume_srv_desc(bad, 0, 2);
                EXPECT(expect_null(vol.p, &srv));
                ++total_null_descriptors;
            }
        }

        // 2. Relaxed castable-format 3D volumes via CreateCommittedResource3.
        {
            const DXGI_FORMAT cast_typed[] = {DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, DXGI_FORMAT_R8G8B8A8_UINT,
                                              DXGI_FORMAT_R32_UINT, DXGI_FORMAT_B8G8R8A8_UNORM};
            const D3D12_RESOURCE_DESC1 d1 = volume_desc1(DXGI_FORMAT_R8G8B8A8_UNORM, 4, 4, 4, 1);
            Ref<ID3D12Resource> vol_typed;
            EXPECT(SUCCEEDED(device10->CreateCommittedResource3(
                &default_heap, D3D12_HEAP_FLAG_NONE, &d1, D3D12_BARRIER_LAYOUT_COPY_DEST, nullptr, nullptr,
                UINT32(std::size(cast_typed)), cast_typed, IID_PPV_ARGS(&vol_typed.p))));

            // Default SRV on typed castable volume.
            device->CreateShaderResourceView(vol_typed.p, nullptr, slot_handle(0));
            g::Image img{};
            EXPECT(decode_slot(0, img) && img.type == g::ImageType::image_3d &&
                   img.format == g::format_from_vulkan(VK_FORMAT_R8G8B8A8_UNORM));
            ++total_valid_descriptors;

            // Listed casts (same-family and cross-family / cross-order).
            const ViewExpect cast_views[] = {
                {DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, VK_FORMAT_R8G8B8A8_SRGB, g::channels_rgba},
                {DXGI_FORMAT_R8G8B8A8_UINT, VK_FORMAT_R8G8B8A8_UINT, g::channels_rgba},
                {DXGI_FORMAT_R32_UINT, VK_FORMAT_R32_UINT, g::channels_r},
                {DXGI_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_B8G8R8A8_UNORM, g::channels_bgra},
            };
            for (const ViewExpect& v : cast_views) {
                const auto srv = volume_srv_desc(v.dxgi, 0, 1);
                device->CreateShaderResourceView(vol_typed.p, &srv, slot_handle(0));
                EXPECT(decode_slot(0, img) && img.type == g::ImageType::image_3d &&
                       img.format == g::format_from_vulkan(v.vk) && img.swizzle == v.swizzle);
                ++total_valid_descriptors;
            }
            // Unlisted format and typeless format must give null descriptors.
            for (const DXGI_FORMAT bad : {DXGI_FORMAT_R8G8B8A8_SINT, DXGI_FORMAT_R8G8B8A8_TYPELESS}) {
                const auto srv = volume_srv_desc(bad, 0, 1);
                EXPECT(expect_null(vol_typed.p, &srv));
                ++total_null_descriptors;
            }

            // Typeless 3D volume with a restricted castable-format list.
            const DXGI_FORMAT cast_typeless[] = {DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R32_UINT};
            const D3D12_RESOURCE_DESC1 d1_tl = volume_desc1(DXGI_FORMAT_R8G8B8A8_TYPELESS, 4, 4, 4, 1);
            Ref<ID3D12Resource> vol_tl;
            EXPECT(SUCCEEDED(device10->CreateCommittedResource3(
                &default_heap, D3D12_HEAP_FLAG_NONE, &d1_tl, D3D12_BARRIER_LAYOUT_COPY_DEST, nullptr, nullptr,
                UINT32(std::size(cast_typeless)), cast_typeless, IID_PPV_ARGS(&vol_tl.p))));
            for (const ViewExpect& v : {ViewExpect{DXGI_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM, g::channels_rgba},
                                        ViewExpect{DXGI_FORMAT_R32_UINT, VK_FORMAT_R32_UINT, g::channels_r}}) {
                const auto srv = volume_srv_desc(v.dxgi, 0, 1);
                device->CreateShaderResourceView(vol_tl.p, &srv, slot_handle(0));
                EXPECT(decode_slot(0, img) && img.type == g::ImageType::image_3d &&
                       img.format == g::format_from_vulkan(v.vk) && img.swizzle == v.swizzle);
                ++total_valid_descriptors;
            }
            EXPECT(expect_null(vol_tl.p, nullptr));
            ++total_null_descriptors;
            for (const DXGI_FORMAT bad : {DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_R8G8B8A8_TYPELESS,
                                          DXGI_FORMAT_R8G8B8A8_UINT}) {
                const auto srv = volume_srv_desc(bad, 0, 1);
                EXPECT(expect_null(vol_tl.p, &srv));
                ++total_null_descriptors;
            }
        }

        std::printf("volume SRV descriptors: %u valid, %u negative nulls verified\n", total_valid_descriptors,
                    total_null_descriptors);

        // 3. GPU upload, bindless set 0 flush, and texelFetch verification across typeless & castable 3D views.
        if (d.bindless) {
            Ref<ID3D12Resource> vol_rgba, vol_r32, vol_bgra, vol_cast;
            const D3D12_RESOURCE_DESC d_rgba = volume_desc(DXGI_FORMAT_R8G8B8A8_TYPELESS, 4, 4, 4, 1);
            const D3D12_RESOURCE_DESC d_r32 = volume_desc(DXGI_FORMAT_R32_TYPELESS, 4, 4, 4, 1);
            const D3D12_RESOURCE_DESC d_bgra = volume_desc(DXGI_FORMAT_B8G8R8A8_TYPELESS, 4, 4, 4, 1);
            const D3D12_RESOURCE_DESC1 d_cast1 = volume_desc1(DXGI_FORMAT_R8G8B8A8_UNORM, 4, 4, 4, 1);
            const DXGI_FORMAT cast_list[] = {DXGI_FORMAT_R32_UINT, DXGI_FORMAT_B8G8R8A8_UNORM};
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &d_rgba,
                                                             D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                             IID_PPV_ARGS(&vol_rgba.p))));
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &d_r32,
                                                             D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                             IID_PPV_ARGS(&vol_r32.p))));
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &d_bgra,
                                                             D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                             IID_PPV_ARGS(&vol_bgra.p))));
            EXPECT(SUCCEEDED(device10->CreateCommittedResource3(
                &default_heap, D3D12_HEAP_FLAG_NONE, &d_cast1, D3D12_BARRIER_LAYOUT_COPY_DEST, nullptr, nullptr,
                UINT32(std::size(cast_list)), cast_list, IID_PPV_ARGS(&vol_cast.p))));

            // Staging upload buffer for all four 4x4x4x4B volumes.
            const D3D12_RESOURCE_DESC up_desc = buffer_desc(65536);
            Ref<ID3D12Resource> upload;
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &up_desc,
                                                             D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                             IID_PPV_ARGS(&upload.p))));
            void* mapped = nullptr;
            EXPECT(SUCCEEDED(upload->Map(0, nullptr, &mapped)));

            const auto fill_volume = [&](const D3D12_RESOURCE_DESC& rd, UINT64 base_offset,
                                         const std::array<std::uint8_t, 4>& voxel,
                                         D3D12_PLACED_SUBRESOURCE_FOOTPRINT& placed) {
                device->GetCopyableFootprints(&rd, 0, 1, base_offset, &placed, nullptr, nullptr, nullptr);
                const UINT row_pitch = placed.Footprint.RowPitch;
                const UINT slice_pitch = row_pitch * placed.Footprint.Height;
                for (UINT z = 0; z < placed.Footprint.Depth; ++z)
                    for (UINT y = 0; y < placed.Footprint.Height; ++y)
                        for (UINT x = 0; x < placed.Footprint.Width; ++x)
                            std::memcpy(static_cast<std::uint8_t*>(mapped) + placed.Offset + z * slice_pitch +
                                            y * row_pitch + x * 4,
                                        voxel.data(), 4);
            };

            const std::array<std::uint8_t, 4> rgba_bytes{0x80, 0x40, 0xFF, 0xC0};
            const float f_val = -3.5f;
            std::array<std::uint8_t, 4> r32_bytes{};
            std::memcpy(r32_bytes.data(), &f_val, 4);
            const std::array<std::uint8_t, 4> bgra_bytes{0x20, 0x80, 0xE0, 0x40};  // B, G, R, A in memory
            const std::array<std::uint8_t, 4> cast_bytes{0x11, 0x22, 0x33, 0x44};

            D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp_rgba{}, fp_r32{}, fp_bgra{}, fp_cast{};
            fill_volume(d_rgba, 0, rgba_bytes, fp_rgba);
            fill_volume(d_r32, 16384, r32_bytes, fp_r32);
            fill_volume(d_bgra, 32768, bgra_bytes, fp_bgra);
            const D3D12_RESOURCE_DESC d_cast = volume_desc(DXGI_FORMAT_R8G8B8A8_UNORM, 4, 4, 4, 1);
            fill_volume(d_cast, 49152, cast_bytes, fp_cast);
            upload->Unmap(0, nullptr);

            // Create SRVs in heap slots 0..11.
            struct BoundView {
                ID3D12Resource* res;
                DXGI_FORMAT fmt;
                UINT mode;  // 0: float, 1: uint, 2: sint
            };
            const BoundView bound[] = {
                {vol_rgba.p, DXGI_FORMAT_R8G8B8A8_UNORM, 0},
                {vol_rgba.p, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, 0},
                {vol_rgba.p, DXGI_FORMAT_R8G8B8A8_UINT, 1},
                {vol_rgba.p, DXGI_FORMAT_R8G8B8A8_SINT, 2},
                {vol_rgba.p, DXGI_FORMAT_R8G8B8A8_SNORM, 0},
                {vol_r32.p, DXGI_FORMAT_R32_FLOAT, 0},
                {vol_r32.p, DXGI_FORMAT_R32_UINT, 1},
                {vol_r32.p, DXGI_FORMAT_R32_SINT, 2},
                {vol_bgra.p, DXGI_FORMAT_B8G8R8A8_UNORM, 0},
                {vol_bgra.p, DXGI_FORMAT_B8G8R8A8_UNORM_SRGB, 0},
                {vol_cast.p, DXGI_FORMAT_R32_UINT, 1},
                {vol_cast.p, DXGI_FORMAT_B8G8R8A8_UNORM, 0},
            };
            for (UINT i = 0; i < std::size(bound); ++i) {
                const auto srv = volume_srv_desc(bound[i].fmt, 0, 1);
                device->CreateShaderResourceView(bound[i].res, &srv, slot_handle(i));
                g::Image img{};
                EXPECT(decode_slot(i, img) && img.type == g::ImageType::image_3d);
            }

            // Record CopyTextureRegion + transitions and execute to flush bindless set 0.
            const D3D12_COMMAND_QUEUE_DESC qdesc{D3D12_COMMAND_LIST_TYPE_DIRECT, 0, D3D12_COMMAND_QUEUE_FLAG_NONE, 0};
            Ref<ID3D12CommandQueue> queue;
            Ref<ID3D12CommandAllocator> alloc;
            Ref<ID3D12GraphicsCommandList> cmd;
            Ref<ID3D12Fence> fence;
            EXPECT(SUCCEEDED(device->CreateCommandQueue(&qdesc, IID_PPV_ARGS(&queue.p))));
            EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc.p))));
            EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr,
                                                       IID_PPV_ARGS(&cmd.p))));
            EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

            const auto copy_vol = [&](ID3D12Resource* dst, const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& fp) {
                D3D12_TEXTURE_COPY_LOCATION dloc{};
                dloc.pResource = dst;
                dloc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                dloc.SubresourceIndex = 0;
                D3D12_TEXTURE_COPY_LOCATION sloc{};
                sloc.pResource = upload.p;
                sloc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                sloc.PlacedFootprint = fp;
                cmd->CopyTextureRegion(&dloc, 0, 0, 0, &sloc, nullptr);
            };
            copy_vol(vol_rgba.p, fp_rgba);
            copy_vol(vol_r32.p, fp_r32);
            copy_vol(vol_bgra.p, fp_bgra);
            copy_vol(vol_cast.p, fp_cast);

            const D3D12_RESOURCE_BARRIER barriers[] = {
                transition(vol_rgba.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE),
                transition(vol_r32.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE),
                transition(vol_bgra.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE),
                transition(vol_cast.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE),
            };
            cmd->ResourceBarrier(UINT(std::size(barriers)), barriers);
            EXPECT(SUCCEEDED(cmd->Close()));
            ID3D12CommandList* lists[] = {cmd.p};
            queue->ExecuteCommandLists(1, lists);
            EXPECT(SUCCEEDED(queue->Signal(fence.p, 1)));
            EXPECT(SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));

#ifdef LEONIDA_TEST_CS_VOLUME_SRV_SPV
            const VkDescriptorSet set0 = rt::d3d12::bindless_set(device.p);
            const VkDescriptorSetLayout set0_layout = rt::d3d12::bindless_set_layout(device.p);
            EXPECT(set0 && set0_layout);

            std::vector<FetchParams> queries(std::size(bound));
            for (UINT i = 0; i < std::size(bound); ++i) {
                std::memcpy(queries[i].tsharp, slot_bytes(i), sizeof(queries[i].tsharp));
                queries[i].out_index = i;
                queries[i].mode = bound[i].mode;
                queries[i].x = 2;
                queries[i].y = 1;
                queries[i].z = 3;
            }
            std::vector<std::array<std::uint32_t, 4>> out;
            EXPECT(fetch_volumes(d, set0, set0_layout, queries, out));

            const auto as_float = [&](UINT idx, UINT ch) {
                float f = 0.0f;
                std::memcpy(&f, &out[idx][ch], sizeof(f));
                return f;
            };

            // 0: R8G8B8A8_UNORM
            EXPECT(near_f(as_float(0, 0), 128.0f / 255.0f) && near_f(as_float(0, 1), 64.0f / 255.0f) &&
                   near_f(as_float(0, 2), 1.0f) && near_f(as_float(0, 3), 192.0f / 255.0f));
            // 1: R8G8B8A8_UNORM_SRGB (RGB decoded, alpha linear)
            EXPECT(near_f(as_float(1, 0), srgb_to_linear(0x80)) && near_f(as_float(1, 1), srgb_to_linear(0x40)) &&
                   near_f(as_float(1, 2), 1.0f) && near_f(as_float(1, 3), 192.0f / 255.0f));
            // 2: R8G8B8A8_UINT
            EXPECT(out[2][0] == 0x80u && out[2][1] == 0x40u && out[2][2] == 0xFFu && out[2][3] == 0xC0u);
            // 3: R8G8B8A8_SINT (-128, 64, -1, -64)
            EXPECT(std::int32_t(out[3][0]) == -128 && std::int32_t(out[3][1]) == 64 &&
                   std::int32_t(out[3][2]) == -1 && std::int32_t(out[3][3]) == -64);
            // 4: R8G8B8A8_SNORM (-1.0, 64/127, -1/127, -64/127)
            EXPECT(near_f(as_float(4, 0), -1.0f) && near_f(as_float(4, 1), 64.0f / 127.0f) &&
                   near_f(as_float(4, 2), -1.0f / 127.0f) && near_f(as_float(4, 3), -64.0f / 127.0f));
            // 5: R32_FLOAT
            EXPECT(near_f(as_float(5, 0), f_val, 1e-6f));
            // 6: R32_UINT
            std::uint32_t f_bits = 0;
            std::memcpy(&f_bits, &f_val, 4);
            EXPECT(out[6][0] == f_bits);
            // 7: R32_SINT
            EXPECT(out[7][0] == f_bits);
            // 8: B8G8R8A8_UNORM (memory B=0x20, G=0x80, R=0xE0, A=0x40 -> R=0xE0, G=0x80, B=0x20, A=0x40)
            EXPECT(near_f(as_float(8, 0), 0xE0 / 255.0f) && near_f(as_float(8, 1), 0x80 / 255.0f) &&
                   near_f(as_float(8, 2), 0x20 / 255.0f) && near_f(as_float(8, 3), 0x40 / 255.0f));
            // 9: B8G8R8A8_UNORM_SRGB
            EXPECT(near_f(as_float(9, 0), srgb_to_linear(0xE0)) && near_f(as_float(9, 1), srgb_to_linear(0x80)) &&
                   near_f(as_float(9, 2), srgb_to_linear(0x20)) && near_f(as_float(9, 3), 0x40 / 255.0f));
            // 10: Castable R8G8B8A8_UNORM -> R32_UINT (little-endian 0x44332211)
            EXPECT(out[10][0] == 0x44332211u);
            // 11: Castable R8G8B8A8_UNORM -> B8G8R8A8_UNORM (R/B swapped: 0x33, 0x22, 0x11, 0x44)
            EXPECT(near_f(as_float(11, 0), 0x33 / 255.0f) && near_f(as_float(11, 1), 0x22 / 255.0f) &&
                   near_f(as_float(11, 2), 0x11 / 255.0f) && near_f(as_float(11, 3), 0x44 / 255.0f));

            std::printf("volume SRV GPU readbacks: %zu/12 views matched exact expected values\n", out.size());
#endif
        }
    }

    EXPECT(d.validation_errors() == 0);
    return 0;
}
