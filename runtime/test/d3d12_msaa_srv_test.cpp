// Multisampled typeless color and depth/stencil shader-resource views (TEXTURE2DMS and TEXTURE2DMSARRAY).
// Exercises descriptor encoding (T# format, sample count, layer range, stencil swizzle), bindless set 0 registration
// (including native D24_UNORM_S8_UINT and emulated D24-as-D32_SFLOAT_S8_UINT), negative validation controls, and GPU
// clear + barrier transitions to PIXEL_SHADER_RESOURCE / NON_PIXEL_SHADER_RESOURCE under Vulkan validation.
// Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../src/gfx10_descriptors.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>

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

D3D12_RESOURCE_DESC msaa_texture_desc(DXGI_FORMAT format, UINT16 layers, D3D12_RESOURCE_FLAGS flags) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = 16;
    desc.Height = 16;
    desc.DepthOrArraySize = layers;
    desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc = {4, 0};
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = flags;
    return desc;
}

D3D12_SHADER_RESOURCE_VIEW_DESC msaa_srv_desc(DXGI_FORMAT format, bool array, UINT first_slice = 0,
                                              UINT slice_count = 1) {
    D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
    desc.Format = format;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    if (array) {
        desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY;
        desc.Texture2DMSArray.FirstArraySlice = first_slice;
        desc.Texture2DMSArray.ArraySize = slice_count;
    } else {
        desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMS;
    }
    return desc;
}

bool supports_4x(ID3D12Device* device, DXGI_FORMAT format) {
    D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS q{format, 4};
    return SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &q, sizeof(q))) &&
           q.NumQualityLevels > 0;
}

}  // namespace

int main() {
    namespace g = leonida::gfx10;

    vktest::Device d;
    const auto status = d.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;
    if (!d.bindless) {
        std::printf("d3d12_msaa_srv: device lacks bindless features, skipping\n");
        return 77;
    }

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

        const UINT srv_step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        const UINT rtv_step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        const UINT dsv_step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
        EXPECT(srv_step >= sizeof(g::ImageWords));

        constexpr UINT srv_slots = 64;
        Ref<ID3D12DescriptorHeap> srv_heap, rtv_heap, dsv_heap;
        D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, srv_slots,
                                             D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&srv_heap.p))));
        heap_desc = {D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 8, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&rtv_heap.p))));
        heap_desc = {D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 8, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&dsv_heap.p))));

        const D3D12_CPU_DESCRIPTOR_HANDLE srv_base = srv_heap->GetCPUDescriptorHandleForHeapStart();
        const D3D12_CPU_DESCRIPTOR_HANDLE rtv_base = rtv_heap->GetCPUDescriptorHandleForHeapStart();
        const D3D12_CPU_DESCRIPTOR_HANDLE dsv_base = dsv_heap->GetCPUDescriptorHandleForHeapStart();
        const auto srv_handle = [&](UINT i) {
            return D3D12_CPU_DESCRIPTOR_HANDLE{srv_base.ptr + SIZE_T(i) * srv_step};
        };
        const auto rtv_handle = [&](UINT i) {
            return D3D12_CPU_DESCRIPTOR_HANDLE{rtv_base.ptr + SIZE_T(i) * rtv_step};
        };
        const auto dsv_handle = [&](UINT i) {
            return D3D12_CPU_DESCRIPTOR_HANDLE{dsv_base.ptr + SIZE_T(i) * dsv_step};
        };
        const auto srv_bytes = [&](UINT i) {
            return reinterpret_cast<const std::uint8_t*>(srv_base.ptr + SIZE_T(i) * srv_step);
        };
        const auto is_null = [&](UINT i) {
            const std::uint8_t* p = srv_bytes(i);
            return std::all_of(p, p + sizeof(g::ImageWords), [](std::uint8_t b) { return b == 0; });
        };
        const auto decode_slot = [&](UINT i, g::Image& out, g::ImageWords& words) {
            std::memcpy(words.data(), srv_bytes(i), sizeof(words));
            return g::is_image(words) && g::decode(words, out);
        };

        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        UINT next_srv = 0;

        const auto check_valid_srv = [&](ID3D12Resource* res, const D3D12_SHADER_RESOURCE_VIEW_DESC* desc,
                                         g::ImageType expected_type, std::uint32_t expected_fmt_a,
                                         std::uint32_t expected_fmt_b, UINT expected_first_layer,
                                         UINT expected_layer_count, bool expect_stencil_swizzle = false) {
            const UINT slot = next_srv++;
            EXPECT(slot < srv_slots);
            device->CreateShaderResourceView(res, desc, srv_handle(slot));
            g::Image img{};
            g::ImageWords words{};
            EXPECT(decode_slot(slot, img, words));
            EXPECT(img.type == expected_type);
            EXPECT(img.samples == 4);
            EXPECT(img.first_mip == 0 && img.mip_count == 1);
            EXPECT(img.first_layer == expected_first_layer && img.layer_count == expected_layer_count);
            EXPECT(img.format == expected_fmt_a || (expected_fmt_b != 0 && img.format == expected_fmt_b));
            if (expect_stencil_swizzle) {
                const g::Swizzle stencil_swizzle{g::select_zero, g::select_x, g::select_zero, g::select_one};
                EXPECT(img.swizzle == stencil_swizzle);
            }
            std::uint32_t key[8];
            std::memcpy(key, words.data(), sizeof(key));
            EXPECT(rt::d3d12::bindless_hash_lookup(device.p, key) != 0);
            return 0;
        };

        const auto check_null_srv = [&](ID3D12Resource* res, const D3D12_SHADER_RESOURCE_VIEW_DESC* desc) {
            const UINT slot = srv_slots - 1;
            std::memset(reinterpret_cast<void*>(srv_handle(slot).ptr), 0xCC, srv_step);
            device->CreateShaderResourceView(res, desc, srv_handle(slot));
            EXPECT(is_null(slot));
            return 0;
        };

        // ---------------------------------------------------------------------
        // 1. 4x MSAA R8G8B8A8_TYPELESS (single & 3-layer array)
        // ---------------------------------------------------------------------
        Ref<ID3D12Resource> rgba8_ms, rgba8_ms_arr;
        if (supports_4x(device.p, DXGI_FORMAT_R8G8B8A8_UNORM)) {
            const auto d1 = msaa_texture_desc(DXGI_FORMAT_R8G8B8A8_TYPELESS, 1, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
            const auto d3 = msaa_texture_desc(DXGI_FORMAT_R8G8B8A8_TYPELESS, 3, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &d1,
                                                             D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                             IID_PPV_ARGS(&rgba8_ms.p))));
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &d3,
                                                             D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                             IID_PPV_ARGS(&rgba8_ms_arr.p))));

            struct ColorCase {
                DXGI_FORMAT dxgi;
                VkFormat vk;
            };
            const ColorCase rgba8_cases[] = {
                {DXGI_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM},
                {DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, VK_FORMAT_R8G8B8A8_SRGB},
                {DXGI_FORMAT_R8G8B8A8_UINT, VK_FORMAT_R8G8B8A8_UINT},
                {DXGI_FORMAT_R8G8B8A8_SINT, VK_FORMAT_R8G8B8A8_SINT},
            };
            for (const auto& c : rgba8_cases) {
                const auto v2d = msaa_srv_desc(c.dxgi, false);
                EXPECT(check_valid_srv(rgba8_ms.p, &v2d, g::ImageType::image_2d_msaa,
                                       g::format_from_vulkan(c.vk), 0, 0, 1) == 0);
                const auto varr_all = msaa_srv_desc(c.dxgi, true, 0, 3);
                EXPECT(check_valid_srv(rgba8_ms_arr.p, &varr_all, g::ImageType::image_2d_msaa_array,
                                       g::format_from_vulkan(c.vk), 0, 0, 3) == 0);
                const auto varr_slice = msaa_srv_desc(c.dxgi, true, 1, 2);
                EXPECT(check_valid_srv(rgba8_ms_arr.p, &varr_slice, g::ImageType::image_2d_msaa_array,
                                       g::format_from_vulkan(c.vk), 0, 1, 2) == 0);
                const auto varr_rem = msaa_srv_desc(c.dxgi, true, 1, UINT_MAX);
                EXPECT(check_valid_srv(rgba8_ms_arr.p, &varr_rem, g::ImageType::image_2d_msaa_array,
                                       g::format_from_vulkan(c.vk), 0, 1, 2) == 0);
            }

            // Negative controls on typeless color MSAA
            EXPECT(check_null_srv(rgba8_ms.p, nullptr) == 0);
            const auto v_unknown = msaa_srv_desc(DXGI_FORMAT_UNKNOWN, false);
            EXPECT(check_null_srv(rgba8_ms.p, &v_unknown) == 0);
            const auto v_typeless = msaa_srv_desc(DXGI_FORMAT_R8G8B8A8_TYPELESS, false);
            EXPECT(check_null_srv(rgba8_ms.p, &v_typeless) == 0);
            const auto v_wrong_fam = msaa_srv_desc(DXGI_FORMAT_R16G16B16A16_FLOAT, false);
            EXPECT(check_null_srv(rgba8_ms.p, &v_wrong_fam) == 0);
            const auto v_stencil_on_color = msaa_srv_desc(DXGI_FORMAT_X24_TYPELESS_G8_UINT, false);
            EXPECT(check_null_srv(rgba8_ms.p, &v_stencil_on_color) == 0);
            const auto v_oob1 = msaa_srv_desc(DXGI_FORMAT_R8G8B8A8_UNORM, true, 3, 1);
            EXPECT(check_null_srv(rgba8_ms_arr.p, &v_oob1) == 0);
            const auto v_oob2 = msaa_srv_desc(DXGI_FORMAT_R8G8B8A8_UNORM, true, 1, 3);
            EXPECT(check_null_srv(rgba8_ms_arr.p, &v_oob2) == 0);
            const auto v_zero = msaa_srv_desc(DXGI_FORMAT_R8G8B8A8_UNORM, true, 0, 0);
            EXPECT(check_null_srv(rgba8_ms_arr.p, &v_zero) == 0);
            auto v_bad_map = msaa_srv_desc(DXGI_FORMAT_R8G8B8A8_UNORM, false);
            v_bad_map.Shader4ComponentMapping = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(6, 1, 2, 3);
            EXPECT(check_null_srv(rgba8_ms.p, &v_bad_map) == 0);
        }

        // ---------------------------------------------------------------------
        // 2. 4x MSAA R16G16B16A16_TYPELESS (single & 2-layer array)
        // ---------------------------------------------------------------------
        Ref<ID3D12Resource> rgba16_ms, rgba16_ms_arr;
        if (supports_4x(device.p, DXGI_FORMAT_R16G16B16A16_FLOAT)) {
            const auto d1 =
                msaa_texture_desc(DXGI_FORMAT_R16G16B16A16_TYPELESS, 1, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
            const auto d2 =
                msaa_texture_desc(DXGI_FORMAT_R16G16B16A16_TYPELESS, 2, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &d1,
                                                             D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                             IID_PPV_ARGS(&rgba16_ms.p))));
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &d2,
                                                             D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                             IID_PPV_ARGS(&rgba16_ms_arr.p))));

            const std::pair<DXGI_FORMAT, VkFormat> rgba16_cases[] = {
                {DXGI_FORMAT_R16G16B16A16_FLOAT, VK_FORMAT_R16G16B16A16_SFLOAT},
                {DXGI_FORMAT_R16G16B16A16_UNORM, VK_FORMAT_R16G16B16A16_UNORM},
                {DXGI_FORMAT_R16G16B16A16_UINT, VK_FORMAT_R16G16B16A16_UINT},
            };
            for (const auto& [dxgi, vk] : rgba16_cases) {
                const auto v2d = msaa_srv_desc(dxgi, false);
                EXPECT(check_valid_srv(rgba16_ms.p, &v2d, g::ImageType::image_2d_msaa,
                                       g::format_from_vulkan(vk), 0, 0, 1) == 0);
                const auto varr = msaa_srv_desc(dxgi, true, 0, 2);
                EXPECT(check_valid_srv(rgba16_ms_arr.p, &varr, g::ImageType::image_2d_msaa_array,
                                       g::format_from_vulkan(vk), 0, 0, 2) == 0);
            }
            const auto v_unknown = msaa_srv_desc(DXGI_FORMAT_UNKNOWN, false);
            EXPECT(check_null_srv(rgba16_ms.p, &v_unknown) == 0);
            const auto v_typeless = msaa_srv_desc(DXGI_FORMAT_R16G16B16A16_TYPELESS, false);
            EXPECT(check_null_srv(rgba16_ms.p, &v_typeless) == 0);
            const auto v_wrong = msaa_srv_desc(DXGI_FORMAT_R8G8B8A8_UNORM, false);
            EXPECT(check_null_srv(rgba16_ms.p, &v_wrong) == 0);
        }

        // ---------------------------------------------------------------------
        // 3. 4x MSAA R32_TYPELESS (ALLOW_DEPTH_STENCIL) -> R32_FLOAT
        // ---------------------------------------------------------------------
        Ref<ID3D12Resource> d32_ms, d32_ms_arr;
        if (supports_4x(device.p, DXGI_FORMAT_D32_FLOAT)) {
            const auto d1 = msaa_texture_desc(DXGI_FORMAT_R32_TYPELESS, 1, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
            const auto d3 = msaa_texture_desc(DXGI_FORMAT_R32_TYPELESS, 3, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &d1,
                                                             D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                             IID_PPV_ARGS(&d32_ms.p))));
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &d3,
                                                             D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                             IID_PPV_ARGS(&d32_ms_arr.p))));

            const auto v2d = msaa_srv_desc(DXGI_FORMAT_R32_FLOAT, false);
            EXPECT(check_valid_srv(d32_ms.p, &v2d, g::ImageType::image_2d_msaa,
                                   g::format_from_vulkan(VK_FORMAT_D32_SFLOAT), 0, 0, 1) == 0);
            const auto varr = msaa_srv_desc(DXGI_FORMAT_R32_FLOAT, true, 1, 2);
            EXPECT(check_valid_srv(d32_ms_arr.p, &varr, g::ImageType::image_2d_msaa_array,
                                   g::format_from_vulkan(VK_FORMAT_D32_SFLOAT), 0, 1, 2) == 0);

            // Negative controls on R32_TYPELESS depth MSAA
            EXPECT(check_null_srv(d32_ms.p, nullptr) == 0);
            const auto v_unknown = msaa_srv_desc(DXGI_FORMAT_UNKNOWN, false);
            EXPECT(check_null_srv(d32_ms.p, &v_unknown) == 0);
            const auto v_typeless = msaa_srv_desc(DXGI_FORMAT_R32_TYPELESS, false);
            EXPECT(check_null_srv(d32_ms.p, &v_typeless) == 0);
            const auto v_uint = msaa_srv_desc(DXGI_FORMAT_R32_UINT, false);
            EXPECT(check_null_srv(d32_ms.p, &v_uint) == 0);
            const auto v_color = msaa_srv_desc(DXGI_FORMAT_R8G8B8A8_UNORM, false);
            EXPECT(check_null_srv(d32_ms.p, &v_color) == 0);
            const auto v_stencil = msaa_srv_desc(DXGI_FORMAT_X24_TYPELESS_G8_UINT, false);
            EXPECT(check_null_srv(d32_ms.p, &v_stencil) == 0);
            const auto v_oob = msaa_srv_desc(DXGI_FORMAT_R32_FLOAT, true, 2, 2);
            EXPECT(check_null_srv(d32_ms_arr.p, &v_oob) == 0);
        }

        // ---------------------------------------------------------------------
        // 4. 4x MSAA R24G8_TYPELESS (ALLOW_DEPTH_STENCIL)
        //    Tests BOTH native D24_UNORM_S8_UINT (NVIDIA) and emulated D32_SFLOAT_S8_UINT (AMD)
        // ---------------------------------------------------------------------
        Ref<ID3D12Resource> d24s8_ms, d24s8_ms_arr;
        if (supports_4x(device.p, DXGI_FORMAT_D24_UNORM_S8_UINT)) {
            const auto d1 = msaa_texture_desc(DXGI_FORMAT_R24G8_TYPELESS, 1, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
            const auto d2 = msaa_texture_desc(DXGI_FORMAT_R24G8_TYPELESS, 2, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &d1,
                                                             D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                             IID_PPV_ARGS(&d24s8_ms.p))));
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &d2,
                                                             D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                             IID_PPV_ARGS(&d24s8_ms_arr.p))));

            const std::uint32_t d24_code = g::format_from_vulkan(VK_FORMAT_D24_UNORM_S8_UINT);
            const std::uint32_t d32s8_code = g::format_from_vulkan(VK_FORMAT_D32_SFLOAT_S8_UINT);

            // Depth plane
            const auto v_depth_2d = msaa_srv_desc(DXGI_FORMAT_R24_UNORM_X8_TYPELESS, false);
            EXPECT(check_valid_srv(d24s8_ms.p, &v_depth_2d, g::ImageType::image_2d_msaa,
                                   d24_code, d32s8_code, 0, 1) == 0);
            const auto v_depth_arr = msaa_srv_desc(DXGI_FORMAT_R24_UNORM_X8_TYPELESS, true, 0, 2);
            EXPECT(check_valid_srv(d24s8_ms_arr.p, &v_depth_arr, g::ImageType::image_2d_msaa_array,
                                   d24_code, d32s8_code, 0, 2) == 0);

            // Stencil plane (plane 1 inferred from X24_TYPELESS_G8_UINT)
            const auto v_stencil_2d = msaa_srv_desc(DXGI_FORMAT_X24_TYPELESS_G8_UINT, false);
            EXPECT(check_valid_srv(d24s8_ms.p, &v_stencil_2d, g::ImageType::image_2d_msaa,
                                   g::stencil_format, 0, 0, 1, true) == 0);
            const auto v_stencil_arr = msaa_srv_desc(DXGI_FORMAT_X24_TYPELESS_G8_UINT, true, 1, 1);
            EXPECT(check_valid_srv(d24s8_ms_arr.p, &v_stencil_arr, g::ImageType::image_2d_msaa_array,
                                   g::stencil_format, 0, 1, 1, true) == 0);

            // Negative controls on R24G8_TYPELESS
            EXPECT(check_null_srv(d24s8_ms.p, nullptr) == 0);
            const auto v_unknown = msaa_srv_desc(DXGI_FORMAT_UNKNOWN, false);
            EXPECT(check_null_srv(d24s8_ms.p, &v_unknown) == 0);
            const auto v_typeless = msaa_srv_desc(DXGI_FORMAT_R24G8_TYPELESS, false);
            EXPECT(check_null_srv(d24s8_ms.p, &v_typeless) == 0);
            const auto v_wrong_d = msaa_srv_desc(DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS, false);
            EXPECT(check_null_srv(d24s8_ms.p, &v_wrong_d) == 0);
            const auto v_wrong_s = msaa_srv_desc(DXGI_FORMAT_X32_TYPELESS_G8X24_UINT, false);
            EXPECT(check_null_srv(d24s8_ms.p, &v_wrong_s) == 0);
            const auto v_color = msaa_srv_desc(DXGI_FORMAT_R8G8B8A8_UNORM, false);
            EXPECT(check_null_srv(d24s8_ms.p, &v_color) == 0);
        }

        // ---------------------------------------------------------------------
        // 5. 4x MSAA R32G8X24_TYPELESS (ALLOW_DEPTH_STENCIL)
        // ---------------------------------------------------------------------
        Ref<ID3D12Resource> d32s8_ms, d32s8_ms_arr;
        if (supports_4x(device.p, DXGI_FORMAT_D32_FLOAT_S8X24_UINT)) {
            const auto d1 =
                msaa_texture_desc(DXGI_FORMAT_R32G8X24_TYPELESS, 1, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
            const auto d2 =
                msaa_texture_desc(DXGI_FORMAT_R32G8X24_TYPELESS, 2, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &d1,
                                                             D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                             IID_PPV_ARGS(&d32s8_ms.p))));
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &d2,
                                                             D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                             IID_PPV_ARGS(&d32s8_ms_arr.p))));

            const std::uint32_t d32s8_code = g::format_from_vulkan(VK_FORMAT_D32_SFLOAT_S8_UINT);

            // Depth plane
            const auto v_depth_2d = msaa_srv_desc(DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS, false);
            EXPECT(check_valid_srv(d32s8_ms.p, &v_depth_2d, g::ImageType::image_2d_msaa,
                                   d32s8_code, 0, 0, 1) == 0);
            const auto v_depth_arr = msaa_srv_desc(DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS, true, 0, 2);
            EXPECT(check_valid_srv(d32s8_ms_arr.p, &v_depth_arr, g::ImageType::image_2d_msaa_array,
                                   d32s8_code, 0, 0, 2) == 0);

            // Stencil plane (plane 1 inferred from X32_TYPELESS_G8X24_UINT)
            const auto v_stencil_2d = msaa_srv_desc(DXGI_FORMAT_X32_TYPELESS_G8X24_UINT, false);
            EXPECT(check_valid_srv(d32s8_ms.p, &v_stencil_2d, g::ImageType::image_2d_msaa,
                                   g::stencil_format, 0, 0, 1, true) == 0);
            const auto v_stencil_arr = msaa_srv_desc(DXGI_FORMAT_X32_TYPELESS_G8X24_UINT, true, 0, 2);
            EXPECT(check_valid_srv(d32s8_ms_arr.p, &v_stencil_arr, g::ImageType::image_2d_msaa_array,
                                   g::stencil_format, 0, 0, 2, true) == 0);

            // Negative controls on R32G8X24_TYPELESS
            EXPECT(check_null_srv(d32s8_ms.p, nullptr) == 0);
            const auto v_unknown = msaa_srv_desc(DXGI_FORMAT_UNKNOWN, false);
            EXPECT(check_null_srv(d32s8_ms.p, &v_unknown) == 0);
            const auto v_typeless = msaa_srv_desc(DXGI_FORMAT_R32G8X24_TYPELESS, false);
            EXPECT(check_null_srv(d32s8_ms.p, &v_typeless) == 0);
            const auto v_wrong_d = msaa_srv_desc(DXGI_FORMAT_R24_UNORM_X8_TYPELESS, false);
            EXPECT(check_null_srv(d32s8_ms.p, &v_wrong_d) == 0);
            const auto v_wrong_s = msaa_srv_desc(DXGI_FORMAT_X24_TYPELESS_G8_UINT, false);
            EXPECT(check_null_srv(d32s8_ms.p, &v_wrong_s) == 0);
        }

        // ---------------------------------------------------------------------
        // 6. GPU clears + transitions to PIXEL_SHADER_RESOURCE / NON_PIXEL_SHADER_RESOURCE
        //    Also flushes bindless set 0 with all created MSAA views under Vulkan validation.
        // ---------------------------------------------------------------------
        Ref<ID3D12CommandQueue> queue;
        D3D12_COMMAND_QUEUE_DESC qdesc{D3D12_COMMAND_LIST_TYPE_DIRECT};
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&qdesc, IID_PPV_ARGS(&queue.p))));
        Ref<ID3D12CommandAllocator> alloc;
        EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc.p))));
        Ref<ID3D12GraphicsCommandList> list;
        EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr,
                                                   IID_PPV_ARGS(&list.p))));

        const float clear_rgba[4] = {0.25f, 0.5f, 0.75f, 1.0f};
        UINT rtv_idx = 0;
        const auto clear_and_transition_color = [&](ID3D12Resource* res, DXGI_FORMAT rtv_fmt) {
            if (!res) return;
            D3D12_RENDER_TARGET_VIEW_DESC rtv{};
            rtv.Format = rtv_fmt;
            rtv.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DMS;
            const auto handle = rtv_handle(rtv_idx++);
            device->CreateRenderTargetView(res, &rtv, handle);

            const auto b1 = transition(res, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET);
            list->ResourceBarrier(1, &b1);
            list->ClearRenderTargetView(handle, clear_rgba, 0, nullptr);
            const auto b2 =
                transition(res, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            list->ResourceBarrier(1, &b2);
            const auto b3 = transition(res, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            list->ResourceBarrier(1, &b3);
        };

        UINT dsv_idx = 0;
        const auto clear_and_transition_depth = [&](ID3D12Resource* res, DXGI_FORMAT dsv_fmt, bool has_stencil) {
            if (!res) return;
            D3D12_DEPTH_STENCIL_VIEW_DESC dsv{};
            dsv.Format = dsv_fmt;
            dsv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DMS;
            const auto handle = dsv_handle(dsv_idx++);
            device->CreateDepthStencilView(res, &dsv, handle);

            const auto b1 = transition(res, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_DEPTH_WRITE);
            list->ResourceBarrier(1, &b1);
            const D3D12_CLEAR_FLAGS flags =
                has_stencil ? (D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL) : D3D12_CLEAR_FLAG_DEPTH;
            list->ClearDepthStencilView(handle, flags, 0.5f, 0x5A, 0, nullptr);
            const auto b2 =
                transition(res, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            list->ResourceBarrier(1, &b2);
            const auto b3 = transition(res, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            list->ResourceBarrier(1, &b3);
        };

        clear_and_transition_color(rgba8_ms.p, DXGI_FORMAT_R8G8B8A8_UNORM);
        clear_and_transition_color(rgba16_ms.p, DXGI_FORMAT_R16G16B16A16_FLOAT);
        clear_and_transition_depth(d32_ms.p, DXGI_FORMAT_D32_FLOAT, false);
        clear_and_transition_depth(d24s8_ms.p, DXGI_FORMAT_D24_UNORM_S8_UINT, true);
        clear_and_transition_depth(d32s8_ms.p, DXGI_FORMAT_D32_FLOAT_S8X24_UINT, true);

        EXPECT(SUCCEEDED(list->Close()));
        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);

        Ref<ID3D12Fence> fence;
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
        HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        EXPECT(event != nullptr);
        EXPECT(SUCCEEDED(queue->Signal(fence.p, 1)));
        EXPECT(SUCCEEDED(fence->SetEventOnCompletion(1, event)));
        EXPECT(WaitForSingleObject(event, 10000) == WAIT_OBJECT_0);
        CloseHandle(event);

        std::printf("d3d12_msaa_srv: validated %u MSAA typeless color and depth/stencil SRVs + GPU clears/barriers\n",
                    next_srv);
    }

    EXPECT(d.validation_errors() == 0);
    return 0;
}
