// LEONIDA_STUB_COMPUTE_WORDS=<n> (diagnostic): a compute PSO whose translated module is over n SPIR-V words is
// created as a no-op stub, never registered, instead of being compiled. A module of exactly n words, an unset switch,
// "0" and a value that is not a number all compile it as before.
// The program is a synthetic FSSM blob and the module is test/glsl/cs_user_data.comp, as in d3d12_compute_test.
// No game data.
#include "shader_fixture_path.h"
#include "test_temp_dir.h"
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../src/shader_library.h"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#ifndef LEONIDA_TEST_CS_SPV
#error "LEONIDA_TEST_CS_SPV must name the compiled test shader"
#endif

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() {
        if (p) p->Release();
    }
};

void put32(std::vector<std::uint8_t>& out, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) out.push_back(std::uint8_t(v >> (8 * i)));
}

void set_limit(const wchar_t* value) { SetEnvironmentVariableW(L"LEONIDA_STUB_COMPUTE_WORDS", value); }

// S_OK and a pipeline: 1 when it was compiled (registered as a compute pipeline), 0 when it is a stub; -1 on failure.
int create(ID3D12Device* device, const D3D12_COMPUTE_PIPELINE_STATE_DESC& desc) {
    Ref<ID3D12PipelineState> pso;
    if (device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso.p)) != S_OK || !pso.p) return -1;
    return rt::d3d12::compute_pipeline_id(pso.p) != 0 ? 1 : 0;
}

}  // namespace

int main() {
    set_limit(nullptr);
    vktest::Device d;
    const auto device_status = d.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) {
        std::fprintf(stderr, "SKIP: no usable Vulkan 1.3 device\n");
        return 77;
    }
    EXPECT(device_status == vktest::DeviceStatus::ready);

    // A bare FSSM program (the slot-67 shape that CreateComputePipelineState also takes) and an index that maps its
    // hash to the test module.
    std::vector<std::uint8_t> fssm = {'F', 'S', 'S', 'M'};
    put32(fssm, 5000);
    for (std::uint32_t i = 0; i < 14; ++i) put32(fssm, 0xbf810000u + i);
    const testfiles::ScratchDir scratch("leonida_stub_compute_words_test");
    const auto& library = scratch.path();
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_CS_SPV), library / "cs_user_data.spv"));
    const auto module_bytes = std::filesystem::file_size(library / "cs_user_data.spv");
    EXPECT(module_bytes >= 20 && module_bytes % 4 == 0);
    const std::uintmax_t words = module_bytes / 4;
    {
        std::ofstream index(library / "index.json", std::ios::binary);
        index << "{\"cs_user_data\": {\"stage\": \"cs\", \"fssm_sha256\": \"" << leonida::sha256_hex(fssm)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"local_size\": [64, 1, 1], "
                 "\"uses_code_base\": true, \"wave64\": false}}";
    }

    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = d.instance_proc;
    info.instance = d.instance;
    info.physical = d.physical;
    info.device = d.device;
    info.queue = d.queue;
    info.queue_family = d.family;
    info.buffer_device_address = true;
    info.subgroup = d.subgroup;
    info.shader_dir = library.string();
    Ref<ID3D12Device> device;
    EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))));

    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
    desc.CS = {fssm.data(), fssm.size()};
    const std::wstring at = std::to_wstring(words), below = std::to_wstring(words - 1);

    EXPECT(create(device.p, desc) == 1);  // unset: compiled
    set_limit(at.c_str());
    EXPECT(create(device.p, desc) == 1);  // exactly n words is not over n
    set_limit(below.c_str());
    EXPECT(create(device.p, desc) == 0);  // over n: a stub that binds as no compute pipeline
    set_limit(L"1");
    EXPECT(create(device.p, desc) == 0);
    set_limit(L"0");
    EXPECT(create(device.p, desc) == 1);  // 0 is off
    set_limit(L"100k");
    EXPECT(create(device.p, desc) == 1);  // not a number: off
    set_limit(L"99999999999999999999");
    EXPECT(create(device.p, desc) == 1);  // too long to be a word count: off

    // Real refusals stay refusals: an unknown program is E_INVALIDARG with the switch on.
    set_limit(L"1");
    auto other = fssm;
    other[8] ^= 1;
    D3D12_COMPUTE_PIPELINE_STATE_DESC unknown_desc{};
    unknown_desc.CS = {other.data(), other.size()};
    Ref<ID3D12PipelineState> unknown;
    EXPECT(device.p->CreateComputePipelineState(&unknown_desc, IID_PPV_ARGS(&unknown.p)) == E_INVALIDARG &&
           !unknown.p);
    set_limit(nullptr);

    std::printf("PASS: %llu-word module compiled at n=%llu and stubbed at n=%llu\n",
                static_cast<unsigned long long>(words), static_cast<unsigned long long>(words),
                static_cast<unsigned long long>(words - 1));
    return 0;
}
