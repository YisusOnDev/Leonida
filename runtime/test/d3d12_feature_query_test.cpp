// ID3D12Device::CheckFeatureSupport on every D3D12_FEATURE value from 0 to 80 and on sizes around the one each
// structure has, compared with the host D3D12 on the hardware adapter and on WARP: which queries are answered, which
// sizes are refused and with what, and the payload of the queries every adapter answers the same way. FORMAT_SUPPORT,
// FORMAT_INFO and MULTISAMPLE_QUALITY_LEVELS (descriptions, not sizes), OPTIONS (#4026), OPTIONS6 and OPTIONS10
// (variable rate shading, #469) and the per-adapter values of the OPTIONS structures are not compared; a query only some host
// adapters answer is not compared. No shaders or game data.
// LEONIDA_PROBE (any value) prints every row.
#include "vk_test_device.h"
#include "host_adapter.h"
#include "host_wine.h"
#include "../shims/d3d12_com.h"
#include <dxgi1_4.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {
template<class T>
struct Ref {
    T* p = nullptr;
    Ref() = default;
    Ref(const Ref&) = delete;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
};

constexpr int kFeatureCount = 81;
constexpr UINT kMaxSize = 160;

// What a device says about one feature: the smallest size it accepts (0: none up to kMaxSize), the HRESULT it gives
// at sizes around that, and the bytes it writes.
struct Answer {
    UINT size = 0;
    std::vector<std::uint32_t> around;  // HRESULTs at 0, size - 4, size - 1, size + 1, size + 4, 2 * size
    std::vector<unsigned char> bytes;
};

HRESULT ask(ID3D12Device* d, int feature, UINT size, std::vector<unsigned char>& buffer) {
    buffer.assign(kMaxSize * 4, 0);
    return d->CheckFeatureSupport(static_cast<D3D12_FEATURE>(feature), buffer.data(), size);
}

Answer measure(ID3D12Device* d, int feature) {
    Answer a;
    std::vector<unsigned char> buffer;
    for (UINT size = 1; size <= kMaxSize; ++size) {
        if (SUCCEEDED(ask(d, feature, size, buffer))) {
            a.size = size;
            // The payload is read from a buffer poisoned with a nonzero byte, so a query that stops writing its output
            // (whose expected value is zero) shows as 0xA5 instead of matching the zero-filled buffer.
            // Features 17 and 55 have input fields the host reads (it refuses 17 and leaves the input parts of 55
            // alone when they are 0xA5), so they are read from the zero-filled buffer.
            buffer.assign(kMaxSize * 4, feature == 17 || feature == 55 ? 0 : 0xA5);
            if (SUCCEEDED(d->CheckFeatureSupport(static_cast<D3D12_FEATURE>(feature), buffer.data(), size)))
                a.bytes.assign(buffer.begin(), buffer.begin() + size);
            break;
        }
    }
    const UINT probes[] = {0, a.size > 4 ? a.size - 4 : 4, a.size > 1 ? a.size - 1 : 1, a.size + 1, a.size + 4, a.size * 2 + 8};
    for (UINT s : probes) a.around.push_back(static_cast<std::uint32_t>(ask(d, feature, s, buffer)));
    return a;
}

// Features whose payload is the same on every adapter, so the layer's must be too. The rest are tiers and limits that
// depend on the GPU.
bool payload_compared(int feature) {
    switch (feature) {
    case 17: case 54: case 55: case 58: case 56: return true;
    default: return false;
    }
}

// 0: OPTIONS, which the layer answers for the console's larger structure too (#4026), the host does not.
// 3, 4, 5: descriptions, not sizes. 30 and 39: OPTIONS6 and OPTIONS10, with the variable-rate-shading work (#469).
bool features_skipped(int feature) {
    return feature == 0 || feature == 3 || feature == 4 || feature == 5 || feature == 30 || feature == 39;
}

struct Row {
    std::string name;
    std::int64_t code;
};

std::int64_t digest(const Answer& a, bool with_payload) {
    std::int64_t v = a.size;
    for (std::uint32_t h : a.around) v = v * 1000003 ^ h;
    if (with_payload)
        for (unsigned char b : a.bytes) v = v * 1000003 ^ b;
    return v;
}
}  // namespace

int main() {
    vktest::Device vk;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;
    const bool probe = GetEnvironmentVariableA("LEONIDA_PROBE", nullptr, 0) != 0;
    Ref<ID3D12Device> ours;
    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = vk.instance_proc;
    info.instance = vk.instance;
    info.physical = vk.physical;
    info.device = vk.device;
    info.queue = vk.queue;
    info.queue_family = vk.family;
    if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&ours.p)))) return 1;
    std::vector<Answer> mine;
    for (int f = 0; f < kFeatureCount; ++f) mine.push_back(measure(ours.p, f));
    std::vector<std::vector<Answer>> hosts;
    Ref<IDXGIFactory4> factory;
    if (!hostwine::host_is_wine() && SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p)))) {
        for (const bool warp : {false, true}) {
            Ref<IDXGIAdapter> adapter;
            Ref<ID3D12Device> host;
            std::string name;
            if (!hostadapter::reference(factory.p, vk, warp, &adapter.p, name)) continue;
            if (FAILED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) continue;
            std::vector<Answer> answers;
            for (int f = 0; f < kFeatureCount; ++f) answers.push_back(measure(host.p, f));
            hosts.push_back(std::move(answers));
        }
    }
    int compared = 0, mismatches = 0;
    for (int f = 0; f < kFeatureCount; ++f) {
        if (features_skipped(f)) continue;
        bool agree = !hosts.empty();
        for (const auto& h : hosts) agree = agree && h[f].size == hosts[0][f].size && h[f].around == hosts[0][f].around;
        const bool with_payload = payload_compared(f);
        if (agree && with_payload)
            for (const auto& h : hosts) agree = agree && h[f].bytes == hosts[0][f].bytes;
        const bool differs = agree && digest(mine[f], with_payload) != digest(hosts[0][f], with_payload);
        if (probe) {
            std::printf("feature %d: layer size %u, host size %u%s\n", f, mine[f].size, hosts.empty() ? 0 : hosts[0][f].size,
                        differs ? "  <-- DIFF" : agree ? "" : "  (hosts differ)");
        }
        if (!agree) continue;
        ++compared;
        if (differs) {
            ++mismatches;
            std::fprintf(stderr, "feature %d: layer size %u (hr", f, mine[f].size);
            for (auto h : mine[f].around) std::fprintf(stderr, " %08x", h);
            std::fprintf(stderr, "), host size %u (hr", hosts[0][f].size);
            for (auto h : hosts[0][f].around) std::fprintf(stderr, " %08x", h);
            std::fprintf(stderr, ")\n");
            for (int k = 0; k < 2; ++k) {
                const auto& b = k ? hosts[0][f].bytes : mine[f].bytes;
                std::fprintf(stderr, "  %s bytes:", k ? "host" : "layer");
                for (unsigned char c : b) std::fprintf(stderr, " %02x", c);
                std::fprintf(stderr, "\n");
            }
        }
    }
    std::printf("feature queries: %d features compared with %zu host adapters, %d differ\n", compared, hosts.size(),
                mismatches);
    if (vk.validation_errors()) return 1;
    return mismatches ? 1 : 0;
}
