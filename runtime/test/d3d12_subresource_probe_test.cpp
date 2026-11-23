// WriteToSubresource / ReadFromSubresource on odd-sized block-compressed and mipped textures, against the host
// (PROBE_WARP=1: WARP). Every HRESULT and a hash of every block row read back must match. The host takes a
// block-compressed box in whole blocks up to the block-padded mip edge ((0,0)-(8,8) on a 6x6 mip) and refuses one that
// stops at an unpadded edge inside a block ((0,0)-(6,6)); NVIDIA 610-series and WARP answer alike.
#include "../shims/d3d12_com.h"
#include <dxgi1_4.h>

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {
template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
};

struct Tex {
    const char* name;
    D3D12_RESOURCE_DIMENSION dim;
    DXGI_FORMAT format;
    UINT w, h;
    UINT16 d, mips;
    UINT bytes, block;
};

struct Op {
    UINT sub;
    bool has_box;
    D3D12_BOX box;
};

std::vector<std::string> run(ID3D12Device* device, const Tex& t, const std::vector<Op>& ops) {
    std::vector<std::string> log;
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_CUSTOM;
    heap.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
    heap.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = t.dim;
    desc.Width = t.w;
    desc.Height = t.h;
    desc.DepthOrArraySize = t.d;
    desc.MipLevels = t.mips;
    desc.Format = t.format;
    desc.SampleDesc.Count = 1;
    Ref<ID3D12Resource> r;
    HRESULT hr = device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON,
                                                 nullptr, IID_PPV_ARGS(&r.p));
    if (FAILED(hr)) {
        log.push_back("create failed");
        return log;
    }
    r.p->Map(0, nullptr, nullptr);
    const bool volume = t.dim == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    const UINT layers = volume ? 1 : t.d;
    const UINT n = t.mips * layers;
    constexpr UINT pitch = 1024, slice = 65536;
    std::vector<std::uint8_t> src(slice * 8), dst(slice * 8);
    unsigned seed = 1;
    // Zero every subresource first.
    std::vector<std::uint8_t> zero(slice * 8, 0);
    for (UINT s = 0; s < n; ++s) r.p->WriteToSubresource(s, nullptr, zero.data(), pitch, slice);
    for (const Op& op : ops) {
        for (auto& b : src) b = std::uint8_t((seed = seed * 1103515245u + 12345u) >> 16);
        hr = r.p->WriteToSubresource(op.sub, op.has_box ? &op.box : nullptr, src.data(), pitch, slice);
        char line[128];
        std::snprintf(line, sizeof(line), "write sub %u box %d (%u,%u,%u)-(%u,%u,%u): 0x%08lx", op.sub, op.has_box,
                      op.box.left, op.box.top, op.box.front, op.box.right, op.box.bottom, op.box.back,
                      static_cast<unsigned long>(hr));
        log.push_back(line);
    }
    // Read every subresource whole and hash per row.
    for (UINT s = 0; s < n; ++s) {
        std::fill(dst.begin(), dst.end(), std::uint8_t(0xcd));
        hr = r.p->ReadFromSubresource(dst.data(), pitch, slice, s, nullptr);
        const UINT mip = s % t.mips;
        const UINT w = std::max(1u, t.w >> mip), h = std::max(1u, t.h >> mip);
        const UINT d = volume ? std::max(1u, UINT(t.d) >> mip) : 1u;
        const UINT rows = (h + t.block - 1) / t.block, row_bytes = (w + t.block - 1) / t.block * t.bytes;
        std::string text = "read sub " + std::to_string(s) + " hr " + std::to_string(hr) + ":";
        for (UINT z = 0; z < d; ++z)
            for (UINT y = 0; y < rows; ++y) {
                char hex[4];
                std::uint32_t hsh = 2166136261u;
                for (UINT x = 0; x < row_bytes; ++x) hsh = (hsh ^ dst[z * slice + y * pitch + x]) * 16777619u;
                std::snprintf(hex, sizeof(hex), "%02x", hsh & 0xff);
                text += " ";
                text += hex;
            }
        log.push_back(text);
    }
    return log;
}
}  // namespace

int main() {
    Ref<ID3D12Device> ours, host;
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&ours.p))) || !ours.p) return 77;
    Ref<IDXGIAdapter> adapter;
    if (std::getenv("PROBE_WARP")) {
        Ref<IDXGIFactory4> factory;
        if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p))) ||
            FAILED(factory.p->EnumWarpAdapter(IID_PPV_ARGS(&adapter.p))))
            return 77;
    }
    if (FAILED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) return 77;
    constexpr auto t2 = D3D12_RESOURCE_DIMENSION_TEXTURE2D, t3 = D3D12_RESOURCE_DIMENSION_TEXTURE3D,
                   t1 = D3D12_RESOURCE_DIMENSION_TEXTURE1D;
    struct Case {
        Tex t;
        std::vector<Op> ops;
    };
    const std::vector<Case> cases = {
        {{"BC1 12x12, 3 mips", t2, DXGI_FORMAT_BC1_UNORM, 12, 12, 1, 3, 8, 4},
         {{0, false, {}}, {1, false, {}}, {2, false, {}}, {1, true, {4, 4, 0, 6, 6, 1}}, {0, true, {4, 8, 0, 12, 12, 1}},
          {1, true, {0, 0, 0, 8, 8, 1}}, {2, true, {0, 0, 0, 4, 4, 1}}, {1, true, {0, 0, 0, 6, 6, 1}},
          {2, true, {0, 0, 0, 3, 3, 1}}, {1, true, {4, 0, 0, 8, 4, 1}}, {1, true, {0, 0, 0, 12, 4, 1}},
          {0, true, {0, 0, 0, 16, 4, 1}}}},
        {{"BC7 20x12 x2, 3 mips", t2, DXGI_FORMAT_BC7_UNORM, 20, 12, 2, 3, 16, 4},
         {{0, false, {}}, {4, false, {}}, {5, true, {0, 0, 0, 3, 3, 1}}, {1, true, {8, 4, 0, 10, 6, 1}}}},
        // One mip of a volume: the host refuses CPU copies into a mipped volume or a 96-bit texture outright.
        {{"BC1 3D 12x10x5, 1 mip", t3, DXGI_FORMAT_BC1_UNORM, 12, 10, 5, 1, 8, 4},
         {{0, false, {}}, {0, true, {4, 8, 1, 12, 12, 4}}, {0, true, {4, 8, 1, 12, 10, 4}}}},
        {{"RGBA8 3D 9x5x7, 1 mip", t3, DXGI_FORMAT_R8G8B8A8_UNORM, 9, 5, 7, 1, 4, 1},
         {{0, false, {}}, {0, true, {1, 1, 1, 4, 2, 3}}}},
        {{"R16 1D 77 x3, 3 mips", t1, DXGI_FORMAT_R16_UNORM, 77, 1, 3, 3, 2, 1},
         {{0, false, {}}, {4, true, {5, 0, 0, 30, 1, 1}}, {8, false, {}}}},
        {{"RGBA8 2D 13x7 2 mips", t2, DXGI_FORMAT_R8G8B8A8_UNORM, 13, 7, 1, 2, 4, 1},
         {{0, false, {}}, {1, true, {1, 1, 0, 6, 3, 1}}}},
    };
    int bad = 0;
    for (const Case& c : cases) {
        const auto a = run(ours.p, c.t, c.ops), b = run(host.p, c.t, c.ops);
        bool same = a == b;
        if (!same) {
            ++bad;
            std::fprintf(stderr, "%s differs:\n", c.t.name);
            for (size_t i = 0; i < std::max(a.size(), b.size()); ++i) {
                const std::string& x = i < a.size() ? a[i] : std::string("-");
                const std::string& y = i < b.size() ? b[i] : std::string("-");
                std::fprintf(stderr, "  %s ours: %s\n  %s host: %s\n", x == y ? " " : "*", x.c_str(), x == y ? " " : "*",
                             y.c_str());
            }
        } else {
            std::printf("%s: matches (%zu steps)\n", c.t.name, a.size());
        }
    }
    std::printf("subresource probe: %d cases differ\n", bad);
    return bad ? 1 : 0;
}
