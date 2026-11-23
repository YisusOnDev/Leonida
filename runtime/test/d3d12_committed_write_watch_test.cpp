// CreateCommittedResource with D3D12_HEAP_FLAG_ALLOW_WRITE_WATCH, against the host (PROBE_WARP=1: WARP). The host
// takes the flag on a heap the CPU can write (UPLOAD, READBACK, CUSTOM with a CPU page property), on buffers and
// textures, keeps it in GetHeapProperties, and refuses it on DEFAULT and on CUSTOM NOT_AVAILABLE. RTX 5070 Ti and WARP
// answer alike.
#include "../shims/d3d12_com.h"
#include <dxgi1_4.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {
template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
};

D3D12_RESOURCE_DESC buffer_desc() {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = 4096;
    d.Height = 1;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return d;
}

D3D12_RESOURCE_DESC texture_desc() {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Width = 16;
    d.Height = 16;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    d.SampleDesc.Count = 1;
    return d;
}

std::string run_case(ID3D12Device* dev, D3D12_HEAP_PROPERTIES hp, D3D12_HEAP_FLAGS hf, const D3D12_RESOURCE_DESC& d,
                     D3D12_RESOURCE_STATES state) {
    Ref<ID3D12Resource> r;
    HRESULT hr = dev->CreateCommittedResource(&hp, hf, &d, state, nullptr, IID_PPV_ARGS(&r.p));
    char line[160];
    std::snprintf(line, sizeof(line), "hr %08lx", static_cast<unsigned long>(hr));
    std::string s = line;
    if (SUCCEEDED(hr) && r.p) {
        D3D12_HEAP_PROPERTIES p{};
        D3D12_HEAP_FLAGS f{};
        hr = r.p->GetHeapProperties(&p, &f);
        std::snprintf(line, sizeof(line), " props %08lx type %d flags %x", static_cast<unsigned long>(hr), p.Type,
                      static_cast<unsigned>(f));
        s += line;
        if (d.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
            void* m = nullptr;
            hr = r.p->Map(0, nullptr, &m);
            std::snprintf(line, sizeof(line), " map %08lx", static_cast<unsigned long>(hr));
            s += line;
            if (SUCCEEDED(hr)) r.p->Unmap(0, nullptr);
        }
    }
    return s;
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

    const auto ww = D3D12_HEAP_FLAG_ALLOW_WRITE_WATCH;
    const auto gr = D3D12_RESOURCE_STATE_GENERIC_READ, cd = D3D12_RESOURCE_STATE_COPY_DEST,
               common = D3D12_RESOURCE_STATE_COMMON;
    struct Case {
        const char* name;
        D3D12_HEAP_PROPERTIES heap;
        D3D12_HEAP_FLAGS flags;
        D3D12_RESOURCE_DESC desc;
        D3D12_RESOURCE_STATES state;
    };
    const D3D12_HEAP_PROPERTIES def{D3D12_HEAP_TYPE_DEFAULT}, up{D3D12_HEAP_TYPE_UPLOAD}, rb{D3D12_HEAP_TYPE_READBACK},
        wc{D3D12_HEAP_TYPE_CUSTOM, D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE, D3D12_MEMORY_POOL_L0},
        wb{D3D12_HEAP_TYPE_CUSTOM, D3D12_CPU_PAGE_PROPERTY_WRITE_BACK, D3D12_MEMORY_POOL_L0},
        na{D3D12_HEAP_TYPE_CUSTOM, D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE, D3D12_MEMORY_POOL_L0};
    const std::vector<Case> cases = {
        {"upload buffer", up, ww, buffer_desc(), gr},
        {"upload buffer, not zeroed", up, D3D12_HEAP_FLAGS(ww | D3D12_HEAP_FLAG_CREATE_NOT_ZEROED), buffer_desc(), gr},
        {"upload buffer, not resident", up, D3D12_HEAP_FLAGS(ww | D3D12_HEAP_FLAG_CREATE_NOT_RESIDENT), buffer_desc(),
         gr},
        {"readback buffer", rb, ww, buffer_desc(), cd},
        {"custom write-combine buffer", wc, ww, buffer_desc(), common},
        {"custom write-back buffer", wb, ww, buffer_desc(), common},
        {"custom write-back texture", wb, ww, texture_desc(), common},
        {"custom write-combine texture", wc, ww, texture_desc(), common},
        // Refused on both.
        {"default buffer", def, ww, buffer_desc(), common},
        {"default texture", def, ww, texture_desc(), common},
        {"custom not-available buffer", na, ww, buffer_desc(), common},
        {"upload buffer, deny buffers", up, D3D12_HEAP_FLAGS(ww | D3D12_HEAP_FLAG_DENY_BUFFERS), buffer_desc(), gr},
    };
    int bad = 0;
    for (const Case& c : cases) {
        const std::string a = run_case(ours.p, c.heap, c.flags, c.desc, c.state);
        const std::string b = run_case(host.p, c.heap, c.flags, c.desc, c.state);
        if (a != b) {
            ++bad;
            std::printf("%s differs:\n  ours: %s\n  host: %s\n", c.name, a.c_str(), b.c_str());
        } else {
            std::printf("%s: %s\n", c.name, a.c_str());
        }
    }
    std::printf("committed write watch: %d cases differ\n", bad);
    return bad ? 1 : 0;
}
