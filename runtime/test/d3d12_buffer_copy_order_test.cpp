// Native D3D12 WARP reference: consecutive writes to COPY_DEST are ordered without an app barrier.
// Original synthetic bytes only; no game data or shaders.
#include <d3d12.h>
#include <dxgi1_4.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

template<class T> struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
};

static D3D12_RESOURCE_DESC buffer_desc(UINT64 bytes) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = bytes;
    d.Height = d.DepthOrArraySize = d.MipLevels = 1;
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return d;
}

int main() {
    constexpr UINT64 bytes = 4096;
    Ref<IDXGIFactory4> factory;
    Ref<IDXGIAdapter> adapter;
    Ref<ID3D12Device> device;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p))) ||
        FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter.p))) ||
        FAILED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device.p)))) {
        std::printf("SKIP: native D3D12 WARP unavailable\n");
        return 77;
    }
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    if (FAILED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))) ||
        FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))) ||
        FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                         IID_PPV_ARGS(&list.p))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p)))) return 1;

    D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
    D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
    const auto upload_desc = buffer_desc(bytes * 2), dst_desc = buffer_desc(bytes);
    Ref<ID3D12Resource> upload, dst, readback;
    if (FAILED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &upload_desc,
                                                D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                IID_PPV_ARGS(&upload.p))) ||
        FAILED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &dst_desc,
                                                D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                IID_PPV_ARGS(&dst.p))) ||
        FAILED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &dst_desc,
                                                D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                IID_PPV_ARGS(&readback.p)))) return 1;
    void* mapped = nullptr;
    if (FAILED(upload->Map(0, nullptr, &mapped))) return 1;
    std::memset(mapped, 0x11, bytes);
    std::memset(static_cast<std::uint8_t*>(mapped) + bytes, 0x22, bytes);
    upload->Unmap(0, nullptr);

    list->CopyBufferRegion(dst.p, 0, upload.p, 0, bytes);
    list->CopyBufferRegion(dst.p, 0, upload.p, bytes, bytes);
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = dst.p;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    list->ResourceBarrier(1, &barrier);
    list->CopyBufferRegion(readback.p, 0, dst.p, 0, bytes);
    const HRESULT closed = list->Close();
    if (closed != S_OK) {
        std::fprintf(stderr, "native WARP Close=0x%08x\n", unsigned(closed));
        return 1;
    }
    ID3D12CommandList* command_lists[] = {list.p};
    queue->ExecuteCommandLists(1, command_lists);
    if (FAILED(queue->Signal(fence.p, 1))) return 1;
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event) return 1;
    const bool signaled = SUCCEEDED(fence->SetEventOnCompletion(1, event)) &&
                          WaitForSingleObject(event, 10000) == WAIT_OBJECT_0;
    CloseHandle(event);
    if (!signaled || device->GetDeviceRemovedReason() != S_OK ||
        FAILED(readback->Map(0, nullptr, &mapped))) return 1;
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < bytes; ++i)
        mismatches += static_cast<const std::uint8_t*>(mapped)[i] != 0x22;
    readback->Unmap(0, nullptr);
    std::printf("native WARP: consecutive overlapping CopyBufferRegion writes, %zu byte mismatches\n", mismatches);
    return mismatches ? 1 : 0;
}
