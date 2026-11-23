// A default-heap buffer released while a submission that reads it is still in flight (D3D-475, docs/d3d12_vulkan.md
// sections 3.3 and 4.4). Such buffers are ranges of the GPU address arena, zero-filled through its host mapping when
// they are made, so a released buffer's range must not go to the next buffer before the GPU is done with it.
//
// The read is held back by the GPU, not by the CPU: queue A waits for a fence that queue C signals after a run of
// 64 MiB copies between host buffers (the gate, sized from a measured run to last about 300 ms). C's Signal is
// submitted first, so A's batch is submitted at once with a Vulkan wait (nothing is held, queue_holds.h). C is a COPY
// queue; on a device that gives a single queue (llvmpipe) the gate runs on A itself, DIRECT, ahead of the read. While
// the gate is closed the buffer is released and one of the same size is made. Checked:
//   1. the new buffer gets another range and A reads the released buffer's data, not the new buffer's zeros;
//   2. once A has completed (and the submission after the release), the next buffer gets the released range back;
//   3. with a 16 MiB arena: a 12 MiB buffer made and released four times with nothing submitted gets the same range
//      each time, and one made while a released 12 MiB buffer still waits for a gated read waits for the GPU and
//      gets that range, instead of failing or taking it under the read.
// No game data.
#include "../shims/d3d12_com.h"

#include <wrl/client.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {

constexpr UINT64 kGateBytes = 64ull << 20;
constexpr double kGateMs = 300;
bool skipped = false;  // no Vulkan device or no DIRECT queue: exit 77

#define CHECK(c)                                                         \
    do {                                                                 \
        if (!(c)) {                                                      \
            std::fprintf(stderr, "line %d: check failed: %s\n", __LINE__, #c); \
            return false;                                                \
        }                                                                \
    } while (0)

std::uint32_t pattern(std::uint32_t seed, std::uint64_t i) {
    std::uint32_t x = seed * 0x9e3779b1u + std::uint32_t(i) * 0x85ebca6bu + 0x1234567u;
    x ^= x >> 15; x *= 0x2c1b3c6du; x ^= x >> 12; x *= 0x297a2d39u; x ^= x >> 15;
    return x | 0x01000000u;  // never zero, so a zero-filled range cannot match
}

double ms_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

ComPtr<ID3D12Resource> make_buffer(ID3D12Device* device, D3D12_HEAP_TYPE type, UINT64 bytes) {
    D3D12_HEAP_PROPERTIES heap{type};
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    const D3D12_RESOURCE_STATES state = type == D3D12_HEAP_TYPE_UPLOAD     ? D3D12_RESOURCE_STATE_GENERIC_READ
                                        : type == D3D12_HEAP_TYPE_READBACK ? D3D12_RESOURCE_STATE_COPY_DEST
                                                                           : D3D12_RESOURCE_STATE_COMMON;
    ComPtr<ID3D12Resource> resource;
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                               IID_PPV_ARGS(&resource))))
        return {};
    return resource;
}

bool fill(ID3D12Resource* upload, std::uint32_t seed, UINT64 bytes) {
    std::uint32_t* words = nullptr;
    const D3D12_RANGE none{0, 0};
    if (FAILED(upload->Map(0, &none, reinterpret_cast<void**>(&words)))) return false;
    for (UINT64 i = 0; i < bytes / 4; ++i) words[i] = pattern(seed, i);
    upload->Unmap(0, nullptr);
    return true;
}

// The number of words of `readback` that differ from the pattern (3 printed).
UINT64 differences(ID3D12Resource* readback, std::uint32_t seed, UINT64 bytes) {
    void* data = nullptr;
    if (FAILED(readback->Map(0, nullptr, &data))) return bytes / 4;
    const auto* words = static_cast<const std::uint32_t*>(data);
    UINT64 bad = 0;
    for (UINT64 i = 0; i < bytes / 4; ++i)
        if (words[i] != pattern(seed, i) && bad++ < 3)
            std::fprintf(stderr, "word %llu: got %08x want %08x\n", static_cast<unsigned long long>(i), words[i],
                         pattern(seed, i));
    const D3D12_RANGE none{0, 0};
    readback->Unmap(0, &none);
    return bad;
}

void wait(ID3D12Fence* fence, UINT64 value) {
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (event && SUCCEEDED(fence->SetEventOnCompletion(value, event)) &&
        WaitForSingleObject(event, 30000) == WAIT_OBJECT_0) {
        CloseHandle(event);
        return;
    }
    // A stuck queue can keep the destructors blocked: end the process without cleaning up.
    std::fprintf(stderr, "fence never reached %llu (completed %llu)\n", static_cast<unsigned long long>(value),
                 static_cast<unsigned long long>(fence->GetCompletedValue()));
    std::fflush(stderr);
    std::_Exit(1);
}

D3D12_RESOURCE_BARRIER transition(ID3D12Resource* resource, D3D12_RESOURCE_STATES before,
                                  D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after};
    return barrier;
}

// One device: a COPY queue that runs the gate and a DIRECT queue (the reader) that copies behind it.
struct Rig {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> gate_queue, reader;
    D3D12_COMMAND_LIST_TYPE gate_type = D3D12_COMMAND_LIST_TYPE_COPY;
    ComPtr<ID3D12Fence> gate, done;
    UINT64 gate_value = 0, done_value = 0;
    ComPtr<ID3D12Resource> gate_src, gate_dst;
    UINT gate_copies = 2;
    std::vector<ComPtr<ID3D12CommandAllocator>> allocators;
    std::vector<ComPtr<ID3D12GraphicsCommandList>> lists;

    // 77: no Vulkan device (skipped), 1: failure, 0: ready.
    int init() {
        if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&device)))) return 77;
        D3D12_COMMAND_QUEUE_DESC desc{D3D12_COMMAND_LIST_TYPE_DIRECT};
        if (FAILED(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&reader)))) return 77;
        desc.Type = D3D12_COMMAND_LIST_TYPE_COPY;
        if (FAILED(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&gate_queue)))) {
            gate_queue = reader;  // one queue: the gate goes ahead of the read on it
            gate_type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        }
        std::printf("gate on %s\n", gate_queue == reader ? "the reader's DIRECT queue (no second queue)"
                                                         : "a COPY queue, the read on a DIRECT one");
        if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&gate))) ||
            FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&done))))
            return 1;
        gate_src = make_buffer(device.Get(), D3D12_HEAP_TYPE_UPLOAD, kGateBytes);
        gate_dst = make_buffer(device.Get(), D3D12_HEAP_TYPE_READBACK, kGateBytes);
        return gate_src && gate_dst ? 0 : 1;
    }

    ID3D12GraphicsCommandList* begin(D3D12_COMMAND_LIST_TYPE type) {
        ComPtr<ID3D12CommandAllocator> allocator;
        ComPtr<ID3D12GraphicsCommandList> list;
        if (FAILED(device->CreateCommandAllocator(type, IID_PPV_ARGS(&allocator))) ||
            FAILED(device->CreateCommandList(0, type, allocator.Get(), nullptr, IID_PPV_ARGS(&list))))
            return nullptr;
        allocators.push_back(allocator);
        lists.push_back(list);
        return list.Get();
    }

    // A closed reader list that copies `bytes` from `src` to `dst`, moving a default-heap side out of COMMON and back.
    ID3D12GraphicsCommandList* copy(ID3D12Resource* dst, ID3D12Resource* src, UINT64 bytes, bool dst_default) {
        ID3D12GraphicsCommandList* list = begin(D3D12_COMMAND_LIST_TYPE_DIRECT);
        if (!list) return nullptr;
        ID3D12Resource* side = dst_default ? dst : src;
        const D3D12_RESOURCE_STATES state =
            dst_default ? D3D12_RESOURCE_STATE_COPY_DEST : D3D12_RESOURCE_STATE_COPY_SOURCE;
        const auto in = transition(side, D3D12_RESOURCE_STATE_COMMON, state);
        const auto out = transition(side, state, D3D12_RESOURCE_STATE_COMMON);
        list->ResourceBarrier(1, &in);
        list->CopyBufferRegion(dst, 0, src, 0, bytes);
        list->ResourceBarrier(1, &out);
        return SUCCEEDED(list->Close()) ? list : nullptr;
    }

    // Submits `list` on the reader followed by Signal(done); returns the value, 0 on failure.
    UINT64 submit(ID3D12GraphicsCommandList* list) {
        if (!list) return 0;
        ID3D12CommandList* batch[] = {list};
        reader->ExecuteCommandLists(1, batch);
        return SUCCEEDED(reader->Signal(done.Get(), ++done_value)) ? done_value : 0;
    }

    bool run(ID3D12GraphicsCommandList* list) {
        const UINT64 value = submit(list);
        if (value) wait(done.Get(), value);
        return value != 0;
    }

    // The gate on its queue: `gate_copies` copies of 64 MiB from an upload to a readback buffer, one after the other
    // (a null aliasing barrier between them is a full memory barrier), then Signal(gate). Returns the value.
    UINT64 submit_gate() {
        ID3D12GraphicsCommandList* list = begin(gate_type);
        if (!list) return 0;
        D3D12_RESOURCE_BARRIER between{};
        between.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
        for (UINT i = 0; i < gate_copies; ++i) {
            if (i) list->ResourceBarrier(1, &between);
            list->CopyBufferRegion(gate_dst.Get(), 0, gate_src.Get(), 0, kGateBytes);
        }
        if (FAILED(list->Close())) return 0;
        ID3D12CommandList* batch[] = {list};
        gate_queue->ExecuteCommandLists(1, batch);
        return SUCCEEDED(gate_queue->Signal(gate.Get(), ++gate_value)) ? gate_value : 0;
    }

    // Times two gate copies and sizes the gate to about kGateMs.
    bool calibrate() {
        gate_copies = 2;
        const auto start = std::chrono::steady_clock::now();
        const UINT64 value = submit_gate();
        if (!value) return false;
        wait(gate.Get(), value);
        const double ms = std::max(ms_since(start), 0.5);
        gate_copies = std::clamp(UINT(2 * kGateMs / ms) + 1, 2u, 400u);
        std::printf("gate: 2 copies of 64 MiB took %.1f ms, so %u copies\n", ms, gate_copies);
        return true;
    }

    // Puts `list` (the read) on the reader behind the gate. Returns the read's done value.
    UINT64 gated(ID3D12GraphicsCommandList* list, UINT64& gate_at) {
        gate_at = submit_gate();
        if (!gate_at || FAILED(reader->Wait(gate.Get(), gate_at))) return 0;
        return submit(list);
    }

    bool gate_closed(UINT64 at) const { return gate->GetCompletedValue() < at; }
};

// 1 and 2: a 1 MiB buffer released under a gated read, on the default arena.
bool in_flight_read() {
    Rig rig;
    if (const int status = rig.init()) return skipped = status == 77;
    ID3D12Device* device = rig.device.Get();
    constexpr UINT64 bytes = 1ull << 20;
    auto upload = make_buffer(device, D3D12_HEAP_TYPE_UPLOAD, bytes);
    auto readback = make_buffer(device, D3D12_HEAP_TYPE_READBACK, bytes);
    auto released = make_buffer(device, D3D12_HEAP_TYPE_DEFAULT, bytes);
    CHECK(upload && readback && released && fill(upload.Get(), 1, bytes));
    CHECK(rig.run(rig.copy(released.Get(), upload.Get(), bytes, true)));
    CHECK(rig.calibrate());

    UINT64 gate = 0;
    const UINT64 read = rig.gated(rig.copy(readback.Get(), released.Get(), bytes, false), gate);
    CHECK(read);
    CHECK(rig.gate_closed(gate));  // else the gate is too short to show anything
    const D3D12_GPU_VIRTUAL_ADDRESS va = released->GetGPUVirtualAddress();
    released.Reset();
    auto next = make_buffer(device, D3D12_HEAP_TYPE_DEFAULT, bytes);  // zero-filled through the host mapping
    CHECK(next);
    const D3D12_GPU_VIRTUAL_ADDRESS next_va = next->GetGPUVirtualAddress();
    const bool closed = rig.gate_closed(gate);
    wait(rig.done.Get(), read);
    const UINT64 bad = differences(readback.Get(), 1, bytes);
    std::printf("in-flight read: released buffer at VA 0x%llx; a buffer of the same size made while the gate was %s "
                "got VA 0x%llx; the gated read found %llu of %llu words wrong\n",
                static_cast<unsigned long long>(va), closed ? "closed" : "already open",
                static_cast<unsigned long long>(next_va), static_cast<unsigned long long>(bad),
                static_cast<unsigned long long>(bytes / 4));
    CHECK(closed);
    CHECK(next_va != va);
    CHECK(bad == 0);

    // The read has completed. The release also waits for the submission after the last one made before it: the
    // next run is that one, and the one after it gives the range back when it starts.
    CHECK(rig.run(rig.copy(next.Get(), upload.Get(), bytes, true)));
    CHECK(rig.run(rig.copy(next.Get(), upload.Get(), bytes, true)));
    auto again = make_buffer(device, D3D12_HEAP_TYPE_DEFAULT, bytes);
    CHECK(again);
    std::printf("after the read completed: the next 1 MiB buffer got VA 0x%llx\n",
                static_cast<unsigned long long>(again->GetGPUVirtualAddress()));
    CHECK(again->GetGPUVirtualAddress() == va);
    return true;
}

// 3: a 16 MiB arena, 12 MiB buffers.
bool small_arena() {
    Rig rig;
    _putenv_s("LEONIDA_ARENA_MB", "16");
    const int status = rig.init();
    _putenv_s("LEONIDA_ARENA_MB", "");
    if (status) return skipped = status == 77;
    ID3D12Device* device = rig.device.Get();
    constexpr UINT64 bytes = 12ull << 20;

    D3D12_GPU_VIRTUAL_ADDRESS churn_va = 0;
    for (int i = 0; i < 4; ++i) {  // nothing submitted: each release is free for the next buffer at once
        auto buffer = make_buffer(device, D3D12_HEAP_TYPE_DEFAULT, bytes);
        CHECK(buffer);
        if (!i) churn_va = buffer->GetGPUVirtualAddress();
        CHECK(buffer->GetGPUVirtualAddress() == churn_va);
    }
    std::printf("16 MiB arena: a 12 MiB buffer made and released 4 times with nothing submitted: VA 0x%llx each time\n",
                static_cast<unsigned long long>(churn_va));

    auto upload = make_buffer(device, D3D12_HEAP_TYPE_UPLOAD, bytes);
    auto readback = make_buffer(device, D3D12_HEAP_TYPE_READBACK, bytes);
    auto released = make_buffer(device, D3D12_HEAP_TYPE_DEFAULT, bytes);
    CHECK(upload && readback && released && fill(upload.Get(), 2, bytes));
    CHECK(!make_buffer(device, D3D12_HEAP_TYPE_DEFAULT, bytes));  // the arena holds one such buffer, not two
    CHECK(rig.run(rig.copy(released.Get(), upload.Get(), bytes, true)));
    CHECK(rig.calibrate());

    UINT64 gate = 0;
    const UINT64 read = rig.gated(rig.copy(readback.Get(), released.Get(), bytes, false), gate);
    CHECK(read);
    CHECK(rig.gate_closed(gate));
    const D3D12_GPU_VIRTUAL_ADDRESS va = released->GetGPUVirtualAddress();
    released.Reset();
    const auto start = std::chrono::steady_clock::now();
    auto next = make_buffer(device, D3D12_HEAP_TYPE_DEFAULT, bytes);
    const double ms = ms_since(start);
    const bool waited = !rig.gate_closed(gate);
    CHECK(next);
    wait(rig.done.Get(), read);
    const UINT64 bad = differences(readback.Get(), 2, bytes);
    std::printf("16 MiB arena: a 12 MiB buffer made while a released one waited for a gated read took %.1f ms "
                "(gate %s when it returned), got VA 0x%llx (released: 0x%llx); the read found %llu of %llu words "
                "wrong\n",
                ms, waited ? "open" : "still closed", static_cast<unsigned long long>(next->GetGPUVirtualAddress()),
                static_cast<unsigned long long>(va), static_cast<unsigned long long>(bad),
                static_cast<unsigned long long>(bytes / 4));
    CHECK(waited);
    CHECK(next->GetGPUVirtualAddress() == va);
    CHECK(bad == 0);
    return true;
}

}  // namespace

int main() {
    const bool in_flight = in_flight_read();
    const bool arena_16 = !skipped && small_arena();
    if (skipped) {
        std::puts("no Vulkan device with a DIRECT queue: skipped");
        return 77;
    }
    const bool ok = in_flight && arena_16;
    std::puts(ok ? "arena ranges of released buffers came back only after the GPU was done with them" : "FAILED");
    return ok ? 0 : 1;
}
