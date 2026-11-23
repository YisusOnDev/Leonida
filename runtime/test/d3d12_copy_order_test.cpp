// Two copies that write the same bytes or texels, back to back in one command list with no barrier between them.
// D3D12 runs them in order while the destination stays in COPY_DEST, so the second copy's data is what a readback
// must find. Vulkan orders nothing by itself: the layer has to (copy_layout for images, a memory barrier for buffers
// in vulkan_replay.h). Before the buffer side had one, "texture to buffer, then CopyBufferRegion over it" left the
// texture's rows in the buffer on an AMD Radeon RX 9060 XT, every time.
// Synthetic data; needs a Vulkan GPU (exit 77 without one).

#include "../shims/d3d12_com.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr UINT64 buffer_bytes = 16384, upload_bytes = 65536;
constexpr UINT texels = 64, pitch = texels * 4;  // RGBA8, 64 texels: the row pitch needs no padding
constexpr UINT rounds = 16;

int fail(const char* what, HRESULT hr = S_OK) {
    std::fprintf(stderr, "FAIL: %s (hr=0x%08X)\n", what, static_cast<unsigned>(hr));
    return 1;
}

D3D12_RESOURCE_BARRIER transition(ID3D12Resource* r, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition = {r, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, from, to};
    return b;
}

}  // namespace

int main() {
    ID3D12Device* dev = nullptr;
    HRESULT hr = rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&dev));
    if (FAILED(hr) || !dev) {
        std::printf("SKIP: no Vulkan device (hr=0x%08X)\n", static_cast<unsigned>(hr));
        return 77;
    }

    const auto make_buffer = [&](D3D12_HEAP_TYPE heap, UINT64 bytes, D3D12_RESOURCE_STATES state) {
        D3D12_HEAP_PROPERTIES hp{heap};
        D3D12_RESOURCE_DESC d{};
        d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        d.Width = bytes;
        d.Height = 1;
        d.DepthOrArraySize = 1;
        d.MipLevels = 1;
        d.SampleDesc = {1, 0};
        d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        ID3D12Resource* r = nullptr;
        dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, state, nullptr, IID_PPV_ARGS(&r));
        return r;
    };
    const auto make_texture = [&](D3D12_RESOURCE_STATES state) {
        D3D12_HEAP_PROPERTIES hp{D3D12_HEAP_TYPE_DEFAULT};
        D3D12_RESOURCE_DESC d{};
        d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        d.Width = texels;
        d.Height = texels;
        d.DepthOrArraySize = 1;
        d.MipLevels = 1;
        d.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        d.SampleDesc = {1, 0};
        ID3D12Resource* r = nullptr;
        dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, state, nullptr, IID_PPV_ARGS(&r));
        return r;
    };

    ID3D12Resource* upload = make_buffer(D3D12_HEAP_TYPE_UPLOAD, upload_bytes, D3D12_RESOURCE_STATE_GENERIC_READ);
    ID3D12Resource* readback = make_buffer(D3D12_HEAP_TYPE_READBACK, buffer_bytes + pitch * texels,
                                           D3D12_RESOURCE_STATE_COPY_DEST);
    ID3D12Resource* buffer = make_buffer(D3D12_HEAP_TYPE_DEFAULT, buffer_bytes, D3D12_RESOURCE_STATE_COPY_DEST);
    ID3D12Resource* texture = make_texture(D3D12_RESOURCE_STATE_COPY_DEST);  // written by the pairs
    ID3D12Resource* other = make_texture(D3D12_RESOURCE_STATE_COPY_DEST);    // only read by them
    if (!upload || !readback || !buffer || !texture || !other) return fail("CreateCommittedResource");

    // The CPU's copy of every resource: each recorded copy is applied here too.
    std::vector<std::uint8_t> upload_data(upload_bytes), buffer_data(buffer_bytes), texture_data(pitch * texels),
        other_data(pitch * texels);
    std::uint8_t* mapped = nullptr;
    if (FAILED(hr = upload->Map(0, nullptr, reinterpret_cast<void**>(&mapped))) || !mapped) return fail("Map(upload)", hr);
    for (UINT64 i = 0; i < upload_bytes; ++i)
        upload_data[i] = mapped[i] = static_cast<std::uint8_t>((i * 2654435761ull >> 11) ^ (i >> 3) ^ 0x5a);
    upload->Unmap(0, nullptr);

    D3D12_COMMAND_QUEUE_DESC qd{};
    ID3D12CommandQueue* queue = nullptr;
    ID3D12CommandAllocator* alloc = nullptr;
    ID3D12GraphicsCommandList* cmd = nullptr;
    ID3D12Fence* fence = nullptr;
    if (FAILED(hr = dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue)))) return fail("CreateCommandQueue", hr);
    if (FAILED(hr = dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc))))
        return fail("CreateCommandAllocator", hr);
    if (FAILED(hr = dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, nullptr, IID_PPV_ARGS(&cmd))))
        return fail("CreateCommandList", hr);
    if (FAILED(hr = dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)))) return fail("CreateFence", hr);
    HANDLE done = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    UINT64 fence_value = 0;
    const auto execute = [&]() -> bool {
        if (FAILED(cmd->Close())) return false;
        ID3D12CommandList* lists[] = {cmd};
        queue->ExecuteCommandLists(1, lists);
        if (FAILED(queue->Signal(fence, ++fence_value))) return false;
        if (fence->GetCompletedValue() < fence_value) {
            fence->SetEventOnCompletion(fence_value, done);
            if (WaitForSingleObject(done, 60000) != WAIT_OBJECT_0) return false;
        }
        return SUCCEEDED(alloc->Reset()) && SUCCEEDED(cmd->Reset(alloc, nullptr));
    };

    // The four copies. Each writes [4096, 4096 + 30 rows) of the buffer or the 56x56 texels at (4, 4) of the texture.
    const auto location = [](ID3D12Resource* r, UINT64 offset, UINT w, UINT h) {
        D3D12_TEXTURE_COPY_LOCATION l{r, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
        l.PlacedFootprint = {offset, {DXGI_FORMAT_R8G8B8A8_UNORM, w, h, 1, pitch}};
        return l;
    };
    const auto subresource = [](ID3D12Resource* r) {
        D3D12_TEXTURE_COPY_LOCATION l{r, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
        return l;
    };
    const auto buffer_copy = [&](UINT64 source) {
        cmd->CopyBufferRegion(buffer, 4096, upload, source, 30 * pitch);
        std::memcpy(buffer_data.data() + 4096, upload_data.data() + source, 30 * pitch);
    };
    const auto texture_to_buffer = [&](UINT x, UINT y) {
        const D3D12_TEXTURE_COPY_LOCATION dst = location(buffer, 4096, 48, 30), src = subresource(other);
        const D3D12_BOX box{x, y, 0, x + 48, y + 30, 1};
        cmd->CopyTextureRegion(&dst, 0, 0, 0, &src, &box);
        for (UINT row = 0; row < 30; ++row)
            std::memcpy(buffer_data.data() + 4096 + row * pitch, other_data.data() + (y + row) * pitch + x * 4, 48 * 4);
    };
    const auto buffer_to_texture = [&](UINT64 source) {
        const D3D12_TEXTURE_COPY_LOCATION dst = subresource(texture), src = location(upload, source, 56, 56);
        cmd->CopyTextureRegion(&dst, 4, 4, 0, &src, nullptr);
        for (UINT row = 0; row < 56; ++row)
            std::memcpy(texture_data.data() + (4 + row) * pitch + 4 * 4, upload_data.data() + source + row * pitch, 56 * 4);
    };
    const auto texture_copy = [&](UINT x, UINT y) {
        const D3D12_TEXTURE_COPY_LOCATION dst = subresource(texture), src = subresource(other);
        const D3D12_BOX box{x, y, 0, x + 56, y + 56, 1};
        cmd->CopyTextureRegion(&dst, 4, 4, 0, &src, &box);
        for (UINT row = 0; row < 56; ++row)
            std::memcpy(texture_data.data() + (4 + row) * pitch + 4 * 4, other_data.data() + (y + row) * pitch + x * 4, 56 * 4);
    };
    const auto write = [&](int kind, UINT v) {
        if (kind == 0) buffer_copy(1024 + v * 16);
        else if (kind == 1) texture_to_buffer(v % 8, v % 5);
        else if (kind == 2) buffer_to_texture(2048 + UINT64(v % 16) * 512);
        else texture_copy(v % 8, v % 8);
    };

    // `other` gets its contents once and is a copy source from then on.
    {
        const D3D12_TEXTURE_COPY_LOCATION dst = subresource(other), src = location(upload, 32768, texels, texels);
        cmd->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        std::memcpy(other_data.data(), upload_data.data() + 32768, pitch * texels);
        const D3D12_RESOURCE_BARRIER b = transition(other, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
        cmd->ResourceBarrier(1, &b);
        if (!execute()) return fail("ExecuteCommandLists (setup)");
    }

    struct Pair { const char* name; int first, second; };
    const Pair pairs[] = {
        {"texture to buffer, then buffer copy", 1, 0},   {"buffer copy, then texture to buffer", 0, 1},
        {"buffer copy twice", 0, 0},                     {"texture to buffer twice", 1, 1},
        {"buffer to texture, then texture copy", 2, 3},  {"texture copy, then buffer to texture", 3, 2},
        {"buffer to texture twice", 2, 2},               {"texture copy twice", 3, 3},
    };
    UINT wrong_pairs = 0;
    for (const Pair& pair : pairs) {
        UINT wrong = 0;
        for (UINT round = 0; round < rounds; ++round) {
            // New contents first; the pair's first copy follows them with no barrier either.
            cmd->CopyBufferRegion(buffer, 0, upload, 8192 + round * 64, buffer_bytes);
            std::memcpy(buffer_data.data(), upload_data.data() + 8192 + round * 64, buffer_bytes);
            const D3D12_TEXTURE_COPY_LOCATION whole = subresource(texture),
                                              fresh = location(upload, 16384 + UINT64(round) * 512, texels, texels);
            cmd->CopyTextureRegion(&whole, 0, 0, 0, &fresh, nullptr);
            std::memcpy(texture_data.data(), upload_data.data() + 16384 + UINT64(round) * 512, pitch * texels);

            write(pair.first, round * 7);
            write(pair.second, round * 7 + 3);

            D3D12_RESOURCE_BARRIER to_source[] = {
                transition(buffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE),
                transition(texture, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE)};
            cmd->ResourceBarrier(2, to_source);
            cmd->CopyBufferRegion(readback, 0, buffer, 0, buffer_bytes);
            const D3D12_TEXTURE_COPY_LOCATION out = location(readback, buffer_bytes, texels, texels), in = subresource(texture);
            cmd->CopyTextureRegion(&out, 0, 0, 0, &in, nullptr);
            D3D12_RESOURCE_BARRIER to_dest[] = {
                transition(buffer, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST),
                transition(texture, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST)};
            cmd->ResourceBarrier(2, to_dest);
            if (!execute()) return fail("ExecuteCommandLists");

            const std::uint8_t* got = nullptr;
            if (FAILED(hr = readback->Map(0, nullptr, reinterpret_cast<void**>(const_cast<std::uint8_t**>(&got)))) || !got)
                return fail("Map(readback)", hr);
            const bool same = !std::memcmp(got, buffer_data.data(), buffer_bytes) &&
                              !std::memcmp(got + buffer_bytes, texture_data.data(), pitch * texels);
            readback->Unmap(0, nullptr);
            wrong += !same;
        }
        if (wrong) {
            std::fprintf(stderr, "FAIL: %s: the readback is not the second copy's data in %u of %u rounds\n", pair.name,
                         wrong, rounds);
            ++wrong_pairs;
        } else {
            std::printf("ok: %s\n", pair.name);
        }
    }

    CloseHandle(done);
    fence->Release();
    cmd->Release();
    alloc->Release();
    queue->Release();
    other->Release();
    texture->Release();
    buffer->Release();
    readback->Release();
    upload->Release();
    dev->Release();
    if (wrong_pairs) return 1;
    std::printf("PASS: d3d12_copy_order_test (%zu pairs of back-to-back copies, %u rounds each)\n",
                sizeof(pairs) / sizeof(pairs[0]), rounds);
    return 0;
}
