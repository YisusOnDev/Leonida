#pragma once
// Title-chosen buffer VAs (Xbox device slot 133) and the console VA a translated shader reaches them by
// (docs/d3d12_vulkan.md section 3.3, "Title-placed buffers").
//
// A translated shader computes heap_base + console_va, with heap_base the GPU address arena's device address. The
// layer chooses every console VA it hands out, so they all land in the arena, but slot 133 places a buffer at a VA
// the title chose: the CPU address of its own pages (about 1.5 TiB on Windows), far past the arena. A shader that
// adds heap_base to that VA reads nothing the layer made. So where the layer itself writes an address a shader will
// dereference (a root CBV/SRV/UAV, and the V# of a CBV or a buffer SRV/UAV), an address inside a title-placed
// buffer is replaced by the console VA of the buffer's GPU copy: its arena offset when the copy is an arena range
// (the slot-133 shadow), or its device address minus heap_base (mod 2^64) when it has its own memory (an imported
// buffer). Everything else is passed through unchanged. Addresses the title writes into memory itself (a V# it
// builds in its own constant data) are not seen by the layer and keep the title's VA.
//
// The table is process-wide, like the title's address space: a title VA names one buffer at a time. It depends only
// on the standard library, so it is tested without a GPU (title_va_alias_test).
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <map>
#include <mutex>
#include <shared_mutex>

namespace leonida {

class TitleVaAliases {
public:
    // [title_va, title_va + bytes) is reached as [shader_va, shader_va + bytes). A range that overlaps a live one
    // replaces it (the title has placed a new buffer over pages it released). False for an empty or wrapping range.
    bool add(std::uint64_t title_va, std::uint64_t bytes, std::uint64_t shader_va) {
        if (!bytes || title_va > UINT64_MAX - bytes) return false;
        std::unique_lock lock(mutex_);
        erase_overlapping(title_va, title_va + bytes);
        ranges_[title_va] = {title_va + bytes, shader_va};
        count_.store(ranges_.size(), std::memory_order_release);
        return true;
    }

    // Forgets the range that starts at title_va, if it still maps to shader_va (a newer buffer at the same VA keeps
    // its own entry).
    void remove(std::uint64_t title_va, std::uint64_t shader_va) {
        std::unique_lock lock(mutex_);
        const auto it = ranges_.find(title_va);
        if (it != ranges_.end() && it->second.shader_va == shader_va) ranges_.erase(it);
        count_.store(ranges_.size(), std::memory_order_release);
    }

    // The address a shader must be given for `va`: translated inside a title-placed buffer, unchanged elsewhere.
    std::uint64_t shader_va(std::uint64_t va) const {
        if (!count_.load(std::memory_order_acquire)) return va;  // the common case: no title-placed buffer
        std::shared_lock lock(mutex_);
        auto it = ranges_.upper_bound(va);
        if (it == ranges_.begin()) return va;
        --it;
        if (va >= it->second.end) return va;
        return it->second.shader_va + (va - it->first);
    }

    std::size_t size() const { return count_.load(std::memory_order_acquire); }

private:
    struct Range {
        std::uint64_t end;        // exclusive
        std::uint64_t shader_va;  // what title_va becomes
    };

    void erase_overlapping(std::uint64_t begin, std::uint64_t end) {
        auto it = ranges_.lower_bound(begin);
        if (it != ranges_.begin()) {
            auto previous = std::prev(it);
            if (previous->second.end > begin) it = previous;
        }
        while (it != ranges_.end() && it->first < end) it = ranges_.erase(it);
    }

    mutable std::shared_mutex mutex_;
    std::map<std::uint64_t, Range> ranges_;  // by title VA
    std::atomic<std::size_t> count_{0};
};

// The process's table: the slot-133 buffers add and remove their ranges, the command lists and descriptor writers
// read it.
inline TitleVaAliases& title_va_aliases() {
    static TitleVaAliases aliases;
    return aliases;
}

}  // namespace leonida
