// Console (RDNA2, GFX10.3) texture layouts: a texture's size and alignment, where each mip and array slice starts, the
// byte offset of each element, and the runs a copy between tiled memory and linear rows takes. xg_xs.dll's texture
// computer answers the title from this (runtime/shims/xg_xs.cpp).
//
// Origin: a C++ port of crates/victory-formats/src/texture.rs of VIctory (github.com/Marvisak/VIctory, upstream
// 204bdcc, GPL-3.0-or-later), which follows AMD's address library (addrlib, as shipped in Mesa under src/amd/addrlib:
// gfx10addrlib.cpp, addrlib2.cpp and the RB+ micro tile patterns of gfx10SwizzlePattern.h). addrlib's notice:
//
//   Copyright (C) 2007-2026 Advanced Micro Devices, Inc. All rights reserved.
//   SPDX-License-Identifier: MIT
//   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
//   documentation files (the "Software"), to deal in the Software without restriction, including without limitation
//   the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
//   to permit persons to whom the Software is furnished to do so, subject to the following conditions: the above
//   copyright notice and this permission notice shall be included in all copies or substantial portions of the
//   Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.
//
// Units: an element is a texel, or a 4x4 block of a block-compressed format. The mips of one array slice (or one depth
// slab of a thick 3D texture) are stored together, smallest first: the mip tail (the small mips sharing one block) at
// offset 0, mip 0 last. Not covered (as in the original): addressing of the pipe/bank XOR modes (_X, _T) and MSAA,
// compression metadata planes (HTILE, CMASK, FMASK, DCC), 96-bit, 1-bit, packed 4:2:2 and video formats.
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace leonida::console_layout {

// Swizzle modes by hardware number (addrlib's ADDR_SW_* order).
enum : uint32_t {
    kLinear = 0, kSw256bS = 1, kSw256bD = 2, kSw4kbS = 5, kSw4kbD = 6, kSw64kbS = 9, kSw64kbD = 10,
    kSw64kbST = 17, kSw64kbDT = 18, kSw4kbSX = 21, kSw4kbDX = 22, kSw4kbRX = 23, kSw64kbZX = 24,
    kSw64kbSX = 25, kSw64kbDX = 26, kSw64kbRX = 27,
};

// log2 of the block size in bytes (0 linear), -1 for a mode GFX10 lacks.
inline int block_log2(uint32_t s) {
    switch (s) {
    case 0: return 0;
    case 1: case 2: return 8;
    case 5: case 6: case 21: case 22: case 23: return 12;
    case 9: case 10: case 17: case 18: case 24: case 25: case 26: case 27: return 16;
    default: return -1;
    }
}
inline bool standard(uint32_t s) { return s == 1 || s == 5 || s == 9 || s == 17 || s == 21 || s == 25; }
inline bool display(uint32_t s) { return s == 2 || s == 6 || s == 10 || s == 18 || s == 22 || s == 26; }
inline bool z_order(uint32_t s) { return s == 24; }
inline bool xor_mode(uint32_t s) { return s >= 16; }

enum class Dimension { tex1d, tex2d, tex3d };
enum class Error { none, invalid, unsupported, too_small };

struct Desc {
    Dimension dimension = Dimension::tex2d;
    uint32_t width = 0, height = 0;
    uint32_t depth = 1;  // 3D depth, else the array size
    uint32_t mips = 1;
    uint32_t element_bytes = 0;  // 1, 2, 4, 8 or 16
    uint32_t block = 1;          // texels per element side: 1, or 4 for block compression
    uint32_t samples = 1;
    uint32_t swizzle = kLinear;
};

inline uint32_t div_ceil(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
inline uint32_t round_up(uint32_t a, uint32_t b) { return div_ceil(a, b) * b; }
// Elements of one side of a mip as data and copies see it: texels halve rounding down, then round up to elements.
inline uint32_t dim(uint32_t texels, uint32_t mip, uint32_t block) {
    return div_ceil(std::max<uint32_t>(texels >> mip, 1), block);
}
// For placement: addrlib halves mip 0's elements rounding up.
inline uint32_t padded_dim(uint32_t texels, uint32_t mip, uint32_t block) {
    return div_ceil(div_ceil(std::max<uint32_t>(texels, 1), block), 1u << mip);
}

struct Mip {
    uint32_t width = 0, height = 0, depth = 1;    // elements
    uint32_t pitch = 0, rows = 0, depth_pitch = 1;  // padded, elements
    uint64_t offset = 0;                          // byte offset of its first block in the slice (0 in the tail)
    bool in_tail = false;
    uint32_t tail_offset = 0, tail_x = 0, tail_y = 0;  // in the tail: byte offset in the tail block, element position
};

constexpr uint32_t kMaxMips = 16;
constexpr uint32_t kMaxExtent = 1u << 14;  // GFX10 image descriptors hold 16384 texels per side

// 256-byte micro tile patterns by log2 element size: per address bit the element coordinate bit it takes, "--" for
// the bits inside an element. Thin S, thin D, thick S (also the 3D D modes).
inline const char* micro_pattern(int kind, uint32_t element_log2) {
    static const char* const kMicro[3][5] = {
        {"x0x1x2x3y0y1y2y3", "--x0x1x2y0y1y2x3", "----x0x1y0y1y2x2", "------x0y0y1x1x2", "--------y0y1x0x1"},
        {"x0x1x2y1y0y2x3y3", "--x0x1x2y0y1y2x3", "----x0x1y0y1x2y2", "------x0y0x1x2y1", "--------x0y0x1y1"},
        {"x0x1z0y0z1y1x2z2", "--x0z0y0z1y1x1z2", "----x0y0z0y1x1z1", "------x0z0y0x1z1", "--------z0y0x0z1"},
    };
    return kMicro[kind][element_log2];
}

constexpr uint32_t kBlock256_2d[5][2] = {{16, 16}, {16, 8}, {8, 8}, {8, 4}, {4, 4}};
constexpr uint32_t kBlock256_3d[5][3] = {{8, 4, 8}, {4, 4, 8}, {4, 4, 4}, {4, 2, 4}, {2, 2, 4}};
constexpr uint32_t kBlock1k_3d[5][3] = {{16, 8, 8}, {8, 8, 8}, {8, 8, 4}, {8, 4, 4}, {4, 4, 4}};

struct Layout {
    Desc desc;
    uint32_t block[3] = {1, 1, 1};  // block dimensions in elements (linear: the pitch alignment, 1, 1)
    bool thick = false;             // 3D blocks span block[2] depth slices
    uint32_t slices = 1;            // array slices, depth slabs (thick) or depth slices (thin and linear 3D)
    uint64_t slice_size = 0;        // one slice: its whole mip chain
    uint64_t size = 0;
    uint64_t alignment = 1;
    uint32_t first_tail_mip = 0;    // mips.size() without a tail
    std::vector<Mip> mips;
    std::array<std::vector<uint32_t>, 3> pattern;  // byte offset in a block of each coordinate value, per axis

    static std::optional<Layout> make(const Desc& d) {
        const int bl = block_log2(d.swizzle);
        if (bl < 0) return std::nullopt;
        auto pow2 = [](uint32_t v) { return v && !(v & (v - 1)); };
        const bool valid = d.width >= 1 && d.width <= kMaxExtent && d.height >= 1 && d.height <= kMaxExtent &&
                           d.depth >= 1 && d.depth <= kMaxExtent && d.mips >= 1 && d.mips <= kMaxMips &&
                           pow2(d.element_bytes) && d.element_bytes <= 16 && (d.block == 1 || d.block == 4) &&
                           pow2(d.samples) && d.samples <= 8 &&
                           (d.samples == 1 || (d.dimension == Dimension::tex2d && d.mips == 1)) &&
                           (d.dimension != Dimension::tex1d || d.height == 1) &&
                           (d.dimension != Dimension::tex3d || d.swizzle == 0 || d.swizzle == 5 || d.swizzle == 9 ||
                            d.swizzle == 17 || d.swizzle == 21 || (d.swizzle >= 24 && d.swizzle <= 27));
        if (!valid) return std::nullopt;
        return bl == 0 ? linear(d) : tiled(d, static_cast<uint32_t>(bl));
    }

    // HwlComputeSurfaceInfoLinear: rows aligned to 256 bytes, mips packed smallest first, each 3D depth slice holds
    // the chain.
    static Layout linear(const Desc& d) {
        Layout l;
        l.desc = d;
        const uint64_t bpe = d.element_bytes;
        const uint32_t align = 256 / d.element_bytes;
        const bool three_d = d.dimension == Dimension::tex3d;
        l.mips.resize(d.mips);
        uint64_t offset = 0;
        for (uint32_t i = d.mips; i-- > 0;) {
            Mip& m = l.mips[i];
            const uint32_t h = padded_dim(d.height, i, d.block);
            m.width = dim(d.width, i, d.block);
            m.height = dim(d.height, i, d.block);
            m.depth = three_d ? dim(d.depth, i, 1) : 1;
            m.pitch = round_up(padded_dim(d.width, i, d.block), align);
            m.rows = h;
            m.depth_pitch = three_d ? d.depth : 1;
            m.offset = offset;
            offset += uint64_t{m.pitch} * h * bpe;
        }
        l.block[0] = align;
        l.slices = d.depth;
        l.slice_size = offset;
        l.size = offset * d.depth;
        l.alignment = 256;
        l.first_tail_mip = d.mips;
        return l;
    }

    static std::array<std::vector<uint32_t>, 3> make_pattern(int kind, uint32_t element_log2, uint32_t bl) {
        std::array<std::vector<uint32_t>, 3> axes;
        const char* micro = micro_pattern(kind, element_log2);
        for (uint32_t bit = 0; bit < 8; ++bit) {
            const char a = micro[2 * bit], idx = micro[2 * bit + 1];
            const int axis = a == 'x' ? 0 : a == 'y' ? 1 : a == 'z' ? 2 : -1;
            if (axis < 0) continue;
            const size_t index = static_cast<size_t>(idx - '0');
            auto& to = axes[static_cast<size_t>(axis)];
            if (to.size() <= index) to.resize(index + 1, 0);
            to[index] = bit;
        }
        static const int kThin[2] = {1, 0}, kThick[3] = {1, 0, 2};
        const int* order = kind == 2 ? kThick : kThin;
        const size_t n = kind == 2 ? 3 : 2;
        for (uint32_t bit = 8, k = 0; bit < bl; ++bit, ++k) axes[static_cast<size_t>(order[k % n])].push_back(bit);
        std::array<std::vector<uint32_t>, 3> spread;
        for (size_t axis = 0; axis < 3; ++axis) {
            const auto& bits = axes[axis];
            spread[axis].resize(size_t{1} << bits.size());
            for (uint32_t c = 0; c < spread[axis].size(); ++c) {
                uint32_t v = 0;
                for (size_t i = 0; i < bits.size(); ++i) v |= ((c >> i) & 1u) << bits[i];
                spread[axis][c] = v;
            }
        }
        return spread;
    }

    static Layout tiled(const Desc& d, uint32_t bl) {
        Layout l;
        l.desc = d;
        const uint32_t element_log2 = static_cast<uint32_t>(ctz32(d.element_bytes));
        const uint64_t bpe = d.element_bytes;
        const uint64_t block_bytes = uint64_t{1} << bl;
        const bool thick = d.dimension == Dimension::tex3d && (standard(d.swizzle) || display(d.swizzle));
        uint32_t block[3];
        if (thick) {  // ComputeThickBlockDimension
            const uint32_t amp = (bl - 10) / 3, rest = (bl - 10) % 3;
            const uint32_t* b = kBlock1k_3d[element_log2];
            block[0] = b[0] << amp;
            block[1] = b[1] << (amp + rest / 2);
            block[2] = b[2] << (amp + (rest != 0 ? 1u : 0u));
        } else {  // ComputeThinBlockDimension
            const uint32_t samples_log2 = static_cast<uint32_t>(ctz32(d.samples));
            const uint32_t elements_log2 = bl - element_log2 - samples_log2;
            const bool wide = (samples_log2 & 1) == 0 || (bl & 1) != 0;
            const uint32_t w = (elements_log2 + (wide ? 1u : 0u)) / 2;
            block[0] = 1u << w;
            block[1] = 1u << (elements_log2 - w);
            block[2] = 1;
        }
        uint32_t slices = round_up(d.depth, block[2]);
        const bool three_d = d.dimension == Dimension::tex3d;
        auto mip_depth = [&](uint32_t i) { return thick ? padded_dim(d.depth, i, 1) : 1u; };
        auto depth_of = [&](uint32_t i) { return three_d ? dim(d.depth, i, 1) : 1u; };
        auto width = [&](uint32_t i) { return dim(d.width, i, d.block); };
        auto height = [&](uint32_t i) { return dim(d.height, i, d.block); };
        uint32_t first_tail_mip = d.mips;
        uint64_t slice_size = 0;
        if (d.mips == 1) {
            Mip m;
            m.width = width(0);
            m.height = height(0);
            m.depth = depth_of(0);
            m.pitch = round_up(width(0), block[0]);
            m.rows = round_up(height(0), block[1]);
            m.depth_pitch = three_d ? slices : 1;
            l.mips.push_back(m);
            slice_size = uint64_t{m.pitch} * m.rows * bpe * d.samples;
        } else {
            // GetMipTailDim: the largest mip the tail takes.
            uint32_t tail_max[3] = {block[0], block[1], block[2]};
            if (thick) {
                switch (bl % 3) {
                case 0: tail_max[1] >>= 1; break;
                case 1: tail_max[0] >>= 1; break;
                default: tail_max[2] >>= 1; break;
                }
            } else {
                tail_max[0] >>= 1;
            }
            // GetMaxNumMipsInTail; 256-byte blocks have no tail.
            const uint32_t effective = thick ? bl - (bl - 8) / 3 : bl;
            const uint32_t max_in_tail = effective == 8 ? 0 : effective <= 11 ? 1 + (1u << (effective - 9)) : effective - 4;
            // dsMipmapHtileFix: 8- and 16-bit depth enter the tail as 32-bit.
            uint32_t fixed[2] = {tail_max[0], tail_max[1]};
            if (z_order(d.swizzle) && element_log2 <= 1) {
                fixed[0] /= kBlock256_2d[element_log2][0] / kBlock256_2d[2][0];
                fixed[1] /= kBlock256_2d[element_log2][1] / kBlock256_2d[2][1];
            }
            std::vector<uint64_t> sizes;
            for (uint32_t i = 0; i < d.mips; ++i) {
                const uint32_t w = padded_dim(d.width, i, d.block), h = padded_dim(d.height, i, d.block);
                if (w <= fixed[0] && h <= fixed[1] && d.mips - i <= max_in_tail) {
                    first_tail_mip = i;
                    slice_size += block_bytes / block[2];
                    break;
                }
                Mip m;
                m.width = width(i);
                m.height = height(i);
                m.depth = depth_of(i);
                m.pitch = round_up(w, block[0]);
                m.rows = round_up(h, block[1]);
                const uint32_t depth = round_up(mip_depth(i), block[2]);
                const uint64_t one = uint64_t{m.pitch} * m.rows * bpe;
                slice_size += one;
                sizes.push_back(one * block[2]);
                m.depth_pitch = three_d ? depth : 1;
                l.mips.push_back(m);
            }
            // Standard mips from the smallest up, after the tail block.
            uint64_t offset = first_tail_mip < d.mips ? block_bytes : 0;
            for (uint32_t i = first_tail_mip; i-- > 0;) {
                l.mips[i].offset = offset;
                offset += sizes[i];
            }
            uint32_t micro[2];
            if (thick) {
                micro[0] = kBlock256_3d[element_log2][0];
                micro[1] = kBlock256_3d[element_log2][1];
            } else {
                micro[0] = kBlock256_2d[element_log2][0];
                micro[1] = kBlock256_2d[element_log2][1];
            }
            uint32_t pitch = tail_max[0], rows = tail_max[1];
            for (uint32_t i = first_tail_mip; i < d.mips; ++i) {
                const uint32_t m_index = max_in_tail - 1 - (i - first_tail_mip);
                const uint32_t toff = m_index > 6 ? 16u << m_index : m_index << 8;
                // Address bits 8.. of the offset, odd ones x, even ones y.
                auto bits = [&](uint32_t from) {
                    uint32_t v = 0;
                    for (uint32_t b = 0; b < 6; ++b) v |= ((toff >> (from + 2 * b)) & 1u) << b;
                    return v;
                };
                uint32_t x = bits(9), y = bits(8);
                if (bl & 1) {
                    std::swap(x, y);
                    if (element_log2 & 1) {
                        y = (y << 1) | (x & 1);
                        x >>= 1;
                    }
                }
                Mip m;
                m.width = width(i);
                m.height = height(i);
                m.depth = depth_of(i);
                m.pitch = pitch;
                m.rows = rows;
                m.depth_pitch = thick ? round_up(mip_depth(i), kBlock256_3d[element_log2][2]) : 1;
                m.in_tail = true;
                m.tail_offset = toff;
                m.tail_x = x * micro[0];
                m.tail_y = y * micro[1];
                l.mips.push_back(m);
                pitch = std::max(pitch >> 1, micro[0]);
                rows = std::max(rows >> 1, micro[1]);
            }
        }
        // Thick layouts store each depth slab's chain together.
        const uint64_t slab = thick ? block[2] : 1;
        slice_size *= slab;
        slices /= block[2];
        if (!xor_mode(d.swizzle) && d.samples == 1)
            l.pattern = make_pattern(thick ? 2 : (display(d.swizzle) ? 1 : 0), element_log2, bl);
        std::copy(block, block + 3, l.block);
        l.thick = thick;
        l.slices = slices;
        l.slice_size = slice_size;
        l.size = slice_size * slices;
        l.alignment = block_bytes;
        l.first_tail_mip = first_tail_mip;
        return l;
    }

    static int ctz32(uint32_t v) {
        int n = 0;
        while (v && !(v & 1)) v >>= 1, ++n;
        return n;
    }

    // Linear and the modes without pipe or bank XOR, single-sampled.
    bool addressable() const { return desc.swizzle == kLinear || !pattern[0].empty(); }

    void split(uint32_t z, uint32_t& slice, uint32_t& inner) const {
        if (thick) slice = z / block[2], inner = z % block[2];
        else slice = z, inner = 0;
    }

    uint32_t log2_alignment() const {
        uint32_t n = 0;
        while ((uint64_t{1} << n) < alignment) ++n;
        return n;
    }

    // The part of an address that depends on the row (and depth in a thick block).
    uint64_t row(const Mip& m, uint32_t y, uint32_t z) const {
        const uint32_t bw = block[0], bh = block[1];
        y += m.in_tail ? m.tail_y : 0;
        const uint64_t blocks = uint64_t{y / bh} * (m.pitch / bw);
        return (blocks << log2_alignment()) + (pattern[1][y % bh] | pattern[2][z]);
    }

    // The part of an address that depends on the column.
    uint64_t column(const Mip& m, uint32_t x) const {
        const uint32_t bw = block[0];
        x += m.in_tail ? m.tail_x : 0;
        return (uint64_t{x / bw} << log2_alignment()) + pattern[0][x % bw];
    }

    // Byte offset of element (x, y) of mip `mip` in array slice or depth slice z.
    Error offset(uint32_t mip, uint32_t x, uint32_t y, uint32_t z, uint64_t& out) const {
        if (mip >= mips.size()) return Error::invalid;
        const Mip& m = mips[mip];
        const uint32_t depth = desc.dimension == Dimension::tex3d ? m.depth : slices;
        const bool inside = x < m.width && y < m.height && z < depth;
        if (!inside) return Error::invalid;
        if (!addressable()) return Error::unsupported;
        uint32_t slice = 0, inner = 0;
        split(z, slice, inner);
        const uint64_t base = uint64_t{slice} * slice_size + m.offset;
        if (desc.swizzle == kLinear) {
            out = base + (uint64_t{y} * m.pitch + x) * desc.element_bytes;
            return Error::none;
        }
        out = base + row(m, y, inner) + column(m, x);
        return Error::none;
    }

    uint32_t subresource_depth(uint32_t mip) const { return mip < mips.size() ? mips[mip].depth : 0; }
};

// One contiguous run of a row: (tiled byte offset from the row's base, linear byte offset, length).
struct Run {
    uint64_t tiled, linear, length;
};

// The checks and the per-row runs of a copy of mip `mip` of array slice `slice` (3D: every depth slice, slice 0)
// between tiled memory of `tiled_len` bytes and linear rows `row_pitch` and `depth_pitch` apart.
inline Error plan_copy(const Layout& l, size_t tiled_len, uint32_t mip, uint32_t slice, size_t linear_len,
                       size_t row_pitch, size_t depth_pitch, uint32_t& first, std::vector<Run>& runs) {
    if (mip >= l.mips.size()) return Error::invalid;
    const Mip& m = l.mips[mip];
    if (l.desc.dimension == Dimension::tex3d) {
        if (slice != 0) return Error::invalid;
        first = 0;
    } else {
        first = slice;
    }
    if (first >= l.slices) return Error::invalid;
    if (!l.addressable()) return Error::unsupported;
    const size_t n = l.desc.element_bytes;
    const size_t bytes = size_t{m.width} * n;
    const unsigned long long depth_offset = (unsigned long long)(m.depth - 1) * depth_pitch;
    const unsigned long long row_offset = (unsigned long long)(m.height - 1) * row_pitch;
    if ((m.depth > 1 && depth_pitch && depth_offset / depth_pitch != m.depth - 1) ||
        (m.height > 1 && row_pitch && row_offset / row_pitch != m.height - 1))
        return Error::too_small;
    const unsigned long long need = depth_offset + row_offset + bytes;
    if (need < depth_offset || tiled_len < l.size || linear_len < need || row_pitch < bytes) return Error::too_small;
    runs.clear();
    if (l.desc.swizzle == kLinear) {
        runs.push_back({0, 0, bytes});
        return Error::none;
    }
    uint64_t first_linear = 0, first_tiled = l.column(m, 0);
    for (uint32_t x = 1; x < m.width; ++x) {
        const uint64_t off = l.column(m, x), at = uint64_t{x} * n;
        if (off != first_tiled + (at - first_linear)) {
            runs.push_back({first_tiled, first_linear, at - first_linear});
            first_tiled = off;
            first_linear = at;
        }
    }
    runs.push_back({first_tiled, first_linear, bytes - first_linear});
    return Error::none;
}

// The tiled byte offset of row y of depth slice z of a planned copy (linear: rows `pitch` elements apart).
inline uint64_t row_base(const Layout& l, const Mip& m, uint32_t first, uint32_t z, uint32_t y) {
    uint32_t s = 0, inner = 0;
    l.split(first + z, s, inner);
    const uint64_t base = uint64_t{s} * l.slice_size + m.offset;
    if (l.desc.swizzle == kLinear) return base + uint64_t{y} * m.pitch * l.desc.element_bytes;
    return base + l.row(m, y, inner);
}

// What a DXGI format stores per element: bytes, texels per side, and whether stencil is a second plane.
struct Format {
    uint32_t bytes = 0, block = 1;
    bool stencil = false;
};
inline bool format(uint32_t f, Format& out) {
    if (f >= 1 && f <= 4) out = {16, 1, false};
    else if (f >= 9 && f <= 18) out = {8, 1, false};
    else if ((f >= 19 && f <= 22) || (f >= 44 && f <= 47)) out = {4, 1, true};
    else if ((f >= 23 && f <= 43) || f == 67 || (f >= 87 && f <= 93) || f == 116 || f == 117) out = {4, 1, false};
    else if ((f >= 48 && f <= 59) || f == 85 || f == 86 || f == 115) out = {2, 1, false};
    else if (f >= 118 && f <= 120) out = {2, 1, true};
    else if (f >= 60 && f <= 65) out = {1, 1, false};
    else if ((f >= 70 && f <= 72) || (f >= 79 && f <= 81)) out = {8, 4, false};
    else if ((f >= 73 && f <= 78) || (f >= 82 && f <= 84) || (f >= 94 && f <= 99)) out = {16, 4, false};
    else return false;
    return true;
}

struct TextureDesc {
    Dimension dimension = Dimension::tex2d;
    uint32_t width = 0, height = 0, depth_or_array = 1;
    uint32_t mips = 1;  // 0: the full chain
    uint32_t format = 0;
    uint32_t samples = 1;
    uint32_t swizzle = kLinear;
};

struct Plane {
    uint64_t offset = 0;  // from the start of the texture
    bool stencil = false;
    Layout layout;
};

// Every data plane of a texture (depth, then stencil), one after the other.
struct Texture {
    std::vector<Plane> planes;
    uint64_t size = 0, alignment = 1;

    static std::optional<Texture> make(const TextureDesc& t) {
        Format f;
        if (!format(t.format, f)) return std::nullopt;
        const uint32_t depth3d = t.dimension == Dimension::tex3d ? t.depth_or_array : 1;
        uint32_t mips = t.mips;
        if (!mips) {
            const uint32_t largest = std::max({t.width, t.height, depth3d, 1u});
            while (mips < 32 && (largest >> mips)) ++mips;
        }
        auto plane = [&](uint32_t bytes) {
            Desc d;
            d.dimension = t.dimension;
            d.width = t.width;
            d.height = t.height;
            d.depth = t.depth_or_array;
            d.mips = mips;
            d.element_bytes = bytes;
            d.block = f.block;
            d.samples = std::max(t.samples, 1u);
            d.swizzle = t.swizzle;
            return Layout::make(d);
        };
        Texture out;
        auto first = plane(f.bytes);
        if (!first) return std::nullopt;
        out.planes.push_back(Plane{0, false, std::move(*first)});
        if (f.stencil) {
            auto stencil = plane(1);
            if (!stencil) return std::nullopt;
            const uint64_t a = stencil->alignment;
            const uint64_t offset = (out.planes[0].layout.size + a - 1) / a * a;
            out.planes.push_back(Plane{offset, true, std::move(*stencil)});
        }
        const Plane& last = out.planes.back();
        out.size = last.offset + last.layout.size;
        for (const Plane& p : out.planes) out.alignment = std::max(out.alignment, p.layout.alignment);
        return out;
    }
};

}  // namespace leonida::console_layout
