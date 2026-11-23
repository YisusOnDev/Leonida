// Native BCPack decoder (runtime/src/bcpack.cpp) checks.
//
//   bcpack_test                        built-in checks that need nothing else
//   bcpack_test --batch <in> <out>     decode every chunk of a batch file for tests/test_bcpack_native.py
//
// Batch input, repeated to the end of the file (little-endian):
//   u32 codec, u32 out_size, u8 allow_provisional, u32 raw_size, raw_size bytes
// Batch output, one record per input chunk:
//   u8 status, u32 message size, message, u32 label count, (u32 size, label) per label, u32 out size, out bytes
// (out bytes only when the status is ok).

#include "../src/bcpack.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace bcpack = leonida::bcpack;

#define CHECK(condition)                                                \
    do {                                                                \
        if (!(condition)) {                                             \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
            return 1;                                                   \
        }                                                               \
    } while (0)

namespace {

int self_test() {
    std::vector<std::uint8_t> out;

    // An empty chunk decodes to nothing for every codec; an unknown codec has no rebuilder.
    for (int codec : {1, 3, 4, 5, 6, 7}) CHECK(bcpack::decode({}, codec, out));
    CHECK(bcpack::decode({}, 2, out).status == bcpack::Status::unsupported);
    CHECK(bcpack::block_bytes(1) == 8 && bcpack::block_bytes(7) == 16 && bcpack::block_bytes(2) == 0);

    // A size that is not a whole number of blocks, and input that runs out, are malformed.
    out.assign(12, 0);
    auto r = bcpack::decode({}, 1, out);
    CHECK(r.status == bcpack::Status::error && r.message.find("whole number of BC1 blocks") != std::string::npos);
    out.assign(8, 0);
    r = bcpack::decode({}, 1, out);
    CHECK(r.status == bcpack::Status::error && r.message.find("input ran out") != std::string::npos);

    // Leftover non-zero bytes after an empty chunk are refused; zero padding is not.
    const std::uint8_t pad[3] = {0, 0, 0}, junk[3] = {0, 1, 0};
    out.clear();
    CHECK(bcpack::decode(pad, 4, out));
    CHECK(bcpack::decode(junk, 4, out).status == bcpack::Status::error);
    return 0;
}

std::uint32_t read_u32(const std::vector<std::uint8_t>& in, std::size_t& at) {
    std::uint32_t v = 0;
    for (int k = 0; k < 4; ++k) v |= static_cast<std::uint32_t>(in.at(at + k)) << (8 * k);
    at += 4;
    return v;
}

void write_u32(std::string& out, std::uint32_t v) {
    for (int k = 0; k < 4; ++k) out.push_back(static_cast<char>((v >> (8 * k)) & 0xFF));
}

void write_blob(std::string& out, const void* data, std::size_t size) {
    write_u32(out, static_cast<std::uint32_t>(size));
    out.append(static_cast<const char*>(data), size);
}

int batch(const char* in_path, const char* out_path) {
    std::ifstream in_file(in_path, std::ios::binary);
    if (!in_file) {
        std::fprintf(stderr, "cannot read %s\n", in_path);
        return 2;
    }
    const std::vector<std::uint8_t> in((std::istreambuf_iterator<char>(in_file)), std::istreambuf_iterator<char>());
    std::string out;
    std::size_t at = 0;
    while (at < in.size()) {
        const int codec = static_cast<int>(read_u32(in, at));
        const std::uint32_t out_size = read_u32(in, at);
        const bool allow_provisional = in.at(at++) != 0;
        const std::uint32_t raw_size = read_u32(in, at);
        if (at + raw_size > in.size()) {
            std::fprintf(stderr, "truncated batch file\n");
            return 2;
        }
        const std::span<const std::uint8_t> raw(in.data() + at, raw_size);
        at += raw_size;
        std::vector<std::uint8_t> decoded(out_size);
        const auto result = bcpack::decode(raw, codec, decoded, allow_provisional);
        out.push_back(static_cast<char>(result.status));
        write_blob(out, result.message.data(), result.message.size());
        write_u32(out, static_cast<std::uint32_t>(result.provisional.size()));
        for (const auto& label : result.provisional) write_blob(out, label.data(), label.size());
        if (result)
            write_blob(out, decoded.data(), decoded.size());
        else
            write_u32(out, 0);
    }
    std::ofstream out_file(out_path, std::ios::binary);
    out_file.write(out.data(), static_cast<std::streamsize>(out.size()));
    return out_file ? 0 : 2;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 4 && std::strcmp(argv[1], "--batch") == 0) return batch(argv[2], argv[3]);
    if (argc != 1) {
        std::fprintf(stderr, "usage: bcpack_test [--batch <in> <out>]\n");
        return 2;
    }
    const int failed = self_test();
    if (!failed) std::puts("bcpack_test: ok");
    return failed;
}
