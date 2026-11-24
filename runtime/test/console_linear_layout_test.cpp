// The console linear layout's pitch and size rule (src/console_linear_layout.h) that xg_xs.dll's texture computer
// answers. No game data: the sizes are the ones the title's write-back computes.
#include "../src/console_linear_layout.h"

#include <cstdio>

#define CHECK(condition) do {                                        \
    if (!(condition)) {                                              \
        std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
        return 1;                                                    \
    }                                                                \
} while (0)

using namespace leonida;

int main() {
    // Rows round up to 256 bytes; a multiple of 256 stays as it is.
    CHECK(console_linear_pitch(0) == 0);
    CHECK(console_linear_pitch(1) == 256);
    CHECK(console_linear_pitch(16) == 256);
    CHECK(console_linear_pitch(256) == 256);
    CHECK(console_linear_pitch(257) == 512);
    // A 1920x1080 one-byte plane: align256(1920) * 1080, what game_scarlett_master.exe+0x532F140 writes back.
    CHECK(console_linear_pitch(1920) == 2048);
    CHECK(console_linear_size(1920, 1080, 1) == 0x21C000);
    // Its half-size chroma plane: 960 -> 1024-byte rows.
    CHECK(console_linear_size(960, 540, 1) == 1024 * 540);
    // A BC1 texture of 100x60 texels: 25 blocks of 8 bytes per row (200 -> 256), 15 rows of blocks.
    CHECK(console_linear_size(25 * 8, 15, 1) == 256 * 15);
    // Depth slices multiply whole slices.
    CHECK(console_linear_size(64, 4, 3) == 256 * 4 * 3);
    std::printf("console_linear_layout: ok\n");
    return 0;
}
