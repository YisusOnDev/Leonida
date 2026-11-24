"""Compare the native BCPack decoder (runtime/src/bcpack.cpp) with tools/bcpack.py on synthetic chunks.

Random lane symbols for every codec are encoded with the test encoder of tests/test_bcpack.py (single and
multi-fragment lanes, narrowed widths, every q), some of them corrupted, and every chunk is decoded by both
decoders in strict and provisional mode. The status, message, provisional labels and decoded bytes must match.
No game files are used. Also checks that runtime/src/bcpack_tables.inc is current (tools/gen_bcpack_tables.py).

    python -I tests/test_bcpack_native.py [--bin <bcpack_test>] [--seed N] [--chunks N]

Without --bin it looks for bcpack_test under runtime/build and skips the comparison when there is none
(ctest passes --bin).
"""
import glob
import os
import random
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import bcpack  # noqa: E402
import bcpack_tables_bptc as bptc  # noqa: E402
import gen_bcpack_tables  # noqa: E402
import test_bcpack as enc  # noqa: E402

OPTIONS = {"bin": None, "seed": 186, "chunks": 24}

STATUS_OK, STATUS_ERROR, STATUS_PROVISIONAL, STATUS_UNSUPPORTED = 0, 1, 2, 3


def find_binary():
    if OPTIONS["bin"]:
        return OPTIONS["bin"]
    env = os.environ.get("LEONIDA_BCPACK_TEST")
    if env:
        return env
    for name in ("bcpack_test", "bcpack_test.exe"):
        hits = sorted(glob.glob(os.path.join(ROOT, "runtime", "build", "**", name), recursive=True))
        if hits:
            return hits[0]
    return None


# --- Synthetic chunks ------------------------------------------------------------------------------------------------

def _encodable(model, syms):
    """Whether the test encoder can encode every symbol with `model` (a non-zero band frequency after the c == 0
    adjustment)."""
    for s in set(syms):
        p = s.bit_length()
        if p > model.w:
            return False
        f = model.freq[p]
        c = model.base[p] + f * (s - (0 if p == 0 else 1 << (p - 1)))
        if f == 0 or (c == 0 and f <= 1):
            return False
    return True


def _pick_q(rng, w, syms):
    qs = [q for q in range(16) if _encodable(bcpack._get_model(w, q), syms)]
    return rng.choice(qs) if qs else None


def _lane_stream(rng, syms, dw):
    """One lane's byte stream: one flush fragment, or several with narrowed widths and non-flush boundaries."""
    if len(syms) >= 4 and rng.random() < 0.5:
        # The test encoder handles only some fragment chains, so keep a chain only when it decodes back on its own.
        try:
            stream = _multi_fragment_stream(rng, syms, dw)
            lane = enc.interleave_lanes([stream], (dw,), len(syms))
            if bcpack._decode_lanes_raw(memoryview(lane), (dw,), len(syms)) == ([syms], len(lane)):
                return stream
        except (AssertionError, IndexError, bcpack.BcpackError):
            pass
    q = _pick_q(rng, dw, syms)
    assert q is not None, f"no q encodes width {dw}"
    return enc._encode_lane_stream([(syms, dw, q, True)], dw)


def _multi_fragment_stream(rng, syms, dw):
    cuts = sorted(rng.sample(range(1, len(syms)), min(len(syms) - 1, rng.randint(1, 3))))
    pieces = [syms[a:b] for a, b in zip([0] + cuts, cuts + [len(syms)])]
    fragments = []
    for k, piece in enumerate(pieces):
        top = max(piece).bit_length()
        w = rng.choice([w for w in range(max(1, top, dw - 3), dw + 1)])
        q = _pick_q(rng, w, piece)
        if q is None:
            w = dw
            q = _pick_q(rng, w, piece)
        flush = k == len(pieces) - 1 or rng.random() < 0.5
        fragments.append((piece, w, q, flush))
    try:
        return enc._encode_lane_stream(fragments, dw)
    except ValueError:
        # test_bcpack.py's encoder refuses a fragment whose payload comes out empty; one fragment then.
        q = _pick_q(rng, dw, syms)
        return enc._encode_lane_stream([(syms, dw, q, True)], dw)


def encode(rng, lanes, codec):
    widths = bcpack.LANE_WIDTHS[codec]
    streams = [_lane_stream(rng, syms, dw) for syms, dw in zip(lanes, widths)]
    return enc.interleave_lanes(streams, widths, len(lanes[0]))


def _rand_lane(rng, n, width):
    style = rng.random()
    if style < 0.25:
        return [0] * n
    if style < 0.5:  # small deltas around zero (zigzag values)
        return [min((1 << width) - 1, rng.choice((0, 0, 1, 2, 3, 4))) for _ in range(n)]
    return [rng.randrange(1 << width) for _ in range(n)]


def _distance_symbols(rng, i, far, clean=False):
    """A back-reference distance for block i: mostly inside the chunk, sometimes into the built-in history."""
    if far or (not clean and rng.random() < 0.15):
        return rng.randrange(256)
    return 0 if i == 0 or rng.random() < 0.3 else rng.randint(1, min(i, 255))


VALID_BC7_CELLS = None


def _bc7_cells():
    global VALID_BC7_CELLS
    if VALID_BC7_CELLS is None:
        table = bptc.load_bc7(None)
        cfg, rank_of = table["config"], table["rank_of"]
        VALID_BC7_CELLS = [
            (a, b) for a in range(32) for b in range(16)
            if rank_of[a * 16 + b] >= 0 and cfg[rank_of[a * 16 + b]][0] <= 7 and cfg[rank_of[a * 16 + b]][1] != 255
        ]
    return VALID_BC7_CELLS


def make_lanes(rng, codec, nblocks):
    widths = bcpack.LANE_WIDTHS[codec]
    per_block = 1 if codec in (1, 4) else 2
    n = nblocks * per_block
    lanes = [_rand_lane(rng, n, w) for w in widths]
    # A clean chunk stays inside itself and keeps BC6H/BC7 anchor indices and BC7 p-bits consistent, so it
    # decodes; the others mostly exercise refusals.
    clean = rng.random() < 0.5
    far = not clean and rng.random() < 0.2
    if codec in (1, 3, 4, 5) and (clean or rng.random() < 0.9):
        lanes[1] = [0] * n
    if clean and codec in (6, 7):
        for lane in (2, 6, 7, 8, 10, 11, 12, 13, 14) if codec == 7 else (11, 12, 13, 14):
            lanes[lane] = [0] * n
    if codec in (1, 4):
        for i in range(nblocks):
            d = _distance_symbols(rng, i, far, clean)
            lanes[0][i] = enc._zig(d if d < 128 else d - 256)
    elif codec in (3, 5):
        for i in range(nblocks):
            d = _distance_symbols(rng, i, far, clean)
            lanes[0][2 * i], lanes[0][2 * i + 1] = enc._deint4(enc._zig(d if d < 128 else d - 256) & 255)
    else:
        for i in range(nblocks):
            di = _distance_symbols(rng, i, far, clean)
            de = _distance_symbols(rng, i, far, clean)
            delta = (de - di) & 255
            delta = delta - 256 if delta >= 128 else delta
            lanes[1][2 * i], lanes[1][2 * i + 1] = enc._deint4(di)
            lanes[0][2 * i], lanes[0][2 * i + 1] = enc._deint4(enc._zig(delta) & 255)
        if codec == 6:
            for i in range(nblocks):
                lanes[15][2 * i] = rng.randrange(14) if rng.random() < 0.97 else rng.randrange(32)
                lanes[15][2 * i + 1] = rng.randrange(32)
        else:
            cells = _bc7_cells()
            for i in range(nblocks):
                if rng.random() < 0.97:
                    lanes[15][2 * i], lanes[15][2 * i + 1] = rng.choice(cells)
                else:
                    lanes[15][2 * i], lanes[15][2 * i + 1] = rng.randrange(32), rng.randrange(32)
    return lanes


def corrupt(rng, raw):
    kind = rng.randrange(5)
    data = bytearray(raw)
    if kind == 0 and data:
        data[rng.randrange(len(data))] ^= 1 << rng.randrange(8)
    elif kind == 1 and data:
        del data[rng.randrange(len(data)):]
    elif kind == 2:
        data += bytes([rng.randint(1, 255)])
    elif kind == 3:
        data += bytes(rng.randint(1, 9))
    else:
        data = bytearray(rng.randrange(256) for _ in range(rng.randint(1, 64)))
    return bytes(data)


def make_cases(seed, chunks):
    """(codec, out_size, raw) chunks: valid ones, corrupted ones, wrong sizes and unsupported codecs."""
    rng = random.Random(seed)
    cases = []
    for codec in (1, 3, 4, 5, 6, 7):
        bb = bcpack.BLOCK_BYTES[codec]
        for k in range(chunks):
            nblocks = rng.choice((1, 2, 3, 5, 16, 33, 64, 130, 257, 300))
            raw = encode(rng, make_lanes(rng, codec, nblocks), codec)
            cases.append((codec, nblocks * bb, raw))
            if k % 2 == 0:
                cases.append((codec, nblocks * bb, corrupt(rng, raw)))
            if k % 6 == 0:
                cases.append((codec, nblocks * bb + rng.choice((-bb, bb, bb // 2)), raw))
    cases += [(2, 16, b""), (0, 8, b"\x00"), (8, 16, b"")]
    return cases


# --- Both decoders ----------------------------------------------------------------------------------------------------

def python_decode(codec, out_size, raw, allow_provisional):
    try:
        data, prov = bcpack.decode_with_info(raw, codec, out_size, allow_provisional=allow_provisional)
        return STATUS_OK, "", sorted(prov), data
    except bcpack.BcpackUnsupportedCodec as e:
        return STATUS_UNSUPPORTED, str(e), [], b""
    except bcpack.BcpackProvisionalError as e:
        return STATUS_PROVISIONAL, str(e), [], b""
    except bcpack.BcpackError as e:
        return STATUS_ERROR, str(e), [], b""


def native_decode_all(binary, records):
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = os.path.join(tmp, "in.bin"), os.path.join(tmp, "out.bin")
        with open(src, "wb") as f:
            for codec, out_size, raw, allow in records:
                f.write(struct.pack("<IIBI", codec, max(out_size, 0), 1 if allow else 0, len(raw)) + raw)
        # Relative names from inside the folder: the helper's arguments then stay ASCII whatever the temp path is
        # (a Windows user name outside the ANSI code page would otherwise reach a narrow argv mangled).
        subprocess.run([os.path.abspath(binary), "--batch", "in.bin", "out.bin"], cwd=tmp, check=True, timeout=600)
        with open(dst, "rb") as f:
            blob = f.read()
    results, at = [], 0

    def take():
        nonlocal at
        (n,) = struct.unpack_from("<I", blob, at)
        at += 4
        value = blob[at:at + n]
        at += n
        return value

    for _ in records:
        status = blob[at]
        at += 1
        message = take().decode("utf-8")
        (count,) = struct.unpack_from("<I", blob, at)
        at += 4
        labels = [take().decode("utf-8") for _ in range(count)]
        results.append((status, message, labels, take()))
    assert at == len(blob), "trailing bytes in the native batch output"
    return results


def _same_message(codec, py_msg, native_msg):
    # Python reports the first bad BC7 signature in set order, the native decoder in block order.
    if codec == 7 and py_msg.startswith("BC7 signature (") and native_msg.startswith("BC7 signature ("):
        return py_msg.endswith("is outside the signature table") and native_msg.endswith("is outside the signature table")
    return py_msg == native_msg


class NativeTablesTest(unittest.TestCase):
    def test_generated_tables_are_current(self):
        self.assertEqual(gen_bcpack_tables.main(["--check"]), 0,
                         "runtime/src/bcpack_tables.inc is stale: run python -I tools/gen_bcpack_tables.py")


class NativeDecoderTest(unittest.TestCase):
    def test_native_matches_python(self):
        binary = find_binary()
        if not binary or not os.path.isfile(binary):
            self.skipTest("bcpack_test not built (pass --bin or set LEONIDA_BCPACK_TEST)")
        cases = make_cases(OPTIONS["seed"], OPTIONS["chunks"])
        records = [(c, s, r, allow) for c, s, r in cases for allow in (False, True)]
        native = native_decode_all(binary, records)
        tally = {}
        mismatches = []
        for (codec, out_size, raw, allow), got in zip(records, native):
            want = python_decode(codec, out_size, raw, allow)
            key = (codec, want[0])
            tally[key] = tally.get(key, 0) + 1
            same = (want[0] == got[0] and _same_message(codec, want[1], got[1]) and want[2] == got[2]
                    and want[3] == got[3])
            if not same:
                mismatches.append((codec, out_size, allow, len(raw), want[:3], got[:3], want[3] == got[3]))
        summary = ", ".join(
            f"{bcpack.CODEC_NAMES.get(c, c)} {['ok', 'error', 'provisional', 'unsupported'][s]} {n}"
            for (c, s), n in sorted(tally.items())
        )
        print(f"\n{len(records)} decodes (seed {OPTIONS['seed']}): {summary}", file=sys.stderr)
        self.assertFalse(mismatches, f"{len(mismatches)} mismatches:\n" + "\n".join(map(repr, mismatches[:10])))
        for codec in (1, 3, 4, 5, 6, 7):
            self.assertTrue(tally.get((codec, STATUS_OK)), f"no {bcpack.CODEC_NAMES[codec]} chunk decoded; widen the generator")


if __name__ == "__main__":
    args = sys.argv[1:]
    rest = [sys.argv[0]]
    while args:
        a = args.pop(0)
        if a in ("--bin", "--seed", "--chunks") and args:
            v = args.pop(0)
            OPTIONS[a[2:]] = v if a == "--bin" else int(v)
        else:
            rest.append(a)
    unittest.main(argv=rest)
