"""SPIRV-SIZE-1 and SPIRV-SIZE-2: smaller native compute modules with the same values.

SPIRV-SIZE-1: a compute shader's lane index, wave lane and "lane >= 32" are computed once in %entry (%h_li, %h_wl,
%h_up) instead of from LocalInvocationId at every lane-mask read. A wave64 program reads a 64-bit constant mask
through an SGPR pair (bit L for lane L, the high SGPR for lanes 32..63); the hoisted module loads LocalInvocationId
once and every lane gets the same word as with the per-use form.

SPIRV-SIZE-2: the subgroup vote that gives SCC after a lane-mask instruction is left out when that SCC is overwritten
before anything reads it, in native modules too (it was --wave64-emulate only). Three mask instructions write SCC;
only the last one's SCC is read, by an s_cbranch_scc0 in the next block, so one vote stays. The program runs on 64
lanes in the subgroup interpreter with the analysis and without it: same stored words, for a vote that is true
(lane 50 exists) and one that is false (lane 99 does not).
"""
import os
import struct
import tempfile
import unittest
from unittest import mock

import support  # noqa: F401  (puts tools/ on sys.path)
import rdna2spirv
import spirv_mini_interp as si
import wave64_emulate_opt

VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]


def translate(name, insns, local_size=(64, 1, 1)):
    lines = [f"; {name}  synthetic", "; mattr: +wavefrontsize64,-wavefrontsize32"]
    for txt, nb in insns:
        lines.append(f"\t{txt:52s} ; encoding: [{','.join(['0x00'] * nb)}]")
    with tempfile.TemporaryDirectory() as td:
        os.mkdir(os.path.join(td, "disasm"))
        os.mkdir(os.path.join(td, "fssm"))
        with open(os.path.join(td, "disasm", name + ".s"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        with open(os.path.join(td, "fssm", name + ".fssm"), "wb") as f:
            f.write(bytes(0x138) + struct.pack("<4I", 0x8088, *local_size))
        return rdna2spirv.translate(os.path.join(td, "disasm", name + ".s"))


def run64(text):
    memory = {}
    si.run_workgroup(text, 64, 64, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0)},
                     push_constants=VSHARP + [0] * 28, memory=memory)
    return [memory.get(4 * lane) for lane in range(64)]


MASK = 0xF00000000000FFFF
LANE_PROGRAM = [
    ("s_mov_b64 exec, -1", 4),                     # a wave64 program (64-bit EXEC)
    ("s_mov_b32 s4, 0xffff", 8),
    ("s_mov_b32 s5, 0xf0000000", 8),
    ("v_cndmask_b32_e64 v5, 0, 1, s[4:5]", 8),
    ("v_lshlrev_b32_e32 v6, 2, v0", 4),
    ("buffer_store_dword v5, v6, s[0:3], 0 offen", 8),
    ("s_endpgm", 4),
]


def main_function(text):
    return text[text.index("%main = OpFunction"):]


class LaneHoist(unittest.TestCase):
    def test_hoisted_lane_mask_read(self):
        hoisted = translate("cs_lane_hoist", LANE_PROGRAM)
        with mock.patch.object(rdna2spirv.Gen, "lane_hoisted", lambda self: False):
            inline = translate("cs_lane_inline", LANE_PROGRAM)
        self.assertIn("%h_up = OpUGreaterThanEqual %bool %h_wl", hoisted)
        # After %entry's hoisted ids, nothing loads LocalInvocationId again (the prologue's load for v0 is before).
        after = main_function(hoisted).split("%h_up = ", 1)[1]
        self.assertNotIn("OpLoad %v3uint %local_id", after)
        self.assertGreater(main_function(inline).count("OpLoad %v3uint %local_id"), 1)
        self.assertNotIn("%h_", inline)
        want = [(MASK >> lane) & 1 for lane in range(64)]
        self.assertEqual(run64(inline), want)
        self.assertEqual(run64(hoisted), want)

    def test_not_in_a_helper_function(self):
        # A helper function's body (Gen.helper_depth) never gets %main's hoisted ids.
        g = rdna2spirv.Gen("cs")
        g.cs = {"size": (64, 1, 1)}
        g.helper_depth = 1
        self.assertFalse(g.lane_hoisted())


def scc_program(target_lane):
    return [
        ("s_mov_b32 s14, 0", 4),
        ("v_cmp_eq_u32_e64 s[6:7], 45, v0", 8),
        ("s_and_saveexec_b64 s[8:9], s[6:7]", 4),      # SCC overwritten by the next instruction: no vote
        ("s_or_b64 exec, exec, s[8:9]", 4),            # overwritten again: no vote
        (f"v_cmp_eq_u32_e64 s[10:11], {target_lane}, v0", 8),
        ("s_and_saveexec_b64 s[12:13], s[10:11]", 4),  # read by the branch in the next block: vote
        ("s_cbranch_scc0 1", 4),
        ("s_mov_b32 s14, 77", 4),
        ("s_mov_b64 exec, -1", 4),
        ("v_mov_b32_e32 v5, s14", 4),
        ("v_lshlrev_b32_e32 v6, 2, v0", 4),
        ("buffer_store_dword v5, v6, s[0:3], 0 offen", 8),
        ("s_endpgm", 4),
    ]


class NativeSccLiveness(unittest.TestCase):
    def test_dead_votes_dropped_live_vote_kept(self):
        for target, want in ((50, 77), (99, 0)):
            with self.subTest(target=target):
                live = translate("cs_scc_live", scc_program(target))
                with mock.patch.object(wave64_emulate_opt, "scc_live_outs", lambda blocks, succ: None):
                    every = translate("cs_scc_every", scc_program(target))
                self.assertEqual(live.count("OpGroupNonUniformAny"), 1)
                self.assertEqual(every.count("OpGroupNonUniformAny"), 3)
                self.assertEqual(run64(every), [want] * 64)
                self.assertEqual(run64(live), [want] * 64)


if __name__ == "__main__":
    unittest.main()
