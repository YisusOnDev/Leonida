"""SPIRV-SIZE-3: repeated instruction bodies shared as SPIR-V functions (tools/spirv_outline.py).

A hand-written compute program with three typed buffer loads and three legacy multiplies on different registers,
under an EXEC that leaves two of four lanes off, is translated with the outliner and without it. Both modules are
executed by spirv_mini_interp on the same memory: every word they store must be the same, the outlined module must
call shared functions instead of repeating the bodies, and it must be smaller. The words themselves are checked
against a plain-Python legacy multiply (0 * anything = 0, denormals flushed) so the comparison is not between two
wrong answers. The other cases feed the outliner small hand-made spans and check what it refuses: a store under a
condition of the span, a branch out of it, a register pointer used other than by OpLoad / OpStore.
"""
import struct
import unittest
from unittest import mock

import support
import rdna2spirv
import spirv_outline
import spirv_mini_interp as si


def _fssm(rsrc2, group):
    header = bytearray(0x148)
    struct.pack_into("<4I", header, 0x138, rsrc2, *group)
    return bytes(header)


CS_FSSM = _fssm(0x8088, (64, 1, 1))     # 4 user SGPRs (the V#), 64x1x1 threads, as cs_bounds
PROGRAM = [
    "v_cmpx_gt_u32_e32 2, v0",                  # lanes 0 and 1 on, 2 and 3 off
    "v_lshlrev_b32_e32 v1, 2, v0",
    "v_add_nc_u32_e32 v2, 16, v1",
    "v_add_nc_u32_e32 v3, 32, v1",
    "buffer_load_format_x v4, v1, s[0:3], 0 offen",
    "buffer_load_format_x v5, v2, s[0:3], 0 offen",
    "buffer_load_format_x v6, v3, s[0:3], 0 offen",
    "s_waitcnt vmcnt(0)",
    "v_mul_legacy_f32_e32 v7, v4, v5",
    "v_mul_legacy_f32_e32 v8, v5, v6",
    "v_mul_legacy_f32_e32 v9, v6, v4",
    "v_lshlrev_b32_e32 v10, 4, v0",
    "v_add_nc_u32_e32 v10, 64, v10",
    "buffer_store_dwordx4 v[6:9], v10, s[0:3], 0 offen",
    "s_endpgm",
]
# 32_FLOAT typed reads (format 22 in dword 3 bits 12..18 of the V#), raw dword stores; base 0, 256 bytes.
VSHARP = [0, 0, 0x100, 0x31016FAC]
F = {"zero": 0x00000000, "two": 0x40000000, "inf": 0x7F800000, "denorm": 0x00000003, "half": 0x3F000000,
     "neg3": 0xC0400000, "nan": 0x7FC00000, "big": 0x7F000000}
LOADS = {0: F["zero"], 4: F["two"], 8: F["half"], 12: F["two"],                 # v4 of lanes 0..3
         16: F["inf"], 20: F["neg3"], 24: F["two"], 28: F["two"],               # v5
         32: F["denorm"], 36: F["big"], 40: F["two"], 44: F["two"]}             # v6


def _f(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _bits(x):
    try:
        return struct.unpack("<I", struct.pack("<f", x))[0]
    except OverflowError:
        return 0x7F800000 if x > 0 else 0xFF800000


def legacy_mul(a, b):
    """v_mul_legacy_f32: +-0 (or a denormal, flushed) times anything, inf and NaN included, is +0; a denormal result
    is flushed keeping its sign."""
    fa, fb = _f(a), _f(b)
    if a & 0x7F800000 == 0 or b & 0x7F800000 == 0:
        return 0
    r = _bits(fa * fb)
    return r & 0x80000000 if r & 0x7F800000 == 0 else r


def expected():
    out = {}
    for lane in (0, 1):
        v4, v5, v6 = LOADS[4 * lane], LOADS[16 + 4 * lane], LOADS[32 + 4 * lane]
        base = 64 + 16 * lane
        # v_mul_legacy reads its sources as f32 with denormals flushed: v6 of lane 0 is a denormal, stored as is.
        out.update({base: v6, base + 4: legacy_mul(v4, v5), base + 8: legacy_mul(v5, v6),
                    base + 12: legacy_mul(v6, v4)})
    return out


def translate(outline):
    with mock.patch.object(rdna2spirv, "OUTLINE", outline):
        return support.translate_lines("cs_outline", PROGRAM, fssm=CS_FSSM)


def run(text):
    memory = dict(LOADS)
    si.run_workgroup(text, 4, 64, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0)},
                     push_constants=VSHARP + [0] * 28, memory=memory)
    return {a: w for a, w in memory.items() if a >= 64}


class OutlinedProgram(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inline = translate(False)
        cls.outlined = translate(True)

    def test_functions_are_called(self):
        self.assertNotIn("%of0 = OpFunction", self.inline)
        self.assertIn("%of0 = OpFunction", self.outlined)
        calls = self.outlined.count("OpFunctionCall %uint %of") + self.outlined.count("OpFunctionCall %of")
        self.assertGreaterEqual(self.outlined.count(" %of0 ") + self.outlined.count(" %of1 "), 4, calls)
        self.assertLess(len(self.outlined), len(self.inline))

    def test_same_words_as_inline(self):
        want = expected()
        self.assertEqual(run(self.inline), want)
        self.assertEqual(run(self.outlined), want)

    def test_off_lanes_store_nothing(self):
        got = run(self.outlined)
        self.assertFalse([a for a in got if a >= 96])


def _outliner(main_lines, types=None):
    types = types or ["%uint = OpTypeInt 32 0", "%bool = OpTypeBool",
                      "%_ptr_Function_uint = OpTypePointer Function %uint",
                      "%_ptr_Function_arr = OpTypePointer Function %arr"]
    return spirv_outline.Outliner(main_lines, types)


PROLOGUE = ["%entry = OpLabel", "%vgpr = OpVariable %_ptr_Function_arr Function",
            "%reg_v1 = OpAccessChain %_ptr_Function_uint %vgpr %c1",
            "%reg_v2 = OpAccessChain %_ptr_Function_uint %vgpr %c2"]


def _span(extra=()):
    """A 40-word-plus straight span: v2 = ((v1 + 1) * 3 ...) with a chain of adds."""
    lines = ["%a0 = OpLoad %uint %reg_v1"]
    for i in range(1, 12):
        lines.append(f"%a{i} = OpIAdd %uint %a{i - 1} %c1")
    lines += list(extra) + ["OpStore %reg_v2 %a11"]
    return lines


class Refusals(unittest.TestCase):
    def test_plain_span_is_taken(self):
        span = _span()
        o = _outliner(PROLOGUE + span)
        info = o.analyze(span, "%entry")
        self.assertIsNotNone(info)
        self.assertEqual([t for _, t in info["params"]], ["%uint"])          # v1, loaded by the caller
        self.assertEqual(info["results"][0][2], ("store", "%reg_v2"))

    def test_conditional_store_refused(self):
        span = (["%c = OpIEqual %bool %c1 %c1", "OpSelectionMerge %m None", "OpBranchConditional %c %d %m",
                 "%d = OpLabel"] + _span() + ["OpBranch %m", "%m = OpLabel", "%z = OpIAdd %uint %c1 %c1"])
        o = _outliner(PROLOGUE + span)
        self.assertIsNone(o.analyze(span, "%entry"))

    def test_branch_out_refused(self):
        span = _span() + ["OpSelectionMerge %m None", "OpBranchConditional %true %far %m", "%m = OpLabel",
                          "%z = OpIAdd %uint %c1 %c1"]
        o = _outliner(PROLOGUE + span + ["%far = OpLabel"])
        self.assertIsNone(o.analyze(span, "%entry"))

    def test_register_pointer_other_use_refused(self):
        span = _span(["%p = OpAccessChain %_ptr_Function_uint %vgpr %a3"])
        o = _outliner(PROLOGUE + span)
        self.assertIsNone(o.analyze(span, "%entry"))

    def test_value_read_after_span_is_a_result(self):
        span = _span()
        o = _outliner(PROLOGUE + span + ["%later = OpIAdd %uint %a5 %c1"])
        info = o.analyze(span, "%entry")
        self.assertIn(("def", "%a5"), [r[2] for r in info["results"]])

    def test_load_after_store_reads_the_stored_value(self):
        span = _span(["OpStore %reg_v1 %a4", "%b = OpLoad %uint %reg_v1", "%b2 = OpIAdd %uint %b %c1",
                      "OpStore %reg_v1 %b2"])
        o = _outliner(PROLOGUE + span)
        info = o.analyze(span, "%entry")
        body = info["key"][3]
        self.assertFalse([l for l in body if "OpLoad" in l])
        self.assertEqual(len(info["params"]), 1)


if __name__ == "__main__":
    unittest.main()
