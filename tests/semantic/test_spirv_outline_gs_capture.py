"""SPIRV-SIZE-3: outlining stays out of geometry shaders, so the compute capture pass sees every stage input.

tools/spirv_gs_to_compute.py turns a translated geometry shader into a compute capture pass: it removes the stage
input variables and rewrites their accesses inside %main. A repeated instruction body outlined into a shared %of*
function would keep its read of an input such as %gs_in_pos, which the rewrite then deletes, leaving an id that is
never defined. Outlining runs for native compute shaders only (rdna2spirv.outline_main).

The program is runtime/test/shaders/gs_tri.asm (hand-written) with ten ds_read_b128 before its first ds_read2_b32:
ten identical input reads, enough for the outliner to share them if it ran for this stage.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import support  # noqa: E402,F401  (sets the tools path)
from support import ROOT, translate_lines  # noqa: E402
import rdna2spirv  # noqa: E402
import spirv_gs_to_compute  # noqa: E402

SHADERS = os.path.join(ROOT, "runtime", "test", "shaders")


def gs_program():
    """gs_tri's instruction lines with ten ds_read_b128 inserted before the first ds_read2_b32."""
    with open(os.path.join(SHADERS, "gs_tri.asm"), encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith(";")]
    first = next(i for i, ln in enumerate(lines) if ln.startswith("ds_read2_b32"))
    return lines[:first] + ["ds_read_b128 v[20:23], v10"] * 10 + lines[first:]


def gs_container():
    sys.path.insert(0, SHADERS)
    import build                                       # gs_container: gs_tri's ISG1/OSG1
    return build.gs_container()


def undefined_ids(spvasm):
    """Ids the module references but never defines (every id in SPIR-V assembly is some instruction's result)."""
    defined = set(re.findall(r"^\s*(%\w+)\s*=", spvasm, re.M))
    used = set(re.findall(r"%\w+", spvasm))
    return sorted(used - defined)


def validate(spvasm):
    """spirv-as and spirv-val (Vulkan 1.3); None when they are not on PATH, else spirv-val's error text or ''."""
    sa, sv = shutil.which("spirv-as"), shutil.which("spirv-val")
    if not (sa and sv):
        return None
    with tempfile.TemporaryDirectory() as td:
        asm_p, spv_p = os.path.join(td, "mod.spvasm"), os.path.join(td, "mod.spv")
        with open(asm_p, "w", encoding="utf-8", newline="\n") as f:
            f.write(spvasm)
        r = subprocess.run([sa, "--target-env", "vulkan1.3", asm_p, "-o", spv_p], capture_output=True, text=True)
        if r.returncode:
            return r.stderr or r.stdout or "spirv-as failed"
        r = subprocess.run([sv, "--target-env", "vulkan1.3", spv_p], capture_output=True, text=True)
        return (r.stderr or r.stdout or "spirv-val failed") if r.returncode else ""


class OutlineGsCapture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        saved = rdna2spirv.OUTLINE
        rdna2spirv.OUTLINE = True
        try:
            cls.gs = translate_lines("gs_outline_capture", gs_program(), dxbc=gs_container())
        finally:
            rdna2spirv.OUTLINE = saved
        cls.capture, _ = spirv_gs_to_compute.rewrite(cls.gs)

    def test_geometry_shader_is_not_outlined(self):
        self.assertIn("OpEntryPoint Geometry", self.gs)
        self.assertNotRegex(self.gs, r"OpFunctionCall %\w+ %of")

    def test_capture_defines_every_id_it_uses(self):
        self.assertIn("OpEntryPoint GLCompute", self.capture)
        self.assertEqual(undefined_ids(self.capture), [])

    def test_both_stages_validate(self):
        for name, text in (("geometry", self.gs), ("capture", self.capture)):
            err = validate(text)
            if err is None:
                self.skipTest("spirv-as / spirv-val not on PATH")
            self.assertEqual(err, "", f"{name} module: {err}")


if __name__ == "__main__":
    unittest.main()
