"""Synthetic tests for packaging the slot-57 shader cache."""

import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from pack_fspx_gfx import pack, read_required_hashes  # noqa: E402


VERTEX = hashlib.sha256(b"synthetic vertex program").hexdigest()
PIXEL = hashlib.sha256(b"synthetic pixel program").hexdigest()
SPIRV_HEADER = struct.pack("<5I", 0x07230203, 0x00010500, 0, 1, 0)


class PackFspxGraphics(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.translated = self.root / "translated"
        self.translated.mkdir()
        self.out = self.root / "fspx_gfx"
        self.vertex = {"stage": "vs", "fspx_code_sha256": VERTEX, "subgroup_size": 32}
        self.pixel = {"stage": "ps", "fspx_code_sha256": PIXEL, "bindings": [{"binding": 2}]}

    def write_index(self, entries):
        (self.translated / "index.json").write_text(json.dumps(entries), encoding="utf-8", newline="\n")

    def add_module(self, name, contents=SPIRV_HEADER):
        (self.translated / f"{name}.spv").write_bytes(contents)

    def test_pack_rekeys_graphics_and_preserves_metadata(self):
        entries = {"pixel": self.pixel, "compute": {"stage": "cs"},
                   "vertex": dict(self.vertex, fspx_code_sha256=VERTEX.upper(), spv="old.spv")}
        self.write_index(entries)
        self.add_module("pixel", SPIRV_HEADER + b"\0" * 4)
        self.add_module("vertex")
        self.assertEqual(pack(self.translated, self.out), 2)
        index = (self.out / "index.json").read_bytes()
        self.assertEqual(json.loads(index), {VERTEX: self.vertex, PIXEL: self.pixel})
        self.assertEqual((self.out / f"{VERTEX}.spv").read_bytes(), SPIRV_HEADER)
        self.assertEqual((self.out / f"{PIXEL}.spv").read_bytes(), SPIRV_HEADER + b"\0" * 4)
        self.assertEqual({path.name for path in self.out.iterdir()}, {"index.json", f"{VERTEX}.spv", f"{PIXEL}.spv"})

        self.write_index(dict(reversed(list(entries.items()))))
        reordered = self.root / "second-cache"
        pack(self.translated, reordered)
        self.assertEqual((reordered / "index.json").read_bytes(), index)

    def test_pack_requires_every_requested_hash(self):
        self.write_index({"vertex": self.vertex})
        self.add_module("vertex")
        with self.assertRaisesRegex(ValueError, PIXEL):
            pack(self.translated, self.out, {VERTEX, PIXEL})
        self.assertFalse(self.out.exists())

    def test_required_list_selects_only_requested_graphics(self):
        required = self.root / "required.txt"
        required.write_text(f"\n {PIXEL.upper()} \n{PIXEL}\n", encoding="utf-8", newline="\n")
        self.write_index({"vertex": self.vertex, "pixel": self.pixel, "other": {"error": "metadata unavailable"}})
        self.add_module("pixel")
        self.assertEqual(pack(self.translated, self.out, read_required_hashes(required)), 1)
        self.assertEqual(set(json.loads((self.out / "index.json").read_text(encoding="utf-8"))), {PIXEL})

    def test_rejects_empty_or_abbreviated_required_list(self):
        required = self.root / "required.txt"
        for content, error in [("\n", "no required hashes"), (PIXEL[:12], "expected a full SHA-256")]:
            with self.subTest(content=content):
                required.write_text(content, encoding="utf-8", newline="\n")
                with self.assertRaisesRegex(ValueError, error):
                    read_required_hashes(required)

    def test_identical_duplicates_coalesce(self):
        self.write_index({"one": self.vertex, "two": self.vertex})
        self.add_module("one")
        self.add_module("two")
        self.assertEqual(pack(self.translated, self.out), 1)

    def test_rejects_differing_modules_or_metadata_for_the_same_hash(self):
        self.add_module("one")
        for second, contents in [
            (self.vertex, SPIRV_HEADER + b"\0" * 4),
            (dict(self.vertex, stage="ps"), SPIRV_HEADER),
            (dict(self.vertex, subgroup_size=64), SPIRV_HEADER),
        ]:
            with self.subTest(second=second, contents=contents):
                self.write_index({"one": self.vertex, "two": second})
                self.add_module("two", contents)
                with self.assertRaisesRegex(ValueError, "conflicting translations"):
                    pack(self.translated, self.out)
                self.assertFalse(self.out.exists())

    def test_failure_after_a_module_write_removes_partial_cache(self):
        self.write_index({"vertex": self.vertex, "pixel": self.pixel})
        self.add_module("vertex")
        with self.assertRaises(FileNotFoundError):
            pack(self.translated, self.out)
        self.assertFalse(self.out.exists())
        self.add_module("pixel", SPIRV_HEADER + b"\0")
        with self.assertRaisesRegex(ValueError, "invalid SPIR-V"):
            pack(self.translated, self.out)
        self.assertFalse(self.out.exists())

    def test_invalid_digest_or_module_name_is_rejected(self):
        for name, digest, error in [("vertex", VERTEX[:-1], "has no valid fspx_code_sha256"),
                                    ("../vertex", VERTEX, "invalid module name")]:
            with self.subTest(name=name, digest=digest):
                self.write_index({name: dict(self.vertex, fspx_code_sha256=digest)})
                with self.assertRaisesRegex(ValueError, error):
                    pack(self.translated, self.out)
                self.assertFalse(self.out.exists())

    def test_no_graphics_entries_is_an_error(self):
        self.write_index({"compute": {"stage": "cs", "fspx_code_sha256": VERTEX}})
        with self.assertRaisesRegex(ValueError, "no translated FSPX graphics programs"):
            pack(self.translated, self.out)
        self.assertFalse(self.out.exists())

    def test_refuses_to_replace_existing_cache(self):
        self.write_index({"vertex": self.vertex})
        self.add_module("vertex")
        self.out.mkdir()
        old_index = self.out / "index.json"
        old_index.write_bytes(b"{}")
        with self.assertRaises(FileExistsError):
            pack(self.translated, self.out)
        self.assertEqual(list(self.out.iterdir()), [old_index])
        self.assertEqual(old_index.read_bytes(), b"{}")


if __name__ == "__main__":
    unittest.main()
