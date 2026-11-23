"""Pack translated FSPX graphics programs into a runtime shader sub-index."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replace_retry import replace  # noqa: E402


GRAPHICS_STAGES = ("vs", "ps", "ms", "gs")
SHA256 = re.compile(r"[0-9a-fA-F]{64}")
MODULE_STEM = re.compile(r"[A-Za-z0-9_+.-]+")
SPIRV_MAGIC = b"\x03\x02\x23\x07"


def read_required_hashes(path):
    required = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        digest = line.strip()
        if not digest:
            continue
        if not SHA256.fullmatch(digest):
            raise ValueError(f"{path}:{number}: expected a full SHA-256")
        required.add(digest.lower())
    if not required:
        raise ValueError(f"{path}: no required hashes")
    return required


def select_programs(translated, required):
    source_index = translated / "index.json"
    index = json.loads(source_index.read_text(encoding="utf-8"))
    if not isinstance(index, dict):
        raise ValueError(f"{source_index}: expected an object keyed by shader name")

    programs = {}
    for name, meta in index.items():
        if not isinstance(meta, dict):
            raise ValueError(f"{source_index}: {name!r} must be a metadata object")
        digest = meta.get("fspx_code_sha256")
        valid_digest = isinstance(digest, str) and SHA256.fullmatch(digest) is not None
        if required is not None and (not valid_digest or digest.lower() not in required):
            continue
        if not isinstance(meta.get("stage"), str):
            raise ValueError(f"{source_index}: {name!r} has no valid stage")
        if meta["stage"] not in GRAPHICS_STAGES:
            continue
        if not valid_digest:
            raise ValueError(f"{source_index}: {name!r} has no valid fspx_code_sha256")
        digest = digest.lower()
        if name in (".", "..") or not MODULE_STEM.fullmatch(name):
            raise ValueError(f"{source_index}: invalid module name {name!r}")

        entry = dict(meta, fspx_code_sha256=digest)
        entry.pop("spv", None)  # The output filename is always <digest>.spv.
        programs.setdefault(digest, []).append((translated / f"{name}.spv", entry))

    missing = required - programs.keys() if required is not None else set()
    if missing:
        raise ValueError(f"requested graphics hashes missing from {source_index}: {', '.join(sorted(missing))}")
    if not programs:
        raise ValueError(f"{source_index}: no translated FSPX graphics programs")
    return programs


def read_module(path):
    data = path.read_bytes()
    if len(data) < 20 or len(data) % 4 or not data.startswith(SPIRV_MAGIC):
        raise ValueError(f"{path}: invalid SPIR-V header or word alignment")
    return data


def pack(translated, out, required=None):
    programs = select_programs(translated, required)
    index = {}
    with ExitStack() as rollback:
        out.mkdir()  # Reserve a new directory without replacing an existing cache.
        rollback.callback(out.rmdir)
        for digest, entries in programs.items():
            first_path, first_meta = entries[0]
            module = read_module(first_path)
            for path, meta in entries[1:]:
                if meta != first_meta or read_module(path) != module:
                    raise ValueError(f"conflicting translations for {digest}: {first_path.name} and {path.name}")
            dest = out / f"{digest}.spv"
            rollback.callback(dest.unlink, missing_ok=True)
            dest.write_bytes(module)
            index[digest] = first_meta

        staging = out / ".index.json.tmp"
        rollback.callback(staging.unlink, missing_ok=True)
        with staging.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(index, f, indent=2, sort_keys=True)
            f.write("\n")
        # The runtime discovers the cache through index.json, so publish it last.
        replace(staging, out / "index.json")
        rollback.pop_all()
    return len(index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("translated", type=Path, help="rdna2spirv.py output directory with index.json")
    parser.add_argument("--out", type=Path, required=True, help="new fspx_gfx directory in the runtime shader root")
    parser.add_argument("--require-hashes", type=Path, help="pack only these native SHA-256 digests, one per line")
    args = parser.parse_args()
    try:
        required = read_required_hashes(args.require_hashes) if args.require_hashes is not None else None
        count = pack(args.translated, args.out, required)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")
    print(f"Packed {count} graphics programs into {args.out}")


if __name__ == "__main__":
    main()
