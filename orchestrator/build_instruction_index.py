#!/usr/bin/env python3
"""
Build an instruction index JSON from the ARM64 ISA XML specification files.

Parses every XML file under ISA_A64/ to extract the instruction identifier
(instructionsection/@id), mnemonic, instruction class, and brief description.
Maps id → {file, mnemonic, class, title}.

Usage:
    python build_instruction_index.py [--spec-dir PATH] [--version DEFAULT] [--out PATH]
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from xml.etree import ElementTree


def _resolve_version_dir(parent: Path, pattern: str) -> Path:
    """Find the versioned directory inside *parent* matching *pattern*.

    Picks the newest (highest sort order) when multiple directories match.
    Raises FileNotFoundError if none are found.
    """
    candidates = sorted(
        p for p in parent.glob(pattern)
        if p.is_dir() and not p.name.endswith("_diff") and "diff" not in p.stem
    )
    if not candidates:
        raise FileNotFoundError(
            f"No directory matching '{pattern}' found inside {parent}"
        )
    if len(candidates) > 1:
        chosen = candidates[-1]
        names = "\n  ".join(c.name for c in candidates)
        print(
            f"[WARN] Multiple version dirs found, picking newest:\n"
            f"  {names}\n"
            f"  → {chosen.name}\n"
            f"  Remove old directories to silence this warning.",
            file=sys.stderr,
        )
        return chosen
    return candidates[0]


def parse_instructionsection(xml_path: Path) -> dict | None:
    """
    Safely parse a single ISA XML file.

    Returns a dict with keys: id, mnemonic, class, title, file
    or None if the file cannot be parsed or contains no instructionsection.
    """
    try:
        tree = ElementTree.parse(xml_path)
    except ElementTree.ParseError as exc:
        print(f"[WARN] XML parse error in {xml_path.name}: {exc}", file=sys.stderr)
        return None
    except Exception as exc:
        print(f"[WARN] Could not read {xml_path.name}: {exc}", file=sys.stderr)
        return None

    root = tree.getroot()

    # The root element <instructionsection> carries the core metadata.
    if root.tag != "instructionsection":
        print(f"[WARN] {xml_path.name}: root tag is <{root.tag}>, skipping", file=sys.stderr)
        return None

    instr_id = root.get("id")
    if not instr_id:
        print(f"[WARN] {xml_path.name}: missing @id on <instructionsection>, skipping",
              file=sys.stderr)
        return None

    title = root.get("title", "").strip()

    # Extract mnemonic and instr-class from <docvars> inside the root.
    mnemonic = ""
    instr_class = ""
    docvars = root.find("docvars")
    if docvars is not None:
        for dv in docvars.findall("docvar"):
            key = dv.get("key", "")
            if key == "mnemonic":
                mnemonic = (dv.get("value") or "").strip()
            elif key == "instr-class":
                instr_class = (dv.get("value") or "").strip()

    return {
        "id": instr_id,
        "mnemonic": mnemonic,
        "class": instr_class,
        "title": title,
        "file": xml_path.name,
    }


def build_instruction_index(spec_root: str) -> dict:
    """
    Walk the ISA_A64 XML directory and build the complete index.

    Returns a dict mapping instruction id → record.
    """
    spec_root = Path(spec_root).expanduser().resolve()
    xml_dir = _resolve_version_dir(spec_root / "ISA_A64", "ISA_A64_xml_A_profile-*")

    index = {}
    files_scanned = 0
    files_parsed = 0

    xml_files = sorted(f for f in xml_dir.iterdir() if f.suffix == ".xml")
    files_scanned = len(xml_files)

    for xml_path in xml_files:
        record = parse_instructionsection(xml_path)
        if record is None:
            continue
        instr_id = record["id"]
        if instr_id in index:
            print(
                f"[WARN] Duplicate instruction id '{instr_id}' "
                f"({record['file']} vs {index[instr_id]['file']}), "
                f"keeping first.",
                file=sys.stderr,
            )
            continue
        index[instr_id] = record
        files_parsed += 1

    print(
        f"[INFO] Scanned {files_scanned} XML files, "
        f"indexed {files_parsed} instructions.",
        file=sys.stderr,
    )
    return index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build instruction index from ARM64 ISA XML specs."
    )
    parser.add_argument(
        "--spec-dir",
        default=os.environ.get("ARM64_SPEC_DIR", ""),
        help="Path to the arm64_specs directory (may also be set via ARM64_SPEC_DIR env var)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON file path (default: instruction_index.json in current dir)",
    )
    args = parser.parse_args()

    if not args.spec_dir:
        print(
            "[FATAL] --spec-dir is required (or set ARM64_SPEC_DIR env var).",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        index = build_instruction_index(args.spec_dir)
    except Exception as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    out_path = Path(args.out) if args.out else Path("instruction_index.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": {
                    "spec_root": str(Path(args.spec_dir).resolve()),
                    "total_instructions": len(index),
                    "generated_by": "build_instruction_index.py",
                },
                "instructions": index,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[INFO] Wrote {out_path.resolve()} ({len(index)} instructions)")


if __name__ == "__main__":
    main()
