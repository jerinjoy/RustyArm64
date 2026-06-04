#!/usr/bin/env python3
"""
Build a system register index JSON from the ARM64 SysReg XML specification files.

Parses every XML file under SysReg/ and maps each register short name
(e.g. 'SCTLR_EL1', 'TTBR0_EL1') to its containing XML file.
Filters out stub entries and register names that are purely AArch32 aliases.

Usage:
    python build_sysreg_index.py --spec-dir /path/to/arm64_specs [--out PATH]
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


def parse_sysreg_file(xml_path: Path) -> list[dict]:
    """
    Safely parse a single SysReg XML file.

    Each <register> element yields one record dict with keys:
        name      - reg_short_name text
        state     - execution_state (AArch64, AArch32, or None for shared)
        long_name - reg_long_name text
        purpose   - brief purpose text
        stub      - bool, True if is_stub_entry
    """
    records: list[dict] = []

    try:
        tree = ElementTree.parse(xml_path)
    except ElementTree.ParseError as exc:
        print(f"[WARN] XML parse error in {xml_path.name}: {exc}", file=sys.stderr)
        return records
    except Exception as exc:
        print(f"[WARN] Could not read {xml_path.name}: {exc}", file=sys.stderr)
        return records

    root = tree.getroot()
    if root.tag != "register_page":
        print(f"[WARN] {xml_path.name}: root tag is <{root.tag}>, skipping",
              file=sys.stderr)
        return records

    registers_elem = root.find("registers")
    if registers_elem is None:
        print(f"[WARN] {xml_path.name}: no <registers> element, skipping",
              file=sys.stderr)
        return records

    for reg in registers_elem.findall("register"):
        short_name = reg.findtext("reg_short_name", default="").strip()
        if not short_name:
            continue

        long_name = reg.findtext("reg_long_name", default="").strip()
        execution_state = reg.get("execution_state", "").strip() or None
        is_stub = reg.get("is_stub_entry", "False").lower() == "true"

        purpose = ""
        purpose_elem = reg.find("reg_purpose")
        if purpose_elem is not None:
            purpose_text = purpose_elem.find("purpose_text")
            if purpose_text is not None:
                para = purpose_text.find("para")
                if para is not None:
                    text = "".join(para.itertext()).strip()
                    purpose = " ".join(text.split())

        records.append({
            "name": short_name,
            "state": execution_state,
            "long_name": long_name,
            "purpose": purpose,
            "stub": is_stub,
        })

    return records


def build_sysreg_index(spec_root: str) -> dict:
    """
    Walk the SysReg XML directory and map each register name → file.

    Returns a dict like:
        {
            "SCTLR_EL1": {
                "file": "AArch64-sctlr_el1.xml",
                "long_name": "System Control Register (EL1)",
                "purpose": "Provides top-level control of the system..."
            },
            ...
        }
    """
    spec_root = Path(spec_root).expanduser().resolve()
    xml_dir = _resolve_version_dir(spec_root / "SysReg", "SysReg_xml_A_profile-*")

    index: dict[str, dict] = {}
    regs_total = 0
    regs_indexed = 0
    stubs_skipped = 0

    xml_files = sorted(f for f in xml_dir.iterdir() if f.suffix == ".xml")
    files_scanned = len(xml_files)

    for xml_path in xml_files:
        reg_records = parse_sysreg_file(xml_path)
        regs_total += len(reg_records)

        for rec in reg_records:
            name = rec["name"]

            # Skip AArch32-only entries
            if rec["state"] == "AArch32":
                continue

            # Skip stub entries — they point to other files
            if rec["stub"]:
                stubs_skipped += 1
                continue

            entry = {
                "file": xml_path.name,
                "long_name": rec["long_name"],
                "purpose": rec["purpose"],
                "_state": rec["state"],
            }

            if name in index:
                existing_state = index[name].get("_state")
                new_state = rec["state"]
                if existing_state == "AArch64":
                    continue
                if new_state == "AArch64":
                    index[name] = entry
                    continue
                if len(rec["purpose"]) <= len(index[name].get("purpose", "")):
                    continue
            else:
                regs_indexed += 1

            index[name] = entry

    # Strip internal _state key from final output
    for entry in index.values():
        entry.pop("_state", None)

    print(
        f"[INFO] Scanned {files_scanned} XML files, "
        f"found {regs_total} register entries, "
        f"indexed {regs_indexed} unique names, "
        f"skipped {stubs_skipped} stubs.",
        file=sys.stderr,
    )
    return index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build system register index from ARM64 SysReg XML specs."
    )
    parser.add_argument(
        "--spec-dir",
        default=os.environ.get("ARM64_SPEC_DIR", ""),
        help="Path to the arm64_specs directory (may also be set via ARM64_SPEC_DIR env var)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON file path (default: sysreg_index.json in current dir)",
    )
    args = parser.parse_args()

    if not args.spec_dir:
        print(
            "[FATAL] --spec-dir is required (or set ARM64_SPEC_DIR env var).",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        index = build_sysreg_index(args.spec_dir)
    except Exception as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    out_path = Path(args.out) if args.out else Path("sysreg_index.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": {
                    "spec_root": str(Path(args.spec_dir).resolve()),
                    "total_registers": len(index),
                    "generated_by": "build_sysreg_index.py",
                },
                "registers": index,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[INFO] Wrote {out_path.resolve()} ({len(index)} registers)")


if __name__ == "__main__":
    main()
