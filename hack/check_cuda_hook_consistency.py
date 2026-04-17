#!/usr/bin/env python3
"""Static consistency checks for CUDA hook registration chain."""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]

HOOK_FILE = ROOT / "src/cuda/hook.c"
ENUM_FILE = ROOT / "src/include/libcuda_hook.h"
DLSYM_FILE = ROOT / "src/libvgpu.c"
WRAPPER_GLOB = "src/cuda/*.c"

# Symbols intentionally excluded from specific checks.
OPTIONAL_DLSYM_SYMBOLS = {"cuGetExportTable"}
OPTIONAL_WRAPPER_SYMBOLS = {"cuGetExportTable"}


def strip_c_comments(content: str) -> str:
    content = re.sub(r"/\*.*?\*/", "", content, flags=re.S)
    content = re.sub(r"//.*", "", content)
    return content


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path.read_text(encoding="utf-8")


def extract_hook_entries(content: str) -> list[str]:
    return re.findall(r'\{\.name\s*=\s*"(cu[^"]+)"\}', strip_c_comments(content))


def extract_enum_entries(content: str) -> list[str]:
    return re.findall(r"CUDA_OVERRIDE_ENUM\((cu[^)]+)\)", strip_c_comments(content))


def extract_dlsym_entries(content: str) -> list[str]:
    return re.findall(r"DLSYM_HOOK_FUNC\((cu[^)]+)\)", strip_c_comments(content))


def extract_wrapper_definitions(paths: Iterable[Path]) -> set[str]:
    # Match C function definitions with optional prefixes like CUDAAPI/FUNC_ATTR_VISIBLE.
    pattern = re.compile(
        r"^\s*(?:[A-Za-z_][A-Za-z0-9_]*\s+)*(?:CUresult|void\s*\*)\s+"
        r"(?:[A-Za-z_][A-Za-z0-9_]*\s+)*(cu[A-Za-z0-9_]+)\s*\(",
        flags=re.M,
    )
    definitions: set[str] = set()
    for path in paths:
        definitions.update(pattern.findall(read_text(path)))
    return definitions


def format_symbols(symbols: Iterable[str], max_items: int = 20) -> str:
    items = sorted(symbols)
    if len(items) <= max_items:
        return ", ".join(items)
    remaining = len(items) - max_items
    return f"{', '.join(items[:max_items])}, ... (+{remaining} more)"


def report_duplicates(name: str, items: list[str]) -> list[str]:
    duplicates = sorted([symbol for symbol, count in Counter(items).items() if count > 1])
    if duplicates:
        print(f"[ERROR] Duplicate {name}: {format_symbols(duplicates)}")
    return duplicates


def main() -> int:
    hook_entries = extract_hook_entries(read_text(HOOK_FILE))
    enum_entries = extract_enum_entries(read_text(ENUM_FILE))
    dlsym_entries = extract_dlsym_entries(read_text(DLSYM_FILE))
    wrapper_files = [DLSYM_FILE, *sorted(ROOT.glob(WRAPPER_GLOB))]
    wrapper_defs = extract_wrapper_definitions(wrapper_files)

    has_error = False

    if not hook_entries:
        print("[ERROR] No CUDA hook entries found in hook.c.")
        return 1

    if not enum_entries:
        print("[ERROR] No CUDA override enum entries found in libcuda_hook.h.")
        return 1

    if len(hook_entries) != len(enum_entries):
        has_error = True
        print(
            "[ERROR] Entry count mismatch: "
            f"hook.c has {len(hook_entries)}, enum has {len(enum_entries)}."
        )

    hook_set = set(hook_entries)
    enum_set = set(enum_entries)
    dlsym_set = set(dlsym_entries)

    missing_in_enum = hook_set - enum_set
    extra_in_enum = enum_set - hook_set
    if missing_in_enum or extra_in_enum:
        has_error = True
        if missing_in_enum:
            print(f"[ERROR] Missing in enum: {format_symbols(missing_in_enum)}")
        if extra_in_enum:
            print(f"[ERROR] Extra in enum: {format_symbols(extra_in_enum)}")

    for idx, (hook_symbol, enum_symbol) in enumerate(zip(hook_entries, enum_entries)):
        if hook_symbol != enum_symbol:
            has_error = True
            print(
                f"[ERROR] Order mismatch at index {idx}: "
                f"hook.c={hook_symbol}, enum={enum_symbol}"
            )
            break

    required_in_dlsym = hook_set - OPTIONAL_DLSYM_SYMBOLS
    missing_in_dlsym = required_in_dlsym - dlsym_set
    extra_in_dlsym = dlsym_set - hook_set
    if missing_in_dlsym:
        has_error = True
        print(f"[ERROR] Missing DLSYM_HOOK_FUNC entries: {format_symbols(missing_in_dlsym)}")
    if extra_in_dlsym:
        has_error = True
        print(f"[ERROR] Unknown DLSYM_HOOK_FUNC entries: {format_symbols(extra_in_dlsym)}")

    required_wrappers = hook_set - OPTIONAL_WRAPPER_SYMBOLS
    missing_wrappers = required_wrappers - wrapper_defs
    if missing_wrappers:
        has_error = True
        print(f"[ERROR] Missing wrapper function definitions: {format_symbols(missing_wrappers)}")

    duplicate_hook_entries = report_duplicates("hook entries", hook_entries)
    duplicate_enum_entries = report_duplicates("enum entries", enum_entries)
    duplicate_dlsym_entries = report_duplicates("DLSYM_HOOK_FUNC entries", dlsym_entries)
    if duplicate_hook_entries or duplicate_enum_entries or duplicate_dlsym_entries:
        has_error = True

    if has_error:
        print("[FAIL] CUDA hook static consistency check failed.")
        return 1

    print(
        "[PASS] CUDA hook static consistency check passed "
        f"(entries={len(hook_entries)}, wrappers={len(wrapper_defs)})."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
