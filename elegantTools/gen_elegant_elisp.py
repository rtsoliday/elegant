#!/usr/bin/env python3
"""
Generate Emacs Lisp tables for ELEGANT #namelist commands and qualifiers.

Input format: ELEGANT namelist definitions with blocks like:
  #namelist run_setup static
      STRING lattice = NULL;
      long n_steps = 1;
  #end

This script emits an .el file with:
  - elegant-namelist-commands
  - elegant-namelist-qualifiers
  - elegant-namelist-qualifier-types

Usage:
  python3 gen_elegant_elisp.py elegant-namelists.txt
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import OrderedDict
from typing import Dict, List, Tuple


# Matches: #namelist <NAME> ...
RE_NAMELIST_START = re.compile(r"^\s*#namelist\s+([A-Za-z_]\w*)\b")

# Matches qualifier declarations inside a namelist block.
# Examples it should capture:
#   STRING name = NULL;
#   double delta_x = 5e-5;
#   long halton_sequence[3] = {0, 0, 0};
#   int32_t halton_radix[6] = {0, 0, 0, 0, 0, 0};
#   short include_x = 1;
RE_DECL = re.compile(
    r"""^\s*
        (STRING|double|long|short|int32_t)      # type
        \s+
        ([A-Za-z_]\w*)                          # name
        (?:\s*\[[^\]]*\])?                      # optional array suffix [..]
        \s*
        (?:=|;)\s*                              # initializer or end
    """,
    re.VERBOSE,
)

RE_END = re.compile(r"^\s*#end\b")

# Elisp string escaping (minimal)
def elisp_string(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def parse_namelists(text: str) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, str]]]:
    """
    Returns:
      qualifiers: command -> [qualifier names in file order]
      types: command -> {qualifier -> base type}
    """
    qualifiers: Dict[str, List[str]] = OrderedDict()
    types: Dict[str, Dict[str, str]] = OrderedDict()

    current: str | None = None
    seen_in_current: set[str] = set()

    for line in text.splitlines():
        m = RE_NAMELIST_START.match(line)
        if m:
            current = m.group(1)
            if current not in qualifiers:
                qualifiers[current] = []
                types[current] = OrderedDict()
            seen_in_current = set()
            continue

        if current is not None and RE_END.match(line):
            current = None
            seen_in_current = set()
            continue

        if current is None:
            continue

        dm = RE_DECL.match(line)
        if not dm:
            continue

        base_type = dm.group(1)
        name = dm.group(2)

        # De-duplicate within a block, preserving first occurrence order
        if name in seen_in_current:
            continue
        seen_in_current.add(name)

        qualifiers[current].append(name)
        types[current][name] = base_type

    return qualifiers, types


def emit_elisp(qualifiers: Dict[str, List[str]], types: Dict[str, Dict[str, str]]) -> str:
    commands = list(qualifiers.keys())

    out: List[str] = []
    out.append(";;; elegant-namelists.el --- Generated ELEGANT namelist tables -*- lexical-binding: t; -*-")
    out.append(";;;")
    out.append(";;; Auto-generated. Do not edit by hand.")
    out.append(";;;")
    out.append(";;; Provides:")
    out.append(";;;   elegant-namelist-commands")
    out.append(";;;   elegant-namelist-qualifiers")
    out.append(";;;   elegant-namelist-qualifier-types")
    out.append(";;;")
    out.append("")

    # Commands
    out.append("(defconst elegant-namelist-commands")
    out.append("  (list")
    for cmd in commands:
        out.append(f"   {elisp_string(cmd)}")
    out.append("   )")
    out.append('  "List of ELEGANT #namelist command names.")')
    out.append("")

    # Qualifiers (alist)
    out.append("(defconst elegant-namelist-qualifiers")
    out.append("  (list")
    for cmd, qs in qualifiers.items():
        out.append(f"   (cons {elisp_string(cmd)}")
        out.append("         (list")
        for q in qs:
            out.append(f"          {elisp_string(q)}")
        out.append("          ))")
    out.append("   )")
    out.append('  "Alist mapping namelist command -> list of qualifier names.")')
    out.append("")

    # Types (alist of alists)
    out.append("(defconst elegant-namelist-qualifier-types")
    out.append("  (list")
    for cmd, qtypes in types.items():
        out.append(f"   (cons {elisp_string(cmd)}")
        out.append("         (list")
        for q, t in qtypes.items():
            out.append(f"          (cons {elisp_string(q)} {elisp_string(t)})")
        out.append("          ))")
    out.append("   )")
    out.append('  "Alist mapping namelist command -> alist of (qualifier . base-type).")')
    out.append("")

    out.append("(provide 'elegant-namelists)")
    out.append(";;; elegant-namelists.el ends here")
    out.append("")

    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate Emacs Lisp tables (elegant-namelists.el) from ELEGANT namelist definitions.")
    ap.add_argument("input", help="Path to elegant-namelists.txt (namelist definitions).")
    ap.add_argument("-o", "--output", default="elegant-namelists.el", help="Output .el path (default: stdout). Use '-' for stdout.")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    qualifiers, types = parse_namelists(text)
    elisp = emit_elisp(qualifiers, types)

    if args.output == "-" or args.output.strip() == "":
        sys.stdout.write(elisp)
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(elisp)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
