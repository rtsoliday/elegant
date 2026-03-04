#!/usr/bin/env python3
"""
generate_elegant_lattice_data_el_v4.py

Generates `elegant-lattice-data.el` from ELEGANT's `track_data.c`.

Emits:
- elegant-lattice-element-types
- elegant-lattice-element-docs
- elegant-lattice-element-params
- elegant-lattice-param-docs  ; ((TYPE . PARAM) . (UNIT . (PTYPE . DOC)))

Designed to support `=TAB` parameter info (units, type token, description).
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_RE_C_STRING = re.compile(r'"((?:\\.|[^"\\])*)"')

def _unescape_c_string(s: str) -> str:
    s = s.replace(r'\"', '"')
    s = s.replace(r"\\", "\\")
    s = s.replace(r"\n", "\n")
    s = s.replace(r"\t", "\t")
    s = s.replace(r"\r", "\r")
    return s

def extract_c_string_literals(text: str) -> List[str]:
    return [_unescape_c_string(m.group(1)) for m in _RE_C_STRING.finditer(text)]

def find_array_initializer(source: str, array_name: str) -> str:
    pat = re.compile(rf"\b{re.escape(array_name)}\b\s*(?:\[[^\]]*\])?\s*=\s*\{{", re.MULTILINE)
    m = pat.search(source)
    if not m:
        raise ValueError(f"Could not find initializer for '{array_name}'")
    start = m.end() - 1
    depth = 0
    in_str = False
    esc = False
    i = start
    while i < len(source):
        ch = source[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return source[start+1:i]
        i += 1
    raise ValueError(f"Unterminated initializer for '{array_name}'")

def split_top_level_brace_entries(init: str) -> List[str]:
    entries: List[str] = []
    depth = 0
    in_str = False
    esc = False
    cur: List[str] = []
    for ch in init:
        if in_str:
            cur.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
                cur.append(ch)
            elif ch == '{':
                depth += 1
                cur.append(ch)
            elif ch == '}':
                depth -= 1
                cur.append(ch)
                if depth == 0:
                    entries.append(''.join(cur))
                    cur = []
            else:
                cur.append(ch)
    return entries

@dataclass
class EntityDesc:
    n_params: int
    flags: int
    struct_name: str
    param_array: Optional[str]

def parse_entity_description(source: str) -> List[EntityDesc]:
    init = find_array_initializer(source, "entity_description")
    entries = split_top_level_brace_entries(init)
    out: List[EntityDesc] = []
    for ent in entries:
        strings = extract_c_string_literals(ent)
        struct_name = strings[0] if strings else ""
        nums = re.findall(r"[-+]?\d+", ent)
        n_params = int(nums[0]) if len(nums) > 0 else 0
        flags = int(nums[1]) if len(nums) > 1 else 0
        mpa = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*_param)\b", ent)
        param_array = mpa.group(1) if mpa else None
        out.append(EntityDesc(n_params=n_params, flags=flags, struct_name=struct_name, param_array=param_array))
    return out

@dataclass
class Param:
    name: str
    unit: str = ""
    ptype: str = ""
    description: str = ""

_RE_PARAM_HEAD = re.compile(r'\{\s*"(?P<name>(?:\\.|[^"\\])*)"\s*,\s*"(?P<unit>(?:\\.|[^"\\])*)"\s*,\s*(?P<ptype>[A-Za-z_][A-Za-z0-9_]*)\s*,')

def parse_parameter_array(source: str, array_name: str) -> List[Param]:
    init = find_array_initializer(source, array_name)
    entries = split_top_level_brace_entries(init)
    params: List[Param] = []
    for ent in entries:
        m = _RE_PARAM_HEAD.search(ent)
        strings = extract_c_string_literals(ent)
        if not strings:
            continue
        # name/unit from head strings
        name = strings[0].strip()
        unit = strings[1].strip() if len(strings) >= 2 else ""
        # description is usually last string literal in entry
        desc = strings[-1].strip() if len(strings) >= 3 else ""
        ptype = ""
        if m:
            ptype = m.group('ptype').strip()
        params.append(Param(name=name, unit=unit, ptype=ptype, description=desc))
    return params

def elisp_escape(s: str) -> str:
    s = s or ""
    s = s.replace("\\", "\\\\").replace('"', '\\"').replace("\r", "")
    s = s.replace("\n", "\\n")
    return s

def elisp_str(s: str) -> str:
    return f'"{elisp_escape(s)}"'

def emit_elisp(types_sorted: List[str],
               docs: Dict[str, str],
               params: Dict[str, List[str]],
               param_docs: Dict[Tuple[str, str], Tuple[str, str, str]]) -> str:
    types_list = "'(" + " ".join(elisp_str(t) for t in types_sorted) + ")"
    docs_items = "\n".join(f"    ({elisp_str(k)} . {elisp_str(docs.get(k, ''))})" for k in types_sorted)
    docs_alist = "'(\n" + docs_items + "\n  )"
    params_items = []
    for k in types_sorted:
        lst = params.get(k, [])
        params_items.append(f"    ({elisp_str(k)} . (" + " ".join(elisp_str(x) for x in lst) + "))")
    params_alist = "'(\n" + "\n".join(params_items) + "\n  )"

    pd_items = []
    for et in types_sorted:
        for pn in params.get(et, []):
            unit, ptype, doc = param_docs.get((et, pn), ("", "", ""))
            pd_items.append(
                f"    (({elisp_str(et)} . {elisp_str(pn)}) . ({elisp_str(unit)} . ({elisp_str(ptype)} . {elisp_str(doc)})))"
            )
    pd_alist = "'(\n" + "\n".join(pd_items) + "\n  )"

    return f""";;; elegant-lattice-data.el --- Generated completion data for ELEGANT lattice mode -*- lexical-binding: t; -*-
;; Auto-generated from ELEGANT track_data.c. Do not edit by hand.

(defconst elegant-lattice-element-types
  {types_list}
  \"List of available ELEGANT element types.\")

(defconst elegant-lattice-element-docs
  {docs_alist}
  \"Alist mapping element type (string) -> short description.\")

(defconst elegant-lattice-element-params
  {params_alist}
  \"Alist mapping element type (string) -> list of parameter/qualifier names (strings).\")

(defconst elegant-lattice-param-docs
  {pd_alist}
  \"Alist mapping (TYPE . PARAM) -> (UNIT . (PTYPE . DOC)).\")

(provide 'elegant-lattice-data)
;;; elegant-lattice-data.el ends here
"""

def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Generate elegant-lattice-data.el from ELEGANT track_data.c")
    ap.add_argument("track_data_c", type=Path, help="Path to track_data.c")
    ap.add_argument("-o", "--out", type=Path, default=Path("elegant-lattice-data.el"), help="Output .el path")
    ap.add_argument("--encoding", default="utf-8", help="Encoding for track_data.c")
    args = ap.parse_args(argv)

    src = args.track_data_c.read_text(encoding=args.encoding, errors="replace")
    names = extract_c_string_literals(find_array_initializer(src, "entity_name"))
    texts = extract_c_string_literals(find_array_initializer(src, "entity_text"))
    descs = parse_entity_description(src)

    if len(texts) == len(names) - 1:
        try:
            i_line = names.index("LINE")
            texts = texts[:i_line] + [""] + texts[i_line:]
        except ValueError:
            texts = [""] + texts
    if len(texts) < len(names):
        texts += [""] * (len(names) - len(texts))
    if len(texts) > len(names):
        texts = texts[:len(names)]

    if len(descs) < len(names):
        descs += [EntityDesc(0, 0, "", None)] * (len(names) - len(descs))
    if len(descs) > len(names):
        descs = descs[:len(names)]

    docs = {names[i]: texts[i] for i in range(len(names))}

    element_params: Dict[str, List[str]] = {}
    param_docs: Dict[Tuple[str, str], Tuple[str, str, str]] = {}

    for i, etype in enumerate(names):
        parr = descs[i].param_array
        if not parr:
            element_params[etype] = []
            continue
        try:
            plist = parse_parameter_array(src, parr)
        except Exception:
            element_params[etype] = []
            continue
        seen = set()
        pnames: List[str] = []
        for p in plist:
            if not p.name or p.name in seen:
                continue
            seen.add(p.name)
            pnames.append(p.name)
            param_docs[(etype, p.name)] = (p.unit or "", p.ptype or "", p.description or "")
        element_params[etype] = pnames

    types_sorted = sorted(set(names))
    elisp = emit_elisp(types_sorted, docs, element_params, param_docs)
    args.out.write_text(elisp, encoding="utf-8")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

