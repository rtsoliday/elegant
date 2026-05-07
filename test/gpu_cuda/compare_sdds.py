#!/usr/bin/env python3
"""Compare two elegant output directories with sddsdiff."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_EXTENSIONS = {
    ".acc",
    ".cen",
    ".csr",
    ".fin",
    ".mag",
    ".los",
    ".lost",
    ".mat",
    ".out",
    ".par",
    ".param",
    ".sig",
    ".twi",
    ".wake",
    ".BC1BEG",
    ".BC1END",
    ".BC2BEG",
    ".BC2END",
    ".DL1BEG",
    ".DL1END",
    ".DL2END",
    ".XBEG",
}


def find_tool(name: str, explicit: str | None, repo_root: Path) -> str | None:
    if explicit:
        return explicit
    for candidate in (
        repo_root / "../SDDS/bin/Linux-x86_64" / name,
        repo_root / "../SDDS/SDDSaps/O.Linux-x86_64" / name,
    ):
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
    return shutil.which(name)


def files_by_relative_path(root: Path, extensions: set[str]) -> dict[Path, Path]:
    result: dict[Path, Path] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in extensions:
            result[path.relative_to(root)] = path
    return result


def has_particle_id(path: Path, sddsquery: str | None) -> bool:
    if not sddsquery:
        return False
    proc = subprocess.run(
        [sddsquery, "-column", str(path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return proc.returncode == 0 and any(line.strip() == "particleID" for line in proc.stdout.splitlines())


def compare_file(reference: Path, candidate: Path, args: argparse.Namespace, sddsquery: str | None) -> tuple[bool, str]:
    cmd = [
        args.sddsdiff,
        str(reference),
        str(candidate),
        "-compareCommon=column",
        f"-tolerance={args.tolerance}",
        "-ignoreUnits",
    ]
    if has_particle_id(reference, sddsquery) and has_particle_id(candidate, sddsquery):
        cmd.append("-rowlabel=particleID,nocomparison")

    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    output = (proc.stdout + proc.stderr).strip()
    if proc.returncode == 0:
        return True, "ok"
    return False, output or f"sddsdiff exited with {proc.returncode}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--tolerance", default="1e-12", help="sddsdiff absolute tolerance")
    parser.add_argument("--extensions", default=",".join(sorted(DEFAULT_EXTENSIONS)), help="comma-separated file extensions")
    parser.add_argument("--sddsdiff", help="path to sddsdiff")
    parser.add_argument("--sddsquery", help="path to sddsquery")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    args.sddsdiff = find_tool("sddsdiff", args.sddsdiff, repo_root)
    sddsquery = find_tool("sddsquery", args.sddsquery, repo_root)

    if not args.sddsdiff:
        print("sddsdiff not found; install SDDS tools or pass --sddsdiff", file=sys.stderr)
        return 2

    extensions = {item if item.startswith(".") else f".{item}" for item in args.extensions.split(",") if item}
    reference_files = files_by_relative_path(args.reference, extensions)
    candidate_files = files_by_relative_path(args.candidate, extensions)

    missing = sorted(reference_files.keys() - candidate_files.keys())
    extra = sorted(candidate_files.keys() - reference_files.keys())
    common = sorted(reference_files.keys() & candidate_files.keys())

    failures = 0
    for relpath in missing:
        print(f"missing candidate file: {relpath}")
        failures += 1
    for relpath in extra:
        print(f"extra candidate file: {relpath}")
        failures += 1

    for relpath in common:
        ok, message = compare_file(reference_files[relpath], candidate_files[relpath], args, sddsquery)
        if ok:
            print(f"PASS {relpath}")
        else:
            print(f"FAIL {relpath}: {message}")
            failures += 1

    if not common:
        print("no common SDDS-like files found", file=sys.stderr)
        return 2

    if failures:
        print(f"{failures} comparison failure(s)")
        return 1
    print(f"all {len(common)} common file(s) matched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
