#!/usr/bin/env python3
"""Profile magnet features in an elegant production test-set tree.

This is a static scanner.  It is intended to rank likely CUDA coverage work
before spending time on a CUDA kernel for a feature that production inputs do
not use often.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


DEFAULT_TEST_SET = Path("/home/soliday/oag/apps/src/elegantTestSet")
DEFAULT_REPORT = Path("test/gpu_cuda/output/reports/phase17-production-magnet-profile.md")
DEFAULT_TSV = Path("test/gpu_cuda/output/reports/phase17-production-magnet-profile.tsv")

BEND_TYPES = {
    "CSBEND",
    "CSBEN",
    "CSRCSBEND",
    "SBEN",
    "RBEN",
    "CCBEND",
    "LGBEND",
    "NIBEND",
}
MULTIPOLE_TYPES = {
    "KQUAD",
    "KSEXT",
    "KOCT",
    "DQCOR",
    "QUAD",
    "SEXT",
    "OCTU",
    "MULT",
    "FMULT",
    "HKICK",
    "VKICK",
    "HVCOR",
    "KICKER",
}
FIELD_MAP_TYPES = {
    "UKICKMAP",
    "CWIGGLER",
    "WIGGLER",
    "BMAPXY",
    "BMXYZ",
    "BRAT",
    "MAPSOLENOID",
    "FTABLE",
}
CUDA_SIMPLE_MULTIPOLE_TYPES = {"KQUAD", "KSEXT", "KOCT", "DQCOR"}
CUDA_SIMPLE_MULT_TYPES = {"MULT"}
CUDA_SIMPLE_CSBEND_TYPES = {"CSBEND", "CSBEN"}
EXISTING_MATRIX_CUDA_TYPES = {"QUAD", "SEXT", "OCTU", "SBEN", "RBEN", "HKICK", "VKICK", "HVCOR", "KICKER"}
MAGNET_TYPES = BEND_TYPES | MULTIPOLE_TYPES | FIELD_MAP_TYPES

ELEMENT_RE = re.compile(
    r'^\s*(?:"(?P<qname>[^"]+)"|(?P<name>[^:\s,]+))\s*:\s*'
    r"(?P<type>[A-Za-z][A-Za-z0-9_]*)\b"
)
PARAM_RE = re.compile(
    r"\b(?P<name>[A-Za-z][A-Za-z0-9_]*)\s*=\s*"
    r"(?P<value>\"[^\"]*\"|'[^']*'|[^,\s]+)"
)


@dataclass
class Occurrence:
    case: str
    path: Path
    line: int
    name: str
    etype: str
    reasons: list[str]
    candidate: bool
    params: dict[str, str]


@dataclass
class CaseSummary:
    files: set[Path] = field(default_factory=set)
    magnets: int = 0
    candidates: int = 0
    blocked: int = 0
    types: Counter[str] = field(default_factory=Counter)
    reasons: Counter[str] = field(default_factory=Counter)


def strip_comment(line: str) -> str:
    in_single = False
    in_double = False
    for index, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "!" and not in_single and not in_double:
            return line[:index]
    return line


def iter_input_files(root: Path) -> Iterable[Path]:
    for suffix in ("*.ele", "*.lte"):
        yield from root.rglob(suffix)


def iter_statements(path: Path) -> Iterable[tuple[int, str]]:
    """Yield rough elegant statements, preserving enough text for profiling."""
    block: list[str] = []
    start_line = 0
    in_namelist = False

    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for lineno, raw_line in enumerate(stream, 1):
            stripped = strip_comment(raw_line).strip()
            if not stripped:
                continue

            lower = stripped.lower()
            starts_namelist = lower.startswith("&") and lower not in {"&", "&end"}
            ends_namelist = lower in {"&", "&end"} or lower.endswith(" &end")

            if in_namelist or starts_namelist:
                if not block:
                    start_line = lineno
                block.append(stripped.rstrip("&").strip())
                in_namelist = not ends_namelist
                if not in_namelist:
                    yield start_line, " ".join(block)
                    block = []
                continue

            if not block:
                start_line = lineno
            continued = stripped.endswith("&")
            block.append(stripped.rstrip("&").strip())
            if not continued:
                yield start_line, " ".join(block)
                block = []

    if block:
        yield start_line, " ".join(block)


def parse_params(statement: str) -> dict[str, str]:
    params: dict[str, str] = {}
    for match in PARAM_RE.finditer(statement):
        params[match.group("name").upper()] = match.group("value").strip().rstrip(",")
    return params


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    cleaned = value.strip().strip("\"'").replace("D", "E").replace("d", "e")
    cleaned = cleaned.rstrip(",")
    try:
        return float(cleaned)
    except ValueError:
        return None


def value_is_set(value: str | None) -> bool:
    if value is None:
        return False
    cleaned = value.strip().strip("\"'").strip()
    if not cleaned:
        return False
    numeric = parse_float(cleaned)
    if numeric is not None:
        return abs(numeric) > 1e-15
    return cleaned.lower() not in {"0", "0.0", "false", "f", "no", "none", "null", "nil"}


def param_is_set(params: dict[str, str], *names: str) -> bool:
    return any(value_is_set(params.get(name.upper())) for name in names)


def max_int_param(params: dict[str, str], names: Iterable[str], default: int = 0) -> int:
    values: list[int] = []
    for name in names:
        numeric = parse_float(params.get(name.upper()))
        if numeric is not None:
            values.append(int(round(numeric)))
    return max(values) if values else default


def has_multipole_file_or_table(params: dict[str, str]) -> bool:
    for name, value in params.items():
        if "MULTIPOLE" in name and value_is_set(value):
            return True
        if name in {"SYSTEMATIC", "RANDOM", "FILENAME", "FIELD_FILE", "INPUT_FILE"} and value_is_set(value):
            return True
    return False


def classify_element(etype: str, params: dict[str, str]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    upper = etype.upper()

    radiation = param_is_set(
        params,
        "SYNCH_RAD",
        "ISR",
        "ISR1PARTICLE",
        "RADIATION",
        "DISTRIBUTION_BASED_RADIATION",
    )
    advanced_misalignment = param_is_set(params, "PITCH", "YAW", "EPITCH", "EYAW") or param_is_set(
        params, "MALIGN_METHOD"
    )
    slice_analysis = param_is_set(params, "SLICE_ANALYSIS_INTERVAL", "SLICE_INTERVAL")

    if upper in EXISTING_MATRIX_CUDA_TYPES:
        if radiation:
            reasons.append("radiation_or_isr")
        return not reasons, reasons

    if upper in CUDA_SIMPLE_MULTIPOLE_TYPES:
        if radiation:
            reasons.append("radiation_or_isr")
        if advanced_misalignment:
            reasons.append("advanced_misalignment")
        if has_multipole_file_or_table(params):
            reasons.append("multipole_tables_or_files")
        if param_is_set(params, "FIDUCIAL", "STEERING", "FRINGE_TYPE", "FFRINGE"):
            reasons.append("extra_multipole_modes")
        return not reasons, reasons

    if upper in CUDA_SIMPLE_MULT_TYPES:
        if radiation:
            reasons.append("radiation_or_isr")
        if advanced_misalignment:
            reasons.append("advanced_misalignment")
        order = max_int_param(params, ("ORDER",), default=1)
        if order < 0 or order > 3:
            reasons.append("unsupported_high_order_multipole")
        return not reasons, reasons

    if upper in CUDA_SIMPLE_CSBEND_TYPES:
        if radiation:
            reasons.append("radiation_or_isr")
        if advanced_misalignment:
            reasons.append("advanced_misalignment")
        if param_is_set(params, "REFERENCE_CORRECTION", "FSE_CORRECTION"):
            reasons.append("csbend_reference_or_fse_correction")
        edge_mode = max_int_param(params, ("EDGE_EFFECTS", "EDGE1_EFFECTS", "EDGE2_EFFECTS"), default=1)
        if edge_mode >= 2:
            reasons.append("csbend_advanced_fringe")
        if slice_analysis:
            reasons.append("slice_analysis_or_slice_tracking")
        return not reasons, reasons

    if upper == "CSRCSBEND":
        reasons.append("csr_csbend_collective")
        if radiation:
            reasons.append("radiation_or_isr")
        if slice_analysis:
            reasons.append("slice_analysis_or_slice_tracking")
        return False, reasons

    if upper in {"CCBEND", "LGBEND", "NIBEND"}:
        reasons.append("advanced_bend_family")
        if radiation:
            reasons.append("radiation_or_isr")
        return False, reasons

    if upper in FIELD_MAP_TYPES:
        reasons.append("field_map_or_wiggler_family")
        if radiation:
            reasons.append("radiation_or_isr")
        return False, reasons

    if upper in MULTIPOLE_TYPES:
        reasons.append("unsupported_multipole_or_corrector_family")
        if radiation:
            reasons.append("radiation_or_isr")
        if advanced_misalignment:
            reasons.append("advanced_misalignment")
        return False, reasons

    return False, ["unclassified_magnet_family"]


def case_name(root: Path, path: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return "."
    return relative.parts[0] if relative.parts else "."


def scan(root: Path) -> tuple[list[Occurrence], int]:
    occurrences: list[Occurrence] = []
    files_scanned = 0
    for path in sorted(iter_input_files(root)):
        files_scanned += 1
        case = case_name(root, path)
        for line, statement in iter_statements(path):
            match = ELEMENT_RE.match(statement)
            if not match:
                continue
            etype = match.group("type").upper()
            if etype not in MAGNET_TYPES:
                continue
            params = parse_params(statement)
            candidate, reasons = classify_element(etype, params)
            occurrences.append(
                Occurrence(
                    case=case,
                    path=path,
                    line=line,
                    name=(match.group("qname") or match.group("name") or "").strip(),
                    etype=etype,
                    reasons=reasons,
                    candidate=candidate,
                    params=params,
                )
            )
    return occurrences, files_scanned


def summarize(root: Path, occurrences: list[Occurrence]) -> dict[str, CaseSummary]:
    summaries: dict[str, CaseSummary] = defaultdict(CaseSummary)
    for occurrence in occurrences:
        summary = summaries[occurrence.case]
        summary.files.add(occurrence.path)
        summary.magnets += 1
        summary.types[occurrence.etype] += 1
        if occurrence.candidate:
            summary.candidates += 1
        else:
            summary.blocked += 1
            for reason in occurrence.reasons:
                summary.reasons[reason] += 1
    return summaries


def top_counter_text(counter: Counter[str], limit: int = 3) -> str:
    if not counter:
        return ""
    return ", ".join(f"{key}:{value}" for key, value in counter.most_common(limit))


def markdown_table(headers: list[str], rows: Iterable[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def make_report(root: Path, files_scanned: int, occurrences: list[Occurrence], top: int) -> str:
    summaries = summarize(root, occurrences)
    type_counts = Counter(occurrence.etype for occurrence in occurrences)
    reason_counts = Counter(reason for occurrence in occurrences for reason in occurrence.reasons)
    candidates = sum(1 for occurrence in occurrences if occurrence.candidate)
    blocked = len(occurrences) - candidates
    generated = _datetime.datetime.now().isoformat(timespec="seconds")

    by_blockers = sorted(summaries.items(), key=lambda item: (item[1].blocked, item[1].magnets), reverse=True)
    by_candidates = sorted(summaries.items(), key=lambda item: (item[1].candidates, item[1].magnets), reverse=True)
    by_magnets = sorted(summaries.items(), key=lambda item: item[1].magnets, reverse=True)

    examples: dict[str, list[Occurrence]] = defaultdict(list)
    for occurrence in occurrences:
        for reason in occurrence.reasons:
            if len(examples[reason]) < 4:
                examples[reason].append(occurrence)

    lines: list[str] = [
        "# Phase 17 Production Magnet Profile",
        "",
        f"- Source root: `{root}`",
        f"- Generated: `{generated}`",
        f"- Files scanned: {files_scanned}",
        f"- Cases with magnet definitions: {len(summaries)}",
        f"- Magnet definitions: {len(occurrences)}",
        f"- Current simple CUDA candidates: {candidates}",
        f"- Definitions with deferred/blocking features: {blocked}",
        "",
        "This is a static profile of `.ele` and `.lte` files.  It ranks likely CUDA work, but runtime benchmarking and SDDS comparisons are still required before enabling any new path.",
        "",
        "## Top Element Types",
        "",
        markdown_table(["Type", "Definitions"], type_counts.most_common(top)),
        "",
        "## Top Deferred Feature Reasons",
        "",
        markdown_table(["Reason", "Definitions"], reason_counts.most_common(top)),
        "",
        "## Top Cases By Deferred Feature Count",
        "",
        markdown_table(
            ["Case", "Files", "Magnets", "Simple candidates", "Deferred", "Top types", "Top reasons"],
            [
                [
                    case,
                    len(summary.files),
                    summary.magnets,
                    summary.candidates,
                    summary.blocked,
                    top_counter_text(summary.types),
                    top_counter_text(summary.reasons),
                ]
                for case, summary in by_blockers[:top]
            ],
        ),
        "",
        "## Top Cases By Simple CUDA Candidate Count",
        "",
        markdown_table(
            ["Case", "Files", "Magnets", "Simple candidates", "Deferred", "Top types"],
            [
                [
                    case,
                    len(summary.files),
                    summary.magnets,
                    summary.candidates,
                    summary.blocked,
                    top_counter_text(summary.types),
                ]
                for case, summary in by_candidates[:top]
            ],
        ),
        "",
        "## Top Cases By Magnet Count",
        "",
        markdown_table(
            ["Case", "Files", "Magnets", "Simple candidates", "Deferred", "Top types", "Top reasons"],
            [
                [
                    case,
                    len(summary.files),
                    summary.magnets,
                    summary.candidates,
                    summary.blocked,
                    top_counter_text(summary.types),
                    top_counter_text(summary.reasons),
                ]
                for case, summary in by_magnets[:top]
            ],
        ),
        "",
        "## Representative Deferred Examples",
        "",
    ]

    for reason, count in reason_counts.most_common(min(top, 12)):
        lines.append(f"### {reason} ({count})")
        for occurrence in examples[reason]:
            rel_path = occurrence.path
            try:
                rel_path = occurrence.path.relative_to(root)
            except ValueError:
                pass
            lines.append(f"- `{rel_path}:{occurrence.line}` `{occurrence.name}: {occurrence.etype}`")
        lines.append("")

    lines.extend(
        [
            "## Phase 17 Guidance",
            "",
            "- Keep stochastic radiation/ISR and spin tracking out of deterministic magnet enablement until distribution-level validation exists.",
            "- Treat `CSRCSBEND`, radiation-enabled bends, and advanced fringe models as separate measured slices; the static profile shows they are common enough to matter but too semantically coupled for a casual port.",
            "- Use the simple-candidate case list to select additional deterministic timing wrappers before widening CUDA support beyond original-mode misalignments.",
            "- Re-run this profiler after adding production wrappers or changing `GPU_SUPPORT` metadata so the next target is based on current coverage rather than memory.",
            "",
        ]
    )
    return "\n".join(lines)


def write_tsv(root: Path, summaries: dict[str, CaseSummary], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write("case\tfiles\tmagnets\tsimple_candidates\tdeferred\ttop_types\ttop_reasons\n")
        for case, summary in sorted(summaries.items(), key=lambda item: item[0].lower()):
            stream.write(
                "\t".join(
                    [
                        case,
                        str(len(summary.files)),
                        str(summary.magnets),
                        str(summary.candidates),
                        str(summary.blocked),
                        top_counter_text(summary.types, 5),
                        top_counter_text(summary.reasons, 5),
                    ]
                )
                + "\n"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", nargs="?", type=Path, default=DEFAULT_TEST_SET)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT, help="Markdown report path.")
    parser.add_argument("--tsv-output", type=Path, default=DEFAULT_TSV, help="Per-case TSV path.")
    parser.add_argument("--top", type=int, default=20, help="Number of rows in top-N report tables.")
    args = parser.parse_args(argv)

    root = args.source_root.resolve()
    if not root.is_dir():
        print(f"error: source root does not exist: {root}", file=sys.stderr)
        return 2

    occurrences, files_scanned = scan(root)
    report = make_report(root, files_scanned, occurrences, max(1, args.top))
    summaries = summarize(root, occurrences)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report + "\n", encoding="utf-8")
    write_tsv(root, summaries, args.tsv_output)

    print(f"wrote {args.output}")
    print(f"wrote {args.tsv_output}")
    print(
        f"scanned {files_scanned} file(s), found {len(occurrences)} magnet definition(s) in {len(summaries)} case(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
