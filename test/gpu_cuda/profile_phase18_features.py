#!/usr/bin/env python3
"""Profile Phase 18 production candidates in an elegant test-set tree.

The Phase 18 scope spans SCMULT, Poisson-backed field solves, field maps,
wigglers, and ion effects.  This static scanner ranks production inputs before
we choose a CUDA implementation slice.
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
DEFAULT_REPORT = Path("test/gpu_cuda/output/reports/phase18-production-feature-profile.md")
DEFAULT_TSV = Path("test/gpu_cuda/output/reports/phase18-production-feature-profile.tsv")

SCMULT_ELEMENT_TYPES = {"SCMULT"}
FIELD_MAP_TYPES = {
    "BMAPXY",
    "BMAPXYZ",
    "BMXYZ",
    "BRAT",
    "BGGEXP",
    "FTABLE",
    "GKICKMAP",
    "KICKMAP",
    "UKICKMAP",
}
WIGGLER_TYPES = {"CWIGGLER", "GFWIGGLER", "WIGGLER"}
ION_TYPES = {"IONEFFECTS"}
ADJACENT_COLLECTIVE_TYPES = {"BEAMBEAM"}
PHASE18_TYPES = SCMULT_ELEMENT_TYPES | FIELD_MAP_TYPES | WIGGLER_TYPES | ION_TYPES | ADJACENT_COLLECTIVE_TYPES

ELEMENT_RE = re.compile(
    r'^\s*(?:"(?P<qname>[^"]+)"|(?P<name>[^:\s,]+))\s*:\s*'
    r"(?P<type>[A-Za-z][A-Za-z0-9_]*)\b"
)
PARAM_RE = re.compile(
    r"\b(?P<name>[A-Za-z][A-Za-z0-9_]*)\s*=\s*"
    r"(?P<value>\"[^\"]*\"|'[^']*'|[^,\s]+)"
)
NAMELIST_RE = re.compile(r"^\s*&(?P<name>[A-Za-z][A-Za-z0-9_]*)\b")


@dataclass
class CaseFlags:
    files: set[Path] = field(default_factory=set)
    bunched_beam: int = 0
    sdds_bunched_mode: int = 0
    poisson_settings: int = 0


@dataclass
class Occurrence:
    case: str
    path: Path
    line: int
    family: str
    kind: str
    name: str
    flags: list[str]


@dataclass
class CaseSummary:
    files: set[Path] = field(default_factory=set)
    occurrences: int = 0
    families: Counter[str] = field(default_factory=Counter)
    kinds: Counter[str] = field(default_factory=Counter)
    flags: Counter[str] = field(default_factory=Counter)


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
    cleaned = value.strip().strip("\"'").replace("D", "E").replace("d", "e").rstrip(",")
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


def case_name(root: Path, path: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return "."
    return relative.parts[0] if relative.parts else "."


def family_for_type(etype: str) -> str:
    if etype in SCMULT_ELEMENT_TYPES:
        return "SCMULT"
    if etype in FIELD_MAP_TYPES:
        return "Field map"
    if etype in WIGGLER_TYPES:
        return "Wiggler"
    if etype in ION_TYPES:
        return "Ion effects"
    if etype in ADJACENT_COLLECTIVE_TYPES:
        return "Adjacent collective"
    return "Other"


def classify_element(etype: str, params: dict[str, str]) -> list[str]:
    flags: list[str] = []
    if param_is_set(params, "SYNCH_RAD", "ISR", "RADIATION"):
        flags.append("radiation_or_isr")
    if param_is_set(params, "INPUT_FILE", "FILENAME", "FIELD_FILE", "BX_FILE", "BY_FILE", "BZ_FILE"):
        flags.append("external_field_file")
    if param_is_set(params, "FIELD_OUTPUT", "PARTICLE_OUTPUT_FILE", "OUTPUT_FILE", "CHECK_FIELDS"):
        flags.append("field_or_particle_output")
    if param_is_set(params, "DX", "DY", "DZ", "TILT", "YAW"):
        flags.append("misalignment_or_rotation")
    if param_is_set(params, "SINGLE_PERIOD_MAP"):
        flags.append("single_period_map")
    if param_is_set(params, "N_KICKS"):
        flags.append("multi_kick_map")
    if param_is_set(params, "NX_POISSON", "NY_POISSON", "X_POISSON_SPAN", "Y_POISSON_SPAN"):
        flags.append("poisson_grid")
    return flags


def classify_scmult_insert(params: dict[str, str], case_flags: CaseFlags) -> list[str]:
    flags: list[str] = []
    if param_is_set(params, "NONLINEAR"):
        flags.append("scmult_nonlinear")
    else:
        flags.append("scmult_linear")
    if param_is_set(params, "SLICE_DURATION", "N_SLICES"):
        flags.append("scmult_sliced")
    else:
        flags.append("scmult_unsliced")
    if case_flags.sdds_bunched_mode:
        flags.append("bunched_mode_input")
    if param_is_set(params, "SKIP"):
        flags.append("skip_inserted_locations")
    if "scmult_linear" in flags and "scmult_unsliced" in flags and "bunched_mode_input" not in flags:
        flags.append("current_opt_in_candidate_shape")
    return flags


def first_pass(root: Path) -> tuple[dict[str, CaseFlags], int]:
    flags: dict[str, CaseFlags] = defaultdict(CaseFlags)
    files_scanned = 0
    for path in sorted(iter_input_files(root)):
        files_scanned += 1
        case = case_name(root, path)
        flags[case].files.add(path)
        for _line, statement in iter_statements(path):
            params = parse_params(statement)
            namelist_match = NAMELIST_RE.match(statement)
            namelist = namelist_match.group("name").upper() if namelist_match else ""
            if namelist == "BUNCHED_BEAM":
                flags[case].bunched_beam += 1
            if namelist == "SDDS_BEAM" and param_is_set(params, "USE_BUNCHED_MODE"):
                flags[case].sdds_bunched_mode += 1
            if "POISSON" in statement.upper():
                flags[case].poisson_settings += 1
    return flags, files_scanned


def scan(root: Path) -> tuple[list[Occurrence], dict[str, CaseFlags], int]:
    case_flags, files_scanned = first_pass(root)
    occurrences: list[Occurrence] = []

    for path in sorted(iter_input_files(root)):
        case = case_name(root, path)
        for line, statement in iter_statements(path):
            params = parse_params(statement)
            namelist_match = NAMELIST_RE.match(statement)
            namelist = namelist_match.group("name").upper() if namelist_match else ""

            if namelist == "INSERT_SCEFFECTS":
                occurrences.append(
                    Occurrence(
                        case=case,
                        path=path,
                        line=line,
                        family="SCMULT",
                        kind="insert_sceffects",
                        name=params.get("NAME", "*").strip("\"'"),
                        flags=classify_scmult_insert(params, case_flags[case]),
                    )
                )
                continue

            if namelist == "ION_EFFECTS":
                flags = ["ion_effects_command"]
                if case_flags[case].poisson_settings:
                    flags.append("poisson_grid")
                occurrences.append(
                    Occurrence(
                        case=case,
                        path=path,
                        line=line,
                        family="Ion effects",
                        kind="ion_effects_command",
                        name=params.get("NAME", "*").strip("\"'"),
                        flags=flags,
                    )
                )
                continue

            element_match = ELEMENT_RE.match(statement)
            if not element_match:
                if "POISSON" in statement.upper():
                    occurrences.append(
                        Occurrence(
                            case=case,
                            path=path,
                            line=line,
                            family="Poisson",
                            kind="poisson_setting",
                            name="poisson",
                            flags=["poisson_grid"],
                        )
                    )
                continue

            etype = element_match.group("type").upper()
            if etype not in PHASE18_TYPES:
                continue
            occurrences.append(
                Occurrence(
                    case=case,
                    path=path,
                    line=line,
                    family=family_for_type(etype),
                    kind=etype,
                    name=(element_match.group("qname") or element_match.group("name") or "").strip(),
                    flags=classify_element(etype, params),
                )
            )

    return occurrences, case_flags, files_scanned


def summarize(occurrences: Iterable[Occurrence]) -> dict[str, CaseSummary]:
    summaries: dict[str, CaseSummary] = defaultdict(CaseSummary)
    for occurrence in occurrences:
        summary = summaries[occurrence.case]
        summary.files.add(occurrence.path)
        summary.occurrences += 1
        summary.families[occurrence.family] += 1
        summary.kinds[occurrence.kind] += 1
        for flag in occurrence.flags:
            summary.flags[flag] += 1
    return summaries


def top_counter_text(counter: Counter[str], limit: int = 3) -> str:
    if not counter:
        return ""
    return ", ".join(f"{key}:{value}" for key, value in counter.most_common(limit))


def markdown_table(headers: list[str], rows: Iterable[list[object] | tuple[object, ...]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def make_report(root: Path, files_scanned: int, occurrences: list[Occurrence], top: int) -> str:
    summaries = summarize(occurrences)
    family_counts = Counter(occurrence.family for occurrence in occurrences)
    kind_counts = Counter(occurrence.kind for occurrence in occurrences)
    flag_counts = Counter(flag for occurrence in occurrences for flag in occurrence.flags)
    generated = _datetime.datetime.now().isoformat(timespec="seconds")

    by_occurrences = sorted(summaries.items(), key=lambda item: item[1].occurrences, reverse=True)
    scmult_cases = sorted(
        ((case, summary) for case, summary in summaries.items() if summary.families["SCMULT"]),
        key=lambda item: item[1].families["SCMULT"],
        reverse=True,
    )
    field_cases = sorted(
        (
            (case, summary)
            for case, summary in summaries.items()
            if summary.families["Field map"] or summary.families["Wiggler"]
        ),
        key=lambda item: (item[1].families["Field map"] + item[1].families["Wiggler"]),
        reverse=True,
    )

    examples: dict[str, list[Occurrence]] = defaultdict(list)
    for occurrence in occurrences:
        for flag in occurrence.flags:
            if len(examples[flag]) < 4:
                examples[flag].append(occurrence)

    lines: list[str] = [
        "# Phase 18 Production Feature Profile",
        "",
        f"- Source root: `{root}`",
        f"- Generated: `{generated}`",
        f"- Files scanned: {files_scanned}",
        f"- Cases with Phase 18 features: {len(summaries)}",
        f"- Phase 18 feature occurrences: {len(occurrences)}",
        "",
        "This is a static profile of `.ele` and `.lte` files.  It identifies production-like cases for timing and comparison work; it is not a substitute for runtime profiling or SDDS output comparison.",
        "",
        "## Feature Families",
        "",
        markdown_table(["Family", "Occurrences"], family_counts.most_common(top)),
        "",
        "## Feature Types",
        "",
        markdown_table(["Type", "Occurrences"], kind_counts.most_common(top)),
        "",
        "## Feature Flags",
        "",
        markdown_table(["Flag", "Occurrences"], flag_counts.most_common(top)),
        "",
        "## Top Cases By Phase 18 Feature Count",
        "",
        markdown_table(
            ["Case", "Files", "Occurrences", "Families", "Types", "Flags"],
            [
                [
                    case,
                    len(summary.files),
                    summary.occurrences,
                    top_counter_text(summary.families),
                    top_counter_text(summary.kinds),
                    top_counter_text(summary.flags),
                ]
                for case, summary in by_occurrences[:top]
            ],
        ),
        "",
        "## SCMULT Candidate Cases",
        "",
        markdown_table(
            ["Case", "Files", "SCMULT occurrences", "Types", "Flags"],
            [
                [
                    case,
                    len(summary.files),
                    summary.families["SCMULT"],
                    top_counter_text(summary.kinds),
                    top_counter_text(summary.flags),
                ]
                for case, summary in scmult_cases[:top]
            ],
        ),
        "",
        "## Field Map And Wiggler Candidate Cases",
        "",
        markdown_table(
            ["Case", "Files", "Field maps", "Wigglers", "Types", "Flags"],
            [
                [
                    case,
                    len(summary.files),
                    summary.families["Field map"],
                    summary.families["Wiggler"],
                    top_counter_text(summary.kinds),
                    top_counter_text(summary.flags),
                ]
                for case, summary in field_cases[:top]
            ],
        ),
        "",
        "## Representative Flag Examples",
        "",
    ]

    for flag, count in flag_counts.most_common(min(top, 12)):
        lines.append(f"### {flag} ({count})")
        for occurrence in examples[flag]:
            try:
                rel_path = occurrence.path.relative_to(root)
            except ValueError:
                rel_path = occurrence.path
            lines.append(f"- `{rel_path}:{occurrence.line}` `{occurrence.name}: {occurrence.kind}`")
        lines.append("")

    ion_count = family_counts["Ion effects"]
    poisson_count = family_counts["Poisson"] + flag_counts["poisson_grid"]
    lines.extend(
        [
            "## Phase 18 Guidance",
            "",
            "- Measure SCMULT production cases before widening the current guarded single-bucket linear CUDA path.  The scanner distinguishes linear/unsliced insertion from nonlinear, sliced, and bunched-mode shapes.",
            "- Treat `UKICKMAP`/`KICKMAP`, `BMAPXY`/`BMXYZ`, and `CWIGGLER`/`WIGGLER` as separate profiling families; they have different file formats, interpolation paths, radiation options, and output hooks.",
            "- Add field-grid and particle-coordinate comparison outputs before any field-map CUDA port, especially for cases with `field_or_particle_output` or external field files.",
            f"- Ion-effects coverage found in this scan: {ion_count} direct occurrence(s).  Poisson-related static coverage found in this scan: {poisson_count} occurrence(s)/flag(s).  If this remains zero or near-zero, add a small production-like ion/Poisson wrapper before attempting CUDA work in `poisson.cc`.",
            "",
        ]
    )
    return "\n".join(lines)


def write_tsv(summaries: dict[str, CaseSummary], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write("case\tfiles\toccurrences\tfamilies\ttypes\tflags\n")
        for case, summary in sorted(summaries.items(), key=lambda item: item[0].lower()):
            stream.write(
                "\t".join(
                    [
                        case,
                        str(len(summary.files)),
                        str(summary.occurrences),
                        top_counter_text(summary.families, 8),
                        top_counter_text(summary.kinds, 8),
                        top_counter_text(summary.flags, 8),
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

    occurrences, _case_flags, files_scanned = scan(root)
    summaries = summarize(occurrences)
    report = make_report(root, files_scanned, occurrences, max(1, args.top))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report + "\n", encoding="utf-8")
    write_tsv(summaries, args.tsv_output)

    print(f"wrote {args.output}")
    print(f"wrote {args.tsv_output}")
    print(
        f"scanned {files_scanned} file(s), found {len(occurrences)} Phase 18 feature occurrence(s) in {len(summaries)} case(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
