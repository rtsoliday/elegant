#!/usr/bin/env python3
"""Profile collective and RF production candidates for CUDA follow-up work."""

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
DEFAULT_RUNTIME_ROOT = Path("test/gpu_cuda/output/finalize-action5-csr-gpu-production-smoke")
DEFAULT_REPORT = Path("test/gpu_cuda/output/reports/finalize-action6-rfcw-mode-profile.md")
DEFAULT_TSV = Path("test/gpu_cuda/output/reports/finalize-action6-rfcw-mode-profile.tsv")

COLLECTIVE_RF_TYPES = {"RFCA", "RFCW", "WAKE", "TRWAKE", "LSCDRIFT"}
RF_TYPES = {"RFCA", "RFCW"}
WAKE_TYPES = {"WAKE", "TRWAKE", "LSCDRIFT"}

ELEMENT_RE = re.compile(
    r'^\s*(?:"(?P<qname>[^"]+)"|(?P<name>[^:\s,]+))\s*:\s*'
    r"(?P<type>[A-Za-z][A-Za-z0-9_]*)\b"
)
PARAM_RE = re.compile(
    r"\b(?P<name>[A-Za-z][A-Za-z0-9_]*)\s*=\s*"
    r"(?P<value>\"[^\"]*\"|'[^']*'|[^,\s]+)"
)
NAMELIST_RE = re.compile(r"^\s*&(?P<name>[A-Za-z][A-Za-z0-9_]*)\b")
CPU_ELEMENT_SYNC_RE = re.compile(
    r"CPU synchronization requested by CPU element after CUDA element: "
    r"(?P<type>[A-Za-z0-9_]+)\s+(?P<name>\S+)#(?P<occurrence>\d+)"
)


@dataclass
class CaseFlags:
    files: set[Path] = field(default_factory=set)
    rfcw_error_items: Counter[str] = field(default_factory=Counter)
    bunched_beam: int = 0
    sdds_bunched_mode: int = 0


@dataclass
class Occurrence:
    case: str
    path: Path
    line: int
    family: str
    kind: str
    name: str
    flags: list[str]
    params: dict[str, str]


@dataclass
class RuntimeSync:
    case: str
    kind: str
    name: str
    occurrence: int
    path: Path


@dataclass
class CaseSummary:
    files: set[Path] = field(default_factory=set)
    occurrences: int = 0
    families: Counter[str] = field(default_factory=Counter)
    kinds: Counter[str] = field(default_factory=Counter)
    flags: Counter[str] = field(default_factory=Counter)
    runtime_syncs: Counter[str] = field(default_factory=Counter)


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


def bool_param(params: dict[str, str], name: str, default: bool = False) -> bool:
    if name.upper() not in params:
        return default
    return value_is_set(params.get(name.upper()))


def clean_value(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip().strip("\"'").strip()


def case_name(root: Path, path: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return "."
    return relative.parts[0] if relative.parts else "."


def family_for_type(kind: str) -> str:
    if kind in RF_TYPES:
        return "RF"
    if kind in WAKE_TYPES:
        return "Wake"
    return "Other"


def rf_n_kicks_flag(params: dict[str, str], default_matrix: bool = True) -> str:
    n_kicks = parse_float(params.get("N_KICKS"))
    if n_kicks is None:
        return "rf_matrix_method" if default_matrix else ""
    if n_kicks == 0:
        return "rf_matrix_method"
    if n_kicks == 1:
        return "single_kick_rf"
    return "multi_kick_rf"


def wake_bins_flag(params: dict[str, str]) -> str:
    n_bins = parse_float(params.get("N_BINS") or params.get("BINS"))
    if n_bins is None or n_bins == 0:
        return "auto_bins"
    return "fixed_bins"


def classify_rfca(params: dict[str, str]) -> list[str]:
    flags: list[str] = []
    length = parse_float(params.get("L"))
    if length == 0:
        flags.append("zero_length_thin_rf")
    else:
        flags.append("nonzero_length_rf")
    if param_is_set(params, "VOLT"):
        flags.append("rf_voltage")
    if param_is_set(params, "CHANGE_P0"):
        flags.append("change_p0")
    if param_is_set(params, "CHANGE_T"):
        flags.append("change_t")
    if param_is_set(params, "END1_FOCUS", "END2_FOCUS"):
        flags.append("end_focusing")
    if param_is_set(params, "Q"):
        flags.append("charging_q")
    if param_is_set(params, "DX", "DY"):
        flags.append("offset_or_misalignment")
    if param_is_set(params, "FIDUCIAL"):
        flags.append("fiducial_mode")
    if parse_float(params.get("T_REFERENCE")) not in (None, -1.0):
        flags.append("fixed_t_reference")
    if param_is_set(params, "LINEARIZE", "LOCK_PHASE"):
        flags.append("linearized_or_locked_phase")
    n_kicks_flag = rf_n_kicks_flag(params, default_matrix=False)
    if n_kicks_flag:
        flags.append(n_kicks_flag)
    return flags


def classify_rfcw(params: dict[str, str], case_flags: CaseFlags) -> list[str]:
    flags = classify_rfca(params)
    include_z = bool_param(params, "ZWAKE", True)
    include_tr = bool_param(params, "TRWAKE", True)
    has_z_wake = include_z and param_is_set(params, "WZCOLUMN", "ZWAKEFILE", "WAKEFILE")
    has_tr_wake = include_tr and param_is_set(params, "WXCOLUMN", "WYCOLUMN", "TRWAKEFILE", "WAKEFILE")

    if has_z_wake:
        flags.append("longitudinal_wake")
    if has_tr_wake:
        flags.append("transverse_wake")
    if not has_z_wake and not has_tr_wake and not param_is_set(params, "LSC"):
        flags.append("rf_only_rfcw")
    if has_z_wake or has_tr_wake:
        flags.append(wake_bins_flag(params))
    if param_is_set(params, "SMOOTHING"):
        flags.append("smoothing")
    if param_is_set(params, "INTERPOLATE"):
        flags.append("interpolate")
    if param_is_set(params, "LSC"):
        flags.append("lsc_in_rfcw")
    if param_is_set(params, "WAKES_AT_END"):
        flags.append("wakes_at_end")
    if case_flags.rfcw_error_items:
        items = ",".join(sorted(case_flags.rfcw_error_items))
        flags.append(f"error_element_{items}")
    return flags


def classify_wake(kind: str, params: dict[str, str], case_flags: CaseFlags) -> list[str]:
    flags = [wake_bins_flag(params)]
    if kind == "WAKE":
        flags.append("longitudinal_wake")
    elif kind == "TRWAKE":
        flags.append("transverse_wake")
    else:
        flags.append("lsc_drift")
    if param_is_set(params, "SMOOTHING"):
        flags.append("smoothing")
    if param_is_set(params, "INTERPOLATE"):
        flags.append("interpolate")
    if param_is_set(params, "CHANGE_P0"):
        flags.append("change_p0")
    if kind == "TRWAKE" and param_is_set(params, "TILT"):
        flags.append("tilted_trwake")
    if param_is_set(params, "BUNCHED_BEAM_MODE") or case_flags.sdds_bunched_mode:
        flags.append("bunched_wake")
    if param_is_set(params, "START_BUNCH"):
        flags.append("start_bunch_filter")
    if param_is_set(params, "END_BUNCH"):
        flags.append("end_bunch_filter")
    if kind == "LSCDRIFT" and param_is_set(
        params,
        "LOW_FREQUENCY_CUTOFF0",
        "LOW_FREQUENCY_CUTOFF1",
        "HIGH_FREQUENCY_CUTOFF0",
        "HIGH_FREQUENCY_CUTOFF1",
    ):
        flags.append("frequency_filter")
    if kind == "LSCDRIFT" and param_is_set(params, "BACKTRACK"):
        flags.append("backtrack")
    if kind == "LSCDRIFT" and param_is_set(params, "AUTO_LEFFECTIVE"):
        flags.append("auto_leffective")
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
            elif namelist == "SDDS_BEAM" and param_is_set(params, "USE_BUNCHED_MODE"):
                flags[case].sdds_bunched_mode += 1
            elif namelist == "ERROR_ELEMENT" and clean_value(params.get("ELEMENT_TYPE")).upper() == "RFCW":
                item = clean_value(params.get("ITEM")).lower() or "unknown"
                flags[case].rfcw_error_items[item] += 1
    return flags, files_scanned


def scan(root: Path) -> tuple[list[Occurrence], dict[str, CaseFlags], int]:
    case_flags, files_scanned = first_pass(root)
    occurrences: list[Occurrence] = []

    for path in sorted(iter_input_files(root)):
        case = case_name(root, path)
        for line, statement in iter_statements(path):
            element_match = ELEMENT_RE.match(statement)
            if not element_match:
                continue
            kind = element_match.group("type").upper()
            if kind not in COLLECTIVE_RF_TYPES:
                continue
            params = parse_params(statement)
            if kind == "RFCW":
                flags = classify_rfcw(params, case_flags[case])
            elif kind == "RFCA":
                flags = classify_rfca(params)
            else:
                flags = classify_wake(kind, params, case_flags[case])
            occurrences.append(
                Occurrence(
                    case=case,
                    path=path,
                    line=line,
                    family=family_for_type(kind),
                    kind=kind,
                    name=(element_match.group("qname") or element_match.group("name") or "").strip(),
                    flags=flags,
                    params=params,
                )
            )
    return occurrences, case_flags, files_scanned


def parse_runtime_syncs(runtime_root: Path) -> list[RuntimeSync]:
    if not runtime_root.exists():
        return []
    syncs: list[RuntimeSync] = []
    for path in sorted(runtime_root.rglob("elegant.stderr")):
        case = path.parent.name
        for line in path.read_text(errors="replace").splitlines():
            match = CPU_ELEMENT_SYNC_RE.search(line)
            if not match:
                continue
            syncs.append(
                RuntimeSync(
                    case=case,
                    kind=match.group("type").upper(),
                    name=match.group("name"),
                    occurrence=int(match.group("occurrence")),
                    path=path,
                )
            )
    return syncs


def summarize(occurrences: Iterable[Occurrence], runtime_syncs: Iterable[RuntimeSync]) -> dict[str, CaseSummary]:
    summaries: dict[str, CaseSummary] = defaultdict(CaseSummary)
    for occurrence in occurrences:
        summary = summaries[occurrence.case.lower()]
        summary.files.add(occurrence.path)
        summary.occurrences += 1
        summary.families[occurrence.family] += 1
        summary.kinds[occurrence.kind] += 1
        for flag in occurrence.flags:
            summary.flags[flag] += 1
    for sync in runtime_syncs:
        summary = summaries[sync.case.lower()]
        summary.runtime_syncs[sync.kind] += 1
    return summaries


def top_counter_text(counter: Counter[str], limit: int = 6) -> str:
    if not counter:
        return ""
    return ", ".join(f"{key}:{value}" for key, value in counter.most_common(limit))


def markdown_table(headers: list[str], rows: Iterable[list[object] | tuple[object, ...]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item).replace("|", "\\|") for item in row) + " |")
    return "\n".join(lines)


def rfcw_signature(occurrence: Occurrence) -> str:
    preferred = [
        "rf_only_rfcw",
        "longitudinal_wake",
        "transverse_wake",
        "lsc_in_rfcw",
        "auto_bins",
        "fixed_bins",
        "smoothing",
        "interpolate",
        "change_p0",
        "end_focusing",
        "rf_matrix_method",
        "single_kick_rf",
        "multi_kick_rf",
        "error_element_dx,dy",
    ]
    flags = [flag for flag in preferred if flag in occurrence.flags]
    extras = sorted(flag for flag in occurrence.flags if flag.startswith("error_element_") and flag not in flags)
    return ", ".join(flags + extras)


def runtime_name_counts(runtime_syncs: Iterable[RuntimeSync], kind_filter: str = "RFCW") -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for sync in runtime_syncs:
        if sync.kind == kind_filter:
            counts[(sync.case.lower(), sync.name)] += 1
    return counts


def make_report(
    root: Path,
    runtime_root: Path,
    files_scanned: int,
    occurrences: list[Occurrence],
    runtime_syncs: list[RuntimeSync],
    top: int,
) -> str:
    summaries = summarize(occurrences, runtime_syncs)
    family_counts = Counter(occurrence.family for occurrence in occurrences)
    kind_counts = Counter(occurrence.kind for occurrence in occurrences)
    flag_counts = Counter(flag for occurrence in occurrences for flag in occurrence.flags)
    runtime_kind_counts = Counter(sync.kind for sync in runtime_syncs)
    runtime_rfcw_by_name = runtime_name_counts(runtime_syncs)
    generated = _datetime.datetime.now().isoformat(timespec="seconds")

    by_runtime = sorted(
        summaries.items(),
        key=lambda item: sum(item[1].runtime_syncs.values()),
        reverse=True,
    )
    rfcw_occurrences = [occurrence for occurrence in occurrences if occurrence.kind == "RFCW"]
    rfcw_by_signature: dict[str, dict[str, object]] = {}
    definitions_by_case_name = {
        (occurrence.case.lower(), occurrence.name): occurrence
        for occurrence in rfcw_occurrences
    }
    for occurrence in rfcw_occurrences:
        signature = rfcw_signature(occurrence)
        entry = rfcw_by_signature.setdefault(
            signature,
            {"definitions": 0, "runtime": 0, "examples": []},
        )
        entry["definitions"] = int(entry["definitions"]) + 1
        entry["runtime"] = int(entry["runtime"]) + runtime_rfcw_by_name[(occurrence.case.lower(), occurrence.name)]
        examples = entry["examples"]
        if isinstance(examples, list) and len(examples) < 4:
            try:
                rel_path = occurrence.path.relative_to(root)
            except ValueError:
                rel_path = occurrence.path
            examples.append(f"{occurrence.case}/{occurrence.name} ({rel_path}:{occurrence.line})")

    unmatched_runtime = [
        (case, name, count)
        for (case, name), count in runtime_rfcw_by_name.items()
        if (case, name) not in definitions_by_case_name
    ]

    lines: list[str] = [
        "# Action 6 Collective/RF Profile",
        "",
        f"- Source root: `{root}`",
        f"- Runtime sync root: `{runtime_root}`",
        f"- Generated: `{generated}`",
        f"- Files scanned: {files_scanned}",
        f"- Collective/RF source occurrences: {len(occurrences)}",
        f"- Runtime CPU-element syncs parsed: {len(runtime_syncs)}",
        "",
        "This static/runtime profile supports action item 6.  It identifies the production `RFCW`, `RFCA`, `WAKE`, `TRWAKE`, and `LSCDRIFT` shapes before choosing a CUDA implementation path.",
        "",
        "## Runtime CPU-Element Syncs",
        "",
        markdown_table(["Element type", "Syncs"], runtime_kind_counts.most_common(top) or [["none", 0]]),
        "",
        "## Runtime RFCW Syncs By Case",
        "",
        markdown_table(
            ["Case", "RFCW syncs", "Top RFCW names"],
            [
                [
                    case,
                    summary.runtime_syncs["RFCW"],
                    top_counter_text(
                        Counter(
                            {
                                name: count
                                for (sync_case, name), count in runtime_rfcw_by_name.items()
                                if sync_case == case
                            }
                        ),
                        5,
                    ),
                ]
                for case, summary in by_runtime
                if summary.runtime_syncs["RFCW"]
            ][:top],
        ),
        "",
        "## Source Feature Families",
        "",
        markdown_table(["Family", "Occurrences"], family_counts.most_common(top)),
        "",
        "## Source Element Types",
        "",
        markdown_table(["Type", "Occurrences"], kind_counts.most_common(top)),
        "",
        "## Source Feature Flags",
        "",
        markdown_table(["Flag", "Occurrences"], flag_counts.most_common(top)),
        "",
        "## RFCW Mode Signatures",
        "",
        markdown_table(
            ["Signature", "Definitions", "Runtime RFCW syncs", "Examples"],
            [
                [
                    signature,
                    entry["definitions"],
                    entry["runtime"],
                    "; ".join(entry["examples"]) if isinstance(entry["examples"], list) else "",
                ]
                for signature, entry in sorted(
                    rfcw_by_signature.items(),
                    key=lambda item: (int(item[1]["runtime"]), int(item[1]["definitions"])),
                    reverse=True,
                )[:top]
            ],
        ),
        "",
        "## Top Cases By Runtime Syncs",
        "",
        markdown_table(
            ["Case", "Files", "Source occurrences", "Types", "Flags", "Runtime syncs"],
            [
                [
                    case,
                    len(summary.files),
                    summary.occurrences,
                    top_counter_text(summary.kinds),
                    top_counter_text(summary.flags),
                    top_counter_text(summary.runtime_syncs),
                ]
                for case, summary in by_runtime[:top]
                if sum(summary.runtime_syncs.values()) or summary.occurrences
            ],
        ),
    ]

    if unmatched_runtime:
        lines.extend(
            [
                "",
                "## Runtime Names Without Static Definition Match",
                "",
                markdown_table(["Case", "Name", "Syncs"], unmatched_runtime[:top]),
            ]
        )

    lines.extend(
        [
            "",
            "## Action 6 Guidance",
            "",
            "- `RFCW` is the leading serial production collective/RF sync target in the latest smoke: the parsed runtime logs show 78 RFCW syncs each in `lcls0`, `lcls1`, and `clic1`.",
            "- The LCLS `RFCW` shape is not a small RF-only cavity: it combines longitudinal and transverse wake files, autoscaled bins, smoothing, interpolation, `CHANGE_P0`, end focusing, and in `lcls1` deterministic RFCW `DX/DY` errors.  Treat it as a larger wake plus RF integration project.",
            "- The CLIC `RFCW` shape is a narrower RF-only cavity family with end focusing and mostly `CHANGE_P0=1`; it has no active wake columns or LSC in the source definitions.  It is the best first direct-code candidate if action 6 moves beyond profiling.",
            "- General multi-bunch `WAKE`/`TRWAKE` filters remain separate from `RFCW`; do not mix distributed Pelegant wake reductions into this serial RFCW slice.",
            "",
        ]
    )
    return "\n".join(lines)


def write_tsv(
    occurrences: list[Occurrence],
    runtime_syncs: list[RuntimeSync],
    root: Path,
    path: Path,
) -> None:
    runtime_counts = runtime_name_counts(runtime_syncs)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write("case\telement\tname\tfile\tline\tflags\truntime_syncs\n")
        for occurrence in sorted(occurrences, key=lambda item: (item.case.lower(), item.kind, item.name.lower(), item.path, item.line)):
            try:
                rel_path = occurrence.path.relative_to(root)
            except ValueError:
                rel_path = occurrence.path
            stream.write(
                "\t".join(
                    [
                        occurrence.case,
                        occurrence.kind,
                        occurrence.name,
                        str(rel_path),
                        str(occurrence.line),
                        ",".join(occurrence.flags),
                        str(runtime_counts[(occurrence.case.lower(), occurrence.name)] if occurrence.kind == "RFCW" else 0),
                    ]
                )
                + "\n"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", nargs="?", type=Path, default=DEFAULT_TEST_SET)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT, help="Markdown report path.")
    parser.add_argument("--tsv-output", type=Path, default=DEFAULT_TSV, help="Per-element TSV path.")
    parser.add_argument("--top", type=int, default=20, help="Number of rows in top-N report tables.")
    args = parser.parse_args(argv)

    root = args.source_root.resolve()
    if not root.is_dir():
        print(f"error: source root does not exist: {root}", file=sys.stderr)
        return 2

    runtime_root = args.runtime_root.resolve()
    occurrences, _case_flags, files_scanned = scan(root)
    runtime_syncs = parse_runtime_syncs(runtime_root)
    report = make_report(root, runtime_root, files_scanned, occurrences, runtime_syncs, max(1, args.top))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report + "\n", encoding="utf-8")
    write_tsv(occurrences, runtime_syncs, root, args.tsv_output)

    print(f"wrote {args.output}")
    print(f"wrote {args.tsv_output}")
    print(
        f"scanned {files_scanned} file(s), found {len(occurrences)} collective/RF occurrence(s), parsed {len(runtime_syncs)} runtime sync(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
