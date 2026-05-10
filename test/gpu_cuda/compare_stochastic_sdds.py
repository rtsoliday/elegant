#!/usr/bin/env python3
"""Compare stochastic elegant output distributions across fixed-seed runs."""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_EXTENSIONS = {
    ".acc",
    ".cen",
    ".fin",
    ".los",
    ".lost",
    ".out",
    ".sig",
    ".twi",
}

DEFAULT_COLUMNS = {
    "x",
    "xp",
    "y",
    "yp",
    "t",
    "p",
    "Cx",
    "Cxp",
    "Cy",
    "Cyp",
    "Cs",
    "Cdelta",
    "Particles",
    "pCentral",
    "Charge",
    "Sx",
    "Sxp",
    "Sy",
    "Syp",
    "Ss",
    "Sdelta",
    "St",
    "ex",
    "enx",
    "ey",
    "eny",
    "spx",
    "spy",
    "spz",
    "Cspx",
    "Cspy",
    "Cspz",
    "Sspxx",
    "Sspxy",
    "Sspxz",
    "Sspyy",
    "Sspyz",
    "Sspzz",
}

DEFAULT_HISTOGRAM_COLUMNS = {"x", "xp", "y", "yp", "t", "p", "spx", "spy", "spz"}


@dataclass(frozen=True)
class RunPair:
    reference: Path
    candidate: Path


@dataclass
class Series:
    reference: list[float] = field(default_factory=list)
    candidate: list[float] = field(default_factory=list)


@dataclass
class CheckResult:
    relpath: Path
    metric: str
    reference_n: int
    candidate_n: int
    reference_mean: float
    candidate_mean: float
    reference_sigma: float
    candidate_sigma: float
    mean_delta: float
    mean_limit: float
    sigma_delta: float
    sigma_limit: float
    ks_delta: float | None = None
    ks_limit: float | None = None
    histogram_delta: float | None = None
    histogram_limit: float | None = None
    status: str = "PASS"
    note: str = ""


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


def parse_csv_set(text: str) -> set[str]:
    return {item.strip() for item in text.split(",") if item.strip()}


def parse_pair(text: str) -> RunPair:
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"expected REF=CAND, got {text!r}")
    reference, candidate = text.split("=", 1)
    if not reference or not candidate:
        raise argparse.ArgumentTypeError(f"expected REF=CAND, got {text!r}")
    return RunPair(Path(reference), Path(candidate))


def files_by_relative_path(root: Path, extensions: set[str]) -> dict[Path, Path]:
    result: dict[Path, Path] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in extensions:
            result[path.relative_to(root)] = path
    return result


def run_sdds(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def query_columns(path: Path, sddsquery: str) -> set[str]:
    proc = run_sdds([sddsquery, "-column", str(path)])
    if proc.returncode != 0:
        return set()
    return {line.strip() for line in proc.stdout.splitlines() if line.strip()}


def row_count(path: Path, sdds2stream: str) -> int:
    proc = run_sdds([sdds2stream, str(path), "-rows=bare"])
    if proc.returncode != 0:
        return 0
    total = 0
    for token in proc.stdout.split():
        try:
            total += int(token)
        except ValueError:
            continue
    return total


def read_numeric_columns(path: Path, columns: list[str], sdds2stream: str) -> dict[str, list[float]]:
    if not columns:
        return {}
    proc = run_sdds([
        sdds2stream,
        str(path),
        f"-columns={','.join(columns)}",
        "-delimiter=\\t",
        "-noquotes",
    ])
    if proc.returncode != 0:
        return {}

    values: dict[str, list[float]] = {column: [] for column in columns}
    for raw_line in proc.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != len(columns):
            parts = line.split()
        if len(parts) != len(columns):
            continue
        for column, value in zip(columns, parts):
            try:
                number = float(value)
            except ValueError:
                values.pop(column, None)
                continue
            if math.isfinite(number) and column in values:
                values[column].append(number)
    return values


def stats(values: list[float]) -> tuple[int, float, float]:
    count = len(values)
    if count == 0:
        return 0, 0.0, 0.0
    mean = math.fsum(values) / count
    variance = math.fsum((value - mean) ** 2 for value in values) / count
    return count, mean, math.sqrt(max(variance, 0.0))


def tolerance(abs_tol: float, rel_tol: float, scale_a: float, scale_b: float) -> float:
    scale = max(abs(scale_a), abs(scale_b), 1.0)
    return abs_tol + rel_tol * scale


def ks_distance(reference: list[float], candidate: list[float]) -> float:
    if not reference or not candidate:
        return 0.0
    ref = sorted(reference)
    cand = sorted(candidate)
    i = j = 0
    max_delta = 0.0
    while i < len(ref) or j < len(cand):
        if j >= len(cand) or (i < len(ref) and ref[i] <= cand[j]):
            value = ref[i]
        else:
            value = cand[j]
        while i < len(ref) and ref[i] <= value:
            i += 1
        while j < len(cand) and cand[j] <= value:
            j += 1
        max_delta = max(max_delta, abs(i / len(ref) - j / len(cand)))
    return max_delta


def histogram_distance(reference: list[float], candidate: list[float], bins: int) -> float:
    if not reference or not candidate or bins <= 0:
        return 0.0
    lo = min(min(reference), min(candidate))
    hi = max(max(reference), max(candidate))
    if not math.isfinite(lo) or not math.isfinite(hi) or lo == hi:
        return 0.0
    width = (hi - lo) / bins
    ref_bins = [0] * bins
    cand_bins = [0] * bins
    for source, target in ((reference, ref_bins), (candidate, cand_bins)):
        for value in source:
            index = int((value - lo) / width)
            if index >= bins:
                index = bins - 1
            if index < 0:
                index = 0
            target[index] += 1
    return max(
        abs(ref_bins[i] / len(reference) - cand_bins[i] / len(candidate))
        for i in range(bins)
    )


def compare_series(relpath: Path, metric: str, series: Series, args: argparse.Namespace) -> CheckResult:
    ref_n, ref_mean, ref_sigma = stats(series.reference)
    cand_n, cand_mean, cand_sigma = stats(series.candidate)
    count_limit = tolerance(args.count_abs_tolerance, args.count_rel_tolerance, ref_n, cand_n)
    count_delta = abs(ref_n - cand_n)
    if metric == "rows":
        mean_limit = tolerance(args.count_abs_tolerance, args.count_rel_tolerance, ref_mean, cand_mean)
        sigma_limit = tolerance(args.count_abs_tolerance, args.count_rel_tolerance, ref_sigma, cand_sigma)
    else:
        mean_limit = tolerance(args.mean_abs_tolerance, args.mean_rel_tolerance, ref_mean, cand_mean)
        sigma_limit = tolerance(args.sigma_abs_tolerance, args.sigma_rel_tolerance, ref_sigma, cand_sigma)
    result = CheckResult(
        relpath=relpath,
        metric=metric,
        reference_n=ref_n,
        candidate_n=cand_n,
        reference_mean=ref_mean,
        candidate_mean=cand_mean,
        reference_sigma=ref_sigma,
        candidate_sigma=cand_sigma,
        mean_delta=abs(ref_mean - cand_mean),
        mean_limit=mean_limit,
        sigma_delta=abs(ref_sigma - cand_sigma),
        sigma_limit=sigma_limit,
    )
    if ref_n == 0 or cand_n == 0:
        result.status = "FAIL"
        result.note = "empty reference or candidate series"
        return result
    notes: list[str] = []
    if count_delta > count_limit:
        notes.append(f"finite sample count outside tolerance: delta={count_delta}, limit={count_limit:.6g}")
    if result.mean_delta > result.mean_limit:
        result.status = "FAIL"
        notes.append("mean outside tolerance")
    if result.sigma_delta > result.sigma_limit:
        result.status = "FAIL"
        notes.append("sigma outside tolerance")
    if notes:
        result.status = "FAIL"
        result.note = "; ".join(notes)

    if metric in args.histogram_columns and ref_n > 1 and cand_n > 1:
        result.ks_delta = ks_distance(series.reference, series.candidate)
        result.ks_limit = args.ks_tolerance
        result.histogram_delta = histogram_distance(series.reference, series.candidate, args.histogram_bins)
        result.histogram_limit = args.histogram_tolerance
        if result.ks_delta > result.ks_limit:
            result.status = "FAIL"
            result.note = "; ".join(filter(None, [result.note, "KS distance outside tolerance"]))
        if result.histogram_delta > result.histogram_limit:
            result.status = "FAIL"
            result.note = "; ".join(filter(None, [result.note, "histogram distance outside tolerance"]))
    return result


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_markdown(pairs: list[RunPair], results: list[CheckResult], notes: list[str]) -> str:
    failures = [result for result in results if result.status != "PASS"]
    lines = [
        "# Stochastic SDDS Distribution Comparison",
        "",
        "## Summary",
        "",
        markdown_table(
            ["Item", "Value"],
            [
                ["Run pairs", str(len(pairs))],
                ["Metrics checked", str(len(results))],
                ["Status", "failed" if failures else "passed"],
                ["Failures", str(len(failures))],
            ],
        ),
        "",
        "## Run Pairs",
        "",
    ]
    lines.append(markdown_table(
        ["Reference", "Candidate"],
        [[str(pair.reference), str(pair.candidate)] for pair in pairs],
    ))
    if notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in notes)
    lines.extend(["", "## Metrics", ""])
    rows: list[list[str]] = []
    for result in sorted(results, key=lambda item: (item.status, str(item.relpath), item.metric)):
        rows.append([
            result.status,
            str(result.relpath),
            result.metric,
            str(result.reference_n),
            str(result.candidate_n),
            format_float(result.reference_mean),
            format_float(result.candidate_mean),
            format_float(result.mean_delta),
            format_float(result.mean_limit),
            format_float(result.reference_sigma),
            format_float(result.candidate_sigma),
            format_float(result.sigma_delta),
            format_float(result.sigma_limit),
            format_float(result.ks_delta),
            format_float(result.histogram_delta),
            result.note,
        ])
    lines.append(markdown_table(
        [
            "Status",
            "File",
            "Metric",
            "Ref n",
            "Cand n",
            "Ref mean",
            "Cand mean",
            "Mean delta",
            "Mean limit",
            "Ref sigma",
            "Cand sigma",
            "Sigma delta",
            "Sigma limit",
            "KS",
            "Histogram",
            "Note",
        ],
        rows or [["FAIL", "", "", "0", "0", "", "", "", "", "", "", "", "", "", "", "no metrics checked"]],
    ))
    lines.append("")
    return "\n".join(lines)


def write_tsv(path: Path, results: list[CheckResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write(
            "status\tfile\tmetric\treference_n\tcandidate_n\treference_mean\t"
            "candidate_mean\tmean_delta\tmean_limit\treference_sigma\t"
            "candidate_sigma\tsigma_delta\tsigma_limit\tks_delta\t"
            "ks_limit\thistogram_delta\thistogram_limit\tnote\n"
        )
        for result in results:
            stream.write(
                "\t".join([
                    result.status,
                    str(result.relpath),
                    result.metric,
                    str(result.reference_n),
                    str(result.candidate_n),
                    format_float(result.reference_mean),
                    format_float(result.candidate_mean),
                    format_float(result.mean_delta),
                    format_float(result.mean_limit),
                    format_float(result.reference_sigma),
                    format_float(result.candidate_sigma),
                    format_float(result.sigma_delta),
                    format_float(result.sigma_limit),
                    format_float(result.ks_delta),
                    format_float(result.ks_limit),
                    format_float(result.histogram_delta),
                    format_float(result.histogram_limit),
                    result.note or "-",
                ])
                + "\n"
            )


def build_pairs(args: argparse.Namespace) -> list[RunPair]:
    pairs: list[RunPair] = []
    if args.reference or args.candidate:
        if not (args.reference and args.candidate):
            raise ValueError("provide both positional reference and candidate directories")
        pairs.append(RunPair(args.reference, args.candidate))
    pairs.extend(args.pair)
    if not pairs:
        raise ValueError("provide a reference/candidate pair or at least one --pair REF=CAND")
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", nargs="?", type=Path, help="reference output root")
    parser.add_argument("candidate", nargs="?", type=Path, help="candidate output root")
    parser.add_argument("--pair", action="append", type=parse_pair, default=[], help="additional REF=CAND output-root pair")
    parser.add_argument("--extensions", default=",".join(sorted(DEFAULT_EXTENSIONS)), help="comma-separated SDDS extensions")
    parser.add_argument("--columns", default="auto", help="comma-separated columns, 'auto', or 'all'")
    parser.add_argument("--histogram-columns", default=",".join(sorted(DEFAULT_HISTOGRAM_COLUMNS)), help="columns for KS and histogram checks")
    parser.add_argument("--histogram-bins", type=int, default=32)
    parser.add_argument("--mean-abs-tolerance", type=float, default=1e-12)
    parser.add_argument("--mean-rel-tolerance", type=float, default=5e-3)
    parser.add_argument("--sigma-abs-tolerance", type=float, default=1e-12)
    parser.add_argument("--sigma-rel-tolerance", type=float, default=5e-3)
    parser.add_argument("--count-abs-tolerance", type=float, default=2.0)
    parser.add_argument("--count-rel-tolerance", type=float, default=2e-2)
    parser.add_argument("--ks-tolerance", type=float, default=5e-2)
    parser.add_argument("--histogram-tolerance", type=float, default=5e-2)
    parser.add_argument("--sddsquery", help="path to sddsquery")
    parser.add_argument("--sdds2stream", help="path to sdds2stream")
    parser.add_argument("--output", type=Path, help="Markdown report output path")
    parser.add_argument("--tsv", type=Path, help="TSV metric output path")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    sddsquery = find_tool("sddsquery", args.sddsquery, repo_root)
    sdds2stream = find_tool("sdds2stream", args.sdds2stream, repo_root)
    if not sddsquery or not sdds2stream:
        print("sddsquery and sdds2stream are required; install SDDS tools or pass explicit paths", file=sys.stderr)
        return 2

    try:
        pairs = build_pairs(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    extensions = {item if item.startswith(".") else f".{item}" for item in parse_csv_set(args.extensions)}
    explicit_columns = args.columns not in {"auto", "all"}
    requested_columns = parse_csv_set(args.columns) if explicit_columns else set()
    args.histogram_columns = parse_csv_set(args.histogram_columns)

    series_by_metric: dict[tuple[Path, str], Series] = defaultdict(Series)
    notes: list[str] = []
    for pair in pairs:
        reference_files = files_by_relative_path(pair.reference, extensions)
        candidate_files = files_by_relative_path(pair.candidate, extensions)
        for relpath in sorted(reference_files.keys() - candidate_files.keys()):
            notes.append(f"missing candidate file for {relpath} in {pair.candidate}")
        for relpath in sorted(candidate_files.keys() - reference_files.keys()):
            notes.append(f"extra candidate file for {relpath} in {pair.candidate}")
        for relpath in sorted(reference_files.keys() & candidate_files.keys()):
            reference = reference_files[relpath]
            candidate = candidate_files[relpath]
            series_by_metric[(relpath, "rows")].reference.append(float(row_count(reference, sdds2stream)))
            series_by_metric[(relpath, "rows")].candidate.append(float(row_count(candidate, sdds2stream)))

            reference_columns = query_columns(reference, sddsquery)
            candidate_columns = query_columns(candidate, sddsquery)
            common_columns = reference_columns & candidate_columns
            if args.columns == "all":
                columns = sorted(common_columns)
            elif explicit_columns:
                missing = sorted(requested_columns - common_columns)
                for column in missing:
                    notes.append(f"missing requested column {column} for {relpath}")
                columns = sorted(requested_columns & common_columns)
            else:
                columns = sorted(DEFAULT_COLUMNS & common_columns)
            reference_values = read_numeric_columns(reference, columns, sdds2stream)
            candidate_values = read_numeric_columns(candidate, columns, sdds2stream)
            for column in sorted(set(reference_values) & set(candidate_values)):
                series_by_metric[(relpath, column)].reference.extend(reference_values[column])
                series_by_metric[(relpath, column)].candidate.extend(candidate_values[column])

    results = [
        compare_series(relpath, metric, series, args)
        for (relpath, metric), series in sorted(series_by_metric.items(), key=lambda item: (str(item[0][0]), item[0][1]))
    ]
    if notes:
        results.append(CheckResult(
            relpath=Path(""),
            metric="file-set",
            reference_n=0,
            candidate_n=0,
            reference_mean=0.0,
            candidate_mean=0.0,
            reference_sigma=0.0,
            candidate_sigma=0.0,
            mean_delta=0.0,
            mean_limit=0.0,
            sigma_delta=0.0,
            sigma_limit=0.0,
            status="FAIL",
            note=f"{len(notes)} file/column note(s)",
        ))

    report = render_markdown(pairs, results, notes)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report)
    if args.tsv:
        write_tsv(args.tsv, results)

    failures = [result for result in results if result.status != "PASS"]
    if failures:
        print(f"{len(failures)} stochastic distribution check(s) failed", file=sys.stderr)
        return 1
    print(f"all {len(results)} stochastic distribution check(s) passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
