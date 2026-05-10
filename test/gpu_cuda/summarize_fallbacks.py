#!/usr/bin/env python3
"""Summarize CUDA CPU fallback and synchronization logs."""

from __future__ import annotations

import argparse
import datetime as _dt
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


SYNC_REASON_RE = re.compile(
    r"elegant CUDA: (?:read-only )?CPU(?: row)? synchronization requested by "
    r"(?P<reason>.+?)(?: \(\d+ rows?\))?\.?$"
)
SYNC_SUMMARY_RE = re.compile(
    r"sync\s+requests=(?P<requests>\d+)\s+"
    r"copies=(?P<copies>\d+)\s+"
    r"(?:(?:readOnly=(?P<read_only>\d+)\s+mutable=(?P<mutable>\d+)\s+))?"
    r"output=(?P<output>\d+)\s+"
    r"cpuElement=(?P<cpu_element>\d+)\s+"
    r"apertureLoss=(?P<aperture_loss>\d+)\s+"
    r"mpi=(?P<mpi>\d+)\s+"
    r"verification=(?P<verification>\d+)\s+"
    r"collective=(?P<collective>\d+)\s+"
    r"reductions=(?P<reductions>\d+)\s+"
    r"dealloc=(?P<dealloc>\d+)\s+"
    r"other=(?P<other>\d+)"
)
SHORT_ISLAND_RE = re.compile(
    r"short GPU island CPU skips=(?P<skips>\d+)\s+maxElements=(?P<max_elements>\d+)"
)

FALLBACK_MARKERS = (
    "using CPU fallback",
    "CUDA support requested in required mode",
    "CPU fallback active",
    "no usable CUDA device found",
)


@dataclass
class LogSummary:
    run_label: str
    case: str
    path: Path
    sync_reasons: Counter[str] = field(default_factory=Counter)
    fallback_messages: Counter[str] = field(default_factory=Counter)
    sync_summary: dict[str, int] = field(default_factory=dict)
    short_island_skips: int = 0
    short_island_max_elements: int = 0

    @property
    def sync_request_count(self) -> int:
        if self.sync_summary:
            return self.sync_summary.get("requests", 0)
        return sum(self.sync_reasons.values())

    @property
    def top_reason(self) -> tuple[str, int]:
        if not self.sync_reasons:
            return ("", 0)
        return self.sync_reasons.most_common(1)[0]


def parse_int(value: str | None) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except ValueError:
        return 0


def normalize_reason(reason: str) -> str:
    reason = reason.strip().rstrip(".")
    cpu_prefix = "CPU element after CUDA element:"
    if reason.startswith(cpu_prefix):
        detail = reason[len(cpu_prefix):].strip()
        element_type = detail.split(None, 1)[0] if detail else "unknown"
        return f"CPU element: {element_type}"
    return reason


def discover_stderr_files(
    output_root: Path,
    label_prefixes: list[str],
    run_dirs: list[Path],
) -> list[Path]:
    roots: list[Path] = []
    if run_dirs:
        roots.extend(run_dirs)
    elif label_prefixes:
        for prefix in label_prefixes:
            roots.extend(sorted(output_root.glob(f"{prefix}*")))
    else:
        roots.append(output_root)

    files: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("elegant.stderr")):
            resolved = path.resolve()
            if resolved not in seen:
                files.append(path)
                seen.add(resolved)
    return files


def summarize_file(path: Path, output_root: Path) -> LogSummary:
    try:
        rel = path.relative_to(output_root)
        parts = rel.parts
        run_label = parts[0] if len(parts) >= 3 else path.parent.parent.name
    except ValueError:
        run_label = path.parent.parent.name
    case = path.parent.name
    summary = LogSummary(run_label=run_label, case=case, path=path)

    for line in path.read_text(errors="replace").splitlines():
        reason_match = SYNC_REASON_RE.search(line)
        if reason_match:
            summary.sync_reasons[normalize_reason(reason_match.group("reason"))] += 1
            continue

        sync_match = SYNC_SUMMARY_RE.search(line)
        if sync_match:
            summary.sync_summary = {
                key: parse_int(value)
                for key, value in sync_match.groupdict().items()
            }
            continue

        short_match = SHORT_ISLAND_RE.search(line)
        if short_match:
            summary.short_island_skips = parse_int(short_match.group("skips"))
            summary.short_island_max_elements = parse_int(short_match.group("max_elements"))
            continue

        if any(marker in line for marker in FALLBACK_MARKERS):
            summary.fallback_messages[line.strip()] += 1

    return summary


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def format_counter(counter: Counter[str], limit: int) -> list[list[str]]:
    return [[key, str(value)] for key, value in counter.most_common(limit)]


def render_markdown(
    summaries: list[LogSummary],
    output_root: Path,
    label_prefixes: list[str],
    run_dirs: list[Path],
    max_rows: int,
) -> str:
    reason_counts: Counter[str] = Counter()
    fallback_counts: Counter[str] = Counter()
    for item in summaries:
        reason_counts.update(item.sync_reasons)
        fallback_counts.update(item.fallback_messages)

    label_text = ", ".join(label_prefixes) if label_prefixes else "none"
    if run_dirs:
        run_dir_text = ", ".join(str(path) for path in run_dirs)
    elif label_prefixes:
        run_dir_text = "discovered from label prefixes"
    else:
        run_dir_text = "all"

    lines: list[str] = [
        "# CUDA Fallback Summary",
        "",
        f"Generated: {_dt.datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Output root: `{output_root}`",
        f"Label prefixes: `{label_text}`",
        f"Run directories: `{run_dir_text}`",
        f"Scanned stderr files: {len(summaries)}",
        "",
        "## Top Synchronization Reasons",
        "",
    ]
    reason_rows = format_counter(reason_counts, max_rows)
    lines.append(markdown_table(["Reason", "Count"], reason_rows or [["none", "0"]]))
    lines.extend(["", "## Fallback Messages", ""])
    fallback_rows = format_counter(fallback_counts, max_rows)
    lines.append(markdown_table(["Message", "Count"], fallback_rows or [["none", "0"]]))
    lines.extend(["", "## Per-Case Summary", ""])

    per_case_rows: list[list[str]] = []
    for item in sorted(
        summaries,
        key=lambda entry: (entry.sync_request_count, sum(entry.fallback_messages.values())),
        reverse=True,
    )[:max_rows]:
        top_reason, top_count = item.top_reason
        fallback_text = "; ".join(
            f"{message} ({count})"
            for message, count in item.fallback_messages.most_common(3)
        )
        per_case_rows.append([
            item.run_label,
            item.case,
            str(item.sync_request_count),
            str(item.sync_summary.get("copies", 0)),
            str(item.sync_summary.get("read_only", 0)),
            str(item.sync_summary.get("mutable", 0)),
            str(item.short_island_skips),
            f"{top_reason} ({top_count})" if top_reason else "",
            fallback_text,
        ])
    lines.append(markdown_table(
        [
            "Run",
            "Case",
            "Sync requests",
            "Copies",
            "Read-only",
            "Mutable",
            "Short skips",
            "Top sync reason",
            "Fallback messages",
        ],
        per_case_rows or [["none", "", "0", "0", "0", "0", "0", "", ""]],
    ))
    lines.append("")
    return "\n".join(lines)


def write_tsv(path: Path, summaries: list[LogSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write(
            "run\tcase\tstderr\tsync_requests\tcopies\tread_only\tmutable\t"
            "short_island_skips\ttop_reason\ttop_reason_count\tfallback_messages\n"
        )
        for item in summaries:
            top_reason, top_count = item.top_reason
            fallback_text = "; ".join(
                f"{message} ({count})"
                for message, count in item.fallback_messages.most_common()
            )
            stream.write(
                "\t".join([
                    item.run_label,
                    item.case,
                    str(item.path),
                    str(item.sync_request_count),
                    str(item.sync_summary.get("copies", 0)),
                    str(item.sync_summary.get("read_only", 0)),
                    str(item.sync_summary.get("mutable", 0)),
                    str(item.short_island_skips),
                    top_reason,
                    str(top_count),
                    fallback_text,
                ])
                + "\n"
            )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
        help="Benchmark output root to scan.",
    )
    parser.add_argument(
        "--label-prefix",
        action="append",
        default=[],
        help="Run-label prefix to scan. May be repeated. Defaults to all output.",
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        type=Path,
        default=[],
        help="Specific run directory to scan. May be repeated.",
    )
    parser.add_argument("--output", type=Path, help="Markdown output path.")
    parser.add_argument("--tsv", type=Path, help="Optional TSV detail output path.")
    parser.add_argument("--max-rows", type=int, default=50)
    args = parser.parse_args(argv)

    output_root = args.output_root.resolve()
    run_dirs = [
        path if path.is_absolute() else (output_root / path)
        for path in args.run_dir
    ]
    stderr_files = discover_stderr_files(output_root, args.label_prefix, run_dirs)
    summaries = [summarize_file(path, output_root) for path in stderr_files]
    text = render_markdown(summaries, output_root, args.label_prefix, run_dirs, args.max_rows)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text)

    if args.tsv:
        write_tsv(args.tsv, summaries)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
