#!/usr/bin/env python3
"""Generate a Markdown report from elegant CUDA benchmark outputs."""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


CUDA_SUMMARY_RE = re.compile(
    r"elements=(?P<elements>\d+)\s+"
    r"passive=(?P<passive>\d+)\s+"
    r"matrix=(?P<matrix>\d+)\s+"
    r"exactDrift=(?P<exact_drift>\d+)\s+"
    r"helpers=(?P<helpers>\d+)\s+"
    r"reductions=(?P<reductions>\d+)\s+"
    r"apertures=(?P<apertures>\d+)\s+"
    r"magnets=(?P<magnets>\d+)\s+"
    r"wakes=(?P<wakes>\d+)\s+"
    r"lsc=(?P<lsc>\d+)\s+"
    r"csr=(?P<csr>\d+)\s+"
    r"scmult=(?P<scmult>\d+)\s+"
    r"wall=(?P<wall>[0-9.]+)s\s+"
    r"kernel=(?P<kernel>[0-9.]+)s\s+"
    r"h2d=(?P<h2d>[0-9.]+)s\s+"
    r"d2h=(?P<d2h>[0-9.]+)s"
)

CUDA_SYNC_RE = re.compile(
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

CUDA_SHORT_ISLAND_RE = re.compile(
    r"short GPU island CPU skips=(?P<skips>\d+)\s+"
    r"maxElements=(?P<max_elements>\d+)"
)

SYNC_CATEGORY_LABELS = [
    ("output", "output/diagnostics"),
    ("cpu_element", "CPU-only element"),
    ("aperture_loss", "aperture/loss fallback"),
    ("mpi", "MPI scatter/gather"),
    ("verification", "verification"),
    ("collective", "collective host work"),
    ("reductions", "reductions"),
    ("dealloc", "deallocation"),
    ("other", "other"),
]


@dataclass
class ManifestRow:
    case: str
    mode: str
    mpi_ranks: str
    particles: float
    passes: float
    extra_macros: str
    status: str
    real_seconds: float
    output_dir: Path

    @classmethod
    def from_dict(cls, row: dict[str, str]) -> "ManifestRow":
        return cls(
            case=row.get("case", ""),
            mode=row.get("mode", ""),
            mpi_ranks=row.get("mpi_ranks", ""),
            particles=parse_float(row.get("particles", "")),
            passes=parse_float(row.get("passes", "")),
            extra_macros=row.get("extra_macros", ""),
            status=row.get("status", ""),
            real_seconds=parse_float(row.get("real_seconds", "")),
            output_dir=Path(row.get("output_dir", "")),
        )

    @property
    def work_units(self) -> float:
        particle_scale = self.particles
        if particle_scale <= 0:
            particle_scale = parse_macro_float(self.extra_macros, "sample_fraction", default=1.0)
        passes = self.passes if self.passes > 0 else 1.0
        return particle_scale * passes


@dataclass
class CudaLog:
    selected_device: str = ""
    fallback: str = ""
    summary: dict[str, float] = field(default_factory=dict)
    sync_summary: dict[str, float] = field(default_factory=dict)
    sync_reasons: Counter[str] = field(default_factory=Counter)
    sync_groups: Counter[str] = field(default_factory=Counter)
    short_gpu_island_cpu_skips: int = 0
    short_gpu_island_max_elements: int = 0


def parse_float(value: str | None, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def parse_macro_float(macros: str, name: str, default: float = 0.0) -> float:
    for item in macros.split(","):
        key, sep, value = item.partition("=")
        if sep and key.strip() == name:
            return parse_float(value.strip(), default)
    return default


def read_manifest(path: Path) -> list[ManifestRow]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        return [ManifestRow.from_dict(row) for row in reader]


def parse_cuda_log(output_dir: Path) -> CudaLog:
    log = CudaLog()
    path = output_dir / "elegant.stderr"
    if not path.exists():
        return log

    for line in path.read_text(errors="replace").splitlines():
        if "selected device" in line:
            match = re.search(r"selected device\s+\d+\s+([^;]+)", line)
            log.selected_device = match.group(1).strip() if match else line.strip()
        elif "no usable CUDA device found" in line or "using CPU fallback" in line:
            log.fallback = line.strip()
        elif "CPU synchronization requested by" in line:
            reason = line.split("CPU synchronization requested by", 1)[1].strip().rstrip(".")
            log.sync_reasons[reason] += 1
            log.sync_groups[normalize_sync_reason(reason)] += 1
        elif "elegant CUDA: elements=" in line:
            match = CUDA_SUMMARY_RE.search(line)
            if match:
                summary: dict[str, float] = {}
                for key, value in match.groupdict().items():
                    summary[key] = parse_float(value)
                log.summary = summary
        elif "elegant CUDA: sync requests=" in line:
            match = CUDA_SYNC_RE.search(line)
            if match:
                sync_summary: dict[str, float] = {}
                for key, value in match.groupdict().items():
                    sync_summary[key] = parse_float(value)
                log.sync_summary = sync_summary
        elif "elegant CUDA: short GPU island CPU skips=" in line:
            match = CUDA_SHORT_ISLAND_RE.search(line)
            if match:
                log.short_gpu_island_cpu_skips = int(parse_float(match.group("skips")))
                log.short_gpu_island_max_elements = int(parse_float(match.group("max_elements")))
    return log


def normalize_sync_reason(reason: str) -> str:
    cpu_prefix = "CPU element after CUDA element:"
    if reason.startswith(cpu_prefix):
        detail = reason[len(cpu_prefix):].strip()
        element_type = detail.split(None, 1)[0] if detail else "unknown"
        return f"CPU element: {element_type}"
    return reason


def run_capture(cmd: list[str], cwd: Path | None = None, timeout: int = 10) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)
    return proc.returncode, proc.stdout.strip()


def first_line(path: Path, prefix: str) -> str:
    try:
        for line in path.read_text(errors="replace").splitlines():
            if line.startswith(prefix):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


def collect_environment(
    repo_root: Path,
    cuda_arch: str,
    build_commands: list[str],
    cuda_logs: list[CudaLog],
    gpu_binary: Path | None,
    metadata: list[str],
) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    rows.append(("Report time", _dt.datetime.now().astimezone().isoformat(timespec="seconds")))

    code, git_rev = run_capture(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root)
    if code == 0:
        rows.append(("Git revision", git_rev))
    code, git_status = run_capture(["git", "status", "--short"], cwd=repo_root)
    rows.append(("Git worktree", "dirty" if git_status else "clean"))

    cpu_model = first_line(Path("/proc/cpuinfo"), "model name")
    cpu_count = os.cpu_count()
    if cpu_model:
        cpu_value = cpu_model
        if cpu_count:
            cpu_value += f" ({cpu_count} logical CPUs)"
        rows.append(("CPU", cpu_value))

    nvidia_smi = shutil.which("nvidia-smi")
    gpu_recorded = False
    if nvidia_smi:
        code, gpu_query = run_capture(
            [nvidia_smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            timeout=10,
        )
        if code == 0 and gpu_query:
            rows.append(("GPU", gpu_query.splitlines()[0]))
            gpu_recorded = True
        code, smi_text = run_capture([nvidia_smi], timeout=10)
        if code == 0:
            match = re.search(r"CUDA Version:\s*([0-9.]+)", smi_text)
            if match:
                rows.append(("Driver CUDA capability", match.group(1)))

    if not gpu_recorded:
        for log in cuda_logs:
            if log.selected_device:
                rows.append(("GPU", f"{log.selected_device} (from CUDA verbose log)"))
                break

    nvcc = os.environ.get("NVCC") or shutil.which("nvcc")
    for candidate in ("/usr/local/cuda-12.4/bin/nvcc", "/usr/local/cuda/bin/nvcc"):
        if not nvcc and Path(candidate).exists():
            nvcc = candidate
    if nvcc:
        code, nvcc_text = run_capture([nvcc, "--version"], timeout=10)
        if code == 0:
            release = re.search(r"release\s+([^,\s]+)", nvcc_text)
            rows.append(("CUDA toolkit", f"{nvcc} ({release.group(1) if release else 'version unknown'})"))

    if gpu_binary and gpu_binary.exists():
        code, ldd_text = run_capture(["ldd", str(gpu_binary)], timeout=10)
        if code == 0:
            for line in ldd_text.splitlines():
                if "libcudart" in line:
                    rows.append(("CUDA runtime", line.strip()))
                    break

    rows.append(("CUDA_ARCH", cuda_arch or os.environ.get("CUDA_ARCH", "not recorded")))
    for name in ("ELEGANT_GPU_MODE", "ELEGANT_GPU_MIN_PARTICLES", "ELEGANT_GPU_VERBOSE", "ELEGANT_GPU_VERIFY"):
        if name in os.environ:
            rows.append((name, os.environ[name]))
    for item in metadata:
        key, sep, value = item.partition("=")
        if sep:
            rows.append((key, value))
        else:
            rows.append(("Metadata", item))
    for command in build_commands:
        rows.append(("Build command", command))
    return rows


def format_float(value: float, digits: int = 3) -> str:
    if value <= 0:
        return ""
    return f"{value:.{digits}f}"


def compact_number(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value)


def format_speedup(value: float) -> str:
    if value <= 0:
        return ""
    return f"{value:.2f}x"


def summarize_speedups(speedups: list[tuple[str, float]]) -> str:
    positive = [(case, value) for case, value in speedups if value > 0]
    if not positive:
        return "not applicable"
    best_case, best_value = max(positive, key=lambda item: item[1])
    worst_case, worst_value = min(positive, key=lambda item: item[1])
    if best_case == worst_case:
        return f"{best_case} {best_value:.2f}x"
    return f"{worst_case} {worst_value:.2f}x to {best_case} {best_value:.2f}x"


def summarize_sync_categories(sync_summary: dict[str, float], limit: int = 3) -> str:
    if not sync_summary:
        return ""
    categories = [
        (label, int(sync_summary.get(key, 0)))
        for key, label in SYNC_CATEGORY_LABELS
        if sync_summary.get(key, 0) > 0
    ]
    if not categories:
        return ""
    categories.sort(key=lambda item: item[1], reverse=True)
    return ", ".join(f"{label} ({count})" for label, count in categories[:limit])


def same_workload(cpu: ManifestRow, gpu: ManifestRow) -> bool:
    return (
        cpu.mode == gpu.mode
        and cpu.mpi_ranks == gpu.mpi_ranks
        and cpu.particles == gpu.particles
        and cpu.passes == gpu.passes
        and cpu.extra_macros == gpu.extra_macros
    )


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(escape_cell(value) for value in row) + " |")
    return lines


def escape_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def compare_status(compare_text: str, compare_returncode: int | None) -> tuple[str, str]:
    if compare_returncode is None and not compare_text:
        return "not provided", ""
    if compare_returncode is not None and compare_returncode != 0:
        return f"failed ({compare_returncode})", ""
    match = re.search(r"all\s+\d+\s+common file\(s\) matched", compare_text)
    if match:
        return "passed", match.group(0)
    if compare_returncode == 0:
        return "passed", "comparison command exited successfully"
    return "provided", ""


def run_compare(args: argparse.Namespace, repo_root: Path) -> tuple[int, str]:
    cmd = [
        sys.executable,
        str(repo_root / "test/gpu_cuda/compare_sdds.py"),
        str(args.cpu_output_root),
        str(args.gpu_output_root),
        "--tolerance",
        args.tolerance,
    ]
    if args.compare_extensions:
        cmd.extend(["--extensions", args.compare_extensions])
    return run_capture(cmd, cwd=repo_root, timeout=args.compare_timeout)


def build_report(args: argparse.Namespace) -> str:
    repo_root = Path(__file__).resolve().parents[2]
    cpu_rows = read_manifest(args.cpu_manifest)
    gpu_rows = read_manifest(args.gpu_manifest)
    cpu_by_case = {row.case: row for row in cpu_rows}
    gpu_by_case = {row.case: row for row in gpu_rows}
    common_cases = sorted(cpu_by_case.keys() & gpu_by_case.keys())

    compare_text = ""
    compare_returncode: int | None = None
    if args.compare_output:
        compare_text = args.compare_output.read_text(errors="replace")
        compare_returncode = 0 if re.search(r"all\s+\d+\s+common file\(s\) matched", compare_text) else None
    if args.run_compare:
        compare_returncode, compare_text = run_compare(args, repo_root)

    status, compare_summary = compare_status(compare_text, compare_returncode)
    cuda_logs_by_case = {case: parse_cuda_log(gpu_by_case[case].output_dir) for case in common_cases}
    env_rows = collect_environment(
        repo_root,
        args.cuda_arch,
        args.build_command,
        list(cuda_logs_by_case.values()),
        args.gpu_binary,
        args.metadata,
    )

    lines: list[str] = []
    lines.append(f"# {args.title}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.extend(
        markdown_table(
            ["Item", "Value"],
            [
                ["CPU manifest", str(args.cpu_manifest)],
                ["GPU manifest", str(args.gpu_manifest)],
                ["Tolerance", args.tolerance],
                ["Comparison status", status],
                ["Comparison summary", compare_summary],
            ],
        )
    )
    lines.append("")
    lines.append("## Environment")
    lines.append("")
    lines.extend(markdown_table(["Item", "Value"], env_rows))
    lines.append("")

    workload_rows: list[list[str]] = []
    cuda_rows: list[list[str]] = []
    same_workload_speedups: list[tuple[str, float]] = []
    throughput_speedups: list[tuple[str, float]] = []
    for case in common_cases:
        cpu = cpu_by_case[case]
        gpu = gpu_by_case[case]
        cuda = cuda_logs_by_case[case]
        same = same_workload(cpu, gpu)
        same_speedup = cpu.real_seconds / gpu.real_seconds if same and gpu.real_seconds > 0 else 0.0
        cpu_throughput = cpu.work_units / cpu.real_seconds if cpu.real_seconds > 0 else 0.0
        gpu_throughput = gpu.work_units / gpu.real_seconds if gpu.real_seconds > 0 else 0.0
        throughput_speedup = gpu_throughput / cpu_throughput if cpu_throughput > 0 else 0.0
        if same_speedup > 0:
            same_workload_speedups.append((case, same_speedup))
        if throughput_speedup > 0:
            throughput_speedups.append((case, throughput_speedup))
        workload_rows.append(
            [
                case,
                cpu.mode,
                compact_number(cpu.particles),
                compact_number(cpu.passes),
                compact_number(gpu.particles),
                compact_number(gpu.passes),
                cpu.extra_macros if cpu.extra_macros == gpu.extra_macros else f"CPU: {cpu.extra_macros}; GPU: {gpu.extra_macros}",
                format_float(cpu.real_seconds),
                format_float(gpu.real_seconds),
                format_speedup(same_speedup),
                format_speedup(throughput_speedup),
                f"{cpu.status}/{gpu.status}",
            ]
        )

        summary = cuda.summary
        top_reasons = ", ".join(f"{reason} ({count})" for reason, count in cuda.sync_reasons.most_common(2))
        top_groups = ", ".join(f"{reason} ({count})" for reason, count in cuda.sync_groups.most_common(2))
        sync_request_copy = ""
        if cuda.sync_summary:
            sync_request_copy = (
                f"{int(cuda.sync_summary.get('requests', 0))}/"
                f"{int(cuda.sync_summary.get('copies', 0))}/"
                f"{int(cuda.sync_summary.get('read_only', 0))}"
            )
        cuda_rows.append(
            [
                case,
                cuda.selected_device or ("CPU fallback" if cuda.fallback else "not recorded"),
                str(int(summary.get("elements", 0))),
                str(cuda.short_gpu_island_cpu_skips) if cuda.short_gpu_island_cpu_skips else "",
                str(int(summary.get("matrix", 0))),
                str(int(summary.get("csr", 0))),
                str(int(summary.get("apertures", 0))),
                format_float(summary.get("wall", 0.0)),
                format_float(summary.get("kernel", 0.0)),
                format_float(summary.get("h2d", 0.0)),
                format_float(summary.get("d2h", 0.0)),
                sync_request_copy,
                summarize_sync_categories(cuda.sync_summary),
                top_groups,
                top_reasons,
            ]
        )

    lines.append("## Release Notes Summary")
    lines.append("")
    lines.append(f"- Compared {len(common_cases)} CPU/GPU case(s) with tolerance `{args.tolerance}`; correctness status: {status}.")
    if compare_summary:
        lines.append(f"- SDDS comparison: {compare_summary}.")
    lines.append(f"- Same-workload speedup range: {summarize_speedups(same_workload_speedups)}.")
    lines.append(f"- Throughput speedup range: {summarize_speedups(throughput_speedups)}.")
    if any(log.sync_reasons for log in cuda_logs_by_case.values()):
        total_reasons: Counter[str] = Counter()
        for log in cuda_logs_by_case.values():
            total_reasons.update(log.sync_reasons)
        top_reasons = ", ".join(f"{reason} ({count})" for reason, count in total_reasons.most_common(3))
        lines.append(f"- Most frequent CUDA synchronization reasons: {top_reasons}.")
    if any(log.sync_groups for log in cuda_logs_by_case.values()):
        total_groups: Counter[str] = Counter()
        for log in cuda_logs_by_case.values():
            total_groups.update(log.sync_groups)
        top_groups = ", ".join(f"{reason} ({count})" for reason, count in total_groups.most_common(5))
        lines.append(f"- Most frequent CUDA synchronization targets: {top_groups}.")
    if any(log.sync_summary for log in cuda_logs_by_case.values()):
        total_categories: Counter[str] = Counter()
        total_requests = 0
        total_copies = 0
        total_read_only = 0
        total_mutable = 0
        for log in cuda_logs_by_case.values():
            total_requests += int(log.sync_summary.get("requests", 0))
            total_copies += int(log.sync_summary.get("copies", 0))
            total_read_only += int(log.sync_summary.get("read_only", 0))
            total_mutable += int(log.sync_summary.get("mutable", 0))
            for key, label in SYNC_CATEGORY_LABELS:
                total_categories[label] += int(log.sync_summary.get(key, 0))
        top_categories = ", ".join(
            f"{label} ({count})" for label, count in total_categories.most_common(3) if count > 0
        )
        lines.append(
            f"- CUDA synchronization accounting: {total_requests} request(s), {total_copies} device-to-host copy event(s)"
            + (f", {total_read_only} read-only and {total_mutable} mutable request(s)" if total_read_only or total_mutable else "")
            + (f"; top categories: {top_categories}." if top_categories else ".")
        )
    total_short_island_skips = sum(log.short_gpu_island_cpu_skips for log in cuda_logs_by_case.values())
    if total_short_island_skips:
        max_settings = sorted(
            {log.short_gpu_island_max_elements for log in cuda_logs_by_case.values() if log.short_gpu_island_max_elements}
        )
        setting_text = f" with maxElements={','.join(str(value) for value in max_settings)}" if max_settings else ""
        lines.append(f"- Short GPU island avoidance kept {total_short_island_skips} simple-matrix element(s) on CPU{setting_text}.")
    lines.append("")

    lines.append("## Workload Results")
    lines.append("")
    lines.extend(
        markdown_table(
            [
                "Case",
                "Mode",
                "CPU particles",
                "CPU passes",
                "GPU particles",
                "GPU passes",
                "Extra macros",
                "CPU s",
                "GPU s",
                "Same-workload speedup",
                "Throughput speedup",
                "Status",
            ],
            workload_rows,
        )
    )
    lines.append("")
    lines.append("## CUDA Log Summary")
    lines.append("")
    lines.extend(
        markdown_table(
            [
                "Case",
                "Device",
                "Elements",
                "Short CPU skips",
                "Matrix",
                "CSR",
                "Apertures",
                "CUDA wall s",
                "Kernel s",
                "H2D s",
                "D2H s",
                "Sync req/copy/RO",
                "Top sync categories",
                "Top sync targets",
                "Top sync reasons",
            ],
            cuda_rows,
        )
    )

    if compare_text:
        lines.append("")
        lines.append("## Compare Output")
        lines.append("")
        lines.append("```text")
        lines.append(compare_text.strip())
        lines.append("```")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-manifest", required=True, type=Path)
    parser.add_argument("--gpu-manifest", required=True, type=Path)
    parser.add_argument("--cpu-output-root", type=Path)
    parser.add_argument("--gpu-output-root", type=Path)
    parser.add_argument("--compare-output", type=Path)
    parser.add_argument("--run-compare", action="store_true", help="Run compare_sdds.py on the output roots and include the result")
    parser.add_argument("--compare-extensions", help="Comma-separated extensions to pass to compare_sdds.py when --run-compare is used")
    parser.add_argument("--compare-timeout", type=int, default=120)
    parser.add_argument("--tolerance", default="1e-11")
    parser.add_argument("--title", default="elegant CUDA Benchmark Report")
    parser.add_argument("--cuda-arch", default=os.environ.get("CUDA_ARCH", "not recorded"))
    parser.add_argument("--gpu-binary", type=Path, help="gpu-elegant binary used for runtime library metadata")
    parser.add_argument("--build-command", action="append", default=[])
    parser.add_argument("--metadata", action="append", default=[], help="Additional report metadata as KEY=VALUE")
    parser.add_argument("--output", type=Path, help="Markdown output path; stdout if omitted")
    args = parser.parse_args()

    args.cpu_manifest = args.cpu_manifest.resolve()
    args.gpu_manifest = args.gpu_manifest.resolve()
    if args.cpu_output_root is None:
        args.cpu_output_root = args.cpu_manifest.parent
    if args.gpu_output_root is None:
        args.gpu_output_root = args.gpu_manifest.parent
    if args.gpu_binary is None:
        default_gpu_binary = Path(__file__).resolve().parents[2] / "bin/Linux-x86_64-gpu/gpu-elegant"
        args.gpu_binary = default_gpu_binary if default_gpu_binary.exists() else None

    report = build_report(args)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
    else:
        print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
