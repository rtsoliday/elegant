"""Tkinter benchmark cycle layered on the existing elegant regression CLI.

The command planner and artifact checks are separate from Tk so they can be
checked without running elegant or opening a display.
"""

from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
import json
import math
import os
from pathlib import Path
import queue
import re
import shlex
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import webbrowser
from typing import Any, Callable

import elegant_test_regression as regression


@dataclass(frozen=True)
class CycleInputs:
    suite: Path
    output: Path
    cases: tuple[str, ...]
    mode: str  # quick or timed
    cpu_source: str  # run or reuse
    cpu_value: str
    prior_source: str  # none, run, or reuse
    prior_value: str
    candidate: str
    targets: tuple[str, ...]
    overrides: dict[str, float]


@dataclass(frozen=True)
class Stage:
    name: str
    output: Path
    command: tuple[str, ...]
    cancel_file: Path
    comparison: bool = False


@dataclass
class PreparedCycle:
    inputs: CycleInputs
    metadata: dict[str, Any]
    artifacts: dict[str, Path]
    executable_hashes: dict[str, str]
    stages: list[Stage]
    workflow: dict[str, Any]


def parse_overrides(values: dict[str, str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, text in values.items():
        if not text.strip():
            continue
        try:
            value = float(text)
        except ValueError as exc:
            raise regression.RegressionError(
                f"{key.replace('_', ' ')} must be a number"
            ) from exc
        if not math.isfinite(value) or value < 0:
            raise regression.RegressionError(
                f"{key.replace('_', ' ')} must be finite and nonnegative"
            )
        if (
            key in {"non_target_regression_limit", "suite_regression_limit"}
            and value > 1
        ):
            raise regression.RegressionError(
                f"{key.replace('_', ' ')} must be at most 1"
            )
        result[key] = value
    return result


def _hardware_identity(hardware: dict[str, Any]) -> tuple[Any, ...]:
    gpus = hardware.get("gpus", [])
    if not isinstance(gpus, list):
        gpus = []
    return (
        hardware.get("hostname"),
        hardware.get("cpu_model"),
        tuple(
            sorted(
                (str(gpu.get("uuid", "")), str(gpu.get("driver", ""))) for gpu in gpus
            )
        ),
    )


def validate_reused_artifact(
    path: Path,
    *,
    role: str,
    metadata: dict[str, Any],
    cases: tuple[str, ...],
    timed: bool,
    current_hardware: dict[str, Any] | None,
) -> dict[str, Any]:
    manifest = regression.load_baseline(path)
    regression.validate_test_set_identity(
        metadata, manifest["test_set"], actual_label=role
    )
    names = {item["name"] for item in manifest.get("tests", [])}
    if names != set(cases):
        raise regression.RegressionError(
            f"{role} artifact case list differs from the selected cases"
        )
    if role == "CPU" and any(
        (item.get("gpu_usage") or {}).get("total_elements", 0) > 0
        for item in manifest["tests"]
    ):
        raise regression.RegressionError("CPU artifact reports GPU activity")
    if role == "prior GPU" and any(
        (item.get("gpu_usage") or {}).get("total_elements", 0) <= 0
        for item in manifest["tests"]
    ):
        raise regression.RegressionError(
            "prior GPU artifact has a case without GPU activity"
        )
    if timed:
        options = manifest.get("run_options", {})
        if (
            options.get("correctness_only")
            or options.get("jobs") != 1
            or options.get("warmup_runs") != regression.GPU_PERFORMANCE_WARMUP_RUNS
            or options.get("repetitions") != regression.GPU_PERFORMANCE_REPETITIONS
            or options.get("additional_repetitions_if_mad_exceeds_percent")
            != regression.NOISY_ADDITIONAL_REPETITIONS
            or options.get("maximum_timing_dispersion_percent")
            != regression.MAX_TIMING_DISPERSION_PERCENT
            or options.get("timing_metric") != "execution_seconds"
            or options.get("executable_arguments") != []
            or any(item.get("timing_repetitions", 0) < 5 for item in manifest["tests"])
        ):
            raise regression.RegressionError(
                f"{role} artifact lacks the serial 1 warm-up / 5 measurement protocol"
            )
        summary_path = path / "baseline-summary.json"
        try:
            hardware = json.loads(summary_path.read_text())["hardware"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise regression.RegressionError(
                f"{role} artifact lacks benchmark hardware metadata: {summary_path}"
            ) from exc
        for source in (hardware, current_hardware):
            if not source or not source.get("hostname") or not source.get("cpu_model"):
                raise regression.RegressionError(
                    f"{role} artifact lacks a complete host and CPU identity"
                )
            if not source.get("gpus") or any(
                not gpu.get("uuid") or not gpu.get("driver") for gpu in source["gpus"]
            ):
                raise regression.RegressionError(
                    f"{role} artifact lacks a complete GPU identity and driver"
                )
        if _hardware_identity(hardware) != _hardware_identity(current_hardware):
            raise regression.RegressionError(
                f"{role} artifact hardware differs from this host, CPU, GPU, or driver"
            )
    return manifest


def _baseline_stage(
    name: str, executable: Path, inputs: CycleInputs, script: Path
) -> Stage:
    output = inputs.output / name
    cancel_file = inputs.output / f".{name}.cancel"
    command = [
        sys.executable,
        str(script),
        "baseline",
        "--test-set",
        str(inputs.suite),
        "--elegant",
        str(executable),
        "--output",
        str(output),
        "--jobs",
        "1",
        "--cancel-file",
        str(cancel_file),
    ]
    if inputs.mode == "quick":
        command.append("--correctness-only")
        command.extend(["--warmup-runs", "0", "--repetitions", "1"])
    else:
        command.extend(["--warmup-runs", "1", "--repetitions", "5"])
    command.extend(inputs.cases)
    return Stage(name, output, tuple(command), cancel_file)


def _comparison_stage(
    name: str,
    baseline: Path,
    candidate: Path,
    inputs: CycleInputs,
    script: Path,
    *,
    prior_for_guards: Path | None = None,
) -> Stage:
    output = inputs.output / name
    cancel_file = inputs.output / f".{name}.cancel"
    command = [
        sys.executable,
        str(script),
        "compare-existing",
        "--baseline",
        str(baseline),
        "--candidate",
        str(candidate),
        "--output",
        str(output),
        "--cancel-file",
        str(cancel_file),
    ]
    if inputs.mode == "timed":
        if name == "cpu-comparison" and inputs.targets:
            for target in inputs.targets:
                command.extend(["--target-test", target])
            if "minimum_speedup" in inputs.overrides:
                command.extend(
                    ["--minimum-speedup", str(inputs.overrides["minimum_speedup"])]
                )
        else:
            command.append("--guard-only")
        if prior_for_guards is not None:
            command.extend(["--pre-change-gpu", str(prior_for_guards)])
        if name == "cpu-comparison":
            for key, flag in (
                ("non_target_regression_limit", "--non-target-regression-limit"),
                ("suite_regression_limit", "--suite-regression-limit"),
            ):
                if key in inputs.overrides:
                    command.extend([flag, str(inputs.overrides[key])])
    for key, flag in (
        ("gpu_noise_absolute_tolerance", "--gpu-noise-absolute-tolerance"),
        ("gpu_noise_relative_tolerance", "--gpu-noise-relative-tolerance"),
    ):
        if key in inputs.overrides:
            command.extend([flag, str(inputs.overrides[key])])
    return Stage(name, output, tuple(command), cancel_file, comparison=True)


def prepare_cycle(inputs: CycleInputs, script: Path) -> PreparedCycle:
    if inputs.mode not in {"quick", "timed"}:
        raise regression.RegressionError("choose Quick Check or Timed Guard")
    if inputs.cpu_source not in {"run", "reuse"} or inputs.prior_source not in {
        "none",
        "run",
        "reuse",
    }:
        raise regression.RegressionError("invalid CPU or prior GPU source")
    if not inputs.cases:
        raise regression.RegressionError("select at least one benchmark case")
    if inputs.targets and inputs.mode != "timed":
        raise regression.RegressionError("speedup targets require Timed Guard")
    if set(inputs.targets) - set(inputs.cases):
        raise regression.RegressionError(
            "speedup targets must be selected benchmark cases"
        )
    if "minimum_speedup" in inputs.overrides and not inputs.targets:
        raise regression.RegressionError(
            "minimum speedup requires selected target cases"
        )
    if inputs.output.exists():
        raise regression.RegressionError(
            f"new cycle directory already exists: {inputs.output}"
        )
    regression.require_commands()
    metadata = regression.test_set_metadata(inputs.suite)
    if not regression.is_gpu_performance_suite(metadata):
        raise regression.RegressionError(
            "Benchmark cycle requires a gpu-performance suite"
        )
    names, _excluded = regression.select_tests(inputs.suite, list(inputs.cases))
    if tuple(names) != inputs.cases:
        raise regression.RegressionError(
            "selected benchmark cases must be unique and sorted"
        )
    current_hardware = (
        regression.hardware_metadata() if inputs.mode == "timed" else None
    )
    artifacts: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    executables: dict[str, Path] = {}
    for role, source, value in (
        ("cpu", inputs.cpu_source, inputs.cpu_value),
        ("prior-gpu", inputs.prior_source, inputs.prior_value),
        ("candidate-gpu", "run", inputs.candidate),
    ):
        if source == "none":
            continue
        if not value.strip():
            raise regression.RegressionError(
                f"select the {role} executable or artifact"
            )
        if source == "reuse":
            path = Path(value).expanduser().resolve()
            reused = validate_reused_artifact(
                path,
                role="CPU" if role == "cpu" else "prior GPU",
                metadata=metadata,
                cases=inputs.cases,
                timed=inputs.mode == "timed",
                current_hardware=current_hardware,
            )
            artifacts[role] = path
            hashes[role] = reused.get("executable", {}).get("sha256", "")
        else:
            executable = regression.resolve_executable(value)
            executables[role] = executable
            hashes[role] = regression.sha256_file(executable)
            artifacts[role] = inputs.output / role
    if (
        inputs.prior_source == "run"
        and executables["prior-gpu"] == executables["candidate-gpu"]
    ):
        raise regression.RegressionError(
            "prior and candidate GPU runs need distinct executable paths"
        )
    stages: list[Stage] = []
    for role in ("cpu", "prior-gpu", "candidate-gpu"):
        if role in executables:
            stages.append(_baseline_stage(role, executables[role], inputs, script))
    stages.append(
        _comparison_stage(
            "cpu-comparison",
            artifacts["cpu"],
            artifacts["candidate-gpu"],
            inputs,
            script,
            prior_for_guards=(
                artifacts.get("prior-gpu") if inputs.mode == "timed" else None
            ),
        )
    )
    if "prior-gpu" in artifacts:
        stages.append(
            _comparison_stage(
                "prior-gpu-comparison",
                artifacts["prior-gpu"],
                artifacts["candidate-gpu"],
                inputs,
                script,
            )
        )
    stage_records = {
        role: {"status": "reused", "output": str(path)}
        for role, path in artifacts.items()
        if role not in executables
    }
    stage_records.update(
        {
            stage.name: {
                "status": "pending",
                "output": str(stage.output),
                "command": list(stage.command),
            }
            for stage in stages
        }
    )
    workflow = {
        "format_version": 1,
        "created_at": regression.utc_timestamp(),
        "status": "pending",
        "mode": inputs.mode,
        "log": str(inputs.output / "workflow.log"),
        "scratch": str(inputs.output / "scratch"),
        "suite": {"path": str(inputs.suite), **metadata},
        "cases": list(inputs.cases),
        "targets": list(inputs.targets),
        "overrides": inputs.overrides,
        "inputs": {
            "cpu": {
                "source": inputs.cpu_source,
                "path": inputs.cpu_value,
                "sha256": hashes.get("cpu"),
            },
            "prior_gpu": {
                "source": inputs.prior_source,
                "path": inputs.prior_value,
                "sha256": hashes.get("prior-gpu"),
            },
            "candidate_gpu": {
                "source": "run",
                "path": inputs.candidate,
                "sha256": hashes["candidate-gpu"],
            },
        },
        "stages": stage_records,
    }
    return PreparedCycle(inputs, metadata, artifacts, hashes, stages, workflow)


def read_results(cycle_root: Path) -> dict[str, Any]:
    cpu_path = cycle_root / "cpu-comparison" / "manifest.json"
    cpu = json.loads(cpu_path.read_text())
    prior_path = cycle_root / "prior-gpu-comparison" / "manifest.json"
    prior = json.loads(prior_path.read_text()) if prior_path.is_file() else None
    candidate = json.loads((cycle_root / "candidate-gpu" / "manifest.json").read_text())
    cpu_baseline_path = (cpu.get("baseline") or {}).get("path")
    prior_baseline_path = (prior.get("baseline") or {}).get("path") if prior else None
    cpu_baseline = (
        json.loads((Path(cpu_baseline_path) / "manifest.json").read_text())
        if cpu_baseline_path
        else {}
    )
    prior_baseline = (
        json.loads((Path(prior_baseline_path) / "manifest.json").read_text())
        if prior_baseline_path
        else {}
    )
    cpu_runs = {row["name"]: row for row in cpu_baseline.get("tests", [])}
    prior_runs = {row["name"]: row for row in prior_baseline.get("tests", [])}
    cpu_cases = {row["name"]: row for row in cpu["comparisons"]}
    prior_cases = {row["name"]: row for row in prior["comparisons"]} if prior else {}
    candidate_cases = {row["name"]: row for row in candidate["tests"]}
    timing = {
        row["name"]: row
        for row in (cpu.get("performance_comparison") or {}).get("tests", [])
    }
    rows = []
    for name in sorted(cpu_cases):
        run = candidate_cases[name]
        trow = timing.get(name, {})
        cpu_run = cpu_runs.get(name, {})
        prior_run = prior_runs.get(name, {})
        cpu_seconds = trow.get("baseline_seconds", cpu_run.get("execution_seconds"))
        prior_seconds = trow.get(
            "pre_change_gpu_seconds", prior_run.get("execution_seconds")
        )
        candidate_seconds = trow.get("candidate_seconds", run.get("execution_seconds"))
        speedup = trow.get("speedup")
        if speedup is None and cpu_seconds and candidate_seconds:
            speedup = cpu_seconds / candidate_seconds
        findings = []
        for label, comparison in (
            ("CPU", cpu_cases[name]),
            ("Prior GPU", prior_cases.get(name)),
        ):
            if comparison:
                for change in comparison.get("changes", []):
                    findings.append(
                        f"{label}: {change.get('path') or 'case'} — "
                        f"{change.get('detail') or change.get('status', 'changed')}"
                    )
        findings.extend(trow.get("reasons", []))
        rows.append(
            {
                "name": name,
                "findings": findings,
                "cpu_status": cpu_cases[name]["status"],
                "prior_status": prior_cases.get(name, {}).get("status", ""),
                "cpu_seconds": cpu_seconds,
                "prior_seconds": prior_seconds,
                "candidate_seconds": candidate_seconds,
                "cpu_speedup": speedup,
                "timing_status": trow.get("status", "informational"),
                "gpu_elements": (run.get("gpu_usage") or {}).get("total_elements", 0),
                "reasons": trow.get("reasons", []),
            }
        )
    return {
        "complete": bool(cpu.get("complete"))
        and (prior is None or bool(prior.get("complete"))),
        "rows": rows,
        "cpu_report": cpu_path.parent / "comparison.txt",
        "prior_report": prior_path.parent / "comparison.txt" if prior else None,
        "performance": cpu.get("performance_comparison"),
        "cpu_assessment": cpu.get("gpu_significance_assessment", {}).get("summary", {}),
    }


class CycleRunner:
    """Run one prepared cycle off the Tk thread and publish UI events."""

    def __init__(self, emit: Callable[[str, Any], None]):
        self.emit = emit
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.active_cancel_file: Path | None = None
        self.log_file: Any = None

    def stop(self) -> None:
        self.stop_event.set()
        with self.lock:
            cancel_file = self.active_cancel_file
        if cancel_file is not None and cancel_file.parent.is_dir():
            cancel_file.write_text("Stop requested from benchmark GUI.\n")

    def _write(self, prepared: PreparedCycle) -> None:
        regression.write_manifest(
            prepared.inputs.output / "workflow.json", prepared.workflow
        )

    def _log(self, line: str) -> None:
        self.emit("line", line)
        if self.log_file is not None:
            self.log_file.write(line)
            self.log_file.flush()

    @staticmethod
    def _repository_root() -> Path:
        return Path(__file__).resolve().parents[3]

    def _run_stage(self, prepared: PreparedCycle, stage: Stage) -> None:
        record = prepared.workflow["stages"][stage.name]
        record["status"] = "running"
        record["started_at"] = regression.utc_timestamp()
        self._write(prepared)
        self.emit("stage", stage.name)
        env = os.environ.copy()
        env[regression.TEMP_DIRECTORY_ENVIRONMENT] = str(
            prepared.inputs.output / "scratch"
        )
        with self.lock:
            self.active_cancel_file = stage.cancel_file
        if self.stop_event.is_set():
            self.stop()
            raise regression.CancellationRequested("cycle stopped before stage launch")
        self._log("$ " + shlex.join(stage.command) + "\n")
        process: subprocess.Popen[str] | None = None
        try:
            process = subprocess.Popen(
                list(stage.command),
                cwd=self._repository_root(),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                self._log(line)
                match = re.match(r"\[(\d+)/(\d+)\]", line)
                if match:
                    self.emit("progress", (int(match.group(1)), int(match.group(2))))
            exit_code = process.wait()
        finally:
            if process is not None and process.poll() is None:
                stage.cancel_file.write_text("Stage controller stopped unexpectedly.\n")
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            with self.lock:
                self.active_cancel_file = None
        record["exit_code"] = exit_code
        record["finished_at"] = regression.utc_timestamp()
        if self.stop_event.is_set() or stage.cancel_file.exists():
            record["status"] = "cancelled"
            self._write(prepared)
            raise regression.CancellationRequested(f"stopped during {stage.name}")
        if exit_code not in (0, 1 if stage.comparison else 0):
            record["status"] = "failed"
            self._write(prepared)
            raise regression.RegressionError(
                f"{stage.name} exited with status {exit_code}; see its output above"
            )
        manifest_path = stage.output / "manifest.json"
        if stage.comparison:
            try:
                report = json.loads(manifest_path.read_text())
            except (OSError, ValueError) as exc:
                raise regression.RegressionError(
                    f"comparison did not write a readable manifest: {manifest_path}"
                ) from exc
            if report.get("mode") != "existing-comparison":
                raise regression.RegressionError(
                    f"unexpected comparison manifest: {manifest_path}"
                )
            record["status"] = "passed" if report.get("complete") else "needs_review"
        else:
            role = stage.name
            artifact = regression.load_baseline(stage.output)
            if (
                artifact.get("executable", {}).get("sha256")
                != prepared.executable_hashes[role]
            ):
                raise regression.RegressionError(
                    f"{role} executable changed while the benchmark was running"
                )
            if role in {"cpu", "prior-gpu"}:
                regression.validate_test_set_identity(
                    prepared.metadata, artifact["test_set"], actual_label=role
                )
                if role == "cpu" and any(
                    (item.get("gpu_usage") or {}).get("total_elements", 0) > 0
                    for item in artifact["tests"]
                ):
                    raise regression.RegressionError(
                        "CPU run unexpectedly reported GPU activity"
                    )
                if role == "prior-gpu" and any(
                    (item.get("gpu_usage") or {}).get("total_elements", 0) <= 0
                    for item in artifact["tests"]
                ):
                    raise regression.RegressionError(
                        "prior GPU run lacked GPU activity"
                    )
            record["status"] = "passed"
        self._write(prepared)

    def run(self, inputs: CycleInputs, script: Path) -> None:
        prepared: PreparedCycle | None = None
        try:
            self.emit("stage", "Checking inputs and artifacts")
            prepared = prepare_cycle(inputs, script)
            if self.stop_event.is_set():
                raise regression.CancellationRequested("cycle stopped during preflight")
            inputs.output.mkdir(parents=True)
            prepared.workflow["status"] = "running"
            self._write(prepared)
            self.emit("root", inputs.output)
            with (inputs.output / "workflow.log").open("w") as log_file:
                self.log_file = log_file
                try:
                    for stage in prepared.stages:
                        if self.stop_event.is_set():
                            raise regression.CancellationRequested(
                                "cycle stopped before next stage"
                            )
                        self._run_stage(prepared, stage)
                finally:
                    self.log_file = None
            result = read_results(inputs.output)
            prepared.workflow["status"] = (
                "passed" if result["complete"] else "needs_review"
            )
            prepared.workflow["finished_at"] = regression.utc_timestamp()
            self._write(prepared)
            self.emit("result", result)
            self.emit("finished", prepared.workflow["status"])
        except regression.CancellationRequested as exc:
            if prepared is not None and inputs.output.is_dir():
                for record in prepared.workflow["stages"].values():
                    if record["status"] == "running":
                        record["status"] = "cancelled"
                prepared.workflow["status"] = "cancelled"
                prepared.workflow["finished_at"] = regression.utc_timestamp()
                self._write(prepared)
            self.emit("finished", f"cancelled: {exc}")
        except Exception as exc:
            if prepared is not None and inputs.output.is_dir():
                for record in prepared.workflow["stages"].values():
                    if record["status"] == "running":
                        record["status"] = "failed"
                prepared.workflow["status"] = "failed"
                prepared.workflow["error"] = str(exc)
                prepared.workflow["finished_at"] = regression.utc_timestamp()
                self._write(prepared)
            self.emit("finished", f"failed: {exc}")


class BenchmarkTab:
    def __init__(self, notebook: ttk.Notebook, repository: Path, script: Path):
        self.repository = repository
        self.script = script
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.runner: CycleRunner | None = None
        self.is_running = False
        self.result: dict[str, Any] | None = None
        self.cycle_root: Path | None = None

        self.frame = ttk.Frame(notebook)
        notebook.add(self.frame, text="Benchmark cycle")
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(0, weight=1)
        self.pages = ttk.Notebook(self.frame)
        self.pages.grid(row=0, column=0, sticky="nsew")
        self.setup = ttk.Frame(self.pages, padding=10)
        self.results = ttk.Frame(self.pages, padding=10)
        self.log_page = ttk.Frame(self.pages, padding=10)
        self.pages.add(self.setup, text="Setup")
        self.pages.add(self.results, text="Results")
        self.pages.add(self.log_page, text="Log")

        self.mode = tk.StringVar(value="quick")
        self.suite = tk.StringVar(value=str(repository / "src/gpu/test-set"))
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
        self.output = tk.StringVar(
            value=str(repository / "benchmark-results" / f"benchmark-{stamp}")
        )
        self.cpu_source = tk.StringVar(value="run")
        cpu_default = repository / "bin/Linux-x86_64/elegant"
        self.cpu_exe = tk.StringVar(
            value=str(cpu_default) if cpu_default.is_file() else ""
        )
        self.cpu_artifact = tk.StringVar()
        self.prior_source = tk.StringVar(value="none")
        self.prior_exe = tk.StringVar()
        self.prior_artifact = tk.StringVar()
        gpu_default = repository / "bin/Linux-x86_64-gpu/gpu-elegant"
        self.candidate = tk.StringVar(
            value=str(gpu_default) if gpu_default.is_file() else ""
        )
        self.targets_enabled = tk.BooleanVar(value=False)
        self.thresholds = {
            key: tk.StringVar()
            for key in (
                "minimum_speedup",
                "non_target_regression_limit",
                "suite_regression_limit",
                "gpu_noise_absolute_tolerance",
                "gpu_noise_relative_tolerance",
            )
        }
        self.policy_text = tk.StringVar()
        self.status = tk.StringVar(value="Ready")
        self.summary = tk.StringVar(value="No benchmark cycle has completed.")
        self._build_setup()
        self._build_results()
        self._build_log()
        self._load_cases()
        self.mode.trace_add("write", self._update_target_state)
        self.targets_enabled.trace_add("write", self._update_target_state)
        self._update_target_state()
        self.frame.after(100, self._drain_events)

    def _path_row(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
        directory: bool = False,
        new_directory: bool = False,
    ) -> None:
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky="w", padx=5, pady=3
        )
        ttk.Entry(parent, textvariable=variable).grid(
            row=row, column=1, sticky="ew", padx=5, pady=3
        )

        def browse() -> None:
            initial = variable.get() or str(self.repository)
            if new_directory:
                chosen = filedialog.asksaveasfilename(
                    parent=self.frame,
                    title="Choose a new benchmark cycle directory",
                    initialdir=str(Path(initial).parent),
                )
            elif directory:
                chosen = filedialog.askdirectory(
                    parent=self.frame,
                    title=f"Choose {label.lower()}",
                    initialdir=(
                        initial if Path(initial).is_dir() else str(Path(initial).parent)
                    ),
                )
            else:
                chosen = filedialog.askopenfilename(
                    parent=self.frame,
                    title=f"Choose {label.lower()}",
                    initialdir=str(Path(initial).parent),
                )
            if chosen:
                variable.set(chosen)
                if variable is self.suite:
                    self._load_cases()

        ttk.Button(parent, text="Browse…", command=browse).grid(
            row=row, column=2, sticky="ew", padx=5, pady=3
        )

    def _build_setup(self) -> None:
        setup = self.setup
        setup.columnconfigure(0, weight=1)
        setup.rowconfigure(3, weight=1)
        ttk.Label(
            setup,
            text="Quick Check tests outputs and GPU activity. Timed Guard adds repeated measurements.",
        ).grid(row=0, column=0, sticky="w", pady=(0, 7))
        modes = ttk.Frame(setup)
        modes.grid(row=1, column=0, sticky="ew")
        ttk.Radiobutton(
            modes, text="Quick Check (default)", variable=self.mode, value="quick"
        ).pack(side="left", padx=(0, 18))
        ttk.Radiobutton(
            modes, text="Timed Guard", variable=self.mode, value="timed"
        ).pack(side="left")

        paths = ttk.LabelFrame(setup, text="Inputs and artifact directory")
        paths.grid(row=2, column=0, sticky="ew", pady=6)
        paths.columnconfigure(1, weight=1)
        self._path_row(paths, 0, "GPU performance suite", self.suite, directory=True)
        self._path_row(paths, 1, "New cycle directory", self.output, new_directory=True)
        cpu_choices = ttk.Frame(paths)
        cpu_choices.grid(row=2, column=1, sticky="w", padx=5)
        ttk.Label(paths, text="CPU source").grid(row=2, column=0, sticky="w", padx=5)
        for label, value in (
            ("Run executable", "run"),
            ("Reuse completed artifact", "reuse"),
        ):
            ttk.Radiobutton(
                cpu_choices, text=label, value=value, variable=self.cpu_source
            ).pack(side="left", padx=(0, 12))
        self._path_row(paths, 3, "CPU executable", self.cpu_exe)
        self._path_row(paths, 4, "CPU artifact", self.cpu_artifact, directory=True)
        prior_choices = ttk.Frame(paths)
        prior_choices.grid(row=5, column=1, sticky="w", padx=5)
        ttk.Label(paths, text="Prior GPU source").grid(
            row=5, column=0, sticky="w", padx=5
        )
        for label, value in (
            ("None", "none"),
            ("Run executable", "run"),
            ("Reuse artifact", "reuse"),
        ):
            ttk.Radiobutton(
                prior_choices, text=label, value=value, variable=self.prior_source
            ).pack(side="left", padx=(0, 12))
        self._path_row(paths, 6, "Prior GPU executable", self.prior_exe)
        self._path_row(
            paths, 7, "Prior GPU artifact", self.prior_artifact, directory=True
        )
        self._path_row(paths, 8, "Candidate GPU executable", self.candidate)

        selections = ttk.LabelFrame(setup, text="Benchmark cases")
        selections.grid(row=3, column=0, sticky="nsew", pady=6)
        selections.columnconfigure(0, weight=1)
        selections.columnconfigure(1, weight=1)
        selections.rowconfigure(1, weight=1)
        ttk.Label(selections, text="Cases to run (all selected initially)").grid(
            row=0, column=0, sticky="w", padx=5
        )
        self.target_check = ttk.Checkbutton(
            selections,
            text="Assess selected speedup targets (optional)",
            variable=self.targets_enabled,
        )
        self.target_check.grid(row=0, column=1, sticky="w", padx=5)
        case_box = ttk.Frame(selections)
        case_box.grid(row=1, column=0, sticky="nsew", padx=5)
        case_box.columnconfigure(0, weight=1)
        case_box.rowconfigure(0, weight=1)
        self.case_list = tk.Listbox(
            case_box, selectmode=tk.MULTIPLE, exportselection=False, height=10
        )
        self.case_list.grid(row=0, column=0, sticky="nsew")
        case_scroll = ttk.Scrollbar(
            case_box, orient="vertical", command=self.case_list.yview
        )
        case_scroll.grid(row=0, column=1, sticky="ns")
        self.case_list.configure(yscrollcommand=case_scroll.set)
        target_box = ttk.Frame(selections)
        target_box.grid(row=1, column=1, sticky="nsew", padx=5)
        target_box.columnconfigure(0, weight=1)
        target_box.rowconfigure(0, weight=1)
        self.target_list = tk.Listbox(
            target_box, selectmode=tk.MULTIPLE, exportselection=False, height=10
        )
        self.target_list.grid(row=0, column=0, sticky="nsew")
        target_scroll = ttk.Scrollbar(
            target_box, orient="vertical", command=self.target_list.yview
        )
        target_scroll.grid(row=0, column=1, sticky="ns")
        self.target_list.configure(yscrollcommand=target_scroll.set)
        case_buttons = ttk.Frame(selections)
        case_buttons.grid(row=2, column=0, sticky="w", padx=5, pady=3)
        ttk.Button(
            case_buttons,
            text="Select all",
            command=lambda: self.case_list.selection_set(0, tk.END),
        ).pack(side="left")
        ttk.Button(
            case_buttons,
            text="Clear",
            command=lambda: self.case_list.selection_clear(0, tk.END),
        ).pack(side="left", padx=5)
        ttk.Button(case_buttons, text="Reload suite", command=self._load_cases).pack(
            side="left"
        )

        advanced = ttk.LabelFrame(
            setup, text="Advanced overrides (blank uses suite policy)"
        )
        advanced.grid(row=4, column=0, sticky="ew", pady=6)
        for column in range(5):
            advanced.columnconfigure(column, weight=1)
        labels = (
            ("minimum_speedup", "Minimum speedup"),
            ("non_target_regression_limit", "Case slowdown fraction"),
            ("suite_regression_limit", "Suite slowdown fraction"),
            ("gpu_noise_absolute_tolerance", "GPU absolute screen"),
            ("gpu_noise_relative_tolerance", "GPU relative screen"),
        )
        for column, (key, label) in enumerate(labels):
            ttk.Label(advanced, text=label).grid(
                row=0, column=column, sticky="w", padx=5
            )
            ttk.Entry(advanced, textvariable=self.thresholds[key], width=17).grid(
                row=1, column=column, sticky="ew", padx=5, pady=(0, 5)
            )
        ttk.Label(advanced, textvariable=self.policy_text).grid(
            row=2, column=0, columnspan=5, sticky="w", padx=5, pady=3
        )
        ttk.Label(
            advanced,
            text="Raw SDDS output comparison remains exact. Overrides are recorded in workflow.json and comparison manifests.",
        ).grid(row=3, column=0, columnspan=5, sticky="w", padx=5, pady=(0, 5))

        controls = ttk.Frame(setup)
        controls.grid(row=5, column=0, sticky="ew", pady=6)
        self.run_button = ttk.Button(
            controls, text="Run benchmark cycle", command=self._start
        )
        self.run_button.pack(side="left")
        self.stop_button = ttk.Button(
            controls, text="Stop safely", command=self._stop, state="disabled"
        )
        self.stop_button.pack(side="left", padx=8)
        ttk.Label(controls, textvariable=self.status).pack(side="left", padx=10)
        self.progress = ttk.Progressbar(setup, mode="determinate")
        self.progress.grid(row=6, column=0, sticky="ew", pady=(0, 8))

    def _build_results(self) -> None:
        self.results.columnconfigure(0, weight=1)
        self.results.rowconfigure(1, weight=1)
        ttk.Label(self.results, textvariable=self.summary, wraplength=1050).grid(
            row=0, column=0, sticky="ew", pady=(0, 8)
        )
        columns = (
            "case",
            "cpu",
            "prior",
            "cpu_seconds",
            "prior_seconds",
            "seconds",
            "speedup",
            "gpu",
            "timing",
        )
        self.tree = ttk.Treeview(self.results, columns=columns, show="headings")
        headings = (
            "Case",
            "CPU comparison",
            "Prior GPU comparison",
            "CPU s",
            "Prior s",
            "Candidate s",
            "CPU/GPU",
            "GPU elements",
            "Timing",
        )
        widths = (255, 128, 155, 75, 75, 90, 82, 93, 105)
        for key, label, width in zip(columns, headings, widths):
            self.tree.heading(key, text=label)
            self.tree.column(key, width=width, stretch=key == "case")
        self.tree.grid(row=1, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(self.results, orient="vertical", command=self.tree.yview)
        scroll.grid(row=1, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scroll.set)
        self.case_details = tk.StringVar(value="Select a case to inspect its findings.")
        self.tree.bind("<<TreeviewSelect>>", self._show_case_details)
        ttk.Label(
            self.results,
            textvariable=self.case_details,
            wraplength=1050,
            justify="left",
        ).grid(row=2, column=0, sticky="ew", pady=(6, 0))
        report_buttons = ttk.Frame(self.results)
        report_buttons.grid(row=3, column=0, sticky="w", pady=8)
        ttk.Button(
            report_buttons,
            text="Open cycle directory",
            command=lambda: self._open_path(self.cycle_root),
        ).pack(side="left")
        ttk.Button(
            report_buttons,
            text="Open CPU comparison report",
            command=lambda: self._open_path(
                self.result["cpu_report"] if self.result else None
            ),
        ).pack(side="left", padx=8)
        ttk.Button(
            report_buttons,
            text="Open prior GPU report",
            command=lambda: self._open_path(
                self.result["prior_report"] if self.result else None
            ),
        ).pack(side="left")

    def _build_log(self) -> None:
        self.log_page.columnconfigure(0, weight=1)
        self.log_page.rowconfigure(0, weight=1)
        self.log = tk.Text(self.log_page, wrap="none", state="disabled")
        self.log.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(self.log_page, orient="vertical", command=self.log.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scroll.set)

    @staticmethod
    def _open_path(path: Path | None) -> None:
        if path is not None and path.exists():
            webbrowser.open(path.as_uri())
        else:
            messagebox.showinfo("No report", "This report is not available yet.")

    def _update_target_state(self, *_unused: Any) -> None:
        timed = self.mode.get() == "timed"
        if not timed and self.targets_enabled.get():
            self.targets_enabled.set(False)
        self.target_check.configure(state="normal" if timed else "disabled")
        self.target_list.configure(
            state="normal" if timed and self.targets_enabled.get() else "disabled"
        )

    def _load_cases(self) -> None:
        try:
            suite = Path(self.suite.get()).expanduser().resolve()
            names, _excluded = regression.select_tests(suite, [])
            configuration = regression.local_suite_configuration(suite)
            self.case_list.delete(0, tk.END)
            self.target_list.delete(0, tk.END)
            for name in names:
                self.case_list.insert(tk.END, name)
                self.target_list.insert(tk.END, name)
            self.case_list.selection_set(0, tk.END)
            self.policy_text.set(
                f"Suite defaults: speedup {configuration.get('minimum_speedup', 0):g}x; "
                f"case slowdown {regression.DEFAULT_NON_TARGET_REGRESSION_LIMIT:.0%}; "
                f"suite slowdown {regression.DEFAULT_SUITE_REGRESSION_LIMIT:.0%}; "
                f"GPU screen abs {configuration.get('gpu_noise_absolute_tolerance', regression.DEFAULT_GPU_NOISE_ABSOLUTE_TOLERANCE):g}, "
                f"rel {configuration.get('gpu_noise_relative_tolerance', regression.DEFAULT_GPU_NOISE_RELATIVE_TOLERANCE):g}"
            )
        except (OSError, regression.RegressionError) as exc:
            self.policy_text.set(f"Unable to load suite: {exc}")
            self.case_list.delete(0, tk.END)
            self.target_list.delete(0, tk.END)

    def _inputs(self) -> CycleInputs:
        cases = tuple(
            sorted(self.case_list.get(index) for index in self.case_list.curselection())
        )
        targets = (
            tuple(
                sorted(
                    self.target_list.get(index)
                    for index in self.target_list.curselection()
                )
            )
            if self.targets_enabled.get()
            else ()
        )
        overrides = parse_overrides(
            {key: value.get() for key, value in self.thresholds.items()}
        )
        if self.targets_enabled.get() and self.mode.get() == "quick":
            raise regression.RegressionError("speedup targets require Timed Guard")
        if self.targets_enabled.get() and not targets:
            raise regression.RegressionError("select at least one speedup target")
        artifact_root = (self.repository / "benchmark-results").resolve()
        output = Path(self.output.get()).expanduser().resolve()
        if output == artifact_root or artifact_root not in output.parents:
            raise regression.RegressionError(
                f"new cycle directory must be under {artifact_root}"
            )
        return CycleInputs(
            suite=Path(self.suite.get()).expanduser().resolve(),
            output=output,
            cases=cases,
            mode=self.mode.get(),
            cpu_source=self.cpu_source.get(),
            cpu_value=(
                self.cpu_artifact.get()
                if self.cpu_source.get() == "reuse"
                else self.cpu_exe.get()
            ).strip(),
            prior_source=self.prior_source.get(),
            prior_value=(
                self.prior_artifact.get()
                if self.prior_source.get() == "reuse"
                else self.prior_exe.get() if self.prior_source.get() == "run" else ""
            ).strip(),
            candidate=self.candidate.get().strip(),
            targets=targets,
            overrides=overrides,
        )

    def _emit(self, kind: str, value: Any) -> None:
        self.events.put((kind, value))

    def _start(self) -> None:
        if self.is_running:
            return
        try:
            inputs = self._inputs()
        except regression.RegressionError as exc:
            messagebox.showerror("Invalid benchmark setup", str(exc))
            return
        self.is_running = True
        self.result = None
        self.cycle_root = None
        self.tree.delete(*self.tree.get_children())
        self.summary.set("Benchmark cycle running…")
        self.log.configure(state="normal")
        self.log.delete("1.0", tk.END)
        self.log.configure(state="disabled")
        self.run_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.progress.configure(value=0, maximum=max(1, len(inputs.cases)))
        self.status.set("Checking inputs…")
        self.runner = CycleRunner(self._emit)
        threading.Thread(
            target=self.runner.run, args=(inputs, self.script), daemon=True
        ).start()

    def _stop(self) -> None:
        if self.runner is not None:
            self.runner.stop()
            self.status.set("Stopping the active stage…")
            self.stop_button.configure(state="disabled")

    def _append_log(self, value: str) -> None:
        self.log.configure(state="normal")
        self.log.insert(tk.END, value)
        self.log.see(tk.END)
        self.log.configure(state="disabled")

    def _show_case_details(self, _event: Any = None) -> None:
        if not self.result or not self.tree.selection():
            return
        selected = self.tree.item(self.tree.selection()[0], "values")[0]
        row = next(
            (item for item in self.result["rows"] if item["name"] == selected), None
        )
        if row is None:
            return
        findings = row["findings"]
        self.case_details.set(
            f"{selected}: " + ("; ".join(findings) if findings else "No findings.")
        )

    def _show_result(self, result: dict[str, Any]) -> None:
        self.result = result
        self.tree.delete(*self.tree.get_children())
        self.case_details.set("Select a case to inspect its findings.")
        output_labels = {
            "unchanged": "Unchanged",
            "probably_expected_gpu_roundoff": "Screened roundoff",
            "changed": "Needs review",
            "": "—",
        }
        timing_labels = {
            "meets_target": "Target passed",
            "meets_guard": "Guard passed",
            "needs_review": "Needs review",
            "informational": "Info only",
        }
        for row in result["rows"]:
            speedup = row["cpu_speedup"]
            seconds = row["candidate_seconds"]
            self.tree.insert(
                "",
                tk.END,
                values=(
                    row["name"],
                    output_labels.get(row["cpu_status"], row["cpu_status"]),
                    output_labels.get(row["prior_status"], row["prior_status"]),
                    (
                        f"{row['cpu_seconds']:.4g}"
                        if isinstance(row["cpu_seconds"], (int, float))
                        else ""
                    ),
                    (
                        f"{row['prior_seconds']:.4g}"
                        if isinstance(row["prior_seconds"], (int, float))
                        else ""
                    ),
                    f"{seconds:.4g}" if isinstance(seconds, (int, float)) else "",
                    f"{speedup:.3f}x" if isinstance(speedup, (int, float)) else "—",
                    row["gpu_elements"],
                    timing_labels.get(row["timing_status"], row["timing_status"]),
                ),
            )
        assessment = result["cpu_assessment"]
        performance = result["performance"]
        details = (
            f"{len(result['rows'])} cases; "
            f"{assessment.get('potentially_significant_tests', 0)} significant CPU/GPU differences; "
            f"{assessment.get('probably_expected_gpu_roundoff_tests', 0)} cases with screened GPU roundoff"
        )
        if performance:
            gate_label = {
                "guard_only": "slowdown guards",
                "targets": "selected speedup targets and guards",
                "informational": "information only",
            }.get(performance.get("gate_mode"), "suite policy")
            details += f"; timing: {gate_label}"
            details += (
                f"; {performance['summary']['tests_needing_review']} cases need review"
            )
        if performance and performance.get("suite_reasons"):
            details += "; " + "; ".join(performance["suite_reasons"])
        self.summary.set(
            ("PASS — " if result["complete"] else "NEEDS REVIEW — ") + details
        )
        self.pages.select(self.results)

    def _drain_events(self) -> None:
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "stage":
                    self.status.set(str(value))
                    self.progress.configure(value=0)
                elif kind == "root":
                    self.cycle_root = value
                elif kind == "line":
                    self._append_log(str(value))
                elif kind == "progress":
                    done, total = value
                    self.progress.configure(maximum=total, value=done)
                elif kind == "result":
                    self._show_result(value)
                elif kind == "finished":
                    self.is_running = False
                    self.run_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                    self.status.set(str(value))
                    self._append_log(f"\n[workflow {value}]\n")
                    if str(value).startswith(("failed", "cancelled")):
                        self.summary.set(str(value))
                    if Path(self.output.get()).expanduser().exists():
                        stamp = dt.datetime.now(dt.timezone.utc).strftime(
                            "%Y%m%d-%H%M%S"
                        )
                        base = (
                            self.repository / "benchmark-results" / f"benchmark-{stamp}"
                        )
                        suggestion = base
                        suffix = 2
                        while suggestion.exists():
                            suggestion = Path(f"{base}-{suffix}")
                            suffix += 1
                        self.output.set(str(suggestion))
        except queue.Empty:
            pass
        self.frame.after(100, self._drain_events)


def create_benchmark_tab(
    notebook: ttk.Notebook, repository: Path, script: Path
) -> BenchmarkTab:
    return BenchmarkTab(notebook, repository, script)
