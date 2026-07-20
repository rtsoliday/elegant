#!/usr/bin/env python3
"""Create and compare elegant regression-test output baselines.

The test set may be a clean Subversion working copy or a fingerprinted local
suite. Tests are copied to disposable directories before they are run, so the
source tests are not modified. Completed baseline artifact directories can also
be compared without rerunning either executable.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime as dt
import fnmatch
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import queue
import re
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Iterable


FORMAT_VERSION = 1
TEMP_DIRECTORY_ENVIRONMENT = "ELEGANT_REGRESSION_TMPDIR"
MAX_PARALLEL_TESTS = 8
DEFAULT_JOBS = min(MAX_PARALLEL_TESTS, os.cpu_count() or 1)
DEFAULT_TEST_SECONDS = 15 * 60
MAX_TEST_SECONDS = 60 * 60
DEFAULT_GPU_NOISE_ABSOLUTE_TOLERANCE = 1e-15
DEFAULT_GPU_NOISE_RELATIVE_TOLERANCE = 1e-12
GPU_ASSESSMENT_VERSION = 1
PERFORMANCE_ASSESSMENT_VERSION = 1
GPU_PERFORMANCE_SUITE_TYPE = "gpu-performance"
DEFAULT_MINIMUM_GPU_SPEEDUP = 1.05
DEFAULT_WARMUP_RUNS = 0
DEFAULT_REPETITIONS = 1
GPU_PERFORMANCE_WARMUP_RUNS = 1
GPU_PERFORMANCE_REPETITIONS = 5
NOISY_ADDITIONAL_REPETITIONS = 5
MAX_REPETITIONS = 20
MAX_WARMUP_RUNS = 10
GPU_LOG_DETAIL_LIMIT = 20
MAX_TIMING_DISPERSION_PERCENT = 5.0
DEFAULT_NON_TARGET_REGRESSION_LIMIT = 0.05
DEFAULT_SUITE_REGRESSION_LIMIT = 0.02
DEFAULT_EXCLUDED_TESTS = {
    "brat4": "has an impractically long runtime with its BRAT maps present",
    "cwigglerMoments3": "has a highly variable runtime under concurrent load",
    "daOpt1": "has an impractically long dynamic-aperture optimization",
    "lsc3": "has an impractically long run while tracking about two million particles",
    "offMomentum3": "has an impractically long multi-step tracking run",
    "radDamping4": "has an impractically long multi-step tracking run",
    "sddsBeam2": "required reference/run.out1 input is absent from SVN",
    "touschek2": "elegant terminates with SIGSEGV in touschek_scatter",
}
REFERENCE_DIRECTORIES = {"reference", "preference", "greference"}
DEFAULT_IGNORED_PARAMETERS = (
    "SVNVersion",
    "CPU",
    "ElapsedCoreTime",
    "ElapsedTime",
    "MEM",
    "MemoryUsage",
)
DEFAULT_IGNORED_COLUMNS = (
    "CPU",
    "ElapsedCoreTime",
    "ElapsedTime",
    "MEM",
    "MemoryUsage",
)
REQUIRED_COMMANDS = (
    "svn",
    "svnversion",
    "sddscheck",
    "sddsquery",
    "sddsdiff",
)
DEPENDENCY_PATTERN = re.compile(rb"\.\./([A-Za-z0-9_.-]+)")
CSH_SHEBANG_PATTERN = re.compile(r"^#!\s*\S*/(?:t?csh)(?:\s|$)")
SCRIPT_TEMP_PATTERN = re.compile(r"^tmp\d+\.\d+\.(?:in|out)$")
LOCAL_FINGERPRINT_EXCLUDED_FILES = {"README.md"}
SDDS_FLOAT_TYPES = {"float", "double", "longdouble"}
SDDS_VALUE_OPTIONS = {
    "column": "-columns",
    "parameter": "-parameters",
    "array": "-arrays",
}


class RegressionError(RuntimeError):
    """An error that should be reported without a Python traceback."""


def configure_temp_directory() -> Path:
    """Keep large disposable test trees off the root filesystem when possible."""
    repository = Path(__file__).resolve().parents[3]
    requested = os.environ.get(TEMP_DIRECTORY_ENVIRONMENT)
    if requested:
        temporary_root = Path(requested).expanduser()
    else:
        gpu_testing = repository / "GPU-Testing"
        temporary_root = (
            gpu_testing / "tmp" if gpu_testing.is_dir() else Path(tempfile.gettempdir())
        )

    try:
        temporary_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RegressionError(
            f"unable to create regression temporary directory: {temporary_root}"
        ) from exc
    if not temporary_root.is_dir() or not os.access(temporary_root, os.W_OK):
        raise RegressionError(
            f"regression temporary directory is not writable: {temporary_root}"
        )

    temporary_root = temporary_root.resolve()
    tempfile.tempdir = str(temporary_root)
    os.environ["TMPDIR"] = str(temporary_root)
    return temporary_root


@dataclasses.dataclass(frozen=True)
class FileStamp:
    size: int
    mtime_ns: int
    mode: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_checked(command: list[str], *, cwd: Path | None = None) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RegressionError(f"required command not found: {command[0]}") from exc
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        raise RegressionError(
            f"command failed ({completed.returncode}): {shlex.join(command)}"
            + (f"\n{detail}" if detail else "")
        )
    return completed.stdout.strip()


def require_commands(commands: Iterable[str] = REQUIRED_COMMANDS) -> None:
    missing = [command for command in commands if not shutil.which(command)]
    if missing:
        raise RegressionError(
            "required commands are not on PATH: " + ", ".join(missing)
        )


def resolve_executable(value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.parent != Path(".") or os.sep in value:
        candidate = candidate.resolve()
        if not candidate.is_file():
            raise RegressionError(f"elegant executable not found: {candidate}")
    else:
        found = shutil.which(value)
        if not found:
            raise RegressionError(f"elegant executable not found on PATH: {value}")
        candidate = Path(found).resolve()
    if not os.access(candidate, os.X_OK):
        raise RegressionError(f"elegant executable is not executable: {candidate}")
    return candidate


def svn_metadata(test_set: Path) -> dict[str, str]:
    if not test_set.is_dir():
        raise RegressionError(f"elegantTestSet directory not found: {test_set}")
    url = run_checked(["svn", "info", "--show-item", "url", str(test_set)])
    working_copy_state = run_checked(["svnversion", str(test_set)])
    status = run_checked(["svn", "status", "--ignore-externals", str(test_set)])
    if status:
        preview = "\n".join(status.splitlines()[:20])
        raise RegressionError(
            "elegantTestSet must be a clean SVN working copy before a run:\n" + preview
        )
    revision_match = re.fullmatch(r"(\d+)(P?)", working_copy_state)
    if not revision_match:
        raise RegressionError(
            "elegantTestSet must be at one uniform SVN revision; "
            f"svnversion reported {working_copy_state!r}"
        )
    return {
        "url": url,
        "revision": revision_match.group(1),
        "working_copy_state": working_copy_state,
    }


def local_test_set_fingerprint(test_set: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    file_count = 0
    for path in sorted(test_set.rglob("*")):
        relative = path.relative_to(test_set)
        if any(part in {".git", ".svn", "__pycache__"} for part in relative.parts):
            continue
        if relative.as_posix() in LOCAL_FINGERPRINT_EXCLUDED_FILES:
            continue
        if path.is_symlink():
            data = os.fsencode(os.readlink(path))
            kind = b"link"
        elif path.is_file():
            data = bytes.fromhex(sha256_file(path))
            kind = b"file"
        else:
            continue
        encoded_relative = relative.as_posix().encode("utf-8", errors="surrogateescape")
        digest.update(kind + b"\0" + encoded_relative + b"\0" + data + b"\0")
        file_count += 1
    return digest.hexdigest(), file_count


def local_suite_configuration(test_set: Path) -> dict[str, Any]:
    configuration_path = test_set / "suite.json"
    if not configuration_path.is_file():
        return {}
    try:
        configuration = json.loads(configuration_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RegressionError(
            f"unable to read local test-suite configuration: {configuration_path}"
        ) from exc
    if not isinstance(configuration, dict):
        raise RegressionError(
            f"local suite configuration must be an object: {configuration_path}"
        )
    if configuration.get("suite_type") == GPU_PERFORMANCE_SUITE_TYPE:
        configuration.setdefault("minimum_speedup", DEFAULT_MINIMUM_GPU_SPEEDUP)
    minimum_speedup = configuration.get("minimum_speedup", 0)
    if (
        isinstance(minimum_speedup, bool)
        or not isinstance(minimum_speedup, (int, float))
        or minimum_speedup < 0
    ):
        raise RegressionError(
            f"suite minimum_speedup must be nonnegative: {configuration_path}"
        )
    recommended_jobs = configuration.get("recommended_jobs")
    if recommended_jobs is not None and (
        not isinstance(recommended_jobs, int)
        or not 1 <= recommended_jobs <= MAX_PARALLEL_TESTS
    ):
        raise RegressionError(
            f"suite recommended_jobs must be 1 through {MAX_PARALLEL_TESTS}: "
            f"{configuration_path}"
        )
    require_gpu_activity = configuration.get("require_gpu_activity", False)
    if not isinstance(require_gpu_activity, bool):
        raise RegressionError(
            f"suite require_gpu_activity must be true or false: {configuration_path}"
        )
    gpu_environment = configuration.get("gpu_environment", {})
    if not isinstance(gpu_environment, dict) or any(
        not isinstance(key, str)
        or not key.startswith("ELEGANT_GPU_")
        or not isinstance(value, str)
        for key, value in gpu_environment.items()
    ):
        raise RegressionError(
            "suite gpu_environment must map ELEGANT_GPU_* names to string values: "
            f"{configuration_path}"
        )
    for setting in (
        "gpu_noise_absolute_tolerance",
        "gpu_noise_relative_tolerance",
    ):
        value = configuration.get(setting, 0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise RegressionError(
                f"suite {setting} must be nonnegative: {configuration_path}"
            )
    return configuration


def test_set_metadata(test_set: Path) -> dict[str, Any]:
    if not test_set.is_dir():
        raise RegressionError(f"test-set directory not found: {test_set}")
    if (test_set / ".svn").is_dir():
        return {"kind": "svn", **svn_metadata(test_set)}
    fingerprint, file_count = local_test_set_fingerprint(test_set)
    return {
        "kind": "local",
        "fingerprint": fingerprint,
        "file_count": file_count,
        "fingerprint_excluded_files": sorted(LOCAL_FINGERPRINT_EXCLUDED_FILES),
        "suite": local_suite_configuration(test_set),
    }


def test_set_kind(metadata: dict[str, Any]) -> str:
    if metadata.get("kind") in {"svn", "local"}:
        return metadata["kind"]
    return "svn" if metadata.get("url") else "local"


def validate_test_set_identity(
    expected: dict[str, Any], actual: dict[str, Any], *, actual_label: str
) -> None:
    expected_kind = test_set_kind(expected)
    actual_kind = test_set_kind(actual)
    if expected_kind != actual_kind:
        raise RegressionError(
            f"test-set source type changed: baseline={expected_kind}, "
            f"{actual_label}={actual_kind}"
        )
    if expected_kind == "svn":
        if expected.get("url") != actual.get("url") or expected.get(
            "revision"
        ) != actual.get("revision"):
            raise RegressionError(
                "runs must use the same SVN test-set URL and revision\n"
                f"baseline: {expected.get('url')} at {expected.get('revision')}\n"
                f"{actual_label}: {actual.get('url')} at {actual.get('revision')}"
            )
        return
    if expected.get("fingerprint") != actual.get("fingerprint"):
        raise RegressionError(
            "runs must use identical local test-set content\n"
            f"baseline fingerprint: {expected.get('fingerprint')}\n"
            f"{actual_label} fingerprint: {actual.get('fingerprint')}"
        )


def test_set_description(metadata: dict[str, Any]) -> str:
    if test_set_kind(metadata) == "svn":
        return f"SVN {metadata.get('revision', 'unknown')}"
    suite = metadata.get("suite", {})
    name = suite.get("name") if isinstance(suite, dict) else None
    return f"local suite {name!r}" if name else "local test set"


def validate_suite_run_settings(metadata: dict[str, Any], jobs: int) -> None:
    suite = metadata.get("suite", {})
    if not isinstance(suite, dict):
        return
    recommended_jobs = suite.get("recommended_jobs")
    if recommended_jobs and jobs != recommended_jobs:
        raise RegressionError(
            f"this suite requires --jobs {recommended_jobs} for comparable timing; "
            f"got --jobs {jobs}"
        )


def suite_environment(metadata: dict[str, Any]) -> dict[str, str]:
    suite = metadata.get("suite", {})
    if not isinstance(suite, dict):
        return {}
    environment = suite.get("gpu_environment", {})
    return dict(environment) if isinstance(environment, dict) else {}


def is_gpu_performance_suite(metadata: dict[str, Any]) -> bool:
    suite = metadata.get("suite", {})
    return (
        isinstance(suite, dict)
        and suite.get("suite_type") == GPU_PERFORMANCE_SUITE_TYPE
    )


def timing_run_options(
    metadata: dict[str, Any],
    requested_warmups: int | None,
    requested_repetitions: int | None,
) -> tuple[int, int, bool]:
    performance_suite = is_gpu_performance_suite(metadata)
    warmups = (
        requested_warmups
        if requested_warmups is not None
        else (GPU_PERFORMANCE_WARMUP_RUNS if performance_suite else DEFAULT_WARMUP_RUNS)
    )
    repetitions = (
        requested_repetitions
        if requested_repetitions is not None
        else (GPU_PERFORMANCE_REPETITIONS if performance_suite else DEFAULT_REPETITIONS)
    )
    return warmups, repetitions, performance_suite and repetitions >= 5


def hardware_metadata() -> dict[str, Any]:
    cpu_model = ""
    try:
        for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass

    gpu_query: list[dict[str, str]] = []
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            completed = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=name,uuid,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=15,
                check=False,
            )
            if completed.returncode == 0:
                for line in completed.stdout.splitlines():
                    fields = [field.strip() for field in line.split(",", 2)]
                    if len(fields) == 3:
                        gpu_query.append(
                            {"name": fields[0], "uuid": fields[1], "driver": fields[2]}
                        )
        except (OSError, subprocess.TimeoutExpired):
            pass

    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "cpu_model": cpu_model,
        "logical_cpu_count": os.cpu_count(),
        "gpus": gpu_query,
    }


def executable_metadata(executable: Path) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [str(executable)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=15,
            check=False,
        )
        version_output = completed.stdout[:20000]
    except subprocess.TimeoutExpired as exc:
        version_output = (exc.stdout or "")[:20000]
    return {
        "path": str(executable),
        "sha256": sha256_file(executable),
        "version_output": version_output,
    }


def is_runner_directory(path: Path) -> bool:
    if (path / "runScript").is_file() or (path / "runTemplate").is_file():
        return True
    return any(path.glob("run*.ele"))


def select_tests(
    test_set: Path, requested: list[str], *, include_excluded: bool = False
) -> tuple[list[str], list[str]]:
    if requested:
        names = sorted(set(requested))
        for name in names:
            if Path(name).name != name or name in {".", ".."}:
                raise RegressionError(f"invalid test name: {name!r}")
            path = test_set / name
            if not path.is_dir():
                raise RegressionError(f"test directory not found: {path}")
            if not is_runner_directory(path):
                raise RegressionError(f"no runnable input found in test: {name}")
        # Explicitly naming a test is an intentional override for diagnosing
        # or rechecking a test after its underlying problem is repaired.
        return names, []

    runnable = [
        path.name
        for path in test_set.iterdir()
        if path.is_dir() and is_runner_directory(path)
    ]
    reference_tests = [
        name for name in runnable if (test_set / name / "reference").is_dir()
    ]
    # Preserve elegantTestSet's reference-backed discovery. Standalone local
    # suites such as src/gpu/test-set intentionally have no reference directories.
    names = reference_tests or runnable
    if not names:
        raise RegressionError(f"no runnable test directories found in {test_set}")
    excluded = (
        []
        if include_excluded
        else sorted(set(names).intersection(DEFAULT_EXCLUDED_TESTS))
    )
    selected = sorted(set(names).difference(excluded))
    if not selected:
        raise RegressionError("all discovered tests are excluded by default")
    return selected, excluded


def excluded_runtime_path(relative: str) -> bool:
    path = Path(relative)
    if path.parts and path.parts[0] in REFERENCE_DIRECTORIES:
        return True
    name = path.name
    return (
        name in {"started", "done"}
        or name == "compare.done"
        or name.endswith(".done")
        or name.endswith(".time")
        or name.startswith("compare.")
        or name.startswith("comp.")
        or bool(SCRIPT_TEMP_PATTERN.fullmatch(name))
    )


def scan_regular_files(root: Path) -> dict[str, FileStamp]:
    result: dict[str, FileStamp] = {}
    for directory, subdirectories, filenames in os.walk(root, followlinks=False):
        subdirectories[:] = [
            name
            for name in subdirectories
            if name not in REFERENCE_DIRECTORIES
            and not (Path(directory) / name).is_symlink()
        ]
        for filename in filenames:
            path = Path(directory) / filename
            if path.is_symlink() or not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            if excluded_runtime_path(relative):
                continue
            stat = path.stat()
            result[relative] = FileStamp(
                size=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
                mode=stat.st_mode & 0o777,
            )
    return result


def discover_dependencies(
    exported_test: Path, test_set: Path, test_name: str
) -> list[str]:
    dependencies: set[str] = set()
    for path in exported_test.rglob("*"):
        if path.is_symlink():
            try:
                data = os.fsencode(os.readlink(path))
            except OSError:
                continue
        elif not path.is_file():
            continue
        else:
            try:
                if path.stat().st_size > 64 * 1024 * 1024:
                    continue
                data = path.read_bytes()
            except OSError:
                continue
        for match in DEPENDENCY_PATTERN.finditer(data):
            name = match.group(1).decode("ascii")
            if (
                name not in {".", "..", test_name}
                and Path(name).name == name
                and (test_set / name).is_dir()
            ):
                dependencies.add(name)
    return sorted(dependencies)


def export_test_source(source: Path, destination: Path, source_kind: str) -> None:
    if source_kind == "svn":
        run_checked(["svn", "export", "--quiet", str(source), str(destination)])
        return
    if source_kind != "local":
        raise RegressionError(f"unsupported test-set source type: {source_kind!r}")
    shutil.copytree(source, destination)


def run_process(
    command: list[str], *, cwd: Path, env: dict[str, str], timeout: float
) -> tuple[int, bytes, bool]:
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=(os.name != "nt"),
        )
    except OSError as exc:
        return 127, str(exc).encode(), False
    try:
        output, _ = process.communicate(timeout=timeout)
        return process.returncode, output, False
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        output, _ = process.communicate()
        return process.returncode, output, True


def translate_simple_csh_runner(script: str, source: Path) -> str:
    """Translate the small csh subset used by elegantTestSet runners to sh."""
    lines = script.splitlines(keepends=True)
    if not lines or not CSH_SHEBANG_PATTERN.match(lines[0]):
        return script

    translated = ["#!/bin/sh\n"]
    for line_number, line in enumerate(lines[1:], start=2):
        ending = "\n" if line.endswith("\n") else ""
        body = line[:-1] if ending else line
        stripped = body.strip()
        indentation = body[: len(body) - len(body.lstrip())]

        if stripped == "set nonomatch":
            # Unmatched globs remain literal in a default POSIX shell, which
            # is the behavior requested by csh's nonomatch option.
            continue

        match = re.fullmatch(r"if\s*\(\s*(-[a-zA-Z])\s+(.+?)\s*\)\s*then", stripped)
        if match:
            operator, operand = match.groups()
            if operator not in {"-b", "-c", "-d", "-e", "-f", "-r", "-w", "-x"}:
                raise RegressionError(
                    f"unsupported csh file test in {source}:{line_number}: {stripped}"
                )
            translated.append(f"{indentation}if [ {operator} {operand} ]; then{ending}")
            continue

        match = re.fullmatch(
            r"foreach\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*", stripped
        )
        if match:
            variable, values = match.groups()
            translated.append(f"{indentation}for {variable} in {values}; do{ending}")
            continue

        if stripped == "endif":
            translated.append(f"{indentation}fi{ending}")
            continue
        if stripped == "end":
            translated.append(f"{indentation}done{ending}")
            continue

        unsupported = re.match(
            r"(?:@|alias|breaksw|case|default|else\s+if|foreach|goto|if\s*\(|"
            r"onintr|repeat|set|source|switch|unalias|unset|while)\b",
            stripped,
        )
        if unsupported or re.search(r"\$(?:#?argv|status)(?:\b|\[)", stripped):
            raise RegressionError(
                f"unsupported csh syntax in {source}:{line_number}: {stripped}"
            )

        translated.append(body.replace(">!", ">") + ending)
    return "".join(translated)


def prepare_runner_command(runner: Path, arguments: list[str]) -> tuple[list[str], str]:
    script = runner.read_text(errors="surrogateescape")
    if not CSH_SHEBANG_PATTERN.match(script):
        return [str(runner), *arguments], "native"

    csh = shutil.which("csh") or shutil.which("tcsh")
    if csh:
        return [csh, "-f", str(runner), *arguments], str(Path(csh).resolve())

    translated_runner = runner.with_name(runner.name + ".sh")
    translated_runner.write_text(
        translate_simple_csh_runner(script, runner), errors="surrogateescape"
    )
    translated_runner.chmod(translated_runner.stat().st_mode | 0o700)
    return [str(translated_runner), *arguments], "POSIX csh compatibility"


def translate_supported_csh_helpers(root: Path) -> tuple[list[str], list[str]]:
    if shutil.which("csh") or shutil.which("tcsh"):
        return [], []

    translated: list[str] = []
    unsupported: list[str] = []
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            if path.stat().st_size > 1024 * 1024:
                continue
            script = path.read_text(errors="surrogateescape")
        except OSError:
            continue
        if not CSH_SHEBANG_PATTERN.match(script):
            continue
        relative = path.relative_to(root).as_posix()
        try:
            replacement = translate_simple_csh_runner(script, path)
        except RegressionError:
            unsupported.append(relative)
            continue
        path.write_text(replacement, errors="surrogateescape")
        translated.append(relative)
    return sorted(translated), sorted(unsupported)


def prepare_commands(
    test_dir: Path, executable: Path
) -> tuple[list[list[str]], bool, str]:
    run_template = test_dir / "runTemplate"
    run_script = test_dir / "runScript"
    if run_template.is_file():
        run_job = test_dir / "runJob"
        template = run_template.read_text(errors="surrogateescape")
        run_job.write_text(
            template.replace("<executable>", shlex.quote(str(executable))),
            errors="surrogateescape",
        )
        run_job.chmod(run_job.stat().st_mode | 0o700)
        command, runner_mode = prepare_runner_command(run_job, [])
        return [command], True, runner_mode
    if run_script.is_file():
        # frfmode1 and frfmode2 use this file only as the source for their
        # final run.done semaphore, but it is absent from the SVN directories.
        # Supplying an empty source lets the runner reach its intended
        # completion semaphore instead of failing at its final `cat` command.
        script = run_script.read_text(errors="surrogateescape")
        if "cat done > run.done" in script and not (test_dir / "done").exists():
            (test_dir / "done").touch()
        command, runner_mode = prepare_runner_command(run_script, [str(executable)])
        return [command], True, runner_mode
    run_file = test_dir / "run.ele"
    if run_file.is_file():
        return [[str(executable), run_file.name]], False, "direct"
    inputs = sorted(test_dir.glob("run*.ele"))
    if not inputs:
        raise RegressionError(f"no runnable input found in {test_dir.name}")
    return [[str(executable), path.name] for path in inputs], False, "direct"


def installed_oag_top_dir(environment: dict[str, str], test_set: Path) -> Path | None:
    configured = environment.get("OAG_TOP_DIR")
    if configured:
        return Path(configured)

    candidates = [
        ancestor.parent
        for ancestor in test_set.parents
        if ancestor.name == "oag" and (ancestor / "apps").is_dir()
    ]
    candidates.extend([Path("/usr/local"), Path("/usr")])
    host_arch = environment.get("HOST_ARCH", "")
    for candidate in candidates:
        library_root = candidate / "oag/apps/lib"
        if host_arch and (library_root / host_arch).exists():
            return candidate
        if library_root.is_dir() and any(library_root.iterdir()):
            return candidate
    return None


def install_csh_command_shim(shim_dir: Path) -> str:
    """Make elegant's simple generated .csh command files executable."""
    csh = shutil.which("csh")
    if csh:
        return str(Path(csh).resolve())
    tcsh = shutil.which("tcsh")
    target = Path(tcsh).resolve() if tcsh else Path("/bin/sh")
    (shim_dir / "csh").symlink_to(target)
    if tcsh:
        return str(target)
    return "POSIX csh compatibility"


def is_sdds(path: Path) -> bool:
    checker = shutil.which("sddscheck")
    if not checker:
        return False
    completed = subprocess.run(
        [checker, str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return completed.stdout.strip() == "ok"


def artifact_entries(test_dir: Path, relatives: Iterable[str]) -> list[dict[str, Any]]:
    entries = []
    for relative in sorted(relatives):
        path = test_dir / relative
        entries.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
                "kind": "sdds" if is_sdds(path) else "raw",
            }
        )
    return entries


def gpu_usage_from_log(log_data: bytes) -> dict[str, Any] | None:
    summaries: list[dict[str, int | float]] = []
    synchronization_summaries: list[dict[str, int | float]] = []
    synchronization_reasons: list[str] = []
    fallback_messages: list[str] = []
    for raw_line in log_data.splitlines():
        line = raw_line.decode("utf-8", errors="replace")
        if line.startswith("elegant CUDA: CPU synchronization requested by "):
            synchronization_reasons.append(line.split(" by ", 1)[1].rstrip("."))
            continue
        if re.search(r"\b(?:fallback|falling back|unsupported)\b", line, re.I):
            fallback_messages.append(line.strip())
        if line.startswith("elegant CUDA: sync requests="):
            target = synchronization_summaries
        elif line.startswith("elegant CUDA: elements="):
            target = summaries
        else:
            continue
        summary: dict[str, int | float] = {}
        for name, value in re.findall(r"([A-Za-z][A-Za-z0-9]*)=([-+0-9.eE]+)s?", line):
            try:
                summary[name] = (
                    float(value) if any(c in value for c in ".eE") else int(value)
                )
            except ValueError:
                continue
        if summary:
            target.append(summary)
    if not summaries and not synchronization_summaries and not synchronization_reasons:
        return None

    final_synchronization = (
        max(
            synchronization_summaries,
            key=lambda item: int(item.get("requests", 0)),
        )
        if synchronization_summaries
        else {}
    )
    cpu_element_reasons = [
        reason
        for reason in synchronization_reasons
        if reason.startswith("CPU element after CUDA element:")
    ]

    def bounded_details(items: list[Any]) -> list[Any]:
        if len(items) <= GPU_LOG_DETAIL_LIMIT:
            return items
        half = GPU_LOG_DETAIL_LIMIT // 2
        return items[:half] + items[-half:]

    usage = {
        "summary_count": len(summaries),
        "summaries": bounded_details(summaries),
        "summaries_truncated": len(summaries) > GPU_LOG_DETAIL_LIMIT,
        "total_elements": sum(int(item.get("elements", 0)) for item in summaries),
        "total_kernel_seconds": round(
            sum(float(item.get("kernel", 0.0)) for item in summaries), 6
        ),
        "total_gpu_wall_seconds": round(
            sum(float(item.get("wall", 0.0)) for item in summaries), 6
        ),
        "synchronization_summary_count": len(synchronization_summaries),
        "synchronization_summaries": bounded_details(synchronization_summaries),
        "synchronization_summaries_truncated": (
            len(synchronization_summaries) > GPU_LOG_DETAIL_LIMIT
        ),
        "synchronization_counts": final_synchronization,
        "synchronization_reason_count": len(synchronization_reasons),
        "synchronization_reasons": bounded_details(synchronization_reasons),
        "synchronization_reasons_truncated": (
            len(synchronization_reasons) > GPU_LOG_DETAIL_LIMIT
        ),
        "cpu_element_synchronizations": len(cpu_element_reasons),
        "cpu_element_synchronization_reasons": bounded_details(cpu_element_reasons),
        "cpu_element_synchronization_reasons_truncated": (
            len(cpu_element_reasons) > GPU_LOG_DETAIL_LIMIT
        ),
        "fallback_message_count": len(fallback_messages),
        "fallback_messages": bounded_details(fallback_messages),
        "fallback_messages_truncated": (len(fallback_messages) > GPU_LOG_DETAIL_LIMIT),
    }
    return usage


def run_test(
    *,
    name: str,
    test_set: Path,
    source_kind: str,
    executable: Path,
    artifact_root: Path,
    timeout: float,
    keep_work: bool,
    environment_overrides: dict[str, str],
) -> dict[str, Any]:
    started = time.monotonic()
    log_path = artifact_root / "logs" / f"{name}.log"
    output_root = artifact_root / "outputs" / name
    if keep_work:
        work_root = artifact_root / "work" / name
        work_root.mkdir(parents=True)
        remove_work = False
    else:
        work_root = Path(tempfile.mkdtemp(prefix=f"elegant-{name}-"))
        remove_work = True

    result: dict[str, Any] = {
        "name": name,
        "status": "failed",
        "elapsed_seconds": 0.0,
        "execution_seconds": 0.0,
        "harness_overhead_seconds": 0.0,
        "commands": [],
        "command_timings": [],
        "dependencies": [],
        "outputs": [],
        "deleted_inputs": [],
        "runner_mode": "",
        "subprocess_shell": "",
        "translated_csh_helpers": [],
        "unsupported_csh_helpers": [],
        "environment_overrides": dict(environment_overrides),
        "log": f"logs/{name}.log",
    }
    log_chunks: list[bytes] = []
    try:
        test_dir = work_root / name
        export_test_source(test_set / name, test_dir, source_kind)
        dependencies = discover_dependencies(test_dir, test_set, name)
        result["dependencies"] = dependencies
        for dependency in dependencies:
            export_test_source(
                test_set / dependency, work_root / dependency, source_kind
            )

        commands, require_done, runner_mode = prepare_commands(test_dir, executable)
        result["runner_mode"] = runner_mode
        translated_helpers, unsupported_helpers = translate_supported_csh_helpers(
            work_root
        )
        result["translated_csh_helpers"] = translated_helpers
        result["unsupported_csh_helpers"] = unsupported_helpers
        # Snapshot after generating harness-only runner files so they are not
        # mistaken for outputs from elegant.
        before = scan_regular_files(test_dir)

        shim_dir = work_root / ".candidate-bin"
        shim_dir.mkdir()
        (shim_dir / "elegant").symlink_to(executable)
        result["subprocess_shell"] = install_csh_command_shim(shim_dir)
        environment = os.environ.copy()
        environment.update(environment_overrides)
        search_paths = [shim_dir, test_dir, executable.parent]
        if executable.parent.name.endswith("-gpu"):
            cpu_bin = executable.parent.with_name(
                executable.parent.name.removesuffix("-gpu")
            )
            if cpu_bin.is_dir():
                search_paths.append(cpu_bin)
        oag_top_dir = installed_oag_top_dir(environment, test_set)
        if oag_top_dir:
            environment["OAG_TOP_DIR"] = str(oag_top_dir)
            oag_tcl_bin = oag_top_dir / "oag/apps/bin/oagtcltk/usr/bin"
            if oag_tcl_bin.is_dir():
                search_paths.append(oag_tcl_bin)
        environment["PATH"] = os.pathsep.join(
            [*(str(path) for path in search_paths), environment.get("PATH", "")]
        )
        environment["ELEGANT_EXECUTABLE"] = str(executable)

        deadline = time.monotonic() + timeout
        for command in commands:
            result["commands"].append(command)
            log_chunks.append(f"$ {shlex.join(command)}\n".encode())
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                result["error"] = f"test exceeded timeout of {timeout:g} seconds"
                result["status"] = "timed_out"
                break
            command_started = time.monotonic()
            returncode, output, timed_out = run_process(
                command,
                cwd=test_dir,
                env=environment,
                timeout=remaining,
            )
            command_elapsed = time.monotonic() - command_started
            result["execution_seconds"] += command_elapsed
            result["command_timings"].append(
                {
                    "command": command,
                    "elapsed_seconds": round(command_elapsed, 6),
                }
            )
            log_chunks.append(output)
            log_chunks.append(
                f"\n[exit status {returncode}; elapsed {command_elapsed:.6f}s]\n".encode()
            )
            if timed_out:
                result["error"] = f"test exceeded timeout of {timeout:g} seconds"
                result["status"] = "timed_out"
                break
            if returncode:
                result["error"] = f"command exited with status {returncode}"
                break
        else:
            if require_done and not (test_dir / "run.done").exists():
                result["error"] = "runner did not create run.done"
            else:
                result["status"] = "passed"

        after = scan_regular_files(test_dir)
        changed = [
            relative
            for relative, stamp in after.items()
            if relative not in before or before[relative] != stamp
        ]
        deleted = sorted(set(before) - set(after))
        result["deleted_inputs"] = deleted
        entries = artifact_entries(test_dir, changed)
        result["outputs"] = entries
        for entry in entries:
            source = test_dir / entry["path"]
            destination = output_root / entry["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        if result["status"] == "passed" and not entries:
            result["warning"] = "test completed without comparable output files"
    except Exception as exc:  # Preserve other test results and the manifest.
        result["error"] = str(exc)
        log_chunks.append(f"\n[harness error] {exc}\n".encode())
    finally:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_data = b"".join(log_chunks)
        log_path.write_bytes(log_data)
        gpu_usage = gpu_usage_from_log(log_data)
        if gpu_usage:
            result["gpu_usage"] = gpu_usage
        total_elapsed = time.monotonic() - started
        result["execution_seconds"] = round(result["execution_seconds"], 6)
        result["elapsed_seconds"] = round(total_elapsed, 3)
        result["harness_overhead_seconds"] = round(
            max(0.0, total_elapsed - result["execution_seconds"]), 6
        )
        if remove_work:
            shutil.rmtree(work_root, ignore_errors=True)
    return result


def run_test_repeated(
    *,
    name: str,
    test_set: Path,
    source_kind: str,
    executable: Path,
    artifact_root: Path,
    timeout: float,
    keep_work: bool,
    environment_overrides: dict[str, str],
    warmup_runs: int,
    repetitions: int,
    extend_noisy_samples: bool,
) -> dict[str, Any]:
    run_roots: list[Path] = []
    warmup_results: list[tuple[Path, dict[str, Any]]] = []
    measured_results: list[tuple[Path, dict[str, Any]]] = []

    def execute(kind: str, index: int) -> tuple[Path, dict[str, Any]]:
        sample_root = Path(
            tempfile.mkdtemp(
                prefix=f".timing-{name}-{kind}-{index}-", dir=artifact_root
            )
        )
        run_roots.append(sample_root)
        return sample_root, run_test(
            name=name,
            test_set=test_set,
            source_kind=source_kind,
            executable=executable,
            artifact_root=sample_root,
            timeout=timeout,
            keep_work=keep_work,
            environment_overrides=environment_overrides,
        )

    def publish(
        sample_root: Path, result: dict[str, Any], measured_count: int
    ) -> dict[str, Any]:
        source_output = sample_root / "outputs" / name
        if source_output.is_dir():
            destination_output = artifact_root / "outputs" / name
            destination_output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_output, destination_output)
        source_log = sample_root / "logs" / f"{name}.log"
        destination_log = artifact_root / "logs" / f"{name}.log"
        destination_log.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_log, destination_log)
        if keep_work:
            source_work = sample_root / "work" / name
            if source_work.is_dir():
                destination_work = artifact_root / "work" / name
                destination_work.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source_work, destination_work)

        if warmup_runs or measured_count > 1:
            timing_log_root = artifact_root / "logs" / "timing" / name
            timing_log_root.mkdir(parents=True, exist_ok=True)
            for index, (root, _warmup) in enumerate(warmup_results, start=1):
                shutil.copy2(
                    root / "logs" / f"{name}.log",
                    timing_log_root / f"warmup-{index}.log",
                )
            for index, (root, _sample) in enumerate(measured_results, start=1):
                shutil.copy2(
                    root / "logs" / f"{name}.log",
                    timing_log_root / f"measurement-{index}.log",
                )
        return result

    try:
        for index in range(1, warmup_runs + 1):
            sample = execute("warmup", index)
            warmup_results.append(sample)
            if sample[1]["status"] != "passed":
                failed = publish(sample[0], sample[1], 0)
                failed["error"] = f"warm-up {index} failed: " + failed.get(
                    "error", "unknown error"
                )
                failed["warmup_runs"] = warmup_runs
                failed["timing_repetitions"] = 0
                return failed

        for index in range(1, repetitions + 1):
            sample = execute("measurement", index)
            measured_results.append(sample)
            if sample[1]["status"] != "passed":
                failed = publish(sample[0], sample[1], len(measured_results))
                failed["error"] = f"measurement {index} failed: " + failed.get(
                    "error", "unknown error"
                )
                failed["warmup_runs"] = warmup_runs
                failed["timing_repetitions"] = len(measured_results)
                return failed

        initial_seconds = [
            result["execution_seconds"] for _root, result in measured_results
        ]
        initial_median = statistics.median(initial_seconds)
        initial_mad = statistics.median(
            abs(value - initial_median) for value in initial_seconds
        )
        initial_dispersion = (
            100 * initial_mad / initial_median if initial_median > 0 else 0.0
        )
        if extend_noisy_samples and initial_dispersion > MAX_TIMING_DISPERSION_PERCENT:
            for offset in range(1, NOISY_ADDITIONAL_REPETITIONS + 1):
                index = repetitions + offset
                sample = execute("measurement", index)
                measured_results.append(sample)
                if sample[1]["status"] != "passed":
                    failed = publish(sample[0], sample[1], len(measured_results))
                    failed["error"] = (
                        f"additional measurement {index} failed: "
                        + failed.get("error", "unknown error")
                    )
                    failed["warmup_runs"] = warmup_runs
                    failed["timing_repetitions"] = len(measured_results)
                    return failed

        execution_samples = [
            float(result["execution_seconds"]) for _root, result in measured_results
        ]
        elapsed_samples = [
            float(result["elapsed_seconds"]) for _root, result in measured_results
        ]
        execution_median = statistics.median(execution_samples)
        execution_mad = statistics.median(
            abs(value - execution_median) for value in execution_samples
        )
        elapsed_median = statistics.median(elapsed_samples)
        representative_index = min(
            range(len(measured_results)),
            key=lambda index: abs(execution_samples[index] - execution_median),
        )
        representative_root, representative_result = measured_results[
            representative_index
        ]
        result = publish(
            representative_root, representative_result, len(measured_results)
        )
        result["execution_seconds"] = round(execution_median, 6)
        result["elapsed_seconds"] = round(elapsed_median, 6)
        result["execution_median_seconds"] = round(execution_median, 6)
        result["execution_mad_seconds"] = round(execution_mad, 6)
        result["timing_dispersion_percent"] = round(
            100 * execution_mad / execution_median if execution_median > 0 else 0.0,
            6,
        )
        result["warmup_runs"] = warmup_runs
        result["timing_repetitions"] = len(measured_results)
        result["representative_sample"] = representative_index + 1
        result["timing_samples"] = [
            {
                "sample": index,
                "execution_seconds": sample["execution_seconds"],
                "elapsed_seconds": sample["elapsed_seconds"],
                "gpu_usage": sample.get("gpu_usage"),
            }
            for index, (_root, sample) in enumerate(measured_results, start=1)
        ]
        result["warmup_samples"] = [
            {
                "sample": index,
                "execution_seconds": sample["execution_seconds"],
                "elapsed_seconds": sample["elapsed_seconds"],
            }
            for index, (_root, sample) in enumerate(warmup_results, start=1)
        ]
        return result
    finally:
        for run_root in run_roots:
            shutil.rmtree(run_root, ignore_errors=True)


def create_artifact_root(path: Path) -> None:
    if path.exists():
        raise RegressionError(
            f"output path already exists; choose a new directory: {path}"
        )
    path.mkdir(parents=True)


def run_tests(
    *,
    names: list[str],
    test_set: Path,
    source_kind: str,
    executable: Path,
    artifact_root: Path,
    timeout: float,
    jobs: int,
    keep_work: bool,
    environment_overrides: dict[str, str],
    warmup_runs: int,
    repetitions: int,
    extend_noisy_samples: bool,
) -> list[dict[str, Any]]:
    if not 1 <= jobs <= MAX_PARALLEL_TESTS:
        raise RegressionError(
            f"jobs must be between 1 and {MAX_PARALLEL_TESTS}, got {jobs}"
        )
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(
                run_test_repeated,
                name=name,
                test_set=test_set,
                source_kind=source_kind,
                executable=executable,
                artifact_root=artifact_root,
                timeout=timeout,
                keep_work=keep_work,
                environment_overrides=environment_overrides,
                warmup_runs=warmup_runs,
                repetitions=repetitions,
                extend_noisy_samples=extend_noisy_samples,
            ): name
            for name in names
        }
        completed_count = 0
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            completed_count += 1
            marker = {
                "passed": "PASS",
                "timed_out": "TIMEOUT",
            }.get(result["status"], "FAIL")
            print(
                f"[{completed_count}/{len(names)}] {marker} {result['name']} "
                f"({result['execution_seconds']:.3f}s execution, "
                f"{result['elapsed_seconds']:.3f}s total, "
                f"{len(result['outputs'])} outputs)",
                flush=True,
            )
    return sorted(results, key=lambda item: item["name"])


def write_timeout_report(
    artifact_root: Path, results: list[dict[str, Any]], timeout: float
) -> list[str]:
    timed_out = [result for result in results if result["status"] == "timed_out"]
    if not timed_out:
        return []
    names = [result["name"] for result in timed_out]
    lines = [
        f"Tests terminated after exceeding the {timeout:g}-second limit:",
        "",
    ]
    for result in timed_out:
        lines.append(f"{result['name']}: {result['error']} " f"(log: {result['log']})")
    (artifact_root / "timed_out_tests.txt").write_text("\n".join(lines) + "\n")
    print(
        f"Recorded {len(names)} timed-out test(s) in "
        f"{artifact_root / 'timed_out_tests.txt'}",
        flush=True,
    )
    return names


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_performance_baseline_summary(
    artifact_root: Path, manifest: dict[str, Any]
) -> None:
    if not is_gpu_performance_suite(manifest.get("test_set", {})):
        return
    tests: list[dict[str, Any]] = []
    for result in manifest.get("tests", []):
        samples = result.get("timing_samples")
        if not isinstance(samples, list) or not samples:
            samples = [
                {
                    "sample": 1,
                    "execution_seconds": result.get("execution_seconds"),
                    "elapsed_seconds": result.get("elapsed_seconds"),
                    "gpu_usage": result.get("gpu_usage"),
                }
            ]
        tests.append(
            {
                "name": result.get("name"),
                "status": result.get("status"),
                "execution_samples_seconds": [
                    sample.get("execution_seconds") for sample in samples
                ],
                "execution_median_seconds": result.get(
                    "execution_median_seconds", result.get("execution_seconds")
                ),
                "execution_mad_seconds": result.get("execution_mad_seconds", 0.0),
                "dispersion_percent": result.get("timing_dispersion_percent", 0.0),
                "gpu_activity_samples": [sample.get("gpu_usage") for sample in samples],
                "representative_gpu_activity": result.get("gpu_usage"),
            }
        )

    test_set = manifest.get("test_set", {})
    executable = manifest.get("executable", {})
    summary = {
        "format_version": 1,
        "created_at": manifest.get("created_at"),
        "artifact_mode": manifest.get("mode"),
        "artifact_path": str(artifact_root),
        "test_set": {
            "kind": test_set.get("kind"),
            "path": test_set.get("path"),
            "fingerprint": test_set.get("fingerprint"),
            "file_count": test_set.get("file_count"),
            "suite": test_set.get("suite"),
        },
        "executable": {
            "path": executable.get("path"),
            "sha256": executable.get("sha256"),
        },
        "hardware": hardware_metadata(),
        "run_options": manifest.get("run_options"),
        "summary": {
            "tests": len(tests),
            "passed": sum(test["status"] == "passed" for test in tests),
            "sum_of_test_medians_seconds": round(
                sum(float(test["execution_median_seconds"] or 0.0) for test in tests),
                6,
            ),
            "sum_of_test_mads_seconds": round(
                sum(float(test["execution_mad_seconds"] or 0.0) for test in tests),
                6,
            ),
        },
        "tests": tests,
    }
    write_manifest(artifact_root / "baseline-summary.json", summary)


def baseline_command(args: argparse.Namespace) -> int:
    require_commands()
    test_set = Path(args.test_set).expanduser().resolve()
    source_metadata = test_set_metadata(test_set)
    validate_suite_run_settings(source_metadata, args.jobs)
    environment_overrides = suite_environment(source_metadata)
    warmup_runs, repetitions, extend_noisy_samples = timing_run_options(
        source_metadata, args.warmup_runs, args.repetitions
    )
    executable = resolve_executable(args.elegant)
    names, excluded = select_tests(
        test_set, args.tests, include_excluded=args.include_excluded
    )
    output = Path(args.output).expanduser().resolve()
    create_artifact_root(output)
    if excluded:
        print(f"Skipping {len(excluded)} known-problem tests:", flush=True)
        for name in excluded:
            print(f"  {name}: {DEFAULT_EXCLUDED_TESTS[name]}", flush=True)
    print(
        f"Creating baseline for {len(names)} tests from "
        f"{test_set_description(source_metadata)} with {executable}; "
        f"{warmup_runs} warm-up(s), {repetitions} measured run(s)",
        flush=True,
    )
    results = run_tests(
        names=names,
        test_set=test_set,
        source_kind=test_set_kind(source_metadata),
        executable=executable,
        artifact_root=output,
        timeout=args.timeout,
        jobs=args.jobs,
        keep_work=args.keep_work,
        environment_overrides=environment_overrides,
        warmup_runs=warmup_runs,
        repetitions=repetitions,
        extend_noisy_samples=extend_noisy_samples,
    )
    timed_out = write_timeout_report(output, results, args.timeout)
    failures = [result for result in results if result["status"] != "passed"]
    manifest = {
        "format_version": FORMAT_VERSION,
        "mode": "baseline",
        "created_at": utc_timestamp(),
        "complete": not failures,
        "test_set": {"path": str(test_set), **source_metadata},
        "executable": executable_metadata(executable),
        "run_options": {
            "jobs": args.jobs,
            "timeout_seconds": args.timeout,
            "timing_metric": "execution_seconds",
            "warmup_runs": warmup_runs,
            "repetitions": repetitions,
            "additional_repetitions_if_mad_exceeds_percent": (
                NOISY_ADDITIONAL_REPETITIONS if extend_noisy_samples else 0
            ),
            "maximum_timing_dispersion_percent": MAX_TIMING_DISPERSION_PERCENT,
            "environment_overrides": environment_overrides,
        },
        "ignored_runtime_files": [
            "started",
            "done",
            "compare.done",
            "*.done",
            "*.time",
            "compare.*",
            "comp.*",
            "tmp<process>.<sequence>.in",
            "tmp<process>.<sequence>.out",
        ],
        "default_ignored_sdds_parameters": list(DEFAULT_IGNORED_PARAMETERS),
        "default_ignored_sdds_columns": list(DEFAULT_IGNORED_COLUMNS),
        "default_excluded_tests": DEFAULT_EXCLUDED_TESTS,
        "excluded_tests": excluded,
        "timed_out_tests": timed_out,
        "tests": results,
    }
    write_manifest(output / "manifest.json", manifest)
    write_performance_baseline_summary(output, manifest)
    ordinary_failures = [result for result in failures if result["status"] == "failed"]
    print(
        f"Baseline written to {output}: {len(results) - len(failures)} passed, "
        f"{len(ordinary_failures)} failed, {len(timed_out)} timed out",
        flush=True,
    )
    return 1 if failures else 0


def load_baseline(path: Path) -> dict[str, Any]:
    manifest_path = path / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RegressionError(
            f"unable to read baseline manifest: {manifest_path}"
        ) from exc
    if manifest.get("format_version") != FORMAT_VERSION:
        raise RegressionError(
            f"unsupported baseline format: {manifest.get('format_version')!r}"
        )
    if manifest.get("mode") != "baseline" or not manifest.get("complete"):
        raise RegressionError("baseline is incomplete or contains failed tests")
    for test in manifest.get("tests", []):
        for entry in test.get("outputs", []):
            output = path / "outputs" / test["name"] / entry["path"]
            if not output.is_file():
                raise RegressionError(f"baseline output is missing: {output}")
            if sha256_file(output) != entry["sha256"]:
                raise RegressionError(f"baseline output was modified: {output}")
    return manifest


def sdds_names(path: Path, option: str) -> list[str]:
    query = shutil.which("sddsquery")
    if not query:
        raise RegressionError("sddsquery is required to compare changed SDDS files")
    completed = subprocess.run(
        [query, str(path), option],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RegressionError(
            f"sddsquery failed for {path}: {completed.stderr.strip()}"
        )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def ignored_parameter(name: str, patterns: list[str]) -> bool:
    lowered = name.lower()
    return any(fnmatch.fnmatchcase(lowered, pattern.lower()) for pattern in patterns)


def compare_sdds(
    baseline: Path,
    candidate: Path,
    *,
    ignored_parameters: list[str],
    ignored_columns: list[str],
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[bool, str]:
    baseline_columns = [
        name
        for name in sdds_names(baseline, "-columnlist")
        if not ignored_parameter(name, ignored_columns)
    ]
    candidate_columns = [
        name
        for name in sdds_names(candidate, "-columnlist")
        if not ignored_parameter(name, ignored_columns)
    ]
    baseline_arrays = sdds_names(baseline, "-arraylist")
    candidate_arrays = sdds_names(candidate, "-arraylist")
    baseline_parameters = [
        name
        for name in sdds_names(baseline, "-parameterlist")
        if not ignored_parameter(name, ignored_parameters)
    ]
    candidate_parameters = [
        name
        for name in sdds_names(candidate, "-parameterlist")
        if not ignored_parameter(name, ignored_parameters)
    ]
    schema_pairs = (
        ("columns", baseline_columns, candidate_columns),
        ("arrays", baseline_arrays, candidate_arrays),
        ("parameters", baseline_parameters, candidate_parameters),
    )
    schema_changes = [name for name, old, new in schema_pairs if old != new]
    if schema_changes:
        detail = []
        for name, old, new in schema_pairs:
            if old != new:
                detail.append(f"{name}: baseline={old!r}, candidate={new!r}")
        return False, "SDDS schema changed; " + "; ".join(detail)

    command = [shutil.which("sddsdiff") or ""]
    if not command[0]:
        raise RegressionError("sddsdiff is required to compare changed SDDS files")
    command.extend([str(baseline), str(candidate)])
    if baseline_columns:
        command.append("-columns=" + ",".join(baseline_columns))
    if baseline_parameters:
        command.append("-parameters=" + ",".join(baseline_parameters))
    if baseline_arrays:
        command.append("-arrays=" + ",".join(baseline_arrays))
    if not (baseline_columns or baseline_parameters or baseline_arrays):
        return True, "only ignored SDDS metadata changed"
    # Not all supported SDDS installations provide sddsdiff's relative
    # tolerance option.  Use sddsdiff only for the exact/schema-aware screen,
    # then apply the combined absolute/relative envelope below with
    # sdds2stream.  This also lets us reject integer, string, non-finite,
    # page-count, and array-shape changes consistently on every installation.
    command.append("-exact")
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    output = completed.stdout.strip()
    if len(output) > 50000:
        output = output[:50000] + "\n[comparison output truncated]"
    if "Unknown option" in output or output.startswith("Usage:"):
        raise RegressionError("sddsdiff rejected the comparison options:\n" + output)
    if completed.returncode:
        raise RegressionError(
            f"sddsdiff failed with status {completed.returncode}"
            + (f":\n{output}" if output else "")
        )
    # sddsdiff reports data differences in its output but, unlike operational
    # errors, does not use a nonzero exit status for them.
    if " are different." in output or "Differences found" in output:
        if absolute_tolerance or relative_tolerance:
            assessment = assess_sdds_gpu_difference(
                baseline,
                candidate,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
                ignored_parameters=ignored_parameters,
                ignored_columns=ignored_columns,
            )
            if assessment["classification"] == "probably_expected_gpu_roundoff":
                return (
                    True,
                    "all numeric differences are inside the configured "
                    "absolute/relative tolerance envelope",
                )
            return False, assessment["reason"] + "\n" + output
        return False, output
    if " are identical." not in output:
        return False, output or "sddsdiff returned no comparison result"
    return True, output


def sdds_definition_records(path: Path) -> list[dict[str, str]]:
    """Return SDDS field definitions in class and declaration order."""
    with tempfile.TemporaryDirectory(prefix="elegant-sdds-query-") as temporary:
        query_output = Path(temporary) / "definitions.sdds"
        run_checked(["sddsquery", str(path), f"-sddsOutput={query_output}"])
        classes_text = run_checked(
            ["sdds2stream", str(query_output), "-parameters=Class", "-noquotes"]
        )
        counts_text = run_checked(["sdds2stream", str(query_output), "-rows=bare"])
        fields_text = run_checked(
            [
                "sdds2stream",
                str(query_output),
                "-columns=Name,Type,Units,Symbol,Format,Description,Group",
                "-delimiter=\t",
                "-noquotes",
            ]
        )

    classes = classes_text.splitlines() if classes_text else []
    try:
        counts = [int(value) for value in counts_text.splitlines()]
    except ValueError as exc:
        raise RegressionError(f"invalid sddsquery row count for {path}") from exc
    field_lines = fields_text.splitlines() if fields_text else []
    if len(classes) != len(counts) or sum(counts) != len(field_lines):
        raise RegressionError(f"unable to interpret SDDS definitions for {path}")

    result: list[dict[str, str]] = []
    offset = 0
    metadata_names = (
        "name",
        "type",
        "units",
        "symbol",
        "format",
        "description",
        "group",
    )
    for field_class, count in zip(classes, counts):
        normalized_class = field_class.strip().lower()
        for line in field_lines[offset : offset + count]:
            values = line.split("\t", len(metadata_names) - 1)
            values.extend([""] * (len(metadata_names) - len(values)))
            record = dict(zip(metadata_names, values))
            record["class"] = normalized_class
            record["type"] = record["type"].strip().lower()
            result.append(record)
        offset += count
    return result


def sdds_page_rows(path: Path) -> list[int]:
    output = run_checked(["sdds2stream", str(path), "-rows=bare"])
    try:
        return [int(value) for value in output.splitlines()] if output else []
    except ValueError as exc:
        raise RegressionError(f"invalid SDDS page row count for {path}") from exc


def sdds_description(path: Path) -> str:
    return run_checked(["sdds2stream", str(path), "-description", "-delimiter=\x1f"])


def sdds_field_command(path: Path, field_class: str, name: str) -> list[str]:
    option = SDDS_VALUE_OPTIONS.get(field_class)
    if not option:
        raise RegressionError(f"unsupported SDDS field class: {field_class!r}")
    stream = shutil.which("sdds2stream")
    if not stream:
        raise RegressionError("sdds2stream is required for GPU difference assessment")
    return [
        stream,
        str(path),
        f"{option}={name}",
        "-delimiter=\x1f",
        "-ignoreFormats",
    ]


def sdds_line_values(line: str) -> list[str]:
    values = line.rstrip("\r\n").split("\x1f")
    if values and values[-1] == "":
        values.pop()
    return values


def compare_sdds_field_values(
    baseline: Path,
    candidate: Path,
    definition: dict[str, str],
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    """Compare one SDDS field while retaining compact numerical evidence."""
    commands = (
        sdds_field_command(baseline, definition["class"], definition["name"]),
        sdds_field_command(candidate, definition["class"], definition["name"]),
    )
    error_streams = [tempfile.TemporaryFile(mode="w+t") for _ in commands]
    processes: list[subprocess.Popen[str]] = []
    is_floating = definition["type"] in SDDS_FLOAT_TYPES
    result: dict[str, Any] = {
        "class": definition["class"],
        "name": definition["name"],
        "type": definition["type"],
        "values_compared": 0,
        "different_values": 0,
        "outside_noise_envelope": 0,
        "max_absolute_difference": 0.0,
        "max_relative_difference": 0.0,
        "unbounded_relative_difference": False,
    }
    try:
        for command, error_stream in zip(commands, error_streams):
            processes.append(
                subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=error_stream,
                    text=True,
                    errors="replace",
                )
            )
        assert processes[0].stdout is not None
        assert processes[1].stdout is not None
        value_index = 0
        for baseline_line, candidate_line in itertools.zip_longest(
            processes[0].stdout, processes[1].stdout
        ):
            if baseline_line is None or candidate_line is None:
                result["stream_shape_changed"] = True
                continue
            baseline_values = sdds_line_values(baseline_line)
            candidate_values = sdds_line_values(candidate_line)
            if len(baseline_values) != len(candidate_values):
                result["stream_shape_changed"] = True
            for old_text, new_text in itertools.zip_longest(
                baseline_values, candidate_values
            ):
                value_index += 1
                if old_text is None or new_text is None:
                    continue
                result["values_compared"] += 1
                if not is_floating:
                    if old_text != new_text:
                        result["different_values"] += 1
                        result["outside_noise_envelope"] += 1
                        result.setdefault(
                            "first_outside",
                            {
                                "index": value_index,
                                "baseline": old_text,
                                "candidate": new_text,
                            },
                        )
                    continue

                try:
                    old_value = float(old_text)
                    new_value = float(new_text)
                except ValueError as exc:
                    raise RegressionError(
                        f"unable to parse numeric SDDS values for "
                        f"{definition['class']} {definition['name']!r}"
                    ) from exc
                if math.isnan(old_value) and math.isnan(new_value):
                    continue
                if old_value == new_value:
                    continue
                result["different_values"] += 1
                if not (math.isfinite(old_value) and math.isfinite(new_value)):
                    outside = True
                    absolute_difference = math.inf
                    relative_difference: float | None = None
                else:
                    absolute_difference = abs(old_value - new_value)
                    scale = max(abs(old_value), abs(new_value))
                    if scale:
                        relative_difference = absolute_difference / scale
                    else:
                        relative_difference = None
                    outside = absolute_difference > max(
                        absolute_tolerance, relative_tolerance * scale
                    )
                if math.isfinite(absolute_difference):
                    result["max_absolute_difference"] = max(
                        result["max_absolute_difference"], absolute_difference
                    )
                else:
                    result["nonfinite_difference"] = True
                if relative_difference is None:
                    result["unbounded_relative_difference"] = True
                else:
                    result["max_relative_difference"] = max(
                        result["max_relative_difference"], relative_difference
                    )
                if outside:
                    result["outside_noise_envelope"] += 1
                    result.setdefault(
                        "first_outside",
                        {
                            "index": value_index,
                            "baseline": old_text,
                            "candidate": new_text,
                            "absolute_difference": (
                                absolute_difference
                                if math.isfinite(absolute_difference)
                                else "nonfinite"
                            ),
                            "relative_difference": relative_difference,
                        },
                    )
        for process, error_stream, command in zip(processes, error_streams, commands):
            returncode = process.wait()
            error_stream.seek(0)
            error = error_stream.read().strip()
            if returncode:
                raise RegressionError(
                    f"sdds2stream failed ({returncode}): {shlex.join(command)}"
                    + (f"\n{error}" if error else "")
                )
    except Exception:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
        raise
    finally:
        for error_stream in error_streams:
            error_stream.close()
    return result


def potentially_significant_assessment(reason: str) -> dict[str, Any]:
    return {
        "classification": "potentially_significant",
        "severity": "high",
        "confidence": "high",
        "reason": reason,
    }


def assess_sdds_gpu_difference(
    baseline: Path,
    candidate: Path,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
    ignored_parameters: list[str] | tuple[str, ...] = (),
    ignored_columns: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    """Classify an exact SDDS difference using a strict CPU/GPU noise envelope."""
    try:
        ignored_parameter_patterns = list(ignored_parameters)
        ignored_column_patterns = list(ignored_columns)

        def included(definition: dict[str, str]) -> bool:
            return not (
                definition["class"] == "parameter"
                and ignored_parameter(definition["name"], ignored_parameter_patterns)
                or definition["class"] == "column"
                and ignored_parameter(definition["name"], ignored_column_patterns)
            )

        baseline_definitions = [
            definition
            for definition in sdds_definition_records(baseline)
            if included(definition)
        ]
        candidate_definitions = [
            definition
            for definition in sdds_definition_records(candidate)
            if included(definition)
        ]
        if baseline_definitions != candidate_definitions:
            return potentially_significant_assessment(
                "SDDS field definitions or metadata changed"
            )
        if sdds_description(baseline) != sdds_description(candidate):
            return potentially_significant_assessment(
                "SDDS description or contents metadata changed"
            )
        if sdds_page_rows(baseline) != sdds_page_rows(candidate):
            return potentially_significant_assessment(
                "SDDS page count or rows per page changed"
            )

        numerical_fields: list[dict[str, Any]] = []
        discrete_changes: list[str] = []
        stream_shape_changes: list[str] = []
        for definition in baseline_definitions:
            field_class = definition["class"]
            if field_class not in SDDS_VALUE_OPTIONS:
                return potentially_significant_assessment(
                    f"unsupported changed SDDS field class {field_class!r}"
                )
            metrics = compare_sdds_field_values(
                baseline,
                candidate,
                definition,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            )
            label = f"{field_class} {definition['name']}"
            if metrics.get("stream_shape_changed"):
                stream_shape_changes.append(label)
            if definition["type"] not in SDDS_FLOAT_TYPES:
                if metrics["different_values"]:
                    discrete_changes.append(label)
            elif metrics["different_values"]:
                numerical_fields.append(metrics)

        if stream_shape_changes:
            return potentially_significant_assessment(
                "SDDS value count or array shape changed for: "
                + ", ".join(stream_shape_changes[:10])
            )
        if discrete_changes:
            return potentially_significant_assessment(
                "integer, character, or string data changed for: "
                + ", ".join(discrete_changes[:10])
            )
        if not numerical_fields:
            return potentially_significant_assessment(
                "the exact SDDS comparison changed, but no floating-point "
                "data differences were measurable"
            )

        outside_count = sum(item["outside_noise_envelope"] for item in numerical_fields)
        differing_count = sum(item["different_values"] for item in numerical_fields)
        assessment = {
            "classification": (
                "probably_expected_gpu_roundoff"
                if outside_count == 0
                else "potentially_significant"
            ),
            "severity": "low" if outside_count == 0 else "high",
            "confidence": "medium",
            "reason": (
                "all floating-point changes are inside the configured GPU "
                "roundoff envelope"
                if outside_count == 0
                else f"{outside_count} of {differing_count} changed floating-point "
                "values exceed the configured GPU roundoff envelope"
            ),
            "floating_point_fields": numerical_fields,
            "different_values": differing_count,
            "outside_noise_envelope": outside_count,
        }
        return assessment
    except RegressionError as exc:
        assessment = potentially_significant_assessment(
            f"unable to classify this SDDS difference safely: {exc}"
        )
        assessment["confidence"] = "medium"
        return assessment


def compare_outputs(
    baseline_root: Path,
    baseline_manifest: dict[str, Any],
    candidate_root: Path,
    candidate_results: list[dict[str, Any]],
    *,
    ignored_parameters: list[str],
    ignored_columns: list[str],
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    old_tests = {item["name"]: item for item in baseline_manifest["tests"]}
    new_tests = {item["name"]: item for item in candidate_results}
    comparisons: list[dict[str, Any]] = []
    report: list[str] = []
    for name in sorted(old_tests):
        old = old_tests[name]
        new = new_tests[name]
        test_changes: list[dict[str, str]] = []
        if new["status"] != "passed":
            test_changes.append(
                {"path": "", "status": "test_failed", "detail": new.get("error", "")}
            )
        if old.get("deleted_inputs", []) != new.get("deleted_inputs", []):
            test_changes.append(
                {
                    "path": "",
                    "status": "deleted_input_set_changed",
                    "detail": (
                        f"baseline={old.get('deleted_inputs', [])!r}, "
                        f"candidate={new.get('deleted_inputs', [])!r}"
                    ),
                }
            )
        old_outputs = {
            entry["path"]: entry
            for entry in old["outputs"]
            if not excluded_runtime_path(entry["path"])
        }
        new_outputs = {
            entry["path"]: entry
            for entry in new["outputs"]
            if not excluded_runtime_path(entry["path"])
        }
        for relative in sorted(set(old_outputs) | set(new_outputs)):
            if relative not in old_outputs:
                test_changes.append(
                    {"path": relative, "status": "added", "detail": "new output file"}
                )
                continue
            if relative not in new_outputs:
                test_changes.append(
                    {
                        "path": relative,
                        "status": "missing",
                        "detail": "baseline output not produced",
                    }
                )
                continue
            old_entry = old_outputs[relative]
            new_entry = new_outputs[relative]
            if old_entry["sha256"] == new_entry["sha256"]:
                continue
            old_path = baseline_root / "outputs" / name / relative
            new_path = candidate_root / "outputs" / name / relative
            if old_entry["kind"] == "sdds" and new_entry["kind"] == "sdds":
                equal, detail = compare_sdds(
                    old_path,
                    new_path,
                    ignored_parameters=ignored_parameters,
                    ignored_columns=ignored_columns,
                    absolute_tolerance=absolute_tolerance,
                    relative_tolerance=relative_tolerance,
                )
                if equal:
                    continue
                test_changes.append(
                    {"path": relative, "status": "changed", "detail": detail}
                )
            else:
                test_changes.append(
                    {
                        "path": relative,
                        "status": "changed",
                        "detail": "raw file content differs",
                    }
                )
        status = "changed" if test_changes else "unchanged"
        comparisons.append({"name": name, "status": status, "changes": test_changes})
        if test_changes:
            report.append(f"[{name}]")
            for change in test_changes:
                label = change["path"] or "test"
                report.append(f"  {change['status']}: {label}")
                detail = change.get("detail", "").strip()
                if detail:
                    for line in detail.splitlines():
                        report.append(f"    {line}")
    if not report:
        report.append("No unintended output changes detected.")
    return comparisons, report


def assess_existing_comparisons(
    comparisons: list[dict[str, Any]],
    baseline_root: Path,
    baseline_manifest: dict[str, Any],
    candidate_root: Path,
    candidate_manifest: dict[str, Any],
    *,
    gpu_noise_absolute_tolerance: float,
    gpu_noise_relative_tolerance: float,
    ignored_parameters: list[str],
    ignored_columns: list[str],
) -> dict[str, int]:
    baseline_tests = {item["name"]: item for item in baseline_manifest["tests"]}
    candidate_tests = {item["name"]: item for item in candidate_manifest["tests"]}
    probably_expected_files = 0
    potentially_significant_changes = 0

    for comparison in comparisons:
        name = comparison["name"]
        baseline_outputs = {
            entry["path"]: entry for entry in baseline_tests[name]["outputs"]
        }
        candidate_outputs = {
            entry["path"]: entry for entry in candidate_tests[name]["outputs"]
        }
        has_expected_roundoff = False
        has_potentially_significant = False
        for change in comparison["changes"]:
            relative = change.get("path", "")
            baseline_entry = baseline_outputs.get(relative)
            candidate_entry = candidate_outputs.get(relative)
            if (
                change["status"] == "changed"
                and baseline_entry
                and candidate_entry
                and baseline_entry.get("kind") == "sdds"
                and candidate_entry.get("kind") == "sdds"
            ):
                assessment = assess_sdds_gpu_difference(
                    baseline_root / "outputs" / name / relative,
                    candidate_root / "outputs" / name / relative,
                    absolute_tolerance=gpu_noise_absolute_tolerance,
                    relative_tolerance=gpu_noise_relative_tolerance,
                    ignored_parameters=ignored_parameters,
                    ignored_columns=ignored_columns,
                )
            else:
                reasons = {
                    "added": "the candidate produced a new output file",
                    "missing": "the candidate did not produce a baseline output file",
                    "test_failed": "the candidate test failed",
                    "deleted_input_set_changed": "the set of deleted test inputs changed",
                    "changed": "non-SDDS file content changed",
                }
                assessment = potentially_significant_assessment(
                    reasons.get(change["status"], "the output structure changed")
                )
            change["assessment"] = assessment
            if assessment["classification"] == "probably_expected_gpu_roundoff":
                has_expected_roundoff = True
                probably_expected_files += 1
            else:
                has_potentially_significant = True
                potentially_significant_changes += 1

        if has_potentially_significant:
            comparison["status"] = "changed"
        elif has_expected_roundoff:
            comparison["status"] = "probably_expected_gpu_roundoff"
        else:
            comparison["status"] = "unchanged"

    return {
        "total_tests": len(comparisons),
        "unchanged_tests": sum(item["status"] == "unchanged" for item in comparisons),
        "probably_expected_gpu_roundoff_tests": sum(
            item["status"] == "probably_expected_gpu_roundoff" for item in comparisons
        ),
        "potentially_significant_tests": sum(
            item["status"] == "changed" for item in comparisons
        ),
        "probably_expected_gpu_roundoff_files": probably_expected_files,
        "potentially_significant_changes": potentially_significant_changes,
    }


def format_gpu_field_metrics(metrics: dict[str, Any]) -> str:
    maximum_relative = metrics["max_relative_difference"]
    relative_text = f"{maximum_relative:.6g}"
    if metrics.get("unbounded_relative_difference"):
        relative_text += " (plus a zero-scale difference)"
    return (
        f"{metrics['class']} {metrics['name']}: "
        f"{metrics['different_values']}/{metrics['values_compared']} values differ; "
        f"max abs={metrics['max_absolute_difference']:.6g}, "
        f"max rel={relative_text}, "
        f"outside envelope={metrics['outside_noise_envelope']}"
    )


def existing_comparison_report(
    comparisons: list[dict[str, Any]],
    summary: dict[str, int],
    *,
    gpu_noise_absolute_tolerance: float,
    gpu_noise_relative_tolerance: float,
) -> list[str]:
    report = [
        "CPU/GPU output significance assessment",
        "",
        (
            "A floating-point value is considered probable GPU roundoff when "
            "abs(cpu-gpu) <= max("
            f"{gpu_noise_absolute_tolerance:.6g}, "
            f"{gpu_noise_relative_tolerance:.6g} * max(abs(cpu), abs(gpu)))."
        ),
        (
            "Schema, file-set, page/row, integer/string, non-finite, and raw-file "
            "changes are potentially significant. This is a screening rule, not "
            "a proof of physical equivalence."
        ),
        "",
        "Summary:",
        f"  tests compared: {summary['total_tests']}",
        f"  unchanged tests: {summary['unchanged_tests']}",
        (
            "  tests with only probable GPU roundoff: "
            f"{summary['probably_expected_gpu_roundoff_tests']}"
        ),
        ("  tests needing review: " f"{summary['potentially_significant_tests']}"),
        (
            "  files classified as probable GPU roundoff: "
            f"{summary['probably_expected_gpu_roundoff_files']}"
        ),
        (
            "  potentially significant changes: "
            f"{summary['potentially_significant_changes']}"
        ),
    ]
    changed_comparisons = [
        item for item in comparisons if item["status"] != "unchanged"
    ]
    if not changed_comparisons:
        report.extend(["", "No output changes detected."])
        return report

    for comparison in changed_comparisons:
        report.extend(["", f"[{comparison['name']}] {comparison['status']}"])
        for change in comparison["changes"]:
            label = change.get("path") or "test"
            assessment = change["assessment"]
            report.append(f"  {assessment['severity']}: {change['status']} {label}")
            report.append(
                f"    classification: {assessment['classification']} "
                f"(confidence: {assessment['confidence']})"
            )
            report.append(f"    reason: {assessment['reason']}")
            floating_fields = sorted(
                assessment.get("floating_point_fields", []),
                key=lambda item: (
                    item["outside_noise_envelope"],
                    item["max_relative_difference"],
                    item["max_absolute_difference"],
                ),
                reverse=True,
            )
            for metrics in floating_fields[:20]:
                report.append("    " + format_gpu_field_metrics(metrics))
            if len(floating_fields) > 20:
                report.append(
                    f"    ... {len(floating_fields) - 20} more changed floating-point "
                    "fields are recorded in manifest.json"
                )
            detail = change.get("detail", "").strip()
            if detail and assessment["classification"] == "potentially_significant":
                report.append("    exact comparison detail:")
                for line in detail.splitlines():
                    report.append(f"      {line}")
    return report


def recorded_test_runtime(result: dict[str, Any]) -> tuple[float, str]:
    median = result.get("execution_median_seconds")
    if isinstance(median, (int, float)) and median > 0:
        return float(median), "execution_median_seconds"
    execution = result.get("execution_seconds")
    if isinstance(execution, (int, float)) and execution > 0:
        return float(execution), "execution_seconds"
    elapsed = result.get("elapsed_seconds")
    if isinstance(elapsed, (int, float)) and elapsed > 0:
        return float(elapsed), "elapsed_seconds_legacy"
    raise RegressionError(f"test {result.get('name')!r} has no positive runtime")


def performance_policy(
    baseline_manifest: dict[str, Any], requested_minimum_speedup: float | None
) -> tuple[bool, float, bool]:
    test_set = baseline_manifest.get("test_set", {})
    suite = test_set.get("suite", {}) if isinstance(test_set, dict) else {}
    if not isinstance(suite, dict):
        suite = {}
    is_performance_suite = suite.get("suite_type") == GPU_PERFORMANCE_SUITE_TYPE
    enabled = is_performance_suite or requested_minimum_speedup is not None
    minimum_speedup = (
        float(requested_minimum_speedup)
        if requested_minimum_speedup is not None
        else float(suite.get("minimum_speedup", 0))
    )
    require_gpu_activity = bool(suite.get("require_gpu_activity", False))
    return enabled, minimum_speedup, require_gpu_activity


def gpu_noise_policy(
    baseline_manifest: dict[str, Any],
    requested_absolute_tolerance: float | None,
    requested_relative_tolerance: float | None,
) -> tuple[float, float]:
    test_set = baseline_manifest.get("test_set", {})
    suite = test_set.get("suite", {}) if isinstance(test_set, dict) else {}
    if not isinstance(suite, dict):
        suite = {}
    absolute_tolerance = (
        float(requested_absolute_tolerance)
        if requested_absolute_tolerance is not None
        else float(
            suite.get(
                "gpu_noise_absolute_tolerance",
                DEFAULT_GPU_NOISE_ABSOLUTE_TOLERANCE,
            )
        )
    )
    relative_tolerance = (
        float(requested_relative_tolerance)
        if requested_relative_tolerance is not None
        else float(
            suite.get(
                "gpu_noise_relative_tolerance",
                DEFAULT_GPU_NOISE_RELATIVE_TOLERANCE,
            )
        )
    )
    return absolute_tolerance, relative_tolerance


def assess_runtime_performance(
    baseline_manifest: dict[str, Any],
    candidate_manifest: dict[str, Any],
    *,
    minimum_speedup: float,
    require_gpu_activity: bool,
    preceding_gpu_manifest: dict[str, Any] | None = None,
    target_tests: set[str] | None = None,
    non_target_regression_limit: float = DEFAULT_NON_TARGET_REGRESSION_LIMIT,
    suite_regression_limit: float = DEFAULT_SUITE_REGRESSION_LIMIT,
) -> dict[str, Any]:
    baseline_tests = {item["name"]: item for item in baseline_manifest["tests"]}
    candidate_tests = {item["name"]: item for item in candidate_manifest["tests"]}
    preceding_tests = (
        {item["name"]: item for item in preceding_gpu_manifest["tests"]}
        if preceding_gpu_manifest is not None
        else None
    )
    targeted = set(baseline_tests) if not target_tests else set(target_tests)
    unknown_targets = targeted - set(baseline_tests)
    if unknown_targets:
        raise RegressionError(
            "targeted performance tests are absent from the completed runs: "
            + ", ".join(sorted(unknown_targets))
        )
    rows: list[dict[str, Any]] = []
    timing_metrics: set[str] = set()
    for name in sorted(baseline_tests):
        baseline_seconds, baseline_metric = recorded_test_runtime(baseline_tests[name])
        candidate_seconds, candidate_metric = recorded_test_runtime(
            candidate_tests[name]
        )
        timing_metrics.update((baseline_metric, candidate_metric))
        speedup = baseline_seconds / candidate_seconds
        preceding_seconds = None
        incremental_speedup = None
        if preceding_tests is not None:
            preceding_seconds, preceding_metric = recorded_test_runtime(
                preceding_tests[name]
            )
            timing_metrics.add(preceding_metric)
            incremental_speedup = preceding_seconds / candidate_seconds
        gpu_usage = candidate_tests[name].get("gpu_usage", {})
        gpu_elements = int(gpu_usage.get("total_elements", 0)) if gpu_usage else 0
        reasons: list[str] = []
        if name in targeted:
            if speedup < minimum_speedup:
                reasons.append(
                    f"CPU/candidate speedup {speedup:.3f}x is below the "
                    f"{minimum_speedup:.3f}x target"
                )
            if (
                incremental_speedup is not None
                and incremental_speedup < minimum_speedup
            ):
                reasons.append(
                    f"pre-change/candidate speedup {incremental_speedup:.3f}x is "
                    f"below the {minimum_speedup:.3f}x target"
                )
        elif preceding_seconds is not None:
            slowdown = candidate_seconds / preceding_seconds - 1
            if slowdown > non_target_regression_limit:
                reasons.append(
                    f"non-target slowdown {slowdown * 100:.3f}% exceeds the "
                    f"{non_target_regression_limit * 100:.3f}% limit"
                )
        if require_gpu_activity and gpu_elements <= 0:
            reasons.append("candidate log did not report any GPU-tracked elements")
        rows.append(
            {
                "name": name,
                "targeted": name in targeted,
                "baseline_seconds": round(baseline_seconds, 6),
                "pre_change_gpu_seconds": (
                    round(preceding_seconds, 6)
                    if preceding_seconds is not None
                    else None
                ),
                "candidate_seconds": round(candidate_seconds, 6),
                "speedup": round(speedup, 6),
                "incremental_speedup": (
                    round(incremental_speedup, 6)
                    if incremental_speedup is not None
                    else None
                ),
                "candidate_percent_faster": round(
                    (1 - candidate_seconds / baseline_seconds) * 100, 3
                ),
                "candidate_gpu_elements": gpu_elements,
                "status": "meets_target" if not reasons else "needs_review",
                "reasons": reasons,
            }
        )

    speedups = [row["speedup"] for row in rows]
    baseline_total = sum(row["baseline_seconds"] for row in rows)
    candidate_total = sum(row["candidate_seconds"] for row in rows)
    preceding_total = (
        sum(float(row["pre_change_gpu_seconds"]) for row in rows)
        if preceding_tests is not None
        else None
    )
    needs_review = [row for row in rows if row["status"] == "needs_review"]
    suite_reasons: list[str] = []
    if preceding_total is not None and candidate_total > preceding_total * (
        1 + suite_regression_limit
    ):
        suite_reasons.append(
            f"candidate suite time is {(candidate_total / preceding_total - 1) * 100:.3f}% "
            f"slower than pre-change, exceeding the "
            f"{suite_regression_limit * 100:.3f}% limit"
        )

    def sample_counts(tests: dict[str, dict[str, Any]]) -> list[int]:
        return sorted(
            {
                int(
                    test.get(
                        "timing_repetitions",
                        len(test.get("timing_samples", [])) or 1,
                    )
                )
                for test in tests.values()
            }
        )

    return {
        "version": PERFORMANCE_ASSESSMENT_VERSION,
        "metric": (
            next(iter(timing_metrics))
            if len(timing_metrics) == 1
            else "mixed (median preferred, legacy timing present)"
        ),
        "samples_per_test": {
            "cpu_baseline": sample_counts(baseline_tests),
            "candidate_gpu": sample_counts(candidate_tests),
            "pre_change_gpu": (
                sample_counts(preceding_tests) if preceding_tests is not None else None
            ),
        },
        "minimum_speedup": minimum_speedup,
        "require_gpu_activity": require_gpu_activity,
        "target_tests": sorted(targeted),
        "non_target_regression_limit": non_target_regression_limit,
        "suite_regression_limit": suite_regression_limit,
        "suite_reasons": suite_reasons,
        "complete": not needs_review and not suite_reasons,
        "summary": {
            "tests": len(rows),
            "targeted_tests": len(targeted),
            "tests_passing_gates": len(rows) - len(needs_review),
            "tests_needing_review": len(needs_review),
            "baseline_total_seconds": round(baseline_total, 6),
            "pre_change_gpu_total_seconds": (
                round(preceding_total, 6) if preceding_total is not None else None
            ),
            "candidate_total_seconds": round(candidate_total, 6),
            "total_speedup": round(baseline_total / candidate_total, 6),
            "total_incremental_speedup": (
                round(preceding_total / candidate_total, 6)
                if preceding_total is not None
                else None
            ),
            "geometric_mean_speedup": round(
                math.exp(sum(math.log(value) for value in speedups) / len(speedups)),
                6,
            ),
            "median_speedup": round(statistics.median(speedups), 6),
        },
        "tests": rows,
    }


def runtime_performance_report(performance: dict[str, Any]) -> list[str]:
    summary = performance["summary"]
    lines = [
        "",
        "Runtime performance assessment",
        "",
        (
            "CPU speedup is CPU baseline / candidate GPU time. Targeted tests "
            f"must reach {performance['minimum_speedup']:.3f}x."
        ),
        (
            f"Timing metric: {performance['metric']}; sample counts: "
            f"{performance['samples_per_test']}. Medians are compared; run with "
            "--jobs 1 on an idle host."
        ),
        "",
        "Summary:",
        f"  tests measured: {summary['tests']}",
        f"  tests passing performance gates: {summary['tests_passing_gates']}",
        f"  tests needing review: {summary['tests_needing_review']}",
        f"  baseline total: {summary['baseline_total_seconds']:.6g}s",
        *(
            [
                "  pre-change GPU total: "
                f"{summary['pre_change_gpu_total_seconds']:.6g}s",
                "  total pre-change/candidate speedup: "
                f"{summary['total_incremental_speedup']:.3f}x",
            ]
            if summary["pre_change_gpu_total_seconds"] is not None
            else []
        ),
        f"  candidate total: {summary['candidate_total_seconds']:.6g}s",
        f"  total speedup: {summary['total_speedup']:.3f}x",
        f"  geometric-mean speedup: {summary['geometric_mean_speedup']:.3f}x",
        f"  median speedup: {summary['median_speedup']:.3f}x",
        "",
        "Per-test timing:",
    ]
    for row in performance["tests"]:
        pre_change = (
            f", pre-change={row['pre_change_gpu_seconds']:.6g}s, "
            f"pre-change/candidate={row['incremental_speedup']:.3f}x"
            if row["pre_change_gpu_seconds"] is not None
            else ""
        )
        lines.append(
            f"  {row['status']}: {row['name']} "
            f"({'target' if row['targeted'] else 'guard'}): "
            f"baseline={row['baseline_seconds']:.6g}s, "
            f"candidate={row['candidate_seconds']:.6g}s, "
            f"CPU/candidate={row['speedup']:.3f}x{pre_change}, "
            f"candidate GPU elements={row['candidate_gpu_elements']}"
        )
        for reason in row["reasons"]:
            lines.append(f"    {reason}")
    for reason in performance.get("suite_reasons", []):
        lines.append(f"  suite needs_review: {reason}")
    return lines


def comparison_ignore_lists(
    args: argparse.Namespace, *manifests: dict[str, Any]
) -> tuple[list[str], list[str]]:
    ignored_parameters = list(DEFAULT_IGNORED_PARAMETERS)
    ignored_columns = list(DEFAULT_IGNORED_COLUMNS)
    for manifest in manifests:
        ignored_parameters.extend(manifest.get("default_ignored_sdds_parameters", []))
        ignored_columns.extend(manifest.get("default_ignored_sdds_columns", []))
    ignored_parameters.extend(args.ignore_parameter)
    ignored_columns.extend(args.ignore_column)
    return (
        list(dict.fromkeys(ignored_parameters)),
        list(dict.fromkeys(ignored_columns)),
    )


def baseline_test_names(manifest: dict[str, Any], label: str) -> set[str]:
    names = [test.get("name") for test in manifest.get("tests", [])]
    if any(not isinstance(name, str) or not name for name in names):
        raise RegressionError(f"{label} manifest contains a test without a valid name")
    if len(names) != len(set(names)):
        raise RegressionError(f"{label} manifest contains duplicate test names")
    return set(names)


def name_preview(names: set[str]) -> str:
    ordered = sorted(names)
    preview = ", ".join(ordered[:20])
    if len(ordered) > 20:
        preview += f", ... ({len(ordered)} total)"
    return preview or "none"


def validate_existing_baselines(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> None:
    expected = baseline.get("test_set", {})
    actual = candidate.get("test_set", {})
    validate_test_set_identity(expected, actual, actual_label="candidate")

    baseline_names = baseline_test_names(baseline, "baseline")
    candidate_names = baseline_test_names(candidate, "candidate")
    if baseline_names != candidate_names:
        raise RegressionError(
            "completed runs contain different test sets\n"
            f"missing from candidate: {name_preview(baseline_names - candidate_names)}\n"
            f"only in candidate: {name_preview(candidate_names - baseline_names)}"
        )


def compare_command(args: argparse.Namespace) -> int:
    require_commands()
    baseline_root = Path(args.baseline).expanduser().resolve()
    baseline = load_baseline(baseline_root)
    test_set = Path(args.test_set).expanduser().resolve()
    source_metadata = test_set_metadata(test_set)
    validate_suite_run_settings(source_metadata, args.jobs)
    environment_overrides = suite_environment(source_metadata)
    warmup_runs, repetitions, extend_noisy_samples = timing_run_options(
        source_metadata, args.warmup_runs, args.repetitions
    )
    expected = baseline["test_set"]
    validate_test_set_identity(expected, source_metadata, actual_label="candidate")
    executable = resolve_executable(args.elegant)
    names = [item["name"] for item in baseline["tests"]]
    output = Path(args.output).expanduser().resolve()
    create_artifact_root(output)
    print(
        f"Running {len(names)} baseline tests with candidate {executable}", flush=True
    )
    results = run_tests(
        names=names,
        test_set=test_set,
        source_kind=test_set_kind(source_metadata),
        executable=executable,
        artifact_root=output,
        timeout=args.timeout,
        jobs=args.jobs,
        keep_work=args.keep_work,
        environment_overrides=environment_overrides,
        warmup_runs=warmup_runs,
        repetitions=repetitions,
        extend_noisy_samples=extend_noisy_samples,
    )
    timed_out = write_timeout_report(output, results, args.timeout)
    ignored, ignored_columns = comparison_ignore_lists(args, baseline)
    comparisons, report = compare_outputs(
        baseline_root,
        baseline,
        output,
        results,
        ignored_parameters=ignored,
        ignored_columns=ignored_columns,
        absolute_tolerance=args.absolute_tolerance,
        relative_tolerance=args.relative_tolerance,
    )
    changed = [item for item in comparisons if item["status"] == "changed"]
    manifest = {
        "format_version": FORMAT_VERSION,
        "mode": "candidate",
        "created_at": utc_timestamp(),
        "complete": not changed,
        "baseline": str(baseline_root),
        "test_set": {"path": str(test_set), **source_metadata},
        "executable": executable_metadata(executable),
        "run_options": {
            "jobs": args.jobs,
            "timeout_seconds": args.timeout,
            "timing_metric": "execution_seconds",
            "warmup_runs": warmup_runs,
            "repetitions": repetitions,
            "additional_repetitions_if_mad_exceeds_percent": (
                NOISY_ADDITIONAL_REPETITIONS if extend_noisy_samples else 0
            ),
            "maximum_timing_dispersion_percent": MAX_TIMING_DISPERSION_PERCENT,
            "environment_overrides": environment_overrides,
        },
        "comparison_options": {
            "ignored_sdds_parameters": ignored,
            "ignored_sdds_columns": ignored_columns,
            "absolute_tolerance": args.absolute_tolerance,
            "relative_tolerance": args.relative_tolerance,
        },
        "timed_out_tests": timed_out,
        "tests": results,
        "comparisons": comparisons,
    }
    write_manifest(output / "manifest.json", manifest)
    write_performance_baseline_summary(output, manifest)
    report_text = "\n".join(report) + "\n"
    (output / "comparison.txt").write_text(report_text)
    if changed:
        print(f"Detected output changes in {len(changed)} test(s):")
        for comparison in changed:
            for change in comparison["changes"]:
                label = change["path"] or "test"
                print(f"  {comparison['name']}: {change['status']} {label}")
        print("See comparison.txt for the detailed SDDS differences.")
    else:
        print(report_text, end="")
    print(f"Comparison artifacts written to {output}")
    return 1 if changed else 0


def compare_existing_command(args: argparse.Namespace) -> int:
    require_commands(("sddsquery", "sddsdiff", "sdds2stream"))
    baseline_root = Path(args.baseline).expanduser().resolve()
    candidate_root = Path(args.candidate).expanduser().resolve()
    if baseline_root == candidate_root:
        print("Baseline and candidate refer to the same completed run.", flush=True)

    baseline = load_baseline(baseline_root)
    candidate = load_baseline(candidate_root)
    validate_existing_baselines(baseline, candidate)
    pre_change_root = None
    pre_change = None
    if args.pre_change_gpu:
        pre_change_root = Path(args.pre_change_gpu).expanduser().resolve()
        pre_change = load_baseline(pre_change_root)
        validate_existing_baselines(baseline, pre_change)
        validate_existing_baselines(pre_change, candidate)
    gpu_noise_absolute_tolerance, gpu_noise_relative_tolerance = gpu_noise_policy(
        baseline,
        args.gpu_noise_absolute_tolerance,
        args.gpu_noise_relative_tolerance,
    )

    output = Path(args.output).expanduser().resolve()
    create_artifact_root(output)
    ignored, ignored_columns = comparison_ignore_lists(args, baseline, candidate)
    print(
        f"Comparing {len(baseline['tests'])} completed tests from\n"
        f"  baseline:  {baseline_root}\n"
        + (f"  pre-change GPU: {pre_change_root}\n" if pre_change_root else "")
        + f"  candidate: {candidate_root}",
        flush=True,
    )
    comparisons, _exact_report = compare_outputs(
        baseline_root,
        baseline,
        candidate_root,
        candidate["tests"],
        ignored_parameters=ignored,
        ignored_columns=ignored_columns,
        absolute_tolerance=args.absolute_tolerance,
        relative_tolerance=args.relative_tolerance,
    )
    assessment_summary = assess_existing_comparisons(
        comparisons,
        baseline_root,
        baseline,
        candidate_root,
        candidate,
        gpu_noise_absolute_tolerance=gpu_noise_absolute_tolerance,
        gpu_noise_relative_tolerance=gpu_noise_relative_tolerance,
        ignored_parameters=ignored,
        ignored_columns=ignored_columns,
    )
    report = existing_comparison_report(
        comparisons,
        assessment_summary,
        gpu_noise_absolute_tolerance=gpu_noise_absolute_tolerance,
        gpu_noise_relative_tolerance=gpu_noise_relative_tolerance,
    )
    performance_enabled, minimum_speedup, require_gpu_activity = performance_policy(
        baseline, args.minimum_speedup
    )
    performance_enabled = performance_enabled or pre_change is not None
    performance = None
    if performance_enabled:
        performance = assess_runtime_performance(
            baseline,
            candidate,
            minimum_speedup=minimum_speedup,
            require_gpu_activity=require_gpu_activity,
            preceding_gpu_manifest=pre_change,
            target_tests=set(args.target_test) if args.target_test else None,
            non_target_regression_limit=args.non_target_regression_limit,
            suite_regression_limit=args.suite_regression_limit,
        )
        report.extend(runtime_performance_report(performance))
    changed = [item for item in comparisons if item["status"] == "changed"]
    expected_roundoff = [
        item
        for item in comparisons
        if item["status"] == "probably_expected_gpu_roundoff"
    ]
    manifest = {
        "format_version": FORMAT_VERSION,
        "mode": "existing-comparison",
        "created_at": utc_timestamp(),
        "complete": not changed and (performance is None or performance["complete"]),
        "baseline": {
            "path": str(baseline_root),
            "created_at": baseline.get("created_at"),
            "executable": baseline.get("executable"),
        },
        "candidate": {
            "path": str(candidate_root),
            "created_at": candidate.get("created_at"),
            "executable": candidate.get("executable"),
        },
        "pre_change_gpu": (
            {
                "path": str(pre_change_root),
                "created_at": pre_change.get("created_at"),
                "executable": pre_change.get("executable"),
            }
            if pre_change is not None
            else None
        ),
        "test_set": {
            key: value for key, value in baseline["test_set"].items() if key != "path"
        },
        "comparison_options": {
            "ignored_sdds_parameters": ignored,
            "ignored_sdds_columns": ignored_columns,
            "absolute_tolerance": args.absolute_tolerance,
            "relative_tolerance": args.relative_tolerance,
        },
        "gpu_significance_assessment": {
            "version": GPU_ASSESSMENT_VERSION,
            "absolute_tolerance": gpu_noise_absolute_tolerance,
            "relative_tolerance": gpu_noise_relative_tolerance,
            "rule": (
                "abs(cpu-gpu) <= max(absolute_tolerance, "
                "relative_tolerance * max(abs(cpu), abs(gpu)))"
            ),
            "summary": assessment_summary,
        },
        "comparisons": comparisons,
    }
    if performance is not None:
        manifest["performance_comparison"] = performance
    write_manifest(output / "manifest.json", manifest)
    report_text = "\n".join(report) + "\n"
    (output / "comparison.txt").write_text(report_text)
    if expected_roundoff:
        print(
            "Classified exact floating-point differences as probable GPU "
            f"roundoff in {len(expected_roundoff)} test(s)."
        )
    if changed:
        print(f"Potentially significant output changes in {len(changed)} test(s):")
        for comparison in changed:
            count = sum(
                change["assessment"]["classification"] == "potentially_significant"
                for change in comparison["changes"]
            )
            suffix = "change" if count == 1 else "changes"
            print(f"  {comparison['name']}: {count} {suffix}")
        print("See comparison.txt for the significance assessment and details.")
    else:
        print(
            "No potentially significant output changes detected. "
            "See comparison.txt for the assessment."
        )
    if performance is not None:
        performance_summary = performance["summary"]
        print(
            f"Runtime: {performance_summary['tests_passing_gates']}/"
            f"{performance_summary['tests']} test(s) met the "
            "target/guard performance gates; "
            f"total speedup {performance_summary['total_speedup']:.3f}x."
        )
    print(f"Comparison report written to {output}")
    performance_failed = performance is not None and not performance["complete"]
    return 1 if changed or performance_failed else 0


def parallel_jobs(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= MAX_PARALLEL_TESTS:
        raise argparse.ArgumentTypeError(f"must be between 1 and {MAX_PARALLEL_TESTS}")
    return parsed


def test_timeout(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed <= MAX_TEST_SECONDS:
        raise argparse.ArgumentTypeError(
            f"must be greater than zero and no more than {MAX_TEST_SECONDS} seconds"
        )
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def fraction(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def warmup_count(value: str) -> int:
    parsed = int(value)
    if not 0 <= parsed <= MAX_WARMUP_RUNS:
        raise argparse.ArgumentTypeError(f"must be between 0 and {MAX_WARMUP_RUNS}")
    return parsed


def repetition_count(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= MAX_REPETITIONS:
        raise argparse.ArgumentTypeError(f"must be between 1 and {MAX_REPETITIONS}")
    return parsed


def add_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--test-set",
        default=os.environ.get("ELEGANT_TEST_SET", "elegantTestSet"),
        help=(
            "clean elegantTestSet SVN working copy or a fingerprinted local "
            "suite such as src/gpu/test-set (default: %(default)s)"
        ),
    )
    parser.add_argument("--elegant", required=True, help="elegant executable to run")
    parser.add_argument(
        "--output", required=True, help="new artifact directory to create"
    )
    parser.add_argument(
        "--jobs",
        type=parallel_jobs,
        default=DEFAULT_JOBS,
        help=(
            f"tests to run concurrently, up to {MAX_PARALLEL_TESTS} "
            f"(default: {DEFAULT_JOBS})"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=test_timeout,
        default=DEFAULT_TEST_SECONDS,
        help=(
            f"per-test timeout in seconds; the hard maximum is {MAX_TEST_SECONDS} "
            f"(default: {DEFAULT_TEST_SECONDS})"
        ),
    )
    parser.add_argument(
        "--keep-work", action="store_true", help="retain exported test work directories"
    )
    parser.add_argument(
        "--warmup-runs",
        type=warmup_count,
        default=None,
        help=(
            "unmeasured warm-up runs per test (default: 1 for GPU performance suites, "
            "otherwise 0)"
        ),
    )
    parser.add_argument(
        "--repetitions",
        type=repetition_count,
        default=None,
        help=(
            "measured runs per test; manifests store samples, median, and MAD "
            "(default: 5 for GPU performance suites, otherwise 1)"
        ),
    )


def add_comparison_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--absolute-tolerance",
        type=nonnegative_float,
        default=0,
        help="allowed absolute SDDS numeric difference (default: exact)",
    )
    parser.add_argument(
        "--relative-tolerance",
        type=nonnegative_float,
        default=0,
        help="allowed relative SDDS numeric difference (default: exact)",
    )
    parser.add_argument(
        "--ignore-parameter",
        action="append",
        default=[],
        metavar="GLOB",
        help="additional SDDS parameter glob to ignore; repeat as needed",
    )
    parser.add_argument(
        "--ignore-column",
        action="append",
        default=[],
        metavar="GLOB",
        help="additional SDDS column glob to ignore; repeat as needed",
    )


def add_gpu_assessment_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--gpu-noise-absolute-tolerance",
        type=nonnegative_float,
        default=None,
        help=(
            "near-zero absolute floor for probable GPU roundoff "
            "(default: suite value or "
            f"{DEFAULT_GPU_NOISE_ABSOLUTE_TOLERANCE:g})"
        ),
    )
    parser.add_argument(
        "--gpu-noise-relative-tolerance",
        type=nonnegative_float,
        default=None,
        help=(
            "relative envelope for probable GPU roundoff "
            "(default: suite value or "
            f"{DEFAULT_GPU_NOISE_RELATIVE_TOLERANCE:g})"
        ),
    )
    parser.add_argument(
        "--minimum-speedup",
        type=nonnegative_float,
        default=None,
        help=(
            "minimum baseline/candidate runtime ratio for every test; local "
            "GPU performance suites provide their own default"
        ),
    )


def gui_subprocess_command(
    *,
    script: Path,
    operation: str,
    output: str,
    test_set: str = "",
    executable: str = "",
    jobs: int = DEFAULT_JOBS,
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
    repetitions: int = DEFAULT_REPETITIONS,
    keep_work: bool = False,
    baseline: str = "",
    candidate: str = "",
    pre_change_gpu: str = "",
    include_excluded: bool = False,
    tests: str = "",
) -> list[str]:
    command = [sys.executable, str(script), operation, "--output", output]
    if operation == "compare-existing":
        command.extend(["--baseline", baseline, "--candidate", candidate])
        if pre_change_gpu:
            command.extend(["--pre-change-gpu", pre_change_gpu])
        for test in shlex.split(tests):
            command.extend(["--target-test", test])
        return command
    if operation not in {"baseline", "compare"}:
        raise RegressionError(f"unsupported GUI operation: {operation}")

    command.extend(
        [
            "--test-set",
            test_set,
            "--elegant",
            executable,
            "--jobs",
            str(jobs),
            "--warmup-runs",
            str(warmup_runs),
            "--repetitions",
            str(repetitions),
        ]
    )
    if keep_work:
        command.append("--keep-work")
    if operation == "compare":
        command.extend(["--baseline", baseline])
    else:
        if include_excluded:
            command.append("--include-excluded")
        if tests:
            command.extend(shlex.split(tests))
    return command


def gui_command(args: argparse.Namespace) -> int:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except ImportError as exc:
        raise RegressionError("Tkinter is required for the GUI") from exc

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise RegressionError(f"unable to open the GUI: {exc}") from exc

    repository = Path(__file__).resolve().parents[3]
    local_executables = sorted((repository / "bin").glob("*/elegant"))
    default_executable = str(local_executables[0]) if local_executables else ""

    root.title("elegant regression baseline comparison")
    root.geometry("960x700")
    root.minsize(800, 560)
    root.columnconfigure(1, weight=1)
    root.rowconfigure(11, weight=1)

    mode = tk.StringVar(value="baseline")
    test_set = tk.StringVar(
        value=os.environ.get("ELEGANT_TEST_SET", str(repository / "elegantTestSet"))
    )
    executable = tk.StringVar(value=default_executable)
    output = tk.StringVar()
    baseline = tk.StringVar()
    candidate = tk.StringVar()
    pre_change_gpu = tk.StringVar()
    tests = tk.StringVar()
    jobs = tk.IntVar(value=DEFAULT_JOBS)
    warmup_runs = tk.IntVar(value=DEFAULT_WARMUP_RUNS)
    repetitions = tk.IntVar(value=DEFAULT_REPETITIONS)
    keep_work = tk.BooleanVar(value=False)
    include_excluded = tk.BooleanVar(value=False)
    operation_note = tk.StringVar(
        value=f"Each test is terminated after {DEFAULT_TEST_SECONDS // 60} minutes"
    )
    status = tk.StringVar(value="Ready")
    messages: queue.Queue[tuple[str, Any]] = queue.Queue()
    running_process: dict[str, subprocess.Popen[str] | None] = {"value": None}

    padding = {"padx": 8, "pady": 5}
    mode_frame = ttk.LabelFrame(root, text="Operation")
    mode_frame.grid(row=0, column=0, columnspan=3, sticky="ew", **padding)
    ttk.Radiobutton(
        mode_frame, text="Create baseline", variable=mode, value="baseline"
    ).pack(side="left", padx=8, pady=5)
    ttk.Radiobutton(
        mode_frame, text="Compare candidate", variable=mode, value="compare"
    ).pack(side="left", padx=8, pady=5)
    ttk.Radiobutton(
        mode_frame,
        text="Compare completed runs",
        variable=mode,
        value="compare-existing",
    ).pack(side="left", padx=8, pady=5)

    def add_path_row(
        row: int,
        label: str,
        variable: Any,
        browse: Any,
    ) -> tuple[Any, Any]:
        ttk.Label(root, text=label).grid(row=row, column=0, sticky="w", **padding)
        entry = ttk.Entry(root, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", **padding)
        button = ttk.Button(root, text="Browse…", command=browse)
        button.grid(row=row, column=2, sticky="ew", **padding)
        return entry, button

    def choose_test_set() -> None:
        selected = filedialog.askdirectory(
            title="Choose elegantTestSet or a local GPU test set",
            initialdir=test_set.get() or str(repository),
        )
        if selected:
            test_set.set(selected)
            try:
                suite = local_suite_configuration(Path(selected))
                if suite.get("recommended_jobs"):
                    jobs.set(suite["recommended_jobs"])
                if suite.get("suite_type") == GPU_PERFORMANCE_SUITE_TYPE:
                    warmup_runs.set(GPU_PERFORMANCE_WARMUP_RUNS)
                    repetitions.set(GPU_PERFORMANCE_REPETITIONS)
                else:
                    warmup_runs.set(DEFAULT_WARMUP_RUNS)
                    repetitions.set(DEFAULT_REPETITIONS)
            except RegressionError as exc:
                messagebox.showerror("Invalid test set", str(exc))

    def choose_executable() -> None:
        selected = filedialog.askopenfilename(
            title="Choose elegant or gpu-elegant executable",
            initialdir=(
                str(Path(executable.get()).parent)
                if executable.get()
                else str(repository)
            ),
        )
        if selected:
            executable.set(selected)

    def choose_output() -> None:
        selected = filedialog.asksaveasfilename(
            title="Choose a new artifact directory",
            initialdir=(
                str(Path(output.get()).parent)
                if output.get()
                else str(repository.parent)
            ),
        )
        if selected:
            output.set(selected)

    def choose_baseline() -> None:
        selected = filedialog.askdirectory(
            title="Choose baseline artifact directory",
            initialdir=baseline.get() or str(repository.parent),
        )
        if selected:
            baseline.set(selected)

    def choose_candidate() -> None:
        selected = filedialog.askdirectory(
            title="Choose candidate artifact directory",
            initialdir=candidate.get() or baseline.get() or str(repository.parent),
        )
        if selected:
            candidate.set(selected)

    def choose_pre_change_gpu() -> None:
        selected = filedialog.askdirectory(
            title="Choose pre-change gpu-elegant artifact directory",
            initialdir=(
                pre_change_gpu.get()
                or candidate.get()
                or baseline.get()
                or str(repository.parent)
            ),
        )
        if selected:
            pre_change_gpu.set(selected)

    test_set_entry, test_set_button = add_path_row(
        1, "Test set", test_set, choose_test_set
    )
    executable_entry, executable_button = add_path_row(
        2, "Executable", executable, choose_executable
    )
    add_path_row(3, "New output directory", output, choose_output)
    baseline_entry, baseline_button = add_path_row(
        4, "Baseline directory", baseline, choose_baseline
    )
    candidate_entry, candidate_button = add_path_row(
        5, "Candidate artifact directory", candidate, choose_candidate
    )
    pre_change_entry, pre_change_button = add_path_row(
        6, "Pre-change GPU directory", pre_change_gpu, choose_pre_change_gpu
    )

    tests_label = ttk.Label(root, text="Focused tests")
    tests_label.grid(row=7, column=0, sticky="w", **padding)
    tests_entry = ttk.Entry(root, textvariable=tests)
    tests_entry.grid(row=7, column=1, sticky="ew", **padding)
    tests_hint = ttk.Label(root, text="space-separated; blank runs all")
    tests_hint.grid(row=7, column=2, sticky="w", **padding)

    options = ttk.Frame(root)
    options.grid(row=8, column=0, columnspan=3, sticky="ew", **padding)
    ttk.Label(options, text="Concurrent tests").pack(side="left")
    jobs_spinbox = ttk.Spinbox(
        options,
        from_=1,
        to=MAX_PARALLEL_TESTS,
        textvariable=jobs,
        width=4,
        state="readonly",
    )
    jobs_spinbox.pack(side="left", padx=(6, 20))
    ttk.Label(options, text="Warm-ups").pack(side="left")
    warmup_spinbox = ttk.Spinbox(
        options,
        from_=0,
        to=MAX_WARMUP_RUNS,
        textvariable=warmup_runs,
        width=3,
        state="readonly",
    )
    warmup_spinbox.pack(side="left", padx=(6, 12))
    ttk.Label(options, text="Measured runs").pack(side="left")
    repetitions_spinbox = ttk.Spinbox(
        options,
        from_=1,
        to=MAX_REPETITIONS,
        textvariable=repetitions,
        width=3,
        state="readonly",
    )
    repetitions_spinbox.pack(side="left", padx=(6, 20))
    keep_work_check = ttk.Checkbutton(
        options, text="Keep exported work", variable=keep_work
    )
    keep_work_check.pack(side="left")
    include_excluded_check = ttk.Checkbutton(
        options,
        text=f"Include {len(DEFAULT_EXCLUDED_TESTS)} excluded tests",
        variable=include_excluded,
    )
    include_excluded_check.pack(side="left", padx=(20, 0))
    ttk.Label(options, textvariable=operation_note).pack(side="right")

    run_button = ttk.Button(root, text="Run")
    run_button.grid(row=9, column=0, sticky="w", **padding)
    ttk.Label(root, textvariable=status).grid(
        row=9, column=1, columnspan=2, sticky="w", **padding
    )

    log_frame = ttk.LabelFrame(root, text="Run output")
    log_frame.grid(row=11, column=0, columnspan=3, sticky="nsew", **padding)
    log_frame.columnconfigure(0, weight=1)
    log_frame.rowconfigure(0, weight=1)
    log = tk.Text(log_frame, wrap="none", state="disabled")
    log.grid(row=0, column=0, sticky="nsew")
    scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=log.yview)
    scrollbar.grid(row=0, column=1, sticky="ns")
    log.configure(yscrollcommand=scrollbar.set)

    def append_log(text: str) -> None:
        log.configure(state="normal")
        log.insert("end", text)
        log.see("end")
        log.configure(state="disabled")

    def set_path_state(entry: Any, button: Any, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        entry.configure(state=state)
        button.configure(state=state)

    def update_mode(*_unused: Any) -> None:
        operation = mode.get()
        comparing = operation == "compare"
        comparing_existing = operation == "compare-existing"
        running_tests = not comparing_existing
        set_path_state(test_set_entry, test_set_button, running_tests)
        set_path_state(executable_entry, executable_button, running_tests)
        set_path_state(baseline_entry, baseline_button, comparing or comparing_existing)
        set_path_state(candidate_entry, candidate_button, comparing_existing)
        set_path_state(pre_change_entry, pre_change_button, comparing_existing)
        tests_entry.configure(
            state=(
                "normal"
                if operation in {"baseline", "compare-existing"}
                else "disabled"
            )
        )
        tests_label.configure(
            text="Target tests" if comparing_existing else "Focused tests"
        )
        tests_hint.configure(
            text=(
                "space-separated; blank targets all"
                if comparing_existing
                else "space-separated; blank runs all"
            )
        )
        jobs_spinbox.configure(state="readonly" if running_tests else "disabled")
        warmup_spinbox.configure(state="readonly" if running_tests else "disabled")
        repetitions_spinbox.configure(state="readonly" if running_tests else "disabled")
        keep_work_check.configure(state="normal" if running_tests else "disabled")
        include_excluded_check.configure(
            state="normal" if operation == "baseline" else "disabled"
        )
        operation_note.set(
            f"Each test is terminated after {DEFAULT_TEST_SECONDS // 60} minutes"
            if running_tests
            else (
                "GPU screening (suite metadata may override defaults): "
                f"abs {DEFAULT_GPU_NOISE_ABSOLUTE_TOLERANCE:g}, "
                f"rel {DEFAULT_GPU_NOISE_RELATIVE_TOLERANCE:g}; no tests are rerun"
            )
        )

    def watch_process(process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            messages.put(("output", line))
        messages.put(("done", process.wait()))

    def start_run() -> None:
        if running_process["value"] is not None:
            return
        operation = mode.get()
        comparing_existing = operation == "compare-existing"
        job_count = DEFAULT_JOBS
        warmup_run_count = DEFAULT_WARMUP_RUNS
        repetition_run_count = DEFAULT_REPETITIONS
        if not comparing_existing:
            try:
                job_count = int(jobs.get())
                warmup_run_count = int(warmup_runs.get())
                repetition_run_count = int(repetitions.get())
            except (ValueError, tk.TclError):
                messagebox.showerror(
                    "Invalid timing options",
                    "Jobs, warm-ups, and measured runs must be integers.",
                )
                return
            if not 1 <= job_count <= MAX_PARALLEL_TESTS:
                messagebox.showerror(
                    "Invalid jobs", "Concurrent tests must be 1 through 8."
                )
                return
            if not 0 <= warmup_run_count <= MAX_WARMUP_RUNS:
                messagebox.showerror(
                    "Invalid warm-ups",
                    f"Warm-up runs must be 0 through {MAX_WARMUP_RUNS}.",
                )
                return
            if not 1 <= repetition_run_count <= MAX_REPETITIONS:
                messagebox.showerror(
                    "Invalid measured runs",
                    f"Measured runs must be 1 through {MAX_REPETITIONS}.",
                )
                return

        required = {"New output directory": output.get().strip()}
        if comparing_existing:
            required["Baseline directory"] = baseline.get().strip()
            required["Candidate artifact directory"] = candidate.get().strip()
        else:
            required["Test set"] = test_set.get().strip()
            required["Executable"] = executable.get().strip()
            if operation == "compare":
                required["Baseline directory"] = baseline.get().strip()
        missing = [label for label, value in required.items() if not value]
        if missing:
            messagebox.showerror("Missing value", "Required: " + ", ".join(missing))
            return

        try:
            command = gui_subprocess_command(
                script=Path(__file__).resolve(),
                operation=operation,
                output=output.get().strip(),
                test_set=test_set.get().strip(),
                executable=executable.get().strip(),
                jobs=job_count,
                warmup_runs=warmup_run_count,
                repetitions=repetition_run_count,
                keep_work=keep_work.get(),
                baseline=baseline.get().strip(),
                candidate=candidate.get().strip(),
                pre_change_gpu=pre_change_gpu.get().strip(),
                include_excluded=include_excluded.get(),
                tests=tests.get().strip(),
            )
        except (RegressionError, ValueError) as exc:
            messagebox.showerror("Invalid operation", str(exc))
            return

        append_log(f"$ {shlex.join(command)}\n")
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                bufsize=1,
            )
        except OSError as exc:
            messagebox.showerror("Unable to start", str(exc))
            return
        running_process["value"] = process
        run_button.configure(state="disabled")
        status.set("Comparing completed runs…" if comparing_existing else "Running…")
        threading.Thread(target=watch_process, args=(process,), daemon=True).start()

    def drain_messages() -> None:
        try:
            while True:
                kind, value = messages.get_nowait()
                if kind == "output":
                    append_log(value)
                else:
                    running_process["value"] = None
                    run_button.configure(state="normal")
                    status.set(f"Finished with exit status {value}")
                    if value == 0:
                        messagebox.showinfo(
                            "Operation complete",
                            "The operation completed successfully.",
                        )
                    else:
                        messagebox.showwarning(
                            "Run needs review",
                            f"The run exited with status {value}. Review the output artifacts.",
                        )
        except queue.Empty:
            pass
        root.after(100, drain_messages)

    def close_window() -> None:
        if running_process["value"] is not None:
            messagebox.showwarning(
                "Run in progress", "Wait for the current regression run to finish."
            )
            return
        root.destroy()

    run_button.configure(command=start_run)
    mode.trace_add("write", update_mode)
    update_mode()
    root.protocol("WM_DELETE_WINDOW", close_window)
    root.after(100, drain_messages)
    root.mainloop()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create and compare output baselines for elegantTestSet."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline = subparsers.add_parser(
        "baseline", help="run tests and create a baseline output set"
    )
    add_run_options(baseline)
    baseline.add_argument(
        "--include-excluded",
        action="store_true",
        help=(
            f"include the {len(DEFAULT_EXCLUDED_TESTS)} known-problem tests that "
            "are skipped by default"
        ),
    )
    baseline.add_argument(
        "tests",
        nargs="*",
        help="test directory names (default: all discovered runnable tests)",
    )
    baseline.set_defaults(handler=baseline_command)

    compare = subparsers.add_parser(
        "compare", help="run a candidate elegant and compare it with a baseline"
    )
    add_run_options(compare)
    compare.add_argument(
        "--baseline", required=True, help="baseline artifact directory"
    )
    add_comparison_options(compare)
    compare.set_defaults(handler=compare_command)

    compare_existing = subparsers.add_parser(
        "compare-existing",
        help=(
            "compare completed CPU/GPU artifact directories and assess "
            "significance without rerunning"
        ),
    )
    compare_existing.add_argument(
        "--baseline", required=True, help="completed baseline artifact directory"
    )
    compare_existing.add_argument(
        "--candidate", required=True, help="completed candidate artifact directory"
    )
    compare_existing.add_argument(
        "--pre-change-gpu",
        help=(
            "completed immediately preceding gpu-elegant run; enables the second "
            "2x gate and non-target regression guards"
        ),
    )
    compare_existing.add_argument(
        "--output", required=True, help="new comparison report directory to create"
    )
    add_comparison_options(compare_existing)
    add_gpu_assessment_options(compare_existing)
    compare_existing.add_argument(
        "--target-test",
        action="append",
        default=[],
        metavar="NAME",
        help=("test subject to both 2x gates; repeat as needed (default: all tests)"),
    )
    compare_existing.add_argument(
        "--non-target-regression-limit",
        type=fraction,
        default=DEFAULT_NON_TARGET_REGRESSION_LIMIT,
        metavar="FRACTION",
        help=(
            "maximum slowdown versus pre-change GPU for guard tests "
            f"(default: {DEFAULT_NON_TARGET_REGRESSION_LIMIT:g})"
        ),
    )
    compare_existing.add_argument(
        "--suite-regression-limit",
        type=fraction,
        default=DEFAULT_SUITE_REGRESSION_LIMIT,
        metavar="FRACTION",
        help=(
            "maximum total-suite slowdown versus pre-change GPU "
            f"(default: {DEFAULT_SUITE_REGRESSION_LIMIT:g})"
        ),
    )
    compare_existing.set_defaults(handler=compare_existing_command)

    gui = subparsers.add_parser("gui", help="open the graphical baseline launcher")
    gui.set_defaults(handler=gui_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        configure_temp_directory()
        args = build_parser().parse_args(argv)
        return args.handler(args)
    except RegressionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
