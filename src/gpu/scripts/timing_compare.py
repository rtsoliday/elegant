#!/usr/bin/env python3
"""Timing comparison of elegant, gpu-elegant and gpu-elegant with host threads.

The procedure is described in GPU-Testing/TIMING-PROCEDURE.md.  In short:

    timing_compare.py run    --cases CASES.json --output RESULTS_DIR
    timing_compare.py report --output RESULTS_DIR

"run" times every case of the list under three configurations, one job at a time on fixed cores,
once each (more with --repeats), and compares the output files of the configurations.  It can be stopped and
started again: finished runs are kept in RESULTS_DIR/results.json and skipped.  "report" writes
RESULTS_DIR/REPORT.md from that file.

Configurations:
    elegant       the CPU program, one core
    gpu           gpu-elegant, one core
    gpu-threads   gpu-elegant -ompThreads=N, N cores
"""
import argparse
import datetime
import hashlib
import json
import os
import platform
import resource
import threading
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import elegant_test_regression as harness  # the comparison rules of the regression harness

CONFIGS = ("elegant", "gpu", "gpu-threads")
REPO = Path(__file__).resolve().parents[3]
RUNTIME_LIBS = "/local/oagmgr-rhel8/runtime-libs/elegant-rhel8"


def load_cases(path):
    data = json.load(open(path))
    for case in data["cases"]:
        root, _, name = case["source"].partition(":")
        case["directory"] = str(Path(data["roots"][root]) / name)
        if not Path(case["directory"]).is_dir():
            sys.exit("case %s: no directory %s" % (case["name"], case["directory"]))
        if "commands" not in case:
            found = sorted(p.name for p in Path(case["directory"]).glob("*.ele"))
            case["commands"] = ["run.ele"] if "run.ele" in found else found
        for command in case["commands"]:
            if not (Path(case["directory"]) / command).is_file():
                sys.exit("case %s: no input file %s in %s" % (case["name"], command, case["directory"]))
    names = [case["name"] for case in data["cases"]]
    if len(set(names)) != len(names):
        sys.exit("the case list has a name twice")
    return data


def selected(case, only):
    return not only or case["name"] in only or case["group"] in only


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def command_output(command):
    try:
        return subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                              text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def environment_record(args):
    cpu = ""
    for line in open("/proc/cpuinfo"):
        if line.startswith("model name"):
            cpu = line.split(":", 1)[1].strip()
            break
    status = command_output(["git", "-C", str(REPO), "status", "--porcelain"])
    return {
        "date": datetime.date.today().isoformat(),
        "host": platform.node(),
        "cpu": cpu,
        "gpu": command_output(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]),
        "commit": command_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"]),
        "locally_changed_files": len([line for line in status.splitlines() if line.strip()]),
        "elegant": {"path": args.elegant, "sha256": sha256(args.elegant)},
        "gpu_elegant": {"path": args.gpu_elegant, "sha256": sha256(args.gpu_elegant)},
        "threads": args.threads,
        "single_core": args.single_core,
        "thread_cores": args.thread_cores,
        "repeats": args.repeats,
        "more_repeats": args.more_repeats,
    }


def run_once(case, config, args, work):
    """One timed run in a fresh copy of the case.  Returns its record; the copy is left in work."""
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(case["directory"], work, symlinks=True,
                    ignore=shutil.ignore_patterns("reference", "preference", "greference"))
    env = dict(os.environ, LD_LIBRARY_PATH=RUNTIME_LIBS)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env.pop(name, None)
    env.update(case.get("environment", {}))      # settings one case needs, such as a forced GPU threshold
    executable = args.elegant if config == "elegant" else args.gpu_elegant
    extra = ["-ompThreads=%d" % args.threads] if config == "gpu-threads" else []
    cores = args.thread_cores if config == "gpu-threads" else args.single_core
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.time()
    status = 0
    with open(work / "timing.log", "w") as log:
        for command in case["commands"]:
            # A plain wait() blocks until the program ends.  subprocess.run(timeout=...) polls for it
            # with sleeps of up to 50 ms, and the wall time then comes out up to 50 ms too long.
            process = subprocess.Popen(["taskset", "-c", cores, executable, command] + extra, cwd=work,
                                       env=env, stdout=log, stderr=subprocess.STDOUT)
            timed_out = []
            timer = threading.Timer(args.timeout, lambda: (timed_out.append(1), process.kill()))
            timer.start()
            try:
                status = process.wait()
            finally:
                timer.cancel()
            if timed_out:
                status = -1
            if status:
                break
    wall = time.time() - start
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {"wall": wall, "cpu": after.ru_utime - before.ru_utime + after.ru_stime - before.ru_stime,
            "status": status}


def output_files(case, kept):
    """SDDS files a run made: those in its directory that the case's own directory doesn't have."""
    names = []
    for path in sorted(kept.iterdir()):
        if not path.is_file() or (Path(case["directory"]) / path.name).exists() or path.name == "timing.log":
            continue
        with open(path, "rb") as stream:
            if stream.read(4) == b"SDDS":
                names.append(path.name)
    return names


def compare_outputs(case, baseline, candidate):
    """identical, rounding (inside the harness's GPU envelope) or review, with the files that decide it"""
    a, b = set(output_files(case, baseline)), set(output_files(case, candidate))
    if not a or a != b:
        return {"result": "review", "detail": "the two runs made different sets of output files"
                if a != b else "no SDDS output files"}
    rounding, review = [], []
    for name in sorted(a):
        try:
            same, _ = harness.compare_sdds(
                baseline / name, candidate / name,
                ignored_parameters=list(harness.DEFAULT_IGNORED_PARAMETERS),
                ignored_columns=list(harness.DEFAULT_IGNORED_COLUMNS),
                absolute_tolerance=0, relative_tolerance=0)
            if same:
                continue
            assessment = harness.assess_sdds_gpu_difference(
                baseline / name, candidate / name,
                absolute_tolerance=harness.DEFAULT_GPU_NOISE_ABSOLUTE_TOLERANCE, relative_tolerance=1e-9,
                ignored_parameters=harness.DEFAULT_IGNORED_PARAMETERS,
                ignored_columns=harness.DEFAULT_IGNORED_COLUMNS)
            if assessment["classification"] == "probably_expected_gpu_roundoff":
                rounding.append(name)
            else:
                review.append(name)
        except harness.RegressionError as error:
            review.append("%s (%s)" % (name, str(error).splitlines()[0][:80]))
    if review:
        return {"result": "review", "files": len(a), "detail": ", ".join(review)}
    if rounding:
        return {"result": "rounding", "files": len(a), "detail": ", ".join(rounding)}
    return {"result": "identical", "files": len(a), "detail": ""}


def median(runs, key):
    values = [run[key] for run in runs if run["status"] == 0]
    return statistics.median(values) if values else None


def close_to_one(results, case):
    """The configurations of this case that are in a speed ratio within 10% of 1, which asks for more runs"""
    walls = {config: median(results.get(case["name"], {}).get(config, {}).get("runs", []), "wall")
             for config in CONFIGS}
    close = set()
    for a, b in (("elegant", "gpu"), ("gpu", "gpu-threads")):
        if walls[a] and walls[b] and 0.9 <= walls[a] / walls[b] <= 1.1:
            close.update((a, b))
    return close


def do_run(args):
    data = load_cases(args.cases)
    output = Path(args.output).resolve()
    (output / "work").mkdir(parents=True, exist_ok=True)
    (output / "outputs").mkdir(exist_ok=True)
    results_path = output / "results.json"
    state = json.load(open(results_path)) if results_path.exists() else {"cases": {}}
    state["environment"] = environment_record(args)
    state["groups"] = data["groups"]
    state["case_list"] = [{k: c.get(k, "") for k in ("name", "group", "note", "source")} for c in data["cases"]]
    results = state["cases"]
    only = set(args.only or [])
    configs = [c for c in CONFIGS if c in args.configs.split(",")]

    def save():
        temporary = results_path.with_suffix(".tmp")
        json.dump(state, open(temporary, "w"), indent=1)
        temporary.replace(results_path)

    def time_case(case, wanted_default, which=None):
        for config in configs:
            if which is not None and config not in which:
                continue
            record = results.setdefault(case["name"], {}).setdefault(config, {"runs": []})
            wanted = case.get("repeats", {}).get(config, wanted_default)
            while len(record["runs"]) < wanted:
                index = len(record["runs"]) + 1
                work = output / "work" / ("%s-%s" % (case["name"], config))
                print("%s  %-18s %-11s run %d of %d ..." % (
                    time.strftime("%H:%M:%S"), case["name"], config, index, wanted), end="", flush=True)
                run = run_once(case, config, args, work)
                print(" wall %.2f s, cpu %.2f s%s" % (
                    run["wall"], run["cpu"], "" if run["status"] == 0 else ", FAILED (status %d)" % run["status"]),
                      flush=True)
                record["runs"].append(run)
                kept = output / "outputs" / case["name"] / config
                if index == 1 or run["status"]:
                    # the first run's files are the ones compared; a failed run's log is kept too
                    target = kept if index == 1 else kept.with_name(config + "-failed-run%d" % index)
                    if target.exists():
                        shutil.rmtree(target)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(work), str(target))
                else:
                    shutil.rmtree(work)
                save()
                if run["status"]:
                    break

    for case in data["cases"]:
        if not selected(case, only):
            continue
        time_case(case, args.repeats)
        close = close_to_one(results, case) if args.more_repeats > args.repeats else set()
        if close:
            print("%s  %-18s a ratio is within 10%% of 1: %d runs of %s" % (
                time.strftime("%H:%M:%S"), case["name"], args.more_repeats, ", ".join(sorted(close))), flush=True)
            time_case(case, args.more_repeats, close)
        kept = output / "outputs" / case["name"]
        comparisons = results[case["name"]].setdefault("comparisons", {})
        for label, a, b in (("gpu_vs_elegant", "elegant", "gpu"), ("threads_vs_gpu", "gpu", "gpu-threads")):
            if (kept / a).is_dir() and (kept / b).is_dir() and label not in comparisons:
                comparisons[label] = compare_outputs(case, kept / a, kept / b)
                print("%s  %-18s outputs, %s: %s %s" % (
                    time.strftime("%H:%M:%S"), case["name"], label.replace("_", " "),
                    comparisons[label]["result"], comparisons[label]["detail"][:100]), flush=True)
                save()
    save()
    print("Results are in %s; make the report with: %s report --output %s" % (results_path, sys.argv[0], output))


def seconds(value):
    if value is None:
        return "failed"
    return "%.1f s" % value if value < 100 else "%.0f s" % value


def ratio(a, b):
    return "%.2fx" % (a / b) if a and b else ""


def spread(runs):
    values = sorted(run["wall"] for run in runs if run["status"] == 0)
    if len(values) < 2 or not values[0]:
        return ""
    return "%.0f%%" % (100 * (values[-1] - values[0]) / statistics.median(values))


def do_report(args):
    output = Path(args.output).resolve()
    state = json.load(open(output / "results.json"))
    env, results = state["environment"], state["cases"]
    lines = ["# Timing of elegant, gpu-elegant and gpu-elegant with host threads", "",
             "Made %s on %s by `src/gpu/scripts/timing_compare.py` from `%s`. "
             "Procedure: `GPU-Testing/TIMING-PROCEDURE.md`." % (env["date"], env["host"], output / "results.json"),
             ""]
    summary, flagged, failed = [], [], []
    tables = []
    single = all(len(record.get(c, {}).get("runs", [])) <= 1 for record in results.values() for c in CONFIGS)
    for group, title in state["groups"].items():
        rows, speed_gpu, speed_threads, totals = [], [], [], {config: 0.0 for config in CONFIGS}
        complete = True
        for case in state["case_list"]:
            if case["group"] != group or case["name"] not in results:
                continue
            record = results[case["name"]]
            wall = {c: median(record.get(c, {}).get("runs", []), "wall") for c in CONFIGS}
            cpu = {c: median(record.get(c, {}).get("runs", []), "cpu") for c in CONFIGS}
            runs = {c: record.get(c, {}).get("runs", []) for c in CONFIGS}
            for c in CONFIGS:
                if any(run["status"] for run in runs[c]):
                    failed.append("`%s`, %s" % (case["name"], c))
                if wall[c] is None:
                    complete = False
                else:
                    totals[c] += wall[c]
            if wall["elegant"] and wall["gpu"]:
                speed_gpu.append(wall["elegant"] / wall["gpu"])
            if wall["elegant"] and wall["gpu-threads"]:
                speed_threads.append(wall["elegant"] / wall["gpu-threads"])
            comparisons = record.get("comparisons", {})
            checks = []
            for label in ("gpu_vs_elegant", "threads_vs_gpu"):
                result = comparisons.get(label, {}).get("result", "not compared")
                checks.append(result)
                if result == "review" or (label == "threads_vs_gpu" and result == "rounding"):
                    flagged.append("`%s`, %s: %s (%s)" % (
                        case["name"], label.replace("_", " "), result, comparisons[label].get("detail", "")))
            rows.append("| `%s` | %s (%s) | %s (%s) | %s | %s (%s) | %s | %s | %s | %s | %s | %s |" % (
                case["name"],
                seconds(wall["elegant"]), seconds(cpu["elegant"]),
                seconds(wall["gpu"]), seconds(cpu["gpu"]), ratio(wall["elegant"], wall["gpu"]),
                seconds(wall["gpu-threads"]), seconds(cpu["gpu-threads"]),
                ratio(wall["elegant"], wall["gpu-threads"]), ratio(wall["gpu"], wall["gpu-threads"]),
                "/".join(str(len(runs[c])) for c in CONFIGS),
                "/".join(spread(runs[c]) or "-" for c in CONFIGS), checks[0], checks[1]))
        if not rows:
            continue
        tables += ["## Group %s: %s" % (group, title), "",
                   "Wall time, with CPU time in brackets%s." % (
                       "; one run each" if single else "; medians of the runs"), "",
                   "| Case | elegant | gpu-elegant | Speedup | gpu-elegant, %d threads | Speedup over elegant | "
                   "Gain from threads | Runs | Spread of wall time | Outputs, GPU against elegant | "
                   "Outputs, threads against one thread |" % env["threads"],
                   "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |"] + rows + [""]
        notes = [c for c in state["case_list"] if c["group"] == group and c.get("note") and c["name"] in results]
        if notes:
            tables += ["- `%s`: %s" % (c["name"], c["note"]) for c in notes] + [""]
        summary.append("| %s | %s | %d | %s | %s | %s | %s | %s |" % (
            group, title, len(rows),
            "%.2fx" % statistics.median(speed_gpu) if speed_gpu else "",
            "%.2fx" % statistics.median(speed_threads) if speed_threads else "",
            seconds(totals["elegant"]) if complete else "incomplete",
            seconds(totals["gpu"]) if complete else "incomplete",
            seconds(totals["gpu-threads"]) if complete else "incomplete"))
    lines += ["## Summary", "",
              "| Group | What it is | Cases | Median speedup, gpu-elegant | Median speedup, %d threads | "
              "Total, elegant | Total, gpu-elegant | Total, %d threads |" % (env["threads"], env["threads"]),
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"] + summary + [""]
    lines += ["The groups answer different questions and are not combined into one number.", ""]
    lines += ["## Things to look at", ""]
    if failed:
        lines += ["- Runs that failed: " + "; ".join(sorted(set(failed))) + "."]
    if flagged:
        lines += ["- Output differences to review:"] + ["  - " + item for item in flagged]
    if not failed and not flagged:
        lines += ["- No run failed, no GPU output is outside the rounding envelope, and every threaded run "
                  "gave the one-thread output."]
    lines += [""] + tables
    lines += ["## How it was run", "",
              "- One job at a time. elegant and gpu-elegant on core %s; gpu-elegant `-ompThreads=%d` on cores %s. "
              "`OMP_NUM_THREADS` and `MKL_NUM_THREADS` unset." % (env["single_core"], env["threads"], env["thread_cores"]),
              ("- One run of each case and configuration, so there is no measure of run-to-run variation: treat "
               "differences under 10% as no difference."
               if single else
               "- %d runs of each case and configuration%s, fewer where the case list says so. Times are medians; "
               "the spread is (longest - shortest) / median." % (
                   env["repeats"], ", %d when a speed ratio is within 10%% of 1" % env["more_repeats"]
                   if env["more_repeats"] > env["repeats"] else "")),
              "- Wall time is from starting the program to its exit, with reading input and writing output. "
              "CPU time is user plus system time of the process and its threads.",
              "- Outputs are those of each configuration's first run. \"identical\": every SDDS value the same, "
              "leaving out run times and memory use. \"rounding\": all differences within 1e-15 absolute or 1e-9 "
              "relative, the regression harness's GPU envelope. \"review\": something beyond that.",
              "", "## What was run", "",
              "| | |", "| --- | --- |",
              "| Host | %s |" % env["host"], "| CPU | %s |" % env["cpu"], "| GPU and driver | %s |" % env["gpu"],
              "| Source | commit `%s` with %d locally changed files |" % (env["commit"], env["locally_changed_files"]),
              "| elegant | `%s`, sha256 `%s` |" % (env["elegant"]["path"], env["elegant"]["sha256"][:16]),
              "| gpu-elegant | `%s`, sha256 `%s` |" % (env["gpu_elegant"]["path"], env["gpu_elegant"]["sha256"][:16]),
              ""]
    report = output / "REPORT.md"
    report.write_text("\n".join(lines))
    print("Wrote %s" % report)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run", help="time the cases and compare their outputs")
    run.add_argument("--cases", required=True, help="the case list (JSON)")
    run.add_argument("--output", required=True, help="results directory; an existing one is continued")
    run.add_argument("--only", nargs="*", help="case names or group letters to run (default: all)")
    run.add_argument("--configs", default=",".join(CONFIGS), help="comma-separated: " + ", ".join(CONFIGS))
    run.add_argument("--elegant", default=str(REPO / "bin/Linux-x86_64/elegant"))
    run.add_argument("--gpu-elegant", default=str(REPO / "bin/Linux-x86_64-gpu/gpu-elegant"))
    run.add_argument("--threads", type=int, default=8, help="host threads of the third configuration")
    run.add_argument("--single-core", default="2", help="core for the one-thread configurations")
    run.add_argument("--thread-cores", default="0-7", help="cores for the threaded configuration")
    run.add_argument("--repeats", type=int, default=1, help="runs of each case and configuration (default 1)")
    run.add_argument("--more-repeats", type=int, default=0,
                     help="if larger than --repeats: runs when a speed ratio is within 10%% of 1")
    run.add_argument("--timeout", type=float, default=6 * 3600, help="seconds allowed for one program run")
    run.set_defaults(function=do_run)
    report = sub.add_parser("report", help="write REPORT.md from results.json")
    report.add_argument("--output", required=True)
    report.set_defaults(function=do_report)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
