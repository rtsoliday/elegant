#!/usr/bin/env python3
"""Survey of many decks: where the time goes by element type, in elegant and gpu-elegant.

    element_survey.py list   --output LIST.json [--minimum 10]
    element_survey.py run    --list LIST.json --config elegant --core 2 --output RESULTS_DIR
    element_survey.py report --output RESULTS_DIR

"list" picks the elegantTestSet decks that took --minimum seconds or more in elegant or gpu-elegant in
an earlier complete run of the test harness (their times are in its manifest.json), and adds the decks
with a run.ele and no `reference` directory, which the harness never runs.  "run" runs every deck of
the list once in one configuration with `show_element_timing = 1`, on one core; it can be stopped and
started again.  Run it once per configuration, into the same RESULTS_DIR (--part splits the list so
that two cores can share a configuration; never run two gpu-elegant jobs at once).  "report" writes SURVEY.md:
the seconds by element type added up over all the decks, the time outside tracking split by the command
that used it, and each deck's own times.

In gpu-elegant a cheap element's work can be paid for at a later element, so its per-type times say
where the program waited.  Totals over many decks are less affected than single rows.

The procedure is described in GPU-Testing/TIMING-PROCEDURE.md.
"""
import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import timing_compare as tc
import predict_runtime as pr

TEST_SET = "/local/elegantTestSet"
CPU_RUN = "/local/GPU-Testing/results/2026-11-11_csbend-drift/cpu-660"
GPU_RUN = "/local/GPU-Testing/results/2026-11-11_csbend-drift/gpu-660"


def do_list(args):
    cpu = {t["name"]: t for t in json.load(open(Path(args.cpu_run) / "manifest.json"))["tests"]}
    gpu = {t["name"]: t for t in json.load(open(Path(args.gpu_run) / "manifest.json"))["tests"]}
    decks, skipped = [], []
    for name, test in sorted(cpu.items()):
        seconds = max(test.get("execution_seconds") or 0, (gpu.get(name) or {}).get("execution_seconds") or 0)
        if seconds < args.minimum:
            continue
        commands = test["commands"]
        if test["dependencies"] or any(os.path.basename(c[0]) != "elegant" or len(c) != 2 for c in commands):
            skipped.append(name)
            continue
        decks.append({"name": name, "directory": str(Path(args.test_set) / name),
                      "commands": [c[1] for c in commands], "group": "in the test harness"})
    for directory in sorted(Path(args.test_set).iterdir()):
        if (directory / "run.ele").is_file() and not (directory / "reference").is_dir():
            decks.append({"name": directory.name, "directory": str(directory), "commands": ["run.ele"],
                          "group": "not run by the harness"})
    json.dump({"decks": decks, "skipped": skipped, "minimum_seconds": args.minimum,
               "cpu_run": args.cpu_run, "gpu_run": args.gpu_run}, open(args.output, "w"), indent=1)
    print("%d decks (%d not run by the harness); skipped, for needing another deck or a script: %s"
          % (len(decks), sum(d["group"] != "in the test harness" for d in decks), ", ".join(skipped) or "none"))


def do_run(args):
    data = json.load(open(args.list))
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / ("results-%s-%s.json" % (args.config, args.part))
    results = json.load(open(path)) if path.exists() else {}
    (output / "logs" / args.config).mkdir(parents=True, exist_ok=True)
    args.single_core = args.thread_cores = args.core
    for deck in data["decks"]:
        name = deck["name"]
        if name in results:
            continue
        if args.part != "all" and (deck["group"] == "in the test harness") != (args.part == "harness"):
            continue
        source = output / "decks" / args.config / name
        work = output / "work" / args.config / name
        if source.exists():
            shutil.rmtree(source)
        shutil.copytree(deck["directory"], source, symlinks=True,
                        ignore=shutil.ignore_patterns("reference", "preference", "greference"))
        for command in deck["commands"]:
            text = (source / command).read_text(errors="replace")
            (source / command).write_text(pr.set_value(text, "run_setup", "show_element_timing", 1))
        record = tc.run_once({"directory": str(source), "commands": deck["commands"]}, args.config, args, work)
        record["elements"] = pr.element_times(work / "timing.log")
        record["group"] = deck["group"]
        shutil.copy(work / "timing.log", output / "logs" / args.config / (name + ".log"))
        shutil.rmtree(work)
        shutil.rmtree(source)
        results[name] = record
        json.dump(results, open(path, "w"), indent=1)
        print("%-28s %-8s wall %8.2f s%s" % (name, args.config, record["wall"],
                                              "" if record["status"] == 0 else
                                              ("   TIMED OUT" if record["status"] == -1 else "   FAILED, status %d" % record["status"])),
              flush=True)
    print("Results are in %s" % path)


# The command that a deck's time outside tracking is put down to: the first of these that the deck uses.
COMMANDS = [("touschek_scatter", "Touschek scattering, `&touschek_scatter`"),
            ("momentum_aperture", "Momentum aperture, `&momentum_aperture`"),
            ("find_aperture", "Dynamic aperture, `&find_aperture`"),
            ("frequency_map", "Frequency map, `&frequency_map`"),
            ("tune_footprint", "Tune footprint, `&tune_footprint`"),
            ("analyze_map", "Map analysis, `&analyze_map`"),
            ("optimize", "Optimizer, `&optimize`"),
            ("track", "Twiss, matrix, moments, orbit and correction commands before `&track`")]
NO_COMMAND = "Other commands"


def command_class(deck):
    """Which command a deck's time outside tracking belongs to, from the commands in its input files."""
    used = set()
    for command in deck["commands"]:
        path = Path(deck["directory"]) / command
        if path.is_file():
            used.update(name.lower() for name in re.findall(r"&(\w+)", pr.masked(path.read_text(errors="replace"))))
    for name, title in COMMANDS:
        if name in used:
            return title
    return NO_COMMAND


def do_report(args):
    output = Path(args.output).resolve()
    listed = args.list or str(output / "list.json")
    decks = {d["name"]: d for d in json.load(open(listed))["decks"]} if Path(listed).is_file() else {}
    results = {}
    for path in sorted(output.glob("results-*.json")):      # results-<configuration>-<part>.json
        config = path.stem[len("results-"):].rsplit("-", 1)[0]
        results.setdefault(config, {}).update(json.load(open(path)))
    configs = [c for c in tc.CONFIGS if c in results]
    titles = {"elegant": "elegant", "gpu": "gpu-elegant", "gpu-threads": "gpu-elegant with threads"}
    names = sorted(set.intersection(*[set(results[c]) for c in configs]))
    good = [n for n in names if all(results[c][n]["status"] == 0 for c in configs)]
    bad = [n for n in names if n not in good]
    OUTSIDE = "(outside tracking)"

    def parts(config, name):
        record = results[config][name]
        elements = {k: v[1] for k, v in record["elements"].items()}
        elements[OUTSIDE] = max(0.0, record["wall"] - sum(elements.values()))
        return elements

    totals = {c: {} for c in configs}
    decks_with = {}
    for name in good:
        for c in configs:
            for kind, seconds in parts(c, name).items():
                totals[c][kind] = totals[c].get(kind, 0.0) + seconds
                if seconds > 0.005:
                    decks_with.setdefault(kind, set()).add(name)
    kinds = sorted(set().union(*[set(totals[c]) for c in configs]), key=lambda k: -totals[configs[-1]].get(k, 0))
    wall = {c: sum(results[c][n]["wall"] for n in good) for c in configs}
    waiting = {c: sum(max(0.0, results[c][n]["wall"] - results[c][n]["cpu"]) for n in good) for c in configs}
    lines = ["# Survey: where the time goes by element type", "",
             "Made by `src/gpu/scripts/element_survey.py`. %d decks ran in every configuration (%d more failed or "
             "timed out in one and are listed at the end). One run each. Seconds by element type are elegant's own "
             "`show_element_timing`, which only `&track`, `&frequency_map` and `&tune_footprint` report; \"%s\" is "
             "everything else: reading input, writing output, and every other command, including those that track "
             "particles themselves. It is split by command in the second table."
             % (len(good), len(bad), OUTSIDE), "",
             "Whole runs added up: " + ", ".join("%s %.0f s" % (titles[c], wall[c]) for c in configs) + ". Of that, "
             + ", ".join("%.0f s in %s" % (waiting[c], titles[c]) for c in configs)
             + " was waiting, not computing (wall time minus CPU time), mostly for the disk. The waiting is "
             "counted in the rows where it happened.", "",
             "## By element type, all decks added up", "",
             "Sorted by the time in %s." % titles[configs[-1]], "",
             "| Element type | Decks | " + " | ".join(titles[c] for c in configs) + " | Share of "
             + titles[configs[-1]] + " | " + ("elegant / gpu-elegant | " if "gpu" in configs and "elegant" in configs else "")
             + "Most time in (" + titles[configs[-1]] + ") |",
             "| --- | ---: | " + " | ".join("---:" for _ in configs) + " | ---: | "
             + ("---: | " if "gpu" in configs and "elegant" in configs else "") + "--- |"]
    last = configs[-1]
    for kind in kinds:
        if max(totals[c].get(kind, 0) for c in configs) < 0.5:
            continue
        top = sorted(good, key=lambda n: -parts(last, n).get(kind, 0))[:3]
        top = ", ".join("`%s` %.0f s" % (n, parts(last, n).get(kind, 0)) for n in top if parts(last, n).get(kind, 0) >= 0.5)
        ratio = ""
        if "gpu" in configs and "elegant" in configs:
            e, g = totals["elegant"].get(kind, 0), totals["gpu"].get(kind, 0)
            ratio = ("%.1fx | " % (e / g)) if g > 0.05 else "- | "
        lines.append("| %s | %d | %s | %.0f%% | %s%s |" % (
            kind, len(decks_with.get(kind, ())), " | ".join("%.1f s" % totals[c].get(kind, 0) for c in configs),
            100 * totals[last].get(kind, 0) / wall[last], ratio, top))
    if decks:
        groups = {}
        for name in good:
            if name in decks:
                groups.setdefault(command_class(decks[name]), []).append(name)
        outside = {c: {n: parts(c, n)[OUTSIDE] for n in good} for c in configs}
        computing = {n: min(outside[last][n], max(0.0, results[last][n]["cpu"]
                                                    - sum(v[1] for v in results[last][n]["elements"].values())))
                     for n in good}
        lines += ["", "## The time outside tracking, by command", "",
                  "Each deck's time outside tracking is put down to one command: the first in this list that the "
                  "deck uses. \"Computing\" is the part of the %s time that is CPU time; the rest is waiting." % titles[last],
                  "",
                  "| Command | Decks | " + " | ".join(titles[c] for c in configs) + " | Computing | Share of "
                  + titles[last] + " | " + ("elegant / gpu-elegant | " if "gpu" in configs and "elegant" in configs else "")
                  + "Most time in (" + titles[last] + ") |",
                  "| --- | ---: | " + " | ".join("---:" for _ in configs) + " | ---: | ---: | "
                  + ("---: | " if "gpu" in configs and "elegant" in configs else "") + "--- |"]
        for title in sorted(groups, key=lambda t: -sum(outside[last][n] for n in groups[t])):
            members = groups[title]
            total = {c: sum(outside[c][n] for n in members) for c in configs}
            if max(total.values()) < 0.5:
                continue
            top = sorted(members, key=lambda n: -outside[last][n])[:3]
            top = ", ".join("`%s` %.0f s" % (n, outside[last][n]) for n in top if outside[last][n] >= 0.5)
            ratio = ""
            if "gpu" in configs and "elegant" in configs:
                ratio = ("%.1fx | " % (total["elegant"] / total["gpu"])) if total["gpu"] > 0.05 else "- | "
            lines.append("| %s | %d | %s | %.1f s | %.0f%% | %s%s |" % (
                title, sum(outside[last][n] >= 0.5 for n in members), " | ".join("%.1f s" % total[c] for c in configs),
                sum(computing[n] for n in members), 100 * total[last] / wall[last], ratio, top))
    lines += ["", "## By deck", "", "Sorted by the time in %s. \"Waiting\" is %s's wall time minus its CPU time."
              % (titles[last], titles[last]), "",
              "| Deck | " + " | ".join(titles[c] for c in configs)
              + (" | elegant / gpu-elegant" if "gpu" in configs and "elegant" in configs else "")
              + " | Waiting | Where %s spends it |" % titles[last],
              "| --- | " + " | ".join("---:" for _ in configs)
              + (" | ---:" if "gpu" in configs and "elegant" in configs else "") + " | ---: | --- |"]
    for name in sorted(good, key=lambda n: -results[last][n]["wall"]):
        where = sorted(parts(last, name).items(), key=lambda kv: -kv[1])[:3]
        where = ", ".join("%s %.0f%%" % (k, 100 * v / results[last][name]["wall"]) for k, v in where
                          if v >= 0.05 * results[last][name]["wall"])
        ratio = ""
        if "gpu" in configs and "elegant" in configs:
            ratio = " | %.2fx" % (results["elegant"][name]["wall"] / results["gpu"][name]["wall"])
        mark = "" if results[last][name]["group"] == "in the test harness" else " *"
        lines.append("| `%s`%s | %s%s | %.1f s | %s |" % (
            name, mark, " | ".join("%.1f s" % results[c][name]["wall"] for c in configs), ratio,
            max(0.0, results[last][name]["wall"] - results[last][name]["cpu"]), where))
    lines += ["", "\\* not run by the test harness (no `reference` directory).", ""]
    if bad:
        lines += ["## Failed or timed out", ""]
        for name in bad:
            lines.append("- `%s`: " % name + ", ".join(
                "%s %s" % (titles[c], ("ran in %.0f s" % results[c][name]["wall"]) if results[c][name]["status"] == 0 else
                           (("timed out after %.0f s" % results[c][name]["wall"]) if results[c][name]["status"] == -1
                            else "failed")) for c in configs))
        lines.append("")
    (output / "SURVEY.md").write_text("\n".join(lines))
    print("\n".join(lines[:60]))
    print("Wrote %s" % (output / "SURVEY.md"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    make = sub.add_parser("list")
    make.add_argument("--output", required=True)
    make.add_argument("--minimum", type=float, default=10)
    make.add_argument("--test-set", default=TEST_SET)
    make.add_argument("--cpu-run", default=CPU_RUN)
    make.add_argument("--gpu-run", default=GPU_RUN)
    run = sub.add_parser("run")
    run.add_argument("--list", required=True)
    run.add_argument("--config", required=True, choices=tc.CONFIGS)
    run.add_argument("--core", required=True, help="the core, or cores for gpu-threads, to run on")
    run.add_argument("--output", required=True)
    run.add_argument("--elegant", default=str(tc.REPO / "bin/Linux-x86_64/elegant"))
    run.add_argument("--gpu-elegant", default=str(tc.REPO / "bin/Linux-x86_64-gpu/gpu-elegant"))
    run.add_argument("--threads", type=int, default=8)
    run.add_argument("--timeout", type=float, default=900)
    run.add_argument("--part", default="all", choices=["all", "harness", "other"],
                     help="the decks the test harness runs, the others, or all (to split the work over cores)")
    report = sub.add_parser("report")
    report.add_argument("--output", required=True)
    report.add_argument("--list", help="the list the runs used, for the table by command (default: list.json in --output)")
    args = parser.parse_args()
    {"list": do_list, "run": do_run, "report": do_report}[args.action](args)


if __name__ == "__main__":
    main()
