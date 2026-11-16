#!/usr/bin/env python3
"""Where a deck's time goes, by element type, in elegant, gpu-elegant and gpu-elegant with threads.

    element_breakdown.py DECK_DIRECTORY/run.ele --output RESULTS_DIR [--configs elegant,gpu,gpu-threads]

The deck is copied, `show_element_timing = 1` is added to its &run_setup, and it is run once in each
configuration (one job at a time, on fixed cores, as in timing_compare.py).  BREAKDOWN.md gets the
wall time and the seconds elegant itself reports for each element type.

In gpu-elegant a cheap element's work can be queued and paid for at a later element, so read its
per-type times as "where the program waited", and trust the totals more than the single rows.

The procedure is described in GPU-Testing/TIMING-PROCEDURE.md.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import timing_compare as tc
import predict_runtime as pr


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("deck")
    parser.add_argument("--output", required=True)
    parser.add_argument("--configs", default=",".join(tc.CONFIGS))
    parser.add_argument("--elegant", default=str(tc.REPO / "bin/Linux-x86_64/elegant"))
    parser.add_argument("--gpu-elegant", default=str(tc.REPO / "bin/Linux-x86_64-gpu/gpu-elegant"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--single-core", default="2")
    parser.add_argument("--thread-cores", default="0-7")
    parser.add_argument("--timeout", type=float, default=6 * 3600)
    args = parser.parse_args()
    deck = Path(args.deck).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = output / "deck"
    if source.exists():
        shutil.rmtree(source)
    shutil.copytree(deck.parent, source, symlinks=True, ignore=shutil.ignore_patterns("reference", "preference", "greference"))
    (source / deck.name).write_text(pr.set_value(deck.read_text(), "run_setup", "show_element_timing", 1))
    results = {}
    for config in args.configs.split(","):
        work = output / "work" / config
        record = tc.run_once({"directory": str(source), "commands": [deck.name]}, config, args, work)
        record["elements"] = pr.element_times(work / "timing.log")
        results[config] = record
        print("%-12s wall %8.2f s, cpu %8.2f s%s" % (config, record["wall"], record["cpu"],
                                                    "" if record["status"] == 0 else "   FAILED, status %d" % record["status"]),
              flush=True)
        json.dump(results, open(output / "breakdown.json", "w"), indent=1)
    configs = list(results)
    titles = {"elegant": "elegant", "gpu": "gpu-elegant", "gpu-threads": "gpu-elegant, %d threads" % args.threads}
    types = sorted({t for r in results.values() for t in r["elements"]},
                   key=lambda t: -max(r["elements"].get(t, [0, 0])[1] for r in results.values()))
    lines = ["# Where the time goes in `%s`" % deck, "",
             "Made by `src/gpu/scripts/element_breakdown.py`. One run each. Seconds by element type are elegant's own "
             "`show_element_timing`; in gpu-elegant a cheap element's work can be paid for at a later element.", "",
             "| Element type | Calls | " + " | ".join(titles[c] for c in configs) + " |",
             "| --- | ---: | " + " | ".join("---:" for _ in configs) + " |"]
    for kind in types:
        calls = max(r["elements"].get(kind, [0, 0])[0] for r in results.values())
        if max(r["elements"].get(kind, [0, 0])[1] for r in results.values()) < 0.005:
            continue
        lines.append("| %s | %s | %s |" % (kind, format(calls, ","), " | ".join(
            "%.2f s" % results[c]["elements"].get(kind, [0, 0])[1] for c in configs)))
    tracked = {c: sum(v[1] for v in results[c]["elements"].values()) for c in configs}
    lines.append("| **All element types** | | %s |" % " | ".join("**%.2f s**" % tracked[c] for c in configs))
    lines.append("| Outside tracking | | %s |" % " | ".join("%.2f s" % max(0.0, results[c]["wall"] - tracked[c]) for c in configs))
    lines.append("| **Wall time** | | %s |" % " | ".join("**%.2f s**" % results[c]["wall"] for c in configs))
    lines.append("| CPU time, all threads | | %s |" % " | ".join("%.2f s" % results[c]["cpu"] for c in configs))
    lines.append("")
    (output / "BREAKDOWN.md").write_text("\n".join(lines))
    print("\n".join(lines[4:]))
    print("Wrote %s" % (output / "BREAKDOWN.md"))


if __name__ == "__main__":
    main()
