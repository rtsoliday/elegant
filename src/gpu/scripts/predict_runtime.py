#!/usr/bin/env python3
"""Predict how long an elegant deck will run, from short runs of the same deck.

    predict_runtime.py DECK_DIRECTORY/run.ele --output RESULTS_DIR [--knob passes] [--check]

The deck is copied, one "knob" that sets its length is turned down, and it is run at two or more
short settings in each configuration (elegant, gpu-elegant, gpu-elegant with host threads).  The
time is taken as  fixed + rate * setting  and extrapolated to the full setting.  With --check the
full deck is run as well and the error of the prediction is shown.

Knobs (--knob; "auto" picks the first that applies, in this order):
    grid        nx*ny of &frequency_map or &find_aperture (n_lines for the line modes)
    passes      n_passes of &run_control
    steps       n_steps of &run_control
    particles   n_particles_per_bunch of &bunched_beam, or sample_interval of &sdds_beam

The procedure is described in GPU-Testing/TIMING-PROCEDURE.md.
"""
import argparse
import json
import math
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import timing_compare as tc

CONFIG_TITLES = {"elegant": "elegant", "gpu": "gpu-elegant", "gpu-threads": "gpu-elegant, %d threads"}


# ---------------------------------------------------------------- namelist reading and editing

def masked(text):
    """The text with comments and quoted strings blanked out, same length."""
    out, quote, comment = [], False, False
    for ch in text:
        if comment:
            comment = ch != "\n"
            out.append("\n" if ch == "\n" else " ")
        elif quote:
            quote = ch != '"'
            out.append(" ")
        elif ch == "!":
            comment = True
            out.append(" ")
        elif ch == '"':
            quote = True
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


def blocks(text, name):
    """(start, end) of the body of every &name ... &end block."""
    mask = masked(text)
    found = []
    for m in re.finditer(r"&%s\b" % re.escape(name), mask, re.I):
        end = re.search(r"&end\b", mask[m.end():], re.I)
        if end:
            found.append((m.end(), m.end() + end.start()))
    return found


def value_span(text, start, end, key):
    m = re.search(r"(?<![\w\[\]])%s\s*=" % re.escape(key), masked(text)[start:end], re.I)
    if not m:
        return None
    first = start + m.end()
    while first < end and text[first] in " \t":     # not through the mask: a quoted value is blank there
        first += 1
    stop = re.search(r"[,\n]", masked(text)[first:end])
    return first, (first + stop.start() if stop else end)


def get_value(text, block, key, default=None):
    for start, end in blocks(text, block):
        span = value_span(text, start, end, key)
        if span:
            return text[span[0]:span[1]].strip()
        return default
    return None


def set_value(text, block, key, value):
    """Set key in every block of that name (added when it is not there)."""
    for index in range(len(blocks(text, block))):
        start, end = blocks(text, block)[index]
        span = value_span(text, start, end, key)
        if span:
            text = text[:span[0]] + str(value) + text[span[1]:]
        else:
            text = text[:start] + "\n  %s = %s," % (key, value) + text[start:]
    return text


def number(value, name):
    try:
        return float(value)
    except (TypeError, ValueError):
        sys.exit("%s is %r, not a plain number; this knob can't be used" % (name, value))


# ---------------------------------------------------------------- knobs

class Knob:
    """full: the deck's setting.  minimum: the smallest short setting.  apply(text, s) -> (text, actual s)."""


def knob_passes(text):
    value = get_value(text, "run_control", "n_passes", "1")
    if value is None:
        return None
    k = Knob()
    k.name, k.unit, k.one = "passes", "passes", "pass"
    k.full, k.minimum = number(value, "n_passes"), 2
    k.apply = lambda t, s: (set_value(t, "run_control", "n_passes", int(round(s))), int(round(s)))
    return k


def knob_steps(text):
    value = get_value(text, "run_control", "n_steps", "1")
    if value is None:
        return None
    k = Knob()
    k.name, k.unit, k.one = "steps", "steps", "step"
    k.full, k.minimum = number(value, "n_steps"), 2
    k.apply = lambda t, s: (set_value(t, "run_control", "n_steps", int(round(s))), int(round(s)))
    return k


def knob_grid(text):
    for block, defaults in (("frequency_map", (20, 21)), ("find_aperture", (21, 11))):
        if not blocks(text, block):
            continue
        k = Knob()
        k.name = "grid"
        mode = (get_value(text, block, "mode", "") or "").lower()
        if block == "find_aperture" and "line" in mode:
            lines = number(get_value(text, block, "n_lines", "11"), "n_lines")
            k.unit, k.one, k.full, k.minimum, k.threads_minimum = "lines", "line", lines, 3, 64
            k.apply = lambda t, s, b=block: (set_value(t, b, "n_lines", int(round(s))), int(round(s)))
            return k
        nx = number(get_value(text, block, "nx", str(defaults[0])), "nx")
        ny = number(get_value(text, block, "ny", str(defaults[1])), "ny")
        k.unit, k.one, k.full, k.minimum, k.threads_minimum = "grid points", "grid point", nx * ny, 9, 256

        def apply(t, s, b=block, nx=nx, ny=ny):
            f = math.sqrt(s / (nx * ny))
            mx, my = max(2, int(round(nx * f))), max(2, int(round(ny * f)))
            return set_value(set_value(t, b, "nx", mx), b, "ny", my), mx * my

        k.apply = apply
        return k
    return None


def knob_particles(text):
    if blocks(text, "bunched_beam"):
        k = Knob()
        k.name, k.unit, k.one = "particles", "particles", "particle"
        k.full = number(get_value(text, "bunched_beam", "n_particles_per_bunch", "1"), "n_particles_per_bunch")
        k.minimum = max(1000, k.full / 64)
        k.gpu_minimum = k.full / 4
        k.apply = lambda t, s: (set_value(t, "bunched_beam", "n_particles_per_bunch", int(round(s))), int(round(s)))
        return k
    if blocks(text, "sdds_beam"):
        interval = number(get_value(text, "sdds_beam", "sample_interval", "1"), "sample_interval")
        k = Knob()
        k.name, k.unit, k.one = "particles", "of the input beam", "whole beam"
        k.full, k.minimum, k.gpu_minimum = 1.0, 1.0 / 64, 1.0 / 4

        def apply(t, s, interval=interval):
            n = max(2, int(round(1.0 / s)))
            return set_value(t, "sdds_beam", "sample_interval", int(interval * n)), 1.0 / n

        k.apply = apply
        return k
    return None


KNOBS = {"passes": knob_passes, "grid": knob_grid, "steps": knob_steps, "particles": knob_particles}


def choose_knob(text, wanted):
    if wanted != "auto":
        knob = KNOBS[wanted](text)
        if not knob or knob.full < 4 * knob.minimum:
            sys.exit("knob %s can't shorten this deck (its full setting is %s)" %
                     (wanted, "absent" if not knob else "%g" % knob.full))
        return knob
    for name in ("grid", "passes", "steps", "particles"):   # a grid is the straightest line, particles the least
        knob = KNOBS[name](text)
        if knob and knob.full >= 16 * knob.minimum:
            return knob
    found = ["%s %g" % (name, KNOBS[name](text).full) for name in KNOBS if KNOBS[name](text)]
    sys.exit("found nothing that shortens this deck by 16 times or more (%s).  Its time is probably in a command "
             "this script can't shorten (an optimizer, a momentum aperture, a scan).  Give --knob to force one, or "
             "just run the deck." % ", ".join(found))


# ---------------------------------------------------------------- running

def element_times(log):
    """Sum of the 'Time spent in different elements' reports: {type: [calls, seconds]}."""
    times, inside = {}, False
    for line in open(log, errors="replace"):
        if line.startswith("Time spent in different elements"):
            inside = True
            continue
        m = re.match(r"\s*(\S+):\s+(\d+) times,\s+(\S+) s,", line) if inside else None
        if m:
            entry = times.setdefault(m.group(1), [0, 0.0])
            entry[0] += int(m.group(2))
            entry[1] += float(m.group(3))
        else:
            inside = False
    return times


def timed(text, ele, source, knob, setting, config, args, output):
    """Run the deck at one setting.  Returns {"setting", "wall", "cpu", "status", "elements"}."""
    edited, actual = (text, knob.full) if setting is None else knob.apply(text, setting)
    edited = set_value(edited, "run_setup", "show_element_timing", 1)
    deck = output / "decks" / ("%s-%g" % (config, actual))
    if deck.exists():
        shutil.rmtree(deck)
    shutil.copytree(source, deck, symlinks=True, ignore=shutil.ignore_patterns("reference", "preference", "greference"))
    (deck / ele).write_text(edited)
    work = output / "work" / ("%s-%g" % (config, actual))
    record = tc.run_once({"directory": str(deck), "commands": [ele]}, config, args, work)
    record["setting"] = actual
    record["elements"] = element_times(work / "timing.log")
    shutil.rmtree(deck)
    print("  %-12s %12g %-12s wall %8.2f s, cpu %8.2f s%s" %
          (config, actual, knob.unit, record["wall"], record["cpu"],
           "" if record["status"] == 0 else "   FAILED, status %d" % record["status"]), flush=True)
    return record


def predict(runs, full):
    """Line through the last two runs.  Returns (prediction, fixed, rate)."""
    (s1, t1), (s2, t2) = [(r["setting"], r["wall"]) for r in runs[-2:]]
    rate = (t2 - t1) / (s2 - s1)
    if rate <= 0:
        return max(t1, t2), max(t1, t2), 0.0
    fixed = max(0.0, t2 - rate * s2)
    return fixed + rate * full, fixed, rate


def do_config(text, ele, source, knob, config, args, output):
    on_gpu = config != "elegant" and hasattr(knob, "gpu_minimum")
    # gpu-elegant leaves small beams on the CPU, so its short runs stay near the full particle count
    runs, setting = [], float(knob.gpu_minimum if on_gpu else knob.minimum)
    growth = 2.0 if on_gpu else args.growth
    if config == "gpu-threads" and hasattr(knob, "threads_minimum"):
        # host threads share out a batch of 64 or more particles: smaller short runs would not use them
        if knob.threads_minimum > knob.full / 2:
            print("  %-12s not predicted: threads need a batch of 64 or more, which this deck only has at full size"
                  % config, flush=True)
            return {"runs": [], "failed": True, "reason": "threads need a batch of 64 or more particles, which this "
                    "deck only has at full size"}
        setting = float(knob.threads_minimum)
    while True:
        record = timed(text, ele, source, knob, setting, config, args, output)
        if record["status"] != 0:
            return {"runs": runs + [record], "failed": True}
        if not runs or record["setting"] > runs[-1]["setting"]:
            runs.append(record)
        following = setting * growth
        if (len(runs) >= 2 and record["wall"] >= args.target) or following > knob.full / 2:
            break
        setting = following
    if len(runs) < 2:
        return {"runs": runs, "failed": True}
    result = {"runs": runs}
    result["prediction"], result["fixed"], result["rate"] = predict(runs, knob.full)
    if len(runs) >= 3:   # how well did the two runs before the last predict the last one?
        earlier = predict(runs[:-1], runs[-1]["setting"])[0]
        result["self_check"] = earlier / runs[-1]["wall"] - 1
    if args.check:
        result["full"] = timed(text, ele, source, knob, None, config, args, output)
    return result


def report(results, knob, args, ele, source, output):
    lines = ["# Predicted run time of `%s`" % (Path(source) / ele), "",
             "Made by `src/gpu/scripts/predict_runtime.py`. The deck was shortened with the knob **%s** "
             "(full setting %g %s), run at short settings, and the time extrapolated as fixed + rate x setting."
             % (knob.name, knob.full, knob.unit), "",
             "| Configuration | Short runs (setting: wall time) | Fixed part | Rate | Predicted | Check of the method |"
             + (" Measured | Error |" if args.check else ""),
             "| --- | --- | ---: | ---: | ---: | ---: |" + (" ---: | ---: |" if args.check else "")]
    for config, r in results.items():
        title = CONFIG_TITLES[config] % args.threads if "%d" in CONFIG_TITLES[config] else CONFIG_TITLES[config]
        shorts = ", ".join("%g: %.2f s" % (x["setting"], x["wall"]) for x in r["runs"])
        if r.get("failed"):
            lines.append("| %s | %s | | | %s | |" % (title, shorts, "not predicted: " + r["reason"] if r.get("reason")
                                                    else "failed") + (" | |" if args.check else ""))
            continue
        row = "| %s | %s | %s | %s per %s | **%s** | %s |" % (
            title, shorts, tc.seconds(r["fixed"]), "%.3g s" % r["rate"], knob.one,
            tc.seconds(r["prediction"]), "%+.0f%%" % (100 * r["self_check"]) if "self_check" in r else "-")
        if args.check:
            full = r["full"]
            row += (" %s | %+.0f%% |" % (tc.seconds(full["wall"]), 100 * (r["prediction"] / full["wall"] - 1))
                    if full["status"] == 0 else " failed | |")
        lines.append(row)
    doubtful = [c for c, r in results.items() if abs(r.get("self_check", 0)) > 0.15]
    if doubtful:
        lines += ["", "**Doubtful: %s.** The short runs don't lie on a straight line there (see \"Check of the method\"), "
                  "so that prediction can be far off. Try another `--knob`, or a larger `--target`." % ", ".join(doubtful)]
    lines += ["", "\"Check of the method\": the error when the two runs before the last are used to predict the last "
              "short run. A large value means the time is not a straight line in this knob, and the prediction is "
              "less sure.", "", "## Where the time goes", "",
              "From `show_element_timing` in the longest short run of each configuration: the share of that run's "
              "wall time spent in each element type. \"Outside tracking\" is reading input, set-up, other commands "
              "and writing output.", ""]
    for config, r in results.items():
        if r.get("failed"):
            continue
        last = r["runs"][-1]
        tracked = sum(v[1] for v in last["elements"].values())
        parts = sorted(last["elements"].items(), key=lambda item: -item[1][1])[:6]
        text = ", ".join("%s %.0f%%" % (name, 100 * v[1] / last["wall"]) for name, v in parts if v[1] > 0.005 * last["wall"])
        lines.append("- **%s**: %s%soutside tracking %.0f%%" %
                     (config, text, ", " if text else "", 100 * max(0.0, 1 - tracked / last["wall"])))
    lines += ["", "## Limits", "",
              "- The prediction assumes the same cost per %s all the way. Beam loss, an optimizer, or a deck that "
              "changes behaviour with time breaks that." % knob.one,
              "- One run at each setting; wall times vary by a few percent, more when large files are written.",
              "- gpu-elegant changes paths at particle-count thresholds, so the **particles** knob is the least "
              "reliable for it.", ""]
    (output / "PREDICTION.md").write_text("\n".join(lines))
    print("\n".join(lines[2:4 + 2 + len(results) + (2 if doubtful else 0)]))
    print("\nWrote %s" % (output / "PREDICTION.md"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("deck", help="the .ele file, in its directory with everything it needs")
    parser.add_argument("--output", required=True, help="directory for the copies, logs and PREDICTION.md")
    parser.add_argument("--knob", default="auto", choices=["auto"] + list(KNOBS))
    parser.add_argument("--configs", default=",".join(tc.CONFIGS))
    parser.add_argument("--target", type=float, default=10.0, help="lengthen the short runs until one takes this long (s)")
    parser.add_argument("--growth", type=float, default=4.0, help="factor between short settings")
    parser.add_argument("--check", action="store_true", help="also run the full deck and show the error")
    parser.add_argument("--elegant", default=str(tc.REPO / "bin/Linux-x86_64/elegant"))
    parser.add_argument("--gpu-elegant", default=str(tc.REPO / "bin/Linux-x86_64-gpu/gpu-elegant"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--single-core", default="2")
    parser.add_argument("--thread-cores", default="0-7")
    parser.add_argument("--timeout", type=float, default=6 * 3600)
    args = parser.parse_args()
    deck = Path(args.deck).resolve()
    if not deck.is_file():
        sys.exit("no such file: %s" % deck)
    text = deck.read_text()
    knob = choose_knob(text, args.knob)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    print("Knob: %s, full setting %g %s" % (knob.name, knob.full, knob.unit), flush=True)
    results = {}
    for config in args.configs.split(","):
        results[config] = do_config(text, deck.name, deck.parent, knob, config, args, output)
        json.dump({"deck": str(deck), "knob": knob.name, "full": knob.full, "results": results},
                  open(output / "prediction.json", "w"), indent=1)
    report(results, knob, args, deck.name, deck.parent, output)


if __name__ == "__main__":
    main()
