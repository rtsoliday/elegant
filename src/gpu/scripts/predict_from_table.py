#!/usr/bin/env python3
"""Predict the run time of an elegant deck from the feature cost table, without running it.

    predict_from_table.py DECK_DIRECTORY/run.ele --table RESULTS_DIR/feature_costs.json [--particles N]

The lattice is read and its elements counted by type; each count is multiplied by the cost per call
that `feature_timing.py` measured, by the passes and the steps.  The result is a table of where the
time is expected to go, for elegant, gpu-elegant and gpu-elegant with host threads, and a list of
what the cost table doesn't cover.  For a closer number run `predict_runtime.py`, which runs the deck
itself at short settings.

The procedure is described in GPU-Testing/TIMING-PROCEDURE.md.
"""
import argparse
import json
import math
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import predict_runtime as pr

CONFIGS = ("elegant", "gpu", "gpu-threads")
DEFAULT_WORK = {"KQUAD": 4, "KSEXT": 4, "KOCT": 4, "MULT": 1, "CSBEND": 4, "CSRCSBEND": 4, "CCBEND": 4, "LGBEND": 4}
WORK_KEYS = ("N_SLICES", "N_KICKS", "N_STEPS")
# Elements whose work is in a file the predictor doesn't read (planes of a field expansion, modes of a
# cavity): taken to be as large as the file of the benchmark deck, and said so in the report.
FILE_WORK = {"BGGEXP": "planes", "FRFMODE": "modes", "FTRFMODE": "modes"}
BGGEXP_PLANES = 2001        # of the benchmark's file; an element's Z_INTERVAL divides it
WORK_NAMES = {"BGGEXP": "planes", "FRFMODE": "modes", "FTRFMODE": "modes", "FTABLE": "kicks", "LSRMDLTR": "steps"}
NOT_PREDICTED = ("optimize", "momentum_aperture", "correct", "closed_orbit", "chromaticity", "tunes", "moments_output",
                 "touschek_scatter", "ion_effects", "insert_sceffects", "tune_footprint", "analyze_map",
                 "coupled_twiss_output", "rf_setup", "link_elements", "sasefel", "slice_analysis", "matrix_output")


# element types costed like another type that is in the table
ALIASES = {"VKICK": "HKICK", "KICKER": "HKICK", "HVCOR": "HKICK", "EVKICK": "EHKICK", "EKICKER": "EHKICK",
           "HMON": "MONI", "VMON": "MONI", "QUADRUPOLE": "QUAD", "SEXTUPOLE": "SEXT", "SBEND": "SBEN", "RBEN": "SBEN",
           "RBEND": "SBEN", "BEND": "SBEN", "ZTRANSVERSE": "ZLONGIT",
           "CSRDRIF": "CSRDRIFT", "KQUSE": "KQUAD", "NIBEND": "CSBEND", "KSBEND": "CSBEND", "EMATRIX": "QUAD",
           "MATR": "QUAD", "SOLE": "QUAD", "CLEAN": "RCOL", "PEPPOT": "RCOL", "TAPERAPC": "ECOL", "TAPERAPR": "RCOL",
           "TAPERAPE": "ECOL", "SPEEDBUMP": "SCRAPER", "APCONTOUR": "ECOL"}
NEGLIGIBLE = ("MARK", "CHARGE", "MALIGN", "CENTER", "ENERGY", "TWISS", "FLOOR", "PFILTER", "RECIRC", "ROTATE", "ROTA",
              "REFLECT", "REMCOR", "MHISTOGRAM", "MAGNIFY")

# ---------------------------------------------------------------- lattice

def statements(path, seen=None):
    seen = seen or set()
    path = Path(path)
    if path in seen or not path.is_file():
        return
    seen.add(path)
    pending = ""
    for raw in path.read_text(errors="replace").splitlines():
        line = pr.masked(raw).rstrip()
        text = raw[:len(line)]
        if text.rstrip().endswith("&"):
            pending += text.rstrip()[:-1]
            continue
        text, pending = pending + text, ""
        if not text.strip() or text.lstrip().startswith("%"):
            continue
        include = re.match(r'\s*#include\s*:\s*"?([^"\s]+)"?', text, re.I)
        if include:
            yield from statements(path.parent / include.group(1), seen)
        else:
            yield text


def split_commas(text):
    parts, depth, quote, current = [], 0, False, ""
    for ch in text:
        if ch == '"':
            quote = not quote
        if not quote and ch in "([":
            depth += 1
        if not quote and ch in ")]":
            depth -= 1
        if ch == "," and not quote and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += ch
    return parts + [current]


def read_lattice(path):
    elements, lines, last = {}, {}, None
    for text in statements(path):
        m = re.match(r'\s*"([^"]+)"\s*:\s*(.*)$', text) or re.match(r'\s*([^":\s]+)\s*:\s*(.*)$', text)
        if not m:
            continue
        name, rest = m.group(1).strip().upper(), m.group(2).strip()
        line = re.match(r"LINE\s*=\s*\((.*)\)\s*$", rest, re.I)
        if line:
            lines[name] = line.group(1)
            last = name
            continue
        parts = split_commas(rest)
        params = {}
        for part in parts[1:]:
            key, _, value = part.partition("=")
            params[key.strip().upper()] = value.strip().strip('"')
        elements[name] = (parts[0].strip().upper(), params)
    return elements, lines, last


def expand(name, lines, memo):
    """Counter of element names in a beamline."""
    if name in memo:
        return memo[name]
    total = Counter()
    for item in split_commas(lines[name]):
        item = item.strip()
        if not item:
            continue
        m = re.match(r"(?:(\d+)\s*\*\s*)?-?\s*(.*)$", item)
        count, body = int(m.group(1) or 1), m.group(2).strip()
        if body.startswith("("):
            lines["(%s)" % body] = body[1:-1]
            inner = expand("(%s)" % body, lines, memo)
        else:
            body = body.strip().strip('"').upper()
            inner = expand(body, lines, memo) if body in lines else Counter({body: 1})
        for key, value in inner.items():
            total[key] += count * value
    memo[name] = total
    return total


def flag(params, key):
    try:
        return float(params.get(key, 0)) != 0
    except ValueError:
        return True


def feature_of(kind, params, table):
    """(feature name or None, work of this element)."""
    types = {}
    for name, entry in table["features"].items():
        types.setdefault(entry["type"], []).append(name)
    kind = ALIASES.get(kind, kind)
    if kind in ("FRFMODE", "FTRFMODE") and kind not in types:      # a table from before these had rows
        kind = kind[1:]
    match = [t for t in types if kind.startswith(t) or t.startswith(kind)]
    if not match:
        return None, 1, (), ""
    canonical = max(match, key=len)
    work = DEFAULT_WORK.get(canonical, 1)
    for key in WORK_KEYS:
        try:
            if float(params.get(key, 0)) > 0:
                work = float(params[key])
                break
        except ValueError:
            pass
    names = types[canonical]
    wanted, ratios, notes = names[0], [], []
    if canonical in FILE_WORK:
        if canonical == "BGGEXP" and flag(params, "SYMPLECTIC") and "bggexp-symplectic" in table["features"]:
            wanted = "bggexp-symplectic"
            notes.append("symplectic")
        work = table["features"][wanted]["work"]
        if canonical == "BGGEXP":
            work = BGGEXP_PLANES
            try:
                work /= max(1.0, float(params.get("Z_INTERVAL", 1)))
            except ValueError:
                pass
            notes.append("a file of %d planes assumed" % BGGEXP_PLANES)
        else:
            notes.append("as many %s assumed as in the benchmark" % FILE_WORK[canonical])
    radiation = "isr" if flag(params, "ISR") else "synch-rad" if flag(params, "SYNCH_RAD") else ""
    if canonical in ("KQUAD", "CSBEND", "LGBEND"):
        own = canonical.lower() + ("-" + radiation if radiation else "")
        if own in table["features"]:
            wanted = own
        else:
            wanted = canonical.lower()
            if radiation:
                ratios.append(("csbend-" + radiation, "csbend"))
    elif canonical in ("KSEXT", "KOCT", "CCBEND", "MULT", "CSRCSBEND") and radiation:
        ratios.append((("csbend-" if "BEND" in canonical else "kquad-") + radiation, "csbend" if "BEND" in canonical else "kquad"))
    if radiation and canonical in ("KQUAD", "CSBEND", "LGBEND", "KSEXT", "KOCT", "CCBEND", "MULT", "CSRCSBEND"):
        notes.append("ISR" if radiation == "isr" else "radiation")
    try:
        order = int(float(params.get("INTEGRATION_ORDER", 6 if canonical == "LGBEND" else 4)))
    except ValueError:
        order = 4
    if canonical in ("KQUAD", "KSEXT", "KOCT") and order == 6:
        ratios.append(("kquad-order6", "kquad"))
        notes.append("order 6")
    elif canonical in ("CSBEND", "CCBEND", "CSRCSBEND") and order == 6:
        ratios.append(("csbend-order6", "csbend"))
        notes.append("order 6")
    elif canonical == "LGBEND" and order == 4:
        ratios.append(("csbend", "csbend-order6"))
        notes.append("order 4")
    if canonical == "RFCA" and flag(params, "CHANGE_P0"):
        wanted = "rfca-change-p0"
    elif canonical == "WATCH":
        mode = params.get("MODE", "coordinate").lower()
        wanted = "watch-" + ("parameter" if mode.startswith("p") else "centroid" if mode.startswith("ce") else "coordinate")
    elif canonical == "QUAD":
        wanted = "quad-matrix"
    elif canonical == "DRIF":
        wanted = "drift"
    return (wanted if wanted in table["features"] else names[0]), work, tuple(ratios), ", ".join(notes)


def altered(elements, text):
    """Apply the deck's &alter_elements commands to the element parameters."""
    import fnmatch
    for start, end in pr.blocks(text, "alter_elements"):
        def item(key, default):
            span = pr.value_span(text, start, end, key)
            return text[span[0]:span[1]].strip().strip('"') if span else default
        name, kind, what, value = item("name", "*"), item("type", "*"), item("item", ""), item("value", "")
        for element, (etype, params) in elements.items():
            if fnmatch.fnmatchcase(element.upper(), name.upper()) and fnmatch.fnmatchcase(etype, kind.upper()):
                params[what.upper()] = value


# ---------------------------------------------------------------- costs

def interpolate(points, particles):
    """points: {"1000": seconds, ...}.  Log-log between measured counts, proportional beyond the last.

    A 0 in the table is a measurement (too small to measure), not a missing one: between a 0 and its
    neighbour the cost goes linearly in the logarithm of the particle count, and beyond a last 0 it is 0.
    """
    xs = sorted((int(k), v) for k, v in points.items())
    if not xs:
        return 0.0
    if particles <= xs[0][0]:
        return xs[0][1]
    if particles >= xs[-1][0]:
        return xs[-1][1] * particles / xs[-1][0] if len(xs) > 1 and xs[-1][1] > 1.5 * xs[-2][1] else xs[-1][1]
    for (n1, t1), (n2, t2) in zip(xs, xs[1:]):
        if n1 <= particles <= n2:
            f = math.log(particles / n1) / math.log(n2 / n1)
            if t1 > 0 and t2 > 0:
                return math.exp(math.log(t1) + f * (math.log(t2) - math.log(t1)))
            return t1 + f * (t2 - t1)


def call_cost(table, name, work, config, particles, ratios=()):
    entry = table["features"][name]
    points = entry["seconds_per_call"].get(config)
    if not points:
        return None
    cost = interpolate(points, particles) * (work / entry["work"] if entry["work"] > 1 else 1)
    for above, below in ratios:      # an option measured on another element type, applied as a factor
        if above in table["features"] and below in table["features"]:
            a = table["features"][above]["seconds_per_call"].get(config)
            b = table["features"][below]["seconds_per_call"].get(config)
            if a and b and interpolate(b, particles) > 0:
                cost *= interpolate(a, particles) / interpolate(b, particles)
    return cost


def beam_particles(text, directory, given):
    if given:
        return given, "given"
    if pr.blocks(text, "bunched_beam"):
        n = float(pr.get_value(text, "bunched_beam", "n_particles_per_bunch", "1"))
        bunches = float(pr.get_value(text, "bunched_beam", "bunches", "1") or 1) if False else 1
        return int(n * bunches), "&bunched_beam"
    if pr.blocks(text, "sdds_beam"):
        name = (pr.get_value(text, "sdds_beam", "input", "") or "").strip('"')
        interval = float(pr.get_value(text, "sdds_beam", "sample_interval", "1"))
        try:
            out = subprocess.run(["sdds2stream", "-rows=bare", str(Path(directory) / name)], capture_output=True,
                                 text=True, timeout=60).stdout.split()
            return int(sum(int(x) for x in out) / interval), "rows of %s" % name
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return None, ""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("deck")
    parser.add_argument("--table", required=True, help="feature_costs.json from feature_timing.py report")
    parser.add_argument("--particles", type=int, help="particles in the beam, when the deck doesn't say")
    parser.add_argument("--output", help="also write the prediction to this Markdown file")
    args = parser.parse_args()
    deck = Path(args.deck).resolve()
    text = deck.read_text()
    table = json.load(open(args.table))
    threads = table["environment"]["threads"]
    titles = {"elegant": "elegant", "gpu": "gpu-elegant", "gpu-threads": "gpu-elegant, %d threads" % threads}

    lattice_name = (pr.get_value(text, "run_setup", "lattice", "") or "").strip('"')
    elements, lines, last = read_lattice(deck.parent / lattice_name)
    for name, (kind, params) in list(elements.items()):      # an element defined as a copy of another
        if kind in elements:
            elements[name] = (elements[kind][0], dict(elements[kind][1], **params))
    altered(elements, text)
    use = (pr.get_value(text, "run_setup", "use_beamline", "") or "").strip('"').upper() or last
    if use not in lines:
        sys.exit("can't find beamline %r in %s" % (use, lattice_name))
    counts = expand(use, lines, {})
    def largest(key):       # a deck can have several &run_control commands; the longest one decides
        values = []
        for start, end in pr.blocks(text, "run_control"):
            span = pr.value_span(text, start, end, key)
            try:
                values.append(float(text[span[0]:span[1]]) if span else 1.0)
            except ValueError:
                values.append(1.0)
        return max(values or [1.0])

    passes, steps = largest("n_passes"), largest("n_steps")
    particles, source = beam_particles(text, deck.parent, args.particles)
    commands = [name for name in re.findall(r"&(\w+)", pr.masked(text)) if name.lower() != "end"]
    tracking = any(c.lower() == "track" for c in commands)
    searches = [c.lower() for c in commands if c.lower() in ("frequency_map", "find_aperture")]
    if tracking and particles is None:
        sys.exit("the number of particles is not in the deck; give --particles")

    # calls by feature
    by_feature, unknown, every_nth = {}, Counter(), {}
    for name, count in counts.items():
        kind, params = elements.get(name, ("?", {}))
        if kind in NEGLIGIBLE:
            continue
        feature, work, ratios, note = feature_of(kind, params, table)
        if feature is None:
            unknown[kind] += count
            long = False
            try:
                long = float(params.get("L", 0)) > 0
            except ValueError:
                long = True
            feature, work = ("quad-matrix" if long else "monitor"), 1
            key = (kind + " (assumed like a %s)" % ("matrix quadrupole" if long else "monitor"), feature, work, ())
        else:
            kind_name = table["features"][feature]["type"]
            label = kind_name + (", %g %s" % (work, WORK_NAMES.get(kind_name, "slices"))
                                 if table["features"][feature]["work"] > 1 else "")
            if table["features"][feature]["type"] in ("WATCH", "RFCA", "QUAD", "DRIF") or kind != table["features"][feature]["type"]:
                label = "%s (%s)" % (kind, feature)
            if table["features"][feature]["type"] == "WATCH":
                try:
                    interval = max(1.0, float(params.get("INTERVAL", 1)))
                except ValueError:
                    interval = 1.0
                if interval > 1:
                    label += ", every %g passes" % interval
                    every_nth[label] = interval
            key = (label + (", " + note if note else ""), feature, work, ratios)
        by_feature[key] = by_feature.get(key, 0) + count
    total_calls = sum(by_feature.values())

    out = ["# Predicted run time of `%s` from the feature cost table" % deck, "",
           "Beamline `%s`: %s elements a pass, %g passes, %g steps%s." %
           (use, format(sum(counts.values()), ","), passes, steps,
            ", %s particles (%s)" % (format(particles, ","), source) if particles else ""), ""]
    totals = {c: 0.0 for c in CONFIGS}
    rows = []
    if tracking:
        for (label, feature, work, ratios), count in by_feature.items():
            row = [label, count]
            for config in CONFIGS:
                cost = call_cost(table, feature, work, config, particles, ratios)
                seconds = (cost or 0) * count * passes * steps / every_nth.get(label, 1)
                # what each copy of the element pays once in a run, where the table has it (gpu-elegant)
                first = table["features"][feature].get("once_per_element", {}).get(config)
                if first:
                    seconds += interpolate(first, particles) * count * steps
                totals[config] += seconds
                row.append(seconds)
            rows.append(row)
        # output files and the fixed part
        extra = []
        every = sum(counts.values())      # the sums are made after every element, markers too
        for name, key in (("output-sigma", "sigma"), ("output-centroid", "centroid")):
            if pr.get_value(text, "run_setup", key) and name in table["whole_run"]:
                extra.append((table["whole_run"][name]["title"], name, every * passes * steps))
                break                     # the sigma sums include the centroid sums
        if pr.get_value(text, "run_setup", "output") and "output-final-beam" in table["whole_run"]:
            extra.append((table["whole_run"]["output-final-beam"]["title"], "output-final-beam", particles * steps))
        for title, name, multiplier in extra:
            row = [title, ""]
            for config in CONFIGS:
                points = table["whole_run"][name]["seconds"].get(config, {})
                per = interpolate(points, particles)
                if table["whole_run"][name]["per"] == "particle":
                    per = interpolate(points, particles) if points else 0
                seconds = per * multiplier
                totals[config] += seconds
                row.append(seconds)
            rows.append(row)
        row = ["start-up and making the beam", ""]
        for config in CONFIGS:
            seconds = interpolate(table["overhead"].get(config, {}), particles)
            totals[config] += seconds
            row.append(seconds)
        rows.append(row)

    for search in searches:
        bench = table["search"].get("frequency-map" if search == "frequency_map" else "dynamic-aperture")
        if not bench:
            continue
        mode = (pr.get_value(text, search, "mode", "") or "").lower()
        nx = float(pr.get_value(text, search, "nx", "20" if search == "frequency_map" else "21"))
        ny = float(pr.get_value(text, search, "ny", "21" if search == "frequency_map" else "11"))
        points = float(pr.get_value(text, search, "n_lines", "11")) * nx if "line" in mode else nx * ny
        one = sum((call_cost(table, feature, work, "elegant", 1, ratios) or 0) * count
                  for (_, feature, work, ratios), count in by_feature.items())
        bench_one = sum((call_cost(table, f, 8, "elegant", 1) or 0) * n
                        for f, n in (("kquad", 40), ("ksext", 20), ("csbend", 20), ("drift", 80)))
        row = ["&%s, %g points, %g turns" % (search, points, passes), ""]
        for config in CONFIGS:
            if config not in bench["wall"] or not bench_one:
                row.append(0.0)
                continue
            factor = bench["wall"][config] / (bench["points"] * bench["turns"] * bench_one)
            seconds = factor * points * passes * one * steps
            totals[config] += seconds
            row.append(seconds)
        rows.append(row)

    out += ["| Part | Calls a pass | " + " | ".join(titles[c] for c in CONFIGS) + " |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in sorted(rows, key=lambda r: -r[2]):
        out.append("| %s | %s | %s |" % (row[0], format(row[1], ",") if row[1] != "" else "", " | ".join(
            "%.3g s" % v for v in row[2:])))
    out.append("| **Total** | | %s |" % " | ".join("**%.3g s**" % totals[c] for c in CONFIGS))
    out += ["", "## Not covered by the table", ""]
    notes = ["- Element type `%s` (%d calls a pass): %s." % (kind, n, "no benchmark, cost assumed as shown above")
             for kind, n in unknown.items()]
    notes += ["- Command `&%s`: not predicted." % c for c in dict.fromkeys(c.lower() for c in commands) if c in NOT_PREDICTED]
    if searches:
        notes.append("- Searches are scaled from one benchmark lattice; particles that are lost early make the real "
                     "time shorter.")
    if steps > 1 or any(c.lower() in ("error_element", "vary_element") for c in commands):
        notes.append("- Every step is taken to cost the same.")
    notes.append("- Beam loss, and gpu-elegant's copies between elements it runs on the GPU and elements it leaves on the "
                 "CPU, are not in the table.")
    out += notes + [""]
    print("\n".join(out))
    if args.output:
        Path(args.output).write_text("\n".join(out))


if __name__ == "__main__":
    main()
