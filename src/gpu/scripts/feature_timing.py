#!/usr/bin/env python3
"""Cost of each main feature of elegant, for elegant, gpu-elegant and gpu-elegant with host threads.

    feature_timing.py make   --cases CASES_DIR             write the benchmark decks
    feature_timing.py run    --cases CASES_DIR --output RESULTS_DIR [--only names] [--configs ...]
    feature_timing.py report --output RESULTS_DIR [--note TEXT]   write FEATURES.md and feature_costs.json

To rerun only some runs (gpu-elegant after a change to it, say), copy runs.json into a new RESULTS_DIR,
delete the keys to be rerun, and run again: runs already recorded there are kept.

Each feature has one small deck: a FODO line of matrix quadrupoles and drifts with the element under
test put into every half cell.  The deck is tracked at several particle counts and the time of the
element type is read from elegant's own `show_element_timing` report, so the numbers are seconds per
call of the element.  "Whole-run" features (output files, commands) are timed by wall time instead.

For gpu-elegant an element's time is a wall-clock slope: the deck and the line without the element (the
scaffold) are each run with two numbers of passes, and the time the deck gains over the scaffold is divided
by the calls it gains (slope_runs, and the note above SLOPE_ROUNDS).  Those runs have "+slope" or "+fixed"
in their keys.  The plain runs of gpu-elegant only say whether a deck stays on the CPU and how many passes
the slope needs.

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

CELLS = 50                     # FODO cells; the element under test appears twice per cell
COUNTS = {"elegant": (1, 1000, 10000, 100000, 1000000), "gpu": (1, 1000, 10000, 100000, 1000000)}
# elegant at 1,000,000 particles tracks a copy of the deck with a tenth of the cells (at least one), in the
# folder e1000000: its time per call is the same, and its runs take a tenth as long
ELEGANT_LARGE = 1000000
COUNTS["gpu-threads"] = COUNTS["gpu"]
PASSES = {1: 2000, 1000: 20, 10000: 4, 100000: 2, 1000000: 1}
LGBEND_CONFIG = Path("/local/GPU-Testing/cases/timing/ring-n0064000-p3/AM1.config")

# name: (type as elegant prints it, definition of X, definition of Y (the second copy), options)
#   options: work = the number the cost scales with (kicks, slices); charge = needs a CHARGE element;
#            min, max = smallest and largest particle count; files = function writing extra files;
#            group = report section;
#            cells = FODO cells if not 50; one = one copy, at the end of the line; pair = an element put
#            after that one copy; before = an element put before each copy; needs = a file that must exist; passes = passes whatever the particle count;
#            grows = bytes the element writes per particle per call; fixed = the passes can't be changed
#            (the loss decks: one pass, so that the losses don't stop; IONEFFECTS: its cost grows with the turns);
#            host = the element itself runs on the CPU in gpu-elegant (said under the table);
#            stages = elements spread along the line; setup = lines for &run_setup; first = commands
#            before &run_setup; commands = commands before the beam is made; beta_y = the beam's, if not 5 m;
#            listed = the type as it is listed in track_data.c, when elegant prints another (RBEN is tracked as SBEN);
#            remark = a sentence put after the definition in the report
F = {}


def feature(name, kind, x, y=None, **options):
    F[name] = dict(type=kind, x=x, y=y or x, **options)


G = "Matrix elements"
feature("drift", "DRIF", "DRIF, L=0.2", group=G)
feature("quad-matrix", "QUAD", "QUAD, L=0.2, K1=0.3", "QUAD, L=0.2, K1=-0.3", group=G)
feature("sext-matrix", "SEXT", "SEXT, L=0.2, K2=5", "SEXT, L=0.2, K2=-5", group=G)
feature("sbend-matrix", "SBEN", "SBEN, L=0.5, ANGLE=0.01", "SBEN, L=0.5, ANGLE=-0.01", group=G)
feature("ilmatrix", "ILMATRIX", "ILMATRIX, NUX=0.01, NUY=0.02, BETAX=5, BETAY=5", group=G)
feature("hkick", "HKICK", "HKICK, L=0.1, KICK=1e-6", "HKICK, L=0.1, KICK=-1e-6", group=G)
feature("ehkick", "EHKICK", "EHKICK, L=0.1, KICK=1e-6", "EHKICK, L=0.1, KICK=-1e-6", group=G)
feature("monitor", "MONI", "MONI, L=0.1", group=G)

G = "Exact and kick elements"
feature("edrift", "EDRIFT", "EDRIFT, L=0.2", group=G)
feature("kquad", "KQUAD", "KQUAD, L=0.2, K1=0.3, N_SLICES=8", "KQUAD, L=0.2, K1=-0.3, N_SLICES=8", work=8, group=G)
feature("kquad-synch-rad", "KQUAD", "KQUAD, L=0.2, K1=0.3, N_SLICES=8, SYNCH_RAD=1",
        "KQUAD, L=0.2, K1=-0.3, N_SLICES=8, SYNCH_RAD=1", work=8, group=G)
feature("kquad-isr", "KQUAD", "KQUAD, L=0.2, K1=0.3, N_SLICES=8, SYNCH_RAD=1, ISR=1",
        "KQUAD, L=0.2, K1=-0.3, N_SLICES=8, SYNCH_RAD=1, ISR=1", work=8, min=1000, group=G)
feature("kquad-order6", "KQUAD", "KQUAD, L=0.2, K1=0.3, N_SLICES=8, INTEGRATION_ORDER=6",
        "KQUAD, L=0.2, K1=-0.3, N_SLICES=8, INTEGRATION_ORDER=6", work=8, group=G)
feature("ksext", "KSEXT", "KSEXT, L=0.2, K2=5, N_SLICES=8", "KSEXT, L=0.2, K2=-5, N_SLICES=8", work=8, group=G)
feature("koct", "KOCT", "KOCT, L=0.2, K3=50, N_SLICES=8", "KOCT, L=0.2, K3=-50, N_SLICES=8", work=8, group=G)
feature("mult", "MULT", "MULT, L=0.2, KNL=0.5, ORDER=2, N_KICKS=8", "MULT, L=0.2, KNL=-0.5, ORDER=2, N_KICKS=8", work=8, group=G)
feature("csbend", "CSBEND", "CSBEND, L=0.5, ANGLE=0.01, N_SLICES=8", "CSBEND, L=0.5, ANGLE=-0.01, N_SLICES=8", work=8, group=G)
feature("csbend-synch-rad", "CSBEND", "CSBEND, L=0.5, ANGLE=0.01, N_SLICES=8, SYNCH_RAD=1",
        "CSBEND, L=0.5, ANGLE=-0.01, N_SLICES=8, SYNCH_RAD=1", work=8, group=G)
feature("csbend-isr", "CSBEND", "CSBEND, L=0.5, ANGLE=0.01, N_SLICES=8, SYNCH_RAD=1, ISR=1",
        "CSBEND, L=0.5, ANGLE=-0.01, N_SLICES=8, SYNCH_RAD=1, ISR=1", work=8, min=1000, group=G)
feature("csbend-order6", "CSBEND", "CSBEND, L=0.5, ANGLE=0.01, N_SLICES=8, INTEGRATION_ORDER=6",
        "CSBEND, L=0.5, ANGLE=-0.01, N_SLICES=8, INTEGRATION_ORDER=6", work=8, group=G)
feature("ccbend", "CCBEND", "CCBEND, L=0.2, ANGLE=0.002, K1=4, N_SLICES=8", "CCBEND, L=0.2, ANGLE=-0.002, K1=-4, N_SLICES=8", work=8, group=G)


def lgbend_files(directory):
    if LGBEND_CONFIG.is_file():
        shutil.copy(LGBEND_CONFIG, directory / "AM1.config")


feature("lgbend", "LGBEND", 'LGBEND, CONFIGURATION="AM1.config", N_SLICES=20, INTEGRATION_ORDER=6, OPTIMIZE_FSE=1, COMPENSATEKN=1',
        work=20, files=lgbend_files, needs=LGBEND_CONFIG, cells=10, group=G)
feature("lgbend-isr", "LGBEND", 'LGBEND, CONFIGURATION="AM1.config", N_SLICES=20, INTEGRATION_ORDER=6, OPTIMIZE_FSE=1, '
        'COMPENSATEKN=1, SYNCH_RAD=1, ISR=1', work=20, files=lgbend_files, needs=LGBEND_CONFIG, cells=10, min=1000, group=G)
feature("wiggler", "WIGGLER", "WIGGLER, L=1, K=1, POLES=20", group=G)

G = "RF"
feature("rfca", "RFCA", "RFCA, L=0.2, VOLT=1e5, PHASE=0, FREQ=352e6", group=G)
feature("rfca-change-p0", "RFCA", "RFCA, L=0.2, VOLT=1e5, PHASE=90, FREQ=352e6, CHANGE_P0=1", group=G)
feature("rfdf", "RFDF", "RFDF, L=0.2, VOLTAGE=1e4, PHASE=0, FREQUENCY=2815e6", cells=10, group=G)


def wake_files(directory):
    """A short resonator-like wake: 2,000 points over 100 ps."""
    import math
    rows = []
    for i in range(2000):
        t = i * 5e-14
        w = 1e13 * math.exp(-t / 2e-11) * math.cos(2 * math.pi * 3e10 * t)
        rows.append("%.6e %.6e %.6e" % (t, w, w * 100))
    (directory / "wake.sdds").write_text(
        "SDDS1\n&column name=t, units=s, type=double &end\n&column name=W, units=V/C, type=double &end\n"
        "&column name=Wt, units=V/C/m, type=double &end\n&data mode=ascii, no_row_counts=1 &end\n" + "\n".join(rows) + "\n")


G = "Collective effects"
feature("wake", "WAKE", 'WAKE, INPUTFILE="wake.sdds", TCOLUMN="t", WCOLUMN="W", N_BINS=1024', charge=1, min=1000,
        files=wake_files, group=G)
feature("trwake", "TRWAKE", 'TRWAKE, INPUTFILE="wake.sdds", TCOLUMN="t", WXCOLUMN="Wt", WYCOLUMN="Wt", N_BINS=1024',
        charge=1, min=1000, files=wake_files, group=G)
feature("rfcw", "RFCW", 'RFCW, L=0.2, CELL_LENGTH=0.1, VOLT=1e5, PHASE=0, FREQ=1.3e9, ZWAKEFILE="wake.sdds", '
        'TCOLUMN="t", WZCOLUMN="W", TRWAKEFILE="wake.sdds", WXCOLUMN="Wt", WYCOLUMN="Wt", N_BINS=1024',
        charge=1, min=1000, files=wake_files, group=G)
feature("zlongit", "ZLONGIT", "ZLONGIT, BROAD_BAND=1, RA=1e3, Q=1, FREQ=3e10, N_BINS=1024, BIN_SIZE=1e-13",
        charge=1, min=1000, group=G)
feature("rfmode", "RFMODE", "RFMODE, RA=1e6, Q=1e4, FREQ=352e6, N_BINS=1024, BIN_SIZE=1e-12", charge=1, min=1000, group=G)
feature("trfmode", "TRFMODE", "TRFMODE, RA=1e6, Q=1e4, FREQ=500e6, N_BINS=1024, BIN_SIZE=1e-12", charge=1, min=1000, group=G)
feature("lscdrift", "LSCDRIFT", "LSCDRIFT, L=0.2, BINS=1024, HIGH_FREQUENCY_CUTOFF0=0.2, HIGH_FREQUENCY_CUTOFF1=0.3",
        charge=1, min=1000, group=G)
feature("csrcsbend", "CSRCSBEND", "CSRCSBEND, L=0.5, ANGLE=0.01, N_SLICES=8, BINS=512, SG_HALFWIDTH=1",
        "CSRCSBEND, L=0.5, ANGLE=-0.01, N_SLICES=8, BINS=512, SG_HALFWIDTH=1", charge=1, min=1000, work=8, group=G)
feature("csrdrift", "CSRDRIFT", "CSRDRIFT, L=0.2, USE_STUPAKOV=1, N_KICKS=1", charge=1, min=1000, group=G,
        before="CSRCSBEND, L=0.5, ANGLE=0.01, N_SLICES=8, BINS=512, SG_HALFWIDTH=1")
feature("sreffects", "SREFFECTS", "SREFFECTS, JX=1, JY=1, JDELTA=2, EXREF=1e-10, EYREF=1e-11, SDELTAREF=1e-3, "
        "DDELTAREF=-1e-5, PREF=11742", min=1000, group=G)

G = "Apertures and diagnostics"
feature("rcol", "RCOL", "RCOL, X_MAX=1, Y_MAX=1", group=G)
feature("ecol", "ECOL", "ECOL, X_MAX=1, Y_MAX=1", group=G)
feature("maxamp", "MAXAMP", "MAXAMP, X_MAX=1, Y_MAX=1", group=G)
feature("scraper", "SCRAPER", 'SCRAPER, POSITION=1, INSERT_FROM="+x"', group=G)
feature("watch-parameter", "WATCH", 'WATCH, FILENAME="%s.wp", MODE="parameter"', group=G, one=1, slope_cells=1, disk=1)
feature("watch-centroid", "WATCH", 'WATCH, FILENAME="%s.wc", MODE="centroid"', group=G, one=1, slope_cells=1, disk=1)
feature("watch-coordinate", "WATCH", 'WATCH, FILENAME="%s.wx", MODE="coordinate"', group=G, one=1, grows=56)

# Added after the survey of November 12, 2026 (TIMING-PROCEDURE.md, section 12): the element types that
# took the most time in elegantTestSet's slow decks and had no row here.  Their field and mode files are
# the test set's own.
TEST_SET = Path("/local/elegantTestSet")
GGE = TEST_SET / "bggexp2" / "Q98.ggrad"                  # 6 multipoles on 2,001 planes
FIELD_MAP = TEST_SET / "ftable2" / "Q98.sdds"             # 1,923,201 points
MONOPOLE = TEST_SET / "feedback6" / "monopole.40tile.modes"    # 120 modes
DIPOLE = TEST_SET / "feedback6" / "dipole.40tile.modes"        # 168 modes

G = "Field maps and integrators"
# BGGEXP takes about 3 ms a particle with every plane, so it stops at 10,000 particles and one pass.  Without
# SYMPLECTIC=1 it cannot use fewer planes: with Z_INTERVAL above 1 it loses every particle that is off the axis.
feature("bggexp", "BGGEXP", 'BGGEXP, L=0.238, LFIELD=1.0, FILENAME="%s", STRENGTH=0.01' % GGE,
        'BGGEXP, L=0.238, LFIELD=1.0, FILENAME="%s", STRENGTH=-0.01' % GGE, work=2001, needs=GGE, cells=1,
        passes=1, max=10000, group=G)
feature("bggexp-symplectic", "BGGEXP",
        'BGGEXP, L=0.238, LFIELD=1.0, FILENAME="%s", STRENGTH=0.01, Z_INTERVAL=40, SYMPLECTIC=1' % GGE,
        'BGGEXP, L=0.238, LFIELD=1.0, FILENAME="%s", STRENGTH=-0.01, Z_INTERVAL=40, SYMPLECTIC=1' % GGE,
        work=51, needs=GGE, cells=1, max=100000, group=G)
feature("ftable", "FTABLE", 'FTABLE, L=0.8, INPUT_FILE="%s", N_KICKS=100, SIMPLE_INPUT=1, FACTOR=0.01' % FIELD_MAP,
        'FTABLE, L=0.8, INPUT_FILE="%s", N_KICKS=100, SIMPLE_INPUT=1, FACTOR=-0.01' % FIELD_MAP,
        work=100, needs=FIELD_MAP, cells=1, group=G)
feature("lsrmdltr", "LSRMDLTR", 'LSRMDLTR, L=0.5, PERIODS=10, BU=0.5, METHOD="non-adaptive", N_STEPS=100, ACCURACY=1e-3, '
        'FIELD_EXPANSION="ideal", LASER_WAVELENGTH=8e-7, LASER_PEAK_POWER=1e6, LASER_W0=3e-3', work=100, cells=1, group=G)

G = "Collective effects from files and commands"
feature("frfmode", "FRFMODE", 'FRFMODE, FILENAME="%s", N_CAVITIES=1, BIN_SIZE=1e-12, N_BINS=1024' % MONOPOLE,
        charge=1, min=1000, work=120, needs=MONOPOLE, group=G)
feature("ftrfmode", "FTRFMODE", 'FTRFMODE, FILENAME="%s", N_CAVITIES=1, BIN_SIZE=1e-12, N_BINS=1024' % DIPOLE,
        charge=1, min=1000, work=168, needs=DIPOLE, group=G)
TWISS_MATCHED = "&twiss_output matched = 1, radiation_integrals = 1 &end\n"
TWISS_GIVEN = "&twiss_output matched = 0, beta_x = 5, beta_y = 5 &end\n"
feature("ibscatter", "IBSCATTER", "IBSCATTER", charge=1, min=1000, one=1, commands=TWISS_MATCHED, group=G)
feature("scmult-nonlinear", "SCMULT", "MARK", charge=1, min=1000, commands=TWISS_GIVEN, group=G,
        first='&insert_sceffects name = "[XY]", horizontal = 1, vertical = 1, nonlinear = 1, skip = 1 &end\n')


def contour_files(directory):
    """A square of 1 mm half width, which XFACTOR and YFACTOR narrow."""
    points = [(-1e-3, -1e-3), (1e-3, -1e-3), (1e-3, 1e-3), (-1e-3, 1e-3), (-1e-3, -1e-3)]
    (directory / "contour.sdds").write_text(
        "SDDS1\n&column name=x, units=m, type=double &end\n&column name=y, units=m, type=double &end\n"
        "&data mode=ascii, no_row_counts=1 &end\n" + "\n".join("%.6e %.6e" % p for p in points) + "\n")


# A line that loses particles all along.  Its kick quadrupoles are strong, so the beam is widest at them,
# and the beam comes in matched (beta_y = 2.4 m).  An aperture that narrows in five steps, every ten
# cells, takes about a tenth of the beam.  One pass, so that the losses don't stop.  Particles are lost
# at the drifts and matrix quadrupoles too, and all of the line's extra cost is put down to the KQUAD.
# The beam is at most 48.3 um wide and 13.8 um high (rms) along this line; the limits are 2.58, 2.33, 2.17,
# 2.05 and 1.96 times that.
LIMITS = ((124.4e-6, 35.5e-6), (112.3e-6, 32.1e-6), (104.8e-6, 29.9e-6), (99.2e-6, 28.3e-6), (94.7e-6, 27.0e-6))
KQ = ("KQUAD, L=0.2, K1=2, N_SLICES=8", "KQUAD, L=0.2, K1=-2, N_SLICES=8")
G = "Particle losses"
feature("kquad-losses-none", "KQUAD", *KQ, work=8, min=1000, passes=1, fixed=1, beta_y=2.4, group=G)
feature("kquad-losses-maxamp", "KQUAD", *KQ, work=8, min=1000, passes=1, fixed=1, beta_y=2.4, group=G,
        stages=["MAXAMP, X_MAX=%g, Y_MAX=%g" % limit for limit in LIMITS])
feature("kquad-losses-maxamp-file", "KQUAD", *KQ, work=8, min=1000, passes=1, fixed=1, beta_y=2.4, group=G,
        setup="losses = %s.los,", stages=["MAXAMP, X_MAX=%g, Y_MAX=%g" % limit for limit in LIMITS])
feature("kquad-losses-apcontour-file", "KQUAD", *KQ, work=8, min=1000, passes=1, fixed=1, beta_y=2.4, group=G,
        setup="losses = %s.los,", files=contour_files,
        stages=['APCONTOUR, FILENAME="contour.sdds", XCOLUMN="x", YCOLUMN="y", STICKY=1, XFACTOR=%g, YFACTOR=%g'
                % (x / 1e-3, y / 1e-3) for x, y in LIMITS])

# Added 2026-11-13: the element types the table had no row for.  Their input files are those of the GPU
# feature set (src/gpu/feature-set) and of elegantTestSet.
FEATURE_SET = Path(__file__).resolve().parent.parent / "feature-set"
ION_TEST = TEST_SET / "ionEffects1"


def copies_of(*sources):
    """A `files` function that puts copies of these files beside the deck."""
    def copier(directory):
        for source in sources:
            shutil.copy(source, directory / Path(source).name)
    return copier


def ramp_files(directory):
    """Waveforms that change nothing (factors of 1, no added phase), 1,001 points from 0 to 1 s."""
    rows = ["%.6e 1 0 1 1" % (i * 1e-3) for i in range(1001)]
    (directory / "ramp.sdds").write_text(
        "SDDS1\n&column name=t, units=s, type=double &end\n&column name=Voltage, type=double &end\n"
        "&column name=Phase, type=double &end\n&column name=Frequency, type=double &end\n"
        "&column name=Ratio, type=double &end\n&data mode=ascii, no_row_counts=1 &end\n" + "\n".join(rows) + "\n")


G = "Exact and kick elements"
FMULT_FILE = FEATURE_SET / "magnet-fmult" / "fmult-sext.sdds"
feature("fmult", "FMULT", 'FMULT, L=0.2, N_KICKS=8, FILENAME="fmult-sext.sdds"', work=8, files=copies_of(FMULT_FILE),
        needs=FMULT_FILE, group=G)

G = "Field maps and integrators"
CWIGGLER_FILE = FEATURE_SET / "data" / "cwig_by_normal.sdds"
UKICKMAP_FILE = FEATURE_SET / "magnet-ukickmap-loss-compaction" / "ukickmap.sdds"
GKICKMAP_FILE = FEATURE_SET / "data" / "kickmap.sdds"
feature("cwiggler", "CWIGGLER", 'CWIGGLER, L=1, PERIODS=10, STEPS_PER_PERIOD=12, INTEGRATION_ORDER=4, B_MAX=0.5, '
        'BY_FILE="cwig_by_normal.sdds"', work=120, files=copies_of(CWIGGLER_FILE), needs=CWIGGLER_FILE, cells=1, group=G)
feature("ukickmap", "UKICKMAP", 'UKICKMAP, L=0.4, INPUT_FILE="ukickmap.sdds", N_KICKS=16, PERIODS=16, KREF=2.0',
        work=16, files=copies_of(UKICKMAP_FILE), needs=UKICKMAP_FILE, group=G)
feature("gkickmap", "GKICKMAP", 'GKICKMAP, L=0.2, INPUT_FILE="kickmap.sdds", N_KICKS=4', work=4,
        files=copies_of(GKICKMAP_FILE), needs=GKICKMAP_FILE, group=G)

G = "Apertures and diagnostics"
feature("apcontour", "APCONTOUR", 'APCONTOUR, FILENAME="contour.sdds", XCOLUMN="x", YCOLUMN="y"', files=contour_files, group=G)
feature("apcontour-sticky", "APCONTOUR", 'APCONTOUR, FILENAME="contour.sdds", XCOLUMN="x", YCOLUMN="y", STICKY=1',
        files=contour_files, group=G)
feature("taperapc", "TAPERAPC", "TAPERAPC, L=0.04, RSTART=0.02, REND=0.018", group=G)
feature("taperapr", "TAPERAPR", "TAPERAPR, L=0.04, XSTART=0.018, XEND=0.016, YSTART=0.014, YEND=0.012", group=G)
feature("speedbump", "SPEEDBUMP", 'SPEEDBUMP, L=0.04, CHORD=0.03, HEIGHT=0.001, POSITION=0.02, INSERT_FROM="+x"', group=G)

G = "Feedback, ramps and ions"
# A pickup and its driver, once in the line: the time is the pair's, per pass.
feature("tfeedback", "TFBPICKUP", 'TFBPICKUP, ID="FY", PLANE="y", A0=1, A1=-1, BUNCHED_BEAM_MODE=0',
        pair='TFBDRIVER, ID="FY", STRENGTH=-0.04, DELAY=8, A0=1, GAIN_FACTOR=1, BUNCHED_BEAM_MODE=0', one=1, slope_cells=1, group=G)
feature("ramprf", "RAMPRF", 'RAMPRF, L=0.2, VOLT=1e5, PHASE=0, FREQ=352e6, VOLT_WAVEFORM="ramp.sdds=t+Voltage", '
        'PHASE_WAVEFORM="ramp.sdds=t+Phase", FREQ_WAVEFORM="ramp.sdds=t+Frequency"', files=ramp_files, group=G)
feature("rampp", "RAMPP", 'RAMPP, WAVEFORM="ramp.sdds=t+Ratio"', files=ramp_files, group=G)
# IONEFFECTS: its cost grows with the ions it has made, so the passes are the same at every particle count,
# and the time is the mean over them (`fixed`: where the rest of the deck is on the GPU, the deck is compared
# with the line at the same passes).  One bunch; the gas and the ion properties are elegantTestSet's.
ION_FILES = (ION_TEST / "gasMix.sdds", ION_TEST / "ionProperties.sdds")
feature("ioneffects", "IONEFFECTS", "IONEFFECTS", charge=1, min=1000, max=100000, passes=20, cells=10, fixed=1,
        remark="Since 2026-11-15 gpu-elegant tracks the ions on a host copy of the beam, as elegant does, and makes the "
               "Gaussian field's kicks to the electrons on the GPU",
        files=copies_of(*ION_FILES), needs=ION_FILES[0], group=G,
        commands='&ion_effects pressure_profile = "gasMix.sdds", ion_properties = "ionProperties.sdds", '
                 'field_calculation_method = "gaussian", macro_ions = 1, ion_span[0] = 0.011, 0.011 &end\n')

# Added on the evening of 2026-11-14 (table 11): a row for every element type the table had none for, so that
# it holds all of elegant's types.  Settings are mild, so that the beam survives the 100 copies; where an input
# file is needed it is elegantTestSet's or the GPU feature set's.
DATA = FEATURE_SET / "data"
SET_FILES = {
    "helper.rmat": FEATURE_SET / "matrix-helpers" / "helper.rmat",
    "poly.sdds": FEATURE_SET / "magnet-polyseries-few-particles" / "poly.sdds",
    "srquad.bmapxy": TEST_SET / "bmapxy3" / "srquad.bmapxy",
    "bmxyz_field.sdds": DATA / "bmxyz_field.sdds",
    "brat.sdds": TEST_SET / "brat1" / "GRID_for_M1_L_bend_5_pole_24_1st_9_24_2nd_8_14.sdds",
    "Q1.gradient.plus": TEST_SET / "boffaxe1" / "Q1.gradient.plus",
    "mapsolenoid_map.sdds": DATA / "mapsolenoid_map.sdds",
    "rftmez0_ez.sdds": DATA / "rftmez0_ez.sdds",
    "impedance.sdds": DATA / "impedance.sdds",
    "wake_cwake.sdds": DATA / "wake_cwake.sdds",
    "lrwake.sdds": DATA / "lrwake.sdds",
    "gaussian.sdds": TEST_SET / "dscatter1" / "gaussian.sdds",
}


def set_file(name):
    """The `files` and `needs` options for one of SET_FILES, put beside the deck under its own name."""
    source = SET_FILES[name]

    def copier(directory):
        shutil.copy(source, directory / name)
    return dict(files=copier, needs=source)


G = "Matrix elements"
feature("rben", "SBEN", "RBEN, L=0.5, ANGLE=0.01", "RBEN, L=0.5, ANGLE=-0.01", listed="RBEN", group=G,
        remark="elegant makes an `RBEN` into an `SBEN` with its edge angles when it reads the lattice, and prints it "
               "as one")
feature("octu", "OCTU", "OCTU, L=0.2, K3=50", "OCTU, L=0.2, K3=-50", group=G)
feature("sole", "SOLE", "SOLE, L=0.2, KS=0.5", "SOLE, L=0.2, KS=-0.5", group=G)
feature("vkick", "VKICK", "VKICK, L=0.1, KICK=1e-6", "VKICK, L=0.1, KICK=-1e-6", group=G)
feature("kicker", "KICKER", "KICKER, L=0.1, HKICK=1e-6, VKICK=1e-6", "KICKER, L=0.1, HKICK=-1e-6, VKICK=-1e-6", group=G)
feature("hmon", "HMON", "HMON, L=0.1", group=G)
feature("vmon", "VMON", "VMON, L=0.1", group=G)
feature("qufringe", "QUFRINGE", "QUFRINGE, L=0.03, K1=0.25, DIRECTION=1", "QUFRINGE, L=0.03, K1=0.25, DIRECTION=-1", group=G)
feature("magnify", "MAGNIFY", "MAGNIFY, MX=1.00002, MXP=0.99998", "MAGNIFY, MX=0.99998, MXP=1.00002", group=G)
feature("reflect", "REFLECT", "REFLECT", group=G)
feature("lthinlens", "LTHINLENS", "LTHINLENS, FX=100, FY=100", "LTHINLENS, FX=-100, FY=-100", group=G)
feature("lmirror", "LMIRROR", "LMIRROR, RX=1e4, RY=1e4", group=G)
feature("bedge", "BEDGE", "BEDGE, BETA=0.015, RHO=30, EXIT=0", "BEDGE, BETA=0.015, RHO=30, EXIT=1", group=G)
feature("ematrix", "EMATRIX", "EMATRIX, L=0.12, ORDER=1, R11=1, R12=0.12, R22=1, R33=1, R34=0.12, R44=1, R55=1, R66=1", group=G)
feature("matr", "MATR", 'MATR, L=0.15, ORDER=1, FILENAME="helper.rmat"', group=G, **set_file("helper.rmat"))
feature("alph", "ALPH", "ALPH, XMAX=1, ORDER=2", group=G)

G = "Exact and kick elements"
feature("evkick", "EVKICK", "EVKICK, L=0.1, KICK=1e-6", "EVKICK, L=0.1, KICK=-1e-6", group=G)
feature("ekicker", "EKICKER", "EKICKER, L=0.1, HKICK=1e-6, VKICK=1e-6", "EKICKER, L=0.1, HKICK=-1e-6, VKICK=-1e-6", group=G)
feature("kquse", "KQUSE", "KQUSE, L=0.2, K1=0.3, K2=5, N_SLICES=8", "KQUSE, L=0.2, K1=-0.3, K2=-5, N_SLICES=8", work=8, group=G)
feature("dqcor", "DQCOR", "DQCOR, L=0.2, K1=0.3, N_SLICES=8", "DQCOR, L=0.2, K1=-0.3, N_SLICES=8", work=8, group=G)
feature("ksbend", "KSBEND", "KSBEND, L=0.5, ANGLE=0.01, N_KICKS=8", "KSBEND, L=0.5, ANGLE=-0.01, N_KICKS=8", work=8, group=G)
feature("kpoly", "KPOLY", 'KPOLY, COEFFICIENT=1e-3, ORDER=2, PLANE="x"', 'KPOLY, COEFFICIENT=-1e-3, ORDER=2, PLANE="y"', group=G)
feature("hkpoly", "HKPOLY", "HKPOLY, K30=-1, K12=3", "HKPOLY, K30=1, K12=-3", group=G)
feature("stray", "STRAY", "STRAY, L=0.2, LBX=1e-6, LBY=1e-6", "STRAY, L=0.2, LBX=-1e-6, LBY=-1e-6", group=G)
# The map is one turn of a ring (the feature set's PAR map): one copy a cell, as two in a row lose the beam
feature("polyseries", "POLYSERIES", 'POLYSERIES, L=30.667, FILENAME="poly.sdds"', "MARK", cells=10, group=G, **set_file("poly.sdds"))

G = "Field maps and integrators"
# The Lorentz family's fixed step is ACCURACY times the length, so ACCURACY=1e-2 is about 200 steps
NI = 'MODEL="hard-edge", METHOD="non-adaptive runge-kutta", ACCURACY=1e-2, ADJUST_BOUNDARY=0, ADJUST_FIELD=0, FUDGE_PATH_LENGTH=0'
feature("nibend", "NIBEND", "NIBEND, L=0.5, ANGLE=0.01, " + NI, work=200, cells=1, group=G)
feature("nisept", "NISEPT", 'NISEPT, L=0.5, ANGLE=0.01, FLEN=0.05, ACCURACY=1e-2, METHOD="non-adaptive runge-kutta", MODEL="linear"',
        work=200, cells=1, group=G)
feature("bmapxy", "BMAPXY", 'BMAPXY, L=0.01, METHOD="non-adaptive runge-kutta", ACCURACY=1e-2, FILENAME="srquad.bmapxy", STRENGTH=1',
        'BMAPXY, L=0.01, METHOD="non-adaptive runge-kutta", ACCURACY=1e-2, FILENAME="srquad.bmapxy", STRENGTH=-1',
        work=200, cells=1, group=G, **set_file("srquad.bmapxy"))
feature("bmxyz", "BMXYZ", 'BMXYZ, L=0.8, ACCURACY=1e-2, METHOD="non-adaptive runge-kutta", FILENAME="bmxyz_field.sdds", '
        'STRENGTH=0.1, XY_INTERPOLATION_ORDER=1',
        'BMXYZ, L=0.8, ACCURACY=1e-2, METHOD="non-adaptive runge-kutta", FILENAME="bmxyz_field.sdds", '
        'STRENGTH=-0.1, XY_INTERPOLATION_ORDER=1', work=200, cells=1, group=G, **set_file("bmxyz_field.sdds"))
# BRAT takes about 36 ms a particle (elegantTestSet's brat1 geometry, ACCURACY=1e-12), so 1,000 particles at most
BRAT = ('BRAT, L=2.100479068951827, ANGLE=0.024357075, XVERTEX=0.9005e-2, ZVERTEX=-29.6096e-2, XENTRY=-0.00276374013551943, '
        'ZENTRY=-1.05028438609493, XEXIT=-0.00278009446616334, ZEXIT=1.05015287200494, FSE=4.805708870196813e-04, '
        'DZMAP=-8.512060517799165e-06, ACCURACY=1e-12, FILENAME="brat.sdds"')
feature("brat", "BRAT", BRAT, cells=1, max=1000, passes=1, group=G, **set_file("brat.sdds"))
feature("boffaxe", "BOFFAXE", 'BOFFAXE, L=0.25, FILENAME="Q1.gradient.plus", FIELD_COLUMN="Gradient", Z_COLUMN="z", EXPANSION_ORDER=7, '
        'STRENGTH=0.01', 'BOFFAXE, L=0.25, FILENAME="Q1.gradient.plus", FIELD_COLUMN="Gradient", Z_COLUMN="z", '
        'EXPANSION_ORDER=7, STRENGTH=-0.01', cells=1, group=G, **set_file("Q1.gradient.plus"))
# gpu_from: the one-cell deck is on the GPU from 10,000 particles (since 2026-11-15), but there the CUDA start-up,
# charged to its first matrix element, outweighs elegant's whole tracking, which the test below takes for a deck
# that stayed on the CPU
feature("gfwiggler", "GFWIGGLER", "GFWIGGLER, L=1, B_MAX=0.5, PERIODS=10", cells=1, gpu_from=10000, group=G)
feature("mapsolenoid", "MAPSOLENOID", 'MAPSOLENOID, L=0.1991984, RCOLUMN="r", ZCOLUMN="z", BRCOLUMN="Br", BZCOLUMN="Bz", '
        'METHOD="non-adaptive runge-kutta", FACTOR=0.5, N_STEPS=100, INPUTFILE="mapsolenoid_map.sdds"', work=100, cells=1, group=G,
        **set_file("mapsolenoid_map.sdds"))

G = "RF"
feature("rftmez0", "RFTMEZ0", 'RFTMEZ0, L=0.3184, FREQUENCY=115e6, EZ_PEAK=1e5, PHASE=0, INPUTFILE="rftmez0_ez.sdds", ZCOLUMN="z", '
        'EZCOLUMN="Ez", METHOD="non-adaptive", N_STEPS=50', work=50, cells=1, group=G, **set_file("rftmez0_ez.sdds"))
feature("twla", "TWLA", 'TWLA, L=0.3149, PHASE=0, EZ=1e5, METHOD="non-adaptive", N_STEPS=100, ACCURACY=1e-9', work=100, cells=1, group=G)
feature("twmta", "TWMTA", 'TWMTA, L=0.3, FREQUENCY=2856e6, EZ=1e5, KX=1, BETA_WAVE=0.99, METHOD="non-adaptive runge-kutta", '
        'N_STEPS=100, ACCURACY=1e-6', work=100, cells=1, group=G)
feature("tmcf", "TMCF", 'TMCF, L=0.2, EZ=1e5, METHOD="non-adaptive runge-kutta", N_STEPS=100, ACCURACY=1e-9', work=100, cells=1, group=G)
feature("cepl", "CEPL", 'CEPL, L=0.2, RAMP_TIME=1e-9, VOLTAGE=1e3, GAP=0.01, METHOD="non-adaptive runge-kutta", N_STEPS=100, '
        'ACCURACY=1e-9', work=100, cells=1, group=G)
feature("twpl", "TWPL", 'TWPL, L=0.2, RAMP_TIME=1e-9, VOLTAGE=1e3, GAP=0.01, METHOD="non-adaptive runge-kutta", N_STEPS=100, '
        'ACCURACY=1e-9', work=100, cells=1, group=G)
feature("rmdf", "RMDF", "RMDF, L=0.2, RAMP_TIME=1, VOLTAGE=1e3, GAP=0.01", "RMDF, L=0.2, RAMP_TIME=1, VOLTAGE=-1e3, GAP=0.01", group=G)
feature("modrf", "MODRF", "MODRF, L=0.2, VOLT=1e5, PHASE=0, FREQ=352e6, PMMAG=1, PMFREQ=1e3", group=G)
feature("rftm110", "RFTM110", "RFTM110, FREQUENCY=2856e6, VOLTAGE=1e3, PHASE=0", group=G)
feature("mrfdf", "MRFDF", "MRFDF, A1=1e3, FREQUENCY1=2856e6", group=G)
feature("shrfdf", "SHRFDF", "SHRFDF, V0=1e3, PERIOD_LENGTH=0.1", group=G)

G = "Collective effects"
feature("ztransverse", "ZTRANSVERSE", "ZTRANSVERSE, BROAD_BAND=1, RS=1e3, Q=1, FREQ=3e10, N_BINS=1024, BIN_SIZE=1e-13",
        charge=1, min=1000, group=G)
def impedance_files(directory):
    """A weak broad-band resonator (3e10 Hz, Q 1, 10 ohm, 1 kohm/m) sampled every 0.5 GHz, so the wake spans 2 ns."""
    import math
    rows = []
    for i in range(4097):
        f = i * 5e8
        x = 1 if f == 0 else 1 / (1 + (3e10 / f - f / 3e10) ** 2)
        zr, zi = 10 * x, 0 if f == 0 else -10 * x * (3e10 / f - f / 3e10)
        rows.append("%.6e %.6e %.6e %.6e %.6e" % (f, zr, zi, zr * 100, zi * 100))
    (directory / "impedance.sdds").write_text(
        "SDDS1\n&column name=f, units=Hz, type=double &end\n&column name=ReZL, type=double &end\n"
        "&column name=ImZL, type=double &end\n&column name=ReZt, type=double &end\n&column name=ImZt, type=double &end\n"
        "&data mode=ascii, no_row_counts=1 &end\n" + "\n".join(rows) + "\n")


feature("impedance", "IMPEDANCE", 'IMPEDANCE, INPUTFILE="impedance.sdds", FREQCOLUMN="f", ZL_REAL="ReZL", ZL_IMAG="ImZL", '
        'ZXD_REAL="ReZt", ZXD_IMAG="ImZt", ZYD_REAL="ReZt", ZYD_IMAG="ImZt"', charge=1, min=1000, group=G,
        files=impedance_files)
feature("cwake", "CWAKE", 'CWAKE, INPUTFILE="wake_cwake.sdds", TCOLUMN="t", WZCOLUMN="Wz", WXCOLUMN="Wx", WYCOLUMN="Wy", '
        'QXCOLUMN="Qx", QYCOLUMN="Qy", CXCOLUMN="Cx", CYCOLUMN="Cy", N_BINS=1024', charge=1, min=1000, group=G,
        **set_file("wake_cwake.sdds"))
feature("lrwake", "LRWAKE", 'LRWAKE, INPUTFILE="lrwake.sdds", TCOLUMN="time", WXCOLUMN="WxDipole", WYCOLUMN="WyDipole", '
        'WZCOLUMN="Wz", QXCOLUMN="WxQuadrupole", QYCOLUMN="WyQuadrupole", TURNS_TO_KEEP=3', charge=1, min=1000, group=G,
        **set_file("lrwake.sdds"))
feature("corgpipe", "CORGPIPE", "CORGPIPE, L=0.2, RADIUS=3e-3, DEPTH=450e-6, PERIOD=1e-3, GAP=750e-6, N_BINS=400",
        charge=1, min=1000, group=G)
feature("corgplates", "CORGPLATES", "CORGPLATES, L=0.2, HALFGAP=2e-3, PERIOD=0.5e-3, DEPTH=0.5e-3, TMAX=0",
        charge=1, min=1000, group=G)
feature("beambeam", "BEAMBEAM", "BEAMBEAM, CHARGE=1e-9, XSIZE=30e-6, YSIZE=5e-6", group=G)

G = "Apertures and diagnostics"
feature("taperape", "TAPERAPE", "TAPERAPE, L=0.04, ASTART=0.02, AEND=0.018, BSTART=0.014, BEND=0.012", group=G)
feature("histogram", "HISTOGRAM", 'HISTOGRAM, FILENAME="%s.h", BINS=100', group=G, one=1, slope_cells=1, disk=1)
feature("slice", "SLICE", 'SLICE, N_SLICES=10, FILENAME="%s.sl"', group=G, one=1, slope_cells=1, disk=1)


def mhistogram_files(directory):
    """MHISTOGRAM's bin counts (it needs INPUT_BINS: without it elegant stops with a segmentation fault): 100 bins
    for each 1-D histogram, 50 by 50 for the 2-D ones, 10 and 5 a coordinate for the 4-D and 6-D ones."""
    (directory / "mhist-bins.sdds").write_text(
        "SDDS1\n&column name=Bins_1D, type=long &end\n&column name=Bins_2D, type=long &end\n"
        "&column name=Bins_4D, type=long &end\n&column name=Bins_6D, type=long &end\n"
        "&data mode=ascii, no_row_counts=1 &end\n" + "100 50 10 5\n" * 6)


feature("mhistogram", "MHISTOGRAM", 'MHISTOGRAM, FILE1D="%s.mh1", FILE2DH="%s.mh2", INPUT_BINS="mhist-bins.sdds"',
        group=G, one=1, slope_cells=1, disk=1, files=mhistogram_files)

G = "Beam adjustment and filters"
feature("malign", "MALIGN", "MALIGN, DX=1e-6", "MALIGN, DX=-1e-6", group=G)
feature("rotate", "ROTATE", "ROTATE, TILT=0.01", "ROTATE, TILT=-0.01", group=G)
feature("center", "CENTER", "CENTER", group=G)
feature("energy", "ENERGY", "ENERGY", group=G)
feature("emittance", "EMITTANCE", "EMITTANCE, EMITX=1e-10, EMITY=1e-11", group=G)
feature("remcor", "REMCOR", "REMCOR, X=1, XP=1, Y=1, YP=1, WITH=6", group=G)
# FROM_BEAM=1 needs more than one particle
feature("twiss", "TWISS", "TWISS, BETAX=5, BETAY=5, FROM_BEAM=1, APPLY_ONCE=0", min=1000, group=G)
feature("sample", "SAMPLE", "SAMPLE, FRACTION=1", group=G)
feature("pfilter", "PFILTER", "PFILTER, DELTALIMIT=0.1", group=G)
feature("clean", "CLEAN", 'CLEAN, MODE="absvalue", XLIMIT=1, XPLIMIT=1, YLIMIT=1, YPLIMIT=1, TLIMIT=1, DELTALIMIT=1', group=G)

G = "Scattering and material"
feature("scatter", "SCATTER", "SCATTER, XP=1e-9, YP=1e-9", group=G)
feature("dscatter", "DSCATTER", 'DSCATTER, PLANE="xp", FILENAME="gaussian.sdds", VALUENAME="z", PDFNAME="g", FACTOR=1e-9',
        group=G, **set_file("gaussian.sdds"))
feature("matter", "MATTER", "MATTER, L=1e-7, Z=13, A=26.98, XO=0.089, RHO=2700, PLIMIT=0.1, ENERGY_DECAY=1", group=G)
feature("cbscat", "CBSCAT", "CBSCAT, LASER_WAVELENGTH=8e-7, PULSE_ENERGY=1e-6, SIGMA_X=1e-4, SIGMA_Y=1e-4, LASER_PULSE_LENGTH=1e-12",
        group=G)
# PEPPOT reads its holes' positions from the lines after its definition: one hole on the axis
feature("peppot", "PEPPOT", "PEPPOT, L=1e-3, RADII=1e-3, TRANSMISSION=1, N_HOLES=1\n0 0", group=G)
feature("tscatter", "TSCATTER", "TSCATTER", group=G)

def flat_waveform_files(directory):
    """A kicker waveform of 1,000 points, flat at 1, from -1 to 1 s.  Each element sets its time origin on its
    first call, so the particles' times stay within microseconds of 0 for the whole run: every call kicks, and
    finds its point after the same 500 steps of the element's linear search, as a call at the middle of a
    1,000-point pulse does.  (With the test set's pulse of 135 ns, every call after the first falls outside
    it and only drifts.)"""
    rows = ["%.9e 1" % (-1 + 2 * i / 999) for i in range(1000)]
    (directory / "flat-waveform.sdds").write_text(
        "SDDS1\n&column name=t, units=s, type=double &end\n&column name=A, type=double &end\n"
        "&data mode=ascii, no_row_counts=1 &end\n" + "\n".join(rows) + "\n")


FLAT_REMARK = ("The waveform has 1,000 points, flat at 1 from -1 to 1 s, so that every call kicks and searches as far "
               "as at the middle of a 1,000-point pulse")
G = "Feedback, ramps and ions"
feature("bumper", "BUMPER", 'BUMPER, L=0.5, ANGLE=1e-6, WAVEFORM="flat-waveform.sdds=t+A"', group=G,
        files=flat_waveform_files, remark=FLAT_REMARK)
feature("mbumper", "MBUMPER", 'MBUMPER, L=0.05, STRENGTH=5e-3, N_KICKS=4, WAVEFORM="flat-waveform.sdds=t+A", ORDER=1',
        group=G, files=flat_waveform_files, remark=FLAT_REMARK)
# An optical stochastic cooling pickup and its kicker, once in the line: the time is the pair's, per pass.
# (LAMBDA_RAD is 0 by default; INCOHERENT_MODE=0 leaves out the sum over all pairs of particles)
feature("cooler", "CPICKUP", 'CPICKUP, ID="OSC"', pair='CKICKER, ID="OSC", STRENGTH=1e-9, LAMBDA_RAD=1e-6, NU=10, '
        'TRANSVERSE_MODE=0, INCOHERENT_MODE=0', one=1, slope_cells=1, group=G)

G = "Markers, branches and scripts"
feature("mark", "MARK", "MARK", group=G)
feature("charge", "CHARGE", "CHARGE, TOTAL=1e-9", one=1, slope_cells=1, group=G)
feature("recirc", "RECIRC", "RECIRC", one=1, slope_cells=1, group=G)
feature("floor", "FLOOR", "FLOOR", group=G)
feature("trcount", "TRCOUNT", "TRCOUNT", one=1, slope_cells=1, group=G)
feature("branch", "BRANCH", "BRANCH, COUNTER=0", group=G)
feature("rimult", "RIMULT", "RIMULT", group=G)
feature("polar", "POLAR", "POLAR", group=G,
        remark="It acts on its `ON_PASS` (0) only; the time per call is that of the other passes, and in gpu-elegant "
               "the pass it acts on, with the beam's trip to the host, is in the table of costs paid once")
feature("script", "SCRIPT", 'SCRIPT, COMMAND="cp %i %o"', one=1, slope_cells=1, group=G)

# Element types with no row, and why (the report lists them)
NO_ROW = {
    "TUBEND": "loses every particle on its first call with any settings tried; not in elegant's dictionary",
    "TFBDRIVER": "timed with its TFBPICKUP, in the `tfeedback` row",
    "CKICKER": "timed with its CPICKUP, in the `cooler` row",
}

# Whole-run features: the extra wall time over the same deck without the feature, per element call.
W = {
    "output-centroid": dict(setup="centroid = %s.cen,", title="`centroid` file in &run_setup"),
    "output-sigma": dict(setup="sigma = %s.sig,", title="`sigma` file in &run_setup"),
    "output-final-beam": dict(setup="output = %s.out,", title="`output` (final beam) file in &run_setup", per="particle"),
    "twiss": dict(command="&twiss_output filename = %s.twi, matched = 0, beta_x = 5, beta_y = 5 &end", title="&twiss_output",
                  per="element", no_track=1),
}

# Searches: wall time per (grid point x turn x element), against the cost of tracking one particle.
S = {
    "frequency-map": dict(points=400, turns=256, command=
                          "&frequency_map output = %s.fma, xmin = 1e-5, xmax = 1e-3, ymin = 1e-5, ymax = 1e-3, nx = 20, ny = 20 &end"),
    "dynamic-aperture": dict(points=11 * 21, turns=256, command=
                             '&find_aperture output = %s.aper, mode = "n-lines", n_lines = 11, nx = 21, xmax = 0.02, ymax = 0.02 &end'),
}


def lattice(spec, single_watch=False):
    lines = ["Q1: QUAD, L=0.2, K1=0.9", "Q2: QUAD, L=0.2, K1=-0.9", "D: DRIF, L=0.4",
             "X: " + spec["x"], "Y: " + spec["y"], "C: CHARGE, TOTAL=1e-9", "M0: MARK"]
    if spec.get("before"):
        lines.append("B: " + spec["before"])
    if spec.get("pair"):
        lines.append("W: " + spec["pair"])
    x = "B, X" if spec.get("before") else "X"
    y = "B, Y" if spec.get("before") else "Y"
    if spec.get("scaffold"):
        lines.append("CELL: LINE=(Q1, D, %sD, Q2, D, %sD)" % (("B, ", "B, ") if spec.get("before") else ("", "")))
        lines.append("BL: LINE=(%s%d*CELL)" % ("C, " if spec.get("charge") else "", spec.get("cells", CELLS)))
    elif spec.get("one"):      # one copy at the end of the line (output elements)
        lines.append("CELL: LINE=(Q1, D, D, Q2, D, D)")
        lines.append("BL: LINE=(%s%d*CELL, X%s)" % ("C, " if spec.get("charge") else "", spec.get("cells", CELLS),
                                                  ", W" if spec.get("pair") else ""))
    else:
        lines.append("CELL: LINE=(Q1, D, %s, D, Q2, D, %s, D)" % (x, y))
        body = "%d*CELL" % spec.get("cells", CELLS)
        if spec.get("stages"):     # elements spread evenly along the line, one before each group of cells
            stages = spec["stages"]
            lines += ["A%d: %s" % (i + 1, definition) for i, definition in enumerate(stages)]
            body = ", ".join("A%d, %d*CELL" % (i + 1, spec.get("cells", CELLS) // len(stages)) for i in range(len(stages)))
        lines.append("BL: LINE=(%s%s)" % ("C, " if spec.get("charge") else "", body))
    return "\n".join(lines) + "\n"


ELE = """%(first)s&run_setup
  lattice = run.lte,
  use_beamline = BL,
  p_central = 11742,
  default_order = 2,
  random_number_seed = 987654321,
  show_element_timing = 1,
  %(setup)s
&end

&run_control
  n_passes = %(passes)d
&end
%(before_beam)s
&bunched_beam
  n_particles_per_bunch = %(particles)d,
  emit_x = 1e-10, emit_y = 1e-11, beta_x = 5, beta_y = %(beta_y)g,
  sigma_dp = 1e-3, sigma_s = 1e-3,
  distribution_type[0] = 3*"gaussian", distribution_cutoff[0] = 3*3,
  enforce_rms_values[0] = 3*1
&end
%(after_beam)s
"""


def write_deck(directory, spec, particles, setup="", command="", track=True):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "run.lte").write_text(lattice(spec))
    if spec.get("files"):
        spec["files"](directory)
    passes = spec.get("passes", PASSES[particles])
    (directory / "run.ele").write_text(ELE % dict(
        first=spec.get("first", ""), setup=setup or spec.get("setup", ""), passes=passes, particles=particles,
        beta_y=spec.get("beta_y", 5),
        before_beam=command if not track else spec.get("commands", ""), after_beam=("&track &end\n" if track else "")))


def search_deck(directory, name):
    """A FODO ring-like line with kick quadrupoles, sextupoles and bends for the search commands."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "run.lte").write_text(
        "Q1: KQUAD, L=0.2, K1=0.9, N_SLICES=8\nQ2: KQUAD, L=0.2, K1=-0.9, N_SLICES=8\nD: DRIF, L=0.4\n"
        "X: KSEXT, L=0.2, K2=5, N_SLICES=8\nY: CSBEND, L=0.5, ANGLE=0.01, N_SLICES=8\n"
        "CELL: LINE=(Q1, D, X, D, Q2, D, Y, D)\nBL: LINE=(20*CELL)\n")
    (directory / "run.ele").write_text(
        "&run_setup\n  lattice = run.lte,\n  use_beamline = BL,\n  p_central = 11742,\n  default_order = 2\n&end\n\n"
        "&run_control\n  n_passes = %d\n&end\n\n%s\n" % (S[name]["turns"], S[name]["command"]))


def do_make(args):
    root = Path(args.cases)
    made = 0
    for name, spec in F.items():
        if spec.get("needs") and not Path(spec["needs"]).is_file():
            print("skipped %s: %s is not there" % (name, spec["needs"]))
            continue
        for particles in COUNTS["gpu"]:
            if particles >= spec.get("min", 1):
                write_deck(root / name / ("n%07d" % particles), spec, particles)
                made += 1
        if ELEGANT_LARGE >= spec.get("min", 1) and ELEGANT_LARGE <= spec.get("max", ELEGANT_LARGE):
            write_deck(root / name / ("e%07d" % ELEGANT_LARGE),
                       dict(spec, cells=max(1, spec.get("cells", CELLS) // 10)), ELEGANT_LARGE)
    for particles in COUNTS["gpu"]:     # the line without any element under test
        write_deck(root / "scaffold" / ("n%07d" % particles), dict(x="MARK", y="MARK", scaffold=1), particles)
    base = F["drift"]
    for particles in (1000, 100000):
        write_deck(root / "whole-run" / "baseline" / ("n%07d" % particles), base, particles)
        for name, w in W.items():
            if not w.get("no_track"):
                write_deck(root / "whole-run" / name / ("n%07d" % particles), base, particles, setup=w["setup"])
    write_deck(root / "whole-run" / "twiss" / "n0001000", base, 1000, command=W["twiss"]["command"], track=False)
    for name in S:
        search_deck(root / "search" / name, name)
    print("Wrote %d element decks, the whole-run decks and %d search decks under %s" % (made, len(S), root))


def run_deck(deck, config, args, output, label, environment=None, flush=False):
    work = output / "work" / label / config
    if flush:      # start with nothing waiting to be written, so this run's file isn't held up by the last one's
        os.sync()
    record = tc.run_once({"directory": str(deck), "commands": ["run.ele"], "environment": environment or {}},
                         config, args, work)
    record["elements"] = pr.element_times(work / "timing.log")
    if record["status"] != 0:
        failed = output / "failed" / label / config
        if failed.exists():
            shutil.rmtree(failed)
        shutil.copytree(work, failed)
    shutil.rmtree(work)
    return record


SCAFFOLD_REPEATS = 7

# How a gpu-elegant cell is measured.
#
# gpu-elegant queues cheap elements and launches its kernels without waiting for them, so its own
# per-element times put cost in the wrong place, and up to about a thousand launches of work can be
# missing from them altogether at the end of a run.  A run also holds about 0.1 s of start-up, which
# differs by 10 ms and more from one run to the next, and costs that an element pays once (opening a file,
# the beam's first trip to the host).  So the cost of an element is taken from wall-clock time, as a slope:
#
#     the deck with `lo` and with `hi` passes, and the line without the element (the scaffold, same
#     cells) with `lo` and `hi` passes; cost per call = (what the deck gained from lo to hi, less what
#     the scaffold gained) / (calls the deck gained).
#
# Everything that doesn't grow with the passes drops out.  The four runs are made SLOPE_ROUNDS times in
# turn and the median of the rounds' values is used.  A deck whose passes can't change (the loss decks) is
# compared with a scaffold of the same passes instead, FIXED_ROUNDS times.
#
# How many passes.  A run's start-up depends on what the machine did in the seconds before it, and can sit
# START_SHIFT off for minutes at a time, which the scatter of the rounds doesn't show (it put 4 us on every
# 100,000-particle cell of one group of decks).  So `hi` is chosen for the element to gain TARGET of wall time
# from `lo` to `hi` (passes_for): the first time from the plain runs' estimate, and again from the measured
# value where that estimate was too high (a "thin" cell: see cell()).  A deck with one copy of its element
# (`slope_cells`) is run in a line of one cell, where thousands of passes cost little.  A time is believed
# no closer than START_SHIFT spread over the calls gained.
SLOPE_ROUNDS = 3
GROWS_ROUNDS = 5               # for a deck that writes to a file at every call (`grows`, `disk`): disk time scatters more
FIXED_ROUNDS = 7
TARGET = 0.5
START_SHIFT = 0.02
FACTORS = (2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000)      # hi / lo
LONGEST = 5.0                  # seconds the line alone may take in its many-pass run
MATRIX_CALL = {1000: 5e-6, 10000: 7e-6, 100000: 36e-6, 1000000: 350e-6}    # a matrix element on the GPU, roughly
ONCE_CLEAR = 0.05              # a deck's one-time difference from its line is believed from here up
FILE_LIMIT = 1e8               # bytes a deck's own output file may reach in a slope run
# The line alone stays on the CPU at 1,000 particles, where a deck with kick magnets goes to the GPU.
# Such a deck is compared with a line made to run on the GPU by these thresholds.
FORCE_GPU = {"ELEGANT_GPU_MIN_PARTICLES": "64", "ELEGANT_GPU_MIN_MATRIX_DRIFT_PARTICLES": "64"}


def tracking(record):
    return sum(v[1] for v in record["elements"].values())


def median(values):
    values = sorted(values)
    return values[len(values) // 2] if len(values) % 2 else (values[len(values) // 2 - 1] + values[len(values) // 2]) / 2


def getter(runs, failed=None):
    """get(label, config): the record of a run that worked, or None; failed runs are noted in `failed`."""
    def lookup(label, config):
        record = runs.get(label + "|" + config)
        if record and record["status"] != 0:
            if failed is not None:
                failed.append(label + " (" + config + ")")
            return None
        return record
    return lookup


def passes_for(lo, gain_per_pass, particles, cells):
    """`hi`: the element gains TARGET of wall time from lo to hi passes, with the line alone under LONGEST."""
    limit = LONGEST / (6 * cells * MATRIX_CALL[particles])
    hi = lo * FACTORS[0]
    for factor in FACTORS:
        if lo * factor > limit:
            break
        hi = lo * factor
        if (hi - lo) * gain_per_pass >= TARGET:
            break
    return hi


def slope_cells(spec, forced):
    """Cells of the line in the wall-clock runs.  Against a line forced to the GPU the deck keeps its own."""
    own = spec.get("cells", CELLS)
    return own if forced else spec.get("slope_cells", own)


def feature_calls(name, calls):
    """Calls of the element under test: the scaffold has drifts and quadrupoles of its own."""
    return calls / (3 if name == "drift" else 2 if name == "quad-matrix" else 1)


def cell(get, name, spec, particles, config):
    """How one number of the table is made, or None if its deck wasn't run.

    Returns {"value": seconds per call, "how": ...} where "how" is
      "own"       the element's own time: elegant, and a gpu-elegant deck that stayed on the CPU ("on_cpu");
      "slope"     from the wall-clock slope ("spread": the scatter of the rounds' values; "limit" if the
                  value is below it, and the spread is then given as the value; "once": what each copy of
                  the element costs once in a run on top of its calls);
      "fixed"     the same for a deck whose passes are fixed;
      "subtract"  not measured that way yet: the plain deck's element times less the standard scaffold's.
                  "needs" then says what to run.
    """
    record = get("%s/n%07d" % (name, particles), config)
    if not record or spec["type"] not in record["elements"]:
        return None
    calls, total = record["elements"][spec["type"]]
    if spec.get("pair"):    # the element put after the one copy is part of the row: gpu-elegant's slope holds both
        total += record["elements"].get(spec["pair"].split(",")[0].strip(), [0, 0.0])[1]
    result = {"value": total / calls, "how": "own"}
    if config == "elegant":
        return result
    # Did the whole deck stay on the CPU?  Then its tracking takes as long as in elegant, the line's cheap
    # matrix elements included, and the element's own time is the right one.
    # (No elegant run at 1,000,000 particles: the verdict at 100,000 is used.)
    reference = min(particles, 100000)
    same = get("%s/n%07d" % (name, reference), "elegant")
    mine = record if particles <= 100000 else get("%s/n%07d" % (name, reference), config)
    if same and mine and tracking(mine) > 0.8 * tracking(same) and particles < spec.get("gpu_from", float("inf")):
        result["on_cpu"] = True
        return result
    standard = get("scaffold/n%07d" % particles, config)
    if not standard:
        return result

    def finish(values, how, shift=0.0, **more):
        # The scatter of the rounds: the median distance from their median, which one bad round doesn't move.
        # What the runs resolve is that or `shift` (a start-up shift over the calls gained), whichever is more.
        value = median(values)
        spread = max(1.5 * median([abs(v - value) for v in values]), shift)
        result.update(how=how, value=max(0.0, value), spread=spread, **more)
        if value < spread:      # a value well below zero says the comparison itself is off by that much
            result.update(limit=True, value=max(spread, abs(value)))
        return result

    slope = get("%s+slope/n%07d" % (name, particles), config)
    if slope:
        lo, hi = slope["lo"], slope["hi"]
        made = feature_calls(name, slope["calls_hi"] - slope["calls_lo"])
        # What the element costs once in a run on top of its calls (a set-up on its first call, reading its
        # file): the deck's wall time carried back to no passes, less the scaffold's, for each copy in the line.
        copies = made / (hi - lo)
        firsts = [(r[0] - lo * (r[1] - r[0]) / (hi - lo)) - (r[2] - lo * (r[3] - r[2]) / (hi - lo)) for r in slope["rounds"]]
        once = median(firsts)
        # Believed where it is well above what a start-up shift does, the rounds agree, and the line wasn't
        # forced to the GPU (a forced line starts differently from a deck that goes there by itself).
        forced = "+gpu" in slope.get("scaffold", "")
        clear = once >= ONCE_CLEAR and all(0.5 * once <= v <= 2 * once for v in firsts) and not forced
        values = [((r[1] - r[0]) - (r[3] - r[2])) / made for r in slope["rounds"]]
        finish(values, "slope", shift=START_SHIFT / made, passes=[lo, hi], once=once / copies, once_clear=clear)
        # A thin cell: the element gained too little between the two runs for a start-up shift not to matter,
        # and more passes are allowed.  It is measured again with them.
        if not spec.get("before") and not spec.get("grows") and not spec.get("fixed"):
            cells = slope_cells(spec, forced)
            better = passes_for(lo, max(median(values), 0.0) * copies, particles, cells)
            if median(values) * made < 0.6 * TARGET and better >= 2 * hi:
                result["needs"] = dict(kind="slope", lo=lo, hi=better, cells=cells, forced=forced, before="",
                                       rounds=GROWS_ROUNDS if spec.get("disk") else SLOPE_ROUNDS)
        return result
    fixed = get("%s+fixed/n%07d" % (name, particles), config)
    if fixed:
        made = feature_calls(name, fixed["calls"])
        return finish([(r[0] - r[1]) / made for r in fixed["rounds"]], "fixed", passes=[fixed["lo"]])

    # Not measured yet.  The old difference serves as a first estimate, and says how many passes to take.
    own = spec.get("cells", CELLS)
    lo = spec.get("passes", PASSES[particles])
    plain = get("scaffold/n%07d" % reference, "elegant")
    there = standard if particles <= 100000 else get("scaffold/n%07d" % reference, config)
    forced = particles >= 1000 and not (plain and there and tracking(there) < 0.5 * tracking(plain))
    estimate = tracking(record) - (0.1 if forced else own / CELLS * tracking(standard))
    cells = slope_cells(spec, forced)
    if spec.get("before"):      # the deck also holds the element put in front of each copy: the copy's own time
        estimate = total
    result.update(how="subtract", value=max(0.0, estimate / feature_calls(name, calls)))
    if particles >= 1000:
        hi = passes_for(lo, max(estimate, 0.002) / lo, particles, cells)
        if spec.get("grows"):      # bytes written per particle per call
            hi = max(3 * lo, min(hi, int(FILE_LIMIT / (spec["grows"] * particles))))
        result["needs"] = dict(kind="fixed" if spec.get("fixed") else "slope", lo=lo, hi=hi, cells=cells,
                               forced=forced, before=name if spec.get("before") else "",
                               rounds=FIXED_ROUNDS if spec.get("fixed") else
                               GROWS_ROUNDS if spec.get("grows") or spec.get("disk") else SLOPE_ROUNDS)
    return result


def slope_runs(args, state, path, output):
    """The runs behind the gpu-elegant cells: see the note above SLOPE_ROUNDS."""
    get = getter(state["runs"])

    def show(label, config, wall, status):
        print("%-34s %-12s wall %8.2f s%s" % (label, config, wall, "" if status == 0 else "   FAILED, status %d" % status),
              flush=True)

    def measure(config):
        """One round of measurements for the cells that need one.  Returns how many groups of decks ran."""
        groups = {}
        for name, spec in F.items():
            if args.only and not any(name.startswith(o) for o in args.only):
                continue
            for particles in COUNTS[config]:
                made = cell(get, name, spec, particles, config)
                if made and made.get("needs"):
                    n = made["needs"]
                    key = (particles, n["kind"], n["lo"], n["hi"], n["cells"], n["forced"], n["before"], n["rounds"])
                    groups.setdefault(key, []).append(name)
        for (particles, kind, lo, hi, cells, forced, before, rounds), names in sorted(groups.items()):
            passes = (lo,) if kind == "fixed" else (lo, hi)
            scaffold = dict(x="MARK", y="MARK", scaffold=1, cells=cells)
            if before:      # the element that the one under test needs in front of it stays in the scaffold
                scaffold.update(before=F[before]["before"], charge=F[before].get("charge"))
            tag = "p%s%s%s%s" % ("-".join(str(p) for p in passes), "" if cells == CELLS else "+c%d" % cells,
                                 "+gpu" if forced else "", "+" + before if before else "")
            decks = {}
            for p in passes:
                decks[("scaffold", p)] = output / "slope-decks" / ("scaffold+" + tag) / ("n%07d-p%d" % (particles, p))
                write_deck(decks[("scaffold", p)], dict(scaffold, passes=p), particles)
                for name in names:
                    decks[(name, p)] = output / "slope-decks" / name / ("n%07d-p%d" % (particles, p))
                    write_deck(decks[(name, p)], dict(F[name], passes=p, cells=cells), particles)

            def run(who, p):
                label = "%s+%s/n%07d-p%d" % (who, kind, particles, p)
                record = run_deck(decks[(who, p)], config, args, output, label,
                                  FORCE_GPU if forced and who == "scaffold" else None,
                                  flush=bool(who != "scaffold" and (F[who].get("grows") or F[who].get("disk"))))
                return record

            run("scaffold", passes[-1])       # the first runs after a pause are slower: not used
            results = {name: {"status": 0, "rounds": []} for name in names}
            for _ in range(rounds):
                base = [run("scaffold", p) for p in passes]
                for name in names:
                    mine = [run(name, p) for p in passes]
                    entry = results[name]
                    entry["status"] = entry["status"] or next((r["status"] for r in mine + base if r["status"]), 0)
                    entry["rounds"].append([r["wall"] for r in mine] + [r["wall"] for r in base])
                    entry["elements"] = mine[-1]["elements"]
                    entry["wall"] = mine[-1]["wall"]
                    kinds = [r["elements"].get(F[name]["type"], [0, 0.0])[0] for r in mine]
                    if kind == "fixed":
                        entry.update(calls=kinds[0], lo=lo)
                    else:
                        entry.update(calls_lo=kinds[0], calls_hi=kinds[1], lo=lo, hi=hi)
            for name in names:
                label = "%s+%s/n%07d" % (name, kind, particles)
                results[name]["scaffold"] = tag
                state["runs"][label + "|" + config] = results[name]
                show(label, config, results[name]["wall"], results[name]["status"])
            json.dump(state, open(path, "w"), indent=1)
        return len(groups)

    for config in args.configs.split(","):
        if config != "elegant":
            for attempt in range(3):      # a measured value can show that a cheap element needs more passes
                if not measure(config):
                    break

    # The fixed cost of a run: the wall time at no passes of a line of one cell, from runs of 1 and 5 passes.
    # (One cell: the passes then cost little beside the start, in elegant too.)
    for config in args.configs.split(","):
        for particles in COUNTS[config]:
            key = "run-cost/n%07d|%s" % (particles, config)
            if args.only or (key in state["runs"] and state["runs"][key]["status"] == 0):
                continue
            lo, hi = 1, 5
            decks = {}
            for p in (lo, hi):
                decks[p] = output / "slope-decks" / "scaffold+run-cost" / ("n%07d-p%d" % (particles, p))
                write_deck(decks[p], dict(x="MARK", y="MARK", scaffold=1, passes=p, cells=1), particles)
            run_deck(decks[hi], config, args, output, "run-cost/warm-up")
            entry = {"status": 0, "rounds": [], "lo": lo, "hi": hi}
            for _ in range(FIXED_ROUNDS):
                pair = [run_deck(decks[p], config, args, output, "run-cost/n%07d-p%d" % (particles, p)) for p in (lo, hi)]
                entry["status"] = entry["status"] or next((r["status"] for r in pair if r["status"]), 0)
                entry["rounds"].append([r["wall"] for r in pair])
                entry["wall"] = pair[-1]["wall"]
            state["runs"][key] = entry
            show("run-cost/n%07d" % particles, config, entry["wall"], entry["status"])
            json.dump(state, open(path, "w"), indent=1)


def do_run(args):
    cases, output = Path(args.cases), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "runs.json"
    state = json.load(open(path)) if path.exists() else {"runs": {}}
    state["environment"] = tc.environment_record(args)
    todo = []
    for name in F:
        for config in args.configs.split(","):
            for particles in COUNTS[config]:
                deck = cases / name / ("n%07d" % particles)
                if config == "elegant" and particles == ELEGANT_LARGE:
                    deck = cases / name / ("e%07d" % particles)
                if deck.is_dir() and F[name].get("min", 1) <= particles <= F[name].get("max", particles):
                    todo.append(("%s/n%07d" % (name, particles), deck, config))
    for deck in sorted((cases / "scaffold").glob("n*")):
        for config in args.configs.split(","):
            if int(deck.name[1:]) in COUNTS[config]:
                todo.append(("scaffold/" + deck.name, deck, config))
    for deck in sorted((cases / "whole-run").glob("*/n*")) + sorted((cases / "search").glob("*")):
        for config in args.configs.split(","):
            todo.append((str(deck.relative_to(cases)), deck, config))
    for label, deck, config in todo:
        if args.only and not any(label.startswith(o + "/") or label.startswith(o) for o in args.only):
            continue
        key = label + "|" + config
        if key in state["runs"] and state["runs"][key]["status"] == 0:
            continue
        record = run_deck(deck, config, args, output, label)
        if label.startswith("scaffold/") and config != "elegant" and record["status"] == 0:
            # Every gpu-elegant row of a particle count is this run's tracking time subtracted from a
            # feature deck's.  It holds start-up paid inside the first element, which varies by a few ms
            # from run to run, so take the median of several.  It is also about 12 ms more when the GPU
            # has been idle for a few minutes than right after a long run, which no repeat removes: the
            # scaffold must be run in the same session as the feature decks, right after them, as here.
            repeats = [record] + [run_deck(deck, config, args, output, label) for _ in range(SCAFFOLD_REPEATS - 1)]
            repeats = sorted((r for r in repeats if r["status"] == 0), key=tracking)
            record = dict(repeats[len(repeats) // 2])
            record["tracking_repeats"] = [tracking(r) for r in repeats]
        state["runs"][key] = record
        print("%-34s %-12s wall %8.2f s%s" % (label, config, record["wall"],
                                              "" if record["status"] == 0 else "   FAILED, status %d" % record["status"]),
              flush=True)
        json.dump(state, open(path, "w"), indent=1)
    slope_runs(args, state, path, output)
    print("Results are in %s" % path)


def describe(spec):
    """The element's definition, and whatever else the deck has for it (one line, shown as code)."""
    parts = [spec["x"].replace("\n", "; next line: ")]
    if spec.get("stages"):
        parts.append("along the line, %d of: %s ... %s" % (len(spec["stages"]), spec["stages"][0], spec["stages"][-1]))
    if spec.get("pair"):
        parts.append("then " + spec["pair"])
    for key in ("first", "commands", "setup"):
        if spec.get(key):
            parts.append(spec[key].strip().rstrip(","))
    if spec.get("passes"):
        parts.append("%d pass%s at every particle count" % (spec["passes"], "" if spec["passes"] == 1 else "es"))
    return "; ".join(parts)


# The CPU-only elements that leave the beam on the GPU on passes where they don't act (gpuIdlePassElement())
IDLE_PASS = (("RIMULT", "`RIMULT` (which tracking ignores)"), ("BRANCH", "`BRANCH` (it only picks the next element)"),
             ("TWISS", "`TWISS` with `APPLY_ONCE=1` after its first pass"), ("POLAR", "`POLAR` off its `ON_PASS`"),
             ("SCATTER", "`SCATTER` outside `STARTONPASS`..`ENDONPASS`"))

HOST_NOTE = (". The element itself runs on the CPU in gpu-elegant: where the rest of the deck is on the GPU, its time "
             "holds the beam's trip to the host and back.")


# The element types with no GPU code, by kind, for the report's list.  The list itself is read from
# src/track_data.c (cpu_only_elements()); a type found there that is not sorted here is shown under "Not sorted",
# and a type sorted here that has GPU code by then is left out.
CPU_ONLY_KINDS = (
    ("Collective effects", "ZLONGIT ZTRANSVERSE LRWAKE CORGPIPE CORGPLATES IBSCATTER IONEFFECTS BEAMBEAM"),
    ("RF and time-dependent kickers", "RFTMEZ0 RFTM110 RMDF MRFDF SHRFDF TMCF CEPL TWPL TWLA TWMTA MODRF BUMPER MBUMPER"),
    ("Magnets, field maps and integrators", "ALPH STRAY KPOLY HKPOLY MAPSOLENOID BRAT BOFFAXE GFWIGGLER LSRMDLTR KSBEND TUBEND"),
    ("Apertures, filters, scattering and material", "MATTER SCATTER DSCATTER CBSCAT PEPPOT PFILTER SAMPLE CLEAN TAPERAPE"),
    ("Beam adjustment", "CENTER EMITTANCE TWISS REMCOR POLAR"),
    ("Diagnostics, cooling and other", "SLICE MHISTOGRAM RIMULT CPICKUP CKICKER BRANCH SCRIPT"),
)
# The day the list was last compared with the code that decides where an element is tracked
# (gpuElementEligible() and gpuPassiveElementSupported() in src/gpu/gpu_stub.c)
CPU_ONLY_CHECKED = "2026-11-15"
# Rows whose element has GPU code but not for the setting the deck uses (none since 2026-11-15, when FTABLE took
# any N_KICKS and BGGEXP SYMPLECTIC=1 moved to the GPU)
CPU_BY_SETTING = {}


def cpu_only_elements():
    """The element types without GPU_SUPPORT in entity_description[] of src/track_data.c.

    Returns (names, hidden, total): the types' names as elegant prints them, those of them that elegant's
    dictionary leaves out (NO_DICT_OUTPUT), and the number of element types in all."""
    text = (tc.REPO / "src/track_data.c").read_text(errors="replace")
    start = text.index("char *entity_name[N_TYPES]")
    names = re.findall(r'"([A-Z0-9_]+)"', text[start:text.index("};", start)])
    i = text.index("{", text.index("ELEMENT_DESCRIPTION entity_description[N_TYPES]")) + 1
    entries, depth, begin = [], 0, 0
    while True:                                   # one {...} per type, in the order of the names
        if text[i] == "{":
            depth += 1
            begin = i + 1
        elif text[i] == "}":
            if depth == 0:
                break
            depth -= 1
            entries.append(text[begin:i])
        i += 1
    if len(entries) != len(names):
        raise SystemExit("track_data.c: %d element names and %d descriptions" % (len(names), len(entries)))
    found, hidden, total = [], [], 0
    for name, entry in zip(names, entries):
        parameters, rest = entry.split(",", 1)
        if parameters.strip() == "0":             # LINE and ELSE: not elements
            continue
        total += 1
        flags = set(re.findall(r"[A-Z][A-Z_0-9]+", rest.split("sizeof")[0]))
        if "GPU_SUPPORT" not in flags:
            found.append(name)
            if "NO_DICT_OUTPUT" in flags:
                hidden.append(name)
    return found, hidden, total


def cpu_only_section(table):
    """The report's section on the element types that gpu-elegant always tracks on the CPU."""
    found, hidden, total = cpu_only_elements()
    code = lambda names: ", ".join("`%s`" % name for name in names)
    spoken = lambda names: " and ".join(filter(None, [code(names[:-1]), code(names[-1:])]))     # a, b and c
    rows = [name for name, entry in table["features"].items() if entry["type"] in found]
    lines = ["## Elements that always run on the CPU in gpu-elegant", "",
             "%d of elegant's %d element types have no GPU code. gpu-elegant tracks them with elegant's code, so a "
             "call costs what it costs in elegant. In a line that is otherwise on the GPU, each call also brings the "
             "beam to the host and sends it back, which adds from about 0.1 ms at 10,000 particles to about 10 ms at "
             "1,000,000.%s" % (
                 len(found), total,
                 " Of the rows above, %s %s of this kind." % (spoken(rows), "is" if len(rows) == 1 else "are")
                 if rows else ""), "",
             "| Kind | Elements |", "| --- | --- |"]
    left = list(found)
    for kind, names in CPU_ONLY_KINDS:
        here = [name for name in names.split() if name in left]
        left = [name for name in left if name not in here]
        if here:
            lines.append("| %s | %s |" % (kind, code(here)))
    if left:
        lines.append("| Not sorted | %s |" % code(left))
    remarks = []
    if hidden:
        remarks.append("%s %s not in elegant's dictionary." % (spoken(hidden), "is" if len(hidden) == 1 else "are"))
    if "SLICE" in found:
        remarks.append("`SLICE` does its sums on the host, where gpu-elegant finds each slice's particles in one "
                       "pass over the beam.")
    lines += ["", " ".join(remarks), ""] if remarks else [""]
    lines += ["The list is every element type without `GPU_SUPPORT` in `src/track_data.c`, read when this report was "
              "made; `&print_dictionary` shows the same flag (\"GPU capable?\", `GPUCapable` in its SDDS form). On %s "
              "it was compared with the code "
              "that decides where an element is tracked (`gpuElementEligible()` in `src/gpu/gpu_stub.c`): none of "
              "these types goes to the GPU, and every other type can." % CPU_ONLY_CHECKED, "",
              "Not in the list:", ""]
    by_setting = ["`%s` (%s)" % (name, why) for name, why in CPU_BY_SETTING.items() if name in table["features"]]
    lines.append("- Elements that have GPU code and go to the CPU with some settings%s." % (
        ". Of the rows above: " + " and ".join(by_setting) if by_setting else ""))
    lines += ["- `MARK`, `CHARGE`, `FLOOR`, `RECIRC`, `TRCOUNT` and `TSCATTER`, which don't move particles: the beam "
              "stays on the GPU through them.", ""]
    idle = [text for name, text in IDLE_PASS if name in found]
    if idle:
        lines[-1:-1] = ["- Passes on which some of the listed elements leave the beam alone (since 2026-11-14): the beam "
                        "stays on the GPU through %s." % (", ".join(idle[:-1]) + " and " + idle[-1] if len(idle) > 1
                                                          else idle[0])]
    return lines


def all_element_types():
    """elegant's element types in track_data.c's order, LINE and ELSE left out."""
    text = (tc.REPO / "src" / "track_data.c").read_text(errors="replace")
    start = text.index("char *entity_name[N_TYPES]")
    names = re.findall(r'"([A-Z0-9_]+)"', text[start:text.index("};", start)])
    return [name for name in names if name not in ("LINE", "ELSE")]


def no_row_section(table):
    """The element types that have no row of their own, and why."""
    have = {entry.get("listed", entry["type"]) for entry in table["features"].values()}
    missing = [name for name in all_element_types() if name not in have]
    lines = ["## Element types without a row of their own", "",
             "%d of elegant's %d element types have a row above (several have more than one)." %
             (len(all_element_types()) - len(missing), len(all_element_types()))]
    if missing:
        lines[-1] += " The others:"
        lines += [""] + ["- `%s`: %s." % (name, NO_ROW.get(name, "no row yet")) for name in missing]
    return lines + [""]


def cost_text(value):
    if value is None:
        return "-"
    for limit, scale, unit in ((1e-6, 1e9, "ns"), (1e-3, 1e6, "us"), (1, 1e3, "ms")):
        if value < limit:
            return "%.3g %s" % (value * scale, unit)
    return "%.3g s" % value


def do_report(args):
    output = Path(args.output)
    state = json.load(open(output / "runs.json"))
    runs, env = state["runs"], state["environment"]
    table = {"environment": env, "features": {}, "whole_run": {}, "search": {}, "overhead": {}, "failed": []}

    get = getter(runs, table["failed"])
    for name, spec in F.items():
        entry = {"type": spec["type"], "work": spec.get("work", 1), "group": spec["group"], "definition": describe(spec),
                 "seconds_per_call": {}}
        if spec.get("host"):
            entry["runs_on_host"] = True
        for key in ("listed", "remark"):
            if spec.get(key):
                entry[key] = spec[key]
        for config in tc.CONFIGS:
            for particles in COUNTS[config]:
                made = cell(get, name, spec, particles, config)
                if not made:
                    continue
                entry["seconds_per_call"].setdefault(config, {})[str(particles)] = made["value"]
                if made.get("on_cpu") and config != "elegant":
                    entry.setdefault("deck_on_cpu", {}).setdefault(config, []).append(particles)
                if made["how"] in ("slope", "fixed"):       # from wall-clock time: its passes, and the rounds' scatter
                    entry.setdefault("passes", {}).setdefault(config, {})[str(particles)] = made["passes"]
                    entry.setdefault("spread", {}).setdefault(config, {})[str(particles)] = made["spread"]
                    if "once" in made:
                        entry.setdefault("once_measured", {}).setdefault(config, {})[str(particles)] = made["once"]
                        if made["once_clear"]:
                            entry.setdefault("once_per_element", {}).setdefault(config, {})[str(particles)] = made["once"]
                    if made.get("limit"):
                        entry.setdefault("below_resolution", {}).setdefault(config, []).append(particles)
                elif made["how"] == "subtract" and particles >= 1000:
                    entry.setdefault("not_resolved", {}).setdefault(config, []).append(particles)
        if entry["seconds_per_call"]:
            table["features"][name] = entry
    # The fixed cost of a run: the wall time at no passes of a line of one cell (the intercept of two numbers
    # of passes).  In tables made before those runs existed: the same from the 50-cell line, or the wall time
    # outside tracking in the drift deck.
    for config in tc.CONFIGS:
        for particles in COUNTS[config]:
            fixed = get("run-cost/n%07d" % particles, config) or get("fixed-cost/n%07d" % particles, config)
            record = get("drift/n%07d" % particles, config)
            if fixed:
                lo, hi = fixed["lo"], fixed["hi"]
                value = median([r[0] - lo * (r[1] - r[0]) / (hi - lo) for r in fixed["rounds"]])
            elif record:
                value = record["wall"] - tracking(record)
            else:
                continue
            table["overhead"].setdefault(config, {})[str(particles)] = max(0.0, value)
    calls_per_pass = 4 * CELLS + 2 * CELLS + 1     # elements of the baseline deck (drift feature)
    for name, w in W.items():
        entry = {"title": w["title"], "per": w.get("per", "element call"), "seconds": {}}
        for config in tc.CONFIGS:
            for particles in (1000, 100000):
                with_it = get("whole-run/%s/n%07d" % (name, particles), config)
                base = get("whole-run/baseline/n%07d" % particles, config)
                if not with_it or (not base and not w.get("no_track")):
                    continue
                if w.get("no_track"):
                    value = with_it["wall"] / calls_per_pass
                elif entry["per"] == "particle":
                    value = (with_it["wall"] - base["wall"]) / particles
                else:
                    value = (with_it["wall"] - base["wall"]) / (calls_per_pass * PASSES[particles])
                entry["seconds"].setdefault(config, {})[str(particles)] = max(0.0, value)
        if entry["seconds"]:
            table["whole_run"][name] = entry
    for name, s in S.items():
        entry = {"points": s["points"], "turns": s["turns"], "elements": 20 * 8, "wall": {}}
        for config in tc.CONFIGS:
            record = get("search/" + name, config)
            if record:
                entry["wall"][config] = record["wall"]
        if entry["wall"]:
            table["search"][name] = entry
    json.dump(table, open(output / "feature_costs.json", "w"), indent=1)

    threads = env["threads"]
    lines = ["# Cost of elegant's main features", "",
             "Made %s on %s by `src/gpu/scripts/feature_timing.py`. Procedure: `GPU-Testing/TIMING-PROCEDURE.md`."
             % (env["date"], env["host"]), ""]
    if getattr(args, "note", None):
        lines += [args.note, ""]
    lines += ["Times are per call of one element (one element, one pass, the whole beam). For elegant they are read from "
             "`show_element_timing`. gpu-elegant queues cheap elements and launches its kernels without waiting for "
             "them, so its own element times put cost in the wrong place and can miss work still queued when the run "
             "ends. For it the time is a wall-clock slope: the deck and the same line without the element are each run "
             "with two numbers of passes, %d times in turn, and the wall time the deck gains over the line is divided "
             "by the calls it gains. Start-up, and anything else paid once in a run, drops out. The passes are chosen "
             "for the element to gain about half a second between the two runs; a deck with one copy of its element "
             "(the `WATCH` rows, `tfeedback`) is run in a line of one cell for this. **<** means the element was below "
             "what the runs resolve (the scatter of the rounds, or a 20 ms shift in one run's start-up spread over the "
             "calls gained), and the number is that limit. The loss decks and `ioneffects` keep their passes: the deck "
             "is compared with the line at the same passes, %d times in turn. **(CPU)** after a gpu-elegant time means "
             "the deck ran at elegant's speed at that particle count, because the whole deck stayed on the CPU or because "
             "the element under test did and takes nearly all of the deck's time; the time shown is then the element's "
             "own. An element that runs on the CPU in a deck that is otherwise on the GPU (`ioneffects`)"
             " is timed like any other, so its time holds the beam's trip to the host and back, and its "
             "speedup can be below one although the deck as a whole is faster. At 1,000 particles the line alone stays on the CPU; a deck that goes to the GPU "
             "there is compared with the line made to go too. `us` is microseconds. \"Speedup\" is elegant's time over "
             "gpu-elegant's at the same particle count. elegant is run at 1,000,000 particles too, on a copy of the "
             "deck with a tenth of the cells (its time per call doesn't depend on how many copies the line has); "
             "\"elegant, per particle\" is its time per call at the largest count it ran, over the particles, and a "
             "speedup marked **~** takes elegant's time there as ten times its time at 100,000. The last column is "
             "gpu-elegant's time with one host thread over its time with %d, and reads 1.0x where the two differ by "
             "less than what their runs resolve. **-** means no value: the elements that need a bunch are not tracked "
             "with one particle, the slowest stop below 1,000,000, and no thread ratio is given where a time is a "
             "limit." % (SLOPE_ROUNDS, FIXED_ROUNDS, threads), ""]
    groups = []
    for entry in table["features"].values():
        if entry["group"] not in groups:
            groups.append(entry["group"])
    for group in groups:
        lines += ["## " + group, "",
                  "| Feature | elegant, per particle | elegant, 1 particle | gpu-elegant: 1,000 | 10,000 | 100,000 | 1,000,000 | "
                  "Speedup at 10,000 | at 100,000 | at 1,000,000 | %d threads at 100,000 |" % threads,
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for name, entry in table["features"].items():
            if entry["group"] != group:
                continue
            e = entry["seconds_per_call"].get("elegant", {})
            g = entry["seconds_per_call"].get("gpu", {})
            t = entry["seconds_per_call"].get("gpu-threads", {})
            most = max((int(n) for n in e), default=0)      # 100,000, or fewer for the slowest elements
            per_particle = e[str(most)] / most if most > 1 else None

            below = entry.get("below_resolution", {})

            def speed(n, e=e, g=g, below=below):
                reference, about = e.get(n), ""
                if reference is None and n == "1000000" and "100000" in e:     # elegant didn't run there
                    reference, about = 10 * e["100000"], "~"
                more = "> " if int(n) in below.get("gpu", []) else ""
                return "%s%s%.1fx" % (more, about, reference / g[n]) if reference and g.get(n) else "-"

            cpu = entry.get("deck_on_cpu", {}).get("gpu", [])

            def gpu_text(n, g=g, cpu=cpu, below=below):
                return (("< " if int(n) in below.get("gpu", []) else "") + cost_text(g.get(n)) +
                        (" (CPU)" if int(n) in cpu else ""))

            # With 8 host threads against one, at 100,000 particles.  Both numbers come from a few rounds, so
            # a ratio is given only where they differ by more than the rounds' scatter; otherwise 1.0x.
            both = (g.get("100000") and t.get("100000") and
                    100000 not in below.get("gpu", []) + below.get("gpu-threads", []))
            threads_text = "-"
            if both:
                ratio = g["100000"] / t["100000"]
                scatter = sum(entry.get("spread", {}).get(c, {}).get("100000", 0.0) / v
                              for c, v in (("gpu", g["100000"]), ("gpu-threads", t["100000"])))
                # (a deck on the CPU: single runs of the element's own timer, good to a few percent)
                threads_text = "%.2fx" % ratio if abs(ratio - 1) > max(2 * scatter, 0.05 if 100000 in cpu else 0.03) else "1.0x"

            lines.append("| `%s` | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                name, cost_text(per_particle), cost_text(e.get("1")), gpu_text("1000"), gpu_text("10000"),
                gpu_text("100000"), gpu_text("1000000"), speed("10000"), speed("100000"), speed("1000000"),
                threads_text))
        lines += [""] + ["- `%s`: `%s`%s%s" % (name, entry["definition"], HOST_NOTE if entry.get("runs_on_host") else "",
                                               ". " + entry["remark"] + "." if entry.get("remark") else "")
                         for name, entry in table["features"].items() if entry["group"] == group] + [""]
    once = [(name, entry["once_per_element"]["gpu"]) for name, entry in table["features"].items()
            if entry.get("once_per_element", {}).get("gpu")]
    if once:
        lines += ["## Paid once by each element in gpu-elegant", "",
                  "What one copy of an element costs once in a run, on top of its calls (a set-up on its first call, "
                  "for one). In a line that the beam goes through once, every element pays it. It comes from the same "
                  "runs as the times above: the deck's wall time carried back to no passes, less the line's, over the "
                  "copies in the deck. A run's start-up can sit 20 ms off, so it is shown only where the deck's "
                  "difference is 50 ms or more and the rounds agree within a factor of two (0.5 ms an element in a deck "
                  "of 100 copies); **-** means smaller than that, or not measured (1,000 particles).", "",
                  "| Feature | 1,000 | 10,000 | 100,000 | 1,000,000 |", "| --- | ---: | ---: | ---: | ---: |"]
        for name, values in once:
            lines.append("| `%s` | %s |" % (name, " | ".join(cost_text(values.get(n)) for n in ("1000", "10000", "100000", "1000000"))))
        lines.append("")
    lines += no_row_section(table)
    lines += cpu_only_section(table)
    lines += ["## Output files and commands", "",
              "Extra wall time over the same deck without the feature.", "",
              "| Feature | Per | elegant, 1,000 | elegant, 100,000 | gpu-elegant, 1,000 | gpu-elegant, 100,000 |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for name, entry in table["whole_run"].items():
        e, g = entry["seconds"].get("elegant", {}), entry["seconds"].get("gpu", {})
        lines.append("| %s | %s | %s | %s | %s | %s |" % (entry["title"], entry["per"], cost_text(e.get("1000")),
                                                        cost_text(e.get("100000")), cost_text(g.get("1000")),
                                                        cost_text(g.get("100000"))))
    lines += ["", "## Searches", "",
              "A line of 20 cells with kick quadrupoles, sextupoles and bends (160 elements), 256 turns.", "",
              "| Command | Points | elegant | gpu-elegant | gpu-elegant, %d threads |" % threads, "| --- | ---: | ---: | ---: | ---: |"]
    for name, entry in table["search"].items():
        lines.append("| %s | %d | %s | %s | %s |" % (name, entry["points"], *[
            tc.seconds(entry["wall"][c]) if c in entry["wall"] else "-" for c in tc.CONFIGS]))
    lines += ["", "## Fixed cost of a run", "",
              "Wall time that does not grow with the passes (starting the program, making the beam, closing files): "
              "a line of one cell (six matrix elements) run with 1 and 5 passes, %d times in turn, and its wall time "
              "carried back to no passes." % FIXED_ROUNDS, "",
              "| Particles | elegant | gpu-elegant | gpu-elegant, %d threads |" % threads, "| ---: | ---: | ---: | ---: |"]
    for particles in COUNTS["gpu"]:
        lines.append("| %s | %s |" % (format(particles, ","), " | ".join(
            cost_text(table["overhead"].get(c, {}).get(str(particles))) for c in tc.CONFIGS)))
    if table["failed"]:
        lines += ["", "## Failed runs", ""] + ["- " + item for item in sorted(set(table["failed"]))]
    lines += ["", "## What was run", "", "| | |", "| --- | --- |", "| Host | %s |" % env["host"], "| CPU | %s |" % env["cpu"],
              "| GPU and driver | %s |" % env["gpu"],
              "| Source | commit `%s` with %d locally changed files |" % (env["commit"], env["locally_changed_files"]),
              "| elegant | `%s`, sha256 `%s` |" % (env["elegant"]["path"], env["elegant"]["sha256"][:16]),
              "| gpu-elegant | `%s`, sha256 `%s` |" % (env["gpu_elegant"]["path"], env["gpu_elegant"]["sha256"][:16]), ""]
    (output / "FEATURES.md").write_text("\n".join(lines))
    print("Wrote %s and %s" % (output / "FEATURES.md", output / "feature_costs.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    make = sub.add_parser("make")
    make.add_argument("--cases", required=True)
    run = sub.add_parser("run")
    run.add_argument("--cases", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--only", nargs="*")
    run.add_argument("--configs", default=",".join(tc.CONFIGS))
    run.add_argument("--elegant", default=str(tc.REPO / "bin/Linux-x86_64/elegant"))
    run.add_argument("--gpu-elegant", default=str(tc.REPO / "bin/Linux-x86_64-gpu/gpu-elegant"))
    run.add_argument("--threads", type=int, default=8)
    run.add_argument("--single-core", default="2")
    run.add_argument("--thread-cores", default="0-7")
    run.add_argument("--timeout", type=float, default=1800)
    run.set_defaults(repeats=1, more_repeats=0)
    report = sub.add_parser("report")
    report.add_argument("--output", required=True)
    report.add_argument("--note", help="a line printed under the heading, such as where reused runs came from")
    args = parser.parse_args()
    {"make": do_make, "run": do_run, "report": do_report}[args.action](args)


if __name__ == "__main__":
    main()
