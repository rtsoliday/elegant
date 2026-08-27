SDDS_REPO = $(firstword $(wildcard ../../SDDS ../../../../../epics/extensions/src/SDDS))
ifeq ($(SDDS_REPO),)
  $(error SDDS source code not found. Run 'git clone https://github.com/rtsoliday/SDDS.git' next to the elegant repository)
endif

include ../Makefile.rules

LIBRARY = oagphy
LIBRARY_SRC = brightness.c pressureData.c lindbergBrightness.c coupledBeamAnalysis.c

INC = oagphy.h

SCRIPTS = spectraCLI spectra2sdds

# -DUSE_GSL enables the GSL-based eigen-analysis in coupledBeamAnalysis.c;
# the GSL headers live under $(SDDS_REPO)/gsl (so "gsl/gsl_eigen.h" resolves).
CFLAGS += -I$(SDDS_REPO)/include -I$(SDDS_REPO)/gsl -DUSE_GSL

ifeq ($(OS), Linux)
endif

ifeq ($(OS), Darwin)
endif

ifeq ($(OS), Windows)
  CFLAGS += -I$(SDDS_REPO)/lzma -I../../gsl -wd4244
  NO_DLL = 1
endif

include ../Makefile.build

