SDDS_REPO = $(firstword $(wildcard ../../SDDS ../../../../../epics/extensions/src/SDDS))
ifeq ($(SDDS_REPO),)
  $(error SDDS source code not found. Run 'git clone https://github.com/rtsoliday/SDDS.git' next to the elegant repository)
endif

include ../Makefile.rules

LIBRARY = oagphy
LIBRARY_SRC = brightness.c pressureData.c lindbergBrightness.c

SCRIPTS = spectraCLI spectra2sdds

CFLAGS += -I$(SDDS_REPO)/include

ifeq ($(OS), Linux)
endif

ifeq ($(OS), Darwin)
endif

ifeq ($(OS), Windows)
  CFLAGS += -I$(SDDS_REPO)/lzma -wd4244
  NO_DLL = 1
endif

include ../Makefile.build

