.DEFAULT_GOAL := all
SDDS_REPO = $(firstword $(wildcard ../../SDDS ../../../../../epics/extensions/src/SDDS))
ifeq ($(SDDS_REPO),)
  $(error SDDS source code not found. Run 'git clone https://github.com/rtsoliday/SDDS.git' next to the elegant repository)
endif

ENABLE_OPENMP ?= 1
BUNDLE_OPENMP ?= 1
include ../Makefile.rules
include $(RULES_DIR)Makefile.elegant
LIBRARY = xrl
LIBRARY_SRC = atomiclevelwidth.c \
        atomicweight.c \
        auger_trans.c \
        comptonprofiles.c \
        coskron.c \
        cross_sections.c \
        crystal_diffraction.c \
        cs_barns.c \
        cs_cp.c \
        cs_line.c \
        densities.c \
        edges.c \
        fi.c \
        fii.c \
        fluor_lines.c \
        fluor_yield.c \
        jump.c \
        kissel_pe.c \
        polarized.c \
        radrate.c \
        refractive_indices.c \
        scattering.c \
        splint.c \
        xrayfiles_inline.c \
        xrayglob_inline.c \
        xraylib-aux.c \
        xraylib-nist-compounds.c \
        xraylib-parser.c \
        xraylib-radionuclides.c \
        xrayvars.c \
        xrf_cross_sections_aux.c

CFLAGS += -DHAVE_STRDUP=1 -I.

ifeq ($(OS), Linux)
endif

ifeq ($(OS), Darwin)
endif

ifeq ($(OS), Windows)
  CFLAGS += -wd4047 -wd4244
  NO_DLL = 1
endif

include ../Makefile.build
$(BUILD_GENERATED): | $(OBJ_DIR)