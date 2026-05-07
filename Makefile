# Detect OS and Architecture
OS := $(shell uname -s)
ifeq ($(findstring CYGWIN, $(OS)),CYGWIN)
    OS := Windows
endif

# Check for external gsl repository needed on Windows
ifeq ($(OS), Windows)
  GSL_REPO = $(wildcard ../gsl)
  ifeq ($(GSL_REPO),)
    $(error GSL source code not found. Run 'git clone https://github.com/rtsoliday/gsl.git' next to the elegant repository)
  endif
endif

# Check for external SDDS repository
SDDS_REPO = $(firstword $(wildcard ../SDDS ../../../../epics/extensions/src/SDDS))
ifeq ($(SDDS_REPO),)
  $(error SDDS source code not found. Run 'git clone https://github.com/rtsoliday/SDDS.git' next to the elegant repository)
endif

ifeq ($(OS), Linux)
  GSL_LOCAL = $(wildcard $(SDDS_REPO)/gsl)
endif

include Makefile.rules

ifeq ($(MDEBUG),1)
  DEBUGOPTIONS = MDEBUG=1
endif

MPI_AVAILABLE =
ifneq ($(WIN_MPI),)
  MPI_AVAILABLE = 1
else ifneq ($(MPI_CC),)
  ifneq ($(MPI_CCC),)
    MPI_AVAILABLE = 1
  endif
endif

CUDA_AUTO ?= 1
CUDA_REQUESTED := $(filter 1 yes YES true TRUE on ON,$(HAVE_CUDA) $(HAVE_GPU))
CUDA_DISABLED := $(filter 0 no NO false FALSE off OFF,$(HAVE_CUDA) $(HAVE_GPU) $(CUDA_AUTO))
CUDA_NVCC_FROM_NVCC := $(if $(NVCC),$(shell command -v $(NVCC) 2>/dev/null))
CUDA_NVCC_FROM_PATH := $(shell command -v nvcc 2>/dev/null)
CUDA_NVCC_COMMON := /usr/local/cuda-12.4/bin/nvcc /usr/local/cuda/bin/nvcc
CUDA_NVCC_VERSIONED := $(sort $(wildcard /usr/local/cuda-*/bin/nvcc))
CUDA_NVCC_CANDIDATES := $(strip $(CUDA_NVCC_FROM_NVCC) $(NVCC) $(CUDA_NVCC_FROM_PATH) $(wildcard $(CUDA_NVCC_COMMON)) $(CUDA_NVCC_VERSIONED))
ifeq ($(strip $(NVCC)),)
  CUDA_AUTO_NVCC := $(firstword $(call CUDA_SUPPORTED_NVCCS,$(CUDA_NVCC_CANDIDATES)))
else
  CUDA_AUTO_NVCC := $(firstword $(call CUDA_SUPPORTED_NVCCS,$(CUDA_NVCC_FROM_NVCC) $(NVCC)))
endif
CUDA_AUTO_HOME := $(patsubst %/bin/nvcc,%,$(CUDA_AUTO_NVCC))
CUDA_AUTO_CUDART := $(firstword $(wildcard $(CUDA_AUTO_HOME)/lib64/libcudart.so) $(wildcard $(CUDA_AUTO_HOME)/lib64/libcudart.a))
CUDA_AVAILABLE =
ifneq ($(CUDA_REQUESTED),)
  CUDA_AVAILABLE = 1
else ifeq ($(CUDA_DISABLED),)
  ifneq ($(CUDA_AUTO_CUDART),)
    CUDA_AVAILABLE = 1
  endif
endif

DIRS = $(GSL_REPO)
DIRS += $(GSL_LOCAL)
DIRS += $(SDDS_REPO)/include
DIRS += $(SDDS_REPO)/meschach
DIRS += $(SDDS_REPO)/zlib
DIRS += $(SDDS_REPO)/lzma
DIRS += $(SDDS_REPO)/mdblib
DIRS += $(SDDS_REPO)/mdbmth
DIRS += $(SDDS_REPO)/rpns/code
DIRS += $(SDDS_REPO)/namelist
DIRS += $(SDDS_REPO)/SDDSlib
DIRS += $(SDDS_REPO)/fftpack
DIRS += $(SDDS_REPO)/matlib
DIRS += $(SDDS_REPO)/mdbcommon
ifneq ($(MPI_AVAILABLE),)
DIRS += $(SDDS_REPO)/pgapack
endif
DIRS += physics
DIRS += xraylib
DIRS += src
DIRS += elegantTools
DIRS += sddsbrightness

.PHONY: all $(DIRS) clean distclean

install: all

all: $(DIRS)
ifneq ($(GSL_REPO),)
  $(GSL_REPO):
	$(MAKE) -C $@ -f Makefile.MSVC all
endif
ifneq ($(GSL_LOCAL),)
  $(GSL_LOCAL):
	$(MAKE) -C $@ all
endif
$(SDDS_REPO)/include: $(GSL_REPO) $(GSL_LOCAL)
	$(MAKE) -C $@
$(SDDS_REPO)/meschach: $(SDDS_REPO)/include
	$(MAKE) -C $@
$(SDDS_REPO)/zlib: $(SDDS_REPO)/meschach
	$(MAKE) -C $@
$(SDDS_REPO)/lzma: $(SDDS_REPO)/zlib
	$(MAKE) -C $@
$(SDDS_REPO)/mdblib: $(SDDS_REPO)/lzma
	$(MAKE) -C $@
$(SDDS_REPO)/mdbmth: $(SDDS_REPO)/mdblib
	$(MAKE) -C $@
$(SDDS_REPO)/rpns/code: $(SDDS_REPO)/mdbmth $(GSL_REPO) $(GSL_LOCAL)
	$(MAKE) -C $@
$(SDDS_REPO)/namelist: $(SDDS_REPO)/rpns/code
	$(MAKE) -C $@
ifeq ($(MPI_AVAILABLE),)
$(SDDS_REPO)/SDDSlib: $(SDDS_REPO)/namelist
	$(MAKE) -C $@
else
$(SDDS_REPO)/SDDSlib: $(SDDS_REPO)/namelist
	$(MAKE) -C $@
	$(MAKE) -C $@ -f Makefile.mpi
endif
$(SDDS_REPO)/fftpack: $(SDDS_REPO)/SDDSlib
	$(MAKE) -C $@
$(SDDS_REPO)/matlib: $(SDDS_REPO)/fftpack
	$(MAKE) -C $@
$(SDDS_REPO)/mdbcommon: $(SDDS_REPO)/matlib
	$(MAKE) -C $@
ifneq ($(MPI_AVAILABLE),)
$(SDDS_REPO)/pgapack: $(SDDS_REPO)/mdbcommon
	$(MAKE) -C $@
endif
physics: $(SDDS_REPO)/mdbcommon
	$(MAKE) -C $@
xraylib: physics
	$(MAKE) -C $@
src: xraylib
	$(MAKE) $(DEBUGOPTIONS) -C $@ HAVE_CUDA= HAVE_GPU=
ifneq ($(CUDA_AVAILABLE),)
	$(MAKE) $(DEBUGOPTIONS) -C $@ HAVE_CUDA=1
endif
ifneq ($(MPI_AVAILABLE),)
	$(MAKE) $(DEBUGOPTIONS) -C $@ -f Makefile.mpi HAVE_CUDA= HAVE_GPU=
ifneq ($(CUDA_AVAILABLE),)
	$(MAKE) $(DEBUGOPTIONS) -C $@ -f Makefile.mpi HAVE_CUDA=1
endif
endif
elegantTools: src
	$(MAKE) -C $@
sddsbrightness: elegantTools
	$(MAKE) -C $@

clean:
	$(MAKE) -C physics clean
	$(MAKE) -C xraylib clean
	$(MAKE) -C src clean
	$(MAKE) -C src HAVE_CUDA=1 clean
	$(MAKE) -C src HAVE_CUDA=1 GPU_VERIFY=1 clean
	$(MAKE) -C src -f Makefile.mpi clean
	$(MAKE) -C src -f Makefile.mpi HAVE_CUDA=1 clean
	$(MAKE) -C src -f Makefile.mpi HAVE_CUDA=1 GPU_VERIFY=1 clean
	$(MAKE) -C elegantTools clean
	$(MAKE) -C sddsbrightness clean

distclean: clean
	rm -rf bin/$(OS)-$(ARCH)
	rm -rf bin/$(OS)-$(ARCH)-gpu
	rm -rf bin/$(OS)-$(ARCH)-gpu-verify
	rm -rf lib/$(OS)-$(ARCH)
	rm -rf lib/$(OS)-$(ARCH)-gpu
	rm -rf lib/$(OS)-$(ARCH)-gpu-verify
