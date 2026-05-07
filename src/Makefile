SDDS_REPO = $(firstword $(wildcard ../../SDDS ../../../../../epics/extensions/src/SDDS))
ifeq ($(SDDS_REPO),)
  $(error SDDS source code not found. Run 'git clone https://github.com/rtsoliday/SDDS.git' next to the elegant repository)
endif

include ../Makefile.rules

PROD = elegant elegantto sddssasefel ibsEmittance abrat trimda iirFilterTest

elegant_SRC = alpha_data.c \
	alpha_matrix.c \
	alter.c \
	amplif.c \
	analyze.c \
	aperture_search.c \
	apple.c \
	bend_matrix.c \
	bratSubroutines.c \
	bunched_beam.c \
	ccbend.c \
	cfgets.c \
	chaosMap.c \
	chbook.c \
	check_duplic.c \
	chrom.c \
	circles.c \
	closed_orbit.c \
	compose_fn.c \
	computeMatricesFromTracking.c \
	compute_centroids.c \
	compute_matrices.c \
	concat_beamline.c \
	concat_mat.c \
	context.c \
	cooler.c \
	copy_particles.c \
	cormon_stats_output.c \
	correct.c \
	corrector_output.c \
	counter.c \
	coupled_twiss.c \
	csbend.c \
	divideElements.c \
	do_tracking.c \
	drand_oag.c \
	elasticScattering.c \
	elegant.c \
	engeCoef.c \
	error.c \
	exactCorrector.c \
	extend_list.c \
	Faddeeva.cc \
	final_props.c \
	find_elem.c \
	floor.c \
	frequencyMap.c \
	frfmode.c \
	fringe.c \
	ftrfmode.c \
	generate_bunch.c \
	get_beamline.c \
	global_settings.c \
	gwig.c \
	GWigSymplecticPass.c \
	ibs_tracking.c \
	ignoreElements.c \
	iirFilter.c \
	inelasticScattering.c \
	insertSCeffects.cc \
	insert_elements.c \
	ionEffects.cc \
	kahan.c \
	kick_poly.c \
	kick_sbend.c \
	kicker.c \
	kickmap.c \
	lgbend.c \
	limit_amplitudes.c \
	link_date.c \
	link_elements.c \
	load_parameters.c \
	lorentz.c \
	lorentzian.c \
	lrwake.c \
	lsc.c \
	mad_parse.c \
	magneticFieldFromGeneralizedGradient.c \
	magneticFieldOffAxisExpansion.c \
	makeInitialParticleEnsemble.c \
	malign_mat.c \
	matrix.c \
	matrixOp.c \
	matrix_output.c \
	matter.c \
	mod_rfca.c \
	modulate.c \
	moments.c \
	momentumAperture.c \
	motion.cc \
	multipole.c \
	noiseGroup.c \
	obstructionData.c \
	optimize.c \
	orbtraj_output.c \
	output_magnets.c \
	particleTunes.c \
	patterns.c \
	pepper_pot.c \
	phase_reference.c \
	pointInsideContour.c \
	poisson.cc \
	poissonBuffer.cc \
	pressureData.c \
	print_line.c \
	quad_matrix.c \
	ramp.c \
	ramp_momentum.c \
	ramped_rfca.c \
	regexCode.cc \
	replace_elements.c \
	response.c \
	rfmode.cc \
	run_rpnexpr.c \
	sample.c \
	sasefel.c \
	sasefelmx.c \
	sdds_beam.c \
	sdds_support.c \
	sdds_support_common.c \
	shrfdf.c \
	simple_rfca.c \
	sliceAnalysis.c \
	slicePoint.c \
	SReffects.c \
	spinUpdate.c \
	subprocess.c \
	taylorSeries.c \
	tfeedback.cc \
	tilt_matrices.c \
	touschekScatter.c \
	trace.c \
	track_data.c \
	track_ramp.c \
	track_rf.c \
	transformBeamWithScript.c \
	transmuteElements.c \
	trfmode.c \
	trimApertureSearchResult.c \
	trwake.c \
	tubend.c \
	tune.c \
	tuneFootprint.c \
	twiss.cc \
	ukickmap.c \
	vary.c \
	wake.c \
	warnings.c \
	zibs.c \
	zlongit.c \
	ztransverse.c

elegantto_SRC = bombElegant.c \
	cfgets.c \
	chbook.c \
	check_duplic.c \
	compose_fn.c \
	context.c \
	convert_to_cosy.c \
	convert_to_mad8.c \
	convert_to_matlab.c \
	convert_to_parmela.c \
	convert_to_patpet.c \
	convert_to_patricia.c \
	convert_to_transport.c \
	convert_to_xorbit.c \
	divideElements.c \
	drand_oag.c \
	elegantto.c \
	engeCoef.c \
	extend_list.c \
	find_elem.c \
	get_beamline.c \
	ignoreElements.c \
	insert_elements.c \
	mad_parse.c \
	matrix.c \
	output_magnets.c \
	patterns.c \
	print_line.c \
	replace_elements.c \
	sdds_strength_output.c \
	sdds_support_common.c \
	trace.c \
	track_data.c \
	transmuteElements.c \
	warnings.c

sddssasefel_SRC = sddssasefel.c sasefelmx.c bombElegant.c

ibsEmittance_SRC = ibsEmittance.c zibs.c bombElegant.c

abrat_SRC = abrat.c track_data.c warnings.c context.c bombElegant.c

trimda_SRC = trimda.c trimApertureSearchResult.c

iirFilterTest_SRC = iirFilterTest.c iirFilter.c bombElegant.c sdds_support_common.c track_data.c

ifneq ($(wildcard $(DD)../.svn $(DD)../../../../.svn),)
  SVN_VERSION=$(shell svnversion .)
  ifeq ($(SVN_VERSION),)
    SVN_VERSION=unknown
  endif
else
  SVN_VERSION=GIT_$(shell git log --oneline | wc -l | sed 's/^[[:space:]]*//')
endif


CFLAGS +=  -I$(SDDS_REPO)/include -I$(OBJ_DIR) -DIEEE_MATH -DMINIMIZE_MEMORY -DUSE_KAHAN -DUSE_GSL -DSVN_VERSION=\"$(SVN_VERSION)\" -Og
CCFLAGS +=  -I$(SDDS_REPO)/include -I$(OBJ_DIR) -DIEEE_MATH -DMINIMIZE_MEMORY -DUSE_KAHAN -DUSE_GSL -DSVN_VERSION=\"$(SVN_VERSION)\" -Og

ELEGANT_INSTALL_NAME = elegant$(EXEEXT)
CUDA_REQUESTED := $(filter 1 yes YES true TRUE on ON,$(HAVE_CUDA) $(HAVE_GPU))
ifneq ($(CUDA_REQUESTED),)
  HAVE_CUDA := 1
  HAVE_GPU := 1
  ELEGANT_INSTALL_NAME = gpu-elegant$(EXEEXT)
  CUDA_NVCC_FROM_PATH := $(shell command -v nvcc 2>/dev/null)
  CUDA_NVCC_COMMON := /usr/local/cuda-12.4/bin/nvcc /usr/local/cuda/bin/nvcc
  CUDA_NVCC_VERSIONED := $(sort $(wildcard /usr/local/cuda-*/bin/nvcc))
  CUDA_NVCC_CANDIDATES := $(strip $(CUDA_NVCC_FROM_PATH) $(wildcard $(CUDA_NVCC_COMMON)) $(CUDA_NVCC_VERSIONED))
  ifeq ($(strip $(NVCC)),)
    NVCC := $(firstword $(call CUDA_SUPPORTED_NVCCS,$(CUDA_NVCC_CANDIDATES)))
  endif
  ifneq ($(strip $(NVCC)),)
    CUDA_NVCC_VERSION := $(call CUDA_NVCC_RELEASE_VERSION,$(NVCC))
    CUDA_NVCC_MAJOR := $(call CUDA_NVCC_RELEASE_MAJOR,$(NVCC))
    CUDA_NVCC_SUPPORTED := $(call CUDA_NVCC_IS_SUPPORTED,$(NVCC))
  endif
  ifeq ($(filter clean distclean,$(MAKECMDGOALS)),)
    ifeq ($(strip $(NVCC)),)
      ifeq ($(strip $(CUDA_NVCC_CANDIDATES)),)
      $(error CUDA support requested, but nvcc was not found. Set NVCC=/path/to/nvcc, for example NVCC=/usr/local/cuda-12.4/bin/nvcc)
      else
      $(error CUDA support requested, but no nvcc with CUDA release >= $(CUDA_NVCC_MINIMUM_MAJOR) was found. Set NVCC=/path/to/nvcc, for example NVCC=/usr/local/cuda-12.4/bin/nvcc)
      endif
    endif
    ifeq ($(strip $(CUDA_NVCC_SUPPORTED)),)
      $(error CUDA support requested, but $(NVCC) reports CUDA release $(if $(CUDA_NVCC_VERSION),$(CUDA_NVCC_VERSION),unknown); nvcc release $(CUDA_NVCC_MINIMUM_MAJOR) or newer is required)
    endif
  endif
  CUDA_HOME ?= $(patsubst %/bin/nvcc,%,$(NVCC))
  CUDA_ARCH ?= sm_86
  OBJ_DIR := $(OBJ_DIR).gpu
  BIN_DIR := $(BIN_DIR)-gpu
  ifeq ($(GPU_VERIFY),1)
    OBJ_DIR := $(OBJ_DIR).verify
    BIN_DIR := $(BIN_DIR)-verify
    CFLAGS += -DGPU_VERIFY=1
    CCFLAGS += -DGPU_VERIFY=1
    NVCCFLAGS += -DGPU_VERIFY=1
  endif
  CFLAGS += -O3 -DHAVE_GPU=1 -Igpu -I$(CUDA_HOME)/include
  CCFLAGS += -O3 -DHAVE_GPU=1 -Igpu -I$(CUDA_HOME)/include
  NVCCFLAGS += -O3 -g -arch=$(CUDA_ARCH) --fmad=false -Xcompiler -fPIC -I$(SDDS_REPO)/include -I$(OBJ_DIR) -Igpu -I$(CUDA_HOME)/include -DIEEE_MATH -DMINIMIZE_MEMORY -DUSE_KAHAN -DUSE_GSL -DSVN_VERSION=\"$(SVN_VERSION)\" -DHAVE_GPU=1
  NVCCDEPFLAGS ?= $(DEPFLAGS)
  LDFLAGS += -L$(CUDA_HOME)/lib64 -Wl,-rpath,$(CUDA_HOME)/lib64
  PROD_SYS_LIBS += -lcudart
  elegant_SRC += gpu/gpu_stub.c gpu/gpu_cuda_runtime.cu
  elegantto_SRC += gpu/gpu_stub.c gpu/gpu_cuda_runtime.cu
endif

ifeq ($(MDEBUG),1)
  CFLAGS += -DMPI_DEBUG=1 -Og 
  CCFLAGS += -DMPI_DEBUG=1 -Og
endif

ifeq ($(OS), Linux)
  CFLAGS += -std=gnu11 $(MKL_CFLAG) $(MKL_INCLUDE) $(LAPACK_CFLAG) $(LAPACK_INCLUDE) $(EPICS_EXTENSIONS_GSL_INC_LOCAL)
  CCFLAGS += $(MKL_CFLAG) $(MKL_INCLUDE) $(LAPACK_CFLAG) $(LAPACK_INCLUDE) $(EPICS_EXTENSIONS_GSL_INC_LOCAL)
  LDFLAGS := -L$(SDDS_REPO)/lib/$(OS)-$(ARCH) -fopenmp $(LDFLAGS)
  PROD_SYS_LIBS := $(LZMA_LIB) $(GSL_LIB) $(GSLCBLAS_LIB) $(Z_LIB) $(FFTW3_LIB)  $(MKL_LIB) $(LAPACK_LIB) $(PROD_SYS_LIBS)
  PROD_LIBS_SDDS = -lmdbcommon -lmatlib -lfftpack -lSDDS1 -lnamelist -lrpnlib -lmdbmth -lmdblib
endif

ifeq ($(OS), Darwin)
  CFLAGS += -DCLAPACK
  CCFLAGS += -DCLAPACK
  LDFLAGS := -L$(SDDS_REPO)/lib/$(OS)-$(ARCH) $(LDFLAGS)
  PROD_SYS_LIBS := -framework Accelerate $(LZMA_LIB) $(GSL_LIB) $(GSLCBLAS_LIB) $(Z_LIB) $(FFTW3_LIB)  $(MKL_LIB) $(LAPACK_LIB) $(PROD_SYS_LIBS)
  PROD_LIBS_SDDS = -lmdbcommon -lmatlib -lfftpack -lSDDS1 -lnamelist -lrpnlib -lmdbmth -lmdblib
endif

ifeq ($(OS), Windows)
  CFLAGS += -I$(SDDS_REPO)/lzma -I../../gsl $(MKL_CFLAG) $(MKL_INCLUDE) $(MKL_INCLUDE)/fftw -wd4018 -wd4244 -wd4267 -wd4311
  CCFLAGS += -I$(SDDS_REPO)/lzma -I../../gsl $(MKL_CFLAG) $(MKL_INCLUDE) $(MKL_INCLUDE)/fftw -wd4018 -wd4244 -wd4267 -wd4311
  PROD_SYS_LIBS := gsl.lib gslcblas.lib $(MKL_LIB) $(PROD_SYS_LIBS)
  PROD_LIBS_SDDS = mdbcommon.lib matlib.lib fftpack.lib SDDS1.lib namelist.lib rpnlib.lib mdbmth.lib mdblib.lib lzma.lib z.lib
  LIB_LINK_DIRS += -LIBPATH:$(SDDS_REPO)/lib/$(OS)-$(ARCH)
endif

ifeq ($(MDEBUG),1)
  LDFLAGS += -g
endif

ifeq ($(MPROFILE),1)
  CFLAGS += -g -pg 
  CCFLAGS += -g -pg 
  LDFLAGS += -g -pg
endif

include ../Makefile.build

$(OBJ_DIR)/elegant$(EXEEXT): $(elegant_OBJS) $(PROD_DEPS) | $(BIN_DIR)
	$(LINKEXE) $(OUTPUTEXE) $(elegant_OBJS) $(LDFLAGS) $(LIB_LINK_DIRS) $(PROD_LIBS) $(PROD_LIBS_SDDS) $(PROD_SYS_LIBS)
	cp -f $@ $(BIN_DIR)/$(ELEGANT_INSTALL_NAME)
	@if [ -n "$(EPICS_BIN_DIR)" ]; then echo cp -f $@ $(EPICS_BIN_DIR)/$(ELEGANT_INSTALL_NAME); fi
	@if [ -n "$(EPICS_BIN_DIR)" ]; then cp -f $@ $(EPICS_BIN_DIR)/$(ELEGANT_INSTALL_NAME); fi

$(OBJ_DIR)/elegantto$(EXEEXT): $(elegantto_OBJS) $(PROD_DEPS) | $(BIN_DIR)
	$(LINKEXE) $(OUTPUTEXE) $(elegantto_OBJS) $(LDFLAGS) $(LIB_LINK_DIRS) $(PROD_LIBS) $(PROD_LIBS_SDDS) $(PROD_SYS_LIBS)
	cp -f $@ $(BIN_DIR)/
	@if [ -n "$(EPICS_BIN_DIR)" ]; then echo cp -f $@ $(EPICS_BIN_DIR)/; fi
	@if [ -n "$(EPICS_BIN_DIR)" ]; then cp -f $@ $(EPICS_BIN_DIR)/; fi

$(OBJ_DIR)/sddssasefel$(EXEEXT): $(sddssasefel_OBJS) $(PROD_DEPS) | $(BIN_DIR)
	$(LINKEXE) $(OUTPUTEXE) $(sddssasefel_OBJS) $(LDFLAGS) $(LIB_LINK_DIRS) $(PROD_LIBS) $(PROD_LIBS_SDDS) $(PROD_SYS_LIBS)
	cp -f $@ $(BIN_DIR)/
	@if [ -n "$(EPICS_BIN_DIR)" ]; then echo cp -f $@ $(EPICS_BIN_DIR)/; fi
	@if [ -n "$(EPICS_BIN_DIR)" ]; then cp -f $@ $(EPICS_BIN_DIR)/; fi

$(OBJ_DIR)/ibsEmittance$(EXEEXT): $(ibsEmittance_OBJS) $(PROD_DEPS) | $(BIN_DIR)
	$(LINKEXE) $(OUTPUTEXE) $(ibsEmittance_OBJS) $(LDFLAGS) $(LIB_LINK_DIRS) $(PROD_LIBS) $(PROD_LIBS_SDDS) $(PROD_SYS_LIBS)
	cp -f $@ $(BIN_DIR)/
	@if [ -n "$(EPICS_BIN_DIR)" ]; then echo cp -f $@ $(EPICS_BIN_DIR)/; fi
	@if [ -n "$(EPICS_BIN_DIR)" ]; then cp -f $@ $(EPICS_BIN_DIR)/; fi

$(OBJ_DIR)/abrat$(EXEEXT): $(abrat_OBJS) $(PROD_DEPS) | $(BIN_DIR)
	$(LINKEXE) $(OUTPUTEXE) $(abrat_OBJS) $(LDFLAGS) $(LIB_LINK_DIRS) $(PROD_LIBS) $(PROD_LIBS_SDDS) $(PROD_SYS_LIBS)
	cp -f $@ $(BIN_DIR)/
	@if [ -n "$(EPICS_BIN_DIR)" ]; then echo cp -f $@ $(EPICS_BIN_DIR)/; fi
	@if [ -n "$(EPICS_BIN_DIR)" ]; then cp -f $@ $(EPICS_BIN_DIR)/; fi

$(OBJ_DIR)/trimda$(EXEEXT): $(trimda_OBJS) $(PROD_DEPS) | $(BIN_DIR)
	$(LINKEXE) $(OUTPUTEXE) $(trimda_OBJS) $(LDFLAGS) $(LIB_LINK_DIRS) $(PROD_LIBS) $(PROD_LIBS_SDDS) $(PROD_SYS_LIBS)
	cp -f $@ $(BIN_DIR)/
	@if [ -n "$(EPICS_BIN_DIR)" ]; then echo cp -f $@ $(EPICS_BIN_DIR)/; fi
	@if [ -n "$(EPICS_BIN_DIR)" ]; then cp -f $@ $(EPICS_BIN_DIR)/; fi

$(OBJ_DIR)/iirFilterTest$(EXEEXT): $(iirFilterTest_OBJS) $(PROD_DEPS) | $(BIN_DIR)
	$(LINKEXE) $(OUTPUTEXE) $(iirFilterTest_OBJS) $(LDFLAGS) $(LIB_LINK_DIRS) $(PROD_LIBS) $(PROD_LIBS_SDDS) $(PROD_SYS_LIBS)
	cp -f $@ $(BIN_DIR)/
	@if [ -n "$(EPICS_BIN_DIR)" ]; then echo cp -f $@ $(EPICS_BIN_DIR)/; fi
	@if [ -n "$(EPICS_BIN_DIR)" ]; then cp -f $@ $(EPICS_BIN_DIR)/; fi

$(OBJ_DIR)/insertSCeffects.$(OBJEXT): $(OBJ_DIR)/insertSCeffects.h

$(OBJ_DIR)/insertSCeffects.h: insertSCeffects.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp -suppressSummaryVariables insertSCeffects.nl $(OBJ_DIR)/insertSCeffects.h

$(OBJ_DIR)/amplif.$(OBJEXT): $(OBJ_DIR)/amplif.h

$(OBJ_DIR)/amplif.h: amplif.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp -suppressSummaryVariables amplif.nl $(OBJ_DIR)/amplif.h

$(OBJ_DIR)/momentumAperture.$(OBJEXT): $(OBJ_DIR)/momentumAperture.h

$(OBJ_DIR)/momentumAperture.h: momentumAperture.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp -suppressSummaryVariables momentumAperture.nl $(OBJ_DIR)/momentumAperture.h

$(OBJ_DIR)/sasefel.$(OBJEXT): $(OBJ_DIR)/sasefel.h

$(OBJ_DIR)/sasefel.h: sasefel.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp -suppressSummaryVariables sasefel.nl $(OBJ_DIR)/sasefel.h

$(OBJ_DIR)/matrix_output.$(OBJEXT): $(OBJ_DIR)/matrix_output.h

$(OBJ_DIR)/matrix_output.h: matrix_output.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp -suppressSummaryVariables matrix_output.nl $(OBJ_DIR)/matrix_output.h

$(OBJ_DIR)/insert_elements.$(OBJEXT): $(OBJ_DIR)/insert_elements.h

$(OBJ_DIR)/insert_elements.h: insert_elements.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp -suppressSummaryVariables insert_elements.nl $(OBJ_DIR)/insert_elements.h

$(OBJ_DIR)/analyze.$(OBJEXT): $(OBJ_DIR)/analyze.h

$(OBJ_DIR)/analyze.h: analyze.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp -suppressSummaryVariables analyze.nl $(OBJ_DIR)/analyze.h

$(OBJ_DIR)/replace_elements.$(OBJEXT): $(OBJ_DIR)/replace_elements.h

$(OBJ_DIR)/replace_elements.h: replace_elements.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp -suppressSummaryVariables replace_elements.nl $(OBJ_DIR)/replace_elements.h

$(OBJ_DIR)/tune.$(OBJEXT): $(OBJ_DIR)/tune.h

$(OBJ_DIR)/tune.h: tune.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp -suppressSummaryVariables tune.nl $(OBJ_DIR)/tune.h

$(OBJ_DIR)/floor.$(OBJEXT): $(OBJ_DIR)/floor.h

$(OBJ_DIR)/floor.h: floor.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp -suppressSummaryVariables floor.nl $(OBJ_DIR)/floor.h

$(OBJ_DIR)/run_rpnexpr.$(OBJEXT): $(OBJ_DIR)/run_rpnexpr.h

$(OBJ_DIR)/run_rpnexpr.h: run_rpnexpr.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp -suppressSummaryVariables run_rpnexpr.nl $(OBJ_DIR)/run_rpnexpr.h

$(OBJ_DIR)/ionEffects.$(OBJEXT): $(OBJ_DIR)/ionEffects.h

$(OBJ_DIR)/ionEffects.h: ionEffects.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp -suppressSummaryVariables ionEffects.nl $(OBJ_DIR)/ionEffects.h

$(OBJ_DIR)/subprocess.$(OBJEXT): $(OBJ_DIR)/subprocess.h

$(OBJ_DIR)/subprocess.h: subprocess.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp -suppressSummaryVariables subprocess.nl $(OBJ_DIR)/subprocess.h

$(OBJ_DIR)/trace.$(OBJEXT): $(OBJ_DIR)/trace.h

$(OBJ_DIR)/trace.h: trace.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp -suppressSummaryVariables trace.nl $(OBJ_DIR)/trace.h

$(OBJ_DIR)/divideElements.$(OBJEXT): $(OBJ_DIR)/divideElements.h

$(OBJ_DIR)/divideElements.h: divideElements.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp -suppressSummaryVariables divideElements.nl $(OBJ_DIR)/divideElements.h

$(OBJ_DIR)/load_parameters.$(OBJEXT): $(OBJ_DIR)/load_parameters.h

$(OBJ_DIR)/load_parameters.h: load_parameters.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp -suppressSummaryVariables load_parameters.nl $(OBJ_DIR)/load_parameters.h

$(OBJ_DIR)/optim_covariable.$(OBJEXT): $(OBJ_DIR)/optim_covariable.h

$(OBJ_DIR)/optim_covariable.h: optim_covariable.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp -suppressSummaryVariables optim_covariable.nl $(OBJ_DIR)/optim_covariable.h

$(OBJ_DIR)/ignoreElements.$(OBJEXT): $(OBJ_DIR)/ignoreElements.h

$(OBJ_DIR)/ignoreElements.h: ignoreElements.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp -suppressSummaryVariables ignoreElements.nl $(OBJ_DIR)/ignoreElements.h

$(OBJ_DIR)/moments.$(OBJEXT): $(OBJ_DIR)/moments.h

$(OBJ_DIR)/moments.h: moments.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp -suppressSummaryVariables moments.nl $(OBJ_DIR)/moments.h

$(OBJ_DIR)/chaosMap.$(OBJEXT): $(OBJ_DIR)/chaosMap.h

$(OBJ_DIR)/chaosMap.h: chaosMap.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp -suppressSummaryVariables chaosMap.nl $(OBJ_DIR)/chaosMap.h

$(OBJ_DIR)/response.$(OBJEXT): $(OBJ_DIR)/response.h

$(OBJ_DIR)/response.h: response.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp -suppressSummaryVariables response.nl $(OBJ_DIR)/response.h

$(OBJ_DIR)/correct.$(OBJEXT): $(OBJ_DIR)/correct.h $(OBJ_DIR)/steer_elem.h 

$(OBJ_DIR)/correct.h: correct.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp -suppressSummaryVariables correct.nl $(OBJ_DIR)/correct.h

$(OBJ_DIR)/closed_orbit.$(OBJEXT): $(OBJ_DIR)/closed_orbit.h

$(OBJ_DIR)/closed_orbit.h: closed_orbit.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp -suppressSummaryVariables closed_orbit.nl $(OBJ_DIR)/closed_orbit.h

$(OBJ_DIR)/twiss.$(OBJEXT): $(OBJ_DIR)/twiss.h

$(OBJ_DIR)/twiss.h: twiss.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp -suppressSummaryVariables twiss.nl $(OBJ_DIR)/twiss.h

$(OBJ_DIR)/link_elements.$(OBJEXT): $(OBJ_DIR)/link_elements.h

$(OBJ_DIR)/link_elements.h: link_elements.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp -suppressSummaryVariables link_elements.nl $(OBJ_DIR)/link_elements.h

$(OBJ_DIR)/chrom.$(OBJEXT): $(OBJ_DIR)/chrom.h

$(OBJ_DIR)/chrom.h: chrom.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp -suppressSummaryVariables chrom.nl $(OBJ_DIR)/chrom.h

$(OBJ_DIR)/touschekScatter.$(OBJEXT): $(OBJ_DIR)/touschekScatter.h

$(OBJ_DIR)/touschekScatter.h: touschekScatter.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp -suppressSummaryVariables touschekScatter.nl $(OBJ_DIR)/touschekScatter.h

$(OBJ_DIR)/transmuteElements.$(OBJEXT): $(OBJ_DIR)/transmuteElements.h

$(OBJ_DIR)/transmuteElements.h: transmuteElements.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp -suppressSummaryVariables transmuteElements.nl $(OBJ_DIR)/transmuteElements.h

$(OBJ_DIR)/sdds_beam.$(OBJEXT): $(OBJ_DIR)/sdds_beam.h

$(OBJ_DIR)/sdds_beam.h: sdds_beam.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp -suppressSummaryVariables sdds_beam.nl $(OBJ_DIR)/sdds_beam.h

$(OBJ_DIR)/aperture_search.$(OBJEXT): $(OBJ_DIR)/aperture_search.h

$(OBJ_DIR)/aperture_search.h: aperture_search.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp -suppressSummaryVariables aperture_search.nl $(OBJ_DIR)/aperture_search.h

$(OBJ_DIR)/aperture_data.$(OBJEXT): $(OBJ_DIR)/aperture_data.h

$(OBJ_DIR)/aperture_data.h: aperture_data.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp -suppressSummaryVariables aperture_data.nl $(OBJ_DIR)/aperture_data.h

$(OBJ_DIR)/tuneFootprint.$(OBJEXT): $(OBJ_DIR)/tuneFootprint.h

$(OBJ_DIR)/tuneFootprint.h: tuneFootprint.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp -suppressSummaryVariables tuneFootprint.nl $(OBJ_DIR)/tuneFootprint.h

$(OBJ_DIR)/vary.$(OBJEXT): $(OBJ_DIR)/vary.h

$(OBJ_DIR)/vary.h: vary.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp -suppressSummaryVariables vary.nl $(OBJ_DIR)/vary.h

$(OBJ_DIR)/steer_elem.$(OBJEXT): $(OBJ_DIR)/steer_elem.h

$(OBJ_DIR)/steer_elem.h: steer_elem.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp -suppressSummaryVariables steer_elem.nl $(OBJ_DIR)/steer_elem.h

$(OBJ_DIR)/error.$(OBJEXT): $(OBJ_DIR)/error.h

$(OBJ_DIR)/error.h: error.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp -suppressSummaryVariables error.nl $(OBJ_DIR)/error.h

$(OBJ_DIR)/bunched_beam.$(OBJEXT): $(OBJ_DIR)/bunched_beam.h $(OBJ_DIR)/bunched_beam2.h 

$(OBJ_DIR)/bunched_beam.h: bunched_beam.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp -suppressSummaryVariables bunched_beam.nl $(OBJ_DIR)/bunched_beam.h

$(OBJ_DIR)/ramp.$(OBJEXT): $(OBJ_DIR)/ramp.h

$(OBJ_DIR)/ramp.h: ramp.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp -suppressSummaryVariables ramp.nl $(OBJ_DIR)/ramp.h

$(OBJ_DIR)/inelasticScattering.$(OBJEXT): $(OBJ_DIR)/inelasticScattering.h

$(OBJ_DIR)/inelasticScattering.h: inelasticScattering.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp -suppressSummaryVariables inelasticScattering.nl $(OBJ_DIR)/inelasticScattering.h

$(OBJ_DIR)/bunched_beam2.$(OBJEXT): $(OBJ_DIR)/bunched_beam2.h

$(OBJ_DIR)/bunched_beam2.h: bunched_beam2.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp -suppressSummaryVariables bunched_beam2.nl $(OBJ_DIR)/bunched_beam2.h

$(OBJ_DIR)/optimize.$(OBJEXT): $(OBJ_DIR)/optimize.h $(OBJ_DIR)/optim_covariable.h

$(OBJ_DIR)/optimize.h: optimize.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp -suppressSummaryVariables optimize.nl $(OBJ_DIR)/optimize.h

$(OBJ_DIR)/elasticScattering.$(OBJEXT): $(OBJ_DIR)/elasticScattering.h

$(OBJ_DIR)/elasticScattering.h: elasticScattering.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp -suppressSummaryVariables elasticScattering.nl $(OBJ_DIR)/elasticScattering.h

$(OBJ_DIR)/modulate.$(OBJEXT): $(OBJ_DIR)/modulate.h

$(OBJ_DIR)/modulate.h: modulate.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp -suppressSummaryVariables modulate.nl $(OBJ_DIR)/modulate.h

$(OBJ_DIR)/fitTraces.$(OBJEXT): $(OBJ_DIR)/fitTraces.h

$(OBJ_DIR)/fitTraces.h: fitTraces.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp -suppressSummaryVariables fitTraces.nl $(OBJ_DIR)/fitTraces.h

$(OBJ_DIR)/frequencyMap.$(OBJEXT): $(OBJ_DIR)/frequencyMap.h

$(OBJ_DIR)/frequencyMap.h: frequencyMap.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp -suppressSummaryVariables frequencyMap.nl $(OBJ_DIR)/frequencyMap.h

$(OBJ_DIR)/sliceAnalysis.$(OBJEXT): $(OBJ_DIR)/sliceAnalysis.h

$(OBJ_DIR)/sliceAnalysis.h: sliceAnalysis.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp -suppressSummaryVariables sliceAnalysis.nl $(OBJ_DIR)/sliceAnalysis.h

$(OBJ_DIR)/get_beamline.$(OBJEXT): $(OBJ_DIR)/save_lattice.h

$(OBJ_DIR)/save_lattice.$(OBJEXT): $(OBJ_DIR)/save_lattice.h

$(OBJ_DIR)/save_lattice.h: save_lattice.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp -suppressSummaryVariables save_lattice.nl $(OBJ_DIR)/save_lattice.h

$(OBJ_DIR)/alter.$(OBJEXT): $(OBJ_DIR)/alter.h

$(OBJ_DIR)/alter.h: alter.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp -suppressSummaryVariables alter.nl $(OBJ_DIR)/alter.h

$(OBJ_DIR)/elegant.$(OBJEXT): $(OBJ_DIR)/elegant.h $(OBJ_DIR)/aperture_data.h

$(OBJ_DIR)/elegant.h: elegant.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp elegant.nl $(OBJ_DIR)/elegant.h

$(OBJ_DIR)/coupled_twiss.$(OBJEXT): $(OBJ_DIR)/coupled_twiss.h

$(OBJ_DIR)/coupled_twiss.h: coupled_twiss.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp -suppressSummaryVariables coupled_twiss.nl $(OBJ_DIR)/coupled_twiss.h

$(OBJ_DIR)/obstructionData.$(OBJEXT): $(OBJ_DIR)/obstructionData.h

$(OBJ_DIR)/obstructionData.h: obstructionData.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp -suppressSummaryVariables obstructionData.nl $(OBJ_DIR)/obstructionData.h

$(OBJ_DIR)/global_settings.$(OBJEXT): $(OBJ_DIR)/global_settings.h

$(OBJ_DIR)/global_settings.h: global_settings.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp -suppressSummaryVariables global_settings.nl $(OBJ_DIR)/global_settings.h

$(OBJ_DIR)/particleTunes.$(OBJEXT): $(OBJ_DIR)/particleTunes.h

$(OBJ_DIR)/particleTunes.h: particleTunes.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp -suppressSummaryVariables particleTunes.nl $(OBJ_DIR)/particleTunes.h
