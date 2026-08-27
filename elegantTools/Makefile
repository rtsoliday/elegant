SDDS_REPO = $(firstword $(wildcard ../../SDDS ../../../../../epics/extensions/src/SDDS))
ifeq ($(SDDS_REPO),)
  $(error SDDS source code not found. Run 'git clone https://github.com/rtsoliday/SDDS.git' next to the SDDS-Python-module repository)
endif

include ../Makefile.rules

EIGEN3_DIR = $(firstword $(wildcard /opt/local/include/eigen3 /opt/homebrew/include/eigen3 /usr/local/include/eigen3 /usr/include/eigen3))
EIGEN3_INC = $(if $(EIGEN3_DIR),-isystem $(EIGEN3_DIR),)

SCRIPTS = analyzeMagnets \
	beamLifetimeCalc \
	bremsstrahlungLifetime \
	bremsstrahlungLifetimeDetailed \
	bucketParameters \
	computeCoherentFraction \
	computeQuadFringeIntegrals \
	computeSCTuneSpread \
	computeTwissBeats \
	coreEmittance \
	correctCoupling \
	doubleDist6 \
	elegant2astra \
	elegant2shower \
	elegant2track \
	elegantRingAnalysis \
	fin2param \
	fracEmittance \
	generateBunch \
	generateBunchTrain \
	impact2elegant \
	impact2sdds \
	ionTrapping \
	km2sdds \
	LFBFirSetup \
	longitCalcs \
	longitCmd \
	makeSkewResponseCP \
	makeSummedCsrZ \
	makeWigglerFromBends \
	parmela2elegant \
	plotTwissBeamsize \
	pop2param \
	prepareTAPAs \
	radiationEnvelope \
	removeBackDrifts \
	reorganizeMmap \
	scaleRingErrors \
	shower2elegant \
	smoothDist6s \
	spiffe2elegant \
	spin_resonances \
	swapOutPolarization \
	TFBFirSetup \
	track2mag \
	track2sdds \
	trwake2impedance \
	view3dGeometry \
	wake2impedance \
	weightedBunch

PROD = astra2elegant \
	centroid2floor \
	computeCBGGE \
	computeLBGGE \
	computeRBGGE \
	csrImpedance \
	curvedDipoleFringeCalc \
	elasticScatteringAnalysis \
	haissinski \
	inelasticScatteringAnalysis \
	offMidplaneExpansion \
	quantumLifetime \
	recurseSetup \
	rfgun2elegant \
	sdds4x4sigmaproc \
	sdds5x5sigmaproc \
	sddsTouschekInteg \
	sddsanalyzebeam \
	sddsbunchingfactor \
	sddsemitmeas \
	sddsemitproc \
	sddsfindresonances \
	sddsmatchmoments \
	sddsmatchtwiss \
	sddsrandmult \
	sddsresdiag \
	straightDipoleFringeCalc \
	touschekLifetime


ifneq ($(wildcard $(DD)../.svn $(DD)../../../../.svn),)
  SVN_VERSION=$(shell svnversion .)
  ifeq ($(SVN_VERSION),)
    SVN_VERSION=unknown
  endif
else
  SVN_VERSION=GIT_$(shell git log --oneline | wc -l | sed 's/^[[:space:]]*//')
endif

CFLAGS +=  -I../physics -I$(SDDS_REPO)/include -I$(SDDS_REPO)/meschach -I$(OBJ_DIR) -DUSE_GSL -DSVN_VERSION=\"$(SVN_VERSION)\" 
CCFLAGS +=  -I../physics -I$(SDDS_REPO)/include -I$(SDDS_REPO)/meschach -I$(OBJ_DIR) -DUSE_GSL -DSVN_VERSION=\"$(SVN_VERSION)\" -I.

ifneq ($(OS), Linux)
  PROD += sddsupsample
endif

ifeq ($(OS), Linux)
	ifneq ($(wildcard $(EIGEN3_DIR)/Eigen/Dense),)
    PROD += sddsupsample
  endif
  CFLAGS += -fopenmp $(EPICS_EXTENSIONS_GSL_INC_LOCAL)
	CCFLAGS += -fopenmp $(EIGEN3_INC) $(GSL_INC_LOCAL)
  LDFLAGS := -L$(SDDS_REPO)/lib/$(OS)-$(ARCH) -L../lib/$(OS)-$(ARCH) -fopenmp $(LDFLAGS)
  PROD_SYS_LIBS := $(LZMA_LIB) $(GSL_LIB) $(GSLCBLAS_LIB) $(Z_LIB) $(PROD_SYS_LIBS)
  PROD_LIBS = -loagphy
  PROD_LIBS_SDDS = -lmdbcommon -lmatlib -lfftpack -lSDDS1 -lnamelist -lrpnlib -lmdbmth -lmdblib -lmeschach
endif

ifeq ($(OS), Darwin)
	CCFLAGS += $(EIGEN3_INC)
  LDFLAGS := -L$(SDDS_REPO)/lib/$(OS)-$(ARCH) -L../lib/$(OS)-$(ARCH) $(LDFLAGS)
  PROD_SYS_LIBS := $(LZMA_LIB) $(GSL_LIB) $(GSLCBLAS_LIB) $(Z_LIB) $(PROD_SYS_LIBS)
  PROD_LIBS = -loagphy
  PROD_LIBS_SDDS = -lmdbcommon -lmatlib -lfftpack -lSDDS1 -lnamelist -lrpnlib -lmdbmth -lmdblib -lmeschach
endif

ifeq ($(OS), Windows)
	PROD := $(filter-out sddsupsample,$(PROD))
  CFLAGS += -I$(SDDS_REPO)/lzma -I../../gsl -wd4244
  CCFLAGS += -I$(SDDS_REPO)/lzma -I../../gsl -wd4244
  PROD_LIBS = oagphy.lib
  PROD_SYS_LIBS += gsl.lib gslcblas.lib
  PROD_LIBS_SDDS = mdbcommon.lib matlib.lib fftpack.lib SDDS1.lib namelist.lib rpnlib.lib mdbmth.lib mdblib.lib lzma.lib z.lib meschach.lib
  LIB_LINK_DIRS += -LIBPATH:$(SDDS_REPO)/lib/$(OS)-$(ARCH) -LIBPATH:../lib/$(OS)-$(ARCH)
endif

include ../Makefile.build

$(OBJ_DIR)/sddsrandmult.$(OBJEXT): $(OBJ_DIR)/sddsrandmult.h

$(OBJ_DIR)/sddsrandmult.h: sddsrandmult.nl
	$(SDDS_REPO)/bin/$(OS)-$(ARCH)/nlpp -suppressSummaryVariables sddsrandmult.nl $(OBJ_DIR)/sddsrandmult.h
