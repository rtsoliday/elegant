SDDS_REPO = $(firstword $(wildcard ../../SDDS ../../../../../epics/extensions/src/SDDS))
ifeq ($(SDDS_REPO),)
  $(error SDDS source code not found. Run 'git clone https://github.com/rtsoliday/SDDS.git' next to the SDDS-Python-module repository)
endif

include ../Makefile.rules

SCRIPTS = analyzeMagnets \
	beamLifetimeCalc \
	bremsstrahlungLifetime \
	bremsstrahlungLifetimeDetailed \
	bucketParameters \
	computeCoherentFraction \
	computeGeneralizedGradients \
	computeQuadFringeIntegrals \
	computeSCTuneSpread \
	computeTwissBeats \
	coreEmittance \
	correctCoupling \
	doubleDist6 \
	elasticScatteringLifetime \
	elasticScatteringLifetimeDetailed \
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
	sddsupsample \
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

CFLAGS +=  -I$(SDDS_REPO)/include -I$(SDDS_REPO)/meschach -I../physics -I$(OBJ_DIR) -DUSE_GSL -DSVN_VERSION=\"$(SVN_VERSION)\" 
CCFLAGS +=  -I$(SDDS_REPO)/include -I$(SDDS_REPO)/meschach -I../physics -I$(OBJ_DIR) -DUSE_GSL -DSVN_VERSION=\"$(SVN_VERSION)\" -I.

ifeq ($(OS), Linux)
  CFLAGS += -fopenmp $(EPICS_EXTENSIONS_GSL_INC_LOCAL)
  CCFLAGS += -fopenmp -I/usr/include/eigen3 $(GSL_INC_LOCAL)
  LDFLAGS := -L$(SDDS_REPO)/lib/$(OS)-$(ARCH) -L../lib/$(OS)-$(ARCH) -fopenmp $(LDFLAGS)
  PROD_SYS_LIBS := $(LZMA_LIB) $(GSL_LIB) $(GSLCBLAS_LIB) $(Z_LIB) $(PROD_SYS_LIBS)
  PROD_LIBS = -loagphy
  PROD_LIBS_SDDS = -lmdbcommon -lmatlib -lfftpack -lSDDS1 -lnamelist -lrpnlib -lmdbmth -lmdblib -lmeschach
endif

ifeq ($(OS), Darwin)
  LDFLAGS := -L$(SDDS_REPO)/lib/$(OS)-$(ARCH) -L../lib/$(OS)-$(ARCH) $(LDFLAGS)
  PROD_SYS_LIBS := $(LZMA_LIB) $(GSL_LIB) $(GSLCBLAS_LIB) $(Z_LIB) $(PROD_SYS_LIBS)
  PROD_LIBS = -loagphy
  PROD_LIBS_SDDS = -lmdbcommon -lmatlib -lfftpack -lSDDS1 -lnamelist -lrpnlib -lmdbmth -lmdblib -lmeschach
endif

ifeq ($(OS), Windows)
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

