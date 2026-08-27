Summary:	Accelerator code
Name:		elegant
License:	EPICS Open license http://www.aps.anl.gov/epics/license/open.php
Group:		Applications/Databases
URL:		https://www.aps.anl.gov/Accelerator-Operations-Physics
Packager:	Robert Soliday <soliday@aps.anl.gov>
Prefix:		%{_bindir}
Autoreq:	0
Version:	2026.4.0
Release:	1
Source:		elegant-2026.4.0.tar.gz

%define debug_package %{nil}
%undefine __check_files
%description
Binary package for Elegant. Elegant is an accelerator code that
computes beta fuctions, matrices, orbits, floor coordinates,
amplifications factors, dynamic aperture, and more. It does
6-D tracking with matricies and/or canonical integrators, and supports
a variety of time-dependent elements. It also does optimization
(e.g., matching), including optimization of tracking results.
It is the principle accelerator code used at APS. 

%prep
%setup

%build
%install
mkdir -p %{buildroot}%{_bindir}
mkdir -p %{buildroot}%{_prefix}/local/oag/apps/configData/elegant/ringAnalysisTemplates
mkdir -p %{buildroot}%{_prefix}/local/oag/apps/configData/spectraCLI

install -s -m 755 abrat %{buildroot}%{_bindir}/abrat
install    -m 755 analyzeMagnets %{buildroot}%{_bindir}/analyzeMagnets
install -s -m 755 astra2elegant %{buildroot}%{_bindir}/astra2elegant
install    -m 755 beamLifetimeCalc %{buildroot}%{_bindir}/beamLifetimeCalc
install    -m 755 bremsstrahlungLifetime %{buildroot}%{_bindir}/bremsstrahlungLifetime
install    -m 755 bremsstrahlungLifetimeDetailed %{buildroot}%{_bindir}/bremsstrahlungLifetimeDetailed
install    -m 755 brightnessEnvelope %{buildroot}%{_bindir}/brightnessEnvelope
install    -m 755 bucketParameters %{buildroot}%{_bindir}/bucketParameters
install -s -m 755 centroid2floor %{buildroot}%{_bindir}/centroid2floor
install    -m 755 computeCoherentFraction %{buildroot}%{_bindir}/computeCoherentFraction
install    -m 755 computeGeneralizedGradients %{buildroot}%{_bindir}/computeGeneralizedGradients
install    -m 755 computeQuadFringeIntegrals %{buildroot}%{_bindir}/computeQuadFringeIntegrals
install -s -m 755 computeCBGGE %{buildroot}%{_bindir}/computeCBGGE
install -s -m 755 computeRBGGE %{buildroot}%{_bindir}/computeRBGGE
install    -m 755 computeSCTuneSpread %{buildroot}%{_bindir}/computeSCTuneSpread
install    -m 755 computeTwissBeats %{buildroot}%{_bindir}/computeTwissBeats
install    -m 755 coreEmittance %{buildroot}%{_bindir}/coreEmittance
install    -m 755 correctCoupling %{buildroot}%{_bindir}/correctCoupling
install -s -m 755 csrImpedance %{buildroot}%{_bindir}/csrImpedance
install -s -m 755 curvedDipoleFringeCalc %{buildroot}%{_bindir}/curvedDipoleFringeCalc
install    -m 755 doubleDist6 %{buildroot}%{_bindir}/doubleDist6
install -s -m 755 elasticScatteringAnalysis %{buildroot}%{_bindir}/elasticScatteringAnalysis
install -s -m 755 elegant %{buildroot}%{_bindir}/elegant
install -s -m 755 elegantto %{buildroot}%{_bindir}/elegantto
install    -m 755 elegant2astra %{buildroot}%{_bindir}/elegant2astra
install    -m 755 elegant2shower %{buildroot}%{_bindir}/elegant2shower
install    -m 755 elegant2track %{buildroot}%{_bindir}/elegant2track
install    -m 755 elegantRingAnalysis %{buildroot}%{_bindir}/elegantRingAnalysis
install    -m 755 fin2param %{buildroot}%{_bindir}/fin2param
install    -m 755 fracEmittance %{buildroot}%{_bindir}/fracEmittance
install    -m 755 generateBunch %{buildroot}%{_bindir}/generateBunch
install    -m 755 generateBunchTrain %{buildroot}%{_bindir}/generateBunchTrain
install -s -m 755 haissinski %{buildroot}%{_bindir}/haissinski
install -s -m 755 ibsEmittance %{buildroot}%{_bindir}/ibsEmittance
install -s -m 755 iirFilterTest %{buildroot}%{_bindir}/iirFilterTest
install    -m 755 impact2elegant %{buildroot}%{_bindir}/impact2elegant
install    -m 755 impact2sdds %{buildroot}%{_bindir}/impact2sdds
install -s -m 755 inelasticScatteringAnalysis %{buildroot}%{_bindir}/inelasticScatteringAnalysis
install    -m 755 ionTrapping %{buildroot}%{_bindir}/ionTrapping
install    -m 755 km2sdds %{buildroot}%{_bindir}/km2sdds
install    -m 755 LFBFirSetup %{buildroot}%{_bindir}/LFBFirSetup
install    -m 755 longitCalcs %{buildroot}%{_bindir}/longitCalcs
install    -m 755 longitCmd %{buildroot}%{_bindir}/longitCmd
install    -m 755 makeSkewResponseCP %{buildroot}%{_bindir}/makeSkewResponseCP
install    -m 755 makeSummedCsrZ %{buildroot}%{_bindir}/makeSummedCsrZ
install    -m 755 makeWigglerFromBends %{buildroot}%{_bindir}/makeWigglerFromBends
install -s -m 755 offMidplaneExpansion %{buildroot}%{_bindir}/offMidplaneExpansion
install -s -m 755 Pelegant %{buildroot}%{_bindir}/Pelegant
install    -m 755 parmela2elegant %{buildroot}%{_bindir}/parmela2elegant
install    -m 755 plotTwissBeamsize %{buildroot}%{_bindir}/plotTwissBeamsize
install    -m 755 pop2param %{buildroot}%{_bindir}/pop2param
install    -m 755 prepareTAPAs %{buildroot}%{_bindir}/prepareTAPAs
install -s -m 755 quantumLifetime %{buildroot}%{_bindir}/quantumLifetime
install    -m 755 radiationEnvelope %{buildroot}%{_bindir}/radiationEnvelope
install -s -m 755 recurseSetup %{buildroot}%{_bindir}/recurseSetup
install    -m 755 removeBackDrifts %{buildroot}%{_bindir}/removeBackDrifts
install    -m 755 reorganizeMmap %{buildroot}%{_bindir}/reorganizeMmap
install -s -m 755 rfgun2elegant %{buildroot}%{_bindir}/rfgun2elegant
install    -m 755 scaleRingErrors %{buildroot}%{_bindir}/scaleRingErrors
install -s -m 755 sdds4x4sigmaproc %{buildroot}%{_bindir}/sdds4x4sigmaproc
install -s -m 755 sdds5x5sigmaproc %{buildroot}%{_bindir}/sdds5x5sigmaproc
install -s -m 755 sddsanalyzebeam %{buildroot}%{_bindir}/sddsanalyzebeam
install -s -m 755 sddsbrightness %{buildroot}%{_bindir}/sddsbrightness
install -s -m 755 sddsbs %{buildroot}%{_bindir}/sddsbs
install -s -m 755 sddsbunchingfactor %{buildroot}%{_bindir}/sddsbunchingfactor
install -s -m 755 sddscompton %{buildroot}%{_bindir}/sddscompton
install -s -m 755 sddsecon %{buildroot}%{_bindir}/sddsecon
install -s -m 755 sddsemitmeas %{buildroot}%{_bindir}/sddsemitmeas
install -s -m 755 sddsemitproc %{buildroot}%{_bindir}/sddsemitproc
install -s -m 755 sddsfindresonances %{buildroot}%{_bindir}/sddsfindresonances
install -s -m 755 sddsfluxcurve %{buildroot}%{_bindir}/sddsfluxcurve
install -s -m 755 sddsfresnel %{buildroot}%{_bindir}/sddsfresnel
install -s -m 755 sddsmatchtwiss %{buildroot}%{_bindir}/sddsmatchtwiss
install -s -m 755 sddsmatchmoments %{buildroot}%{_bindir}/sddsmatchmoments
install -s -m 755 sddsrandmult %{buildroot}%{_bindir}/sddsrandmult
install -s -m 755 sddsresdiag %{buildroot}%{_bindir}/sddsresdiag
install -s -m 755 sddssasefel %{buildroot}%{_bindir}/sddssasefel
install -s -m 755 sddssyncflux %{buildroot}%{_bindir}/sddssyncflux
install -s -m 755 sddsTouschekInteg %{buildroot}%{_bindir}/sddsTouschekInteg
install -s -m 755 sddsupsample %{buildroot}%{_bindir}/sddsupsample
install -s -m 755 sddsurgent %{buildroot}%{_bindir}/sddsurgent
install -s -m 755 sddsws %{buildroot}%{_bindir}/sddsws
install -s -m 755 sddsxra %{buildroot}%{_bindir}/sddsxra
install -s -m 755 sddsxrf %{buildroot}%{_bindir}/sddsxrf
install    -m 755 shower2elegant %{buildroot}%{_bindir}/shower2elegant
install    -m 755 smoothDist6s %{buildroot}%{_bindir}/smoothDist6s
install    -m 755 spectraCLI %{buildroot}%{_bindir}/spectraCLI
install    -m 755 spectra2sdds %{buildroot}%{_bindir}/spectra2sdds
install    -m 755 spin_resonances %{buildroot}%{_bindir}/spin_resonances
install    -m 755 spiffe2elegant %{buildroot}%{_bindir}/spiffe2elegant
install -s -m 755 straightDipoleFringeCalc %{buildroot}%{_bindir}/straightDipoleFringeCalc
install    -m 755 TFBFirSetup %{buildroot}%{_bindir}/TFBFirSetup
install -s -m 755 touschekLifetime %{buildroot}%{_bindir}/touschekLifetime
install    -m 755 track2mag %{buildroot}%{_bindir}/track2mag
install    -m 755 track2sdds %{buildroot}%{_bindir}/track2sdds
install -s -m 755 trimda %{buildroot}%{_bindir}/trimda
install    -m 755 trwake2impedance %{buildroot}%{_bindir}/trwake2impedance
install    -m 755 view3dGeometry %{buildroot}%{_bindir}/view3dGeometry
install    -m 755 wake2impedance %{buildroot}%{_bindir}/wake2impedance
install    -m 755 weightedBunch %{buildroot}%{_bindir}/weightedBunch
install -s -m 755 xrltest %{buildroot}%{_bindir}/xrltest
install -s -m 755 xrltool %{buildroot}%{_bindir}/xrltool
install -m 444 BasicTemplate.ele %{buildroot}%{_prefix}/local/oag/apps/configData/elegant/ringAnalysisTemplates/BasicTemplate.ele
install -m 444 DynamicApertureErrorsTemplate.ele %{buildroot}%{_prefix}/local/oag/apps/configData/elegant/ringAnalysisTemplates/DynamicApertureErrorsTemplate.ele
install -m 444 DynamicApertureTemplate.ele %{buildroot}%{_prefix}/local/oag/apps/configData/elegant/ringAnalysisTemplates/DynamicApertureTemplate.ele
install -m 444 ErrorParametersTemplate.ele %{buildroot}%{_prefix}/local/oag/apps/configData/elegant/ringAnalysisTemplates/ErrorParametersTemplate.ele
install -m 444 ErrorsTemplate.ele %{buildroot}%{_prefix}/local/oag/apps/configData/elegant/ringAnalysisTemplates/ErrorsTemplate.ele
install -m 444 FineDynamicApertureTemplate.ele %{buildroot}%{_prefix}/local/oag/apps/configData/elegant/ringAnalysisTemplates/FineDynamicApertureTemplate.ele
install -m 444 FrequencyMapDeltaErrorsTemplate.ele %{buildroot}%{_prefix}/local/oag/apps/configData/elegant/ringAnalysisTemplates/FrequencyMapDeltaErrorsTemplate.ele
install -m 444 FrequencyMapDeltaTemplate.ele %{buildroot}%{_prefix}/local/oag/apps/configData/elegant/ringAnalysisTemplates/FrequencyMapDeltaTemplate.ele
install -m 444 FrequencyMapErrorsTemplate.ele %{buildroot}%{_prefix}/local/oag/apps/configData/elegant/ringAnalysisTemplates/FrequencyMapErrorsTemplate.ele
install -m 444 FrequencyMapTemplate.ele %{buildroot}%{_prefix}/local/oag/apps/configData/elegant/ringAnalysisTemplates/FrequencyMapTemplate.ele
install -m 444 FrequencyMap1Template.ele %{buildroot}%{_prefix}/local/oag/apps/configData/elegant/ringAnalysisTemplates/FrequencyMap1Template.ele
install -m 444 HigherOrderDispersionTemplate.ele %{buildroot}%{_prefix}/local/oag/apps/configData/elegant/ringAnalysisTemplates/HigherOrderDispersionTemplate.ele
install -m 444 KickApertureErrorsTemplate.ele %{buildroot}%{_prefix}/local/oag/apps/configData/elegant/ringAnalysisTemplates/KickApertureErrorsTemplate.ele
install -m 444 KickApertureTemplate.ele %{buildroot}%{_prefix}/local/oag/apps/configData/elegant/ringAnalysisTemplates/KickApertureTemplate.ele
install -m 444 MomentumApertureTemplate.ele %{buildroot}%{_prefix}/local/oag/apps/configData/elegant/ringAnalysisTemplates/MomentumApertureTemplate.ele
install -m 444 MomentsTemplate.ele %{buildroot}%{_prefix}/local/oag/apps/configData/elegant/ringAnalysisTemplates/MomentsTemplate.ele
install -m 444 OffMomentumDynamicApertureTemplate.ele %{buildroot}%{_prefix}/local/oag/apps/configData/elegant/ringAnalysisTemplates/OffMomentumDynamicApertureTemplate.ele
install -m 444 OffMomentumTuneTrackingTemplate.ele %{buildroot}%{_prefix}/local/oag/apps/configData/elegant/ringAnalysisTemplates/OffMomentumTuneTrackingTemplate.ele
install -m 444 PhaseSpaceTrackingTemplate.ele %{buildroot}%{_prefix}/local/oag/apps/configData/elegant/ringAnalysisTemplates/PhaseSpaceTrackingTemplate.ele
install -m 444 undulatorFluxDensityKSpectrum.spin %{buildroot}%{_prefix}/local/oag/apps/configData/spectraCLI/undulatorFluxDensityKSpectrum.spin
install -m 444 undulatorPinholeFluxKSpectrum.spin %{buildroot}%{_prefix}/local/oag/apps/configData/spectraCLI/undulatorPinholeFluxKSpectrum.spin
install -m 444 undulatorPinholeFluxSpectrum.spin %{buildroot}%{_prefix}/local/oag/apps/configData/spectraCLI/undulatorPinholeFluxSpectrum.spin
install -m 444 undulatorSpatialFluxDensity.spin %{buildroot}%{_prefix}/local/oag/apps/configData/spectraCLI/undulatorSpatialFluxDensity.spin
install -m 444 undulatorSpatialPowerDensity.spin %{buildroot}%{_prefix}/local/oag/apps/configData/spectraCLI/undulatorSpatialPowerDensity.spin
install -m 444 undulatorSpectrum.spin %{buildroot}%{_prefix}/local/oag/apps/configData/spectraCLI/undulatorSpectrum.spin
install -m 444 undulatorTotalFluxSpectrum.spin %{buildroot}%{_prefix}/local/oag/apps/configData/spectraCLI/undulatorTotalFluxSpectrum.spin
install -m 444 undulatorTuningCharacteristics.spin %{buildroot}%{_prefix}/local/oag/apps/configData/spectraCLI/undulatorTuningCharacteristics.spin
install -m 444 undulatorTuningCurve.spin %{buildroot}%{_prefix}/local/oag/apps/configData/spectraCLI/undulatorTuningCurve.spin

%files

%{_bindir}/abrat
%{_bindir}/analyzeMagnets
%{_bindir}/astra2elegant
%{_bindir}/beamLifetimeCalc
%{_bindir}/bremsstrahlungLifetime
%{_bindir}/bremsstrahlungLifetimeDetailed
%{_bindir}/brightnessEnvelope
%{_bindir}/bucketParameters
%{_bindir}/centroid2floor
%{_bindir}/computeCoherentFraction
%{_bindir}/computeGeneralizedGradients
%{_bindir}/computeQuadFringeIntegrals
%{_bindir}/computeCBGGE
%{_bindir}/computeRBGGE
%{_bindir}/computeSCTuneSpread
%{_bindir}/computeTwissBeats
%{_bindir}/coreEmittance
%{_bindir}/correctCoupling
%{_bindir}/csrImpedance
%{_bindir}/curvedDipoleFringeCalc
%{_bindir}/doubleDist6
%{_bindir}/elasticScatteringAnalysis
%{_bindir}/elegant
%{_bindir}/elegantto
%{_bindir}/elegant2astra
%{_bindir}/elegant2shower
%{_bindir}/elegant2track
%{_bindir}/elegantRingAnalysis
%{_bindir}/fin2param
%{_bindir}/fracEmittance
%{_bindir}/generateBunch
%{_bindir}/generateBunchTrain
%{_bindir}/haissinski
%{_bindir}/ibsEmittance
%{_bindir}/iirFilterTest
%{_bindir}/impact2elegant
%{_bindir}/impact2sdds
%{_bindir}/inelasticScatteringAnalysis
%{_bindir}/ionTrapping
%{_bindir}/km2sdds
%{_bindir}/LFBFirSetup
%{_bindir}/longitCalcs
%{_bindir}/longitCmd
%{_bindir}/makeSkewResponseCP
%{_bindir}/makeSummedCsrZ
%{_bindir}/makeWigglerFromBends
%{_bindir}/offMidplaneExpansion
%{_bindir}/Pelegant
%{_bindir}/parmela2elegant
%{_bindir}/plotTwissBeamsize
%{_bindir}/pop2param
%{_bindir}/prepareTAPAs
%{_bindir}/quantumLifetime
%{_bindir}/radiationEnvelope
%{_bindir}/recurseSetup
%{_bindir}/removeBackDrifts
%{_bindir}/reorganizeMmap
%{_bindir}/rfgun2elegant
%{_bindir}/scaleRingErrors
%{_bindir}/sdds4x4sigmaproc
%{_bindir}/sdds5x5sigmaproc
%{_bindir}/sddsanalyzebeam
%{_bindir}/sddsbrightness
%{_bindir}/sddsbunchingfactor
%{_bindir}/sddsbs
%{_bindir}/sddscompton
%{_bindir}/sddsecon
%{_bindir}/sddsemitmeas
%{_bindir}/sddsemitproc
%{_bindir}/sddsfindresonances
%{_bindir}/sddsfluxcurve
%{_bindir}/sddsfresnel
%{_bindir}/sddsmatchtwiss
%{_bindir}/sddsmatchmoments
%{_bindir}/sddsrandmult
%{_bindir}/sddsresdiag
%{_bindir}/sddssasefel
%{_bindir}/sddssyncflux
%{_bindir}/sddsTouschekInteg
%{_bindir}/sddsupsample
%{_bindir}/sddsurgent
%{_bindir}/sddsws
%{_bindir}/sddsxra
%{_bindir}/sddsxrf
%{_bindir}/shower2elegant
%{_bindir}/smoothDist6s
%{_bindir}/spectraCLI
%{_bindir}/spectra2sdds
%{_bindir}/spin_resonances
%{_bindir}/spiffe2elegant
%{_bindir}/straightDipoleFringeCalc
%{_bindir}/TFBFirSetup
%{_bindir}/touschekLifetime
%{_bindir}/track2mag
%{_bindir}/track2sdds
%{_bindir}/trimda
%{_bindir}/trwake2impedance
%{_bindir}/view3dGeometry
%{_bindir}/wake2impedance
%{_bindir}/weightedBunch
%{_bindir}/xrltest
%{_bindir}/xrltool
%{_prefix}/local/oag/apps/configData/elegant/ringAnalysisTemplates/BasicTemplate.ele
%{_prefix}/local/oag/apps/configData/elegant/ringAnalysisTemplates/DynamicApertureErrorsTemplate.ele
%{_prefix}/local/oag/apps/configData/elegant/ringAnalysisTemplates/DynamicApertureTemplate.ele
%{_prefix}/local/oag/apps/configData/elegant/ringAnalysisTemplates/ErrorParametersTemplate.ele
%{_prefix}/local/oag/apps/configData/elegant/ringAnalysisTemplates/ErrorsTemplate.ele
%{_prefix}/local/oag/apps/configData/elegant/ringAnalysisTemplates/FineDynamicApertureTemplate.ele
%{_prefix}/local/oag/apps/configData/elegant/ringAnalysisTemplates/FrequencyMapDeltaErrorsTemplate.ele
%{_prefix}/local/oag/apps/configData/elegant/ringAnalysisTemplates/FrequencyMapDeltaTemplate.ele
%{_prefix}/local/oag/apps/configData/elegant/ringAnalysisTemplates/FrequencyMapErrorsTemplate.ele
%{_prefix}/local/oag/apps/configData/elegant/ringAnalysisTemplates/FrequencyMapTemplate.ele
%{_prefix}/local/oag/apps/configData/elegant/ringAnalysisTemplates/FrequencyMap1Template.ele
%{_prefix}/local/oag/apps/configData/elegant/ringAnalysisTemplates/HigherOrderDispersionTemplate.ele
%{_prefix}/local/oag/apps/configData/elegant/ringAnalysisTemplates/KickApertureErrorsTemplate.ele
%{_prefix}/local/oag/apps/configData/elegant/ringAnalysisTemplates/KickApertureTemplate.ele
%{_prefix}/local/oag/apps/configData/elegant/ringAnalysisTemplates/MomentumApertureTemplate.ele
%{_prefix}/local/oag/apps/configData/elegant/ringAnalysisTemplates/MomentsTemplate.ele
%{_prefix}/local/oag/apps/configData/elegant/ringAnalysisTemplates/OffMomentumDynamicApertureTemplate.ele
%{_prefix}/local/oag/apps/configData/elegant/ringAnalysisTemplates/OffMomentumTuneTrackingTemplate.ele
%{_prefix}/local/oag/apps/configData/elegant/ringAnalysisTemplates/PhaseSpaceTrackingTemplate.ele
%{_prefix}/local/oag/apps/configData/spectraCLI/undulatorFluxDensityKSpectrum.spin
%{_prefix}/local/oag/apps/configData/spectraCLI/undulatorPinholeFluxKSpectrum.spin
%{_prefix}/local/oag/apps/configData/spectraCLI/undulatorPinholeFluxSpectrum.spin
%{_prefix}/local/oag/apps/configData/spectraCLI/undulatorSpatialFluxDensity.spin
%{_prefix}/local/oag/apps/configData/spectraCLI/undulatorSpatialPowerDensity.spin
%{_prefix}/local/oag/apps/configData/spectraCLI/undulatorSpectrum.spin
%{_prefix}/local/oag/apps/configData/spectraCLI/undulatorTotalFluxSpectrum.spin
%{_prefix}/local/oag/apps/configData/spectraCLI/undulatorTuningCharacteristics.spin
%{_prefix}/local/oag/apps/configData/spectraCLI/undulatorTuningCurve.spin
