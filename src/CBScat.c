/*************************************************************************\
* Copyright (c) 2026 The University of Chicago, as Operator of Argonne
* National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution.
\*************************************************************************/

/* file: CBScat.c
 * contents: track_CBScat()
 *
 * Compton back-scattering element (CBSCAT).  Models Compton back-scattering of a
 * counter-propagating (or crossing-angle) laser off the electron beam, following
 * Pan, Li, Jia, Shen and Wang, "Compton scattering process ...",
 * Phys. Rev. Accel. Beams 22, 040702 (2019), in particular the per-scatter
 * coordinate change of their Eq. (4).
 *
 * For each particle, once per pass:
 *   1. The mean number of Compton scatters mu is computed as a longitudinal
 *      luminosity-overlap integral of the particle's path through a Gaussian
 *      laser pulse.  The integral includes the laser hourglass (z-dependent spot
 *      size) and crossing-angle geometry, and uses each particle's own arrival
 *      time (velocity-corrected, referenced to the bunch centroid; the centroid
 *      is reduced across processors so it is correct under Pelegant).
 *   2. A single scatter is applied with probability mu (mu << 1 is the intended
 *      regime; mu is clamped to 1 with a warning otherwise).
 *   3. The scattered-photon polar angle theta is sampled from the chosen
 *      (Thomson or Klein-Nishina) differential cross section and the azimuth phi
 *      uniformly.  The recoil is then applied to x', y', and delta either by the
 *      default approximate lab-frame mapping of Eq. (4) (valid in the Thomson
 *      limit eps=E'/m_e c^2 << 1), or -- when EXACT_RECOIL is set -- by an exact
 *      per-particle Lorentz boost into the electron rest frame, exact Compton
 *      energy-momentum conservation there, and a boost back to the lab.  The two
 *      agree to O(eps); the exact path matters for hard (large-eps) collisions.
 *   4. Optionally the emitted (back-scattered) photon is written to an SDDS file.
 */

#include "mdb.h"
#include "track.h"

#define CBS_THOMSON 0
#define CBS_KLEIN_NISHINA 1

static SDDS_DATASET *SDDSCBSphotons = NULL;
static long CBSphotonRows = 0;

static double thomsonTotalXsec(void);
static double kleinNishinaTotalXsec(double eps);
static double sampleTheta(long crossSectionType, double eps);
static void lorentzBoost(double Ein, const double pin[3], const double betaVec[3], double gamma,
                         double *Eout, double pout[3]);
static void setUpCBSPhotonOutputFile(CBSCAT *cb, char *rootname, int64_t np, long iOccurence);
static void logCBSPhoton(double Ep_eV, double x, double xp, double y, double yp);

void track_CBScat(double **part, int64_t np, double Po, CBSCAT *cb, long iPass, long iOccurence) {
  double Eh, Nph, cosTheta0, sinTheta0, cosPhi0, sinPhi0;
  double kv[3], e1v[3], e2v[3]; /* laser propagation frame: propagation, in-plane, out-of-plane */
  double zRx, zRy, tMean;
  double *time;
  long crossSectionType, N, i;
  int64_t ip;
  TRACKING_CONTEXT tc;
  static long muWarnings = 0;

  /* pass gating (same convention as SCATTER).  These tests depend only on
     parameters and the pass number, which are identical on every processor, so
     all ranks return together -- safe with respect to the collective below.
     Do NOT add an early return on a rank-local particle count (np): under
     Pelegant every active (slave) rank must reach computeAverage_p() so the
     inter-slave reduction does not deadlock, even if this rank holds no
     particles. */
  if (iPass < cb->startOnPass)
    return;
  if (cb->endOnPass >= 0 && iPass > cb->endOnPass)
    return;
  if (cb->passInterval > 1 && (iPass - cb->startOnPass) % cb->passInterval != 0)
    return;

  /* select differential/total cross section */
  crossSectionType = CBS_KLEIN_NISHINA;
  if (cb->crossSection && strlen(cb->crossSection)) {
    if (strncmp_case_insensitive(cb->crossSection, "thomson", strlen(cb->crossSection)) == 0)
      crossSectionType = CBS_THOMSON;
    else if (strncmp_case_insensitive(cb->crossSection, "klein-nishina", strlen(cb->crossSection)) == 0)
      crossSectionType = CBS_KLEIN_NISHINA;
    else
      bombElegant("CBSCAT: CROSS_SECTION must be \"klein-nishina\" or \"thomson\"", NULL);
  }

  /* incident laser photon energy in the lab, E_h = h c / lambda (MeV) */
  if (cb->laserWavelength <= 0)
    bombElegant("CBSCAT: LASER_WAVELENGTH must be positive", NULL);
  Eh = h_mks * c_mks / cb->laserWavelength / e_mks / 1e6;

  cosTheta0 = cos(cb->theta0);

  /* All kinematics that depend on the electron energy -- the rest-frame photon
     energy E', the total cross section (through eps=E'/m_e c^2), the recoil
     coefficients of Eq. (4), and the collision-geometry factors -- are computed
     per particle inside the tracking loop from each electron's own momentum
     P=(1+delta)*Po, i.e. its individual gamma and beta, not the reference
     values.  This gives the correct per-electron Compton kinematics for a beam
     with finite energy spread. */

  /* laser photon number */
  if (cb->nPhotons > 0)
    Nph = cb->nPhotons;
  else
    Nph = cb->pulseEnergy / (h_mks * c_mks / cb->laserWavelength);

  /* Rayleigh ranges from the focus spot sizes */
  if (cb->sigmax <= 0 || cb->sigmay <= 0 || cb->sigmaz <= 0)
    bombElegant("CBSCAT: SIGMA_X, SIGMA_Y, and LASER_PULSE_LENGTH must be positive", NULL);
  zRx = 4 * PI * cb->sigmax * cb->sigmax / cb->laserWavelength;
  zRy = 4 * PI * cb->sigmay * cb->sigmay / cb->laserWavelength;

  /* Laser propagation frame.  theta0 = COLLISION_ANGLE is the angle between the
     electron direction (+z) and the laser propagation direction; phi0 =
     COLLISION_AZIMUTH orients the crossing plane about z (0 = horizontal x-z
     plane).  The laser-pulse ellipsoid has its length (sigmaz) along the
     propagation axis kv, and its transverse rms sizes SIGMA_X along the in-plane
     axis e1v and SIGMA_Y along the out-of-plane axis e2v.  At theta0=PI this
     reduces to kv=-z, e1v=-x, e2v=+y (the previous head-on convention). */
  sinTheta0 = sin(cb->theta0);
  cosPhi0 = cos(cb->azimuth);
  sinPhi0 = sin(cb->azimuth);
  kv[0] = sinTheta0 * cosPhi0;
  kv[1] = sinTheta0 * sinPhi0;
  kv[2] = cosTheta0;
  e1v[0] = cosTheta0 * cosPhi0;
  e1v[1] = cosTheta0 * sinPhi0;
  e1v[2] = -sinTheta0;
  e2v[0] = -sinPhi0;
  e2v[1] = cosPhi0;
  e2v[2] = 0;

  N = cb->nSteps < 2 ? 2 : cb->nSteps;

  /* set up optional photon output (serial only) */
  getTrackingContext(&tc);
  setUpCBSPhotonOutputFile(cb, tc.rootname, np, iOccurence);

  /* Bunch arrival-time reference.  Use each particle's true arrival time
     t = s*sqrt(P^2+1)/(c*P) (velocity-corrected, not the raw path length
     coordinate[4]), and take the bunch mean with computeAverage_p() so the
     reference is correct under Pelegant when the beam is distributed across
     processors (the mean is reduced over the slave "workers" communicator,
     matching rfmode.cc / impedance.c).  compute_average() is used in the
     serial build. */
  time = tmalloc(sizeof(*time) * (np > 0 ? np : 1));
  computeTimeCoordinatesOnly(time, Po, part, np);
#if USE_MPI
  tMean = computeAverage_p(time, np, workers);
#else
  compute_average(&tMean, time, np);
#endif

  for (ip = 0; ip < np; ip++) {
    double *coord;
    double x0, xp0, y0, yp0;
    double sum, mu, r;
    double theta, phi, cosTheta, sinTheta, cosPhi, sinPhi, coef, ddelta;
    double P, b, t, dp0, dxp, dyp;
    double gammaP, betaP, EeP, EprimeP, epsP, sigmaTotP;
    double cDeltaTau, g[3], d0[3];
    double gk, ge1, ge2, xi0k, xi0e1, xi0e2;
    double A, B, zStar, sigEff, zcut, dz, sf0, dsf;

    coord = part[ip];
    x0 = coord[0];
    xp0 = coord[1];
    y0 = coord[2];
    yp0 = coord[3];

    /* per-particle kinematics from this electron's own momentum P=(1+delta)*Po */
    P = (1 + coord[5]) * Po;
    gammaP = sqrt(P * P + 1);
    betaP = P / gammaP;
    EeP = gammaP * particleMassMV;
    EprimeP = gammaP * Eh * (1 - betaP * cosTheta0); /* rest-frame incident photon energy, Eq. (4) */
    epsP = EprimeP / me_mev;
    sigmaTotP = (crossSectionType == CBS_THOMSON) ? thomsonTotalXsec() : kleinNishinaTotalXsec(epsP);

    /* Luminosity-overlap integral in the laser propagation frame.  The electron
       moves along its own path r_e(z)=(x0+xp0*z, y0+yp0*z, z) and reaches
       beamline coordinate z at lab time (time[ip]-tMean)+z/(beta*c) relative to
       the bunch centroid.  The Gaussian laser pulse is centered at
       R_L(t)=F+c*kv*(t-delay), F=(dx,dy,S0), and travels at c along kv.  The
       separation d(z)=r_e(z)-R_L(t(z)) is linear in z; its projections onto the
       pulse axes (kv,e1v,e2v) give the pulse-frame coordinates.  LASER_DELAY is a
       time in seconds; the laser and the electron-arrival offset both convert to
       length with c. */
    cDeltaTau = c_mks * ((time[ip] - tMean) - cb->delay);
    /* d(z=0) = r_e(0) - R_L(t at element) = (x0-dx, y0-dy, -S0) - kv*cDeltaTau */
    d0[0] = (x0 - cb->dx) - kv[0] * cDeltaTau;
    d0[1] = (y0 - cb->dy) - kv[1] * cDeltaTau;
    d0[2] = -cb->focusPosition - kv[2] * cDeltaTau;
    /* g = dd/dz = dr_e/dz - (c*kv)*(dt/dz) = (xp0,yp0,1) - kv/beta */
    g[0] = xp0 - kv[0] / betaP;
    g[1] = yp0 - kv[1] / betaP;
    g[2] = 1 - kv[2] / betaP;
    gk = g[0] * kv[0] + g[1] * kv[1] + g[2] * kv[2];
    ge1 = g[0] * e1v[0] + g[1] * e1v[1] + g[2] * e1v[2];
    ge2 = g[0] * e2v[0] + g[1] * e2v[1] + g[2] * e2v[2];
    xi0k = d0[0] * kv[0] + d0[1] * kv[1] + d0[2] * kv[2];
    xi0e1 = d0[0] * e1v[0] + d0[1] * e1v[1] + d0[2] * e1v[2];
    xi0e2 = d0[0] * e2v[0] + d0[1] * e2v[1] + d0[2] * e2v[2];

    /* Center and span the quadrature on the integrand peak.  With the hourglass
       widths frozen at the focus sizes the exponent is quadratic in z,
       arg(z)=A*z^2+B*z+C; the peak is at z*=-B/(2A) and the effective rms width
       is 1/sqrt(2A).  Integrate +/-5 sigma_eff about the peak.  A<=0 means the
       electron nearly co-propagates with the pulse (no localized overlap); the
       flux factor is ~0 there, so skip. */
    A = gk * gk / (2 * cb->sigmaz * cb->sigmaz) + ge1 * ge1 / (2 * cb->sigmax * cb->sigmax) +
        ge2 * ge2 / (2 * cb->sigmay * cb->sigmay);
    if (A <= 0)
      continue;
    B = gk * xi0k / (cb->sigmaz * cb->sigmaz) + ge1 * xi0e1 / (cb->sigmax * cb->sigmax) +
        ge2 * xi0e2 / (cb->sigmay * cb->sigmay);
    zStar = -B / (2 * A);
    sigEff = 1 / sqrt(2 * A);
    zcut = 5 * sigEff;
    dz = 2 * zcut / (N - 1);

    /* Distance of the evaluation point from the laser waist (focus F) measured
       along the propagation axis; this sets the hourglass transverse sizes and
       is distinct from xik (distance from the moving pulse center, which sets the
       longitudinal pulse-length envelope).  sf = (r_e(z)-F).kv, F fixed. */
    sf0 = xi0k + cDeltaTau;              /* (r_e(0)-F).kv */
    dsf = xp0 * kv[0] + yp0 * kv[1] + kv[2]; /* d/dz (r_e-F).kv */

    sum = 0;
    for (i = 0; i < N; i++) {
      double z, xik, xie1, xie2, sf, sa, sb, n, w;
      z = zStar - zcut + i * dz;
      xik = xi0k + gk * z;   /* distance from pulse center along propagation */
      xie1 = xi0e1 + ge1 * z;
      xie2 = xi0e2 + ge2 * z;
      sf = sf0 + dsf * z;    /* distance from waist along propagation (hourglass) */
      sa = cb->sigmax * sqrt(1 + sqr(sf / zRx));
      sb = cb->sigmay * sqrt(1 + sqr(sf / zRy));
      n = Nph / (pow(PIx2, 1.5) * sa * sb * cb->sigmaz) *
          exp(-xie1 * xie1 / (2 * sa * sa) - xie2 * xie2 / (2 * sb * sb) -
              xik * xik / (2 * cb->sigmaz * cb->sigmaz));
      w = (i == 0 || i == N - 1) ? 0.5 : 1.0;
      sum += w * n;
    }
    sum *= dz; /* integral of n_ph over the electron path length z */

    /* mu = sigma_tot * FACTOR * (Moeller flux factor)/beta * integral.  The flux
       factor (1-beta*cosTheta0) is exact for any collision angle; the 1/beta
       converts the time integral to the path-length integral above. */
    mu = sigmaTotP * cb->factor * (1 - betaP * cosTheta0) / betaP * sum;
    if (mu > 1) {
      if (muWarnings++ < 10)
        printWarningForTracking("CBSCAT: expected scatters per pass exceeds 1; clamping to 1.",
                                "Reduce PULSE_ENERGY/N_PHOTONS or FACTOR for single-scatter validity.");
      mu = 1;
    }
    if (mu <= 0)
      continue;

    r = random_2(1);
    if (r >= mu)
      continue;

    /* sample scattered-photon angles: polar theta from the chosen differential
       cross section (measured from the incident-photon direction in the electron
       rest frame) and azimuth phi uniformly about it */
    theta = sampleTheta(crossSectionType, epsP);
    cosTheta = cos(theta);
    sinTheta = sin(theta);
    phi = PIx2 * random_2(1);
    cosPhi = cos(phi);
    sinPhi = sin(phi);

    if (!cb->exactRecoil) {
      /* Default: approximate lab-frame recoil, Eq. (4) of Ref. [pan2019].  Exact
         in the Thomson limit eps->0; the transverse terms are the paraxial
         projection x'=px/pz and the energy term uses the incident rest-frame
         photon energy E' (no rest-frame Compton down-shift). */
      coef = EprimeP / (EeP * betaP);
      dxp = -coef * sinTheta * cosPhi;
      dyp = -coef * sinTheta * sinPhi;
      ddelta = (gammaP * gammaP * Eh / (EeP * betaP)) * (cosTheta0 - betaP) - (gammaP * EprimeP / (EeP * betaP)) * cosTheta;

      coord[1] = xp0 + dxp;
      coord[3] = yp0 + dyp;

      /* change delta while preserving arrival time (track_SReffects idiom);
         betaP is this electron's velocity before the kick */
      t = coord[4] / betaP;
      dp0 = coord[5];
      coord[5] += ddelta;
      P = (1 + coord[5]) * Po;
      b = P / sqrt(P * P + 1);
      coord[4] = t * b;

      /* optional emitted-photon record */
      if (cb->photonFileActive) {
        double Eg; /* lab energy carried off by the photon (MeV) */
        Eg = EeP * (dp0 - coord[5]);
        if (Eg > 0)
          logCBSPhoton(Eg * 1e6, x0, (Eg > 0 ? -EeP * dxp / Eg : 0.0), y0, (Eg > 0 ? -EeP * dyp / Eg : 0.0));
      }
    } else {
      /* Exact per-particle kinematics.  Work in units of the particle rest
         energy (energies in m c^2, momenta in m c), so the electron 4-momentum
         is (gammaP, P*uhat).  Boost the incident laser photon into the electron
         rest frame, scatter with exact Compton energy-momentum conservation
         (theta from the incident-photon direction, phi about it), then boost the
         outgoing electron and photon back to the lab.  Removes both the Thomson-
         limit energy approximation and the paraxial transverse approximation. */
      double uhat[3], nrm, EphMc2, mMc2, pgi[3], bvec[3];
      double Ee_r, pe_r[3], Eg_r, pg_r[3], Epr, Es;
      double nin[3], a[3], t1[3], t2[3], nout[3], adotn, tnrm;
      double Ee_out, pe_out[3], Eg_out, pg_out[3];
      double pz, ptot, betaNew, told;

      mMc2 = particleMassMV;          /* particle rest energy (MeV) = energy unit */
      EphMc2 = Eh / mMc2;             /* incident photon energy in m c^2 units */

      /* electron direction (x'=px/pz, y'=py/pz) and incident photon momentum */
      nrm = sqrt(1 + xp0 * xp0 + yp0 * yp0);
      uhat[0] = xp0 / nrm; uhat[1] = yp0 / nrm; uhat[2] = 1 / nrm;
      pgi[0] = EphMc2 * kv[0]; pgi[1] = EphMc2 * kv[1]; pgi[2] = EphMc2 * kv[2];

      /* boost into the electron rest frame (frame velocity beta = betaP*uhat) */
      bvec[0] = betaP * uhat[0]; bvec[1] = betaP * uhat[1]; bvec[2] = betaP * uhat[2];
      {
        double pe_lab[3];
        pe_lab[0] = P * uhat[0]; pe_lab[1] = P * uhat[1]; pe_lab[2] = P * uhat[2];
        lorentzBoost(gammaP, pe_lab, bvec, gammaP, &Ee_r, pe_r); /* -> (1, ~0) */
      }
      lorentzBoost(EphMc2, pgi, bvec, gammaP, &Eg_r, pg_r);      /* Eg_r = E' */
      Epr = Eg_r;

      /* incident-photon direction in the rest frame, and a transverse basis */
      nin[0] = pg_r[0] / Epr; nin[1] = pg_r[1] / Epr; nin[2] = pg_r[2] / Epr;
      if (fabs(nin[0]) < 0.9) {
        a[0] = 1; a[1] = 0; a[2] = 0;
      } else {
        a[0] = 0; a[1] = 1; a[2] = 0;
      }
      adotn = a[0] * nin[0] + a[1] * nin[1] + a[2] * nin[2];
      t1[0] = a[0] - adotn * nin[0]; t1[1] = a[1] - adotn * nin[1]; t1[2] = a[2] - adotn * nin[2];
      tnrm = sqrt(t1[0] * t1[0] + t1[1] * t1[1] + t1[2] * t1[2]);
      t1[0] /= tnrm; t1[1] /= tnrm; t1[2] /= tnrm;
      t2[0] = nin[1] * t1[2] - nin[2] * t1[1];
      t2[1] = nin[2] * t1[0] - nin[0] * t1[2];
      t2[2] = nin[0] * t1[1] - nin[1] * t1[0];

      /* outgoing-photon direction (rotate nin by theta,phi) and exact energy */
      nout[0] = cosTheta * nin[0] + sinTheta * (cosPhi * t1[0] + sinPhi * t2[0]);
      nout[1] = cosTheta * nin[1] + sinTheta * (cosPhi * t1[1] + sinPhi * t2[1]);
      nout[2] = cosTheta * nin[2] + sinTheta * (cosPhi * t1[2] + sinPhi * t2[2]);
      Es = Epr / (1 + Epr * (1 - cosTheta)); /* Compton down-shift in rest frame */

      /* rest-frame outgoing 4-momenta by energy-momentum conservation, boosted
         back to the lab (frame velocity -betaP*uhat) */
      {
        double pgo_r[3], peo_r[3], Eeo_r, bback[3];
        pgo_r[0] = Es * nout[0]; pgo_r[1] = Es * nout[1]; pgo_r[2] = Es * nout[2];
        Eeo_r = Ee_r + Eg_r - Es;
        peo_r[0] = pe_r[0] + pg_r[0] - pgo_r[0];
        peo_r[1] = pe_r[1] + pg_r[1] - pgo_r[1];
        peo_r[2] = pe_r[2] + pg_r[2] - pgo_r[2];
        bback[0] = -bvec[0]; bback[1] = -bvec[1]; bback[2] = -bvec[2];
        lorentzBoost(Eeo_r, peo_r, bback, gammaP, &Ee_out, pe_out);
        lorentzBoost(Es, pgo_r, bback, gammaP, &Eg_out, pg_out);
      }

      /* map the outgoing electron 4-momentum back to elegant coordinates,
         preserving arrival time as in the approximate branch */
      pz = pe_out[2];
      ptot = sqrt(pe_out[0] * pe_out[0] + pe_out[1] * pe_out[1] + pe_out[2] * pe_out[2]);
      told = coord[4] / betaP;
      coord[1] = pe_out[0] / pz;
      coord[3] = pe_out[1] / pz;
      coord[5] = ptot / Po - 1; /* delta = p_tot/Po - 1 */
      betaNew = ptot / sqrt(ptot * ptot + 1);
      coord[4] = told * betaNew;

      /* optional emitted-photon record (exact lab 4-momentum) */
      if (cb->photonFileActive) {
        double Eg = Eg_out * mMc2; /* lab photon energy (MeV) */
        if (Eg > 0 && pz != 0)
          logCBSPhoton(Eg * 1e6, x0, pg_out[0] / pg_out[2], y0, pg_out[1] / pg_out[2]);
      }
    }
  }

  if (cb->photonFileActive && !SDDS_UpdatePage(cb->SDDSphotons, FLUSH_TABLE))
    SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);

  free(time);
}

/* Thomson total cross section, sigma_T = (8 pi/3) r_e^2 */
static double thomsonTotalXsec(void) {
  return 8 * PI / 3 * particleRadius * particleRadius;
}

/* Klein-Nishina total cross section as a function of eps = E'/(m_e c^2).
 * Reduces to sigma_T as eps -> 0. */
static double kleinNishinaTotalXsec(double eps) {
  double pre, term1, term2, term3;
  if (eps < 1e-6)
    return thomsonTotalXsec();
  pre = 2 * PI * particleRadius * particleRadius;
  term1 = (1 + eps) / (eps * eps) * (2 * (1 + eps) / (1 + 2 * eps) - log(1 + 2 * eps) / eps);
  term2 = log(1 + 2 * eps) / (2 * eps);
  term3 = (1 + 3 * eps) / ((1 + 2 * eps) * (1 + 2 * eps));
  return pre * (term1 + term2 - term3);
}

/* Sample the scattered-photon polar angle theta (from the electron direction)
 * by rejection from the chosen differential cross section.  cos(theta) is drawn
 * uniformly and accepted against dsigma/dOmega normalized to its theta=0 value. */
static double sampleTheta(long crossSectionType, double eps) {
  double ct, st2, f, fmax, P;
  fmax = 2.0; /* bound on f below for both Thomson and Klein-Nishina */
  while (1) {
    ct = 2 * random_2(1) - 1;
    if (crossSectionType == CBS_THOMSON) {
      f = 1 + ct * ct;
    } else {
      st2 = 1 - ct * ct;
      P = 1 / (1 + eps * (1 - ct));
      f = P * P * (P + 1 / P - st2);
    }
    if (random_2(1) * fmax <= f)
      break;
  }
  return acos(ct);
}

/* General Lorentz boost of a 4-momentum (Ein, pin) into a frame that moves with
 * velocity betaVec (Lorentz factor gamma) relative to the current frame.  Energy
 * and momentum share units (c=1).  Reduces to the identity as betaVec->0.  Used
 * by the exact CBSCAT recoil to map between the lab and the electron rest frame. */
static void lorentzBoost(double Ein, const double pin[3], const double betaVec[3], double gamma,
                         double *Eout, double pout[3]) {
  double b2, bp, coef;
  b2 = betaVec[0] * betaVec[0] + betaVec[1] * betaVec[1] + betaVec[2] * betaVec[2];
  bp = betaVec[0] * pin[0] + betaVec[1] * pin[1] + betaVec[2] * pin[2];
  *Eout = gamma * (Ein - bp);
  coef = (b2 > 0) ? (((gamma - 1) * bp / b2) - gamma * Ein) : 0.0;
  pout[0] = pin[0] + coef * betaVec[0];
  pout[1] = pin[1] + coef * betaVec[1];
  pout[2] = pin[2] + coef * betaVec[2];
}

static void setUpCBSPhotonOutputFile(CBSCAT *cb, char *rootname, int64_t np, long iOccurence) {
  TRACKING_CONTEXT tc;
#if USE_MPI
  /* photon output not supported in the parallel version */
  cb->photonFileActive = 0;
  SDDSCBSphotons = NULL;
  return;
#endif
  if (!cb->photonOutputFile || !strlen(cb->photonOutputFile)) {
    cb->photonFileActive = 0;
    SDDSCBSphotons = NULL;
    return;
  }
  getTrackingContext(&tc);
  if (!cb->photonFileActive) {
    char *filename;
    filename = compose_filename_occurence(cb->photonOutputFile, rootname, iOccurence);
    cb->SDDSphotons = tmalloc(sizeof(SDDS_DATASET));
    if (!SDDS_InitializeOutputElegant(cb->SDDSphotons, SDDS_BINARY, 1, NULL, NULL, filename) ||
        0 > SDDS_DefineParameter(cb->SDDSphotons, "Step", NULL, NULL, NULL, NULL, SDDS_LONG, NULL) ||
        0 > SDDS_DefineParameter(cb->SDDSphotons, "SVNVersion", NULL, NULL, "SVN version number", NULL, SDDS_STRING, SVN_VERSION) ||
        0 > SDDS_DefineParameter(cb->SDDSphotons, "Particles", NULL, NULL, "Number of charged particles", NULL, SDDS_LONG, NULL) ||
        0 > SDDS_DefineParameter(cb->SDDSphotons, "ElementName", NULL, NULL, NULL, NULL, SDDS_STRING, tc.elementName) ||
        0 > SDDS_DefineParameter(cb->SDDSphotons, "ElementOccurence", NULL, NULL, NULL, NULL, SDDS_LONG, NULL) ||
        !SDDS_DefineSimpleColumn(cb->SDDSphotons, "Ep", "eV", SDDS_FLOAT) ||
        !SDDS_DefineSimpleColumn(cb->SDDSphotons, "x", "m", SDDS_FLOAT) ||
        !SDDS_DefineSimpleColumn(cb->SDDSphotons, "xp", "", SDDS_FLOAT) ||
        !SDDS_DefineSimpleColumn(cb->SDDSphotons, "y", "m", SDDS_FLOAT) ||
        !SDDS_DefineSimpleColumn(cb->SDDSphotons, "yp", "", SDDS_FLOAT) ||
        !SDDS_WriteLayout(cb->SDDSphotons)) {
      SDDS_SetError("Problem setting up photon output file for CBSCAT");
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
    }
    free(filename);
    cb->photonFileActive = 1;
  }
  if (!SDDS_StartPage(cb->SDDSphotons, 10000) ||
      !SDDS_SetParameters(cb->SDDSphotons, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "Particles", (long)np, "Step", tc.step,
                          "ElementName", tc.elementName, "ElementOccurence", tc.elementOccurrence, NULL)) {
    SDDS_SetError("Problem setting up photon output file for CBSCAT");
    SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
  }
  CBSphotonRows = 0;
  SDDSCBSphotons = cb->SDDSphotons;
}

static void logCBSPhoton(double Ep_eV, double x, double xp, double y, double yp) {
  if (!SDDSCBSphotons)
    return;
  if (!SDDS_SetRowValues(SDDSCBSphotons, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, CBSphotonRows++,
                         0, (float)Ep_eV,
                         1, (float)x,
                         2, (float)xp,
                         3, (float)y,
                         4, (float)yp,
                         -1))
    SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
  if (CBSphotonRows % 10000 == 0) {
    if (!SDDS_UpdatePage(SDDSCBSphotons, FLUSH_TABLE))
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
  }
}
