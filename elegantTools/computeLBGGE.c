/* computeLBGGE.c
 *
 * Local-least-squares ("Bmad" method) generalized-gradient expansion.
 *
 * This program implements the fitting method attributed to Bmad in
 *   D. Sagan et al., "Field Map Generalized Gradient Representations,"
 *   IPAC2023, WEPL015 (see elegantTools/generalized-gradients-sagan.pdf).
 *
 * Unlike computeRBGGE (the "elegant" method, a global FFT-in-z / Bessel
 * surface integral over four rectangular boundary planes), this program
 * performs an independent weighted least-squares fit at each z-plane using
 * field samples taken from the surrounding volume within a short z-window.
 * The field data are supplied as a full 3-D field map -- the same file
 * format that computeRBGGE accepts through its -autotune/-evaluate option
 * (SDDS columns x,y,z in m and Bx,By,Bz in T).
 *
 * The output files are byte-compatible with computeRBGGE's output: one page
 * per multipole m, parameters m,xCenter,yCenter,xMax,yMax, column z, and
 * CnmS{n}/dCnmS{n}/dz (normal) or CnmC{n}/dCnmC{n}/dz (skew) columns.  They
 * can therefore be used directly by the elegant BGGEXP element and re-read by
 * computeRBGGE -evaluate.
 *
 * The generalized-gradient field model (paper Eq. 4) and its z-Taylor
 * extrapolation (paper Eq. 8) are reproduced exactly as coded in
 * computeRBGGE's evaluateGGEForFieldMap(), so a fit produced here reconstructs
 * fields through the identical formula used to consume it.
 *
 * -autotune reproduces the feature of the same name in computeRBGGE: it
 * searches for the smallest number of multipoles and derivatives whose fit
 * reproduces the 3-D field map to within a requested significance.  Because
 * this is a least-squares method (the fitted coefficients depend on the set of
 * basis terms), autotune RE-FITS at each candidate (multipoles, derivatives)
 * rather than truncating a single maximum-order fit.  The 3-D map used for the
 * goodness-of-fit test is simply the -fieldMap already supplied, so -autotune
 * takes no filename argument here (unlike computeRBGGE, whose primary input is
 * a set of boundary planes).
 *
 * Coded largely by Claude Code under direction of M. Borland, using some code by
 * Michael Borland, R. Soliday, R. Lindberg (elegant); Bmad method by D. Sagan et al.
 */

#include "mdb.h"
#include "SDDS.h"
#include "scan.h"
#include <math.h>
#include <float.h>

#if defined(linux) || (defined(_WIN32) && !defined(_MINGW))
#  include <omp.h>
#else
#  define NOTHREADS 1
#endif

/* ------------------------------------------------------------------ */
/* 3-D field map (columns x,y,z in m, Bx,By,Bz in T)                   */
typedef struct {
  double *x, *y, *z;
  double *Bx, *By, *Bz;
  long n;
} FIELD_MAP;

/* One (multipole, parity) block of unknowns for the local fit.
 * The unknown vector for a block is C^{[j]}_{m,alpha}(z_i) for the
 * derivative orders j = jmin..jmax; these map onto the output columns as
 * CnmX{j} for even j (the gradient and its even derivatives) and
 * dCnmX{j-1}/dz for odd j (the odd derivatives).                        */
typedef struct {
  short skew;      /* 0 => normal (sin m phi), 1 => skew (cos m phi) */
  short mZero;     /* skew solenoid special case (m==0)              */
  long m;          /* angular harmonic                               */
  long ir;         /* index of this block within its normal/skew set */
  int ndLimit;     /* number of radial terms (ig = 0..ndLimit-1)     */
  int jmin, jmax;  /* range of derivative orders that are unknowns   */
  int nOrders;     /* number of unknowns in this block (jmax-jmin+1) */
  int colOffset;   /* first column of this block in the global vector */
  double **coef;   /* solution store: coef[order][iz], order 0..2*derivatives-1 */
} BLOCK;

/* Everything about the field samples that does not depend on the number of
 * multipoles/derivatives, so it can be prepared once and reused for every
 * candidate model during -autotune.                                        */
typedef struct {
  double z;
  long ip;
} ZINDEX;

typedef struct {
  ZINDEX *samples;         /* in-aperture samples, sorted by z                */
  long nInc;               /* number of in-aperture samples                   */
  double *sr, *sphi, *sw;  /* radius, angle (about center), core weight        */
  double *sBx, *sBy, *sBz; /* measured field at each sample                    */
  long *sIz;               /* z-plane index of each sample                     */
  double *zPlane;          /* distinct, uniformly spaced z planes             */
  long Nz;                 /* number of z planes                              */
  double dz, deltaZ;       /* plane spacing and half-window                   */
} FIT_DATA;

/* Residual measures for one candidate fit (mirrors computeRBGGE). */
typedef struct {
  double max, rms, mad;
  double fracMax, fracRms, fracMad;
} ALL_RESIDUALS;

void readFieldMap(char *fieldMapFile, FIELD_MAP *fmData);
static int gaussSolve(double *M, double *b, int n, double *x);
static BLOCK *buildBlocks(long multipoles, long derivatives, long fundamental,
                          int varyDerivatives, int doSkew, long Nz,
                          long *nBlocksRet, long *NtotRet);
static void freeBlocks(BLOCK *blocks, long nBlocks, long derivatives);
static void fitBlocks(BLOCK *blocks, long nBlocks, long Ntot, long derivatives,
                      FIT_DATA *fd, long *underDeterminedRet, long *singularRet);
static void reconstructField(BLOCK *blocks, long nBlocks, double r, double phi,
                             long iz, double B[3]);
static double evaluateBlocks(BLOCK *blocks, long nBlocks, FIT_DATA *fd,
                             double *coordLimit, double significance,
                             unsigned long flags, ALL_RESIDUALS *res);
static int writeGGEOutput(char *outputFile, short skew, BLOCK *blocks, long nBlocks,
                          long derivatives, double *zPlane, long Nz,
                          double xCenter, double yCenter, double xMax, double yMax);
static int writeEvaluation(char *outputFile, BLOCK *blocks, long nBlocks,
                           FIT_DATA *fd, double xCenter, double yCenter);

int threads = 1;

/* ------------------------------------------------------------------ */
#define SET_FIELDMAP 0
#define SET_NORMAL 1
#define SET_SKEW 2
#define SET_DERIVATIVES 3
#define SET_MULTIPOLES 4
#define SET_FUNDAMENTAL 5
#define SET_VARY_DERIVATIVES 6
#define SET_DELTAZ 7
#define SET_COREWEIGHT 8
#define SET_XMAX 9
#define SET_YMAX 10
#define SET_XCENTER 11
#define SET_YCENTER 12
#define SET_THREADS 13
#define SET_VERBOSE 14
#define SET_AUTO_TUNE 15
#define SET_EVALUATE 16
#define N_OPTIONS 17

char *option[N_OPTIONS] = {
  "fieldmap", "normal", "skew", "derivatives", "multipoles", "fundamental",
  "varyderivatives", "deltaz", "coreweight", "xmax", "ymax", "xcenter",
  "ycenter", "threads", "verbose", "autotune", "evaluate"};

#define USAGE "computeLBGGE -fieldMap=<filename> -normal=<output> [-skew=<output>]\n\
             [-derivatives=<integer>] [-multipoles=<integer>] [-fundamental=<integer>]\n\
             [-varyDerivatives] [-deltaZ=<meters>] [-coreWeight=<value>]\n\
             [-xMax=<meters>] [-yMax=<meters>] [-xCenter=<meters>] [-yCenter=<meters>]\n\
             [-threads=<integer>] [-verbose]\n\
             [-autotune[,significance=<fieldValue>][,minimize={rms|mav|maximum}]\n\
                       [,radiusLimit=<meters>][,xLimit=<meters>][,yLimit=<meters>]\n\
                       [,increaseOnly][,evaluate][,verbose][,log=<filename>]\n\
                       [,minDerivatives=<integer>][,minMultipoles=<integer>]]\n\
             [-evaluate=<filename>]\n\n\
Computes a generalized-gradient expansion from a 3-D magnetic field map using\n\
the local weighted least-squares (\"Bmad\") method of Sagan et al. (IPAC2023).\n\
The output is in the same format as computeRBGGE and is usable by the elegant\n\
BGGEXP element.\n\n\
-fieldMap    (x, y, z, Bx, By, Bz) 3-D field map (x,y,z in m; Bx,By,Bz in T).\n\
             The distinct z values in the map define the output z planes and\n\
             must be uniformly spaced.\n\
-normal      Output file for normal-component generalized gradients.\n\
-skew        Output file for skew-component generalized gradients.  If the\n\
             input data has non-zero Bz on axis, this option is essential.\n\
-derivatives Number of derivatives vs z desired in output. Default: 7\n\
             With -autotune this is the maximum number of derivatives.\n\
-multipoles  Number of multipoles desired in output. Default: 8\n\
             With -autotune this is the maximum number of multipoles.\n\
-fundamental Fundamental multipole of sequence. 0=none (default), 1=dipole, etc.\n\
-varyDerivatives\n\
             If given, the number of derivatives used varies with multipole\n\
             order so as to maintain an approximately consistent maximum\n\
             transverse order.\n\
-deltaZ      Half-width (m) of the z window used for each local fit. Default:\n\
             twice the z-plane spacing (a five-plane window).\n\
-coreWeight  Core weighting factor w_c (>=1) from paper Eq. 10; up-weights\n\
             near-axis samples. Default: 1 (uniform weighting).\n\
-xMax,-yMax  Half-aperture of the expansion (m). Default: map transverse extent.\n\
-xCenter,-yCenter\n\
             Center of the expansion (m). Default: midpoint of map extent.\n\
-threads     Number of threads to use (default: 1).\n\
-verbose     Print progress information.\n\
-autotune    Seeks the smallest number of multipoles and derivatives that fits\n\
             the -fieldMap data to within 'significance'.  The fit is repeated\n\
             for each candidate model (the least-squares coefficients depend on\n\
             the basis) and the goodness of fit is measured against the 3-D map.\n\
             The measure minimized is the maximum error by default, or the rms\n\
             or mean-absolute-value error.  'increaseOnly' scans only toward\n\
             increasing model size relative to the previous best (faster).\n\
             xLimit/yLimit/radiusLimit restrict the goodness-of-fit test to\n\
             samples near the axis (measured from the expansion center).\n\
             'log' writes the residual for every model tried.  'evaluate'\n\
             disables tuning and just reports the residual at the maximum model.\n\
-evaluate    Write the fitted field (Bx,By,Bz) and the map reference values at\n\
             every in-aperture sample to this file, for goodness-of-fit checks.\n\n\
Local (D. Sagan's Bmad-method) Generalized Gradient Expansion, coded by Claude Code under\n\
direction of M. Borland, using some code from computeRBGGE by M. Borland, R. Lindberg,\n\
and R. Soliday.\n\n\
(" __DATE__ " " __TIME__ ", SVN revision: " SVN_VERSION ")\n"

#define AUTOTUNE_VERBOSE 0x0001UL
#define AUTOTUNE_RMS 0x0002UL
#define AUTOTUNE_MAXIMUM 0x0004UL
#define AUTOTUNE_MAV 0x0008UL
#define AUTOTUNE_EVALONLY 0x0010UL
#define AUTOTUNE_MODE_SET 0x0100UL
#define AUTOTUNE_LOG 0x0200UL
#define AUTOTUNE_INCRONLY 0x0400UL
char *modeOption[3] = {"rms", "maximum", "mav"};

/* ------------------------------------------------------------------ */
/* Taylor coefficient dz^p / p! used for the z extrapolation (Eq. 8). */
static double taylorCoef(double dz, int p) {
  return ipow(dz, p) / factorial(p);
}

/* Sort helper: distinct sorted z values. */
static int cmpDouble(const void *a, const void *b) {
  double d = *(const double *)a - *(const double *)b;
  return d < 0 ? -1 : (d > 0 ? 1 : 0);
}

static int cmpZINDEX(const void *a, const void *b) {
  double d = ((const ZINDEX *)a)->z - ((const ZINDEX *)b)->z;
  return d < 0 ? -1 : (d > 0 ? 1 : 0);
}

/* Accumulate this block's contribution to the (Br, Bphi, Bz) design-row
 * coefficients for a single field sample located at radius r, angle phi, and
 * longitudinal offset dz = z_sample - z_plane from the fit plane.  rowBr,
 * rowBphi, rowBz are the full-length (Ntot) design rows being built up; the
 * block writes into columns [colOffset, colOffset+nOrders).                */
static void fillBlockRow(BLOCK *blk, double r, double phi, double dz,
                         double *rowBr, double *rowBphi, double *rowBz) {
  int ig, j;
  long m = blk->m;
  double mfact = dfactorial(m);
  double sin_mphi = sin(m * phi);
  double cos_mphi = cos(m * phi);

  if (blk->skew && blk->mZero) {
    /* Solenoid: on-axis Bz from C^{[1]}, and radial terms for ig>=1.
     * (C^{[0]} is not part of the field model and is not an unknown here.) */
    for (j = 1; j <= blk->jmax; j++) /* ig=0 term: Bz += 1 * C^{[1]}(z_k) */
      rowBz[blk->colOffset + (j - blk->jmin)] += taylorCoef(dz, j - 1);
    for (ig = 1; ig < blk->ndLimit; ig++) {
      double base = ipow(-1, ig) * mfact /
                    (ipow(2, 2 * ig) * factorial(ig) * factorial(ig)) *
                    ipow(r, 2 * ig - 1);
      int n0 = 2 * ig, n1 = 2 * ig + 1;
      for (j = n0; j <= blk->jmax; j++)
        rowBr[blk->colOffset + (j - blk->jmin)] +=
          base * (2 * ig) * taylorCoef(dz, j - n0);
      for (j = n1; j <= blk->jmax; j++)
        rowBz[blk->colOffset + (j - blk->jmin)] +=
          base * r * taylorCoef(dz, j - n1);
    }
    return;
  }

  for (ig = 0; ig < blk->ndLimit; ig++) {
    double base = ipow(-1, ig) * mfact /
                  (ipow(2, 2 * ig) * factorial(ig) * factorial(ig + m)) *
                  ipow(r, 2 * ig + m - 1);
    int n0 = 2 * ig;     /* order feeding Br, Bphi (C^{[2ig]})   */
    int n1 = 2 * ig + 1; /* order feeding Bz     (C^{[2ig+1]})   */
    double sBr, sBphi, sBz;
    if (!blk->skew) {
      /* normal: Br ~ sin(m phi), Bphi ~ cos(m phi) */
      sBr = (2 * ig + m) * sin_mphi;
      sBphi = m * cos_mphi;
      sBz = r * sin_mphi;
    } else {
      /* skew: Br ~ cos(m phi), Bphi ~ -sin(m phi) */
      sBr = (2 * ig + m) * cos_mphi;
      sBphi = -(double)m * sin_mphi;
      sBz = r * cos_mphi;
    }
    for (j = n0; j <= blk->jmax; j++) {
      double c = base * taylorCoef(dz, j - n0);
      int col = blk->colOffset + (j - blk->jmin);
      rowBr[col] += c * sBr;
      rowBphi[col] += c * sBphi;
    }
    for (j = n1; j <= blk->jmax; j++)
      rowBz[blk->colOffset + (j - blk->jmin)] +=
        base * sBz * taylorCoef(dz, j - n1);
  }
}

/* ------------------------------------------------------------------ */
/* Allocate and describe the (multipole, parity) blocks for one model. */
static BLOCK *buildBlocks(long multipoles, long derivatives, long fundamental,
                          int varyDerivatives, int doSkew, long Nz,
                          long *nBlocksRet, long *NtotRet) {
  long nSets = doSkew ? 2 : 1;
  long set, colOffset = 0, ib = 0, nBlocks = nSets * multipoles;
  BLOCK *blocks = tmalloc(sizeof(*blocks) * nBlocks);

  for (set = 0; set < nSets; set++) {
    short skew = (short)set; /* 0 normal, 1 skew */
    for (long ir = 0; ir < multipoles; ir++) {
      BLOCK *blk = &blocks[ib];
      long m;
      int ndLimit, order;
      if (!skew)
        m = fundamental ? fundamental * (2 * ir + 1) : ir + 1;
      else
        m = fundamental ? fundamental * (2 * ir + 1) : ir;
      ndLimit = (int)derivatives;
      if (varyDerivatives)
        ndLimit -= (int)(m / 2);
      if (ndLimit < 1)
        ndLimit = 1;
      blk->skew = skew;
      blk->m = m;
      blk->ir = ir;
      blk->mZero = (skew && m == 0) ? 1 : 0;
      blk->ndLimit = ndLimit;
      blk->jmax = 2 * ndLimit - 1;
      blk->jmin = blk->mZero ? 1 : 0;
      blk->nOrders = blk->jmax - blk->jmin + 1;
      blk->colOffset = (int)colOffset;
      colOffset += blk->nOrders;
      /* Solution store: full 0..2*derivatives-1 so unused orders write 0. */
      blk->coef = tmalloc(sizeof(*blk->coef) * (2 * derivatives));
      for (order = 0; order < 2 * derivatives; order++)
        blk->coef[order] = tmalloc(sizeof(double) * Nz);
      ib++;
    }
  }
  *nBlocksRet = nBlocks;
  *NtotRet = colOffset;
  return blocks;
}

static void freeBlocks(BLOCK *blocks, long nBlocks, long derivatives) {
  long ib;
  int order;
  for (ib = 0; ib < nBlocks; ib++) {
    for (order = 0; order < 2 * derivatives; order++)
      free(blocks[ib].coef[order]);
    free(blocks[ib].coef);
  }
  free(blocks);
}

/* ------------------------------------------------------------------ */
/* Per-plane weighted least-squares fit.  Fills blocks[].coef[order][iz].
 * Returns via out-params the number of planes that were under-determined or
 * near-singular (the caller decides whether to warn).                       */
static void fitBlocks(BLOCK *blocks, long nBlocks, long Ntot, long derivatives,
                      FIT_DATA *fd, long *underDeterminedRet, long *singularRet) {
  long underDetermined = 0, singularPlanes = 0;
  long iz;
  ZINDEX *samples = fd->samples;
  long nInc = fd->nInc;
  double *sr = fd->sr, *sphi = fd->sphi, *sw = fd->sw;
  double *sBx = fd->sBx, *sBy = fd->sBy, *sBz = fd->sBz;
  double deltaZ = fd->deltaZ;

#pragma omp parallel for schedule(dynamic) reduction(+ : underDetermined, singularPlanes)
  for (iz = 0; iz < fd->Nz; iz++) {
    double zi = fd->zPlane[iz];
    double zlo = zi - deltaZ * (1 + 1e-6);
    double zhi = zi + deltaZ * (1 + 1e-6);
    long lo, hi, mid, kk;
    long nObs;
    double *M = tmalloc(sizeof(double) * Ntot * Ntot);
    double *rhs = tmalloc(sizeof(double) * Ntot);
    double *x = tmalloc(sizeof(double) * Ntot);
    double *rowBr = tmalloc(sizeof(double) * Ntot);
    double *rowBphi = tmalloc(sizeof(double) * Ntot);
    double *rowBz = tmalloc(sizeof(double) * Ntot);
    long a, b;

    /* window [lo, hi) via binary search on sorted sample z */
    lo = 0;
    hi = nInc;
    {
      long l = 0, r = nInc;
      while (l < r) { mid = (l + r) / 2; if (samples[mid].z < zlo) l = mid + 1; else r = mid; }
      lo = l;
      l = 0; r = nInc;
      while (l < r) { mid = (l + r) / 2; if (samples[mid].z <= zhi) l = mid + 1; else r = mid; }
      hi = l;
    }
    nObs = (hi - lo) * 3;

    memset(M, 0, sizeof(double) * Ntot * Ntot);
    memset(rhs, 0, sizeof(double) * Ntot);

    for (kk = lo; kk < hi; kk++) {
      double r = sr[kk], phi = sphi[kk], w = sw[kk];
      double dzk = samples[kk].z - zi;
      double cphi = cos(phi), sphi_ = sin(phi);
      int obs;
      double target[3];
      double *rows[3];
      memset(rowBr, 0, sizeof(double) * Ntot);
      memset(rowBphi, 0, sizeof(double) * Ntot);
      memset(rowBz, 0, sizeof(double) * Ntot);
      for (long ibb = 0; ibb < nBlocks; ibb++)
        fillBlockRow(&blocks[ibb], r, phi, dzk, rowBr, rowBphi, rowBz);
      /* Convert (Br, Bphi) design rows into Cartesian; Bz is direct. */
      for (a = 0; a < Ntot; a++) {
        double br = rowBr[a], bp = rowBphi[a];
        rowBr[a] = br * cphi - bp * sphi_; /* now holds Bx row */
        rowBphi[a] = br * sphi_ + bp * cphi; /* now holds By row */
      }
      rows[0] = rowBr;   target[0] = sBx[kk];
      rows[1] = rowBphi; target[1] = sBy[kk];
      rows[2] = rowBz;   target[2] = sBz[kk];
      for (obs = 0; obs < 3; obs++) {
        double *row = rows[obs];
        double t = target[obs];
        for (a = 0; a < Ntot; a++) {
          double wa = w * row[a];
          if (wa == 0)
            continue;
          rhs[a] += wa * t;
          for (b = 0; b < Ntot; b++)
            M[a * Ntot + b] += wa * row[b];
        }
      }
    }

    if (nObs < Ntot)
      underDetermined++;

    if (gaussSolve(M, rhs, (int)Ntot, x))
      singularPlanes++;

    /* Scatter the solution into the per-block coefficient store. */
    for (long ibb = 0; ibb < nBlocks; ibb++) {
      BLOCK *blk = &blocks[ibb];
      int order;
      for (order = 0; order < 2 * derivatives; order++)
        blk->coef[order][iz] = 0.0;
      for (order = blk->jmin; order <= blk->jmax; order++)
        blk->coef[order][iz] = x[blk->colOffset + (order - blk->jmin)];
    }

    free(M);
    free(rhs);
    free(x);
    free(rowBr);
    free(rowBphi);
    free(rowBz);
  }

  *underDeterminedRet = underDetermined;
  *singularRet = singularPlanes;
}

/* ------------------------------------------------------------------ */
/* Reconstruct the field at (r, phi) on z-plane iz from the fitted blocks.
 * This mirrors computeRBGGE's evaluateGGEForFieldMap exactly, reading
 * C^{[2ig]} from coef[2ig][iz] and C^{[2ig+1]} from coef[2ig+1][iz].        */
static void reconstructField(BLOCK *blocks, long nBlocks, double r, double phi,
                             long iz, double B[3]) {
  double Br = 0, Bphi = 0;
  long ib;
  B[0] = B[1] = B[2] = 0;
  for (ib = 0; ib < nBlocks; ib++) {
    BLOCK *blk = &blocks[ib];
    long m = blk->m;
    double mfact = dfactorial(m);
    double sin_mphi = sin(m * phi), cos_mphi = cos(m * phi);
    double term;
    int ig;
    if (blk->skew && blk->mZero) {
      B[2] += blk->coef[1][iz]; /* on-axis Bz from C^{[1]} */
      for (ig = 1; ig < blk->ndLimit; ig++) {
        term = ipow(-1, ig) * mfact / (ipow(2, 2 * ig) * factorial(ig) * factorial(ig + m)) * ipow(r, 2 * ig + m - 1);
        B[2] += term * blk->coef[2 * ig + 1][iz] * r;
        Br += term * (2 * ig + m) * blk->coef[2 * ig][iz];
      }
    } else if (!blk->skew) {
      for (ig = 0; ig < blk->ndLimit; ig++) {
        term = ipow(-1, ig) * mfact / (ipow(2, 2 * ig) * factorial(ig) * factorial(ig + m)) * ipow(r, 2 * ig + m - 1);
        B[2] += term * blk->coef[2 * ig + 1][iz] * r * sin_mphi;
        term *= blk->coef[2 * ig][iz];
        Br += term * (2 * ig + m) * sin_mphi;
        Bphi += m * term * cos_mphi;
      }
    } else {
      for (ig = 0; ig < blk->ndLimit; ig++) {
        term = ipow(-1, ig) * mfact / (ipow(2, 2 * ig) * factorial(ig) * factorial(ig + m)) * ipow(r, 2 * ig + m - 1);
        B[2] += term * blk->coef[2 * ig + 1][iz] * r * cos_mphi;
        term *= blk->coef[2 * ig][iz];
        Br += term * (2 * ig + m) * cos_mphi;
        Bphi -= m * term * sin_mphi;
      }
    }
  }
  B[0] = Br * cos(phi) - Bphi * sin(phi);
  B[1] = Br * sin(phi) + Bphi * cos(phi);
}

/* ------------------------------------------------------------------ */
/* Compute the goodness of fit of the reconstructed field against the map.
 * Fills *res and returns the comparison residual (per the selected mode),
 * or 0.0 if it is at or below 'significance' (mirrors computeRBGGE).        */
static double evaluateBlocks(BLOCK *blocks, long nBlocks, FIT_DATA *fd,
                             double *coordLimit, double significance,
                             unsigned long flags, ALL_RESIDUALS *res) {
  double worst = 0, sum = 0, sum2 = 0, maxField = 0, cmp;
  long count = 0, k;

  for (k = 0; k < fd->nInc; k++) {
    double r = fd->sr[k], phi = fd->sphi[k];
    double xr = r * cos(phi), yr = r * sin(phi);
    double B[3], field, resid;
    if ((coordLimit[0] > 0 && fabs(xr) > coordLimit[0]) ||
        (coordLimit[1] > 0 && fabs(yr) > coordLimit[1]))
      continue;
    if (coordLimit[2] > 0 && r > coordLimit[2])
      continue;
    reconstructField(blocks, nBlocks, r, phi, fd->sIz[k], B);
    field = sqrt(sqr(B[0]) + sqr(B[1]) + sqr(B[2]));
    if (field > maxField)
      maxField = field;
    resid = sqrt(sqr(B[0] - fd->sBx[k]) + sqr(B[1] - fd->sBy[k]) + sqr(B[2] - fd->sBz[k]));
    if (resid > worst)
      worst = resid;
    sum += resid;
    sum2 += sqr(resid);
    count++;
  }

  res->max = worst;
  res->rms = count ? sqrt(sum2 / count) : DBL_MAX;
  res->mad = count ? sum / count : DBL_MAX;
  if (maxField > 0) {
    res->fracRms = res->rms / maxField;
    res->fracMad = res->mad / maxField;
    res->fracMax = res->max / maxField;
  } else
    res->fracRms = res->fracMad = res->fracMax = -DBL_MAX;

  cmp = worst;
  if (flags & AUTOTUNE_RMS)
    cmp = res->rms;
  else if (flags & AUTOTUNE_MAV)
    cmp = res->mad;
  return cmp > significance ? cmp : 0.0;
}

/* ------------------------------------------------------------------ */
/* Write one output file (normal or skew) in computeRBGGE format.      */
static int writeGGEOutput(char *outputFile, short skew, BLOCK *blocks, long nBlocks,
                          long derivatives, double *zPlane, long Nz,
                          double xCenter, double yCenter, double xMax, double yMax) {
  SDDS_DATASET SDDSOutput;
  char name[64], units[64];
  const char *frag = skew ? "CnmC" : "CnmS";
  long Nderiv = 2 * derivatives - 1;
  long ib, iz;
  int32_t n;

  if (SDDS_InitializeOutput(&SDDSOutput, SDDS_BINARY, 1, NULL,
                            skew ? "computeLBGGE skew output" : "computeLBGGE normal output",
                            outputFile) != 1) {
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
    return 1;
  }
  if ((SDDS_DefineSimpleParameter(&SDDSOutput, "m", NULL, SDDS_LONG) != 1) ||
      (SDDS_DefineSimpleParameter(&SDDSOutput, "xCenter", "m", SDDS_DOUBLE) != 1) ||
      (SDDS_DefineSimpleParameter(&SDDSOutput, "yCenter", "m", SDDS_DOUBLE) != 1) ||
      (SDDS_DefineSimpleParameter(&SDDSOutput, "xMax", "m", SDDS_DOUBLE) != 1) ||
      (SDDS_DefineSimpleParameter(&SDDSOutput, "yMax", "m", SDDS_DOUBLE) != 1) ||
      (SDDS_DefineSimpleColumn(&SDDSOutput, "z", "m", SDDS_DOUBLE) != 1)) {
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
    return 1;
  }
  for (n = 0; n < Nderiv; n += 2) {
    sprintf(name, "%s%" PRId32, frag, n);
    if ((2 * n - 1) < 0)
      sprintf(units, "T/m$a(m-%d)$n", -(2 * n - 1));
    else if ((2 * n - 1) == 0)
      sprintf(units, "T/m$am$n");
    else
      sprintf(units, "T/m$a(m+%d)$n", (2 * n - 1));
    if (SDDS_DefineSimpleColumn(&SDDSOutput, name, units, SDDS_DOUBLE) != 1) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      return 1;
    }
  }
  for (n = 0; n < Nderiv; n += 2) {
    sprintf(name, "d%s%" PRId32 "/dz", frag, n);
    if ((2 * n - 2) < 0)
      sprintf(units, "T/m$a(m-%d)$n", -(2 * n - 2));
    else if ((2 * n - 2) == 0)
      sprintf(units, "T/m$am$n");
    else
      sprintf(units, "T/m$a(m+%d)$n", (2 * n - 2));
    if (SDDS_DefineSimpleColumn(&SDDSOutput, name, units, SDDS_DOUBLE) != 1) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      return 1;
    }
  }
  if (SDDS_WriteLayout(&SDDSOutput) != 1) {
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
    return 1;
  }

  for (ib = 0; ib < nBlocks; ib++) {
    BLOCK *blk = &blocks[ib];
    if ((short)blk->skew != skew)
      continue;
    if (SDDS_StartPage(&SDDSOutput, Nz) != 1) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      return 1;
    }
    if (SDDS_SetParameters(&SDDSOutput, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                           "m", (int32_t)blk->m,
                           "xCenter", xCenter, "yCenter", yCenter,
                           "xMax", xMax, "yMax", yMax, NULL) != 1) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      return 1;
    }
    for (iz = 0; iz < Nz; iz++) {
      if (SDDS_SetRowValues(&SDDSOutput, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, iz,
                            "z", zPlane[iz], NULL) != 1) {
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        return 1;
      }
      for (n = 0; n < Nderiv; n += 2) {
        /* CnmX{n} holds C^{[n]}; dCnmX{n}/dz holds C^{[n+1]}. */
        sprintf(name, "%s%" PRId32, frag, n);
        if (SDDS_SetRowValues(&SDDSOutput, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, iz,
                              name, blk->coef[n][iz], NULL) != 1) {
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
          return 1;
        }
        sprintf(name, "d%s%" PRId32 "/dz", frag, n);
        if (SDDS_SetRowValues(&SDDSOutput, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, iz,
                              name, blk->coef[n + 1][iz], NULL) != 1) {
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
          return 1;
        }
      }
    }
    if (SDDS_WritePage(&SDDSOutput) != 1) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      return 1;
    }
  }
  if (SDDS_Terminate(&SDDSOutput) != 1) {
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
    return 1;
  }
  return 0;
}

/* ------------------------------------------------------------------ */
/* Write the reconstructed field and the map reference at each in-aperture
 * sample, for goodness-of-fit inspection (analogue of computeRBGGE -evaluate). */
static int writeEvaluation(char *outputFile, BLOCK *blocks, long nBlocks,
                           FIT_DATA *fd, double xCenter, double yCenter) {
  SDDS_DATASET SDDSout;
  long k;

  if (SDDS_InitializeOutput(&SDDSout, SDDS_BINARY, 1, NULL, "computeLBGGE evaluation", outputFile) != 1 ||
      !SDDS_DefineSimpleColumn(&SDDSout, "x", "m", SDDS_DOUBLE) ||
      !SDDS_DefineSimpleColumn(&SDDSout, "y", "m", SDDS_DOUBLE) ||
      !SDDS_DefineSimpleColumn(&SDDSout, "z", "m", SDDS_DOUBLE) ||
      !SDDS_DefineSimpleColumn(&SDDSout, "Bx", "T", SDDS_DOUBLE) ||
      !SDDS_DefineSimpleColumn(&SDDSout, "By", "T", SDDS_DOUBLE) ||
      !SDDS_DefineSimpleColumn(&SDDSout, "Bz", "T", SDDS_DOUBLE) ||
      !SDDS_DefineSimpleColumn(&SDDSout, "residual", "T", SDDS_DOUBLE) ||
      !SDDS_WriteLayout(&SDDSout) ||
      !SDDS_StartPage(&SDDSout, fd->nInc)) {
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
    return 1;
  }

  for (k = 0; k < fd->nInc; k++) {
    double r = fd->sr[k], phi = fd->sphi[k];
    double B[3], resid;
    reconstructField(blocks, nBlocks, r, phi, fd->sIz[k], B);
    resid = sqrt(sqr(B[0] - fd->sBx[k]) + sqr(B[1] - fd->sBy[k]) + sqr(B[2] - fd->sBz[k]));
    if (SDDS_SetRowValues(&SDDSout, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, k,
                          "x", xCenter + r * cos(phi), "y", yCenter + r * sin(phi),
                          "z", fd->zPlane[fd->sIz[k]],
                          "Bx", B[0], "By", B[1], "Bz", B[2],
                          "residual", resid, NULL) != 1) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      return 1;
    }
  }
  if (SDDS_WritePage(&SDDSout) != 1 || SDDS_Terminate(&SDDSout) != 1) {
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
    return 1;
  }
  return 0;
}

/* ------------------------------------------------------------------ */
int main(int argc, char **argv) {
  SCANNED_ARG *scanned;
  long i_arg;
  char *fieldMapFile = NULL, *normalOutputFile = NULL, *skewOutputFile = NULL;
  char *evaluationOutput = NULL;
  long derivatives = 7, multipoles = 8, fundamental = 0, varyDerivatives = 0, verbose = 0;
  double deltaZ = -1, coreWeight = 1.0;
  double xMax = -1, yMax = -1, xCenter = 0, yCenter = 0;
  short haveXMax = 0, haveYMax = 0, haveXCenter = 0, haveYCenter = 0;
  FIELD_MAP fieldMap;
  double *zPlane = NULL;
  long Nz = 0, ip, k;
  double dz = 0;
  long nBlocks = 0, Ntot = 0;
  BLOCK *blocks = NULL;
  ZINDEX *samples = NULL;
  double *sr = NULL, *sphi = NULL, *sw = NULL, *sBx = NULL, *sBy = NULL, *sBz = NULL;
  long *sIz = NULL;
  long nInc = 0;
  double Rmax2;
  long iz;
  long underDetermined = 0, singularPlanes = 0;
  FIT_DATA fd;
  /* autotune state */
  short doAutotune = 0;
  unsigned long autoTuneFlags = 0;
  double autoTuneSignificance = 1e-12, autoTuneCoordLimit[3] = {0, 0, 0};
  char *autoTuneModeString = NULL, *autoTuneLogFile = NULL;
  int32_t minMultipoles = -1, minDerivatives = -1;
  SDDS_DATASET SDDS_autoTuneLog;
  long iAutoTuneLog = 0;

  argc = scanargs(&scanned, argc, argv);
  if (argc < 2) {
    fprintf(stderr, "%s\n", USAGE);
    return 1;
  }
  for (i_arg = 1; i_arg < argc; i_arg++) {
    if (scanned[i_arg].arg_type == OPTION) {
      switch (match_string(scanned[i_arg].list[0], option, N_OPTIONS, 0)) {
      case SET_FIELDMAP:
        if (scanned[i_arg].n_items != 2)
          SDDS_Bomb("invalid -fieldMap syntax");
        fieldMapFile = scanned[i_arg].list[1];
        break;
      case SET_NORMAL:
        if (scanned[i_arg].n_items != 2)
          SDDS_Bomb("invalid -normal syntax");
        normalOutputFile = scanned[i_arg].list[1];
        break;
      case SET_SKEW:
        if (scanned[i_arg].n_items != 2)
          SDDS_Bomb("invalid -skew syntax");
        skewOutputFile = scanned[i_arg].list[1];
        break;
      case SET_DERIVATIVES:
        if (scanned[i_arg].n_items != 2 ||
            sscanf(scanned[i_arg].list[1], "%ld", &derivatives) != 1 || derivatives <= 0)
          SDDS_Bomb("invalid -derivatives syntax");
        break;
      case SET_MULTIPOLES:
        if (scanned[i_arg].n_items != 2 ||
            sscanf(scanned[i_arg].list[1], "%ld", &multipoles) != 1 || multipoles <= 0)
          SDDS_Bomb("invalid -multipoles syntax");
        break;
      case SET_FUNDAMENTAL:
        if (scanned[i_arg].n_items != 2 ||
            sscanf(scanned[i_arg].list[1], "%ld", &fundamental) != 1 || fundamental < 0)
          SDDS_Bomb("invalid -fundamental syntax");
        break;
      case SET_VARY_DERIVATIVES:
        varyDerivatives = 1;
        break;
      case SET_DELTAZ:
        if (scanned[i_arg].n_items != 2 ||
            sscanf(scanned[i_arg].list[1], "%lf", &deltaZ) != 1 || deltaZ <= 0)
          SDDS_Bomb("invalid -deltaZ syntax");
        break;
      case SET_COREWEIGHT:
        if (scanned[i_arg].n_items != 2 ||
            sscanf(scanned[i_arg].list[1], "%lf", &coreWeight) != 1 || coreWeight < 1)
          SDDS_Bomb("invalid -coreWeight syntax: give a value >= 1");
        break;
      case SET_XMAX:
        if (scanned[i_arg].n_items != 2 ||
            sscanf(scanned[i_arg].list[1], "%lf", &xMax) != 1 || xMax <= 0)
          SDDS_Bomb("invalid -xMax syntax");
        haveXMax = 1;
        break;
      case SET_YMAX:
        if (scanned[i_arg].n_items != 2 ||
            sscanf(scanned[i_arg].list[1], "%lf", &yMax) != 1 || yMax <= 0)
          SDDS_Bomb("invalid -yMax syntax");
        haveYMax = 1;
        break;
      case SET_XCENTER:
        if (scanned[i_arg].n_items != 2 ||
            sscanf(scanned[i_arg].list[1], "%lf", &xCenter) != 1)
          SDDS_Bomb("invalid -xCenter syntax");
        haveXCenter = 1;
        break;
      case SET_YCENTER:
        if (scanned[i_arg].n_items != 2 ||
            sscanf(scanned[i_arg].list[1], "%lf", &yCenter) != 1)
          SDDS_Bomb("invalid -yCenter syntax");
        haveYCenter = 1;
        break;
      case SET_THREADS:
        if (scanned[i_arg].n_items != 2 || sscanf(scanned[i_arg].list[1], "%d", &threads) != 1 ||
            threads <= 0)
          SDDS_Bomb("invalid -threads syntax: give a value greater than 0");
        break;
      case SET_VERBOSE:
        verbose = 1;
        break;
      case SET_AUTO_TUNE:
        doAutotune = 1;
        autoTuneFlags = 0;
        autoTuneSignificance = 1e-12;
        autoTuneCoordLimit[0] = autoTuneCoordLimit[1] = autoTuneCoordLimit[2] = 0;
        minDerivatives = minMultipoles = -1;
        /* Unlike computeRBGGE, there is no map-file argument: the 3-D map is
         * the -fieldMap already given.  Sub-items start at list[1].          */
        scanned[i_arg].n_items -= 1;
        if (scanned[i_arg].n_items > 0 &&
            (!scanItemList(&autoTuneFlags, scanned[i_arg].list + 1, &scanned[i_arg].n_items, 0,
                           "verbose", -1, NULL, 0, AUTOTUNE_VERBOSE,
                           "increaseonly", -1, NULL, 0, AUTOTUNE_INCRONLY,
                           "evaluate", -1, NULL, 0, AUTOTUNE_EVALONLY,
                           "significance", SDDS_DOUBLE, &autoTuneSignificance, 1, 0,
                           "xlimit", SDDS_DOUBLE, &autoTuneCoordLimit[0], 1, 0,
                           "ylimit", SDDS_DOUBLE, &autoTuneCoordLimit[1], 1, 0,
                           "radiuslimit", SDDS_DOUBLE, &autoTuneCoordLimit[2], 1, 0,
                           "minimize", SDDS_STRING, &autoTuneModeString, 1, AUTOTUNE_MODE_SET,
                           "log", SDDS_STRING, &autoTuneLogFile, 1, AUTOTUNE_LOG,
                           "minmultipoles", SDDS_LONG, &minMultipoles, 1, 0,
                           "minderivatives", SDDS_LONG, &minDerivatives, 1, 0,
                           NULL) ||
             autoTuneSignificance <= 0)) {
          fprintf(stderr, "invalid -autotune syntax\n%s\n", USAGE);
          return 1;
        }
        if (autoTuneFlags & AUTOTUNE_MODE_SET) {
          switch (match_string(autoTuneModeString, modeOption, 3, 0)) {
          case 0:
            autoTuneFlags |= AUTOTUNE_RMS;
            break;
          case 1:
            autoTuneFlags |= AUTOTUNE_MAXIMUM;
            break;
          case 2:
            autoTuneFlags |= AUTOTUNE_MAV;
            break;
          default:
            SDDS_Bomb("invalid mode for autotune minimization. Use rms, mav, or maximum.");
            break;
          }
        } else
          autoTuneFlags |= AUTOTUNE_MAXIMUM;
        break;
      case SET_EVALUATE:
        if (scanned[i_arg].n_items != 2)
          SDDS_Bomb("invalid -evaluate syntax");
        evaluationOutput = scanned[i_arg].list[1];
        break;
      default:
        fprintf(stderr, "unknown option given\n%s\n", USAGE);
        return 1;
      }
    } else {
      fprintf(stderr, "too many files listed\n%s\n", USAGE);
      return 1;
    }
  }

  if (!fieldMapFile || !normalOutputFile) {
    fprintf(stderr, "%s\n", USAGE);
    return 1;
  }

#if !defined(NOTHREADS)
  omp_set_num_threads(threads);
#endif

  /* --- Read the 3-D field map --------------------------------------- */
  readFieldMap(fieldMapFile, &fieldMap);
  if (verbose)
    fprintf(stdout, "Read %ld field samples from %s\n", fieldMap.n, fieldMapFile);

  /* --- Geometry: centers and half-apertures ------------------------- */
  {
    double xmn = fieldMap.x[0], xmx = fieldMap.x[0];
    double ymn = fieldMap.y[0], ymx = fieldMap.y[0];
    for (ip = 1; ip < fieldMap.n; ip++) {
      if (fieldMap.x[ip] < xmn) xmn = fieldMap.x[ip];
      if (fieldMap.x[ip] > xmx) xmx = fieldMap.x[ip];
      if (fieldMap.y[ip] < ymn) ymn = fieldMap.y[ip];
      if (fieldMap.y[ip] > ymx) ymx = fieldMap.y[ip];
    }
    if (!haveXCenter) xCenter = 0.5 * (xmn + xmx);
    if (!haveYCenter) yCenter = 0.5 * (ymn + ymx);
    if (!haveXMax) xMax = MAX(xmx - xCenter, xCenter - xmn);
    if (!haveYMax) yMax = MAX(ymx - yCenter, yCenter - ymn);
  }
  Rmax2 = xMax * xMax + yMax * yMax;
  if (Rmax2 <= 0)
    SDDS_Bomb("degenerate aperture (xMax=yMax=0)");

  /* --- Distinct, uniformly spaced z planes -------------------------- */
  {
    double *zc = tmalloc(sizeof(*zc) * fieldMap.n);
    double zrange, epsZ, dzMin = DBL_MAX, last;
    long np = 0;
    memcpy(zc, fieldMap.z, sizeof(*zc) * fieldMap.n);
    qsort(zc, fieldMap.n, sizeof(*zc), cmpDouble);
    zrange = zc[fieldMap.n - 1] - zc[0];
    epsZ = 1e-8 * (zrange > 0 ? zrange : 1.0);
    for (ip = 1; ip < fieldMap.n; ip++) {
      double d = zc[ip] - zc[ip - 1];
      if (d > epsZ && d < dzMin)
        dzMin = d;
    }
    if (dzMin == DBL_MAX)
      SDDS_Bomb("field map has only a single z plane");
    /* Group values into planes separated by > 0.5*dzMin. */
    zPlane = tmalloc(sizeof(*zPlane) * fieldMap.n);
    zPlane[np++] = last = zc[0];
    for (ip = 1; ip < fieldMap.n; ip++) {
      if (zc[ip] - last > 0.5 * dzMin) {
        zPlane[np++] = last = zc[ip];
      }
    }
    Nz = np;
    free(zc);
    if (Nz < 2)
      SDDS_Bomb("field map must contain at least two distinct z planes");
    dz = (zPlane[Nz - 1] - zPlane[0]) / (Nz - 1);
    for (iz = 1; iz < Nz; iz++) {
      double spacing = zPlane[iz] - zPlane[iz - 1];
      if (fabs(spacing - dz) > 1e-3 * dz) {
        fprintf(stderr, "computeLBGGE: the field map z planes are not uniformly spaced\n");
        fprintf(stderr, "  plane %ld spacing=%.8le, expected=%.8le\n", iz, spacing, dz);
        return 1;
      }
    }
  }
  if (deltaZ <= 0)
    deltaZ = 2 * dz; /* five-plane window in the interior */
  if (verbose)
    fprintf(stdout, "Geometry: Nz=%ld, dz=%.6le, deltaZ=%.6le, xCenter=%.6le, yCenter=%.6le, xMax=%.6le, yMax=%.6le\n",
            Nz, dz, deltaZ, xCenter, yCenter, xMax, yMax);

  /* --- Collect in-aperture samples, sorted by z --------------------- */
  samples = tmalloc(sizeof(*samples) * fieldMap.n);
  nInc = 0;
  for (ip = 0; ip < fieldMap.n; ip++) {
    double x = fieldMap.x[ip] - xCenter;
    double y = fieldMap.y[ip] - yCenter;
    if (fabs(x) > xMax * (1 + 1e-6) || fabs(y) > yMax * (1 + 1e-6))
      continue;
    samples[nInc].z = fieldMap.z[ip];
    samples[nInc].ip = ip;
    nInc++;
  }
  if (nInc < 1)
    SDDS_Bomb("no field samples fall within the requested aperture");
  qsort(samples, nInc, sizeof(*samples), cmpZINDEX);

  sr = tmalloc(sizeof(*sr) * nInc);
  sphi = tmalloc(sizeof(*sphi) * nInc);
  sw = tmalloc(sizeof(*sw) * nInc);
  sBx = tmalloc(sizeof(*sBx) * nInc);
  sBy = tmalloc(sizeof(*sBy) * nInc);
  sBz = tmalloc(sizeof(*sBz) * nInc);
  sIz = tmalloc(sizeof(*sIz) * nInc);
  for (k = 0; k < nInc; k++) {
    long p = samples[k].ip;
    double x = fieldMap.x[p] - xCenter;
    double y = fieldMap.y[p] - yCenter;
    double r2 = x * x + y * y;
    long jz = (long)floor((samples[k].z - zPlane[0]) / dz + 0.5);
    if (jz < 0) jz = 0;
    if (jz >= Nz) jz = Nz - 1;
    sr[k] = sqrt(r2);
    sphi[k] = atan2(y, x);
    sw[k] = Rmax2 / (Rmax2 + (coreWeight - 1) * r2); /* paper Eq. 10 */
    sBx[k] = fieldMap.Bx[p];
    sBy[k] = fieldMap.By[p];
    sBz[k] = fieldMap.Bz[p];
    sIz[k] = jz;
  }

  fd.samples = samples;
  fd.nInc = nInc;
  fd.sr = sr; fd.sphi = sphi; fd.sw = sw;
  fd.sBx = sBx; fd.sBy = sBy; fd.sBz = sBz;
  fd.sIz = sIz;
  fd.zPlane = zPlane; fd.Nz = Nz;
  fd.dz = dz; fd.deltaZ = deltaZ;

  if (!doAutotune) {
    /* -------- Single fit at the requested model -------------------- */
    blocks = buildBlocks(multipoles, derivatives, fundamental, (int)varyDerivatives,
                         skewOutputFile ? 1 : 0, Nz, &nBlocks, &Ntot);
    if (verbose)
      fprintf(stdout, "Fitting %ld blocks, %ld unknowns per plane\n", nBlocks, Ntot);
    fitBlocks(blocks, nBlocks, Ntot, derivatives, &fd, &underDetermined, &singularPlanes);
    if (underDetermined)
      fprintf(stderr, "computeLBGGE: warning: %ld of %ld planes had fewer observations than "
                      "unknowns; those fits are regularized and may be unreliable near the ends.\n",
              underDetermined, Nz);
    if (singularPlanes)
      fprintf(stderr, "computeLBGGE: warning: %ld of %ld planes produced a near-singular normal "
                      "matrix; affected coefficients were regularized toward zero.\n",
              singularPlanes, Nz);
  } else {
    /* -------- Autotune: search for the smallest good model --------- */
    long maxDerivatives = derivatives, maxMultipoles = multipoles;
    long bestDerivatives = maxDerivatives, bestMultipoles = maxMultipoles;
    double bestResidual = DBL_MAX;
    ALL_RESIDUALS allResiduals;
    long d, m;

    if (minDerivatives < 1) minDerivatives = 1;
    if (minMultipoles < 1) minMultipoles = 1;
    if (minDerivatives > maxDerivatives) minDerivatives = maxDerivatives;
    if (minMultipoles > maxMultipoles) minMultipoles = maxMultipoles;
    if (autoTuneFlags & AUTOTUNE_EVALONLY) {
      minDerivatives = maxDerivatives;
      minMultipoles = maxMultipoles;
    }

    if (autoTuneFlags & AUTOTUNE_LOG) {
      if (SDDS_InitializeOutput(&SDDS_autoTuneLog, SDDS_BINARY, 1, NULL, "computeLBGGE autotune output", autoTuneLogFile) != 1 ||
          SDDS_DefineParameter(&SDDS_autoTuneLog, "OptimalMultipoles", "m$bopt$n", NULL, "Optimal number of multipoles",
                               NULL, SDDS_LONG, NULL) == -1 ||
          SDDS_DefineParameter(&SDDS_autoTuneLog, "OptimalDerivatives", "d$bopt$n", NULL, "Optimal number of derivatives",
                               NULL, SDDS_LONG, NULL) == -1 ||
          SDDS_DefineParameter(&SDDS_autoTuneLog, "OptimalResidual", "r$bopt$n", "T", "Optimal residual",
                               NULL, SDDS_DOUBLE, NULL) == -1 ||
          !SDDS_DefineSimpleParameter(&SDDS_autoTuneLog, "OptimumLabel", NULL, SDDS_STRING) ||
          SDDS_DefineColumn(&SDDS_autoTuneLog, "m", NULL, NULL, "Number of multipoles", NULL, SDDS_LONG, 0) == -1 ||
          SDDS_DefineColumn(&SDDS_autoTuneLog, "d", NULL, NULL, "Number of derivatives", NULL, SDDS_LONG, 0) == -1 ||
          SDDS_DefineSimpleColumn(&SDDS_autoTuneLog, "RmsError", "T", SDDS_DOUBLE) != 1 ||
          SDDS_DefineSimpleColumn(&SDDS_autoTuneLog, "MaximumError", "T", SDDS_DOUBLE) != 1 ||
          SDDS_DefineSimpleColumn(&SDDS_autoTuneLog, "MadError", "T", SDDS_DOUBLE) != 1 ||
          SDDS_DefineSimpleColumn(&SDDS_autoTuneLog, "FractionalRmsError", NULL, SDDS_DOUBLE) != 1 ||
          SDDS_DefineSimpleColumn(&SDDS_autoTuneLog, "FractionalMaximumError", NULL, SDDS_DOUBLE) != 1 ||
          SDDS_DefineSimpleColumn(&SDDS_autoTuneLog, "FractionalMadError", NULL, SDDS_DOUBLE) != 1 ||
          !SDDS_WriteLayout(&SDDS_autoTuneLog) ||
          !SDDS_StartPage(&SDDS_autoTuneLog, maxMultipoles * maxDerivatives)) {
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        return 1;
      }
    }

    for (d = minDerivatives; d <= maxDerivatives; d++) {
      if (!(autoTuneFlags & AUTOTUNE_INCRONLY)) {
        if (autoTuneFlags & AUTOTUNE_EVALONLY)
          minMultipoles = maxMultipoles;
      }
      for (m = minMultipoles; m <= maxMultipoles; m++) {
        double residual;
        long nB, nT, uDet, sing;
        BLOCK *b = buildBlocks(m, d, fundamental, (int)varyDerivatives,
                               skewOutputFile ? 1 : 0, Nz, &nB, &nT);
        fitBlocks(b, nB, nT, d, &fd, &uDet, &sing);
        residual = evaluateBlocks(b, nB, &fd, autoTuneCoordLimit,
                                  autoTuneSignificance, autoTuneFlags, &allResiduals);
        if (residual < bestResidual) {
          bestResidual = residual;
          bestMultipoles = m;
          bestDerivatives = d;
          if (autoTuneFlags & AUTOTUNE_INCRONLY) {
            minMultipoles = bestMultipoles;
            minDerivatives = bestDerivatives;
          }
          if (autoTuneFlags & AUTOTUNE_VERBOSE) {
            printf("New best residual of %le for m=%ld, d=%ld\n", residual, m, d);
            fflush(stdout);
          }
        } else if (autoTuneFlags & AUTOTUNE_VERBOSE) {
          printf("Goodness of fit (%le) for m=%ld, d=%ld is not better than %le obtained for m=%ld, d=%ld\n",
                 residual, m, d, bestResidual, bestMultipoles, bestDerivatives);
          fflush(stdout);
        }
        if (autoTuneFlags & AUTOTUNE_LOG) {
          if (!SDDS_SetRowValues(&SDDS_autoTuneLog, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                                 iAutoTuneLog++,
                                 "m", m, "d", d, "RmsError", allResiduals.rms,
                                 "MaximumError", allResiduals.max, "MadError", allResiduals.mad,
                                 "FractionalRmsError", allResiduals.fracRms,
                                 "FractionalMadError", allResiduals.fracMad,
                                 "FractionalMaximumError", allResiduals.fracMax,
                                 NULL)) {
            SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
            return 1;
          }
        }
        freeBlocks(b, nB, d);
      }
    }

    if (verbose)
      fprintf(stdout, "Autotune selected multipoles=%ld, derivatives=%ld (residual %le)\n",
              bestMultipoles, bestDerivatives, bestResidual);

    if (autoTuneFlags & AUTOTUNE_LOG) {
      char buffer[1024];
      snprintf(buffer, 1024, "m$bopt$n: %ld  d$bopt$n: %ld  r$bopt$n: %lg T",
               bestMultipoles, bestDerivatives, bestResidual);
      if (SDDS_SetParameters(&SDDS_autoTuneLog, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                             "OptimalMultipoles", bestMultipoles,
                             "OptimalDerivatives", bestDerivatives,
                             "OptimalResidual", bestResidual,
                             "OptimumLabel", buffer, NULL) != 1 ||
          SDDS_WritePage(&SDDS_autoTuneLog) != 1 ||
          SDDS_Terminate(&SDDS_autoTuneLog) != 1) {
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        return 1;
      }
    }

    /* Re-fit at the selected model for the actual output. */
    multipoles = bestMultipoles;
    derivatives = bestDerivatives;
    blocks = buildBlocks(multipoles, derivatives, fundamental, (int)varyDerivatives,
                         skewOutputFile ? 1 : 0, Nz, &nBlocks, &Ntot);
    fitBlocks(blocks, nBlocks, Ntot, derivatives, &fd, &underDetermined, &singularPlanes);
    if (underDetermined)
      fprintf(stderr, "computeLBGGE: warning: %ld of %ld planes had fewer observations than "
                      "unknowns; those fits are regularized and may be unreliable near the ends.\n",
              underDetermined, Nz);
    if (singularPlanes)
      fprintf(stderr, "computeLBGGE: warning: %ld of %ld planes produced a near-singular normal "
                      "matrix; affected coefficients were regularized toward zero.\n",
              singularPlanes, Nz);
  }

  /* --- Write output ------------------------------------------------- */
  if (writeGGEOutput(normalOutputFile, 0, blocks, nBlocks, derivatives, zPlane, Nz,
                     xCenter, yCenter, xMax, yMax))
    return 1;
  if (verbose)
    fprintf(stdout, "Wrote normal GGE to %s\n", normalOutputFile);
  if (skewOutputFile) {
    if (writeGGEOutput(skewOutputFile, 1, blocks, nBlocks, derivatives, zPlane, Nz,
                       xCenter, yCenter, xMax, yMax))
      return 1;
    if (verbose)
      fprintf(stdout, "Wrote skew GGE to %s\n", skewOutputFile);
  }

  if (evaluationOutput) {
    if (writeEvaluation(evaluationOutput, blocks, nBlocks, &fd, xCenter, yCenter))
      return 1;
    if (verbose)
      fprintf(stdout, "Wrote evaluation output to %s\n", evaluationOutput);
  }

  /* --- Cleanup ------------------------------------------------------- */
  freeBlocks(blocks, nBlocks, derivatives);
  free(samples);
  free(sr); free(sphi); free(sw); free(sBx); free(sBy); free(sBz); free(sIz);
  free(zPlane);
  free(fieldMap.x); free(fieldMap.y); free(fieldMap.z);
  free(fieldMap.Bx); free(fieldMap.By); free(fieldMap.Bz);

  return 0;
}

/* ------------------------------------------------------------------ */
/* Solve the symmetric normal-equations system M x = b (M is n x n,
 * row-major).  Uses diagonal (Jacobi) preconditioning plus a tiny ridge
 * so that unconstrained/degenerate directions resolve to ~0 rather than
 * blowing up, followed by Gaussian elimination with partial pivoting.
 * Returns 1 if any diagonal was non-positive (degenerate direction),
 * else 0.  M and b are overwritten.                                    */
static int gaussSolve(double *M, double *b, int n, double *x) {
  int i, j, col, prow, degenerate = 0;
  double *d = tmalloc(sizeof(double) * n);
  const double ridge = 1e-10;

  for (i = 0; i < n; i++) {
    double diag = M[i * n + i];
    if (diag > 0)
      d[i] = sqrt(diag);
    else {
      d[i] = 1.0;
      degenerate = 1;
    }
  }
  /* symmetric diagonal scaling: M'[i][j] = M[i][j]/(d[i] d[j]) */
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++)
      M[i * n + j] /= (d[i] * d[j]);
    b[i] /= d[i];
    M[i * n + i] += ridge; /* regularize; diagonal is ~1 after scaling */
  }

  /* Gaussian elimination with partial pivoting */
  for (col = 0; col < n; col++) {
    double piv, maxv = fabs(M[col * n + col]);
    prow = col;
    for (i = col + 1; i < n; i++) {
      double v = fabs(M[i * n + col]);
      if (v > maxv) { maxv = v; prow = i; }
    }
    if (prow != col) {
      for (j = 0; j < n; j++) {
        double tmp = M[col * n + j];
        M[col * n + j] = M[prow * n + j];
        M[prow * n + j] = tmp;
      }
      { double tmp = b[col]; b[col] = b[prow]; b[prow] = tmp; }
    }
    piv = M[col * n + col];
    if (fabs(piv) < 1e-300) {
      M[col * n + col] = piv = ridge;
      degenerate = 1;
    }
    for (i = col + 1; i < n; i++) {
      double f = M[i * n + col] / piv;
      if (f == 0)
        continue;
      for (j = col; j < n; j++)
        M[i * n + j] -= f * M[col * n + j];
      b[i] -= f * b[col];
    }
  }
  /* back substitution, then undo the diagonal scaling */
  for (i = n - 1; i >= 0; i--) {
    double s = b[i];
    for (j = i + 1; j < n; j++)
      s -= M[i * n + j] * x[j];
    x[i] = s / M[i * n + i];
  }
  for (i = 0; i < n; i++)
    x[i] /= d[i];

  free(d);
  return degenerate;
}

/* ------------------------------------------------------------------ */
/* Read a 3-D field map (columns x,y,z in m; Bx,By,Bz in T).           */
void readFieldMap(char *fieldMapFile, FIELD_MAP *fmData) {
  SDDS_DATASET SDDSin;

  if (!SDDS_InitializeInputFromSearchPath(&SDDSin, fieldMapFile))
    SDDS_Bomb("unable to read field input file");
  if (SDDS_CheckColumn(&SDDSin, "Bx", "T", SDDS_ANY_NUMERIC_TYPE, stderr) != SDDS_CHECK_OKAY ||
      SDDS_CheckColumn(&SDDSin, "By", "T", SDDS_ANY_NUMERIC_TYPE, stderr) != SDDS_CHECK_OKAY ||
      SDDS_CheckColumn(&SDDSin, "Bz", "T", SDDS_ANY_NUMERIC_TYPE, stderr) != SDDS_CHECK_OKAY)
    SDDS_Bomb("Didn't find required field columns Bx, By, Bz in T");
  if (SDDS_CheckColumn(&SDDSin, "x", "m", SDDS_ANY_NUMERIC_TYPE, stderr) != SDDS_CHECK_OKAY ||
      SDDS_CheckColumn(&SDDSin, "y", "m", SDDS_ANY_NUMERIC_TYPE, stderr) != SDDS_CHECK_OKAY ||
      SDDS_CheckColumn(&SDDSin, "z", "m", SDDS_ANY_NUMERIC_TYPE, stderr) != SDDS_CHECK_OKAY)
    SDDS_Bomb("Didn't find required coordinate columns x, y, z in m");
  if (SDDS_ReadPage(&SDDSin) <= 0 ||
      !(fmData->x = SDDS_GetColumnInDoubles(&SDDSin, "x")) ||
      !(fmData->y = SDDS_GetColumnInDoubles(&SDDSin, "y")) ||
      !(fmData->z = SDDS_GetColumnInDoubles(&SDDSin, "z")) ||
      !(fmData->Bx = SDDS_GetColumnInDoubles(&SDDSin, "Bx")) ||
      !(fmData->By = SDDS_GetColumnInDoubles(&SDDSin, "By")) ||
      !(fmData->Bz = SDDS_GetColumnInDoubles(&SDDSin, "Bz")))
    SDDS_Bomb("unable to get data from field input file");
  if (!(fmData->n = SDDS_CountRowsOfInterest(&SDDSin)) || fmData->n < 1)
    SDDS_Bomb("field map file has insufficient data");
  SDDS_Terminate(&SDDSin);
}
