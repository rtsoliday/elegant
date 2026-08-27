/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution.
\*************************************************************************/

/* file: coupledBeamAnalysis.c
 * purpose: coupled (eigen-)emittance and coupled lattice-function analysis of a
 *          beam (sigma) matrix, using the Sigma.J (Wolski invariant) method.
 *
 * Factored out of elegantTools/sddsanalyzebeam.c into the oagphy library so the
 * same routine can be used by elegant itself for analysis of beam distributions.
 *
 * Michael Borland, 2026
 */
#include "mdb.h"
#include "oagphy.h"

#ifdef USE_GSL
#  include "gsl/gsl_math.h"
#  include "gsl/gsl_complex.h"
#  include "gsl/gsl_complex_math.h"
#  include "gsl/gsl_vector.h"
#  include "gsl/gsl_matrix.h"
#  include "gsl/gsl_eigen.h"
#endif

/* Compute the coupled (eigen-)emittances and coupled lattice functions of the
   full 6x6 beam matrix.

   The analysis is carried out in the same phase space as elegant's moments_output,
   i.e. coordinates (x, x', y, y', s, delta) with the longitudinal coordinate
   s=beta*c*t in meters.  The caller passes sLongScale=beta*c so that the incoming
   matrix S (whose row/column 4 are in t, seconds) is rescaled to s on entry; this
   makes the third (longitudinal) eigen-emittance come out in meters, directly
   comparable to moments_output's e3.  A caller that already works in s (e.g. elegant
   internally) should pass sLongScale=1.

   Let S6 be the (rescaled) 6x6 beam matrix and J6 the symplectic form built from
   2x2 blocks [[0,1],[-1,0]] on the (x,x'), (y,y'), and (s,delta) pairs.  The
   eigenvalues of G = S6.J6 are +/- i*e_k for k=1,2,3, where the e_k are the three
   eigen-emittances (invariant under symplectic transport; they reduce to the
   ordinary x, y, and longitudinal emittances when the planes are uncoupled).
   Writing an eigenvector for eigenvalue +i*e_k as v_k = p_k + i q_k, normalized so
   that v_k^dagger J v_k = 2i (the symplectic normalization used by
   NormalizeEigenvectors() in moments.c), the per-mode matrix A_k = p_k p_k^T +
   q_k q_k^T satisfies S6 = sum_k e_k A_k, and its entries are the coupled lattice
   functions (A_k[0][0]=beta_x,k, A_k[0][1]=-alpha_x,k, ..., A_k[4][4]=beta_s,k,
   A_k[4][5]=-alpha_s,k, A_k[5][5]=gamma_s,k, etc.), following the same conventions
   used by elegant's coupled_twiss output.  A_k is invariant under the (arbitrary)
   complex phase of the eigenvector, so the result is well defined.

   The three modes are labeled so that index 0 is x-dominated, 1 is y-dominated, and
   2 is longitudinal-dominated, chosen as the assignment of modes to planes that
   maximizes the total in-plane weight (as SortEigenvalues does in moments_output).

   Relation to routines in elegant proper:  the ALGORITHM here differs from
   elegant's own eigenmode routines because the caller has only a beam (one sigma
   matrix), not a one-turn map.  elegant's computeNaturalEmittances() (src/moments.c)
   and run_coupled_twiss_output()/GetAMatrix() (src/coupled_twiss.c) eigen-decompose
   the transfer/revolution MATRIX (LAPACK dgeev_) and then project or rotate the
   sigma matrix onto those map eigenvectors.  Here we instead take the eigenvectors
   of G = S6.J6, i.e. of the BEAM matrix itself (Wolski invariants; see
   A. Wolski, PRST-AB 9, 024001 (2006)), which needs no map.  What is emulated from
   those routines are the CONVENTIONS, so results are directly comparable:
     - the per-mode A_k = (real)(real)^T + (imag)(imag)^T construction and the
       beta/alpha/gamma and cross-term (A_xy, A_xpy, A_xyp, A_xpyp) readout follow
       GetAMatrix() and setup_coupled_twiss_output() in src/coupled_twiss.c;
     - the block-diagonal symplectic form and v^dagger J v normalization follow
       NormalizeEigenvectors() in src/moments.c;
     - the (x,y,z) mode ordering by dominant in-plane weight follows
       SortEigenvalues()/GetMaxIndex() (src/moments.c, src/coupled_twiss.c);
     - the coordinates and units, incl. the longitudinal s=beta*c*t (m) rescaling
       above, match computeNaturalEmittances() in src/moments.c so that e3/beta_s,k
       coincide with moments_output.
   Because the map-based and beam-based eigen-analyses use different inputs, the
   transverse e1/e2 partition can differ near a linear coupling resonance even though
   both are correct; the Wolski (Sigma.J) eigen-emittances reported here are the
   location-independent invariants.

   Returns 1 on success (result filled), 0 on failure (result left unchanged). */
long ComputeCoupledParameters(COUPLED_RESULTS *result, double S[6][6], double sLongScale) {
#ifdef USE_GSL
  double Ss[6][6];
  gsl_matrix *G;
  gsl_vector_complex *eval;
  gsl_matrix_complex *evec;
  gsl_eigen_nonsymmv_workspace *w;
  int i, j, k, m, nmodes;
  double A[3][6][6], emit[3], weight[3][3];
  int order[3];
  /* the six permutations of (0,1,2): mode assigned to each plane (x,y,s) */
  static const int perm[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
  double bestScore;
  int best;

  /* Rescale the longitudinal coordinate t (index 4) to s=beta*c*t so the analysis
     is done in the (x,x',y,y',s,delta) coordinates used by moments_output. */
  for (i = 0; i < 6; i++)
    for (j = 0; j < 6; j++)
      Ss[i][j] = S[i][j] * (i == 4 ? sLongScale : 1.0) * (j == 4 ? sLongScale : 1.0);

  /* G = S6 . J6 :  columns are -Ss[:,1], Ss[:,0], -Ss[:,3], Ss[:,2], -Ss[:,5], Ss[:,4] */
  G = gsl_matrix_alloc(6, 6);
  for (i = 0; i < 6; i++) {
    gsl_matrix_set(G, i, 0, -Ss[i][1]);
    gsl_matrix_set(G, i, 1, Ss[i][0]);
    gsl_matrix_set(G, i, 2, -Ss[i][3]);
    gsl_matrix_set(G, i, 3, Ss[i][2]);
    gsl_matrix_set(G, i, 4, -Ss[i][5]);
    gsl_matrix_set(G, i, 5, Ss[i][4]);
  }
  eval = gsl_vector_complex_alloc(6);
  evec = gsl_matrix_complex_alloc(6, 6);
  w = gsl_eigen_nonsymmv_alloc(6);
  if (gsl_eigen_nonsymmv(G, eval, evec, w) != 0) {
    gsl_eigen_nonsymmv_free(w);
    gsl_matrix_free(G);
    gsl_vector_complex_free(eval);
    gsl_matrix_complex_free(evec);
    return 0;
  }
  gsl_eigen_nonsymmv_free(w);

  nmodes = 0;
  for (k = 0; k < 6 && nmodes < 3; k++) {
    gsl_complex lambda = gsl_vector_complex_get(eval, k);
    double li = GSL_IMAG(lambda);
    double p[6], q[6], sympNorm, scale;
    if (li <= 0)
      continue; /* use only the +i*e_k member of each conjugate pair */
    for (i = 0; i < 6; i++) {
      gsl_complex z = gsl_matrix_complex_get(evec, i, k);
      p[i] = GSL_REAL(z);
      q[i] = GSL_IMAG(z);
    }
    /* v^dagger J v = i*sympNorm (purely imaginary); Im( conj(v_i) (Jv)_i ) = p_i Jvi - q_i Jvr,
       with Jv = (v1, -v0, v3, -v2, v5, -v4). */
    sympNorm = 0;
    for (i = 0; i < 6; i++) {
      double Jvr, Jvi;
      switch (i) {
      case 0:  Jvr = p[1];  Jvi = q[1];  break;
      case 1:  Jvr = -p[0]; Jvi = -q[0]; break;
      case 2:  Jvr = p[3];  Jvi = q[3];  break;
      case 3:  Jvr = -p[2]; Jvi = -q[2]; break;
      case 4:  Jvr = p[5];  Jvi = q[5];  break;
      default: Jvr = -p[4]; Jvi = -q[4]; break;
      }
      sympNorm += p[i] * Jvi - q[i] * Jvr;
    }
    if (sympNorm <= 0)
      continue;
    scale = sqrt(2.0 / sympNorm);
    for (i = 0; i < 6; i++) {
      p[i] *= scale;
      q[i] *= scale;
    }
    emit[nmodes] = li;
    for (i = 0; i < 6; i++)
      for (j = 0; j < 6; j++)
        A[nmodes][i][j] = p[i] * p[j] + q[i] * q[j];
    /* in-plane weight of this mode in plane 0=x, 1=y, 2=longitudinal */
    weight[nmodes][0] = A[nmodes][0][0] + A[nmodes][1][1];
    weight[nmodes][1] = A[nmodes][2][2] + A[nmodes][3][3];
    weight[nmodes][2] = A[nmodes][4][4] + A[nmodes][5][5];
    nmodes++;
  }

  gsl_matrix_free(G);
  gsl_vector_complex_free(eval);
  gsl_matrix_complex_free(evec);

  if (nmodes < 3)
    return 0;

  /* order modes so index 0 is x-dominated, 1 is y-dominated, 2 is longitudinal-dominated:
     pick the mode->plane assignment maximizing the total in-plane weight. */
  best = 0;
  bestScore = -1;
  for (m = 0; m < 6; m++) {
    double score = weight[perm[m][0]][0] + weight[perm[m][1]][1] + weight[perm[m][2]][2];
    if (score > bestScore) {
      bestScore = score;
      best = m;
    }
  }
  for (m = 0; m < 3; m++)
    order[m] = perm[best][m];

  for (m = 0; m < 3; m++) {
    k = order[m];
    result->emit[m] = emit[k];
    result->betax[m] = A[k][0][0];
    result->alphax[m] = -A[k][0][1];
    result->gammax[m] = A[k][1][1];
    result->betay[m] = A[k][2][2];
    result->alphay[m] = -A[k][2][3];
    result->gammay[m] = A[k][3][3];
    result->betaz[m] = A[k][4][4];
    result->alphaz[m] = -A[k][4][5];
    result->gammaz[m] = A[k][5][5];
    result->A_xy[m] = A[k][0][2];
    result->A_xpy[m] = A[k][1][2];
    result->A_xyp[m] = A[k][0][3];
    result->A_xpyp[m] = A[k][1][3];
  }
  return 1;
#else
  (void)result;
  (void)S;
  (void)sLongScale;
  return 0;
#endif
}
