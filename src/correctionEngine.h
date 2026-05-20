/* file: correctionEngine.h
 * purpose: shared linear-response correction engine used by correct_coupling
 *          and correct_lattice (and any future LOCO-style correction command).
 *
 * The engine factors out the parts that don't depend on the specific
 * observable being corrected: knob/BPM collection, element lookup by
 * (name, occurrence), one-sided-difference response-matrix construction,
 * MPI assembly, SVD pseudo-inversion, and uniform strength-limit clamping.
 *
 * What the engine does NOT do (because they differ between commands):
 *   - the per-iteration loop with its SDDS output files;
 *   - parsing of command-specific namelists;
 *   - selection and weighting of the residual vector.
 * Each calling command keeps that logic in its own .c file.
 */

#ifndef CORRECTION_ENGINE_H
#define CORRECTION_ENGINE_H

/* Caller must include "track.h" before this header (for ELEMENT_LIST,
 * LINE_LIST, RUN). track.h has no include guard, so we deliberately don't
 * include it here -- that would cause duplicate-definition errors on
 * unguarded transitively-included headers (e.g. chbook.h). */

/* A correction knob: an element parameter that the engine may adjust. */
typedef struct {
  ELEMENT_LIST *elem;
  long paramIndex;       /* index into entity_description[type].parameter[] */
  double *valuePtr;      /* live pointer into elem->p_elem */
  double initialValue;   /* parameter value at the start of the correction */
} LRC_Knob;

/* An observation point. */
typedef struct {
  ELEMENT_LIST *elem;
} LRC_Bpm;

/* Observable reader. Writes nObs doubles per BPM into obs (length nObs*nBpm)
 * using the row-major layout
 *     obs[i_obs + nObs*i_bpm]
 * so that each BPM's observables sit contiguously. The reader is responsible
 * for whatever measurement noise simulation the calling command wants; the
 * engine itself never adds noise. The opaque ctx pointer is passed through
 * unchanged. */
typedef void (*LRC_ReaderFn)(LRC_Bpm *bpms, long nBpm, long nObs,
                             double *obs, void *ctx);

/* Free a list of patterns allocated by addPatterns(). */
void LRC_freePatternList(char ***patterns, long *n);

/* Walk the beamline; pick every element whose name matches any of the
 * namePatterns (or all elements if nNamePatterns==0) AND whose type matches
 * any of the typePatterns, AND that exposes a real-valued parameter named
 * `item`. Returns the count; allocates *knobs. */
long LRC_collectKnobs(LINE_LIST *beamline,
                      char **namePatterns, long nNamePatterns,
                      char **typePatterns, long nTypePatterns,
                      char *item, LRC_Knob **knobs);

/* Walk the beamline; pick every element whose name/type matches the lists.
 * Returns the count; allocates *bpms. */
long LRC_collectBpms(LINE_LIST *beamline,
                     char **namePatterns, long nNamePatterns,
                     char **typePatterns, long nTypePatterns,
                     LRC_Bpm **bpms);

/* Linear scan for an element with the given name+occurrence. NULL if absent. */
ELEMENT_LIST *LRC_findElementByNameOccurence(LINE_LIST *beamline,
                                             char *name, long occurence);

/* Force a recompute of the periodic twiss (and of the element matrix that
 * may have changed). Calls update_twiss_parameters() and warns if unstable. */
void LRC_retwiss(RUN *run, LINE_LIST *beamline, ELEMENT_LIST *changed);

/* Build the (nObs*nBpm) x nKnob response matrix R by one-sided finite
 * difference: R[i_obs + nObs*i_bpm][j] = (obs_pert - obs_base)/pert.
 *
 * R must be pre-allocated by the caller (e.g. via czarray_2d) with the
 * stated dimensions. Each knob's perturbation is applied, the twiss is
 * recomputed, the reader is invoked, the knob is restored, and twiss is
 * recomputed back. After the call R is populated on the master processor;
 * its state on non-master processors is undefined.
 *
 * The reader is called both for the baseline measurement and for each
 * perturbed measurement, so simulated measurement noise (if any) is applied
 * naturally in both. */
void LRC_buildResponseMatrix(RUN *run, LINE_LIST *beamline,
                             LRC_Knob *knobs, long nKnob,
                             LRC_Bpm *bpms, long nBpm, long nObs,
                             LRC_ReaderFn reader, void *ctx,
                             double perturbation,
                             double **R);

/* SVD pseudo-inverse: dK = -pinv(R) . y, with R (nRows x nCols), y (nRows),
 * dK (nCols).  Singular values are dropped if  s_i/s_max < svdThreshold;
 * additionally, if nKeep > 0, only the top nKeep are retained.
 * minSV/maxSV/nUsed report the retained singular value set. */
void LRC_svdSolve(double **R, long nRows, long nCols,
                  double *y, double *dK,
                  double svdThreshold, long nKeep,
                  double *minSV, double *maxSV, long *nUsed);

/* Largest uniform scale factor s in [0,1] such that |K[j] + s*dK[j]| <=
 * strengthLimit for every knob. Returns 1.0 when strengthLimit <= 0
 * (no limit). Returns 0 with a warning when no progress is possible. */
double LRC_clampStepToLimit(LRC_Knob *knobs, double *dK, long n, double strengthLimit);

#endif /* CORRECTION_ENGINE_H */
