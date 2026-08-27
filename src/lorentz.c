/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* routine: lorentz()
 * purpose: integrate lorentz equation for electrons in a general
 *          magnetic field.
 *
 * The equations of motion are
 *        -
 *      d w                   -   - -
 *      --- = (1+f)/(1+dp)*S0 w x F(q)
 *      d s
 *
 *        -
 *      d q    -
 *      --- =  w
 *      d s
 *
 * where f is the fractional strength error, dp = (p-po)/po,
 * F is the normalized magnetic field, s is the distance, 
 * q is the coordinate, and w is the rate of change of q with
 * distance.  I use q = (z, x, y).
 *
 * S0 is a strength parameter.  For a bending magnet, it is 1/rho0,
 * rho0 being the central bending radius.   The charge and central momentum
 * are hidden in S0.  The equation for S0 is
 *  
 *    S0 = Q*B0/(m*c*beta0*gamma0)   (MKS)
 *
 * B0*(1+f)*F(q) gives the magnetic field in Tesla.
 *
 * If synchrotron radiation is included, then it is computed by integrating
 *
 *     d (dp)           e^2                                     -
 *     -----  = - --------------- (BETAo*GAMMAo)^3 (1+dp)^4 * (dw/ds)^2
 *      ds        6 Pi EPSo m c^2
 *   
 * This alters dp in the above equation, which is not strictly correct, but
 * should be okay for small energy losses.
 *
 * The routines in this directory support these elegant elements:
 * 1. NIBENDs --- Numerically Integrated BENDs
 *                This is a flat-field magnet with extended-edge effects.
 * 2. NISEPTs --- Numerically Integrated SEPTum-type magnet.
 *                This is a straight, rectangular magnet with an optional 
 *                Cartesian gradient. The entry point of the beam is taken 
 *                to define X=0, and the field is By = Bo*(1 + b1*X), Bx = Bo*b1*Y.  
 *                The entrance edge angle is theta0 (the bending angle).
 *                The coordinates of the entry point are taken to be
 *                     q0=-rhoi*sin(theta0), q1=rhoi*cos(theta0)
 *                Where rhoi is the ideal bending radius (ignoring b1).
 *                The exit plane is
 *                     q0=0
 *                Bo and g are scaled together to get the right bend angle.
 * 3. BMAPXYs --- numerically integrated B-field MAPs in XY plane.
 *                A 2D map of (Fx, Fy) is given as a function of (x, y).
 *                If the field map is, say, (Fx=y, Fy=x) near x=y=0,
 *                then the strength parameter (STRENGTH) is equivalent
 *                to K1.
 *                There's not much more to it. 
 * 4. BMAPXYZ --- Numerically-integrated B-field maps in XYZ.
 *                A 3D map for (Fx, Fy, Fz) is given as a function of (x, y, Z).
 *
 * Michael Borland, 1991, 1993, 1997, 2004
 */
#include "mdb.h"
#include "match_string.h"
#include "track.h"

typedef struct {
  char *filename;
  BMAPXYZ_DATA *data;
  double fieldLength;
  short singlePrecision;
} STORED_BMAPXYZ_DATA;
static STORED_BMAPXYZ_DATA *storedBmapxyzData = NULL;
static long iStoredBmapxyzData = -1;
static long nStoredBmapxyzData = 0;

void lorentz_setup(void *field, long field_type, double **part, int64_t np, double Po);
void lorentz_terminate(void *field, long field_type, double **part, int64_t np, double Po);
void select_lorentz_integrator(char *desired_method);
long do_lorentz_integration(double *coord, void *field);
double nibend_trajectory_error_offset(double offsetp);
double nibend_trajectory_error_fse(double fsep);
double nisept_trajectory_error(double fsep);
static double *traj_err_final_coord;

void engeFunction(double *Fy, double *Fz, double z, double y, double D, double a1, double a2, double a3, long order);
void writeEngeFunctionToFile(char *filename);

/* pointers to functions to be used in integration.  Set by lorentz_setup()
 * and select_lorentz_integrator() */
static long (*integrator)();
static void (*coord_transform)(double *q, double *coord, void *field, long which_end);
/* function to indicate exit from magnet */
static double (*exit_function)(double *qp, double *q, double s);
/* function to calculate derivatives */
static void (*deriv_function)(double *qp, double *q, double s);

/* parameters of element needed for integration--set by lorentz_setup */
static double S0, P0, one_plus_fse, rad_coef;
static double offset, s_offset;
static double x_correction;
static double fse_opt;

/* q0 = q1*entr_slope + entr_intercept defines entrance plane of fringe at entrance: */
static double entr_slope, entr_intercept;
/* q0 = q1*entr_slope + fentr_intercept defines end of fringe region at entrance :*/
static double fentr_intercept;
/* q0 = q1*entr_slope + rentr_intercept defines the reference plane at the entrance: */
static double rentr_intercept;

/* q0 = q1*exit_slope + exit_intercept defines exit plane (end of fringe at exit): */
static double exit_slope, exit_intercept;
/* q0 = q1*exit_slope + fexit_intercept defines start of fringe region at exit:*/
static double fexit_intercept;
/* q0 = q1*exit_slope + rexit_intercept defines the reference plane at the exit: */
static double rexit_intercept;

static double central_length, tolerance;
static long lastParticleID;

/* prototypes and other information for NIBEND element */
void nibend_coord_transform(double *q, double *coord, void *field, long which_end);
double nibend_exit_function(double *qp, double *q, double s);
void nibend_deriv_function(double *qp, double *q, double s);

static long fringe_code;
#define HARD_EDGE_MODEL 0
#define LINEAR_MODEL 1
#define CUBIC_MODEL 2
#define TANH_MODEL 3
#define QUINTIC_MODEL 4
#define ENGE1_MODEL 5
#define ENGE3_MODEL 6
#define ENGE5_MODEL 7
#define N_FRINGE_MODELS 8
static char *fringe_model[N_FRINGE_MODELS] = {
  "hard-edge",
  "linear",
  "cubic-spline",
  "tanh",
  "quintic",
  "enge1",
  "enge3",
  "enge5",
};

/* parameters for the fringe fields: */
static double Fa, Fb, Fc, flen;
static double engeD, engeCoef[3];
static long engeOrder;
static double cos_alpha1, cos_alpha2;
static double sin_alpha1, sin_alpha2;
/* For NIBENDs, there are various field models available:
 * Let f==flen be the extent of the fringe field, which starts at z=0.
 * Let K = Integral[-inf->inf, Fy*(1-Fy) dz]/g (K.Brown's fringe-field
 *    integral).  g is the full gap.  Recommended values for f are
 *    given below that get the correct linear focusing.
 *
 * 1. Linear fringe field:
 *        Fy = z*Fa
 *        Fz = y*Fa
 *    Fa = 1/f
 *    f  = 6*K*g
 *
 * 2. Cubic-spline fringe field:
 *        Fy = Fa*z^2 + Fb*z^3 + y^2*(-Fa - 3*Fb*z)   
 *        Fz = (2*Fa*z + 3*Fb*z^2)*y 
 *    where Fa = 3/f^2 and Fb = -2/f^3
 *    f = 70*K*g/9
 *
 * 3. Tanh-like fringe field:
 *        Fy = (1+tanh(Fa*z))/2 + (Fa*y*sech(Fa*z))^2*tanh(Fa*z)/2
 *              + (Fa*y*sech(Fa*z))^4*sech(Fa*z)(11 sinh(Fa*z) - sinh(3*Fa*z))/24
 *        Fz = Fa*y*sech(Fa*z)^2/2 + (Fa*y*sech(Fa*z))^3*sech(Fa*z)*(2-cosh(2*Fa*z))/6
 *              + (Fa*y*sech(Fa*z))^5*sech(Fa*z)*(33-26 cosh(2*Fa*z) + cosh(4*Fa*z))/120
 *    Fa = 1/(2*K*g)    
 *    f  = 1/Fa = 2*K*g.  The actual extent of the fringe field is undefined, but here
 *    I take it to be |z|<4*f
 *
 * 4. Quintic-spline fringe field, to third order in y:
 *        Fy =     (Fa*z^3 + Fb*z^4 + Fc*z^5) + 
 *             y^2*z*(3*Fa + 6*Fb*z + 10*Fc*z^2)
 *        Fz = y  *(3*Fa*z^2 + 4*Fb*z^3 + 5*Fc*z^4) + 
 *             y^3*(-Fa - 4*Fb*z - 10*Fc*z^2)
 *    where Fa = 10/f^3, Fb = -15/f^4, Fc = 6/f^5
 *    f = 231*K*g/25
 *
 * 5. Enge model with 3 coefficients:
 *        F0 = 1/(1 + exp(a1 + a2*z/D + a3*(z/D)^2))
 *        Fy = F0 - 1/2*y^2*F'' + 1/24*y^4*F''''
 *        Fz = y*F' - 1/6*y^3*F''' + 1/120*y^5*F'''''
 * 
 * alpha[i] is theta/2-beta[i], where beta[i] is the 
 * edge angle for the ith edge.
 */

/* prototypes for NISEPT element */
void nisept_coord_transform(double *q, double *coord, void *field, long which_end);
double nisept_exit_function(double *qp, double *q, double s);
void nisept_deriv_function(double *qp, double *q, double s);

/* prototypes for BMAPXY element */
void bmapxy_coord_transform(double *q, double *coord, void *field, long which_end);
double bmapxy_exit_function(double *qp, double *q, double s);
void bmapxy_deriv_function(double *qp, double *q, double s);
void bmapxy_field_setup(BMAPXY *bmapxy);

/* prototypes for BMAPXYZ element */
double bmapxyz_exit_function(double *qp, double *q, double s);
void bmapxyz_deriv_function(double *qp, double *q, double s);
void bmapxyz_coord_transform(double *q, double *coord, void *field, long which_end);
void bmapxyz_field_setup(BMAPXYZ *bmapxyz);
long interpolate_bmapxyz(double *F0, double *F1, double *F2, BMAPXYZ *bmapxyz, double x, double y, double z);

long method_code = 0;
#define RUNGE_KUTTA 0
#define BULIRSCH_STOER 1
#define MODIFIED_MIDPOINT 2
#define TWO_PASS_MODIFIED_MIDPOINT 3
#define LEAP_FROG 4
#define NA_RUNGE_KUTTA 5
#define N_METHODS 6
static char *method[N_METHODS] = {
  "runge-kutta", "bulirsch-stoer", "modified-midpoint", "two-pass modified-midpoint",
  "leap-frog", "non-adaptive runge-kutta"};
void lorentz_leap_frog(double *Qf, double *Qi, double s, long n_steps, void (*derivs)(double *dQds, double *Q, double sd));

/* minimum number of steps to take */
#define N_INTERIOR_STEPS 100

#ifdef DEBUG
static FILE *fp_field = NULL;
static long field_output_on = 1;
#endif

static long n_lorentz_calls = 0;
static long n_deriv_calls = 0;
static long n_particles_done = 0;
static double length_mult_sum = 0;
static long n_invalid_particles = 0;

/* lost particle coordinate buffer.
                      0   1   2   3       4       5       6     7      8   9 10  11
   In order, we have (q0, q1, q2, dq0/ds, dq1/ds, dq2/ds, path, delta, s,  X, Z, theta 
 */
static double lostParticleCoordinate[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static long isLost = 0;

static MULT_APERTURE_DATA apertureData;

void lorentz_report(void) {
  if (n_lorentz_calls) {
    printf("\nStatistics for numerical integrations by lorentz() module:\n");
    printf("    %ld calls to main module\n    %ld derivative evaluations\n    %ld particles integrated\n",
           n_lorentz_calls, n_deriv_calls, n_particles_done);
    if (length_mult_sum && n_particles_done && !integrator)
      printf("    average length multiplier for non-adaptive integration: %e\n", length_mult_sum / n_particles_done);
    printf("    %ld calls to derivative module had invalid particles\n",
           n_invalid_particles);
    fflush(stdout);
  }
  n_lorentz_calls = n_deriv_calls = n_particles_done = 0;
  length_mult_sum = 0;
}

long lorentz(
  double **part,
  int64_t n_part,
  void *field,
  long field_type,
  double P_central,
  double **accepted,
  MAXAMP *maxamp0, 
  APCONTOUR *apcontour0,
  APERTURE_DATA *apData0
             ) {
  double *coord;
  int64_t i_part, i_top;
  long mod, count;
  TRACKING_CONTEXT context;
  short verbosity = 0;
  
  if (!n_part)
    return (0);

  getTrackingContext(&context);
  if (field_type==T_BMAPXYZ) {
    BMAPXYZ *bmxyz;
    bmxyz = (BMAPXYZ *)field;
    verbosity = bmxyz->verbosity;
    if (bmxyz->apContourElement && strlen(bmxyz->apContourElement))
      setupMultApertureData(&apertureData, 0.0, &bmxyz->apContour, maxamp0, apData0, NULL, context.zStart, context.element);
    else
      setupMultApertureData(&apertureData, 0.0, apcontour0, maxamp0, apData0, NULL, context.zStart, context.element);
  }
  
  if (verbosity) {
#if USE_MPI
    if (myid==1)
      dup2(fdStdout, fileno(stdout));
#endif
  }

  log_entry("lorentz");

  n_lorentz_calls++;

#ifdef DEBUG
  if (!fp_field) {
#  if USE_MPI
    char buffer[100];
    snprintf(buffer, 100, "fields%03d.sdds", myid);
    /* fp_field = fopen_e(buffer, "w", 0); */
#  else
    fp_field = fopen_e("fields.sdds", "w", 0);
#  endif
    if (fp_field) {
      fprintf(fp_field, "SDDS1\n&column name=q0 type=double units=m &end\n");
      fprintf(fp_field, "&column name=q1 type=double units=m &end\n");
      fprintf(fp_field, "&column name=q2 type=double units=m &end\n");
      fprintf(fp_field, "&column name=z type=double units=m &end\n");
      fprintf(fp_field, "&column name=s type=double units=m &end\n");
      fprintf(fp_field, "&column name=F0 type=double  &end\n");
      fprintf(fp_field, "&column name=F1 type=double  &end\n");
      fprintf(fp_field, "&column name=F2 type=double  &end\n");
      fprintf(fp_field, "&data mode=ascii no_row_counts=1 &end\n");
    }
  }
  field_output_on = 0;
#endif

  lorentz_setup(field, field_type, part, n_part, P_central);

  if (verbosity) {
#if USE_MPI
    /* Open file descriptor for rest of processors */
    if (myid>1) 
      dup2(fdStdout, fileno(stdout));
#endif
  }
#ifdef DEBUG
  printf("lorentz_setup finished\n");
  fflush(stdout);
#endif

  if (field_type == T_BMAPXYZ) {
    BMAPXYZ *bmxyz;
    bmxyz = (BMAPXYZ *)field;
    verbosity = bmxyz->verbosity;
    if ((bmxyz->fieldLength > 0 && bmxyz->length != bmxyz->fieldLength) && !bmxyz->injectAtZero)
      exactDrift(part, n_part, (bmxyz->length - bmxyz->fieldLength) / 2);
  }

#ifdef DEBUG
  field_output_on = fp_field ? 1 : 0;
#endif
  i_top = n_part - 1;

  if (verbosity>99)
    mod = n_part/100;
  else
    mod = n_part/10;
  if (mod<=0)
    mod = 1;
  
  count = 0;
  for (i_part = 0; i_part <= i_top; i_part++) {
    count++;
    coord = part[i_part];
    lastParticleID = part[i_part][particleIDIndex];
    if (verbosity>9 && count%mod==0) {
#if USE_MPI
      printf("Worker %d tracking particle %ld of %ld\n", myid, count, n_part);
#else
      printf("Tracking particle %ld of %ld\n", count, n_part);
#endif
      fflush(stdout);
    }
    if (!do_lorentz_integration(coord, field)) {
      if (i_part != i_top) {
        swapParticles(part[i_part], part[i_top]);
        if (accepted)
          swapParticles(accepted[i_part], accepted[i_top]);
      }
      part[i_top][5] = P_central * (1 + part[i_top][5]);
      i_top--;
      i_part--;
    }

    if (field_type == T_BMAPXYZ) {
      BMAPXYZ *bmxyz;
      bmxyz = (BMAPXYZ *)field;
      if (bmxyz->SDDSpo && !SDDS_WritePage(bmxyz->SDDSpo)) {
        SDDS_SetError("Problem writing page in particle output file for BMXYZ");
        SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
      }
      bmxyz->poRow = 0;
    }
  }

  lorentz_terminate(field, field_type, part, n_part, P_central);

  if (field_type == T_BMAPXYZ) {
    BMAPXYZ *bmxyz;
    bmxyz = (BMAPXYZ *)field;
    if ((bmxyz->fieldLength > 0 && bmxyz->length != bmxyz->fieldLength) && !bmxyz->injectAtZero)
      exactDrift(part, i_top + 1, (bmxyz->length - bmxyz->fieldLength) / 2);
    if (bmxyz->discardMap && iStoredBmapxyzData>=0) {
      STORED_BMAPXYZ_DATA mapData;
      long im;
      mapData = storedBmapxyzData[iStoredBmapxyzData];
#if USE_MPI
      if (myid==1) {
#endif
        printf("Discarding field map data from file %s\n", mapData.filename);
        fflush(stdout);
#if USE_MPI
      }
#endif
      free(mapData.filename);
      if (mapData.data->singlePrecision) {
        free(mapData.data->Fx1);
        free(mapData.data->Fy1);
        free(mapData.data->Fz1);
      } else {
        free(mapData.data->Fx);
        free(mapData.data->Fy);
        free(mapData.data->Fz);
      }
      free(mapData.data);
      for (im=iStoredBmapxyzData+1; im<nStoredBmapxyzData; im++)
        storedBmapxyzData[im-1] = storedBmapxyzData[im];
      nStoredBmapxyzData--;
      bmxyz->data = NULL;
    }
  }

  log_exit("lorentz");

#if USE_MPI
  if (verbosity && myid!=0) {
#if defined(_WIN32)
    freopen("NUL", "w", stdout);
#else
    if (!freopen("/dev/null", "w", stdout)) {
      perror("freopen failed");
      exit(EXIT_FAILURE);
    }
#endif
  }
#endif

  return (i_top + 1);
#ifdef DEBUG
  printf("Exiting lorentz()\n");
  fflush(stdout);
#endif
}

long do_lorentz_integration(double *coord, void *field) {
  static long n_eq = 8;
  static double exvalue;
  static double q[8], qout[8]; /* q0,q1,q2,w0,w1,w2,s,(p-po)/po */
  static double qp[8];
  static long misses[8], accmode[8];
  static double accuracy[8], tiny[8];
  double hrec, hmax, length_multiplier;
  double s_start, s_end, exit_toler;
  long int_return, i, n_steps;

  log_entry("do_lorentz_integration");

  n_particles_done++;

  (*coord_transform)(q, coord, field, -1);

  fill_long_array(accmode, 8, 2);
  fill_double_array(tiny, 8, 1e-16);
  fill_double_array(accuracy, 8, tolerance);
  fill_long_array(misses, 8, 0);
  isLost = 0;

  if (integrator != NULL) {
    /* use adaptive integration or another compatible routine */
    s_start = 0;
    s_end = central_length * 2;
    hmax = central_length / N_INTERIOR_STEPS;
    hrec = hmax / 10;
    if ((exit_toler = sqr(tolerance) * s_end) < central_length * 1e-14)
      exit_toler = central_length * 1e-14;
    if (integrator == rk_odeint3_na) {
      hrec = central_length * tolerance;
      int_return = rk_odeint3_na(q, deriv_function, n_eq, accuracy, accmode, tiny, misses,
                                 &s_start, s_end, exit_toler, hrec, hmax, &hrec, exit_function, exit_toler, NULL);
    } else
      int_return = (*integrator)(q, deriv_function, n_eq, accuracy, accmode, tiny, misses,
                                 &s_start, s_end, exit_toler, hrec, hmax, &hrec, exit_function, exit_toler);
    switch (int_return) {
    case DIFFEQ_ZERO_STEPSIZE:
    case DIFFEQ_CANT_TAKE_STEP:
    case DIFFEQ_OUTSIDE_INTERVAL:
    case DIFFEQ_XI_GT_XF:
      printf("Integration failure---may be program bug: %s\n", diffeq_result_description(int_return));
      for (i = 0; i < 6; i++)
        printf("%11.4e  ", coord[i]);
      fflush(stdout);
      exitElegant(1);
      break;
    case DIFFEQ_END_OF_INTERVAL:
      log_exit("do_lorentz_integration");
      printf("Integration reached end of interval\n");
      fflush(stdout);
      return (0);
      break;
    default:
      if ((exvalue = (*exit_function)(NULL, q, central_length)) > exit_toler) {
        char warningText[1024];
        snprintf(warningText, 1024, "Exit value of %e exceeds tolerance of %e--particle lost.", exvalue, exit_toler);
        printWarningForTracking("Problem in numerical integration.", warningText);
        if (!isLost) {
          isLost = 1;
          memcpy(&lostParticleCoordinate, &q, sizeof(q[0]) * 8);
        }
        return (0);
      }
      break;
    }
  } else {
    /* non-adpative integration */
    length_multiplier = 1 + tolerance;
    n_steps = central_length / tolerance + 0.5;
    do {
      s_start = 0;
      s_end = central_length * length_multiplier;
      (*deriv_function)(qp, q, s_start);
      switch (method_code) {
      case MODIFIED_MIDPOINT:
        mmid(q, qp, n_eq, s_start, s_end - s_start, n_steps, qout, deriv_function);
        break;
      case TWO_PASS_MODIFIED_MIDPOINT:
        mmid2(q, qp, n_eq, s_start, s_end - s_start, n_steps, qout, deriv_function);
        break;
      case LEAP_FROG:
        lorentz_leap_frog(qout, q, s_end - s_start, n_steps, deriv_function);
        break;
      default:
        printf("error: unknown non-adaptive integration method code: %ld\n", method_code);
        fflush(stdout);
        exitElegant(1);
        break;
      }
      if ((exvalue = (*exit_function)(NULL, qout, central_length)) > 0)
        length_multiplier += exvalue / central_length;
      else
        break;
    } while (1);
    length_mult_sum += length_multiplier;
    copy_doubles(q, qout, 8);
  }

  if (isLost)
    memcpy(q, lostParticleCoordinate, sizeof(q[0]) * 8);

  (*coord_transform)(q, coord, field, 1);

  if (isLost) {
    coord[4] = lostParticleCoordinate[8]; /* need z, not s */
    if (globalLossCoordOffset > 0)
      memcpy(coord + globalLossCoordOffset, lostParticleCoordinate + 9, 3 * sizeof(double));
  }

#ifdef DEBUG
  printf("length from integration routine: %le\n", s_start);
#endif

  log_exit("do_lorentz_integration");

  if (isLost)
    return 0;

  return (1);
}

static void *field_global;

void lorentz_setup(
  void *field,
  long field_type,
  double **part,
  int64_t np,
  double Po) {
  NIBEND *nibend;
  NISEPT *nisept;
  BMAPXY *bmapxy;
  BMAPXYZ *bmapxyz;
  double alpha, Kg;
  static double last_fse = 0;

  log_entry("lorentz_setup");
  field_global = field;
  Fa = Fb = 0;
  x_correction = s_offset = 0;
  P0 = Po;

  switch (field_type) {
  case T_NIBEND:
    nibend = (NIBEND *)field;
    exit_function = nibend_exit_function;
    deriv_function = nibend_deriv_function;
    coord_transform = nibend_coord_transform;
    select_lorentz_integrator(nibend->method);
    if ((fringe_code = match_string(nibend->model, fringe_model, N_FRINGE_MODELS, 0)) < 0)
      bombElegant("unknown fringe-field model for NIBEND", NULL);
    flen = 0;
    if ((Kg = 2 * nibend->hgap * nibend->fint)) {
      switch (fringe_code) {
      case HARD_EDGE_MODEL:
        Fa = Fb = 0;
        nibend->flen = flen = 0;
        break;
      case LINEAR_MODEL:
        flen = nibend->flen = 6 * Kg;
        Fa = 1 / nibend->flen;
        break;
      case CUBIC_MODEL:
        flen = nibend->flen = (70 * Kg) / 9.;
        Fa = 3 / sqr(nibend->flen);
        Fb = -2 / ipow3(nibend->flen);
        break;
      case TANH_MODEL:
        flen = nibend->flen = 2 * Kg;
        Fa = 1 / nibend->flen;
        flen = nibend->fp1 * nibend->flen;
        break;
      case QUINTIC_MODEL:
        flen = nibend->flen = (231 * Kg) / 25.;
        Fa = 10 / ipow3(nibend->flen);
        Fb = -15 / ipow4(nibend->flen);
        Fc = 6 / ipow5(nibend->flen);
        break;
      case ENGE1_MODEL:
      case ENGE3_MODEL:
      case ENGE5_MODEL:
        engeD = nibend->hgap * 2;
        engeOrder = fringe_code == ENGE1_MODEL ? 1 : (fringe_code == ENGE3_MODEL ? 3 : 5);
        if (!nibend->fp2) {
          /* determine a1, a2, and a3 to match FINT, HGAP, and angle */
          computeEngeCoefficients(engeCoef, fabs(nibend->length / nibend->angle),
                                  nibend->length, engeD, nibend->fint);
        } else {
          engeCoef[0] = nibend->fp2;
          engeCoef[1] = nibend->fp3;
          engeCoef[2] = nibend->fp4;
        }
        flen = nibend->flen = nibend->length;
        break;
      default:
        bombElegant("logic error in fringe field setup for NIBEND (lorentz_setup)", NULL);
        break;
      }
    } else {
      Fa = Fb = nibend->flen = 0;
      fringe_code = HARD_EDGE_MODEL;
    }
    nibend->angleSign = 1;
    if (nibend->angle < 0) {
      nibend->angle *= -1;
      nibend->e[nibend->e1Index] *= -1;
      nibend->e[nibend->e2Index] *= -1;
      nibend->angleSign = -1;
    }
    if (nibend->e[nibend->e1Index] != nibend->e[nibend->e2Index])
      printWarningForTracking("E1!=E2 for NIBEND, which may cause orbit distortions", NULL);
    nibend->rho0 = nibend->length / nibend->angle;
    S0 = 1 / nibend->rho0;
    central_length = nibend->length + flen;
    tolerance = nibend->accuracy;

    /* calculate slope and intercept for entrance plane */
    alpha = nibend->angle / 2 - nibend->e[nibend->e1Index];
    rentr_intercept = -nibend->rho0 * (sin(nibend->angle / 2) - cos(nibend->angle / 2) * tan(alpha));
    switch (nibend->fringePosition) {
    case -1: /* Fringe inside */
      fentr_intercept = rentr_intercept + flen / cos(alpha);
      entr_intercept = rentr_intercept;
      break;
    case 1: /* Fringe outside */
      fentr_intercept = rentr_intercept;
      entr_intercept = rentr_intercept - flen / cos(alpha);
      break;
    case 0: /* Fringe centered */
    default:
      fentr_intercept = rentr_intercept + flen / cos(alpha) / 2;
      entr_intercept = rentr_intercept - flen / cos(alpha) / 2;
      break;
    }

    if (alpha != PIo2)
      entr_slope = -tan(alpha);
    else
      entr_slope = DBL_MAX;
    cos_alpha1 = cos(alpha);
    sin_alpha1 = sin(alpha);

    /* calculate slope and intercept for exit plane */
    alpha = nibend->angle / 2 - nibend->e[nibend->e2Index];
    rexit_intercept = nibend->rho0 * (sin(nibend->angle / 2) - cos(nibend->angle / 2) * tan(alpha));
    switch (nibend->fringePosition) {
    case -1: /* Fringe inside */
      fexit_intercept = rexit_intercept - flen / cos(alpha);
      exit_intercept = rexit_intercept;
      break;
    case 1: /* Fringe outside */
      fexit_intercept = rexit_intercept;
      exit_intercept = rexit_intercept + flen / cos(alpha);
      break;
    case 0: /* Fringe centered */
    default:
      fexit_intercept = rexit_intercept - flen / cos(alpha) / 2;
      exit_intercept = rexit_intercept + flen / cos(alpha) / 2;
      break;
    }

    cos_alpha2 = cos(alpha);
    sin_alpha2 = sin(alpha);
    if (alpha != PIo2)
      exit_slope = tan(alpha);
    else
      exit_slope = DBL_MAX;

#ifdef DEBUG
    if (1) {
      FILE *fp_edges;
      fp_edges = fopen("edges.sdds", "w");
      fprintf(fp_edges, "SDDS1\n&parameter name=Type type=string &end\n");
      fprintf(fp_edges, "&column name=q0 type=double units=m &end\n");
      fprintf(fp_edges, "&column name=q1 type=double units=m &end\n");
      fprintf(fp_edges, "&data mode=ascii no_row_counts=1 &end\n");
      fprintf(fp_edges, "Fringe1Begin\n");
      fprintf(fp_edges, "%le 0\n", entr_intercept);
      fprintf(fp_edges, "%le %le\n\n", entr_intercept + entr_slope * nibend->rho0 * 2, nibend->rho0 * 2);
      fprintf(fp_edges, "Fringe1End\n");
      fprintf(fp_edges, "%le 0\n", fentr_intercept);
      fprintf(fp_edges, "%le %le\n\n", fentr_intercept + entr_slope * nibend->rho0 * 2, nibend->rho0 * 2);
      fprintf(fp_edges, "Reference1\n");
      fprintf(fp_edges, "%le 0\n", rentr_intercept);
      fprintf(fp_edges, "%le %le\n\n", rentr_intercept + entr_slope * nibend->rho0 * 2, nibend->rho0 * 2);

      fprintf(fp_edges, "Fringe2Begin\n");
      fprintf(fp_edges, "%le 0\n", fexit_intercept);
      fprintf(fp_edges, "%le %le\n\n", fexit_intercept + exit_slope * nibend->rho0 * 2, nibend->rho0 * 2);
      fprintf(fp_edges, "Fringe2End\n");
      fprintf(fp_edges, "%le 0\n", exit_intercept);
      fprintf(fp_edges, "%le %le\n\n", exit_intercept + exit_slope * nibend->rho0 * 2, nibend->rho0 * 2);
      fprintf(fp_edges, "Reference2\n");
      fprintf(fp_edges, "%le 0\n", rexit_intercept);
      fprintf(fp_edges, "%le %le\n\n", rexit_intercept + exit_slope * nibend->rho0 * 2, nibend->rho0 * 2);
      fclose(fp_edges);
    }

#endif
    if (!nibend->initialized) {
      /* find zeta offset or fse adjustment to give the right bending angle */
      if (!flen)
        nibend->zeta_offset = nibend->fse_adjust = nibend->x_correction = nibend->s_offset = 0;
      else {
        double save[5];
        /* set values for offset adjustment (no errors, etc) */
        rad_coef = 0;
        x_correction = offset = 0;
        save[0] = nibend->etilt;
        nibend->etilt = 0;
        save[1] = nibend->dx;
        nibend->dx = 0;
        save[2] = nibend->dy;
        nibend->dy = 0;
        save[3] = nibend->dz;
        nibend->dz = 0;
        save[4] = nibend->fse;
        nibend->fse = 0;
        fse_opt = 0;
        one_plus_fse = 1;
        if (nibend->adjustField) {
          fse_opt = zeroNewton(nibend_trajectory_error_fse, 0, fse_opt, 1e-6, 10, 1e-14 * nibend->rho0);
#ifdef IEEE_MATH
          if (isnan(fse_opt) || isinf(fse_opt))
            bombElegant("Newton's method failed to find strength adjustment for NIBEND--decrease accuracy parameter", NULL);
#endif
        } else if (nibend->adjustBoundary) {
          offset = zeroNewton(nibend_trajectory_error_offset, 0, offset, 1e-6, 10, 1e-14 * nibend->rho0);
#ifdef IEEE_MATH
          if (isnan(offset) || isinf(offset))
            bombElegant("Newton's method failed to find coordinate offset for NIBEND--decrease accuracy parameter", NULL);
#endif
        }
        if (fringe_code == ENGE1_MODEL || fringe_code == ENGE3_MODEL || fringe_code == ENGE5_MODEL)
          printf("Enge coefficients: D=%e, a=%e, %e, %e\n",
                 engeD, engeCoef[0], engeCoef[1], engeCoef[2]);
        if (nibend->adjustField) {
          printf("NIBEND FSE adjusted by %e to obtain trajectory error of %e\n",
                 fse_opt, nibend_trajectory_error_fse(fse_opt));
          printf("final coordinates: %e, %e, %e, %e, %e\n",
                 traj_err_final_coord[0], traj_err_final_coord[1], traj_err_final_coord[2],
                 traj_err_final_coord[3], traj_err_final_coord[4]);
          fflush(stdout);
        } else if (nibend->adjustBoundary) {
          printf("NIBEND offset adjusted to %e to obtain trajectory error of %e\n",
                 offset, nibend_trajectory_error_offset(offset));
          printf("(Positive offset adjustment means a shorter magnet.)\n");
          printf("final coordinates: %e, %e, %e, %e, %e\n",
                 traj_err_final_coord[0], traj_err_final_coord[1], traj_err_final_coord[2],
                 traj_err_final_coord[3], traj_err_final_coord[4]);
          if (fringe_code == ENGE1_MODEL || fringe_code == ENGE3_MODEL || fringe_code == ENGE5_MODEL) {
            printf("Equivalent Enge coefficients: D=%e, a=%e, %e, %e\n",
                   engeD,
                   engeCoef[0] + engeCoef[1] * offset / engeD + engeCoef[2] * sqr(offset / engeD),
                   engeCoef[1] + 2 * engeCoef[2] * offset / engeD,
                   engeCoef[2]);
            fflush(stdout);
          }
        }
        nibend->zeta_offset = offset;
        nibend->fse_adjust = fse_opt;
        nibend->x_correction = traj_err_final_coord[0];
        if (nibend->fudgePathLength)
          nibend->s_offset = (nibend->length - traj_err_final_coord[4]) / 2;
        else
          nibend->s_offset = 0;
        nibend->etilt = save[0];
        nibend->dx = save[1];
        nibend->dy = save[2];
        nibend->dz = save[3];
        nibend->fse = save[4];
      }
      nibend->initialized = 1;
    }
    offset = nibend->zeta_offset;
    fse_opt = nibend->fse_adjust;
    one_plus_fse = 1 + nibend->fse + nibend->fse_adjust;
    x_correction = nibend->x_correction;
    s_offset = nibend->s_offset;
    if (nibend->synch_rad)
      rad_coef = sqr(particleCharge / c_mks) * ipow3(Po) / (6 * PI * epsilon_o * particleMass);
    else
      rad_coef = 0;
    /*
#ifdef DEBUG
            printf("entrance: begin fringe intercept = %.16le, end = %.16le, slope = %.16le\n",
                entr_intercept, fentr_intercept, entr_slope);
            fflush(stdout);
            printf("exit    : begin fringe intercept = %.16le, end = %.16le, slope = %.16le\n",
                exit_intercept, fexit_intercept, exit_slope);
            fflush(stdout);
#endif
*/
    break;
  case T_NISEPT:
    nisept = (NISEPT *)field;
    exit_function = nisept_exit_function;
    deriv_function = nisept_deriv_function;
    coord_transform = nisept_coord_transform;
    select_lorentz_integrator(nisept->method);
    nisept->negative_angle = 0;
    if (nisept->angle < 0) {
      nisept->angle *= -1;
      nisept->e1 *= -1;
      nisept->negative_angle = 1;
    }
    nisept->rho0 = nisept->length / nisept->angle;
    S0 = 1 / nisept->rho0;
    central_length = nisept->length;
    tolerance = nisept->accuracy;

    /* calculate slope and intercept for entrance plane */
    entr_intercept = -nisept->rho0 * sin(nisept->e1) - nisept->flen / 2;
    fentr_intercept = entr_intercept + nisept->flen;
    entr_slope = 0;
    cos_alpha1 = cos(nisept->e1);
    sin_alpha1 = sin(nisept->e1);

    /* calculate slope and intercept for exit plane */
    nisept->e2 = nisept->angle - nisept->e1;
    exit_intercept = nisept->rho0 * sin(nisept->e2) + nisept->flen / 2;
    fexit_intercept = exit_intercept - nisept->flen;
    cos_alpha2 = cos(nisept->e2);
    sin_alpha2 = sin(nisept->e2);
    exit_slope = 0;

    /* find fractional strength error to give desired slope at exit, using Newton's method */
    fse_opt = nisept->fse_opt;
    if (fse_opt == 0) {
      /* q1 offset to global coordinates to add to q1_ref */
      nisept->q1_offset = nisept->rho0 * cos(nisept->e1);
      if ((fse_opt = nisept->last_fse_opt) == 0)
        fse_opt = last_fse;
      fse_opt = zeroNewton(nisept_trajectory_error, 0, fse_opt, 1e-6, 10, 1e-14);
#ifdef IEEE_MATH
      if (isnan(fse_opt) || isinf(fse_opt))
        bombElegant("Newton's method failed to find coordinate fse_opt for NISEPT--decrease accuracy parameter", NULL);
#endif
      if (fse_opt == 0)
        fse_opt = DBL_MIN;
      printf("NISEPT fse_opt adjusted to %e to obtain trajectory error of %e\n",
             fse_opt, nisept_trajectory_error(fse_opt));
      fflush(stdout);
      printf("final coordinates: %e, %e, %e, %e, %e\n",
             traj_err_final_coord[0], traj_err_final_coord[1], traj_err_final_coord[2],
             traj_err_final_coord[3], traj_err_final_coord[4]);
      fflush(stdout);
      last_fse = nisept->last_fse_opt = nisept->fse_opt = fse_opt;
    }
    break;
  case T_BMAPXY:
    bmapxy = (BMAPXY *)field;
    exit_function = bmapxy_exit_function;
    deriv_function = bmapxy_deriv_function;
    coord_transform = bmapxy_coord_transform;
    central_length = bmapxy->length;
    select_lorentz_integrator(bmapxy->method);
    tolerance = bmapxy->accuracy;
    if ((!bmapxy->filename && !(bmapxy->FxRpn && bmapxy->FyRpn)) ||
        (bmapxy->filename && (bmapxy->FxRpn || bmapxy->FyRpn)))
      bombElegant("Specify one and only one of filename or (Fx,Fy) for BMAPXY", NULL);
    if (bmapxy->filename) {
      if (!bmapxy->points)
        bmapxy_field_setup(bmapxy);
    } else {
      bmapxy->rpnMem[0] = rpn_create_mem("x", 0);
      bmapxy->rpnMem[1] = rpn_create_mem("y", 0);
    }
    break;
  case T_BMAPXYZ:
    bmapxyz = (BMAPXYZ *)field;
    exit_function = bmapxyz_exit_function;
    deriv_function = bmapxyz_deriv_function;
    coord_transform = bmapxyz_coord_transform;
    select_lorentz_integrator(bmapxyz->method);
    tolerance = bmapxyz->accuracy;
    if (!bmapxyz->filename)
      bombElegant("Specify filename for BMXYZ", NULL);
    if (!bmapxyz->data)
      bmapxyz_field_setup(bmapxyz);
    central_length = bmapxyz->fieldLength > 0 ? bmapxyz->fieldLength : bmapxyz->length;
    rad_coef = 0;
    if (bmapxyz->synchRad)
      rad_coef = sqr(particleCharge / c_mks) * ipow3(Po) / (6 * PI * epsilon_o * particleMass);
#if !USE_MPI
    if (bmapxyz->particleOutputFile && !(bmapxyz->SDDSpo)) {
      TRACKING_CONTEXT tcontext;
      char *filename;
      getTrackingContext(&tcontext);
      bmapxyz->SDDSpo = tmalloc(sizeof(*(bmapxyz->SDDSpo)));
      bmapxyz->poRow = 0;
      filename = compose_filename(bmapxyz->particleOutputFile, tcontext.rootname);
      if (!SDDS_InitializeOutputElegant(bmapxyz->SDDSpo, SDDS_BINARY, 1,
                                        NULL, NULL, filename) ||
          0 > SDDS_DefineParameter(bmapxyz->SDDSpo, "SVNVersion", NULL, NULL, "SVN version number", NULL, SDDS_STRING, SVN_VERSION) ||
          !SDDS_DefineSimpleParameter(bmapxyz->SDDSpo, "particleID", NULL, SDDS_ULONG64) ||
          !SDDS_DefineSimpleParameter(bmapxyz->SDDSpo, "pCentral", "m$be$nc", SDDS_DOUBLE) ||
          (bmapxyz->poIndex[0] = SDDS_DefineColumn(bmapxyz->SDDSpo, "x", NULL, "m", NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
          (bmapxyz->poIndex[1] = SDDS_DefineColumn(bmapxyz->SDDSpo, "xp", NULL, NULL, NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
          (bmapxyz->poIndex[2] = SDDS_DefineColumn(bmapxyz->SDDSpo, "y", NULL, "m", NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
          (bmapxyz->poIndex[3] = SDDS_DefineColumn(bmapxyz->SDDSpo, "py", NULL, NULL, NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
          (bmapxyz->poIndex[4] = SDDS_DefineColumn(bmapxyz->SDDSpo, "z", NULL, "m", NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
          (bmapxyz->poIndex[5] = SDDS_DefineColumn(bmapxyz->SDDSpo, "pz", NULL, NULL, NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
          (bmapxyz->poIndex[6] = SDDS_DefineColumn(bmapxyz->SDDSpo, "Bx", NULL, "T", NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
          (bmapxyz->poIndex[7] = SDDS_DefineColumn(bmapxyz->SDDSpo, "By", NULL, "T", NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
          (bmapxyz->poIndex[8] = SDDS_DefineColumn(bmapxyz->SDDSpo, "Bz", NULL, "T", NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
          !SDDS_WriteLayout(bmapxyz->SDDSpo)) {
        SDDS_SetError("Problem setting up particle output file for BMXYZ");
        SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
      }
    }
#endif
    break;
  default:
    bombElegant("invalid field type (lortenz_setup)", NULL);
    break;
  }
  if (s_offset)
    exactDrift(part, np, s_offset);
  log_exit("lorentz_setup");
}

void lorentz_terminate(
  void *field,
  long field_type,
  double **part,
  int64_t np,
  double Po) {
  NIBEND *nibend;
  BMAPXYZ *bmxyz;

  switch (field_type) {
  case T_NIBEND:
    nibend = (NIBEND *)field;
    nibend->angle *= nibend->angleSign;
    nibend->e[nibend->e1Index] *= nibend->angleSign;
    nibend->e[nibend->e2Index] *= nibend->angleSign;
    break;
  case T_BMAPXYZ:
    bmxyz = (BMAPXYZ *)field;
    if (bmxyz->SDDSpo) {
      if (!SDDS_Terminate(bmxyz->SDDSpo)) {
        SDDS_SetError("Problem closing particle output file for BMXYZ");
        SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
      }
      free(bmxyz->SDDSpo);
      bmxyz->SDDSpo = NULL;
    }
  default:
    break;
  }
  if (s_offset)
    exactDrift(part, np, s_offset);
}

void select_lorentz_integrator(char *desired_method) {
  long i;

  log_entry("select_lorentz_integrator");

  switch (method_code = match_string(desired_method, method, N_METHODS, 0)) {
  case RUNGE_KUTTA:
    integrator = rk_odeint3;
    break;
  case BULIRSCH_STOER:
    integrator = bs_odeint3;
    break;
  case NA_RUNGE_KUTTA:
    integrator = rk_odeint3_na;
    break;
  case MODIFIED_MIDPOINT:
  case TWO_PASS_MODIFIED_MIDPOINT:
  case LEAP_FROG:
    integrator = NULL; /* use to indicate that non-adaptive integration will be used--pretty kludgey */
    break;
  default:
    printf("error: unknown integration method %s requested.\n", desired_method);
    printf("Available methods are:\n");
    for (i = 0; i < N_METHODS; i++)
      printf("    %s\n", method[i]);
    fflush(stdout);
    exitElegant(1);
    break;
  }
  log_exit("select_lorentz_integrator");
}

void lorentz_leap_frog(double *Qf, double *Qi, double s, long n_steps, void (*derivs)(double *dQds, double *Q, double sd)) {
  long step;
  double h, hh;
  double *q, *w, *wp;
  /* double *qp; */
  double F[8];

  q = Qf;
  w = Qf + 3;
  /* qp = F; */
  wp = F + 3;

  copy_doubles(Qf, Qi, 8);

  h = s / n_steps;
  hh = h / 2;
  for (step = 0; step < n_steps; step++) {
    if (step != 0) {
      /* advance position variables by two half-steps */
      q[0] += w[0] * h;
      q[1] += w[1] * h;
      q[2] += w[2] * h;
    } else {
      /* advance position variables by half-step */
      q[0] += w[0] * hh;
      q[1] += w[1] * hh;
      q[2] += w[2] * hh;
    }
    /* find forces */
    derivs(F, Qf, 0.0);
    /* advance momentum variables */
    w[0] += wp[0] * h;
    w[1] += wp[1] * h;
    w[2] += wp[2] * h;
    w[3] += wp[3] * h;
    w[4] += wp[4] * h;
  }
  /* advance position variables by half-step */
  q[0] += w[0] * hh;
  q[1] += w[1] * hh;
  q[2] += w[2] * hh;
}

/* routines for NIBEND element: */

double nibend_trajectory_error_offset(double offsetp) {
  static double coord[6];

  fill_double_array(coord, 6, 0.0);
  offset = offsetp;
  if (!do_lorentz_integration(coord, field_global))
    bombElegant("integration failure in nibend_trajectory_error", NULL);
  traj_err_final_coord = coord;
  return (coord[1]);
}

double nibend_trajectory_error_fse(double fsep) {
  static double coord[6], opf;

  fill_double_array(coord, 6, 0.0);
  opf = one_plus_fse;
  one_plus_fse += fsep;
  if (!do_lorentz_integration(coord, field_global))
    bombElegant("integration failure in nibend_trajectory_error", NULL);
  traj_err_final_coord = coord;
  one_plus_fse = opf;
  return (coord[1]);
}

void nibend_coord_transform(double *q, double *coord, void *field, long which_end) {
  double dxds, dyds, dzds;
  double q0[3], dqds[3], dcoordEtilt[6];
  double cos_ah, sin_ah, tan_ah;
  double q0I, ds, alpha;
  long i;
  double dz, dx, dy;
  NIBEND *nibend = field;

  if (which_end == -1) {
    /* do misalignment */
    dx = -nibend->dx;
    dz = nibend->dz;
    dy = -nibend->dy;
    coord[4] += dz * sqrt(1 + sqr(coord[1]) + sqr(coord[3]));
    coord[0] = coord[0] + dx + dz * coord[1];
    coord[2] = coord[2] + dy + dz * coord[3];
    /* perform coordinate rotation---centroid offset due to error tilt is 
         * added at the end 
         */
    rotateCoordinatesForMisalignment(coord, nibend->tilt + (nibend->angleSign == -1 ? PI : 0) + nibend->etilt);
    /* transform coord (x, x', y, y', s, dp/p) into q (q0,q1,q2,d/ds(q0,q1,q2),s,dp/p) */
    /* convert slopes to normalized velocities */
    dzds = 1 / sqrt(1 + sqr(coord[1]) + sqr(coord[3]));
    dxds = coord[1] * dzds;
    dyds = coord[3] * dzds;
    /* find q coordinates of particle at reference plane */
    q0[0] = -(nibend->rho0 + coord[0]) * (sin_ah = sin(nibend->angle / 2));
    q0[1] = (nibend->rho0 + coord[0]) * (cos_ah = cos(nibend->angle / 2));
    q0[2] = coord[2];
    dqds[0] = dzds * cos_ah - dxds * sin_ah;
    dqds[1] = dzds * sin_ah + dxds * cos_ah;
    dqds[2] = dyds;
    /* find q coordinates of particle at entrance plane */
    alpha = nibend->angle / 2 - nibend->e[nibend->e1Index];
    switch (nibend->fringePosition) {
    case -1:
      q0I = nibend->rho0 * (cos_ah * tan(alpha) - sin_ah);
      break;
    case 1:
      q0I = -flen / cos(alpha) + nibend->rho0 * (cos_ah * tan(alpha) - sin_ah);
      break;
    case 0:
      q0I = -flen / 2 / cos(alpha) + nibend->rho0 * (cos_ah * tan(alpha) - sin_ah);
      break;
    default:
      q0I = 0;
      bombElegant("fringe position error in nibend_coord_transform", NULL);
      break;
    }
    ds = (-q0[1] * tan(alpha) + q0I - q0[0]) / (dqds[0] + dqds[1] * tan(alpha));
#ifdef DEBUG
    printf("Initial ds = %e\n", ds);
#endif
    q[0] = q0[0] + dqds[0] * ds;
    q[1] = q0[1] + dqds[1] * ds;
    q[2] = q0[2] + dqds[2] * ds;
    q[3] = dqds[0];
    q[4] = dqds[1];
    q[5] = dqds[2];
    q[6] = coord[4] + ds;
    q[7] = coord[5];
  } else if (which_end == -2) {
    bombElegant("nibend transform called with which_end==-2", NULL);
    /* transform q into coord at entrance refence plane */
    dqds[0] = q[3];
    dqds[1] = q[4];
    dqds[2] = q[5];
    /* drift back to reference plane */
    tan_ah = tan(nibend->angle / 2);
    ds = -(q[0] + q[1] * tan_ah) / (dqds[0] + dqds[1] * tan_ah);
    q0[0] = q[0] + ds * dqds[0];
    q0[1] = q[1] + ds * dqds[1];
    q0[2] = q[2] + ds * dqds[2];
    /* convert to (x,y,z) system */
    coord[0] = sqrt(sqr(q0[0]) + sqr(q0[1])) - nibend->rho0;
    coord[2] = q0[2];
    cos_ah = cos(nibend->angle / 2);
    sin_ah = sin(nibend->angle / 2);
    dzds = dqds[0] * cos_ah + dqds[1] * sin_ah;
    dxds = -dqds[0] * sin_ah + dqds[1] * cos_ah;
    dyds = dqds[2];
    coord[1] = dxds / dzds;
    coord[3] = dyds / dzds;
    coord[4] = q[6] + ds;
    coord[5] = q[7];
    if (nibend->angleSign == -1) {
      for (i = 0; i < 4; i++)
        coord[i] *= -1;
    }
  } else if (which_end == 1) {
    /* transform q into coord at end of magnet */
    dqds[0] = q[3];
    dqds[1] = q[4];
    dqds[2] = q[5];
    /* drift back to reference plane */
    tan_ah = tan(nibend->angle / 2);
    ds = -(q[0] - q[1] * tan_ah) / (dqds[0] - dqds[1] * tan_ah);
#ifdef DEBUG
    printf("Final ds = %e\n", ds);
#endif
    q0[0] = q[0] + ds * dqds[0];
    q0[1] = q[1] + ds * dqds[1];
    q0[2] = q[2] + ds * dqds[2];
    /* convert to (x,y,z) system */
    coord[0] = sqrt(sqr(q0[0]) + sqr(q0[1])) - nibend->rho0 - x_correction;
    coord[2] = q0[2];
    cos_ah = cos(nibend->angle / 2);
    sin_ah = sin(nibend->angle / 2);
    dzds = dqds[0] * cos_ah - dqds[1] * sin_ah;
    dxds = dqds[0] * sin_ah + dqds[1] * cos_ah;
    dyds = dqds[2];
    coord[1] = dxds / dzds;
    coord[3] = dyds / dzds;
    coord[4] = q[6] + ds;
    coord[5] = q[7];
    /* rotate back and add centroid offset due to error tilt */
    rotateCoordinatesForMisalignment(coord, -(nibend->tilt + (nibend->angleSign == -1 ? PI : 0) + nibend->etilt));
    computeEtiltCentroidOffset(dcoordEtilt, nibend->rho0, nibend->angle, nibend->etilt,
                               nibend->tilt + (nibend->angleSign == -1 ? PI : 0));
    for (i = 0; i < 4; i++)
      coord[i] += dcoordEtilt[i];
    /* undo misalignment */
    dx = nibend->dx * cos(nibend->angleSign * nibend->angle) + nibend->dz * sin(nibend->angleSign * nibend->angle);
    dz = nibend->dx * sin(nibend->angleSign * nibend->angle) - nibend->dz * cos(nibend->angleSign * nibend->angle);
    dy = nibend->dy;
    coord[0] += dx + dz * coord[1];
    coord[2] += dy + dz * coord[3];
    coord[4] += dz * sqrt(1 + sqr(coord[1]) + sqr(coord[3]));
  } else
    bombElegant("unknown coordinate conversion requested (nibend_coord_transform)", NULL);
}

double nibend_exit_function(double *qp, double *q, double s) {
  /* static NIBEND *nibend; */
  /* nibend = (NIBEND*)field_global; */

  /* returns positive if inside, negative if outside, exit fringe region */
  if (exit_slope != DBL_MAX)
    return ((q[1] * exit_slope + exit_intercept) - q[0]);
  else
    return (q[0]);
}

void nibend_deriv_function(double *qp, double *q, double s) {
  static double *w, *wp, F0, F1, F2;
  static double S, dq0, z, z2, z3, q22;
  /* static NIBEND *nibend; */
  static double Fa_y, Fa_z, tanh_Fa_z, sech_Fa_z, cosh_Fa_z;
  static double sinh_Fa_z, sinh_3Fa_z, cosh_2Fa_z, cosh_4Fa_z;
  static double Fa_y_sech, tmp, tmp1;

  /* nibend = (NIBEND*)field_global; */
  n_deriv_calls++;

  w = q + 3;
  wp = qp + 3;
  qp[0] = w[0];
  qp[1] = w[1];
  qp[2] = w[2];

  S = S0 * one_plus_fse / (1 + q[7]);
  qp[6] = 1;
  qp[7] = 0;

  if (fringe_code == TANH_MODEL) {
    if (q[0] < 0) {
      /* positive z goes into the magnet */
      dq0 = q[0] - (q[1] * entr_slope + fentr_intercept);
      z = flen / 2 + dq0 * cos_alpha1 + offset;
    } else {
      /* positive z goes out of the magnet */
      dq0 = q[0] - (q[1] * exit_slope + fexit_intercept);
      z = (-flen / 2 + dq0 * cos_alpha2) - offset;
    }
    if ((q[0] < 0 && z < -flen / 2) || (q[0] > 0 && z > flen / 2))
      F0 = F1 = F2 = 0;
    else {
      Fa_y = Fa * q[2];
      Fa_z = Fa * z;
      tanh_Fa_z = tanh(Fa_z);
      sech_Fa_z = 1 / (cosh_Fa_z = cosh(Fa_z));
      sinh_Fa_z = tanh_Fa_z * cosh_Fa_z;
      cosh_2Fa_z = 1 + 2 * sqr(sinh_Fa_z);
      sinh_3Fa_z = sinh(3 * Fa_z);
      cosh_4Fa_z = 2 * sqr(cosh_2Fa_z) - 1;
      Fa_y_sech = Fa_y * sech_Fa_z;
      F2 = (1 + tanh_Fa_z) / 2 + (tmp = sqr(Fa_y_sech)) * tanh_Fa_z / 2;
      F2 += sqr(tmp) * sech_Fa_z * (11 * sinh_Fa_z - sinh_3Fa_z) / 24;
      F0 = tmp1 = Fa_y_sech * sech_Fa_z / 2;
      F0 += tmp1 * (tmp = sqr(Fa_y_sech)) * (2 - cosh_2Fa_z) / 3;
      F0 += tmp1 * sqr(tmp) * (33 - 26 * cosh_2Fa_z + cosh_4Fa_z) / 60;
      if (q[0] < 0) {
        F2 *= S;
        F0 *= S;
        F1 = F0 * sin_alpha1;
        F0 = F0 * cos_alpha1;
      } else {
        /* The next two lines account for the fact that tanh(a*z) is reversed for the exit */
        F2 = S * (1 - F2);
        F0 *= -S;
        F1 = -F0 * sin_alpha2;
        F0 = F0 * cos_alpha2;
      }
    }
  } else if (fringe_code == ENGE1_MODEL || fringe_code == ENGE3_MODEL || fringe_code == ENGE5_MODEL) {
    if (q[0] < 0) {
      /* entrance---positive z is out of the magnet */
      dq0 = q[0] - (q[1] * entr_slope + rentr_intercept);
      z = -dq0 * cos_alpha1 + offset;
      engeFunction(&F2, &F0, z, q[2], engeD, engeCoef[0], engeCoef[1], engeCoef[2], engeOrder);
      F2 *= S;
      F0 *= -S;
      F1 = F0 * sin_alpha1;
      F0 = F0 * cos_alpha1;
    } else {
      /* exit---positive z is out of the magnet */
      dq0 = q[0] - (q[1] * exit_slope + rexit_intercept);
      z = dq0 * cos_alpha1 + offset;
      engeFunction(&F2, &F0, z, q[2], engeD, engeCoef[0], engeCoef[1], engeCoef[2], engeOrder);
      F2 *= S;
      F0 *= S;
      F1 = -F0 * sin_alpha2;
      F0 = F0 * cos_alpha2;
    }
  } else {
    /* determine which region particle is in */
    /* determine if inside entrance fringe */
    dq0 = q[0] - (q[1] * entr_slope + fentr_intercept);
    z = flen + dq0 * cos_alpha1 + offset;
    if (z < 0)
      F0 = F1 = F2 = 0; /* outside of magnet */
    else if (z < flen) {
      /* in entrance fringe */
      /* positive z goes into the magnet */
      switch (fringe_code) {
      case LINEAR_MODEL:
        F2 = S * z / flen;
        F0 = S * q[2] / flen;
        break;
      case CUBIC_MODEL:
        z2 = sqr(z);
        q22 = sqr(q[2]);
        F2 = S * ((Fa + Fb * z) * z2 + q22 * (-Fa - 3 * Fb * z));
        F0 = S * q[2] * (2 * Fa * z + 3 * Fb * z2);
        break;
      case QUINTIC_MODEL:
        z2 = sqr(z);
        z3 = z2 * z;
        q22 = sqr(q[2]);
        F2 = S * ((Fa + Fb * z + Fc * z2) * z3 - q22 * (3 * Fa + 6 * Fb * z + 10 * Fc * z2) * z);
        F0 = S * q[2] * ((3 * Fa + 4 * Fb * z + 5 * Fc * z2) * z2 - q22 * (Fa + 4 * Fb * z + 10 * Fc * z2));
        break;
      default:
        bombElegant("invalid fringe-field code in nibend_deriv_function", NULL);
        break;
      }
      F1 = F0 * sin_alpha1;
      F0 = F0 * cos_alpha1;
    } else {
      /* determine if inside exit fringe region */
      dq0 = q[0] - (q[1] * exit_slope + fexit_intercept);
      z = dq0 * cos_alpha2 - offset;
      if (z > flen)
        F0 = F1 = F2 = 0; /* outside of magnet */
      else if (z > 0) {
        /* positive z goes out of the magnet */
        switch (fringe_code) {
        case LINEAR_MODEL:
          F2 = S * (1 - z / flen);
          F0 = -S * q[2] / flen;
          break;
        case CUBIC_MODEL:
          z2 = sqr(z);
          q22 = sqr(q[2]);
          F2 = S * (1 - ((Fa + Fb * z) * z2 + q22 * (-Fa - 3 * Fb * z)));
          F0 = -S * q[2] * (2 * Fa * z + 3 * Fb * z2);
          break;
        case QUINTIC_MODEL:
          z2 = sqr(z);
          z3 = z2 * z;
          q22 = sqr(q[2]);
          F2 = S * (1 - ((Fa + Fb * z + Fc * z2) * z3 - q22 * (3 * Fa + 6 * Fb * z + 10 * Fc * z2) * z));
          F0 = -S * q[2] * ((3 * Fa + 4 * Fb * z + 5 * Fc * z2) * z2 - q22 * (Fa + 4 * Fb * z + 10 * Fc * z2));
          break;
        default:
          bombElegant("invalid fringe-field code in nibend_deriv_function", NULL);
          break;
        }
        F1 = -F0 * sin_alpha2;
        F0 = F0 * cos_alpha2;
      } else {
        F0 = F1 = 0;
        F2 = S;
      }
    }
  }

  wp[0] = w[1] * F2 - w[2] * F1;
  wp[1] = w[2] * F0 - w[0] * F2;
  wp[2] = w[0] * F1 - w[1] * F0;
  if (rad_coef)
    qp[7] = -rad_coef * pow4(1 + q[7]) * (sqr(wp[0]) + sqr(wp[1]) + sqr(wp[2]));

#ifdef DEBUG
  if (field_output_on) {
    fprintf(fp_field, "%21.15e %21.15e %21.15e %21.15e %21.15e %21.15e %21.15e %21.15e\n", q[0], q[1], q[2], z, s, F0, F1, F2);
    fflush(fp_field);
  }
#endif
}

/* routines for NISEPT element: */

double nisept_trajectory_error(double fsep) {
  static double coord[6];

  fill_double_array(coord, 6, 0.0);
  fse_opt = fsep;
  if (!do_lorentz_integration(coord, field_global))
    bombElegant("integration failure in nisept_trajectory_error", NULL);
  traj_err_final_coord = coord;
  return (coord[1]);
}

void nisept_coord_transform(double *q, double *coord, void *field, long which_end) {
  double dxds, dyds, dzds;
  double q0[3], dqds[3];
  double sin_e1, cos_e1;
  double sin_e2, cos_e2;
  double ds;
  long i;
  NISEPT *nisept = field;

  if (which_end == -1) {
    if (nisept->negative_angle) {
      for (i = 0; i < 4; i++)
        coord[i] *= -1;
    }
    /* transform coord (x, x', y, y', s, dp/p) into q (q0,q1,q2,d/ds(q0,q1,q2),s,dp/p) */
    /* convert slopes to normalized velocities */
    dzds = 1 / sqrt(1 + sqr(coord[1]) + sqr(coord[3]));
    dxds = coord[1] * dzds;
    dyds = coord[3] * dzds;
    /* find q coordinates of particle at reference plane */
    q0[0] = -(nisept->rho0 + coord[0]) * (sin_e1 = sin(nisept->e1));
    q0[1] = (nisept->rho0 + coord[0]) * (cos_e1 = cos(nisept->e1));
    q0[2] = coord[2];
    dqds[0] = dzds * cos_e1 - dxds * sin_e1;
    dqds[1] = dzds * sin_e1 + dxds * cos_e1;
    dqds[2] = dyds;
    /* find q coordinates of particle at entrance plane */
    ds = (entr_intercept - q0[0]) / dqds[0];
    q[0] = q0[0] + dqds[0] * ds;
    q[1] = q0[1] + dqds[1] * ds;
    q[2] = q0[2] + dqds[2] * ds;
    q[3] = dqds[0];
    q[4] = dqds[1];
    q[5] = dqds[2];
    q[6] = coord[4] + ds;
    q[7] = coord[5];
  } else if (which_end == 1) {
    /* transform q into coord at end of magnet */
    dqds[0] = q[3];
    dqds[1] = q[4];
    dqds[2] = q[5];
    /* drift back to reference plane */
    ds = -(q[0] - q[1] * tan(nisept->e2)) / (dqds[0] - dqds[1] * tan(nisept->e2));
    q0[0] = q[0] + ds * dqds[0];
    q0[1] = q[1] + ds * dqds[1];
    q0[2] = q[2] + ds * dqds[2];
    /* convert to (x,y,z) system */
    coord[0] = sqrt(sqr(q0[0]) + sqr(q0[1])) - nisept->rho0;
    coord[2] = q0[2];
    cos_e2 = cos(nisept->e2);
    sin_e2 = sin(nisept->e2);
    dzds = dqds[0] * cos_e2 - dqds[1] * sin_e2;
    dxds = dqds[0] * sin_e2 + dqds[1] * cos_e2;
    dyds = dqds[2];
    coord[1] = dxds / dzds;
    coord[3] = dyds / dzds;
    coord[4] = q[6] + ds;
    coord[5] = q[7];
    if (nisept->negative_angle) {
      for (i = 0; i < 4; i++)
        coord[i] *= -1;
    }
  } else
    bombElegant("unknown coordinate conversion requested (nisept_coord_transform)", NULL);
}

double nisept_exit_function(double *qp, double *q, double s) {
  /* static NISEPT *nisept; */
  /* nisept = (NISEPT*)field_global; */

  /* returns positive if inside or before, negative if after */
  return (exit_intercept - q[0]);
}

void nisept_deriv_function(double *qp, double *q, double s) {
  static double *w, *wp, F0, F1, F2;
  static double S, dq0, dq1;
  static NISEPT *nisept;

  nisept = (NISEPT *)field_global;
  n_deriv_calls++;

  w = q + 3;
  wp = qp + 3;
  qp[0] = w[0];
  qp[1] = w[1];
  qp[2] = w[2];

  S = S0 * (1 + fse_opt) / (1 + q[7]);
  qp[6] = 1;
  qp[7] = 0;

  /* determine if inside entrance fringe */
  dq0 = q[0] - entr_intercept;
  dq1 = q[1] - (nisept->q1_ref + nisept->q1_offset);
  if (dq0 <= 0)
    F0 = F1 = F2 = 0; /* outside of magnet */
  else if (nisept->flen && dq0 <= nisept->flen) {
    /* in entrance fringe */
    S /= nisept->flen;
    F0 = S * q[2] * (1 + nisept->b1 * dq1);
    F1 = S * q[2] * dq0 * nisept->b1;
    F2 = S * dq0 * (1 + nisept->b1 * dq1);
  } else {
    /* determine if inside exit fringe region */
    dq0 = q[0] - fexit_intercept;
    if (dq0 >= nisept->flen)
      F0 = F1 = F2 = 0; /* outside of magnet */
    else if (nisept->flen && dq0 > 0) {
      S /= nisept->flen;
      F0 = -S * q[2] * (1 + nisept->b1 * dq1);
      F1 = S * q[2] * (nisept->flen - dq0) * nisept->b1;
      F2 = S * (nisept->flen - dq0) * (1 + nisept->b1 * dq1);
    } else {
      /* in interior */
      F0 = 0;
      F1 = S * nisept->b1 * q[2];
      F2 = S * (1 + nisept->b1 * dq1);
    }
  }

  wp[0] = w[1] * F2 - w[2] * F1;
  wp[1] = w[2] * F0 - w[0] * F2;
  wp[2] = w[0] * F1 - w[1] * F0;

#ifdef DEBUG
  if (fp_field && field_output_on) {
    fprintf(fp_field, "%21.15e %21.15e %21.15e %21.15e %21.15e %21.15e %21.15e %21.15e\n", q[0], q[1], q[2], 0.0, s, F0, F1, F2);
    fflush(fp_field);
  }
#endif
}

double bmapxy_exit_function(double *qp, double *q, double s) {
  /* static BMAPXY *bmapxy; */
  /* bmapxy = (BMAPXY*)field_global; */

  /* returns positive if inside or before, negative if after */
  return (central_length - q[0]);
}

void bmapxy_deriv_function(double *qp, double *q, double s) {
  double *w, *wp, F0, F1, F2, Fa, Fb, fx, fy;
  double x, y;
  BMAPXY *bmapxy;
  long ix, iy;

  bmapxy = (BMAPXY *)field_global;
  n_deriv_calls++;

  x = q[1];
  y = q[2];
  if (isnan(x) || isnan(y) || isinf(x) || isinf(y)) {
    for (ix = 0; ix < 7; ix++)
      qp[ix] = 0;
    qp[6] = 1;
    q[0] = central_length;
    n_invalid_particles++;
    return;
  }

  /* w is the velocity */
  w = q + 3;
  wp = qp + 3;
  qp[0] = w[0];
  qp[1] = w[1];
  qp[2] = w[2];
  qp[6] = 1;
  qp[7] = 0;

  /* find field components */
  F0 = 0; /* z component of field. */
  if (bmapxy->points) {
    ix = (x - bmapxy->xmin) / bmapxy->dx;
    iy = (y - bmapxy->ymin) / bmapxy->dy;
    if (ix < 0 || iy < 0 || ix > bmapxy->nx - 1 || iy > bmapxy->ny - 1) {
      F1 = F2 = 0;
      printf("invalid particle: x=%e, y=%e, ix=%ld, iy=%ld\n",
             x, y, ix, iy);
      n_invalid_particles++;
      wp[0] = wp[1] = wp[2] = 0;
    } else {
      fx = (x - (ix * bmapxy->dx + bmapxy->xmin)) / bmapxy->dx;
      fy = (y - (iy * bmapxy->dy + bmapxy->ymin)) / bmapxy->dy;
      Fa = (1 - fy) * bmapxy->Fx[ix + iy * bmapxy->nx] + fy * bmapxy->Fx[ix + (iy + 1) * bmapxy->nx];
      Fb = (1 - fy) * bmapxy->Fx[ix + 1 + iy * bmapxy->nx] + fy * bmapxy->Fx[ix + 1 + (iy + 1) * bmapxy->nx];
      F1 = (1 - fx) * Fa + fx * Fb;
      Fa = (1 - fy) * bmapxy->Fy[ix + iy * bmapxy->nx] + fy * bmapxy->Fy[ix + (iy + 1) * bmapxy->nx];
      Fb = (1 - fy) * bmapxy->Fy[ix + 1 + iy * bmapxy->nx] + fy * bmapxy->Fy[ix + 1 + (iy + 1) * bmapxy->nx];
      F2 = (1 - fx) * Fa + fx * Fb;
      F1 *= bmapxy->strength;
      F2 *= bmapxy->strength;
      /* compute lorentz force */
      wp[0] = (w[1] * F2 - w[2] * F1) / (1 + q[7]);
      wp[1] = (w[2] * F0 - w[0] * F2) / (1 + q[7]);
      wp[2] = (w[0] * F1 - w[1] * F0) / (1 + q[7]);
      if (bmapxy->BGiven) {
        for (ix = 0; ix < 3; ix++)
          wp[ix] *= -particleCharge * particleRelSign / (particleMass * c_mks * P0);
      }
    }
  } else {
    rpn_clear();
    rpn_store(x, NULL, bmapxy->rpnMem[0]);
    rpn_store(y, NULL, bmapxy->rpnMem[1]);
    F1 = bmapxy->strength * rpn(bmapxy->FxRpn);
    F2 = bmapxy->strength * rpn(bmapxy->FyRpn);
  }
#ifdef DEBUG
  if (fp_field && field_output_on) {
    fprintf(fp_field, "%21.15e %21.15e %21.15e %21.15e %21.15e %21.15e %21.15e %21.15e\n", q[0], q[1], q[2], 0.0, s, F0, F1, F2);
    fflush(fp_field);
  }
#endif
}

void bmapxy_coord_transform(double *q, double *coord, void *field, long which_end) {
  double dzds, *dqds;
  /* BMAPXY *bmapxy = field; */

  dqds = q + 3;
  if (which_end == -1) {
    /* transform coord (x, x', y, y', s, dp/p) into q (q0,q1,q2,d/ds(q0,q1,q2),s,dp/p) */
    /* convert slopes to normalized velocities */
    dqds[0] = dzds = 1 / sqrt(1 + sqr(coord[1]) + sqr(coord[3]));
    dqds[1] = coord[1] * dzds;
    dqds[2] = coord[3] * dzds;
    /* find q coordinates of particle at reference plane */
    q[0] = 0;
    q[1] = coord[0];
    q[2] = coord[2];
    q[6] = coord[4];
    q[7] = coord[5];
  } else {
    /* transform q into coord at exit refence plane */
    coord[0] = q[1];
    coord[1] = dqds[1] / dqds[0];
    coord[2] = q[2];
    coord[3] = dqds[2] / dqds[0];
    coord[4] = q[6];
    coord[5] = q[7];
  }
}

void bmapxy_field_setup(BMAPXY *bmapxy) {
  SDDS_DATASET SDDSin;
  double *x = NULL, *y = NULL, *Fx = NULL, *Fy = NULL;
  long nx;

  if (!fexists(bmapxy->filename)) {
    printf("file %s not found for BMAPXY element\n", bmapxy->filename);
    fflush(stdout);
    exitElegant(1);
  }
  if (!SDDS_InitializeInputFromSearchPath(&SDDSin, bmapxy->filename) ||
      SDDS_ReadPage(&SDDSin) <= 0 ||
      !(x = SDDS_GetColumnInDoubles(&SDDSin, "x")) || !(y = SDDS_GetColumnInDoubles(&SDDSin, "y"))) {
    SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
  }
  if (!check_sdds_column(&SDDSin, "x", "m") ||
      !check_sdds_column(&SDDSin, "y", "m")) {
    fprintf(stderr, "BMAPXY input file must have x and y in m (meters)\n");
    exitElegant(1);
  }
  bmapxy->BGiven = 0;
  if (!(Fx = SDDS_GetColumnInDoubles(&SDDSin, "Fx")) || !(Fy = SDDS_GetColumnInDoubles(&SDDSin, "Fy"))) {
    if (!(Fx = SDDS_GetColumnInDoubles(&SDDSin, "Bx")) || !(Fy = SDDS_GetColumnInDoubles(&SDDSin, "By"))) {
      fprintf(stderr, "BMAPXY input file must have both (Fx, Fy) or both (Bx, By)\n");
      exitElegant(1);
    }
    bmapxy->BGiven = 1;
    if (!check_sdds_column(&SDDSin, "Bx", "T") ||
        !check_sdds_column(&SDDSin, "By", "T")) {
      fprintf(stderr, "BMAPXY input file must have Bx and By in T (Tesla)\n");
      exitElegant(1);
    }
  } else {
    if (!check_sdds_column(&SDDSin, "Fx", "") ||
        !check_sdds_column(&SDDSin, "Fy", "")) {
      fprintf(stderr, "BMAPXY input file must have Fx and Fy with no units\n");
      exitElegant(1);
    }
  }

  if (!(bmapxy->points = SDDS_CountRowsOfInterest(&SDDSin)) || bmapxy->points < 2) {
    printf("file %s for BMAPXY element has insufficient data\n", bmapxy->filename);
    fflush(stdout);
    exitElegant(1);
  }
  SDDS_Terminate(&SDDSin);

  /* It is assumed that the data is ordered so that x changes fastest.
   * This can be accomplished with sddssort -column=y,incr -column=x,incr
   * The points are assumed to be equipspaced.
   */
  nx = 1;
  bmapxy->xmin = x[0];
  while (nx < bmapxy->points) {
    if (x[nx - 1] > x[nx])
      break;
    nx++;
  }
  bmapxy->xmax = x[nx - 1];
  bmapxy->dx = (bmapxy->xmax - bmapxy->xmin) / (nx - 1);
  if ((bmapxy->nx = nx) <= 1 || y[0] > y[nx] || (bmapxy->ny = bmapxy->points / nx) <= 1) {
    printf("file %s for BMAPXY element doesn't have correct structure or amount of data\n",
           bmapxy->filename);
    fflush(stdout);
    printf("nx = %ld, ny=%ld\n", bmapxy->nx, bmapxy->ny);
    fflush(stdout);
    exitElegant(1);
  }
  bmapxy->ymin = y[0];
  bmapxy->ymax = y[bmapxy->points - 1];
  bmapxy->dy = (bmapxy->ymax - bmapxy->ymin) / (bmapxy->ny - 1);
  printf("BMAPXY element from file %s: nx=%ld, ny=%ld, dx=%e, dy=%e, x:[%e, %e], y:[%e, %e]\n",
         bmapxy->filename, bmapxy->nx, bmapxy->ny, bmapxy->dx, bmapxy->dy,
         bmapxy->xmin, bmapxy->xmax,
         bmapxy->ymin, bmapxy->ymax);
  free(x);
  free(y);
  bmapxy->Fx = Fx;
  bmapxy->Fy = Fy;
}

void engeFunction(double *Fy, double *Fz, double z, double y, double D, double a1, double a2, double a3, long order) {
  double z2;
  double t1, t2, t3, t4, t5, t6, t7, t8, t9;
  double t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20;
  double t21, t22, t23;
  double b1, b2, b3;
  double y2, y3, y4, y5;

  b1 = a1;
  b2 = a2 / D;
  b3 = a3 / sqr(D);

  if (order > 3) {
    y2 = y * y;
    y3 = y2 * y;
    y4 = y2 * y2;
    y5 = y4 * y;

    z2 = z * z;
    t1 = b3 * z2 + b2 * z + b1;
    t3 = exp(t1);
    t2 = t3 + 1;
    t4 = t3 * t3;
    t5 = t4 * t3;
    t6 = t4 * t4;
    t22 = t4 * t3;
    t7 = 2 * b3 * z + b2;
    t8 = t7 * t7;
    t9 = t8 * t7;
    t10 = t8 * t8;
    t11 = t10 * t7;
    t12 = ipow(t2, -2);
    t13 = t12 / t2;
    t14 = t13 / t2;
    t15 = 4 * b3 * z + 2 * b2;
    t16 = t15 * t15;
    t17 = 6 * b3 * z + 3 * b2;
    t18 = t14 / t2;
    t19 = ipow2(b3);
    t20 = t17 * t17;
    t21 = t16 * t15;
    t23 = t18 / t2;

    *Fy = y4 * (2 * t4 * t13 * t8 * t16 - t3 * t12 * t10 + 2 * t4 * t13 * t10 - 6 * t5 * t14 * t10 + 24 * t6 * t18 * t10 - 6 * t5 * t14 * t17 * t9 + 2 * t4 * t13 * t15 * t9 - 6 * t5 * t14 * t15 * t9 - 12 * b3 * t3 * t12 * t8 + 32 * b3 * t4 * t13 * t8 - 72 * b3 * t5 * t14 * t8 + 20 * b3 * t4 * t13 * t7 * t15 - 12 * t19 * t3 * t12 + 24 * t19 * t4 * t13) / 24.0 - y2 * (-t3 * t12 * t8 + 2 * t4 * t13 * t8 - 2 * b3 * t3 * t12) / 2.0 + 1 / t2;

    *Fz = y5 * (-6 * t5 * t14 * t9 * t20 + 2 * t4 * t13 * t8 * t21 + 2 * t4 * t13 * t9 * t16 - 6 * t5 * t14 * t9 * t16 + 28 * b3 * t4 * t13 * t7 * t16 - t3 * t12 * t11 + 2 * t4 * t13 * t11 - 6 * t5 * t14 * t11 + 24 * t6 * t18 * t11 - 120 * t22 * t23 * t11 + 24 * t6 * t18 * (8 * b3 * z + 4 * b2) * t10 - 6 * t5 * t14 * t17 * t10 + 24 * t6 * t18 * t17 * t10 + 2 * t4 * t13 * t15 * t10 - 6 * t5 * t14 * t15 * t10 + 24 * t6 * t18 * t15 * t10 - 6 * t5 * t14 * t15 * t17 * t9 - 20 * b3 * t3 * t12 * t9 + 48 * b3 * t4 * t13 * t9 - 204 * b3 * t5 * t14 * t9 + 480 * b3 * t6 * t18 * t9 - 108 * b3 * t5 * t14 * t17 * t8 + 60 * b3 * t4 * t13 * t15 * t8 - 96 * b3 * t5 * t14 * t15 * t8 + 64 * t19 * t4 * t13 * t15 - 60 * t19 * t3 * t12 * t7 + 232 * t19 * t4 * t13 * t7 - 360 * t19 * t5 * t14 * t7) / 120.0 - y3 * (-t3 * t12 * t9 + 2 * t4 * t13 * t9 - 6 * t5 * t14 * t9 + 2 * t4 * t13 * t15 * t8 - 6 * b3 * t3 * t12 * t7 + 12 * b3 * t4 * t13 * t7) / 6.0 - t3 * t12 * y * t7;
  } else if (order > 1) {
    z2 = z * z;
    y2 = y * y;
    y3 = y2 * y;

    t1 = b3 * z2 + b2 * z + b1;
    t2 = exp(t1);
    t3 = 1 / (t2 + 1);
    t4 = t3 * t3;
    t5 = t3 * t4;
    t6 = t4 * t4;
    t7 = 2 * b3 * z + b2;
    t8 = t7 * t7;
    t9 = 2 * t7;
    t10 = t2 * t2;
    t11 = t8 * t7;
    t12 = t2 * t10;

    *Fy = t3 - y2 * (-t2 * t4 * t8 + 2 * t10 * t5 * t8 - 2 * b3 * t2 * t4) / 2.0;

    *Fz = -y3 * (-t2 * t4 * t11 + 2 * t10 * t5 * t11 - 6 * t12 * t6 * t11 + 2 * t10 * t5 * t9 * t8 - 6 * b3 * t2 * t4 * t7 + 12 * b3 * t10 * t5 * t7) / 6.0 - t2 * t4 * y * t7;

  } else {
    z2 = z * z;
    t1 = b3 * z2 + b2 * z + b1;
    t2 = exp(t1);
    t3 = 1 / (t2 + 1);
    t4 = t3 * t3;
    t7 = 2 * b3 * z + b2;

    *Fy = t3;
    *Fz = -t2 * t4 * y * t7;
  }
}

void writeEngeFunctionToFile(char *filename) {
  FILE *fp;
  double dz, z;
  double Fy, Fz;

  fp = fopen_e(filename, "w", 0);
  fprintf(fp, "SDDS1\n");
  fprintf(fp, "&column name=z units=m type=double &end\n");
  fprintf(fp, "&column name=Fy type=double &end\n");
  fprintf(fp, "&data mode=ascii no_row_counts=1 &end\n");
  z = -flen;
  dz = flen / 100;
  while (z < flen) {
    engeFunction(&Fy, &Fz, z, 0.0, engeD, engeCoef[0], engeCoef[1], engeCoef[2], 1);
    fprintf(fp, "%21.15e %21.15e\n", z, Fy);
    z += dz;
  }
  fclose(fp);
}

double bmapxyz_exit_function(double *qp, double *q, double s) {
  /* static BMAPXYZ *bmapxyz; */
  /* bmapxyz = (BMAPXYZ*)field_global; */

  /* returns positive if inside or before, negative if after */
  return (central_length - q[0]);
}

void bmapxyz_deriv_function(double *qp, double *q, double s) {
  double *w, *wp;
  double F0, F1, F2; /* z, x, y */
  double x, y, z;
  BMAPXYZ *bmapxyz;
  long ix;

  bmapxyz = (BMAPXYZ *)field_global;
  n_deriv_calls++;

  z = q[0];
  x = q[1];
  y = q[2];

  if (isnan(x) || isnan(y) || isinf(x) || isinf(y)) {
    for (ix = 0; ix < 7; ix++)
      qp[ix] = 0;
    qp[6] = 1;
    q[0] = central_length;
    n_invalid_particles++;
    return;
  }
  if (!isLost) {
    double zOffset, dzHardEdge;
    double xp, yp, zs;
    APCONTOUR *apc;
    zOffset = (bmapxyz->fieldLength - bmapxyz->length) / 2;
    apc = apertureData.apContour;
    if (bmapxyz->data->magnetSymmetry[2])
      /* the field map starts from z=0 instead of z=-fieldLength/2 */
      zs = z - bmapxyz->fieldLength/2 + bmapxyz->data->zmin; 
    else
      zs = z + bmapxyz->data->zmin;
    if (zs<bmapxyz->zMinApContour || zs>bmapxyz->zMaxApContour)
      apertureData.apContour = NULL;
    if (!checkMultAperture(x, y, z, &apertureData)) {
      isLost = 1;
      dzHardEdge = z - zOffset;
    } else if ((dzHardEdge = z - zOffset) >= 0 && dzHardEdge <= bmapxyz->length) {
      xp = q[4] / q[3];
      yp = q[5] / q[3];
      if (insideObstruction_xyz(x, xp, y, yp, lastParticleID, lostParticleCoordinate + 9,
                                bmapxyz->tilt, GLOBAL_LOCAL_MODE_DZ, dzHardEdge, 0, 0))
        isLost = 1;
    }
    apertureData.apContour = apc;
    if (isLost) {
      /*static FILE *fp; */
      TRACKING_CONTEXT tcontext;
      getTrackingContext(&tcontext);
      /* 
         if (!fp) {
         #if USE_MPI
            char buffer[256];
            sprintf(buffer, "lorentz.los-%03d", myid);
            fp = fopen_e(buffer, "w", 0);
#else
            fp = fopen_e("lorentz.los", "w", 0);
#endif
            fprintf(fp, "SDDS1\n&column name=ElementName type=string &end\n&column name=z type=double units=m &end\n");
            fprintf(fp, "&column name=dzHardEdge type=double units=m &end\n");
            fprintf(fp, "&column name=s type=double units=m &end\n");
            fprintf(fp, "&column name=x type=double units=m &end\n");
            fprintf(fp, "&column name=wx type=double &end\n");
            fprintf(fp, "&column name=particleID, type=long &end\n");
            fprintf(fp, "&data mode=ascii no_row_counts=1 &end\n");
          }
          fprintf(fp, "%s %le %le %le %le %le %ld\n", 
                  tcontext.elementName, tcontext.zStart+dzHardEdge, dzHardEdge, q[6], x, q[4], lastParticleID);
          */
      for (ix = 0; ix < 8; ix++)
        lostParticleCoordinate[ix] = q[ix];
      lostParticleCoordinate[ix] = tcontext.zStart + dzHardEdge;
      isLost = 1;
    }
  }

  /* w is the velocity */
  w = q + 3;
  wp = qp + 3;
  qp[0] = w[0];
  qp[1] = w[1];
  qp[2] = w[2];
  qp[6] = 1; /* path length derivative */
  qp[7] = 0; /* momentum offset doesn't change except from SR (below) */

  /* find field components */
  if (!interpolate_bmapxyz(&F0, &F1, &F2, bmapxyz, x, y, z)) {
    wp[0] = wp[1] = wp[2] = 0;
  } else {
    /* compute lorentz force */
    wp[0] = (w[1] * F2 - w[2] * F1) / (1 + q[7]);
    wp[1] = (w[2] * F0 - w[0] * F2) / (1 + q[7]);
    wp[2] = (w[0] * F1 - w[1] * F0) / (1 + q[7]);
    if (bmapxyz->data->BGiven) {
      for (ix = 0; ix < 3; ix++)
        wp[ix] *= -particleCharge * particleRelSign / (particleMass * c_mks * P0);
    }
  }

  if (rad_coef)
    qp[7] = -rad_coef * pow4(1 + q[7]) * (sqr(wp[0]) + sqr(wp[1]) + sqr(wp[2]));

#if !USE_MPI
  if (bmapxyz->SDDSpo) {
    if (bmapxyz->poRow == 0 &&
        !SDDS_StartPage(bmapxyz->SDDSpo, bmapxyz->poRows = 1000)) {
      SDDS_SetError("Problem setting up particle output page for BMXYZ");
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
    }
    if (bmapxyz->poRow >= bmapxyz->poRows &&
        !SDDS_LengthenTable(bmapxyz->SDDSpo, (bmapxyz->poRows += 1000))) {
      SDDS_SetError("Problem lenthening particle output table for BMXYZ");
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
    }
    if (!SDDS_SetRowValues(bmapxyz->SDDSpo, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE,
                           bmapxyz->poRow++,
                           bmapxyz->poIndex[0], q[1],
                           bmapxyz->poIndex[1], qp[1],
                           bmapxyz->poIndex[2], q[2],
                           bmapxyz->poIndex[3], qp[2],
                           bmapxyz->poIndex[4], q[0],
                           bmapxyz->poIndex[5], qp[0],
                           bmapxyz->poIndex[6], F1,
                           bmapxyz->poIndex[7], F2,
                           bmapxyz->poIndex[8], F0,
                           -1)) {
      SDDS_SetError("Problem setting rows in particle output file for BMXYZ");
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
    }
  }
#endif

#ifdef DEBUG
  if (fp_field && field_output_on) {
    fprintf(fp_field, "%21.15e %21.15e %21.15e %21.15e %21.15e %21.15e %21.15e %21.15e\n", q[0], q[1], q[2], 0.0, s, F0, F1, F2);
    fflush(fp_field);
  }
#endif
}

void bmapxyz_coord_transform(double *q, double *coord, void *field, long which_end) {
  double dzds, *dqds;
  BMAPXYZ *bmapxyz = field;

  dqds = q + 3;
  if (which_end == -1) {
    if (bmapxyz->dxError || bmapxyz->dyError || bmapxyz->dzError)
      offsetBeamCoordinatesForMisalignment(&coord, 1, bmapxyz->dxError, bmapxyz->dyError, bmapxyz->dzError);
    if (bmapxyz->tilt)
      rotateBeamCoordinatesForMisalignment(&coord, 1, bmapxyz->tilt);
    /* transform coord (x, x', y, y', s, dp/p) into q (q0,q1,q2,d/ds(q0,q1,q2),s,dp/p) */
    /* convert slopes to normalized velocities */
    dqds[0] = dzds = 1 / sqrt(1 + sqr(coord[1]) + sqr(coord[3]));
    dqds[1] = coord[1] * dzds;
    dqds[2] = coord[3] * dzds;
    /* find q coordinates of particle at reference plane */
    q[0] = 0;
    q[1] = coord[0];
    q[2] = coord[2];
    q[6] = coord[4];
    q[7] = coord[5];
  } else {
    /* transform q into coord at exit refence plane */
    coord[0] = q[1];
    coord[1] = dqds[1] / dqds[0];
    coord[2] = q[2];
    coord[3] = dqds[2] / dqds[0];
    coord[4] = q[6];
    coord[5] = q[7];
    if (bmapxyz->tilt)
      rotateBeamCoordinatesForMisalignment(&coord, 1, -bmapxyz->tilt);
    if (bmapxyz->dxError || bmapxyz->dyError || bmapxyz->dzError)
      offsetBeamCoordinatesForMisalignment(&coord, 1, -bmapxyz->dxError, -bmapxyz->dyError, -bmapxyz->dzError);
  }
}

void bmapxyz_field_setup(BMAPXYZ *bmapxyz) {
  SDDS_DATASET SDDSin;
  long i, nx, ny, imap;
  BMAPXYZ_DATA *data;
  static char *symmetryType[3] = {"none", "odd", "even"};

  for (imap = 0; imap < nStoredBmapxyzData; imap++) {
    if (strcmp(bmapxyz->filename, storedBmapxyzData[imap].filename) == 0)
      break;
  }
  iStoredBmapxyzData = imap;
  if (imap < nStoredBmapxyzData) {
    bmapxyz->data = storedBmapxyzData[imap].data;
    bmapxyz->fieldLength = storedBmapxyzData[imap].fieldLength;
    bmapxyz->singlePrecision = storedBmapxyzData[imap].singlePrecision;
    return;
  }
 
  /*
  if (!fexists(bmapxyz->filename)) {
    printf("file %s not found for BMAPXYZ element\n", bmapxyz->filename);
    fflush(stdout);
    exitElegant(1);
  }
  */

  
  storedBmapxyzData = SDDS_Realloc(storedBmapxyzData, sizeof(*storedBmapxyzData) * (nStoredBmapxyzData + 1));
  bmapxyz->data = data = storedBmapxyzData[imap].data = (BMAPXYZ_DATA *)tmalloc(sizeof(*data));
  nStoredBmapxyzData++;
  cp_str(&(storedBmapxyzData[imap].filename), bmapxyz->filename);

  printf("Reading BMXYZ field data from file %s\n", bmapxyz->filename);
  fflush(stdout);
  if (!bmapxyz->singlePrecision) {
    double *x = NULL, *y = NULL, *z = NULL, *Fx = NULL, *Fy = NULL, *Fz = NULL;
    printf("Data from file %s will be stored in double precision\n", bmapxyz->filename);
    fflush(stdout);
    if (!SDDS_InitializeInputFromSearchPath(&SDDSin, bmapxyz->filename) ||
        SDDS_ReadPage(&SDDSin) <= 0) {
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
    }
    SDDS_SetColumnMemoryMode(&SDDSin, DONT_TRACK_COLUMN_MEMORY_AFTER_ACCESS);
    printf("Checking BMXYZ field data from file %s\n", bmapxyz->filename);
    fflush(stdout);
    if (!check_sdds_column(&SDDSin, "x", "m") ||
        !check_sdds_column(&SDDSin, "y", "m") ||
        !check_sdds_column(&SDDSin, "z", "m")) {
      printf("BMAPXYZ input file must have x, y, and z in m (meters)\n");
      exitElegant(1);
    }

    if (!(data->points = SDDS_CountRowsOfInterest(&SDDSin)) || data->points < 2) {
      printf("file %s for BMAPXYZ element has insufficient data\n", bmapxyz->filename);
      fflush(stdout);
      exitElegant(1);
    }
    printf("File %s for BMAPXYZ element has %ld points\n", bmapxyz->filename, data->points);
    fflush(stdout);

    /* It is assumed that the data is ordered so that x changes fastest.
     * This can be accomplished with sddssort -column=z,incr -column=y,incr -column=x,incr
     * The points are assumed to be equipspaced.
     */
    if (!(x = SDDS_GetColumnInDoubles(&SDDSin, "x")))
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
    nx = 1;
    data->xmin = x[0];
    while (nx < data->points) {
      if (x[nx - 1] > x[nx])
        break;
      nx++;
    }
    if (nx == data->points) {
      printf("file %s for BMAPXYZ element doesn't have correct structure or amount of data (x)\n",
             bmapxyz->filename);
      printf("Use sddssort -column=z -column=y -column=x to sort the file\n");
      exitElegant(1);
    }
    data->xmax = x[nx - 1];
    data->dx = (data->xmax - data->xmin) / (nx - 1);
    free(x);
    
    if (!(y = SDDS_GetColumnInDoubles(&SDDSin, "y")))
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
    ny = 1;
    data->ymin = y[0];
    while (ny < (data->points / nx)) {
      if (y[(ny - 1) * nx] > y[ny * nx])
        break;
      ny++;
    }
    if (ny == data->points / nx) {
      printf("file %s for BMAPXYZ element doesn't have correct structure or amount of data (y)\n",
             bmapxyz->filename);
      printf("Use sddssort -column=z -column=y -column=x to sort the file\n");
      exitElegant(1);
    }
    data->ymax = y[(ny - 1) * nx];
    data->dy = (data->ymax - data->ymin) / (ny - 1);
    free(y);
    
    if (!(z = SDDS_GetColumnInDoubles(&SDDSin, "z")))
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
    if ((data->nx = nx) <= 1 || (data->ny = ny) <= 1 || (data->nz = data->points / (nx * ny)) <= 1 ||
        (data->nx * data->ny * data->nz != data->points)) {
      printf("file %s for BMAPXYZ element doesn't have correct structure or amount of data\n",
             bmapxyz->filename);
      printf("nx = %ld, ny=%ld, nz=%ld, points=%ld\n", data->nx, data->ny, data->nz, data->points);
      exitElegant(1);
    }
    data->zmin = z[0];
    data->zmax = z[data->points - 1];
    data->dz = (data->zmax - data->zmin) / (data->nz - 1);
    free(z);
    printf("BMAPXYZ element from file %s: nx=%ld, ny=%ld, nz=%ld\ndx=%e, dy=%e, dz=%e\nx:[%e, %e], y:[%e, %e], z:[%e, %e]\n",
           bmapxyz->filename,
           data->nx, data->ny, data->nz,
           data->dx, data->dy, data->dz,
           data->xmin, data->xmax,
           data->ymin, data->ymax,
           data->zmin, data->zmax);
    fflush(stdout);
    if (data->zmin < 0) {
      if (bmapxyz->injectAtZero) {
        printWarningForTracking("zmin<0 in BMAPXYZ data and INJECT_AT_Z0 is non-zero.",
                                "Particles will start inside the element. The insertion length (L) and any difference from the field length is ignored.");
      } else {
        data->zmax = data->zmax - data->zmin;
        data->zmin = 0;
      }
    }

    data->BGiven = 0;
    if (!(Fx = SDDS_GetColumnInDoubles(&SDDSin, "Fx")) || !(Fy = SDDS_GetColumnInDoubles(&SDDSin, "Fy")) ||
        !(Fz = SDDS_GetColumnInDoubles(&SDDSin, "Fz"))) {
      if (!(Fx = SDDS_GetColumnInDoubles(&SDDSin, "Bx")) || !(Fy = SDDS_GetColumnInDoubles(&SDDSin, "By")) ||
          !(Fz = SDDS_GetColumnInDoubles(&SDDSin, "Bz"))) {
        fprintf(stderr, "BMAPXYZ input file must have (Fx, Fy, Fz) (dimensionless) or (Bx, By, Bz) (in T)\n");
        exitElegant(1);
      }
      data->BGiven = 1;
      if (!check_sdds_column(&SDDSin, "Bx", "T") ||
          !check_sdds_column(&SDDSin, "By", "T") ||
          !check_sdds_column(&SDDSin, "Bz", "T")) {
        fprintf(stderr, "BMAPXYZ input file must have Bx, By, and Bz in T (Tesla)\n");
        exitElegant(1);
      }
    } else {
      if (!check_sdds_column(&SDDSin, "Fx", "") ||
          !check_sdds_column(&SDDSin, "Fy", "") ||
          !check_sdds_column(&SDDSin, "Fz", "")) {
        fprintf(stderr, "BMAPXYZ input file must have Fx, Fy, and Fz with no units\n");
        exitElegant(1);
      }
    }

    data->Fx1 = data->Fy1 = data->Fz1 = NULL;
    data->Fx = Fx;
    data->Fy = Fy;
    data->Fz = Fz;
  } else {
    /* single precision */
    float *x = NULL, *y = NULL, *z = NULL, *Fx = NULL, *Fy = NULL, *Fz = NULL;
    printf("Data from file %s will be stored in single precision\n", bmapxyz->filename);
    fflush(stdout);
    if (!SDDS_InitializeInputFromSearchPath(&SDDSin, bmapxyz->filename) ||
        SDDS_ReadPage(&SDDSin) <= 0) {
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
    }
    SDDS_SetColumnMemoryMode(&SDDSin, DONT_TRACK_COLUMN_MEMORY_AFTER_ACCESS);
    printf("Checking BMXYZ field data from file %s\n", bmapxyz->filename);
    fflush(stdout);
    if (!check_sdds_column(&SDDSin, "x", "m") ||
        !check_sdds_column(&SDDSin, "y", "m") ||
        !check_sdds_column(&SDDSin, "z", "m")) {
      printf("BMAPXYZ input file must have x, y, and z in m (meters)\n");
      exitElegant(1);
    }

    if (!(data->points = SDDS_CountRowsOfInterest(&SDDSin)) || data->points < 2) {
      printf("file %s for BMAPXYZ element has insufficient data\n", bmapxyz->filename);
      fflush(stdout);
      exitElegant(1);
    }
    printf("File %s for BMAPXYZ element has %ld points\n", bmapxyz->filename, data->points);
    fflush(stdout);
    
    /* It is assumed that the data is ordered so that x changes fastest.
     * This can be accomplished with sddssort -column=z,incr -column=y,incr -column=x,incr
     * The points are assumed to be equipspaced.
     */
    if (!(x = SDDS_GetColumnInFloats(&SDDSin, "x")))
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
    nx = 1;
    data->xmin = x[0];
    while (nx < data->points) {
      if (x[nx - 1] > x[nx])
        break;
      nx++;
    }
    if (nx == data->points) {
      printf("file %s for BMAPXYZ element doesn't have correct structure or amount of data (x)\n",
             bmapxyz->filename);
      printf("Use sddssort -column=z -column=y -column=x to sort the file\n");
      exitElegant(1);
    }
    data->xmax = x[nx - 1];
    data->dx = (data->xmax - data->xmin) / (nx - 1);
    free(x);
    
    if (!(y = SDDS_GetColumnInFloats(&SDDSin, "y")))
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
    ny = 1;
    data->ymin = y[0];
    while (ny < (data->points / nx)) {
      if (y[(ny - 1) * nx] > y[ny * nx])
        break;
      ny++;
    }
    if (ny == data->points / nx) {
      printf("file %s for BMAPXYZ element doesn't have correct structure or amount of data (y)\n",
             bmapxyz->filename);
      printf("Use sddssort -column=z -column=y -column=x to sort the file\n");
      exitElegant(1);
    }
    data->ymax = y[(ny - 1) * nx];
    data->dy = (data->ymax - data->ymin) / (ny - 1);
    free(y);
    
    if (!(z = SDDS_GetColumnInFloats(&SDDSin, "z")))
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
    if ((data->nx = nx) <= 1 || (data->ny = ny) <= 1 || (data->nz = data->points / (nx * ny)) <= 1 ||
        (data->nx * data->ny * data->nz != data->points)) {
      printf("file %s for BMAPXYZ element doesn't have correct structure or amount of data\n",
             bmapxyz->filename);
      printf("nx = %ld, ny=%ld, nz=%ld, points=%ld\n", data->nx, data->ny, data->nz, data->points);
      exitElegant(1);
    }
    data->zmin = z[0];
    data->zmax = z[data->points - 1];
    data->dz = (data->zmax - data->zmin) / (data->nz - 1);
    free(z);
    printf("BMAPXYZ element from file %s: nx=%ld, ny=%ld, nz=%ld\ndx=%e, dy=%e, dz=%e\nx:[%e, %e], y:[%e, %e], z:[%e, %e]\n",
           bmapxyz->filename,
           data->nx, data->ny, data->nz,
           data->dx, data->dy, data->dz,
           data->xmin, data->xmax,
           data->ymin, data->ymax,
           data->zmin, data->zmax);
    fflush(stdout);
    if (data->zmin < 0) {
      if (bmapxyz->injectAtZero) {
        printWarningForTracking("zmin<0 in BMAPXYZ data and INJECT_AT_Z0 is non-zero.",
                                "Particles will start inside the element. The insertion length (L) and any difference from the field length is ignored.");
      } else {
        data->zmax = data->zmax - data->zmin;
        data->zmin = 0;
      }
    }

    data->BGiven = 0;
    if (!(Fx = SDDS_GetColumnInFloats(&SDDSin, "Fx")) || !(Fy = SDDS_GetColumnInFloats(&SDDSin, "Fy")) ||
        !(Fz = SDDS_GetColumnInFloats(&SDDSin, "Fz"))) {
      if (!(Fx = SDDS_GetColumnInFloats(&SDDSin, "Bx")) || !(Fy = SDDS_GetColumnInFloats(&SDDSin, "By")) ||
          !(Fz = SDDS_GetColumnInFloats(&SDDSin, "Bz"))) {
        fprintf(stderr, "BMAPXYZ input file must have (Fx, Fy, Fz) (dimensionless) or (Bx, By, Bz) (in T)\n");
        exitElegant(1);
      }
      data->BGiven = 1;
      if (!check_sdds_column(&SDDSin, "Bx", "T") ||
          !check_sdds_column(&SDDSin, "By", "T") ||
          !check_sdds_column(&SDDSin, "Bz", "T")) {
        fprintf(stderr, "BMAPXYZ input file must have Bx, By, and Bz in T (Tesla)\n");
        exitElegant(1);
      }
    } else {
      if (!check_sdds_column(&SDDSin, "Fx", "") ||
          !check_sdds_column(&SDDSin, "Fy", "") ||
          !check_sdds_column(&SDDSin, "Fz", "")) {
        fprintf(stderr, "BMAPXYZ input file must have Fx, Fy, and Fz with no units\n");
        exitElegant(1);
      }
    }

    data->Fx = data->Fy = data->Fz = NULL;
    data->Fx1 = Fx;
    data->Fy1 = Fy;
    data->Fz1 = Fz;
  }

  for (i=0; i<3; i++) { /* x, y, z */
    char *parameterName[3] = {"xSymmetry", "ySymmetry", "zSymmetry"};
    char *symmetryString;
    bmapxyz->data->magnetSymmetry[i] = 0;
    if (SDDS_GetParameterIndex(&SDDSin, parameterName[i])>=0) {
      if (SDDS_CheckParameter(&SDDSin, parameterName[i], NULL, SDDS_STRING, stdout)!=SDDS_CHECK_OK)
        bombElegantVA("Error: parameter %s in BMXYZ file %s does not have string type\n",
                      parameterName[i], bmapxyz->filename);
      if (!SDDS_GetParameter(&SDDSin, parameterName[i], (void*)&symmetryString)) 
        bombElegantVA("Problem with %s parameter in BRAT file %s: not string type \n", parameterName[i], bmapxyz->filename);
      if ((bmapxyz->data->magnetSymmetry[i] = match_string(symmetryString, symmetryType, 3, 0))<0)
	bombElegantVA("Problem with %s parameter in BRAT file %s: value %s not recognized\n",
                      parameterName[i], bmapxyz->filename, symmetryString);
      printf("Recognized magnet %c symmetry of %s\n",
             parameterName[i][0], symmetryType[bmapxyz->data->magnetSymmetry[i]]);
      free(symmetryString);
    }
  }

  SDDS_Terminate(&SDDSin);

  if (bmapxyz->fieldLength > 0) {
    if (!bmapxyz->data->magnetSymmetry[2]) {
      if (fabs(bmapxyz->fieldLength - (data->zmax - data->zmin)) / bmapxyz->fieldLength > 1e-7) {
	fprintf(stderr, "Error: Mismatch of LFIELD (%21.15e) and data from file %s (range %21.15le)\n",
		bmapxyz->fieldLength, bmapxyz->filename, data->zmax - data->zmin);
	exit(1);
      }
    } else {
      if (fabs(bmapxyz->fieldLength - 2*data->zmax) / bmapxyz->fieldLength > 1e-7) {
	fprintf(stderr, "Error: Mismatch of LFIELD (%21.15e) and data from file %s (range %21.15le)\n",
		bmapxyz->fieldLength, bmapxyz->filename, data->zmax - data->zmin);
	exit(1);
      }
    }
  } else {
    bmapxyz->fieldLength = data->zmax - data->zmin;
    if (bmapxyz->data->magnetSymmetry[2])
      bmapxyz->fieldLength *= 2;
    printf("Set LFIELD for %s to %21.15e\n", bmapxyz->filename, bmapxyz->fieldLength);
    fflush(stdout);
  }
  storedBmapxyzData[imap].fieldLength = bmapxyz->fieldLength;
  storedBmapxyzData[imap].singlePrecision = bmapxyz->singlePrecision;

  if (bmapxyz->checkFields) {
    /* compute the maximum values of |div B| and |curl B| */
    double divB, curlB[3], magCurlB, maxDivB, maxCurlB;
    double B, maxB, maxDxyz;
    long ix, iy, iz;
    maxDivB = maxCurlB = maxB = 0;
    printf("Checking divergence and curl of fields...\n");
    fflush(stdout);
    for (ix = 1; ix < data->nx - 1; ix++) {
      for (iy = 1; iy < data->ny - 1; iy++) {
        for (iz = 1; iz < data->nz - 1; iz++) {
          if (!bmapxyz->singlePrecision)
            B = sqrt(sqr(data->Fx[ix + iy * data->nx + iz * data->nx * data->ny]) +
                     sqr(data->Fy[ix + iy * data->nx + iz * data->nx * data->ny]) +
                     sqr(data->Fz[ix + iy * data->nx + iz * data->nx * data->ny]));
          else
            B = sqrt(sqr(data->Fx1[ix + iy * data->nx + iz * data->nx * data->ny]) +
                     sqr(data->Fy1[ix + iy * data->nx + iz * data->nx * data->ny]) +
                     sqr(data->Fz1[ix + iy * data->nx + iz * data->nx * data->ny]));
          if (B > maxB)
            maxB = B;
          /* compute divergence */
          if (!bmapxyz->singlePrecision)
            divB = fabs(
              (data->Fx[ix + 1 + iy * data->nx + iz * data->nx * data->ny] -
               data->Fx[ix - 1 + iy * data->nx + iz * data->nx * data->ny]) /
                (2 * data->dx) +
              (data->Fy[ix + (iy + 1) * data->nx + iz * data->nx * data->ny] -
               data->Fy[ix + (iy - 1) * data->nx + iz * data->nx * data->ny]) /
                (2 * data->dy) +
              (data->Fz[ix + iy * data->nx + (iz + 1) * data->nx * data->ny] -
               data->Fz[ix + iy * data->nx + (iz - 1) * data->nx * data->ny]) /
                (2 * data->dz));
          else
            divB = fabs(
              (data->Fx1[ix + 1 + iy * data->nx + iz * data->nx * data->ny] -
               data->Fx1[ix - 1 + iy * data->nx + iz * data->nx * data->ny]) /
                (2 * data->dx) +
              (data->Fy1[ix + (iy + 1) * data->nx + iz * data->nx * data->ny] -
               data->Fy1[ix + (iy - 1) * data->nx + iz * data->nx * data->ny]) /
                (2 * data->dy) +
              (data->Fz1[ix + iy * data->nx + (iz + 1) * data->nx * data->ny] -
               data->Fz1[ix + iy * data->nx + (iz - 1) * data->nx * data->ny]) /
                (2 * data->dz));
          if (divB > maxDivB)
            maxDivB = divB;

          /* compute curl */
          if (!bmapxyz->singlePrecision) {
            curlB[0] =
              (data->Fz[ix + (iy + 1) * data->nx + iz * data->nx * data->ny] -
               data->Fz[ix + (iy - 1) * data->nx + iz * data->nx * data->ny]) /
                (2 * data->dy) -
              (data->Fy[ix + iy * data->nx + (iz + 1) * data->nx * data->ny] -
               data->Fy[ix + iy * data->nx + (iz - 1) * data->nx * data->ny]) /
                (2 * data->dz);
            curlB[1] =
              (data->Fx[ix + iy * data->nx + (iz + 1) * data->nx * data->ny] -
               data->Fx[ix + iy * data->nx + (iz - 1) * data->nx * data->ny]) /
                (2 * data->dz) -
              (data->Fz[ix + 1 + iy * data->nx + iz * data->nx * data->ny] -
               data->Fz[ix - 1 + iy * data->nx + iz * data->nx * data->ny]) /
                (2 * data->dx);
            curlB[2] =
              (data->Fy[ix + 1 + iy * data->nx + iz * data->nx * data->ny] -
               data->Fy[ix - 1 + iy * data->nx + iz * data->nx * data->ny]) /
                (2 * data->dx) -
              (data->Fx[ix + (iy + 1) * data->nx + iz * data->nx * data->ny] -
               data->Fx[ix + (iy - 1) * data->nx + iz * data->nx * data->ny]) /
                (2 * data->dy);
          } else {
            curlB[0] =
              (data->Fz1[ix + (iy + 1) * data->nx + iz * data->nx * data->ny] -
               data->Fz1[ix + (iy - 1) * data->nx + iz * data->nx * data->ny]) /
                (2 * data->dy) -
              (data->Fy1[ix + iy * data->nx + (iz + 1) * data->nx * data->ny] -
               data->Fy1[ix + iy * data->nx + (iz - 1) * data->nx * data->ny]) /
                (2 * data->dz);
            curlB[1] =
              (data->Fx1[ix + iy * data->nx + (iz + 1) * data->nx * data->ny] -
               data->Fx1[ix + iy * data->nx + (iz - 1) * data->nx * data->ny]) /
                (2 * data->dz) -
              (data->Fz1[ix + 1 + iy * data->nx + iz * data->nx * data->ny] -
               data->Fz1[ix - 1 + iy * data->nx + iz * data->nx * data->ny]) /
                (2 * data->dx);
            curlB[2] =
              (data->Fy1[ix + 1 + iy * data->nx + iz * data->nx * data->ny] -
               data->Fy1[ix - 1 + iy * data->nx + iz * data->nx * data->ny]) /
                (2 * data->dx) -
              (data->Fx1[ix + (iy + 1) * data->nx + iz * data->nx * data->ny] -
               data->Fx1[ix + (iy - 1) * data->nx + iz * data->nx * data->ny]) /
                (2 * data->dy);
          }

          magCurlB = sqrt(sqr(curlB[0]) + sqr(curlB[1]) + sqr(curlB[2]));
          if (magCurlB > maxCurlB)
            maxCurlB = magCurlB;
        }
      }
    }
    maxDxyz = data->dz;
    if (data->dy > maxDxyz)
      maxDxyz = data->dy;
    if (data->dz > maxDxyz)
      maxDxyz = data->dz;
    printf("Maximum |div B|: %le T/m\n", maxDivB);
    printf("Maximum |div B|/max|B|*max(dx,dy,dz): %le\n", maxDivB / maxB * maxDxyz);
    printf("Maximum |curl B|: %le T/m\n", maxCurlB);
    printf("Maximum |curl B|/max|B|*max(dx,dy,dz): %le\n", maxCurlB / maxB * maxDxyz);
    fflush(stdout);
  }

  printf("Field setup completed for data from file %s\n", bmapxyz->filename);
  fflush(stdout);

}

/*
static FILE *fpdeb = NULL;
*/

long interpolate_bmapxyz(double *F0, double *F1, double *F2,
                         BMAPXYZ *bmapxyz,
                         double x, double y, double z) {
  long ix, iy, iz, iq;
  double fx, fy, fz;
  double Finterp1[2][2], Finterp2[2];
  double Freturn[3];
  double symmetryFactor[3] = {1, 1, 1}; /* Bx, By, Bz */

  if (bmapxyz->data->magnetSymmetry[0] && x<0) {
    /* x */
    x = fabs(x);
    if (bmapxyz->data->magnetSymmetry[0]==1) {
      /* odd symmetry */
      symmetryFactor[1] *= -1;
      symmetryFactor[2] *= -1;
    } else {
      symmetryFactor[0] *= -1;
    }
  }
  if (bmapxyz->data->magnetSymmetry[1] && y<0) {
    /* y */
    y = fabs(y);
    if (bmapxyz->data->magnetSymmetry[1]==1) {
      /* odd symmetry */
      symmetryFactor[0] *= -1;
      symmetryFactor[2] *= -1;
    } else {
      symmetryFactor[1] *= -1;
    }
  }
  if (bmapxyz->data->magnetSymmetry[2]) {
    /* z */
    if (z<bmapxyz->fieldLength/2) {
      z = fabs(z-bmapxyz->fieldLength/2);
      if (bmapxyz->data->magnetSymmetry[2]==1) {
        /* odd symmetry */
        symmetryFactor[0] *= -1;
        symmetryFactor[1] *= -1;
      } else {
        symmetryFactor[2] *= -1;
      }
    } else
      z -= bmapxyz->fieldLength/2;
  }

  ix = (x - bmapxyz->data->xmin) / bmapxyz->data->dx;
  iy = (y - bmapxyz->data->ymin) / bmapxyz->data->dy;
  iz = (z - bmapxyz->data->zmin) / bmapxyz->data->dz;
  *F0 = *F1 = *F2 = 0;
  if (ix < 0 || iy < 0 || iz < 0 || ix >= (bmapxyz->data->nx - 1) || iy >= (bmapxyz->data->ny - 1) || iz >= (bmapxyz->data->nz - 1)) {
    n_invalid_particles++;
    return 0;
  } else {
    fx = (x - (ix * bmapxyz->data->dx + bmapxyz->data->xmin)) / bmapxyz->data->dx;
    fy = (y - (iy * bmapxyz->data->dy + bmapxyz->data->ymin)) / bmapxyz->data->dy;
    fz = (z - (iz * bmapxyz->data->dz + bmapxyz->data->zmin)) / bmapxyz->data->dz;

    if (bmapxyz->xyInterpolationOrder > 1) {
      double FOutput1[3], FOutput2[3];
      long offset;
      offset = iz * bmapxyz->data->nx * bmapxyz->data->ny;
      if (!bmapxyz->singlePrecision)
        interpolate2dFieldMapHigherOrder2(&FOutput1[0],
                                          x, y, bmapxyz->data->dx, bmapxyz->data->dy,
                                          bmapxyz->data->xmin, bmapxyz->data->ymin,
                                          bmapxyz->data->xmax, bmapxyz->data->ymax,
                                          bmapxyz->data->nx, bmapxyz->data->ny,
                                          bmapxyz->data->Fz, bmapxyz->data->Fx, bmapxyz->data->Fy,
                                          offset, bmapxyz->singlePrecision,
                                          bmapxyz->xyInterpolationOrder, bmapxyz->xyGridExcess);
      else
        interpolate2dFieldMapHigherOrder2(&FOutput1[0],
                                          x, y, bmapxyz->data->dx, bmapxyz->data->dy,
                                          bmapxyz->data->xmin, bmapxyz->data->ymin,
                                          bmapxyz->data->xmax, bmapxyz->data->ymax,
                                          bmapxyz->data->nx, bmapxyz->data->ny,
                                          bmapxyz->data->Fz1, bmapxyz->data->Fx1, bmapxyz->data->Fy1,
                                          offset, bmapxyz->singlePrecision,
                                          bmapxyz->xyInterpolationOrder, bmapxyz->xyGridExcess);

      offset = (iz + 1) * bmapxyz->data->nx * bmapxyz->data->ny;
      if (!bmapxyz->singlePrecision)
        interpolate2dFieldMapHigherOrder2(&FOutput2[0],
                                          x, y, bmapxyz->data->dx, bmapxyz->data->dy,
                                          bmapxyz->data->xmin, bmapxyz->data->ymin,
                                          bmapxyz->data->xmax, bmapxyz->data->ymax,
                                          bmapxyz->data->nx, bmapxyz->data->ny,
                                          bmapxyz->data->Fz, bmapxyz->data->Fx, bmapxyz->data->Fy,
                                          offset, bmapxyz->singlePrecision,
                                          bmapxyz->xyInterpolationOrder, bmapxyz->xyGridExcess);
      else
        interpolate2dFieldMapHigherOrder2(&FOutput2[0],
                                          x, y, bmapxyz->data->dx, bmapxyz->data->dy,
                                          bmapxyz->data->xmin, bmapxyz->data->ymin,
                                          bmapxyz->data->xmax, bmapxyz->data->ymax,
                                          bmapxyz->data->nx, bmapxyz->data->ny,
                                          bmapxyz->data->Fz1, bmapxyz->data->Fx1, bmapxyz->data->Fy1,
                                          offset, bmapxyz->singlePrecision,
                                          bmapxyz->xyInterpolationOrder, bmapxyz->xyGridExcess);

      *F0 = bmapxyz->strength * (1 + bmapxyz->fse) * ((1 - fz) * FOutput1[0] + fz * FOutput2[0]);
      *F1 = bmapxyz->strength * (1 + bmapxyz->fse) * ((1 - fz) * FOutput1[1] + fz * FOutput2[1]);
      *F2 = bmapxyz->strength * (1 + bmapxyz->fse) * ((1 - fz) * FOutput1[2] + fz * FOutput2[2]);
    } else {
      for (iq = 0; iq < 3; iq++) {
        /* interpolate vs z to get four points in a x-y grid */
        /* (ix, iy) */
        if (!bmapxyz->singlePrecision) {
          double *Fq[3];
          Fq[0] = bmapxyz->data->Fz;
          Fq[1] = bmapxyz->data->Fx;
          Fq[2] = bmapxyz->data->Fy;
          Finterp1[0][0] = (1 - fz) * (*(Fq[iq] + (ix + 0) + iy * bmapxyz->data->nx + iz * bmapxyz->data->nx * bmapxyz->data->ny)) +
                           fz * (*(Fq[iq] + (ix + 0) + iy * bmapxyz->data->nx + (iz + 1) * bmapxyz->data->nx * bmapxyz->data->ny));
          /* (ix+1, iy) */
          Finterp1[1][0] = (1 - fz) * (*(Fq[iq] + (ix + 1) + (iy + 0) * bmapxyz->data->nx + iz * bmapxyz->data->nx * bmapxyz->data->ny)) +
                           fz * (*(Fq[iq] + (ix + 1) + (iy + 0) * bmapxyz->data->nx + (iz + 1) * bmapxyz->data->nx * bmapxyz->data->ny));
          /* (ix, iy+1) */
          Finterp1[0][1] = (1 - fz) * (*(Fq[iq] + (ix + 0) + (iy + 1) * bmapxyz->data->nx + iz * bmapxyz->data->nx * bmapxyz->data->ny)) +
                           fz * (*(Fq[iq] + (ix + 0) + (iy + 1) * bmapxyz->data->nx + (iz + 1) * bmapxyz->data->nx * bmapxyz->data->ny));
          /* (ix+1, iy+1) */
          Finterp1[1][1] = (1 - fz) * (*(Fq[iq] + (ix + 1) + (iy + 1) * bmapxyz->data->nx + iz * bmapxyz->data->nx * bmapxyz->data->ny)) +
                           fz * (*(Fq[iq] + (ix + 1) + (iy + 1) * bmapxyz->data->nx + (iz + 1) * bmapxyz->data->nx * bmapxyz->data->ny));
        } else {
          float *Fq[3];
          Fq[0] = bmapxyz->data->Fz1;
          Fq[1] = bmapxyz->data->Fx1;
          Fq[2] = bmapxyz->data->Fy1;
          Finterp1[0][0] = (1 - fz) * (*(Fq[iq] + (ix + 0) + iy * bmapxyz->data->nx + iz * bmapxyz->data->nx * bmapxyz->data->ny)) +
                           fz * (*(Fq[iq] + (ix + 0) + iy * bmapxyz->data->nx + (iz + 1) * bmapxyz->data->nx * bmapxyz->data->ny));
          /* (ix+1, iy) */
          Finterp1[1][0] = (1 - fz) * (*(Fq[iq] + (ix + 1) + (iy + 0) * bmapxyz->data->nx + iz * bmapxyz->data->nx * bmapxyz->data->ny)) +
                           fz * (*(Fq[iq] + (ix + 1) + (iy + 0) * bmapxyz->data->nx + (iz + 1) * bmapxyz->data->nx * bmapxyz->data->ny));
          /* (ix, iy+1) */
          Finterp1[0][1] = (1 - fz) * (*(Fq[iq] + (ix + 0) + (iy + 1) * bmapxyz->data->nx + iz * bmapxyz->data->nx * bmapxyz->data->ny)) +
                           fz * (*(Fq[iq] + (ix + 0) + (iy + 1) * bmapxyz->data->nx + (iz + 1) * bmapxyz->data->nx * bmapxyz->data->ny));
          /* (ix+1, iy+1) */
          Finterp1[1][1] = (1 - fz) * (*(Fq[iq] + (ix + 1) + (iy + 1) * bmapxyz->data->nx + iz * bmapxyz->data->nx * bmapxyz->data->ny)) +
                           fz * (*(Fq[iq] + (ix + 1) + (iy + 1) * bmapxyz->data->nx + (iz + 1) * bmapxyz->data->nx * bmapxyz->data->ny));
        }

        /* interpolate vs y to get two points spaced by dx */
        /* ix */
        Finterp2[0] = (1 - fy) * Finterp1[0][0] + fy * Finterp1[0][1];
        /* ix+1 */
        Finterp2[1] = (1 - fy) * Finterp1[1][0] + fy * Finterp1[1][1];

        /* interpolate vs x to get 1 value */
        Freturn[iq] = (1 - fx) * Finterp2[0] + fx * Finterp2[1];
      }

      *F0 = bmapxyz->strength * (1 + bmapxyz->fse) * Freturn[0];
      *F1 = bmapxyz->strength * (1 + bmapxyz->fse) * Freturn[1];
      *F2 = bmapxyz->strength * (1 + bmapxyz->fse) * Freturn[2];
    }

    *F1 *= bmapxyz->BFactor[0];
    *F2 *= bmapxyz->BFactor[1];
    *F0 *= bmapxyz->BFactor[2];

    if (fabs(z - bmapxyz->fieldLength / 2) < bmapxyz->length / 2 &&
        (bmapxyz->xInsideLimit[0]>=bmapxyz->xInsideLimit[1] || (x>=bmapxyz->xInsideLimit[0] && x<=bmapxyz->xInsideLimit[1]))) {
      *F0 += bmapxyz->BInside[2]; /* z */
      *F1 += bmapxyz->BInside[0]; /* x */
      *F2 += bmapxyz->BInside[1]; /* y */
    }

    /*
    if (!fpdeb) {
      fpdeb = fopen("bmapxyz.deb", "w");
      fprintf(fpdeb, "SDDS1\n");
      fprintf(fpdeb, "&column name=ix type=short &end\n");
      fprintf(fpdeb, "&column name=iy type=short &end\n");
      fprintf(fpdeb, "&column name=iz type=short &end\n");
      fprintf(fpdeb, "&column name=x type=float units=m &end\n");
      fprintf(fpdeb, "&column name=y type=float units=m &end\n");
      fprintf(fpdeb, "&column name=z type=float units=m &end\n");
      fprintf(fpdeb, "&column name=fx type=float units=m &end\n");
      fprintf(fpdeb, "&column name=fy type=float units=m &end\n");
      fprintf(fpdeb, "&column name=fz type=float units=m &end\n");
      fprintf(fpdeb, "&column name=Bx type=float  &end\n");
      fprintf(fpdeb, "&column name=By type=float  &end\n");
      fprintf(fpdeb, "&column name=Bz type=float  &end\n");
      fprintf(fpdeb, "&column name=S type=float  &end\n");
      fprintf(fpdeb, "&data mode=ascii no_row_counts=1 &end\n");
    }
   
    fprintf(fpdeb, "%ld %ld %ld %le %le %le %le %le %le %le %le %le %le\n",
            ix, iy, iz,
            x, y, z,
            fx, fy, fz,
            *F1, *F2,*F0, bmapxyz->strength);
    */
  }

  /* note that array order is (x, y, z) while (F1=Bx, F2=By, F0=Bz) */
  *F1 *= symmetryFactor[0];
  *F2 *= symmetryFactor[1];
  *F0 *= symmetryFactor[2];

  return 1;
}
