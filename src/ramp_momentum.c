/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* routine: ramp_momentum()
 * purpose: do momentum ramping
 *
 * Michael Borland, 1992, 1993
 */
#include "mdb.h"
#include "track.h"
#include "table.h"

void set_up_ramp_p(RAMPP *rampp);

void ramp_momentum(
  double **coord,
  int64_t np,
  RAMPP *rampp,
  double *P_central, /* current beta*gamma on input, changed on output */
  long pass) {
  int64_t ip;
  long i_time;
  double P_new, t, t0;

  log_entry("ramp_momentum");

  if (!rampp->t_Pf)
    set_up_ramp_p(rampp);

  if (!rampp->t_Pf || !rampp->Pfactor || rampp->n_pts < 2)
    bombElegant("NULL data for rampp element", NULL);

  if (!rampp->Po)
    rampp->Po = *P_central;

#if defined(DEBUG)
  printf("*P_central = %15.8e\n", *P_central);
  fflush(stdout);
#endif

  for (ip = t0 = 0; ip < np; ip++) {
    t0 += (t = coord[ip][4] / (c_mks * beta_from_delta(*P_central, coord[ip][5])));
#if defined(IEEE_MATH)
    if (isnan(t) || isinf(t)) {
      long i;
      printf("error: bad time coordinate for particle %ld\n", ip);
      for (i = 0; i < 6; i++)
        printf("%15.8e ", coord[ip][i]);
      fputc('\n', stdout);
      fflush(stdout);
      abort();
    }
#endif
  }
#if !USE_MPI
  t0 /= np;
#else
  if (distributedBeam) {
    long np_total;
    double t0_total;

    MPI_Allreduce(&np, &np_total, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&t0, &t0_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    t0 = t0_total / np_total;
  } else {
    t0 /= np;
  }
#endif

  /* macro-scale timing not carried in coord[][4]: sample the momentum ramp at
     the true absolute bunch-mean time (see trackingClockOffset) */
  t0 += trackingClockOffset();

  i_time = find_nearby_array_entry(rampp->t_Pf, rampp->n_pts, t0);
  P_new = rampp->Po * linear_interpolation(rampp->Pfactor, rampp->t_Pf, rampp->n_pts, t0, i_time);

#if defined(DEBUG)
  printf("new momentum for pass %ld, <t> = %.15e s:  %.15e\n", pass, t0, P_new);
  fflush(stdout);
#endif

  for (ip = 0; ip < np; ip++)
    coord[ip][5] = (1 + coord[ip][5]) * (*P_central) / P_new - 1;

  *P_central = P_new;
  log_exit("ramp_momentum");
}

void set_up_ramp_p(RAMPP *rampp) {
  TABLE data;
  long i;

  log_entry("set_up_rampp");

  if (!rampp->waveform)
    bombElegant("no waveform filename given for rampp", NULL);

  if (!getTableFromSearchPath(&data, rampp->waveform, 1, 0))
    bombElegant("unable to read waveform for rampp", NULL);

  if (data.n_data <= 1)
    bombElegant("rampp waveform contains less than 2 points", NULL);

  rampp->Po = 0;
  rampp->t_Pf = data.c1;
  rampp->Pfactor = data.c2;
  rampp->n_pts = data.n_data;
  for (i = 0; i < rampp->n_pts - 1; i++)
    if (rampp->t_Pf[i] > rampp->t_Pf[i + 1])
      bombElegant("time values are not monotonically increasing in rampp waveform", NULL);
  tfree(data.xlab);
  tfree(data.ylab);
  tfree(data.title);
  tfree(data.topline);
  data.xlab = data.ylab = data.title = data.topline = NULL;
  log_exit("set_up_rampp");
}
