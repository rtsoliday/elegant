/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* routine: output_magnets
 * purpose: write mpl format data to give plot of magnets.
 *
 * Michael Borland, 1988, 1991
 */
#include "mdb.h"
#include "track.h"

void output_magnets(char *filename, char *line_name, LINE_LIST *beamline) {
  ELEMENT_LIST *eptr;
  QUAD *qptr;
  BEND *bptr;
  KQUSE *kqsptr;
  KQUAD *kqptr;
  KSBEND *kbptr;
  CSBEND *cbptr;
  CCBEND *crbptr;
  CSRCSBEND *csrbptr;
  LGBEND *lgbptr;
  long iPhase;
  double start, end, dz, value, K2;
  /* double total_length; */
  FILE *fpm;

  log_entry("output_magnets");

#if USE_MPI
#  ifdef MPI_DEBUG
  printf("output_magnets called, myid=%d\n", myid);
  fflush(stdout);
#  endif
  if (myid != 0)
    return;
#endif

  /* total_length = 0; */
  eptr = beamline->elem;

  fpm = fopen_e(filename, "w", 0);

  start = end = sStart;
  fprintf(fpm, "SDDS1\n&description text=\"magnet layout for beamline %s\" &end\n", line_name);
  fprintf(fpm, "&column name=ElementName, type=string &end\n");
  fprintf(fpm, "&column name=ElementType, type=string &end\n");
  fprintf(fpm, "&column name=s, units=m, type=double &end\n&column name=Profile, type=double &end\n");
  fprintf(fpm, "&data mode=ascii, no_row_counts=1 &end\n");

  eptr = beamline->elem;
  fprintf(fpm, "_BEGIN_ MARK %le 0\n", sStart);
  while (eptr) {
    fprintf(fpm, "\"%s\" %s %e 0\n", eptr->name, entity_name[eptr->type], end);
    switch (eptr->type) {
    case T_QUAD:
      qptr = (QUAD *)eptr->p_elem;
      fprintf(fpm, "\"%s\" %s %e  %d\n", eptr->name, entity_name[eptr->type], start, SIGN(qptr->k1));
      end = start + qptr->length;
      fprintf(fpm, "\"%s\" %s %e  %d\n\"%s\" %s %e  0\n\"%s\" %s %e 0\n\"%s\" %s %e 0\n",
              eptr->name, entity_name[eptr->type], end, SIGN(qptr->k1),
              eptr->name, entity_name[eptr->type], end,
              eptr->name, entity_name[eptr->type], start,
              eptr->name, entity_name[eptr->type], end);
      start = end;
      break;
    case T_KQUAD:
      kqptr = (KQUAD *)eptr->p_elem;
      if (kqptr->bore)
        value = kqptr->B;
      else
        value = kqptr->k1;
      fprintf(fpm, "\"%s\" %s %e  %d\n", eptr->name, entity_name[eptr->type], start, SIGN(value));
      end = start + kqptr->length;
      fprintf(fpm, "\"%s\" %s %e  %d\n\"%s\" %s %e  0\n\"%s\" %s %e 0\n\"%s\" %s %e 0\n",
              eptr->name, entity_name[eptr->type], end, SIGN(value),
              eptr->name, entity_name[eptr->type], end,
              eptr->name, entity_name[eptr->type], start,
              eptr->name, entity_name[eptr->type], end);
      start = end;
      break;
    case T_KQUSE:
      kqsptr = (KQUSE *)eptr->p_elem;
      value = kqsptr->k1;
      fprintf(fpm, "\"%s\" %s %e  %d\n", eptr->name, entity_name[eptr->type], start, SIGN(value));
      end = start + kqsptr->length;
      fprintf(fpm, "\"%s\" %s %e  %d\n\"%s\" %s %e  0\n\"%s\" %s %e 0\n\"%s\" %s %e 0\n",
              eptr->name, entity_name[eptr->type], end, SIGN(value),
              eptr->name, entity_name[eptr->type], end,
              eptr->name, entity_name[eptr->type], start,
              eptr->name, entity_name[eptr->type], end);
      start = end;
      break;
    case T_RBEN:
    case T_SBEN:
      bptr = (BEND *)eptr->p_elem;
      end = start + bptr->length;
      if (bptr->angle > 0)
        fprintf(fpm,
                "\"%s\" %s %e .33333333\n\"%s\" %s %e .33333333\n\"%s\" %s %e  0\n\"%s\" %s %e  0\n\"%s\" %s %e  0\n\"%s\" %s %e 0\n",
                eptr->name, entity_name[eptr->type], start, eptr->name, entity_name[eptr->type], end, eptr->name, entity_name[eptr->type], end, eptr->name, entity_name[eptr->type], start,
                eptr->name, entity_name[eptr->type], start, eptr->name, entity_name[eptr->type], end);
      else if (bptr->angle < 0)
        fprintf(fpm,
                "\"%s\" %s %e -.33333333\n\"%s\" %s %e -.33333333\n\"%s\" %s %e  0\n\"%s\" %s %e  0\n\"%s\" %s %e  0\n\"%s\" %s %e 0\n",
                eptr->name, entity_name[eptr->type], start, eptr->name, entity_name[eptr->type], end, eptr->name, entity_name[eptr->type], end, eptr->name, entity_name[eptr->type], start,
                eptr->name, entity_name[eptr->type], start, eptr->name, entity_name[eptr->type], end);
      start = end;
      break;
    case T_KSBEND:
      kbptr = (KSBEND *)eptr->p_elem;
      end = start + kbptr->length;
      if (kbptr->angle > 0)
        fprintf(fpm,
                "\"%s\" %s %e .33333333\n\"%s\" %s %e .33333333\n\"%s\" %s %e  0\n\"%s\" %s %e  0\n\"%s\" %s %e  0\n\"%s\" %s %e 0\n",
                eptr->name, entity_name[eptr->type], start, eptr->name, entity_name[eptr->type], end, eptr->name, entity_name[eptr->type], end, eptr->name, entity_name[eptr->type], start,
                eptr->name, entity_name[eptr->type], start, eptr->name, entity_name[eptr->type], end);
      else if (kbptr->angle < 0)
        fprintf(fpm,
                "\"%s\" %s %e -.33333333\n\"%s\" %s %e -.33333333\n\"%s\" %s %e  0\n\"%s\" %s %e  0\n\"%s\" %s %e  0\n\"%s\" %s %e 0\n",
                eptr->name, entity_name[eptr->type], start, eptr->name, entity_name[eptr->type], end, eptr->name, entity_name[eptr->type], end, eptr->name, entity_name[eptr->type], start,
                eptr->name, entity_name[eptr->type], start, eptr->name, entity_name[eptr->type], end);
      start = end;
      break;
    case T_SEXT:
      end = start + ((SEXT *)eptr->p_elem)->length;
      K2 = ((SEXT *)eptr->p_elem)->k2;
      fprintf(fpm, "\"%s\" %s %e  %e\n\"%s\" %s %e %e\n\"%s\" %s %e 0\n\"%s\" %s %e 0\n\"%s\" %s %e 0\n",
              eptr->name, entity_name[eptr->type], start, 0.5*SIGN(K2),
              eptr->name, entity_name[eptr->type], end, 0.5*SIGN(K2),
              eptr->name, entity_name[eptr->type], end, 
              eptr->name, entity_name[eptr->type], start,
              eptr->name, entity_name[eptr->type], end);
      start = end;
      break;
    case T_KSEXT:
      end = start + ((KSEXT *)eptr->p_elem)->length;
      K2 = ((KSEXT *)eptr->p_elem)->k2;
      fprintf(fpm, "\"%s\" %s %e  %e\n\"%s\" %s %e %e\n\"%s\" %s %e 0\n\"%s\" %s %e 0\n\"%s\" %s %e 0\n",
              eptr->name, entity_name[eptr->type], start, 0.5*SIGN(K2),
              eptr->name, entity_name[eptr->type], end, 0.5*SIGN(K2),
              eptr->name, entity_name[eptr->type], end, 
              eptr->name, entity_name[eptr->type], start,
              eptr->name, entity_name[eptr->type], end);
      start = end;
      break;
    case T_HCOR:
      end = start + ((HCOR *)eptr->p_elem)->length;
      fprintf(fpm, "\"%s\" %s %e .25\n\"%s\" %s %e .25\n\"%s\" %s %e 0\n",
              eptr->name, entity_name[eptr->type], start, eptr->name, entity_name[eptr->type], end, eptr->name, entity_name[eptr->type], end);
      start = end;
      break;
    case T_VCOR:
      end = start + ((VCOR *)eptr->p_elem)->length;
      fprintf(fpm, "\"%s\" %s %e -.25\n\"%s\" %s %e -.25\n\"%s\" %s %e 0\n",
              eptr->name, entity_name[eptr->type], start, eptr->name, entity_name[eptr->type], end, eptr->name, entity_name[eptr->type], end);
      start = end;
      break;
    case T_DQCOR:
      end = start + ((DQCOR *)eptr->p_elem)->length;
      fprintf(fpm, "\"%s\" %s %e .25\n\"%s\" %s %e .25\n\"%s\" %s %e 0\n",
              eptr->name, entity_name[eptr->type], start, eptr->name, entity_name[eptr->type], end, eptr->name, entity_name[eptr->type], end);
      start = end;
      break;
    case T_HVCOR:
      end = start + ((HVCOR *)eptr->p_elem)->length;
      fprintf(fpm, "\"%s\" %s %e .25\n\"%s\" %s %e -.25\n\"%s\" %s %e 0\n",
              eptr->name, entity_name[eptr->type], start, eptr->name, entity_name[eptr->type], end, eptr->name, entity_name[eptr->type], end);
      start = end;
      break;
    case T_EHCOR:
      end = start + ((EHCOR *)eptr->p_elem)->length;
      fprintf(fpm, "\"%s\" %s %e .25\n\"%s\" %s %e .25\n\"%s\" %s %e 0\n",
              eptr->name, entity_name[eptr->type], start, eptr->name, entity_name[eptr->type], end, eptr->name, entity_name[eptr->type], end);
      start = end;
      break;
    case T_EVCOR:
      end = start + ((EVCOR *)eptr->p_elem)->length;
      fprintf(fpm, "\"%s\" %s %e -.25\n\"%s\" %s %e -.25\n\"%s\" %s %e 0\n",
              eptr->name, entity_name[eptr->type], start, eptr->name, entity_name[eptr->type], end, eptr->name, entity_name[eptr->type], end);
      start = end;
      break;
    case T_EHVCOR:
      end = start + ((EHVCOR *)eptr->p_elem)->length;
      fprintf(fpm, "\"%s\" %s %e .25\n\"%s\" %s %e -.25\n\"%s\" %s %e 0\n",
              eptr->name, entity_name[eptr->type], start, eptr->name, entity_name[eptr->type], end, eptr->name, entity_name[eptr->type], end);
      start = end;
      break;
    case T_DRIF:
      start = (end = start + ((DRIFT *)eptr->p_elem)->length);
      fprintf(fpm, "\"%s\" %s %e  0\n", eptr->name, entity_name[eptr->type], end);
      break;
    case T_CSRDRIFT:
      start = (end = start + ((CSRDRIFT *)eptr->p_elem)->length);
      fprintf(fpm, "\"%s\" %s %e  0\n", eptr->name, entity_name[eptr->type], end);
      break;
    case T_EDRIFT:
      start = (end = start + ((EDRIFT *)eptr->p_elem)->length);
      fprintf(fpm, "\"%s\" %s %e  0\n", eptr->name, entity_name[eptr->type], end);
      break;
    case T_HMON:
      dz = ((HMON *)eptr->p_elem)->length / 2;
      fprintf(fpm, "\"%s\" %s %e 0.125\n\"%s\" %s %e 0\n\"%s\" %s %e 0\n\"%s\" %s %e 0\n",
              eptr->name, entity_name[eptr->type], start + dz, eptr->name, entity_name[eptr->type], start + 2 * dz,
              eptr->name, entity_name[eptr->type], start + dz, eptr->name, entity_name[eptr->type], start + 2 * dz);
      start += 2 * dz;
      break;
    case T_VMON:
      dz = ((VMON *)eptr->p_elem)->length / 2;
      fprintf(fpm, "\"%s\" %s %e -0.125\n\"%s\" %s %e 0\n\"%s\" %s %e 0\n\"%s\" %s %e 0\n",
              eptr->name, entity_name[eptr->type], start + dz, eptr->name, entity_name[eptr->type], start + 2 * dz,
              eptr->name, entity_name[eptr->type], start + dz, eptr->name, entity_name[eptr->type], start + 2 * dz);
      start += 2 * dz;
      break;
    case T_MONI:
      dz = ((MONI *)eptr->p_elem)->length / 2;
      fprintf(fpm, "\"%s\" %s %e 0.125\n\"%s\" %s %e 0\n\"%s\" %s %e -0.125\n\"%s\" %s %e 0\n\"%s\" %s %e 0\n",
              eptr->name, entity_name[eptr->type], start + dz, eptr->name, entity_name[eptr->type], start + 2 * dz, eptr->name, entity_name[eptr->type], start + dz,
              eptr->name, entity_name[eptr->type], start, eptr->name, entity_name[eptr->type], start + 2 * dz);
      start += 2 * dz;
      break;
    case T_MULT:
      dz = ((MULT *)eptr->p_elem)->length / 3;
      fprintf(fpm, "\"%s\" %s %e 0.6666\n\"%s\" %s %e 0.6666\n\"%s\" %s %e 0\n\"%s\" %s %e -0.6666\n\"%s\" %s %e -0.6666\n\"%s\" %s %e 0\n\"%s\" %s %e 0\n",
              eptr->name, entity_name[eptr->type], start + dz, eptr->name, entity_name[eptr->type], start + 2 * dz, eptr->name, entity_name[eptr->type], start + 3 * dz,
              eptr->name, entity_name[eptr->type], start + 2 * dz, eptr->name, entity_name[eptr->type], start + dz, eptr->name, entity_name[eptr->type], start, eptr->name, entity_name[eptr->type], start + 3 * dz);
      start += 3 * dz;
      break;
    case T_MARK: /* zero-length drift */
      break;
    case T_CSBEND:
      cbptr = (CSBEND *)eptr->p_elem;
      end = start + cbptr->length;
      if (cbptr->angle > 0)
        fprintf(fpm,
                "\"%s\" %s %e .33333333\n\"%s\" %s %e .33333333\n\"%s\" %s %e  0\n\"%s\" %s %e  0\n\"%s\" %s %e  0\n\"%s\" %s %e 0\n",
                eptr->name, entity_name[eptr->type], start, eptr->name, entity_name[eptr->type], end, eptr->name, entity_name[eptr->type], end,
                eptr->name, entity_name[eptr->type], start, eptr->name, entity_name[eptr->type], start, eptr->name, entity_name[eptr->type], end);
      else if (cbptr->angle < 0)
        fprintf(fpm,
                "\"%s\" %s %e -.33333333\n\"%s\" %s %e -.33333333\n\"%s\" %s %e  0\n\"%s\" %s %e  0\n\"%s\" %s %e  0\n\"%s\" %s %e 0\n",
                eptr->name, entity_name[eptr->type], start, eptr->name, entity_name[eptr->type], end, eptr->name, entity_name[eptr->type], end,
                eptr->name, entity_name[eptr->type], start, eptr->name, entity_name[eptr->type], start, eptr->name, entity_name[eptr->type], end);
      start = end;
      break;
    case T_LGBEND:
      lgbptr = (LGBEND *)eptr->p_elem;
      end = start + lgbptr->length;
      if (lgbptr->angle > 0)
        fprintf(fpm,
                "\"%s\" %s %e .33333333\n\"%s\" %s %e .33333333\n\"%s\" %s %e  0\n\"%s\" %s %e  0\n\"%s\" %s %e  0\n\"%s\" %s %e 0\n",
                eptr->name, entity_name[eptr->type], start, eptr->name, entity_name[eptr->type], end, eptr->name, entity_name[eptr->type], end,
                eptr->name, entity_name[eptr->type], start, eptr->name, entity_name[eptr->type], start, eptr->name, entity_name[eptr->type], end);
      else if (lgbptr->angle < 0)
        fprintf(fpm,
                "\"%s\" %s %e -.33333333\n\"%s\" %s %e -.33333333\n\"%s\" %s %e  0\n\"%s\" %s %e  0\n\"%s\" %s %e  0\n\"%s\" %s %e 0\n",
                eptr->name, entity_name[eptr->type], start, eptr->name, entity_name[eptr->type], end, eptr->name, entity_name[eptr->type], end,
                eptr->name, entity_name[eptr->type], start, eptr->name, entity_name[eptr->type], start, eptr->name, entity_name[eptr->type], end);
      start = end;
      break;
    case T_CCBEND:
      crbptr = (CCBEND *)eptr->p_elem;
      end = start + crbptr->length;
      if (crbptr->angle > 0)
        fprintf(fpm,
                "\"%s\" %s %e .33333333\n\"%s\" %s %e .33333333\n\"%s\" %s %e  0\n\"%s\" %s %e  0\n\"%s\" %s %e  0\n\"%s\" %s %e 0\n",
                eptr->name, entity_name[eptr->type], start, eptr->name, entity_name[eptr->type], end, eptr->name, entity_name[eptr->type], end,
                eptr->name, entity_name[eptr->type], start, eptr->name, entity_name[eptr->type], start, eptr->name, entity_name[eptr->type], end);
      else if (crbptr->angle < 0)
        fprintf(fpm,
                "\"%s\" %s %e -.33333333\n\"%s\" %s %e -.33333333\n\"%s\" %s %e  0\n\"%s\" %s %e  0\n\"%s\" %s %e  0\n\"%s\" %s %e 0\n",
                eptr->name, entity_name[eptr->type], start, eptr->name, entity_name[eptr->type], end, eptr->name, entity_name[eptr->type], end,
                eptr->name, entity_name[eptr->type], start, eptr->name, entity_name[eptr->type], start, eptr->name, entity_name[eptr->type], end);
      start = end;
      break;
    case T_CSRCSBEND:
      csrbptr = (CSRCSBEND *)eptr->p_elem;
      end = start + csrbptr->length;
      if (csrbptr->angle > 0)
        fprintf(fpm,
                "\"%s\" %s %e .33333333\n\"%s\" %s %e .33333333\n\"%s\" %s %e  0\n\"%s\" %s %e  0\n\"%s\" %s %e  0\n\"%s\" %s %e 0\n",
                eptr->name, entity_name[eptr->type], start, eptr->name, entity_name[eptr->type], end, eptr->name, entity_name[eptr->type], end,
                eptr->name, entity_name[eptr->type], start, eptr->name, entity_name[eptr->type], start, eptr->name, entity_name[eptr->type], end);
      else if (csrbptr->angle < 0)
        fprintf(fpm,
                "\"%s\" %s %e -.33333333\n\"%s\" %s %e -.33333333\n\"%s\" %s %e  0\n\"%s\" %s %e  0\n\"%s\" %s %e  0\n\"%s\" %s %e 0\n",
                eptr->name, entity_name[eptr->type], start, eptr->name, entity_name[eptr->type], end, eptr->name, entity_name[eptr->type], end,
                eptr->name, entity_name[eptr->type], start, eptr->name, entity_name[eptr->type], start, eptr->name, entity_name[eptr->type], end);
      start = end;
      break;
    case T_RFCA:
    case T_TWLA:
    case T_RAMPRF:
    case T_RFCW:
    case T_MODRF:
      dz = ((DRIFT *)eptr->p_elem)->length;
      dz /= 8;
      for (iPhase = 0; iPhase < 9; iPhase++) {
        fprintf(fpm, "\"%s\" %s %e %e\n",
                eptr->name, entity_name[eptr->type], start + dz * iPhase,
                0.5 * sin((iPhase / 8.0) * PIx2));
      }
      start += dz * 8;
      break;
    case T_MATR:
    case T_SOLE:
    case T_MAPSOLENOID:
      dz = ((DRIFT *)eptr->p_elem)->length;
      fprintf(fpm, "\"%s\" %s %e %e\n", eptr->name, entity_name[eptr->type], start, 0.5);
      fprintf(fpm, "\"%s\" %s %e %e\n", eptr->name, entity_name[eptr->type], start + dz, -0.5);
      fprintf(fpm, "\"%s\" %s %e %e\n", eptr->name, entity_name[eptr->type], start + dz, 0.5);
      fprintf(fpm, "\"%s\" %s %e %e\n", eptr->name, entity_name[eptr->type], start, -0.5);
      fprintf(fpm, "\"%s\" %s %e %e\n", eptr->name, entity_name[eptr->type], start, 0.0);
      fprintf(fpm, "\"%s\" %s %e %e\n", eptr->name, entity_name[eptr->type], start + dz, 0.0);
      start += dz;
      break;
    default:
      if (entity_description[eptr->type].flags & HAS_LENGTH) {
        dz = ((DRIFT *)eptr->p_elem)->length;
        fprintf(fpm, "\"%s\" %s %e 0\n", eptr->name, entity_name[eptr->type], start += dz);
      }
      break;
    }
    start = end = eptr->end_pos;
    eptr = eptr->succ;
  }
  log_exit("output_magnets");
  fclose(fpm);
}

void output_profile(char *filename, char *line_name, LINE_LIST *beamline) {
  ELEMENT_LIST *eptr;
  long iRow;
  double s, rho, K1, K2, K3, angle, K1L, K2L, K3L, xKick, yKick, FSE, FSEDipole, FSEQuadrupole;
  SDDS_DATASET SDDSout;
  long iElementName, iElementType, iElementOccurence, is, irho, iK1, iK2, iK3, iAngle, iK1L, iK2L, iK3L,
    ixKick, iyKick, iFSE, iFSEDipole, iFSEQuadrupole;

#if USE_MPI
#  ifdef MPI_DEBUG
  printf("output_profile called, myid=%d\n", myid);
  fflush(stdout);
#  endif
  if (myid != 0)
    return;
#endif

  iElementName = iElementType = iElementOccurence = is = irho = iK1 = iK2 = iK3 = iAngle =
    iK1L = iK2L = iK3L = ixKick = iyKick = iFSE = iFSEDipole = iFSEQuadrupole = -1;
  
  if (!SDDS_InitializeOutputElegant(&SDDSout, SDDS_BINARY, 0, NULL, NULL, filename) ||
      (iElementName=SDDS_DefineColumn(&SDDSout, "ElementName", NULL, NULL, NULL, NULL, SDDS_STRING, 0))<0 ||
      (iElementType=SDDS_DefineColumn(&SDDSout, "ElementType", NULL, NULL, NULL, NULL, SDDS_STRING, 0))<0 ||
      (iElementOccurence=SDDS_DefineColumn(&SDDSout, "ElementOccurence", NULL, NULL, NULL, NULL, SDDS_LONG, 0))<0 ||
      (is=SDDS_DefineColumn(&SDDSout, "s", NULL, "m", NULL, NULL, SDDS_DOUBLE, 0))<0 ||
      (irho=SDDS_DefineColumn(&SDDSout, "rho", NULL, "m", NULL, NULL, SDDS_DOUBLE, 0))<0 ||
      (iK1=SDDS_DefineColumn(&SDDSout, "K1", NULL, "m$a2$n", NULL, NULL, SDDS_DOUBLE, 0))<0 ||
      (iK2=SDDS_DefineColumn(&SDDSout, "K2", NULL, "m$a3$n", NULL, NULL, SDDS_DOUBLE, 0))<0 ||
      (iK3=SDDS_DefineColumn(&SDDSout, "K3", NULL, "m$a4$n", NULL, NULL, SDDS_DOUBLE, 0))<0 ||
      (iAngle=SDDS_DefineColumn(&SDDSout, "Angle", NULL, "", NULL, NULL, SDDS_DOUBLE, 0))<0 ||
      (iK1L=SDDS_DefineColumn(&SDDSout, "K1L", NULL, "m$a1$n", NULL, NULL, SDDS_DOUBLE, 0))<0 ||
      (iK2L=SDDS_DefineColumn(&SDDSout, "K2L", NULL, "m$a2$n", NULL, NULL, SDDS_DOUBLE, 0))<0 ||
      (iK3L=SDDS_DefineColumn(&SDDSout, "K3L", NULL, "m$a3$n", NULL, NULL, SDDS_DOUBLE, 0))<0 ||
      (ixKick=SDDS_DefineColumn(&SDDSout, "xKick", NULL, "", NULL, NULL, SDDS_DOUBLE, 0))<0 ||
      (iyKick=SDDS_DefineColumn(&SDDSout, "yKick", NULL, "", NULL, NULL, SDDS_DOUBLE, 0))<0 ||
      (iFSE=SDDS_DefineColumn(&SDDSout, "FSE", NULL, "", NULL, NULL, SDDS_DOUBLE, 0))<0 ||
      (iFSEDipole=SDDS_DefineColumn(&SDDSout, "FSE_DIPOLE", NULL, "", NULL, NULL, SDDS_DOUBLE, 0))<0 ||
      (iFSEQuadrupole=SDDS_DefineColumn(&SDDSout, "FSE_QUADRUPOLE", NULL, "", NULL, NULL, SDDS_DOUBLE, 0))<0 ||
      !SDDS_WriteLayout(&SDDSout) || !SDDS_StartPage(&SDDSout, 2*beamline->n_elems)) {
    printf("Problem setting up profile output file\n");
    fflush(stdout);
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
    exitElegant(1);
  }

  eptr = beamline->elem;
  iRow = 0;
  s = 0;
  while (eptr) {
    rho = K1 = K2 = K3 = angle = K1L = K2L = K3L = xKick = yKick = FSE = FSEDipole = FSEQuadrupole = 0;
    switch (eptr->type) {
    case T_QUAD:
      K1 = ((QUAD*)eptr->p_elem)->k1;
      K1L = K1*((QUAD*)eptr->p_elem)->length;
      xKick = ((QUAD*)eptr->p_elem)->xkick*((QUAD*)eptr->p_elem)->xKickCalibration;
      yKick = ((QUAD*)eptr->p_elem)->ykick*((QUAD*)eptr->p_elem)->yKickCalibration;
      FSE = ((QUAD*)eptr->p_elem)->fse;
      break;
    case T_KQUAD:
      K1 = ((KQUAD*)eptr->p_elem)->k1;
      K1L = K1*((KQUAD*)eptr->p_elem)->length;
      xKick = ((KQUAD*)eptr->p_elem)->xkick*((KQUAD*)eptr->p_elem)->xKickCalibration;
      yKick = ((KQUAD*)eptr->p_elem)->ykick*((KQUAD*)eptr->p_elem)->yKickCalibration;
      FSE = ((KQUAD*)eptr->p_elem)->fse;
      break;
    case T_KQUSE:
      K1 = ((KQUSE*)eptr->p_elem)->k1;
      K2 = ((KQUSE*)eptr->p_elem)->k2;
      K1L = K1*((KQUSE*)eptr->p_elem)->length;
      K2L = K2*((KQUSE*)eptr->p_elem)->length;
      break;
    case T_RBEN:
    case T_SBEN:
      angle = ((BEND*)eptr->p_elem)->angle;
      if (angle!=0)
        rho = ((BEND*)eptr->p_elem)->length/angle;
      K1 = ((BEND*)eptr->p_elem)->k1;
      K2 = ((BEND*)eptr->p_elem)->k2;
      K1L = K1*((BEND*)eptr->p_elem)->length;
      K2L = K2*((BEND*)eptr->p_elem)->length;
      FSE = ((BEND*)eptr->p_elem)->fse;
      break;
    case T_SEXT:
      K2 = ((SEXT*)eptr->p_elem)->k2;
      K2L = K2*((SEXT*)eptr->p_elem)->length;
      FSE = ((SEXT*)eptr->p_elem)->fse;
      break;
    case T_KSEXT:
      K2 = ((KSEXT*)eptr->p_elem)->k2;
      K2L = K2*((KSEXT*)eptr->p_elem)->length;
      xKick = ((KSEXT*)eptr->p_elem)->xkick*((KSEXT*)eptr->p_elem)->xKickCalibration;
      yKick = ((KSEXT*)eptr->p_elem)->ykick*((KSEXT*)eptr->p_elem)->yKickCalibration;
      FSE = ((KSEXT*)eptr->p_elem)->fse;
      break;
    case T_MULT:
      if (((MULT*)eptr->p_elem)->length) {
        switch (((MULT*)eptr->p_elem)->order) {
        case 1:
          K1 = ((MULT*)eptr->p_elem)->KnL/(((MULT*)eptr->p_elem)->length);
          break;
        case 2:
          K2 = ((MULT*)eptr->p_elem)->KnL/(((MULT*)eptr->p_elem)->length);
          break;
        case 3:
          K3 = ((MULT*)eptr->p_elem)->KnL/(((MULT*)eptr->p_elem)->length);
          break;
        default:
          break;
        }
      }
      switch (((MULT*)eptr->p_elem)->order) {
      case 1:
	K1L = ((MULT*)eptr->p_elem)->KnL;
	break;
      case 2:
	K2L = ((MULT*)eptr->p_elem)->KnL;
	break;
      case 3:
	K3L = ((MULT*)eptr->p_elem)->KnL;
	break;
      default:
	break;
      }
      break;
    case T_CSBEND:
      angle = ((CSBEND*)eptr->p_elem)->angle;
      if (angle!=0)
	rho = ((CSBEND*)eptr->p_elem)->length/angle;
      K1 = ((CSBEND*)eptr->p_elem)->k1;
      K1L = K1*((CSBEND*)eptr->p_elem)->length;
      K2 = ((CSBEND*)eptr->p_elem)->k2;
      K2L = K2*((CSBEND*)eptr->p_elem)->length;
      xKick = ((CSBEND*)eptr->p_elem)->xKick;
      yKick = ((CSBEND*)eptr->p_elem)->yKick;
      FSE = ((CSBEND*)eptr->p_elem)->fse;
      FSEDipole = ((CSBEND*)eptr->p_elem)->fseDipole;
      FSEQuadrupole = ((CSBEND*)eptr->p_elem)->fseQuadrupole;
      break;
    case T_CCBEND:
      angle = ((CCBEND*)eptr->p_elem)->angle;
      if (angle!=0)
	rho = ((CCBEND*)eptr->p_elem)->length/angle;
      K1 = ((CCBEND*)eptr->p_elem)->K1;
      K1L = K1*((CCBEND*)eptr->p_elem)->length;
      K2 = ((CCBEND*)eptr->p_elem)->K2;
      K2L = K2*((CCBEND*)eptr->p_elem)->length;
      xKick = ((CCBEND*)eptr->p_elem)->xKick;
      FSE = ((CCBEND*)eptr->p_elem)->fse;
      FSEDipole = ((CCBEND*)eptr->p_elem)->fseDipole;
      FSEQuadrupole = ((CCBEND*)eptr->p_elem)->fseQuadrupole;
      break;
    case T_CSRCSBEND:
      angle = ((CSRCSBEND*)eptr->p_elem)->angle;
      if (angle!=0)
	rho = ((CSRCSBEND*)eptr->p_elem)->length/angle;
      K1 = ((CSRCSBEND*)eptr->p_elem)->k1;
      K1L = K1*((CSRCSBEND*)eptr->p_elem)->length;
      K2 = ((CSRCSBEND*)eptr->p_elem)->k2;
      K2L = K2*((CSRCSBEND*)eptr->p_elem)->length;
      FSE = ((CSRCSBEND*)eptr->p_elem)->fse;
      break;
    case T_OCT:
      K3 = ((OCTU*)eptr->p_elem)->k3;
      K3L = K3*((OCTU*)eptr->p_elem)->length;
      FSE = ((OCTU*)eptr->p_elem)->fse;
      break;
    case T_KOCT:
      K3 = ((KOCT*)eptr->p_elem)->k3;
      K3L = K3*((KOCT*)eptr->p_elem)->length;
      FSE = ((KOCT*)eptr->p_elem)->fse;
      break;
    case T_HCOR:
      xKick = ((HCOR*)eptr->p_elem)->kick;
      break;
    case T_VCOR:
      yKick = ((VCOR*)eptr->p_elem)->kick;
      break;
    case T_HVCOR:
      xKick = ((HVCOR*)eptr->p_elem)->xkick;
      yKick = ((HVCOR*)eptr->p_elem)->ykick;
      break;
    case T_EHCOR:
      xKick = ((EHCOR*)eptr->p_elem)->kick;
      break;
    case T_EVCOR:
      yKick = ((EVCOR*)eptr->p_elem)->kick;
      break;
    case T_EHVCOR:
      xKick = ((EHVCOR*)eptr->p_elem)->xkick;
      yKick = ((EHVCOR*)eptr->p_elem)->ykick;
      break;
    default:
      break;
    }
    if (!SDDS_SetRowValues(&SDDSout, SDDS_SET_BY_INDEX|SDDS_PASS_BY_VALUE, iRow,
                           iElementName, eptr->name, iElementType, entity_name[eptr->type],
                           iElementOccurence, eptr->occurence,
                           is, s, irho, rho, iK1, K1, iK2, K2, iK3, K3,
			   iAngle, angle, iK1L, K1L, iK2L, K2L, iK3L, K3L,
			   ixKick, xKick, iyKick, yKick,
			   iFSE, FSE, iFSEDipole, FSEDipole, iFSEQuadrupole, FSEQuadrupole, -1)) {
      printf("Problem setting values for profile output file\n");
      fflush(stdout);
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }
    iRow++;
    if (!SDDS_SetRowValues(&SDDSout, SDDS_SET_BY_INDEX|SDDS_PASS_BY_VALUE, iRow,
                           iElementName, eptr->name, iElementType, entity_name[eptr->type],
                           iElementOccurence, eptr->occurence,
                           is, s=eptr->end_pos, irho, rho, iK1, K1, iK2, K2, iK3, K3, 
			   iAngle, angle, iK1L, K1L, iK2L, K3L, iK3L, K3L,
			   ixKick, xKick, iyKick, yKick,
			   iFSE, FSE, iFSEDipole, FSEDipole, iFSEQuadrupole, FSEQuadrupole, -1)) {
      printf("Problem setting values for profile output file\n");
      fflush(stdout);
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }
    iRow++;
    s += 1e-12;
    eptr = eptr->succ;
  }
  if (!SDDS_WritePage(&SDDSout) || !SDDS_Terminate(&SDDSout)) {
      printf("Problem writing profile output file\n");
      fflush(stdout);
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
  }
  log_exit("output_profile");
}
