/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* routine: print_line()
 * purpose: print a beam-line with magnet parameters
 * 
 * Michael Borland, 1989
 */
#include "mdb.h"
#include "track.h"

void print_line(FILE *fp, LINE_LIST *lptr) {
  ELEMENT_LIST *eptr;
  long i, j;
  char *ptr;
  PARAMETER *parameter;

  log_entry("print_line");

  eptr = lptr->elem;
  fprintf(fp, "line name: %s\n", lptr->name);
  fprintf(fp, "line has %ld elements\n", lptr->n_elems);

  for (i = 0; i < lptr->n_elems; i++) {
    parameter = entity_description[eptr->type].parameter;
    fprintf(fp, "%s %s at z=%em, theta=%e, part of %s:\n",
            entity_name[eptr->type], eptr->name, eptr->end_pos,
            eptr->end_theta, (eptr->part_of ? eptr->part_of : "?"));
    for (j = 0; j < entity_description[eptr->type].n_params; j++) {
      switch (parameter[j].type) {
      case IS_DOUBLE:
        fprintf(fp, "    %s = %.15g %s\n",
                parameter[j].name,
                *(double *)(eptr->p_elem + parameter[j].offset),
                parameter[j].unit);
        break;
      case IS_LONG:
        fprintf(fp, "    %s = %ld %s\n",
                parameter[j].name,
                *(long *)(eptr->p_elem + parameter[j].offset),
                parameter[j].unit);
        break;
      case IS_INT64:
        fprintf(fp, "    %s = %" PRId64 " %s\n",
                parameter[j].name,
                *(int64_t *)(eptr->p_elem + parameter[j].offset),
                parameter[j].unit);
        break;
      case IS_SHORT:
        fprintf(fp, "    %s = %hd %s\n",
                parameter[j].name,
                *(short *)(eptr->p_elem + parameter[j].offset),
                parameter[j].unit);
        break;
      case IS_STRING:
        if ((ptr = *(char **)(eptr->p_elem + parameter[j].offset)))
          fprintf(fp, "    %s = \"%s\"\n", parameter[j].name, ptr);
        else
          fprintf(fp, "    %s = \"\"\n", parameter[j].name);
        break;
      }
    }
    eptr = eptr->succ;
    if (eptr == NULL && (i + 1) != lptr->n_elems) {
      fputs("line list ends prematurely", stdout);
      exitElegant(1);
    }
  }
  log_exit("print_line");
}

void print_elem(FILE *fp, ELEMENT_LIST *eptr) {
  char *ptr;
  PARAMETER *parameter;
  long j;

  log_entry("print_elem");

  parameter = entity_description[eptr->type].parameter;
  fprintf(fp, "%s %s at z=%em, theta=%e:\n",
          entity_name[eptr->type], eptr->name, eptr->end_pos, eptr->end_theta);
  for (j = 0; j < entity_description[eptr->type].n_params; j++) {
    switch (parameter[j].type) {
    case IS_DOUBLE:
      fprintf(fp, "    %s = %.15g %s\n",
              parameter[j].name,
              *(double *)(eptr->p_elem + parameter[j].offset),
              parameter[j].unit);
      break;
    case IS_LONG:
      fprintf(fp, "    %s = %ld %s\n",
              parameter[j].name,
              *(long *)(eptr->p_elem + parameter[j].offset),
              parameter[j].unit);
      break;
    case IS_INT64:
      fprintf(fp, "    %s = %" PRId64 " %s\n",
              parameter[j].name,
              *(int64_t *)(eptr->p_elem + parameter[j].offset),
              parameter[j].unit);
      break;
    case IS_SHORT:
      fprintf(fp, "    %s = %hd %s\n",
              parameter[j].name,
              *(short *)(eptr->p_elem + parameter[j].offset),
              parameter[j].unit);
      break;
    case IS_STRING:
      if ((ptr = *(char **)(eptr->p_elem + parameter[j].offset)))
        fprintf(fp, "    %s = \"%s\"\n", parameter[j].name, ptr);
      else
        fprintf(fp, "    %s = \"\"\n", parameter[j].name);
      break;
    }
  }

  if (eptr->type == T_LGBEND) {
    LGBEND *lg;
    long iseg;
    lg = (LGBEND *)(eptr->p_elem);
    fprintf(fp, "    initialized = %hd\n", lg->initialized);
    fprintf(fp, "    nSegments = %ld\n", lg->nSegments);
    fprintf(fp, "    angle = %le\n", lg->angle);
    fprintf(fp, "    optimized = %hd\n", lg->optimized);
    fprintf(fp, "    wasFlipped = %hd\n", lg->wasFlipped);
    fprintf(fp, "    predrift = %le\n", lg->predrift);
    fprintf(fp, "    postdrift = %le\n", lg->postdrift);
    for (iseg = 0; iseg < lg->nSegments; iseg++) {
      fprintf(fp, "   segment %ld: length = %le, arcLength = %le, K1 = %le, K2 = %le, angle = %le, FSE=%le, KnDelta=%le\n",
              iseg,
              lg->segment[iseg].length,
              lg->segment[iseg].arcLength,
              lg->segment[iseg].K1,
              lg->segment[iseg].K2,
              lg->segment[iseg].angle,
              lg->fseOpt[iseg], lg->KnDelta[iseg]);
    }
    fflush(fp);
  }

  log_exit("print_elem");
}

void print_elem_list(FILE *fp, ELEMENT_LIST *eptr) {
  long j;
  char *ptr;
  PARAMETER *parameter;

  log_entry("print_elem_list");

  while (eptr) {
    parameter = entity_description[eptr->type].parameter;
    fprintf(fp, "%s %s at z=%em, theta=%e:\n",
            entity_name[eptr->type], eptr->name, eptr->end_pos, eptr->end_theta);
    if (eptr->type == T_MATR) {
      fprintf(fp, "    ORDER = %ld\n", eptr->matrix->order);
    } else {
      for (j = 0; j < entity_description[eptr->type].n_params; j++) {
        switch (parameter[j].type) {
        case IS_DOUBLE:
          fprintf(fp, "    %s = %.15g %s\n",
                  parameter[j].name,
                  *(double *)(eptr->p_elem + parameter[j].offset),
                  parameter[j].unit);
          break;
        case IS_LONG:
          fprintf(fp, "    %s = %ld %s\n",
                  parameter[j].name,
                  *(long *)(eptr->p_elem + parameter[j].offset),
                  parameter[j].unit);
          break;
        case IS_INT64:
          fprintf(fp, "    %s = %" PRId64 " %s\n",
                  parameter[j].name,
                  *(int64_t *)(eptr->p_elem + parameter[j].offset),
                  parameter[j].unit);
          break;
        case IS_SHORT:
          fprintf(fp, "    %s = %hd %s\n",
                  parameter[j].name,
                  *(short *)(eptr->p_elem + parameter[j].offset),
                  parameter[j].unit);
          break;
        case IS_STRING:
          if ((ptr = *(char **)(eptr->p_elem + parameter[j].offset)))
            fprintf(fp, "    %s = \"%s\"\n", parameter[j].name, ptr);
          else
            fprintf(fp, "    %s = \"\"\n", parameter[j].name);
          break;
        }
      }
    }
    eptr = eptr->succ;
  }
  log_exit("print_elem_list");
}

void print_elem_names(FILE *fp, ELEMENT_LIST *eptr, long width) {
  long pos;
  pos = 0;
  while (eptr) {
    if (eptr->name)
      fprintf(fp, "%16s,", eptr->name);
    pos += 16;
    if (pos + 16 > width) {
      fprintf(fp, "\n");
      pos = 0;
    }
    eptr = eptr->succ;
  }
  fprintf(fp, "\n");
}
