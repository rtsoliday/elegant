/*************************************************************************\
* Copyright (c) 2017 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2017 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

#include "mdb.h"
#include "madto.h"
#include <ctype.h>

#define MAXBUF 32768

void convert_to_mad8(char *outputfile, LINE_LIST *beamline, char *header_file, char *ender_file) {
  ELEMENT_LIST *eptr;
  QUAD *quad;
  KQUAD *kquad;
  SEXT *sext;
  KSEXT *ksext;
  BEND *bend;
  DRIFT *drift;
  EDRIFT *edrift;
  CSBEND *csbend;
  CSRCSBEND *csrbend;
  CSRDRIFT *csrdrift;
  HCOR *hcor;
  VCOR *vcor;
  HVCOR *hvcor;
  EHCOR *ehcor;
  EVCOR *evcor;
  EHVCOR *ehvcor;
  FILE *fpi, *fp;
  char s[32768], buffer[32768], name[1024];
  htab *hash_table;
  long count, lcount;

  fp = fopen_e(outputfile, "w", 0);

  hash_table = hcreate(12); /* create a hash table with the size of 2^12, it can grow automatically if necessary */

  if (header_file) {
    fpi = fopen_e(header_file, "r", 1);
    while (fgets(s, 256, fpi))
      fputs(s, fp);
    fclose(fpi);
  }

  eptr = beamline->elem;
  count = 0;
  while (eptr) {
    strcpy(name, eptr->name);
    replace_chars(name, ":.", "cd");
    if (hadd(hash_table, (unsigned char*)name, strlen(name), NULL) == TRUE) {
      switch (eptr->type) {
      case T_QUAD:
        quad = (QUAD *)eptr->p_elem;
        sprintf(s, "%s: QUAD,L=%.15g,K1=%.15g,TILT=%.15g",
                name, quad->length, quad->k1, quad->tilt);
        break;
      case T_KQUAD:
        kquad = (KQUAD *)eptr->p_elem;
        sprintf(s, "%s: QUAD,L=%.15g,K1=%.15g,TILT=%.15g",
                name, kquad->length, kquad->k1, kquad->tilt);
        break;
      case T_SBEN:
        bend = (BEND *)eptr->p_elem;
        sprintf(s, "%s: SBEN,L=%.15g,ANGLE=%.15g,K1=%.15g,E1=%.15g,E2=%.15g,TILT=%.15g,HGAP=%.15g,FINT=%.15g",
                name, bend->length, bend->angle, bend->k1, bend->e[0], bend->e[1], bend->tilt, bend->hgap, bend->fint);
        break;
      case T_CSBEND:
        csbend = (CSBEND *)eptr->p_elem;
        sprintf(s, "%s: SBEN,L=%.15g,ANGLE=%.15g,K1=%.15g,E1=%.15g,E2=%.15g,TILT=%.15g,HGAP=%.15g,FINT=%.15g",
                name, csbend->length, csbend->angle, csbend->k1, csbend->e[0], csbend->e[1], csbend->tilt, csbend->hgap,
                csbend->fintBoth);
        break;
      case T_CSRCSBEND:
        csrbend = (CSRCSBEND *)eptr->p_elem;
        sprintf(s, "%s: SBEN,L=%.15g,ANGLE=%.15g,K1=%.15g,E1=%.15g,E2=%.15g,TILT=%.15g,HGAP=%.15g,FINT=%.15g",
                name, csrbend->length, csrbend->angle, csrbend->k1, csrbend->e[0], csrbend->e[1], csrbend->tilt, csrbend->hgap, csrbend->fint);
        break;
      case T_DRIF:
        drift = (DRIFT *)eptr->p_elem;
        sprintf(s, "%s: DRIF,L=%.15g",
                name, drift->length);
        break;
      case T_EDRIFT:
        edrift = (EDRIFT *)eptr->p_elem;
        sprintf(s, "%s: DRIF,L=%.15g",
                name, edrift->length);
        break;
      case T_CSRDRIFT:
        csrdrift = (CSRDRIFT *)eptr->p_elem;
        sprintf(s, "%s: DRIF,L=%.15g",
                name, csrdrift->length);
        break;
      case T_SEXT:
        sext = (SEXT *)eptr->p_elem;
        sprintf(s, "%s: SEXT,L=%.15g,K2=%.15g,TILT=%.15g",
                name, sext->length, sext->k2, sext->tilt);
        break;
      case T_KSEXT:
        ksext = (KSEXT *)eptr->p_elem;
        sprintf(s, "%s: SEXT,L=%.15g,K2=%.15g,TILT=%.15g",
                name, ksext->length, ksext->k2, ksext->tilt);
        break;
      case T_HCOR:
        hcor = (HCOR *)eptr->p_elem;
        sprintf(s, "%s: HKICKER,L=%.15g,KICK=%.15g",
                name, hcor->length, hcor->kick);
        break;
      case T_VCOR:
        vcor = (VCOR *)eptr->p_elem;
        sprintf(s, "%s: VKICKER,L=%.15g,KICK=%.15g",
                name, vcor->length, vcor->kick);
        break;
      case T_HVCOR:
        hvcor = (HVCOR *)eptr->p_elem;
        sprintf(s, "%s: KICKER,L=%.15g,HKICK=%.15g,VKICK=%.15g,TILT=%.15g",
                name, hvcor->length, hvcor->xkick, hvcor->ykick, hvcor->tilt);
        break;
      case T_EHCOR:
        ehcor = (EHCOR *)eptr->p_elem;
        sprintf(s, "%s: HKICKER,L=%.15g,KICK=%.15g",
                name, ehcor->length, ehcor->kick);
        break;
      case T_EVCOR:
        evcor = (EVCOR *)eptr->p_elem;
        sprintf(s, "%s: VKICKER,L=%.15g,KICK=%.15g",
                name, evcor->length, evcor->kick);
        break;
      case T_EHVCOR:
        ehvcor = (EHVCOR *)eptr->p_elem;
        sprintf(s, "%s: KICKER,L=%.15g,HKICK=%.15g,VKICK=%.15g,TILT=%.15g",
                name, ehvcor->length, ehvcor->xkick, ehvcor->ykick, ehvcor->tilt);
        break;
      case T_MARK:
        sprintf(s, "%s: MARK",
                name);
        break;
      default:
        fprintf(stderr, "warning: entity type %s not implemented---translated to MARK\n",
                entity_name[eptr->type]);
        sprintf(s, "%s: MARK", name);
        break;
      }
      print_with_continuation(fp, s, 79);
      s[0] = 0;
    }
    eptr = eptr->succ;
    count++;
  }

  lcount = 1;
  eptr = beamline->elem;
  s[0] = 0;
  count = 0;
  while (eptr) {
    strcpy(name, eptr->name);
    replace_chars(name, ":.", "cd");
    if (s[0] == 0) {
      sprintf(s, "BL%04ld: LINE=(%s", lcount++, name);
    } else {
      strcat(s, ",");
      strcat(s, name);
    }
    count++;
    if (count == 100) {
      strcat(s, ")");
      print_with_continuation(fp, s, 79);
      s[0] = 0;
      count = 0;
    }
    eptr = eptr->succ;
  }
  if (count) {
    strcat(s, ")");
    print_with_continuation(fp, s, 79);
    s[0] = 0;
    count = 0;
  }

  s[0] = 0;
  for (count = 1; count < lcount; count++) {
    if (count == 1)
      sprintf(s, "bl: line=(BL%04ld", count);
    else {
      sprintf(buffer, ",BL%04ld", count);
      strcat(s, buffer);
      /* sprintf(s, "%s,BL%04ld", s, count); */
    }
  }
  strcat(s, ")");
  print_with_continuation(fp, s, 79);

  if (ender_file) {
    fpi = fopen_e(ender_file, "r", 1);
    while (fgets(s, 256, fpi))
      fputs(s, fp);
    fclose(fpi);
  }
}
