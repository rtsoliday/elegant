/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

#include "mdb.h"
#include "track.h"
#include "match_string.h"
#include "insert_elements.h"

typedef struct {
  char **name, *type, *exclude, *elemDef, *variable;
  long nNames, nskip, add_end, add_start, total, before, occur[100];
  double sStart, sEnd;
  long oStart, oEnd;
} ADD_ELEM;

static ADD_ELEM addElem;
static long add_elem_flag = 0;

#define COPYLEN 1024
/* static char elementDefCopy[COPYLEN+1]; */
static long insertCount = 0;
static long insertStartEndCount = 0;

long getAddElemFlag() {
  return (add_elem_flag);
}

char *getElemDefinition() {
  return (addElem.elemDef);
}

long getAddEndFlag() {
  if (addElem.add_end)
    insertStartEndCount++;
  return (addElem.add_end);
}

long getAddStartFlag() {
  if (addElem.add_start)
    insertStartEndCount++;
  return (addElem.add_start);
}

void do_insert_elements(NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline) {
  long i, nNames;
  char **nameList;

  /* process the namelist text */
  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&insert_elements, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (echoNamelists)
    print_namelist(stdout, &insert_elements);

  if (disable)
    return;
  if (!skip && !add_at_end && !add_at_start && !total_occurrences)
    bombElegant("skip, add_at_end, add_at_start, and total_occurrences can not be zero at the same time", NULL);

  if (total_occurrences) {
    if (skip)
      bombElegant("skip and total_occurrences can not be used together. One must have to be set to zero", NULL);
    if (has_wildcards(name))
      bombElegant("element name has to be specified if you use the occurrence feature", NULL);
  }
  add_elem_flag = 0;

  if ((!name || !strlen(name)) && !type)
    bombElegant("name or type needs to be given", NULL);
  if (!element_def || !strlen(element_def))
    bombElegant("element's definition is not given", NULL);

  nameList = NULL;
  nNames = 0;
  if (name) {
    char *ptr;
    str_toupper(name);
    while ((ptr = get_token_t(name, ", "))) {
      nameList = SDDS_Realloc(nameList, sizeof(*nameList) * (nNames + 1));
      nameList[nNames] = ptr;
      if (has_wildcards(ptr) && strchr(ptr, '-')) {
        nameList[nNames] = expand_ranges(ptr);
        free(ptr);
      } else
        nameList[nNames] = ptr;
      nNames++;
    }
  }

  if (type) {
    str_toupper(type);
    if (has_wildcards(type) && strchr(type, '-'))
      type = expand_ranges(type);
    for (i = 0; i < N_TYPES; i++)
      if (wild_match(entity_name[i], type))
        break;
    if (i == N_TYPES) {
      fprintf(stderr, "type pattern %s does not match any known type", type);
      exitElegant(1);
    }
  }
  if (exclude) {
    str_toupper(exclude);
    if (has_wildcards(exclude) && strchr(exclude, '-'))
      exclude = expand_ranges(exclude);
  }
  str_to_upper_quotes(element_def);

  if (start_at_element && strlen(start_at_element)) {
    ELEMENT_LIST *eptr;
    short matched;
    eptr = beamline->elem;
    matched = 0;
    while (eptr) {
      if (strcmp(eptr->name, start_at_element)==0) {
        matched = 1;
        break;
      }
      eptr = eptr->succ;
    }
    if (!matched)
      bombElegantVA("no match for start_at_element = %s", start_at_element);
    s_start = eptr->end_pos;
    printf("start_at_element = %s => s_start = %le\n", start_at_element, s_start);
  }
  if (end_at_element && strlen(end_at_element)) {
    ELEMENT_LIST *eptr;
    short matched;
    eptr = beamline->elast;
    matched = 0;
    while (eptr) {
      if (strcmp(eptr->name, end_at_element)==0) {
        matched = 1;
        break;
      }
      eptr = eptr->pred;
    }
    if (!matched)
      bombElegantVA("no match for end_at_element = %s", end_at_element);
    s_end = eptr->end_pos;
    printf("end_at_element = %s => s_end = %le\n", end_at_element, s_end);
  }

  add_elem_flag = 1;
  addElem.nskip = skip;
  addElem.add_end = add_at_end;
  addElem.add_start = add_at_start;
  addElem.name = nameList;
  addElem.nNames = nNames;
  addElem.type = type;
  addElem.exclude = exclude;
  addElem.elemDef = element_def;
  addElem.sStart = s_start;
  addElem.sEnd = s_end;
  addElem.oStart = occurrence_start;
  addElem.oEnd = occurrence_end;
  addElem.before = insert_before;
  addElem.variable = insertion_count_variable;
  delete_spaces(addElem.elemDef);
  /* strncpy(elementDefCopy, element_def, COPYLEN); */

  
  addElem.total = total_occurrences;
  for (i = 0; i < addElem.total; i++) {
    addElem.occur[i] = occurrence[i];
  }

  insertCount = 0;
  insertStartEndCount = 0;
  beamline = get_beamline(NULL, beamline->name, run->p_central, 0, 0, NULL, NULL);
  if (run->backtrack)
    beamline->flags |= BEAMLINE_BACKTRACKING;
  compute_end_positions(beamline);

  add_elem_flag = 0;
  if (verbose)
    printf("%ld elements inserted in total\n", insertCount+insertStartEndCount);
  if (addElem.variable && strlen(addElem.variable))
    rpn_store((double)(insertCount+insertStartEndCount), NULL, rpn_create_mem(addElem.variable, 0));
  if (insertCount==0) {
    if (allow_no_matches) {
      char warningBuffer[16384];
      snprintf(warningBuffer, 16384, "Attempting insertion of %s", element_def);
      printWarning("No matches found, nothing inserted for insert_elements", warningBuffer);
    } else
      bombElegant("No matches found, no instances inserted.", NULL);
  }
  if ((insertCount+insertStartEndCount)==0) {
    if (allow_no_insertions) {
      char warningBuffer[16384];
      snprintf(warningBuffer, 16384, "Attempting insertion of %s", element_def);
      printWarning("Nothing inserted for insert_elements", warningBuffer);
    } else
      bombElegant("No instances inserted.", NULL);
  }

  return;
}

long insertElem(char *name, long type, long *skip, long occurPosition, double endPosition) {
  long i;

  if (addElem.exclude && wild_match(name, addElem.exclude))
    return (0);
  if (addElem.type && !wild_match(entity_name[type], addElem.type))
    return (0);
  if (addElem.name) {
    for (i = 0; i < addElem.nNames; i++) {
      if (addElem.name[i] && wild_match(name, addElem.name[i]))
        break;
    }
    if (i == addElem.nNames)
      return (0);
  }

  if ((addElem.sStart >= 0 && endPosition < addElem.sStart) ||
      (addElem.sEnd >= 0 && endPosition > addElem.sEnd))
    return 0;
  if ((addElem.oStart >= 0 && occurPosition < addElem.oStart) ||
      (addElem.oEnd >= 0 && occurPosition > addElem.oEnd))
    return 0;

  if (addElem.total) {
    for (i = 0; i < addElem.total; i++) {
      if (occurPosition == addElem.occur[i]) {
        if (verbose)
          printf("Adding \"%s\" %s occurrence %ld of %s\n",
                 addElem.elemDef, addElem.before ? "before" : "after", occurPosition, name);
        insertCount++;
        return (addElem.before ? 2 : 1);
      }
    }
    return (0);
  }

  if (addElem.nskip) {
    (*skip)++;
    if (*skip < addElem.nskip)
      return (0);
    *skip = 0;
    if (verbose)
      printf("Adding \"%s\" %s occurrence %ld of %s\n",
             addElem.elemDef, addElem.before ? "before" : "after", occurPosition, name);
    insertCount++;
    return (addElem.before ? 2 : 1);
  }
  return (0);
}
