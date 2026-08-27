/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* file    : mad_parse.c
 * contents: fill_line(), fill_elem(), expand(), copy_element(),
 *           copy_line(), tell_type(), get_param_name(), find_param()
 *
 *           Routines to fill LINE and ELEMENT structures by parsing
 *           a line of text.
 *
 * Michael Borland, 1987, 1988, 1989, 1991.
 */
#include "mdb.h"
#include "track.h"
#include <ctype.h>
#include <string.h>
#include <memory.h>

long is_simple(char *s);
void setEdgeIndices(ELEMENT_LIST *e1);
void swapEdgeIndices(ELEMENT_LIST *e1);

long max_name_length = 100;

long set_max_name_length(long length) {
  long tmp;
  if (length <= 0)
    bombElegant("maximum name length set to nonpositive number", NULL);
  tmp = max_name_length;
  max_name_length = length;
  return (tmp);
}

/* routine: fill_line()
 * purpose: fill a LINE_LIST structure by parsing a line of text.
 */

void fill_line(
  LINE_LIST *line,    /* linked list of line definitions */
  long nl,            /* number of line definitions so far */
  ELEMENT_LIST *elem, /* linked list of element definitions */
  long ne,            /* number of element definitions so far */
  char *s             /* text to parse into a new line definition */
) {
  register char *ptr;
  register long i;
  ELEMENT_LIST *leptr;
  LINE_LIST *lptr;
  long quoteCounter;

  log_entry("fill_line");

#ifdef DEBUG
  printf("expanding beamline: %s\n", s);
  fflush(stdout);
#endif

  ptr = s;
  quoteCounter = 0;
  while (*ptr)
    if (*ptr++ == '"')
      quoteCounter++;
  if (quoteCounter % 2 != 0)
    bombElegantVA("Mismatched quotation marks in LINE definition:\n%s\n", s);

  /* get pointer to empty spot in list of LINE definitions */
  lptr = line;
  for (i = 0; i < nl; i++)
    lptr = lptr->succ;
  if (lptr == NULL)
    bombElegant("improperly extended linked-list (fill_line)", NULL);

  /* allocate string in which to save original beamline definition for later use */
  lptr->definition = tmalloc(sizeof(char) * (strlen(s) + 12));

  /* get beamline name */
  if ((ptr = strchr(s, ',')) == NULL) {
    printf("error parsing LINE: %s\n", s);
    fflush(stdout);
    exitElegant(1);
  }
  *ptr = 0;
  lptr->name = get_token_t(s, " ,\011");
#ifdef DEBUG
  printf("line name is %s\n", lptr->name);
  fflush(stdout);
#endif
  if (strpbrk(lptr->name, ":.,/-_+abcdefghijklmnopqrstuvwyxz "))
    sprintf(lptr->definition, "\"%s\": LINE=%s", lptr->name, ptr + 1);
  else
    sprintf(lptr->definition, "%s: LINE=%s", lptr->name, ptr + 1);
  ptr += 2;

  leptr = lptr->elem = tmalloc(sizeof(*leptr));
  lptr->n_elems = 0;
  lptr->flags = 0;

  expand_line(leptr, lptr, ptr, line, nl, elem, ne, lptr->name);
  while (leptr->succ != NULL) {
    leptr = leptr->succ;
  }
  if (leptr->pred == NULL)
    bombElegant("programming error: problem in linked list of beamlines", NULL);
  (leptr->pred)->succ = NULL;
  log_exit("fill_line");
}

extern ELEMENT_LIST *expand_line(
  ELEMENT_LIST *leptr, /* pointer to list of elements for new line */
  LINE_LIST *lptr,     /* pointer to the line structure new line */
  char *s,             /* string defining the new line */
  LINE_LIST *line,     /* pointer to linked-list of line structures for existing lines */
  long nl,             /* number of existing lines */
  ELEMENT_LIST *elem,  /* pointer to linked-list of element structures for defined elements */
  long ne,             /* number of defined elements */
  char *part_of        /* name of the group that elements will be part of if they are not members of sub-lines */
) {
  register char *ptr, *ptr1;
  char *ptrs = NULL, *ptr2 = NULL;
  register long i, l;
  long multiplier = 1, reverse = 0, simple, count1;
  char *line_save = NULL;

  log_entry("expand_line");

  cp_str(&line_save, s);

#ifdef DEBUG
  printf("line is %s\n", s);
  fflush(stdout);
#endif

  count1 = 0;
  while ((ptr2 = get_token_tq(s, ",", ",", "(\"", ")\""))) {
    ptr1 = ptr2;
    ptr = ptr1 + strlen(ptr1) - 1;
    while (isspace(*ptr))
      ptr--;
    if (*ptr == ')')
      *ptr = 0;
    if (*ptr1 == '-')
      delete_bounding(ptr1 + 1, "\"");
    else
      delete_bounding(ptr1, "\"");
#ifdef DEBUG
    printf("ptr1 = %s (1)\n", ptr1);
    fflush(stdout);
#endif
    reverse = 0;
    if (*ptr1 == '-') {
      reverse = 1;
      ptr1++;
    }

    multiplier = 1;
    if (isdigit(*ptr1)) {
      if ((ptr = strchr(ptr1, '*'))) {
        *ptr = 0;
        if (1 != sscanf(ptr1, "%ld", &multiplier)) {
          printf("problem with line: %s\n", line_save);
          fflush(stdout);
          printf("position is %s\n", ptr1);
          fflush(stdout);
          bombElegant("improper element multiplication", NULL);
        }
        ptr1 = ptr + 1;
        if (*ptr1 == '(' || *ptr1 == '"')
          ptr1++;
      }
    }
#ifdef DEBUG
    printf("reverse = %ld, multiplier = %ld\n", reverse, multiplier);
    fflush(stdout);
    printf("ptr1 = %s (2)\n", ptr1);
    fflush(stdout);
#endif

    simple = is_simple(ptr1);
#ifdef DEBUG
    printf("simple = %ld\n", simple);
    fflush(stdout);
#endif
    count1 += multiplier;
    if (simple) {
      /* add simple elements to the line's element list */
      lptr->n_elems +=
        (l = expand_phys(leptr, ptr1, elem, ne, line, nl, reverse, multiplier, part_of));
#ifdef DEBUG
      printf("number of elements now %ld\n", lptr->n_elems);
#endif
      for (i = 0; i < l; i++)
        leptr = leptr->succ;
    } else {
      /* add elements from a parenthesized list to the line's element list */
      cp_str(&ptrs, ptr1);
      for (i = 0; i < multiplier; i++) {
        strcpy_ss(ptr1, ptrs);
        leptr = expand_line(leptr, lptr, ptr1, line, nl, elem, ne, part_of);
      }
#ifdef DEBUG
      printf("number of elements now %ld\n", lptr->n_elems);
#endif
      tfree(ptrs);
      ptrs = NULL;
    }
    free(ptr2);
  }

  if (count1 > lptr->n_elems) {
    printf("Problem with line parsing: expected at least %ld elements, but got %ld\n",
           count1, lptr->n_elems);
    print_line(stdout, lptr);
  }
  if (line_save)
    free(line_save);
  log_exit("fill_line");
  return (leptr);
}

long is_simple(char *s) {
  register char *ptr;
  log_entry("is_simple");
  ptr = s;
  while (*ptr) {
    if (*ptr == '(' || *ptr == ',' || *ptr == '*') {
      log_exit("is_simple");
      return (0);
    }
    ptr++;
  }
  log_exit("is_simple");
  return (1);
}

/* routine: fill_elem()
 * purpose: fill a single physical element slot in an element list
 */

void fill_elem(ELEMENT_LIST *eptr, char *s, long type, FILE *fp_input) {
  BEND *bptr;
  MATR *matr;

  log_entry("fill_elem");
  eptr->end_pos = eptr->flags = 0;
  eptr->ignore = 0;

  if ((eptr->type = type) >= N_TYPES || type <= 0) {
    printf("unknown element type %ld in fill_elem()\n", type);
    fflush(stdout);
    printf("remainder of line is: \n%s\n", s);
    fflush(stdout);
    exitElegant(1);
  }

  eptr->name = get_token_t(s, " ,\011");
  eptr->namelen = strlen(eptr->name);
  if (((long)strlen(eptr->name)) > max_name_length) {
    char warningText[1024];
    snprintf(warningText, 1024, "Element name truncated to %ld  characters",
             max_name_length);
    printWarning(warningText, eptr->name);
    eptr->name[max_name_length] = 0;
  }
  cp_str(&eptr->definition_text, s);
  eptr->matrix = NULL;
  eptr->group = NULL;
  eptr->p_elem = tmalloc(entity_description[type].structure_size);
  eptr->p_elem0 = tmalloc(entity_description[type].structure_size);
  zero_memory(eptr->p_elem, entity_description[type].structure_size);
  zero_memory(eptr->p_elem0, entity_description[type].structure_size);

  parse_element((char *)(eptr->p_elem),
                entity_description[type].parameter,
                entity_description[type].n_params,
                s, eptr, entity_name[type]);

  if (IS_BEND(type) || type == T_NIBEND || type == T_TAPERAPC || type == T_TAPERAPE || type == T_TAPERAPR)
    setEdgeIndices(eptr);

  switch (type) {
  case T_RBEN:
    bptr = (BEND *)(eptr->p_elem);
    bptr->e[0] += bptr->angle / 2;
    bptr->e[1] += bptr->angle / 2;
    type = eptr->type = T_SBEN;
    if (fabs(bptr->angle) > 1e-14)
      bptr->length *= (bptr->angle / 2) / sin(bptr->angle / 2);
    break;
  case T_LGBEND:
    readLGBendConfiguration((LGBEND *)(eptr->p_elem), eptr);
    break;
  case T_PEPPOT:
    parse_pepper_pot((PEPPOT *)(eptr->p_elem), fp_input, eptr->name);
    break;
  case T_MATR:
    matr = (MATR *)eptr->p_elem;
    if (!matr->matrix_read) {
      char *filename;
      FILE *fpm;
      if (!(filename = findFileInSearchPath(matr->filename))) {
        fprintf(stderr, "Unable to find MATR file %s\n", matr->filename);
        exitElegant(1);
      }
      printf("File %s found: %s\n", matr->filename, filename);
      fpm = fopen_e(filename, "r", 0);
      matr->M.order = matr->order;
      initialize_matrices(&(matr->M), matr->order);
      if (!read_matrices(&(matr->M), filename, fpm)) {
        printf("error reading matrix from file %s\n", matr->filename);
        fflush(stdout);
        abort();
      }
      free(filename);
      fclose(fpm);
      matr->matrix_read = 1;
      matr->length = matr->M.C[4];
    }
    break;
  case T_SCRAPER:
    ((SCRAPER *)(eptr->p_elem))->direction = interpretScraperDirection(((SCRAPER *)(eptr->p_elem))->insert_from,
                                                                       ((SCRAPER *)(eptr->p_elem))->oldDirection);
    break;
  case T_SPEEDBUMP:
    ((SPEEDBUMP *)(eptr->p_elem))->direction = interpretScraperDirection(((SPEEDBUMP *)(eptr->p_elem))->insertFrom, -1);
    break;
  default:
    break;
  }

  eptr->flags = PARAMETERS_ARE_STATIC; /* default, to be changed by variation and error settings */

  copy_p_elem(eptr->p_elem0, eptr->p_elem, type);

  log_exit("fill_elem");
}

/* routine: copy_named_element()
 * purpose: fill a single physical element slot in an element list
 *          by copying a named, previously-defined element
 */

void copy_named_element(ELEMENT_LIST *eptr, char *s, ELEMENT_LIST *elem) {
  /* This doesn't appear to be used now. */
  char *name, *match;

  log_entry("copy_named_element");
  eptr->end_pos = eptr->flags = 0;

  if (!(name = get_token_tq(s, "", ":", " \"", " \""))) {
    log_exit("copy_named_element");
    return;
  }
  delete_bounding(name, "\"");

  if (!(match = get_token_tq(s, "", ",=", " \"", " \""))) {
    log_exit("copy_named_element");
    bombElegant("programming error--unable to recover source element name (copy_named_element)", NULL);
  }
  delete_bounding(match, "\"");

  if (((long)strlen(name)) > max_name_length) {
    char warningText[1024];
    snprintf(warningText, 1024, "Element name truncated to %ld  characters",
             max_name_length);
    printWarning(warningText, name);
    name[max_name_length] = 0;
  }

#ifdef DEBUG
  printf("seeking match for name %s to copy for defining name %s\n",
         match, name);
  fflush(stdout);
#endif

  while (elem && elem->name) {
    if (strcmp(elem->name, match) == 0)
      break;
    elem = elem->succ;
  }
  if (!elem->name) {
    log_exit("copy_named_element");
    printf("unable to define %s as copy of %s--source element does not exist\n",
           name, match);
    fflush(stdout);
    exitElegant(1);
  }
#ifdef DEBUG
  printf("match found for %s: %s\n", name, match);
  fflush(stdout);
#endif

  eptr->p_elem = tmalloc(entity_description[elem->type].structure_size);
  eptr->p_elem0 = tmalloc(entity_description[elem->type].structure_size);
  copy_p_elem(eptr->p_elem, elem->p_elem, elem->type);
  copy_p_elem(eptr->p_elem0, eptr->p_elem, elem->type);
  cp_str(&eptr->name, name);
  eptr->group = NULL;
  if (elem->group)
    cp_str(&eptr->group, elem->group);
  eptr->matrix = NULL;
  eptr->flags = PARAMETERS_ARE_STATIC;
  eptr->type = elem->type;
  eptr->ignore = elem->ignore;
  log_exit("copy_named_element");
}

static long *entity_name_length = NULL;

/* routine: expand_phys()
 * purpose: expand a named entity into physical elements 
 */

long expand_phys(
  ELEMENT_LIST *leptr,     /* list being filled with expansion */
  char *entity,            /* entity to expand */
  ELEMENT_LIST *elem_list, /* list of all physical elements */
  long ne,                 /* number of same */
  LINE_LIST *line_list,    /* list of already-existing LINEs */
  long nl,                 /* number of same */
  long reverse,
  long multiplier,
  char *part_of /* name of line this is to be part of */
) {
  long ie, il, i, j, comparison, div;
  char trunc_char, *editCmd;
  double length;

  log_entry("expand_phys");

  /* search for truncated name, for the case that it turns out
   * to be an element */
  trunc_char = 0;
  if (((long)strlen(entity)) > max_name_length) {
    trunc_char = entity[max_name_length];
    entity[max_name_length] = 0;
  }
  div = 1;
  for (ie = 0; ie < ne; ie++) {
    if ((comparison = strcmp(entity, elem_list->name)) == 0) {
#ifdef DEBUG
      printf("adding element %ld*%s\n", multiplier, entity);
      fflush(stdout);
#endif
      if (entity_description[elem_list->type].flags & DIVIDE_OK &&
          entity_description[elem_list->type].flags & HAS_LENGTH &&
          (length = *(double *)(elem_list->p_elem)) > 0) {
        div = elementDivisions(entity, entity_name[elem_list->type], length);
        if (div > 1) {
          elem_list->divisions = div;
          if (elem_list->type==T_CWIGGLER) {
            CWIGGLER *cwig;
            cwig = (CWIGGLER*)(elem_list->p_elem);
            if (cwig->periods%div!=0)
              bombElegantVA("Can't divide %ld-period CWIGGLER %s into %ld parts. Must divide at period boundaries.",
                            cwig->periods, elem_list->name, div);
          }
        }
      }
#ifdef DEBUG
      fprintf(stderr, "Dividing %s %ld times\n",
              entity, div);
#endif
      for (i = 0; i < multiplier; i++) {
        long j;
        for (j = 0; j < div; j++) {
          copy_element(leptr, elem_list, reverse, j, div, NULL);
          leptr->part_of = elem_list->part_of ? elem_list->part_of : part_of;
#ifdef DEBUG
          printf("expand_phys copied: name=%s divisions=%ld first=%hd j=%ld\n",
                 leptr->name, leptr->divisions, leptr->firstOfDivGroup, j);
#endif
          extend_elem_list(&leptr);
        }
      }
      return (multiplier * div);
    }
    if (comparison < 0)
      break;
    elem_list = elem_list->succ;
  }

  /* it isn't an element, so search list of beam-lines for occurence
   * of the full name 
   */

  if (trunc_char)
    entity[max_name_length] = trunc_char;
  if ((editCmd = strstr(entity, "<<"))) {
    /* extract edit command for this line */
    if (strlen(editCmd + 2) == 0) {
      fprintf(stderr, "Problem with edit command for beamline segment: %s\n", entity);
      exitElegant(1);
    }
    *editCmd = 0;
    editCmd += 2;
  }

  for (il = 0; il < nl; il++) {
    if (strcmp(line_list->name, entity) == 0) {
      ie = 0;
      for (i = 0; i < multiplier; i++) {
        ie += copy_line(leptr, line_list->elem, line_list->n_elems, reverse, entity, editCmd);
        for (j = 0; j < line_list->n_elems; j++)
          leptr = leptr->succ;
      }
      log_exit("expand_phys");
      return (ie);
    }
    line_list = line_list->succ;
  }

  printf("no expansion for entity %s\n", entity);
  fflush(stdout);
#if USE_MPI
  MPI_Barrier(MPI_COMM_WORLD); /* Make sure the information can be printed before aborting */
#endif
  exitElegant(1);
  return (0);
}

/* routine: copy_element()
 * purpose: copy an element 
 */

void copy_element(ELEMENT_LIST *e1, ELEMENT_LIST *e2, long reverse, long division, long divisions, char *editCmd) {
  log_entry("copy_element");
  if (editCmd) {
    char *buffer;
    buffer = tmalloc(sizeof(*buffer) * (strlen(e2->name) + 1) * 10);
    strcpy(buffer, e2->name);
    edit_string(buffer, editCmd);
    cp_str(&e1->name, buffer);
    free(buffer);
  } else {
    cp_str(&e1->name, e2->name);
  }
  e1->group = NULL;
  if (e2->group) {
    if (editCmd) {
      char *buffer;
      buffer = tmalloc(sizeof(*buffer) * (strlen(e2->group) + 1) * 10);
      strcpy(buffer, e2->group);
      edit_string(buffer, editCmd);
      cp_str(&e1->group, buffer);
      free(buffer);
    } else {
      cp_str(&e1->group, e2->group);
    }
  }
  e1->end_pos = 0;
  e1->flags = 0;
  e1->matrix = NULL;
  e1->type = e2->type;
  e1->ignore = e2->ignore;
  e1->p_elem = tmalloc(entity_description[e1->type].structure_size);
  e1->p_elem0 = tmalloc(entity_description[e1->type].structure_size);
  if (reverse)
    e1->reversed = !e2->reversed;
  else
    e1->reversed = e2->reversed;
  copy_p_elem(e1->p_elem, e2->p_elem, e1->type);
  copy_p_elem(e1->p_elem0, e2->p_elem, e1->type);
  e1->divisions = 1;
  if (reverse && (IS_BEND(e1->type) || e1->type == T_TAPERAPC || e1->type == T_TAPERAPE || e1->type == T_TAPERAPR))
    swapEdgeIndices(e1);
  if (reverse && e1->type==T_BEDGE) {
    BEDGE *be1, *be2;
    be1 = (BEDGE*)e1->p_elem;
    be2 = (BEDGE*)e2->p_elem;
    be1->exitEdge = !be2->exitEdge;
  }
  e1->firstOfDivGroup = 0;
  if (reverse && divisions <= 1 && e1->type == T_KQUAD) {
    KQUAD *kqptr;
    kqptr = (KQUAD *)e1->p_elem;
    SWAP_LONG(kqptr->edge1_effects, kqptr->edge2_effects);
  }
  if (divisions > 1) {
    if (division == 0)
      e1->firstOfDivGroup = 1;
    /* TODO: use PARAM_DIVISION_RELATED to decide which parameters to divide */
    if (entity_description[e1->type].flags & HAS_LENGTH)
      *(double *)(e1->p_elem) /= divisions;
    if (e1->type == T_CWIGGLER) {
      CWIGGLER *cwig;
      cwig = (CWIGGLER*)e1->p_elem;
      cwig->periods /= divisions;
    } else if (e1->type == T_RFCA) {
      RFCA *rfca;
      rfca = (RFCA *)e1->p_elem;
      rfca->volt /= divisions;
    } else if (e1->type == T_KQUAD) {
      KQUAD *kqptr;
      kqptr = (KQUAD *)e1->p_elem;
      if (kqptr->lEffective != 0) {
        printf("Error: can't have non-zero LEFFECTIVE when dividing KQUAD elements (name %s)\n", e2->name);
        exit(1);
      }
      if (kqptr->edge_multipoles) {
        printf("Error: can't have EDGE_MULTIPOLES when dividing KQUAD elements (name %s)\n", e2->name);
        exit(1);
      }
      if (division != 0)
        kqptr->edge1_effects = 0;
      if (division != (divisions - 1))
        kqptr->edge2_effects = 0;
      if (reverse)
        SWAP_LONG(kqptr->edge1_effects, kqptr->edge2_effects);
    } else if (IS_BEND(e1->type)) {
      BEND *bptr;
      CSBEND *csbptr;
      switch (e1->type) {
      case T_SBEN:
        bptr = (BEND *)e1->p_elem;
        bptr->angle /= divisions;
        break;
      case T_CSBEND:
        csbptr = (CSBEND *)e1->p_elem;
        csbptr->angle /= divisions;
        csbptr->nSlices = csbptr->nSlices / divisions + 1;
        break;
      default:
        printf("Internal error: Attempt to divide angle for element that is not supported (type=%ld)\n",
               e1->type);
        exitElegant(1);
        break;
      }
    }
    e1->divisions = divisions;
  }
  log_exit("copy_element");
}

/* routine: copy_line
 * purpose: copy a series of elements from a linked-list 
 */

long copy_line(ELEMENT_LIST *e1, ELEMENT_LIST *e2, long ne, long reverse, char *part_of, char *editCmd) {
  register long i;

  log_entry("copy_line");

  if (!reverse) {
    for (i = 0; i < ne; i++) {
      copy_element(e1, e2, reverse, 0, 0, editCmd);
      e1->part_of = e2->part_of ? e2->part_of : part_of;
      e1->divisions = e2->divisions;
      e1->occurence = e2->occurence;
      e1->firstOfDivGroup = e2->firstOfDivGroup;
      extend_elem_list(&e1);
      e2 = e2->succ;
    }
  } else {
    for (i = 0; i < (ne - 1); i++)
      e2 = e2->succ;
    for (i = 0; i < ne; i++) {
      copy_element(e1, e2, reverse, 0, 0, editCmd);
      e1->part_of = e2->part_of ? e2->part_of : part_of;
      e1->divisions = e2->divisions;
      e1->occurence = e2->occurence;
      e1->firstOfDivGroup = e2->firstOfDivGroup;
      extend_elem_list(&e1);
      e2 = e2->pred;
    }
  }
  log_exit("copy_line");
  return (ne);
}

/* routine: tell_type()
 * purpose: return the integer type code for a physical element 
 *         
 */

long tell_type(char *s, ELEMENT_LIST *elem) {
  long i, l, match_found, return_value, comparison;
  char *ptr, *ptr1, *name, *name1, *buffer;

  match_found = 0;
  return_value = T_NODEF;

  log_entry("tell_type");

  if (!(name1 = get_token_tq(s, "", ":", " \"", " \""))) {
    log_exit("tell_type");
    return (T_NODEF);
  }
  name = name1;
  delete_bounding(name, "\"");
#ifdef DEBUG
  printf("first token on line: >%s<\n", name);
  fflush(stdout);
  printf("remainder of line: >%s<\n", s);
  fflush(stdout);
#endif
  ptr1 = get_token_tq(s, "", ",=", " \"", " \"");
  ptr = ptr1;
  delete_bounding(ptr, "\"");
#ifdef DEBUG
  printf("second token: >%s<\n", ptr);
  fflush(stdout);
#endif
  if (!ptr || is_blank(ptr)) {
#ifdef DEBUG
    printf("second token is blank\n");
    fflush(stdout);
#endif
    if ((ptr = strchr(name, ','))) {
      strcpy_ss(s, ptr + 1);
      *ptr = 0;
    }
    ptr = name;
    name = NULL;
  } else {
    insert(s, ",");
    insert(s, name);
    /* s now has the form "name,<remainder-of-s>".  The original second token of s is
         * pointed to by ptr, but is not in s. */
  }
#ifdef DEBUG
  printf("seeking to match %s to entity name\n", ptr);
  fflush(stdout);
#endif
  if (entity_name_length == NULL) {
    entity_name_length = tmalloc(sizeof(*entity_name_length) * N_TYPES);
    for (i = 0; i < N_TYPES; i++)
      entity_name_length[i] = strlen(entity_name[i]);
  }
  l = strlen(ptr);
  for (i = 0; i < N_TYPES; i++) {
    if (strncmp(ptr, entity_name[i], MIN(l, entity_name_length[i])) == 0) {
      if (match_found) {
        printf("error: item %s is ambiguous--specify more of the item name\n", ptr);
        fflush(stdout);
        exitElegant(1);
      }
      if (l == entity_name_length[i]) {
        if (name1)
          free(name1);
        if (ptr1)
          free(ptr1);
        log_exit("tell_type");
        return (i);
      }
      return_value = i;
      match_found = 1;
#ifdef DEBUG
      printf("%s matches entity name %s\n", ptr, entity_name[i]);
      fflush(stdout);
#endif
    }
  }
#ifdef DEBUG
  printf("seeking to match %s to command name\n", ptr);
  fflush(stdout);
#endif
  for (i = 1; i < N_MADCOMS; i++) {
    if (strncmp(ptr, madcom_name[i], MIN(l, strlen(madcom_name[i]))) == 0) {
      if (match_found) {
        printf("error: item %s is ambiguous--specify more of the item name\n", ptr);
        fflush(stdout);
        exitElegant(1);
      }
      if (l == ((long)strlen(madcom_name[i]))) {
        if (name1)
          free(name1);
        if (ptr1)
          free(ptr1);
        log_exit("tell_type");
        return (-i - 1);
      }
      return_value = -i - 1;
      match_found = 1;
#ifdef DEBUG
      printf("%s matches command name %s\n", ptr, madcom_name[i]);
      fflush(stdout);
#endif
    }
  }
#ifdef DEBUG
  printf("seeking to match %s to element name\n", ptr);
  fflush(stdout);
#endif
  /* this code depends on the elements being previously sorted into
     * strcmp() order.  This is done by check_duplic_elem
     */
  while (elem && elem->name) {
    if ((comparison = strcmp(elem->name, ptr)) == 0) {
      if (match_found)
        printWarning("Element reference is ambiguous---assuming element copy desired",
                     ptr);
      if (!elem->definition_text)
        bombElegant("element copy with no definition_text--internal error", NULL);
      buffer = tmalloc(sizeof(*buffer) * (strlen(elem->definition_text) + strlen(s) + 1));
      if ((ptr = strchr(s, ','))) {
#ifdef DEBUG
        printf("inserting into: %s\n", s);
        fflush(stdout);
        printf("inserting: %s\n", elem->definition_text);
        fflush(stdout);
#endif
        insert(ptr, elem->definition_text);
      } else {
        strcat(s, ",");
        strcat(s, buffer);
      }
      free(buffer);
#ifdef DEBUG
      printf("modified input text for element %s:\n%s\n", name, s);
      fflush(stdout);
#endif
      if (name1)
        free(name1);
      if (ptr1)
        free(ptr1);
      log_exit("tell_type");
      return (elem->type);
    }
    if (comparison > 1)
      break;
    elem = elem->succ;
  }
  if (name1)
    free(name1);
  if (ptr1)
    free(ptr1);
  log_exit("tell_type");
  return (return_value);
}

/* routine: get_param_name()
 * purpose: extract the parameter name from a string of the type
 *          "L=3.0331".  A pointer to the name is returned and the
 *          string is altered to bring the number to the beginning.
 */

char *get_param_name(char *s) {
  char *ptr, *ptr1 = NULL;

  log_entry("get_param_name");

  if ((ptr = strchr(s, '=')) == NULL) {
    printf("get_param_name(): no parameter name found in string %s\n", s);
    fflush(stdout);
    exitElegant(1);
  }
  *ptr = 0;
  cp_str(&ptr1, s);
  strcpy_ss(s, ptr + 1);
  log_exit("get_param_name");
  return (ptr1);
}

/* routine: find_param
 * purpose: find the element parameter with a given name, and return a
 *          pointer to the name (or NULL if not found).  The string is
 *          altered so that the value to be scanned appears at the
 *          beginning.
 */

char *find_param(char *s, char *param) {
  char *ptr;
  long l;

  log_entry("find_param");

  if (!*s) {
    log_exit("find_param");
    return (NULL);
  }

  l = strlen(param);

  while ((ptr = get_param_name(s))) {
    if (strncmp(param, ptr, l) == 0) {
      log_exit("find_param");
      return (ptr);
    }
  }
  log_exit("find_param");
  return (NULL);
}

void unknown_parameter(char *parameter,
                       char *element, char *type_name, char *caller) {
  printf("error: unknown parameter %s used for %s %s (%s)\n",
         parameter, type_name, element, caller);
  fflush(stdout);
  exitElegant(1);
}

void parse_element(
  char *p_elem,
  PARAMETER *parameter,
  long n_params,
  char *string,
  ELEMENT_LIST *eptr,
  char *type_name) {
  long i, difference, isGroup, pType = 0;
  char *ptr, *ptr1 = NULL, *rpn_token;

  log_entry("parse_element");

  if (!(entity_description[eptr->type].flags & OFFSETS_CHECKED)) {
    if (parameter[0].offset < 0) {
      printf("error: bad initial parameter offset for element type %s\n", type_name);
      fflush(stdout);
      exitElegant(1);
    }
    for (i = 1; i < n_params; i++) {
      /* difference of offsets must be positive and less than size of double */
      if ((difference = parameter[i].offset - parameter[i - 1].offset) <= 0 && !(parameter[i].flags & PARAM_IS_ALIAS)) {
        printf("error: bad parameter offset (retrograde) for element type %s, parameter %ld (%s)\n",
               type_name, i, parameter[i].name ? parameter[i].name : "NULL");
        fflush(stdout);
        exitElegant(1);
      }
      if (parameter[i - 1].flags & PARAM_IS_ALIAS) {
        if (difference != 0) {
          printf("error: bad parameter offset (should be zero) for element type %s, parameter %ld (%s)\n",
                 type_name, i, parameter[i].name ? parameter[i].name : "NULL");
          fflush(stdout);
          exitElegant(1);
        }
      } else {
        if (difference > sizeof(double) && eptr->type != T_TWISSELEMENT && eptr->type != T_EMATRIX) {
          printf("error: bad parameter offset (too large) for element type %s, parameter %ld (%s)\n",
                 type_name, i, parameter[i].name ? parameter[i].name : "NULL");
          fflush(stdout);
          exitElegant(1);
        }
      }
    }
    entity_description[eptr->type].flags |= OFFSETS_CHECKED;
  }

  for (i = 0; i < n_params; i++) {
    switch (parameter[i].type) {
    case IS_DOUBLE:
      *(double *)(p_elem + parameter[i].offset) = parameter[i].number;
      break;
    case IS_LONG:
      *(long *)(p_elem + parameter[i].offset) = parameter[i].integer;
      break;
    case IS_INT64:
      *(int64_t *)(p_elem + parameter[i].offset) = parameter[i].integer;
      break;
    case IS_SHORT:
      *(short *)(p_elem + parameter[i].offset) = parameter[i].integer;
      break;
    case IS_STRING:
      if (parameter[i].string == NULL)
        *(char **)(p_elem + parameter[i].offset) = NULL;
      else
        cp_str((char **)(p_elem + parameter[i].offset),
               parameter[i].string);
      break;
    }
  }

  while ((ptr = get_token(string))) {
#ifdef DEBUG
    printf("Parsing %s\n", ptr);
#endif
    ptr1 = get_param_name(ptr);
    if (!ptr1) {
      printf("error getting parameter name for element %s\n", eptr->name);
      exitElegant(1);
    }
    isGroup = 0;
    if (strcmp(ptr1, "GROUP") == 0) {
      pType = IS_STRING;
      isGroup = 1;
    } else {
      for (i = 0; i < n_params; i++)
        if (strcmp(parameter[i].name, ptr1) == 0)
          break;
      if (i == n_params) {
        /* try again, ignoring underscores */
        for (i = 0; i < n_params; i++) {
          if (strcmp_skip(parameter[i].name, ptr1, "_") == 0)
            break;
        }
      }
      if (i < n_params)
        pType = parameter[i].type;
    }
    if (i == n_params && !isGroup) {
      printf("Error: unknown parameter %s used for %s %s (%s)\n",
             ptr1, eptr->name, type_name, "parse_element");
      fflush(stdout);
      fputs("valid parameters are:", stdout);
      for (i = 0; i < n_params; i++) {
        switch (parameter[i].type) {
        case IS_DOUBLE:
          printf("%s (%.16f %s)\n",
                 parameter[i].name, parameter[i].number,
                 parameter[i].unit);
          fflush(stdout);
          break;
        case IS_LONG:
          printf("%s (%ld %s)\n",
                 parameter[i].name, parameter[i].integer,
                 parameter[i].unit);
          fflush(stdout);
          break;
        case IS_INT64:
          printf("%s (%" PRId64 " %s)\n",
                 parameter[i].name, (int64_t)parameter[i].integer,
                 parameter[i].unit);
          fflush(stdout);
          break;
        case IS_SHORT:
          printf("%s (%hd %s)\n",
                 parameter[i].name, (short)parameter[i].integer,
                 parameter[i].unit);
          fflush(stdout);
          break;
        case IS_STRING:
          printf("%s (\"%s\")\n",
                 parameter[i].name,
                 parameter[i].string == NULL ? "{null}" : parameter[i].string);
          fflush(stdout);
          break;
        }
      }
      exitElegant(1);
    }
    free(ptr1);
    if (!*ptr) {
      printf("Error: missing value for parameter %s of %s\n",
             parameter[i].name, eptr->name);
      exitElegant(1);
    }
    if (!isGroup && parameter[i].flags & PARAM_IS_DEPRECATED) {
      char buffer1[1024], buffer2[1024];
      snprintf(buffer1, 1024, "Parameter %s of %s is deprecated.",
               parameter[i].name, entity_name[eptr->type]);
      snprintf(buffer2, 1024, "Element is %s. Please consult manual for details.", eptr->name);
      printWarning(buffer1, buffer2);
    }
    switch (pType) {
    case IS_DOUBLE:
      if (!isdigit(*ptr) && *ptr != '.' && *ptr != '-' && *ptr != '+') {
        rpn_token = get_token(ptr);
        SDDS_UnescapeQuotes(rpn_token, '"');
        *((double *)(p_elem + parameter[i].offset)) = rpn(rpn_token);
        if (rpn_check_error())
          exitElegant(1);
        printf("computed value for %s.%s is %.15e\n", eptr->name, parameter[i].name,
               *((double *)(p_elem + parameter[i].offset)));
        fflush(stdout);
      } else {
        if (sscanf(ptr, "%lf", (double *)(p_elem + parameter[i].offset)) != 1) {
          printf("Error scanning token %s for double value for parameter %s of %s.  Please check syntax.\n",
                 ptr, parameter[i].name, eptr->name);
          exitElegant(1);
        }
      }
      break;
    case IS_LONG:
      if (!isdigit(*ptr) && *ptr != '-' && *ptr != '+') {
        rpn_token = get_token(ptr);
        SDDS_UnescapeQuotes(rpn_token, '"');
        *((long *)(p_elem + parameter[i].offset)) = rpn(rpn_token);
        if (rpn_check_error())
          exitElegant(1);
        printf("computed value for %s.%s is %ld\n",
               eptr->name, parameter[i].name,
               *((long *)(p_elem + parameter[i].offset)));
        fflush(stdout);
      } else {
        if (sscanf(ptr, "%ld", (long *)(p_elem + parameter[i].offset)) != 1) {
          printf("Error scanning token %s for integer value for parameter %s of %s.  Please check syntax.\n",
                 ptr, parameter[i].name, eptr->name);
          exitElegant(1);
        }
      }
      break;
    case IS_INT64:
      if (!isdigit(*ptr) && *ptr != '-' && *ptr != '+') {
        rpn_token = get_token(ptr);
        SDDS_UnescapeQuotes(rpn_token, '"');
        *((int64_t *)(p_elem + parameter[i].offset)) = rpn(rpn_token);
        if (rpn_check_error())
          exitElegant(1);
        printf("computed value for %s.%s is %" PRId64 "\n",
               eptr->name, parameter[i].name,
               *((int64_t *)(p_elem + parameter[i].offset)));
        fflush(stdout);
      } else {
        if (sscanf(ptr, "%" SCNd64, (int64_t *)(p_elem + parameter[i].offset)) != 1) {
          printf("Error scanning token %s for integer value for parameter %s of %s.  Please check syntax.\n",
                 ptr, parameter[i].name, eptr->name);
          exitElegant(1);
        }
      }
      break;
    case IS_SHORT:
      if (!isdigit(*ptr) && *ptr != '-' && *ptr != '+') {
        rpn_token = get_token(ptr);
        SDDS_UnescapeQuotes(rpn_token, '"');
        *((short *)(p_elem + parameter[i].offset)) = rpn(rpn_token);
        if (rpn_check_error())
          exitElegant(1);
        printf("computed value for %s.%s is %hd\n",
               eptr->name, parameter[i].name,
               *((short *)(p_elem + parameter[i].offset)));
        fflush(stdout);
      } else {
        if (sscanf(ptr, "%hd", (short *)(p_elem + parameter[i].offset)) != 1) {
          printf("Error scanning token %s for integer value for parameter %s of %s.  Please check syntax.\n",
                 ptr, parameter[i].name, eptr->name);
        }
      }
      break;
    case IS_STRING:
      if (!isGroup)
        *(char **)(p_elem + parameter[i].offset) = get_token(ptr);
      else
        eptr->group = get_token(ptr);
      break;
    default:
      bombElegant("unknown data type in parse_element!", NULL);
      break;
    }
    free(ptr);
  }
  log_exit("parse_element");
}

void parse_pepper_pot(
  PEPPOT *peppot,
  FILE *fp,
  char *name) {
  long n_holes, i_hole;
  double *xc, *yc, x, y, cos_tilt, sin_tilt;
  char s[100];

  log_entry("parse_pepper_pot");

  n_holes = peppot->n_holes;
  peppot->x = xc = array_1d(sizeof(double), 0, n_holes - 1);
  peppot->y = yc = array_1d(sizeof(double), 0, n_holes - 1);
  i_hole = 0;
  sin_tilt = sin(peppot->tilt);
  cos_tilt = cos(peppot->tilt);

  for (i_hole = 0; i_hole < n_holes; i_hole++) {
    if (!fgets(s, 100, fp) ||
        !get_double(&x, s) || !get_double(&y, s)) {
      printf("error: data missing for pepper-pot plate %s\n",
             name);
      fflush(stdout);
      exitElegant(1);
    }
    xc[i_hole] = x * cos_tilt + y * sin_tilt;
    yc[i_hole] = -x * sin_tilt + y * cos_tilt;
  }

  /*
    printf("pepper pot plate %s:\n", name);
    fflush(stdout);
    printf("L=%lf  RADII=%lf  N_HOLES=%ld\n", 
            peppot->length, peppot->radii, peppot->n_holes);
    fflush(stdout);
    for (i_hole=0; i_hole<n_holes; i_hole++) 
        printf("x = %lf    y = %lf\n",
                peppot->x[i_hole], peppot->y[i_hole]);
        fflush(stdout);
 */
  log_exit("parse_pepper_pot");
}

unsigned long interpretScraperDirection(char *insert_from, long oldDirectionCode) {
  char sign = 0, plane = 0;
  unsigned long direction = 0;
  if (oldDirectionCode != -1 && (!insert_from || strlen(insert_from) == 0)) {
    switch (oldDirectionCode) {
    case 0:
      return DIRECTION_PLUS_X;
      break;
    case 1:
      return DIRECTION_PLUS_Y;
      break;
    case 2:
      return DIRECTION_MINUS_X;
      break;
    case 3:
      return DIRECTION_MINUS_Y;
      break;
    }
  }
  if (insert_from) {
    if (insert_from[0] == '+' || insert_from[0] == '-') {
      sign = insert_from[0];
      plane = insert_from[1];
    } else
      plane = insert_from[0];
    switch (toupper(plane)) {
    case 'Y':
    case 'V':
      switch (sign) {
      case '+':
        direction = DIRECTION_PLUS_Y;
        break;
      case '-':
        direction = DIRECTION_MINUS_Y;
        break;
      default:
        direction = DIRECTION_Y;
        break;
      }
      break;
    case 'X':
    case 'H':
      switch (sign) {
      case '+':
        direction = DIRECTION_PLUS_X;
        break;
      case '-':
        direction = DIRECTION_MINUS_X;
        break;
      default:
        direction = DIRECTION_X;
        break;
      }
      break;
    default:
      break;
    }
  }
  if (!direction) {
    char warningText[1024];
    snprintf(warningText, 1024,
             "Value is %s, should be one of one of x, h, y, or v, optionally preceeded by + or -",
             insert_from ? insert_from : "NULL");
    printWarning("Invalid insert_from parameter for SCRAPER.", warningText);
  }
  return direction;
}

void copy_p_elem(char *target, char *source, long type) {
  long i;
  PEPPOT *pps, *ppt;

  for (i = 0; i < entity_description[type].n_params; i++) {
    switch (entity_description[type].parameter[i].type) {
    case IS_DOUBLE:
      *((double *)(target + entity_description[type].parameter[i].offset)) =
        *((double *)(source + entity_description[type].parameter[i].offset));
      break;
    case IS_LONG:
      *((long *)(target + entity_description[type].parameter[i].offset)) =
        *((long *)(source + entity_description[type].parameter[i].offset));
      break;
    case IS_INT64:
      *((int64_t *)(target + entity_description[type].parameter[i].offset)) =
        *((int64_t *)(source + entity_description[type].parameter[i].offset));
      break;
    case IS_SHORT:
      *((short *)(target + entity_description[type].parameter[i].offset)) =
        *((short *)(source + entity_description[type].parameter[i].offset));
      break;
    case IS_STRING:
      cp_str(((char **)(target + entity_description[type].parameter[i].offset)),
             *((char **)(source + entity_description[type].parameter[i].offset)));
      break;
    default:
      bombElegant("invalid parameter type (copy_p_elem).  Seek professional help!", NULL);
      break;
    }
  }

  switch (type) {
  case T_SBEN:
  case T_RBEN:
    ((BEND *)target)->e1Index = ((BEND *)source)->e1Index;
    ((BEND *)target)->e2Index = ((BEND *)source)->e2Index;
    break;
  case T_KSBEND:
    ((KSBEND *)target)->e1Index = ((KSBEND *)source)->e1Index;
    ((KSBEND *)target)->e2Index = ((KSBEND *)source)->e2Index;
    break;
  case T_NIBEND:
    ((NIBEND *)target)->e1Index = ((NIBEND *)source)->e1Index;
    ((NIBEND *)target)->e2Index = ((NIBEND *)source)->e2Index;
    break;
  case T_CSBEND:
    ((CSBEND *)target)->e1Index = ((CSBEND *)source)->e1Index;
    ((CSBEND *)target)->e2Index = ((CSBEND *)source)->e2Index;
    ((CSBEND *)target)->edgeFlip = ((CSBEND *)source)->edgeFlip;
    break;
  case T_CSRCSBEND:
    ((CSRCSBEND *)target)->e1Index = ((CSRCSBEND *)source)->e1Index;
    ((CSRCSBEND *)target)->e2Index = ((CSRCSBEND *)source)->e2Index;
    break;
  case T_SCRAPER:
    ((SCRAPER *)target)->direction = ((SCRAPER *)source)->direction;
    break;
  case T_SPEEDBUMP:
    ((SPEEDBUMP *)target)->direction = ((SPEEDBUMP *)source)->direction;
    break;
  case T_CCBEND:
    ((CCBEND *)target)->optimized = ((CCBEND *)source)->optimized;
    ((CCBEND *)target)->fseOffset = ((CCBEND *)source)->fseOffset;
    ((CCBEND *)target)->dxOffset = ((CCBEND *)source)->dxOffset;
    ((CCBEND *)target)->xAdjust = ((CCBEND *)source)->xAdjust;
    ((CCBEND *)target)->KnDelta = ((CCBEND *)source)->KnDelta;
    for (int ii=0; ii<5; ii++)
      ((CCBEND *)target)->referenceData[ii] = ((CCBEND *)source)->referenceData[ii];
    for (int ii=0; ii<5; ii++)
      ((CCBEND *)target)->referenceTrajectory[ii] = ((CCBEND *)source)->referenceTrajectory[ii];
    ((CCBEND *)target)->edgeFlip = ((CCBEND *)source)->edgeFlip;
    break;
  case T_LGBEND:
    copyLGBend(((LGBEND *)target), ((LGBEND *)source));
    break;
  case T_BRAT:
    ((BRAT *)target)->initialized = ((BRAT *)source)->initialized;
    ((BRAT *)target)->dataIndex = ((BRAT *)source)->dataIndex;
    break;
  case T_BGGEXP:
    ((BGGEXP *)target)->initialized = ((BGGEXP *)source)->initialized;
    ((BGGEXP *)target)->dataIndex[0] = ((BGGEXP *)source)->dataIndex[0];
    ((BGGEXP *)target)->dataIndex[1] = ((BGGEXP *)source)->dataIndex[1];
    break;
  case T_TAPERAPC:
    ((TAPERAPC *)target)->e1Index = ((TAPERAPC *)source)->e1Index;
    ((TAPERAPC *)target)->e2Index = ((TAPERAPC *)source)->e2Index;
    break;
  case T_TAPERAPE:
    ((TAPERAPE *)target)->e1Index = ((TAPERAPE *)source)->e1Index;
    ((TAPERAPE *)target)->e2Index = ((TAPERAPE *)source)->e2Index;
    break;
  case T_TAPERAPR:
    ((TAPERAPR *)target)->e1Index = ((TAPERAPR *)source)->e1Index;
    ((TAPERAPR *)target)->e2Index = ((TAPERAPR *)source)->e2Index;
    break;
  case T_PEPPOT:
    /* Need to modernize pepper-pot element to read data from SDDS file */
    pps = (PEPPOT *)source;
    ppt = (PEPPOT *)target;
    ppt->x = array_1d(sizeof(double), 0, ppt->n_holes - 1);
    ppt->y = array_1d(sizeof(double), 0, ppt->n_holes - 1);
    memcpy(ppt->x, pps->x, sizeof(double) * ppt->n_holes);
    memcpy(ppt->y, pps->y, sizeof(double) * ppt->n_holes);
    break;
  }
}

void resetElementToDefaults(char *p_elem, long type) {
  long i, n_params;
  PARAMETER *parameter;
  n_params = entity_description[type].n_params;
  parameter = entity_description[type].parameter;
  for (i = 0; i < n_params; i++) {
    switch (parameter[i].type) {
    case IS_DOUBLE:
      *((double *)(p_elem + parameter[i].offset)) = parameter[i].number;
      break;
    case IS_LONG:
      *((long *)(p_elem + parameter[i].offset)) = parameter[i].integer;
      break;
    case IS_INT64:
      *((int64_t *)(p_elem + parameter[i].offset)) = parameter[i].integer;
      break;
    case IS_SHORT:
      *((short *)(p_elem + parameter[i].offset)) = parameter[i].integer;
      break;
    case IS_STRING:
      if (parameter[i].string == NULL)
        *((char **)(p_elem + parameter[i].offset)) = NULL;
      else
        cp_str(((char **)p_elem + parameter[i].offset),
               parameter[i].string);
      break;
    }
  }
}

void setEdgeIndices(ELEMENT_LIST *e1) {
  BEND *bptr;
  KSBEND *ksbptr;
  CSBEND *csbptr;
  CSRCSBEND *csrbptr;
  NIBEND *nibptr;

  switch (e1->type) {
  case T_SBEN:
  case T_RBEN:
    bptr = (BEND *)e1->p_elem;
    bptr->e1Index = 0;
    bptr->e2Index = 1;
    break;
  case T_KSBEND:
    ksbptr = (KSBEND *)e1->p_elem;
    ksbptr->e1Index = 0;
    ksbptr->e2Index = 1;
    break;
  case T_NIBEND:
    nibptr = (NIBEND *)e1->p_elem;
    nibptr->e1Index = 0;
    nibptr->e2Index = 1;
    break;
  case T_CSBEND:
    csbptr = (CSBEND *)e1->p_elem;
    csbptr->e1Index = 0;
    csbptr->e2Index = 1;
    ((CSBEND *)e1->p_elem)->edgeFlip = 0;
    break;
  case T_CSRCSBEND:
    csrbptr = (CSRCSBEND *)e1->p_elem;
    csrbptr->e1Index = 0;
    csrbptr->e2Index = 1;
    break;
  case T_TAPERAPC:
    ((TAPERAPC *)e1->p_elem)->e1Index = 0;
    ((TAPERAPC *)e1->p_elem)->e2Index = 1;
    break;
  case T_TAPERAPE:
    ((TAPERAPE *)e1->p_elem)->e1Index = 0;
    ((TAPERAPE *)e1->p_elem)->e2Index = 1;
    break;
  case T_TAPERAPR:
    ((TAPERAPR *)e1->p_elem)->e1Index = 0;
    ((TAPERAPR *)e1->p_elem)->e2Index = 1;
    break;
  case T_CCBEND:
    ((CCBEND *)e1->p_elem)->edgeFlip = 0;
    break;
  case T_LGBEND:
    ((LGBEND *)e1->p_elem)->wasFlipped = 0;
    break;
  default:
    break;
  }
}

void swapEdgeIndices(ELEMENT_LIST *e1) {
  BEND *bptr;
  KSBEND *ksbptr;
  CSBEND *csbptr;
  CSRCSBEND *csrbptr;
  NIBEND *nibptr;
  CCBEND *ccbptr;
  LGBEND *lgbptr;
  TAPERAPC *taperapc;
  TAPERAPE *taperape;
  TAPERAPR *taperapr;

  switch (e1->type) {
  case T_SBEN:
  case T_RBEN:
    bptr = (BEND *)e1->p_elem;
    SWAP_LONG(bptr->e1Index, bptr->e2Index);
    break;
  case T_KSBEND:
    ksbptr = (KSBEND *)e1->p_elem;
    SWAP_LONG(ksbptr->e1Index, ksbptr->e2Index);
    break;
  case T_NIBEND:
    nibptr = (NIBEND *)e1->p_elem;
    SWAP_LONG(nibptr->e1Index, nibptr->e2Index);
    break;
  case T_CSBEND:
    csbptr = (CSBEND *)e1->p_elem;
    SWAP_LONG(csbptr->e1Index, csbptr->e2Index);
    csbptr->edgeFlip = !csbptr->edgeFlip;
    break;
  case T_CSRCSBEND:
    csrbptr = (CSRCSBEND *)e1->p_elem;
    SWAP_LONG(csrbptr->e1Index, csrbptr->e2Index);
    break;
  case T_CCBEND:
    ccbptr = (CCBEND *)e1->p_elem;
    ccbptr->edgeFlip = !ccbptr->edgeFlip;
    break;
  case T_LGBEND:
    lgbptr = (LGBEND *)e1->p_elem;
    flipLGBEND(lgbptr);
    break;
  case T_TAPERAPC:
    taperapc = (TAPERAPC *)e1->p_elem;
    SWAP_SHORT(taperapc->e1Index, taperapc->e2Index);
    break;
  case T_TAPERAPE:
    taperape = (TAPERAPE *)e1->p_elem;
    SWAP_SHORT(taperape->e1Index, taperape->e2Index);
    break;
  case T_TAPERAPR:
    taperapr = (TAPERAPR *)e1->p_elem;
    SWAP_SHORT(taperapr->e1Index, taperapr->e2Index);
    break;
  default:
    break;
  }
}

void readLGBendConfiguration(LGBEND *lgbend, ELEMENT_LIST *eptr) {
  SDDS_DATASET SDDSin;
  long readCode, i, index;
  short postdriftSeen, predriftSeen;
  uint64_t iRow, rows;
  double *parameterValue;
  char **parameterName;
  /* These names need to have the same order as the items in LGBEND_SEGMENT */
#define NLGBEND (8 + 2 * N_LGBEND_FRINGE_INT)
  static char *knownParameterName[NLGBEND] = {
    "LONGIT_L",
    "K1",
    "K2",
    "ANGLE",
    "ENTRY_X",
    "ENTRY_ANGLE",
    "EXIT_X",
    "EXIT_ANGLE",
    "FRINGE1K0",
    "FRINGE1I0",
    "FRINGE1K2",
    "FRINGE1I1",
    "FRINGE1K4",
    "FRINGE1K5",
    "FRINGE1K6",
    "FRINGE1K7",
    "FRINGE2K0",
    "FRINGE2I0",
    "FRINGE2K2",
    "FRINGE2I1",
    "FRINGE2K4",
    "FRINGE2K5",
    "FRINGE2K6",
    "FRINGE2K7",
  };
  short required[NLGBEND] = {
    2,
    0,
    0,
    1,
    1,
    1,
    2,
    2,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
  };
  short disallowed[NLGBEND] = {
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
  };
  short provided[NLGBEND];
  short offset[NLGBEND];
  LGBEND_SEGMENT segmentExample;
  LGBEND lgbendExample;

  static char *globalParameterName[8] = {
    "PREDRIFT", "POSTDRIFT", "ZEntry", "XEntry", "ZVertex", "XVertex", "ZExit", "XExit"};
  short offsetGlobal[8];

  offset[0] = 0;
  offset[1] = (short)(&(segmentExample.K1) - &(segmentExample.length));
  offset[2] = (short)(&(segmentExample.K2) - &(segmentExample.length));
  offset[3] = (short)(&(segmentExample.angle) - &(segmentExample.length));
  offset[4] = (short)(&(segmentExample.entryX) - &(segmentExample.length));
  offset[5] = (short)(&(segmentExample.entryAngle) - &(segmentExample.length));
  offset[6] = (short)(&(segmentExample.exitX) - &(segmentExample.length));
  offset[7] = (short)(&(segmentExample.exitAngle) - &(segmentExample.length));
  offset[8] = (short)(&(segmentExample.fringeInt1K0) - &(segmentExample.length));
  offset[9] = (short)(&(segmentExample.fringeInt1I0) - &(segmentExample.length));
  offset[10] = (short)(&(segmentExample.fringeInt1K2) - &(segmentExample.length));
  offset[11] = (short)(&(segmentExample.fringeInt1I1) - &(segmentExample.length));
  offset[12] = (short)(&(segmentExample.fringeInt1K4) - &(segmentExample.length));
  offset[13] = (short)(&(segmentExample.fringeInt1K5) - &(segmentExample.length));
  offset[14] = (short)(&(segmentExample.fringeInt1K6) - &(segmentExample.length));
  offset[15] = (short)(&(segmentExample.fringeInt1K7) - &(segmentExample.length));
  offset[16] = (short)(&(segmentExample.fringeInt2K0) - &(segmentExample.length));
  offset[17] = (short)(&(segmentExample.fringeInt2I0) - &(segmentExample.length));
  offset[18] = (short)(&(segmentExample.fringeInt2K2) - &(segmentExample.length));
  offset[19] = (short)(&(segmentExample.fringeInt2I1) - &(segmentExample.length));
  offset[20] = (short)(&(segmentExample.fringeInt2K4) - &(segmentExample.length));
  offset[21] = (short)(&(segmentExample.fringeInt2K5) - &(segmentExample.length));
  offset[22] = (short)(&(segmentExample.fringeInt2K6) - &(segmentExample.length));
  offset[23] = (short)(&(segmentExample.fringeInt2K7) - &(segmentExample.length));
  for (i = 0; i < NLGBEND; i++)
    offset[i] *= sizeof(double);

  offsetGlobal[0] = (short)(&(lgbendExample.predrift) - &(lgbendExample.length));
  offsetGlobal[1] = (short)(&(lgbendExample.postdrift) - &(lgbendExample.length));
  offsetGlobal[2] = (short)(&(lgbendExample.zEntry) - &(lgbendExample.length));
  offsetGlobal[3] = (short)(&(lgbendExample.xEntry) - &(lgbendExample.length));
  offsetGlobal[4] = (short)(&(lgbendExample.zVertex) - &(lgbendExample.length));
  offsetGlobal[5] = (short)(&(lgbendExample.xVertex) - &(lgbendExample.length));
  offsetGlobal[6] = (short)(&(lgbendExample.zExit) - &(lgbendExample.length));
  offsetGlobal[7] = (short)(&(lgbendExample.xExit) - &(lgbendExample.length));
  for (i = 0; i < 8; i++)
    offsetGlobal[i] *= sizeof(double);

  if (!SDDS_InitializeInputFromSearchPath(&SDDSin, lgbend->configurationFile)) {
    fprintf(stderr, "Error: unable to find or read LGBEND configuration file %s\n", lgbend->configurationFile);
    exitElegant(1);
  }
  if (SDDS_CheckColumn(&SDDSin, "ParameterName", NULL, SDDS_STRING, stdout) != SDDS_CHECK_OK ||
      SDDS_CheckColumn(&SDDSin, "ParameterValue", NULL, SDDS_ANY_FLOATING_TYPE, stdout) != SDDS_CHECK_OK) {
    fprintf(stderr, "Error: required columns with required types not present in LGBEND configuration file %s\n",
            lgbend->configurationFile);
    exitElegant(1);
  }

  lgbend->angle = lgbend->length = 0;
  postdriftSeen = 0;
  while ((readCode = SDDS_ReadPage(&SDDSin)) >= 1) {
    if (postdriftSeen) {
      fprintf(stderr, "Error: POSTDRIFT parameter seen for interior segment reading LGBEND configuration file %s\n",
              lgbend->configurationFile);
      exitElegant(1);
    }
    if (!(lgbend->segment = SDDS_Realloc(lgbend->segment, (lgbend->nSegments + 1) * sizeof(*(lgbend->segment)))) ||
        !(lgbend->fseOpt = SDDS_Realloc(lgbend->fseOpt, (lgbend->nSegments + 1) * sizeof(*(lgbend->fseOpt)))) ||
        !(lgbend->KnDelta = SDDS_Realloc(lgbend->KnDelta, (lgbend->nSegments + 1) * sizeof(*(lgbend->KnDelta))))) {
      fprintf(stderr, "Error: memory allocation failure reading LGBEND configuration file %s\n", lgbend->configurationFile);
      exitElegant(1);
    }
    memset(&(lgbend->segment[lgbend->nSegments]), 0, sizeof(*(lgbend->segment)));
    parameterName = NULL;
    parameterValue = NULL;
    if ((rows = SDDS_RowCount(&SDDSin)) <= 0 ||
        !(parameterName = SDDS_GetColumn(&SDDSin, "ParameterName")) ||
        !(parameterValue = SDDS_GetColumn(&SDDSin, "ParameterValue"))) {
      fprintf(stderr, "Error: problem getting data from LGBEND configuration file %s\n", lgbend->configurationFile);
      exitElegant(1);
    }
    memset(provided, 0, sizeof(provided[0]) * (NLGBEND));
    postdriftSeen = predriftSeen = 0;
    for (iRow = 0; iRow < rows; iRow++) {
      if ((index = match_string(parameterName[iRow], knownParameterName, NLGBEND, EXACT_MATCH)) < 0) {
        if ((index = match_string(parameterName[iRow], globalParameterName, 8, EXACT_MATCH)) < 0) {
          fprintf(stdout, "Error: unrecognized parameter name \"%s\" in LGBEND configuration file %s (page %ld, row %" PRIu64 ")\n",
                  parameterName[iRow], lgbend->configurationFile,
                  readCode, iRow);
          exitElegant(1);
        } else
          *((double *)(((char *)lgbend) + offsetGlobal[index])) = parameterValue[iRow];
      } else {
        provided[index] = 1;
        *((double *)(((char *)&(lgbend->segment[lgbend->nSegments])) + offset[index])) = parameterValue[iRow];
      }
    }
    free(parameterValue);
    SDDS_FreeStringArray(parameterName, rows);
    for (i = 0; i < NLGBEND; i++) {
      if ((readCode == 1 && required[i] && !provided[i]) ||
          (readCode > 1 && required[i] == 2 && !provided[i])) {
        fprintf(stdout, "Error: parameter %s is required but was not provided in page %ld of configuration file %s for LGBEND %s#%ld\n",
                knownParameterName[i], readCode, lgbend->configurationFile, eptr->name, eptr->occurence);
        exitElegant(1);
      }
      if (readCode > 1 && provided[i] && disallowed[i]) {
        fprintf(stdout, "Error: parameter %s is provided in page %ld of configuration file %s for LGBEND %s#%ld. Not meaningful.\n",
                knownParameterName[i], readCode, lgbend->configurationFile, eptr->name, eptr->occurence);
        exitElegant(1);
      }
    }
    lgbend->segment[lgbend->nSegments].has2 = 1;
    lgbend->segment[lgbend->nSegments].has1 = readCode == 1 ? 1 : 0;
    if (lgbend->nSegments > 0) {
      lgbend->segment[lgbend->nSegments].entryX = lgbend->segment[lgbend->nSegments - 1].exitX;
      lgbend->segment[lgbend->nSegments].entryAngle = lgbend->segment[lgbend->nSegments - 1].exitAngle;
    }
    lgbend->angle += lgbend->segment[lgbend->nSegments].angle;
    lgbend->fseOpt[lgbend->nSegments] = 0;
    lgbend->KnDelta[lgbend->nSegments] = 0;
    lgbend->nSegments++;
  }
  SDDS_Terminate(&SDDSin);
  lgbend->initialized = 1;
  lgbend->localApertureData = NULL;
  configureLGBendGeometry(lgbend);

  if (lgbend->verbose > 10) {
    printf("%ld segments found for LGBEND %s#%ld in file %s\n",
           lgbend->nSegments, eptr->name, eptr->occurence, lgbend->configurationFile);
    printf("total angle is %le, total length is %le\n", lgbend->angle, lgbend->length);
    printf("predrift = %le, postdrift = %le\n", lgbend->predrift, lgbend->postdrift);
    printf("nominal entry coordinates in magnet frame: Z=%le, X=%le\n", lgbend->zEntry, lgbend->xEntry);
    printf("nominal exit coordinates in magnet frame: Z=%le, X=%le\n", lgbend->zExit, lgbend->xExit);
    printf("vertex coordinates in magnet frame: Z=%le, X=%le\n", lgbend->zVertex, lgbend->xVertex);
    fflush(stdout);
    for (i = 0; i < lgbend->nSegments; i++) {
      printf("Segment %ld\nLONGIT_L = %le\nK1 = %le\nK2 = %le\n",
             i,
             lgbend->segment[i].length, lgbend->segment[i].K1, lgbend->segment[i].K2);
      printf("ANGLE = %le\nENTRY_X = %le\nENTRY_ANGLE = %le\nEXIT_X = %le\nEXIT_ANGLE = %le\n",
             lgbend->segment[i].angle,
             lgbend->segment[i].entryX, lgbend->segment[i].entryAngle,
             lgbend->segment[i].exitX, lgbend->segment[i].exitAngle);
      if (lgbend->segment[i].has1)
        printf("FRINGE1K0 = %le\nFRINGE1I0 = %le\nFRINGE1K2 = %le\nFRINGE1I1 = %le\nFRINGE1K4 = %le\nFRINGE1K5 = %le\nFRINGE1K6 = %le\nFRINGE1K7 = %le\n",
               lgbend->segment[i].fringeInt1K0,
               lgbend->segment[i].fringeInt1I0,
               lgbend->segment[i].fringeInt1K2,
               lgbend->segment[i].fringeInt1I1,
               lgbend->segment[i].fringeInt1K4,
               lgbend->segment[i].fringeInt1K5,
               lgbend->segment[i].fringeInt1K6,
               lgbend->segment[i].fringeInt1K7);
      if (lgbend->segment[i].has2)
        printf("FRINGE2K0 = %le\nFRINGE2I0 = %le\nFRINGE2K2 = %le\nFRINGE2I1 = %le\nFRINGE2K4 = %le\nFRINGE2K5 = %le\nFRINGE2K6 = %le\nFRINGE2K7 = %le\n",
               lgbend->segment[i].fringeInt2K0,
               lgbend->segment[i].fringeInt2I0,
               lgbend->segment[i].fringeInt2K2,
               lgbend->segment[i].fringeInt2I1,
               lgbend->segment[i].fringeInt2K4,
               lgbend->segment[i].fringeInt2K5,
               lgbend->segment[i].fringeInt2K6,
               lgbend->segment[i].fringeInt2K7);
    }
  }
}

void flipLGBEND(LGBEND *lgbend) {
  long i, j;
  LGBEND_SEGMENT *segment;

  SWAP_DOUBLE(lgbend->xEntry, lgbend->xExit);
  lgbend->zEntry *= -1;
  lgbend->zExit *= -1;
  SWAP_DOUBLE(lgbend->zEntry, lgbend->zExit);
  lgbend->zVertex *= -1;

  /* need to reverse the order of the segments and exchange edge integrals */
  /* - copy the data to temporary memory */
  segment = tmalloc(sizeof(*segment) * lgbend->nSegments);
  for (i = 0; i < lgbend->nSegments; i++)
    memcpy(segment + i, lgbend->segment + i, sizeof(*segment));
  /* - copy back to caller's memory, but in reversed order */
  for (i = 0, j = lgbend->nSegments - 1; i < lgbend->nSegments; i++, j--) {
    if (i != j)
      memcpy(lgbend->segment + i, segment + j, sizeof(*segment));
  }
  free(segment);
  /* - swap edge data and adjust signs as needed */
  for (i = 0; i < lgbend->nSegments; i++) {
    SWAP_DOUBLE(lgbend->segment[i].fringeInt1K0, lgbend->segment[i].fringeInt2K0);
    SWAP_DOUBLE(lgbend->segment[i].fringeInt1I0, lgbend->segment[i].fringeInt2I0);
    SWAP_DOUBLE(lgbend->segment[i].fringeInt1K2, lgbend->segment[i].fringeInt2K2);
    SWAP_DOUBLE(lgbend->segment[i].fringeInt1I1, lgbend->segment[i].fringeInt2I1);
    SWAP_DOUBLE(lgbend->segment[i].fringeInt1K4, lgbend->segment[i].fringeInt2K4);
    SWAP_DOUBLE(lgbend->segment[i].fringeInt1K5, lgbend->segment[i].fringeInt2K5);
    SWAP_DOUBLE(lgbend->segment[i].fringeInt1K6, lgbend->segment[i].fringeInt2K6);
    SWAP_DOUBLE(lgbend->segment[i].fringeInt1K7, lgbend->segment[i].fringeInt2K7);
    lgbend->segment[i].fringeInt1K0 *= -1;
    lgbend->segment[i].fringeInt1I1 *= -1;
    lgbend->segment[i].fringeInt1K5 *= -1;
    lgbend->segment[i].fringeInt2K0 *= -1;
    lgbend->segment[i].fringeInt2I1 *= -1;
    lgbend->segment[i].fringeInt2K5 *= -1;
    SWAP_SHORT(lgbend->segment[i].has1, lgbend->segment[i].has2);
    SWAP_DOUBLE(lgbend->segment[i].entryX, lgbend->segment[i].exitX);
    if (i == 0) {
      lgbend->segment[i].entryAngle = -lgbend->segment[i].exitAngle;
      lgbend->segment[i].exitAngle = lgbend->segment[i].entryAngle - lgbend->segment[i].angle;
    }
  }
  for (i = 1; i < lgbend->nSegments; i++) {
    lgbend->segment[i].entryAngle = lgbend->segment[i - 1].exitAngle;
    lgbend->segment[i].exitAngle = lgbend->segment[i].entryAngle - lgbend->segment[i].angle;
  }
  SWAP_DOUBLE(lgbend->predrift, lgbend->postdrift);
  lgbend->wasFlipped = !lgbend->wasFlipped;
  configureLGBendGeometry(lgbend); /* may be needed if flipLGBEND is called outside of lattice parsing */
}

void copyLGBend(LGBEND *target, LGBEND *source) {
  long i;
  memcpy(target, source, sizeof(*target));
  target->segment = calloc(target->nSegments, sizeof(*(target->segment)));
  target->fseOpt = calloc(target->nSegments, sizeof(*(target->fseOpt)));
  target->KnDelta = calloc(target->nSegments, sizeof(*(target->KnDelta)));
  for (i = 0; i < source->nSegments; i++) {
    memcpy(target->segment + i, source->segment + i, sizeof(*(target->segment)));
    target->fseOpt[i] = source->fseOpt[i];
    target->KnDelta[i] = source->KnDelta[i];
  }
  configureLGBendGeometry(target);
}

void configureLGBendGeometry(LGBEND *lgbend) {
  long i;
  double arcLength, rho, angle, entryAngle, length;
  arcLength = lgbend->predrift;
  for (i = 0; i < lgbend->nSegments; i++) {
    length = lgbend->segment[i].length;
    angle = lgbend->segment[i].angle;
    entryAngle = lgbend->segment[i].entryAngle;
    rho = length / (sin(entryAngle) + sin(angle - entryAngle));
    lgbend->segment[i].arcLengthStart = arcLength;
    arcLength += (lgbend->segment[i].arcLength = rho * angle);
    if (i == 0)
      lgbend->segment[i].zAccumulated = length;
    else
      lgbend->segment[i].zAccumulated = lgbend->segment[i - 1].zAccumulated + length;
  }
  lgbend->length = arcLength + lgbend->postdrift;
}

