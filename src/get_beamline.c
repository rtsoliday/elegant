/*************************************************************************\
 * Copyright (c) 2002 The University of Chicago, as Operator of Argonne
 * National Laboratory.
 * Copyright (c) 2002 The Regents of the University of California, as
 * Operator of Los Alamos National Laboratory.
 * This file is distributed subject to a Software License Agreement found
 * in the file LICENSE that is included with this distribution.
\*************************************************************************/

/* routine: get_beamline()
 * purpose: read a mad-format lattice and return a pointer to a linked
 *          list for the beamline specified.
 *
 *	    It is assumed that the MAD-style input file is in
 *          the usual units of meters, radians, etc.  The
 *          output has the same units.
 *
 * Michael Borland, 1989
 */
#include "mdb.h"
#include "track.h"
#include <ctype.h>
#include "match_string.h"

void show_elem(ELEMENT_LIST *eptr, long type);
void process_rename_request(char *s, char **name, long n_names);
long find_parameter_offset(char *param_name, long elem_type);
void resolveBranchPoints(LINE_LIST *lptr);
void copyEdgeIndices(char *target, long targetType, char *source, long sourceType);
long getAddStartFlag();
void setUpBMapXYZApContour(BMAPXYZ *bmxyz, ELEMENT_LIST *eptr0);

/* elem: root of linked-list of ELEM structures
 * This list contains the definitions of all elements as supplied in the
 * input file.  An important use of this list is to keep track of the lattice
 * that will be saved with save_lattice.
 */
static ELEMENT_LIST *elem = NULL;

/* line: root of linked-list of LINE structures
 * This list contains the definitions of all beamlines as supplied in the
 * input file.  Each beamline contains instances of the elements that it
 * contains.  I.e., it does not refer explicitly to the structures in
 * elem
 */
static LINE_LIST *line;

typedef struct input_object {
  void *ptr; /* points to an ELEMENT_LIST or LINE_LIST */
  long isLine;
  struct input_object *next;
} INPUT_OBJECT;
static INPUT_OBJECT inputObject, *lastInputObject = NULL;

/* All the quantities listed here must be double-precision values */
#define N_TRANSMUTE_ITEMS 32
static char *transmuteItems[N_TRANSMUTE_ITEMS] = {
  "L",
  "K1",
  "K2",
  "K3",
  "ANGLE",
  "DX",
  "DY",
  "DZ",
  "TILT",
  "BORE",
  "E1",
  "E2",
  "H1",
  "H2",
  "FINT",
  "ETILT",
  "B1",
  "B2",
  "FSE",
  "HGAP",
  "X_MAX",
  "Y_MAX",
  "I0P",
  "I1P",
  "I2P",
  "I3P",
  "LAMBDA2P",
  "I0M",
  "I1M",
  "I2M",
  "I3M",
  "LAMBDA2M",
};

void addToInputObjectList(void *ptr, long isLine) {
  if (lastInputObject == NULL)
    lastInputObject = &inputObject;
  else {
    lastInputObject->next = tmalloc(sizeof(INPUT_OBJECT));
    lastInputObject = lastInputObject->next;
  }
  lastInputObject->ptr = ptr;
  lastInputObject->isLine = isLine;
  lastInputObject->next = NULL;
}

void freeInputObjects() {
  INPUT_OBJECT *ptr;
  lastInputObject = inputObject.next;
  while (lastInputObject) {
    ptr = lastInputObject;
    lastInputObject = lastInputObject->next;
    free(ptr);
  }
  lastInputObject = NULL;
}

#define MAX_LINE_LENGTH 128 * 16384
#define MAX_FILE_NESTING 10
LINE_LIST *get_beamline(char *madfile, char *use_beamline, double p_central, long echo, long backtrack,
                        CHANGE_START_SPEC *changeStart, CHANGE_END_SPEC *changeEnd) {
  long type = 0, i;
  long iMad;
  static ELEMENT_LIST *eptr, *eptr1, *eptr_sc, *eptr_add, *eptr_del;
  static LINE_LIST *lptr;
  static long n_elems, n_lines;
  FILE *fp_mad[MAX_FILE_NESTING];
  char *s, *t = NULL, *ptr = NULL;
  ntuple *nBx, *nBy, *nBz;
  double ftable_length;
  htab *occurence_htab;
  long totalElements, uniqueElements, *occurenceCounter, *occurencePtr;

  log_entry("get_beamline");

  if (!(s = malloc(sizeof(*s) * MAX_LINE_LENGTH)) ||
      !(t = malloc(sizeof(*s) * MAX_LINE_LENGTH)))
    bombElegant("memory allocation failure (get_beamline)", NULL);

  if (madfile) {
    char *filename;

    if (echo) {
      printf("reading from file %s\n", madfile);
      fflush(stdout);
    }

    if (!(filename = findFileInSearchPath(madfile))) {
      fprintf(stderr, "Unable to find file %s\n", madfile);
      exitElegant(1);
    }
    fp_mad[0] = fopen_e(filename, "r", 0);
    free(filename);

    iMad = 0;

    elem = tmalloc(sizeof(*elem));
    line = tmalloc(sizeof(*line));

    elem->pred = elem->succ = NULL;
    elem->name = NULL;
    eptr = elem;
    n_elems = 0; /* number of physical elements in linked-list */
    line->pred = line->succ = NULL;
    line->name = NULL;
    lptr = line;
    n_lines = 0; /* number of line definitions in linked-list  */

    /* assemble linked-list of simple elements and a separate linked-list
       of fully expanded line definitions */
    while (iMad >= 0) {
      while (cfgets(s, MAX_LINE_LENGTH, fp_mad[iMad])) {
        if (echo) {
          printf("%s\n", s);
          fflush(stdout);
        }
        if (s[0] == '%') {
          /* rpn command */
          chop_nl(s);
          rpn(s + 1);
        }
        if (s[0] == '#' && strncmp(s, "#INCLUDE:", strlen("#INCLUDE:")) == 0) {
          char *filename;
          if (++iMad == MAX_FILE_NESTING)
            bombElegant("files nested too deeply", NULL);
          ptr = get_token(s + strlen("#INCLUDE:"));
          if (echo) {
            printf("reading file %s\n", ptr);
          }
          if (!(filename = findFileInSearchPath(ptr))) {
            fprintf(stderr, "Error: unable to find file %s\n", ptr);
            exitElegant(1);
          }
          fp_mad[iMad] = fopen_e(filename, "r", 0);
          free(filename);
          continue;
        }
        strcpy_ss(t, s);
        if ((type = tell_type(s, elem)) == T_NODEF) {
          if (!is_blank(s))
            printWarning("no recognized statement on lattice file line", t);
          continue;
        }
#ifdef DEBUG
        printf("type code = %ld\n", type);
        fflush(stdout);
#endif
        if (type == T_RENAME)
          process_rename_request(s, entity_name, N_TYPES);
        else if (type == T_TITLE) {
          if (!fgets(s, MAX_LINE_LENGTH, fp_mad[iMad])) {
            fprintf(stderr, "Error reading line from file\n");
            exit(EXIT_FAILURE);
          }
          compressString(s, " ");
          if (s[i = strlen(s) - 1] == ' ')
            s[i] = 0;
        } else if (type == T_USE || type == T_RETURN)
          break;
        else if (type == T_LINE) {
#ifdef DEBUG
          printf("current element list is:\n");
          fflush(stdout);
          print_elem_list(stdout, elem);
#endif
          fill_line(line, n_lines, elem, n_elems, s);
          addToInputObjectList((void *)lptr, 1);
          if (strchr(lptr->name, '#')) {
            printf("Error: the name %s is invalid for a beamline: # is a reserved character.\n", lptr->name);
            exitElegant(1);
          }
          if (check_duplic_elem(&elem, NULL, lptr->name, n_elems, NULL)) {
            printf("Error: line definition %s conflicts with element of same name\n", lptr->name);
            exitElegant(1);
          }
          check_duplic_line(line, lptr->name, n_lines + 1, 0);
#ifdef DEBUG
          printf("\n****** expanded line %s:\n", lptr->name);
          fflush(stdout);
          print_line(stdout, lptr);
#endif
          extend_line_list(&lptr);
          n_lines++;
        } else {
          if (type == T_ECOPY) {
#ifdef DEBUG
            printf("copying existing element\n");
            fflush(stdout);
#endif
            strcpy_ss(s, t);
            copy_named_element(eptr, s, elem);
            if (strchr(eptr->name, '#')) {
              printf("Error: the name %s is invalid for an element: # is a reserved character.\n", eptr->name);
              exitElegant(1);
            }
          } else {
            long newType;
            double length;
#ifdef DEBUG
            printf("creating new element\n");
            fflush(stdout);
#endif
            fill_elem(eptr, s, type, fp_mad[iMad]);
            addToInputObjectList((void *)eptr, 0);
            if (strchr(eptr->name, '#')) {
              printf("Error: the name %s is invalid for an element: # is a reserved character.\n", eptr->name);
              exitElegant(1);
            }
            length = 0;
            if ((newType = elementTransmutation(eptr->name, eptr->type)) != eptr->type &&
                newType >= 0) {
              int it, ip1, ip2;
              char *pNew;
              if (entity_description[eptr->type].flags & HAS_LENGTH) {
                length = ((DRIFT *)eptr->p_elem)->length;
                if (length && !(entity_description[newType].flags & HAS_LENGTH)) {
                  printf("Error: can't transmute %s %s into %s---would change length of beamline\n",
                         entity_name[eptr->type], eptr->name,
                         entity_name[newType]);
                  exitElegant(1);
                }
              }
              pNew = tmalloc(entity_description[newType].structure_size);
              resetElementToDefaults(pNew, newType);
              for (it = 0; it < N_TRANSMUTE_ITEMS; it++) {
                if ((ip1 = find_parameter_offset(transmuteItems[it], eptr->type)) >= 0 &&
                    (ip2 = find_parameter_offset(transmuteItems[it], newType)) >= 0) {
                  *((double *)(pNew + ip2)) = *((double *)(eptr->p_elem + ip1));
                }
              }
              if (IS_BEND(newType))
                copyEdgeIndices(pNew, newType, eptr->p_elem, eptr->type);
              free(eptr->p_elem);
              eptr->p_elem = pNew;
              eptr->type = newType;
              /*
                if (entity_description[newType].flags&HAS_LENGTH)
                ((DRIFT*)eptr->p_elem)->length = length;
              */
            }
            eptr->ignore = 0;
            if (ignoreElement(eptr->name, eptr->type, 0)) {
              /*
                printf("Ignoring %s in multi-particle tracking\n", eptr->name);
              */
              eptr->ignore = 1;
            } else if (ignoreElement(eptr->name, eptr->type, 1)) {
              /*
                printf("Ignoring %s completely\n", eptr->name);
              */
              eptr->ignore = 2;
            }
            if (check_duplic_line(line, eptr->name, n_lines + 1, 1)) {
              printf("Error: element %s conflicts with line with same name\n", eptr->name);
              exitElegant(1);
            }
            check_duplic_elem(&elem, &eptr, NULL, n_elems, NULL);
          }
#ifdef DEBUG
          print_elem(stdout, elem);
#endif
          extend_elem_list(&eptr);
          n_elems++;
        }
      }
      fclose(fp_mad[iMad]);
      iMad--;
    }
    if (n_elems == 0 || n_lines == 0) {
      printf("Error: insufficient (recognizable) data in file.\n");
      fflush(stdout);
      exitElegant(1);
    }
    if (echo) {
      printf("finished reading from files\n");
      report_stats(stdout, "statistics: ");
      fflush(stdout);
    }

    if (getSCMULTSpecCount()) {
      fill_elem(eptr, scMultName, T_SCMULT, NULL);
      eptr_sc = eptr;
      check_duplic_elem(&elem, &eptr, NULL, n_elems, NULL);
      extend_elem_list(&eptr);
      n_elems++;
    }

    /* since the lists were being extended before it was known that
       the was another object to put in them, must eliminate references
       to the most recently added nodes.
    */
    (eptr->pred)->succ = NULL;
    (lptr->pred)->succ = NULL;
    eptr = eptr->pred;
    lptr = lptr->pred;
  } else {
    s[0] = 0;
    type = T_NODEF;

    if (getAddElemFlag()) {
      ELEMENT_LIST *eptrExisting;
      /* go to the last elements in linked-list */
      if ((eptr = elem)) {
        while (eptr->succ)
          eptr = eptr->succ;
      }
      /* extend the list for accommodating new element */
      extend_elem_list(&eptr);
      /* add new element to linked-list */
      cp_str(&s, getElemDefinition());
      if ((type = tell_type(s, elem)) == T_NODEF) {
        if (!is_blank(s))
          printWarning("no recognized statement on line", s);
      }
      fill_elem(eptr, s, type, NULL);
      if (strchr(eptr->name, '#')) {
        printf("Error: the name %s is invalid for an element: # is a reserved character.\n", eptr->name);
        exitElegant(1);
      }
      if (check_duplic_line(line, eptr->name, n_lines + 1, 1)) {
        printf("Error: element %s conflicts with line with same name\n", eptr->name);
        exitElegant(1);
      }
      if (check_duplic_elem(&elem, NULL, eptr->name, n_elems, &eptrExisting)) {
        printWarning("insert_elements invoked using same new element identical to existing element. The existing definition is used.", NULL);
        if (eptr->pred)
          eptr->pred->succ = NULL;
        free_elements(eptr);
        eptr_add = tmalloc(sizeof(*eptr_add));
        copy_element(eptr_add, eptrExisting, 0, 0, 0, NULL);
      } else {
        /* This will actually insert the new element definition */
        eptr_add = eptr;
        check_duplic_elem(&elem, &eptr, NULL, n_elems, NULL);
        n_elems++;
      }
    }

    if (getDelElemFlag() == 1) {
      /* go to the last elements in linked-list */
      eptr = elem;
      while (eptr->succ)
        eptr = eptr->succ;
      /* extend the list for accommodating new element */
      extend_elem_list(&eptr);
      /* add new element to linked-list */
      s = getElemDefinition1();
      if ((type = tell_type(s, elem)) == T_NODEF) {
        if (!is_blank(s))
          printWarning("No recognized statement on line.", s);
      }
      fill_elem(eptr, s, type, NULL);
      if (strchr(eptr->name, '#')) {
        printf("Error: the name %s is invalid for an element: # is a reserved character.\n", eptr->name);
        exitElegant(1);
      }
      /* This is mis spelled. Should be eptr_replace. */
      eptr_del = eptr;
      if (check_duplic_line(line, eptr->name, n_lines + 1, 1)) {
        printf("Error: element %s conflicts with line with same name\n", eptr->name);
        exitElegant(1);
      }
      check_duplic_elem(&elem, &eptr, NULL, n_elems, NULL);
      n_elems++;
    }
  }

  if (type != T_USE && use_beamline == NULL) {
    char warningBuffer[1024];
    if (n_lines == 0)
      bombElegant("no beam-line defined\n", NULL);
    snprintf(warningBuffer, 1024, "Will use line %s", lptr->name);
    printWarning("No USE statement in lattice file", warningBuffer);
  } else {
    if (!use_beamline) {
      if ((ptr = get_token(s)) == NULL)
        bombElegant("no line named in USE statement", NULL);
    } else
      ptr = str_toupper(use_beamline);
    lptr = line;
    while (lptr) {
      if (strcmp(lptr->name, ptr) == 0)
        break;
      lptr = lptr->succ;
    }
    if (lptr == NULL) {
      printf("Error: no definition of beam-line %s\n", ptr);
      exitElegant(1);
    }
    if (!use_beamline) {
      free(ptr);
    }
  }

  /* these really aren't necessary, since I clear memory upon allocation */
  lptr->elem_recirc = lptr->elem_twiss = lptr->elast = lptr->ecat = NULL;
  lptr->twiss0 = NULL;
  lptr->matrix = NULL;

  if (getSCMULTSpecCount()) {
    long skip = 0;
    long nelem = 0;
    eptr = lptr->elem;
    while (eptr) {
      if (eptr->type == T_SCMULT) { /* the code allow user put scmult explicitly */
        eptr = eptr->succ;
        continue;
      }
      if (insertSCMULT(eptr->name, eptr->type, &skip)) {
        add_element(eptr, eptr_sc);
        eptr = eptr->succ; /* move pointer to new added element */
        nelem++;
      }
      if (eptr->succ == NULL && skip != 0) { /* add space charge element to the end of line */
        add_element(eptr, eptr_sc);
        eptr = eptr->succ; /* this is very impotant to get off the loop */
        nelem++;
      }
      eptr = eptr->succ;
    }
    lptr->n_elems += nelem;
  }

  if (getAddElemFlag()) {
    long skip = 0;
    long nelem = 0;
    eptr = lptr->elem;
    if (getAddStartFlag()) {
      ELEMENT_LIST *eptr2, *next;
      next = eptr->succ;
      eptr2 = tmalloc(sizeof(*eptr2));
      copy_element(eptr2, eptr, 0, 0, 0, NULL);
      copy_element(eptr, eptr_add, 0, 0, 0, NULL);
      eptr->pred = NULL;
      eptr->succ = eptr2;
      eptr2->pred = eptr;
      eptr2->succ = next;
      eptr = eptr2;
      nelem++;
    }
    while (eptr) {
      long code;
      /* The end position will have been set in a previous call to get_beamline(), prior to
         definition of insertions */
      if ((code = insertElem(eptr->name, eptr->type, &skip, eptr->occurence, eptr->end_pos))) {
        if (code == 1) {
          /* insert after */
          add_element(eptr, eptr_add);
          eptr = eptr->succ; /* move pointer to new added element */
        } else if (eptr->pred) {
          /* insert before */
          add_element(eptr->pred, eptr_add);
        }
        nelem++;
      }
      if (eptr->succ == NULL && getAddEndFlag()) { /* add element to the end of line if request */
        add_element(eptr, eptr_add);
        eptr = eptr->succ; /* this is very important to get off the loop */
        nelem++;
      }
      eptr = eptr->succ;
    }
    lptr->n_elems += nelem;
  }

  if (getDelElemFlag()) {
    long skip = 0;
    long flag;
    eptr = lptr->elem;
    while (eptr) {
      flag = replaceElem(eptr->name, eptr->type, &skip, eptr->occurence);
      if (flag == 1) {
        if (!eptr->pred)
          bombElegant("Can not replace the first element in beamline", NULL);
        eptr = replace_element(eptr, eptr_del);
        eptr = eptr->succ;
      } else if (flag == -1) {
        if (!eptr->pred)
          bombElegant("Can not remove the first element in beamline", NULL);
        eptr = rm_element(eptr);
        lptr->n_elems--;
      } else
        eptr = eptr->succ;
    }
  }

  /* go through and remove completely ignored elements */
  eptr = lptr->elem;
  while (eptr) {
    if (eptr->ignore == 2) {
      if (eptr->pred == NULL) {
        printWarning("Can't ignore the first element in the beamline", eptr->name);
        eptr->ignore = 0;
        eptr = eptr->succ;
      } else {
        eptr = rm_element(eptr);
        lptr->n_elems--;
      }
    } else {
      if (eptr->type == T_BMAPXYZ) {
        BMAPXYZ *bmxyz;
        bmxyz = (BMAPXYZ *)eptr->p_elem;
        if (bmxyz->apContourElement && strlen(bmxyz->apContourElement))
          setUpBMapXYZApContour(bmxyz, eptr);
      }
      eptr = eptr->succ;
    }
  }

  if (echo) {
    printf("Beginning organization of lattice input data.\n");
    report_stats(stdout, "statistics: ");
    fflush(stdout);
  }

  if (changeStart && changeStart->active) {
    /* change the starting element */
    ELEMENT_LIST *eptrMiddle, *eptrEnd, *eptrBegin, *eptrLastMatch;
    long countDown = changeStart->elementOccurence;
    eptrMiddle = eptrBegin = lptr->elem;
    eptrLastMatch = NULL;
    while (eptrMiddle) {
      if (strcmp(eptrMiddle->name, changeStart->elementName) == 0) {
        eptrLastMatch = eptrMiddle;
        if ((--countDown) == 0)
          break;
      }
      eptrMiddle = eptrMiddle->succ;
    }
    eptrMiddle = eptrLastMatch;
    if (!eptrMiddle)
      bombElegantVA("Couldn't find element %s#%ld for &change_start\n", changeStart->elementName, changeStart->elementOccurence);
    if (changeStart->deltaPosition > 0) {
      long countDown = changeStart->deltaPosition;
      while (countDown--) {
        if (eptrMiddle->succ)
          eptrMiddle = eptrMiddle->succ;
        else
          bombElegantVA("Couldn't offset start position as requested by %ld positions due to end of beamline",
                        changeStart->deltaPosition);
      }
    } else if (changeStart->deltaPosition < 0) {
      long countDown = -changeStart->deltaPosition;
      while (countDown--) {
        if (eptrMiddle->pred)
          eptrMiddle = eptrMiddle->pred;
        else
          bombElegantVA("Couldn't offset start position as requested by %ld positions due to end of beamline",
                        changeStart->deltaPosition);
      }
    }
    if (changeStart->ringMode) {
      /* If ring mode, put all elements before the new start onto the end of the beamline */
      /* find the last element */
      eptrEnd = eptrMiddle;
      while (eptrEnd->succ)
        eptrEnd = eptrEnd->succ;
      /* retarget linked lists */
      eptrEnd->succ = eptrBegin;
      eptrBegin->pred = eptrEnd;
      if (eptrMiddle->pred)
        eptrMiddle->pred->succ = NULL;
      eptrMiddle->pred = NULL;
    } else {
      if (eptrMiddle->pred) {
        eptrMiddle->pred->succ = NULL;
        eptrMiddle->pred = NULL;
        free_elements(eptrBegin);
      }
    }
    lptr->elem = eptrMiddle;
  }

  if (changeEnd && changeEnd->active) {
    /* change the final element */
    ELEMENT_LIST *eptrEnd, *eptrLastMatch;
    long countDown = changeEnd->elementOccurence;
    eptrEnd = lptr->elem;
    eptrLastMatch = NULL;
    while (eptrEnd) {
      if (strcmp(eptrEnd->name, changeEnd->elementName) == 0) {
        eptrLastMatch = eptrEnd;
        if ((--countDown) == 0)
          break;
      }
      eptrEnd = eptrEnd->succ;
    }
    eptrEnd = eptrLastMatch;
    if (!eptrEnd)
      bombElegantVA("Couldn't find element %s#%ld for &change_end\n", changeEnd->elementName, changeEnd->elementOccurence);
    if (changeEnd->deltaPosition > 0) {
      long countDown = changeEnd->deltaPosition;
      while (countDown--) {
        if (eptrEnd->succ)
          eptrEnd = eptrEnd->succ;
        else
          bombElegantVA("Couldn't offset end position as requested by %ld positions due to end of beamline",
                        changeEnd->deltaPosition);
      }
    } else if (changeEnd->deltaPosition < 0) {
      long countDown = -changeEnd->deltaPosition;
      while (countDown--) {
        if (eptrEnd->pred)
          eptrEnd = eptrEnd->pred;
        else
          bombElegantVA("Couldn't offset end position as requested by %ld positions due to end of beamline",
                        changeEnd->deltaPosition);
      }
    }
    if (eptrEnd->succ) {
      free_elements(eptrEnd->succ);
      eptrEnd->succ = NULL;
    }
  }

  /* go through and do some basic initialization for each element */
  eptr = lptr->elem;
  totalElements = 0;
  lptr->flags = 0;
  while (eptr) {
    eptr->occurence = 0;
    lptr->elast = eptr;
    eptr->Pref_input = eptr->Pref_output = p_central;
    if (eptr->type == T_SREFFECTS)
      lptr->flags |= BEAMLINE_TWISS_WANTED;
    totalElements++;
    if (entity_description[eptr->type].flags & MATRIX_TRACKING)
      lptr->flags |= BEAMLINE_MATRICES_NEEDED;
    if (eptr->type == T_CWIGGLER)
      determineCWigglerEndFlags((CWIGGLER *)eptr->p_elem, eptr);
    eptr = eptr->succ;
  }

  /* Set up hash table to allow computing occurrence numbers quickly */
  occurence_htab = hcreate(12);
  occurenceCounter = tmalloc(sizeof(*occurenceCounter) * totalElements);
  uniqueElements = 0;
  eptr = lptr->elem;
  while (eptr) {
    if (eptr->name != NULL) {
      if (hcount(occurence_htab) == 0 || hfind(occurence_htab, (unsigned char*)eptr->name, strlen(eptr->name)) == FALSE) {
        occurenceCounter[uniqueElements] = 0;
        hadd(occurence_htab, (unsigned char*)eptr->name, strlen(eptr->name), (void *)&occurenceCounter[uniqueElements++]);
        if (echo) {
          printf("Added %s to hash table\n", eptr->name);
          fflush(stdout);
        }
      }
    }
    eptr = eptr->succ;
  }
  lptr->n_elems = totalElements;
  if (echo) {
    printf("Created occurence hash table for %ld unique elements of %ld total elements\n",
           uniqueElements, totalElements);
    report_stats(stdout, "statistics: ");
    fflush(stdout);
  }

  /* assign occurence numbers */
  eptr = lptr->elem;
  while (eptr) {
#ifdef DEBUG
    printf("Setting occurence number for %s\n", eptr->name);
    fflush(stdout);
#endif
    if (hfind(occurence_htab, (unsigned char*)eptr->name, strlen(eptr->name)) == TRUE) {
      occurencePtr = hstuff(occurence_htab);
      eptr->occurence = (*occurencePtr += 1);
    } else
      bombElegant("hash table problem in get_beamline---seek professional help!", NULL);
    eptr = eptr->succ;
  }
  hdestroy(occurence_htab);
  free(occurenceCounter);

  if (backtrack) {
    ELEMENT_LIST ecopy;
    copy_line(&ecopy, lptr->elem, lptr->n_elems, 0, NULL, NULL);
    copy_line(lptr->elem, &ecopy, lptr->n_elems, 1, NULL, NULL);
    /* free_elements(&ecopy); */
    eptr = lptr->elem;
    modify_for_backtracking(eptr);
    lptr->flags |= BEAMLINE_BACKTRACKING;
  }

  if (echo) {
    printf("Step 1 done.\n");
    report_stats(stdout, "statistics: ");
    fflush(stdout);
  }

  eptr = lptr->elem;
  while (eptr) {
    if (eptr->occurence == 1 && eptr->type == T_FTABLE) {
      initializeFTable((FTABLE *)eptr->p_elem);
      nBx = ((FTABLE *)eptr->p_elem)->Bx;
      nBy = ((FTABLE *)eptr->p_elem)->By;
      nBz = ((FTABLE *)eptr->p_elem)->Bz;
      ftable_length = ((FTABLE *)eptr->p_elem)->length;

      eptr1 = eptr->succ;
      while (eptr1) {
        if (eptr1->type == T_FTABLE && eptr1->occurence > 1 && strcmp(eptr->name, eptr1->name) == 0) {
          ((FTABLE *)eptr1->p_elem)->initialized = 1;
          ((FTABLE *)eptr1->p_elem)->dataIsCopy = 1;
          ((FTABLE *)eptr1->p_elem)->length = ftable_length;
          ((FTABLE *)eptr1->p_elem)->Bx = nBx;
          ((FTABLE *)eptr1->p_elem)->By = nBy;
          ((FTABLE *)eptr1->p_elem)->Bz = nBz;
        }
        eptr1 = eptr1->succ;
      }
    }
    eptr = eptr->succ;
  }

  if (echo) {
    printf("Step 2 done.\n");
    report_stats(stdout, "statistics: ");
    fflush(stdout);
  }

  create_load_hash(lptr->elem);

  if (echo) {
    printf("Step 3 done.\n");
    report_stats(stdout, "statistics: ");
    fflush(stdout);
  }

  compute_end_positions(lptr);
  free(s);
  free(t);

  if (echo) {
    printf("Step 4 done.\n");
    report_stats(stdout, "statistics: ");
    fflush(stdout);
  }

  return (lptr);
}

void create_load_hash(ELEMENT_LIST *elem) {
  ELEMENT_LIST *eptr;
  char occurence_s[8], eptr_name[1024];

  /* create/recreate a hash table with the size of 2^12, it can grow automatically if necessary */
  if (load_hash)
    hdestroy(load_hash);
  load_hash = hcreate(12);

  eptr = elem;
  while (eptr) {
    /* use "eptr->name+eptr->occurence" as the key, and eptr's address as the value for hash table*/
    sprintf(occurence_s, "#%ld", eptr->occurence);
    strcpy(eptr_name, eptr->name);
    strcat(eptr_name, occurence_s);
    hadd(load_hash, (unsigned char*)eptr_name, strlen(eptr_name), (void *)eptr);
    eptr = eptr->succ;
  }
}

double compute_end_positions(LINE_LIST *lptr) {
  double z, l, theta, z_recirc;
  static ELEMENT_LIST *eptr;
  long i_elem, recircPresent;
  CSBEND *csbend;

  /* use length data to establish z coordinates at end of each element */
  /* also check for duplicate recirculation elements and set occurence numbers to 0 */
  eptr = lptr->elem;
  z = z_recirc = sStart;
  theta = 0;
  i_elem = 0;
  recircPresent = 0;
  do {
    eptr->beg_pos = z;
    if (!(entity_description[eptr->type].flags & HAS_LENGTH))
      l = 0;
    else
      l = (*((double *)eptr->p_elem));
    if (eptr->type == T_SBEN || eptr->type == T_RBEN)
      theta += ((BEND *)eptr->p_elem)->angle;
    else if (eptr->type == T_KSBEND)
      theta += ((KSBEND *)eptr->p_elem)->angle;
    else if (eptr->type == T_NIBEND)
      theta += ((NIBEND *)eptr->p_elem)->angle;
    else if (eptr->type == T_NISEPT)
      theta += ((NISEPT *)eptr->p_elem)->angle;
    else if (eptr->type == T_CSBEND) {
      csbend = (CSBEND *)eptr->p_elem;
      if (csbend->edge_order > 1 &&
          (csbend->edge_effects[0] == 1 || csbend->edge_effects[1] == 1 ||
           csbend->edge_effects[0] == 2 || csbend->edge_effects[1] == 2)) {
        printWarning("CSBEND edge settings may be non-optimal if symplecticity is important",
                     "EDGE1_EFFECTS and EDGE2_EFFECTS should be 3 or 4 for symplectic tracking with nonlinear effects.");
      }
      theta += ((CSBEND *)eptr->p_elem)->angle;
    } else if (eptr->type == T_EMATRIX)
      theta += ((EMATRIX *)eptr->p_elem)->angle;
    else if (eptr->type == T_FTABLE && ((FTABLE *)eptr->p_elem)->angle) {
      theta += ((FTABLE *)eptr->p_elem)->angle;
      l = ((FTABLE *)eptr->p_elem)->l0 / 2. / sin(((FTABLE *)eptr->p_elem)->angle / 2.) * ((FTABLE *)eptr->p_elem)->angle;
      ((FTABLE *)eptr->p_elem)->arcLength = l;
    } else if (eptr->type == T_RECIRC) {
      if (recircPresent)
        bombElegant("multiple recirculation (RECIRC) elements in beamline--this doesn't make sense", NULL);
      lptr->elem_recirc = eptr;
      lptr->i_recirc = i_elem;
      z_recirc = z;
      recircPresent = 1;
    }
    if (((!(lptr->flags & BEAMLINE_BACKTRACKING) && l < 0) || (lptr->flags & BEAMLINE_BACKTRACKING && l > 0))) {
      char warningText[1024];
      snprintf(warningText, 1024,
               "%s has length %le.", eptr->name, lptr->flags & BEAMLINE_BACKTRACKING ? -l : l);
      printWarning("Element has negative length.", warningText);
    }
    eptr->end_pos = z + l;
    eptr->end_theta = theta;
    z = eptr->end_pos;
    i_elem++;
  } while ((eptr = eptr->succ));

  resolveBranchPoints(lptr);

  /* if ((lptr->flags&BEAMLINE_BACKTRACKING) && z<0) { */
  if ((lptr->flags & BEAMLINE_BACKTRACKING)) {
    printf("Offsetting z coordinates by %le for backtracking\n", -z);
    eptr = lptr->elem;
    do {
      eptr->beg_pos -= z;
      eptr->end_pos -= z;
    } while ((eptr = eptr->succ));
  }

  /* Compute revolution length, respecting BRANCH elements */
  eptr = lptr->elem;
  z = z_recirc = 0;
  theta = 0;
  i_elem = 0;
  recircPresent = 0;
  do {
    if (eptr->type == T_BRANCH) {
      BRANCH *branch;
      branch = (BRANCH *)eptr->p_elem;
      branch->z = z;
      if (branch->counter && !branch->defaultToElse) {
        eptr = branch->beptr1;
        continue;
      } else {
        eptr = branch->beptr2;
        continue;
      }
    }
    if (!(entity_description[eptr->type].flags & HAS_LENGTH))
      l = 0;
    else
      l = (*((double *)eptr->p_elem));
    if (eptr->type == T_FTABLE && ((FTABLE *)eptr->p_elem)->angle)
      l = ((FTABLE *)eptr->p_elem)->arcLength;
    else if (eptr->type == T_RECIRC) {
      if (recircPresent)
        bombElegant("multiple recirculation (RECIRC) elements in beamline--this doesn't make sense", NULL);
      lptr->elem_recirc = eptr;
      lptr->i_recirc = i_elem;
      z_recirc = z;
      recircPresent = 1;
    }
    z += l;
    i_elem++;
  } while ((eptr = eptr->succ));

  lptr->revolution_length = z - z_recirc;

  return lptr->revolution_length;
}

void show_elem(ELEMENT_LIST *eptr, long type) {
  long j;
  char *ptr;
  PARAMETER *parameter;

  log_entry("show_elem");

  parameter = entity_description[type].parameter;
  printf("%s %s at z=%em:\n",
         entity_name[type], eptr->name, eptr->end_pos);
  fflush(stdout);
  for (j = 0; j < entity_description[type].n_params; j++) {
    switch (parameter[j].type) {
    case IS_DOUBLE:
      printf("    %s = %.16e with offset %ld\n",
             parameter[j].name,
             *(double *)(eptr->p_elem + parameter[j].offset),
             parameter[j].offset);
      fflush(stdout);
      break;
    case IS_LONG:
      printf("    %s = %ld with offset %ld\n",
             parameter[j].name,
             *(long *)(eptr->p_elem + parameter[j].offset),
             parameter[j].offset);
      fflush(stdout);
      break;
    case IS_SHORT:
      printf("    %s = %hd with offset %ld\n",
             parameter[j].name,
             *(short *)(eptr->p_elem + parameter[j].offset),
             parameter[j].offset);
      fflush(stdout);
      break;
    case IS_STRING:
      if ((ptr = *(char **)(eptr->p_elem + parameter[j].offset))) {
        printf("    %s = %s\n", parameter[j].name, ptr);
        fflush(stdout);
      } else {
        printf("    %s = <NULL>\n", parameter[j].name);
        fflush(stdout);
      }
      break;
    }
  }
  log_exit("show_elem");
}

void free_elements(ELEMENT_LIST *elemlist) {
  ELEMENT_LIST *eptr;

  log_entry("free_elements");

  if (elemlist) {
    eptr = elemlist;
  } else {
    eptr = elem;
    elem = NULL;
#ifdef DEBUG
    printf("freeing elements in main list\n");
    fflush(stdout);
#endif
  }
  while (eptr) {
#ifdef DEBUG
    printf("freeing memory for element %s of type %s\n",
           eptr->name ? eptr->name : "NULL",
           (eptr->type >= 0 && eptr->type < N_TYPES) ? entity_name[eptr->type] : "NULL");
    fflush(stdout);
#endif
    if (eptr->type == T_MATR) {
      eptr = eptr->succ;
      continue;
    }
    if (eptr->type == T_WATCH) {
      WATCH *wptr;
      if ((wptr = (WATCH *)eptr->p_elem)) {
        if (wptr->initialized && !SDDS_Terminate(wptr->SDDS_table)) {
          SDDS_SetError("Problem terminate watch-point SDDS file (free_elements)");
          SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
        }
      }
    } else if (eptr->type == T_FTABLE) {
      FTABLE *ftable;
      ftable = (FTABLE *)eptr->p_elem;
      if (!ftable->dataIsCopy) {
        free_hbookn(ftable->Bx);
        free_hbookn(ftable->By);
        free_hbookn(ftable->Bz);
      }
    }
    tfree(eptr->p_elem);
    eptr->p_elem = NULL;
    tfree(eptr->p_elem0);
    eptr->p_elem0 = NULL;
    tfree(eptr->name);
    eptr->name = NULL;
    tfree(eptr->definition_text);
    eptr->definition_text = NULL;
    if (entity_description[eptr->type].flags & HAS_MATRIX && eptr->matrix) {
      free_matrices(eptr->matrix);
      free(eptr->matrix);
      eptr->matrix = NULL;
    }
    if (eptr->accumMatrix) {
      free_matrices(eptr->accumMatrix);
      free(eptr->accumMatrix);
      eptr->accumMatrix = NULL;
    }
    if (eptr->succ) {
      eptr = eptr->succ;
      free(eptr->pred);
      eptr->pred = NULL;
    } else {
      free(eptr);
      break;
    }
  }
  log_exit("free_elements");
}

void free_beamlines(LINE_LIST *beamline) {
  LINE_LIST *lptr;

  log_entry("free_beamlines");

  if (beamline) {
    lptr = beamline;
  } else {
    lptr = line;
    line = NULL;
#ifdef DEBUG
    printf("freeing main beamline list\n");
    fflush(stdout);
#endif
  }
  while (lptr) {
#ifdef DEBUG
    printf("*************************\nfreeing memory for beamline %s with %ld elements\n",
           lptr->name ? lptr->name : "NULL", lptr->n_elems);
    fflush(stdout);
#endif
    if (lptr->definition) {
      tfree(lptr->definition);
      lptr->definition = NULL;
    }
    if (lptr->name) {
      tfree(lptr->name);
      lptr->name = NULL;
    }
    if (lptr->n_elems) {
      free_elements(lptr->elem);
      /* should free name etc. for lptr->elem also */
      lptr->elem = NULL;
      lptr->n_elems = 0;
      lptr->flags = 0;
    }
    if (lptr->succ) {
      lptr = lptr->succ;
      tfree(lptr->pred);
      lptr->pred = NULL;
    } else {
      tfree(lptr);
      break;
    }
  }
  if (load_hash) {
    hdestroy(load_hash); /* destroy hash table */
    load_hash = NULL;
  }
  log_exit("free_beamlines");
}

void delete_matrix_data(LINE_LIST *beamline) {
  LINE_LIST *lptr;
  ELEMENT_LIST *eptr;

  log_entry("delete_matrix_data");

  if (beamline) {
    lptr = beamline;
  } else {
    lptr = line;
  }
  while (lptr) {
    if (lptr->n_elems) {
      eptr = lptr->elem;
      while (eptr) {
        if (entity_description[eptr->type].flags & HAS_MATRIX && eptr->matrix) {
          free_matrices(eptr->matrix);
          tfree(eptr->matrix);
          eptr->matrix = NULL;
        }
        if (eptr->accumMatrix) {
          free_matrices(eptr->accumMatrix);
          tfree(eptr->accumMatrix);
          eptr->accumMatrix = NULL;
        }
        eptr = eptr->succ;
      }
      lptr->n_elems = 0;
      lptr->flags = 0;
    }
    lptr = lptr->succ;
  }
  log_exit("delete_matrix_data");
}

/* routine: do_save_lattice()
 * purpose: save the element and beamline definitions to a file
 *
 * Michael Borland, 1991
 */
#include "save_lattice.h"

void do_save_lattice(NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline) {
  FILE *fp;
  ELEMENT_LIST *eptr;
  LINE_LIST *lptr;
  long j;
  double dvalue;
  long lvalue;
  short svalue;
  char *ptr;
  PARAMETER *parameter;
  char s[16384], t[1024], name[1024];
  INPUT_OBJECT *object;

  log_entry("do_save_lattice");

  /* process the namelist text */
  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&save_lattice, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (echoNamelists)
    print_namelist(stdout, &save_lattice);

  /* check for valid data */
  if (filename == NULL)
    bombElegant("no filename given to save lattice to", NULL);

  if (output_seq < 0 || output_seq > 2)
    bombElegant("valid values of output_seq are 0, 1, and 2", NULL);

#if USE_MPI
  if (myid == 0) {
#endif

    if (str_in(filename, "%s"))
      filename = compose_filename(filename, run->rootname);
    fp = fopen_e(filename, "w", FOPEN_INFORM_OF_OPEN);

    if (!output_seq) {
      object = &inputObject;
      do {
        if (!object->isLine) {
          eptr = (ELEMENT_LIST *)(object->ptr);
          parameter = entity_description[eptr->type].parameter;
          if (strpbrk(eptr->name, ":.,/-_+abcdefghijklmnopqrstuvwyxz "))
            sprintf(name, "\"%s\"", eptr->name);
          else
            strcpy_ss(name, eptr->name);
          sprintf(s, "%s: %s,", name, entity_name[eptr->type]);
          if (eptr->group && strlen(eptr->group)) {
            sprintf(t, "GROUP=\"%s\",", eptr->group);
            strcat(s, t);
          }
          for (j = 0; j < entity_description[eptr->type].n_params; j++) {
            if (parameter[j].flags & PARAM_IS_ALIAS)
              continue;
            switch (parameter[j].type) {
            case IS_DOUBLE:
              dvalue = *(double *)(eptr->p_elem + parameter[j].offset);
              /*
                if ((parameter[j].flags&PARAM_DIVISION_RELATED) &&
                eptr->divisions>1) {
                fprintf(stderr, "Multiplying %s by %ld\n",
                parameter[j].name, eptr->divisions);
                dvalue *= eptr->divisions;
                }
              */
              if (!suppress_defaults || dvalue != parameter[j].number) {
                /* value is not the default, so add to output */
                sprintf(t, "%s=%.16g", parameter[j].name, dvalue);
                strcat(s, t);
                if (j != entity_description[eptr->type].n_params - 1)
                  strcat(s, ",");
              }
              break;
            case IS_LONG:
              lvalue = *(long *)(eptr->p_elem + parameter[j].offset);
              if (strcmp(parameter[j].name, "PHASE_REFERENCE") == 0 && lvalue > LONG_MAX / 2)
                lvalue = 0;
              if (!suppress_defaults || lvalue != parameter[j].integer) {
                /* value is not the default, so add to output */
                sprintf(t, "%s=%ld", parameter[j].name, lvalue);
                strcat(s, t);
                if (j != entity_description[eptr->type].n_params - 1)
                  strcat(s, ",");
              }
              break;
            case IS_SHORT:
              svalue = *(short *)(eptr->p_elem + parameter[j].offset);
              if (!suppress_defaults || svalue != parameter[j].integer) {
                /* value is not the default, so add to output */
                sprintf(t, "%s=%hd", parameter[j].name, svalue);
                strcat(s, t);
                if (j != entity_description[eptr->type].n_params - 1)
                  strcat(s, ",");
              }
              break;
            case IS_STRING:
              ptr = *(char **)(eptr->p_elem + parameter[j].offset);
              if (ptr && strlen(ptr) &&
                  (!suppress_defaults || !parameter[j].string || strcmp(ptr, parameter[j].string) != 0)) {
                sprintf(t, "%s=\"%s\"", parameter[j].name, ptr);
                strcat(s, t);
                if (j != entity_description[eptr->type].n_params - 1)
                  strcat(s, ",");
              }
              break;
            }
          }
          if (s[j = strlen(s) - 1] == ',')
            s[j] = 0;
          print_with_continuation(fp, s, 139);
          eptr = eptr->succ;
        } else {
          lptr = (LINE_LIST *)(object->ptr);
          print_with_continuation(fp, lptr->definition, 139);
        }
      } while ((object = object->next));

    } else {
      /* first write element definition */
      long type;
      long nline = 1, nelem = 0;
      for (type = 1; type < N_TYPES; type++) {
        eptr = beamline->elem;
        while (eptr) {
          if ((eptr->occurence == 1) && (eptr->type == type)) {
            parameter = entity_description[eptr->type].parameter;
            if (strpbrk(eptr->name, ":.,/-_+abcdefghijklmnopqrstuvwyxz "))
              sprintf(name, "\"%s\"", eptr->name);
            else
              strcpy_ss(name, eptr->name);
            sprintf(s, "%s: %s,", name, entity_name[eptr->type]);
            if (eptr->group && strlen(eptr->group)) {
              sprintf(t, "GROUP=\"%s\",", eptr->group);
              strcat(s, t);
            }
            for (j = 0; j < entity_description[eptr->type].n_params; j++) {
              if (parameter[j].flags & PARAM_IS_ALIAS)
                continue;
              switch (parameter[j].type) {
              case IS_DOUBLE:
                dvalue = *(double *)(eptr->p_elem + parameter[j].offset);
                /*
                  if ((parameter[j].flags&PARAM_DIVISION_RELATED) &&
                  eptr->divisions>1) {
                  fprintf(stderr, "Multiplying %s by %ld\n",
                  parameter[j].name, eptr->divisions);
                  dvalue *= eptr->divisions;
                  }
                */
                if (!suppress_defaults || dvalue != parameter[j].number) {
                  /* value is not the default, so add to output */
                  sprintf(t, "%s=%.16g", parameter[j].name, dvalue);
                  strcat(s, t);
                  if (j != entity_description[eptr->type].n_params - 1)
                    strcat(s, ",");
                }
                break;
              case IS_LONG:
                lvalue = *(long *)(eptr->p_elem + parameter[j].offset);
                if (strcmp(parameter[j].name, "PHASE_REFERENCE") == 0 && lvalue > LONG_MAX / 2)
                  lvalue = 0;
                if (!suppress_defaults || lvalue != parameter[j].integer) {
                  /* value is not the default, so add to output */
                  sprintf(t, "%s=%ld", parameter[j].name, lvalue);
                  strcat(s, t);
                  if (j != entity_description[eptr->type].n_params - 1)
                    strcat(s, ",");
                }
                break;
              case IS_SHORT:
                svalue = *(short *)(eptr->p_elem + parameter[j].offset);
                if (!suppress_defaults || svalue != parameter[j].integer) {
                  /* value is not the default, so add to output */
                  sprintf(t, "%s=%d", parameter[j].name, svalue);
                  strcat(s, t);
                  if (j != entity_description[eptr->type].n_params - 1)
                    strcat(s, ",");
                }
                break;
              case IS_STRING:
                ptr = *(char **)(eptr->p_elem + parameter[j].offset);
                if (ptr &&
                    (!suppress_defaults || !parameter[j].string || strcmp(ptr, parameter[j].string) != 0)) {
                  sprintf(t, "%s=\"%s\"", parameter[j].name, ptr);
                  strcat(s, t);
                  if (j != entity_description[eptr->type].n_params - 1)
                    strcat(s, ",");
                }
                break;
              }
            }
            if (s[j = strlen(s) - 1] == ',')
              s[j] = 0;
            print_with_continuation(fp, s, 139);
          }
          eptr = eptr->succ;
        }
      }
      /* Write beamline sequence now
       * if output_seq=1, each line has 40 elements limitation
       * otherwise, a single beamline is created
       */
      eptr = beamline->elem;
      sprintf(s, "L%04ld: LINE = (", nline);
      while (eptr) {
        nelem++;
        if (strpbrk(eptr->name, ":.,/-_+abcdefghijklmnopqrstuvwyxz "))
          sprintf(name, "\"%s\"", eptr->name);
        else
          strcpy_ss(name, eptr->name);
        strcat(s, name);
        strcat(s, ",");

        eptr = eptr->succ;
        if (output_seq == 1 && nelem == 40) {
          nline++;
          nelem = 0;
          if (s[j = strlen(s) - 1] == ',')
            s[j] = 0;
          strcat(s, ")");
          print_with_continuation(fp, s, 139);
          sprintf(s, "L%04ld: LINE = (", nline);
        }
      }
      if (nelem) {
        nline++;
        if (s[j = strlen(s) - 1] == ',')
          s[j] = 0;
        strcat(s, ")");
        print_with_continuation(fp, s, 139);
      }

      sprintf(s, "%s: LINE = (", beamline->name);
      for (j = 1; j < nline; j++) {
        sprintf(t, " L%04ld,", j);
        strcat(s, t);
      }
      if (s[j = strlen(s) - 1] == ',')
        s[j] = 0;
      strcat(s, ")");
      print_with_continuation(fp, s, 139);
    }

    if (beamline && beamline->name)
      fprintf(fp, "USE,\"%s\"\n", beamline->name);

    fprintf(fp, "RETURN\n");
    fclose(fp);

#if USE_MPI
  }
#endif

  log_exit("do_save_lattice");
}

void print_with_continuation(FILE *fp, char *s, long endcol) {
  char c, *ptr;
  long l, isContin;

  isContin = 0;
  while ((l = strlen(s))) {
    if (isContin)
      fputc(' ', fp);
    if (l > endcol) {
      ptr = s + endcol - 2;
      while (ptr != s && *ptr != ',')
        ptr--;
      if (ptr == s)
        c = *(ptr = s + endcol - 1);
      else {
        ptr++;
        c = *ptr;
      }
      *ptr = 0;
      fputs(s, fp);
      fputs("&\n", fp);
      isContin = 1;
      s = ptr;
      *ptr = c;
    } else {
      fputs(s, fp);
      fputc('\n', fp);
      log_exit("print_with_continuation");
      return;
    }
  }
}

/* Change defined parameter values in the reference list elem.
 * This routine allows changing a number of parameters for a number of differently-named elements
 */
void change_defined_parameter_values(char **elem_name, long *param_number, long *type,
                                     double *value, long n_elems) {
  ELEMENT_LIST *eptr;
  char *p_elem;
  long i_elem, elem_type, data_type, param;
  double dValue;

  log_entry("change_defined_parameter_values");

  for (i_elem = 0; i_elem < n_elems; i_elem++) {
    eptr = NULL;
    if ((elem_type = type[i_elem]) == T_FREEVAR)
      continue;
    param = param_number[i_elem];
    data_type = entity_description[elem_type].parameter[param].type;
    while (find_element(elem_name[i_elem], &eptr, elem)) {
      p_elem = eptr->p_elem;
      switch (data_type) {
      case IS_DOUBLE:
        dValue = value[i_elem];
        if ((entity_description[elem_type].parameter[param].flags & PARAM_DIVISION_RELATED) &&
            eptr->divisions)
          dValue *= eptr->divisions;
        *((double *)(p_elem + entity_description[elem_type].parameter[param].offset)) = dValue;
#if DEBUG
        printf("   changing parameter %s of %s #%ld to %21.15e\n",
               entity_description[elem_type].parameter[param].name,
               eptr->name, eptr->occurence,
               *((double *)(p_elem + entity_description[elem_type].parameter[param].offset)));
        fflush(stdout);
#endif
        break;
      case IS_LONG:
        *((long *)(p_elem + entity_description[elem_type].parameter[param].offset)) =
          nearestInteger(value[i_elem]);
#if DEBUG
        printf("   changing parameter %s of %s #%ld to %ld\n",
               entity_description[elem_type].parameter[param].name,
               eptr->name, eptr->occurence,
               *((long *)(p_elem + entity_description[elem_type].parameter[param].offset)));
        fflush(stdout);
#endif
        break;
      case IS_SHORT:
        *((short *)(p_elem + entity_description[elem_type].parameter[param].offset)) =
          nearestInteger(value[i_elem]);
#if DEBUG
        printf("   changing parameter %s of %s #%ld to %hd\n",
               entity_description[elem_type].parameter[param].name,
               eptr->name, eptr->occurence,
               *((short *)(p_elem + entity_description[elem_type].parameter[param].offset)));
        fflush(stdout);
#endif
        break;
      case IS_STRING:
      default:
        bombElegant("unknown/invalid variable quantity", NULL);
        exitElegant(1);
      }
    }
  }
  log_exit("change_defined_parameter_values");
}

/* Change defined parameter values in the reference list elem.
 * This routine allows changing a single parameter for a single element name.
 */
void change_defined_parameter_divopt(char *elem_name, long param, long elem_type,
                                     double value, char *valueString, unsigned long mode,
                                     long checkDiv) {
  ELEMENT_LIST *eptr;
  char *p_elem;
  long data_type;
  short wildcards = 0;

  log_entry("change_defined_parameter");

  data_type = entity_description[elem_type].parameter[param].type;
  if (mode & LOAD_FLAG_IGNORE)
    return;

  wildcards = has_wildcards(elem_name);
  eptr = NULL;
  while ((!wildcards && find_element(elem_name, &eptr, elem)) ||
         (wildcards && (eptr = wfind_element(elem_name, &eptr, elem)))) {
    p_elem = eptr->p_elem;
    switch (data_type) {
    case IS_DOUBLE:
      if (valueString) {
        if (!sscanf(valueString, "%lf", &value)) {
          printf("Error (change_defined_parameter): unable to scan double from \"%s\"\n", valueString);
          fflush(stdout);
          exitElegant(1);
        }
      }
      if (checkDiv && eptr->divisions > 1 &&
          (entity_description[elem_type].parameter[param].flags & PARAM_DIVISION_RELATED))
        value /= eptr->divisions;
      if (mode & LOAD_FLAG_VERBOSE)
        printf("Changing definition (mode %s) %s.%s from %21.15e to ",
               (mode & LOAD_FLAG_ABSOLUTE) ? "absolute" : ((mode & LOAD_FLAG_DIFFERENTIAL) ? "differential" : (mode & LOAD_FLAG_FRACTIONAL) ? "fractional"
                                                           : "unknown"),
               elem_name, entity_description[elem_type].parameter[param].name,
               *((double *)(p_elem + entity_description[elem_type].parameter[param].offset)));
      fflush(stdout);
      if (mode & LOAD_FLAG_ABSOLUTE) {
        *((double *)(p_elem + entity_description[elem_type].parameter[param].offset)) = value;
      } else if (mode & LOAD_FLAG_DIFFERENTIAL) {
        *((double *)(p_elem + entity_description[elem_type].parameter[param].offset)) += value;
      } else if (mode & LOAD_FLAG_FRACTIONAL) {
        *((double *)(p_elem + entity_description[elem_type].parameter[param].offset)) *= 1 + value;
      }
      if (mode & LOAD_FLAG_VERBOSE)
        printf("%21.15e\n",
               *((double *)(p_elem + entity_description[elem_type].parameter[param].offset)));
      fflush(stdout);
      break;
    case IS_LONG:
    case IS_SHORT:
      if (valueString) {
        if (!sscanf(valueString, "%lf", &value)) {
          printf("Error (change_defined_parameter): unable to scan double from \"%s\"\n", valueString);
          fflush(stdout);
          exitElegant(1);
        }
      }
      if (mode & LOAD_FLAG_VERBOSE) {
        printf("Changing definition (mode %s) %s.%s ",
               (mode & LOAD_FLAG_ABSOLUTE) ? "absolute" : ((mode & LOAD_FLAG_DIFFERENTIAL) ? "differential" : (mode & LOAD_FLAG_FRACTIONAL) ? "fractional"
                                                           : "unknown"),
               elem_name, entity_description[elem_type].parameter[param].name);
        if (data_type == IS_LONG)
          printf("from %ld to ",
                 *((long *)(p_elem + entity_description[elem_type].parameter[param].offset)));
        else
          printf("from %hd to ",
                 *((short *)(p_elem + entity_description[elem_type].parameter[param].offset)));
        fflush(stdout);
      }
      if (data_type == IS_LONG) {
        if (mode & LOAD_FLAG_ABSOLUTE)
          *((long *)(p_elem + entity_description[elem_type].parameter[param].offset)) =
            nearestInteger(value);
        else if (mode & LOAD_FLAG_DIFFERENTIAL)
          *((long *)(p_elem + entity_description[elem_type].parameter[param].offset)) +=
            nearestInteger(value);
        else if (mode & LOAD_FLAG_FRACTIONAL)
          *((long *)(p_elem + entity_description[elem_type].parameter[param].offset)) *= 1 + value;
      } else {
        if (mode & LOAD_FLAG_ABSOLUTE)
          *((short *)(p_elem + entity_description[elem_type].parameter[param].offset)) =
            nearestInteger(value);
        else if (mode & LOAD_FLAG_DIFFERENTIAL)
          *((short *)(p_elem + entity_description[elem_type].parameter[param].offset)) +=
            nearestInteger(value);
        else if (mode & LOAD_FLAG_FRACTIONAL)
          *((short *)(p_elem + entity_description[elem_type].parameter[param].offset)) *= 1 + value;
      }
      if (mode & LOAD_FLAG_VERBOSE) {
        if (data_type == IS_LONG)
          printf("%ld\n",
                 *((long *)(p_elem + entity_description[elem_type].parameter[param].offset)));
        else
          printf("%hd\n",
                 *((short *)(p_elem + entity_description[elem_type].parameter[param].offset)));
        fflush(stdout);
      }
      break;
    case IS_STRING:
      if (!valueString)
        return;
      if (mode & LOAD_FLAG_VERBOSE)
        printf("Changing definition %s.%s from %s to %s\n",
               elem_name, entity_description[elem_type].parameter[param].name,
               *((char **)(p_elem + entity_description[elem_type].parameter[param].offset)),
               valueString);
      fflush(stdout);
      if (strlen(valueString)) {
        if (!SDDS_CopyString(((char **)(p_elem + entity_description[elem_type].parameter[param].offset)),
                             valueString)) {
          printf("Error (change_defined_parameter): unable to copy string parameter value\n");
          fflush(stdout);
          exitElegant(1);
        }
      } else
        *((char **)(p_elem + entity_description[elem_type].parameter[param].offset)) = NULL;
      break;
    default:
      bombElegant("unknown/invalid variable quantity", NULL);
      exitElegant(1);
    }
  }
  log_exit("change_defined_parameter");
}

void change_defined_parameter(char *elem_name, long param, long elem_type,
                              double value, char *valueString, unsigned long mode) {
  change_defined_parameter_divopt(elem_name, param, elem_type, value, valueString, mode, 0);
}

/* Change used parameter values in the reference list elem.
 * This routine allows changing a single parameter for a single element name.
 */
void change_used_parameter_divopt(LINE_LIST *beamline, char *elem_name, long param, long elem_type,
                                  double value, char *valueString, unsigned long mode,
                                  long checkDiv) {
  ELEMENT_LIST *eptr;
  char *p_elem;
  long data_type;
  short wildcards = 0;

  log_entry("change_used_parameter_divopt");

  eptr = NULL;
  data_type = entity_description[elem_type].parameter[param].type;
  if (mode & LOAD_FLAG_IGNORE)
    return;
  wildcards = has_wildcards(elem_name);
  while ((!wildcards && find_element(elem_name, &eptr, beamline->elem)) ||
         (wildcards && (eptr = wfind_element(elem_name, &eptr, beamline->elem)))) {
    p_elem = eptr->p_elem;
    switch (data_type) {
    case IS_DOUBLE:
      if (valueString) {
        if (!sscanf(valueString, "%lf", &value)) {
          printf("Error (change_used_parameter): unable to scan double from \"%s\"\n", valueString);
          fflush(stdout);
          exitElegant(1);
        }
      }
      if (checkDiv && eptr->divisions > 1 &&
          (entity_description[elem_type].parameter[param].flags & PARAM_DIVISION_RELATED))
        value /= eptr->divisions;
      if (mode & LOAD_FLAG_VERBOSE)
        printf("Changing value (mode %s) %s.%s from %21.15e to ",
               (mode & LOAD_FLAG_ABSOLUTE) ? "absolute" : ((mode & LOAD_FLAG_DIFFERENTIAL) ? "differential" : (mode & LOAD_FLAG_FRACTIONAL) ? "fractional"
                                                           : "unknown"),
               elem_name, entity_description[elem_type].parameter[param].name,
               *((double *)(p_elem + entity_description[elem_type].parameter[param].offset)));
      fflush(stdout);
      if (mode & LOAD_FLAG_ABSOLUTE) {
        *((double *)(p_elem + entity_description[elem_type].parameter[param].offset)) = value;
      } else if (mode & LOAD_FLAG_DIFFERENTIAL) {
        *((double *)(p_elem + entity_description[elem_type].parameter[param].offset)) += value;
      } else if (mode & LOAD_FLAG_FRACTIONAL) {
        *((double *)(p_elem + entity_description[elem_type].parameter[param].offset)) *= 1 + value;
      }
      if (mode & LOAD_FLAG_VERBOSE)
        printf("%21.15e\n",
               *((double *)(p_elem + entity_description[elem_type].parameter[param].offset)));
      fflush(stdout);
      break;
    case IS_LONG:
    case IS_SHORT:
      if (valueString) {
        if (!sscanf(valueString, "%lf", &value)) {
          printf("Error (change_used_parameter): unable to scan double from \"%s\"\n", valueString);
          fflush(stdout);
          exitElegant(1);
        }
      }
      if (mode & LOAD_FLAG_VERBOSE) {
        printf("Changing value (mode %s) %s.%s ",
               (mode & LOAD_FLAG_ABSOLUTE) ? "absolute" : ((mode & LOAD_FLAG_DIFFERENTIAL) ? "differential" : (mode & LOAD_FLAG_FRACTIONAL) ? "fractional"
                                                           : "unknown"),
               elem_name, entity_description[elem_type].parameter[param].name);
        if (data_type == IS_LONG)
          printf("from %ld to ",
                 *((long *)(p_elem + entity_description[elem_type].parameter[param].offset)));
        else
          printf("from %hd to ",
                 *((short *)(p_elem + entity_description[elem_type].parameter[param].offset)));
        fflush(stdout);
      }
      if (data_type == IS_LONG) {
        if (mode & LOAD_FLAG_ABSOLUTE)
          *((long *)(p_elem + entity_description[elem_type].parameter[param].offset)) =
            nearestInteger(value);
        else if (mode & LOAD_FLAG_DIFFERENTIAL)
          *((long *)(p_elem + entity_description[elem_type].parameter[param].offset)) +=
            nearestInteger(value);
        else if (mode & LOAD_FLAG_FRACTIONAL)
          *((long *)(p_elem + entity_description[elem_type].parameter[param].offset)) *= 1 + value;
      } else {
        if (mode & LOAD_FLAG_ABSOLUTE)
          *((short *)(p_elem + entity_description[elem_type].parameter[param].offset)) =
            nearestInteger(value);
        else if (mode & LOAD_FLAG_DIFFERENTIAL)
          *((short *)(p_elem + entity_description[elem_type].parameter[param].offset)) +=
            nearestInteger(value);
        else if (mode & LOAD_FLAG_FRACTIONAL)
          *((short *)(p_elem + entity_description[elem_type].parameter[param].offset)) *= 1 + value;
      }
      if (mode & LOAD_FLAG_VERBOSE) {
        if (data_type == IS_LONG)
          printf("%ld\n",
                 *((long *)(p_elem + entity_description[elem_type].parameter[param].offset)));
        else
          printf("%hd\n",
                 *((short *)(p_elem + entity_description[elem_type].parameter[param].offset)));
        fflush(stdout);
      }
      break;
    case IS_STRING:
      if (!valueString)
        return;
      if (mode & LOAD_FLAG_VERBOSE)
        printf("Changing value %s.%s from %s to %s\n",
               elem_name, entity_description[elem_type].parameter[param].name,
               *((char **)(p_elem + entity_description[elem_type].parameter[param].offset)),
               valueString);
      fflush(stdout);
      if (strlen(valueString)) {
        if (!SDDS_CopyString(((char **)(p_elem + entity_description[elem_type].parameter[param].offset)),
                             valueString)) {
          printf("Error (change_used_parameter): unable to copy string parameter value\n");
          fflush(stdout);
          exitElegant(1);
        }
      } else
        *((char **)(p_elem + entity_description[elem_type].parameter[param].offset)) = NULL;
      break;
    default:
      bombElegant("unknown/invalid variable quantity", NULL);
      exitElegant(1);
    }
  }
  log_exit("change_used_parameter_divopt");
}

void change_used_parameter(LINE_LIST *beamline, char *elem_name, long param, long elem_type,
                           double value, char *valueString, unsigned long mode) {
  change_used_parameter_divopt(beamline, elem_name, param, elem_type, value, valueString, mode, 0);
}

void process_rename_request(char *s, char **name, long n_names) {
  long i;
  char *old, *new, *ptr;
  char warningText[1024];

  log_entry("process_rename_request");
  if (!(ptr = strchr(s, '=')))
    bombElegant("invalid syntax for RENAME", NULL);
  *ptr++ = 0;
  old = s;
  str_toupper(trim_spaces(old = s));
  while (*old == ',' || *old == ' ')
    old++;
  str_toupper(trim_spaces(new = ptr));
  if (match_string(new, name, n_names, EXACT_MATCH) >= 0) {
    printf("Error: can't rename to name %s--already exists\n", new);
    exitElegant(1);
  }
  if ((i = match_string(old, name, n_names, EXACT_MATCH)) < 0) {
    printf("Error: can't rename %s to %s--%s not recognized\n", old, new, old);
    exitElegant(1);
  }
  snprintf(warningText, 1024, "%s is now known as %s", old, new);
  printWarning("Element renaming can result in unexpected behavior.", warningText);
  fflush(stdout);
  cp_str(name + i, new);
  log_exit("process_rename_request");
}

long nearestInteger(double value) {
  if (value < 0)
    return -1 * ((long)(-value + 0.5));
  return (long)(value + 0.5);
}

/* add element "elem1" after "elem0" */
void add_element(ELEMENT_LIST *elem0, ELEMENT_LIST *elem1) {
  ELEMENT_LIST *eptr;
  /* printf("Adding %s after %s\n", elem1->name, elem0->name); */
  eptr = tmalloc(sizeof(*eptr));
  copy_element(eptr, elem1, 0, 0, 0, NULL);

  eptr->pred = elem0;
  eptr->succ = elem0->succ;
  if (elem0->succ)
    (elem0->succ)->pred = eptr;
  elem0->succ = eptr;
}

/* remove element "elem" from the list */
ELEMENT_LIST *rm_element(ELEMENT_LIST *elem) {
  ELEMENT_LIST *eptr0, *pred, *succ;
  eptr0 = elem;
  pred = elem->pred;
  succ = elem->succ;
  if (pred)
    pred->succ = succ;
  if (succ)
    succ->pred = pred;
  free(eptr0);
  return pred ? pred : succ;
}

/* replace element "elem0" with "elem1" */
ELEMENT_LIST *replace_element(ELEMENT_LIST *elem0, ELEMENT_LIST *elem1) {
  ELEMENT_LIST *eptr;
  eptr = tmalloc(sizeof(*eptr));
  copy_element(eptr, elem1, 0, 0, 0, NULL);
  printf("Replacing occurrence %ld of %s with %s\n",
         elem0->occurence, elem0->name, elem1->name);

  (elem0->pred)->succ = eptr;
  if (elem0->succ)
    (elem0->succ)->pred = eptr;
  eptr->pred = elem0->pred;
  eptr->succ = elem0->succ;
  return (eptr);
}
/* This is called at beginning to avoid multiple calls for same element at different locations */
void initializeFTable(FTABLE *ftable) {
  long i;

  if (ftable->simpleInput) {
    readSimpleFtable(ftable);
  } else {
    ftable->Bx = readbookn(ftable->inputFile, 1);
    ftable->By = readbookn(ftable->inputFile, 2);
    ftable->Bz = readbookn(ftable->inputFile, 3);
  }

  if ((ftable->Bx->nD != 3) || (ftable->By->nD != 3) || (ftable->Bz->nD != 3))
    bombElegantVA("ND must be 3 for field table %s.", ftable->inputFile);
  ftable->length = ftable->Bz->xmax[2] - ftable->Bz->xmin[2];
  if (fabs((ftable->l0 + ftable->l1 + ftable->l2) - ftable->length) > 1e-12)
    bombElegantVA("L+L1+L2 != field length in file %s. Must agree to within 1e-12.", ftable->inputFile);

  if (1) {
    double Bmin, Bmax;
    find_min_max(&Bmin, &Bmax, ftable->Bx->value, ftable->Bx->length);
    printf("Bx: [%le, %le]\n", Bmin, Bmax);
    find_min_max(&Bmin, &Bmax, ftable->By->value, ftable->By->length);
    printf("By: [%le, %le]\n", Bmin, Bmax);
    find_min_max(&Bmin, &Bmax, ftable->Bz->value, ftable->Bz->length);
    printf("Bz: [%le, %le]\n", Bmin, Bmax);
  }

  for (i = 0; i < ftable->Bz->nD; i++) {
    ftable->Bx->xmin[i] -= ftable->Bx->dx[i] / 2;
    ftable->By->xmin[i] -= ftable->By->dx[i] / 2;
    ftable->Bz->xmin[i] -= ftable->Bz->dx[i] / 2;
    ftable->Bx->xmax[i] += ftable->Bx->dx[i] / 2;
    ftable->By->xmax[i] += ftable->By->dx[i] / 2;
    ftable->Bz->xmax[i] += ftable->Bz->dx[i] / 2;
  }
  ftable->initialized = 1;
  ftable->dataIsCopy = 0;
  return;
}

long find_parameter_offset(char *param_name, long elem_type) {
  long param;
  if ((param = confirm_parameter(param_name, elem_type)) < 0)
    return (-1);
  return (entity_description[elem_type].parameter[param].offset);
}

void resolveBranchPoints(LINE_LIST *lptr) {
  static ELEMENT_LIST *eptr;
  BRANCH *branch;
  ELEMENT_LIST *eptr2;

  eptr = lptr->elem;
  do {
    if (eptr->type == T_BRANCH) {
      branch = (BRANCH *)eptr->p_elem;
      if (branch->branchTo) {
        /* Resolve branch for non-positive counter */
        eptr2 = eptr->succ;
        while (eptr2) {
          if (strcmp(eptr2->name, branch->branchTo) == 0)
            break;
          eptr2 = eptr2->succ;
        }
        if (!eptr2)
          bombElegantVA("Failed to find downstream target %s for BRANCH %s", branch->branchTo, eptr->name);
        if (eptr2->type != T_MARK)
          bombElegantVA("Branch target %s is not a MARK element", branch->branchTo);
        printf("Found branch point %s for BRANCH %s\n", branch->branchTo, eptr->name);
        branch->beptr1 = eptr2;
      } else
        branch->beptr1 = eptr->succ;
      if (branch->elseTo) {
        /* Resolve branch for positive counter */
        eptr2 = eptr->succ;
        while (eptr2) {
          if (strcmp(eptr2->name, branch->elseTo) == 0)
            break;
          eptr2 = eptr2->succ;
        }
        if (!eptr2)
          bombElegantVA("Failed to find downstream target %s for BRANCH %s", branch->elseTo, eptr->name);
        if (eptr2->type != T_MARK)
          bombElegantVA("Branch target %s is not a MARK element", branch->elseTo);
        printf("Found branch point %s for BRANCH %s\n", branch->elseTo, eptr->name);
        branch->beptr2 = eptr2;
      } else
        branch->beptr2 = eptr->succ;
    }
  } while ((eptr = eptr->succ));
}

void copyEdgeIndices(char *target, long targetType, char *source, long sourceType) {
  long e1Index, e2Index;

  e1Index = 0;
  e2Index = 1;

  switch (sourceType) {
  case T_SBEN:
  case T_RBEN:
    e1Index = ((BEND *)source)->e1Index;
    e2Index = ((BEND *)source)->e2Index;
    break;
  case T_KSBEND:
    e1Index = ((KSBEND *)source)->e1Index;
    e2Index = ((KSBEND *)source)->e2Index;
    break;
  case T_NIBEND:
    e1Index = ((NIBEND *)source)->e1Index;
    e2Index = ((NIBEND *)source)->e2Index;
    break;
  case T_CSBEND:
    e1Index = ((CSBEND *)source)->e1Index;
    e2Index = ((CSBEND *)source)->e2Index;
    break;
  case T_CSRCSBEND:
    e1Index = ((CSRCSBEND *)source)->e1Index;
    e2Index = ((CSRCSBEND *)source)->e2Index;
    break;
  }

  switch (targetType) {
  case T_SBEN:
  case T_RBEN:
    ((BEND *)target)->e1Index = e1Index;
    ((BEND *)target)->e2Index = e2Index;
    break;
  case T_KSBEND:
    ((KSBEND *)target)->e1Index = e1Index;
    ((KSBEND *)target)->e2Index = e2Index;
    break;
  case T_NIBEND:
    ((NIBEND *)target)->e1Index = e1Index;
    ((NIBEND *)target)->e2Index = e2Index;
    break;
  case T_CSBEND:
    ((CSBEND *)target)->e1Index = e1Index;
    ((CSBEND *)target)->e2Index = e2Index;
    break;
  case T_CSRCSBEND:
    ((CSRCSBEND *)target)->e1Index = e1Index;
    ((CSRCSBEND *)target)->e2Index = e2Index;
    break;
  }
}

void print_beamlines(FILE *fp) {
  LINE_LIST *lptr;
  INPUT_OBJECT *object;

  object = &inputObject;
  do {
    if (object->isLine) {
      lptr = (LINE_LIST *)(object->ptr);
      print_with_continuation(fp, lptr->definition, 139);
    }
  } while ((object = object->next));
}

void modify_for_backtracking(ELEMENT_LIST *eptr) {
  ENERGY *en;
  while (eptr) {
    if (!(entity_description[eptr->type].flags & BACKTRACK))
      bombElegantVA("Error: no backtracking method for element %s (type %s)",
                    eptr->name, entity_name[eptr->type]);
    if (entity_description[eptr->type].flags & HAS_LENGTH)
      ((DRIFT *)(eptr->p_elem))->length *= -1;
    switch (eptr->type) {
    case T_SBEN:
      ((BEND *)(eptr->p_elem))->angle *= -1;
      ((BEND *)(eptr->p_elem))->e[0] *= -1;
      ((BEND *)(eptr->p_elem))->e[1] *= -1;
      break;
    case T_CSBEND:
      ((CSBEND *)(eptr->p_elem))->angle *= -1;
      ((CSBEND *)(eptr->p_elem))->e[0] *= -1;
      ((CSBEND *)(eptr->p_elem))->e[1] *= -1;
      break;
    case T_CSRCSBEND:
      if (!((CSRCSBEND *)(eptr->p_elem))->steadyState)
        bombElegant("CSRCSBEND in backtrack mode requires STEADY_STATE=1", NULL);
      ((CSRCSBEND *)(eptr->p_elem))->angle *= -1;
      ((CSRCSBEND *)(eptr->p_elem))->e[0] *= -1;
      ((CSRCSBEND *)(eptr->p_elem))->e[1] *= -1;
      ((CSRCSBEND *)(eptr->p_elem))->backtrack = 1;
      break;
    case T_CCBEND:
      ((CCBEND *)(eptr->p_elem))->angle *= -1;
      ((CCBEND *)(eptr->p_elem))->yaw *= -1;
      break;
    case T_EHCOR:
      ((EHCOR *)(eptr->p_elem))->kick *= -1;
      break;
    case T_EVCOR:
      ((EVCOR *)(eptr->p_elem))->kick *= -1;
      break;
    case T_EHVCOR:
      ((EHVCOR *)(eptr->p_elem))->xkick *= -1;
      ((EHVCOR *)(eptr->p_elem))->ykick *= -1;
      break;
    case T_RFCA:
      ((RFCA *)(eptr->p_elem))->volt *= -1;
      ((RFCA *)(eptr->p_elem))->backtrack = 1;
      break;
    case T_RFCW:
      ((RFCW *)(eptr->p_elem))->volt *= -1;
      ((RFCW *)(eptr->p_elem))->backtrack = 1;
      if (((RFCW *)(eptr->p_elem))->wakesAtEnd)
        bombElegant("RFCW in backtrack mode requires WAKES_AT_END=0", NULL);
      break;
    case T_WAKE:
      ((WAKE *)(eptr->p_elem))->factor *= -1;
      break;
    case T_TRWAKE:
      ((TRWAKE *)(eptr->p_elem))->factor *= -1;
      break;
    case T_MALIGN:
      ((MALIGN *)eptr->p_elem)->dx *= -1;
      ((MALIGN *)eptr->p_elem)->dxp *= -1;
      ((MALIGN *)eptr->p_elem)->dy *= -1;
      ((MALIGN *)eptr->p_elem)->dyp *= -1;
      break;
    case T_ROTATE:
      ((ROTATE *)eptr->p_elem)->tilt *= -1;
      break;
    case T_HCOR:
      ((HCOR *)eptr->p_elem)->kick *= -1;
      break;
    case T_VCOR:
      ((VCOR *)eptr->p_elem)->kick *= -1;
      break;
    case T_HVCOR:
      ((HVCOR *)eptr->p_elem)->xkick *= -1;
      ((HVCOR *)eptr->p_elem)->ykick *= -1;
      break;
    case T_UKICKMAP:
      ((UKICKMAP *)eptr->p_elem)->flipSign = 1;
      break;
    case T_KICKMAP:
      ((KICKMAP *)eptr->p_elem)->flipSign = 1;
      break;
    case T_LSCDRIFT:
      ((LSCDRIFT *)eptr->p_elem)->backtrack = 1;
      break;
    case T_ENERGY:
      en = (ENERGY *)eptr->p_elem;
      if (en->match_beamline || en->match_particles)
        bombElegant("Can't use ENERGY element with MATCH_BEAMLINE=1 or MATCH_PARTICLES=1 in backtracking mode", NULL);
      break;
    default:
      break;
    }
    eptr = eptr->succ;
    if (eptr && eptr->name == NULL) {
      eptr->pred->succ = NULL;
      free(eptr);
      eptr = NULL;
    }
  }
}

void determineCWigglerEndFlags(CWIGGLER *cwig, ELEMENT_LIST *eptr0) {
  ELEMENT_LIST *eptr;

  eptr = eptr0->pred;
  cwig->endFlag[0] = 1;
  while (eptr) {
    if (eptr->type == T_CWIGGLER) {
      cwig->endFlag[0] = 0;
      break;
    }
    if (eptr->type != T_MARK && eptr->type != T_WATCH) {
      cwig->endFlag[0] = 1;
      break;
    } else
      eptr = eptr->pred;
  }

  eptr = eptr0->succ;
  cwig->endFlag[1] = 1;
  while (eptr) {
    if (eptr->type == T_CWIGGLER) {
      cwig->endFlag[1] = 0;
      break;
    }
    if (eptr->type != T_MARK && eptr->type != T_WATCH) {
      cwig->endFlag[1] = 1;
      break;
    } else
      eptr = eptr->succ;
  }
}

void setUpBMapXYZApContour(BMAPXYZ *bmxyz, ELEMENT_LIST *eptr0) {
  ELEMENT_LIST *eptr;
  eptr = elem;
  while (eptr) {
    if (eptr->type == T_APCONTOUR && strcmp(eptr->name, bmxyz->apContourElement) == 0) {
      copy_p_elem((char *)&bmxyz->apContour, eptr->p_elem, T_APCONTOUR);
      initializeApContour(&bmxyz->apContour);
      return;
    }
    eptr = eptr->succ;
  }
  bombElegantVA("Unable to find APCONTOUR element %s referred to by BMXYZ element %s\n",
                bmxyz->apContourElement, eptr0->name);
}

void initializeApContour(APCONTOUR *apcontour) {
  if (!apcontour->initialized && !apcontour->cancel) {
    SDDS_DATASET SDDSin;
    long readCode;
    SDDSin.parallel_io = 0;
    if (apcontour->x)
      free(apcontour->x);
    if (apcontour->y)
      free(apcontour->y);
    apcontour->x = apcontour->y = NULL;
    apcontour->nPoints = NULL;
    apcontour->nContours = 0;
    if (!apcontour->filename || !strlen(apcontour->filename))
      bombElegantVA("Error: No filename given for APCONTOUR\n", apcontour->filename);
    if (!apcontour->xColumn || !strlen(apcontour->xColumn))
      bombElegantVA("Error: No XCOLUMN given for APCONTOUR\n", apcontour->xColumn);
    if (!apcontour->yColumn || !strlen(apcontour->yColumn))
      bombElegantVA("Error: No YCOLUMN given for APCONTOUR\n", apcontour->yColumn);
    if (!SDDS_InitializeInputFromSearchPath(&SDDSin, apcontour->filename))
      bombElegantVA("Error: APCONTOUR file %s is unreadable\n", apcontour->filename);
    while ((readCode = SDDS_ReadPage(&SDDSin)) > 0) {
      if (readCode == 1) {
        if (SDDS_CheckColumn(&SDDSin, apcontour->xColumn, "m", SDDS_ANY_FLOATING_TYPE, stdout) != SDDS_CHECK_OK)
          bombElegantVA("Error: problem with x column (%s) for APCONTOUR file %s---check existence, units, and type\n",
                        apcontour->xColumn, apcontour->filename);
        if (SDDS_CheckColumn(&SDDSin, apcontour->yColumn, "m", SDDS_ANY_FLOATING_TYPE, stdout) != SDDS_CHECK_OK)
          bombElegantVA("Error: problem with y column (%s) for APCONTOUR file %s---check existence, units, and type\n",
                        apcontour->yColumn, apcontour->filename);
        if ((apcontour->hasLogic = SDDS_GetParameterIndex(&SDDSin, "Logic") >= 0)) {
          if (SDDS_CheckParameter(&SDDSin, "Logic", NULL, SDDS_STRING, stdout) != SDDS_CHECK_OK)
            bombElegantVA("Error: parameter \"Logic\" in APCONTOUR file %s must have string type\n", apcontour->filename);
        }
      }
      apcontour->x = SDDS_Realloc(apcontour->x, sizeof(*(apcontour->x)) * (apcontour->nContours + 1));
      apcontour->y = SDDS_Realloc(apcontour->y, sizeof(*(apcontour->y)) * (apcontour->nContours + 1));
      apcontour->logic = SDDS_Realloc(apcontour->logic, sizeof(*(apcontour->logic)) * (apcontour->nContours + 1));
      if (!apcontour->hasLogic)
        apcontour->logic[apcontour->nContours] = NULL;
      else if (!SDDS_GetParameter(&SDDSin, "Logic", &apcontour->logic[apcontour->nContours]))
        bombElegantVA("Error: problem getting parameter \"Logic\" from APCONTOUR file %s\n", apcontour->filename);
      apcontour->nPoints = SDDS_Realloc(apcontour->nPoints, sizeof(*(apcontour->nPoints)) * (apcontour->nContours + 1));
      if ((apcontour->nPoints[apcontour->nContours] = SDDS_RowCount(&SDDSin)) < 3)
        bombElegantVA("Error: APCONTOUR file %s page %d has too few points\n", apcontour->filename, readCode);
      if (!(apcontour->x[apcontour->nContours] = SDDS_GetColumnInDoubles(&SDDSin, apcontour->xColumn)) ||
          !(apcontour->y[apcontour->nContours] = SDDS_GetColumnInDoubles(&SDDSin, apcontour->yColumn)))
        bombElegantVA("Error: failed to get x or y data from APCONTOUR file %s\n", apcontour->filename);
      if (apcontour->x[apcontour->nContours][0] != apcontour->x[apcontour->nContours][apcontour->nPoints[apcontour->nContours] - 1] ||
          apcontour->y[apcontour->nContours][0] != apcontour->y[apcontour->nContours][apcontour->nPoints[apcontour->nContours] - 1])
        bombElegantVA("Error: contour provided in file %s for APCONTOUR is not a closed shape\n", apcontour->filename);
      apcontour->nContours += 1;
    }
    SDDS_Terminate(&SDDSin);
    printf("Read %ld aperture contours from file %s\n", apcontour->nContours, apcontour->filename);
    fflush(stdout);
    apcontour->initialized = 1;
  }
}

typedef struct {
  char *name, *type, *exclude;
  long skip, verbosity;
} SC_SPEC;
static short scSpecActive = 0;
static SC_SPEC *scSpec = NULL;
static long No_scSpec = 0;

long getSCMULTSpecCount() {
  return (No_scSpec);
}

void addSCSpec(char *name, char *type, char *exclude, long skip, long verbosity) {
  if (!(scSpec = (SC_SPEC *)SDDS_Realloc(scSpec,
                                         sizeof(*scSpec) * (No_scSpec + 1))))
    bombElegant((char *)"memory allocation failure", NULL);
  scSpec[No_scSpec].name = NULL;
  scSpec[No_scSpec].type = NULL;
  scSpec[No_scSpec].exclude = NULL;
  if ((name &&
       !SDDS_CopyString(&scSpec[No_scSpec].name, name)) ||
      (type &&
       !SDDS_CopyString(&scSpec[No_scSpec].type, type)) ||
      (exclude &&
       !SDDS_CopyString(&scSpec[No_scSpec].exclude, exclude)))
    bombElegant((char *)"memory allocation failure", NULL);

  scSpec[No_scSpec].skip = skip;
  
  No_scSpec++;
}

void finishSCSpecs() {
  if (No_scSpec)
    scSpecActive = 1;
}

void clearSCSpecs() {
  while (No_scSpec--) {
    if (scSpec[No_scSpec].name)
      free(scSpec[No_scSpec].name);
    if (scSpec[No_scSpec].type)
      free(scSpec[No_scSpec].type);
    if (scSpec[No_scSpec].exclude)
      free(scSpec[No_scSpec].exclude);
  }
  free(scSpec);
  scSpec = NULL;
  scSpecActive = 0;
  
}

long insertSCMULT(char *name, long type, long *occurrence) {
  long i;
  if (scSpecActive)
    return 0;
  for (i = 0; i < No_scSpec; i++) {
    if (scSpec[i].exclude && wild_match(name, scSpec[i].exclude))
      continue;
    if (scSpec[i].name && !wild_match(name, scSpec[i].name))
      continue;
    if (scSpec[i].type && !wild_match(entity_name[type], scSpec[i].type))
      continue;
    (*occurrence)++;
    break;
  }
  if (i==No_scSpec)
    return 0;
  
  if (*occurrence < scSpec[i].skip || scSpec[i].skip == 0)
    return (0);

  if (scSpec[i].verbosity > 0)
    printf("Inserting SCMULT after %s\n", name);
  *occurrence = 0;
  return (1);
}

