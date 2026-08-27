/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution.
\*************************************************************************/

/* file: knobs.c
 *
 * Implements the &load_knobs setup command and the knob registry it
 * populates.  See knobs.h for the public API.
 */

#include "mdb.h"
#include "SDDS.h"
#include "track.h"
#include "knobs.h"
#include "load_knobs.h"

KNOB *knobList = NULL;
long nKnobs = 0;

long find_knob(const char *name) {
  long i;
  if (!name || nKnobs == 0)
    return -1;
  for (i = 0; i < nKnobs; i++) {
    if (strcmp(knobList[i].name, name) == 0)
      return i;
  }
  return -1;
}

double get_knob_value(long knobIndex) {
  if (knobIndex < 0 || knobIndex >= nKnobs)
    return 0.0;
  return knobList[knobIndex].currentValue;
}

void set_knob_value(long knobIndex, double newValue, LINE_LIST *beamline) {
  KNOB *knob;
  double delta;
  long iT;

  if (knobIndex < 0 || knobIndex >= nKnobs)
    bombElegant("set_knob_value: invalid knob index", NULL);

  knob = &knobList[knobIndex];
  delta = newValue - knob->currentValue;

  for (iT = 0; iT < knob->nTargets; iT++) {
    ELEMENT_LIST *eptr = knob->eptr[iT];
    char *p_elem = eptr->p_elem;
    long offset = knob->paramOffset[iT];
    double factor = knob->factor[iT];
    long paramIdx = knob->paramIndex[iT];
    switch (knob->paramType[iT]) {
    case IS_DOUBLE:
      *((double *)(p_elem + offset)) += delta * factor;
      break;
    case IS_LONG:
      *((long *)(p_elem + offset)) += nearestInteger(delta * factor);
      break;
    case IS_INT64:
      *((int64_t *)(p_elem + offset)) += nearestInteger64(delta * factor);
      break;
    case IS_SHORT:
      *((short *)(p_elem + offset)) += nearestInteger(delta * factor);
      break;
    default:
      bombElegant("set_knob_value: unsupported parameter data type", NULL);
    }
    /* Invalidate the cached element matrix when a matrix-relevant parameter
     * moves, mirroring assert_element_links() in link_elements.c. */
    if ((entity_description[eptr->type].parameter[paramIdx].flags & PARAM_CHANGES_MATRIX) &&
        eptr->matrix) {
      free_matrices(eptr->matrix);
      tfree(eptr->matrix);
      eptr->matrix = NULL;
    }
  }

  knob->currentValue = newValue;
  rpn_store(newValue, NULL, knob->rpnMem);

  if (knob->matrixChanges && beamline) {
    beamline->flags &= ~(BEAMLINE_CONCAT_CURRENT | BEAMLINE_TWISS_CURRENT |
                         BEAMLINE_RADINT_CURRENT | BEAMLINE_RADINT_DONE);
  }
}

void perturb_knob(long knobIndex, double delta, LINE_LIST *beamline) {
  if (knobIndex < 0 || knobIndex >= nKnobs)
    bombElegant("perturb_knob: invalid knob index", NULL);
  set_knob_value(knobIndex, knobList[knobIndex].currentValue + delta, beamline);
}

void setup_load_knobs(NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline) {
  SDDS_DATASET SDDSin;
  char *resolvedFilename = NULL;
  long pageCode;
  short hasOccurence = 0;

  log_entry("setup_load_knobs");

  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&load_knobs, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (echoNamelists)
    print_namelist(stdout, &load_knobs);

  if (!filename || !strlen(filename))
    bombElegant("load_knobs: filename is required", NULL);

  resolvedFilename = compose_filename(filename, run->rootname);

  if (!SDDS_InitializeInputFromSearchPath(&SDDSin, resolvedFilename))
    bombElegantVA("load_knobs: cannot open %s\n", resolvedFilename);

  if (SDDS_CheckParameter(&SDDSin, "KnobName", NULL, SDDS_STRING, stderr) != SDDS_CHECK_OK)
    bombElegant("load_knobs: file must have a STRING parameter named KnobName", NULL);
  if (SDDS_CheckColumn(&SDDSin, "ElementName", NULL, SDDS_STRING, stderr) != SDDS_CHECK_OK)
    bombElegant("load_knobs: file must have a STRING column named ElementName", NULL);
  if (SDDS_CheckColumn(&SDDSin, "ElementParameter", NULL, SDDS_STRING, stderr) != SDDS_CHECK_OK)
    bombElegant("load_knobs: file must have a STRING column named ElementParameter", NULL);
  if (SDDS_CheckColumn(&SDDSin, "Factor", NULL, SDDS_ANY_NUMERIC_TYPE, stderr) != SDDS_CHECK_OK)
    bombElegant("load_knobs: file must have a numeric column named Factor", NULL);
  /* ElementOccurence is optional.  When present, each row's ElementName is
   * resolved to a single element matching both name and occurence; rows in
   * the same page may then repeat ElementName values with different
   * occurences.  Wildcards in ElementName are not permitted in this mode. */
  if (SDDS_CheckColumn(&SDDSin, "ElementOccurence", NULL, SDDS_ANY_INTEGER_TYPE, NULL) == SDDS_CHECK_OK)
    hasOccurence = 1;

  while ((pageCode = SDDS_ReadPage(&SDDSin)) > 0) {
    char *knobName = NULL, *knobNameUpper = NULL;
    char **elementName = NULL, **elementParameter = NULL;
    double *factor = NULL;
    int32_t *elementOccurence = NULL;
    long nRows, iRow;
    KNOB *knob;
    char rpnName[256];

    if (!SDDS_GetParameter(&SDDSin, "KnobName", &knobName) || !knobName)
      bombElegantVA("load_knobs: KnobName missing on page %ld of %s\n", pageCode, resolvedFilename);

    knobNameUpper = tmalloc(sizeof(*knobNameUpper) * (strlen(knobName) + 1));
    strcpy(knobNameUpper, knobName);
    str_toupper(knobNameUpper);

    if (find_knob(knobNameUpper) >= 0)
      bombElegantVA("load_knobs: duplicate knob name '%s' on page %ld of %s\n",
                    knobNameUpper, pageCode, resolvedFilename);

    nRows = SDDS_RowCount(&SDDSin);
    if (nRows < 1) {
      char detail[256];
      snprintf(detail, sizeof(detail), "knob '%s' on page %ld has no rows; skipping", knobNameUpper, pageCode);
      printWarning("load_knobs: empty page", detail);
      free(knobName);
      free(knobNameUpper);
      continue;
    }

    if (!(elementName = (char **)SDDS_GetColumn(&SDDSin, "ElementName")))
      bombElegantVA("load_knobs: cannot read ElementName on page %ld\n", pageCode);
    if (!(elementParameter = (char **)SDDS_GetColumn(&SDDSin, "ElementParameter")))
      bombElegantVA("load_knobs: cannot read ElementParameter on page %ld\n", pageCode);
    if (!(factor = SDDS_GetColumnInDoubles(&SDDSin, "Factor")))
      bombElegantVA("load_knobs: cannot read Factor on page %ld\n", pageCode);
    if (hasOccurence) {
      int32_t nValues;
      if (!(elementOccurence = SDDS_GetColumnInLong(&SDDSin, "ElementOccurence")))
        bombElegantVA("load_knobs: cannot read ElementOccurence on page %ld\n", pageCode);
      nValues = nRows;
      /* Reject wildcards in ElementName when ElementOccurence is present:
       * occurence specifies a unique element so wildcards are incompatible. */
      for (iRow = 0; iRow < nValues; iRow++) {
        if (elementName[iRow] && has_wildcards(elementName[iRow]))
          bombElegantVA("load_knobs: knob '%s' row %ld: ElementName '%s' contains wildcards, which is not permitted when ElementOccurence is supplied\n",
                        knobNameUpper, iRow, elementName[iRow]);
      }
    }

    knobList = (KNOB *)trealloc(knobList, sizeof(*knobList) * (nKnobs + 1));
    knob = &knobList[nKnobs];
    memset(knob, 0, sizeof(*knob));
    knob->name = knobNameUpper;
    knob->currentValue = 0.0;

    for (iRow = 0; iRow < nRows; iRow++) {
      ELEMENT_LIST *context = NULL;
      char *eName = elementName[iRow];
      char *pName = elementParameter[iRow];
      long matchedThisRow = 0;
      char *expandedName;

      if (!eName || !strlen(eName) || !pName || !strlen(pName)) {
        char detail[256];
        snprintf(detail, sizeof(detail), "knob '%s' row %ld: empty ElementName or ElementParameter; skipping",
                 knobNameUpper, iRow);
        printWarning("load_knobs: empty row entry", detail);
        continue;
      }

      expandedName = eName;
      if (has_wildcards(expandedName) && strchr(expandedName, '-'))
        expandedName = expand_ranges(expandedName);
      str_toupper(expandedName);
      str_toupper(pName);

      while ((context = wfind_element(expandedName, &context, beamline->elem))) {
        long pIdx, pType, pOffset;
        if (hasOccurence && context->occurence != elementOccurence[iRow])
          continue;
        if ((pIdx = confirm_parameter(pName, context->type)) < 0) {
          /* Skip elements whose type doesn't have this parameter (allowed
           * when wildcards span multiple element types). */
          continue;
        }
        if (entity_description[context->type].parameter[pIdx].flags & PARAM_IS_LOCKED) {
          bombElegantVA("load_knobs: knob '%s' row %ld: parameter %s of %s is locked\n",
                        knobNameUpper, iRow, pName, context->name);
        }
        pType = entity_description[context->type].parameter[pIdx].type;
        if (pType != IS_DOUBLE && pType != IS_LONG && pType != IS_INT64 && pType != IS_SHORT) {
          bombElegantVA("load_knobs: knob '%s' row %ld: parameter %s of %s is not numeric\n",
                        knobNameUpper, iRow, pName, context->name);
        }
        pOffset = entity_description[context->type].parameter[pIdx].offset;

        knob->eptr = (ELEMENT_LIST **)trealloc(knob->eptr, sizeof(*knob->eptr) * (knob->nTargets + 1));
        knob->paramIndex = (long *)trealloc(knob->paramIndex, sizeof(*knob->paramIndex) * (knob->nTargets + 1));
        knob->paramOffset = (long *)trealloc(knob->paramOffset, sizeof(*knob->paramOffset) * (knob->nTargets + 1));
        knob->paramType = (short *)trealloc(knob->paramType, sizeof(*knob->paramType) * (knob->nTargets + 1));
        knob->factor = (double *)trealloc(knob->factor, sizeof(*knob->factor) * (knob->nTargets + 1));

        knob->eptr[knob->nTargets] = context;
        knob->paramIndex[knob->nTargets] = pIdx;
        knob->paramOffset[knob->nTargets] = pOffset;
        knob->paramType[knob->nTargets] = (short)pType;
        knob->factor[knob->nTargets] = factor[iRow];
        if (entity_description[context->type].parameter[pIdx].flags & PARAM_CHANGES_MATRIX)
          knob->matrixChanges = 1;
        knob->nTargets++;
        matchedThisRow++;
      }

      if (!matchedThisRow) {
        char detail[256];
        snprintf(detail, sizeof(detail),
                 "knob '%s' row %ld: pattern '%s' matched no element with parameter '%s'",
                 knobNameUpper, iRow, eName, pName);
        printWarning("load_knobs: row matched no elements", detail);
      }
    }

    snprintf(rpnName, sizeof(rpnName), "%s.value", knobNameUpper);
    knob->rpnMem = rpn_create_mem(rpnName, 0);
    rpn_store(0.0, NULL, knob->rpnMem);

    if (verbose) {
      printf("load_knobs: registered knob '%s' with %ld element-parameter target(s)\n",
             knobNameUpper, knob->nTargets);
      fflush(stdout);
    }

    nKnobs++;

    SDDS_FreeStringArray(elementName, nRows);
    free(elementName);
    SDDS_FreeStringArray(elementParameter, nRows);
    free(elementParameter);
    free(factor);
    if (elementOccurence)
      free(elementOccurence);
    free(knobName);
    /* knobNameUpper is owned by knob->name now -- do not free. */
  }

  if (!SDDS_Terminate(&SDDSin))
    bombElegantVA("load_knobs: SDDS_Terminate failed on %s\n", resolvedFilename);

  log_exit("setup_load_knobs");
}
