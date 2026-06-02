#include "mdb.h"
#include "track.h"

typedef struct {
  char *text;
  long count;
} WARNING_RECORD;

WARNING_RECORD **warningRecord = NULL;
static long warnings = 0;
static FILE *fpWarn = NULL;
static htab *hash_table = NULL;
/* Nestable suppression counter.  When >0, printWarning prints nothing and
 * does not bump per-text counters; used by correction routines that
 * intentionally try a step that might destabilise the lattice and roll back. */
static long warningSuppressDepth = 0;

void setWarningFilePointer(FILE *fp) {
  fpWarn = fp;
}

void pushWarningSuppression(void) {
  warningSuppressDepth++;
}

void popWarningSuppression(void) {
  if (warningSuppressDepth > 0) warningSuppressDepth--;
}

void printWarning(char *text, char *detail) {
  printWarningWithContext(NULL, NULL, text, detail);
}

void printWarningForTracking(char *text, char *detail)
/* extracts element info from tracking context */
{
  TRACKING_CONTEXT trackingContext;
  char warningBuffer[1024];
  getTrackingContext(&trackingContext);
  if (!strlen(trackingContext.elementName) || !trackingContext.elementOccurrence || !trackingContext.element)
    printWarning(text, detail);
  else {
    snprintf(warningBuffer, 1024, "%s#%ld", trackingContext.elementName, trackingContext.elementOccurrence);
    printWarningWithContext(entity_name[trackingContext.element->type], warningBuffer,
                            text, detail);
  }
}

void printWarningWithContext(char *context1, char *context2, char *text, char *detail) {
  WARNING_RECORD *wrPointer = NULL;

  if (warningSuppressDepth > 0) return;
  if (!hash_table)
    hash_table = hcreate(12);
  if (hcount(hash_table) == 0 ||
      hfind(hash_table, (unsigned char*)text, strlen(text)) == FALSE) {
    if (!(warningRecord = SDDS_Realloc(warningRecord, sizeof(*warningRecord) * (warnings + 1))))
      bombElegant("Memory allocation error in printWarning\n", NULL);
    warningRecord[warnings] = malloc(sizeof(**warningRecord));
    cp_str(&(warningRecord[warnings]->text), text);
    warningRecord[warnings]->count = 1;
    wrPointer = warningRecord[warnings];
    hadd(hash_table, (unsigned char*)warningRecord[warnings]->text, strlen(warningRecord[warnings]->text),
         (void *)warningRecord[warnings]);
    warnings++;
  } else {
    wrPointer = hstuff(hash_table);
    if (!wrPointer)
      bombElegantVA("Warning counter pointer undefined. Text is %s.\n",
                    text);
    wrPointer->count += 1;
  }

  if (!fpWarn)
    fpWarn = stdout;
  if (warningCountLimit < 0 || (wrPointer->count - 1) < warningCountLimit) {
    fprintf(fpWarn, "*** Warning: %s", text);
    if (context1 && strlen(context1)) {
      if (context2 && strlen(context2)) {
        /* context1 is typically the element type name, context2 is typically the element name */
        if (detail && strlen(detail))
          fprintf(fpWarn, " --- %s %s: %s\n", context1, context2, detail);
        else
          fprintf(fpWarn, " --- %s %s\n", context1, context2);
      } else {
        /* context1 is typically the command name or subroutine name */
        if (detail && strlen(detail))
          fprintf(fpWarn, " --- %s: %s\n", context1, detail);
        else
          fprintf(fpWarn, " --- %s\n", context1);
      }
    } else {
      if (detail && strlen(detail))
        fprintf(fpWarn, " --- %s\n", detail);
      else
        fprintf(fpWarn, "\n");
    }
    if (warningCountLimit > 0 && wrPointer->count == warningCountLimit)
      fprintf(fpWarn, "Further warnings of this type will be suppressed. Increase warning_limit with global_settings to see more, or set to zero to see everything.\n");
    fflush(fpWarn);
  }
}

void summarizeWarnings() {
  long i;
  if (!fpWarn)
    fpWarn = stdout;
  if (warnings) {
    fprintf(fpWarn, "\n************************** Summary of warnings ****************************\n");
    fprintf(fpWarn, "%ld types of warnings were recorded:\n", warnings);
    for (i = 0; i < warnings; i++) {
      fprintf(fpWarn, "%ld* %s\n",
              warningRecord[i]->count, warningRecord[i]->text);
      free(warningRecord[i]->text);
      free(warningRecord[i]);
    }
    fprintf(fpWarn, "*****************************************************************************\n\n");

    warnings = 0;
    hdestroy(hash_table);
    hash_table = NULL;
    free(warningRecord);
  }
}
