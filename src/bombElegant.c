/* This is not used by elegant.  It is used by other programs that use some of elegant's
 * subroutines.  Elegant.c has a copy that is used by elegant itself.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

void bombElegant(char *error, char *usage) {
  if (error)
    fprintf(stderr, "error: %s\n", error);
  if (usage)
    fprintf(stderr, "usage: %s\n", usage);
  exit(1);
}

void exitElegant(long status) {
  exit(status);
}

void bombElegantVA(char *template, ...) {
  va_list argp;

  va_start(argp, template);
  vprintf(template, argp);
  va_end(argp);
  exit(1);
}
