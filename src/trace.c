/*************************************************************************\
 * Copyright (c) 2002 The University of Chicago, as Operator of Argonne
 * National Laboratory.
 * Copyright (c) 2002 The Regents of the University of California, as
 * Operator of Los Alamos National Laboratory.
 * This file is distributed subject to a Software License Agreement found
 * in the file LICENSE that is included with this distribution.
\*************************************************************************/

/* file: trace.c
 * purpose: routines for tracing program calls
 * method: a file is maintained that contains the calling sequence.  It is
 *         updated by file positioning using a stack of file positions.
 *         In addition, a list of strings is maintained so that a trace-back
 *         can be given in event of a crash.
 *
 * M.Borland, 1992
 */
#include "mdb.h"
#include "track.h"
#include "trace.h"
#include <signal.h>

#define MAX_LENGTH 78

static long trace_level = 0;
static long max_level = 0;
static char **routine_name = NULL;
static FILE *fp = NULL;
static FILE *fpmem = NULL;
static long *file_pos = NULL;
static char blank[MAX_LENGTH];
static long in_trace_routine = 0;
static long memory_level = 0;

void process_trace_request(NAMELIST_TEXT *nltext) {
  long i;

  if (fp)
    fclose(fp);
  fp = NULL;

  /* process the namelist text */
  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&trace, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (echoNamelists)
    print_namelist(stdout, &trace);

  if (record_allocation && filename)
    keep_alloc_record(filename);

  trace_mode = 0;

  if (!filename && trace_on)
    bombElegant("provide filename for program trace", NULL);
  if (memory_log)
    fpmem = fopen_e(memory_log, "w", 0);
  else
    fpmem = NULL;
  trace_mode += (memory_log ? TRACE_MEMORY_LEVEL : 0);
  trace_mode += (traceback_on ? TRACEBACK_ON : 0);
  if (trace_on) {
    fp = fopen_e(filename, "w", 0);
    for (i = 0; i < MAX_LENGTH; i++)
      blank[i] = ' ';
    trace_mode += TRACE_ENTRY + (heap_verify_depth ? TRACE_HEAP_VERIFY : 0);
    printf("trace activated into file %s\n", filename);
  }
}

void log_entry(const char *routine) {
  long len;

  if (!trace_mode)
    return;

  in_trace_routine = 1;

  if (immediate) {
    char s[1000];
    sprintf(s, "Entering %s\n", routine);
    printMessageAndTime(stdout, s);
  }

  if (trace_level < 0) {
    printf("error: trace level is negative (log_entry)\n");
    fflush(stdout);
    printf("calling routine is %s\n", routine);
    fflush(stdout);
    exitElegant(1);
  }

  if (trace_mode & TRACE_MEMORY_LEVEL)
    memory_level = memoryUsage();

  if (trace_mode & TRACE_ENTRY) {
    /* keep track of calls in a file and in an internal list */
    if (max_level <= trace_level) {
      file_pos = trealloc(file_pos, sizeof(*file_pos) * (max_level = trace_level + 1));
      routine_name = trealloc(routine_name, sizeof(*routine_name) * (max_level));
      file_pos[trace_level] = ftell(fp);
      routine_name[trace_level] = tmalloc(sizeof(**routine_name) * (MAX_LENGTH + 1));
    } else {
      fseek(fp, file_pos[trace_level], 0);
    }
    strncpy(routine_name[trace_level], routine, MAX_LENGTH);
    if ((len = strlen(routine)) > MAX_LENGTH)
      len = MAX_LENGTH;
    fwrite(routine, sizeof(*routine), len = strlen(routine), fp);
    if (len < MAX_LENGTH)
      fwrite(blank, sizeof(*blank), MAX_LENGTH - len, fp);
    fputc('\n', fp);
    fflush(fp);
  }
  if (trace_mode & TRACEBACK_ON) {
    /* keep track of calls internally */
    if (max_level <= trace_level) {
      routine_name = trealloc(routine_name, sizeof(*routine_name) * (max_level = trace_level + 1));
      routine_name[trace_level] = tmalloc(sizeof(**routine_name) * (MAX_LENGTH + 1));
    }
    strncpy(routine_name[trace_level], routine, MAX_LENGTH);
  }
  trace_level++;
  in_trace_routine = 0;
}

void log_exit(const char *routine) {
  long memlev;

  if (!trace_mode)
    return;

  if (immediate) {
    char s[1000];
    sprintf(s, "Exiting %s\n", routine);
    printMessageAndTime(stdout, s);
  }

  in_trace_routine = 2;
  if (trace_mode & TRACE_MEMORY_LEVEL) {
    memlev = memoryUsage();
    if (memlev != memory_level) {
      fprintf(fpmem, "memory changed to %ld inside %s\n", memlev,
              routine_name[trace_level - 1] ? routine_name[trace_level - 1] : "{NULL}");
      fflush(fpmem);
    }
    memory_level = memlev;
  }
  if (trace_mode & TRACE_ENTRY) {
    if (trace_level <= 0) {
      printf("error: trace level is nonpositive (log_exit)\n");
      fflush(stdout);
      printf("calling routine is %s\n", routine);
      fflush(stdout);
      exitElegant(1);
    }
    fseek(fp, file_pos[trace_level - 1], 0);
    fwrite(blank, sizeof(*blank), MAX_LENGTH, fp);
    fflush(fp);
  }
  trace_level--;
  in_trace_routine = 0;
}

void traceback_handler(int sig) {
  long i;
  switch (sig) {
#if !defined(_WIN32)
  case SIGHUP:
    printf("\nTerminated by SIGHUP\n");
    break;
  case SIGQUIT:
    printf("\nTerminated by SIGQUIT\n");
    break;
  case SIGTRAP:
    printf("\nTerminated by SIGTRAP\n");
    break;
  case SIGBUS:
    printf("\nTerminated by SIGBUS");
    break;
#endif
  case SIGINT:
    printf("\nTerminated by SIGINT\n");
    break;
  case SIGABRT:
    printf("\nTerminated by SIGABRT\n");
    break;
  case SIGILL:
    printf("\nTerminated by SIGILL\n");
    break;
  case SIGFPE:
    printf("\nTerminated by SIGFPE");
    break;
  case SIGSEGV:
    printf("\nTerminated by SIGSEGV");
    break;
  default:
    printf("\nTerminated by unknown signal\n");
    break;
  }
  printf("\nProgram trace-back:\n");
  for (i = 0; i < trace_level; i++) {
    fputs(routine_name[i], stdout);
    fputc('\n', stdout);
  }
  if (in_trace_routine == 1)
    printf("log_entry\n");
  else if (in_trace_routine == 2)
    printf("log_exit\n");
  fflush(stdout); /* to force flushing of output sent to stdout by other parts of the code */
  exitElegant(1);
}

void show_traceback(FILE *fp) {
  long i;
  if (!traceback_on && !trace_on)
    return;
  printf("\nProgram trace-back:\n");
  for (i = 0; i < trace_level; i++) {
    fputs(routine_name[i], stdout);
    fputc('\n', stdout);
  }
  if (in_trace_routine == 1)
    printf("log_entry\n");
  else if (in_trace_routine == 2)
    printf("log_exit\n");
  fflush(stdout); /* to force flushing of output sent to stdout by other parts of the code */
}

void printMessageAndTime(FILE *fp, char *message) {
  char *tString;
  tString = mtimes();
  fprintf(fp, "%s: %s", tString, message);
  fflush(fp);
  free(tString);
}

long memoryUsage()
/* Memory used across all cores */
{
  long memoryUsed;

  memoryUsed = memory_count();

#if USE_MPI
  return memoryUsed;
  /* collect memory information from all cores---unfortunately, this can sometimes hang
     because in some contexts and processor may finish its work and exit a loop to wait
     for other processors
  long memoryUsedTotal;
  MPI_Allreduce(&memoryUsed, &memoryUsedTotal, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  memoryUsed = memoryUsedTotal;
 */
#endif

  return memoryUsed;
}
