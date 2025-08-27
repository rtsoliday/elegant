/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* file    : extend_list.c
 * contents: extend_line_list(), extend_elem_list()
 *           Routines to extend linked lists for program mtt.c
 *
 * Michael Borland, 1987.
 */
#include "mdb.h"
#include "track.h"

void extend_line_list(LINE_LIST **lptr) {
  log_entry("extend_line_list");
  (*lptr)->succ = tmalloc((unsigned)sizeof(**lptr));
  ((*lptr)->succ)->pred = *lptr;
  *lptr = (*lptr)->succ;
  (*lptr)->succ = NULL;
  (*lptr)->links = NULL;
  log_exit("extend_line_list");
}

void extend_elem_list(ELEMENT_LIST **eptr) {
  (*eptr)->succ = calloc(1, sizeof(**eptr));
  ((*eptr)->succ)->pred = *eptr;
  *eptr = (*eptr)->succ;
  (*eptr)->succ = NULL;

  (*eptr)->matrix = (*eptr)->accumMatrix = (*eptr)->Mld = NULL;
  (*eptr)->type = (*eptr)->flags = (*eptr)->end_pos = 0;
  (*eptr)->p_elem = (*eptr)->name = (*eptr)->group = (*eptr)->definition_text = NULL;
  (*eptr)->D = (*eptr)->accumD = (*eptr)->DIbs = NULL;
  (*eptr)->twiss = NULL;
  (*eptr)->part_of = NULL;
  (*eptr)->sigmaMatrix = NULL;
}
