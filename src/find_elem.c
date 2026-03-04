/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* file: find_elem.c
 *
 * Michael Borland, 1989-94
 */
#include "mdb.h"
#include "track.h"

ELEMENT_LIST *find_element(char *elem_name, ELEMENT_LIST **context, ELEMENT_LIST *elem) {
  ELEMENT_LIST *eptr;

  log_entry("find_element");
  if (!elem_name)
    bombElegant("elem_name is NULL (find_element)", NULL);
  if (!elem)
    bombElegant("elem is NULL (find_element)", NULL);

  if (!context || *context == NULL)
    eptr = elem;
  else
    eptr = (*context)->succ;
  size_t search_len = strlen(elem_name);
  while (eptr) {
     if (eptr->namelen == 0)
       eptr->namelen = strlen(eptr->name);
    // else if (eptr->namelen != strlen(eptr->name))
    //   bombElegant("elem_name name length mismatch (find_element)", NULL);
    if ((eptr->namelen == search_len) && strcmp(eptr->name, elem_name) == 0)
      break;
    eptr = eptr->succ;
  }
  log_exit("find_element");
  if (context)
    return *context = eptr;
  else
    return eptr;
}

ELEMENT_LIST *find_element_hash(char *elem_name, long occurence, ELEMENT_LIST **context, ELEMENT_LIST *elem) {
  ELEMENT_LIST *eptr;
  static char occurence_s[8], name_occurence[1024];

  log_entry("find_element");
  if (!elem_name)
    bombElegant("elem_name is NULL (find_element)", NULL);
  if (!elem)
    bombElegant("elem is NULL (find_element)", NULL);

  if (!context || *context == NULL)
    eptr = elem;
  else
    eptr = (*context)->succ;

  sprintf(occurence_s, "#%ld", occurence);
  strcpy(name_occurence, elem_name);
  strcat(name_occurence, occurence_s);
  /* use hash table */
  if (hfind(load_hash, (unsigned char*)name_occurence, strlen(name_occurence)))
    eptr = (ELEMENT_LIST *)hstuff(load_hash);
  else
    eptr = NULL;

  log_exit("find_element");
  if (context)
    return *context = eptr;
  else
    return eptr;
}

ELEMENT_LIST *find_element_index(char *elem_name, ELEMENT_LIST **context, ELEMENT_LIST *elem, long *index) {
  ELEMENT_LIST *eptr;

  if (!elem_name)
    bombElegant("elem_name is NULL (find_element)", NULL);
  if (!elem)
    bombElegant("elem is NULL (find_element)", NULL);

  if (!context || *context == NULL) {
    *index = 0;
    eptr = elem;
  } else {
    *index += 1;
    eptr = (*context)->succ;
  }
  while (eptr && strcmp(eptr->name, elem_name) != 0) {
    *index += 1;
    eptr = eptr->succ;
  }
  if (!eptr)
    *index = -1;
  if (context)
    return *context = eptr;
  else
    return eptr;
}

ELEMENT_LIST *wfind_element(char *elem_name, ELEMENT_LIST **context, ELEMENT_LIST *elem) {
  ELEMENT_LIST *eptr;

  log_entry("wfind_element");
  if (!elem_name)
    bombElegant("elem_name is NULL (wfind_element)", NULL);
  if (!elem)
    bombElegant("elem is NULL (wfind_element)", NULL);

  if (!context || *context == NULL)
    eptr = elem;
  else
    eptr = (*context)->succ;
  while (eptr && !wild_match(eptr->name, elem_name))
    eptr = eptr->succ;

  log_exit("wfind_element");
  if (context)
    return *context = eptr;
  else
    return eptr;
}

long confirm_parameter(char *item_name, long type) {
  long i;
  PARAMETER *param;

  log_entry("confirm_parameter");

  if (!item_name)
    bombElegant("item_name is NULL (confirm_parameter)", NULL);

  param = entity_description[type].parameter;
  for (i = 0; i < entity_description[type].n_params; i++)
    if (strcmp(item_name, param[i].name) == 0) {
      log_exit("confirm_parameter");
      return (i);
    }
  log_exit("confirm_parameter");
  return (-1);
}
