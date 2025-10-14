/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* program: elegantto
 * purpose: convert elegant lattice files to other formats
 * 
 * Michael Borland, 1993
 */
#include "mdb.h"
#include "scan.h"
#include "match_string.h"
#include <ctype.h>
#include "madto.h"

char *USAGE = "elegantto inputfile outputfile \n\
 { -EmmaMatlab | -patpet | -patricia | -parmela[=quad_ap(mm),sext_ap(mm),p(MeV/c)]\n\
  -transport[=quad_ap(mm),sext_ap(mm),p(GeV/c)] | -sdds=[p(GeV/c)] | [-xorbit] \n\
  -cosy=quad_ap(mm),sext_ap(mm),p(MeV/c),order | -mad8 | -sad}\n\
 [-angle_tolerance=value] [-flip_k_signs] [-magnets=filename]\n\
 [-header=filename] [-ender=filename]\n\n\
elegantto converts elegant's accelerator lattice format to various other formats.\n\
N.B.: the units for various quantities are not the same for different options!.\n\
Program by Michael Borland.  (This is Version 8, July 2025.)\n";

#define SET_CONVERT_TO_PATRICIA 0
#define SET_CONVERT_TO_TRANSPORT 1
#define SET_CONVERT_TO_PARMELA 2
#define SET_CONVERT_TO_PATPET 3
#define SET_CONVERT_TO_SDDS 4
#define SET_CONVERT_TO_XORBIT 5
#define SET_ANGLE_TOLERANCE 6
#define SET_FLIP_K_SIGNS 7
#define SET_MAGNET_OUTPUT 8
#define SET_HEADER_FILE 9
#define SET_ENDER_FILE 10
#define SET_CONVERT_TO_EMMAMATLAB 11
#define SET_CONVERT_TO_COSY 12
#define SET_CONVERT_TO_MAD8 13
#define SET_CONVERT_TO_SAD 14
#define N_OPTIONS 15
char *option[N_OPTIONS] = {
  "patricia", "transport", "parmela", "patpet", "sdds", "xorbit",
  "angle_tolerance", "flip_k_signs", "magnets", "header", "ender",
  "emmamatlab", "cosy", "mad8", "sad"};

#define TRANSPORT_OUTPUT 1
#define PATRICIA_OUTPUT 2
#define PATPET_OUTPUT 3
#define PARMELA_OUTPUT 4
#define SDDS_OUTPUT 5
#define XORBIT_OUTPUT 6
#define EMMAMATLAB_OUTPUT 7
#define COSY_OUTPUT 8
#define MAD8_OUTPUT 9
#define SAD_OUTPUT 10

void convert_to_mad8(char *outputfile, LINE_LIST *beamline, char *header_file, char *ender_file);
void convert_to_sad(char *outputfile, LINE_LIST *beamline, char *header_file, char *ender_file);

/* Needed by insertSCeffects.cc */
long isSlave;
long notSinglePart;     
void index_bunch_assignments(double **part, long np, long idSlotsPerBunch, double P0,
                             double **time, long **ibParticle, long ***ipBunch, long **npBunch,
                             long *nBunches, long lastNBunches
) {
}

int main(int argc, char **argv) {
  LINE_LIST *beamline;
  char *input, *output, *magnets;
  long i, flip_k_signs, output_mode;
  char *header_file, *ender_file;
  double angle_tolerance;
  SCANNED_ARG *scanned;
  double quad_ap, sext_ap, p_cent;
  long order;

  if (argc < 3 || argc > 7)
    bomb(NULL, USAGE);

  scanargs(&scanned, argc, argv);
  header_file = ender_file = NULL;
  input = output = magnets = NULL;
  flip_k_signs = output_mode = 0;
  quad_ap = sext_ap = 50;
  p_cent = 1e9;

  for (i = 1; i < argc; i++) {
    if (scanned[i].arg_type == OPTION) {
      switch (match_string(scanned[i].list[0], option, N_OPTIONS, 0)) {
      case SET_CONVERT_TO_PATRICIA:
        if (output_mode)
          bomb("can only convert to one format at a time", USAGE);
        output_mode = PATRICIA_OUTPUT;
        break;
      case SET_CONVERT_TO_PATPET:
        if (output_mode)
          bomb("can only convert to one format at a time", USAGE);
        output_mode = PATPET_OUTPUT;
        break;
      case SET_CONVERT_TO_TRANSPORT:
        if (output_mode)
          bomb("can only convert to one format at a time", USAGE);
        output_mode = TRANSPORT_OUTPUT;
        if (scanned[i].n_items != 1 && scanned[i].n_items != 4)
          bomb("wrong number of values with -transport option", USAGE);
        if (scanned[i].n_items == 4 &&
            (sscanf(scanned[i].list[1], "%lf", &quad_ap) != 1 || quad_ap <= 0 ||
             sscanf(scanned[i].list[2], "%lf", &sext_ap) != 1 || sext_ap <= 0 ||
             sscanf(scanned[i].list[3], "%lf", &p_cent) != 1 || p_cent <= 0))
          bomb("invalid values for -transport option", USAGE);
        if (scanned[i].n_items == 4)
          p_cent *= 1e9;
        break;
      case SET_CONVERT_TO_PARMELA:
        if (output_mode)
          bomb("can only convert to one format at a time", USAGE);
        output_mode = PARMELA_OUTPUT;
        if (scanned[i].n_items != 1 && scanned[i].n_items != 4)
          bomb("wrong number of values with -parmela option", USAGE);
        if (scanned[i].n_items == 4 &&
            (sscanf(scanned[i].list[1], "%lf", &quad_ap) != 1 || quad_ap <= 0 ||
             sscanf(scanned[i].list[2], "%lf", &sext_ap) != 1 || sext_ap <= 0 ||
             sscanf(scanned[i].list[3], "%lf", &p_cent) != 1 || p_cent <= 0))
          bomb("invalid values for -parmela option", USAGE);
        if (scanned[i].n_items == 4)
          p_cent *= 1e6;
        break;
      case SET_CONVERT_TO_SDDS:
        if (output_mode)
          bomb("can only convert to one format at a time", USAGE);
        output_mode = SDDS_OUTPUT;
        if (scanned[i].n_items < 1 || scanned[i].n_items > 2)
          bomb("wrong number of values with -sdds option", USAGE);
        if (scanned[i].n_items == 2) {
          if (sscanf(scanned[i].list[1], "%lf", &p_cent) != 1 || p_cent < 0)
            bomb("invalid values for -sdds option", USAGE);
          p_cent *= 1e3;
        }
        break;
      case SET_CONVERT_TO_XORBIT:
        if (output_mode)
          bomb("can only convert to one format at a time", USAGE);
        output_mode = XORBIT_OUTPUT;
        break;
      case SET_ANGLE_TOLERANCE:
        if (scanned[i].n_items != 2)
          bomb(NULL, USAGE);
        if (1 != sscanf(scanned[i].list[1], "%lf", &angle_tolerance))
          bomb(NULL, USAGE);
        break;
      case SET_FLIP_K_SIGNS:
        flip_k_signs = 1;
        break;
      case SET_MAGNET_OUTPUT:
        if (scanned[i].n_items != 2)
          bomb("invalid -magnet_output syntax", USAGE);
        magnets = scanned[i].list[1];
        break;
      case SET_ENDER_FILE:
        if (ender_file)
          bomb("duplicate -ender argument", USAGE);
        ender_file = scanned[i].list[1];
        break;
      case SET_HEADER_FILE:
        if (header_file)
          bomb("duplicate -header argument", USAGE);
        header_file = scanned[i].list[1];
        break;
      case SET_CONVERT_TO_EMMAMATLAB:
        if (output_mode)
          bomb("can only convert to one format at a time", USAGE);
        output_mode = EMMAMATLAB_OUTPUT;
        if (scanned[i].n_items != 1)
          bomb("wrong number of values with -EmmaMatlab option", USAGE);
        break;
      case SET_CONVERT_TO_COSY:
        if (output_mode)
          bomb("can only convert to one format at a time", USAGE);
        output_mode = COSY_OUTPUT;
        if (scanned[i].n_items != 5)
          bomb("wrong number of values with -cosy option", USAGE);
        if (sscanf(scanned[i].list[1], "%lf", &quad_ap) != 1 || quad_ap <= 0 ||
            sscanf(scanned[i].list[2], "%lf", &sext_ap) != 1 || sext_ap <= 0 ||
            sscanf(scanned[i].list[3], "%lf", &p_cent) != 1 || p_cent <= 0 ||
            sscanf(scanned[i].list[4], "%ld", &order) != 1 || order <= 0)
          bomb("invalid values for -cosy option", USAGE);
        break;
      case SET_CONVERT_TO_MAD8:
        if (output_mode)
          bomb("can only convert to one format at a time", USAGE);
        output_mode = MAD8_OUTPUT;
        break;
      case SET_CONVERT_TO_SAD:
        if (output_mode)
          bomb("can only convert to one format at a time", USAGE);
        output_mode = SAD_OUTPUT;
        break;
      default:
        bomb("unknown option given", USAGE);
        break;
      }
    } else {
      /* filenames */
      if (!input)
        input = scanned[i].list[0];
      else if (!output)
        output = scanned[i].list[0];
      else
        bomb("too many file names listed", USAGE);
    }
  }

  if (!input || !output)
    bomb("too few file names listed", USAGE);
  if (!output_mode)
    bomb("output type not specified", USAGE);

  compute_offsets();
  if (getenv("RPN_DEFNS"))
    rpn(getenv("RPN_DEFNS"));

  switch (output_mode) {
  case TRANSPORT_OUTPUT:
    beamline = get_beamline(input, NULL, p_cent / (1e6 * particleMassMV), 0, 0, NULL, NULL);
    if (magnets)
      output_magnets(magnets, beamline->name, beamline);
    convert_to_transport(output, beamline, flip_k_signs, angle_tolerance,
                         header_file, ender_file, quad_ap, sext_ap, p_cent);
    break;
  case PATRICIA_OUTPUT:
    set_max_name_length(4);
    beamline = get_beamline(input, NULL, p_cent / (1e6 * particleMassMV), 0, 0, NULL, NULL);
    if (magnets)
      output_magnets(magnets, beamline->name, beamline);
    convert_to_patricia(output, beamline, flip_k_signs, angle_tolerance, header_file, ender_file);
    break;
  case PATPET_OUTPUT:
    set_max_name_length(4);
    beamline = get_beamline(input, NULL, p_cent / (1e6 * particleMassMV), 0, 0, NULL, NULL);
    if (magnets)
      output_magnets(magnets, beamline->name, beamline);
    convert_to_patpet(output, beamline, flip_k_signs, angle_tolerance, header_file, ender_file);
    break;
  case PARMELA_OUTPUT:
    beamline = get_beamline(input, NULL, p_cent / (1e6 * particleMassMV), 0, 0, NULL, NULL);
    if (magnets)
      output_magnets(magnets, beamline->name, beamline);
    convert_to_parmela(output, beamline, flip_k_signs, angle_tolerance, header_file, ender_file,
                       quad_ap, sext_ap, p_cent * 1e3);
    break;
  case SDDS_OUTPUT:
    set_max_name_length(12);
    beamline = get_beamline(input, NULL, p_cent / (1e6 * particleMassMV), 0, 0, NULL, NULL);
    if (magnets)
      output_magnets(magnets, beamline->name, beamline);
    sdds_strength_output(output, beamline, input);
    break;
  case XORBIT_OUTPUT:
    beamline = get_beamline(input, NULL, p_cent / (1e6 * particleMassMV), 0, 0, NULL, NULL);
    if (magnets)
      output_magnets(magnets, beamline->name, beamline);
    convert_to_xorbit(output, beamline, flip_k_signs, header_file, ender_file);
    break;
  case EMMAMATLAB_OUTPUT:
    beamline = get_beamline(input, NULL, p_cent / (1e6 * particleMassMV), 0, 0, NULL, NULL);
    if (magnets)
      output_magnets(magnets, beamline->name, beamline);
    convert_to_EmmaMatlab(output, beamline, header_file, ender_file);
    break;
  case COSY_OUTPUT:
    beamline = get_beamline(input, NULL, p_cent / particleMassMV, 0, 0, NULL, NULL);
    if (magnets)
      output_magnets(magnets, beamline->name, beamline);
    convert_to_cosy(output, beamline, order, p_cent / particleMassMV, quad_ap / 1e3, sext_ap / 1e3);
    break;
  case MAD8_OUTPUT:
    beamline = get_beamline(input, NULL, p_cent / (1e6 * particleMassMV), 0, 0, NULL, NULL);
    if (magnets)
      output_magnets(magnets, beamline->name, beamline);
    convert_to_mad8(output, beamline, header_file, ender_file);
    break;
  case SAD_OUTPUT:
    beamline = get_beamline(input, NULL, p_cent / (1e6 * particleMassMV), 0, 0, NULL, NULL);
    if (magnets)
      output_magnets(magnets, beamline->name, beamline);
    convert_to_sad(output, beamline, header_file, ender_file);
    break;
  default:
    bomb("internal error--unknown output mode", NULL);
    break;
  }
  return (0);
}

char *quoted_label(char *s) {
  char *ptr;
  ptr = malloc((unsigned)sizeof(*ptr) * (strlen(s) + 3));
  sprintf(ptr, "'%s'", s);
  return (ptr);
}

void do_output(FILE *fp, char *s) {
  char *ptr;

  ptr = s;
  while (*ptr) {
    if (islower(*ptr))
      *ptr = toupper(*ptr);
    ptr++;
  }

  fputs(s, fp);
}

long nearly_equal(double x1, double x2, double frac) {
  double tmp;

  if (x1 == 0) {
    if (x2 == 0)
      return (1);
    tmp = x1;
    x1 = x2;
    x2 = tmp;
  }
  if (fabs(x1 - x2) / x1 < frac)
    return (1);
  return (0);
}

