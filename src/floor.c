/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* file: floor.c
 * purpose: computation/output of floor coordinates
 *
 * Michael Borland, 1993
 */
#include "mdb.h"
#include "track.h"
#include "matlib.h"
#include "matrixOp.h"

static SDDS_TABLE SDDS_floor;

#define IC_S 0
#define IC_DS 1
#define IC_X 2
#define IC_Y 3
#define IC_Z 4
#define IC_THETA 5
#define IC_PHI 6
#define IC_PSI 7
#define IC_ELEMENT 8
#define IC_OCCURENCE 9
#define IC_TYPE 10
#define IC_NEXT_ELEMENT 11
#define IC_NEXT_TYPE 12
#ifdef INCLUDE_WIJ
#  define IC_W11 13
#  define N_COLUMNS 22
#else
#  define N_COLUMNS 13
#endif
static SDDS_DEFINITION column_definition[N_COLUMNS] = {
  {"s", "&column name=s, units=m, type=double, description=\"Distance\" &end"},
  {"ds", "&column name=ds, units=m, type=double, description=\"Distance change\" &end"},
  {"X", "&column name=X, units=m, type=double, description=\"Transverse survey coordinate\" &end"},
  {"Y", "&column name=Y, units=m, type=double, description=\"Transverse survey coordinate\" &end"},
  {"Z", "&column name=Z, units=m, type=double, description=\"Longitudinal survey coordinate\" &end"},
  {"theta", "&column name=theta, symbol=\"$gq$r\", units=radians, type=double, description=\"Survey angle\" &end"},
  {"phi", "&column name=phi, symbol=\"$gf$r\", units=radians, type=double, description=\"Survey angle\" &end"},
  {"psi", "&column name=psi, symbol=\"$gy$r\", units=radians, type=double, description=\"Roll angle\" &end"},
  {"ElementName", "&column name=ElementName, type=string, description=\"Element name\", format_string=%10s &end"},
  {"ElementOccurence",
   "&column name=ElementOccurence, type=long, description=\"Occurence of element\", format_string=%6ld &end"},
  {"ElementType", "&column name=ElementType, type=string, description=\"Element-type name\", format_string=%10s &end"},
  {"NextElementName", "&column name=NextElementName, type=string, description=\"Next element name\", format_string=%10s &end"},
  {"NextElementType", "&column name=NextElementType, type=string, description=\"Element-type name for next element\", format_string=%10s &end"},
#ifdef INCLUDE_WIJ
  {"W11", "&column name=W11 type=double &end\n"},
  {"W12", "&column name=W12 type=double &end\n"},
  {"W13", "&column name=W13 type=double &end\n"},
  {"W21", "&column name=W21 type=double &end\n"},
  {"W22", "&column name=W22 type=double &end\n"},
  {"W23", "&column name=W23 type=double &end\n"},
  {"W31", "&column name=W31 type=double &end\n"},
  {"W32", "&column name=W32 type=double &end\n"},
  {"W33", "&column name=W33 type=double &end\n"},
#endif
};

#include "floor.h"

ELEMENT_LIST *bendPreceeds(ELEMENT_LIST *elem);
ELEMENT_LIST *bendFollows(ELEMENT_LIST *elem);
long advanceFloorCoordinates(MATRIX *V1, MATRIX *W1, MATRIX *V0, MATRIX *W0,
                             double *theta, double *phi, double *psi, double *s,
                             ELEMENT_LIST *elem, ELEMENT_LIST *last_elem,
                             SDDS_DATASET *SDDS_floor, long row_index);
void computeSurveyAngles(double *theta, double *phi, double *psi, MATRIX *W, char *name);
double nearbyAngle(double angle, double reference);
void store_vertex_floor_coordinates(char *name, long occurence, double *ve, double *angle);
void store_center_floor_coordinates(char *name, long occurence, double *ve, double *angle);

void setupSurveyAngleMatrix(MATRIX *W0, double theta0, double phi0, double psi0) {
  MATRIX *temp, *Theta, *Phi, *Psi;

  m_alloc(&Theta, 3, 3);
  m_alloc(&Phi, 3, 3);
  m_alloc(&Psi, 3, 3);
  m_zero(Theta);
  m_zero(Phi);
  m_zero(Psi);
  m_alloc(&temp, 3, 3);

  Theta->a[0][0] = Theta->a[2][2] = cos(theta0);
  Theta->a[0][2] = -(Theta->a[2][0] = -sin(theta0));
  Theta->a[1][1] = 1;

  Phi->a[1][1] = Phi->a[2][2] = cos(phi0);
  Phi->a[1][2] = -(Phi->a[2][1] = -sin(phi0));
  Phi->a[0][0] = 1;

  Psi->a[0][0] = Psi->a[1][1] = cos(psi0);
  Psi->a[0][1] = -(Psi->a[1][0] = sin(psi0));
  Psi->a[2][2] = 1;

  m_mult(temp, Theta, Phi);
  m_mult(W0, temp, Psi);

  m_free(&Theta);
  m_free(&Phi);
  m_free(&Psi);
  m_free(&temp);
}

void output_floor_coordinates(NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline) {
  ELEMENT_LIST *elem, *last_elem;
  long n_points, row_index;
  MATRIX *V0, *V1;
  MATRIX *W0, *W1;
  double theta, phi, psi, s;
#ifdef INCLUDE_WIJ
  long iw, jw;
#endif

  log_entry("output_floor_coordinates");

  /* process namelist input */
  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&floor_coordinates, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (echoNamelists)
    print_namelist(stdout, &floor_coordinates);

  if (magnet_centers && vertices_only)
    bombElegant("you can't simultaneously request magnet centers and vertices-only output", NULL);
  if (include_arc_centers && vertices_only)
    bombElegant("you can't simultaneously request arc centers and vertices-only output", NULL);
  if (isMaster) {
    if (filename)
      filename = compose_filename(filename, run->rootname);
    else
      bombElegant("filename must be given for floor coordinates", NULL);

    SDDS_ElegantOutputSetup(&SDDS_floor, filename, SDDS_BINARY, 1, "floor coordinates",
                            run->runfile, run->lattice, NULL, 0,
                            column_definition, N_COLUMNS, "floor coordinates",
                            SDDS_EOS_NEWFILE | SDDS_EOS_COMPLETE);
  }

  n_points = beamline->n_elems + 1;
  if (vertices_only)
    n_points = 2;
  last_elem = NULL;
  if (include_vertices || vertices_only || include_arc_centers) {
    elem = beamline->elem;
    while (elem) {
      switch (elem->type) {
      case T_RBEN:
      case T_SBEN:
      case T_KSBEND:
      case T_CSBEND:
      case T_CSRCSBEND:
      case T_CCBEND:
      case T_BRAT:
	if (include_vertices || vertices_only)
	  n_points++;
	if (include_arc_centers)
	  n_points++;
        break;
      case T_FTABLE:
        if (((FTABLE *)elem->p_elem)->angle)
          n_points++;
        break;
      default:
        break;
      }
      last_elem = elem;
      elem = elem->succ;
    }
  }

  if (isMaster && !SDDS_StartTable(&SDDS_floor, 2 * n_points)) {
    SDDS_SetError("Unable to start SDDS table (output_floor_coordinates)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }

  m_alloc(&V0, 3, 1);
  m_alloc(&V1, 3, 1);
  V0->a[0][0] = X0;
  V0->a[1][0] = Y0;
  V0->a[2][0] = Z0;

  m_alloc(&W0, 3, 3);
  m_alloc(&W1, 3, 3);
  setupSurveyAngleMatrix(W0, theta = theta0, phi = phi0, psi = psi0);

  s = sStart;

  elem = beamline->elem;

  row_index = 0;
  if (isMaster && !SDDS_SetRowValues(&SDDS_floor, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, row_index,
                                     IC_S, (double)s, IC_DS, (double)0.0, IC_X, X0, IC_Y, Y0, IC_Z, Z0,
                                     IC_THETA, theta0, IC_PHI, phi0, IC_PSI, psi0,
                                     IC_ELEMENT, "_BEG_", IC_OCCURENCE, (long)1, IC_TYPE, "MARK",
                                     IC_NEXT_ELEMENT, elem->succ ? elem->succ->name : "_END_", IC_NEXT_TYPE, "MARK",
                                     -1)) {
    SDDS_SetError("Unable to set SDDS row (output_floor_coordinates.0)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
#ifdef INCLUDE_WIJ
  if (isMaster)
    for (iw = 0; iw < 3; iw++)
      for (jw = 0; jw < 3; jw++)
        if (!SDDS_SetRowValues(&SDDS_floor, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, row_index,
                               IC_W11 + iw * 3 + jw, W0->a[iw][jw], -1)) {
          SDDS_SetError("Unable to set SDDS row (output_floor_coordinates.0a)");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
#endif
  row_index++;

  while (elem) {
    if (elem->type == T_STRAY) {
      STRAY *stray;
      stray = (STRAY *)elem->p_elem;
      if (!stray->WiInitialized) {
        m_alloc((MATRIX **)(&(stray->Wi)), 3, 3);
        stray->WiInitialized = 1;
      }
      m_invert((MATRIX *)stray->Wi, W0);
    }
    row_index = advanceFloorCoordinates(V1, W1, V0, W0, &theta, &phi, &psi, &s,
                                        elem, last_elem, &SDDS_floor, row_index);
    m_copy(W0, W1);
    m_copy(V0, V1);
    if (elem->succ == NULL && IS_BEND(elem->type) && include_vertices) {
      /* Get final vertex, if needed */
      row_index = advanceFloorCoordinates(V1, W1, V0, W0, &theta, &phi, &psi, &s,
                                          NULL, last_elem, &SDDS_floor, row_index);
    }
    elem = elem->succ;
  }
  if (isMaster) {
    if (!SDDS_WriteTable(&SDDS_floor)) {
      SDDS_SetError("Unable to write floor coordinate data (output_floor_coordinates)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    if (!inhibitFileSync)
      SDDS_DoFSync(&SDDS_floor);
    if (!SDDS_Terminate(&SDDS_floor)) {
      SDDS_SetError("Unable to terminate SDDS file (output_floor_coordinates)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
  }
  m_free(&V0);
  m_free(&V1);
  m_free(&W0);
  m_free(&W1);
}

long advanceFloorCoordinates(MATRIX *V1, MATRIX *W1, MATRIX *V0, MATRIX *W0,
                             double *theta, double *phi, double *psi, double *s,
                             ELEMENT_LIST *elem, ELEMENT_LIST *last_elem,
                             SDDS_DATASET *SDDS_floor, long row_index) {
  double dX, dY, dZ, rho = 0.0, angle, coord[3], sangle[3], length;
  long is_bend, is_misalignment, is_magnet, is_rotation, i, is_alpha, is_mirror;
  BEND *bend;
  KSBEND *ksbend;
  CSBEND *csbend;
  CCBEND *ccbend;
  MALIGN *malign;
  CSRCSBEND *csrbend;
  ROTATE *rotate;
  ALPH *alpha;
  FTABLE *ftable;
  BRAT *brat;
  LGBEND *lgbend;
  char label[200];
  static MATRIX *temp33, *tempV, *R, *S, *T, *TInv;
  static long matricesAllocated = 0;
  double theta0, phi0, psi0, tilt;
  static VEC *x1, *x2, *x3, *x4;
  static short x1x2Ready = 0;
  static double sVertex = 0, anglesVertex[3];
#ifdef INCLUDE_WIJ
  long iw, jw;
#endif

  if (!isMaster)
    SDDS_floor = NULL;

  if (!matricesAllocated) {
    matricesAllocated = 1;
    m_alloc(&temp33, 3, 3);
    m_alloc(&tempV, 3, 1);
    m_alloc(&R, 3, 1);
    m_alloc(&S, 3, 3);
    m_alloc(&T, 3, 3);
    m_alloc(&TInv, 3, 3);
    /* These are used to determine the vertex point in the general case */
    x1 = vec_get(3);
    x2 = vec_get(3);
    x3 = vec_get(3);
    x4 = vec_get(3);
  }

  is_bend = is_magnet = is_rotation = is_misalignment = is_alpha = is_mirror = 0;
  length = dX = dY = dZ = tilt = angle = 0;
  brat = NULL;
  lgbend = NULL;
  if (elem) {
    switch (elem->type) {
    case T_RBEN:
    case T_SBEN:
      is_bend = 1;
      bend = (BEND *)elem->p_elem;
      angle = bend->angle;
      rho = (length = bend->length) / angle;
      tilt = bend->tilt;
      break;
    case T_KSBEND:
      is_bend = 1;
      ksbend = (KSBEND *)elem->p_elem;
      angle = ksbend->angle;
      rho = (length = ksbend->length) / angle;
      tilt = ksbend->tilt;
      break;
    case T_CSBEND:
      is_bend = 1;
      csbend = (CSBEND *)elem->p_elem;
      angle = csbend->angle;
      rho = (length = csbend->length) / angle;
      tilt = csbend->tilt;
      break;
    case T_CCBEND:
      is_bend = 1;
      ccbend = (CCBEND *)elem->p_elem;
      angle = ccbend->angle;
      rho = (length = ccbend->length) / angle;
      tilt = ccbend->tilt;
      break;
    case T_CSRCSBEND:
      is_bend = 1;
      csrbend = (CSRCSBEND *)elem->p_elem;
      angle = csrbend->angle;
      rho = (length = csrbend->length) / angle;
      tilt = csrbend->tilt;
      break;
    case T_FTABLE:
      ftable = (FTABLE *)elem->p_elem;
      if (ftable->angle) {
        is_bend = 1;
        angle = ftable->angle;
        rho = ftable->l0 / 2. / sin(angle / 2.);
        ftable->arcLength = length = rho * angle;
        tilt = ftable->tilt;
      }
      break;
    case T_MALIGN:
      malign = (MALIGN *)elem->p_elem;
      if (malign->floor) {
        dX = malign->dx;
        dY = malign->dy;
        dZ = malign->dz;
        angle = atan(sqrt(sqr(malign->dxp) + sqr(malign->dyp)));
        tilt = atan2(malign->dyp, -malign->dxp);
        is_misalignment = 1;
      } else {
        dX = dY = dZ = angle = tilt = 0;
        is_misalignment = 0;
      }
      break;
    case T_ALPH:
      is_alpha = 1;
      alpha = (ALPH *)elem->p_elem;
      tilt = alpha->tilt;
      switch (alpha->part) {
      case 1:
        dX = alpha->xmax * sin(ALPHA_ANGLE);
        dZ = alpha->xmax * cos(ALPHA_ANGLE);
        angle = -(ALPHA_ANGLE + PI / 2);
        break;
      case 2:
        dX = alpha->xmax;
        angle = -(ALPHA_ANGLE + PI / 2);
        break;
      case 3:
        dX = 0;
        angle = 0;
        break;
      default:
        angle = -(2 * ALPHA_ANGLE + PI);
        break;
      }
      break;
    case T_ROTATE:
      rotate = (ROTATE *)elem->p_elem;
      if (!rotate->excludeFloor) {
        tilt = rotate->tilt;
        is_rotation = 1;
      } else {
        tilt = 0;
        is_rotation = 0;
      }
      break;
    case T_LMIRROR:
      angle = PI - 2 * (((LMIRROR *)elem->p_elem)->theta);
      tilt = (((LMIRROR *)elem->p_elem)->tilt);
      length = 0;
      is_mirror = 1;
      break;
    case T_BRAT:
      brat = ((BRAT *)elem->p_elem);
      angle = brat->angle;
      break;
    case T_LGBEND:
      is_bend = 1;
      lgbend = (LGBEND *)elem->p_elem;
      length = lgbend->length;
      angle = lgbend->angle;
      tilt = lgbend->tilt;
      break;
    default:
      if (entity_description[elem->type].flags & HAS_LENGTH)
        length = dZ = *((double *)(elem->p_elem));
      if (entity_description[elem->type].flags & IS_MAGNET)
        is_magnet = 1;
      break;
    }
  }

  if (include_arc_centers && SDDS_floor && IS_BEND(elem->type)) {
    /* compute the arc center and save */
    double theta1, phi1, psi1;
    R->a[0][0] = -SIGN(angle)*rho;
    R->a[1][0] = R->a[2][0] = 0;
    m_identity(T);
    T->a[0][0] = T->a[1][1] = cos(tilt);
    T->a[1][0] = -(T->a[0][1] = -sin(tilt));
    m_mult(tempV, T, R);
    m_mult(R, W0, tempV);
    m_add(tempV, V0, R);
    theta1 = *theta;
    phi1 = *phi;
    psi1 = *psi;
    computeSurveyAngles(&theta1, &phi1, &psi1, W0, elem->name);
    sprintf(label, "%s-AC", elem->name);
    if (!SDDS_SetRowValues(SDDS_floor, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, row_index++,
			   IC_S, 0.0, IC_DS, 0.0, IC_X, tempV->a[0][0], IC_Y, tempV->a[1][0], IC_Z, tempV->a[2][0],
			   IC_THETA, theta1, IC_PHI, phi1, IC_PSI, theta1,
			   IC_ELEMENT, label, IC_OCCURENCE, elem->occurence, IC_TYPE,
			   "ARC-CENTER", IC_NEXT_ELEMENT, "", IC_NEXT_TYPE, "",
			   -1)) {
      SDDS_SetError("Unable to set SDDS row (output_floor_coordinates.1)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
  }
  
  if (include_vertices || vertices_only) {
    ELEMENT_LIST *preceeds;

    if (is_bend || brat) {
      /* If this element is a bend, may need to record data for vertex computation */
#ifdef DEBUG
      printf("%s is a bend\n", elem->name);
#endif
      if (elem->pred == NULL || bendPreceeds(elem) == NULL) {
#ifdef DEBUG
        printf("Setting x1 and x2\n");
#endif
        x1->ve[0] = V0->a[0][0];
        x1->ve[1] = V0->a[1][0];
        x1->ve[2] = V0->a[2][0];
        x2->ve[0] = V0->a[0][0] - W0->a[0][2];
        x2->ve[1] = V0->a[1][0] - W0->a[1][2];
        x2->ve[2] = V0->a[2][0] - W0->a[2][2];
        if (elem->pred)
          sVertex = elem->pred->end_pos;
        else
          sVertex = 0;
        anglesVertex[0] = *theta;
        anglesVertex[1] = *phi;
        anglesVertex[2] = *psi;
        x1x2Ready = 1;
      }
    }
    if ((!elem && (IS_BEND(last_elem->type) || last_elem->type == T_BRAT)) || (elem && !is_bend && elem->type != T_MARK && elem->type != T_WATCH && elem->type != T_MAXAMP)) {
#ifdef DEBUG
      printf("preparing vertex output after %s\n", elem ? elem->name : "END");
#endif
      /* If this element is not a bend but previous was a bend and next element is not a bend, have all the required information for vertex computation */
      preceeds = NULL;
      if (!elem || (elem->pred != NULL && (preceeds = bendPreceeds(elem)) != NULL)) {
        if (!preceeds)
          preceeds = last_elem;
        /* See http://mathworld.wolfram.com/Line-LineIntersection.html for an explanation of the method.
           Also Hill, F. S. Jr. "The Pleasures of 'Perp Dot' Products." Ch. II.5 in Graphics Gems IV (Ed. P. S. Heckbert). San Diego: Academic Press, pp. 138-148, 1994. 
           */
        x3->ve[0] = V0->a[0][0];
        x3->ve[1] = V0->a[1][0];
        x3->ve[2] = V0->a[2][0];
        x4->ve[0] = V0->a[0][0] + W0->a[0][2];
        x4->ve[1] = V0->a[1][0] + W0->a[1][2];
        x4->ve[2] = V0->a[2][0] + W0->a[2][2];
        if (!x1x2Ready)
          bombElegant("Problem with x1, x2 preparation for vertex computation", NULL);
        else {
          VEC *a, *b, *c, *cxb, *axb;
          double ds;
#ifdef DEBUG
          printf("*** Computing vertex for %s\n", preceeds->name);
#endif
          x1x2Ready = 0;

          a = vec_get(3);
          b = vec_get(3);
          c = vec_get(3);
          vec_addition(x2, x1, 1, -1, a);
          vec_addition(x4, x3, 1, -1, b);
          vec_addition(x3, x1, 1, -1, c);
          axb = vec_get(3);
          cxb = vec_get(3);
          vec_cross(c, b, cxb);
          vec_cross(a, b, axb);

#ifdef DEBUG
          printf("x1 = (%le, %le, %le)\n", x1->ve[0], x1->ve[1], x1->ve[2]);
          printf("x2 = (%le, %le, %le)\n", x2->ve[0], x2->ve[1], x2->ve[2]);
          printf("x3 = (%le, %le, %le)\n", x3->ve[0], x3->ve[1], x3->ve[2]);
          printf("x4 = (%le, %le, %le)\n", x4->ve[0], x4->ve[1], x4->ve[2]);
          printf("a = (%le, %le, %le)\n", a->ve[0], a->ve[1], a->ve[2]);
          printf("b = (%le, %le, %le)\n", b->ve[0], b->ve[1], b->ve[2]);
          printf("c = (%le, %le, %le)\n", c->ve[0], c->ve[1], c->ve[2]);
          printf("axb = (%le, %le, %le)\n", axb->ve[0], axb->ve[1], axb->ve[2]);
          printf("cxb = (%le, %le, %le)\n", cxb->ve[0], cxb->ve[1], cxb->ve[2]);
#endif

          ds = vec_dot(cxb, axb) / vec_dot(axb, axb);
          vec_addition(x1, a, 1, ds, c);
          sprintf(label, "%s-VP", preceeds->name);
          ds *= -sqrt(vec_dot(a, a));
          if (store_vertices)
            store_vertex_floor_coordinates(preceeds->name, preceeds->occurence, c->ve, anglesVertex);
          if (SDDS_floor) {
            if (!SDDS_SetRowValues(SDDS_floor, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, row_index,
                                   IC_S, sVertex + ds, IC_DS, ds,
                                   IC_X, c->ve[0], IC_Y, c->ve[1], IC_Z, c->ve[2],
                                   IC_THETA, anglesVertex[0], IC_PHI, anglesVertex[1], IC_PSI, anglesVertex[2],
                                   IC_ELEMENT, label, IC_OCCURENCE, preceeds->occurence, IC_TYPE, "VERTEX-POINT",
                                   IC_NEXT_ELEMENT, "", IC_NEXT_TYPE, "",
                                   -1)) {
              SDDS_SetError("Unable to set SDDS row (output_floor_coordinates.1)");
              SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
            }
          }
#ifdef DEBUG
          printf("skewness check (should be 0): %le\n",
                 vec_dot(c, axb));
#endif
          row_index++;
          vec_free(a);
          vec_free(b);
          vec_free(c);
          vec_free(axb);
          vec_free(cxb);
        }
      }
    }
  }
  if (!elem)
    return row_index;
  if (elem->type == T_FLOORELEMENT) {
    /* floor coordinate reset */
    long i;
    FLOORELEMENT *fep;
    fep = (FLOORELEMENT *)(elem->p_elem);
    for (i = 0; i < 3; i++) {
      V1->a[i][0] = coord[i] = fep->position[i];
      sangle[i] = fep->angle[i];
    }
    strcpy_ss(label, elem->name);
    setupSurveyAngleMatrix(W1, fep->angle[0], fep->angle[1], fep->angle[2]);
  } else {
    theta0 = *theta;
    phi0 = *phi;
    psi0 = *psi;
    m_identity(S);
    if ((is_bend || is_mirror) && !lgbend) {
      if (angle && !isnan(rho)) {
        if (!is_mirror) {
          R->a[0][0] = rho * (cos(angle) - 1);
          R->a[1][0] = 0;
          R->a[2][0] = rho * sin(angle);
        } else {
          R->a[0][0] = R->a[1][0] = R->a[2][0] = 0;
        }
        S->a[0][0] = S->a[2][2] = cos(angle);
        S->a[2][0] = -(S->a[0][2] = -sin(angle));
      } else {
        R->a[0][0] = R->a[1][0] = 0;
        R->a[2][0] = length;
      }
    } else if (is_misalignment || is_alpha) {
      R->a[0][0] = dX;
      R->a[1][0] = dY;
      R->a[2][0] = dZ;
      S->a[0][0] = S->a[2][2] = cos(angle);
      S->a[2][0] = -(S->a[0][2] = -sin(angle));
    } else if (brat) {
      double t1;
      if ((angle > 0 && (brat->xVertex - brat->xEntry) < 0) || (angle < 0 && (brat->xVertex - brat->xEntry) > 0))
        bombElegantVA("Error: ANGLE and XVERTEX-XENTRY are inconsistent in sign for BRAT element %s\n", elem->name);
      t1 = atan2(brat->xVertex - brat->xEntry, brat->zVertex - brat->zEntry);
      dZ = cos(t1) * (brat->zExit - brat->zEntry) + sin(t1) * (brat->xExit - brat->xEntry);
      dX = -sin(t1) * (brat->zExit - brat->zEntry) + cos(t1) * (brat->xExit - brat->xEntry);
      R->a[0][0] = dX;
      R->a[1][0] = 0;
      R->a[2][0] = dZ;
      S->a[0][0] = S->a[2][2] = cos(angle);
      S->a[2][0] = -(S->a[0][2] = -sin(angle));
    } else if (lgbend) {
      double t1;
      if ((angle > 0 && (lgbend->xVertex - lgbend->xEntry) < 0) || (angle < 0 && (lgbend->xVertex - lgbend->xEntry) > 0))
        bombElegantVA("Error: ANGLE and XVERTEX-XENTRY are inconsistent in sign for LGBEND element %s\n", elem->name);
      t1 = atan2(lgbend->xVertex - lgbend->xEntry, lgbend->zVertex - lgbend->zEntry);
      dZ = cos(t1) * (lgbend->zExit - lgbend->zEntry) + sin(t1) * (lgbend->xExit - lgbend->xEntry);
      dX = -sin(t1) * (lgbend->zExit - lgbend->zEntry) + cos(t1) * (lgbend->xExit - lgbend->xEntry);
      R->a[0][0] = dX;
      R->a[1][0] = 0;
      R->a[2][0] = dZ;
      S->a[0][0] = S->a[2][2] = cos(angle);
      S->a[2][0] = -(S->a[0][2] = -sin(angle));
    } else {
      R->a[0][0] = R->a[1][0] = 0;
      R->a[2][0] = length;
    }

    if (tilt && !is_rotation) {
      m_identity(T);
      T->a[0][0] = T->a[1][1] = cos(tilt);
      T->a[1][0] = -(T->a[0][1] = -sin(tilt));
      if (!m_mult(tempV, T, R) ||
          !m_copy(R, tempV) ||
          !m_mult(temp33, T, S) ||
          !m_invert(TInv, T) ||
          !m_mult(S, temp33, TInv))
        m_error("making tilt transformation");
    }
    if (is_rotation) {
      m_zero(R);
      m_identity(S);
      S->a[0][0] = S->a[1][1] = cos(tilt);
      S->a[1][0] = -(S->a[0][1] = -sin(tilt));
    }
    m_mult(tempV, W0, R);
    m_add(V1, tempV, V0);
    m_mult(W1, W0, S);
    computeSurveyAngles(theta, phi, psi, W1, elem->name);
    if ((is_bend || is_magnet) && (magnet_centers || store_centers)) {
      for (i = 0; i < 3; i++)
        coord[i] = V0->a[i][0] + (V1->a[i][0] - V0->a[i][0]) / 2;
      sangle[0] = (*theta + theta0) / 2;
      sangle[1] = (*phi + phi0) / 2;
      sangle[2] = (*psi + psi0) / 2;
      if (store_centers)
        store_center_floor_coordinates(elem->name, elem->occurence, coord, sangle);
      if (magnet_centers) {
        sprintf(label, "%s-C", elem->name);
        if (SDDS_floor &&
            (!vertices_only || (!last_elem || elem == last_elem))) {
          if (!SDDS_SetRowValues(SDDS_floor, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, row_index,
                                 IC_S, *s + length / 2, IC_DS, length, IC_X, coord[0], IC_Y, coord[1], IC_Z, coord[2],
                                 IC_THETA, sangle[0], IC_PHI, sangle[1], IC_PSI, sangle[2],
                                 IC_ELEMENT, label, IC_OCCURENCE, elem->occurence, IC_TYPE,
                                 entity_name[elem->type], IC_NEXT_ELEMENT, "", IC_NEXT_TYPE, "",
                                 -1)) {
            SDDS_SetError("Unable to set SDDS row (output_floor_coordinates.2)");
            SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
          }
        }
      }
      row_index++;
    }
    for (i = 0; i < 3; i++)
      coord[i] = V1->a[i][0];
    sangle[0] = *theta;
    sangle[1] = *phi;
    sangle[2] = *psi;
    strcpy_ss(label, elem->name);
  }
  for (i = 0; i < 3; i++) {
    elem->floorCoord[i] = coord[i];
    elem->floorAngle[i] = sangle[i];
  }
  if (SDDS_floor &&
      (!vertices_only || (!last_elem || elem == last_elem))) {
    if (!SDDS_SetRowValues(SDDS_floor, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, row_index,
                           IC_S, *s + length, IC_DS, length, IC_X, coord[0], IC_Y, coord[1], IC_Z, coord[2],
                           IC_THETA, sangle[0], IC_PHI, sangle[1], IC_PSI, sangle[2],
                           IC_ELEMENT, label, IC_OCCURENCE, elem->occurence, IC_TYPE, entity_name[elem->type],
                           IC_NEXT_ELEMENT, elem->succ ? elem->succ->name : "_END_", IC_NEXT_TYPE,
                           elem->succ ? entity_name[elem->succ->type] : "MARK",
                           -1)) {
      SDDS_SetError("Unable to set SDDS row (output_floor_coordinates.2)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
#ifdef INCLUDE_WIJ
    for (iw = 0; iw < 3; iw++)
      for (jw = 0; jw < 3; jw++)
        if (!SDDS_SetRowValues(SDDS_floor, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, row_index,
                               IC_W11 + iw * 3 + jw, W1->a[iw][jw], -1)) {
          SDDS_SetError("Unable to set SDDS row (output_floor_coordinates.2a)");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
#endif
    row_index++;
  }
  *s += length;
  return row_index;
}

void final_floor_coordinates(LINE_LIST *beamline, double *XYZ, double *Angle,
                             double *XYZMin, double *XYZMax) {
  ELEMENT_LIST *elem, *last_elem;
  /* double X, Y, Z; */
  double theta, phi, psi, s;
  MATRIX *V0, *V1;
  MATRIX *Theta, *Phi, *Psi, *W0, *W1, *temp;
  long i;

  elem = beamline->elem;
  /* X = X0; */
  /* Y = Y0; */
  /* Z = Z0; */
  s = 0;

  m_alloc(&V0, 3, 1);
  m_alloc(&V1, 3, 1);
  V0->a[0][0] = X0;
  V0->a[1][0] = Y0;
  V0->a[2][0] = Z0;

  m_alloc(&Theta, 3, 3);
  m_alloc(&Phi, 3, 3);
  m_alloc(&Psi, 3, 3);
  m_zero(Theta);
  m_zero(Phi);
  m_zero(Psi);

  theta = theta0;
  Theta->a[0][0] = Theta->a[2][2] = cos(theta0);
  Theta->a[0][2] = -(Theta->a[2][0] = -sin(theta0));
  Theta->a[1][1] = 1;

  phi = phi0;
  Phi->a[1][1] = Phi->a[2][2] = cos(phi0);
  Phi->a[1][2] = -(Phi->a[2][1] = -sin(phi0));
  Phi->a[0][0] = 1;

  psi = psi0;
  Psi->a[0][0] = Psi->a[1][1] = cos(psi0);
  Psi->a[0][1] = -(Psi->a[1][0] = sin(psi0));
  Psi->a[2][2] = 1;

  m_alloc(&W0, 3, 3);
  m_alloc(&temp, 3, 3);
  m_mult(temp, Theta, Phi);
  m_mult(W0, temp, Psi);
  m_alloc(&W1, 3, 3);

  if (XYZMin) {
    XYZMin[0] = X0;
    XYZMin[1] = Y0;
    XYZMin[2] = Z0;
  }
  if (XYZMax) {
    XYZMax[0] = X0;
    XYZMax[1] = Y0;
    XYZMax[2] = Z0;
  }

  last_elem = NULL;
  while (elem) {
    advanceFloorCoordinates(V1, W1, V0, W0, &theta, &phi, &psi, &s, elem, last_elem, NULL, 0);
    if (elem->type != T_FLOORELEMENT) {
      long i;
      if (XYZMin)
        for (i = 0; i < 3; i++)
          XYZMin[i] = MIN(V1->a[i][0], XYZMin[i]);
      if (XYZMax)
        for (i = 0; i < 3; i++)
          XYZMax[i] = MAX(V1->a[i][0], XYZMax[i]);
    } else {
      long i;
      if (XYZMin)
        for (i = 0; i < 3; i++)
          XYZMin[i] = V1->a[i][0];
      if (XYZMax)
        for (i = 0; i < 3; i++)
          XYZMax[i] = V1->a[i][0];
    }
    if (elem->type == T_MARK && ((MARK *)elem->p_elem)->fitpoint) {
      MARK *mark;
      char t[100];
      static char *suffix[7] = {"X", "Y", "Z", "theta", "phi", "psi", "s"};
      mark = (MARK *)(elem->p_elem);
      if (!(mark->init_flags & 4)) {
        mark->floor_mem = tmalloc(sizeof(*mark->floor_mem) * 7);
        for (i = 0; i < 7; i++) {
          sprintf(t, "%s#%ld.%s", elem->name, elem->occurence,
                  suffix[i]);
          mark->floor_mem[i] = rpn_create_mem(t, 0);
        }
        mark->init_flags |= 4;
      }
      rpn_store(V1->a[0][0], NULL, mark->floor_mem[0]);
      rpn_store(V1->a[1][0], NULL, mark->floor_mem[1]);
      rpn_store(V1->a[2][0], NULL, mark->floor_mem[2]);
      rpn_store(theta, NULL, mark->floor_mem[3]);
      rpn_store(phi, NULL, mark->floor_mem[4]);
      rpn_store(psi, NULL, mark->floor_mem[5]);
      rpn_store(s, NULL, mark->floor_mem[6]);
    }
    m_copy(V0, V1);
    m_copy(W0, W1);
    /* elem->end_pos = s; */
    last_elem = elem;
    elem = elem->succ;
  }
  beamline->revolution_length = s;
  for (i = 0; i < 3; i++)
    XYZ[i] = V0->a[i][0];
  Angle[0] = theta;
  Angle[1] = phi;
  Angle[2] = psi;
  if (elem == NULL && IS_BEND(last_elem->type) && include_vertices) {
    /* Get final vertex, if needed */
    advanceFloorCoordinates(V1, W1, V0, W0, &theta, &phi, &psi, &s,
                            NULL, last_elem, NULL, 0);
  }

  /*
  if (beamline->flags&BEAMLINE_BACKTRACKING && s<0) {
    ELEMENT_LIST *eptr;
    printf("Offseting z coordinates by %le for backtracking\n", -s);
    eptr = beamline->elem;
    do {
      eptr->end_pos -= s;
    } while ((eptr=eptr->succ));
  }
  */

  m_free(&V0);
  m_free(&V1);
  m_free(&Theta);
  m_free(&Phi);
  m_free(&Psi);
  m_free(&W0);
  m_free(&temp);
  m_free(&W1);
}

double nearbyAngle(double angle, double reference) {
  double diff;
  return angle;
  /* double bestAngle, minDiff; */

  if ((diff = reference - angle) > 0)
    return angle + PIx2 * ((long)(diff / PIx2 + 0.5));
  else
    return angle + PIx2 * ((long)(diff / PIx2 - 0.5));

  /*

  bestAngle = angle;
  minDiff = fabs(reference-angle);

  angle += PIx2;
  diff = fabs(reference-angle);
  if (diff<minDiff) {
    minDiff = diff;
    bestAngle = angle;
  }

  angle -= 2*PIx2;
  diff = fabs(reference-angle);
  if (diff<minDiff) {
    minDiff = diff;
    bestAngle = angle;
  }
  return bestAngle;
  */
}

void computeSurveyAngles(double *theta, double *phi, double *psi, MATRIX *W, char *name) {
  double arg;
#ifdef DEBUG
  long iw, jw;
  double Wc[3][3];
#endif

  arg = sqrt(sqr(W->a[1][0]) + sqr(W->a[1][1])); /* |cos(phi)| */
  arg = SIGN(cos(*phi)) * arg;
  *phi = atan2(W->a[1][2], arg);
  if (fabs(arg) > 1e-20) {
    *theta = nearbyAngle(atan2(W->a[0][2], W->a[2][2]), *theta);
    *psi = nearbyAngle(atan2(W->a[1][0], W->a[1][1]), *psi);
  } else {
    *psi = nearbyAngle(atan2(-W->a[0][1], W->a[0][0]) - *theta, *psi);
  }

#ifdef DEBUG
  Wc[0][0] = cos(*psi) * cos(*theta) - sin(*phi) * sin(*psi) * sin(*theta);
  Wc[0][1] = -sin(*phi) * cos(*psi) * sin(*theta) - sin(*psi) * cos(*theta);
  Wc[0][2] = cos(*phi) * sin(*theta);
  Wc[1][0] = cos(*phi) * sin(*psi);
  Wc[1][1] = cos(*phi) * cos(*psi);
  Wc[1][2] = sin(*phi);
  Wc[2][0] = -cos(*psi) * sin(*theta) - sin(*phi) * sin(*psi) * cos(*theta);
  Wc[2][1] = sin(*psi) * sin(*theta) - sin(*phi) * cos(*psi) * cos(*theta);
  Wc[2][2] = cos(*phi) * cos(*theta);

  printf("%s: theta=%13.6e, phi=%13.6e, psi=%13.6e\n", name, *theta, *phi, *psi);
  for (iw = 0; iw < 3; iw++) {
    for (jw = 0; jw < 3; jw++) {
      printf("%13.6e ", W->a[iw][jw]);
    }
    printf(" * ");
    for (jw = 0; jw < 3; jw++) {
      printf("%13.6e ", W->a[iw][jw] - Wc[iw][jw]);
    }
    printf("\n");
  }
#endif
}

ELEMENT_LIST *bendPreceeds(ELEMENT_LIST *elem) {
  ELEMENT_LIST *eptr;
  eptr = elem->pred;
#ifdef DEBUG
  printf("Checking bendPreceeds for %s\n", elem->name);
#endif
  while (eptr) {
    if (IS_BEND(eptr->type) || eptr->type == T_BRAT) {
#ifdef DEBUG
      printf("Returning bendPreceeds=1 for %s\n", elem->name);
#endif
      return eptr;
    }
    if (eptr->type == T_MARK || eptr->type == T_WATCH || eptr->type == T_MAXAMP) {
#ifdef DEBUG
      printf("MARK/WATCH/MAXAMP found for %s---continuing\n", elem->name);
#endif
      eptr = eptr->pred;
    } else {
#ifdef DEBUG
      printf("Returning bendPreceeds=0 for %s (%s)\n", elem->name, eptr->name);
#endif
      return NULL;
    }
  }
#ifdef DEBUG
  printf("Returning bendPreceeds=0 for %s (end of line)\n", elem->name);
#endif
  return NULL;
}

ELEMENT_LIST *bendFollows(ELEMENT_LIST *elem) {
  ELEMENT_LIST *eptr;
  eptr = elem->succ;
#ifdef DEBUG
  printf("Checking bendFollows for %s\n", elem->name);
#endif
  while (eptr) {
    if (IS_BEND(eptr->type) || eptr->type == T_BRAT) {
#ifdef DEBUG
      printf("Returning bendFollows=1 for %s\n", elem->name);
#endif
      return eptr;
    }
    if (eptr->type == T_MARK || eptr->type == T_WATCH || eptr->type == T_MAXAMP) {
#ifdef DEBUG
      printf("MARK/WATCH/MAXAMP found for %s---continuing\n", elem->name);
#endif
      eptr = eptr->succ;
    } else {
#ifdef DEBUG
      printf("Returning bendFollows=0 for %s (%s)\n", elem->name, eptr->name);
#endif
      return NULL;
    }
  }
#ifdef DEBUG
  printf("Returning bendFollows=0 for %s (end of line)\n", elem->name);
#endif
  return NULL;
}

void store_vertex_floor_coordinates(char *name, long occurence, double *ve, double *angle) {
  char *buffer;
  long i;
  char *vTag[3] = {"X", "Y", "Z"};
  char *angleTag[3] = {"theta", "phi", "psi"};

  buffer = tmalloc(sizeof(*name) * (strlen(name) + 1000));
  for (i = 0; i < 3; i++) {
    sprintf(buffer, "%s#%ld-VP.%s", name, occurence, vTag[i]);
    rpn_store(ve[i], NULL, rpn_create_mem(buffer, 0));
    sprintf(buffer, "%s#%ld-VP.%s", name, occurence, angleTag[i]);
    rpn_store(angle[i], NULL, rpn_create_mem(buffer, 0));
  }
}

void store_center_floor_coordinates(char *name, long occurence, double *coord, double *angle) {
  char *buffer;
  long i;
  char *vTag[3] = {"X", "Y", "Z"};
  char *angleTag[3] = {"theta", "phi", "psi"};

  buffer = tmalloc(sizeof(*name) * (strlen(name) + 1000));
  for (i = 0; i < 3; i++) {
    sprintf(buffer, "%s#%ld-C.%s", name, occurence, vTag[i]);
    rpn_store(coord[i], NULL, rpn_create_mem(buffer, 0));
    sprintf(buffer, "%s#%ld-C.%s", name, occurence, angleTag[i]);
    rpn_store(angle[i], NULL, rpn_create_mem(buffer, 0));
  }
}

static double lastGlobalParticleCoordinates[3];
void getLastGlobalParticleCoordinates(double *buffer) {
  memcpy(buffer, lastGlobalParticleCoordinates, 3 * sizeof(double));
}

void convertLocalCoordinatesToGlobal(
  double *Z, double *X, double *Y,
  double *thetaX,
  short mode,
  double *coord,
  ELEMENT_LIST *eptr,
  double dz,                   /* mode==GLOBAL_LOCAL_MODE_DZ: distance to particle z-plane relative to start of element */
  long segment, long nSegments /* mode==GLOBAL_LOCAL_MODE_SEG: distance to z plane is segment/nSegments*length */
) {
  double theta1;
  double dZ, dX, Z1, X1, length;
#ifdef DEBUG
  static FILE *fpdeb = NULL;
#endif
  /* convert (s, x, y, z) coordinates to (Z, X, Y) */
  /* For now, we assume that the beamline is flat ! */

#ifdef DEBUG
  if (fpdeb == NULL) {
    fpdeb = fopen("global.sdds", "w");
    fprintf(fpdeb, "SDDS1\n&column name=Z type=float units=m &end\n");
    fprintf(fpdeb, "&column name=X type=float units=m &end\n");
    fprintf(fpdeb, "&column name=dZ type=float &end\n");
    fprintf(fpdeb, "&column name=thetaX type=float units=m &end\n");
    fprintf(fpdeb, "&column name=x type=float units=m &end\n");
    fprintf(fpdeb, "&column name=xp type=float units=m &end\n");
    fprintf(fpdeb, "&column name=ElementName type=string &end\n");
    fprintf(fpdeb, "&data mode=ascii no_row_counts=1 &end\n");
  }
#endif

  if (mode == GLOBAL_LOCAL_MODE_END) {
    /* We are at the end of the element, so use the floor coordinates there as the reference point */
    X1 = eptr->floorCoord[0];
    Z1 = eptr->floorCoord[2];
    dX = coord[0];
    theta1 = -eptr->floorAngle[0];
    *Z = Z1 + dX * sin(theta1);
    *X = X1 + dX * cos(theta1);
    *Y = coord[2];
    *thetaX = eptr->floorAngle[0] + atan(coord[1]);
    return;
  }

  if (eptr->pred) {
    if (mode==GLOBAL_LOCAL_MODE_DZPM) {
      Z1 = eptr->floorCoord[2];
      X1 = eptr->floorCoord[0];
      theta1 = -eptr->floorAngle[0];
    } else {
      Z1 = eptr->pred->floorCoord[2];
      X1 = eptr->pred->floorCoord[0];
      theta1 = -eptr->pred->floorAngle[0];
    }
  } else {
    Z1 = Z0;
    X1 = X0;
    theta1 = -theta0;
  }

  if (eptr->type == T_LGBEND) {
    LGBEND *lgptr;
    lgptr = (LGBEND *)(eptr->p_elem);
    /* Compute coordinates relative to the entrance point in frame of magnet */
    dX = coord[0] - lgptr->xEntry;
    dZ = dz;
    theta1 += lgptr->segment[0].entryAngle;
    /* Rotate and displace into global coordinate system  */
    *Z = Z1 + dX * sin(theta1) + dZ * cos(theta1);
    *X = X1 + dX * cos(theta1) - dZ * sin(theta1);
    *Y = coord[2];
    *thetaX = -theta1 + atan(coord[1]);
#ifdef DEBUG
    if (!lgptr->optimizeFse || lgptr->optimized == 1) {
      fprintf(fpdeb, "%le %le %le %le  %le %le %s\n",
              *Z, *X, dZ, *thetaX, coord[0], coord[1], eptr->name);
      fflush(fpdeb);
    }
#endif
    if (eptr->end_pos > eptr->beg_pos && (dZ < -1e-6 || (dZ - (eptr->end_pos - eptr->beg_pos)) > 1e-6)) {
#if USE_MPI
      dup2(fdStdout, fileno(stdout));
#endif
      printf("Problem (1) converting to global for particle %ld:\nZ1 = %21.15le, X1 = %21.15le, dZ = %21.15le, dX = %21.15le, theta1 = %21.15le\n -> Z = %21.15le, X = %21.15le\n",
             (long)coord[6], Z1, X1, dZ, dX, theta1, *Z, *X);
      printf("Beg pos: %21.15le, End pos: %21.15le\n", eptr->beg_pos, eptr->end_pos);
      exit(1);
    }
  } else if (eptr->type == T_CCBEND) {
    CCBEND *ccptr;
    ccptr = (CCBEND *)(eptr->p_elem);
    dX = coord[0] - ccptr->dxOffset;
    dZ = dz;
    theta1 += ccptr->angle / 2;
    /* Rotate and displace into global coordinate system  */
    *Z = Z1 + dX * sin(theta1) + dZ * cos(theta1);
    *X = X1 + dX * cos(theta1) - dZ * sin(theta1);
    *Y = coord[2];
    *thetaX = -theta1 + cos(ccptr->extraTilt)*atan(coord[1]);
#ifdef DEBUG
    if ((!ccptr->optimizeFse && !ccptr->optimizeDx) || ccptr->optimized == 1) {
      fprintf(fpdeb, "%le %le %le %le  %le %le %s\n",
              *Z, *X, dZ, *thetaX, coord[0], coord[1], eptr->name);
      fflush(fpdeb);
    }
#endif
    if (eptr->end_pos > eptr->beg_pos && (dZ < -1e-6 || (dZ - (eptr->end_pos - eptr->beg_pos)) > 1e-6)) {
#if USE_MPI
      dup2(fdStdout, fileno(stdout));
#endif
      printf("Problem (2) converting to global for particle %ld:\nZ1 = %21.15le, X1 = %21.15le, dZ = %21.15le, dX = %21.15le, theta1 = %21.15le\n -> Z = %21.15le, X = %21.15le\n",
             (long)coord[6], Z1, X1, dZ, dX, theta1, *Z, *X);
      printf("Beg pos: %21.15le, End pos: %21.15le\n", eptr->beg_pos, eptr->end_pos);
      exit(1);
    }
  } else {
    if (IS_BEND(eptr->type)) {
      double dtheta, angle, rho;
      length = 0;
      angle = 0;
      switch (eptr->type) {
      case T_SBEN:
        length = ((BEND *)eptr->p_elem)->length;
        angle = ((BEND *)eptr->p_elem)->angle;
        break;
      case T_CSBEND:
        length = ((CSBEND *)eptr->p_elem)->length;
        angle = ((CSBEND *)eptr->p_elem)->angle;
        break;
      default:
        bombElegantVA("Error: at present, can't provide global loss coordinates for %s element\n",
                      entity_name[eptr->type]);
        break;
      }
      if (length <= 0)
        bombElegantVA("Error: length<=0 for %s element %s---can't compute global loss coordinates\n",
                      entity_name[eptr->type], eptr->name);
      rho = length / angle;
      /* compute floor coordinate offsets in frame of the magnet (initial trajectory parallel to line X=Y=0 */
      dtheta = 0;
      switch (mode) {
      case GLOBAL_LOCAL_MODE_SEG:
        if (nSegments > 0)
          dtheta = (angle * segment) / nSegments;
        else
          bombElegant("programming error for local-to-global coordinates (1)---seek help from developers", NULL);
        break;
      case GLOBAL_LOCAL_MODE_DZ:
      case GLOBAL_LOCAL_MODE_DZPM:
        dtheta = angle * dz / length;
        break;
      default:
        bombElegant("programming error for local-to-global coordinates (2)---seek help from developers", NULL);
        break;
      }
      dX = (rho + coord[0]) * cos(dtheta) - rho;
      dZ = (rho + coord[0]) * sin(dtheta);

      *Z = Z1 + dX * sin(theta1) + dZ * cos(theta1);
      *X = X1 + dX * cos(theta1) - dZ * sin(theta1);
      *Y = coord[2];
      *thetaX = theta1 - atan(coord[1]);
      /* *thetaX = 888; */
    } else {
      /* Not a simple bending element */
      if (entity_description[eptr->type].flags & HAS_LENGTH)
        length = *((double *)(eptr->p_elem));
      else
        length = 0;
      dZ = 0;
      switch (mode) {
      case GLOBAL_LOCAL_MODE_SEG:
        if (nSegments > 0)
          dZ = (length * segment) / nSegments;
        else
          bombElegant("programming error for local-to-global coordinates (3)---seek help from developers", NULL);
        break;
      case GLOBAL_LOCAL_MODE_DZ:
      case GLOBAL_LOCAL_MODE_DZPM:
        /* someone provides the dz value for us */
        dZ = dz;
        break;
      default:
        bombElegant("programming error for local-to-global coordinates (4)---seek help from developers", NULL);
        break;
      }
      dX = coord[0];
      *Z = Z1 + dX * sin(theta1) + dZ * cos(theta1);
      *X = X1 + dX * cos(theta1) - dZ * sin(theta1);
      *Y = coord[2];
      *thetaX = -theta1 + atan(coord[1]);
      if (eptr->end_pos > eptr->beg_pos && (dZ < -1e-6 || (dZ - (eptr->end_pos - eptr->beg_pos)) > 1e-6) &&
          mode!=GLOBAL_LOCAL_MODE_DZPM) {
#if USE_MPI
        dup2(fdStdout, fileno(stdout));
#endif
        printf("Problem (3) converting to global for particle %ld:\nZ1 = %21.15le, X1 = %21.15le, dZ = %21.15le, dX = %21.15le, theta1 = %21.15le\n -> Z = %21.15le, X = %21.15le\n",
               (long)coord[6], Z1, X1, dZ, dX, theta1, *Z, *X);
        printf("Beg pos: %21.15le, End pos: %21.15le\n", eptr->beg_pos, eptr->end_pos);
        exit(1);
      }
    }
  }
}
