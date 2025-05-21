#ifndef _SDDSBRIGHTNESS_

#include <math.h>

#define MAXIMUM_E 10001 /*maximum energy points */
#define MAXIMUM_H 20  /*maximum harmonics */
#define MAXIMUM_K 1000 /*maximum k size for calculating brightness, i.e., 
                         the # of rows in output */
#define minB (1.0E10)  /* minimum brilliance */

/*physical constants */
#undef PI
#define PI   3.141592653589793238462643

/* physical constants from 1988 Particle Properties Data Booklet */
    /* speed of light */
#define c_mks   (2.99792458e8)
    /* unit charge */
#define e_mks   (1.60217733e-19)
    /* electron mass */
#define me_mks (9.1093897e-31)
#define me_mev (0.51099906)
    /* Planck's constant */
#define h_mks (6.6260755e-34)

#define mu0 (1.2566370614e-6) /*Permeability of vacuum [NA-2] */
#define epsilon0 (8.854187817e-12) /*Permittivity of vacuum [Fm-1]*/

#define c_evang (h_mks*c_mks/e_mks*1.0e10)
#define ptot_fac (PI/3.0*e_mks/epsilon0/(me_mev*me_mev)*1.0e6)
#define pd_fac (21.0/(16.0*PI*me_mev*me_mev)*ptot_fac)

#define PLANARK(k) ((k)*(pow(k,6)+24.0*pow((k),4)/7.0+4.0*(k)*(k)+16.0/7.0)/pow((1.0+(k)*(k)),3.5))
#define HELICAK(k) (32.0/7.0*(k)/pow((1.0+(k)*(k)),3.0))

/*method:
  dejus:             Non-zero emittance
                     infinite-N +convolution (Dejus' approach)
  walkerinfinite:    Non-zero emittance;
              i      nfinite-N +convolution (Walker's approach)
  walkerfinite:      Non-zero emittance; finite-N (Walker's)
*/
#define BORLAND 0
#define DEJUS 1
#define WALKER_INF 2
#define WALKER_FIN 3
#define LINDBERG 4
#define METHOD_OPTIONS 5
char *method_option[METHOD_OPTIONS]={
  "borland", "dejus", "walkerinfinite", "walkerfinite", "lindberg",
  };

/*device of type: regular planar or helical */
#define PLANAR 0     
#define HELICAL 1
#define DEVICE_OPTIONS 2
char *device_option[DEVICE_OPTIONS]={
  "planar", "helical"
  };

#endif
