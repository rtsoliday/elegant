#include <math.h>
#include "fresnel.h"

/* Simple implementation of Fresnel integrals using series expansion */

#define PI 3.14159265358979323846

/* Fresnel integral C(x) */
double fresnel_c(double x) {
    double x2, x4, term, sum;
    int n;
    
    if (x == 0.0) return 0.0;
    
    x2 = x * x;
    x4 = x2 * x2;
    
    /* For small x, use series expansion */
    if (fabs(x) < 2.0) {
        sum = x;
        term = x;
        for (n = 1; n < 50; n++) {
            term *= -x4 / ((4*n-1) * (4*n) * (2*n-1) * (2*n));
            sum += term;
            if (fabs(term) < 1e-15) break;
        }
        return sum;
    }
    
    /* For larger x, use asymptotic expansion */
    double f = 1.0 / (PI * x2);
    double s = sin(PI * x2 / 2.0);
    double c = cos(PI * x2 / 2.0);
    
    return 0.5 + f * s - f * f * c / 2.0;
}

/* Fresnel integral S(x) */
double fresnel_s(double x) {
    double x2, x4, term, sum;
    int n;
    
    if (x == 0.0) return 0.0;
    
    x2 = x * x;
    x4 = x2 * x2;
    
    /* For small x, use series expansion */
    if (fabs(x) < 2.0) {
        sum = PI * x2 * x / 6.0;
        term = sum;
        for (n = 1; n < 50; n++) {
            term *= -x4 / ((4*n+1) * (4*n+2) * (2*n) * (2*n+1));
            sum += term;
            if (fabs(term) < 1e-15) break;
        }
        return sum;
    }
    
    /* For larger x, use asymptotic expansion */
    double f = 1.0 / (PI * x2);
    double s = sin(PI * x2 / 2.0);
    double c = cos(PI * x2 / 2.0);
    
    return 0.5 - f * c - f * f * s / 2.0;
}
