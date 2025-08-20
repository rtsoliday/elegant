#ifndef GSL_FRESNEL_H
#define GSL_FRESNEL_H

#ifdef __cplusplus
extern "C" {
#endif

/* Fresnel integrals C(x) and S(x) */
double fresnel_c(double x);
double fresnel_s(double x);

#ifdef __cplusplus
}
#endif

#endif /* GSL_FRESNEL_H */
