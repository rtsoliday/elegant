/* file: undulator_brightness.nl
 * purpose: namelist for computing Lindberg undulator brightness as an
 *          RPN scalar (one per invocation, named "<tag>.brightness").
 *
 * The handler only registers the spec and zeros the RPN scalar; the
 * brightness is computed at every Twiss refresh -- end of run_twiss_output
 * and inside the optimization_function loop -- so the value stays current
 * during optimization.
 */
#include "namelist.h"

#namelist undulator_brightness static
    STRING tag = NULL;
    double wavelength = 0;
    double photon_energy = 0;
    long harmonic = 1;
    double detuning = -0.5;
    double period_length = 0;
    long n_periods = 0;
    double total_length = 0;
    double current = 0;
    long use_twiss_output_values = 0;
    double coupling = 0;
    STRING twiss_element = NULL;
    long twiss_occurence = 1;
    double emitx = 0;
    double emity = 0;
    double betax = 0;
    double alphax = 0;
    double betay = 0;
    double alphay = 0;
    double etax = 0;
    double etaxp = 0;
    double etay = 0;
    double etayp = 0;
    double Sdelta = 0;
#end
