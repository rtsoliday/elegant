/* file: sceff.nl
 * purpose: namelist for tracking with space charge effects
 * 
 * Aimin Xiao, 2005
 */
#include "namelist.h"

#namelist insert_sceffects static
        STRING name = NULL;
        STRING type = NULL;
        STRING exclude = NULL;
        long disable = 0;
        long clear = 0;
        STRING element_prefix = (char*)"MYSC";
        long skip = 0;
        long vertical = 0;
        long horizontal = 0;
        long longitudinal = 0;
        long nonlinear = 0;
	long uniform_distribution = 0; 
	double slice_duration = 0;
	long slice_threshold = 100;
	long slice_interpolation = 0;
        long verbosity = 0;
        double averaging_factor = 1;
#end
