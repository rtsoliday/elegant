/* file: alter.nl
 * purpose: namelist for altering element properties
 * 
 * Michael Borland, 1999
 */
#include "namelist.h"

#namelist alter_elements static
        STRING name = NULL;
        STRING item = NULL;
        STRING type = NULL;
        STRING exclude = NULL;
        double value = 0;
        STRING string_value = NULL;
	long disable = 0;
        long differential = 0;
        long multiplicative = 0;
        long alter_at_each_step = 0;
        long alter_before_load_parameters = 0;
        long verbose = 0;
        long allow_missing_elements = 0;
        long allow_missing_parameters = 0;
        long start_occurence = 0;
        long end_occurence = 0;
        long occurence_step = 1;
        double s_start = -1;
        double s_end = -1;
        STRING after = NULL;
        STRING before = NULL;
#end


