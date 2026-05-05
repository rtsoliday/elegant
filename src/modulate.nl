/* file: error.nl
 * purpose: namelist for random errors
 * 
 * Michael Borland, 1991
 */
#include "namelist.h"

#namelist modulate_elements static
    STRING name = NULL;
    STRING item = NULL;
    STRING type = NULL;
    STRING expression = NULL;
    STRING filename = NULL;
    STRING time_column = NULL;
    long convert_pass_to_time = 0;
    STRING amplitude_column = NULL;
    long refresh_matrix = 0;
    long differential = 1;
    long multiplicative = 0;
    double factor = 1;
    long start_pass = 0;
    long end_pass = LONG_MAX;
    long pass_delay = 0;
    double time_delay = 0;
    long start_occurence = 0;
    long end_occurence = 0;
    double s_start = -1;
    double s_end = -1;
    STRING before = NULL;
    STRING after = NULL;
    long verbose = 0;
    double verbose_threshold = 0;
    STRING record = NULL;
    long flush_record = 1;
#end

