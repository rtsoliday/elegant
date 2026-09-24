/* file: steer_elem.nl 
 * purpose: namelist definition for adding correction elements
 *
 * Michael Borland, 1991
 */
#include "namelist.h"

#namelist steering_element static
    STRING name = NULL;
    STRING element_type = NULL;
    STRING item = NULL;
    STRING plane = "h";
    double tweek = 1e-6;
    double limit = 0;
    long start_occurence = 0;
    long end_occurence = 0;
    long occurence_step = 1;
    double s_start = -1;
    double s_end = -1;
    STRING after = NULL;
    STRING before = NULL;
    STRING target = "correct";
    long verbose = 0;
#end

