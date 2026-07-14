/* file: save_lattice.nl
 * purpose: namelist for saving lattice
 * 
 * Michael Borland, 1991
 */
#include "namelist.h"

#namelist save_lattice static
    STRING filename = NULL;
    long suppress_defaults = 1;
    long output_seq = 0;
    STRING comment = NULL;
#end
