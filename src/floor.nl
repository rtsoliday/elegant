/* file: floor.nl
 * contents: namelist for floor coordinates
 * 
 * Michael Borland, 1993
 */
#include "namelist.h"

#namelist floor_coordinates static
    STRING filename = NULL;
    double X0 = 0.0;
    double Y0 = 0.0;
    double Z0 = 0.0;
    double theta0 = 0.0;
    double phi0 = 0.0;
    double psi0 = 0.0;
    long include_vertices = 0;
    long include_arc_centers = 0;
    long vertices_only = 0;
    long magnet_centers = 0;
    long store_vertices = 0;
    long store_centers = 0;
#end

