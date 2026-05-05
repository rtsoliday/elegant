/* file: particleTunes.nl
 * contents: namelist for particle-by-particle tunes
 * 
 * Michael Borland, 2024
 */
#include "namelist.h"

#namelist particle_tunes
          STRING filename = NULL;
          long start_pid = -1;
          long end_pid = -1;
          long pid_interval = 1;
          short include_x = 1;
          short include_y = 1;
          short include_s = 0;
	  short include_spin = 0;
          long start_pass = 0;
          long segment_length = 0;
#end
