/* file: load_knobs.nl
 * purpose: namelist for loading knob definitions from an SDDS file.
 *
 * Each page of the file defines one knob (page parameter KnobName).
 * The knob is a scalar V; when V is set, each (ElementName,
 * ElementParameter) target listed on the page is offset by V * Factor.
 * The current scalar value is exposed as the RPN variable
 * "<KnobName>.value".  Knobs may be the target of &vary_element,
 * &optimization_variable, and &error_element (the `item` parameter on
 * those namelists is ignored when `name` matches a knob).
 */
#include "namelist.h"

#namelist load_knobs static
    STRING filename = NULL;
    long verbose = 0;
#end
