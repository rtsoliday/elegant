#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <ctype.h>

#include "xraylib.h"
#include "xraylib-parser.h"

/* Define global variables */
#define MAXVAR 16
int   verbose = 0;		/* Print out control */
char  units[17];		/* Units of the output quantities */
char  strVar[MAXVAR][64];	/* Command line arguments in strings */
int   intVar[MAXVAR];		/* Command line arguments in integers */
float floatVar[MAXVAR];		/* Command line arguments in floating numbers */

typedef struct {
  int    Z;
  char   *formula;
  char   *name;
  int    group;  // column
  int    period; // row
  double weight;
  double density;
  double melt;
  double boil;
  double heat;
  double negtivity;
  double abundance;
} MatEntry;
  
/* Element data are from wikipedia */
#define MAX_ENTRY 102
MatEntry materialTable[MAX_ENTRY] = {
/* Z, formula, name,    group, row, mass,    density,        melt,    boil,    heat,  neg,  abundance */
  { 1, "H",   "Hydrogen",    1,  1,  1.00794,    0.00008988,   14.175,  20.28, 14.304,  2.20,  1400  },
  { 2, "He",  "Helium",     18, 1,   4.002602,   0.0001785,     0.0,    4.22,   5.193,  0.0,   0.008 },
  { 3, "Li",  "Lithium",    1,  2,   6.941,      0.534,       453.85,  1615,    3.582,  0.98,  20    },
  { 4, "Be",  "Beryllium",  2,  2,   9.012182,   1.85,       1560.15,  2742,    1.825,  1.57,  2.8   },
  { 5, "B",   "Boron",      13, 2,  10.811,      2.34,       2573.15,  4200,    1.026,  2.04,  10    },
  { 6, "C",   "Diamond",    14, 2,  12.0107,     3.52,       3948.157, 4300,    0.709,  2.55,  200   },
  { 6, "C",   "Graphite",   14, 2,  12.0107,     2.267,      3948.157, 4300,    0.709,  2.55,  200   },
  { 6, "C",   "Carbon",     14, 2,  12.0107,     2.267,      3948.157, 4300,    0.709,  2.55,  200   },
  { 7, "N",   "Nitrogen",   15, 2,  14.0067,     0.0012506,    63.29,    77.36, 1.04,   3.04,  19    },
  { 8, "O",   "Oxygen",     16, 2,  15.9994,     0.001429,     50.5,     90.20, 0.918,  3.44,  461000},
  { 9, "F",   "Fluorine",   17, 2,  18.9984032,  0.001696,    53.63,     85.03, 0.824,  3.98,  585   },
  {10, "Ne",  "Neon",       18, 2,  20.1797,     0.0008999,    24.703,   27.07, 1.03,   0.00,  0.005 },
  {11, "Na",  "Sodium",     1,  3,  22.98976928, 0.971,       371.15,  1156,    1.228,  0.93,  23600 },
  {12, "Mg",  "Magnesium",  2,  3,  24.3050,     1.738,       923.15,  1363,    1.023,  1.31,  23300 },
  {13, "Al",  "Aluminium",  13, 3,  26.9815386,  2.698,       933.4,   2792,    0.897,  1.61,  82300 },
  {14, "Si",  "Silicon",    14, 3,  28.0855,     2.3296,     1683.15,  3538,    0.705,  1.9,   282000},
  {15, "P",   "Phosphorus", 15, 3,  30.973762,   1.82,        317.25,   553,    0.769,  2.19,  1050  },
  {16, "S",   "Sulfur",     16, 3,  32.065,      2.067,       388.51,   717.8,  0.71,   2.58,  350   },
  {17, "Cl",  "Chlorine",   17, 3,  35.453,      0.003214,    172.31,   239.11, 0.479,  3.16,  145   },
  {18, "Ar",  "Argon",      18, 3,  39.948,      0.0017837,    83.96,    87.30, 0.52,   0.0,   3.5   },
  {19, "K",   "Potassium",  1,  4,  39.0983,     0.862,       336.5,   1032,    0.757,  0.82,  20900 },
  {20, "Ca",  "Calcium",    2,  4,  40.078,      1.54,       1112.15,  1757,    0.647,  1,     41500 },
  {21, "Sc",  "Scandium",   3,  4,  44.955912,   2.989,      1812.15,  3109,    0.568,  1.36,  22,   },
  {22, "Ti",  "Titanium",   4,  4,  47.867,      4.54,       1933.15,  3560,    0.523,  1.54,  5650  },
  {23, "V",   "Vanadium",   5,  4,  50.9415,     6.11,       2175.15,  3680,    0.489,  1.63,  120   },
  {24, "Cr",  "Chromium",   6,  4,  51.9961,     7.15,       2130.15,  2944,    0.449,  1.66,  102   },
  {25, "Mn",  "Manganese",  7,  4,  54.938045,   7.44,       1519.15,  2334,    0.479,  1.55,  950   },
  {26, "Fe",  "Iron",       8,  4,  55.845,      7.874,      1808.15,  3134,    0.449,  1.83,  56300 },
  {27, "Co",  "Cobalt",     9,  4,  58.933195,   8.86,       1768.15,  3200,    0.421,  1.88,  25    },
  {28, "Ni",  "Nickel",    10,  4,  58.6934,     8.912,      1726.15,  3186,    0.444,  1.91,  84    },
  {29, "Cu",  "Copper",    11,  4,  63.546,      8.96,       1357.75,  2835,    0.385,  1.9,   60    },
  {30, "Zn",  "Zinc",      12,  4,  65.38,       7.134,       692.88,  1180,    0.388,  1.65,  70    },
  {31, "Ga",  "Gallium",   13,  4,  69.723,      5.907,       302.91,  2477,    0.371,  1.81,  19    },
  {32, "Ge",  "Germanium", 14,  4,  72.64,       5.323,      1211.45,  3106,    0.32,   2.01,  1.5   },
  {33, "As",  "Arsenic",   15,  4,  74.92160,    5.776,      1090.157,  887,    0.329,  2.18,  1.8   },
  {34, "Se",  "Selenium",  16,  4,  78.96,       4.809,       494.15,   958,    0.321,  2.55,  0.05  },
  {35, "Br",  "Bromine",   17,  4,  79.904,      3.122,       266.05,   332.0,  0.474,  2.96,  2.4   },
  {36, "Kr",  "Krypton",   18,  4,  83.798,      0.003733,    115.93,   119.93, 0.248,  3,     0.001 },
  {37, "Rb",  "Rubidium",   1,  5,  85.4678,     1.532,       312.79,   961,    0.363,  0.82,  90    },
  {38, "Sr",  "Strontium",  2,  5,  87.62,       2.64,       1042.15,  1655,    0.301,  0.95,  370   },
  {39, "Y",   "Yttrium",    3,  5,  88.90585,    4.469,      1799.15,  3609,    0.298,  1.22,  33    },
  {40, "Zr",  "Zirconium",  4,  5,  91.224,      6.506,      2125.15,  4682,    0.278,  1.33,  165   },
  {41, "Nb",  "Niobium",    5,  5,  92.90638,    8.57,       2741.15,  5017,    0.265,  1.6,   20    },
  {42, "Mo",  "Molybdenum", 6,  5,  95.96,      10.22,       2890.15,  4912,    0.251,  2.16,  1.2   },
  {43, "Tc",  "Technetium", 7,  5,  98,         11.5,        2473.15,  5150,    0.0,    1.9,   0.001 },
  {44, "Ru",  "Ruthenium",  8,  5, 101.07,      12.37,       2523.15,  4423,    0.238,  2.2,   0.001 },
  {45, "Rh",  "Rhodium",    9,  5, 102.90550,   12.41,       2239.15,  3968,    0.243,  2.28,  0.001 },
  {46, "Pd",  "Palladium", 10,  5, 106.42,      12.02,       1825.15,  3236,    0.244,  2.2,   0.015 },
  {47, "Ag",  "Silver",    11,  5, 107.8682,    10.501,      1234.15,  2435,    0.235,  1.93,  0.075 },
  {48, "Cd",  "Cadmium",   12,  5, 112.411,      8.69,        594.33,  1040,    0.232,  1.69,  0.159 },
  {49, "In",  "Indium",    13,  5, 114.818,      7.31,        429.91,  2345,    0.233,  1.78,  0.25  },
  {50, "Sn",  "Tin",       14,  5, 118.710,      7.287,       505.21,  2875,    0.228,  1.96,  2.3   },
  {51, "Sb",  "Antimony",  15,  5, 121.760,      6.685,       904.05,  1860,    0.207,  2.05,  0.2   },
  {52, "Te",  "Tellurium", 16,  5, 127.60,       6.232,       722.8,   1261,    0.202,  2.1,   0.001 },
  {53, "I",   "Iodine",    17,  5, 126.90447,    4.93,        386.65,   457.4,  0.214,  2.66,  0.45  },
  {54, "Xe",  "Xenon",     18,  5, 131.293,      0.005887,    161.45,   165.03, 0.158,  2.6,   0.001 },
  {55, "Cs",  "Caesium",    1,  6, 132.9054519,  1.873,       301.7,    944,    0.242,  0.79,  3     },
  {56, "Ba",  "Barium",     2,  6, 137.327,      3.594,      1002.15,  2170,    0.204,  0.89,  425   },
  {57, "La",  "Lanthanum",  0,  6, 138.90547,    6.145,      1193.15,  3737,    0.195,  1.1,   39    },
  {58, "Ce",  "Cerium",     0,  6, 140.116,      6.77,       1071.15,  3716,    0.192,  1.12,  66.5  },
  {59, "Pr",  "Praseodymium", 0, 6,140.90765,    6.773,      1204.15,  3793,    0.193,  1.13,  9.2   },
  {60, "Nd",  "Neodymium",  0,  6, 144.242,      7.007,      1289.15,  3347,    0.19,   1.14,  41.5  },
  {61, "Pm",  "Promethium", 0,  6, 145,          7.26,       1204.15,  3273,    0,      0,     0.001 },
  {62, "Sm",  "Samarium",   0,  6, 150.36,       7.52,       1345.15,  2067,    0.197,  1.17,  7.05  },
  {63, "Eu",  "Europium",   0,  6, 151.964,      5.243,      1095.15,  1802,    0.182,  1.2,   2     },
  {64, "Gd",  "Gadolinium", 0,  6, 157.25,       7.895,      1585.15,  3546,    0.236,  1.2,   6.2   },
  {65, "Tb",  "Terbium",    0,  6, 158.92535,    8.229,      1630.15,  3503,    0.182,  1.2,   1.2   },
  {66, "Dy",  "Dysprosium", 0,  6, 162.500,      8.55,       1680.15,  2840,    0.17,   1.22,  5.2   },
  {67, "Ho",  "Holmium",    0,  6, 164.93032,    8.795,      1743.15,  2993,    0.165,  1.23,  1.3   },
  {68, "Er",  "Erbium",     0,  6, 167.259,      9.066,      1795.15,  3503,    0.168,  1.24,  3.5   },
  {69, "Tm",  "Thulium",    0,  6, 168.93421,    9.321,      1818.15,  2223,    0.16,   1.25,  0.52  },
  {70, "Yb",  "Ytterbium",  0,  6, 173.054,      6.965,      1097.15,  1469,    0.155,  1.1,   3.2   },
  {71, "Lu",  "Lutetium",   3,  6, 174.9668,     9.84,       1936.15,  3675,    0.154,  1.27,  0.8   },
  {72, "Hf",  "Hafnium",    4,  6, 178.49,      13.31,       2500.15,  4876,    0.144,  1.3,   3     },
  {73, "Ta",  "Tantalum",   5,  6, 180.94788,   16.654,      3269.15,  5731,    0.14,   1.5,   2     },
  {74, "W",   "Tungsten",   6,  6, 183.84,      19.25,       3680.15,  5828,    0.132,  2.36,  1.3   },
  {75, "Re",  "Rhenium",    7,  6, 186.207,     21.02,       3453.15,  5869,    0.137,  1.9,   0.001 },
  {76, "Os",  "Osmium",     8,  6, 190.23,      22.61,       3300.15,  5285,    0.13,   2.2,   0.002 },
  {77, "Ir",  "Iridium",    9,  6, 192.217,     22.56,       2716.15,  4701,    0.131,  2.2,   0.001 },
  {78, "Pt",  "Platinum",  10,  6, 195.084,     21.46,       2045.15,  4098,    0.133,  2.28,  0.005 },
  {79, "Au",  "Gold",      11,  6, 196.966569,  19.282,      1337.73,  3129,    0.129,  2.54,  0.004 },
  {80, "Hg",  "Mercury",   12,  6, 200.59,      13.5336,      234.43,   630,    0.14,   2,     0.085 },
  {81, "Tl",  "Thallium",  13,  6, 204.3833,    11.85,        577.15,  1746,    0.129,  1.62,  0.85  },
  {82, "Pb",  "Lead",      14,  6, 207.2,       11.342,       600.75,  2022,    0.129,  2.33,  14    },
  {83, "Bi",  "Bismuth",   15,  6, 208.98040,   9.807,        544.67,  1837,    0.122,  2.02,  0.009 },
  {84, "Po",  "Polonium",  16,  6, 210,         9.32,         527.15,  1235,    0,      2,     0.001 },
  {85, "At",  "Astatine",  17,  6, 210,         7,            575.15,  610,     0,      2.2,   0.001 },
  {86, "Rn",  "Radon",     18,  6, 222,         0.00973,      202.15,  211.3,   0.094,  0,     0.001 },
  {87, "Fr",  "Francium",   1,  7, 223,         1.87,         300.15,  950,     0,      0.7,   0.001 },
  {88, "Ra",  "Radium",     2,  7, 226,         5.5,          973.15,  2010,    0,      0.9,   0.001 },
  {89, "Ac",  "Actinium",   0,  7, 227,        10.07,        1323.15,  3471,    0.12,   1.1,   0.001 },
  {90, "Th",  "Thorium",    0,  7, 232.03806,  11.72,        2028.15,  5061,    0.113,  1.3,   9.6   },
  {91, "Pa",  "Protactinium", 0,7, 231.03588,  15.37,        1873.15,  4300,    0,      1.5,   0.001 },
  {92, "U",   "Uranium",    0,  7, 238.02891,  18.95,        1405.15,  4404,    0.116,  1.38,  2.7   },
  {93, "Np",  "Neptunium",  0,  7, 237,        20.45,         913.15,  4273,    0,      1.36,  0.001 },
  {94, "Pu",  "Plutonium",  0,  7, 244,        19.84,         913.15,  3501,    0,      1.28,  0.001 },
  {95, "Am",  "Americium",  0,  7, 243,        13.69,        1267.15,  2880,    0,      1.3,   0     },
  {96, "Cm",  "Curium",     0,  7, 247,        13.51,        1340.15,  3383,    0,      1.3,   0     },
  {97, "Bk",  "Berkelium",  0,  7, 247,        14.79,        1259.15,   983,    0,      1.3,   0     },
  {98, "Cf",  "Californium", 0, 7, 251,        15.1,         1925.15,  1173,    0,      1.3,   0     },
  {99, "Es",  "Einsteinium", 0, 7, 252,        13.5,         1133.15,     0,    0,      1.3,   0     },
  {18, "H2O", "Water",      0,  0,  18.016,    1.00,          271.15,   371.15, 0.0,    0.0,   0.0   },
};
  


/************************************
  getXRLFunctions (command)
************************************/
int getXRLFunctions (char *command)
{
  const char delimiters[] = "( ),";
  char *token, *endp, *saveptr;
  int  k;
  
  // printf( "getXRLFunctions: %s\n", command);
  
  k = 0;
  token = strtok_r (command, delimiters, &saveptr);
  while( token != NULL ) {
    // printf( "token is \"%s\"\n", token );
    snprintf(strVar[k], 64, "%s", token);
    /* Convert the token to number */
    if ( isdigit(token[0]) == 0 ) {
      // printf("%s is not a number.\n", token);
      intVar[k] = 0;
      floatVar[k] = 0;
    } else {
      // printf("%s is a number.\n", token);
      floatVar[k] = strtod(token, &endp);
      intVar[k] = floatVar[k];
    }

    token = strtok_r( NULL, delimiters, &saveptr );
    k++;
  }

  return 0;
}


/************************************
  printXrl 
************************************/
char* getUnits (char *name)
{
  snprintf (units, sizeof(units), " ");
  if(strstr(name, "CSb") == name)          { snprintf (units, sizeof(units), "barn/atom");
  } else if(strstr(name, "DCSb") == name)  { snprintf (units, sizeof(units), "barn/atom/sterad");
  } else if(strstr(name, "DCSPb") == name) { snprintf (units, sizeof(units), "barn/atom/sterad");
  } else if(strcmp(strVar[0], "CS_KN") == 0)   { snprintf (units, sizeof(units), "barn/atom");
  } else if(strcmp(strVar[0], "DCS_KN") == 0)  { snprintf (units, sizeof(units), "barn/atom");
  } else if(strcmp(strVar[0], "DCSP_KN") == 0) { snprintf (units, sizeof(units), "barn/atom");
  } else if(strcmp(strVar[0], "DCS_Thoms") == 0)  { snprintf (units, sizeof(units), "barn/atom");
  } else if(strcmp(strVar[0], "DCSP_Thoms") == 0) { snprintf (units, sizeof(units), "barn/atom");
  } else if(strstr(name, "CS") == name)     { snprintf (units, sizeof(units), "cm2/g");
  } else if(strstr(name, "DCS") == name)    { snprintf (units, sizeof(units), "cm2/g");
  } else if(strstr(name, "DCSP") == name)   { snprintf (units, sizeof(units), "cm2/g");
  } else if(strstr(name, "AtomicWeight") == name)    { snprintf (units, sizeof(units), "AMU");
  } else if(strstr(name, "MomentTransf") != NULL)    { snprintf (units, sizeof(units), "Ang^-1");
  } else if(strstr(name, "Energy") != NULL)          { snprintf (units, sizeof(units), "keV");
  } else if(strstr(name, "AtomicLevelWidth") != NULL){ snprintf (units, sizeof(units), "keV");
  } else if(strcmp(strVar[0], "ElectronConfig") == 0) { snprintf (units, sizeof(units), "electron");
  } else if(strcmp(strVar[0], "FF_Rayl") == 0)  { snprintf (units, sizeof(units), "electron");
  } else if(strcmp(strVar[0], "SF_Compt") == 0) { snprintf (units, sizeof(units), "electron");
  } else if(strcmp(strVar[0], "Fi") == 0)  { snprintf (units, sizeof(units), "electron");
  } else if(strcmp(strVar[0], "Fii") == 0) { snprintf (units, sizeof(units), "electron");
  } else if(strstr(name, "ComptonProfile") != NULL){ snprintf (units, sizeof(units), "electron?");

  }
  return units;
}

int printXrlFloat (float value)
{
  // printf( "xrltest option: verbose = %d\n", verbose);
  if(verbose <= 0)         { printf( "%f\n", value);
  } else if(verbose == 1)  { printf( "%s = %f\n", strVar[0], value);
  } else if(verbose >  1)  { printf( "%s = %f (%s)\n", strVar[0], value, getUnits(strVar[0]) );
  }
  exit (0);
}

int printXrlStr (char *value)
{
  // printf( "xrltest option: verbose = %d\n", verbose);
  if(verbose <= 0)         { printf( "%s\n", value);
  } else if(verbose == 1)  { printf( "%s = %s\n", strVar[0], value);
  } else if(verbose >  1)  { printf( "%s = %s (%s)\n", strVar[0], value, getUnits(strVar[0]) );
  }
  exit (0);
}


/************************************
  getShellID (name)
************************************/
int getShellID (char *name)
{
  int value;
  
  if(strcmp(name, "K") == 0)  { value = K_SHELL;
  } else   if(strcmp(name, "L1") == 0)  { value = L1_SHELL;
  } else   if(strcmp(name, "L2") == 0)  { value = L2_SHELL;
  } else   if(strcmp(name, "L3") == 0)  { value = L3_SHELL;
  } else   if(strcmp(name, "M1") == 0)  { value = M1_SHELL;
  } else   if(strcmp(name, "M2") == 0)  { value = M2_SHELL;
  } else   if(strcmp(name, "M3") == 0)  { value = M3_SHELL;
  } else   if(strcmp(name, "M4") == 0)  { value = M4_SHELL;
  } else   if(strcmp(name, "M5") == 0)  { value = M5_SHELL;
  } else   if(strcmp(name, "N1") == 0)  { value = N1_SHELL;
  } else   if(strcmp(name, "N2") == 0)  { value = N2_SHELL;
  } else   if(strcmp(name, "N3") == 0)  { value = N3_SHELL;
  } else   if(strcmp(name, "N4") == 0)  { value = N4_SHELL;
  } else   if(strcmp(name, "N5") == 0)  { value = N5_SHELL;
  } else   if(strcmp(name, "N6") == 0)  { value = N6_SHELL;
  } else   if(strcmp(name, "N7") == 0)  { value = N7_SHELL;
  } else   if(strcmp(name, "O1") == 0)  { value = O1_SHELL;
  } else   if(strcmp(name, "O2") == 0)  { value = O2_SHELL;
  } else   if(strcmp(name, "O3") == 0)  { value = O3_SHELL;
  } else   if(strcmp(name, "O4") == 0)  { value = O4_SHELL;
  } else   if(strcmp(name, "O5") == 0)  { value = O5_SHELL;
  } else   if(strcmp(name, "O6") == 0)  { value = O6_SHELL;
  } else   if(strcmp(name, "O7") == 0)  { value = O7_SHELL;
  } else   if(strcmp(name, "P1") == 0)  { value = P1_SHELL;
  } else   if(strcmp(name, "P2") == 0)  { value = P2_SHELL;
  } else   if(strcmp(name, "P3") == 0)  { value = P3_SHELL;
  } else   if(strcmp(name, "P4") == 0)  { value = P4_SHELL;
  } else   if(strcmp(name, "P5") == 0)  { value = P5_SHELL;
  } else { value = 0;
  }

  return value;
}



/************************************
  getMatPropertyFloat (name)
************************************/
double getMatPropertyFloat (char *label, char *property)
{
  int    i, test=-1;
  double value=0.0;

  /* Search for the entry */
  for (i = 0; i < MAX_ENTRY; i++) {
    test = strcmp(label, materialTable[i].name);
    if (test == 0) break;
    test = strcmp(label, materialTable[i].formula);
    if (test == 0) break;
  }
  if (test != 0) return -1.0;  /* faile to find the match */
  
  value = materialTable[i].Z;
  if(strcmp(property, "Z") == 0         || strcmp(property, "z") == 0)         { value = materialTable[i].Z; };
  if(strcmp(property, "Group") == 0     || strcmp(property, "group") == 0)     { value = materialTable[i].group; };
  if(strcmp(property, "Period") == 0    || strcmp(property, "period") == 0)    { value = materialTable[i].period; };
  if(strcmp(property, "Weight") == 0    || strcmp(property, "weight") == 0)    { value = materialTable[i].weight; };
  if(strcmp(property, "Density") == 0   || strcmp(property, "density") == 0)   { value = materialTable[i].density; };
  if(strcmp(property, "Melt") == 0      || strcmp(property, "melt") == 0)      { value = materialTable[i].melt; };
  if(strcmp(property, "Boil") == 0      || strcmp(property, "boil") == 0)      { value = materialTable[i].boil; };
  if(strcmp(property, "Heat") == 0      || strcmp(property, "heat") == 0)      { value = materialTable[i].heat; };
  if(strcmp(property, "Negtivity") == 0 || strcmp(property, "negtivity") == 0) { value = materialTable[i].negtivity; };
  if(strcmp(property, "Abundance") == 0 || strcmp(property, "abundance") == 0) { value = materialTable[i].abundance; };
   
  // printf("i = %d, label = %s, property = %s, value = %f\n", i, label, property, value);
  return value;
}

/************************************
  getMatPropertyString (name)
************************************/
char * getMatPropertyString (char *label, char *property)
{
  static char outputString[1024];  
  char   tmpString[512];
  int    i, test=-1;

  /* Default: Output string = input string. Use tmpString to allow using same string for input and output */
  strcpy( tmpString, label );
  strcpy( outputString, tmpString );

  /* Search for the entry */
  for (i = 0; i < MAX_ENTRY; i++) {
    test = strcmp(label, materialTable[i].name);
    if (test == 0) break;
    test = strcmp(label, materialTable[i].formula);
    if (test == 0) break;
  }
  if (test != 0) { return outputString; } 	/* faile to find the match, return input string as output */
  
  if(strcmp(property, "Formula") == 0 || strcmp(property, "formula") == 0) { strcpy( outputString,  materialTable[i].formula); }
  if(strcmp(property, "Name") == 0    || strcmp(property, "name") == 0)    { strcpy( outputString,  materialTable[i].name); }
  return outputString;
}


/************************************
  getLineID (name)
************************************/
int getLineID (char *name)
{
  int value;
  
  if(strcmp(name, "KA") == 0)  { value = KA_LINE;
  } else   if(strcmp(name, "KB") == 0)  { value = KB_LINE;
  } else   if(strcmp(name, "LA") == 0)  { value = LA_LINE;
  } else   if(strcmp(name, "LB") == 0)  { value = LB_LINE;
  /* Single lines */
  } else   if(strcmp(name, "KA1") == 0) { value = KA1_LINE;
  } else   if(strcmp(name, "KA2") == 0) { value = KA2_LINE;
  } else   if(strcmp(name, "KB1") == 0) { value = KB1_LINE;
  } else   if(strcmp(name, "KB2") == 0) { value = KB2_LINE;
  } else   if(strcmp(name, "KB3") == 0) { value = KB3_LINE;
  } else   if(strcmp(name, "KB4") == 0) { value = KB4_LINE;
  } else   if(strcmp(name, "KB5") == 0) { value = KB5_LINE;
  } else   if(strcmp(name, "LA1") == 0) { value = LA1_LINE;
  } else   if(strcmp(name, "LA2") == 0) { value = LA2_LINE;
  } else   if(strcmp(name, "LB1") == 0) { value = LB1_LINE;
  } else   if(strcmp(name, "LB2") == 0) { value = LB2_LINE;
  } else   if(strcmp(name, "LB3") == 0) { value = LB3_LINE;
  } else   if(strcmp(name, "LB4") == 0) { value = LB4_LINE;
  } else   if(strcmp(name, "LB5") == 0) { value = LB5_LINE;
  } else   if(strcmp(name, "LB6") == 0) { value = LB6_LINE;
  } else   if(strcmp(name, "LB7") == 0) { value = LB7_LINE;
  } else   if(strcmp(name, "LB9") == 0) { value = LB9_LINE;
  } else   if(strcmp(name, "LB10") == 0){ value = LB10_LINE;
  } else   if(strcmp(name, "LB15") == 0){ value = LB15_LINE;
  } else   if(strcmp(name, "LB17") == 0){ value = LB17_LINE;
  } else   if(strcmp(name, "LG1") == 0) { value = LG1_LINE;
  } else   if(strcmp(name, "LG2") == 0) { value = LG2_LINE;
  } else   if(strcmp(name, "LG3") == 0) { value = LG3_LINE;
  } else   if(strcmp(name, "LG4") == 0) { value = LG4_LINE;
  } else   if(strcmp(name, "LG5") == 0) { value = LG5_LINE;
  } else   if(strcmp(name, "LG6") == 0) { value = LG6_LINE;
  } else   if(strcmp(name, "LG8") == 0) { value = LG8_LINE;
  } else   if(strcmp(name, "LE") == 0) { value = LE_LINE;
  } else   if(strcmp(name, "LL") == 0) { value = LL_LINE;
  } else   if(strcmp(name, "LS") == 0) { value = LS_LINE;
  } else   if(strcmp(name, "LT") == 0) { value = LT_LINE;
  } else   if(strcmp(name, "LU") == 0) { value = LU_LINE;
  } else   if(strcmp(name, "LV") == 0) { value = LV_LINE;
  } else   if(strcmp(name, "MA1") == 0){ value = MA1_LINE;
  } else   if(strcmp(name, "MA2") == 0){ value = MA2_LINE;
  } else   if(strcmp(name, "MB") == 0) { value = MB_LINE;
  } else   if(strcmp(name, "MG") == 0) { value = MG_LINE;
  } else   if(strcmp(name, "KL1") == 0)  { value = KL1_LINE;
  } else   if(strcmp(name, "KL2") == 0)  { value = KL2_LINE;
  } else   if(strcmp(name, "KL3") == 0)  { value = KL3_LINE;
  } else   if(strcmp(name, "KM1") == 0)  { value = KM1_LINE;
  } else   if(strcmp(name, "KM2") == 0)  { value = KM2_LINE;
  } else   if(strcmp(name, "KM3") == 0)  { value = KM3_LINE;
  } else   if(strcmp(name, "KM4") == 0)  { value = KM4_LINE;
  } else   if(strcmp(name, "KM5") == 0)  { value = KM5_LINE;
  } else   if(strcmp(name, "KN1") == 0)  { value = KN1_LINE;
  } else   if(strcmp(name, "KN2") == 0)  { value = KN2_LINE;
  } else   if(strcmp(name, "KN3") == 0)  { value = KN3_LINE;
  } else   if(strcmp(name, "KN4") == 0)  { value = KN4_LINE;
  } else   if(strcmp(name, "KN5") == 0)  { value = KN5_LINE;
  } else   if(strcmp(name, "KN6") == 0)  { value = KN6_LINE;
  } else   if(strcmp(name, "KN7") == 0)  { value = KN7_LINE;
  } else   if(strcmp(name, "KO1") == 0)  { value = KO1_LINE;
  } else   if(strcmp(name, "KO2") == 0)  { value = KO2_LINE;
  } else   if(strcmp(name, "KO3") == 0)  { value = KO3_LINE;
  } else   if(strcmp(name, "KO4") == 0)  { value = KO4_LINE;
  } else   if(strcmp(name, "KO5") == 0)  { value = KO5_LINE;
  } else   if(strcmp(name, "KO6") == 0)  { value = KO6_LINE;
  } else   if(strcmp(name, "KO7") == 0)  { value = KO7_LINE;
  } else   if(strcmp(name, "KP1") == 0)  { value = KP1_LINE;
  } else   if(strcmp(name, "KP2") == 0)  { value = KP2_LINE;
  } else   if(strcmp(name, "KP3") == 0)  { value = KP3_LINE;
  } else   if(strcmp(name, "KP4") == 0)  { value = KP4_LINE;
  } else   if(strcmp(name, "KP5") == 0)  { value = KP5_LINE;
  } else   if(strcmp(name, "L1L2") == 0)  { value = L1L2_LINE;
  } else   if(strcmp(name, "L1L3") == 0)  { value = L1L3_LINE;
  } else   if(strcmp(name, "L1M1") == 0)  { value = L1M1_LINE;
  } else   if(strcmp(name, "L1M2") == 0)  { value = L1M2_LINE;
  } else   if(strcmp(name, "L1M3") == 0)  { value = L1M3_LINE;
  } else   if(strcmp(name, "L1M4") == 0)  { value = L1M4_LINE;
  } else   if(strcmp(name, "L1M5") == 0)  { value = L1M5_LINE;
  } else   if(strcmp(name, "L1N1") == 0)  { value = L1N1_LINE;
  } else   if(strcmp(name, "L1N2") == 0)  { value = L1N2_LINE;
  } else   if(strcmp(name, "L1N3") == 0)  { value = L1N3_LINE;
  } else   if(strcmp(name, "L1N4") == 0)  { value = L1N4_LINE;
  } else   if(strcmp(name, "L1N5") == 0)  { value = L1N5_LINE;
  } else   if(strcmp(name, "L1N6") == 0)  { value = L1N6_LINE;
  } else   if(strcmp(name, "L1N7") == 0)  { value = L1N7_LINE;
  } else   if(strcmp(name, "L1N67") == 0) { value = L1N67_LINE;
  } else   if(strcmp(name, "L1O1") == 0)  { value = L1O1_LINE;
  } else   if(strcmp(name, "L1O2") == 0)  { value = L1O2_LINE;
  } else   if(strcmp(name, "L1O3") == 0)  { value = L1O3_LINE;
  } else   if(strcmp(name, "L1O4") == 0)  { value = L1O4_LINE;
  } else   if(strcmp(name, "L1O45") == 0) { value = L1O45_LINE;
  } else   if(strcmp(name, "L1O5") == 0)  { value = L1O5_LINE;
  } else   if(strcmp(name, "L1O6") == 0)  { value = L1O6_LINE;
  } else   if(strcmp(name, "L1O7") == 0)  { value = L1O7_LINE;
  } else   if(strcmp(name, "L1P1") == 0)  { value = L1P1_LINE;
  } else   if(strcmp(name, "L1P2") == 0)  { value = L1P2_LINE;
  } else   if(strcmp(name, "L1P23") == 0) { value = L1P23_LINE;
  } else   if(strcmp(name, "L1P3") == 0)  { value = L1P3_LINE;
  } else   if(strcmp(name, "L1P4") == 0)  { value = L1P4_LINE;
  } else   if(strcmp(name, "L1P5") == 0)  { value = L1P5_LINE;
  } else   if(strcmp(name, "L2L3") == 0)  { value = L2L3_LINE;
  } else   if(strcmp(name, "L2M1") == 0)  { value = L2M1_LINE;
  } else   if(strcmp(name, "L2M2") == 0)  { value = L2M2_LINE;
  } else   if(strcmp(name, "L2M3") == 0)  { value = L2M3_LINE;
  } else   if(strcmp(name, "L2M4") == 0)  { value = L2M4_LINE;
  } else   if(strcmp(name, "L2M5") == 0)  { value = L2M5_LINE;
  } else   if(strcmp(name, "L2N1") == 0)  { value = L2N1_LINE;
  } else   if(strcmp(name, "L2N2") == 0)  { value = L2N2_LINE;
  } else   if(strcmp(name, "L2N3") == 0)  { value = L2N3_LINE;
  } else   if(strcmp(name, "L2N4") == 0)  { value = L2N4_LINE;
  } else   if(strcmp(name, "L2N5") == 0)  { value = L2N5_LINE;
  } else   if(strcmp(name, "L2N6") == 0)  { value = L2N6_LINE;
  } else   if(strcmp(name, "L2N7") == 0)  { value = L2N7_LINE;
  } else   if(strcmp(name, "L2O1") == 0)  { value = L2O1_LINE;
  } else   if(strcmp(name, "L2O2") == 0)  { value = L2O2_LINE;
  } else   if(strcmp(name, "L2O3") == 0)  { value = L2O3_LINE;
  } else   if(strcmp(name, "L2O4") == 0)  { value = L2O4_LINE;
  } else   if(strcmp(name, "L2O5") == 0)  { value = L2O5_LINE;
  } else   if(strcmp(name, "L2O6") == 0)  { value = L2O6_LINE;
  } else   if(strcmp(name, "L2O7") == 0)  { value = L2O7_LINE;
  } else   if(strcmp(name, "L2P1") == 0)  { value = L2P1_LINE;
  } else   if(strcmp(name, "L2P2") == 0)  { value = L2P2_LINE;
  } else   if(strcmp(name, "L2P23") == 0) { value = L2P23_LINE;
  } else   if(strcmp(name, "L2P3") == 0)  { value = L2P3_LINE;
  } else   if(strcmp(name, "L2P4") == 0)  { value = L2P4_LINE;
  } else   if(strcmp(name, "L2P5") == 0)  { value = L2P5_LINE;
  } else   if(strcmp(name, "L2Q1") == 0)  { value = L2Q1_LINE;
  } else   if(strcmp(name, "L3M1") == 0)  { value = L3M1_LINE;
  } else   if(strcmp(name, "L3M2") == 0)  { value = L3M2_LINE;
  } else   if(strcmp(name, "L3M3") == 0)  { value = L3M3_LINE;
  } else   if(strcmp(name, "L3M4") == 0)  { value = L3M4_LINE;
  } else   if(strcmp(name, "L3M5") == 0)  { value = L3M5_LINE;
  } else   if(strcmp(name, "L3N1") == 0)  { value = L3N1_LINE;
  } else   if(strcmp(name, "L3N2") == 0)  { value = L3N2_LINE;
  } else   if(strcmp(name, "L3N3") == 0)  { value = L3N3_LINE;
  } else   if(strcmp(name, "L3N4") == 0)  { value = L3N4_LINE;
  } else   if(strcmp(name, "L3N5") == 0)  { value = L3N5_LINE;
  } else   if(strcmp(name, "L3N6") == 0)  { value = L3N6_LINE;
  } else   if(strcmp(name, "L3N7") == 0)  { value = L3N7_LINE;
  } else   if(strcmp(name, "L3O1") == 0)  { value = L3O1_LINE;
  } else   if(strcmp(name, "L3O2") == 0)  { value = L3O2_LINE;
  } else   if(strcmp(name, "L3O3") == 0)  { value = L3O3_LINE;
  } else   if(strcmp(name, "L3O4") == 0)  { value = L3O4_LINE;
  } else   if(strcmp(name, "L3O45") == 0) { value = L3O45_LINE;
  } else   if(strcmp(name, "L3O5") == 0)  { value = L3O5_LINE;
  } else   if(strcmp(name, "L3O6") == 0)  { value = L3O6_LINE;
  } else   if(strcmp(name, "L3O7") == 0)  { value = L3O7_LINE;
  } else   if(strcmp(name, "L3P1") == 0)  { value = L3P1_LINE;
  } else   if(strcmp(name, "L3P2") == 0)  { value = L3P2_LINE;
  } else   if(strcmp(name, "L3P23") == 0) { value = L3P23_LINE;
  } else   if(strcmp(name, "L3P3") == 0)  { value = L3P3_LINE;
  } else   if(strcmp(name, "L3P4") == 0)  { value = L3P4_LINE;
  } else   if(strcmp(name, "L3P45") == 0) { value = L3P45_LINE;
  } else   if(strcmp(name, "L3P5") == 0)  { value = L3P5_LINE;
  } else   if(strcmp(name, "L3Q1") == 0)  { value = L3Q1_LINE;
  } else   if(strcmp(name, "M1M2") == 0)  { value = M1M2_LINE;
  } else   if(strcmp(name, "M1M3") == 0)  { value = M1M3_LINE;
  } else   if(strcmp(name, "M1M4") == 0)  { value = M1M4_LINE;
  } else   if(strcmp(name, "M1M5") == 0)  { value = M1M5_LINE;
  } else   if(strcmp(name, "M1N1") == 0)  { value = M1N1_LINE;
  } else   if(strcmp(name, "M1N2") == 0)  { value = M1N2_LINE;
  } else   if(strcmp(name, "M1N3") == 0)  { value = M1N3_LINE;
  } else   if(strcmp(name, "M1N4") == 0)  { value = M1N4_LINE;
  } else   if(strcmp(name, "M1N5") == 0)  { value = M1N5_LINE;
  } else   if(strcmp(name, "M1N6") == 0)  { value = M1N6_LINE;
  } else   if(strcmp(name, "M1N7") == 0)  { value = M1N7_LINE;
  } else   if(strcmp(name, "M1O1") == 0)  { value = M1O1_LINE;
  } else   if(strcmp(name, "M1O2") == 0)  { value = M1O2_LINE;
  } else   if(strcmp(name, "M1O3") == 0)  { value = M1O3_LINE;
  } else   if(strcmp(name, "M1O4") == 0)  { value = M1O4_LINE;
  } else   if(strcmp(name, "M1O5") == 0)  { value = M1O5_LINE;
  } else   if(strcmp(name, "M1O6") == 0)  { value = M1O6_LINE;
  } else   if(strcmp(name, "M1O7") == 0)  { value = M1O7_LINE;
  } else   if(strcmp(name, "M1P1") == 0)  { value = M1P1_LINE;
  } else   if(strcmp(name, "M1P2") == 0)  { value = M1P2_LINE;
  } else   if(strcmp(name, "M1P3") == 0)  { value = M1P3_LINE;
  } else   if(strcmp(name, "M1P4") == 0)  { value = M1P4_LINE;
  } else   if(strcmp(name, "M1P5") == 0)  { value = M1P5_LINE;
  } else   if(strcmp(name, "M2M3") == 0)  { value = M2M3_LINE;
  } else   if(strcmp(name, "M2M4") == 0)  { value = M2M4_LINE;
  } else   if(strcmp(name, "M2M5") == 0)  { value = M2M5_LINE;
  } else   if(strcmp(name, "M2N1") == 0)  { value = M2N1_LINE;
  } else   if(strcmp(name, "M2N2") == 0)  { value = M2N2_LINE;
  } else   if(strcmp(name, "M2N3") == 0)  { value = M2N3_LINE;
  } else   if(strcmp(name, "M2N4") == 0)  { value = M2N4_LINE;
  } else   if(strcmp(name, "M2N5") == 0)  { value = M2N5_LINE;
  } else   if(strcmp(name, "M2N6") == 0)  { value = M2N6_LINE;
  } else   if(strcmp(name, "M2N7") == 0)  { value = M2N7_LINE;
  } else   if(strcmp(name, "M2O1") == 0)  { value = M2O1_LINE;
  } else   if(strcmp(name, "M2O2") == 0)  { value = M2O2_LINE;
  } else   if(strcmp(name, "M2O3") == 0)  { value = M2O3_LINE;
  } else   if(strcmp(name, "M2O4") == 0)  { value = M2O4_LINE;
  } else   if(strcmp(name, "M2O5") == 0)  { value = M2O5_LINE;
  } else   if(strcmp(name, "M2O6") == 0)  { value = M2O6_LINE;
  } else   if(strcmp(name, "M2O7") == 0)  { value = M2O7_LINE;
  } else   if(strcmp(name, "M2P1") == 0)  { value = M2P1_LINE;
  } else   if(strcmp(name, "M2P2") == 0)  { value = M2P2_LINE;
  } else   if(strcmp(name, "M2P3") == 0)  { value = M2P3_LINE;
  } else   if(strcmp(name, "M2P4") == 0)  { value = M2P4_LINE;
  } else   if(strcmp(name, "M2P5") == 0)  { value = M2P5_LINE;
  } else   if(strcmp(name, "M3M4") == 0)  { value = M3M4_LINE;
  } else   if(strcmp(name, "M3M5") == 0)  { value = M3M5_LINE;
  } else   if(strcmp(name, "M3N1") == 0)  { value = M3N1_LINE;
  } else   if(strcmp(name, "M3N2") == 0)  { value = M3N2_LINE;
  } else   if(strcmp(name, "M3N3") == 0)  { value = M3N3_LINE;
  } else   if(strcmp(name, "M3N4") == 0)  { value = M3N4_LINE;
  } else   if(strcmp(name, "M3N5") == 0)  { value = M3N5_LINE;
  } else   if(strcmp(name, "M3N6") == 0)  { value = M3N6_LINE;
  } else   if(strcmp(name, "M3N7") == 0)  { value = M3N7_LINE;
  } else   if(strcmp(name, "M3O1") == 0)  { value = M3O1_LINE;
  } else   if(strcmp(name, "M3O2") == 0)  { value = M3O2_LINE;
  } else   if(strcmp(name, "M3O3") == 0)  { value = M3O3_LINE;
  } else   if(strcmp(name, "M3O4") == 0)  { value = M3O4_LINE;
  } else   if(strcmp(name, "M3O5") == 0)  { value = M3O5_LINE;
  } else   if(strcmp(name, "M3O6") == 0)  { value = M3O6_LINE;
  } else   if(strcmp(name, "M3O7") == 0)  { value = M3O7_LINE;
  } else   if(strcmp(name, "M3P1") == 0)  { value = M3P1_LINE;
  } else   if(strcmp(name, "M3P2") == 0)  { value = M3P2_LINE;
  } else   if(strcmp(name, "M3P3") == 0)  { value = M3P3_LINE;
  } else   if(strcmp(name, "M3P4") == 0)  { value = M3P4_LINE;
  } else   if(strcmp(name, "M3P5") == 0)  { value = M3P5_LINE;
  } else   if(strcmp(name, "M3Q1") == 0)  { value = M3Q1_LINE;
  } else   if(strcmp(name, "M4M5") == 0)  { value = M4M5_LINE;
  } else   if(strcmp(name, "M4N1") == 0)  { value = M4N1_LINE;
  } else   if(strcmp(name, "M4N2") == 0)  { value = M4N2_LINE;
  } else   if(strcmp(name, "M4N3") == 0)  { value = M4N3_LINE;
  } else   if(strcmp(name, "M4N4") == 0)  { value = M4N4_LINE;
  } else   if(strcmp(name, "M4N5") == 0)  { value = M4N5_LINE;
  } else   if(strcmp(name, "M4N6") == 0)  { value = M4N6_LINE;
  } else   if(strcmp(name, "M4N7") == 0)  { value = M4N7_LINE;
  } else   if(strcmp(name, "M4O1") == 0)  { value = M4O1_LINE;
  } else   if(strcmp(name, "M4O2") == 0)  { value = M4O2_LINE;
  } else   if(strcmp(name, "M4O3") == 0)  { value = M4O3_LINE;
  } else   if(strcmp(name, "M4O4") == 0)  { value = M4O4_LINE;
  } else   if(strcmp(name, "M4O5") == 0)  { value = M4O5_LINE;
  } else   if(strcmp(name, "M4O6") == 0)  { value = M4O6_LINE;
  } else   if(strcmp(name, "M4O7") == 0)  { value = M4O7_LINE;
  } else   if(strcmp(name, "M4P1") == 0)  { value = M4P1_LINE;
  } else   if(strcmp(name, "M4P2") == 0)  { value = M4P2_LINE;
  } else   if(strcmp(name, "M4P3") == 0)  { value = M4P3_LINE;
  } else   if(strcmp(name, "M4P4") == 0)  { value = M4P4_LINE;
  } else   if(strcmp(name, "M4P5") == 0)  { value = M4P5_LINE;
  } else   if(strcmp(name, "M5N1") == 0)  { value = M5N1_LINE;
  } else   if(strcmp(name, "M5N2") == 0)  { value = M5N2_LINE;
  } else   if(strcmp(name, "M5N3") == 0)  { value = M5N3_LINE;
  } else   if(strcmp(name, "M5N4") == 0)  { value = M5N4_LINE;
  } else   if(strcmp(name, "M5N5") == 0)  { value = M5N5_LINE;
  } else   if(strcmp(name, "M5N6") == 0)  { value = M5N6_LINE;
  } else   if(strcmp(name, "M5N7") == 0)  { value = M5N7_LINE;
  } else   if(strcmp(name, "M5O1") == 0)  { value = M5O1_LINE;
  } else   if(strcmp(name, "M5O2") == 0)  { value = M5O2_LINE;
  } else   if(strcmp(name, "M5O3") == 0)  { value = M5O3_LINE;
  } else   if(strcmp(name, "M5O4") == 0)  { value = M5O4_LINE;
  } else   if(strcmp(name, "M5O5") == 0)  { value = M5O5_LINE;
  } else   if(strcmp(name, "M5O6") == 0)  { value = M5O6_LINE;
  } else   if(strcmp(name, "M5O7") == 0)  { value = M5O7_LINE;
  } else   if(strcmp(name, "M5P1") == 0)  { value = M5P1_LINE;
  } else   if(strcmp(name, "M5P2") == 0)  { value = M5P2_LINE;
  } else   if(strcmp(name, "M5P3") == 0)  { value = M5P3_LINE;
  } else   if(strcmp(name, "M5P4") == 0)  { value = M5P4_LINE;
  } else   if(strcmp(name, "M5P5") == 0)  { value = M5P5_LINE;
  } else   if(strcmp(name, "N1N2") == 0)  { value = N1N2_LINE;
  } else   if(strcmp(name, "N1N3") == 0)  { value = N1N3_LINE;
  } else   if(strcmp(name, "N1N4") == 0)  { value = N1N4_LINE;
  } else   if(strcmp(name, "N1N5") == 0)  { value = N1N5_LINE;
  } else   if(strcmp(name, "N1N6") == 0)  { value = N1N6_LINE;
  } else   if(strcmp(name, "N1N7") == 0)  { value = N1N7_LINE;
  } else   if(strcmp(name, "N1O1") == 0)  { value = N1O1_LINE;
  } else   if(strcmp(name, "N1O2") == 0)  { value = N1O2_LINE;
  } else   if(strcmp(name, "N1O3") == 0)  { value = N1O3_LINE;
  } else   if(strcmp(name, "N1O4") == 0)  { value = N1O4_LINE;
  } else   if(strcmp(name, "N1O5") == 0)  { value = N1O5_LINE;
  } else   if(strcmp(name, "N1O6") == 0)  { value = N1O6_LINE;
  } else   if(strcmp(name, "N1O7") == 0)  { value = N1O7_LINE;
  } else   if(strcmp(name, "N1P1") == 0)  { value = N1P1_LINE;
  } else   if(strcmp(name, "N1P2") == 0)  { value = N1P2_LINE;
  } else   if(strcmp(name, "N1P3") == 0)  { value = N1P3_LINE;
  } else   if(strcmp(name, "N1P4") == 0)  { value = N1P4_LINE;
  } else   if(strcmp(name, "N1P5") == 0)  { value = N1P5_LINE;
  } else   if(strcmp(name, "N2N3") == 0)  { value = N2N3_LINE;
  } else   if(strcmp(name, "N2N4") == 0)  { value = N2N4_LINE;
  } else   if(strcmp(name, "N2N5") == 0)  { value = N2N5_LINE;
  } else   if(strcmp(name, "N2N6") == 0)  { value = N2N6_LINE;
  } else   if(strcmp(name, "N2N7") == 0)  { value = N2N7_LINE;
  } else   if(strcmp(name, "N2O1") == 0)  { value = N2O1_LINE;
  } else   if(strcmp(name, "N2O2") == 0)  { value = N2O2_LINE;
  } else   if(strcmp(name, "N2O3") == 0)  { value = N2O3_LINE;
  } else   if(strcmp(name, "N2O4") == 0)  { value = N2O4_LINE;
  } else   if(strcmp(name, "N2O5") == 0)  { value = N2O5_LINE;
  } else   if(strcmp(name, "N2O6") == 0)  { value = N2O6_LINE;
  } else   if(strcmp(name, "N2O7") == 0)  { value = N2O7_LINE;
  } else   if(strcmp(name, "N2P1") == 0)  { value = N2P1_LINE;
  } else   if(strcmp(name, "N2P2") == 0)  { value = N2P2_LINE;
  } else   if(strcmp(name, "N2P3") == 0)  { value = N2P3_LINE;
  } else   if(strcmp(name, "N2P4") == 0)  { value = N2P4_LINE;
  } else   if(strcmp(name, "N2P5") == 0)  { value = N2P5_LINE;
  } else   if(strcmp(name, "N3N4") == 0)  { value = N3N4_LINE;
  } else   if(strcmp(name, "N3N5") == 0)  { value = N3N5_LINE;
  } else   if(strcmp(name, "N3N6") == 0)  { value = N3N6_LINE;
  } else   if(strcmp(name, "N3N7") == 0)  { value = N3N7_LINE;
  } else   if(strcmp(name, "N3O1") == 0)  { value = N3O1_LINE;
  } else   if(strcmp(name, "N3O2") == 0)  { value = N3O2_LINE;
  } else   if(strcmp(name, "N3O3") == 0)  { value = N3O3_LINE;
  } else   if(strcmp(name, "N3O4") == 0)  { value = N3O4_LINE;
  } else   if(strcmp(name, "N3O5") == 0)  { value = N3O5_LINE;
  } else   if(strcmp(name, "N3O6") == 0)  { value = N3O6_LINE;
  } else   if(strcmp(name, "N3O7") == 0)  { value = N3O7_LINE;
  } else   if(strcmp(name, "N3P1") == 0)  { value = N3P1_LINE;
  } else   if(strcmp(name, "N3P2") == 0)  { value = N3P2_LINE;
  } else   if(strcmp(name, "N3P3") == 0)  { value = N3P3_LINE;
  } else   if(strcmp(name, "N3P4") == 0)  { value = N3P4_LINE;
  } else   if(strcmp(name, "N3P5") == 0)  { value = N3P5_LINE;
  } else   if(strcmp(name, "N4N5") == 0)  { value = N4N5_LINE;
  } else   if(strcmp(name, "N4N6") == 0)  { value = N4N6_LINE;
  } else   if(strcmp(name, "N4N7") == 0)  { value = N4N7_LINE;
  } else   if(strcmp(name, "N4O1") == 0)  { value = N4O1_LINE;
  } else   if(strcmp(name, "N4O2") == 0)  { value = N4O2_LINE;
  } else   if(strcmp(name, "N4O3") == 0)  { value = N4O3_LINE;
  } else   if(strcmp(name, "N4O4") == 0)  { value = N4O4_LINE;
  } else   if(strcmp(name, "N4O5") == 0)  { value = N4O5_LINE;
  } else   if(strcmp(name, "N4O6") == 0)  { value = N4O6_LINE;
  } else   if(strcmp(name, "N4O7") == 0)  { value = N4O7_LINE;
  } else   if(strcmp(name, "N4P1") == 0)  { value = N4P1_LINE;
  } else   if(strcmp(name, "N4P2") == 0)  { value = N4P2_LINE;
  } else   if(strcmp(name, "N4P3") == 0)  { value = N4P3_LINE;
  } else   if(strcmp(name, "N4P4") == 0)  { value = N4P4_LINE;
  } else   if(strcmp(name, "N4P5") == 0)  { value = N4P5_LINE;
  } else   if(strcmp(name, "N5N6") == 0)  { value = N5N6_LINE;
  } else   if(strcmp(name, "N5N7") == 0)  { value = N5N7_LINE;
  } else   if(strcmp(name, "N5O1") == 0)  { value = N5O1_LINE;
  } else   if(strcmp(name, "N5O2") == 0)  { value = N5O2_LINE;
  } else   if(strcmp(name, "N5O3") == 0)  { value = N5O3_LINE;
  } else   if(strcmp(name, "N5O4") == 0)  { value = N5O4_LINE;
  } else   if(strcmp(name, "N5O5") == 0)  { value = N5O5_LINE;
  } else   if(strcmp(name, "N5O6") == 0)  { value = N5O6_LINE;
  } else   if(strcmp(name, "N5O7") == 0)  { value = N5O7_LINE;
  } else   if(strcmp(name, "N5P1") == 0)  { value = N5P1_LINE;
  } else   if(strcmp(name, "N5P2") == 0)  { value = N5P2_LINE;
  } else   if(strcmp(name, "N5P3") == 0)  { value = N5P3_LINE;
  } else   if(strcmp(name, "N5P4") == 0)  { value = N5P4_LINE;
  } else   if(strcmp(name, "N5P5") == 0)  { value = N5P5_LINE;
  } else   if(strcmp(name, "N6N7") == 0)  { value = N6N7_LINE;
  } else   if(strcmp(name, "N6O1") == 0)  { value = N6O1_LINE;
  } else   if(strcmp(name, "N6O2") == 0)  { value = N6O2_LINE;
  } else   if(strcmp(name, "N6O3") == 0)  { value = N6O3_LINE;
  } else   if(strcmp(name, "N6O4") == 0)  { value = N6O4_LINE;
  } else   if(strcmp(name, "N6O5") == 0)  { value = N6O5_LINE;
  } else   if(strcmp(name, "N6O6") == 0)  { value = N6O6_LINE;
  } else   if(strcmp(name, "N6O7") == 0)  { value = N6O7_LINE;
  } else   if(strcmp(name, "N6P1") == 0)  { value = N6P1_LINE;
  } else   if(strcmp(name, "N6P2") == 0)  { value = N6P2_LINE;
  } else   if(strcmp(name, "N6P3") == 0)  { value = N6P3_LINE;
  } else   if(strcmp(name, "N6P4") == 0)  { value = N6P4_LINE;
  } else   if(strcmp(name, "N6P5") == 0)  { value = N6P5_LINE;
  } else   if(strcmp(name, "N7O1") == 0)  { value = N7O1_LINE;
  } else   if(strcmp(name, "N7O2") == 0)  { value = N7O2_LINE;
  } else   if(strcmp(name, "N7O3") == 0)  { value = N7O3_LINE;
  } else   if(strcmp(name, "N7O4") == 0)  { value = N7O4_LINE;
  } else   if(strcmp(name, "N7O5") == 0)  { value = N7O5_LINE;
  } else   if(strcmp(name, "N7O6") == 0)  { value = N7O6_LINE;
  } else   if(strcmp(name, "N7O7") == 0)  { value = N7O7_LINE;
  } else   if(strcmp(name, "N7P1") == 0)  { value = N7P1_LINE;
  } else   if(strcmp(name, "N7P2") == 0)  { value = N7P2_LINE;
  } else   if(strcmp(name, "N7P3") == 0)  { value = N7P3_LINE;
  } else   if(strcmp(name, "N7P4") == 0)  { value = N7P4_LINE;
  } else   if(strcmp(name, "N7P5") == 0)  { value = N7P5_LINE;
  } else { value = 0;
  }

  return value;
}


/************************************
  getAugerTransID (name)
************************************/
int getAugerTransID (char *name)
{
  int value;
  
  if(strcmp(name, "K_L1L1") == 0)  { value = K_L1L1_AUGER;
  } else   if(strcmp(name, "K_L1L2") == 0)  { value = K_L1L2_AUGER;
  } else   if(strcmp(name, "K_L1L3") == 0)  { value = K_L1L3_AUGER;
  } else   if(strcmp(name, "K_L1M1") == 0)  { value = K_L1M1_AUGER;
  } else   if(strcmp(name, "K_L1M2") == 0)  { value = K_L1M2_AUGER;
  } else   if(strcmp(name, "K_L1M3") == 0)  { value = K_L1M3_AUGER;
  } else   if(strcmp(name, "K_L1M4") == 0)  { value = K_L1M4_AUGER;
  } else   if(strcmp(name, "K_L1M5") == 0)  { value = K_L1M5_AUGER;
  } else   if(strcmp(name, "K_L2L1") == 0)  { value = K_L2L1_AUGER;
  } else   if(strcmp(name, "K_L2L2") == 0)  { value = K_L2L2_AUGER;
  } else   if(strcmp(name, "K_L2L3") == 0)  { value = K_L2L3_AUGER;
  } else   if(strcmp(name, "K_L2M1") == 0)  { value = K_L2M1_AUGER;
  } else   if(strcmp(name, "K_L2M2") == 0)  { value = K_L2M2_AUGER;
  } else   if(strcmp(name, "K_L2M3") == 0)  { value = K_L2M3_AUGER;
  } else   if(strcmp(name, "K_L2M4") == 0)  { value = K_L2M4_AUGER;
  } else   if(strcmp(name, "K_L2M5") == 0)  { value = K_L2M5_AUGER;
  } else   if(strcmp(name, "K_L3L1") == 0)  { value = K_L3L1_AUGER;
  } else   if(strcmp(name, "K_L3L2") == 0)  { value = K_L3L2_AUGER;
  } else   if(strcmp(name, "K_L3L3") == 0)  { value = K_L3L3_AUGER;
  } else   if(strcmp(name, "K_L3M1") == 0)  { value = K_L3M1_AUGER;
  } else   if(strcmp(name, "K_L3M2") == 0)  { value = K_L3M2_AUGER;
  } else   if(strcmp(name, "K_L3M3") == 0)  { value = K_L3M3_AUGER;
  } else   if(strcmp(name, "K_L3M4") == 0)  { value = K_L3M4_AUGER;
  } else   if(strcmp(name, "K_L3M5") == 0)  { value = K_L3M5_AUGER;
  } else   if(strcmp(name, "K_M1L1") == 0)  { value = K_M1L1_AUGER;
  } else   if(strcmp(name, "K_M1L2") == 0)  { value = K_M1L2_AUGER;
  } else   if(strcmp(name, "K_M1L3") == 0)  { value = K_M1L3_AUGER;
  } else   if(strcmp(name, "K_M1M1") == 0)  { value = K_M1M1_AUGER;
  } else   if(strcmp(name, "K_M1M2") == 0)  { value = K_M1M2_AUGER;
  } else   if(strcmp(name, "K_M1M3") == 0)  { value = K_M1M3_AUGER;
  } else   if(strcmp(name, "K_M1M4") == 0)  { value = K_M1M4_AUGER;
  } else   if(strcmp(name, "K_M1M5") == 0)  { value = K_M1M5_AUGER;
  } else   if(strcmp(name, "K_M2L1") == 0)  { value = K_M2L1_AUGER;
  } else   if(strcmp(name, "K_M2L2") == 0)  { value = K_M2L2_AUGER;
  } else   if(strcmp(name, "K_M2L3") == 0)  { value = K_M2L3_AUGER;
  } else   if(strcmp(name, "K_M2M1") == 0)  { value = K_M2M1_AUGER;
  } else   if(strcmp(name, "K_M2M2") == 0)  { value = K_M2M2_AUGER;
  } else   if(strcmp(name, "K_M2M3") == 0)  { value = K_M2M3_AUGER;
  } else   if(strcmp(name, "K_M2M4") == 0)  { value = K_M2M4_AUGER;
  } else   if(strcmp(name, "K_M2M5") == 0)  { value = K_M2M5_AUGER;
  } else   if(strcmp(name, "K_M3L1") == 0)  { value = K_M3L1_AUGER;
  } else   if(strcmp(name, "K_M3L2") == 0)  { value = K_M3L2_AUGER;
  } else   if(strcmp(name, "K_M3L3") == 0)  { value = K_M3L3_AUGER;
  } else   if(strcmp(name, "K_M3M1") == 0)  { value = K_M3M1_AUGER;
  } else   if(strcmp(name, "K_M3M2") == 0)  { value = K_M3M2_AUGER;
  } else   if(strcmp(name, "K_M3M3") == 0)  { value = K_M3M3_AUGER;
  } else   if(strcmp(name, "K_M3M4") == 0)  { value = K_M3M4_AUGER;
  } else   if(strcmp(name, "K_M3M5") == 0)  { value = K_M3M5_AUGER;
  } else   if(strcmp(name, "K_M4L1") == 0)  { value = K_M4L1_AUGER;
  } else   if(strcmp(name, "K_M4L2") == 0)  { value = K_M4L2_AUGER;
  } else   if(strcmp(name, "K_M4L3") == 0)  { value = K_M4L3_AUGER;
  } else   if(strcmp(name, "K_M4M1") == 0)  { value = K_M4M1_AUGER;
  } else   if(strcmp(name, "K_M4M2") == 0)  { value = K_M4M2_AUGER;
  } else   if(strcmp(name, "K_M4M3") == 0)  { value = K_M4M3_AUGER;
  } else   if(strcmp(name, "K_M4M4") == 0)  { value = K_M4M4_AUGER;
  } else   if(strcmp(name, "K_M4M5") == 0)  { value = K_M4M5_AUGER;
  } else   if(strcmp(name, "K_M5L1") == 0)  { value = K_M5L1_AUGER;
  } else   if(strcmp(name, "K_M5L2") == 0)  { value = K_M5L2_AUGER;
  } else   if(strcmp(name, "K_M5L3") == 0)  { value = K_M5L3_AUGER;
  } else   if(strcmp(name, "K_M5M1") == 0)  { value = K_M5M1_AUGER;
  } else   if(strcmp(name, "K_M5M2") == 0)  { value = K_M5M2_AUGER;
  } else   if(strcmp(name, "K_M5M3") == 0)  { value = K_M5M3_AUGER;
  } else   if(strcmp(name, "K_M5M4") == 0)  { value = K_M5M4_AUGER;
  } else   if(strcmp(name, "K_M5M5") == 0)  { value = K_M5M5_AUGER;
  } else   if(strcmp(name, "L1_L2L2") == 0)  { value = L1_L2L2_AUGER;
  } else   if(strcmp(name, "L1_L2L3") == 0)  { value = L1_L2L3_AUGER;
  } else   if(strcmp(name, "L1_L2M1") == 0)  { value = L1_L2M1_AUGER;
  } else   if(strcmp(name, "L1_L2M2") == 0)  { value = L1_L2M2_AUGER;
  } else   if(strcmp(name, "L1_L2M3") == 0)  { value = L1_L2M3_AUGER;
  } else   if(strcmp(name, "L1_L2M4") == 0)  { value = L1_L2M4_AUGER;
  } else   if(strcmp(name, "L1_L2M5") == 0)  { value = L1_L2M5_AUGER;
  } else   if(strcmp(name, "L1_L3L2") == 0)  { value = L1_L3L2_AUGER;
  } else   if(strcmp(name, "L1_L3L3") == 0)  { value = L1_L3L3_AUGER;
  } else   if(strcmp(name, "L1_L3M1") == 0)  { value = L1_L3M1_AUGER;
  } else   if(strcmp(name, "L1_L3M2") == 0)  { value = L1_L3M2_AUGER;
  } else   if(strcmp(name, "L1_L3M3") == 0)  { value = L1_L3M3_AUGER;
  } else   if(strcmp(name, "L1_L3M4") == 0)  { value = L1_L3M4_AUGER;
  } else   if(strcmp(name, "L1_L3M5") == 0)  { value = L1_L3M5_AUGER;
  } else   if(strcmp(name, "L1_M1L2") == 0)  { value = L1_M1L2_AUGER;
  } else   if(strcmp(name, "L1_M1L3") == 0)  { value = L1_M1L3_AUGER;
  } else   if(strcmp(name, "L1_M1M1") == 0)  { value = L1_M1M1_AUGER;
  } else   if(strcmp(name, "L1_M1M2") == 0)  { value = L1_M1M2_AUGER;
  } else   if(strcmp(name, "L1_M1M3") == 0)  { value = L1_M1M3_AUGER;
  } else   if(strcmp(name, "L1_M2M4") == 0)  { value = L1_M1M4_AUGER;
  } else   if(strcmp(name, "L1_M2M5") == 0)  { value = L1_M1M5_AUGER;
  } else   if(strcmp(name, "L1_M2L2") == 0)  { value = L1_M2L2_AUGER;
  } else   if(strcmp(name, "L1_M2L3") == 0)  { value = L1_M2L3_AUGER;
  } else   if(strcmp(name, "L1_M2M1") == 0)  { value = L1_M2M1_AUGER;
  } else   if(strcmp(name, "L1_M2M2") == 0)  { value = L1_M2M2_AUGER;
  } else   if(strcmp(name, "L1_M2M3") == 0)  { value = L1_M2M3_AUGER;
  } else   if(strcmp(name, "L1_M2M4") == 0)  { value = L1_M2M4_AUGER;
  } else   if(strcmp(name, "L1_M2M5") == 0)  { value = L1_M2M5_AUGER;
  } else   if(strcmp(name, "L1_M3L2") == 0)  { value = L1_M3L2_AUGER;
  } else   if(strcmp(name, "L1_M3L3") == 0)  { value = L1_M3L3_AUGER;
  } else   if(strcmp(name, "L1_M3M1") == 0)  { value = L1_M3M1_AUGER;
  } else   if(strcmp(name, "L1_M3M2") == 0)  { value = L1_M3M2_AUGER;
  } else   if(strcmp(name, "L1_M3M3") == 0)  { value = L1_M3M3_AUGER;
  } else   if(strcmp(name, "L1_M3M4") == 0)  { value = L1_M3M4_AUGER;
  } else   if(strcmp(name, "L1_M3M5") == 0)  { value = L1_M3M5_AUGER;
  } else   if(strcmp(name, "L1_M4L2") == 0)  { value = L1_M4L2_AUGER;
  } else   if(strcmp(name, "L1_M4L3") == 0)  { value = L1_M4L3_AUGER;
  } else   if(strcmp(name, "L1_M4M1") == 0)  { value = L1_M4M1_AUGER;
  } else   if(strcmp(name, "L1_M4M2") == 0)  { value = L1_M4M2_AUGER;
  } else   if(strcmp(name, "L1_M4M3") == 0)  { value = L1_M4M3_AUGER;
  } else   if(strcmp(name, "L1_M4M4") == 0)  { value = L1_M4M4_AUGER;
  } else   if(strcmp(name, "L1_M4M5") == 0)  { value = L1_M4M5_AUGER;
  } else   if(strcmp(name, "L1_M5L2") == 0)  { value = L1_M5L2_AUGER;
  } else   if(strcmp(name, "L1_M5L3") == 0)  { value = L1_M5L3_AUGER;
  } else   if(strcmp(name, "L1_M5M1") == 0)  { value = L1_M5M1_AUGER;
  } else   if(strcmp(name, "L1_M5M2") == 0)  { value = L1_M5M2_AUGER;
  } else   if(strcmp(name, "L1_M5M3") == 0)  { value = L1_M5M3_AUGER;
  } else   if(strcmp(name, "L1_M5M4") == 0)  { value = L1_M5M4_AUGER;
  } else   if(strcmp(name, "L1_M5M5") == 0)  { value = L1_M5M5_AUGER;
  } else   if(strcmp(name, "L2_L3L3") == 0)  { value = L2_L3L3_AUGER;
  } else   if(strcmp(name, "L2_L3M1") == 0)  { value = L2_L3M1_AUGER;
  } else   if(strcmp(name, "L2_L3M2") == 0)  { value = L2_L3M2_AUGER;
  } else   if(strcmp(name, "L2_L3M3") == 0)  { value = L2_L3M3_AUGER;
  } else   if(strcmp(name, "L2_L3M4") == 0)  { value = L2_L3M4_AUGER;
  } else   if(strcmp(name, "L2_L3M5") == 0)  { value = L2_L3M5_AUGER;
  } else   if(strcmp(name, "L2_M1L3") == 0)  { value = L2_M1L3_AUGER;
  } else   if(strcmp(name, "L2_M1M1") == 0)  { value = L2_M1M1_AUGER;
  } else   if(strcmp(name, "L2_M1M2") == 0)  { value = L2_M1M2_AUGER;
  } else   if(strcmp(name, "L2_M1M3") == 0)  { value = L2_M1M3_AUGER;
  } else   if(strcmp(name, "L2_M1M4") == 0)  { value = L2_M1M4_AUGER;
  } else   if(strcmp(name, "L2_M1M5") == 0)  { value = L2_M1M5_AUGER;
  } else   if(strcmp(name, "L2_M2L3") == 0)  { value = L2_M2L3_AUGER;
  } else   if(strcmp(name, "L2_M2M1") == 0)  { value = L2_M2M1_AUGER;
  } else   if(strcmp(name, "L2_M2M2") == 0)  { value = L2_M2M2_AUGER;
  } else   if(strcmp(name, "L2_M2M3") == 0)  { value = L2_M2M3_AUGER;
  } else   if(strcmp(name, "L2_M2M4") == 0)  { value = L2_M2M4_AUGER;
  } else   if(strcmp(name, "L2_M2M5") == 0)  { value = L2_M2M5_AUGER;
  } else   if(strcmp(name, "L2_M3L3") == 0)  { value = L2_M3L3_AUGER;
  } else   if(strcmp(name, "L2_M3M1") == 0)  { value = L2_M3M1_AUGER;
  } else   if(strcmp(name, "L2_M3M2") == 0)  { value = L2_M3M2_AUGER;
  } else   if(strcmp(name, "L2_M3M3") == 0)  { value = L2_M3M3_AUGER;
  } else   if(strcmp(name, "L2_M3M4") == 0)  { value = L2_M3M4_AUGER;
  } else   if(strcmp(name, "L2_M3M5") == 0)  { value = L2_M3M5_AUGER;
  } else   if(strcmp(name, "L2_M4L3") == 0)  { value = L2_M4L3_AUGER;
  } else   if(strcmp(name, "L2_M4M1") == 0)  { value = L2_M4M1_AUGER;
  } else   if(strcmp(name, "L2_M4M2") == 0)  { value = L2_M4M2_AUGER;
  } else   if(strcmp(name, "L2_M4M3") == 0)  { value = L2_M4M3_AUGER;
  } else   if(strcmp(name, "L2_M4M4") == 0)  { value = L2_M4M4_AUGER;
  } else   if(strcmp(name, "L2_M4M5") == 0)  { value = L2_M4M5_AUGER;
  } else   if(strcmp(name, "L2_M5L3") == 0)  { value = L2_M5L3_AUGER;
  } else   if(strcmp(name, "L2_M5M1") == 0)  { value = L2_M5M1_AUGER;
  } else   if(strcmp(name, "L2_M5M2") == 0)  { value = L2_M5M2_AUGER;
  } else   if(strcmp(name, "L2_M5M3") == 0)  { value = L2_M5M3_AUGER;
  } else   if(strcmp(name, "L2_M5M4") == 0)  { value = L2_M5M4_AUGER;
  } else   if(strcmp(name, "L2_M5M5") == 0)  { value = L2_M5M5_AUGER;
  } else   if(strcmp(name, "L3_M1M1") == 0)  { value = L3_M1M1_AUGER;
  } else   if(strcmp(name, "L3_M1M2") == 0)  { value = L3_M1M2_AUGER;
  } else   if(strcmp(name, "L3_M1M3") == 0)  { value = L3_M1M3_AUGER;
  } else   if(strcmp(name, "L3_M1M4") == 0)  { value = L3_M1M4_AUGER;
  } else   if(strcmp(name, "L3_M1M5") == 0)  { value = L3_M1M5_AUGER;
  } else   if(strcmp(name, "L3_M2M1") == 0)  { value = L3_M2M1_AUGER;
  } else   if(strcmp(name, "L3_M2M2") == 0)  { value = L3_M2M2_AUGER;
  } else   if(strcmp(name, "L3_M2M3") == 0)  { value = L3_M2M3_AUGER;
  } else   if(strcmp(name, "L3_M2M4") == 0)  { value = L3_M2M4_AUGER;
  } else   if(strcmp(name, "L3_M2M5") == 0)  { value = L3_M2M5_AUGER;
  } else   if(strcmp(name, "L3_M3M1") == 0)  { value = L3_M3M1_AUGER;
  } else   if(strcmp(name, "L3_M3M2") == 0)  { value = L3_M3M2_AUGER;
  } else   if(strcmp(name, "L3_M3M3") == 0)  { value = L3_M3M3_AUGER;
  } else   if(strcmp(name, "L3_M3M4") == 0)  { value = L3_M3M4_AUGER;
  } else   if(strcmp(name, "L3_M3M5") == 0)  { value = L3_M3M5_AUGER;
  } else   if(strcmp(name, "L3_M4M1") == 0)  { value = L3_M4M1_AUGER;
  } else   if(strcmp(name, "L3_M4M2") == 0)  { value = L3_M4M2_AUGER;
  } else   if(strcmp(name, "L3_M4M3") == 0)  { value = L3_M4M3_AUGER;
  } else   if(strcmp(name, "L3_M4M4") == 0)  { value = L3_M4M4_AUGER;
  } else   if(strcmp(name, "L3_M4M5") == 0)  { value = L3_M4M5_AUGER;
  } else   if(strcmp(name, "L3_M5M1") == 0)  { value = L3_M5M1_AUGER;
  } else   if(strcmp(name, "L3_M5M2") == 0)  { value = L3_M5M2_AUGER;
  } else   if(strcmp(name, "L3_M5M3") == 0)  { value = L3_M5M3_AUGER;
  } else   if(strcmp(name, "L3_M5M4") == 0)  { value = L3_M5M4_AUGER;
  } else   if(strcmp(name, "L3_M5M5") == 0)  { value = L3_M5M5_AUGER;
  } else   if(strcmp(name, "M1_M2M2") == 0)  { value = M1_M2M2_AUGER;
  } else   if(strcmp(name, "M1_M2M3") == 0)  { value = M1_M2M3_AUGER;
  } else   if(strcmp(name, "M1_M2M4") == 0)  { value = M1_M2M4_AUGER;
  } else   if(strcmp(name, "M1_M2M5") == 0)  { value = M1_M2M5_AUGER;
  } else   if(strcmp(name, "M1_M3M2") == 0)  { value = M1_M3M2_AUGER;
  } else   if(strcmp(name, "M1_M3M3") == 0)  { value = M1_M3M3_AUGER;
  } else   if(strcmp(name, "M1_M3M4") == 0)  { value = M1_M3M4_AUGER;
  } else   if(strcmp(name, "M1_M3M5") == 0)  { value = M1_M3M5_AUGER;
  } else   if(strcmp(name, "M1_M4M2") == 0)  { value = M1_M4M2_AUGER;
  } else   if(strcmp(name, "M1_M4M3") == 0)  { value = M1_M4M3_AUGER;
  } else   if(strcmp(name, "M1_M4M4") == 0)  { value = M1_M4M4_AUGER;
  } else   if(strcmp(name, "M1_M4M5") == 0)  { value = M1_M4M5_AUGER;
  } else   if(strcmp(name, "M1_M5M2") == 0)  { value = M1_M5M2_AUGER;
  } else   if(strcmp(name, "M1_M5M3") == 0)  { value = M1_M5M3_AUGER;
  } else   if(strcmp(name, "M1_M5M4") == 0)  { value = M1_M5M4_AUGER;
  } else   if(strcmp(name, "M1_M5M5") == 0)  { value = M1_M5M5_AUGER;
  } else   if(strcmp(name, "M2_M3M3") == 0)  { value = M2_M3M3_AUGER;
  } else   if(strcmp(name, "M2_M3M4") == 0)  { value = M2_M3M4_AUGER;
  } else   if(strcmp(name, "M2_M3M5") == 0)  { value = M2_M3M5_AUGER;
  } else   if(strcmp(name, "M2_M4M3") == 0)  { value = M2_M4M3_AUGER;
  } else   if(strcmp(name, "M2_M4M4") == 0)  { value = M2_M4M4_AUGER;
  } else   if(strcmp(name, "M2_M4M5") == 0)  { value = M2_M4M5_AUGER;
  } else   if(strcmp(name, "M2_M5M3") == 0)  { value = M2_M5M3_AUGER;
  } else   if(strcmp(name, "M2_M5M4") == 0)  { value = M2_M5M4_AUGER;
  } else   if(strcmp(name, "M2_M5M5") == 0)  { value = M2_M5M5_AUGER;
  } else   if(strcmp(name, "M3_M4M4") == 0)  { value = M3_M4M4_AUGER;
  } else   if(strcmp(name, "M3_M4M5") == 0)  { value = M3_M4M5_AUGER;
  } else   if(strcmp(name, "M3_M5M4") == 0)  { value = M3_M5M4_AUGER;
  } else   if(strcmp(name, "M3_M5M5") == 0)  { value = M3_M5M5_AUGER;
  } else   if(strcmp(name, "M4_M5M5") == 0)  { value = M4_M5M5_AUGER;
  } else { value = 0;
  }

  return value;
}



/************************************
  getAugerTransID (name)
************************************/
int getCKTransID (char *name)
{
  int value;
  
  if(strcmp(name, "F1") == 0)  { value = F1_TRANS;
  } else   if(strcmp(name, "F12") == 0)  { value = F12_TRANS;
  } else   if(strcmp(name, "F13") == 0)  { value = F13_TRANS;
  } else   if(strcmp(name, "FP13") == 0) { value = FP13_TRANS;
  } else   if(strcmp(name, "F23") == 0)  { value = F23_TRANS;
  } else   if(strcmp(name, "FL12") == 0) { value = FL12_TRANS;
  } else   if(strcmp(name, "FL13") == 0) { value = FL13_TRANS;
  } else   if(strcmp(name, "FLP13") == 0){ value = FLP13_TRANS;
  } else   if(strcmp(name, "FL23") == 0) { value = FL23_TRANS;
  } else   if(strcmp(name, "FM12") == 0) { value = FM12_TRANS;
  } else   if(strcmp(name, "FM13") == 0) { value = FM13_TRANS;
  } else   if(strcmp(name, "FM14") == 0) { value = FM14_TRANS;
  } else   if(strcmp(name, "FM15") == 0) { value = FM15_TRANS;
  } else   if(strcmp(name, "FM23") == 0) { value = FM23_TRANS;
  } else   if(strcmp(name, "FM24") == 0) { value = FM24_TRANS;
  } else   if(strcmp(name, "FM25") == 0) { value = FM25_TRANS;
  } else   if(strcmp(name, "FM34") == 0) { value = FM34_TRANS;
  } else   if(strcmp(name, "FM35") == 0) { value = FM35_TRANS;
  } else   if(strcmp(name, "FM45") == 0) { value = FM45_TRANS;
  } else { value = 0;
  }

  return value;
}


/******************
  MAIN PROGRAM
******************/

int main ( int argc, char *argv[] )
{
  char   xrlCommand[256], *endp;
  int    i;

  /* Print flag if only one argument */
  if ( argc < 2 ) {
    /* We print argv[0] assuming it is the program name */
    printf( "usage:   xrltest <full_function_call> \n");
    printf( "Example: xrltest \"CSb_Total(15, 2000.0)\"\n\n");
    return 0;
  }
  
  /* parse the xraylib command as the second argument */
  strcpy(xrlCommand, argv[1]);
  if ( getXRLFunctions (xrlCommand) != 0 ) {
    printf( "xrltest error: parsing xraylib function failed: %s\n", xrlCommand);
    return -999;
  }

  /* parse command line options */
  if (argc > 2 ) {
    for ( i = 2; i < argc; i++ ) {
      // printf( "Command line option No. %d: %s\n", i, argv[i]);
      if (strstr(argv[i], "-verb") == argv[i]) { 
        i++; if (isdigit(argv[i][0])) { verbose = strtod(argv[i], &endp); }
      }
    }
  }

  // printf( "Function %s, variables = %g, \n", strVar[0], floatVar[1]);
  /*  Related APS functions */
  if(strcmp(strVar[0], "getMatPropertyFloat") == 0)        { 
    printXrlFloat ( getMatPropertyFloat ( strVar[1], strVar[2] )); 
  } else if(strcmp(strVar[0], "getMatPropertyString") == 0) { 
    printXrlStr ( getMatPropertyString ( strVar[1], strVar[2] )); 
  }

  /* Initialize xraylib */
  XRayInit();

  /*  Atomic number and symbol conversion */
  if(strcmp(strVar[0], "SymbolToAtomicNumber") == 0)        { printXrlFloat ( SymbolToAtomicNumber ( strVar[1] )); 
  } else if(strcmp(strVar[0], "AtomicNumberToSymbol") == 0) { printXrlStr   ( AtomicNumberToSymbol ( intVar[1] )); 
  /*  Atomic Weight */
  } else if(strcmp(strVar[0], "AtomicWeight") == 0)    { printXrlFloat ( AtomicWeight ( intVar[1] )); 
  } else if(strcmp(strVar[0], "AtomicWeight_CP") == 0) { printXrlFloat ( AtomicWeight ( SymbolToAtomicNumber(strVar[1]) )); 
  /*  ELECTRONIC CONFIGURATION */
  } else if(strcmp(strVar[0], "ElectronConfig") == 0)    { printXrlFloat ( ElectronConfig ( intVar[1], getShellID(strVar[2]) )); 
  } else if(strcmp(strVar[0], "ElectronConfig_CP") == 0) { printXrlFloat ( ElectronConfig ( SymbolToAtomicNumber(strVar[1]), getShellID(strVar[2]) ));
  /*  SCATTERING FUNCTIONS AND FORM FACTORS */
  } else if(strcmp(strVar[0], "MomentTransf") == 0)  { printXrlFloat ( MomentTransf ( floatVar[1], floatVar[2] )); 
  } else if(strcmp(strVar[0], "ComptonEnergy") == 0) { printXrlFloat ( ComptonEnergy( floatVar[1], floatVar[2] )); 
  } else if(strcmp(strVar[0], "FF_Rayl") == 0)       { printXrlFloat ( FF_Rayl ( intVar[1], floatVar[2] )); 
  } else if(strcmp(strVar[0], "SF_Compt") == 0)      { printXrlFloat ( SF_Compt ( intVar[1], floatVar[2] )); 
  /*  ANOMALOUS SCATTERING FACTOR */
  } else if(strcmp(strVar[0], "Fi") == 0)   { printXrlFloat ( Fi  ( intVar[1], floatVar[2] )); 
  } else if(strcmp(strVar[0], "Fii") == 0)  { printXrlFloat ( Fii ( intVar[1], floatVar[2] )); 
  /*  REFRACTIVE INDICES FUNCTIONS */
  } else if(strcmp(strVar[0], "Refractive_Index_Re") == 0)   { printXrlFloat ( Refractive_Index_Re ( strVar[1], floatVar[2], floatVar[3] )); 
  } else if(strcmp(strVar[0], "Refractive_Index_Im") == 0)   { printXrlFloat ( Refractive_Index_Im ( strVar[1], floatVar[2], floatVar[3] )); 
  /*  COMPTON PROFILES */
  } else if(strcmp(strVar[0], "ComptonProfile") == 0)         { printXrlFloat ( ComptonProfile ( intVar[1], floatVar[2] )); 
  } else if(strcmp(strVar[0], "ComptonProfile_Partial") == 0) { printXrlFloat ( ComptonProfile_Partial ( intVar[1], getShellID(strVar[2]), floatVar[2] )); 
  /*  ABSORPTION EDGES AND TRANSITIONS DATA */
  } else if(strcmp(strVar[0], "EdgeEnergy") == 0)    { printXrlFloat ( EdgeEnergy ( intVar[1], getShellID(strVar[2]) )); 
  } else if(strcmp(strVar[0], "EdgeEnergy_CP") == 0) { printXrlFloat ( EdgeEnergy ( SymbolToAtomicNumber(strVar[1]), getShellID(strVar[2]) ));
  } else if(strcmp(strVar[0], "FluorYield") == 0)    { printXrlFloat ( FluorYield ( intVar[1], getShellID(strVar[2]) )); 
  } else if(strcmp(strVar[0], "FluorYield_CP") == 0) { printXrlFloat ( FluorYield ( SymbolToAtomicNumber(strVar[1]), getShellID(strVar[2]) ));
  } else if(strcmp(strVar[0], "JumpFactor") == 0)    { printXrlFloat ( JumpFactor ( intVar[1], getShellID(strVar[2]) )); 
  } else if(strcmp(strVar[0], "JumpFactor_CP") == 0) { printXrlFloat ( JumpFactor ( SymbolToAtomicNumber(strVar[1]), getShellID(strVar[2]) ));
  } else if(strcmp(strVar[0], "AtomicLevelWidth") == 0)    { printXrlFloat ( AtomicLevelWidth ( intVar[1], getShellID(strVar[2]) )); 
  } else if(strcmp(strVar[0], "AtomicLevelWidth_CP") == 0) { printXrlFloat ( AtomicLevelWidth ( SymbolToAtomicNumber(strVar[1]), getShellID(strVar[2]) ));
  } else if(strcmp(strVar[0], "LineEnergy") == 0)    { printXrlFloat ( LineEnergy ( intVar[1], getLineID(strVar[2]) )); 
  } else if(strcmp(strVar[0], "LineEnergy_CP") == 0) { printXrlFloat ( LineEnergy ( SymbolToAtomicNumber(strVar[1]), getLineID(strVar[2]) ));
  } else if(strcmp(strVar[0], "RadRate") == 0)       { printXrlFloat ( RadRate ( intVar[1], getLineID(strVar[2]) )); 
  } else if(strcmp(strVar[0], "RadRate_CP") == 0)    { printXrlFloat ( RadRate ( SymbolToAtomicNumber(strVar[1]), getLineID(strVar[2]) ));
  /*  CROSS SECTIONS: (cm2/g) */
  } else if(strcmp(strVar[0], "CS_Total") == 0)    { printXrlFloat ( CS_Total ( intVar[1], floatVar[2] )); 
  } else if(strcmp(strVar[0], "CS_Photo") == 0)    { printXrlFloat ( CS_Photo ( intVar[1], floatVar[2] )); 
  } else if(strcmp(strVar[0], "CS_Rayl")  == 0)    { printXrlFloat ( CS_Rayl  ( intVar[1], floatVar[2] )); 
  } else if(strcmp(strVar[0], "CS_Compt") == 0)    { printXrlFloat ( CS_Compt ( intVar[1], floatVar[2] )); 
  } else if(strcmp(strVar[0], "CS_FluorLine") == 0){ printXrlFloat ( CS_FluorLine  ( intVar[1], getLineID(strVar[2]), floatVar[3] )); 
  } else if(strcmp(strVar[0], "CS_Total_CP") == 0) { printXrlFloat ( CS_Total_CP ( strVar[1], floatVar[2] )); 
  } else if(strcmp(strVar[0], "CS_Photo_CP") == 0) { printXrlFloat ( CS_Photo_CP ( strVar[1], floatVar[2] )); 
  } else if(strcmp(strVar[0], "CS_Rayl_CP")  == 0) { printXrlFloat ( CS_Rayl_CP  ( strVar[1], floatVar[2] )); 
  } else if(strcmp(strVar[0], "CS_Compt_CP") == 0) { printXrlFloat ( CS_Compt_CP ( strVar[1], floatVar[2] )); 
  /*  CROSS SECTIONS: (barn/atom) */
  } else if(strcmp(strVar[0], "CS_KN") == 0)        { printXrlFloat ( CS_KN     ( floatVar[1] )); 
  } else if(strcmp(strVar[0], "CSb_Total") == 0)    { printXrlFloat ( CSb_Total ( intVar[1], floatVar[2] )); 
  } else if(strcmp(strVar[0], "CSb_Photo") == 0)    { printXrlFloat ( CSb_Photo ( intVar[1], floatVar[2] )); 
  } else if(strcmp(strVar[0], "CSb_Rayl")  == 0)    { printXrlFloat ( CSb_Rayl  ( intVar[1], floatVar[2] )); 
  } else if(strcmp(strVar[0], "CSb_Compt") == 0)    { printXrlFloat ( CSb_Compt ( intVar[1], floatVar[2] )); 
  } else if(strcmp(strVar[0], "CSb_FluorLine") == 0){ printXrlFloat ( CSb_FluorLine ( intVar[1], getLineID(strVar[2]), floatVar[3] )); 
  } else if(strcmp(strVar[0], "CSb_Total_CP") == 0) { printXrlFloat ( CSb_Total_CP ( strVar[1], floatVar[2] )); 
  } else if(strcmp(strVar[0], "CSb_Photo_CP") == 0) { printXrlFloat ( CSb_Photo_CP ( strVar[1], floatVar[2] )); 
  } else if(strcmp(strVar[0], "CSb_Rayl_CP")  == 0) { printXrlFloat ( CSb_Rayl_CP  ( strVar[1], floatVar[2] )); 
  } else if(strcmp(strVar[0], "CSb_Compt_CP") == 0) { printXrlFloat ( CSb_Compt_CP ( strVar[1], floatVar[2] )); 
  /*  DIFFERENTIAL UNPOLARIZED CROSS SECTIONS: (cm2/g/sterad) */
  } else if(strcmp(strVar[0], "DCS_Rayl") == 0)     { printXrlFloat ( DCS_Rayl  ( intVar[1],   floatVar[2], floatVar[3] )); 
  } else if(strcmp(strVar[0], "DCS_Compt") == 0)    { printXrlFloat ( DCS_Compt ( intVar[1],   floatVar[2], floatVar[3] )); 
  } else if(strcmp(strVar[0], "DCS_Rayl_CP") == 0)  { printXrlFloat ( DCS_Rayl_CP  ( strVar[1],   floatVar[2], floatVar[3] )); 
  } else if(strcmp(strVar[0], "DCS_Compt_CP") == 0) { printXrlFloat ( DCS_Compt_CP ( strVar[1],   floatVar[2], floatVar[3] )); 
  /*  DIFFERENTIAL UNPOLARIZED CROSS SECTIONS: (barns/atom/sterad) */
  } else if(strcmp(strVar[0], "DCS_Thoms") == 0)    { printXrlFloat ( DCS_Thoms ( floatVar[1] )); 
  } else if(strcmp(strVar[0], "DCS_KN") == 0)       { printXrlFloat ( DCS_KN    ( floatVar[1], floatVar[2] )); 
  } else if(strcmp(strVar[0], "DCSb_Rayl") == 0)    { printXrlFloat ( DCSb_Rayl ( intVar[1],   floatVar[2], floatVar[3] )); 
  } else if(strcmp(strVar[0], "DCSb_Compt") == 0)   { printXrlFloat ( DCSb_Compt( intVar[1],   floatVar[2], floatVar[3] )); 
  } else if(strcmp(strVar[0], "DCSb_Rayl_CP") == 0) { printXrlFloat ( DCSb_Rayl_CP  ( strVar[1],   floatVar[2], floatVar[3] )); 
  } else if(strcmp(strVar[0], "DCSb_Compt_CP") == 0){ printXrlFloat ( DCSb_Compt_CP ( strVar[1],   floatVar[2], floatVar[3] )); 
  /*  DIFFERENTIAL POLARIZED CROSS SECTIONS: (cm2/g/sterad) */
  } else if(strcmp(strVar[0], "DCSP_Rayl") == 0)     { printXrlFloat ( DCSP_Rayl  ( intVar[1], floatVar[2], floatVar[3], floatVar[4] )); 
  } else if(strcmp(strVar[0], "DCSP_Compt") == 0)    { printXrlFloat ( DCSP_Compt ( intVar[1], floatVar[2], floatVar[3], floatVar[4]  )); 
  } else if(strcmp(strVar[0], "DCSP_Rayl_CP") == 0)  { printXrlFloat ( DCSP_Rayl_CP  ( strVar[1], floatVar[2], floatVar[3], floatVar[4] )); 
  } else if(strcmp(strVar[0], "DCSP_Compt_CP") == 0) { printXrlFloat ( DCSP_Compt_CP ( strVar[1], floatVar[2], floatVar[3], floatVar[4]  )); 
  /*  DIFFERENTIAL POLARIZED CROSS SECTIONS: (barns/atom/sterad) */
  } else if(strcmp(strVar[0], "DCSP_Thoms") == 0)    { printXrlFloat ( DCSP_Thoms ( floatVar[1], floatVar[2]  )); 
  } else if(strcmp(strVar[0], "DCSP_KN") == 0)       { printXrlFloat ( DCSP_KN    ( floatVar[1], floatVar[2], floatVar[3]  )); 
  } else if(strcmp(strVar[0], "DCSPb_Rayl") == 0)    { printXrlFloat ( DCSPb_Rayl ( intVar[1], floatVar[2], floatVar[3], floatVar[4]  )); 
  } else if(strcmp(strVar[0], "DCSPb_Compt") == 0)   { printXrlFloat ( DCSPb_Compt( intVar[1], floatVar[2], floatVar[3], floatVar[4]  )); 
  } else if(strcmp(strVar[0], "DCSPb_Rayl_CP") == 0) { printXrlFloat ( DCSPb_Rayl_CP ( strVar[1], floatVar[2], floatVar[3], floatVar[4]  )); 
  } else if(strcmp(strVar[0], "DCSPb_Compt_CP") == 0){ printXrlFloat ( DCSPb_Compt_CP( strVar[1], floatVar[2], floatVar[3], floatVar[4]  )); 
  /*  KISSEL PHOTOELECTRIC CROSS SECTIONS  (cm2/g) */
  } else if(strcmp(strVar[0], "CS_Total_Kissel") == 0)  { printXrlFloat ( CS_Total_Kissel  ( intVar[1], floatVar[2] )); 
  } else if(strcmp(strVar[0], "CS_Photo_Total") == 0)   { printXrlFloat ( CS_Photo_Total   ( intVar[1], floatVar[2] )); 
  } else if(strcmp(strVar[0], "CS_Photo_Partial") == 0) { printXrlFloat ( CS_Photo_Partial ( intVar[1], getShellID(strVar[2]), floatVar[3] )); 
  } else if(strcmp(strVar[0], "CS_Photo_Total_CP") == 0){ printXrlFloat ( CS_Photo_Total_CP ( strVar[1], floatVar[2] )); 
  /*  KISSEL PHOTOELECTRIC CROSS SECTIONS  (barn/atom) */
  } else if(strcmp(strVar[0], "CSb_Total_Kissel") == 0) { printXrlFloat ( CSb_Total_Kissel ( intVar[1], floatVar[2] )); 
  } else if(strcmp(strVar[0], "CSb_Photo_Total") == 0)  { printXrlFloat ( CSb_Photo_Total  ( intVar[1], floatVar[2] )); 
  } else if(strcmp(strVar[0], "CSb_Photo_Partial") == 0){ printXrlFloat ( CSb_Photo_Partial( intVar[1], getShellID(strVar[2]), floatVar[3] )); 
  } else if(strcmp(strVar[0], "CSb_Photo_Total_CP") == 0)  { printXrlFloat ( CSb_Photo_Total_CP  ( strVar[1], floatVar[2] )); 
  /*  XRF CROSS SECTIONS USING KISSEL PARTIAL PHOTOELECTRIC CROSS SECTIONS  (cm2/g) */
  } else if(strcmp(strVar[0], "CS_FluorLine_Kissel") == 0)             { printXrlFloat ( CS_FluorLine_Kissel ( intVar[1], getLineID(strVar[2]), floatVar[3] ));
  } else if(strcmp(strVar[0], "CS_FluorLine_Kissel_no_Cascade") == 0)  { printXrlFloat ( CS_FluorLine_Kissel_no_Cascade ( intVar[1], getLineID(strVar[2]), floatVar[3] ));
  } else if(strcmp(strVar[0], "CS_FluorLine_Kissel_Cascade") == 0)     { printXrlFloat ( CS_FluorLine_Kissel_Cascade ( intVar[1], getLineID(strVar[2]), floatVar[3] ));
  } else if(strcmp(strVar[0], "CS_FluorLine_Kissel_Nonradiative_Cascade") == 0){ printXrlFloat ( CS_FluorLine_Kissel_Nonradiative_Cascade ( intVar[1], getLineID(strVar[2]), floatVar[3] ));
  } else if(strcmp(strVar[0], "CS_FluorLine_Kissel_Radiative_Cascade") == 0)   { printXrlFloat ( CS_FluorLine_Kissel_Radiative_Cascade ( intVar[1], getLineID(strVar[2]), floatVar[3] ));
  /*  XRF CROSS SECTIONS USING KISSEL PARTIAL PHOTOELECTRIC CROSS SECTIONS  (barn/atom) */
  } else if(strcmp(strVar[0], "CSb_FluorLine_Kissel") == 0)            { printXrlFloat ( CSb_FluorLine_Kissel ( intVar[1], getLineID(strVar[2]), floatVar[3] ));
  } else if(strcmp(strVar[0], "CSb_FluorLine_Kissel_no_Cascade") == 0) { printXrlFloat ( CSb_FluorLine_Kissel_no_Cascade ( intVar[1], getLineID(strVar[2]), floatVar[3] ));
  } else if(strcmp(strVar[0], "CSb_FluorLine_Kissel_Cascade") == 0)    { printXrlFloat ( CSb_FluorLine_Kissel_Cascade ( intVar[1], getLineID(strVar[2]), floatVar[3] ));
  } else if(strcmp(strVar[0], "CSb_FluorLine_Kissel_Nonradiative_Cascade") == 0)  { printXrlFloat ( CSb_FluorLine_Kissel_Nonradiative_Cascade ( intVar[1], getLineID(strVar[2]), floatVar[3] ));
  } else if(strcmp(strVar[0], "CSb_FluorLine_Kissel_Radiative_Cascade") == 0)     { printXrlFloat ( CSb_FluorLine_Kissel_Radiative_Cascade ( intVar[1], getLineID(strVar[2]), floatVar[3] ));
  /*  NONRADIATIVE RATE */
  } else if(strcmp(strVar[0], "AugerRate") == 0)        { printXrlFloat ( AugerRate ( intVar[1], getAugerTransID(strVar[2]) ));
  } else if(strcmp(strVar[0], "CosKronTransProb") == 0) { printXrlFloat ( CosKronTransProb ( intVar[1], getCKTransID(strVar[2]) ));
  /*  UNRECOGNIZED FUNCTION */
  } else {
    printf( "xrltest error: XRL function not recognized, %s\n", strVar[0]);
    return -999;
  }

  return 0;
}


