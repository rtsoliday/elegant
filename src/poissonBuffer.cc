#include "poissonBuffer.h"
#include "poisson.h"
#include "manual.h"

void testFunc() {
  return;
}

void poissonSolverWrapper(double **ion2dDensity, double **ionPotential, long N_x, long N_y,
                          double **xkick, double **ykick, double delta[2]) {

  double x_max, y_max;

  x_max = N_x * delta[0] / 2;
  y_max = N_y * delta[1] / 2;

  /*
  vector<vector<double>> x(N_x, vector<double> (N_y, 0));
  vector<vector<double>> y(N_x, vector<double> (N_y, 0));
  meshgrid(x_max, y_max, N_x, N_y, x, y);
  */

  /*
  vector<vector<double> > u_rhs(N_x, vector<double> (N_y, 0));
  for (int i=0; i<N_x; i++) {
    for (int j=0; j<N_y; j++) {
      u_rhs[i][j] = ion2dDensity[i][j];
    }
  }
  */


  double x_domain = 2.0 * x_max;
  double y_domain = 2.0 * y_max;
 
  //vector<vector<double> > u_fft;
  //poisson_solver(u_rhs, x_domain, y_domain, N_x, N_y);
  poisson_solver(ion2dDensity, x_domain, y_domain, N_x, N_y, ionPotential);

  double d1 = 1.0 / delta[0];
  double d2 = 1.0 / delta[1];
  for (int i=0; i<N_x-1; i++) {
    for (int j=0; j<N_y-1; j++) {
      xkick[i][j] =  (ionPotential[i+1][j] - ionPotential[i][j]) * d1;
    }
  }
  for (int i=0; i<N_x-1; i++) {
    for (int j=0; j<N_y-1; j++) {
      ykick[i][j] = (ionPotential[i][j+1] - ionPotential[i][j]) * d2;
    }
  }
  for (int i=N_x-1; i<N_x; i++) {
    for (int j=N_y-1; j<N_y; j++) {
        xkick[i][j] = 0;
        ykick[i][j] = 0;
    }
  }

  //write_to_file(N_x, N_y, u_fft, "u_fft.txt");
  //write_to_file(N_x, N_y, dx, "dx.txt");
  //write_to_file(N_x, N_y, dy, "dy.txt");

}
