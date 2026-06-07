/*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*
**                                                                                  **
** This file forms part of the Underworld geophysics modelling application.         **
**                                                                                  **
** For full license and copyright information, please refer to the LICENSE.md file  **
** located at the project root, or contact the authors.                             **
**                                                                                  **
**~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*/

/* Thin UW3 adapters around the verbatim Velic SolCx kernel (solCx.c).
   _Velic_solCx fills caller-provided arrays; pass NULL for components we
   don't want. The viscosity is the analytic step (the kernel doesn't return
   it as a field). */

#include "solCx.h"
#include "AnalyticSolCx.h"

vec2 SolCx_velocity( double eta_A, double eta_B, double x_c, int n, const double x, const double z )
{
    double pos[2];
    double vel[2];
    vec2 out;
    pos[0] = x;
    pos[1] = z;
    _Velic_solCx( pos, eta_A, eta_B, x_c, n, vel, NULL, NULL, NULL );
    out.x = vel[0];
    out.z = vel[1];
    return out;
}

double SolCx_pressure( double eta_A, double eta_B, double x_c, int n, const double x, const double z )
{
    double pos[2];
    double pressure;
    pos[0] = x;
    pos[1] = z;
    _Velic_solCx( pos, eta_A, eta_B, x_c, n, NULL, &pressure, NULL, NULL );
    return pressure;
}

double SolCx_viscosity( double eta_A, double eta_B, double x_c, int n, const double x, const double z )
{
    (void) n; (void) z;
    return ( x < x_c ) ? eta_A : eta_B;
}
