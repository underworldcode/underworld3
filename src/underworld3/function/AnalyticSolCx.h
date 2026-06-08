/*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*
**                                                                                  **
** This file forms part of the Underworld geophysics modelling application.         **
**                                                                                  **
** For full license and copyright information, please refer to the LICENSE.md file  **
** located at the project root, or contact the authors.                             **
**                                                                                  **
**~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*/

/* UW3 wrapper for the classic Velic "SolCx" analytic Stokes solution:
   isoviscous step in x (eta_A for x<x_c, eta_B for x>=x_c), trigonometric
   density forcing  f = (0, +cos(pi x) sin(n pi z))  (UW3 momentum-sign
   convention; UW2 docs quote -cos) on a unit box with free-slip walls.
   The heavy lifting is the verbatim Velic kernel in solCx.c
   (_Velic_solCx); these thin adapters expose it in the same vec2/scalar shape
   as AnalyticSolNL so analytic.pyx can wrap it identically. */

#ifndef __Underworld_Function_AnalyticSolCx_h__
#define __Underworld_Function_AnalyticSolCx_h__

#include "AnalyticSolNL.h"   /* for the shared vec2 typedef */

/* See AnalyticSolNL.h for the rationale behind __attribute__((const)):
   the result depends solely on the arguments, so redundant calls (e.g. the
   x and z components evaluated separately) can be collapsed by the compiler. */

vec2   SolCx_velocity(  double eta_A, double eta_B, double x_c, int n, const double x, const double z ) __attribute__((const));
double SolCx_pressure(  double eta_A, double eta_B, double x_c, int n, const double x, const double z ) __attribute__((const));
double SolCx_viscosity( double eta_A, double eta_B, double x_c, int n, const double x, const double z ) __attribute__((const));

#endif
