/*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*
**                                                                                  **
** This file forms part of the Underworld geophysics modelling application.         **
**                                                                                  **
** For full license and copyright information, please refer to the LICENSE.md file  **
** located at the project root, or contact the authors.                             **
**                                                                                  **
**~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*/

#ifndef __Underworld_Function_AnalyticSolNL_h__
#define __Underworld_Function_AnalyticSolNL_h__

typedef struct
{
    double x;
    double z;
} vec2;

typedef struct
{
    double xx;
    double zz;
    double xz;
} tensor2;

vec2    SolNL_velocity(   double eta0, unsigned n, double r, const double x, const double z ) __attribute__((const));
double  SolNL_pressure(   double eta0, unsigned n, double r, const double x, const double z ) __attribute__((const));
tensor2 SolNL_stress(     double eta0, unsigned n, double r, const double x, const double z ) __attribute__((const));
tensor2 SolNL_strainrate( double eta0, unsigned n, double r, const double x, const double z ) __attribute__((const));
double  SolNL_viscosity(  double eta0, unsigned n, double r, const double x, const double z ) __attribute__((const));
vec2    SolNL_bodyforce(  double eta0, unsigned n, double r, const double x, const double z ) __attribute__((const));

#endif
