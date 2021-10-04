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


// The __attribute__ ((const)) means that the return value is solely a function of the arguments, 
// and if any of the arguments are pointers, then the pointers must not be dereferenced. 
// A const function is always pure.

// Basically, the compiler is told that if the input parameters are the same, the return result will be the same, 
// and therefore it should reuse the result where possible. 

// Specifically, the way we write out the extension code results in vectors being evaluated 
// as `solNL_velocity().x + solNL_velocity().z` (for example), so these flags prevent 
// `solNL_velocity()` being evaluated twice. Also, some analytic solutions (in particular SolH) 
// can become *very* expensive to compute, so we definitely don't want them to be computed redundantly.

// This flag may cause issues on certain compilers, in which case it should be #deffed in or out. 

vec2    SolNL_velocity(   double eta0, unsigned n, double r, const double x, const double z ) __attribute__((const));
double  SolNL_pressure(   double eta0, unsigned n, double r, const double x, const double z ) __attribute__((const));
tensor2 SolNL_stress(     double eta0, unsigned n, double r, const double x, const double z ) __attribute__((const));
tensor2 SolNL_strainrate( double eta0, unsigned n, double r, const double x, const double z ) __attribute__((const));
double  SolNL_viscosity(  double eta0, unsigned n, double r, const double x, const double z ) __attribute__((const));
vec2    SolNL_bodyforce(  double eta0, unsigned n, double r, const double x, const double z ) __attribute__((const));

#endif

