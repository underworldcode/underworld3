#include <petsc/private/dmpleximpl.h>   /*I "petscdmplex.h" I*/
#include <petsc/private/snesimpl.h>     /*I "petscsnes.h"   I*/
#include <petscds.h>
#include <petscblaslapack.h>
#include <petsc/private/petscimpl.h>
#include <petsc/private/petscfeimpl.h>

PetscErrorCode DMInterpolationSetUp_UW(DMInterpolationInfo ctx, DM dm, PetscBool redundantPoints, PetscBool ignoreOutsideDomain, size_t* owning_cell);
PetscErrorCode DMInterpolationEvaluate_UW(DMInterpolationInfo ctx, DM dm, Vec x, Vec v);