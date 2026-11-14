#include <petsc/private/dmpleximpl.h>   /*I "petscdmplex.h" I*/
#include <petsc/private/snesimpl.h>     /*I "petscsnes.h"   I*/
#include <petscds.h>
#include <petscblaslapack.h>
#include <petsc/private/petscimpl.h>
#include <petsc/private/petscfeimpl.h>

PetscErrorCode DMInterpolationSetUp_UW(DMInterpolationInfo ctx, DM dm, PetscBool redundantPoints, PetscBool ignoreOutsideDomain, size_t* owning_cell, PetscBool hintAuthoritative);
PetscErrorCode DMInterpolationEvaluate_UW(DMInterpolationInfo ctx, DM dm, Vec x, Vec v);
PetscErrorCode DMInterpolationGetUnlocated_UW(DMInterpolationInfo ctx, PetscInt n, signed char* mask);
PetscErrorCode DMInterpolationGetCells_UW(DMInterpolationInfo ctx, PetscInt n, PetscInt64 *cells);
PetscErrorCode DMPlexCellClosureFieldRange_UW(DM dm, Vec x, PetscInt field, PetscInt nc, PetscInt n, const PetscInt64 *cells, PetscReal *lo, PetscReal *hi);
