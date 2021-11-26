
#include "petsc.h"

PetscErrorCode PetscDSAddBoundary_UW(DM dm, DMBoundaryConditionType type, const char name[], const char labelname[], PetscInt field, PetscInt numcomps, const PetscInt *comps, void (*bcFunc)(void), void (*bcFunc_t)(void), PetscInt numids, const PetscInt *ids, void *ctx)
{
#if ( (PETSC_VERSION_MAJOR==3) && (PETSC_VERSION_MINOR<=15) )
    PetscDS ds;
    DMGetDS(dm, &ds);
    return PetscDSAddBoundary(ds, type, name, labelname, field, numcomps, comps, bcFunc, bcFunc_t, numids, ids, ctx);
#else
    DMLabel label;
    DMGetLabel(dm, labelname, &label);
    PetscInt bd;  // This is a return value that we do nothing with.
    return DMAddBoundary(dm, type, name, label, numids, ids, field, numcomps, comps, bcFunc, bcFunc_t, ctx, &bd);
#endif
}
