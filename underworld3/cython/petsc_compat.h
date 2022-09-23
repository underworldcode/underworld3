
#include "petsc.h"

PetscErrorCode PetscDSAddBoundary_UW(DM dm, DMBoundaryConditionType type, const char name[], const char labelname[], PetscInt field, PetscInt numcomps, const PetscInt *comps, void (*bcFunc)(void), void (*bcFunc_t)(void), PetscInt numids, const PetscInt *ids, void *ctx)
{
#if PETSC_VERSION_LE(3,15,0)
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

PetscErrorCode DMSetAuxiliaryVec_UW(DM dm, DMLabel label, PetscInt value, PetscInt part, Vec aux)
{
#if PETSC_VERSION_LE(3,16,4) 
    return DMSetAuxiliaryVec(dm, label, value, aux);
#else
    return DMSetAuxiliaryVec(dm, label, value, part, aux);
#endif
}
