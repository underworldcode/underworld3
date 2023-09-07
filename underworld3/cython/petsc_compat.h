#include "petsc.h"

// Add 1 boundary condition at a time (1 boundary, 1 component etc etc)

PetscErrorCode PetscDSAddBoundary_UW(DM dm,
                                     DMBoundaryConditionType type,
                                     const char name[],
                                     const char labelname[],
                                     PetscInt field,
                                     const PetscInt component,
                                     void (*bcFunc)(void),
                                     void (*bcFunc_t)(void),
                                     const PetscInt ids,
                                     const PetscInt *id_values,
                                     void *ctx)
{

    DMLabel label;
    DMGetLabel(dm, labelname, &label);

    PetscInt bd; // This is a return value that we pass back.
    PetscInt components[1];
    PetscDS ds;

    components[0] = component;

    DMAddBoundary(dm, type, name, label, ids, id_values, field, 1, &component, bcFunc, bcFunc_t, ctx, &bd);

    // fprintf(stdout, "Adding in boundary ... %d\n", bd);

    return bd;
}

PetscErrorCode DMSetAuxiliaryVec_UW(DM dm, DMLabel label, PetscInt value, PetscInt part, Vec aux)
{
    return DMSetAuxiliaryVec(dm, label, value, part, aux);
}

PetscErrorCode UW_PetscDSSetBdResidual(PetscDS ds, DMLabel label, PetscInt label_val, PetscInt bd,
                                       PetscInt f, PetscInt part,
                                       PetscInt idx0, void (*bcFunc_f0)(void),
                                       PetscInt idx1, void (*bcFunc_f1)(void))
{

    PetscWeakForm wf;

    // PetscCall(PetscDSGetBoundary(ds, bd, &wf1, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
    PetscCall(PetscDSGetWeakForm(ds, &wf));
    PetscCall(PetscWeakFormSetIndexBdResidual(wf, label, label_val, f, part, idx0, bcFunc_f0, idx1, bcFunc_f1));

    return 1;
}

PetscErrorCode UW_PetscDSViewWF(PetscDS ds)
{

    PetscWeakForm wf;

    // PetscCall(PetscDSGetBoundary(ds, bd, &wf1, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
    PetscCall(PetscDSGetWeakForm(ds, &wf));
    PetscCall(PetscWeakFormView(wf, NULL));

    return 1;
}

// PetscErrorCode UW_PetscDSSetBdJacobian(PetscDS ds, DMLabel label, PetscInt label_val, PetscInt bd,
//                                        PetscInt f, PetscInt part,
//                                        PetscInt idx0, void (*bcFunc_f0)(void),
//                                        PetscInt idx1, void (*bcFunc_f1)(void))
// {

//     PetscWeakForm wf;

//     // PetscCall(PetscDSGetBoundary(ds, bd, &wf1, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
//     PetscCall(PetscDSGetWeakForm(ds, &wf));
//     PetscCall(PetscWeakFormSetIndexBdJacobian(wf, label, label_val, f, part, idx0, bcFunc_f0, idx1, bcFunc_f1));

//     return 1;
// }
