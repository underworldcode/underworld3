#include "petsc.h"

// Add 1 boundary condition at a time (1 boundary, 1 component etc etc)

PetscErrorCode PetscDSAddBoundary_UW(DM dm,
                                     DMBoundaryConditionType type,
                                     const char name[],
                                     const char labelname[],
                                     PetscInt field,
                                     PetscInt num_const_components,
                                     const PetscInt *components,
                                     void (*bcFunc)(void),
                                     void (*bcFunc_t)(void),
                                     const PetscInt ids,
                                     const PetscInt *id_values,
                                     void *ctx)
{

    DMLabel label;
    DMGetLabel(dm, labelname, &label);

    PetscInt bd; // This is a return value that we pass back.
    PetscDS ds;

    DMAddBoundary(dm, type, name, label, ids, id_values, field, num_const_components, components, bcFunc, bcFunc_t, ctx, &bd);

    return bd;
}

PetscErrorCode DMSetAuxiliaryVec_UW(DM dm, DMLabel label, PetscInt value, PetscInt part, Vec aux)
{
    return DMSetAuxiliaryVec(dm, label, value, part, aux);
}

PetscErrorCode UW_PetscDSSetBdTerms(PetscDS ds, DMLabel label, PetscInt label_val, PetscInt bd,
                                    PetscInt f, PetscInt g, PetscInt part,
                                    PetscInt idx0, void (*bcFunc_f0)(void),
                                    PetscInt idx1, void (*bcFunc_f1)(void),
                                    PetscInt idxg0, void (*bcFunc_g0)(void),
                                    PetscInt idxg1, void (*bcFunc_g1)(void),
                                    PetscInt idxg2, void (*bcFunc_g2)(void),
                                    PetscInt idxg3, void (*bcFunc_g3)(void))

{
    PetscWeakForm wf;

    PetscCall(PetscDSGetBoundary(ds, bd, &wf, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
    PetscCall(PetscWeakFormSetIndexBdResidual(wf, label, label_val, f, part, idx0, bcFunc_f0, idx1, bcFunc_f1));
    PetscCall(PetscWeakFormSetIndexBdJacobian(wf, label, label_val, f, g, part, idxg0, bcFunc_g0, idxg1, bcFunc_g1, idxg2, bcFunc_g2, idxg3, bcFunc_g3));
    PetscCall(PetscWeakFormSetIndexBdJacobianPreconditioner(wf, label, label_val, f, g, part, idxg0, bcFunc_g0, idxg1, bcFunc_g1, idxg2, bcFunc_g2, idxg3, bcFunc_g3));

    return 1;
}

PetscErrorCode UW_PetscDSSetBdResidual(PetscDS ds, DMLabel label, PetscInt label_val, PetscInt bd,
                                       PetscInt f, PetscInt part,
                                       PetscInt idx0, void (*bcFunc_f0)(void),
                                       PetscInt idx1, void (*bcFunc_f1)(void))
{

    PetscWeakForm wf;
    PetscCall(PetscDSGetBoundary(ds, bd, &wf, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
    PetscCall(PetscWeakFormSetIndexBdResidual(wf, label, label_val, f, part, idx0, bcFunc_f0, 0, NULL));

    return 1;
}

PetscErrorCode UW_PetscDSSetBdJacobian(PetscDS ds, DMLabel label, PetscInt label_val, PetscInt bd,
                                       PetscInt f, PetscInt g, PetscInt part,
                                       PetscInt idx0, void (*bcFunc_g0)(void),
                                       PetscInt idx1, void (*bcFunc_g1)(void),
                                       PetscInt idx2, void (*bcFunc_g2)(void),
                                       PetscInt idx3, void (*bcFunc_g3)(void))
{
    PetscWeakForm wf;

    PetscCall(PetscDSGetBoundary(ds, bd, &wf, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
    PetscCall(PetscWeakFormSetIndexBdJacobian(wf, label, label_val, f, g, part, idx0, bcFunc_g0, idx1, bcFunc_g1, idx2, bcFunc_g2, idx3, bcFunc_g3));

    return 1;
}

PetscErrorCode UW_PetscDSSetBdJacobianPreconditioner(PetscDS ds, DMLabel label, PetscInt label_val, PetscInt bd,
                                                     PetscInt f, PetscInt g, PetscInt part,
                                                     PetscInt idx0, void (*bcFunc_g0)(void),
                                                     PetscInt idx1, void (*bcFunc_g1)(void),
                                                     PetscInt idx2, void (*bcFunc_g2)(void),
                                                     PetscInt idx3, void (*bcFunc_g3)(void))
{
    PetscWeakForm wf;

    PetscCall(PetscDSGetBoundary(ds, bd, &wf, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
    PetscCall(PetscWeakFormSetIndexBdJacobianPreconditioner(wf, label, label_val, f, g, part, idx0, bcFunc_g0, idx1, bcFunc_g1, idx2, bcFunc_g2, idx3, bcFunc_g3));

    return 1;
}

PetscErrorCode UW_PetscDSViewWF(PetscDS ds)
{

    PetscWeakForm wf;

    PetscCall(PetscDSGetWeakForm(ds, &wf));
    PetscCall(PetscWeakFormView(wf, NULL));

    return 1;
}

PetscErrorCode UW_PetscDSViewBdWF(PetscDS ds, PetscInt bd)
{

    PetscWeakForm wf;

    PetscCall(PetscDSGetBoundary(ds, bd, &wf, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
    PetscCall(PetscWeakFormView(wf, NULL));

    return 1;
}

// PetscErrorCode UW_PetscVecConcatenate(PetscInt nx, Vec inputVecs[], Vec *outputVec)
// {
//     IS *x_is;

//     PetscErrorCode VecConcatenate(nx, inputVecs, outputVec, &x_is);

//     return 1;
// }
