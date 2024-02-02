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

    DMAddBoundary(dm, type, name, label, ids, id_values, field, num_const_components, components, bcFunc, bcFunc_t, ctx, &bd);

    return bd;
}

PetscErrorCode DMSetAuxiliaryVec_UW(DM dm, DMLabel label, PetscInt value, PetscInt part, Vec aux)
{
    return DMSetAuxiliaryVec(dm, label, value, part, aux);
}

// copy paste function signitures from $PETSC_DIR/include/petscds.h - would be nice to automate this.
#define UW_SIG_F0 PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]
#define UW_SIG_G0 PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]

PetscErrorCode UW_PetscDSSetBdTerms(PetscDS ds, DMLabel label, PetscInt label_val, PetscInt bd,
                                    PetscInt f, PetscInt g, PetscInt part,
                                    void (*bcFunc_f0)(UW_SIG_F0),
                                    void (*bcFunc_f1)(UW_SIG_F0),
                                    void (*bcFunc_g0)(UW_SIG_G0),
                                    void (*bcFunc_g1)(UW_SIG_G0),
                                    void (*bcFunc_g2)(UW_SIG_G0),
                                    void (*bcFunc_g3)(UW_SIG_G0))

{
    PetscWeakForm wf;

    int idx0 = 0;
    int idx1 = 0;

    PetscCall(PetscDSGetBoundary(ds, bd, &wf, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
    // PetscCall(PetscDSGetWeakForm(ds, &wf));
    PetscCall(PetscWeakFormAddBdResidual(wf, label, label_val, f, part, bcFunc_f0, bcFunc_f1));
    // PetscCall(PetscWeakFormSetIndexBdResidual(wf, label, label_val, f, part, idx0, bcFunc_f0, idx1, bcFunc_f1));
    //  PetscCall(PetscWeakFormAddBdJacobian(wf, label, label_val, f, g, part, bcFunc_g0, bcFunc_g1, bcFunc_g2, bcFunc_g3));
    //  PetscCall(PetscWeakFormAddBdJacobianPreconditioner(wf, label, label_val, f, g, part, bcFunc_g0, bcFunc_g1, bcFunc_g2, bcFunc_g3));

    return 1;
}

// These use the older interface :

PetscErrorCode
UW_PetscDSSetBdResidual(PetscDS ds, DMLabel label, PetscInt label_val, PetscInt bd,
                        PetscInt f, PetscInt part,
                        void (*bcFunc_f0)(UW_SIG_F0),
                        void (*bcFunc_f1)(UW_SIG_F0))
{

    PetscWeakForm wf;

    PetscCall(PetscDSGetBoundary(ds, bd, &wf, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
    PetscCall(PetscWeakFormAddBdResidual(wf, label, label_val, f, part, bcFunc_f0, bcFunc_f1));

    return 1;
}

PetscErrorCode UW_PetscDSSetBdJacobian(PetscDS ds, DMLabel label, PetscInt label_val, PetscInt bd,
                                       PetscInt f, PetscInt g, PetscInt part,
                                       void (*bcFunc_g0)(UW_SIG_G0),
                                       void (*bcFunc_g1)(UW_SIG_G0),
                                       void (*bcFunc_g2)(UW_SIG_G0),
                                       void (*bcFunc_g3)(UW_SIG_G0))
{
    PetscWeakForm wf;

    PetscCall(PetscDSGetBoundary(ds, bd, &wf, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
    PetscCall(PetscWeakFormAddBdJacobian(wf, label, label_val, f, g, part, bcFunc_g0, bcFunc_g1, bcFunc_g2, bcFunc_g3));

    return 1;
}

PetscErrorCode UW_PetscDSSetBdJacobianPreconditioner(PetscDS ds, DMLabel label, PetscInt label_val, PetscInt bd,
                                                     PetscInt f, PetscInt g, PetscInt part,
                                                     void (*bcFunc_g0)(UW_SIG_G0),
                                                     void (*bcFunc_g1)(UW_SIG_G0),
                                                     void (*bcFunc_g2)(UW_SIG_G0),
                                                     void (*bcFunc_g3)(UW_SIG_G0))
{
    PetscWeakForm wf;

    PetscCall(PetscDSGetBoundary(ds, bd, &wf, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
    PetscCall(PetscWeakFormAddBdJacobianPreconditioner(wf, label, label_val, f, g, part, bcFunc_g0, bcFunc_g1, bcFunc_g2, bcFunc_g3));

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
