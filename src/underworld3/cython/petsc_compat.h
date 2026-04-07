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

    // int idx0 = 0;
    // int idx1 = 0;

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

// Set the time value on a DM. This is passed as `petsc_t` to all
// pointwise residual and Jacobian functions during assembly.
// PETSc stores this internally but petsc4py doesn't expose it.
PetscErrorCode UW_DMSetTime(DM dm, PetscReal time)
{
    // DMSetOutputSequenceNumber stores (step, time) on the DM.
    // The time component is what DMPlexComputeResidual_Internal
    // passes as petsc_t to the pointwise functions.
    PetscInt step;
    PetscReal old_time;
    PetscCall(DMGetOutputSequenceNumber(dm, &step, &old_time));
    PetscCall(DMSetOutputSequenceNumber(dm, step, time));
    return PETSC_SUCCESS;
}

PetscErrorCode UW_DMPlexSetSNESLocalFEM(DM dm, PetscBool flag, void *ctx)
{

#if PETSC_VERSION_LE(3, 20, 5)
    return DMPlexSetSNESLocalFEM(dm, NULL, NULL, NULL);
#else
    return DMPlexSetSNESLocalFEM(dm, flag, NULL);
#endif
}

// Simplified wrapper for DMPlexComputeBdIntegral.
// Takes a single boundary pointwise function (for field 0) instead of an Nf-element array.
//
// PETSc's DMPlexComputeBdIntegral returns only the local contribution (no MPI
// reduction), so this wrapper adds an Allreduce to produce the global integral.
//
// Ghost facet filtering is handled by our PETSc patch
// (plexfem-internal-boundary-ownership-fix.patch) which filters SF leaves
// inside DMPlexComputeBdIntegral, DMPlexComputeBdResidual_Internal, and
// DMPlexComputeBdJacobian_Internal. The patch also fixes part-consistent
// assembly (support[key.part]) for internal boundary residuals/Jacobians.
PetscErrorCode UW_DMPlexComputeBdIntegral(DM dm, Vec X,
                                          DMLabel label, PetscInt numVals, const PetscInt vals[],
                                          void (*func)(UW_SIG_F0),
                                          PetscScalar *result,
                                          void *ctx)
{
    PetscSection  section;
    PetscInt      Nf;
    PetscInt      localCount = 0;

    PetscFunctionBeginUser;

    PetscCall(DMGetLocalSection(dm, &section));
    PetscCall(PetscSectionGetNumFields(section, &Nf));

    // If the label is NULL or the requested boundary has no local entities on
    // this rank, contribute 0 but still participate in the MPI Allreduce to
    // avoid hangs. Parallel DMPlex boundary assembly can deadlock if some
    // ranks enter with an empty local stratum.
    if (label) {
        for (PetscInt i = 0; i < numVals; ++i) {
            PetscInt stratumSize = 0;
            PetscCall(DMLabelGetStratumSize(label, vals[i], &stratumSize));
            localCount += stratumSize;
        }
    }

    if (!label || localCount == 0) {
        PetscScalar zero = 0.0;
        PetscCallMPI(MPIU_Allreduce(&zero, result, 1, MPIU_SCALAR, MPIU_SUM,
                                    PetscObjectComm((PetscObject)dm)));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    // Build Nf-element function pointer array (only field 0 has a callback)
    void (**funcs)(UW_SIG_F0);
    PetscCall(PetscCalloc1(Nf, &funcs));
    funcs[0] = func;

    PetscScalar *integral;
    PetscCall(PetscCalloc1(Nf, &integral));

    // PETSc changed DMPlexComputeBdIntegral signature in v3.22.0:
    //   <= 3.21.x: void (*func)(...)     — single function pointer
    //   >= 3.22.0: void (**funcs)(...)    — array of Nf function pointers
#if PETSC_VERSION_GE(3, 22, 0)
    PetscCall(DMPlexComputeBdIntegral(dm, X, label, numVals, vals, funcs, integral, ctx));
#else
    PetscCall(DMPlexComputeBdIntegral(dm, X, label, numVals, vals, funcs[0], integral, ctx));
#endif

    // MPI reduction — sum local owned contributions across all ranks
    PetscScalar global_val;
    PetscCallMPI(MPIU_Allreduce(&integral[0], &global_val, 1, MPIU_SCALAR, MPIU_SUM,
                                PetscObjectComm((PetscObject)dm)));
    *result = global_val;

    PetscCall(PetscFree(funcs));
    PetscCall(PetscFree(integral));

    PetscFunctionReturn(PETSC_SUCCESS);
}
