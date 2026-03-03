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
// Fixes two issues in DMPlexComputeBdIntegral:
//   1. Missing MPI Allreduce (plexfem.c returns local contribution only,
//      unlike DMPlexComputeIntegralFEM which reduces at plexfem.c:2633).
//   2. Ghost facet double-counting: DMPlexComputeBdIntegral iterates over ALL
//      label stratum points including ghost facets, so shared boundary facets
//      get integrated on both the owning rank and the ghost rank. We create a
//      temporary label containing only owned (non-ghost) boundary points.
PetscErrorCode UW_DMPlexComputeBdIntegral(DM dm, Vec X,
                                          DMLabel label, PetscInt numVals, const PetscInt vals[],
                                          void (*func)(UW_SIG_F0),
                                          PetscScalar *result,
                                          void *ctx)
{
    PetscSection  section;
    PetscInt      Nf, pStart, pEnd;

    PetscFunctionBeginUser;

    PetscCall(DMGetLocalSection(dm, &section));
    PetscCall(PetscSectionGetNumFields(section, &Nf));

    // --- Build a ghost-point bitset from the point SF ---
    PetscSF   sf;
    PetscInt  nleaves;
    const PetscInt *ilocal;
    PetscCall(DMGetPointSF(dm, &sf));
    PetscCall(PetscSFGetGraph(sf, NULL, &nleaves, &ilocal, NULL));
    PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));

    PetscBT ghostBT;
    PetscCall(PetscBTCreate(pEnd - pStart, &ghostBT));
    if (ilocal) {
        for (PetscInt i = 0; i < nleaves; i++) {
            PetscCall(PetscBTSet(ghostBT, ilocal[i] - pStart));
        }
    }

    // --- Create a temporary label with only owned boundary points ---
    DMLabel ownedLabel;
    PetscCall(DMLabelCreate(PetscObjectComm((PetscObject)dm), "uw_owned_bd", &ownedLabel));

    for (PetscInt v = 0; v < numVals; v++) {
        IS origIS;
        PetscCall(DMLabelGetStratumIS(label, vals[v], &origIS));
        if (!origIS) continue;

        PetscInt        n;
        const PetscInt *indices;
        PetscCall(ISGetLocalSize(origIS, &n));
        PetscCall(ISGetIndices(origIS, &indices));

        for (PetscInt i = 0; i < n; i++) {
            if (!PetscBTLookup(ghostBT, indices[i] - pStart)) {
                PetscCall(DMLabelSetValue(ownedLabel, indices[i], vals[v]));
            }
        }
        PetscCall(ISRestoreIndices(origIS, &indices));
        PetscCall(ISDestroy(&origIS));
    }
    PetscCall(PetscBTDestroy(&ghostBT));

    // --- Compute boundary integral over owned points only ---
    void (**funcs)(UW_SIG_F0);
    PetscCall(PetscCalloc1(Nf, &funcs));
    funcs[0] = func;

    PetscScalar *integral;
    PetscCall(PetscCalloc1(Nf, &integral));

    PetscCall(DMPlexComputeBdIntegral(dm, X, ownedLabel, numVals, vals, funcs, integral, ctx));

    // --- MPI reduction (sum local owned contributions across all ranks) ---
    PetscScalar global_val;
    PetscCallMPI(MPIU_Allreduce(&integral[0], &global_val, 1, MPIU_SCALAR, MPIU_SUM,
                                PetscObjectComm((PetscObject)dm)));
    *result = global_val;

    PetscCall(DMLabelDestroy(&ownedLabel));
    PetscCall(PetscFree(funcs));
    PetscCall(PetscFree(integral));

    PetscFunctionReturn(PETSC_SUCCESS);
}
