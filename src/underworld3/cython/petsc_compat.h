#include "petsc.h"

// Version-compatible wrapper for DMPlexFilter.
// PETSc 3.25 added an MPI_Comm argument before the SF pointer.
static inline PetscErrorCode UW_DMPlexFilter(DM dm, DMLabel label, PetscInt value,
                                             PetscBool useClosure, PetscBool ignoreClosure,
                                             DM *subdm)
{
#if PETSC_VERSION_GE(3, 25, 0)
    return DMPlexFilter(dm, label, value, useClosure, ignoreClosure,
                        PetscObjectComm((PetscObject)dm), NULL, subdm);
#else
    return DMPlexFilter(dm, label, value, useClosure, ignoreClosure, NULL, subdm);
#endif
}

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

PetscErrorCode UW_PetscDSGetBoundaryWeakForm(PetscDS ds, PetscInt bd, PetscWeakForm *wf)
{
    PetscCall(PetscDSGetBoundary(ds, bd, wf, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
    return PETSC_SUCCESS;
}

PetscErrorCode UW_DMPlexComputeResidualByKeyVolumeOnly(DM dm, PetscFormKey key, IS cellIS, PetscReal time, Vec locX, Vec locX_t, PetscReal t, Vec locF, void *ctx)
{
    DM vdm = NULL;
    PetscSection section = NULL;
    PetscSection global_section = NULL;
    PetscDS ds = NULL;
    PetscDS vds = NULL;
    Vec aux = NULL;

    PetscCall(DMClone(dm, &vdm));
    PetscCall(DMGetLocalSection(dm, &section));
    PetscCall(DMSetLocalSection(vdm, section));
    PetscCall(DMGetGlobalSection(dm, &global_section));
    PetscCall(DMSetGlobalSection(vdm, global_section));
    PetscCall(DMCopyFields(dm, PETSC_DETERMINE, PETSC_DETERMINE, vdm));
    PetscCall(DMCreateDS(vdm));
    PetscCall(DMGetDS(dm, &ds));
    PetscCall(DMGetDS(vdm, &vds));
    PetscCall(PetscDSCopyConstants(ds, vds));
    PetscCall(PetscDSCopyEquations(ds, vds));
    PetscCall(DMGetAuxiliaryVec(dm, key.label, key.value, key.part, &aux));
    if (aux) PetscCall(DMSetAuxiliaryVec(vdm, key.label, key.value, key.part, aux));
    PetscCall(DMPlexComputeResidualByKey(vdm, key, cellIS, time, locX, locX_t, t, locF, ctx));
    PetscCall(DMDestroy(&vdm));
    return PETSC_SUCCESS;
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

// Issue #96 fix: Force coordinate field creation on a DM and strip
// boundary labels from the coordinate DM so they don't cause MPI errors
// in DMCompleteBCLabels_Internal during lazy coordinate field recreation.
//
// Must be called AFTER createCoordinateSpace and AFTER labels are added.
// The coordinate field is created NOW (while we can clean the coord DM)
// and cached, preventing PETSc from lazily recreating it later.
PetscErrorCode UW_DMForceCoordinateField(DM dm)
{
    DMField        coordField;
    DM             cdm;
    PetscInt       numLabels, i;

    PetscFunctionBeginUser;

    // Force coordinate field creation (triggers DMCreateCoordinateField_Plex)
    PetscCall(DMGetCoordinateField(dm, &coordField));

    // Now strip non-essential labels from the coordinate DM
    // (DMClone copied all of mesh.dm's labels, including boundary labels)
    PetscCall(DMGetCoordinateDM(dm, &cdm));
    PetscCall(DMGetNumLabels(cdm, &numLabels));
    for (i = numLabels - 1; i >= 0; --i) {
        const char *name;
        PetscBool   isDepth, isCelltype;
        PetscCall(DMGetLabelName(cdm, i, &name));
        PetscCall(PetscStrcmp(name, "depth", &isDepth));
        PetscCall(PetscStrcmp(name, "celltype", &isCelltype));
        if (!isDepth && !isCelltype) {
            PetscCall(DMRemoveLabel(cdm, name, NULL));
        }
    }

    PetscFunctionReturn(PETSC_SUCCESS);
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
// IMPORTANT: DMPlexComputeBdIntegral is collective (it calls DMGlobalToLocal
// internally). All ranks MUST call it, even if their local boundary stratum is
// empty — PETSc handles empty strata gracefully (skips the per-face loop).
// Skipping the call on empty ranks causes a deadlock because the other ranks
// block inside DMGlobalToLocal waiting for the missing participants.
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

    PetscFunctionBeginUser;

    PetscCall(DMGetLocalSection(dm, &section));
    PetscCall(PetscSectionGetNumFields(section, &Nf));

    // NULL label means no boundary was found at all — no rank has work.
    // Safe to return early since no rank enters the collective PETSc call.
    if (!label) {
        *result = 0.0;
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    // Build Nf-element function pointer array (only field 0 has a callback)
    void (**funcs)(UW_SIG_F0);
    PetscCall(PetscCalloc1(Nf, &funcs));
    funcs[0] = func;

    PetscScalar *integral;
    PetscCall(PetscCalloc1(Nf, &integral));

    // All ranks must call DMPlexComputeBdIntegral — it is collective.
    // Ranks with empty local strata will simply contribute 0.
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

// Create a sandbox DM for BdIntegral that won't contaminate the
// original DM's shared DM_Plex caches.
//
// DMClone shares DM_Plex by refcount, and DMPlexComputeBdIntegral
// lazily initialises height-trace FE caches on the shared struct.
// This corrupts any solver DM cloned from the same mesh.
//
// Strategy: DMClone (cheap, shares topology) then rebuild the
// coordinate field from scratch.  The fresh coordinate field has its
// own DMField_DS with empty height-trace caches, so the lazy init
// during BdIntegral writes to the sandbox's caches, not the original's.
PetscErrorCode UW_DMCreateBdIntegralSandbox(DM src, DM *sandbox)
{
    DM            sdm;
    PetscInt      Nf, i;
    PetscSection  srcSec;
    Vec           auxVec;

    PetscFunctionBeginUser;

    // DMClone shares DM_Plex topology, point SF, and labels by reference.
    // The topology itself is read-only — the problem is the coordinate
    // DMField's lazily-cached height-trace FEs.  We replace the coordinate
    // field below so those caches are independent.
    PetscCall(DMClone(src, &sdm));

    // Rebuild coordinate space from scratch — creates a new coordinate
    // DMField_DS with empty height-trace caches, independent of src's.
    // Degree 1 (P1/Q1) matches UW3's mesh coordinate convention.
    // Signature changed in PETSc 3.25: (dm, degree, localized, project)
    // vs 3.24: (dm, degree, project, snapFunc)
#if PETSC_VERSION_GE(3, 25, 0)
    PetscCall(DMPlexCreateCoordinateSpace(sdm, 1, PETSC_FALSE, PETSC_TRUE));
#else
    PetscCall(DMPlexCreateCoordinateSpace(sdm, 1, PETSC_FALSE, NULL));
#endif
    PetscCall(UW_DMForceCoordinateField(sdm));

    // Replicate the field layout and section from src so that the
    // solution vector is compatible with the sandbox's DS.
    PetscCall(DMGetLocalSection(src, &srcSec));
    PetscCall(PetscSectionGetNumFields(srcSec, &Nf));
    for (i = 0; i < Nf; ++i) {
        PetscObject obj;
        DMLabel     label;
        PetscCall(DMGetField(src, i, &label, &obj));
        PetscCall(DMSetField(sdm, i, label, obj));
    }
    PetscCall(DMCreateDS(sdm));
    PetscCall(DMSetLocalSection(sdm, srcSec));

    // Copy auxiliary data (viscosity, etc.) — just a pointer, no mutation
    PetscCall(DMGetAuxiliaryVec(src, NULL, 0, 0, &auxVec));
    if (auxVec) {
        PetscCall(DMSetAuxiliaryVec(sdm, NULL, 0, 0, auxVec));
    }

    *sandbox = sdm;
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Interpolation between two DMs on ONE plex (a coordinate DM and a basis DM
// cloned from it, a variable's sub-DM and a scratch clone of the mesh): the
// matrix DMCreateInterpolation builds through DMPlexComputeInterpolatorNested
// (isRefined = PETSC_FALSE), copied from PETSc v3.25.0 (BSD-2-Clause) with one
// change (#902). PETSc runs its preallocation loop only on a rank with local
// rows or columns (`if (locRows || locCols)`, since b4ec2976050, v3.24.0) but
// its fill loop on every rank that holds cells. A shared point belongs to the
// highest rank holding it, so a rank whose cells lie inside higher ranks'
// regions owns no rows and no columns; its cells' entries then reach owners
// that never preallocated them ("New nonzero caused a malloc" in
// MatAssemblyEnd) and the ranks that did not raise wait in the assembly's
// collective. Here both loops run over the same cells. Finite-element fields
// only, which are the only ones UW3 interpolates this way. Drop this for
// DMCreateInterpolation once PETSc runs the two loops alike.
PetscErrorCode UW_DMCreateInterpolationWithinPlex(DM dmc, DM dmf, Mat *In)
{
    PetscSection fsection, fglobalSection, csection, cglobalSection;
    PetscScalar *elemMat, *vals;
    PetscFE     *fe;
    PetscInt     dim, Nf, f, fieldI, fieldJ, offsetI, offsetJ, cStart, cEnd, c;
    PetscInt     cTotDim = 0, rTotDim = 0, m, n, locRows, locCols;
    MatType      mtype;
    PetscBool    ismatis;
    Mat          preallocator;

    PetscFunctionBeginUser;
    PetscCall(DMGetGlobalSection(dmf, &fglobalSection));
    PetscCall(PetscSectionGetConstrainedStorageSize(fglobalSection, &m));
    PetscCall(DMGetGlobalSection(dmc, &cglobalSection));
    PetscCall(PetscSectionGetConstrainedStorageSize(cglobalSection, &n));
    PetscCall(DMGetMatType(dmc, &mtype));
    PetscCall(PetscStrcmp(mtype, MATIS, &ismatis));
    PetscCall(MatCreate(PetscObjectComm((PetscObject)dmc), In));
    PetscCall(MatSetSizes(*In, m, n, PETSC_DETERMINE, PETSC_DETERMINE));
    PetscCall(MatSetType(*In, ismatis ? MATAIJ : mtype));

    PetscCall(DMGetDimension(dmf, &dim));
    PetscCall(DMGetLocalSection(dmf, &fsection));
    PetscCall(DMGetLocalSection(dmc, &csection));
    PetscCall(PetscSectionGetNumFields(fsection, &Nf));
    PetscCall(DMPlexGetSimplexOrBoxCells(dmc, 0, &cStart, &cEnd));
    PetscCall(PetscCalloc1(Nf, &fe));
    for (f = 0; f < Nf; ++f) {
        PetscObject  obj, objc;
        PetscClassId id, idc;
        PetscInt     rNb, cNb;

        PetscCall(DMGetField(dmf, f, NULL, &obj));
        PetscCall(PetscObjectGetClassId(obj, &id));
        PetscCall(DMGetField(dmc, f, NULL, &objc));
        PetscCall(PetscObjectGetClassId(objc, &idc));
        PetscCheck(id == PETSCFE_CLASSID && idc == PETSCFE_CLASSID, PetscObjectComm((PetscObject)dmc),
                   PETSC_ERR_SUP, "UW_DMCreateInterpolationWithinPlex: field %" PetscInt_FMT " is not a PetscFE", f);
        fe[f] = (PetscFE)obj;
        PetscCall(PetscFEGetDimension(fe[f], &rNb));
        PetscCall(PetscFEGetDimension((PetscFE)objc, &cNb));
        rTotDim += rNb;
        cTotDim += cNb;
    }
    PetscCall(PetscCalloc1(rTotDim * cTotDim, &elemMat));
    for (fieldI = 0, offsetI = 0; fieldI < Nf; ++fieldI) {
        PetscDualSpace   Qref;
        PetscQuadrature  quad;
        const PetscReal *qpoints, *qweights;
        PetscReal       *points;
        PetscInt         npoints = 0, Nc, Np, fpdim, i, k, p, d;

        // the fine field's dual basis: every functional's quadrature points
        PetscCall(PetscFEGetDualSpace(fe[fieldI], &Qref));
        PetscCall(PetscFEGetNumComponents(fe[fieldI], &Nc));
        PetscCall(PetscDualSpaceGetDimension(Qref, &fpdim));
        for (i = 0; i < fpdim; ++i) {
            PetscCall(PetscDualSpaceGetFunctional(Qref, i, &quad));
            PetscCall(PetscQuadratureGetData(quad, NULL, NULL, &Np, NULL, NULL));
            npoints += Np;
        }
        PetscCall(PetscMalloc1(npoints * dim, &points));
        for (i = 0, k = 0; i < fpdim; ++i) {
            PetscCall(PetscDualSpaceGetFunctional(Qref, i, &quad));
            PetscCall(PetscQuadratureGetData(quad, NULL, NULL, &Np, &qpoints, NULL));
            for (p = 0; p < Np; ++p, ++k)
                for (d = 0; d < dim; ++d) points[k * dim + d] = qpoints[p * dim + d];
        }
        for (fieldJ = 0, offsetJ = 0; fieldJ < Nf; ++fieldJ) {
            PetscObject     obj;
            PetscFE         fec;
            PetscTabulation T = NULL;
            PetscInt        NcJ, cpdim, j, qNc;

            PetscCall(DMGetField(dmc, fieldJ, NULL, &obj));
            fec = (PetscFE)obj;
            PetscCall(PetscFEGetNumComponents(fec, &NcJ));
            PetscCall(PetscFEGetDimension(fec, &cpdim));
            // fields only interpolate themselves: the coarse basis evaluated at
            // the fine functionals' points, weighted by the functionals
            if (fieldI == fieldJ) {
                PetscCheck(Nc == NcJ, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of components in fine space field %" PetscInt_FMT " does not match coarse field %" PetscInt_FMT, Nc, NcJ);
                PetscCall(PetscFECreateTabulation(fec, 1, npoints, points, 0, &T));
                for (i = 0, k = 0; i < fpdim; ++i) {
                    PetscCall(PetscDualSpaceGetFunctional(Qref, i, &quad));
                    PetscCall(PetscQuadratureGetData(quad, NULL, &qNc, &Np, NULL, &qweights));
                    PetscCheck(qNc == NcJ, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of components in quadrature %" PetscInt_FMT " does not match coarse field %" PetscInt_FMT, qNc, NcJ);
                    for (p = 0; p < Np; ++p, ++k)
                        for (j = 0; j < cpdim; ++j)
                            for (c = 0; c < Nc; ++c) elemMat[(offsetI + i) * cTotDim + offsetJ + j] += T->T[0][k * cpdim * NcJ + j * Nc + c] * qweights[p * qNc + c];
                }
                PetscCall(PetscTabulationDestroy(&T));
            }
            offsetJ += cpdim;
        }
        offsetI += fpdim;
        PetscCall(PetscFree(points));
    }

    // Preallocate from every cell this rank holds, as the fill below does,
    // whether or not the rank owns any rows or columns (the one change).
    PetscCall(MatGetLocalSize(*In, &locRows, &locCols));
    PetscCall(MatCreate(PetscObjectComm((PetscObject)*In), &preallocator));
    PetscCall(MatSetType(preallocator, MATPREALLOCATOR));
    PetscCall(MatSetSizes(preallocator, locRows, locCols, PETSC_DETERMINE, PETSC_DETERMINE));
    PetscCall(MatSetUp(preallocator));
    PetscCall(PetscCalloc1(rTotDim * cTotDim, &vals));
    for (c = cStart; c < cEnd; ++c)
        PetscCall(DMPlexMatSetClosureGeneral(dmf, fsection, fglobalSection, PETSC_FALSE, dmc, csection, cglobalSection, PETSC_FALSE, preallocator, c, vals, INSERT_VALUES));
    PetscCall(PetscFree(vals));
    PetscCall(MatAssemblyBegin(preallocator, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(preallocator, MAT_FINAL_ASSEMBLY));
    PetscCall(MatPreallocatorPreallocate(preallocator, PETSC_TRUE, *In));
    PetscCall(MatDestroy(&preallocator));

    PetscCall(MatZeroEntries(*In));
    for (c = cStart; c < cEnd; ++c)
        PetscCall(DMPlexMatSetClosureGeneral(dmf, fsection, fglobalSection, PETSC_FALSE, dmc, csection, cglobalSection, PETSC_FALSE, *In, c, elemMat, INSERT_VALUES));
    PetscCall(PetscFree(fe));
    PetscCall(PetscFree(elemMat));
    PetscCall(MatAssemblyBegin(*In, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*In, MAT_FINAL_ASSEMBLY));
    PetscCall(MatViewFromOptions(*In, NULL, "-interp_mat_view"));
    PetscFunctionReturn(PETSC_SUCCESS);
}
