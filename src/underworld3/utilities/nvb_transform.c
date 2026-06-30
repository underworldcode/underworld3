/*
  Newest-vertex bisection (NVB) as a native DMPlexTransform — UW extension.

  Route B of docs/developer/design/NVB_GRADED_ADAPT.md: a graded, parallel,
  SF-preserving simplicial refinement transform registered into PETSc from UW
  (no PETSc rebuild). NVB is a small delta from PETSc's SBR transform
  (src/dm/impls/plex/transform/impls/refine/sbr/plexrefsbr.c): identical
  bisection geometry + the same DMLabelPropagate cross-rank closure; the only
  NVB-specific change is the edge chosen to split (newest-vertex refinement edge
  via a per-vertex "age" label, vs SBR's geometric longest edge).

  STAGE 1 (this file, current): the INFRA GATE — register "uwnvb" and prove the
  build / registration / private-header / parallel-SF stack with an identity
  transform. The two PETSc helpers the real transform needs that are NOT exported
  from libpetsc (SetDimensions, MapCoordinatesBarycenter) are reimplemented here
  (both are a few lines); the closure (DMLabelPropagate*) and registration are
  exported and linked directly. STAGE 2 will replace SetUp_UWNVB / the cell
  transform with the newest-vertex bisection logic.
*/
#include <petsc/private/dmplextransformimpl.h> /*I "petscdmplextransform.h" I*/
#include <petscsf.h>

typedef struct {
  DMLabel splitPoints; /* STAGE 2: edges (1) / triangles (2) to split */
} DMPlexTransform_UWNVB;

/* --- reimplemented (non-exported) helpers ------------------------------- */
static PetscErrorCode DMPlexTransformSetDimensions_UWNVB(DMPlexTransform tr, DM dm, DM tdm)
{
  PetscInt dim, cdim;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMSetDimension(tdm, dim));
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCall(DMSetCoordinateDim(tdm, cdim));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* New vertices land at the barycentre of their source point's vertices — for an
   edge that is exactly the midpoint (NVB's bisection vertex). */
static PetscErrorCode DMPlexTransformMapCoordinates_UWNVB(DMPlexTransform tr, DMPolytopeType pct, DMPolytopeType ct, PetscInt p, PetscInt r, PetscInt Nv, PetscInt dE, const PetscScalar in[], PetscScalar out[])
{
  PetscInt v, d;

  PetscFunctionBeginHot;
  PetscCheck(ct == DM_POLYTOPE_POINT, PETSC_COMM_SELF, PETSC_ERR_SUP, "Not for refined point type %s", DMPolytopeTypes[ct]);
  for (d = 0; d < dE; ++d) out[d] = 0.0;
  for (v = 0; v < Nv; ++v)
    for (d = 0; d < dE; ++d) out[d] += in[v * dE + d];
  for (d = 0; d < dE; ++d) out[d] /= Nv;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* --- transform ops ------------------------------------------------------ */
static PetscErrorCode DMPlexTransformSetUp_UWNVB(DMPlexTransform tr)
{
  /* STAGE 1: identity — no split points. STAGE 2: build the newest-vertex
     splitPoints label + run the DMLabelPropagate cross-rank closure (see the
     SBR template + NVB_GRADED_ADAPT.md). */
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTransformDestroy_UWNVB(DMPlexTransform tr)
{
  DMPlexTransform_UWNVB *nvb = (DMPlexTransform_UWNVB *)tr->data;

  PetscFunctionBegin;
  PetscCall(DMLabelDestroy(&nvb->splitPoints));
  PetscCall(PetscFree(tr->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTransformInitialize_UWNVB(DMPlexTransform tr)
{
  PetscFunctionBegin;
  tr->ops->setup                 = DMPlexTransformSetUp_UWNVB;
  tr->ops->destroy               = DMPlexTransformDestroy_UWNVB;
  tr->ops->setdimensions         = DMPlexTransformSetDimensions_UWNVB;
  tr->ops->celltransform         = DMPlexTransformCellTransformIdentity;        /* STAGE 2: NVB bisection */
  tr->ops->getsubcellorientation = DMPlexTransformGetSubcellOrientationIdentity;
  tr->ops->mapcoordinates        = DMPlexTransformMapCoordinates_UWNVB;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode DMPlexTransformCreate_UWNVB(DMPlexTransform tr)
{
  DMPlexTransform_UWNVB *nvb;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscCall(PetscNew(&nvb));
  tr->data = nvb;
  PetscCall(DMPlexTransformInitialize_UWNVB(tr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Idempotent registration entry called from the Cython module on import. */
PetscErrorCode UWNVBTransformRegister(void)
{
  PetscFunctionBegin;
  PetscCall(DMPlexTransformRegister("uwnvb", DMPlexTransformCreate_UWNVB));
  PetscFunctionReturn(PETSC_SUCCESS);
}
