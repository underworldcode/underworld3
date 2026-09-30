/*
 * UWDELTA: a PetscSpace of Kronecker deltas at the points of a quadrature rule.
 *
 * Registered as a PetscSpace type ("uwdelta") through PETSc's plugin API, so
 * it works on any PETSc build, stock conda packages included. It is the
 * prime space of the quadrature-point finite element: tabulated on its own
 * rule the basis is the identity, tabulated anywhere else (a face rule, a
 * different cell rule) it is zero, and derivatives are always zero.
 *
 * PETSc's own PETSCSPACEPOINT is the same idea but (as of 3.25) it errors
 * unless asked for exactly its own points in its own order, which breaks
 * PetscFESetUp (one point per functional), face tabulation in PetscDSSetUp
 * and boundary integrals over auxiliary fields. That is why this type exists.
 *
 * Header-only: include from exactly one extension module.
 */
#ifndef UW_DELTA_SPACE_H
#define UW_DELTA_SPACE_H

#include <petsc.h>
#include <petsc/private/petscfeimpl.h>

#define UWDELTA_TOL 1.0e-10

typedef struct {
  PetscQuadrature quad; /* the rule whose points carry the deltas */
} UWDeltaSpace;

static PetscErrorCode UWDeltaSpace_Destroy(PetscSpace sp)
{
  UWDeltaSpace *dl = (UWDeltaSpace *)sp->data;

  PetscFunctionBegin;
  PetscCall(PetscQuadratureDestroy(&dl->quad));
  PetscCall(PetscFree(dl));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode UWDeltaSpace_SetUp(PetscSpace sp)
{
  UWDeltaSpace *dl = (UWDeltaSpace *)sp->data;

  PetscFunctionBegin;
  PetscCheck(dl->quad, PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_WRONGSTATE, "UWDELTA space has no points: call UWDeltaSpaceSetPoints() first");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode UWDeltaSpace_View(PetscSpace sp, PetscViewer viewer)
{
  UWDeltaSpace *dl = (UWDeltaSpace *)sp->data;
  PetscBool     isascii;
  PetscInt      Nq = 0;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    if (dl->quad) PetscCall(PetscQuadratureGetData(dl->quad, NULL, NULL, &Nq, NULL, NULL));
    PetscCall(PetscViewerASCIIPrintf(viewer, "UWDELTA space in dimension %" PetscInt_FMT " on %" PetscInt_FMT " points\n", sp->Nv, Nq));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode UWDeltaSpace_GetDimension(PetscSpace sp, PetscInt *dim)
{
  UWDeltaSpace *dl = (UWDeltaSpace *)sp->data;
  PetscInt      Nq = 0;

  PetscFunctionBegin;
  if (dl->quad) PetscCall(PetscQuadratureGetData(dl->quad, NULL, NULL, &Nq, NULL, NULL));
  *dim = Nq;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* B is laid out [point][basis][component] as for every PetscSpace. Basis i is
   the delta at the space's point i: a requested point coincident with point i
   gives e_i, any other point gives zero. All components share the basis. */
static PetscErrorCode UWDeltaSpace_Evaluate(PetscSpace sp, PetscInt npoints, const PetscReal points[], PetscReal B[], PetscReal D[], PetscReal H[])
{
  UWDeltaSpace    *dl  = (UWDeltaSpace *)sp->data;
  const PetscInt   dim = sp->Nv, Nc = sp->Nc;
  const PetscReal *qp;
  PetscInt         pdim = 0, p, i, d, c;

  PetscFunctionBegin;
  PetscCheck(dl->quad, PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_WRONGSTATE, "UWDELTA space has no points");
  PetscCall(PetscQuadratureGetData(dl->quad, NULL, NULL, &pdim, &qp, NULL));
  if (B) {
    PetscCall(PetscArrayzero(B, npoints * pdim * Nc));
    for (p = 0; p < npoints; ++p) {
      for (i = 0; i < pdim; ++i) {
        for (d = 0; d < dim; ++d) {
          if (PetscAbsReal(points[p * dim + d] - qp[i * dim + d]) > UWDELTA_TOL) break;
        }
        if (d >= dim) {
          for (c = 0; c < Nc; ++c) B[(p * pdim + i) * Nc + c] = 1.0;
          break;
        }
      }
    }
  }
  if (D) PetscCall(PetscArrayzero(D, npoints * pdim * Nc * dim));
  if (H) PetscCall(PetscArrayzero(H, npoints * pdim * Nc * dim * dim));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSpaceCreate_UWDelta(PetscSpace sp)
{
  UWDeltaSpace *dl;

  PetscFunctionBegin;
  PetscCall(PetscNew(&dl));
  dl->quad      = NULL;
  sp->data      = dl;
  sp->maxDegree = PETSC_INT_MAX;

  sp->ops->setfromoptions    = NULL;
  sp->ops->setup             = UWDeltaSpace_SetUp;
  sp->ops->view              = UWDeltaSpace_View;
  sp->ops->destroy           = UWDeltaSpace_Destroy;
  sp->ops->getdimension      = UWDeltaSpace_GetDimension;
  sp->ops->evaluate          = UWDeltaSpace_Evaluate;
  sp->ops->getheightsubspace = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Idempotent: PETSc's function list rejects nothing on re-registration, but
   registering once per process keeps the list clean. */
static PetscErrorCode UWDeltaSpaceRegister(void)
{
  static PetscBool registered = PETSC_FALSE;

  PetscFunctionBegin;
  if (!registered) {
    PetscCall(PetscSpaceRegister("uwdelta", PetscSpaceCreate_UWDelta));
    registered = PETSC_TRUE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Set the rule whose points carry the deltas (duplicated; caller keeps its own). */
static PetscErrorCode UWDeltaSpaceSetPoints(PetscSpace sp, PetscQuadrature q)
{
  UWDeltaSpace *dl = (UWDeltaSpace *)sp->data;
  PetscBool     isdelta;
  PetscInt      qdim;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)sp, "uwdelta", &isdelta));
  PetscCheck(isdelta, PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_WRONG, "Space is not of type uwdelta");
  PetscCall(PetscQuadratureGetData(q, &qdim, NULL, NULL, NULL, NULL));
  PetscCheck(qdim == sp->Nv, PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_INCOMP, "Rule dimension %" PetscInt_FMT " != space variables %" PetscInt_FMT, qdim, sp->Nv);
  PetscCall(PetscQuadratureDestroy(&dl->quad));
  PetscCall(PetscQuadratureDuplicate(q, &dl->quad));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#endif /* UW_DELTA_SPACE_H */
