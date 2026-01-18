#include "petsc_tools.h"

/*@C
  DMInterpolationSetUp - Compute spatial indices for point location during interpolation

  Collective on ctx

  Input Parameters:
+ ctx - the context
. dm  - the DM for the function space used for interpolation
. redundantPoints - If PETSC_TRUE, all processes are passing in the same array of points. Otherwise, points need to be communicated among processes.
- ignoreOutsideDomain - If PETSC_TRUE, ignore points outside the domain, otherwise return an error

  Level: intermediate

.seealso: DMInterpolationEvaluate(), DMInterpolationAddPoints(), DMInterpolationCreate()
@*/

PetscErrorCode DMInterpolationSetUp_UW(DMInterpolationInfo ctx, DM dm, PetscBool redundantPoints, PetscBool ignoreOutsideDomain, size_t *owning_cell)
{
  MPI_Comm           comm = ctx->comm;
  PetscScalar       *a;
  PetscInt           p, q, i;
  PetscMPIInt        rank, size;
  Vec                pointVec;
  PetscSF            cellSF;
  PetscLayout        layout;
  PetscReal         *globalPoints;
  PetscScalar       *globalPointsScalar;
  const PetscInt    *ranges;
  PetscMPIInt       *counts, *displs;
  const PetscSFNode *foundCells;
  const PetscInt    *foundPoints;
  PetscMPIInt       *foundProcs, *globalProcs;
  PetscInt           n, N, numFound;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCheck(ctx->dim >= 0, comm, PETSC_ERR_ARG_WRONGSTATE, "The spatial dimension has not been set");
  /* Locate points */
  n = ctx->nInput;
  if (!redundantPoints) {
    PetscCall(PetscLayoutCreate(comm, &layout));
    PetscCall(PetscLayoutSetBlockSize(layout, 1));
    PetscCall(PetscLayoutSetLocalSize(layout, n));
    PetscCall(PetscLayoutSetUp(layout));
    PetscCall(PetscLayoutGetSize(layout, &N));
    /* Communicate all points to all processes */
    PetscCall(PetscMalloc3(N * ctx->dim, &globalPoints, size, &counts, size, &displs));
    PetscCall(PetscLayoutGetRanges(layout, &ranges));
    for (p = 0; p < size; ++p) {
      counts[p] = (ranges[p + 1] - ranges[p]) * ctx->dim;
      displs[p] = ranges[p] * ctx->dim;
    }
    PetscCallMPI(MPI_Allgatherv(ctx->points, n * ctx->dim, MPIU_REAL, globalPoints, counts, displs, MPIU_REAL, comm));
  } else {
    N            = n;
    globalPoints = ctx->points;
    counts = displs = NULL;
    layout          = NULL;
  }
#if 0
  PetscCall(PetscMalloc3(N,&foundCells,N,&foundProcs,N,&globalProcs));
  /* foundCells[p] = m->locatePoint(&globalPoints[p*ctx->dim]); */
#else
  #if defined(PETSC_USE_COMPLEX)
  PetscCall(PetscMalloc1(N * ctx->dim, &globalPointsScalar));
  for (i = 0; i < N * ctx->dim; i++) globalPointsScalar[i] = globalPoints[i];
  #else
  globalPointsScalar = globalPoints;
  #endif
  PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF, ctx->dim, N * ctx->dim, globalPointsScalar, &pointVec));
  PetscCall(PetscMalloc2(N, &foundProcs, N, &globalProcs));
  for (p = 0; p < N; ++p) foundProcs[p] = size;
  cellSF = NULL;
  /* the Underworld code is used to find good guesses for the owning cells */
  if (owning_cell)
  {
    PetscSFNode *sf_cells;
    ierr = PetscMalloc1(N, &sf_cells);
    CHKERRQ(ierr);
    size_t range = 0;
    for (size_t p = 0; p < (size_t)N; p++)
    {
      sf_cells[p].rank = 0;
      sf_cells[p].index = owning_cell[p];
      if (owning_cell[p] > range)
      {
        range = owning_cell[p];
      }
    }
    ierr = PetscSFCreate(PETSC_COMM_SELF, &cellSF);
    CHKERRQ(ierr);
    // PETSC_OWN_POINTER => sf_cells memory control goes to cellSF
    // nroots must be > max(iremote.index), so use range + 1
    ierr = PetscSFSetGraph(cellSF, range + 1, N, NULL, PETSC_OWN_POINTER, sf_cells, PETSC_OWN_POINTER);
    CHKERRQ(ierr);
  }
  PetscCall(DMLocatePoints(dm, pointVec, DM_POINTLOCATION_REMOVE, &cellSF));
  PetscCall(PetscSFGetGraph(cellSF, NULL, &numFound, &foundPoints, &foundCells));
#endif
  for (p = 0; p < numFound; ++p) {
    if (foundCells[p].index >= 0) foundProcs[foundPoints ? foundPoints[p] : p] = rank;
  }
  /* Let the lowest rank process own each point */
  PetscCall(MPIU_Allreduce(foundProcs, globalProcs, N, MPI_INT, MPI_MIN, comm));
  ctx->n = 0;
  for (p = 0; p < N; ++p) {
    if (globalProcs[p] == size) {
      PetscCheck(ignoreOutsideDomain, comm, PETSC_ERR_PLIB, "Point %" PetscInt_FMT ": %g %g %g not located in mesh", p, (double)globalPoints[p * ctx->dim + 0], (double)(ctx->dim > 1 ? globalPoints[p * ctx->dim + 1] : 0.0),
                 (double)(ctx->dim > 2 ? globalPoints[p * ctx->dim + 2] : 0.0));
      if (rank == 0) ++ctx->n;
    } else if (globalProcs[p] == rank) ++ctx->n;
  }
  /* Create coordinates vector and array of owned cells */
  PetscCall(PetscMalloc1(ctx->n, &ctx->cells));
  PetscCall(VecCreate(comm, &ctx->coords));
  PetscCall(VecSetSizes(ctx->coords, ctx->n * ctx->dim, PETSC_DECIDE));
  PetscCall(VecSetBlockSize(ctx->coords, ctx->dim));
  PetscCall(VecSetType(ctx->coords, VECSTANDARD));
  PetscCall(VecGetArray(ctx->coords, &a));
  for (p = 0, q = 0, i = 0; p < N; ++p) {
    if (globalProcs[p] == rank) {
      PetscInt d;

      for (d = 0; d < ctx->dim; ++d, ++i) a[i] = globalPoints[p * ctx->dim + d];
      ctx->cells[q] = foundCells[q].index;
      ++q;
    }
    if (globalProcs[p] == size && rank == 0) {
      PetscInt d;

      for (d = 0; d < ctx->dim; ++d, ++i) a[i] = 0.;
      ctx->cells[q] = -1;
      ++q;
    }
  }
  PetscCall(VecRestoreArray(ctx->coords, &a));
#if 0
  PetscCall(PetscFree3(foundCells,foundProcs,globalProcs));
#else
  PetscCall(PetscFree2(foundProcs, globalProcs));
  PetscCall(PetscSFDestroy(&cellSF));
  PetscCall(VecDestroy(&pointVec));
#endif
  if ((void *)globalPointsScalar != (void *)globalPoints) PetscCall(PetscFree(globalPointsScalar));
  if (!redundantPoints) PetscCall(PetscFree3(globalPoints, counts, displs));
  PetscCall(PetscLayoutDestroy(&layout));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMInterpolationEvaluate - Using the input from dm and x, calculates interpolated field values at the interpolation points.

  Input Parameters:
+ ctx - The DMInterpolationInfo context
. dm  - The DM
- x   - The local vector containing the field to be interpolated

  Output Parameters:
. v   - The vector containing the interpolated values

  Note: A suitable v can be obtained using DMInterpolationGetVector().

  Level: beginner

.seealso: DMInterpolationGetVector(), DMInterpolationAddPoints(), DMInterpolationCreate()
@*/
PetscErrorCode DMInterpolationEvaluate_UW(DMInterpolationInfo ctx, DM dm, Vec x, Vec v)
{
  PetscDS   ds;
  PetscInt  n, p, Nf, field;
  PetscBool useDS = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 3);
  PetscValidHeaderSpecific(v, VEC_CLASSID, 4);
  PetscCall(VecGetLocalSize(v, &n));
  PetscCheck(n == ctx->n * ctx->dof, ctx->comm, PETSC_ERR_ARG_SIZ, "Invalid input vector size %" PetscInt_FMT " should be %" PetscInt_FMT, n, ctx->n * ctx->dof);
  if (!n) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMGetDS(dm, &ds));
  if (ds) {
    useDS = PETSC_TRUE;
    PetscCall(PetscDSGetNumFields(ds, &Nf));
    for (field = 0; field < Nf; ++field) {
      PetscObject  obj;
      PetscClassId id;

      PetscCall(PetscDSGetDiscretization(ds, field, &obj));
      PetscCall(PetscObjectGetClassId(obj, &id));
      if (id != PETSCFE_CLASSID && id != PETSCFV_CLASSID) {
        useDS = PETSC_FALSE;
        break;
      }
    }
  }
  if (useDS) {
    const PetscScalar *coords;
    PetscScalar       *interpolant;
    PetscInt           cdim, d;

    PetscCall(DMGetCoordinateDim(dm, &cdim));
    PetscCall(VecGetArrayRead(ctx->coords, &coords));
    PetscCall(VecGetArrayWrite(v, &interpolant));
    for (p = 0; p < ctx->n; ++p) {
      PetscReal    pcoords[3], xi[3];
      PetscScalar *xa   = NULL;
      PetscInt     coff = 0, foff = 0, clSize;

      if (ctx->cells[p] < 0) continue;
      for (d = 0; d < cdim; ++d) pcoords[d] = PetscRealPart(coords[p * cdim + d]);
      PetscCall(DMPlexCoordinatesToReference(dm, ctx->cells[p], 1, pcoords, xi));
      PetscCall(DMPlexVecGetClosure(dm, NULL, x, ctx->cells[p], &clSize, &xa));
      for (field = 0; field < Nf; ++field) {
        PetscTabulation T;
        PetscObject     obj;
        PetscClassId    id;

        PetscCall(PetscDSGetDiscretization(ds, field, &obj));
        PetscCall(PetscObjectGetClassId(obj, &id));
        if (id == PETSCFE_CLASSID) {
          PetscFE fe = (PetscFE)obj;

          PetscCall(PetscFECreateTabulation(fe, 1, 1, xi, 0, &T));
          {
            const PetscReal *basis = T->T[0];
            const PetscInt   Nb    = T->Nb;
            const PetscInt   Nc    = T->Nc;

            for (PetscInt fc = 0; fc < Nc; ++fc) {
              interpolant[p * ctx->dof + coff + fc] = 0.0;
              for (PetscInt f = 0; f < Nb; ++f) interpolant[p * ctx->dof + coff + fc] += xa[foff + f] * basis[(0 * Nb + f) * Nc + fc];
            }
            coff += Nc;
            foff += Nb;
          }
          PetscCall(PetscTabulationDestroy(&T));
        } else if (id == PETSCFV_CLASSID) {
          PetscFV  fv = (PetscFV)obj;
          PetscInt Nc;

          // TODO Could use reconstruction if available
          PetscCall(PetscFVGetNumComponents(fv, &Nc));
          for (PetscInt fc = 0; fc < Nc; ++fc) interpolant[p * ctx->dof + coff + fc] = xa[foff + fc];
          coff += Nc;
          foff += Nc;
        }
      }
      PetscCall(DMPlexVecRestoreClosure(dm, NULL, x, ctx->cells[p], &clSize, &xa));
      PetscCheck(coff == ctx->dof, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Total components %" PetscInt_FMT " != %" PetscInt_FMT " dof specified for interpolation", coff, ctx->dof);
      PetscCheck(foff == clSize, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Total FE/FV space size %" PetscInt_FMT " != %" PetscInt_FMT " closure size", foff, clSize);
    }
    PetscCall(VecRestoreArrayRead(ctx->coords, &coords));
    PetscCall(VecRestoreArrayWrite(v, &interpolant));
  } else {
    PetscAssert(0,PETSC_COMM_WORLD, 1, "Underworld3 interpolation code shouldn't reach here");
    // for (PetscInt p = 0; p < ctx->n; ++p) {
    //   const PetscInt cell = ctx->cells[p];
    //   DMPolytopeType ct;

    //   PetscCall(DMPlexGetCellType(dm, cell, &ct));
    //   switch (ct) {
    //   case DM_POLYTOPE_SEGMENT:
    //     PetscCall(DMInterpolate_Segment_Private(ctx, dm, p, x, v));
    //     break;
    //   case DM_POLYTOPE_TRIANGLE:
    //     PetscCall(DMInterpolate_Triangle_Private(ctx, dm, p, x, v));
    //     break;
    //   case DM_POLYTOPE_QUADRILATERAL:
    //     PetscCall(DMInterpolate_Quad_Private(ctx, dm, p, x, v));
    //     break;
    //   case DM_POLYTOPE_TETRAHEDRON:
    //     PetscCall(DMInterpolate_Tetrahedron_Private(ctx, dm, p, x, v));
    //     break;
    //   case DM_POLYTOPE_HEXAHEDRON:
    //     PetscCall(DMInterpolate_Hex_Private(ctx, dm, cell, x, v));
    //     break;
    //   default:
    //     SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "No support for cell type %s", DMPolytopeTypes[PetscMax(0, PetscMin(ct, DM_NUM_POLYTOPES))]);
    //   }
    // }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
