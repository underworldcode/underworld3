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

PetscErrorCode DMInterpolationSetUp_UW(DMInterpolationInfo ctx, DM dm, PetscBool redundantPoints, PetscBool ignoreOutsideDomain, size_t* owning_cell)
{
  MPI_Comm          comm = ctx->comm;
  PetscScalar       *a;
  PetscInt          p, q, i;
  PetscMPIInt       rank, size;
  PetscErrorCode    ierr;
  Vec               pointVec;
  PetscLayout       layout;
  PetscReal         *globalPoints;
  PetscScalar       *globalPointsScalar;
  const PetscInt    *ranges;
  PetscMPIInt       *counts, *displs;
  const PetscSFNode *foundCells;
  const PetscInt    *foundPoints;
  PetscMPIInt       *foundProcs, *globalProcs;
  PetscInt          n, N, numFound;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = MPI_Comm_size(comm, &size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  if (ctx->dim < 0) SETERRQ(comm, PETSC_ERR_ARG_WRONGSTATE, "The spatial dimension has not been set");
  /* Locate points */
  n = ctx->nInput;
  if (!redundantPoints) {
    ierr = PetscLayoutCreate(comm, &layout);CHKERRQ(ierr);
    ierr = PetscLayoutSetBlockSize(layout, 1);CHKERRQ(ierr);
    ierr = PetscLayoutSetLocalSize(layout, n);CHKERRQ(ierr);
    ierr = PetscLayoutSetUp(layout);CHKERRQ(ierr);
    ierr = PetscLayoutGetSize(layout, &N);CHKERRQ(ierr);
    /* Communicate all points to all processes */
    ierr = PetscMalloc3(N*ctx->dim,&globalPoints,size,&counts,size,&displs);CHKERRQ(ierr);
    ierr = PetscLayoutGetRanges(layout, &ranges);CHKERRQ(ierr);
    for (p = 0; p < size; ++p) {
      counts[p] = (ranges[p+1] - ranges[p])*ctx->dim;
      displs[p] = ranges[p]*ctx->dim;
    }
    ierr = MPI_Allgatherv(ctx->points, n*ctx->dim, MPIU_REAL, globalPoints, counts, displs, MPIU_REAL, comm);CHKERRMPI(ierr);
  } else {
    N = n;
    globalPoints = ctx->points;
    counts = displs = NULL;
    layout = NULL;
  }
#if 0
  ierr = PetscMalloc3(N,&foundCells,N,&foundProcs,N,&globalProcs);CHKERRQ(ierr);
  /* foundCells[p] = m->locatePoint(&globalPoints[p*ctx->dim]); */
#else
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscMalloc1(N*ctx->dim,&globalPointsScalar);CHKERRQ(ierr);
  for (i=0; i<N*ctx->dim; i++) globalPointsScalar[i] = globalPoints[i];
#else
  globalPointsScalar = globalPoints;
#endif
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, ctx->dim, N*ctx->dim, globalPointsScalar, &pointVec);CHKERRQ(ierr);
  ierr = PetscMalloc2(N,&foundProcs,N,&globalProcs);CHKERRQ(ierr);
  for (p = 0; p < N; ++p) {foundProcs[p] = size;}
  PetscSF cellSF = NULL;
  if (owning_cell){
    PetscSFNode    *sf_cells;
    ierr = PetscMalloc1(N, &sf_cells);CHKERRQ(ierr);
    size_t range = 0;
    for (size_t p=0; p<(size_t)N; p++) {
      sf_cells[p].rank  = 0;
      sf_cells[p].index = owning_cell[p];
      if (owning_cell[p] > range) {
        range = owning_cell[p];
      }
    }
    ierr = PetscSFCreate(PETSC_COMM_SELF,&cellSF);CHKERRQ(ierr);
    ierr = PetscSFSetGraph(cellSF, range, N, NULL, PETSC_OWN_POINTER, sf_cells, PETSC_OWN_POINTER);CHKERRQ(ierr);
  }  
  ierr = DMLocatePoints(dm, pointVec, DM_POINTLOCATION_REMOVE, &cellSF);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(cellSF,NULL,&numFound,&foundPoints,&foundCells);CHKERRQ(ierr);
#endif
  for (p = 0; p < numFound; ++p) {
    if (foundCells[p].index >= 0) foundProcs[foundPoints ? foundPoints[p] : p] = rank;
  }
  /* Let the lowest rank process own each point */
  ierr   = MPIU_Allreduce(foundProcs, globalProcs, N, MPI_INT, MPI_MIN, comm);CHKERRQ(ierr);
  ctx->n = 0;
  for (p = 0; p < N; ++p) {
    if (globalProcs[p] == size) {
      if (!ignoreOutsideDomain) SETERRQ4(comm, PETSC_ERR_PLIB, "Point %d: %g %g %g not located in mesh", p, (double)globalPoints[p*ctx->dim+0], (double)(ctx->dim > 1 ? globalPoints[p*ctx->dim+1] : 0.0), (double)(ctx->dim > 2 ? globalPoints[p*ctx->dim+2] : 0.0));
      else if (!rank) ++ctx->n;
    } else if (globalProcs[p] == rank) ++ctx->n;
  }
  /* Create coordinates vector and array of owned cells */
  ierr = PetscMalloc1(ctx->n, &ctx->cells);CHKERRQ(ierr);
  ierr = VecCreate(comm, &ctx->coords);CHKERRQ(ierr);
  ierr = VecSetSizes(ctx->coords, ctx->n*ctx->dim, PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetBlockSize(ctx->coords, ctx->dim);CHKERRQ(ierr);
  ierr = VecSetType(ctx->coords,VECSTANDARD);CHKERRQ(ierr);
  ierr = VecGetArray(ctx->coords, &a);CHKERRQ(ierr);
  for (p = 0, q = 0, i = 0; p < N; ++p) {
    if (globalProcs[p] == rank) {
      PetscInt d;

      for (d = 0; d < ctx->dim; ++d, ++i) a[i] = globalPoints[p*ctx->dim+d];
      ctx->cells[q] = foundCells[q].index;
      ++q;
    }
    if (globalProcs[p] == size && !rank) {
      PetscInt d;

      for (d = 0; d < ctx->dim; ++d, ++i) a[i] = 0.;
      ctx->cells[q] = -1;
      ++q;
    }
  }
  ierr = VecRestoreArray(ctx->coords, &a);CHKERRQ(ierr);
#if 0
  ierr = PetscFree3(foundCells,foundProcs,globalProcs);CHKERRQ(ierr);
#else
  ierr = PetscFree2(foundProcs,globalProcs);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&cellSF);CHKERRQ(ierr);
  ierr = VecDestroy(&pointVec);CHKERRQ(ierr);
#endif
  if ((void*)globalPointsScalar != (void*)globalPoints) {ierr = PetscFree(globalPointsScalar);CHKERRQ(ierr);}
  if (!redundantPoints) {ierr = PetscFree3(globalPoints,counts,displs);CHKERRQ(ierr);}
  ierr = PetscLayoutDestroy(&layout);CHKERRQ(ierr);
  PetscFunctionReturn(0);
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
  PetscInt       n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 3);
  PetscValidHeaderSpecific(v, VEC_CLASSID, 4);
  ierr = VecGetLocalSize(v, &n);CHKERRQ(ierr);
  if (n != ctx->n*ctx->dof) SETERRQ2(ctx->comm, PETSC_ERR_ARG_SIZ, "Invalid input vector size %D should be %D", n, ctx->n*ctx->dof);
  if (n) {
    PetscDS        ds;
    // PetscBool      done = PETSC_FALSE;

    ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);
    if (ds) {
      const PetscScalar *coords;
      PetscScalar       *interpolant;
      PetscInt           cdim, d, p, Nf, field;

      ierr = VecGetArrayRead(ctx->coords, &coords);CHKERRQ(ierr);
      ierr = VecGetArrayWrite(v, &interpolant);CHKERRQ(ierr);

      ierr = DMGetCoordinateDim(dm, &cdim);CHKERRQ(ierr);
      ierr = PetscDSGetNumFields(ds, &Nf);CHKERRQ(ierr);
      for (p = 0; p < ctx->n; ++p) {
        PetscReal    xi[3];
        PetscScalar *xa = NULL;
        PetscReal    pcoords[3];
        PetscInt     c;

        if (ctx->cells[p] < 0) continue;
        for (d = 0; d < cdim; ++d) pcoords[d] = PetscRealPart(coords[p*cdim+d]);
        ierr = DMPlexCoordinatesToReference(dm, ctx->cells[p], 1, pcoords, xi);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(dm, NULL, x, ctx->cells[p], NULL, &xa);CHKERRQ(ierr);

        PetscInt d = 0;

        c = 0;

        for (field = 0; field < Nf; ++field) {
          PetscTabulation T;
          PetscFE         fe;
          PetscClassId    id;
          PetscInt        Nc, f, fc;

          ierr = PetscDSGetDiscretization(ds, field, (PetscObject *) &fe);CHKERRQ(ierr);
          ierr = PetscObjectGetClassId((PetscObject) fe, &id);CHKERRQ(ierr);
          if (id != PETSCFE_CLASSID) break;
          ierr = PetscFEGetNumComponents(fe, &Nc);CHKERRQ(ierr);

          ierr = PetscFECreateTabulation(fe, 1, 1, xi, 0, &T);CHKERRQ(ierr);
          {
            const PetscReal *basis = T->T[0];
            const PetscInt   Nb    = T->Nb;
            const PetscInt   Nc    = T->Nc;
            for (fc = 0; fc < Nc; ++fc) {
              interpolant[p*ctx->dof+c+fc] = 0.0;
              for (f = 0; f < Nb; ++f) {
                interpolant[p*ctx->dof+c+fc] += xa[d+f]*basis[(0*Nb + f)*Nc + fc];
              }
            }
            d+=Nb;
          }
          ierr = PetscTabulationDestroy(&T);CHKERRQ(ierr);
          c += Nc;
        }
      if (field == Nf) {
        // done = PETSC_TRUE;
        if (c != ctx->dof) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Total components %D != %D dof specified for interpolation", c, ctx->dof);
      }
      ierr = DMPlexVecRestoreClosure(dm, NULL, x, ctx->cells[p], NULL, &xa);CHKERRQ(ierr);

      }
      ierr = VecRestoreArrayWrite(v, &interpolant);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(ctx->coords, &coords);CHKERRQ(ierr);
    }
    // if (!done) {
    //   DMPolytopeType ct;
    //   /* TODO Check each cell individually */
    //   ierr = DMPlexGetCellType(dm, ctx->cells[0], &ct);CHKERRQ(ierr);
    //   switch (ct) {
    //     case DM_POLYTOPE_TRIANGLE:      ierr = DMInterpolate_Triangle_Private(ctx, dm, x, v);CHKERRQ(ierr);break;
    //     case DM_POLYTOPE_QUADRILATERAL: ierr = DMInterpolate_Quad_Private(ctx, dm, x, v);CHKERRQ(ierr);break;
    //     case DM_POLYTOPE_TETRAHEDRON:   ierr = DMInterpolate_Tetrahedron_Private(ctx, dm, x, v);CHKERRQ(ierr);break;
    //     case DM_POLYTOPE_HEXAHEDRON:    ierr = DMInterpolate_Hex_Private(ctx, dm, x, v);CHKERRQ(ierr);break;
    //     default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "No support fpr cell type %s", DMPolytopeTypes[ct]);
    //   }
    // }
  }
  PetscFunctionReturn(0);
}
