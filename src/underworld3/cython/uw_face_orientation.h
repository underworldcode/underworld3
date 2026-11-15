/*
 * Boundary terms on triangular faces that are not at orientation 0 in their cell (#1188).
 *
 * PETSc's boundary integrators (PetscFEIntegrateBdResidual_Basic,
 * PetscFEIntegrateBdJacobian_Basic and PetscFEIntegrateBd_Basic, PETSc 3.25 and main)
 * evaluate the integrand's coordinates and normal at the physical point of face
 * quadrature point q, mapped through the face's own vertex order, and the basis and
 * fields at the cell's tabulation point q (q reversed for a negative orientation; the
 * boundary integral ignores the orientation). The two are the same point only on a
 * face at orientation 0 in its cell. That is the rule for a segment, which has two
 * orientations; a triangle has six. A term that reads the coordinates (a load or a
 * normal written in x) is then evaluated at one point and spread with the basis of
 * another; the total of the load is kept, its moments are not.
 *
 * For each face at a non-zero orientation, the wrappers below replace the face's
 * affine map x(xi) by x(T_o(xi)), where T_o is the reference triangle's own map for
 * orientation o (DMPlexOrientPoint on the reference cell), so that the face is
 * described in its cell's frame, and set the orientation to 0. They then call
 * PETSc's own integrator, and restore the geometry and orientation. The basis,
 * fields, weights, Jacobian determinant and normal are untouched: the determinant
 * and normal are per face and the same in either frame.
 *
 * A PETSc that pairs the points itself is not disturbed: the face it is given is
 * at orientation 0, in its cell's frame.
 *
 * Installed on the PetscFE objects of 3-D simplex meshes only (UW_DMPairFace-
 * QuadratureByOrientation). PETSc's integrator is kept on each PetscFE object, so a
 * PetscFE reached from two extension modules is wrapped once.
 *
 * Not covered: quadrilateral faces of hexahedral meshes (no wrapper is installed),
 * and curved faces (a 3-D mesh with coordinates of degree other than 1 is refused at
 * install, on every rank together). An auxiliary field must live on the cells, as
 * every one UW3 sets does (the mesh's own local vector); the wrapper asserts it.
 * Nothing here is copied from PETSc.
 */
#ifndef UW_FACE_ORIENTATION_H
#define UW_FACE_ORIENTATION_H

#include <petsc.h>
#include <petsc/private/petscfeimpl.h>

typedef PetscErrorCode (*UW_BdResidualFn)(PetscDS, PetscWeakForm, PetscFormKey, PetscInt, PetscFEGeom *, const PetscScalar[], const PetscScalar[], PetscDS, const PetscScalar[], PetscReal, PetscScalar[]);
typedef PetscErrorCode (*UW_BdJacobianFn)(PetscDS, PetscWeakForm, PetscFEJacobianType, PetscFormKey, PetscInt, PetscFEGeom *, const PetscScalar[], const PetscScalar[], PetscDS, const PetscScalar[], PetscReal, PetscReal, PetscScalar[]);
typedef PetscErrorCode (*UW_BdIntegralFn)(PetscDS, PetscInt, PetscBdPointFn *, PetscInt, PetscFEGeom *, const PetscScalar[], PetscDS, const PetscScalar[], PetscScalar[]);

#define UW_PETSC_BD_RESIDUAL "UWPetscFEIntegrateBdResidual_C"
#define UW_PETSC_BD_JACOBIAN "UWPetscFEIntegrateBdJacobian_C"
#define UW_PETSC_BD_INTEGRAL "UWPetscFEIntegrateBd_C"

/* The reference triangle's map for orientation o in [-3, 3):
   xi -> uw_triangle_v0[o + 3] + uw_triangle_J[o + 3] (xi - (-1, -1)). */
static PetscBool uw_triangle_maps_ready = PETSC_FALSE;
static PetscReal uw_triangle_v0[6][2], uw_triangle_J[6][4];

static PetscErrorCode UW_TriangleOrientationMaps(void)
{
  PetscFunctionBegin;
  if (uw_triangle_maps_ready) PetscFunctionReturn(PETSC_SUCCESS);
  for (PetscInt o = -3; o < 3; ++o) {
    DM        refdm;
    PetscReal v0[3], J[9], detJ;

    PetscCall(DMPlexCreateReferenceCell(PETSC_COMM_SELF, DM_POLYTOPE_TRIANGLE, &refdm));
    PetscCall(DMPlexOrientPoint(refdm, 0, o));
    PetscCall(DMPlexComputeCellGeometryFEM(refdm, 0, NULL, v0, J, NULL, &detJ));
    PetscCall(DMDestroy(&refdm));
    for (PetscInt i = 0; i < 2; ++i) uw_triangle_v0[o + 3][i] = v0[i];
    for (PetscInt i = 0; i < 4; ++i) uw_triangle_J[o + 3][i] = J[i];
  }
  uw_triangle_maps_ready = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Describe every face of the chunk that is not at orientation 0 in its cell's frame.
   The replaced maps and orientations go to `saved` and `ornt` for UW_FaceMapsRestore;
   both stay NULL when there is nothing to replace. */
static PetscErrorCode UW_FaceMapsToCellFrame(PetscFE fe, PetscDS dsAux, PetscInt Ne, PetscFEGeom *fgeom, PetscReal **saved, PetscInt **ornt)
{
  const PetscInt   dE = fgeom->dimEmbed, Np = fgeom->numPoints, stride = dE + dE * dE;
  const PetscReal  m1[2] = {-1.0, -1.0};
  const PetscReal *xi;
  PetscInt         e;

  PetscFunctionBegin;
  *saved = NULL;
  *ornt  = NULL;
  for (e = 0; e < Ne; ++e)
    if (fgeom->face[e][1]) break;
  if (e == Ne) PetscFunctionReturn(PETSC_SUCCESS);
  // Installed only where every face is a triangle with straight sides (checked at
  // install) and every auxiliary field lives on the cells; these hold by construction.
  PetscCheck(dE == 3 && fgeom->isAffine, PETSC_COMM_SELF, PETSC_ERR_PLIB, "A face that is not a straight-sided triangle reached the boundary-term pairing (#1188)");
  if (dsAux) {
    PetscInt dimAux, dim;

    PetscCall(PetscDSGetSpatialDimension(dsAux, &dimAux));
    PetscCall(PetscFEGetSpatialDimension(fe, &dim));
    PetscCheck(dimAux == dim, PETSC_COMM_SELF, PETSC_ERR_PLIB, "An auxiliary field on the boundary reached the boundary-term pairing (#1188)");
  }
  for (e = 0; e < Ne; ++e) PetscCheck(fgeom->face[e][1] >= -3 && fgeom->face[e][1] < 3, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Orientation %" PetscInt_FMT " is not a triangle's", fgeom->face[e][1]);
  PetscCall(UW_TriangleOrientationMaps());
  xi = fgeom->xi ? fgeom->xi : m1;
  PetscCall(PetscMalloc2(Ne * stride, saved, Ne, ornt));
  for (e = 0; e < Ne; ++e) {
    const PetscInt   o = fgeom->face[e][1];
    PetscReal       *v = &fgeom->v[e * Np * dE];
    PetscReal       *J = &fgeom->J[e * Np * dE * dE];
    const PetscReal *T0, *TJ;
    PetscReal        shift[2], JT[3][2];

    (*ornt)[e] = o;
    PetscCall(PetscArraycpy(&(*saved)[e * stride], v, dE));
    PetscCall(PetscArraycpy(&(*saved)[e * stride + dE], J, dE * dE));
    if (!o) continue;
    T0 = uw_triangle_v0[o + 3];
    TJ = uw_triangle_J[o + 3];
    // x(T(xi')) = v + J (T(xi') - xi), with T(xi') = T0 + TJ (xi' - m1), is v' + J' (xi' - xi)
    for (PetscInt a = 0; a < 2; ++a) shift[a] = T0[a] - xi[a] + TJ[a * 2 + 0] * (xi[0] - m1[0]) + TJ[a * 2 + 1] * (xi[1] - m1[1]);
    for (PetscInt i = 0; i < dE; ++i) {
      v[i] += J[i * dE + 0] * shift[0] + J[i * dE + 1] * shift[1];
      for (PetscInt b = 0; b < 2; ++b) JT[i][b] = J[i * dE + 0] * TJ[0 * 2 + b] + J[i * dE + 1] * TJ[1 * 2 + b];
    }
    for (PetscInt i = 0; i < dE; ++i)
      for (PetscInt b = 0; b < 2; ++b) J[i * dE + b] = JT[i][b];
    fgeom->face[e][1] = 0;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode UW_FaceMapsRestore(PetscInt Ne, PetscFEGeom *fgeom, PetscReal **saved, PetscInt **ornt)
{
  const PetscInt dE = fgeom->dimEmbed, Np = fgeom->numPoints, stride = dE + dE * dE;

  PetscFunctionBegin;
  if (!*saved) PetscFunctionReturn(PETSC_SUCCESS);
  for (PetscInt e = 0; e < Ne; ++e) {
    PetscCall(PetscArraycpy(&fgeom->v[e * Np * dE], &(*saved)[e * stride], dE));
    PetscCall(PetscArraycpy(&fgeom->J[e * Np * dE * dE], &(*saved)[e * stride + dE], dE * dE));
    fgeom->face[e][1] = (*ornt)[e];
  }
  PetscCall(PetscFree2(*saved, *ornt));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode UW_PetscFEIntegrateBdResidual(PetscDS ds, PetscWeakForm wf, PetscFormKey key, PetscInt Ne, PetscFEGeom *fgeom, const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS dsAux, const PetscScalar coefficientsAux[], PetscReal t, PetscScalar elemVec[])
{
  PetscFE         fe;
  UW_BdResidualFn petsc_integrator;
  PetscReal      *saved;
  PetscInt       *ornt;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscCall(PetscDSGetDiscretization(ds, key.field, (PetscObject *)&fe));
  PetscCall(PetscObjectQueryFunction((PetscObject)fe, UW_PETSC_BD_RESIDUAL, &petsc_integrator));
  PetscCall(UW_FaceMapsToCellFrame(fe, dsAux, Ne, fgeom, &saved, &ornt));
  ierr = petsc_integrator(ds, wf, key, Ne, fgeom, coefficients, coefficients_t, dsAux, coefficientsAux, t, elemVec);
  PetscCall(UW_FaceMapsRestore(Ne, fgeom, &saved, &ornt));
  PetscCall(ierr);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode UW_PetscFEIntegrateBdJacobian(PetscDS ds, PetscWeakForm wf, PetscFEJacobianType jtype, PetscFormKey key, PetscInt Ne, PetscFEGeom *fgeom, const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS dsAux, const PetscScalar coefficientsAux[], PetscReal t, PetscReal u_tshift, PetscScalar elemMat[])
{
  PetscFE         fe;
  PetscInt        Nf;
  UW_BdJacobianFn petsc_integrator;
  PetscReal      *saved;
  PetscInt       *ornt;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  // the dispatch's own choice: key.field numbers the pair (row field, column field)
  PetscCall(PetscDSGetNumFields(ds, &Nf));
  PetscCall(PetscDSGetDiscretization(ds, key.field / Nf, (PetscObject *)&fe));
  PetscCall(PetscObjectQueryFunction((PetscObject)fe, UW_PETSC_BD_JACOBIAN, &petsc_integrator));
  PetscCall(UW_FaceMapsToCellFrame(fe, dsAux, Ne, fgeom, &saved, &ornt));
  ierr = petsc_integrator(ds, wf, jtype, key, Ne, fgeom, coefficients, coefficients_t, dsAux, coefficientsAux, t, u_tshift, elemMat);
  PetscCall(UW_FaceMapsRestore(Ne, fgeom, &saved, &ornt));
  PetscCall(ierr);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode UW_PetscFEIntegrateBd(PetscDS ds, PetscInt field, PetscBdPointFn *obj_func, PetscInt Ne, PetscFEGeom *fgeom, const PetscScalar coefficients[], PetscDS dsAux, const PetscScalar coefficientsAux[], PetscScalar integral[])
{
  PetscFE         fe;
  UW_BdIntegralFn petsc_integrator;
  PetscReal      *saved;
  PetscInt       *ornt;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscCall(PetscDSGetDiscretization(ds, field, (PetscObject *)&fe));
  PetscCall(PetscObjectQueryFunction((PetscObject)fe, UW_PETSC_BD_INTEGRAL, &petsc_integrator));
  PetscCall(UW_FaceMapsToCellFrame(fe, dsAux, Ne, fgeom, &saved, &ornt));
  ierr = petsc_integrator(ds, field, obj_func, Ne, fgeom, coefficients, dsAux, coefficientsAux, integral);
  PetscCall(UW_FaceMapsRestore(Ne, fgeom, &saved, &ornt));
  PetscCall(ierr);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Route one PetscFE's three boundary integrators through the wrappers above. A
   PetscFE whose face quadrature is not on a triangle is refused: the wrappers remap
   triangles only. The PetscFE's set-up is the same on every rank. */
static PetscErrorCode UW_PetscFEPairFaceQuadratureByOrientation(PetscFE fe)
{
  UW_BdResidualFn residual;
  PetscQuadrature fquad;
  DMPolytopeType  ct;

  PetscFunctionBegin;
  PetscCall(PetscObjectQueryFunction((PetscObject)fe, UW_PETSC_BD_RESIDUAL, &residual));
  if (residual) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscFEGetFaceQuadrature(fe, &fquad));
  PetscCall(PetscQuadratureGetCellType(fquad, &ct));
  PetscCheck(ct == DM_POLYTOPE_TRIANGLE, PetscObjectComm((PetscObject)fe), PETSC_ERR_SUP, "Boundary terms on a 3-D simplex mesh: a face quadrature on %s, not a triangle (#1188)", DMPolytopeTypes[ct]);
  PetscCheck(fe->ops->integratebdresidual && fe->ops->integratebdjacobian && fe->ops->integratebd, PetscObjectComm((PetscObject)fe), PETSC_ERR_SUP, "Boundary terms on a 3-D simplex mesh: a PetscFE without PETSc's boundary integrators (#1188)");
  PetscCall(PetscObjectComposeFunction((PetscObject)fe, UW_PETSC_BD_RESIDUAL, fe->ops->integratebdresidual));
  PetscCall(PetscObjectComposeFunction((PetscObject)fe, UW_PETSC_BD_JACOBIAN, fe->ops->integratebdjacobian));
  PetscCall(PetscObjectComposeFunction((PetscObject)fe, UW_PETSC_BD_INTEGRAL, fe->ops->integratebd));
  fe->ops->integratebdresidual = UW_PetscFEIntegrateBdResidual;
  fe->ops->integratebdjacobian = UW_PetscFEIntegrateBdJacobian;
  fe->ops->integratebd         = UW_PetscFEIntegrateBd;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Every finite-element field of every DS of a 3-D simplex mesh. Other meshes are
   left with PETSc's integrators. Call after DMCreateDS, on every rank.

   The pairing remaps affine faces only. A 3-D mesh whose coordinates are not of
   degree 1 is refused here; the coordinate space is a property of the plex that
   every rank holds alike, so every rank refuses together. (Mesh refuses such a
   mesh first, #978.) A rank with no cells may not see a simplex; it installs
   nothing and has no faces to integrate. */
PetscErrorCode UW_DMPairFaceQuadratureByOrientation(DM dm)
{
  PetscInt  dim, Nds, Ncf;
  PetscBool simplex;
  DM        cdm;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  if (dim != 3) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetNumFields(cdm, &Ncf));
  if (Ncf) {
    PetscObject  disc;
    PetscClassId id;

    PetscCall(DMGetField(cdm, 0, NULL, &disc));
    PetscCall(PetscObjectGetClassId(disc, &id));
    if (id == PETSCFE_CLASSID) {
      PetscSpace sp;
      PetscInt   degree;

      PetscCall(PetscFEGetBasisSpace((PetscFE)disc, &sp));
      PetscCall(PetscSpaceGetDegree(sp, &degree, NULL));
      PetscCheck(degree == 1, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Boundary terms on a 3-D mesh need coordinates of degree 1, not %" PetscInt_FMT ": curved faces are not paired (#1188)", degree);
    }
  }
  PetscCall(DMPlexIsSimplex(dm, &simplex));
  if (!simplex) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMGetNumDS(dm, &Nds));
  for (PetscInt s = 0; s < Nds; ++s) {
    PetscDS  ds;
    PetscInt Nf;

    PetscCall(DMGetRegionNumDS(dm, s, NULL, NULL, &ds, NULL));
    PetscCall(PetscDSGetNumFields(ds, &Nf));
    for (PetscInt f = 0; f < Nf; ++f) {
      PetscObject  obj;
      PetscClassId id;

      PetscCall(PetscDSGetDiscretization(ds, f, &obj));
      PetscCall(PetscObjectGetClassId(obj, &id));
      if (id == PETSCFE_CLASSID) PetscCall(UW_PetscFEPairFaceQuadratureByOrientation((PetscFE)obj));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#endif
