static char help[] = "Stokes Problem on an annular section with finite elements.\n\
We solve the Stokes problem on an annular domain, using a parallel unstructured mesh.\n\n\n";

/*
  - We can get higher order geometry by
    1) Makiing a new DM/DS for some orders
    2) Projecting the original coordinates, but for marked boundaries we should evaluate from the NURB
    3) Replace the coordinate DM
*/

/*
The isoviscous Stokes problem, which we discretize using the finite
element method on an unstructured mesh. The weak form equations are

  < \nabla v, \nabla u + {\nabla u}^T > - < \nabla\cdot v, p > + < v, f > = 0
  < q, \nabla\cdot u >                                                    = 0

Viewing:

To produce nice output, use

  -dm_refine 3 -dm_view hdf5:sol1.h5 -sol_vec_view hdf5:sol1.h5::append

You can get a LaTeX view of the mesh, with point numbering using

  -dm_view :mesh.tex:ascii_latex -dm_plex_view_scale 8.0

The data layout can be viewed using

  -dm_petscsection_view

Lots of information about the FEM assembly can be printed using

  -dm_plex_print_fem 2

Field Data:

  DMPLEX data is organized by point, and the closure operation just stacks up the
data from each mesh point in the closure, and segregates by field. Thus, for a
P_2-P_1 Stokes element, we have from DMPlexVecGetClosure()

  cl{e} = {f e_0 e_1 e_2 v_0 v_1 v_2}
  x'    = [u_{e_0} v_{e_0} u_{e_1} v_{e_1} u_{e_2} v_{e_2} u_{v_0} v_{v_0} u_{v_1} v_{v_1} u_{v_2} v_{v_2} p_{v_0} p_{v_1} p_{v_2}]

Likewise, DMPlexVecSetClosure() takes data partitioned by field.
*/

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscds.h>

typedef struct {
  /* Domain and mesh definition */
  PetscInt  dim;     /* The topological mesh dimension */
  PetscBool simplex; /* Use simplices or tensor product cells */
  PetscBool annular; /* Use an annular region */
  PetscInt  elements[3];
} AppCtx;

PetscErrorCode zero(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt c;
  for (c = 0; c < Nc; ++c) u[c] = 0.0;
  return 0;
}

PetscErrorCode one(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt c;
  for (c = 0; c < Nc; ++c) u[c] = 1.0;
  return 0;
}

void pressure(PetscInt dim, PetscInt Nf, PetscInt NfAux,
              const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
              const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
              PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar p[])
{
  p[0] = u[uOff[1]];
}

void f0_error_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt Nc = uOff[1] - uOff[0], c;
  for (c = 0; c < Nc; ++c) f0[c] = 0.0 - u[c];
}

void f0_error_p(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = 0.0 - u[uOff[1]];
}

void f0_boussinesq_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                     PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscScalar Ra = constants[0];
  const PetscScalar T  = a[0];
  PetscScalar mag;
  PetscInt d;

  f0[dim-1] = +1 * T;
  for(d=0; d<dim-1; d++) f0[d]=0;
}

void f1_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
          PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscInt Nc = uOff[1] - uOff[0];
  PetscInt       c, d;

  for (c = 0; c < Nc; ++c) {
    for (d = 0; d < dim; ++d) {
      f1[c*dim+d] = 0.5*(u_x[c*dim+d] + u_x[d*dim+c]);
    }
    f1[c*dim+c] -= u[uOff[1]];
  }
}

void f0_p(PetscInt dim, PetscInt Nf, PetscInt NfAux,
          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
          PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;
  for (d = 0, f0[0] = 0.0; d < dim; ++d) f0[0] += u_x[d*dim+d];
}

void g1_pu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
           PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g1[d*dim+d] = 1.0;
}

void g2_up(PetscInt dim, PetscInt Nf, PetscInt NfAux,
           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
           PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g2[d*dim+d] = -1.0;
}

void g3_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
           PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  const PetscInt Nc = uOff[1] - uOff[0];
  PetscInt       c, d;

  for (c = 0; c < Nc; ++c) {
    for (d = 0; d < dim; ++d) {
      g3[((c*Nc+c)*dim+d)*dim+d] = 0.5;
      g3[((c*Nc+d)*dim+c)*dim+d] = 0.5;
    }
  }
}

static void g0_pp(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  const PetscReal mu = 1.0;
  g0[0] = 1.0/mu;
}

PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;
  PetscBool specified;

  PetscFunctionBeginUser;
  options->dim     = 2;
  options->simplex = PETSC_TRUE;
  options->annular = PETSC_FALSE;
  PetscInt cells[3] = {4,4,4};

  ierr = PetscOptionsBegin(comm, NULL, "Annular Stokes Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex4.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "Use simplices or tensor product cells", "ex4.c", options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-annular", "Use an annular region", "ex4.c", options->annular, &options->annular, NULL);CHKERRQ(ierr);
    PetscInt n;
  ierr = PetscOptionsIntArray("-cells", "element count (default: 5,5,5)", NULL,cells,&n,&specified);CHKERRQ(ierr);
  memcpy(options->elements, cells, 3*sizeof(PetscInt));
  if(!options->annular) {
    if(n!=0) options->dim = n;
  }
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  {
    double min[3] = {-1.0,-1.0,-1.0};
    double max[3] = {1.0,1.0,1.0};
    ierr = DMPlexCreateBoxMesh(comm, user->dim, 
        user->simplex, user->elements, NULL, NULL, 
        NULL, PETSC_TRUE, dm);CHKERRQ(ierr);
  }
  {
    DM               pdm = NULL;
    PetscPartitioner part;

    ierr = DMPlexGetPartitioner(*dm, &part);CHKERRQ(ierr);
    ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);
    ierr = DMPlexDistribute(*dm, 0, NULL, &pdm);CHKERRQ(ierr);
    if (pdm) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = pdm;
    }
  }
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode circle_shape(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  AppCtx* user = (AppCtx*)ctx; 
  PetscInt d;
  PetscScalar circle, r, center[3];
  
  center[0] = 0.5;
  center[1] = 0.5;
  r = 0.2;

  circle = pow(x[0]-center[0], 2) + pow(x[1]-center[1], 2);

  if( circle < pow(0.1, 2) ){ u[0] = 1.5; }
  else                      { u[0] = 0.;  }

  return 0;
}

static PetscErrorCode SetupMaterial(DM dm, DM dmAux, AppCtx *user)
{
  Vec            paramVec;
  PetscInt       cStart, cEnd, cEndInterior;
  PetscErrorCode ierr;
  PetscErrorCode (*matFuncs[1])( PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx) = {circle_shape};
  void* ctx_array[1] = {user};

  PetscFunctionBeginUser;

  ierr = DMCreateLocalVector(dmAux, &paramVec);CHKERRQ(ierr);
  DMProjectFunctionLocal(dmAux, 0.0, matFuncs, ctx_array, INSERT_ALL_VALUES, paramVec);
  ierr = PetscObjectCompose((PetscObject) dm, "A", (PetscObject) paramVec);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dmAux, NULL, "-dm_aux_view");CHKERRQ(ierr);
  {
    Vec gvec;

    ierr = DMGetGlobalVector(dmAux, &gvec);CHKERRQ(ierr);
    ierr = DMLocalToGlobalBegin(dmAux, paramVec, INSERT_VALUES, gvec);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(dmAux, paramVec, INSERT_VALUES, gvec);CHKERRQ(ierr);
    ierr = VecViewFromOptions(gvec, NULL, "-vec_aux_view");CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dmAux, &gvec);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&paramVec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SetupProblem(DM dm, AppCtx *user)
{
  PetscDS        ds;
  PetscInt       id, comp;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);
  ierr = PetscDSSetResidual(ds, 0, f0_boussinesq_u, f1_u);CHKERRQ(ierr);
  ierr = PetscDSSetResidual(ds, 1, f0_p, NULL);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(ds, 0, 0, NULL, NULL,  NULL,  g3_uu);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(ds, 0, 1, NULL, NULL,  g2_up, NULL);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(ds, 1, 0, NULL, g1_pu, NULL,  NULL);CHKERRQ(ierr);
  ierr = PetscDSSetJacobianPreconditioner(ds, 0, 0, NULL, NULL, NULL, g3_uu);CHKERRQ(ierr);
  ierr = PetscDSSetJacobianPreconditioner(ds, 0, 1, NULL, NULL, g2_up, NULL);CHKERRQ(ierr);
  ierr = PetscDSSetJacobianPreconditioner(ds, 1, 0, NULL, g1_pu, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscDSSetJacobianPreconditioner(ds, 1, 1, g0_pp, NULL, NULL, NULL);CHKERRQ(ierr);

  {
    PetscInt ids[4] = {1, 2,3,4};
    
    // no-slip dirichlet bc
    ierr = PetscDSAddBoundary(ds, DM_BC_ESSENTIAL, 
            "wall", "marker", 0, 0, NULL, (void (*)(void)) zero, 4, ids, user);CHKERRQ(ierr);
  }
#if 0
  ierr = PetscDSSetExactSolution(ds, 0, user->exactFuncs[0], user);CHKERRQ(ierr);
  ierr = PetscDSSetExactSolution(ds, 1, user->exactFuncs[1], user);CHKERRQ(ierr);
#endif
  {
    PetscScalar constants[1];

    constants[0] = 1.0; /* Rayleigh number */
    ierr = PetscDSSetConstants(ds, 1, constants);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SetupDiscretization(DM dm, AppCtx *user)
{
  DM              cdm = dm, dmAux;
  PetscFE         fe[2], feAux;
  PetscQuadrature q;
  MPI_Comm        comm;
  PetscInt        dim;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(comm, dim, dim, user->simplex, "vel_", PETSC_DEFAULT, &fe[0]);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe[0], "velocity");CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fe[0], &q);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(comm, dim, 1, user->simplex, "pres_", PETSC_DEFAULT, &fe[1]);CHKERRQ(ierr);
  ierr = PetscFESetQuadrature(fe[1], q);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe[1], "pressure");CHKERRQ(ierr);

  ierr = PetscFECreateDefault(comm, dim, 1, user->simplex, "temp_", PETSC_DEFAULT, &feAux);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) feAux, "temperature");CHKERRQ(ierr);
  ierr = PetscFESetQuadrature(feAux, q);CHKERRQ(ierr);

  ierr = DMSetField(dm, 0, NULL, (PetscObject) fe[0]);CHKERRQ(ierr);
  ierr = DMSetField(dm, 1, NULL, (PetscObject) fe[1]);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  ierr = SetupProblem(dm, user);CHKERRQ(ierr);
  while (cdm) {
    ierr = DMCopyDisc(dm, cdm);CHKERRQ(ierr);

    ierr = DMClone(cdm, &dmAux);CHKERRQ(ierr);
    ierr = DMPlexCopyCoordinates(cdm, dmAux);CHKERRQ(ierr);
    ierr = DMSetField(dmAux, 0, NULL, (PetscObject) feAux);CHKERRQ(ierr);
    ierr = DMCreateDS(dmAux);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject) cdm, "dmAux", (PetscObject) dmAux);CHKERRQ(ierr);
    ierr = SetupMaterial(cdm, dmAux, user);CHKERRQ(ierr);
    ierr = DMDestroy(&dmAux);CHKERRQ(ierr);

    ierr = DMGetCoarseDM(cdm, &cdm);CHKERRQ(ierr);
  }
  ierr = PetscFEDestroy(&fe[0]);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe[1]);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&feAux);CHKERRQ(ierr);
  {
    PetscObject  pressure;
    MatNullSpace nullSpacePres;

    ierr = DMGetField(dm, 1, NULL, &pressure);CHKERRQ(ierr);
    ierr = MatNullSpaceCreate(PetscObjectComm(pressure), PETSC_TRUE, 0, NULL, &nullSpacePres);CHKERRQ(ierr);
    ierr = PetscObjectCompose(pressure, "nullspace", (PetscObject) nullSpacePres);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nullSpacePres);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode CreatePressureNullSpace(DM dm, AppCtx *user, Vec *v, MatNullSpace *nullSpace)
{
  Vec              vec;
  PetscErrorCode (*funcs[2])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void* ctx) = {zero, one};
  PetscErrorCode   ierr;

  PetscFunctionBeginUser;
  ierr = DMCreateGlobalVector(dm, &vec);CHKERRQ(ierr);
  ierr = DMProjectFunction(dm, 0.0, funcs, NULL, INSERT_ALL_VALUES, vec);CHKERRQ(ierr);
  ierr = VecNormalize(vec, NULL);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) vec, "Pressure Null Space");CHKERRQ(ierr);
  ierr = VecViewFromOptions(vec, NULL, "-null_space_vec_view");CHKERRQ(ierr);
  ierr = MatNullSpaceCreate(PetscObjectComm((PetscObject) dm), PETSC_FALSE, 1, &vec, nullSpace);CHKERRQ(ierr);
  if (v) {*v = vec;}
  else   {ierr = VecDestroy(&vec);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* Add a vector in the nullspace to make the continuum integral 0.

   If int(u) = a and int(n) = b, then int(u - a/b n) = a - a/b b = 0
*/
static PetscErrorCode CorrectDiscretePressure(DM dm, MatNullSpace nullspace, Vec u, AppCtx *user)
{
  PetscDS        prob;
  const Vec     *nullvecs;
  PetscScalar    pintd, intc[2], intn[2];
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSSetObjective(prob, 1, pressure);CHKERRQ(ierr);
  ierr = MatNullSpaceGetVecs(nullspace, NULL, NULL, &nullvecs);CHKERRQ(ierr);
  ierr = VecDot(nullvecs[0], u, &pintd);CHKERRQ(ierr);
  if (PetscAbsScalar(pintd) > 1.0e-10) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Discrete integral of pressure: %g\n", (double) PetscRealPart(pintd));
  ierr = DMPlexComputeIntegralFEM(dm, nullvecs[0], intn, user);CHKERRQ(ierr);
  ierr = DMPlexComputeIntegralFEM(dm, u, intc, user);CHKERRQ(ierr);
  ierr = VecAXPY(u, -intc[1]/intn[1], nullvecs[0]);CHKERRQ(ierr);
  ierr = DMPlexComputeIntegralFEM(dm, u, intc, user);CHKERRQ(ierr);
  if (PetscAbsScalar(intc[1]) > 1.0e-10) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Continuum integral of pressure after correction: %g\n", (double) PetscRealPart(intc[1]));
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESConvergenceCorrectPressure(SNES snes, PetscInt it, PetscReal xnorm, PetscReal gnorm, PetscReal f, SNESConvergedReason *reason, void *user)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = SNESConvergedDefault(snes, it, xnorm, gnorm, f, reason, user);CHKERRQ(ierr);
  if (*reason > 0) {
    DM           dm;
    Mat          J;
    Vec          u;
    MatNullSpace nullspace;

    ierr = SNESGetDM(snes, &dm);CHKERRQ(ierr);
    ierr = SNESGetSolution(snes, &u);CHKERRQ(ierr);
    ierr = SNESGetJacobian(snes, &J, NULL, NULL, NULL);CHKERRQ(ierr);
    ierr = MatGetNullSpace(J, &nullspace);CHKERRQ(ierr);
    ierr = CorrectDiscretePressure(dm, nullspace, u, (AppCtx *) user);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  SNES           snes;
  DM             dm;
  Mat            J;
  MatNullSpace   nullspace;
  Vec            u, r;
  AppCtx         user;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = SNESCreate(PETSC_COMM_WORLD, &snes);CHKERRQ(ierr);
  ierr = SNESSetDM(snes, dm);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(dm, &user);CHKERRQ(ierr);

  ierr = SetupDiscretization(dm, &user);CHKERRQ(ierr);
  ierr = DMPlexCreateClosureIndex(dm, NULL);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) u, "Solution");CHKERRQ(ierr);
  ierr = VecDuplicate(u, &r);CHKERRQ(ierr);

  ierr = DMPlexSetSNESLocalFEM(dm,&user,&user,&user);CHKERRQ(ierr);
  ierr = CreatePressureNullSpace(dm, &user, NULL, &nullspace);CHKERRQ(ierr);
  ierr = SNESSetConvergenceTest(snes, SNESConvergenceCorrectPressure, &user, NULL);CHKERRQ(ierr);

  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  ierr = SNESSetUp(snes);CHKERRQ(ierr);
  ierr = SNESGetJacobian(snes, &J, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = MatSetNullSpace(J, nullspace);CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&nullspace);CHKERRQ(ierr);

  ierr = SNESSolve(snes, NULL, u);CHKERRQ(ierr);
  ierr = VecViewFromOptions(u, NULL, "-sol_vec_view");CHKERRQ(ierr);
  {
    Vec e;
    void (*funcs[2])(PetscInt, PetscInt, PetscInt,
                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                     PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]) = {f0_error_u, f0_error_p};

    ierr = DMGetGlobalVector(dm, &e);CHKERRQ(ierr);
    ierr = DMProjectField(dm, 0.0, u, funcs, INSERT_ALL_VALUES, e);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) e, "Solution Error");CHKERRQ(ierr);
    ierr = VecViewFromOptions(e, NULL, "-error_vec_view");CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dm, &e);CHKERRQ(ierr);
  }

  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
