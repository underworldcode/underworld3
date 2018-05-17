static char help[] = "Convection-diffusion Problem in 3D with FEM.\n\
We solve the convection-diffusion problem in a spherical\n\
shell, using a parallel unstructured mesh (DMPLEX) to discretize it.\n\
This example supports discretized auxiliary fields as well as\n\
multilevel nonlinear solvers.\n\n\n";

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscds.h>

typedef enum {RUN_ANALYTIC_SIMPLE, RUN_ANALYTIC_FREE_SLIP, RUN_CONVECTION} RunType;

typedef struct {
  char    filename[2048]; /* The mesh file */
  RunType runType;        /* Type of run: analytic solution, convection */
} AppCtx;

static PetscErrorCode zero_scalar(PetscInt dim, PetscReal time, const PetscReal coords[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = 0.0;
  return 0;
}
static PetscErrorCode one_scalar(PetscInt dim, PetscReal time, const PetscReal coords[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = 1.0;
  return 0;
}
static PetscErrorCode zero_vector(PetscInt dim, PetscReal time, const PetscReal coords[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  PetscInt d;
  for (d = 0; d < dim; ++d) u[d] = 0.0;
  return 0;
}

static void f0_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt c;
  PetscScalar circle, force=0;
  circle = sqrt(   x[0]*x[0] 
                +  x[1]*x[1] 
                + (x[2]-7)*(x[2]-7) );
  if (circle < 1.3) { force = 1; }
  circle = sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);

  for (c = 0; c < dim; ++c) f0[c] = force/circle * x[c];
}

/* [P] The pointwise functions below describe all the problem physics */

/* gradU[comp*dim+d] = {u_x, u_y, v_x, v_y} or {u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y, w_z}
   u[Ncomp]          = {p} */
static void f1_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscInt Ncomp = dim;
  PetscInt       comp, d;

  for (comp = 0; comp < Ncomp; ++comp) {
    for (d = 0; d < dim; ++d) {
      /* f1[comp*dim+d] = 0.5*(gradU[comp*dim+d] + gradU[d*dim+comp]); */
      f1[comp*dim+d] = u_x[comp*dim+d];
    }
    f1[comp*dim+comp] -= u[Ncomp];
  }
}

/* gradU[comp*dim+d] = {u_x, u_y, v_x, v_y} or {u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y, w_z} */
static void f0_p(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;
  for (d = 0, f0[0] = 0.0; d < dim; ++d) f0[0] += u_x[d*dim+d];
}

/* < q, \nabla\cdot u >
   NcompI = 1, NcompJ = dim */
static void g1_pu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g1[d*dim+d] = 1.0; /* \frac{\partial\phi^{u_d}}{\partial x_d} */
}

/* -< \nabla\cdot v, p >
    NcompI = dim, NcompJ = 1 */
static void g2_up(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g2[d*dim+d] = -1.0; /* \frac{\partial\psi^{u_d}}{\partial x_d} */
}

/* < \nabla v, \nabla u + {\nabla u}^T >
   This just gives \nabla u, give the perdiagonal for the transpose */
static void g3_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  const PetscInt Nc = dim;
  PetscInt       c, d;

  for (c = 0; c < Nc; ++c) {
    for (d = 0; d < dim; ++d) {
      g3[((c*Nc+c)*dim+d)*dim+d] = 1.0;
    }
  }
}

static void j3_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  const PetscReal nu  = 1.;//PetscExpReal(2.0*PetscRealPart(a[2])*x[0]);
  PetscInt        cI, d;

  for (cI = 0; cI < dim; ++cI) {
    for (d = 0; d < dim; ++d) {
      g3[((cI*dim+cI)*dim+d)*dim+d] += nu; /*g3[cI, cI, d, d]*/
      g3[((cI*dim+d)*dim+d)*dim+cI] += nu; /*g3[cI, d, d, cI]*/
    }
  }
}

static void g0_pp(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  const PetscInt Nc = dim;
  PetscInt       c;

  for (c = 0; c < Nc; ++c) g0[c*Nc+c] = 1.0;
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  const char    *runTypes[3] = {"analytic_simple", "analytic_free_slip", "convection"};
  PetscInt       run;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->filename[0] = '\0';
  options->runType     = RUN_ANALYTIC_SIMPLE;

  ierr = PetscOptionsBegin(comm, "", "Convection-Diffusion Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsString("-f", "Mesh filename to read", "ex1.c", options->filename, options->filename, sizeof(options->filename), &flg);CHKERRQ(ierr);
  run  = options->runType;
  ierr = PetscOptionsEList("-run_type", "The run type", "ex1.c", runTypes, 3, runTypes[options->runType], &run, NULL);CHKERRQ(ierr);
  options->runType = (RunType) run;
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

/* [T] This function creates the mesh topolgoy and geometry from a file */
static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  const char    *filename = user->filename;
  size_t         len;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscStrlen(filename, &len);CHKERRQ(ierr);
  if (!len) SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Must supply a mesh filename");
  ierr = DMPlexCreateFromFile(comm, filename, PETSC_FALSE, dm);CHKERRQ(ierr);
  /* Distribute mesh over processes */
  {
    PetscPartitioner part;
    DM               pdm = NULL;

    ierr = DMPlexGetPartitioner(*dm, &part);CHKERRQ(ierr);
    ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);
    ierr = DMPlexDistribute(*dm, 0, NULL, &pdm);CHKERRQ(ierr);
    if (pdm) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = pdm;
    }
  }
  /* Enable conversion to p4est */
  {
    char      convType[256];
    PetscBool flg;

    ierr = PetscOptionsBegin(comm, "", "Mesh conversion options", "DMPLEX");CHKERRQ(ierr);
    ierr = PetscOptionsFList("-dm_plex_convert_type","Convert DMPlex to another format","ex1",DMList,DMPLEX,convType,256,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();
    if (flg) {
      DM dmConv;

      ierr = DMConvert(*dm,convType,&dmConv);CHKERRQ(ierr);
      if (dmConv) {
        ierr = DMDestroy(dm);CHKERRQ(ierr);
        *dm  = dmConv;
      }
    }
  }
  ierr = DMLocalizeCoordinates(*dm);CHKERRQ(ierr); /* needed for periodic */
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  /* [O] The mesh is output to HDF5 using options */
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupProblem(DM dm, PetscDS prob, AppCtx *user)
{
  PetscInt                ids[2] = {1, 2}; // the two components made by mesh generator
  PetscErrorCode          ierr;

  PetscFunctionBeginUser;
  ierr = PetscDSSetResidual(prob, 0, f0_u, f1_u);CHKERRQ(ierr);
  ierr = PetscDSSetResidual(prob, 1, f0_p, NULL);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL,  NULL,  g3_uu);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 0, 1, NULL, NULL,  g2_up, NULL);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 1, 0, NULL, g1_pu, NULL,  NULL);CHKERRQ(ierr);
  /*
  ierr = PetscDSSetJacobianPreconditioner(prob, 0, 0, NULL,  NULL,  NULL,  j3_uu);CHKERRQ(ierr);
  ierr = PetscDSSetJacobianPreconditioner(prob, 0, 1, NULL,  NULL,  g2_up, NULL);CHKERRQ(ierr);
  ierr = PetscDSSetJacobianPreconditioner(prob, 1, 0, NULL,  g1_pu, NULL,  NULL);CHKERRQ(ierr);
  ierr = PetscDSSetJacobianPreconditioner(prob, 1, 1, g0_pp, NULL, NULL,  NULL);CHKERRQ(ierr);
  */

  ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "wall", "marker", 
          0, 0, NULL, (void (*)(void)) zero_vector, 2, ids, user);CHKERRQ(ierr);
  /*
    Problem: We want to constrain the radial direction, but our global/local unknowns are x, y, z.

    Solution: Change the global unknowns to the natural mesh coordinate system

      We moving from the global to the local space, we must rotate the vector. We can do this with a GlobalToLocalHook,
      and the same thing for LocalToGlobal. Unfortunately, we would like to insert boundary values before the rotation.

      What if we indicate in the DS that a field is a vector field. Then a rotation matrix could be supplied, or a callback,
      which moves between the global and local spaces.

    Normals: The normal to an edge/face is given by the normal to the gradient of the coordinate field \nabla x at a point
      on the boundary of a cell. You can get this by taking the determinant of the gradient matrix with (1 1 1) appended.
  */
  PetscFunctionReturn(0);
}

/* [D] This function creates a PetscFE object for each field */
static PetscErrorCode SetupDiscretization(DM dm, AppCtx *user)
{
  DM              cdm = dm;
  PetscFE         fe[2];
  PetscQuadrature q;
  PetscDS         prob;
  PetscInt        dim;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  /* Create finite element */
  ierr = PetscFECreateDefault(dm, dim, dim, PETSC_FALSE, "velocity_", PETSC_DEFAULT, &fe[0]);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe[0], "velocity");CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fe[0], &q);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(dm, dim, 1, PETSC_FALSE, "pressure_", PETSC_DEFAULT, &fe[1]);CHKERRQ(ierr);
  ierr = PetscFESetQuadrature(fe[1], q);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe[1], "pressure");CHKERRQ(ierr);
  /* Set discretization and boundary conditions for each mesh */
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSSetDiscretization(prob, 0, (PetscObject) fe[0]);CHKERRQ(ierr);
  ierr = PetscDSSetDiscretization(prob, 1, (PetscObject) fe[1]);CHKERRQ(ierr);
  ierr = SetupProblem(dm, prob, user);CHKERRQ(ierr);
  while (cdm) {
    ierr = DMSetDS(cdm, prob);CHKERRQ(ierr);
    ierr = DMGetCoarseDM(cdm, &cdm);CHKERRQ(ierr);
  }
  ierr = PetscFEDestroy(&fe[0]);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe[1]);CHKERRQ(ierr);
  {
    PetscObject  pressure;
    MatNullSpace nullSpacePres;

    ierr = DMGetField(dm, 1, &pressure);CHKERRQ(ierr);
    ierr = MatNullSpaceCreate(PetscObjectComm(pressure), PETSC_TRUE, 0, NULL, &nullSpacePres);CHKERRQ(ierr);
    ierr = PetscObjectCompose(pressure, "nullspace", (PetscObject) nullSpacePres);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nullSpacePres);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode CreatePressureNullSpace(DM dm, AppCtx *user, Vec *v, MatNullSpace *nullSpace)
{
  Vec              vec;
  PetscErrorCode (*funcs[2])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void* ctx) = {zero_vector, one_scalar};
  PetscErrorCode   ierr;

  PetscFunctionBeginUser;
  ierr = DMGetGlobalVector(dm, &vec);CHKERRQ(ierr);
  ierr = DMProjectFunction(dm, 0.0, funcs, NULL, INSERT_ALL_VALUES, vec);CHKERRQ(ierr);
  ierr = VecNormalize(vec, NULL);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) vec, "Pressure Null Space");CHKERRQ(ierr);
  ierr = VecViewFromOptions(vec, NULL, "-null_space_vec_view");CHKERRQ(ierr);
  ierr = MatNullSpaceCreate(PetscObjectComm((PetscObject) dm), PETSC_FALSE, 1, &vec, nullSpace);CHKERRQ(ierr);
  if (v) {
    ierr = DMCreateGlobalVector(dm, v);CHKERRQ(ierr);
    ierr = VecCopy(vec, *v);CHKERRQ(ierr);
  }
  ierr = DMRestoreGlobalVector(dm, &vec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM               dm;          /* Problem specification */
  SNES             snes;        /* nonlinear solver */
  Vec              u;           /* solution vector */
  AppCtx           user;        /* user-defined work context */
  PetscErrorCode (*exactFuncs[2])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *) = {0,0};
  PetscReal        ferrors[2];
  PetscErrorCode   ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = SNESCreate(PETSC_COMM_WORLD, &snes);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = SNESSetDM(snes, dm);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(dm, &user);CHKERRQ(ierr);

  ierr = SetupDiscretization(dm, &user);CHKERRQ(ierr);
  ierr = DMPlexCreateClosureIndex(dm, NULL);CHKERRQ(ierr);

  ierr = DMPlexSetSNESLocalFEM(dm, &user, &user, &user);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(dm, &u);CHKERRQ(ierr);
  /* [S] The solver is constructed dynamically from command-line arguments */
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  /* Solve */
  if (exactFuncs[0]) {
    ierr = DMProjectFunction(dm, 0.0, exactFuncs, NULL, INSERT_ALL_VALUES, u);CHKERRQ(ierr);
    ierr = DMSNESCheckFromOptions(snes, u, exactFuncs, NULL);CHKERRQ(ierr);
  }
  {
    PetscErrorCode (*initialGuess[2])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void* ctx) = {zero_vector, zero_scalar};
    MatNullSpace     nullSpace;
    Mat              J;
    Vec              nullVec;
    PetscReal        pint;

    ierr = CreatePressureNullSpace(dm, &user, &nullVec, &nullSpace);CHKERRQ(ierr);
    ierr = DMCreateMatrix(dm, &J);CHKERRQ(ierr);
    ierr = MatSetNullSpace(J, nullSpace);CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes, J, J, NULL, NULL);CHKERRQ(ierr);

    ierr = DMProjectFunction(dm, 0.0, initialGuess, NULL, INSERT_VALUES, u);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) u, "Initial Solution");CHKERRQ(ierr);
    ierr = VecViewFromOptions(u, NULL, "-initial_vec_view");CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) u, "Solution");CHKERRQ(ierr);
    ierr = SNESSolve(snes, NULL, u);CHKERRQ(ierr);
    /* [O] The solution is output to HDF5 using options */
    ierr = VecViewFromOptions(u, NULL, "-sol_vec_view");CHKERRQ(ierr);

    ierr = VecDot(nullVec, u, &pint);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Integral of pressure: %g\n", (double) (PetscAbsScalar(pint) < 1.0e-14 ? 0.0 : PetscRealPart(pint)));CHKERRQ(ierr);
    ierr = VecDestroy(&nullVec);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nullSpace);CHKERRQ(ierr);
    ierr = MatDestroy(&J);CHKERRQ(ierr);
  }
  if (exactFuncs[0]) {
    ierr = DMComputeL2FieldDiff(dm, 0.0, exactFuncs, NULL, u, ferrors);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Error: [%g, %g]\n", ferrors[0], ferrors[1]);CHKERRQ(ierr);
  }

  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
