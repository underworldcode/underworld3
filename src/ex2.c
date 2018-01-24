static char help[] = "Convection-diffusion Problem in 3D with FEM.\n\
We solve the convection-diffusion problem in a spherical\n\
shell, using a parallel unstructured mesh (DMPLEX) to discretize it.\n\
This example supports discretized auxiliary fields as well as\n\
multilevel nonlinear solvers.\n\n\n";

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscds.h>

typedef struct {
  char filename[2048]; /* The mesh file */
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
static PetscErrorCode exact(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = 10.0/PetscSqrtReal(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]) - 1.0;
  return 0;
}

/* < \nabla v, \nabla u + {\nabla u}^T >
   This just gives \nabla u, give the perdiagonal for the transpose */
static void g3_tt(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  PetscInt       d;

  for (d = 0; d < dim; ++d) { g3[d*dim+d] = 1.0; }
}

static void f0_t(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = 0;
}

static void f1_t(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f1[d] = u_x[d];
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->filename[0] = '\0';

  ierr = PetscOptionsBegin(comm, "", "Convection-Diffusion Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsString("-f", "Mesh filename to read", "ex1.c", options->filename, options->filename, sizeof(options->filename), &flg);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  const char    *filename = user->filename;
  size_t         len;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscStrlen(filename, &len);CHKERRQ(ierr);
  if (!len) SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Must supply a mesh filename");
  ierr = DMPlexCreateFromFile(comm, filename, PETSC_TRUE, dm);CHKERRQ(ierr);
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
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupProblem(DM dm, PetscDS prob, AppCtx *user)
{
  PetscInt                ids[2] = {1, 2};
  PetscErrorCode          ierr;

  PetscFunctionBeginUser;
  ierr = PetscDSSetResidual(prob, 0, f0_t, f1_t);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL,  NULL,  g3_tt);CHKERRQ(ierr);
  ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "wall_inner", "marker", 0, 0, 0, (void (*)(void)) one_scalar, 1, ids, user);CHKERRQ(ierr);
  ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "wall_outer", "marker", 0, 0, 0, (void (*)(void)) zero_scalar, 1, &ids[1], user);CHKERRQ(ierr);
  ierr = PetscDSSetExactSolution(prob, 0, exact);CHKERRQ(ierr);
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

static PetscErrorCode SetupDiscretization(DM dm, AppCtx *user)
{
  DM              cdm = dm;
  PetscFE         fe;
  PetscQuadrature q;
  PetscDS         prob;
  PetscInt        dim;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  /* Create finite element */
  ierr = PetscFECreateDefault(dm, dim, 1, PETSC_FALSE, "temperature_", PETSC_DEFAULT, &fe);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe, "temperature");CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fe, &q);CHKERRQ(ierr);
  /* Set discretization and boundary conditions for each mesh */
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSSetDiscretization(prob, 0, (PetscObject) fe);CHKERRQ(ierr);
  ierr = SetupProblem(dm, prob, user);CHKERRQ(ierr);
  while (cdm) {
    ierr = DMSetDS(cdm, prob);CHKERRQ(ierr);
    ierr = DMGetCoarseDM(cdm, &cdm);CHKERRQ(ierr);
  }
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;          /* Problem specification */
  SNES           snes;        /* nonlinear solver */
  Vec            u;           /* solution vector */
  AppCtx         user;        /* user-defined work context */
  PetscErrorCode ierr;

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
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  /* Solve */
  {
    PetscErrorCode (*initialGuess[1])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void* ctx) = {zero_scalar};
    Vec              lu;
    
    ierr = DMProjectFunction(dm, 0.0, initialGuess, NULL, INSERT_VALUES, u);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) u, "Initial Solution");CHKERRQ(ierr);
    ierr = VecViewFromOptions(u, NULL, "-initial_vec_view");CHKERRQ(ierr);
    ierr = DMGetLocalVector(dm, &lu);CHKERRQ(ierr);
    ierr = DMPlexInsertBoundaryValues(dm, PETSC_TRUE, lu, 0.0, NULL, NULL, NULL);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dm, u, INSERT_VALUES, lu);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm, u, INSERT_VALUES, lu);CHKERRQ(ierr);
    ierr = VecViewFromOptions(lu, NULL, "-local_vec_view");CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm, &lu);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) u, "Solution");CHKERRQ(ierr);
    ierr = SNESSolve(snes, NULL, u);CHKERRQ(ierr);
    ierr = VecViewFromOptions(u, NULL, "-sol_vec_view");CHKERRQ(ierr);
}
{
  PetscDS prob;
  DM       dmerr;
  PetscSection s;
  Vec      error;
  PetscInt cStart, cEnd, c;
  PetscErrorCode (*exactFuncs[1])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *);
  
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetExactSolution(prob, 0, &exactFuncs[0]);CHKERRQ(ierr);
  ierr = DMSNESCheckFromOptions(snes, u, exactFuncs, NULL);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PETSC_COMM_SELF, &s);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(s, 1);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(s, cStart, cEnd);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    ierr = PetscSectionSetDof(s, c, 1);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldDof(s, c, 0, 1);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(s);CHKERRQ(ierr);
  ierr = DMClone(dm, &dmerr);CHKERRQ(ierr);
  ierr = DMSetDefaultSection(dmerr, s);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&s);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dmerr, &error);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) error, "error");CHKERRQ(ierr);
  ierr = DMPlexComputeL2DiffVec(dm, 0.0, exactFuncs, NULL, u, error);CHKERRQ(ierr);
  ierr = VecViewFromOptions(error, NULL, "-error_vec_view");CHKERRQ(ierr);
  ierr = VecDestroy(&error);CHKERRQ(ierr);
}

  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
