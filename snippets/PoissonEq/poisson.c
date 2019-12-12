#include <stdio.h>
#include <petsc.h>
#include <petscviewerhdf5.h>
#include <mpi.h>

/*
This example uses a 2D-DMDA to create a distributed vector.
On the vector we do a collective addition on it.
Very straight forward and elegant

command line args: -da_grid_x <M> -da_grid_y <N>
*/

typedef struct {
  char filename[2048]; /* The mesh file */
  PetscBool simplex;
  PetscScalar y0, y1, T0, T1, k, h;
} AppCtx;

static PetscErrorCode top_bc(PetscInt dim, PetscReal time, const PetscReal coords[], 
                                  PetscInt Nf, PetscScalar *u, void *ctx)
{
  AppCtx* model = (AppCtx*)ctx;
  u[0] = model->T1;
  return 0;
}
static PetscErrorCode analytic(PetscInt dim, PetscReal time, const PetscReal coords[], 
                               PetscInt Nf, PetscScalar *u, void *ctx)
{
  AppCtx* model = (AppCtx*)ctx;
  double y0 = model->y0; 
  double y1 = model->y1;
  double T0 = model->T0;
  double T1 = model->T1;
  double k = model->k;
  double h = model->h;
  double y = coords[1];
  double c0, c1;

  c0 = (T1-T0+h/(2*k)*(y1*y1-y0*y0)) / (y1-y0);
  c1 = T1 + h/(2*k)*y1*y1 - c0*y1;

  u[0] = -h /(2*k) * y * y + c0 * y + c1;

  return 0;
}

static PetscErrorCode fn_x(PetscInt dim, PetscReal time, const PetscReal coords[], 
                                  PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = coords[0];
  return 0;
}

static PetscErrorCode bottom_bc(PetscInt dim, PetscReal time, const PetscReal coords[], 
                                 PetscInt Nf, PetscScalar *u, void *ctx)
{
  AppCtx* model = (AppCtx*)ctx;
  u[0] = model->T0;
  return 0;
}


static void f0_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = -1*constants[1];
}

static void f1_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d_i;
  for( d_i=0; d_i<dim; ++d_i ) f0[d_i] = constants[0] * u_x[d_i];
}

/* < \nabla v, \nabla u + {\nabla u}^T >
   This just gives \nabla u, give the perdiagonal for the transpose */
static void g3_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  PetscInt d_i;

  for ( d_i=0; d_i<dim; ++d_i ) { g3[d_i*dim+d_i] = 1.0; }
}

static PetscErrorCode SetupProblem(DM dm, PetscDS prob, AppCtx *user)
{
  const PetscInt          comp   = 0; /* scalar */
  PetscInt                ids[4] = {1,2,3,4};
  PetscErrorCode          ierr;

  PetscFunctionBeginUser;

  { 
    PetscScalar constants[2];
    constants[0] = user->k;
    constants[1] = user->h;

    ierr = PetscDSSetConstants(prob, 2, constants);CHKERRQ(ierr);
  }

  ierr = PetscDSSetResidual(prob, 0, f0_u, f1_u);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL,  NULL,  g3_uu);CHKERRQ(ierr);
  
  /* with -dm_plex_separate_marker we split the wall markers- closer to uw feel */
  ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, NULL, "marker", 
                            0, 0, NULL, /* field to constain and number of constained components */
                            (void (*)(void)) top_bc, 1, &ids[2], user);CHKERRQ(ierr);
                            
  ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, NULL, "marker", 
                            0, 0, NULL, /* field to constain and number of constained components */
                            (void (*)(void)) bottom_bc, 1, &ids[0], user);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, AppCtx *user)
{
  DM              cdm = dm;
  PetscFE         fe;
  //PetscQuadrature q;
  PetscDS         prob;
  PetscSpace      space;
  PetscInt        dim;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  /* Create finite element */
  ierr = PetscFECreateDefault(PetscObjectComm((PetscObject)dm), dim, 1, user->simplex, "temperature_", PETSC_DEFAULT, &fe);CHKERRQ(ierr);
  ierr = PetscFEGetBasisSpace(fe, &space);CHKERRQ(ierr);
  /* Set discretization and boundary conditions for each mesh */
  ierr = DMSetField(dm,0,NULL,(PetscObject)fe);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);
  ierr = DMGetDS(dm, &prob);
  ierr = SetupProblem(dm, prob, user);CHKERRQ(ierr);
  while (cdm) {
    ierr = DMSetUp(cdm);
    ierr = DMGetCoarseDM(cdm, &cdm);CHKERRQ(ierr);
  }
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char**argv) {

    PetscInt size;
    DM dm;
    Vec u;
    AppCtx user;
    SNES snes;
    DMLabel label;
    Vec gVec, lVec, xy;
    PetscErrorCode ierr;    
    PetscInt retCode=1;
    PetscInt dim = 2;
    PetscReal max[2],min[2];
    PetscInt elements[2] = {4,3};
    double l2_error, tol=1e-1;

    PetscInitialize( &argc, &argv, (char*)0, NULL );

    // define constants on the 'user' data structure
    user.y0 = -0.6;
    user.y1 = 1.3;
    user.k  = 0.5;
    user.h  = 10;
    user.T0  = 4;
    user.T1  = 8;
    user.simplex = PETSC_FALSE;

    // read in some command line options
    PetscOptionsBegin(PETSC_COMM_WORLD, "", "Julian 2D Poisson test", PETSC_NULL); // must call before options
    /* Can force termperture element type with the following
    ierr = PetscOptionsSetValue(NULL, "-temperature_petscspace_degree", "1");CHKERRQ(ierr); */
    ierr=PetscOptionsBool("-simplex", "use simplicies", "n/a", user.simplex, &user.simplex, NULL);CHKERRQ(ierr);
    ierr=PetscOptionsIntArray("-elRes", "element count (default: 4,4)", "n/a", elements, &dim, NULL);CHKERRQ(ierr);
    PetscOptionsEnd();

    ierr = SNESCreate(PETSC_COMM_WORLD, &snes);CHKERRQ(ierr);
    
    // user can only define y extent for now. Need to generalise.
    min[0] = 0; max[0] = 1;
    min[1] = user.y0; max[1] = user.y1;

    DMPlexCreateBoxMesh(PETSC_COMM_WORLD, 2, 
                        user.simplex,       
                        elements, min, max,   
                        NULL, PETSC_TRUE,&dm);
    /* Distribute mesh over processes */
    {
      PetscPartitioner part;
      DM               pdm = NULL;

      ierr = DMPlexGetPartitioner(dm, &part);CHKERRQ(ierr);
      ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);
      ierr = DMPlexDistribute(dm, 0, NULL, &pdm);CHKERRQ(ierr);
      if (pdm) {
        ierr = DMDestroy(&dm);CHKERRQ(ierr);
        dm  = pdm;
      }
    }
    ierr = DMLocalizeCoordinates(dm);CHKERRQ(ierr); /* needed for periodic only */
    ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
    
    /* The mesh is output to HDF5 using options */
    ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);

    SetupDiscretization(dm, &user);

    /* Calculates the index of the 'default' section, should improve performance */
    ierr = DMPlexCreateClosureIndex(dm, NULL);CHKERRQ(ierr);
    /* Sets the fem routines for boundary, residual and Jacobian point wise operations */
    ierr = DMPlexSetSNESLocalFEM(dm, NULL, NULL, NULL);CHKERRQ(ierr);
    /* Get global vector */
    ierr = DMCreateGlobalVector(dm, &u);CHKERRQ(ierr);

    /* Update SNES */
    ierr = SNESSetDM(snes, dm); CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
    
    /* Solve and output*/
    {
      PetscErrorCode (*initialGuess[1])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void* ctx) = {fn_x};
      PetscErrorCode (*exactFunc[1])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void* ctx) = {analytic};
      AppCtx *ctxs[1];
      Vec    lu;
      ctxs[0] = &user;
      
      ierr = DMProjectFunction(dm, 0.0, initialGuess, (void**)ctxs, INSERT_VALUES, u);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) u, "Initial Solution");CHKERRQ(ierr);
      ierr = VecViewFromOptions(u, NULL, "-initial_vec_view");CHKERRQ(ierr);

      ierr = SNESSolve(snes, NULL, u);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) u, "Solution");CHKERRQ(ierr);
      ierr = VecViewFromOptions(u, NULL, "-sol_vec_view");CHKERRQ(ierr);
      
      // build an analytic field
      Vec r;
      ierr = DMGetGlobalVector(dm, &r);
      ierr = DMProjectFunction(dm, 0.0, exactFunc, (void**)ctxs, INSERT_ALL_VALUES, r);
      ierr = PetscObjectSetName((PetscObject) r, "Analytic");CHKERRQ(ierr);
      ierr = VecViewFromOptions(r, NULL, "-ana_vec_view");CHKERRQ(ierr);
      ierr = DMRestoreGlobalVector(dm, &r);
 
      // test the L2 error
      ierr = DMComputeL2Diff(dm, 0, exactFunc,(void**)ctxs, u, &l2_error );
   }
    /*************************************
    // Output dm and temperature solution
    // to a given filename
    {
      PetscViewer h5viewer;
      PetscViewerHDF5Open(PETSC_COMM_WORLD, "sol.h5", FILE_MODE_WRITE, &h5viewer);
      PetscViewerSetFromOptions(h5viewer);
      DMView(dm, h5viewer);
      PetscViewerDestroy(&h5viewer);

      PetscViewerHDF5Open(PETSC_COMM_WORLD, "sol.h5", FILE_MODE_APPEND, &h5viewer);
      PetscViewerSetFromOptions(h5viewer);
      VecView(u, h5viewer);
      PetscViewerDestroy(&h5viewer);
    }
    **************************************/
	      
    VecDestroy(&u);
    SNESDestroy(&snes);
    DMDestroy(&dm);

    if( l2_error > tol ) { 
      ierr = PetscPrintf(PETSC_COMM_WORLD, "\n*** L2 Error %.5e > tolerance (1e-10)\n", l2_error);
      retCode = 1;
    } else {
      ierr = PetscPrintf(PETSC_COMM_WORLD, "\n*** L2 Error %.5e\n", l2_error); 
      retCode = 0;
    }

    PetscFinalize();
    return retCode;
}
