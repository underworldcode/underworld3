#include "poisson_setup.h"

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

PetscErrorCode SetupProblem(DM dm, PetscDS prob, void *_user)
{
  const PetscInt          comp   = 0; /* scalar */
  PetscInt                ids[4] = {1,2,3,4};
  PetscErrorCode          ierr;
  AppCtx*                 user = (AppCtx*) _user;

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

PetscErrorCode SetupDiscretization(DM dm, AppCtx *user)
{
  DM              cdm = dm;
  PetscFE         fe;
  //PetscQuadrature q;
  PetscDS         prob;
  PetscSpace      space;
  PetscInt        dim;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  ierr = DMSetApplicationContext(dm,user);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  /* Create finite element */
  ierr = PetscFECreateDefault(PetscObjectComm((PetscObject)dm), dim, 1, user->simplex, "temperature_", PETSC_DEFAULT, &fe);CHKERRQ(ierr);
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
