#include "poisson_setup.h"

static PetscErrorCode top_bc(PetscInt dim, PetscReal time, const PetscReal coords[], 
                                  PetscInt Nf, PetscScalar *u, void *ctx)
{
  AppCtx* model = (AppCtx*)ctx;
  u[0] = model->T1;
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
