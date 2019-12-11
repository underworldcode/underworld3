#include <petsc.h>

typedef struct {
  PetscBool simplex;
  PetscScalar y0, y1, T0, T1, k, h;
} AppCtx;

static PetscErrorCode top_bc(PetscInt dim, PetscReal time, const PetscReal coords[], PetscInt Nf, PetscScalar *u, void *ctx);
static void f0_u(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[],
                 const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[],
                 const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[],
                 PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[]);
static void f1_u(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[],
                 const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[],
                 const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[],
                 PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[]);
PetscErrorCode SetupProblem(DM dm, PetscDS prob, void *user);

