#include <petsc.h>

typedef struct {
  PetscBool simplex;
  PetscScalar y0, y1, T0, T1, k, h;
} AppCtx;

PetscErrorCode SetupDiscretization(DM dm, AppCtx *user);
PetscErrorCode PetscJulesUseDM(AppCtx* crap, DM _dm);
PetscErrorCode PetscJules(AppCtx *user);
//PetscErrorCode PetscJules();

