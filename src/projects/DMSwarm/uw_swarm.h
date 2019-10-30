#include <petsc.h>
#include <petscsys.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmplex.h>
#include <petscdmswarm.h>
#include <petscviewerhdf5.h>

// use this wrapper ?
PetscErrorCode BuildSwarm2(DM dm, PetscInt ppcell, DM swarm);
PetscErrorCode BuildMesh( PetscInt dim, PetscInt* elements, PetscBool use_plex, PetscBool is_simplex, DM* dm);
PetscErrorCode BuildSwarm(DM dm, PetscInt nfields, const char* fieldnames[], PetscInt ppcell, DM* swarm);
PetscErrorCode DMBuildVelocityPressureFields(DM dm, PetscBool is_simplex);
PetscErrorCode DMGetGlobalElementCount(DM dm, PetscInt* elCount );
PetscErrorCode SwarmAdvectRK1(DM dm, Vec gridvel, DM swarm, PetscReal dt);
PetscErrorCode init_scalar(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
PetscErrorCode init_vector(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
PetscErrorCode main(int argc,char **args);
PetscErrorCode swarm_metric(DM swarm);
PetscErrorCode vec_print(Vec* vec, const char *idstr);

