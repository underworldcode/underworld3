
static char help[] = "DMSwarm demonstrator of points in a DM \n\
Options: \n\
-elements 2,3   : number of elements and dimensions for mesh\n\
-ppcell 2       : gauss point particles per cell, -ve number is for subdivision point distribution\n\
-simplex 0      : use simplicies for mesh\n\
-steps 3        : number of advection steps\n";

#include <petsc.h>
#include <petscsys.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmplex.h>
#include <petscdmswarm.h>
#include <petscviewerhdf5.h>

PetscErrorCode vec_print(Vec* vec, const char *idstr);
PetscErrorCode init_vector(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  double factor = 1.; //PetscCosReal(2*PETSC_PI*time/(1e-1*15)); // time dependent option

  u[0] = -(x[1]-0.5);
  u[1] =  (x[0]-0.5);
  return 0;
}

PetscErrorCode init_scalar(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = PetscSqrtReal( (x[1]-0.3)*(x[1]-0.3)+(x[0]-0.5)*(x[0]-0.5) );
  return 0;
}

PetscErrorCode DMBuildVelocityPressureFields(DM dm, PetscBool is_simplex) {

  /* create a 2 "fields": velocity and pressure. Velocity will advect the swarm */
  PetscFunctionBegin;
  PetscDS ds;
  PetscFE fe[2];
  PetscInt dim;
  PetscErrorCode ierr;
  PetscQuadrature q;
  MPI_Comm comm;

  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
 
  // build velocity PetscFE
  ierr = DMGetDimension(dm, &dim);
  ierr = PetscFECreateDefault(comm, dim, dim, is_simplex, "vel_", PETSC_DEFAULT, &fe[0]);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe[0], "velocity");CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fe[0], &q);CHKERRQ(ierr);

  // build pressure PetscFE
  ierr = PetscFECreateDefault(comm, dim, 1, is_simplex, "pres_", PETSC_DEFAULT, &fe[1]);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe[1], "pressure");CHKERRQ(ierr);
  ierr = PetscFESetQuadrature(fe[1], q);CHKERRQ(ierr);

  // set the PetscFEs as fields to the DM
  ierr = DMSetField(dm, 0, NULL, (PetscObject) fe[0]);CHKERRQ(ierr);
  ierr = DMSetField(dm, 1, NULL, (PetscObject) fe[1]);CHKERRQ(ierr);
  
  // MUST create DS, although we won't use it
  ierr = DMCreateDS(dm);CHKERRQ(ierr);

  // can destroy PetscFE now
  ierr = PetscFEDestroy(&fe[0]);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe[1]);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

PetscErrorCode vec_print(Vec* vec, const char *idstr) {
  /*
   * Creates a hdf5 file of the vector and associate dm.
   * File is called 'idstr'
   */

  PetscViewer h5viewer;
  DM dm;

  // need to include geometry in .h5 - it's wasteful
  VecGetDM(*vec, &dm);
  PetscViewerHDF5Open(PETSC_COMM_WORLD, idstr, FILE_MODE_WRITE, &h5viewer);
  PetscViewerSetFromOptions(h5viewer);
  DMView(dm, h5viewer);
  PetscViewerDestroy(&h5viewer);

  PetscViewerHDF5Open(PETSC_COMM_WORLD, idstr, FILE_MODE_APPEND, &h5viewer);
  PetscViewerSetFromOptions(h5viewer);
  VecView(*vec, h5viewer);
  PetscViewerDestroy(&h5viewer);

  return(0);
}

PetscErrorCode SwarmAdvectRK1(DM dm, Vec gridvel, DM swarm, PetscReal dt) {
  /*
   * Advects a DMSwarm, dt in time, using a velocity vector
   */

  PetscFunctionBegin;
  DMInterpolationInfo ipInfo;
  Vec                 pvel;
  PetscReal           *coords;
  PetscInt            lsize, p_i, dim, blockSize;
  const PetscScalar   *pv;
  PetscErrorCode      ierr;

  /* get a vector fields on the swarm */
  ierr = DMSwarmVectorDefineField(swarm, DMSwarmPICField_coor);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(swarm, &pvel);CHKERRQ(ierr);
  ierr = DMSwarmGetLocalSize(swarm, &lsize);CHKERRQ(ierr);

  // create interpolation
  ierr = DMInterpolationCreate(PetscObjectComm((PetscObject)dm), &ipInfo);
  ierr = DMGetDimension(dm, &dim);
  ierr = DMInterpolationSetDim(ipInfo, dim);
  ierr = DMInterpolationSetDof(ipInfo, dim);

  // add points
  ierr = DMSwarmGetField(swarm, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
  ierr = DMInterpolationAddPoints(ipInfo, lsize, coords);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(swarm, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);

  ierr = DMInterpolationSetUp(ipInfo, dm, PETSC_FALSE);CHKERRQ(ierr);
  ierr = DMInterpolationEvaluate(ipInfo, dm, gridvel, pvel);CHKERRQ(ierr);
  ierr = DMInterpolationDestroy(&ipInfo);CHKERRQ(ierr);

  // update locations of points
  ierr = DMSwarmGetField(swarm, DMSwarmPICField_coor, &blockSize, NULL, (void**)&coords);CHKERRQ(ierr);
  ierr = VecGetArrayRead(pvel, &pv);CHKERRQ(ierr);
  for(p_i=0; p_i<lsize; p_i++) {
    coords[p_i*(dim)+0] += pv[p_i*blockSize+0] * dt;
    coords[p_i*(dim)+1] += pv[p_i*blockSize+1] * dt;
  }
  ierr = VecRestoreArray(pvel, (PetscScalar**)&pv);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(swarm, DMSwarmPICField_coor, NULL, NULL, (void**)&coords);CHKERRQ(ierr);
  ierr = DMSwarmMigrate(swarm, PETSC_TRUE);CHKERRQ(ierr);
  ierr = VecDestroy(&pvel);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode BuildMesh( PetscInt dim,
                          PetscInt* elements, 
                          PetscBool use_plex,
                          PetscBool is_simplex, 
                          DM* dm) {

  PetscErrorCode ierr;
  DM             distributedMesh = NULL;

  PetscFunctionBegin;

  /* Create the background cell DM */
  if(use_plex) {

    ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dim, is_simplex, elements, 
            NULL, NULL, NULL, PETSC_TRUE, dm);CHKERRQ(ierr);

    /* Distribute mesh over processes */
    ierr = DMPlexDistribute(*dm,0,NULL,&distributedMesh);CHKERRQ(ierr);
    if (distributedMesh) {
        ierr = DMDestroy(dm);CHKERRQ(ierr);
        *dm = distributedMesh;
    }
  } else {

    exit(1); // DISABLED for now
    PetscInt stencil_width=1,dof=2;
    ierr = DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,
            10,8,PETSC_DECIDE,PETSC_DECIDE,dof,stencil_width,NULL,NULL,dm);CHKERRQ(ierr);

    // must set coordinates on the mesh manually
    ierr = DMDASetUniformCoordinates(*dm,0.0,1,0.0,1.0,0.0,0.0);CHKERRQ(ierr);
  }

  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMSetUp(*dm);CHKERRQ(ierr);

  return 0;

}

PetscErrorCode DMGetGlobalElementCount(DM dm, PetscInt* elCount ) {
  /*
   *  get the number of cells from the mesh dm
   */
  PetscInt nel;
  PetscBool isdmplex, isdmda;
  PetscErrorCode ierr;
  DMType   dmtype;

  // get the DM type
  ierr = DMGetType(dm, &dmtype);CHKERRQ(ierr);
  ierr = PetscStrcmp(dmtype, DMPLEX, &isdmplex);CHKERRQ(ierr);
  ierr = PetscStrcmp(dmtype, DMDA  , &isdmda);CHKERRQ(ierr);

  if( isdmplex ) {
      PetscInt eStart,eEnd;
      ierr = DMPlexGetHeightStratum(dm, 0, &eStart, &eEnd);CHKERRQ(ierr);
      nel = eEnd - eStart;
  } else if( isdmda ) {
      PetscInt ne,nen;
      const PetscInt *elist;
      ierr = DMDAGetElements(dm, &ne, &nen, &elist);
      nel  = ne;
      ierr = DMDARestoreElements(dm, &ne, &nen, &elist);
  } else {
    return 666;
  }
  return 0;
}

PetscErrorCode swarm_metric(DM swarm) {
  /*
   * Print the global and local swarm populations
   */
  PetscInt size, lsize;
  PetscMPIInt rank;
  PetscErrorCode ierr;

  MPI_Comm_rank( PETSC_COMM_WORLD, &rank );
  ierr = DMSwarmGetSize(swarm, &size);CHKERRQ(ierr);
  ierr = DMSwarmGetLocalSize(swarm, &lsize);CHKERRQ(ierr);
  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,
          "\trank %d DMSwarm global size & local size: %d %d\n",
          rank, size, lsize );CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT);CHKERRQ(ierr);
}


PetscErrorCode BuildSwarm(DM dm, PetscInt nfields, 
                          const char* fieldnames[], 
                          PetscInt ppcell, 
                          DM* swarm) {
  DM          swarmcelldm;
  PetscInt    elCount,dim;
  PetscErrorCode ierr;

  ierr = DMGetDimension(dm, &dim );
  ierr = DMGetGlobalElementCount(dm, &elCount );

  ierr = DMCreate(PETSC_COMM_WORLD,swarm);CHKERRQ(ierr);
  ierr = DMSetType(*swarm,DMSWARM);CHKERRQ(ierr);
  ierr = DMSetDimension(*swarm,dim);CHKERRQ(ierr);

  ierr = DMSwarmSetType(*swarm,DMSWARM_PIC);CHKERRQ(ierr);
  ierr = DMSwarmSetCellDM(*swarm,dm);CHKERRQ(ierr);

  /* Register two scalar fields within the DMSwarm */
  ierr = DMSwarmRegisterPetscDatatypeField(*swarm,fieldnames[0],1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(*swarm,fieldnames[1],1,PETSC_INT);CHKERRQ(ierr);
  ierr = DMSwarmFinalizeFieldRegister(*swarm);CHKERRQ(ierr);

  if (ppcell > 0) {
      /* Set initial local sizes of the DMSwarm with a buffer length of zero */
      ierr = DMSwarmSetLocalSizes(*swarm,elCount*ppcell,0);CHKERRQ(ierr);
      /* Insert swarm coordinates cell-wise */
      ierr = DMSwarmInsertPointsUsingCellDM(*swarm,DMSWARMPIC_LAYOUT_GAUSS,ppcell);CHKERRQ(ierr);
  } else {
      ierr = DMSwarmSetLocalSizes(*swarm, elCount*abs(ppcell), 250 );
      ierr = DMSwarmInsertPointsUsingCellDM(*swarm,DMSWARMPIC_LAYOUT_SUBDIVISION,abs(ppcell));CHKERRQ(ierr);
  }

  { // initialise swarm variables 
    PetscReal *coords,*eta,x,y,z;
    PetscInt   p_i,t,bs,lsize,dim,*rank0;
    PetscMPIInt rank;
    PetscErrorCode ierr;
    MPI_Comm_rank( PETSC_COMM_WORLD, &rank );

    /* DMSwarmPICField_coord is a special field registered with DMSwarm during DMSwarmSetType() 
    * Special fields are DMSwarmField_pid, DMSwarmField_rank, DMSwarmPICField_coor, DMSwarmPICField_cellid*/
    ierr = DMSwarmGetField(*swarm, DMSwarmPICField_coor, &bs, NULL, (void**)&coords );CHKERRQ(ierr);
    ierr = DMSwarmGetField(*swarm, fieldnames[0], NULL, NULL, (void**)&eta );CHKERRQ(ierr);
    ierr = DMSwarmGetField(*swarm, fieldnames[1], NULL, NULL, (void**)&rank0 );CHKERRQ(ierr);

    ierr = DMSwarmGetLocalSize(*swarm, &lsize);CHKERRQ(ierr);
    for(p_i=0;p_i<lsize;p_i++) {
      x = coords[p_i*(bs)+0];
      y = coords[p_i*(bs)+1];
      if (dim == 3) z = coords[p_i*(bs)+2];

      rank0[p_i] = rank;
      eta[p_i] = PetscCosReal(4*PETSC_PI*x)*PetscCosReal(2*PETSC_PI*y);
    }
    ierr = DMSwarmRestoreField(*swarm, DMSwarmPICField_coor, &bs, NULL, (void**)&coords );CHKERRQ(ierr);
    ierr = DMSwarmRestoreField(*swarm, fieldnames[0], NULL, NULL, (void**)&eta );CHKERRQ(ierr);
    ierr = DMSwarmRestoreField(*swarm, fieldnames[1], NULL, NULL, (void**)&rank0 );CHKERRQ(ierr);
  }

}

int main(int argc,char **args)
{
  PetscErrorCode ierr;
  PetscInt       dim = 3, steps=1;
  PetscInt       elCount, elements[3], ppcell = 2;
  PetscBool      use_plex = PETSC_TRUE;
  PetscBool      simplex = PETSC_FALSE;

  DM  dm, swarmdm;
  Vec vec;
  PetscErrorCode (*initFuncs[2])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void* ctx) = {init_vector, init_scalar};
  ierr = PetscInitialize(&argc,&args,(char*)0,help);if(ierr) return ierr;

  { // grab cmd options
    PetscBool found;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","DMSwarm example options","");
    ierr = PetscOptionsIntArray("-elements", "initial elements/dimensions",NULL,elements,&dim,&found);CHKERRQ(ierr);
    if(!found) { dim=2; elements[0]=elements[1]=3; } // default options
    ierr = PetscOptionsInt("-ppcell", "particles per cell",NULL,ppcell,&ppcell,NULL);CHKERRQ(ierr);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-steps", "number of advection steps",NULL,steps,&steps,NULL);CHKERRQ(ierr);
    //ierr = PetscOptionsBool("-use_plex","if true use dmplex else dmda(untested)",NULL,use_plex,&use_plex,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-simplex","if true use simplicies else tensor products",NULL,simplex,&simplex,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();
  }

  PetscPrintf(PETSC_COMM_WORLD, "Build DM, a.k.a Mesh ... ");
  ierr = BuildMesh( dim, elements, use_plex, simplex, &dm);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD, "Done\n");

  PetscPrintf(PETSC_COMM_WORLD, "Build fields velocity and pressure ... ");
  ierr = DMBuildVelocityPressureFields(dm, simplex);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD, "Done\n");

  // create and initialise GlobalVector 'vec' with callbacks, initFuncs
  PetscPrintf(PETSC_COMM_WORLD, "Initialise velocity and pressure ... ");
  ierr = DMCreateGlobalVector(dm, &vec);CHKERRQ(ierr);
  ierr = DMProjectFunction(dm, 0.0, initFuncs, NULL, INSERT_VALUES, vec);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) vec, "solution");CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD, "Done\n");

  // save the result
  ierr = vec_print( &vec, "mesh.h5"); 

  char prefix[PETSC_MAX_PATH_LEN];
  int t_i=0;
  PetscInt nfields = 2;
  PetscScalar time,dt=1e-1;
  const char *fieldnames[] = {"viscosity","rank0"};
  // build swarm
  ierr = BuildSwarm(dm, nfields, fieldnames, ppcell, &swarmdm);
  
  // save
  PetscSNPrintf( prefix, PETSC_MAX_PATH_LEN-1, "swarm-%05d.xmf", t_i);
  ierr = DMSwarmViewFieldsXDMF(swarmdm,prefix,nfields,fieldnames);CHKERRQ(ierr);

  DM        vdm;
  IS        vis;
  PetscInt  vf[1] = {0};
  Vec       locvel,vel;
  
  // get sub vec for velocity and copy values to local vec
  ierr = DMCreateSubDM(dm, 1, vf, &vis, &vdm);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(vdm, &locvel);CHKERRQ(ierr);
  ierr = VecGetSubVector(vec, vis, &vel);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(vdm, vel, INSERT_VALUES, locvel);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(vdm, vel, INSERT_VALUES, locvel);CHKERRQ(ierr);
  ierr = VecRestoreSubVector(vec,vis,&vel);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm,&vec);CHKERRQ(ierr);

  for( t_i = 0 ; t_i < steps ; t_i++ ) {
    // save current timestep of particles
    PetscPrintf( PETSC_COMM_WORLD, "\nstep %05d save and advect", t_i);

    PetscSNPrintf( prefix, PETSC_MAX_PATH_LEN-1, "swarm-%05d.xmf", t_i);
    ierr = DMSwarmViewFieldsXDMF(swarmdm,prefix,nfields,fieldnames);CHKERRQ(ierr);

    SwarmAdvectRK1( vdm, locvel, swarmdm, dt );
    PetscPrintf( PETSC_COMM_WORLD, " ... Done\n");

    swarm_metric(swarmdm);
    time += dt;
  }
  PetscPrintf(PETSC_COMM_WORLD, "Finished model see output\n");
  // delete global vectors
  ISDestroy(&vis);CHKERRQ(ierr);
  VecDestroy(&locvel);CHKERRQ(ierr);
  VecDestroy(&vec);
  DMDestroy(&vdm);CHKERRQ(ierr);
  DMDestroy(&swarmdm);
  DMDestroy(&dm);
  ierr = PetscFinalize();
  return ierr;
}
