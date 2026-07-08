static char help[] =
  "Reproducer: DMSetCoordinateDisc breaks DMPlexComputeBdIntegral on distributed meshes.\n\n"
  "After calling DMSetCoordinateDisc on a distributed gmsh-loaded DM,\n"
  "DMPlexComputeBdIntegral segfaults (--with-debugging=0) or reports\n"
  "'Null Pointer: Parameter #2' in PetscSpaceCreateSubspace (--with-debugging=1).\n"
  "The null pointer comes from PetscDualSpaceGetPointSubspace returning NULL\n"
  "for boundary face points on the coordinate FE created by DMSetCoordinateDisc.\n\n"
  "Without DMSetCoordinateDisc (using the coordinate space from the gmsh load),\n"
  "everything works. DMPlexCreateCoordinateSpace also works as a replacement.\n\n"
  "Requires a gmsh .msh file as first argument.\n"
  "Generate one with: gmsh -2 -clmax 0.125 -o box.msh <(echo '\n"
  "  SetFactory(\"OpenCASCADE\");\n"
  "  Rectangle(1) = {0,0,0,1,1};\n"
  "  Physical Surface(99) = {1};\n"
  "  Physical Curve(11) = {1}; // Bottom\n"
  "  Physical Curve(12) = {3}; // Top\n"
  "')\n\n"
  "Run:  mpirun -np 2 ./reproducer box.msh\n"
  "  --with-debugging=1: 'Null Pointer: Parameter #2' in PetscSpaceCreateSubspace\n"
  "  --with-debugging=0: SIGSEGV or MPI deadlock\n";

#include <petscdmplex.h>
#include <petscds.h>

/* Trivial boundary integrand: constant 1 */
static void bd_f0(PetscInt dim, PetscInt Nf, PetscInt NfAux,
    const PetscInt uOff[], const PetscInt uOff_x[],
    const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
    const PetscInt aOff[], const PetscInt aOff_x[],
    const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
    PetscReal t, const PetscReal x[], const PetscReal n[],
    PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = 1.0;
}

int main(int argc, char **argv)
{
  DM          dm;
  const char *meshfile;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  /* Get mesh filename from command line */
  PetscCheck(argc > 1, PETSC_COMM_WORLD, PETSC_ERR_USER, "Usage: %s <mesh.msh>", argv[0]);
  meshfile = argv[1];

  /* Load gmsh mesh and distribute */
  PetscCall(DMPlexCreateGmshFromFile(PETSC_COMM_WORLD, meshfile, PETSC_TRUE, &dm));
  {
    DM distDM = NULL;
    PetscCall(DMPlexDistribute(dm, 0, NULL, &distDM));
    if (distDM) { PetscCall(DMDestroy(&dm)); dm = distDM; }
  }
  {
    PetscInt pStart, pEnd;
    PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "  chart = (%d, %d)\n", pStart, pEnd));
    PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT));
  }

  /* ── THIS TRIGGERS THE BUG ──
   * Call DMSetCoordinateDisc with a user-created PetscFE.
   * This clears the cached coordinate field (DMSetCoordinateField(dm, NULL)).
   * When DMPlexComputeBdIntegral later calls DMGetCoordinateField, PETSc
   * lazily recreates it via DMCreateCoordinateField_Plex. The new coordinate
   * field's FE (set by DMSetCoordinateDisc) has broken dual space point
   * subspaces — PetscDualSpaceGetPointSubspace returns NULL for boundary
   * face points, causing PetscFECreatePointTrace to segfault.
   *
   * WORKAROUND: use DMPlexCreateCoordinateSpace(dm, 1, FALSE, FALSE) instead.
   * It creates the FE internally with correct subspace initialisation.
   */
  {
    PetscFE coordFE;
    PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, 2, 2, PETSC_TRUE, "coord_", 3, &coordFE));
    PetscCall(DMSetCoordinateDisc(dm, coordFE, PETSC_FALSE, PETSC_FALSE));
    PetscCall(PetscFEDestroy(&coordFE));
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "DMSetCoordinateDisc done\n"));

  /* Add a field and create DS */
  {
    PetscFE fe;
    PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, 2, 1, PETSC_TRUE, "field_", -1, &fe));
    PetscCall(DMSetField(dm, 0, NULL, (PetscObject)fe));
    PetscCall(PetscFEDestroy(&fe));
    PetscCall(DMCreateDS(dm));
  }

  /* DMPlexComputeBdIntegral — crashes here */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "DMPlexComputeBdIntegral...\n"));
  {
    Vec          gvec;
    DMLabel      label;
    PetscInt     Nf;
    PetscSection sec;

    PetscCall(DMGetLocalSection(dm, &sec));
    PetscCall(PetscSectionGetNumFields(sec, &Nf));
    PetscCall(DMCreateGlobalVector(dm, &gvec));
    PetscCall(VecSet(gvec, 1.0));
    PetscCall(DMGetLabel(dm, "Face Sets", &label));
    if (label) {
      void (**funcs)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[],
                     const PetscScalar[], const PetscScalar[], const PetscScalar[],
                     const PetscInt[], const PetscInt[],
                     const PetscScalar[], const PetscScalar[], const PetscScalar[],
                     PetscReal, const PetscReal[], const PetscReal[],
                     PetscInt, const PetscScalar[], PetscScalar[]);
      PetscScalar *integral, global_val;
      PetscInt     val = 11; /* physical group tag for bottom boundary */

      PetscCall(PetscCalloc1(Nf, &funcs));
      funcs[0] = (typeof(funcs[0]))bd_f0;
      PetscCall(PetscCalloc1(Nf, &integral));
      PetscCall(DMPlexComputeBdIntegral(dm, gvec, label, 1, &val, funcs, integral, NULL));

      PetscCallMPI(MPIU_Allreduce(&integral[0], &global_val, 1, MPIU_SCALAR, MPIU_SUM,
                                  PetscObjectComm((PetscObject)dm)));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  integral = %g\n", (double)global_val));
      PetscCall(PetscFree(funcs));
      PetscCall(PetscFree(integral));
    } else {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  no 'Face Sets' label found\n"));
    }
    PetscCall(VecDestroy(&gvec));
  }

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Done.\n"));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}
