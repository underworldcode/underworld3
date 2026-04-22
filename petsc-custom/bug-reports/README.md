# PETSc Bug Reports

## DMSetCoordinateDisc breaks DMPlexComputeBdIntegral (2026-04-14)

**PETSc GitLab**: (link to filed issue)

**PETSc version**: 3.24.2 (release branch)

### Summary

`DMSetCoordinateDisc(dm, userFE, FALSE, FALSE)` on a distributed gmsh-loaded
DM causes `DMPlexComputeBdIntegral` to crash. The coordinate FE created by
`DMSetCoordinateDisc` has broken dual space point subspaces —
`PetscDualSpaceGetPointSubspace` returns NULL for boundary face points,
causing a null pointer dereference in `PetscSpaceCreateSubspace`.

- `--with-debugging=1`: clean error "Null Pointer: Parameter #2"
- `--with-debugging=0`: SIGSEGV or MPI deadlock

**Workaround**: Use `DMPlexCreateCoordinateSpace(dm, 1, FALSE, FALSE)` instead
of `DMSetCoordinateDisc`. It creates the FE internally with correct subspace
initialisation.

### Files

- `petsc_issue_dmsetcoordinatedisc.c` — self-contained C reproducer
- `box.msh` — gmsh mesh file (or generate with the command in the .c file)

### Build and run

```bash
# Generate mesh (if box.msh is not available)
gmsh -2 -clmax 0.125 -o box.msh -parse_string '
  SetFactory("OpenCASCADE");
  Rectangle(1) = {0,0,0,1,1};
  Physical Surface(99) = {1};
  Physical Curve(11) = {1};
  Physical Curve(12) = {3};
'

# Compile
mpicc -o repro petsc_issue_dmsetcoordinatedisc.c \
  -I$PETSC_DIR/include -I$PETSC_DIR/$PETSC_ARCH/include \
  -L$PETSC_DIR/$PETSC_ARCH/lib -lpetsc \
  -Wl,-rpath,$PETSC_DIR/$PETSC_ARCH/lib

# Run (crashes with np >= 1)
mpirun -np 2 ./repro box.msh
```

### Call chain

```
DMPlexComputeBdIntegral
  → DMPlexComputeBdIntegral_Internal
    → DMGetCoordinateField (field cache is NULL — cleared by DMSetCoordinateDisc)
      → DMCreateCoordinateField_Plex
        → DMFieldCreateDS → DMFieldDSGetHeightDisc
          → PetscFECreateHeightTrace → PetscFECreatePointTrace
            → PetscDualSpaceGetPointSubspace → returns NULL
            → PetscSpaceCreateSubspace(bsp, NULL, ...) → CRASH
```

### Related: Underworld3 issue #96

This bug was discovered during investigation of
[underworldcode/underworld3#96](https://github.com/underworldcode/underworld3/issues/96).
The UW3 workaround (`UW_DMForceCoordinateField`) forces coordinate field creation
and strips inherited boundary labels from the coordinate DM after
`DMPlexCreateCoordinateSpace`.
