# CHANGES: Underworld3

## 2026-05-20

  - **Snapshot toolkit (PRs #195, #196, #198)**: unified
    state-capture mechanism for Underworld3 models.

    User-facing API on `Model`:

    ```python
    token = model.save_state()                  # in-memory "stash for timesteps"
    model.load_state(token)                     # exact restore from token

    model.save_state(file="step42.snap.h5")     # persistent on-disk snapshot
    model.load_state("step42.snap.h5")          # restore from disk
    ```

    Same call, two storage modes. Captures the full model — every
    registered mesh and mesh-variable, every swarm with per-particle
    data, every solver-internal state-bearer (`ModelTracker`, `DDt`
    instances, anything exposing the `Snapshottable` contract). User
    guide: `docs/advanced/snapshot-restore.md`.

    - **In-memory mode** (#195): bit-exact discard-of-a-step
      guarantee, proven through real PETSc solves; parallel-correct
      under MPI at any fixed rank count (recovers from genuine
      cross-rank particle loss); rebuild-on-restore semantics for
      swarms (the discarded step *is* what restore exists to undo).
    - **`Model.tracker`** (#196): model-dwelling, snapshot-managed
      record of where a run is. Holds `time` / `step` / `dt` plus
      any quantity the user parks on it (`model.tracker.foo = ...`)
      — anything on the tracker reverts with the model. Solvers do
      not depend on it; using it is optional. A loose Python
      variable is not reverted by `load_state`; the same value on
      the tracker is.
    - **On-disk mode (v1.1, #198)**: HDF5 wrapper file + companion
      `.bulk/` directory. Wrapper is `h5ls`-inspectable without UW3
      in the loop (carries run name, schema version, sim time, step,
      MPI rank count, mesh/swarm/variable inventories). Bulk data
      uses PR #146's PETSc DMPlex primitives for mesh + meshvars;
      swarms get per-rank h5py sidecars for parallel correctness.
      Same-rank-count restart contract; clean errors on rank-count
      mismatch.

    Related API changes:

    - `MeshVariable.read_timestep` is now format-aware: detects
      whether the file is a legacy `write_timestep` per-variable
      file or a v1.1 snapshot wrapper and dispatches internally.
      Existing scripts that call `var.read_timestep(...)` work
      transparently against new files via a KDTree bridge over
      `MeshVariable.read_checkpoint` (#146).
    - `mesh.write_timestep()` / `mesh.write_checkpoint()` (PR #146)
      remain unchanged — they serve different use cases
      (visualisation + flexible/cross-resolution restart;
      memory-efficient same-rank PETSc reload for postprocessing).
      See "Choosing between paths" in the user guide.

    State-as-dataclass contract for solver helpers:
    `docs/developer/guides/state-as-dataclass.md` — declare
    mutable evolution state as a `SnapshottableState` dataclass
    and the snapshot mechanism captures/restores it with no extra
    plumbing. Retrofitted for all five DDt flavors in this work
    (`Symbolic`, `Eulerian`, `SemiLagrangian`, `Lagrangian`,
    `Lagrangian_Swarm`).

  - **Fix(ddt): `Lagrangian.__init__` typo (`uw.swarm.UWSwarm` →
    `uw.swarm.Swarm`)** (PR #184). Lagrangian DDt had been
    unconstructible since commit `0778b7d` (2025-07-07) — typo
    introduced during the unrelated `evalf` cleanup. Surfaced
    during the snapshot toolkit's retrofit work.

## 2026-03-14

  - **Release v3.0.0**: Merged development (398 commits) to main, tagged v3.0.0
  - **Release v0.99**: Tagged pixi-compatible snapshot of previous main for JOSS compatibility
  - Binder infrastructure overhaul:
    - Four versioned binder links: v0.99, v3.0.0, main, development
    - CI workflow handles nested branch names and `v*` tag builds
    - Manual dispatch overrides (`uw3_branch`, `image_tag`) for building old tags
    - Launcher dispatch payload fixed (field name mismatch prevented branch creation)
    - Dockerfile made resilient to different dependency versions (no hardcoded paths)
    - Deleted obsolete `uw3-release-candidate` branch
  - README updated: versioned binder badges, fixed CI badge filenames (.yml → .yaml)
  - `petsc_save_checkpoint()` now delegates to `write_timestep()` (fixes #80):
    - Gains vertex/cell compat groups, field projection, tensor repacking
    - Single checkpoint code path for easier maintenance
  - Cleaned up 10 merged feature/bugfix branches

## 2026-03-13

  - New `uw.maths.BdIntegral` for boundary and surface integrals (closes #47):
    - Wraps PETSc `DMPlexComputeBdIntegral` with MPI Allreduce and units support
    - Works on external boundaries, internal boundaries (`AnnulusInternalBoundary`, etc.)
    - Integrand can reference outward unit normal via `mesh.Gamma` / `mesh.Gamma_N`
    - Handles PETSc API change in v3.22.0 (function pointer signature)
  - PETSc patch for internal boundary assembly in parallel (fixes #77):
    - `plexfem-internal-boundary-ownership-fix.patch`
    - Ghost facet filtering in boundary integral, residual, and Jacobian paths
    - Part-consistent assembly (`support[key.part]`) with support-size guards
    - Fixes rank-dependent L2 norms for internal boundary natural BCs
  - MPI regression test: `tests/parallel/test_0765_internal_boundary_integral_mpi.py`
  - Boundary integral tests: `tests/test_0502_boundary_integrals.py`
  - Fixed binder image building from stale branch (fixes #71):
    - Dockerfile now uses build-arg for branch instead of hardcoded `uw3-release-candidate`
    - CI workflow passes triggering branch name to Docker build
  - Prevented worktree symlinks from being accidentally committed:
    - `.gitignore` patterns match both directories and symlinks
    - `./uw worktree create` writes exclusions to `.git/info/exclude`

## 2025-12-21

  - PETSc 3.24 compatibility verified (conda-forge petsc 3.24.2 works correctly)
  - Enhanced `uw.doctor()` diagnostics to detect PETSc version mismatches:
    - New check compares compile-time vs runtime PETSc library versions
    - Detects when extensions link to wrong `libpetsc.X.Y.dylib`
    - Provides clear fix instructions for rebuild
  - Updated `./uw doctor` to detect PETSc library mismatches even when import fails
  - Migrated all example files from `uw.options.get*` to new `uw.Params` pattern
  - Added `uw.pause()` for interactive notebook development:
    - Displays tidy, formatted message without Python tracebacks
    - Automatically skipped in non-notebook environments (scripts, HPC)
    - Same notebook runs interactively or unattended without modification
    - Override with `UW_NOTEBOOK_EMULATION` environment variable
  - Fixed problematic example files:
    - `Ex_SphericalStokes_Visualise.py`: Added configurable checkpoint paths
    - `Theory_VE_NavierStokes.py`: Marked WIP implementation sections
    - `Ex_Sheared_Layer_Test.py`: Fixed undefined `strain` variable, added `uw.pause()` for development review points

## 2024-09-01

  - Add Notebooks for the quickstart guide (quarto backend for rendering)
  - html5 embedded renders for 3D examples (low res, can be checked in)
  - Update the version numbering to match 3.x.x (consistent with UW1 and UW2)
  - uw expressions - updated interface
  - many bug fixes including advection-diffusion (fixed severe error in parallel implementation)

## 2024-06-15

  - Add JOSS submission information
  - Add licence file (licensing code and documentation in line with UW1 and UW2)
  - Add quickstart guide (and test on binder)
  - Clean up repository files

## 2024-04-30

  - uw expressions (sympy symbols that contain other expressions / functions)
  - use uw expressions for viscoplastic / viscoelastic constitutive models.

## 2024-01-31

 - Introduce particle sub-steps along path to improve accuracy in advection schemes
 - Bug fixes associated with benchmarking
 - Finalise the ability to create Meshvariables / Swarmvariables (with proxies) on the fly


## 2023-12-12

  - Surface integrals implementation in weak form (integrated into boundary conditions)
  - DDt interface (Semi-Lagrangian, Lagrangian)
  - Solver base class, add an Unknowns class to manage generic form of unknowns

## 2023-08-27

  - Geometrical multigrid (necessary changes to DM setup)

## 2023-05-20

  - Rewrite the constitutive model to understand the concept of materials
  - function.evalf is the rbf equivalent to function.evaluate it is fast but approximate

## 2023-03-29

  - Swarm reading using kd-tree to speed up locations
  - Swarm cycling now reverts to positions defined on the mesh and uses randomness to avoid
      unexpected jamming of particles in stagnant regions
  - viscoplasticity seems to be doing the right thing


## 2023-03-01

 - Use dmplex / dm adaptor to refine meshes
 - use kd-tree to find points across a partitioned mesh
 - use local kd-tree distances for fast, local rbf interpolants (approximate)
 - swarm cycling version of pop control
 - integrals are working ok


## 2023-01-15

 - >10000 core runs / timings
 - swarm checkpointing
 - mesh checkpointing
 - read back from mesh using kd-tree in order to provide flexible reading across different decompositions / mesh resolutions


## 2022-09-01

 - Release 0.3.0
 - Quarterly tidy up

## 2021-08-12

* Added our own kdtree functionality.

## 2021-03-12


* Added a mesh-variable proxy for swarm variables.
  This variable is automatically kept in sync with
  the swarm variable. Currently we use the SciPy
  kdtree methods to map from swarm variables to
  mesh variable nodes.

* Added the `Stateful` mixin which helps to keep
  track of the state of objects.


## 2021-03-11

* Added `MeshVariable.coord` attribute. Mesh variables
  now record their vertex coordinates array directly.
* Added `parse_cmd_line_options()` routine which
  ingests PETSc command line options.

## Release 0.0.2 []

* Addition of `underworld3.maths.Integral` class for calculating integrals via PETSc & UW3 JIT method.
* Rearrangement of UW3 classes to closer align with UW2.
* Addition of Rayleigh-Taylor model.


## Release 0.0.1 []

* Big rework of PETSc API usage. Now all
  systems create their own private solve
  PETSc variables, and all user facing variables
  (as encapsulated by the MeshVariable class)
  are effectively Aux variables.
* Systems retain public versions of their solution
 variables (stokes.u, stokes.p, poisson.u). These
  are copies of the actual solution variables
  (which are private).
* All variable read access must be done within
 the `mesh.access()` context manager. Write
 access is achieved by supplying a list of
  writeable variables (`mesh.access(stokes.u)`).
  Let's have a play with this and see if it feels
  like the way forward. It is a bit cumbersome
  for read access.
* Stokes velocity variable is now a vector instead
  of being a flat array.
* Swarm variable `project_from()` function. Not
  sure if we'll retain this one, but it's there for
  now. It uses a least squares approach.
* Documention updates.
* Introduced Python3 annotated parameters
  for testing.
* Model updates for interface changes.
* Update lavavu/plot prototype for swarm.
* Init commit of RT example. WIP. Need to fix
  fix interpolation routines which currently take
  20x the solve time.
* Updates for dockerfile and setup.py.
* Added `CHANGES.md`


---


## Development milestones

Reproduce the existing UW2 examples and extend to spherical / cylindrical

- [x] Spherical stokes
- [x] Buoyancy driven stokes (various geometries)
- [x] Compositional Buoyancy (Rayleigh-Taylor) level set
- [x] Compositional Buoyancy (Rayleigh-Taylor) via swarms (benchmark)
- [x] Advection/diffusion (slcn)
- [x] Advection/diffusion (swarm)
- [x] Constant viscosity convection
- [x] Convection, strongly temp-dep viscosity (stagnant lid)
- [x] Non-linear viscosity convection
- [ ] Quantitative Convection benchmarks (various geometries)
- [ ] Viscoelasticity (linear) benchmarks
- [x] Inertial terms (Navier-Stokes benchmarks)
- [x] Anisotropic viscosity

## Repository milestones

 - [x] pip install
 - [ ] conda install
 - [x] auto-formatting (e.g. black)
 - [x] pytest setup
 - [ ] pytest full-coverage
 - [x] pytest on commit / PR
 - [x] api docs (pdoc3)
 - [ ] jupyterbook docs (autobuild / publish)
 - [ ] JOSS compatibility:
   - [ ] LICENCE
   - [ ] citation txt
   - [ ] PR / Commit templates
   - [ ] Policies

### Checklist

Ingredients in achieving the above

Outcomes of Dec 22 Canberra Catch up

- [ ] Better constraint interface
- [ ] Small technical document on penalty constraint applications, ellipse geometry !!!
- [ ] UW2 examples reproduction.
- [ ] Mesh Variable constraints based on masks, using labels to apply penalty constraints.
 - requires label generation to use constraints.
- [ ] work out timing model and create user guide


[[T](https://github.com/underworldcode/underworld3/blob/master/src/ex1.c#L174)] Topology & Meshing

- [x] spherical, annulus
- [x] Cartesian
- [x] Different element types (at least Linear / Quadratic & Hex, Tet)
- [ ] Sandbox-style deforming mesh
  - [ ] Sandbox-style deforming mesh *with particles*
- [ ] Remeshing examples / adaptivity
- [ ] Earth topography / plate boundary adapted mesh

[[D](https://github.com/underworldcode/underworld3/blob/master/src/ex1.c#L268)] Disc

- [x] Cont Galerkin
- [ ] ~Disc Galerkin~
- [x] Semi-lagrangian
- [x] Free-slip BC on surface
  - [ ] Penalty - Needs improved interface for users)


[[P](https://github.com/underworldcode/underworld3/blob/master/src/ex1.c#L73)] Physics

- [x] Stokes-Boussinesq
- [x] Temp-dep rheology
- [x] Buoyancy driven convection
- [x] Non-linear viscosity / yielding
- [ ] Viscoelasticity
- [x] Navier-Stokes / interial terms
- [ ] Energy equation, resolve bdry layers
- [ ]
- [ ] ~kermit the 🐸~

[[S](https://github.com/underworldcode/underworld3/blob/master/src/ex1.c#L354)] Solvers

- [x] SNES - generic vector / scalar
- [x] Block Stokes solvers
- [x] Semi-lagrangian
- [x] Swarm-projected history terms
- [x] Projection solvers for function (sympy / variables) evaluation
- [ ] ~TS~  (address this later)

PIC for composition

- [x] Viscosity, buoyancy, ...
- [x] Nearest neighbour (k-d tree ? 🌳 )
- [ ] ~2D - L2 projection into FEM space (Petsc shall provide)~
- [ ] ~3D - L2 projection into FEM space (Petsc shall provide but not in 3D)~
- [x] Petsc Integrals
- [x] uw.function evaluate (for Sympy functions)

[[O1](https://github.com/underworldcode/underworld3/blob/master/src/ex1.c#L218) [O2](https://github.com/underworldcode/underworld3/blob/master/src/ex1.c#L382)] Output

- [x] HDF5 -> XDMF -> Paraview
- [x] pyvista (serial)
- [ ] LavaVu (or pyvista parallel workflow)

[[V](https://github.com/underworldcode/underworld3/blob/master/src/ex1.c#L35)] Exact solutions
- [ ] MMS
- [ ] Analytical
  - https://www.solid-earth-discuss.net/se-2017-71/se-2017-71.pdf
  -https://www.researchgate.net/publication/304784132_Benchmark_solutions_for_Stokes_flows_in_cylindrical_and_spherical_geometry


### Tasks

  - [x] Solver options - robust for viscosity contrasts, customisable and quick.
  - [ ] Investigate generalising context managers
  - [ ] ~Proper quadratic mesh interpolations for deformed meshes.~
  - [ ] DMLabels for higher order meshes, ie. using a label to set values in a Vec. How do you label mid-points?
  - [ ] Further integrals/reduction operators on fields variables.
  - [x] nKK nanoflann exposure.
  - [ ] create developer docs for software stack and general development strategy.
