CHANGES: Underworld3
====================


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


2022-09-01
----------

## Release 0.3.0 


 * Um



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


* Addition of `underworld3.maths.Integral` class for calculating
  integrals via PETSc & UW3 JIT method. 
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
