CHANGES: Underworld3
====================

2023-03-29
----------

  - Swarm reading using kd-tree to speed up locations
  - Swarm cycling now reverts to positions defined on the mesh and uses randomness to avoid 
      unexpected jamming of particles in stagnant regions
  - viscoplasticity seems to be doing the right thing




2023-03-01
----------

 - Use dmplex / dm adaptor to refine meshes
 - use kd-tree to find points across a partitioned mesh
 - use local kd-tree distances for fast, local rbf interpolants
 - swarm cycling version of pop control
 - integrals are working ok


2023-01-15
----------

 - >10000 core runs / timings
 - swarm checkpointing
 - mesh checkpointing
 - read back from mesh using kd-tree in order to provide flexible reading across different decompositions / mesh resolutions


2022-09-01
----------

Release 0.3.0 
-------------

 * Um



2021-08-12
----------
* Added our own kdtree functionality.

2021-03-12
----------
* Added a mesh-variable proxy for swarm variables.
  This variable is automatically kept in sync with
  the swarm variable. Currently we use the SciPy
  kdtree methods to map from swarm variables to 
  mesh variable nodes. 
* Added the `Stateful` mixin which helps to keep 
  track of the state of objects. 


2021-03-11
----------
* Added `MeshVariable.coord` attribute. Mesh variables
  now record their vertex coordinates array directly. 
* Added `parse_cmd_line_options()` routine which 
  ingests PETSc command line options.

Release 0.0.2 []
----------------
* Addition of `underworld3.maths.Integral` class for calculating
  integrals via PETSc & UW3 JIT method. 
* Rearrangement of UW3 classes to closer align with UW2.
* Addition of Rayleigh-Taylor model.


Release 0.0.1 []
----------------
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