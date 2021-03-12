CHANGES: Underworld3
====================
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