CHANGES: Underworld3
====================

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