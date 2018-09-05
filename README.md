# underworld3

**Milestones**

- [ ] Spherical stokes

- [x] Buoyancy drive stokes

- [ ] Advection diffusion

- [ ] High Ra, constant viscosity

- [ ] Highly temp-dep viscosity

**Objectives**

[[T](https://github.com/underworldcode/underworld3/blob/master/src/ex1.c#L174)] Topology & Meshing
- [x] spherical annulus - https://github.com/julesghub/cubie
- [x] Cartesian
- [x] Different element types (at least Linear / Quadratic & Hex, Tet)

[[D](https://github.com/underworldcode/underworld3/blob/master/src/ex1.c#L268)] Disc 
- [ ] Cont Galerkin 
- [ ] ~Disc Galerkin~
- [ ] Semi-lagrangian
- [ ] Free-slip BC on surface

[[P](https://github.com/underworldcode/underworld3/blob/master/src/ex1.c#L73)] Physics
- [ ] Buoyancy driven convection
- [ ] Stokes-Boussinesq
- [ ] Temp-dep rheology
- [ ] Energy equation, resolve bdry layers

[[S](https://github.com/underworldcode/underworld3/blob/master/src/ex1.c#L354)] Solvers
- [ ] Block Stokes solvers
- [ ] Semi-lagrangian
- [ ] TS?

PIC for composition
- [ ] Viscosity
- [ ] Nearest neighbour?
- [ ] L2 projection into FEM space?

[[O1](https://github.com/underworldcode/underworld3/blob/master/src/ex1.c#L218) [O2](https://github.com/underworldcode/underworld3/blob/master/src/ex1.c#L382)] Output
- [ ] HDF5 -> XDMF -> Paraview
- [ ] LavaVu

[[V](https://github.com/underworldcode/underworld3/blob/master/src/ex1.c#L35)] Exact solutions
- [ ] MMS
- [ ] Analytical 
  - https://www.solid-earth-discuss.net/se-2017-71/se-2017-71.pdf
  - https://www.researchgate.net/publication/304784132_Benchmark_solutions_for_Stokes_flows_in_cylindrical_and_spherical_geometry
