# Blog Posts for underworldcode.org (Ghost)

Building blocks for the two GMD papers and the 3.0.0 release announcement.
Each post is self-contained but feeds into larger pieces.

## Status Key
- **draft**: outline or raw notes exist
- **ready**: content written, needs review
- **published**: live on underworldcode.org

---

## The Release Anchor

1. **"Underworld3 Reaches 3.0.0"**
   Status: draft (uw3-release-announcement.md — to be written last)
   Feeds into: index for all other posts
   Content: What changed since 0.99/JOSS. New features, fixed bugs, API changes.
   Links out to the explanatory posts below.

## Origin Story

2. **"Our Journey from Underworld2 to Underworld3"**
   Status: **published** (2026-03-23)
   Feeds into: Paper 1 introduction, release post
   Content: UW1 and UW2 as the same engine in different clothes. Why we rewrote:
   PETSc maturity, StGermain liability, hex-only meshes, SWIG agony, SNES unlock.
   What SymPy gave us. What UW3 can do that UW2 could not (curved BCs, coordinates,
   constitutive models without C, swappable time derivatives).

## Symbolic Machinery (→ Paper 1)

3. **"How Underworld3 Turns SymPy into C"**
   Status: **published** (2026-04-01)
   Feeds into: Paper 1 core (sections 4–6)
   Content: The six-stage JIT pipeline: strong form templates, automatic Jacobians,
   symbolic wrappers, unwrapping/non-dimensionalisation, C code generation,
   per-function caching, PETSc callback registration.
   URL: https://www.underworldcode.org/how-underworld3-turns-sympy-into-c/

4. **"Automatic Jacobians for Free"**
   Status: not started
   Feeds into: Paper 1 (solver integration)
   Content: How symbolic differentiation eliminates hand-coded Newton derivatives.
   Why this was impossible in UW2 (opaque C Functions). SNES integration.

5. **"Constitutive Models as Symbolic Objects"**
   Status: not started
   Feeds into: Paper 1 (constitutive models)
   Content: Composable rheology: viscous, plastic, elastic, transverse isotropic.
   How SymPy lets you mix and simplify material laws interactively.
   Contrast with UW2's fn.rheology (one C class per model).

6. **"Constants That Aren't Constant"**
   Status: not started
   Feeds into: Paper 1 (expression system)
   Content: The PetscDS constants mechanism for routing UWexpressions to C variables.
   Why this matters for time-dependent coefficients (BDF/AM ramp, continuation).

7. **"Time Derivatives You Can See"**
   Status: not started
   Feeds into: Paper 1 (time discretisation)
   Content: The DDt hierarchy: Lagrangian, Semi-Lagrangian, Eulerian, Symbolic.
   How BDF/AM schemes appear as symbolic rewrites. Contrast with UW2's fixed RK4.

8. **"Natural Mathematical Syntax in Scientific Python"**
   Status: not started
   Feeds into: Paper 1 (mathematical objects)
   Content: The MathematicalMixin: why `density * velocity` works.
   How operator overloading connects to SymPy's Matrix API.

## Units & Scaling (→ Paper 1)

9. **"Physical Units in Computational Geodynamics"**
   Status: not started
   Feeds into: Paper 1 (units system)
   Content: The Pint-based units system. String input, object storage,
   transparent container principle. Why units not dimensionality.

10. **"Non-dimensionalisation Without Tears"**
    Status: not started
    Feeds into: Paper 1 (units system)
    Content: Reference quantities, nondimensional unwrapping mode,
    solver works in scaled space while user thinks in physical units.

## Particles & Surfaces (→ Paper 2)

11. **"Finding Particles in a Parallel Mesh"**
    Status: not started
    Feeds into: Paper 2 core
    Content: How particle-to-processor assignment works: DMSwarm migration,
    spatial indexing, the handshake between PETSc's mesh decomposition and
    particle ownership. What happens when particles cross processor boundaries.

12. **"Particles That Know Calculus"**
    Status: not started
    Feeds into: Paper 2 (proxy variables)
    Content: Swarm variables as first-class symbolic objects. Proxy mesh
    variables via RBF projection. How particle data participates in weak forms.

13. **"Stress Has a History"**
    Status: not started
    Feeds into: Paper 2 (material history)
    Content: Viscoelastic stress storage and advection. Explicit stress
    history architecture. Order-1 vs order-2 validation.

14. **"Ghost Boundaries Done Right"**
    Status: not started
    Feeds into: Paper 2 (boundary integrals)
    Content: Internal and external boundary integrals in parallel.
    The ownership problem, PETSc patches, MPI-correct integration.

15. **"Tracking Interfaces in Large Deformation"**
    Status: not started
    Feeds into: Paper 2 (surfaces)
    Content: Material interfaces, level-set weighted composites,
    population control.

## Geometry & Meshing (→ Paper 2)

16. **"Meshing for Planetary Scale"**
    Status: not started
    Feeds into: Paper 2 (meshing)
    Content: Cubed-sphere construction, RegionalSphericalBox,
    geodetic projections. Why lat-lon grids have singularities.

17. **"Geographic Meshes for Regional Geodynamics"**
    Status: not started
    Feeds into: Paper 2 (meshing)
    Content: Ellipsoidal Earth geometry with topography,
    RegionalGeographicBox, boundary labelling, coordinate transforms.
    Real-world regional modelling with proper geodesy.

18. **"Symbolic Geometry: Differential Operators in Curvilinear Coordinates"**
    Status: not started
    Feeds into: Paper 1 or 2 (coordinates)
    Content: How gradient/divergence/curl auto-adjust for
    spherical/cylindrical via the coordinate system factory.

19. **"Adaptive Meshes That Follow the Physics"**
    Status: not started
    Feeds into: Paper 2 (adaptivity)
    Content: Metric-tensor-based adaptation, swarm-mediated variable
    transfer during remeshing.

## Infrastructure & Craft

20. **"Mesh Variables and PETSc Vectors: Keeping Arrays in Sync"**
    Status: not started
    Feeds into: Paper 1 & 2
    Content: The self-validating data cache, NDArray_With_Callback,
    why `with mesh.access()` is gone and direct `.data[...]` works now.
    Contrast with UW2's context managers.

21. **"Safe Parallelism Without MPI Expertise"**
    Status: not started
    Feeds into: Paper 2
    Content: uw.pprint(), selective_ranks(), why PETSc handles the
    hard parts. The MPICH-on-macOS scaling bug story.

22. **"From Notebook to Supercomputer in Zero Edits"**
    Status: not started
    Feeds into: Paper 1 (introspection)
    Content: How the same Jupyter notebook runs on a laptop and
    12,000 cores. Mathematical display, literate computing philosophy.

23. **"Interactive 3D Geodynamics in a Browser"**
    Status: not started
    Feeds into: neither paper directly
    Content: PyVista/trame integration, P2 field visualisation,
    server proxy detection. Contrast with UW2's LavaVu.

## Development Process

24. **"AI and Scientific Software: What We Learned Rebuilding Underworld3"**
    Status: **published** (2026-03-23)
    Feeds into: standalone (high interest)
    Content: Co-evolution of code and AI tools. Four phases from first contact
    to productivity jumps. What works, what doesn't, the units system slog.
    URL: https://www.underworldcode.org/ai-and-scientific-software-what-we-learned-rebuilding-underworld3/

25. **"Notation Gymnastics in Continuum Mechanics"**
    Status: not started
    Feeds into: Paper 1 (tensors)
    Content: Voigt, Mandel, full tensor. Why Mandel preserves inner
    products. Dimension-independent indexing.

---

## Suggested Writing Order

Priority based on: feeds into papers, standalone interest, dependency chain.

| # | Post | Why first |
|---|------|-----------|
| 1 | Journey from UW2 to UW3 (#2) | Sets up everything else |
| 2 | SymPy into C (#3) | Paper 1 centrepiece |
| 3 | Finding particles (#11) | Paper 2 centrepiece |
| 4 | Units (#9) | Foundation, standalone interest |
| 5 | AI strategy (#24) | Standalone, high external interest |
| 6 | Arrays in sync (#20) | Practical, bridges both papers |
| 7 | Release announcement (#1) | Written last, links to all others |
