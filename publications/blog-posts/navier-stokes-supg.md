---
title: "Navier-Stokes on the Grid"
status: draft (held back; a later post)
feeds_into: [paper-2, release-post]
target: underworldcode.org (Ghost)
tags: [underworld, navier-stokes, supg, transport, benchmarks, geodynamics]
---

Underworld3 has had a Navier-Stokes solver for a while, built on the semi-Lagrangian time derivative that we use for most transport problems. It works, and it has the useful property that the advection is handled by tracing characteristics back through the flow, so the linear system that we solve at each step is the Stokes system that we already know how to precondition. This post is about a second solver that we have just added, which assembles the advection term on the grid and stabilises it with the streamline-upwind Petrov-Galerkin (SUPG) method. We describe why we built it, the design decisions in it, and what the benchmarks showed us, including one measurement that went somewhere we did not expect.

## Why a second solver

Earlier this year we added an Eulerian, implicit SUPG solver for scalar transport (`uw.systems.AdvDiffusion`) as a drop-in alternative to the semi-Lagrangian scheme. On the mantle-convection benchmarks the two schemes give equivalent answers, but the Eulerian one costs about a tenth as much per step, is exactly conservative, and gives the same answer on any number of processors. The semi-Lagrangian scheme keeps its advantage at large Courant numbers, where the grid scheme's Crank-Nicolson step starts to ring. So we now recommend SUPG for convection and the semi-Lagrangian scheme when the time step is much larger than the cell-crossing time.

The scalar solver has one limitation that matters for us: it moves scalars. The semi-Lagrangian scheme extends to vectors and tensors with no extra work because the trace-back is the same for every component. A grid scheme needs the vector residual and a stabilisation parameter for the vector equation, and that extra work is what stands between the scalar solver and a stress-transport scheme for viscoelastic materials. Momentum is the natural first vector problem. It has exact solutions and a literature of benchmarks, and it is the problem where the advection term is nonlinear, so it exercises everything the viscoelastic case will need.

## The residual, and what goes into the stabilisation

The solver is `uw.systems.NavierStokes`. It is a subclass of the Stokes saddle-point solver, so it inherits the block preconditioning, the constitutive models, and the boundary condition machinery. What it adds is a momentum residual with the material derivative in it:

$$
\mathbf{f} _ 0 = \rho \left( \dot{\mathbf{u}} + (\mathbf{a}\cdot\nabla)\mathbf{u} \right) - \mathbf{f},
\qquad
\mathbf{F} _ 1 = \boldsymbol{\sigma} - p\mathbf{I} + \tau _ s \, \mathbf{R} \otimes \mathbf{a} .
$$

The time derivative comes from the Eulerian history manager that the scalar solver uses (Crank-Nicolson or BDF2 from the stored velocity levels), $\mathbf{a}$ is the advecting velocity, and the last term in the flux is the SUPG term: the strong residual $\mathbf{R}$ weighted by the streamline direction, with $\tau _ s$ the usual stabilisation parameter built from the time step, the cell size and the local velocity, and the viscosity $\nu = \eta / \rho$.

The strong residual is the place where we made a mistake first time round. $\mathbf{R}$ is what the momentum equation would evaluate to at the current iterate if it were evaluated pointwise, and the SUPG term adds diffusion along streamlines in proportion to it. That is what makes the method consistent: at the exact solution $\mathbf{R}$ vanishes and the stabilisation switches itself off. Our first version used $\mathbf{f} _ 0$ as the residual, which is the natural thing to do when you are copying the scalar solver, where $\mathbf{f} _ 0$ is the whole strong form. For momentum it is not. The pressure gradient balances the advection at the exact solution, and leaving it out leaves $\mathbf{R}$ of order one everywhere the flow turns. The Kovasznay error at $h = 1/16$ dropped by a factor of three when we put $\nabla p$ in. The viscous term is still missing, because the pointwise kernels see first derivatives only, and that is the residual inconsistency we can measure in the spatial convergence rates below.

## No stress history

The semi-Lagrangian Navier-Stokes solver carries a history of the viscous stress as well as the velocity. It has to. Its Crank-Nicolson step needs the old viscous flux at the departure points of the characteristics, and the only way to have it there is to store the stress on the mesh and trace it back with the velocity. On the grid, the old flux is needed where it was formed, so we rebuild it from the stored velocity level, $\boldsymbol{\sigma}^n = 2\eta\,\dot{\boldsymbol{\varepsilon}}(\mathbf{u}^n)$, through the constitutive model. For a constant viscosity this is exact. For a strain-rate dependent one we recommend BDF2, which puts every spatial term at the new time level. The pressure has no history at all.

The point of this is not the memory saved. A stress that depends on its own history is a viscoelastic material, and the place for that is the constitutive model, not the transport scheme. Keeping the solver free of stress history keeps that boundary clean for when we come to build it.

## Choosing the advecting velocity

The momentum equation is nonlinear in the velocity. The solver lets you choose how to treat that, with the `advection` argument:

```python
ns = uw.systems.NavierStokes(mesh, v, p, rho=1.0, order=1,
                                 advection="extrapolated")
ns.constitutive_model = uw.constitutive_models.ViscousFlowModel
ns.constitutive_model.Parameters.shear_viscosity_0 = 1.0 / Re
```

The default takes $\mathbf{a} = 2\mathbf{u}^n - \mathbf{u}^{n-1}$, the second-order extrapolation of the stored levels. Each step is then a single linear Oseen solve through the Stokes fieldsplit, with a lag that is second order in the time step and no explicit stability limit. Setting `picard_iterations=n` re-solves the step with the latest iterate in the advecting velocity, which converges on the fully implicit fixed point without ever forming a tangent. Setting `advection="implicit"` puts $\mathbf{u}^{n+1}$ in the term and lets the nonlinear solver take Newton steps with the symbolic Jacobian, which we get for free from SymPy.

At a steady state all three coincide, and the Kovasznay runs confirm that to every printed digit. They differ on a time-dependent wake, and we quantify that below. There is also a regime, which we found in the lid-driven cavity at Re 400 with a Courant number of two, where the extrapolated step alone never settles: a lagged coefficient fed by the corner singularity sets up an alternating mode while the interior sits still. One Picard pass removes it, at about 20% more cost per step, and so does halving the time step. That is the case the Picard option is for.

## What the benchmarks showed

We ran the three standard tests for a laminar Navier-Stokes solver, and a fourth that measures the time accuracy directly.

**Kovasznay flow** is an exact steady solution at Re 40 on a rectangle. Marching from the exact solution to the discrete steady state and measuring the velocity error against the exact one gives the spatial error of the scheme. The Galerkin form (stabilisation off) converges at third order, which is the P2 interpolation error. SUPG converges at between 1.4 and 2 because of the missing viscous term in the residual; at $h = 1/32$ it is 2.6e-4 against 1.6e-5 for Galerkin. The semi-Lagrangian scheme on the same meshes is at first order and a factor of ten less accurate than SUPG, at seven times the cost per step. At this Reynolds number the element Reynolds number is below three on every mesh and the stabilisation is not needed, so what this benchmark shows is the price of it when it is not.

**The lid-driven cavity** is the harder test because the lid is singular at the corners. At Re 100 on a $1/32$ mesh the centreline velocity extrema are within 4% of Ghia, Ghia and Shin (1982) and the solution reaches an exact fixed point. At Re 400 on $1/48$ the extrema are 5 to 6% below the reference, steady to four digits, and the same for the extrapolated step and the Picard-corrected one, so that is the mesh talking. The Re 1000 case is still a transient after 1200 steps on a $1/64$ mesh with geometric multigrid on the velocity block and is a long-run comparison for another day.

**The cylinder wake** is the DFG 2D-2 benchmark of Schaefer and Turek (1996): a cylinder in a channel at Re 100, with the shedding frequency, the peak drag and lift coefficients and the pressure difference across the cylinder as the reference quantities. The shedding frequency comes out on the reference (Strouhal number 0.298 on the coarse mesh, 0.304 on the fine one, against 0.295 to 0.305), and so does the pressure difference. The lift peak is 18% low on the coarse mesh and 11% low on the fine one. The drag peak is 28% low on the coarse mesh and 23% low on the fine one, and the second number is the interesting one, because a discretisation error should have closed more than that when the cell size halved. That sent us looking for a cause that does not scale with the mesh.

[CYLINDER TAU SWEEP RESULT: to be written from the sweep.]

The semi-Lagrangian solver on the same mesh and time step has the shedding 13% too slow, with a drag closer to the reference and a lower lift peak, at three times the cost per step. The frequency is the quantity that the time integration owns, and there the Eulerian scheme is the accurate one.

**Vortex decay** is the Taylor-Green solution, a lattice of counter-rotating vortices that decays in place with the viscosity, $\mathbf{u}(t) = \mathbf{u}(0)\,e^{-2\nu t}$. It is the cleanest test of the time stepping because the exact solution is known at every instant, and on a $[0, \pi]^2$ box the walls carry no normal flow and no tangential stress, so free-slip walls are exact and there is nothing time-dependent on the boundary.

[VORTEX DECAY RESULT: to be written from the campaign.]

## Looking at the wake

We spent some time on the pictures, because a vortex street is the kind of flow that is easy to render badly. Two rules came out of it. The vorticity is a derivative of a P2 field, and evaluating it pointwise gives a piecewise-discontinuous mess. We form the curl in the right-hand side of a projection and project it to a P1 scalar, which is what we save and plot:

```python
omega = uw.discretisation.MeshVariable("omega", mesh, 1, degree=1)
omega_proj = uw.systems.Projection(mesh, omega)
omega_proj.uw_function = v.sym[1].diff(x) - v.sym[0].diff(y)
omega_proj.solve()
```

The second rule is not to clip the colour range. A colour scale that saturates at some fraction of the extreme vorticity paints the cylinder's boundary layer as a flat block, and a linear scale that reaches the extremes shows nothing in the wake. An inverse hyperbolic sine stretch of the field, with the range set just above the extremes, shows both. We save the velocity and the projected vorticity every step, which is cheap, and render from the checkpoints afterwards.

![Vorticity in the wake of the cylinder at Re 100 on the finer mesh, an inverse hyperbolic sine colour stretch on a muted red-blue scale, with passive tracer streaks released at the inlet.](figures/navier-stokes-supg/cylinder-wake-tracers.png)

The tracers are a passive swarm, seeded at the inlet every step in a narrow band around the centre line and advected with the flow. They show where the shedding tears the incoming streaks apart, which the vorticity field alone does not.

## Defects found on the way

A new solver runs through code paths that the old ones did not, and this one found five things that were already there:

- `mesh.cell_size()` was partition-dependent. The cell radius came from a nearest-centroid search that only saw the rank's own cells, so $\tau _ s$ differed across a partition seam and a two-rank Kovasznay run differed from the serial one by 5e-4. It also read stale coordinates after a mesh deformation. The field now reports each cell's own radius from the DM's coordinates (#687).
- A mesh variable that goes out of scope leaves its PETSc field in the DM, and two places assumed the variable registry and the DM's field list line up by position. Every later variable was then packed into the wrong slots. Fixed by packing by name.
- A mesh built with `refinement>=1` reloads from its checkpoint as a corrupt DM, so a restart from a refined mesh is silently wrong (#690). The checkpoint reader itself is fine; it matches by coordinates and does not care about ordering.
- The semi-Lagrangian solvers' `preconditioner` property has no effect, because their `solve()` runs the setup stages before the property is read (#683).
- Advecting a swarm in parallel fails when the swarm was filled with `add_particles_with_coordinates` (#693). This is why the tracer movies are serial runs. The defect dates from July and blocks parallel tracers until it is fixed.

## What comes next

The list, in the order we intend to take it: [REMAINING LIST: to be written from the plan after the sweeps.]
