# Turbulence Modelling for Underworld3 Navier-Stokes Solver

**Date**: 2026-02-12
**Status**: Design / Planning — not yet implemented
**Solver**: `SNES_NavierStokes` (`src/underworld3/systems/solvers.py`)

---

## Context

The `SNES_NavierStokes` solver resolves the full laminar Navier-Stokes equations with semi-Lagrangian advection (SLCN method, Spiegelman & Katz 2006). It uses a pluggable constitutive model that provides the stress tensor via a `.flux` property. Turbulence models that modify the **effective viscosity** slot directly into this existing architecture with minimal solver changes.

### Current Solver Structure

```
SNES_NavierStokes
├── F0: -bodyforce + ρ * DuDt.bdf(1) / Δt         (inertial + source)
├── F1: DFDt.adams_moulton_flux() - pI + penalty    (stress + pressure)
├── PF0: div(u)                                      (incompressibility)
└── constitutive_model.flux → viscous stress tensor
```

The constitutive model already supports nonlinear (strain-rate-dependent) viscosity via `ViscoPlasticFlowModel`. A turbulence model that adds an eddy viscosity follows exactly the same pattern.

---

## Recommended First Implementation: Smagorinsky LES

### The Model

Replace molecular viscosity with an effective viscosity:

$$\eta_{\text{eff}} = \eta + \eta_t, \quad \eta_t = \rho \left( C_s \Delta \right)^2 |\dot{\varepsilon}|$$

where:
- $C_s \approx 0.1$–$0.2$ is the Smagorinsky constant
- $\Delta$ is the local grid scale (element size)
- $|\dot{\varepsilon}|$ is the second invariant of the strain rate tensor

### Why This Model

1. **Constitutive model pattern** — It is a nonlinear viscosity. Can be implemented as a new `SmagorinskyFlowModel` constitutive model, or set up symbolically using the existing `ViscousFlowModel`:

   ```python
   edot_II = stokes.strainrate_1d
   eta_t = rho * (Cs * delta)**2 * sympy.sqrt(2 * edot_II.dot(edot_II))
   solver.constitutive_model.Parameters.viscosity = eta_molecular + eta_t
   ```

2. **No extra equations** — No additional PDEs, no new mesh variables, no transport equations. The SNES nonlinear solver already handles strain-rate-dependent viscosity.

3. **No extra history variables** — Purely algebraic. No modification to the `DuDt`/`DFDt` semi-Lagrangian machinery.

4. **Geodynamics heritage** — Strain-rate-dependent viscosity is standard for geodynamics codes. The infrastructure is battle-tested.

5. **Element size** is available from the mesh for computing $\Delta$.

### Implementation Approach

**Option A: New constitutive model class** (cleanest)

```python
class SmagorinskyFlowModel(ViscousFlowModel):
    """Smagorinsky LES turbulence model.

    Adds subgrid-scale eddy viscosity based on local strain rate
    and grid scale.
    """
    class Parameters:
        viscosity = ...       # molecular viscosity
        Cs = 0.17             # Smagorinsky constant
        delta = None          # grid scale (auto from mesh if None)

    @property
    def flux(self):
        eta_eff = self.viscosity + self._eta_turbulent
        return eta_eff * self.strainrate
```

**Option B: User-level symbolic recipe** (simplest, no code changes)

```python
# Users can do this today with existing infrastructure:
Cs = uw.expression("C_s", 0.17)
delta = uw.expression(r"\Delta", element_size)
edot = solver.strainrate_1d
eta_t = rho * (Cs * delta)**2 * sympy.sqrt(2 * edot.dot(edot))
solver.constitutive_model.Parameters.viscosity = eta_mol + eta_t
```

Option B demonstrates that the framework already supports this physics — Option A just packages it for convenience and discoverability.

---

## Alternative Models Considered

### 1. Spalart-Allmaras (One-Equation RANS)

Solves one transport equation for turbulent viscosity $\tilde{\nu}$:

$$\frac{\partial \tilde{\nu}}{\partial t} + u_j \frac{\partial \tilde{\nu}}{\partial x_j} = \text{production} - \text{destruction} + \text{diffusion}$$

| Aspect | Assessment |
|--------|-----------|
| Extra PDEs | 1 (scalar advection-diffusion) |
| Fits constitutive pattern? | Partially — needs coupled solver |
| Implementation effort | Medium |
| Best for | Wall-bounded flows, boundary layers |

**Pros**: More physically grounded for wall-bounded flows. The transport equation could use the existing `SNES_AdvectionDiffusion` solver.

**Cons**: Wall distance function needed (non-trivial for complex geometries). Adds coupling complexity to the timestepping loop. Several empirical constants.

### 2. k-epsilon (Two-Equation RANS)

Two transport equations for turbulent kinetic energy $k$ and dissipation rate $\varepsilon$:

$$\eta_t = \rho C_\mu \frac{k^2}{\varepsilon}$$

| Aspect | Assessment |
|--------|-----------|
| Extra PDEs | 2 (both advection-diffusion) |
| Fits constitutive pattern? | No — significant solver infrastructure needed |
| Implementation effort | High |
| Best for | General-purpose RANS, free shear flows |

**Pros**: Industry standard, well-validated. Good for mixing (plumes, subduction).

**Cons**: Two coupled transport equations plus wall treatment. Stiff near-wall behaviour. Significant implementation and validation effort.

### 3. Dynamic Smagorinsky

Same as Smagorinsky but $C_s$ is computed dynamically from the resolved field using the Germano identity (test-filter approach).

| Aspect | Assessment |
|--------|-----------|
| Extra PDEs | 0 |
| Fits constitutive pattern? | Yes |
| Implementation effort | Medium-high |
| Best for | Transitional flows, self-calibrating |

**Pros**: No constant tuning. Correctly predicts $\eta_t \to 0$ in laminar regions.

**Cons**: Requires a test filter (spatial averaging at 2x grid scale) — non-trivial on unstructured FEM meshes. Can produce negative viscosities requiring clipping.

### 4. Variational Multiscale (VMS)

The "turbulence model" emerges from the FEM formulation itself — unresolved fine scales modelled as a function of the residual.

| Aspect | Assessment |
|--------|-----------|
| Extra PDEs | 0 |
| Fits constitutive pattern? | No — modifies weak form assembly |
| Implementation effort | High |
| Best for | Mathematically rigorous LES in FEM |

**Pros**: No empirical constants. Mathematically consistent with FEM. Naturally stabilises advection (like SUPG).

**Cons**: Requires changes in `petsc_generic_snes_solvers.pyx` (Cython/PETSc layer). Less intuitive for geophysics users. Limited geodynamics validation.

---

## Comparison Summary

| Model | Extra PDEs | Fits Constitutive Pattern? | Effort | Best For |
|-------|-----------|--------------------------|--------|----------|
| **Smagorinsky** | 0 | Yes (perfectly) | Low | First implementation |
| Spalart-Allmaras | 1 | Partially | Medium | Wall-bounded flows |
| k-epsilon | 2 | No | High | General-purpose RANS |
| Dynamic Smagorinsky | 0 | Yes | Medium-high | Transitional flows |
| VMS | 0 | No (modifies weak form) | High | Rigorous FEM-native LES |

## Suggested Sequencing

1. **Smagorinsky first** — near-trivially implementable given existing nonlinear viscosity infrastructure.
2. **Spalart-Allmaras** if wall-bounded flows become important — leverages existing advection-diffusion solver.
3. **Dynamic Smagorinsky** as natural upgrade from basic Smagorinsky.

---

## Validation Benchmarks

Three benchmarks in order of priority, progressing from canonical fluid dynamics to geodynamics application.

### 1. Lid-Driven Cavity (Re = 1,000 → 10,000)

The standard first test for any NS/turbulence implementation. Compare centreline velocity profiles against Ghia, Ghia & Shin (1982) tabulated reference data. At Re = 1,000 the flow is steady and laminar (turbulence model should be nearly inactive); at Re = 10,000 the flow is unsteady and the eddy viscosity contribution becomes significant.

**What it tests**: Basic correctness — the model activates at high Re without corrupting the solution at moderate Re.

**Reference**: Ghia, U., Ghia, K. N., & Shin, C. T. (1982). High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method. *Journal of Computational Physics*, 48(3), 387–411.

### 2. Taylor-Green Vortex Decay (Free-Slip Box)

Decaying vortex flow with analytical initial condition and known DNS reference data for energy dissipation rate and enstrophy evolution.

**Domain and boundary conditions**: The TGV is solved on the **impermeable box** $[0, \pi]^3$ with **free-slip (stress-free) walls** — no periodic boundary conditions required. Brachet et al. (1983) distinguish two domains:

- **Impermeable box** $[0, \pi]^3$: the physical domain, bounded by stress-free walls. The TGV symmetry group naturally confines the flow here.
- **Periodicity box** $[0, 2\pi]^3$: eight reflected copies of the impermeable box, tiled by the TGV symmetry group. Used by spectral methods (which require periodicity) as a computational convenience.

The DNS reference data from Brachet et al. describes the impermeable box dynamics — the periodicity is an artefact of the spectral method, not the physics. This means a free-slip FEM computation on $[0, \pi]^3$ can be compared directly against their published energy dissipation curves.

**Initial condition** (in the impermeable box):

$$u = \cos(x)\sin(y)\cos(z), \quad v = -\sin(x)\cos(y)\cos(z), \quad w = 0$$

**What it tests**: Whether the eddy viscosity dissipates energy at the correct rate. Key observables:
- Kinetic energy decay $E_k(t)$
- Energy dissipation rate $-dE_k/dt$ (peak location and magnitude)
- Enstrophy $\mathcal{E}(t)$ (related to dissipation via $dE_k/dt = -2\nu\mathcal{E}$)

**Reynolds numbers**: Brachet et al. provide reference data at Re up to 3000. At Re $\geq$ 1000 the small scales become nearly isotropic near peak dissipation; at Re = 3000 the flow exhibits features of fully developed turbulence.

**Caveat**: The Brachet et al. DNS preserves the TGV symmetry group exactly (enforced by the spectral method). An FEM computation on the free-slip box does not enforce this symmetry — numerical perturbations could break it at very high Re or long integration times. For the benchmark Reynolds numbers (Re $\leq$ 3000) this is not expected to be a problem.

**Why free-slip matters for UW3**: Double periodicity in FEM is problematic (null spaces, shared corner constraints). The free-slip formulation avoids these issues entirely — free-slip is the natural ("do nothing") boundary condition in the weak form.

**References**:
- Brachet, M. E., Meiron, D. I., Orszag, S. A., Nickel, B. G., Morf, R. H., & Frisch, U. (1983). Small-scale structure of the Taylor-Green vortex. *Journal of Fluid Mechanics*, 130, 411–452.
- Brachet, M. E. (1991). The Taylor-Green vortex and fully developed turbulence. *Journal of Statistical Physics*, 34, 1049–1063.

### 3. Turbulent Lava Tube/Channel Flow

Geodynamics-relevant application benchmark. Low-viscosity basaltic or komatiitic melts flowing in tubes or channels can reach genuinely turbulent regimes (Re > 10,000 for komatiites). The problem combines turbulent channel flow (well-characterised engineering reference data) with heat transfer and potential solidification at walls.

**What it tests**: Whether the turbulence model produces physically meaningful results for a real geodynamics application. Key observables:
- Mean velocity profile and flow rate vs pressure gradient
- Heat transfer rate to tube walls (controls thermal erosion rate)
- Comparison of turbulent vs laminar predictions for the same conditions

**Why this matters**: Thermal erosion rate by turbulent lava is a significant scientific question — laminar models systematically underpredict erosion for low-viscosity melts. The turbulence model directly affects the predicted erosion rate.

**References**:
- Hulme, G. (1973). Turbulent lava flow and the formation of lunar sinuous rilles. *Modern Geology*, 4, 107–117.
- Williams, D. A., Kerr, R. C., & Lesher, C. M. (1998). Emplacement and erosion by Archean komatiite lava flows at Kambalda: Revisited. *Journal of Geophysical Research*, 103(B11), 27533–27549.
- Kerr, R. C. (2001). Thermal erosion by laminar lava flows. *Journal of Geophysical Research*, 106(B11), 26453–26465.

**Geometry**: 2D channel (Cartesian mesh) or annular cross-section (annulus mesh) — both natural for Underworld3.

---

## References

- Smagorinsky, J. (1963). General circulation experiments with the primitive equations. *Monthly Weather Review*, 91(3), 99–164.
- Spiegelman, M., & Katz, R. F. (2006). A semi-Lagrangian Crank-Nicolson algorithm for the numerical solution of advection-diffusion problems. *Geochemistry, Geophysics, Geosystems*, 7(4).
- Germano, M., Piomelli, U., Moin, P., & Cabot, W. H. (1991). A dynamic subgrid-scale eddy viscosity model. *Physics of Fluids A*, 3(7), 1760–1765.
- Hughes, T. J. R., Feijóo, G. R., Mazzei, L., & Quincy, J.-B. (1998). The variational multiscale method — a paradigm for computational mechanics. *Computer Methods in Applied Mechanics and Engineering*, 166(1–2), 3–24.
