# Solver Unification Design

> Status: **Proposed** — for implementation after VEP validation is complete

## Goal

Eliminate the need for separate `VE_Stokes` and (future) `VE_NavierStokes` solver
classes. The constitutive model declares what infrastructure it needs; the solver
creates it lazily.

## Current Architecture

| Solver | DuDt (velocity) | DFDt (flux/stress) | Constitutive models |
|--------|-----------------|-------------------|-------------------|
| `Stokes` | — | — | Viscous, VP |
| `VE_Stokes` | — | SemiLagrangian (stress history) | VEP |
| `NavierStokes` | SemiLagrangian (velocity) | SemiLagrangian (AM flux) | Viscous, VP |
| `VE_NavierStokes` | does not exist | — | — |

Problem: user must choose the correct solver class based on the constitutive model.
Using VEP on plain Stokes silently drops stress history (now caught by barrier in PR #95).

## Proposed Architecture

Two solver classes: `Stokes` and `NavierStokes`. Each detects whether the
constitutive model requires stress history and creates DFDt infrastructure lazily.
No separate VE variants needed.

### Constitutive model contract

```python
class Constitutive_Model:
    @property
    def requires_stress_history(self):
        return False  # Viscous, VP

class ViscoElasticPlasticFlowModel(ViscousFlowModel):
    @property
    def requires_stress_history(self):
        return True  # VEP
```

### Solver behaviour

```python
@constitutive_model.setter
def constitutive_model(self, model):
    # ... existing setup ...
    if model.requires_stress_history and self.Unknowns.DFDt is None:
        self._create_stress_history_ddt(order=model.order)
```

### Constitutive model is assigned once

Changing parameters (viscosity, yield stress, modulus) is fine — they flow through
UWexpressions and PetscDS constants[]. Swapping the constitutive model class after
the first solve is not supported (DFDt allocation, JIT structure changes).

### VE_Stokes becomes a backward-compat alias

```python
class VE_Stokes(Stokes):
    """Deprecated: use Stokes directly with VEP constitutive model."""
    def __init__(self, mesh, order=2, **kwargs):
        super().__init__(mesh, **kwargs)
        self._create_stress_history_ddt(order=order)
```

### solve() hooks

```python
def solve(self, timestep=None, ...):
    if self.Unknowns.DFDt is not None:
        if timestep is None:
            raise ValueError("timestep required for viscoelastic solve")
        self.constitutive_model._update_bdf_coefficients()
        self.DFDt.update_pre_solve(timestep, store_result=False)

    # PETSc solve
    self._snes_solve(...)

    if self.Unknowns.DFDt is not None:
        self._post_solve_stress_history(timestep)
```

### tau property

```python
@property
def tau(self):
    if self.Unknowns.DFDt is not None:
        return self.DFDt.psi_star[0]  # stored actual stress
    else:
        return self._lazy_tau_projection()  # on-demand projection
```

## NavierStokes Considerations

NS already uses DFDt for Adams-Moulton (Crank-Nicolson) stabilisation of the
viscous flux. The AM scheme stores `η·∇u` at previous timesteps for flux averaging:

$$F^{n+1/2} = \theta \cdot F^{n+1} + (1-\theta) \cdot F^{n*}$$

For VE-NS, the DFDt must serve both purposes:
- AM flux averaging for time integration stability
- VE stress history for the Maxwell constitutive law

### Possible unification

If DFDt stores the **actual deviatoric stress** (as PR #89 implements for VE_Stokes),
the AM scheme can be reformulated to read from it:

$$F^{n+1/2} = \theta \cdot \sigma^{n+1} + (1-\theta) \cdot \sigma^{n*}$$

where σ^{n*} is the advected stress from `psi_star[0]`. This unifies the two uses:
the DFDt stores actual stress, and both AM stabilisation and VE constitutive law
read from the same history chain.

### Open questions for VE-NS

- Is order-1 AM sufficient alongside VE stress history? The VE history already
  provides temporal accuracy for the elastic part; AM only needs to stabilise the
  viscous/advection part.
- Can the AM flux and VE stress contributions simply be added symbolically in
  the F0/F1 expressions? SymPy handles the algebra; the JIT compiles the combined
  expression. This might avoid needing a separate DFDt entirely — the NS solver's
  existing DFDt carries the AM flux, and the VE stress history is a separate DFDt
  created by the constitutive model.
- This needs to be driven by physics requirements (a problem that demands VE-NS),
  not implemented speculatively.

### Recommendation

Implement VE-NS only after:
1. VEP on Stokes is fully validated (current priority)
2. Solver unification for Stokes is complete and tested
3. NS benchmarks are passing as a baseline
4. A physics problem demands VE-NS

## Implementation Order

1. ~~PR #95: Barrier + is_viscoplastic fix~~ (done)
2. Move DFDt creation from VE_Stokes.__init__ to Stokes.constitutive_model setter
3. Move pre/post solve hooks from VE_Stokes.solve() to Stokes.solve()
4. VE_Stokes becomes backward-compat alias
5. Same pattern for NavierStokes (when physics demands it)
