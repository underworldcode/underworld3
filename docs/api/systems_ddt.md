# Time Derivatives

Time derivative operators approximate $D\phi/Dt$ or $DF/Dt$ for transient
solvers.  All operators share a common interface: ``update_pre_solve(dt)``
before each timestep, ``bdf()`` for the BDF approximation in the weak form,
and ``update_post_solve(dt)`` after the solve completes.

History is initialised automatically on the first solve call, and BDF order
ramps from 1 up to the requested ``order`` over the first few timesteps.

For analytical-IC benchmarks (no startup transient) or checkpoint restarts,
``set_initial_history(values, dt=...)`` plants the BDF history directly and
bypasses the order ramp, so the first solve runs at full BDF order.

## Base Class

### Symbolic

```{eval-rst}
.. autoclass:: underworld3.systems.ddt.Symbolic
   :members:
   :show-inheritance:
```

## Eulerian Derivatives

### Eulerian

```{eval-rst}
.. autoclass:: underworld3.systems.ddt.Eulerian
   :members:
   :show-inheritance:
```

## Lagrangian Derivatives

### SemiLagrangian

```{eval-rst}
.. autofunction:: underworld3.systems.ddt.SemiLagrangian
```

### BackwardNodesSemiLagrangian

```{eval-rst}
.. autoclass:: underworld3.systems.ddt.BackwardNodesSemiLagrangian
   :members:
   :show-inheritance:
```

### BackwardIntegrationPointsSemiLagrangian

```{eval-rst}
.. autoclass:: underworld3.systems.ddt.BackwardIntegrationPointsSemiLagrangian
   :members:
   :show-inheritance:
```

### ForwardIntegrationPointsSemiLagrangian

```{eval-rst}
.. autoclass:: underworld3.systems.ddt.ForwardIntegrationPointsSemiLagrangian
   :members:
   :show-inheritance:
```

### ForwardNodesSemiLagrangian

```{eval-rst}
.. autoclass:: underworld3.systems.ddt.ForwardNodesSemiLagrangian
   :members:
   :show-inheritance:
```

### Lagrangian

```{eval-rst}
.. autoclass:: underworld3.systems.ddt.Lagrangian
   :members:
   :show-inheritance:
```

### Lagrangian_Swarm

```{eval-rst}
.. autoclass:: underworld3.systems.ddt.Lagrangian_Swarm
   :members:
   :show-inheritance:
```

## Convenience Aliases

The following aliases are available via ``underworld3.systems``:

- ``Lagrangian_DDt`` → {class}`~underworld3.systems.ddt.Lagrangian`
- ``SemiLagragian_DDt`` → {class}`~underworld3.systems.ddt.BackwardNodesSemiLagrangian`
- ``Lagrangian_Swarm_DDt`` → {class}`~underworld3.systems.ddt.Lagrangian_Swarm`
- ``Eulerian_DDt`` → {class}`~underworld3.systems.ddt.Eulerian`
