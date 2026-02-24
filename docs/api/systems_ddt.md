# Time Derivatives

```{eval-rst}
.. automodule:: underworld3.systems.ddt
   :no-members:
```

Time derivative operators approximate $D\phi/Dt$ or $DF/Dt$ for transient
solvers.  All operators share a common interface: ``update_pre_solve(dt)``
before each timestep, ``bdf()`` for the BDF approximation in the weak form,
and ``update_post_solve(dt)`` after the solve completes.

History is initialised automatically on the first solve call, and BDF order
ramps from 1 up to the requested ``order`` over the first few timesteps.

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
.. autoclass:: underworld3.systems.ddt.SemiLagrangian
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
- ``SemiLagragian_DDt`` → {class}`~underworld3.systems.ddt.SemiLagrangian`
- ``Lagrangian_Swarm_DDt`` → {class}`~underworld3.systems.ddt.Lagrangian_Swarm`
- ``Eulerian_DDt`` → {class}`~underworld3.systems.ddt.Eulerian`
