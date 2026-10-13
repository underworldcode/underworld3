# Materials

Underworld3 keeps two questions apart: **what** a material is, and **where** it
is.

*What* is a {py:class}`~underworld3.MaterialRegistry` entry — a name and a
table of properties, where a property may be a number, a quantity, or a *law*
such as `eta_0 * sympy.exp(-T.sym[0])`. A registry knows nothing about
geometry, so it can be built before there is a mesh and shared between models.

*Where* is a **distribution**, and there are two:

- {py:class}`~underworld3.swarm.MaterialSwarm` — carried by particles, so the
  materials advect with the flow.
- {py:class}`~underworld3.MaterialRegions` — tied to the mesh, from gmsh
  physical groups or from a geometric condition. Exact, and needs no particles.

Both present the same face to a solver — `stokes.materials = materials`, which
sets every constitutive-model parameter the materials declare and the model
recognises — and both build it from the same partition of unity, one level set
per material.

See {doc}`../advanced/particle-population-and-materials` for the user guide.

```{eval-rst}
.. automodule:: underworld3.materials
   :no-members:
```

## Defining materials

```{eval-rst}
.. autoclass:: underworld3.MaterialRegistry
   :members:
   :show-inheritance:

.. autoclass:: underworld3.MaterialDefinition
   :members:
   :show-inheritance:

.. autoclass:: underworld3.MaterialProperty
   :members:
   :show-inheritance:
```

## Distributing materials

```{eval-rst}
.. autoclass:: underworld3.swarm.MaterialSwarm
   :members:
   :show-inheritance:

.. autoclass:: underworld3.MaterialRegions
   :members:
   :show-inheritance:

.. autoclass:: underworld3.materials.MaterialDistribution
   :members:
   :show-inheritance:

.. autoclass:: underworld3.materials.BoundMaterial
   :members:
```

## Different laws per material

A distribution blends per-material *parameter values* into one constitutive
model. When the materials need genuinely different constitutive **laws** —
one viscous, one viscoelastic — compose the models instead:

```{eval-rst}
.. autoclass:: underworld3.MultiMaterialConstitutiveModel
   :members:
   :show-inheritance:
```
