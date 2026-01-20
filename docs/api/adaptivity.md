# Adaptive Mesh Refinement

```{eval-rst}
.. automodule:: underworld3.adaptivity
   :no-members:
```

## Mesh Adaptation

Tools for adaptive mesh refinement (AMR) in Underworld3.

**Note**: AMR features require the `amr` environment with custom PETSc build.

### mesh_adapt_meshVar

Perform mesh adaptation based on a mesh variable field.

```{eval-rst}
.. autofunction:: underworld3.adaptivity.mesh_adapt_meshVar
```

### mesh2mesh_meshVariable

Transfer mesh variable data between meshes.

```{eval-rst}
.. autofunction:: underworld3.adaptivity.mesh2mesh_meshVariable
```

### mesh2mesh_swarm

Transfer swarm data between meshes.

```{eval-rst}
.. autofunction:: underworld3.adaptivity.mesh2mesh_swarm
```

## Metric Functions

Functions for computing adaptation metrics that guide mesh refinement.

### create_metric

Create a metric field for mesh adaptation.

```{eval-rst}
.. autofunction:: underworld3.adaptivity.create_metric
```

### metric_from_field

Compute an adaptation metric from a scalar field.

```{eval-rst}
.. autofunction:: underworld3.adaptivity.metric_from_field
```

### metric_from_gradient

Compute an adaptation metric from field gradients.

```{eval-rst}
.. autofunction:: underworld3.adaptivity.metric_from_gradient
```
