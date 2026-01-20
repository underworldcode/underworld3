# Meshing

```{eval-rst}
.. automodule:: underworld3.meshing
   :no-members:
```

## Cartesian Meshes

### StructuredQuadBox

```{eval-rst}
.. autofunction:: underworld3.meshing.StructuredQuadBox
```

### UnstructuredSimplexBox

```{eval-rst}
.. autofunction:: underworld3.meshing.UnstructuredSimplexBox
```

### BoxInternalBoundary

```{eval-rst}
.. autofunction:: underworld3.meshing.BoxInternalBoundary
```

## Annulus Meshes

### Annulus

```{eval-rst}
.. autofunction:: underworld3.meshing.Annulus
```

### QuarterAnnulus

```{eval-rst}
.. autofunction:: underworld3.meshing.QuarterAnnulus
```

### SegmentofAnnulus

```{eval-rst}
.. autofunction:: underworld3.meshing.SegmentofAnnulus
```

### AnnulusWithSpokes

```{eval-rst}
.. autofunction:: underworld3.meshing.AnnulusWithSpokes
```

### AnnulusInternalBoundary

```{eval-rst}
.. autofunction:: underworld3.meshing.AnnulusInternalBoundary
```

### DiscInternalBoundaries

```{eval-rst}
.. autofunction:: underworld3.meshing.DiscInternalBoundaries
```

## Spherical Meshes

### SphericalShell

```{eval-rst}
.. autofunction:: underworld3.meshing.SphericalShell
```

### CubedSphere

```{eval-rst}
.. autofunction:: underworld3.meshing.CubedSphere
```

### SegmentofSphere

```{eval-rst}
.. autofunction:: underworld3.meshing.SegmentofSphere
```

### SphericalShellInternalBoundary

```{eval-rst}
.. autofunction:: underworld3.meshing.SphericalShellInternalBoundary
```

## Geographic Meshes

Regional meshes defined using longitude/latitude coordinates.

### RegionalSphericalBox

```{eval-rst}
.. autofunction:: underworld3.meshing.RegionalSphericalBox
```

### RegionalGeographicBox

```{eval-rst}
.. autofunction:: underworld3.meshing.RegionalGeographicBox
```

## Segmented Meshes

Meshes with internal segment boundaries for multi-region problems.

### SegmentedSphericalShell

```{eval-rst}
.. autofunction:: underworld3.meshing.SegmentedSphericalShell
```

### SegmentedSphericalSurface2D

```{eval-rst}
.. autofunction:: underworld3.meshing.SegmentedSphericalSurface2D
```

### SegmentedSphericalBall

```{eval-rst}
.. autofunction:: underworld3.meshing.SegmentedSphericalBall
```

## Faults and Internal Boundaries

Tools for creating meshes with fault surfaces and internal boundaries.

### FaultSurface

```{eval-rst}
.. autoclass:: underworld3.meshing.FaultSurface
   :members:
   :show-inheritance:
```

### FaultCollection

```{eval-rst}
.. autoclass:: underworld3.meshing.FaultCollection
   :members:
   :show-inheritance:
```

## Surfaces

Tools for working with surface meshes and embedded boundaries.

### Surface

```{eval-rst}
.. autoclass:: underworld3.meshing.Surface
   :members:
   :show-inheritance:
```

### SurfaceCollection

```{eval-rst}
.. autoclass:: underworld3.meshing.SurfaceCollection
   :members:
   :show-inheritance:
```

### SurfaceVariable

Variables defined on surface meshes.

```{eval-rst}
.. autoclass:: underworld3.meshing.SurfaceVariable
   :members:
   :show-inheritance:
```
