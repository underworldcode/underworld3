# Utilities

```{eval-rst}
.. automodule:: underworld3.utilities
   :no-members:
```

## I/O Utilities

### XDMF Generation

```{eval-rst}
.. autofunction:: underworld3.utilities.generateXdmf
```

```{eval-rst}
.. autofunction:: underworld3.utilities.generate_uw_Xdmf
```

### Swarm I/O

```{eval-rst}
.. autofunction:: underworld3.utilities.swarm_h5
```

```{eval-rst}
.. autofunction:: underworld3.utilities.swarm_xdmf
```

## Mesh Import

```{eval-rst}
.. autofunction:: underworld3.utilities.read_medit_ascii
```

```{eval-rst}
.. autofunction:: underworld3.utilities.create_dmplex_from_medit
```

## Development Utilities

```{eval-rst}
.. autoclass:: underworld3.utilities.CaptureStdout
   :members:
```

```{eval-rst}
.. autofunction:: underworld3.utilities.h5_scan
```

```{eval-rst}
.. autofunction:: underworld3.utilities.mem_footprint
```

## Geometry Utilities

Functions for geometric computations useful for mesh and swarm operations.

### Distance Calculations

```{eval-rst}
.. autofunction:: underworld3.utilities.distance_pointcloud_linesegment
```

```{eval-rst}
.. autofunction:: underworld3.utilities.distance_pointcloud_polyline
```

```{eval-rst}
.. autofunction:: underworld3.utilities.distance_pointcloud_triangle
```

### Signed Distance Functions

```{eval-rst}
.. autofunction:: underworld3.utilities.signed_distance_pointcloud_linesegment_2d
```

```{eval-rst}
.. autofunction:: underworld3.utilities.signed_distance_pointcloud_polyline_2d
```

### Point-in-Simplex Tests

```{eval-rst}
.. autofunction:: underworld3.utilities.points_in_simplex2D
```

```{eval-rst}
.. autofunction:: underworld3.utilities.points_in_simplex3D
```

### Normals

```{eval-rst}
.. autofunction:: underworld3.utilities.linesegment_normals_2d
```

## Array Utilities

### UnitAwareArray

Array with attached unit information for dimensional quantities.

```{eval-rst}
.. autoclass:: underworld3.utilities.UnitAwareArray
   :members:
   :show-inheritance:
```

### NDArray_With_Callback

Array wrapper that triggers callbacks on modification.

```{eval-rst}
.. autoclass:: underworld3.utilities.NDArray_With_Callback
   :members:
   :show-inheritance:
```
