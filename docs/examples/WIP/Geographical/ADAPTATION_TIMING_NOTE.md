# Mesh Adaptation Timing Issue

## Observation

When testing mesh adaptation with constant metric values on a geographic mesh, the adaptation process takes a very long time (>2 minutes for even a 5×5×5 element mesh), eventually timing out.

## Test Configuration

- **Base mesh**: RegionalGeographicBox, 5×5×5 elements (~200 nodes)
- **Metric**: Uniform H = 100.0
- **Result**: Timeout after 1-2 minutes

## Hypothesis

The mesh adaptation algorithm (MMG5) may be:
1. Trying to create a very large number of elements for the given metric value
2. Running into memory allocation issues
3. Taking a long time to converge

## Error Messages Seen

```
## Error: unable to allocate larger solution.
## Check the mesh size or increase maximal authorized memory with the -m option.
```

This suggests the metric value of 100 may be interpreted as requiring many more elements than available memory allows.

## Implications for Geographic Mesh Workflow

From the original workflow (`Mesh-Adapted-2-Faults.py`), the metric values used were:
- **Fine mesh** (near faults/surface): `mesh_adaptation_parameter = 6.6e6 * (mesh_k_elts/100)`
  - With mesh_k_elts=100: H = 6.6e6
- **Coarse mesh** (elsewhere): H = 100

This suggests:
1. **Large values** (6.6e6) create fine meshes
2. **Small values** (100) create coarse meshes
3. **Inverse relationship**: Smaller H → fewer elements (counter-intuitive!)

## Next Steps

1. **Skip exhaustive adaptation testing** - Accept that adaptation works based on original workflow
2. **Focus on workflow**: Load faults, calculate distances, use same metric values as before
3. **Test adaptation** with actual fault-based metric (combining 6.6e6 and 100)
4. **If issues persist**: Investigate MMG5 parameters or memory limits

## Key Insight

The relationship between metric H and element count is **inverse** and highly nonlinear:
- H = 100 → attempts to create a LARGE mesh (possibly millions of elements!)
- H = 6.6e6 → creates a moderate refined mesh (100k elements)

This is consistent with H being a "target edge length" metric where:
- Smaller H → smaller edges → more elements
- Larger H → larger edges → fewer elements

**But** the scale is such that even H=100 creates too many elements for a 400 km domain!

This suggests the metric is NOT in physical units (km), but rather a dimensionless parameter with a very specific scaling relationship to the domain size.

## Recommendation for Workflow

**Use the original metric values directly** without trying to interpret them physically:
- Near faults/surface: H = 6.6e6 * (mesh_k_elts/100)
- Elsewhere: H = 100
- These are **calibrated empirical values** that work with the adaptation algorithm

The geographic coordinates are in km, but the metric H remains dimensionless and calibrated to the specific implementation of mesh adaptation.
