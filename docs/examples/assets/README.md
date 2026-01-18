# Example Assets Directory

This directory contains **curated images and media** for documentation purposes that should be version controlled.

## Purpose

Unlike the `output/` directory which excludes everything, `assets/` contains:
- **Documentation images** - Figures that illustrate concepts in example notebooks
- **Reference results** - Expected output images for validation
- **Diagrams** - Conceptual illustrations, flowcharts, schemas
- **Icons** - Visual elements for documentation

## Organization

### By Physics Domain
```
assets/
├── heat_transfer/
│   ├── temperature_distribution_concept.png
│   ├── diffusion_equation_diagram.svg
│   └── expected_results/
├── fluid_mechanics/
│   ├── stokes_flow_streamlines.png
│   ├── boundary_conditions_diagram.svg
│   └── benchmarks/
├── convection/
│   ├── rayleigh_benard_cells.png
│   └── convection_regimes_chart.png
└── ...
```

### File Types

**Preferred formats:**
- `*.png` - Screenshots, plots (compress with `optipng`)
- `*.svg` - Vector graphics, diagrams (scalable)
- `*.jpg` - Photos, complex images (compressed)

**Avoid:**
- Large uncompressed images
- Temporary visualization outputs
- Files that can be regenerated from examples

## Guidelines

### Adding New Assets

1. **Optimize file size**:
   ```bash
   # PNG optimization
   optipng -o7 image.png
   
   # SVG optimization  
   svgo image.svg
   ```

2. **Use descriptive names**:
   ```
   ✅ heat_conduction_boundary_conditions.png
   ✅ stokes_flow_around_cylinder_expected.png
   ❌ figure1.png
   ❌ output_plot.png
   ```

3. **Document purpose**:
   - Add comment in notebook explaining the asset
   - Include in example README if it's a key reference

### Reference vs Generated

**Reference images** (store in assets/):
- Concept diagrams drawn by developers
- Expected benchmark results
- Illustrations of physics concepts
- Validation targets

**Generated images** (exclude via output/):
- Runtime visualization from examples
- User-specific parameter variations
- Temporary debugging plots
- Large datasets visualizations

## Integration with Examples

Reference assets in notebooks using relative paths:

```python
# Display reference image
from IPython.display import Image
Image("../assets/heat_transfer/expected_steady_state.png")
```

```markdown
# Concept Illustration
![Boundary Conditions](../assets/fluid_mechanics/boundary_conditions.svg)
```

## Review Process

Before committing assets:
1. ✅ Is this needed for documentation?
2. ✅ Is the file size reasonable (<1MB)?
3. ✅ Is it optimized (compressed)?
4. ✅ Is the filename descriptive?
5. ✅ Is it referenced from example documentation?