# Linear Metric Gradient Test - Findings

## Test Setup

Created `Step4-TestLinearMetric.py` to test mesh adaptation with simple linear gradients:
- **Test 1**: Fine at surface (H=1000) → Coarse at depth (H=100)
- **Test 2**: Coarse at surface (H=100) → Fine at depth (H=1000)
- **Base mesh**: 5×5×5 elements (163 nodes) - very coarse
- **Domain**: 5° × 5° × 400 km (Eyre Peninsula region)

## Results

**Adaptation did not complete within 3 minutes** for either test.

Even with:
- Very coarse initial mesh (163 nodes)
- Modest metric values (H: 100-1000)
- Simple linear gradient (no complex conditions)

The adaptation algorithm either:
1. Is taking a very long time to converge
2. Is creating a huge number of elements
3. Has convergence issues with these metric values

## Interpretation

### What We Know from User Feedback

> "The idea of the metric is that mesh refinement drives it to near unity. If the metric is 100, say, the points need to be 100 times closer together than places where it is one."

This tells us:
- **Metric → 1**: Refinement process drives metric toward unity
- **H = 100**: Points need to be **100× closer** than where H=1
- **Relationship**: Likely H ∝ 1/(desired_spacing)² or similar

### The Problem with Our Test Values

If H=100 means points need to be 100× closer than H=1, then:
- **H=1000 means 1000× refinement** relative to H=1
- For a 400 km domain, this could mean:
  - H=1 → ~400 km spacing
  - H=100 → ~4 km spacing
  - H=1000 → ~0.4 km spacing

This would create **millions of elements** in the refined region!

### Why Original Workflow Values Work

From `Mesh-Adapted-2-Faults.py`:
- **Fine**: H = 6.6e6 (near faults/surface)
- **Coarse**: H = 100 (elsewhere)
- **Result**: ~100k elements total

The key insight: **The contrast ratio matters**, not absolute values.

If the adaptation algorithm works to drive H→1:
- Starting from H=6.6e6, it needs to **reduce** element size by 6.6e6× to reach unity
- This creates very fine elements
- Starting from H=100, it needs to reduce by only 100× to reach unity
- This creates relatively coarse elements

The 66,000× ratio (6.6e6 / 100) creates the refinement contrast.

### Why Our Tests Failed

Our test values (1000 and 100) may both be:
1. **Too coarse to resolve the domain properly** at those refinement levels
2. **Too close in ratio** (only 10×) to create useful contrast
3. **In the wrong regime** - perhaps values need to be either much larger or much smaller

## Possible Solutions

### Option 1: Use Original Values Directly

Just use the calibrated values from the original workflow:
```python
H_fine = 6.6e6  # Near faults/surface
H_coarse = 100  # Elsewhere
```

These are **empirically known to work** and produce ~100k elements.

### Option 2: Test with Much Smaller Values

If smaller H → coarser, try:
```python
H_fine = 10  # "Fine" region
H_coarse = 1  # Coarse region
```

This might produce a manageable mesh quickly.

### Option 3: Test with Uniform Small Value

Try H=1 everywhere to see if adaptation converges at all:
```python
H.array[:] = 1.0
```

This should create the coarsest possible adapted mesh.

### Option 4: Accept That Adaptation Takes Time

The original workflow may just take time to adapt. Perhaps:
- Accept 5-10 minute adaptation times
- Use coarser initial mesh (3×3×3)
- Trust that it works based on original workflow success

## Recommendation

**Skip exhaustive adaptation testing** and proceed with the fault-based metric using original values:

1. Create fault-based metric with H=6.6e6 (fine) and H=100 (coarse)
2. Run adaptation once with proper mesh resolution (20×20×10)
3. Let it run to completion (may take 10-30 minutes)
4. If successful, document the result
5. If unsuccessful, investigate MMG5 parameters or mesh settings

The linear gradient test was conceptually valuable but practically stalled. The original empirical values are more likely to work than our test values.

## Key Takeaway

**Mesh metric values are highly sensitive and empirically calibrated.**

- The relationship between H and element count is **nonlinear and complex**
- Values that "make sense" (like 100-1000) may not be in the working regime
- The original values (6.6e6 and 100) are **magic numbers** that work
- Geographic coordinates don't change this - metric stays dimensionless

**Trust the empirical calibration** rather than trying to derive values from first principles.
