# Migration Status from underworld3-documentation

## Summary
The `underworld3-documentation` repository contains legacy documentation and examples that are being progressively migrated or superseded.

## Already Migrated ✅

### Theory Documentation
- **ConstitutiveModels_Introduction.qmd** → `underworld3/docs/developer/subsystems/constitutive-models-theory.qmd`
- **ConstitutiveModels_Anisotropy.qmd** → `underworld3/docs/developer/subsystems/constitutive-models-anisotropy.qmd`

## Still Valuable - Should Migrate 🔄

### Solver Documentation (Technical)
Located in `Documentation/`:
- **Solvers_Equation_Systems.md** - PDE formulation and PIC approach
- **Solvers_SNES_interface.md** - PETSc pointwise functions and weak forms
- **Numerical_Methods.md** - Core numerical methodology
- **UnderworldDocumentation.bib** - Comprehensive bibliography

### Example Notebooks
Located in `Notebooks/Examples-*/`:

#### High Priority Examples
- **Examples-Utilities/**
  - `Ex_ConstitutiveTensors.py` - Demonstrates tensor mechanics
  - `Ex_Anisotropy.py` - Anisotropic material examples
  - `Ex_scaling.py` - Non-dimensionalization examples
  - Benchmark cases (Kramer, Thieulot)

#### Physics Examples
- **Examples-Convection/** - Thermal convection models
- **Examples-NavierStokes/** - Navier-Stokes implementations
- **Examples-FreeSurface/** - Free surface boundary conditions
- **Examples-Sandbox-VEP/** - Visco-elasto-plastic sandbox models
- **Examples-PorousFlow/** - Porous media flow

#### Utility Examples
- **Examples-Meshing/** - Advanced meshing techniques
- **Examples-PostProcessing/** - Visualization and analysis
- **Examples-SwarmAndParticles/** - Particle management

## Can Be Deprecated ❌

### Superseded Content
- **Jupyterbook/** - Old book structure (replaced by Quarto)
- **Developers/** - Mostly empty or superseded
- **Examples-UW-2to3/** - Migration guides from UW2 (less relevant now)
- **WIP/** - Work in progress, likely outdated

## Recommended Actions

### Immediate Actions
1. **Preserve solver documentation** - Move to `underworld3/docs/developer/subsystems/solvers-theory.qmd`
2. **Convert key examples** - Priority on constitutive tensors, anisotropy, scaling
3. **Extract bibliography** - Merge references into main documentation

### Medium Term
1. **Convert physics examples** to percent format notebooks in `underworld3/docs/user/Notebooks/`
2. **Create example index** with clear categorization
3. **Benchmark suite** - Establish formal benchmark directory

### Long Term
1. **Full deprecation** of underworld3-documentation repository
2. **Single source of truth** in underworld3 repository
3. **Automated example testing** as part of CI/CD

## Directory Size Analysis
```bash
# Approximate sizes
Documentation/    ~1MB  (mostly text/markdown)
Notebooks/        ~50MB (includes output cells, images)
Jupyterbook/      ~10MB (can be removed)
```

## Migration Command Examples
```bash
# For converting .py examples to percent format
jupytext --to py:percent Examples-Utilities/Ex_ConstitutiveTensors.py

# For moving to underworld3
cp Notebooks/Examples-Utilities/Ex_*.py ../underworld3/docs/user/Notebooks/Examples/Utilities/
```

## Notes
- Many notebooks have valuable physics implementations not yet in main docs
- Benchmark cases should be preserved for validation
- Consider creating a "legacy examples" archive before deletion