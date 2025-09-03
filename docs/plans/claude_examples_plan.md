# Claude-Parseable Examples for Underworld3

**Objective**: Create a structured library of examples that Claude can parse, understand, and help users adapt for their specific geophysical modeling needs.

## Project Overview

This plan outlines creating a comprehensive example library that enables Claude to:
- Understand Underworld3 code patterns and physics
- Help users adapt examples for their specific problems
- Suggest parameter modifications and physics extensions
- Guide users through learning progressions

## Directory Structure

```
underworld3/examples/
├── metadata.yaml           # Claude-parseable metadata for all examples
├── templates/             # Base templates (leverage existing solver_template.py)
│   ├── solver_template.py  # Move/link existing template here
│   ├── basic_poisson.py    # Minimal working examples
│   └── basic_stokes.py
├── basic/                 # Simple, focused examples
│   ├── diffusion/
│   ├── flow/
│   └── meshing/
├── intermediate/          # Multi-physics examples
│   ├── thermal_convection/
│   ├── porous_flow/
│   └── free_surface/
├── advanced/              # Complex research-level examples
│   ├── mantle_convection/
│   ├── crustal_deformation/
│   └── multiphase_flow/
└── tools/                 # Automation and analysis tools
    ├── metadata_generator.py
    ├── example_validator.py
    └── claude_integration.py
```

## Phase 1: Foundation (Weeks 1-2)

### 1.1 Set Up Example Structure
- [ ] Create `underworld3/examples/` directory
- [ ] Move `solver_template.py` to `templates/` directory
- [ ] Create initial directory structure
- [ ] Set up basic tooling framework

### 1.0 Existing Examples Integration Strategy

Before creating new examples, we need to systematically process the extensive existing example collection in `underworld3-documentation/Notebooks/`. This collection contains valuable, working examples across multiple physics domains that should be the foundation of our Claude-parseable library.

#### 1.0.1 Existing Example Inventory

**Current structure analysis:**
```
underworld3-documentation/Notebooks/
├── Examples-Annulus-Stokes/        # Cylindrical geometry Stokes problems
├── Examples-Convection/            # Thermal convection examples  
├── Examples-FreeSurface/           # Surface deformation problems
├── Examples-Loop-Mesh/             # Advanced meshing techniques
├── Examples-Meshing/               # Basic mesh generation examples
├── Examples-NavierStokes/          # Non-linear fluid dynamics
├── Examples-PoissonEquation/       # Heat/diffusion equation examples
├── Examples-PorousFlow/            # Groundwater/magma flow
├── Examples-PostProcessing/        # Visualization and analysis
├── Examples-Sandbox-VEP/          # Visco-elasto-plastic materials
├── Examples-Spherical-Stokes/     # Planetary-scale flow problems
├── Examples-Stokes_Kernels/       # Advanced Stokes formulations
├── Examples-StokesFlow/            # Basic Stokes flow examples
├── Examples-SwarmAndParticles/     # Lagrangian particle methods
├── Examples-Utilities/             # Helper functions and tools
└── Examples-UW-2to3/              # Migration examples from UW2
```

**File format analysis:**
- Mix of `.py` files (often jupytext format) and `.ipynb` notebooks
- **Target format**: Convert all to python percent format (`.py` files with `# %%` cells)
- Many examples are research-quality with complex physics
- Some have visualization/post-processing scripts
- Variable documentation quality and consistency

#### 1.0.2 Migration and Enhancement Strategy

**Step 1: Automated Analysis and Cataloging**
```python
# underworld3/examples/tools/legacy_analyzer.py
class LegacyExampleAnalyzer:
    def scan_documentation_examples(self, docs_path="../underworld3-documentation/Notebooks"):
        """Scan all existing examples and extract metadata"""
        
    def categorize_by_physics(self, example_files):
        """Group examples by physics domain (Poisson, Stokes, etc.)"""
        
    def assess_claude_readiness(self, example_file):
        """Rate how close each example is to Claude-parseable format"""
        return {
            'documentation_quality': score,  # 1-5 based on comments/docstrings
            'code_clarity': score,           # 1-5 based on structure
            'parameter_adaptability': score, # 1-5 based on hardcoded vs configurable
            'educational_value': score       # 1-5 based on learning potential
        }
        
    def identify_conversion_priority(self):
        """Rank examples for conversion based on educational value + code quality"""
```

**Step 2: Three-Tier Conversion Strategy**

**Tier 1: Direct Enhancement (High Priority - 15-20 examples)**
- Examples already well-structured and documented
- Require minimal changes to meet Claude standards
- Focus on adding metadata headers and parameter annotations

*Candidates from existing examples:*
- `Examples-PoissonEquation/Ex_Poisson_Cartesian.py` - Basic heat diffusion
- `Examples-StokesFlow/Ex_Stokes_Sinker.py` - Falling sphere benchmark
- `Examples-Convection/Ex_Thermal_Convection_2D.py` - Rayleigh-Benard convection
- `Examples-Meshing/Ex_Mesh_Types_Comparison.py` - Different mesh strategies

**Enhancement process:**
1. Convert to python percent format with `# %%` cell separators
2. Add comprehensive markdown cells with metadata format
3. Insert `# SECTION:` and `# PARAM:` annotations
4. Move parameters to constants at top of notebook
5. Add Claude adaptation hints in markdown cells
6. Ensure consistent variable naming
7. Add basic error handling

**Tier 2: Significant Refactoring (Medium Priority - 20-30 examples)**
- Good physics but needs code restructuring
- May combine multiple notebooks into single coherent example
- Focus on clarity and educational progression

*Example conversion:*
```python
# BEFORE: Examples-Convection/complex_convection_notebook.py
# - Mixed physics setup
# - Hardcoded parameters throughout
# - Minimal documentation
# - Complex visualization code mixed with physics

# AFTER: intermediate/thermal_convection_rayleigh_benard.py  
"""
TITLE: Rayleigh-Benard Thermal Convection
CATEGORY: convection
DIFFICULTY: intermediate
PHYSICS: buoyancy-driven flow with heat transfer
...
"""

# Clear parameter section at top
RAYLEIGH_NUMBER = 1e4    # PARAM: ra - controls convection vigor
PRANDTL_NUMBER = 1.0     # PARAM: pr - thermal vs momentum diffusion
ASPECT_RATIO = 2.0       # PARAM: aspect - domain width/height

# Separated physics setup, solve, and post-processing sections
```

**Tier 3: Complete Redesign (Lower Priority - Research Examples)**
- Complex research codes that need significant simplification
- May extract key concepts into multiple simpler examples
- Create educational versions alongside original research code

#### 1.0.3 Conversion Workflow

**Automated Pre-processing:**
```python
# underworld3/examples/tools/example_converter.py
class ExampleConverter:
    def extract_physics_core(self, legacy_file):
        """Remove visualization, extract core physics setup"""
        
    def identify_adaptable_parameters(self, code_ast):
        """Find hardcoded values that should be parameters"""
        
    def generate_metadata_template(self, analysis_results):
        """Create YAML metadata entry for manual completion"""
        
    def suggest_claude_adaptations(self, code_structure):
        """Recommend parameter variations and extensions"""
```

**Manual Enhancement Process:**
1. **Format Conversion**: Convert to python percent format with proper cell structure
2. **Physics Review**: Ensure educational clarity and correctness
3. **Documentation**: Add comprehensive markdown cells with educational content  
4. **Parameter Extraction**: Move hardcoded values to constants at notebook top
5. **Section Organization**: Structure code into logical notebook cells
6. **Claude Integration**: Add adaptation hints and learning objectives in markdown
7. **Testing**: Verify example runs successfully in both .py and notebook formats

#### 1.0.4 Specific Integration Plan

**Week 1: Setup and Analysis**
- [ ] Run automated analysis on all existing examples
- [ ] Create priority ranking based on quality and educational value
- [ ] Set up conversion pipeline and tools
- [ ] Create template for enhanced example format

**Week 2: High-Priority Conversions**
- [ ] Convert 5 Tier 1 examples (direct enhancement)
- [ ] Create metadata entries for converted examples
- [ ] Test Claude's ability to understand and adapt converted examples
- [ ] Refine conversion process based on results

**Ongoing: Systematic Conversion**
- [ ] Process 3-5 examples per week through conversion pipeline
- [ ] Maintain quality standards and educational coherence
- [ ] Update metadata.yaml with each new conversion
- [ ] Cross-reference with new examples to avoid duplication

#### 1.0.5 Quality Assurance for Converted Examples

**Conversion Standards Checklist:**
- [ ] Comprehensive metadata docstring with all required fields
- [ ] Clear parameter annotations with `# PARAM:` markers
- [ ] Structured code sections with `# SECTION:` markers
- [ ] Educational learning objectives explicitly stated
- [ ] Claude adaptation hints included
- [ ] Error handling for common parameter ranges
- [ ] Consistent import structure and variable naming
- [ ] Execution time documented and reasonable (< 5 min)
- [ ] Basic visualization or output verification included

**Integration Testing:**
```python
# Automated testing for converted examples
def validate_converted_example(example_file):
    """Ensure converted example meets Claude-parseable standards"""
    checks = {
        'has_metadata_docstring': check_docstring_format(),
        'has_parameter_markers': check_param_annotations(),
        'runs_successfully': test_execution(),
        'claude_parseable': test_claude_understanding(),
        'educational_value': assess_learning_objectives()
    }
    return all(checks.values())
```

This integration strategy ensures we build on the substantial existing work while creating a coherent, Claude-optimized example library. The existing examples provide physics accuracy and real-world applicability, while our enhancements add the structure and metadata needed for effective Claude assistance.

### 1.2 Define Metadata Schema
Create `metadata.yaml` with this structure:

```yaml
metadata_version: "1.0"
generated_date: "2024-XX-XX"

# Claude integration configuration
claude_integration:
  parameter_patterns:
    mesh_resolution:
      keywords: ["cellSize", "resolution", "h_", "meshSize"]
      claude_hint: "Controls mesh resolution - smaller = higher accuracy but slower"
      typical_range: [0.01, 0.2]
      
    material_properties:
      keywords: ["diffusivity", "viscosity", "density", "conductivity"]
      claude_hint: "Physical material properties - check literature for realistic values"
      
    boundary_conditions:
      keywords: ["add_essential_bc", "add_natural_bc", "DirichletBC"]
      claude_hint: "Defines physics at domain boundaries"
      
  common_adaptations:
    geometry:
      description: "Modify domain shape, size, or mesh type"
      difficulty: "basic"
      examples: ["Change minCoords/maxCoords", "Try StructuredQuadBox vs UnstructuredSimplexBox"]
      
    physics_extensions:
      description: "Add coupled physics or time dependence"
      difficulty: "advanced"
      examples: ["Add time stepping", "Couple multiple equations", "Nonlinear materials"]

# Example library
examples:
  # Format for each example
  example_name.py:
    title: "Human-readable title"
    category: "poisson|stokes|convection|meshing|utilities"
    difficulty: "basic|intermediate|advanced"
    physics: "Description of physics being solved"
    mesh_types: ["structured", "unstructured", "spherical"]
    estimated_runtime: "30 seconds"
    memory_usage: "< 100MB"
    
    # Auto-extractable metadata
    solver_class: "uw.systems.Poisson"
    mesh_creation: "uw.meshing.UnstructuredSimplexBox"
    dependencies: ["underworld3", "numpy", "sympy"]
    
    # Claude guidance
    adaptable_parameters:
      parameter_name: "Description of what changing this does"
      
    adaptation_suggestions:
      easy: ["Simple parameter changes"]
      medium: ["Physics modifications"]
      hard: ["Significant extensions"]
      
    claude_prompts:
      beginner: |
        Multi-line prompt text for beginners
        explaining what to try modifying
        
      advanced: |
        Advanced modification suggestions
        and extension possibilities
        
    learning_objectives:
      - "What users will learn from this example"
      
    related_examples:
      - "other_example.py"

# Learning paths
learning_paths:
  geophysics_beginner:
    title: "Introduction to Computational Geophysics"
    description: "Step-by-step introduction for geoscientists"
    target_audience: "Undergraduate/graduate geoscience students"
    prerequisites: ["Basic Python", "Undergraduate math/physics"]
    
    sequence:
      - step: 1
        example: "basic_diffusion_2d.py"
        concepts: ["Finite elements", "Boundary conditions", "Steady state"]
        exercises:
          - "Modify domain size and boundary temperatures"
          - "Change mesh resolution and observe accuracy"
          
      - step: 2
        example: "stokes_driven_cavity.py"
        concepts: ["Fluid mechanics", "Saddle point problems", "Velocity-pressure coupling"]
        builds_on: ["Finite elements from previous step"]

# Physics categories for organization
physics_categories:
  heat_transfer:
    examples: ["basic_diffusion_2d.py", "thermal_convection_2d.py"]
    theory_background: "https://en.wikipedia.org/wiki/Heat_equation"
    applications: ["Geothermal systems", "Planetary thermal evolution"]
    
  fluid_mechanics:
    examples: ["stokes_driven_cavity.py", "navier_stokes_channel.py"]
    prerequisites: ["basic vector calculus", "continuum mechanics"]
    applications: ["Mantle convection", "Magma flow"]
```

### 1.3 Create Core Examples (5-6 examples)
**Priority examples to implement:**

- [ ] `basic_diffusion_2d.py` - Simple steady-state heat conduction
- [ ] `stokes_driven_cavity.py` - Classical lid-driven cavity flow
- [ ] `mesh_types_demo.py` - Comparison of different mesh types
- [ ] `boundary_conditions_demo.py` - Different BC types and effects
- [ ] `time_stepping_diffusion.py` - Basic time-dependent problem

**Example format standard (Python percent format for Jupyter/Jupytext compatibility):**
```python
# %% [markdown]
"""
# Basic 2D Heat Diffusion

**CATEGORY:** poisson  
**DIFFICULTY:** basic  
**PHYSICS:** steady-state heat conduction  
**MESH_TYPE:** unstructured  
**RUNTIME:** ~30 seconds  

## Description
Solves steady-state heat equation with mixed boundary conditions.
Demonstrates basic Underworld3 workflow and finite element concepts.

## Learning Objectives
- Understand finite element mesh creation
- Learn boundary condition specification
- Visualize scalar field solutions

## Adaptable Parameters
- `thermal_diffusivity`: controls heat transport rate (try 0.1 to 10)
- `domain_size`: affects overall solution scale (try 0.5 to 2.0)  
- `mesh_resolution`: accuracy vs speed tradeoff (try 0.02 to 0.1)
- `boundary_temperatures`: driving forces for heat flow

## Claude Adaptation Hints

**Easy modifications:**
- Change boundary temperature values
- Modify domain dimensions
- Adjust thermal diffusivity

**Medium modifications:**
- Add internal heat sources
- Try temperature-dependent diffusivity
- Change to different boundary condition types

**Advanced modifications:**
- Couple with fluid flow for convection
- Add time dependence for transient heating
- Implement anisotropic materials
"""

# %% [markdown]
"""
## Parameter Setup
Define key parameters at the top for easy modification.
"""

# %%
# Parameter constants - easily modifiable
DOMAIN_MIN = (0.0, 0.0)         # PARAM: domain geometry
DOMAIN_MAX = (1.0, 1.0)         # PARAM: domain geometry  
MESH_RESOLUTION = 0.05          # PARAM: mesh resolution
THERMAL_DIFFUSIVITY = 1.0       # PARAM: material property
HOT_BOUNDARY_TEMP = 1.0         # PARAM: boundary condition
COLD_BOUNDARY_TEMP = 0.0        # PARAM: boundary condition

# %%
import underworld3 as uw
import numpy as np
import sympy

# %% [markdown]
"""
## Mesh Creation
Create computational domain with specified resolution.
"""

# %%
# SECTION: Mesh Creation
# CLAUDE_ADAPTABLE: Domain geometry and resolution
mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=DOMAIN_MIN,           # References parameter above
    maxCoords=DOMAIN_MAX,           # References parameter above
    cellSize=MESH_RESOLUTION,       # References parameter above
    qdegree=3
)

# %% [markdown]
"""
## Variable Definition
Set up temperature field on the mesh.
"""

# %%
# SECTION: Variable Definition
temperature = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)

# %% [markdown]
"""
## Solver Setup
Configure Poisson solver with diffusion physics.
"""

# %%
# SECTION: Solver Setup
# CLAUDE_ADAPTABLE: Physics parameters
poisson_solver = uw.systems.Poisson(mesh, u_Field=temperature)
poisson_solver.constitutive_model = uw.constitutive_models.DiffusionModel
poisson_solver.constitutive_model.Parameters.diffusivity = THERMAL_DIFFUSIVITY

# %% [markdown]
"""
## Boundary Conditions
Apply temperature constraints at domain boundaries.
"""

# %%
# SECTION: Boundary Conditions  
# CLAUDE_ADAPTABLE: Boundary values and types
poisson_solver.add_essential_bc([HOT_BOUNDARY_TEMP], "Bottom")
poisson_solver.add_essential_bc([COLD_BOUNDARY_TEMP], "Top")

# %% [markdown]
"""
## Solve System
Compute steady-state temperature distribution.
"""

# %%
# SECTION: Solve
poisson_solver.solve()

# %% [markdown]
"""
## Results and Visualization
Examine the solution and create basic plots.
"""

# %%
# SECTION: Post-processing
if uw.mpi.size == 1:
    import matplotlib.pyplot as plt
    # Basic visualization code here
    print(f"Solution range: {temperature.data.min():.3f} to {temperature.data.max():.3f}")
```

## Phase 2: Automation Tools (Weeks 3-4)

### 2.1 Metadata Generator
```python
# underworld3/examples/tools/metadata_generator.py
class MetadataGenerator:
    def scan_examples(self, examples_dir):
        """Automatically extract metadata from example files"""
        
    def parse_docstring_metadata(self, file_path):
        """Extract TITLE, CATEGORY, etc. from docstrings"""
        
    def analyze_code_structure(self, ast_tree):
        """Identify solver types, mesh creation, parameters"""
        
    def generate_yaml(self, output_path):
        """Create metadata.yaml from analyzed examples"""
```

### 2.2 Example Validator
```python  
# underworld3/examples/tools/example_validator.py
class ExampleValidator:
    def validate_syntax(self, example_file):
        """Check Python syntax and imports"""
        
    def check_underworld_patterns(self, example_file):
        """Verify proper UW3 usage patterns"""
        
    def test_execution(self, example_file, timeout=60):
        """Run example and check for successful completion"""
        
    def validate_metadata_consistency(self):
        """Check metadata.yaml matches actual example files"""
```

### 2.3 Claude Integration Generator
```python
# underworld3/examples/tools/claude_integration.py  
class ClaudeIntegrationGenerator:
    def generate_adaptation_hints(self, example_metadata):
        """Create parameter variation suggestions"""
        
    def build_learning_progressions(self, all_examples):
        """Generate learning path sequences"""
        
    def create_similarity_map(self, examples):
        """Find related examples based on code similarity"""
```

## Phase 3: Example Library Expansion (Weeks 5-8)

### 3.1 Basic Examples (Week 5)
- [ ] `poisson_nonlinear.py` - Temperature-dependent diffusivity
- [ ] `poisson_anisotropic.py` - Directional material properties  
- [ ] `stokes_natural_convection.py` - Buoyancy-driven flow
- [ ] `mesh_refinement_demo.py` - Adaptive mesh strategies
- [ ] `swarm_advection.py` - Particle tracking basics

### 3.2 Intermediate Examples (Week 6)  
- [ ] `thermal_convection_2d.py` - Rayleigh-Benard convection
- [ ] `stokes_variable_viscosity.py` - Temperature-dependent rheology
- [ ] `free_surface_gravity.py` - Surface deformation under gravity
- [ ] `porous_flow_darcy.py` - Groundwater/magma flow
- [ ] `phase_change_melting.py` - Solid-liquid transitions

### 3.3 Advanced Examples (Week 7)
- [ ] `mantle_convection_spherical.py` - Planetary-scale convection
- [ ] `subduction_zone_2d.py` - Tectonic process modeling  
- [ ] `magma_chamber_dynamics.py` - Multiphase volcanic systems
- [ ] `crustal_deformation_elastic.py` - Earthquake/fault modeling
- [ ] `planetary_differentiation.py` - Core formation processes

### 3.4 Utility Examples (Week 8)
- [ ] `mesh_convergence_study.py` - Systematic accuracy testing
- [ ] `parallel_scaling_analysis.py` - Performance optimization
- [ ] `boundary_condition_cookbook.py` - Comprehensive BC examples
- [ ] `post_processing_gallery.py` - Visualization techniques
- [ ] `parameter_sensitivity.py` - Uncertainty quantification

## Phase 4: Integration & Documentation (Weeks 9-10)

### 4.1 Automated Integration
- [ ] Set up CI/CD to validate examples automatically
- [ ] Create automated metadata.yaml updates
- [ ] Build example relationship mapping
- [ ] Generate Claude prompt templates

### 4.2 User Interface  
- [ ] Create example browser/search tool
- [ ] Build interactive parameter exploration
- [ ] Add example comparison features
- [ ] Integrate with existing documentation

### 4.3 Claude Training Data
- [ ] Export examples in Claude-optimized format
- [ ] Create adaptation pattern database
- [ ] Build physics knowledge connections
- [ ] Generate user interaction scenarios

## Implementation Guidelines

### Code Standards
- **Python percent format**: All examples must be in "python percent format" compatible with Jupyter/Jupytext
- **Notebook structure**: Use `# %%` cell separators for logical sections
- **Markdown cells**: Include `# %% [markdown]` cells for educational explanations
- **Consistent imports**: Always use `import underworld3 as uw`
- **Clear sections**: Use `# SECTION:` comments to mark workflow steps
- **Parameter markers**: Use `# PARAM:` for Claude-adaptable values  
- **Documentation**: Comprehensive docstrings with physics explanations
- **Error handling**: Graceful failures with helpful messages

### Metadata Standards
- **YAML format**: Human-readable, version-controllable
- **Consistent vocabulary**: Standardized categories and tags
- **Claude optimization**: Include adaptation hints and prompts
- **Learning integration**: Connect examples into educational sequences
- **Physics context**: Link to theory and applications

### Testing Requirements
- [ ] All examples must run successfully
- [ ] Execution time < 5 minutes on standard hardware
- [ ] Memory usage documented and reasonable
- [ ] Cross-platform compatibility (Linux, macOS, Windows)
- [ ] MPI compatibility where applicable

## Success Metrics

### Technical Metrics
- **Coverage**: Examples for all major UW3 solver types
- **Quality**: 95%+ examples pass automated validation  
- **Performance**: Average example runtime < 2 minutes
- **Accessibility**: Examples run on standard laptop hardware

### Educational Metrics  
- **Completeness**: Learning paths for major geophysics domains
- **Progression**: Clear skill building from basic to advanced
- **Adaptability**: Claude can suggest meaningful modifications
- **Documentation**: Self-contained examples with physics context

### User Experience Metrics
- **Discoverability**: Users can find relevant examples quickly
- **Adaptability**: Examples easily modified for user needs
- **Learning**: Progressive skill building through example sequences
- **Support**: Claude provides helpful adaptation guidance

## Long-term Vision

This example library becomes the foundation for:
- **Interactive tutorials** that adapt to user skill level
- **Research scaffolding** that helps scientists build complex models
- **Educational courseware** for computational geophysics
- **Community contributions** with standardized quality and documentation

The system should scale to handle hundreds of examples while maintaining educational coherence and Claude's ability to provide intelligent assistance.

## Next Steps

1. **Review and refine** this plan based on your priorities
2. **Set up initial directory structure** and metadata schema
3. **Create first 3-5 core examples** to validate the approach
4. **Build basic automation tools** for metadata generation
5. **Iterate and expand** based on user feedback and Claude performance

---

*This plan provides a roadmap for creating a comprehensive, Claude-parseable example library that will significantly enhance Underworld3's accessibility and educational value.*