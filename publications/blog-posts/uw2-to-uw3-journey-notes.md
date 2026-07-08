---
title: Raw notes for "Our Journey from Underworld2 to Underworld3"
status: notes
feeds_into: [paper-1, release-post]
---

# Source Material: Louis's Account

## The Three Eras

### Underworld1: XML-composed modular C code
- Modular design with XML tools to compose modules into a working code
- **Pro**: Completely deterministic
- **Con**: Deterministic → inflexible, difficult to work with

### Underworld2: Python wrapper over C engine
- Original motivation: interoperability with other codes/tools
- Functions and variables made for a near-Python programming experience
- That carried forward into parallel very nicely
- UWGeodynamics (Beucher) provided a higher-level structured interface

### Underworld3: Clean break
- PETSc DMPlex directly, no StGermain
- SymPy symbolic layer
- JIT compilation

## The Motivations (in Louis's priority order)

### 1. PETSc maturity gap
- Started using PETSc for UW1 when it was very immature
- Had our own FE engine on top
- Solvers and meshing were very primitive as a result
- Too much maintenance — wanted to leverage modern PETSc capabilities
- **Could not use SNES (nonlinear solvers) with UW2 strategy**

### 2. StGermain was a liability
- UW3 was an opportunity to set it aside
- XML component loading, C object lifecycle, dlopen gymnastics
- A framework from a different era

### 3. Meshing limitations
- Only hex meshes with regular shapes in UW2
- Wanted spherical and elliptical meshes — "not possible with UW2 (believe me we tried)"
- DMPlex in modern PETSc provided unstructured mesh support

### 4. Build system / deployment
- "The agony of SWIG"
- Impossibility of maintaining the software stack
- Parallel / HPC horrors
- The sheer number of .py files from SWIG meant loading on HPC filesystems would often randomly fail

### 5. Introspection (a bonus, not the original driver)
- SymPy + PETSc pointwise C injection made this "a fabulous step up"
- But it was NOT originally the motivation — it emerged from the design choice
- The ability to see, simplify, and differentiate the mathematical structure came for free once SymPy was chosen

## What Was Preserved
- The particle-in-cell philosophy
- Python-first user interface
- Parallel safety by design
- The idea of composable mathematical objects (Functions → SymPy expressions)
- PETSc underneath (but now used properly)

## What We Still Miss from UW2
- **Particle machinery** — UW2's particle infrastructure is still ahead in some respects:
  - Population control is not yet as good in UW3
  - The move to level-set-based material representations (MultiMaterialConstitutiveModel)
    is more general but less intuitive than UW2's integer-key material mapping
    (`fn.branching.map` with swarm variable keys was very natural)

## How the Design Evolved After the Break
- **Lazy evaluation became central** — started as an implementation detail, now
  defines the architecture. Expressions are built symbolically, derivatives are
  deferred, compilation happens only when the solver needs C callbacks.
- **Lazy derivatives** — the ability to symbolically differentiate constitutive
  models (for Jacobians) without evaluating them is what makes SNES integration
  seamless. This was not an original design goal but became the keystone.
- **Templated problem descriptions** — lazy evaluation enables symbolic problem
  templates (Stokes, Navier-Stokes, Darcy) where the user fills in constitutive
  laws and the framework handles the rest. The template *is* the mathematics.
- **We leaned in** — once introspection and lazy evaluation proved powerful, the
  design deliberately embraced them. SymPy went from "convenient expression
  language" to "the way the code thinks about physics".
- **AI-assisted discovery** — tracing the logic of lazy derivatives and the
  compilation pipeline was one of the early Claude-assisted wins. AI could
  follow the symbolic chain from user expression through unwrapping to generated
  C code, identifying bugs and simplification opportunities that were hard to
  see manually. This fed back into making the pipeline more transparent.

## Key Narrative Points for the Blog Post
- The rewrite was driven by **infrastructure pain** (StGermain, SWIG, primitive PETSc usage, hex-only meshes) more than by feature ambition
- SymPy introspection was a **happy consequence** of good design choices, not the goal — but then we leaned in hard
- Lazy evaluation and lazy derivatives became the architectural keystone — enabling symbolic templates, automatic Jacobians, and SNES integration
- The Function concept from UW2 was the right idea — but implemented in the wrong layer (opaque C). Moving it to SymPy made it inspectable and composable
- SNES was the solver unlock — UW2's solver strategy couldn't leverage PETSc's nonlinear solver framework
- Particle machinery is still catching up — an honest assessment, not everything is better yet
- AI assistance accelerated the maturation of the symbolic pipeline — a concrete example for the AI strategy post
