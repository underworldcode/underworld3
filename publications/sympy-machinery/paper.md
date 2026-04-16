---
title: "Symbolic PDE Assembly in Underworld3: From SymPy Expressions to Parallel Finite Element Solvers"
short_title: Symbolic PDE Assembly in Underworld3
subject: Methods for Geoscientific Model Development
abstract: |
  We describe the design and implementation of the symbolic equation-assembly
  pipeline in Underworld3, an open-source Python framework for geodynamics
  modelling. Users specify governing equations in strong form using SymPy,
  the Python computer-algebra system. Underworld3 automatically derives the
  corresponding finite element weak forms, computes exact Jacobians for
  Newton iteration, and just-in-time compiles the resulting expressions to C
  callbacks consumed by PETSc. The pipeline preserves full symbolic
  introspection at every stage, so that the mathematical structure of a
  model — constitutive laws, boundary conditions, time discretisation — can
  be displayed, simplified, and validated interactively in a Jupyter notebook
  before the same script is deployed on a parallel cluster. We present the
  architecture of this pipeline, discuss design trade-offs between symbolic
  generality and numerical performance, and illustrate the approach with
  examples drawn from mantle convection, lithospheric deformation, and
  porous-media flow.
date: 2026-03-22
---

# Introduction

<!-- Context: the gap between textbook equations and parallel FE code -->

Geodynamics modelling requires solving coupled, nonlinear partial differential
equations (PDEs) with complex, spatially varying constitutive laws. The
distance between the textbook statement of a problem and the numerical code
that solves it is large: users must manually derive weak forms, compute
Jacobian contributions for Newton solvers, and express the result in a
low-level language that the solver library can consume. Each of these steps
is error-prone, opaque to collaborators, and a barrier to rapid prototyping.

Several projects have addressed parts of this gap. The Unified Form Language
(UFL) and FEniCS ecosystem [@loggAutomatedSolutionDifferential2012] allow
users to write weak forms symbolically in Python and compile them to efficient
C/C++ kernels. Firedrake [@daviesAutomaticFiniteelementMethods2022] extends
this with composable solvers via PETSc. TerraFERMA
[@wilsonTerraFERMARansparent2017] provides a high-level options system for
multiphysics coupling. In all of these, the user works at the level of the
*weak form*, which already requires a non-trivial derivation from the
governing equations.

Underworld3 takes a different approach. Users write the *strong form* of the
governing equations as standard SymPy expressions — the same notation found
in textbooks and journal papers. The framework then:

1. maps these expressions onto the PETSc pointwise-function template
   [@knepleyAchievingHighPerformance2013],
2. symbolically differentiates to produce exact Jacobian contributions,
3. JIT-compiles the result to C shared libraries via Cython
   [@behnel2011cython], and
4. hands function pointers to PETSc for parallel assembly and solution.

The entire pipeline is transparent: at any point the user can inspect the
SymPy expression tree, view the generated C code, or display the
mathematical form in a Jupyter notebook. This paper describes the design,
implementation, and rationale for this pipeline.


# Design Philosophy

<!-- Why strong form? Why SymPy rather than a DSL? Why runtime compilation? -->

## Strong-form-first

## SymPy as the algebra engine

## Deferred evaluation and lazy compilation


# The PETSc Pointwise-Function Template

<!-- Recap Knepley et al. — the F0/F1/G structure that UW3 targets -->

The PETSc `DMPlex` finite element infrastructure provides a template for
residual and Jacobian assembly based on pointwise callback functions
[@knepleyAchievingHighPerformance2013; @balayPETScTAOUsers2024]. Rather
than requiring users to write element-level integration routines, PETSc
decomposes the weak form into contributions that depend on the trial
function value ($f_0$) and its gradient ($f_1$), and similarly for the
Jacobian ($g_0, g_1, g_2, g_3$).

$$
\mathcal{F}(u) \sim \sum_e \epsilon_e^T \left[
  B^T\, W\, f_0(u^q, \nabla u^q)
  + D^T\, W\, f_1(u^q, \nabla u^q)
\right] = 0
$$ (eq:petsc-weak-form)

The user's responsibility reduces to providing C functions for $f_0$, $f_1$
and the corresponding Jacobian blocks $g_0 \ldots g_3$. This is precisely
what Underworld3 generates from the symbolic strong form.


# From Strong Form to Weak Form: The Symbolic Pipeline

## UWexpression: the symbolic wrapper

<!-- Lazy evaluation, symbol disambiguation, unit propagation -->

## Expression unwrapping and compilation

<!-- _unwrap_for_compilation, coordinate handling -->

## Constitutive models as symbolic objects

<!-- ViscousFlowModel etc. — composable, auto-differentiated -->

## Time discretisation

<!-- DDt hierarchy, BDF/AM as symbolic rewrites, history terms -->


# JIT Compilation

## Cython code generation

## Symbol-to-C mapping

## Shared-library loading and PETSc callback registration


# Solver Integration

## Template solvers: Stokes, Navier–Stokes, Darcy, Poisson

## Nonlinear iteration and automatic Jacobians

## Boundary conditions as symbolic expressions


# Introspection and the Notebook Experience

## Mathematical display of assembled forms

## Dimensional analysis via the units system

## From notebook prototype to HPC script


# Examples

## Mantle convection with composite rheology

## Lithospheric extension with plasticity

## Darcy flow with heterogeneous permeability


# Discussion

<!-- Trade-offs: generality vs performance, SymPy limitations,
     comparison with UFL/Firedrake approach, future directions -->


# Code and data availability

Underworld3 is open-source software released under the LGPLv3 licence.
Source code, documentation, and example notebooks are available at
[https://github.com/underworldcode/underworld3](https://github.com/underworldcode/underworld3).

# Author contributions

<!-- CRediT roles -->

# Competing interests

The authors declare that they have no conflict of interest.

# Acknowledgements

AuScope provides direct support for the core development team behind the
Underworld codes. AuScope is funded by the Australian Government through
the National Collaborative Research Infrastructure Strategy, NCRIS.

# References
