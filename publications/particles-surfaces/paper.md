---
title: "Lagrangian Particles and Surface Tracking in Underworld3: Symbolic Integration of Material History"
short_title: Particles and Surfaces in Underworld3
subject: Methods for Geoscientific Model Development
abstract: |
  Placeholder abstract.
date: 2026-03-22
---

# Introduction

<!-- Lagrangian particle methods for geodynamics; the need for
     material tracking in large-deformation problems -->


# Background: Particle-in-Cell and Lagrangian Integration Points

<!-- Moresi et al. 2003; evolution from UW1 → UW2 → UW3 -->


# Swarm Implementation

## PETSc DMSwarm

## Parallel decomposition and migration

## Population control


# Proxy Mesh Variables

## RBF projection from particles to mesh

## Symbolic participation: swarm variables as UWexpressions


# Material History and Time Derivatives

## Stress history for viscoelasticity

## Advection of stored fields

## BDF/Adams–Moulton schemes on particles


# Boundary and Surface Integrals

## External boundary integrals

## Internal surfaces and interfaces

## The ghost-ownership problem in parallel


# Examples

## Subduction with material tracking

## Viscoelastic lithosphere with stress history

## Two-phase flow with a tracked interface


# Discussion


# Code and data availability

Underworld3 is open-source software released under the LGPLv3 licence.
Source code, documentation, and example notebooks are available at
[https://github.com/underworldcode/underworld3](https://github.com/underworldcode/underworld3).

# Author contributions

# Competing interests

The authors declare that they have no conflict of interest.

# Acknowledgements

# References
