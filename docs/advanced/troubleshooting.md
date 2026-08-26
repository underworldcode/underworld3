---
title: "Troubleshooting"
---

# Troubleshooting

Notes on behaviour changes and common pitfalls that show up as "my old script
runs but gives different answers / different performance".

## Stokes penalty is now viscosity-scaled (June 2026)

The Stokes augmented-Lagrangian (grad-div) incompressibility penalty changed
from a bare constant to a **viscosity-scaled** term. The operator changed
from

$$
\sigma + \lambda \, (\nabla \cdot \mathbf{u}) \, I
$$

to

$$
\sigma + \lambda \, \mu \, (\nabla \cdot \mathbf{u}) \, I
$$

where $\mu$ is the local viscosity (`constitutive_model.K`) and $\lambda$ is
the value you set with `stokes.penalty`.

Why: with spatially variable viscosity, a constant penalty is enormous
relative to the stress in low-viscosity regions (over-stiffening them into
velocity locking) and negligible in high-viscosity regions. Scaling by the
local viscosity keeps the penalty proportional to the local stress scale
everywhere.

**What to change in older scripts:**

1. `stokes.penalty` is now a *dimensionless* number of order 1. Scripts that
   tuned a large constant against the viscosity magnitude (say `penalty=1e6`
   for a model with $\mu \sim 1$, or a value chosen to sit above the largest
   viscosity in a contrast model) should drop that tuning — use `1.0` (or a
   modest multiple) instead. Keeping the old large value multiplies it by
   the viscosity again and can lock the velocity field or destroy the
   conditioning of the velocity block.
2. A penalty is still **off by default** (`penalty = 0`) and usually
   unnecessary: the pressure Schur complement is already preconditioned with
   the local $1/\mu$ scaling.
3. The penalty term is part of the operator, so converged pressures include
   the grad-div contribution; diagnostics that compared pressure against a
   constant-penalty run will differ at the level of the divergence residual.

See the `Stokes.penalty` property docstring for the full description, and the
`CONSTRAINED_FREESLIP_MULTIPLIER` design note for the derivation.

## A long output path can segfault parallel HDF5 output (#645)

**Symptom.** A native segmentation fault inside `mesh.write()` or
`mesh.write_timestep()` on an HPC filesystem, with no Python traceback. The mesh,
labels, fields, solve, MPI size and output calls are all valid, and the same script
succeeds when only the output directory is renamed.

**Cause.** The failure follows the *length of the full generated filename*, not the
validity of the path. UW3 appends its own suffixes to whatever you pass — a mesh write
becomes `output.mesh.00000.h5`, and a P2 velocity variable becomes
`output.mesh.U.00000.h5` — so a descriptive case directory can push the complete path
past what the native PETSc/HDF5/MPI-I/O stack tolerates.

```{warning}
This happens **well below** the advertised limits. In the reported case every path
component was under `NAME_MAX` (255), the filesystem allowed `PATH_MAX=4096`, and PETSc
was configured with `PETSC_MAX_PATH_LEN=4096`. A 286-character filename still crashed.
```

**Reported thresholds.** On Gadi (PETSc 3.25.4, HDF5 1.12.2p, Open MPI 4.1.7, MPI-enabled
h5py 3.16.0, GPFS) a staged test failed at a **259-character** filename, and 252
characters is the shortest length observed to fail on that stack. The exact first unsafe
length was not established, so treat these as observations on one stack rather than a
portable threshold. A macOS stack (PETSc 3.25.0, HDF5 1.14.6, Open MPI 5.0.10) did not
reproduce the original failure, though the full threshold matrix was not repeated there.

**What to do.**

1. Keep the output root and the generated case identifier **compact**, especially on HPC.
2. Put fixed configuration in HDF5 metadata or a metrics file rather than encoding every
   parameter in the directory name. A directory named for the two or three parameters
   that actually vary across a campaign is enough to tell runs apart.
3. If output segfaults natively, **print the full generated filename first** — including
   the UW3 suffixes — before investigating the mesh, the labels or the solver.

Shortening the longest filename from 286 to 189 characters made an otherwise unchanged
eight-rank benchmark pass end to end, and a subsequent 192-rank run at 1/64 resolution
completed every mesh, Stokes, HDF5, XDMF and postprocessing stage.
