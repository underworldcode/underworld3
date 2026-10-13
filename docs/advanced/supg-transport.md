# SUPG Scalar Transport

This page is obsolete. See
[Eulerian advection-diffusion](eulerian-advection-diffusion.md) for
`uw.systems.AdvDiffusion` and its transport managers: the default
`uw.systems.ddt.EulerianSUPG` for implicit CN/BDF transport and
`uw.systems.ddt.EulerianSUPGPC` for `method="citcoms"` or
`method="pc_converged"`, supplied through `DuDt=`. The guide covers the PC
algorithm, accuracy limitations, timestep policies and checkpoint state.
