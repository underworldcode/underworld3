# SUPG Scalar Transport

The general implicit solver and the CitcomS predictor-corrector now share
one implementation and one public class, `uw.systems.AdvDiffusionSUPG`.

See [Eulerian advection-diffusion](eulerian-advection-diffusion.md) for the
method table, CitcomS mode, timestep policies, and restart requirements.

Existing scripts selecting `time_integrator="citcoms"` retain that method.
The general default is now Crank-Nicolson; select `theta=1.0` or
`time_integrator="bdf"` explicitly for backward Euler.
