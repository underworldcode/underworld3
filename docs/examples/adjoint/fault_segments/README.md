# Friction on a listric fault from surface observations

The example for the technical note *A Discrete Adjoint from the Run Record*
(UWTN 2026-020). Four friction coefficients on a listric fault are recovered
from the surface uplift rate and the shear stress at five interior points,
with the gradient from the solver's own adjoint.

| file | what it does |
|---|---|
| `fault_friction.ipynb` | the example as a notebook |
| `fault_friction.py` | the same, as a script in jupytext percent format; `python fault_friction.py -uw_check_only 1` runs the gradient check alone |
| `plot_fault_segments.py` | the figure: the weak plane, the uplift profiles, and the path of the coefficients, from the `_data.npz` a run writes |
| `plot_convergence.py` | convergence under the three observation sets |
| `plot_noise.py` | recovered coefficients against the noise, the bounds and the prior |
| `render_fault_friction.py` | PyVista renders of the truth: the plane's viscosity, the slip rate, the uplift and the pressure |

The gradient check takes about three minutes and the inversion about ten on
a laptop. The transcript of a run is written to `transcripts/<started>/`,
and `uw.transcript_figure(...)` draws it.

The problem is stated in kilometres, pascal seconds and millimetres per year,
and the solver works in units of the depth, the viscosity and the
convergence rate. The observation sets, the noise and the prior are switches
in the parameter block at the top of the notebook.
