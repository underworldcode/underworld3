"""The MMPDE mover must not stall far from the minimum of its energy.

On an annulus with a sharp thermal layer the default mover stopped at the fifth
outer iteration of 150, its energy 9 % above what plain descent reaches. Two causes:
a failed conjugate direction was taken for convergence, and the metric, interpolated
from its nearest nodes, made the energy jump where those nodes change, so even
steepest descent could not go further.
"""
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


def test_the_default_mover_reaches_its_minimum(capsys):
    """From the first step's energy: stalled, it fell 0.5 % in 3 outer iterations;
    converged, 13 % in 38."""
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=1 / 16, qdegree=3)
    x, y = mesh.X
    r = sympy.sqrt(x ** 2 + y ** 2)
    T = uw.discretisation.MeshVariable("T0765", mesh, 1, degree=2)
    field = 0.5 * (1 - sympy.tanh((r - 0.9) / 0.03)) * (1 + 0.3 * sympy.cos(4 * sympy.atan2(y, x)))
    T.array[:, 0, 0] = uw.function.evaluate(field, T.coords).reshape(-1)
    density = uw.meshing.metric_density_from_gradient(
        mesh, T, refinement=5, coarsening="auto", metric_choice="front-following")
    capsys.readouterr()
    uw.meshing.node_redistribution(
        mesh, density, verbose=True,
        method_kwargs=dict(step_frac=0.2, accel="cg", momentum=0.0),
        slip_surfaces=True, skip_threshold=None)
    trace = capsys.readouterr().out
    energy = [float(line.split("I=")[1].split()[0]) for line in trace.splitlines()
              if "mmpde outer" in line]
    drop = (energy[0] - energy[-1]) / energy[0]
    assert drop > 0.08, f"energy fell {drop:.1%} in {len(energy)} outer iterations"
