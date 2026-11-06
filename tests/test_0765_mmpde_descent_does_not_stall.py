"""The MMPDE mover must not report a stalled line search as convergence.

With the default ``accel="cg"`` a conjugate direction can fail to descend. On an
annulus with a sharp thermal layer every backtrack then failed at the fifth outer
iteration of 150 and the mover stopped, having done a fifth of the energy drop
that plain descent achieves on the same metric. A failed conjugate direction says
nothing about the minimum, so the mover has to retry along steepest descent and
stop only when that fails too.
"""
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _energy_trace(accel, capsys):
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=1 / 16, qdegree=3)
    x, y = mesh.X
    r = sympy.sqrt(x ** 2 + y ** 2)
    T = uw.discretisation.MeshVariable(f"T_{accel}", mesh, 1, degree=2)
    field = 0.5 * (1 - sympy.tanh((r - 0.9) / 0.03)) * (1 + 0.3 * sympy.cos(4 * sympy.atan2(y, x)))
    T.array[:, 0, 0] = uw.function.evaluate(field, T.coords).reshape(-1)
    rho = uw.meshing.metric_density_from_gradient(
        mesh, T, refinement=5, coarsening="auto", metric_choice="front-following")
    capsys.readouterr()
    uw.meshing.node_redistribution(
        mesh, rho, verbose=True,
        method_kwargs=dict(step_frac=0.2, accel=accel, momentum=0.0, n_outer=60),
        slip_surfaces=True, skip_threshold=None)
    trace = capsys.readouterr().out
    energy = [float(line.split("I=")[1].split()[0]) for line in trace.splitlines()
              if "mmpde outer" in line]
    return energy


def test_cg_reaches_at_least_the_energy_of_plain_descent(capsys):
    cg = _energy_trace("cg", capsys)
    plain = _energy_trace("none", capsys)
    assert cg[-1] <= plain[-1], (
        f"cg stopped at I={cg[-1]:.4f} after {len(cg)} outer iterations; "
        f"plain descent reached I={plain[-1]:.4f} in {len(plain)}")
