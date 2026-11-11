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
    # Only rank 0 prints the trace; every rank parses its copy.
    trace = uw.mpi.comm.bcast(capsys.readouterr().out, root=0)
    energy = [float(line.split("I=")[1].split()[0]) for line in trace.splitlines()
              if "mmpde outer" in line]
    drop = (energy[0] - energy[-1]) / energy[0]
    assert drop > 0.08, f"energy fell {drop:.1%} in {len(energy)} outer iterations"


def _frozen_metric_fields(monkeypatch):
    """Record the name of every MeshVariable built while the mover runs."""
    made = []
    cls = uw.discretisation.MeshVariable
    real_init = cls.__init__

    def recording(self, varname=None, *args, **kwargs):
        made.append(varname)
        real_init(self, varname, *args, **kwargs)

    monkeypatch.setattr(cls, "__init__", recording)
    return made


def test_a_constant_metric_builds_no_frozen_field(monkeypatch):
    """``Mesh.relax`` passes a constant metric, which the mover reads exactly at any
    point. It skips the frozen copy of the mesh that a spatial metric needs. The
    control is the same mover on a metric that varies in space."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25, qdegree=2)
    made = _frozen_metric_fields(monkeypatch)
    mesh.relax()
    assert "M_frozen" not in made, made

    x, y = mesh.X
    uw.meshing.node_redistribution(mesh, 1.0 + 4.0 * sympy.exp(-((y - 0.5) / 0.1) ** 2))
    assert "M_frozen" in made, made

