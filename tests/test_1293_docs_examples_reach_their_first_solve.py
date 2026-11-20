"""A curated set of docs/examples scripts runs to its first solve (#944, part C).

The examples are not tested otherwise, and on 2026-11-20 every changed one
failed on development: an API change breaks them silently (#944 part A, #1289).
This runs each listed script in a subprocess, in a copy of its own folder, and
stops it cleanly after its first solve returns: the script has built its mesh,
variables, solver and conditions with the current API, and the solve has run.
What follows the first solve (plots, time loops, output) is not checked.

Selection rule: scripts that ran to completion in #944's runs, a few per
physics family, each at most 30 s there to its end (macOS, Apple M4, serial),
needing no optional package (``assess``, ``cmcrameri``) and no data from
outside the repository. A script that changes so that it no longer reaches a
solve fails here, as it would for a reader; a new example joins the list when
it meets the rule.
"""
import os
import pathlib
import shutil
import subprocess
import sys
import textwrap

import pytest

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b, pytest.mark.timeout(1800)]

EXAMPLES = pathlib.Path(__file__).resolve().parent.parent / "docs" / "examples"

SCRIPTS = [
    # Poisson, diffusion and the semi-Lagrangian heat solvers
    "heat_transfer/intermediate/Ex_Poisson_Cartesian.py",
    "heat_transfer/intermediate/Ex_DiffusionSLCN_Cartesian.py",
    "heat_transfer/advanced/Ex_Poisson_v.SLCN.py",
    # Stokes: Cartesian, internal boundaries, a nonlinear analytic case, the annulus
    "fluid_mechanics/intermediate/Ex_Stokes_Flow_Obstruction.py",
    "fluid_mechanics/intermediate/Ex_Stokes_Flow_Internal_BC.py",
    "fluid_mechanics/advanced/Ex_Stokes_Cartesian_SolNL.py",
    "fluid_mechanics/advanced/Ex_Stokes_Annulus_Kernels.py",
    # advection-diffusion and convection
    "convection/advanced/Ex_AdvectionDiffusion_1dBlock.py",
    "convection/advanced/Ex_AdvectionDiffusionSLCN_RotationTest.py",
    "convection/advanced/Ex_Convection_Disc_InternalHeat.py",
    # porous flow, plasticity
    "porous_flow/basic/Ex_Darcy_1D_benchmark-Swarm.py",
    "solid_mechanics/advanced/Ex_Shear_Band_Plasticity_PS.py",
    # boundary conditions from fields, integrals, projections
    "utilities/intermediate/Ex_BCs_from_meshVariables.py",
    "utilities/intermediate/Ex_Projection_Function_Eval.py",
    "utilities/advanced/Ex_Integrals_on_Meshes.py",
    # units, and the adjoint transcript
    "Tutorial_Thermal_Convection_Units.py",
    "adjoint/sinker_transcript/generate_target_transcript.py",
]

# Wraps the solve of every solver class in uw.systems so that the first solve to
# return ends the script with status 0; then runs the script as __main__.
BOOTSTRAP = textwrap.dedent("""
    import runpy, sys
    import underworld3 as uw

    def _stop_after(solve):
        def wrapped(self, *args, **kwargs):
            result = solve(self, *args, **kwargs)
            print("REACHED_FIRST_SOLVE", type(self).__name__, flush=True)
            raise SystemExit(0)
        return wrapped

    for name in dir(uw.systems):
        cls = getattr(uw.systems, name)
        if isinstance(cls, type) and callable(getattr(cls, "solve", None)):
            try:
                cls.solve = _stop_after(cls.solve)
            except (TypeError, AttributeError):
                pass
    sys.argv = [sys.argv[1]]
    runpy.run_path(sys.argv[0], run_name="__main__")
    print("ENDED_WITHOUT_A_SOLVE", flush=True)
""")


def test_every_listed_script_exists():
    missing = [s for s in SCRIPTS if not (EXAMPLES / s).is_file()]
    assert not missing, missing


@pytest.mark.parametrize("script", SCRIPTS)
def test_the_script_reaches_its_first_solve(script, tmp_path):
    source = EXAMPLES / script
    work = tmp_path / source.parent.name
    shutil.copytree(source.parent, work,
                    ignore=shutil.ignore_patterns("output", "*.h5", "*.xdmf", "__pycache__"))
    env = dict(os.environ, OMP_NUM_THREADS="1", MPLBACKEND="Agg", PYVISTA_OFF_SCREEN="true")
    run = subprocess.run([sys.executable, "-c", BOOTSTRAP, source.name], cwd=work, env=env,
                         capture_output=True, text=True, timeout=600)
    tail = (run.stdout + run.stderr)[-3000:]
    assert "REACHED_FIRST_SOLVE" in run.stdout, tail
    assert run.returncode == 0, tail
