"""Independent exact-solution gates for P1 CitcomS predictor-corrector transport.

No Stokes solve or fine numerical reference is used. The channel pulse is the
translated, initially smoothed pulse of Calhoun & LeVeque (2000), section 6.1,
equations 45-48, with rescaled width, origin and time:
https://doi.org/10.1006/jcph.1999.6369

The rotation uses uw.analytic.RotatingGaussian. The spherical test follows
directly from (r*T)_t = kappa*(r*T)_rr. All spatial refinements use one fixed
timestep selected from the most restrictive mesh. Set UW_PC2_RESULTS to retain
small HDF5 metrics files; serial and MPI runs must share UW_MESH_CACHE_DIR.
"""

import math
import os
from pathlib import Path
import time

import numpy as np
import pytest
import sympy
from mpi4py import MPI

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


def _pulse(x, time_value, speed, diffusivity):
    # Nonzero initial smoothing avoids the unresolved top-hat singularity.
    width = sympy.sqrt(4.0 * (0.01 + diffusivity * time_value))
    distance = x - speed * time_value
    return (sympy.erf((0.25 - distance) / width)
            + sympy.erf((0.25 + distance) / width)) / 2


def _problem(case, h, dim=2, speed=0.0, diffusivity=0.0):
    if case == "shell":
        mesh = uw.meshing.SphericalShell(
            radiusInner=0.55, radiusOuter=1.0, cellSize=h, qdegree=4)
    else:
        lower = (-2.0, -2.0) if case == "rotation" else (-1.5,) + (-0.25,) * (dim - 1)
        upper = tuple(-value for value in lower)
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=lower, maxCoords=upper, cellSize=h,
            qdegree=4, regular=False)
    temperature = uw.discretisation.MeshVariable("T", mesh, 1, degree=1)
    velocity = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=1)
    velocity.array[...] = 0.0

    if case == "rotation":
        oracle = uw.analytic.RotatingGaussian(
            mesh, sigma=0.2, centre_radius=0.5, omega=1.0,
            diffusivity=diffusivity)
        initial = oracle.at(0.0)
        end_time = float(sympy.pi / 2)
        exact = oracle.at(end_time)
        velocity.array[:, 0, 0] = -velocity.coords[:, 1]
        velocity.array[:, 0, 1] = velocity.coords[:, 0]
        boundaries = ("Left", "Right", "Top", "Bottom")
        # Max Gaussian tail on the square throughout this quarter turn.
        variance = 0.2**2 + 2 * diffusivity * end_time
        boundary_tail = 0.2**2 / variance * math.exp(-1.5**2 / (2 * variance))
    elif case == "shell":
        radius = sympy.sqrt(sum(x**2 for x in mesh.X))
        initial = 0.55 / radius * sympy.sin(sympy.pi * (radius - 0.55) / 0.45)
        end_time = 0.2
        exact = initial * sympy.exp(-diffusivity * (sympy.pi / 0.45)**2 * end_time)
        boundaries = ("Lower", "Upper")
        boundary_tail = 0.0
    else:
        initial = _pulse(mesh.X[0], 0.0, speed, diffusivity)
        end_time = 0.2
        exact = _pulse(mesh.X[0], end_time, speed, diffusivity)
        velocity.array[:, 0, 0] = speed
        # Zero transverse diffusive flux is exact. End-wall tails are bounded
        # analytically, not silently treated as exactly zero.
        boundaries = ("Left", "Right")
        boundary_tail = math.erfc(
            (1.5 - 0.25 - abs(speed) * end_time)
            / math.sqrt(4 * (0.01 + diffusivity * end_time)))
    assert boundary_tail < 1e-10, boundary_tail
    temperature.array[:, 0, 0] = uw.function.evaluate(
        initial, temperature.coords).reshape(-1)
    thermal = uw.systems.AdvDiffusionSUPG(
        mesh, temperature, velocity.sym, time_integrator="citcoms",
        adv_gamma=0.5, corrector_steps=2)
    thermal.constitutive_model.Parameters.diffusivity = diffusivity
    for boundary in boundaries:
        thermal.add_dirichlet_bc(0.0, boundary)
    return mesh, temperature, thermal, initial, exact, end_time, boundary_tail


def _integral(mesh, expression):
    return float(uw.maths.Integral(mesh, fn=expression).evaluate())


def _save_result(name, metrics):
    """Optional rank-zero output, with write failures propagated collectively."""
    error = None
    if uw.mpi.rank == 0:
        print("PC2_ANALYTICAL " + name + " " + " ".join(
            f"{key}={value:.12g}" for key, value in metrics.items()), flush=True)
        directory = os.environ.get("UW_PC2_RESULTS")
        if directory:
            try:
                import h5py

                target = Path(directory) / f"ncpus_{uw.mpi.size}"
                target.mkdir(parents=True, exist_ok=True)
                with h5py.File(target / f"{name}.h5", "w") as output:
                    output.attrs["method"] = "citcoms_pc2"
                    for key, value in metrics.items():
                        output[key] = value
            except Exception as exc:
                error = f"Cannot write PC2 analytical metrics: {exc}"
    error = uw.mpi.comm.bcast(error, root=0)
    assert error is None, error


def _spatial_refinement(case, sizes, dim=2, speed=0.0, diffusivity=0.0):
    problems = [_problem(case, h, dim, speed, diffusivity) for h in sizes]
    dt_limit = min(float(problem[2].estimate_dt()) for problem in problems)
    end_time = problems[0][5]
    # Round down before choosing an integer number of steps, avoiding a
    # partition-dependent ceil when a stability estimate differs by roundoff.
    dt_cap = 2.0**math.floor(math.log2(min(0.005, 0.5 * dt_limit)))
    steps = math.ceil(end_time / dt_cap)
    timestep = end_time / steps
    errors = []
    for h, (mesh, temperature, thermal, initial, exact, _, tail) in zip(sizes, problems):
        norm_squared = _integral(mesh, exact**2)
        initial_norm_squared = _integral(mesh, initial**2)
        interpolation_error = math.sqrt(_integral(
            mesh, (temperature.sym[0] - initial)**2) / initial_norm_squared)
        start = time.perf_counter()
        for _ in range(steps):
            thermal.solve(timestep=timestep)
        solve_seconds = uw.mpi.comm.allreduce(time.perf_counter() - start, op=MPI.MAX)
        error = math.sqrt(_integral(mesh, (temperature.sym[0] - exact)**2) / norm_squared)
        heat = _integral(mesh, temperature.sym[0])
        exact_heat = _integral(mesh, exact)
        nodal = temperature.array[:, 0, 0]
        minimum = uw.mpi.comm.allreduce(float(np.min(nodal, initial=np.inf)), op=MPI.MIN)
        maximum = uw.mpi.comm.allreduce(float(np.max(nodal, initial=-np.inf)), op=MPI.MAX)
        metrics = dict(
            cellsize=h, dim=mesh.dim, ncpus=uw.mpi.size, timestep=timestep,
            steps=steps, end_time=end_time, relative_l2=error,
            initial_relative_l2=interpolation_error,
            temperature_integral=heat, exact_temperature_integral=exact_heat,
            relative_heat_error=(heat - exact_heat) / exact_heat,
            minimum=minimum, maximum=maximum, solve_seconds=solve_seconds,
            boundary_tail_bound=tail, volume=_integral(mesh, sympy.Integer(1)))
        if case == "rotation":
            centre = [_integral(mesh, coordinate * temperature.sym[0]) / heat
                      for coordinate in mesh.X]
            metrics["phase_error_radians"] = math.atan2(centre[1], centre[0]) - end_time
        if speed != 0 or case == "rotation":
            tau = uw.function.evaluate(thermal.tau, mesh._centroids)
            assert uw.mpi.comm.allreduce(bool(np.any(tau > 0)), op=MPI.LOR)
        name = f"{case}_{mesh.dim}d_u{speed:g}_k{diffusivity:g}_h{h:g}"
        _save_result(name, metrics)
        assert np.isfinite(error) and minimum > -0.05 and maximum < 1.05, metrics
        errors.append(error)
    return errors


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("speed,diffusivity", [(1.0, 0.0), (0.0, 0.01), (1.0, 0.01)])
def test_pc2_exact_channel_pulse(dim, speed, diffusivity):
    errors = _spatial_refinement("pulse", (0.125, 0.0625), dim, speed, diffusivity)
    assert errors[1] < 0.7 * errors[0], errors
    assert errors[1] < 0.05, errors


def test_pc2_exact_rotating_gaussian():
    errors = _spatial_refinement("rotation", (0.125, 0.0625))
    assert errors[1] < 0.7 * errors[0], errors
    assert errors[1] < 0.10, errors


def test_pc2_exact_spherical_diffusion():
    errors = _spatial_refinement("shell", (0.25, 0.125), dim=3, diffusivity=0.02)
    assert errors[1] < 0.7 * errors[0], errors
    assert errors[1] < 0.08, errors


def test_exact_spherical_diffusion_satisfies_radial_heat_equation():
    r, t, ri, thickness, kappa = sympy.symbols("r t ri d kappa", positive=True)
    exact = ri / r * sympy.sin(sympy.pi * (r - ri) / thickness) * sympy.exp(
        -kappa * (sympy.pi / thickness)**2 * t)
    residual = sympy.diff(exact, t) - kappa * (
        sympy.diff(exact, r, 2) + 2 / r * sympy.diff(exact, r))
    assert sympy.simplify(residual) == 0
    assert sympy.simplify(exact.subs(r, ri)) == 0
    assert sympy.simplify(exact.subs(r, ri + thickness)) == 0
