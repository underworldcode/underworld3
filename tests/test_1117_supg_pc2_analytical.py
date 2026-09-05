"""Independent exact-solution gates for P1 CitcomS predictor-corrector transport.

No Stokes solve or fine numerical reference is used. The channel pulse is the
translated, initially smoothed pulse of Calhoun & LeVeque (2000), section 6.1,
equations 45-48, with rescaled width, origin and time:
https://doi.org/10.1006/jcph.1999.6369

The rotation uses uw.analytic.RotatingGaussian. The spherical test follows
directly from (r*T)_t = kappa*(r*T)_rr. All spatial refinements use one fixed
timestep selected from the most restrictive mesh. Set UW_PC2_RESULTS to retain
small HDF5 metrics files. Separate jobs use separate UW_MESH_CACHE_DIR paths;
Gmsh SHA256 fingerprints must match before comparing their numerical results.
"""

import hashlib
import math
import os
from pathlib import Path
import time

import numpy as np
import pytest
import sympy
from mpi4py import MPI

import underworld3 as uw
from underworld3.meshing._mesh_files import mesh_file_path

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


def _pulse(x, time_value, speed, diffusivity):
    # Nonzero initial smoothing avoids the unresolved top-hat singularity.
    width = sympy.sqrt(4.0 * (0.01 + diffusivity * time_value))
    distance = x - speed * time_value
    return (sympy.erf((0.25 - distance) / width)
            + sympy.erf((0.25 + distance) / width)) / 2


def _problem(case, h, dim=2, speed=0.0, diffusivity=0.0):
    filename = mesh_file_path(f"pc2_{case}_{dim}d_h{h:g}.msh")
    if case == "shell":
        mesh = uw.meshing.SphericalShell(
            radiusInner=0.55, radiusOuter=1.0, cellSize=h, qdegree=4, filename=filename)
    else:
        lower = (-2.0, -2.0) if case == "rotation" else (-1.5,) + (-0.25,) * (dim - 1)
        upper = tuple(-value for value in lower)
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=lower, maxCoords=upper, cellSize=h,
            qdegree=4, regular=False, filename=filename)
    fingerprint = None
    if uw.mpi.rank == 0:
        try:
            fingerprint = (None, hashlib.sha256(Path(filename).read_bytes()).hexdigest())
        except OSError as exc:
            fingerprint = (str(exc), None)
    failure, mesh_sha256 = uw.mpi.comm.bcast(fingerprint, root=0)
    assert failure is None, failure
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
    return mesh, temperature, thermal, initial, exact, end_time, boundary_tail, mesh_sha256


def _integral(mesh, expression):
    return float(uw.maths.Integral(mesh, fn=expression).evaluate())


def _save_result(name, metrics):
    """Optional rank-zero output, with write failures propagated collectively."""
    error = None
    if uw.mpi.rank == 0:
        print("PC2_ANALYTICAL " + name + " " + " ".join(
            f"{key}={value}" if isinstance(value, str) else f"{key}={value:.12g}"
            for key, value in metrics.items()), flush=True)
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


def _step_count(problems):
    dt_limit = min(float(problem[2].estimate_dt()) for problem in problems)
    end_time = problems[0][5]
    # Round down before choosing an integer number of steps, avoiding a
    # partition-dependent ceil when a stability estimate differs by roundoff.
    dt_cap = 2.0**math.floor(math.log2(min(0.005, 0.5 * dt_limit)))
    return math.ceil(end_time / dt_cap)


def _advance(problem, h, steps):
    mesh, temperature, thermal, initial, exact, end_time, tail, mesh_sha256 = problem
    timestep = end_time / steps
    norm_squared = _integral(mesh, exact**2)
    interpolation_error = math.sqrt(_integral(
        mesh, (temperature.sym[0] - initial)**2) / _integral(mesh, initial**2))
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
    return dict(
        cellsize=h, dim=mesh.dim, ncpus=uw.mpi.size, timestep=timestep,
        steps=steps, end_time=end_time, relative_l2=error,
        initial_relative_l2=interpolation_error,
        temperature_integral=heat, exact_temperature_integral=exact_heat,
        relative_heat_error=(heat - exact_heat) / exact_heat,
        minimum=minimum, maximum=maximum, solve_seconds=solve_seconds,
        boundary_tail_bound=tail, volume=_integral(mesh, sympy.Integer(1)),
        mesh_sha256=mesh_sha256)


def _spatial_refinement(case, sizes, dim=2, speed=0.0, diffusivity=0.0):
    problems = [_problem(case, h, dim, speed, diffusivity) for h in sizes]
    steps = _step_count(problems)
    errors = []
    for h, problem in zip(sizes, problems):
        mesh, temperature, thermal = problem[:3]
        metrics = _advance(problem, h, steps)
        if case == "rotation":
            centre = [_integral(mesh, coordinate * temperature.sym[0]) / metrics["temperature_integral"]
                      for coordinate in mesh.X]
            metrics["phase_error_radians"] = math.atan2(centre[1], centre[0]) - metrics["end_time"]
        if speed != 0 or case == "rotation":
            tau = uw.function.evaluate(thermal.tau, mesh._centroids)
            assert uw.mpi.comm.allreduce(bool(np.any(tau > 0)), op=MPI.LOR)
        name = f"{case}_{mesh.dim}d_u{speed:g}_k{diffusivity:g}_h{h:g}"
        _save_result(name, metrics)
        assert (np.isfinite(metrics["relative_l2"])
                and metrics["minimum"] > -0.05 and metrics["maximum"] < 1.05), metrics
        errors.append(metrics["relative_l2"])
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
    errors = _spatial_refinement("shell", (0.125, 0.0625), dim=3, diffusivity=0.02)
    assert errors[1] < 0.7 * errors[0], errors
    assert errors[1] < 0.08, errors


def test_pc2_spherical_diffusion_timestep_sensitivity():
    """Separate finite-step changes from the continuum spatial error at h=1/8."""
    problem = _problem("shell", 0.125, dim=3, diffusivity=0.02)
    mesh, temperature, thermal = problem[:3]
    difference = uw.discretisation.MeshVariable("T_difference", mesh, 1, degree=1)
    initial_values = np.array(temperature.array)
    initial_state = thermal.state
    base_steps = _step_count([problem])
    metrics, solutions = [], []
    for factor in (1, 2, 4):
        temperature.array[...] = initial_values
        thermal.temperature_rate.array[...] = 0.0
        thermal.state = initial_state
        metrics.append(_advance(problem, 0.125, base_steps * factor))
        solutions.append(np.array(temperature.array))
    norm_squared = _integral(mesh, problem[4]**2)
    changes = []
    for index in (0, 1):
        difference.array[...] = solutions[index] - solutions[index + 1]
        changes.append(math.sqrt(_integral(mesh, difference.sym[0]**2) / norm_squared))
        metrics[index]["relative_difference_to_next_dt"] = changes[-1]
    if min(changes) > 0:
        metrics[-1]["observed_time_order"] = math.log2(changes[0] / changes[1])
    for factor, result in zip((1, 2, 4), metrics):
        _save_result(f"shell_time_h0.125_dtdiv{factor}", result)
    assert all(np.isfinite(item["relative_l2"]) for item in metrics), metrics
    assert all(item["minimum"] > -0.05 and item["maximum"] < 1.05 for item in metrics), metrics
    assert changes[1] < changes[0], changes
    assert changes[1] < 0.05 * metrics[-1]["relative_l2"], (changes, metrics)


def test_exact_spherical_diffusion_satisfies_radial_heat_equation():
    r, t, ri, thickness, kappa = sympy.symbols("r t ri d kappa", positive=True)
    exact = ri / r * sympy.sin(sympy.pi * (r - ri) / thickness) * sympy.exp(
        -kappa * (sympy.pi / thickness)**2 * t)
    residual = sympy.diff(exact, t) - kappa * (
        sympy.diff(exact, r, 2) + 2 / r * sympy.diff(exact, r))
    assert sympy.simplify(residual) == 0
    assert sympy.simplify(exact.subs(r, ri)) == 0
    assert sympy.simplify(exact.subs(r, ri + thickness)) == 0
