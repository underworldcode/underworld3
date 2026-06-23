"""Adaptive-mesh stagnant-lid annulus convection as an underworld3 workflow.

A worked example of ``underworld3.workflows`` driving an *adaptive-mesh*
thermal convection run on the fixed mmpde mover base (PRs #259 + #264 +
#266: deformed-mesh point-location + monotone RBF metric bake + SPD
floor). It is the no-fault baseline; a fault variant builds on the same
config + steps.

Structure (the ``underworld3.workflows`` pattern):

  * ``AdaptiveConvectionConfig`` — a pydantic ``WorkflowConfig`` holding
    every parameter, with identity fields (mesh + physics + adaptation)
    separated from operational fields (cadence, step caps).
  * ``@workflow_step`` DAG — ``create_mesh`` → ``create_solvers`` →
    ``evolve`` → ``summarise_run``, resolved by a ``WorkflowRunner``.
  * the run-directory product (``Run``) owns the bytes on disk: manifest,
    append-only ``timeseries.csv``, h5 checkpoints, steady-state summary.

The adaptation recipe (see the ``adaptive-meshing`` skill and
``project_mmpde_holes_real_root_cause``):

  mover    ``smooth_mesh_interior(method="mmpde", step_frac=0.2,
           accel="cg", momentum=0.0, slip_surfaces=True)`` — variational,
           non-folding; owns field transfer.
  metric   ``metric_density_from_gradient(refinement=R,
           metric_choice="front-following")`` — R≈5 grading from |∇T|.
  Stokes   isotropic Frank-Kamenetskii ``η = exp(θ(1−T))`` (θ=ln Δη),
           buoyancy ``Ra·(T−T_cond)·r̂``, no-slip inner, free-slip outer
           (penalty), ``snes_type=ksponly``.
  advect   ``AdvDiffusionSLCN(theta=1, monotone_mode='clamp')``;
           ``dt = estimate_dt(percentile=50)·dt_mult`` — SLCN is
           unconditionally stable so the small adapted cell need not
           govern dt.

Run it via ``simulate.py`` (auto-CLI) or from Python:

    import config as ac
    from underworld3.workflows import WorkflowRunner
    cfg = ac.AdaptiveConvectionConfig(output_dir="...", max_steps=80)
    runner = WorkflowRunner(ac, cfg)
    runner.build("run_summary")
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import sympy
from pydantic import Field

from underworld3.workflows import (
    RUN_NAME,
    Run,
    WorkflowConfig,
    config_cache_key,
    config_snapshot,
    workflow_step,
)

# diagnostics.py sits next to this module; the workflow module and the
# simulate CLI run with this directory on sys.path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diagnostics as diag  # noqa: E402


# Identity fields — a change to any of these invalidates an on-disk run.
_IDENTITY_FIELDS = (
    "r_inner", "r_outer", "cellsize", "qdegree", "refinement", "T_degree",
    "rayleigh", "delta_eta", "diffusivity",
    "resolution_ratio", "adapt_every", "adapt_start", "mover_accel",
    "freeslip", "pert_mode", "pert_amplitude",
    "seed_kind", "seed_theta_deg", "seed_depth", "seed_width",
)


def view():
    """Display the workflow steps and config classes in this module."""
    from underworld3.workflows import view as _wf_view
    _wf_view(sys.modules[__name__])


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class AdaptiveConvectionConfig(WorkflowConfig):
    """Parameters for adaptive-mesh stagnant-lid annulus convection."""

    workflow_name: str = "adaptive_convection"
    description: str = (
        "Adaptive-mesh (mmpde) stagnant-lid Frank-Kamenetskii convection "
        "in an annulus, on the fixed mover base"
    )

    _identity_fields = _IDENTITY_FIELDS

    # Identity — geometry / mesh
    r_inner: float = Field(default=0.5, gt=0)
    r_outer: float = Field(default=1.0, gt=0)
    cellsize: float = Field(default=1.0 / 24, gt=0,
                            description="Uniform base cell size (1/res).")
    qdegree: int = Field(default=3, ge=1)
    refinement: int = Field(default=0, ge=0,
                            description="Uniform refinement levels for an "
                                        "FMG dm_hierarchy (0 = GAMG).")
    T_degree: int = Field(default=3, ge=1)

    # Identity — physics
    rayleigh: float = Field(default=1e6, gt=0)
    delta_eta: float = Field(default=1e3, gt=0,
                             description="Frank-Kamenetskii viscosity "
                                         "contrast η(cold)/η(hot).")
    diffusivity: float = Field(default=1.0, gt=0)

    # Identity — initial condition
    seed_kind: Literal["mode", "blob"] = Field(
        default="mode",
        description="Initial T perturbation shape. 'mode' = global azimuthal "
                    "sin(pert_mode·θ) (plumes everywhere). 'blob' = a single "
                    "localized gaussian hot anomaly at (seed_theta_deg, "
                    "seed_depth) — use to seed convection AWAY from a fault so "
                    "the rising plume does not sit on it.")
    pert_mode: int = Field(default=5, ge=1,
                           description="Azimuthal wavenumber of the initial "
                                       "T perturbation (seed_kind='mode').")
    pert_amplitude: float = Field(default=0.05,
                                  description="Perturbation amplitude (peak ΔT "
                                              "of the mode or the blob).")
    seed_theta_deg: float = Field(default=270.0,
                                  description="Azimuth of the blob seed "
                                              "(seed_kind='blob'); 90 = the "
                                              "default fault trace, 270 = "
                                              "directly opposite.")
    seed_depth: float = Field(default=0.5, ge=0.0, le=1.0,
                              description="Fractional radial position of the "
                                          "blob (0 = inner/hot boundary, 1 = "
                                          "outer/cold).")
    seed_width: float = Field(default=0.12, gt=0,
                              description="Gaussian half-width of the blob "
                                          "(model coords).")

    # Identity — adaptation
    resolution_ratio: float = Field(default=5.0, ge=0,
                                    description="Metric grading ratio R "
                                                "(refinement= to "
                                                "metric_density_from_gradient; "
                                                "0 disables adaptation).")
    adapt_every: int = Field(default=1, ge=1,
                             description="Adapt every N steps (1 = strongest "
                                         "mesh test).")
    adapt_start: int = Field(default=0, ge=0)
    mover_accel: Literal["cg", "heavy", "none"] = "cg"

    # Operational — free slip
    freeslip: Literal["penalty", "nitsche"] = "penalty"
    kfs: float = Field(default=1e6, gt=0,
                       description="Penalty free-slip stiffness.")
    nitsche_gamma: float = Field(default=10.0, gt=0)

    # Operational — adaptation / timestep
    skip_threshold: float = Field(default=99.0,
                                  description=">10 = never skip the move; "
                                              "~0.9 = skip when aligned.")
    grad_smooth_h0: float = Field(default=0.0, ge=0,
                                  description="gradient_smoothing_length as a "
                                              "multiple of mean h0 (0 = off).")
    dt_mult: float = Field(default=4.0, gt=0,
                           description="Multiplier on estimate_dt(p50).")
    cold_stokes: bool = False

    # Operational — run control / cadence
    max_steps: int = Field(default=80, gt=0)
    max_t: float = Field(default=0.06, ge=0,
                         description="Stop when sim time reaches this "
                                     "(0 = ignore).")
    save_every: int = Field(default=5, gt=0)

    # Steady-state detection (operational)
    steady_window: float = Field(default=0.3, gt=0, le=1.0)
    steady_tol_mean: float = Field(default=0.02, gt=0)
    steady_tol_cv: float = Field(default=0.05, gt=0)
    steady_min_window: int = Field(default=30, ge=5)

    # Output
    output_dir: str = "output/adaptive_convection/run"
    restart_policy: Literal["error", "fresh", "seed_from_old"] = "error"


# ---------------------------------------------------------------------------
# Identity / on-disk state
# ---------------------------------------------------------------------------
def _config_hash(config: AdaptiveConvectionConfig) -> str:
    return config_cache_key(config, _IDENTITY_FIELDS)


def _identity_snapshot(config: AdaptiveConvectionConfig) -> dict:
    return config_snapshot(config, _IDENTITY_FIELDS)


def _check_compatibility(config: AdaptiveConvectionConfig, output_dir: Path):
    """Compare current config to an existing manifest; archive on mismatch."""
    run = Run(output_dir)
    manifest = run.manifest
    if manifest is None:
        return "fresh", None
    if manifest.config_hash == _config_hash(config):
        return "continue", None
    snap = manifest.config_snapshot
    changed = [f"{f}: {snap.get(f)!r} -> {getattr(config, f)!r}"
               for f in _IDENTITY_FIELDS
               if snap.get(f) != getattr(config, f, None)]
    if config.restart_policy == "error":
        raise RuntimeError(
            f"Config mismatch in {output_dir}\n  Changed identity fields:\n    "
            + "\n    ".join(changed)
            + "\n  Set restart_policy='fresh' or 'seed_from_old' (both archive, "
            "never delete).")
    archive = run.archive()
    output_dir.mkdir(parents=True, exist_ok=True)
    return "fresh", archive


# Timeseries schema — uw.workflows owns 'step'/'t'/'dt'; the rest are ours.
_TS_FIELDS = ("step", "t", "dt", "wall", "Nu", "vrms", "Tmin", "Tmax",
              "adapted", "misalign", "folded", "area_ratio", "aspect",
              "min_area", "bl_ratio")


# ---------------------------------------------------------------------------
# Steady-state test (Nu + vrms stationary)
# ---------------------------------------------------------------------------
def _is_steady(timeseries: list[dict], config: AdaptiveConvectionConfig) -> bool:
    n = len(timeseries)
    n_window = max(int(n * config.steady_window), config.steady_min_window)
    if n < n_window:
        return False
    window = timeseries[-n_window:]
    nu = np.array([r["Nu"] for r in window])
    vrms = np.array([r["vrms"] for r in window])
    for series in (nu, vrms):
        s = series[np.isfinite(series)]
        if s.size < 4:
            continue
        half = len(s) // 2
        a, b = s[:half].mean(), s[half:].mean()
        mean_overall = max(abs(s.mean()), 1e-30)
        if abs(a - b) / mean_overall > config.steady_tol_mean:
            return False
        if s.std() / mean_overall > config.steady_tol_cv:
            return False
    return True


# ---------------------------------------------------------------------------
# Workflow steps — the DAG
# ---------------------------------------------------------------------------
@workflow_step(
    description="Build (or reload) the annulus mesh; apply restart_policy on mismatch",
    produces=["mesh"],
)
def create_mesh(config: AdaptiveConvectionConfig):
    import underworld3 as uw

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    action, archive = _check_compatibility(config, output_dir)
    mesh_h5 = output_dir / f"{RUN_NAME}.mesh.00000.h5"

    if action == "continue" and mesh_h5.exists():
        mesh = uw.discretisation.Mesh(str(mesh_h5), qdegree=config.qdegree)
    else:
        ref = config.refinement if config.refinement > 0 else None
        mesh = uw.meshing.Annulus(
            radiusOuter=config.r_outer, radiusInner=config.r_inner,
            cellSize=config.cellsize, qdegree=config.qdegree, refinement=ref)
    mesh._uw_run_archive = archive
    return mesh


@workflow_step(
    description="Build Stokes + AdvDiffusion solvers and their MeshVariables",
    produces=["stokes", "adv_diff", "T", "v", "p"],
    requires=["mesh"],
)
def create_solvers(mesh, config: AdaptiveConvectionConfig):
    """Isotropic Frank-Kamenetskii Stokes (free-slip outer, no-slip inner)
    + SLCN advection-diffusion.  Live objects — runner cache only."""
    import underworld3 as uw

    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=config.T_degree,
                                       continuous=True, varsymbol="T")
    v = uw.discretisation.MeshVariable("v", mesh, mesh.dim, degree=2,
                                       continuous=True, varsymbol=r"\mathbf{v}")
    p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1,
                                       continuous=True, varsymbol="p")

    X = mesh.CoordinateSystem.X
    r_sym = sympy.sqrt(X[0] ** 2 + X[1] ** 2)
    unit_r = mesh.CoordinateSystem.unit_e_0
    T_cond = sympy.log(r_sym / config.r_outer) / sympy.log(config.r_inner / config.r_outer)
    theta_FK = float(np.log(config.delta_eta))
    eta_FK = sympy.exp(theta_FK * (1 - T.sym[0]))

    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = eta_FK
    stokes.tolerance = 1.0e-5
    stokes.penalty = 0.0
    stokes.petsc_options["snes_type"] = "ksponly"   # linear ⇒ one KSP solve
    stokes.add_essential_bc((0.0, 0.0), mesh.boundaries.Lower.name)
    if config.freeslip == "nitsche":
        stokes.add_nitsche_bc(mesh.boundaries.Upper.name, gamma=config.nitsche_gamma)
    else:
        stokes.add_natural_bc(config.kfs * v.sym.dot(unit_r) * unit_r,
                              mesh.boundaries.Upper.name)
    stokes.bodyforce = config.rayleigh * (T.sym[0] - T_cond) * unit_r

    adv = uw.systems.AdvDiffusionSLCN(mesh, u_Field=T, V_fn=v.sym,
                                      theta=1.0, monotone_mode="clamp",
                                      verbose=False)
    adv.constitutive_model = uw.constitutive_models.DiffusionModel
    adv.constitutive_model.Parameters.diffusivity = config.diffusivity
    adv.tolerance = 1.0e-4
    adv.add_dirichlet_bc(1.0, mesh.boundaries.Lower.name)   # T=1 hot inner
    adv.add_dirichlet_bc(0.0, mesh.boundaries.Upper.name)   # T=0 cold outer

    return {"stokes": stokes, "adv_diff": adv, "T": T, "v": v, "p": p}


def _write_ic(T, mesh, config: AdaptiveConvectionConfig):
    """Conductive profile + initial perturbation.

    ``seed_kind='mode'`` (default) adds a global azimuthal-mode perturbation
    (plumes form everywhere). ``seed_kind='blob'`` adds a single localized
    gaussian hot anomaly at (``seed_theta_deg``, ``seed_depth``) so convection
    is seeded AWAY from a fault — the rising plume does not sit on it.
    """
    import underworld3 as uw
    X = mesh.CoordinateSystem.X
    r_sym = sympy.sqrt(X[0] ** 2 + X[1] ** 2)
    th_sym = sympy.atan2(X[1], X[0])
    r_i, r_o = config.r_inner, config.r_outer
    T_cond = sympy.log(r_sym / r_o) / sympy.log(r_i / r_o)
    if config.seed_kind == "blob":
        th0 = float(np.deg2rad(config.seed_theta_deg))
        r0 = r_i + config.seed_depth * (r_o - r_i)
        x0, y0 = r0 * np.cos(th0), r0 * np.sin(th0)
        blob = sympy.exp(-((X[0] - x0) ** 2 + (X[1] - y0) ** 2)
                         / config.seed_width ** 2)
        init = T_cond + config.pert_amplitude * blob
    else:
        init = (config.pert_amplitude * sympy.sin(config.pert_mode * th_sym)
                * sympy.sin(np.pi * (r_sym - r_i) / (r_o - r_i)) + T_cond)
    T.data[...] = np.asarray(uw.function.evaluate(init, T.coords)).reshape(-1, 1)


def _make_adapt_step(mesh, stokes, adv_diff, T, v, p, config):
    """Return an ``adapt()`` closure: build the |∇T| metric, move with mmpde,
    cold-restart the flow.  Returns (moved, misalignment)."""
    import underworld3 as uw

    accel = None if config.mover_accel == "none" else config.mover_accel

    def adapt():
        old_X = np.asarray(mesh.X.coords).copy()
        h0 = float(mesh._radii.mean()) if hasattr(mesh, "_radii") else None
        grad_L = (config.grad_smooth_h0 * h0
                  if (config.grad_smooth_h0 > 0 and h0) else None)
        sk = None if config.skip_threshold > 10.0 else config.skip_threshold

        rho = uw.meshing.metric_density_from_gradient(
            mesh, T, refinement=float(config.resolution_ratio),
            coarsening="auto", metric_choice="front-following", name="loop",
            gradient_smoothing_length=grad_L)
        mm = uw.meshing.mesh_metric_mismatch(
            mesh, rho, resolution_ratio=config.resolution_ratio)
        misalign = float(mm["misalignment"])

        uw.meshing.smooth_mesh_interior(
            mesh, metric=rho, method="mmpde",
            method_kwargs=dict(step_frac=0.2, accel=accel, momentum=0.0),
            slip_surfaces=True, skip_threshold=sk, verbose=False)

        if np.allclose(np.asarray(mesh.X.coords), old_X):
            return False, misalign
        v.data[...] = 0.0
        p.data[...] = 0.0
        return True, misalign

    return adapt


@workflow_step(
    description="Adaptive time loop until steady state or max_steps; append-only, resumable",
    produces=["run_directory", "evolution_log"],
    requires=["mesh", "stokes", "adv_diff", "T", "v", "p"],
)
def evolve(mesh, stokes, adv_diff, T, v, p, config: AdaptiveConvectionConfig):
    import underworld3 as uw

    output_dir = Path(config.output_dir)
    run = Run.create(output_dir)

    summary = run.summary
    if summary is not None and summary.get("status") == "steady":
        uw.pprint(f"[evolve] {output_dir}: steady (n_steps={summary.get('n_steps')}). "
                  "No work.")
        return {"run_directory": run, "evolution_log": run.timeseries}

    timeseries = run.timeseries
    saved_steps = run.steps
    nusselt = diag.NusseltSurface(mesh, T, mesh.boundaries.Upper.name,
                                  r_inner=config.r_inner, r_outer=config.r_outer)
    adapt = _make_adapt_step(mesh, stokes, adv_diff, T, v, p, config)
    do_adapt = config.resolution_ratio > 0

    if not saved_steps:
        archive = getattr(mesh, "_uw_run_archive", None)
        if archive is not None and config.restart_policy == "seed_from_old":
            old = Run(archive).steps
            if old:
                T.read_timestep(data_filename=RUN_NAME, data_name="T",
                                index=max(old), outputPath=str(archive))
            else:
                _write_ic(T, mesh, config)
        else:
            _write_ic(T, mesh, config)
        stokes.solve(zero_init_guess=True)
        run.write_manifest({
            "workflow": "adaptive_convection",
            "config_hash": _config_hash(config),
            "config_snapshot": _identity_snapshot(config),
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "rayleigh": config.rayleigh, "delta_eta": config.delta_eta,
        })
        q = diag.mesh_quality(mesh)
        nn = diag.nn_spacing_ratios(mesh, r_inner=config.r_inner, r_outer=config.r_outer)
        diag0 = dict(wall=0.0, Nu=nusselt(), vrms=diag.vrms(mesh, v),
                     Tmin=float(T.data.min()), Tmax=float(T.data.max()),
                     adapted=0, misalign=float("nan"), folded=q["folded"],
                     area_ratio=q["area_ratio"], aspect=q["aspect"],
                     min_area=q["min_area"], bl_ratio=nn["bl_ratio"])
        run.append_step(step=0, t=0.0, dt=0.0, mesh=mesh, mesh_vars=[v, T, p],
                        diags=diag0, fields=_TS_FIELDS, mesh_updates=True)
        timeseries.append({"step": 0, "t": 0.0, "dt": 0.0, **diag0})
        timestep, elapsed = 0, 0.0
        uw.pprint(f"[evolve] {output_dir} fresh start Ra={config.rayleigh:.0e} "
                  f"Δη={config.delta_eta:.0e} R={config.resolution_ratio}", flush=True)
    else:
        last = max(saved_steps)
        T.read_timestep(data_filename=RUN_NAME, data_name="T", index=last,
                        outputPath=str(output_dir))
        v.read_timestep(data_filename=RUN_NAME, data_name="v", index=last,
                        outputPath=str(output_dir))
        timestep = last
        elapsed = timeseries[-1]["t"] if timeseries else 0.0
        uw.pprint(f"[evolve] {output_dir} resume from step {last}", flush=True)

    while timestep < config.max_steps:
        t_step = time.perf_counter()
        did_adapt, misalign = False, float("nan")
        if (do_adapt and (timestep + 1) >= config.adapt_start
                and ((timestep + 1) % config.adapt_every == 0)):
            did_adapt, misalign = adapt()

        stokes.solve(zero_init_guess=(did_adapt or config.cold_stokes))
        dt = adv_diff.estimate_dt(direction_aware=True, percentile=50.0) \
            * float(config.dt_mult)
        adv_diff.solve(timestep=dt, zero_init_guess=False)
        timestep += 1
        elapsed += float(dt)
        wall = time.perf_counter() - t_step

        T_arr = T.data[:, 0]
        Tmin, Tmax = float(T_arr.min()), float(T_arr.max())
        q = diag.mesh_quality(mesh)
        nn = diag.nn_spacing_ratios(mesh, r_inner=config.r_inner, r_outer=config.r_outer)
        d = dict(wall=wall, Nu=nusselt(), vrms=diag.vrms(mesh, v),
                 Tmin=Tmin, Tmax=Tmax, adapted=int(did_adapt),
                 misalign=misalign, folded=q["folded"],
                 area_ratio=q["area_ratio"], aspect=q["aspect"],
                 min_area=q["min_area"], bl_ratio=nn["bl_ratio"])
        row = {"step": timestep, "t": elapsed, "dt": float(dt), **d}
        run.append_timeseries_row(row, _TS_FIELDS)
        timeseries.append(row)
        uw.pprint(
            f"[step {timestep:4d}] t={elapsed:.5f} dt={dt:.2e} {wall:5.1f}s "
            f"Nu={d['Nu']:+.3f} vrms={d['vrms']:.3e} T[{Tmin:+.3f},{Tmax:+.3f}] "
            f"fold={q['folded']} arat={q['area_ratio']:.1f} "
            f"{'ADAPT' if did_adapt else ''}", flush=True)

        if not np.isfinite(T_arr).all() or Tmax > 1.1 or Tmin < -0.1:
            uw.pprint(f"[evolve] step {timestep}: T out of range — ABORT", flush=True)
            break
        if timestep % config.save_every == 0:
            mesh.write_timestep(filename=RUN_NAME, index=timestep,
                                outputPath=str(output_dir), meshVars=[v, T, p], meshUpdates=True)
            if _is_steady(timeseries, config):
                uw.pprint(f"[evolve] steady at step {timestep}.", flush=True)
                break
        if config.max_t > 0 and elapsed >= config.max_t:
            uw.pprint(f"[evolve] reached max_t={config.max_t} at step {timestep}.",
                      flush=True)
            if timestep % config.save_every != 0:
                mesh.write_timestep(filename=RUN_NAME, index=timestep,
                                    outputPath=str(output_dir), meshVars=[v, T, p], meshUpdates=True)
            break

    return {"run_directory": run, "evolution_log": timeseries}


@workflow_step(
    description="Average the steady window and write run_summary.yaml",
    produces=["run_summary"],
    requires=["run_directory"],
)
def summarise_run(run_directory: Run, config: AdaptiveConvectionConfig):
    cached = run_directory.summary
    if cached is not None and cached.get("status") == "steady":
        return cached
    log = run_directory.timeseries
    last_step = log[-1]["step"] if log else 0
    if _is_steady(log, config):
        status = "steady"
    elif last_step >= config.max_steps:
        status = "stalled"
    else:
        status = "incomplete"
    if not log:
        return {"status": status, "n_steps": 0}

    n_window = max(int(len(log) * config.steady_window),
                   min(config.steady_min_window, len(log)))
    window = log[-n_window:]
    nu = np.array([r["Nu"] for r in window])
    vr = np.array([r["vrms"] for r in window])
    fold = np.array([r["folded"] for r in window])
    arat = np.array([r["area_ratio"] for r in window])
    summary = {
        "status": status, "n_steps": last_step,
        "rayleigh": config.rayleigh, "delta_eta": config.delta_eta,
        "resolution_ratio": config.resolution_ratio,
        "Nu_mean": float(np.nanmean(nu)), "Nu_std": float(np.nanstd(nu)),
        "vrms_mean": float(np.nanmean(vr)), "vrms_std": float(np.nanstd(vr)),
        "max_folded": int(np.nanmax(fold)),
        "area_ratio_mean": float(np.nanmean(arat)),
        "elapsed_t": float(window[-1]["t"]),
    }
    if status == "steady":
        run_directory.write_summary(summary)
    return summary
