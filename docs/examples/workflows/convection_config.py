"""Thermal convection workflow — idempotent, resumable, steady-state-driven.

Each invocation of ``runner.build("run_summary")`` either:

* short-circuits if ``run_summary.yaml`` already records ``status="steady"``
  for this output directory; or
* extends the run from the latest h5 checkpoint until either the steady-state
  test passes (writes the summary) or ``config.max_steps`` is reached
  (returns the timeseries; no summary written so a future call with a larger
  ``max_steps`` resumes seamlessly).

The run never destroys data.  Only ``timeseries.csv`` is appended to and only
new h5 timesteps are written.  When config fields that affect the mesh or
physics change, the existing directory is **archived** (``run_summary`` policy
``"fresh"`` or ``"seed_from_old"``) or the run **fails fast** (default ``"error"``).

Output layout per run::

    <output_dir>/
        manifest.yaml               run state + config snapshot + identity hash
        run.mesh.00000.h5           mesh, written once
        run.mesh.<var>.NNNNN.h5     per-saved-step variable values
        run.mesh.NNNNN.xdmf         ParaView metadata
        timeseries.csv              one row per saved step
                                      (step, t, dt, Nu_top, Nu_bot, Vrms, mean_T)
        run_summary.yaml            ONLY when status == "steady"

The summary file is the "done" marker — its presence makes the runner
skip work.  Stalled runs (hit ``max_steps`` without converging) do
**not** write the summary, so simply re-running with a higher
``max_steps`` resumes from the latest checkpoint and extends.  Remove
``run_summary.yaml`` to force a re-evaluation of an already-steady run
(or call ``runner.rebuild("run_summary")``).
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import sympy
from pydantic import Field

from underworld3.workflows import (  # noqa: F401  (RUN_NAME re-exported for viz)
    RUN_NAME,
    Run,
    WorkflowConfig,
    config_cache_key,
    config_snapshot,
    workflow_step,
)

# Fields whose change invalidates the existing on-disk run.
_IDENTITY_FIELDS = (
    "aspect_ratio", "cellsize", "qdegree", "regular",
    "T_degree",
    "rayleigh", "viscosity", "diffusivity",
    "T_top", "T_bottom",
)


def view():
    """Display the workflow steps and config classes in this module."""
    import sys
    from underworld3.workflows import view as _wf_view

    _wf_view(sys.modules[__name__])


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ConvectionConfig(WorkflowConfig):
    """Parameters for steady-state-driven Rayleigh-Bénard convection.

    Identity (mesh + physics) fields are hashed to detect config changes
    against an existing on-disk run.  Operational fields (steady-state
    tolerances, batch sizing) can change between invocations without
    triggering a mismatch.
    """

    workflow_name: str = "rayleigh_benard"
    description: str = (
        "Steady-state-driven, resumable Rayleigh-Bénard convection in a "
        "Cartesian box"
    )

    # Identity fields — mesh + discretisation + physics.  Operational
    # fields (steady tolerances, step caps, save cadence) are excluded
    # so they can change between invocations without invalidating
    # cached products.  Mirror of the module-level ``_IDENTITY_FIELDS``
    # tuple defined above.
    _identity_fields = _IDENTITY_FIELDS

    # Identity — mesh
    cellsize: float = Field(default=1.0 / 16, gt=0)
    aspect_ratio: float = Field(default=1.0, gt=0)
    qdegree: int = Field(default=3, ge=1)
    regular: bool = False

    # Identity — discretisation orders
    # ``T_degree`` controls the polynomial order of the temperature
    # MeshVariable (degree=3 by default, matching the previous hardcode).
    # Changing it forces a fresh run because the on-disk h5 layout and
    # the JIT cache key both depend on the FE space.
    T_degree: int = Field(default=3, ge=1)

    # Identity — physics
    rayleigh: float = Field(default=1e6, gt=0)
    viscosity: float = Field(default=1.0, gt=0)
    diffusivity: float = Field(default=1.0, gt=0)
    # NOTE: AdvDiffusionSLCN exhibits a monotonic amplitude-loss
    # artefact (the solution drifts toward zero everywhere over
    # time).  With T ∈ [0, 1] the physical mean is 0.5 and the drift
    # shows up as a downward trend in <T>.  With T ∈ [-0.5, 0.5] the
    # physical mean is 0 and the drift is invisible (drift toward
    # zero == drift toward the mean).  ΔT = 1 either way, so Ra and
    # Nu are unchanged — but for clean steady-state diagnostics the
    # symmetric scaling is preferred.
    T_top: float = 0.0
    T_bottom: float = 1.0

    # Steady-state detection (operational; can change without mismatch)
    steady_window: float = Field(default=0.3, gt=0, le=1.0)
    steady_tol_mean: float = Field(default=0.02, gt=0)
    steady_tol_cv: float = Field(default=0.05, gt=0)
    steady_min_window: int = Field(default=50, ge=10)

    # Run extension (operational)
    batch_steps: int = Field(default=200, gt=0)
    max_steps: int = Field(default=5000, gt=0)
    save_every: int = Field(default=10, gt=0)
    dt_factor: float = Field(default=2.0, gt=0)
    # Diagnostic cadence: how often to do the heavy volume integrals
    # (Vrms, mean_T).  0 = match save_every; 1 = every step (low-Ra
    # quick runs); >1 = every N steps.  Nu/Vmax always run every step.
    diag_every: int = Field(default=0, ge=0)
    # Optional: clip T to [T_top, T_bottom] after each AdvDiff solve.
    # Helpful as a safety belt during the violent early transient at
    # high Ra — without this, AdvDiff overshoots can drive T outside
    # the physical range, which pumps spurious buoyancy and crashes
    # the Stokes solve.  False by default (don't change converged
    # solutions); set True for warm-start / high-Ra cold-start runs.
    clip_T_range: bool = False

    # Output
    output_dir: str = "output/convection/run"
    restart_policy: Literal["error", "fresh", "seed_from_old"] = "error"


# ---------------------------------------------------------------------------
# Identity / on-disk state
# ---------------------------------------------------------------------------


def _config_hash(config: ConvectionConfig) -> str:
    """16-char hex cache key over the identity fields."""
    return config_cache_key(config, _IDENTITY_FIELDS)


def _identity_snapshot(config: ConvectionConfig) -> dict:
    return config_snapshot(config, _IDENTITY_FIELDS)


def _check_compatibility(config: ConvectionConfig, output_dir: Path):
    """Compare current config to an existing manifest.

    Returns
    -------
    action : str
        ``"continue"`` — config matches; reuse the directory.
        ``"fresh"`` — directory was archived (or didn't exist); start clean.
    archive : Path or None
        Path to the archived old directory, if any.
    """
    run = Run(output_dir)
    manifest = run.manifest
    if manifest is None:
        return "fresh", None

    if manifest.config_hash == _config_hash(config):
        return "continue", None

    snap = manifest.config_snapshot
    changed = [
        f"{f}: {snap.get(f)!r} -> {getattr(config, f)!r}"
        for f in _IDENTITY_FIELDS
        if snap.get(f) != getattr(config, f, None)
    ]

    if config.restart_policy == "error":
        raise RuntimeError(
            f"Config mismatch in {output_dir}\n"
            f"  Changed identity fields:\n    "
            + "\n    ".join(changed)
            + "\n  Set config.restart_policy='fresh' (build new) or "
            "'seed_from_old' (use old T as IC).\n"
            "  Both options archive the existing directory rather than "
            "delete it."
        )

    archive = run.archive()
    output_dir.mkdir(parents=True, exist_ok=True)
    return "fresh", archive


# ---------------------------------------------------------------------------
# Timeseries schema (workflow-specific column order)
# ---------------------------------------------------------------------------


_TS_FIELDS = ("step", "t", "dt", "Nu_top", "Nu_bot", "Vrms", "Vmax", "mean_T")


# ---------------------------------------------------------------------------
# Steady-state test
# ---------------------------------------------------------------------------


def _is_steady(timeseries: list[dict], config: ConvectionConfig) -> bool:
    """Pass when both Nu(t) and Vrms(t) are statistically stationary.

    Tested over the last ``steady_window`` fraction of points (with a
    floor of ``steady_min_window`` samples).  Both the change in mean
    between halves and the coefficient of variation must be below the
    configured tolerances.

    Vrms may be NaN on rows where the volume integrals were skipped
    for cost; those rows are dropped from the Vrms test (which then
    only sees the checkpoint-cadence rows).  Nu is computed every
    step so its test sees every row.
    """
    n = len(timeseries)
    n_window = max(int(n * config.steady_window), config.steady_min_window)
    if n < n_window:
        return False

    window = timeseries[-n_window:]
    nu = np.array([(r["Nu_top"] + r["Nu_bot"]) / 2 for r in window])
    vrms_raw = np.array([r["Vrms"] for r in window])
    vrms = vrms_raw[np.isfinite(vrms_raw)]

    series_to_test = [nu]
    if vrms.size >= 4:
        series_to_test.append(vrms)

    for series in series_to_test:
        half = len(series) // 2
        a, b = series[:half].mean(), series[half:].mean()
        mean_overall = max(abs(series.mean()), 1e-30)
        if abs(a - b) / mean_overall > config.steady_tol_mean:
            return False
        if series.std() / mean_overall > config.steady_tol_cv:
            return False
    return True


# ---------------------------------------------------------------------------
# Diagnostics (Nu, Vrms, <T>)
# ---------------------------------------------------------------------------


# Module-level cache for the diagnostic objects.  Keyed by id(mesh) so
# different runs in the same process don't collide.  Each entry holds
# the Integral and flux SymPy expression built once for that mesh; we
# reuse the SAME Python objects across every diagnostic call so the JIT
# cache hits instead of recompiling on every step.
_DIAG_CACHE: dict = {}


def _diag_state(mesh, T, v, config: ConvectionConfig) -> dict:
    """Build (or fetch) the cached diagnostic helpers for this mesh.

    The heavy diagnostic — the boundary heat flux — is computed via a
    one-shot ``Projection`` of the (conductive + advective) flux
    expression onto a scalar MeshVariable.  The Projection is created
    once and warm-started from the previous step's solution (the flux
    field varies smoothly between steps, so warm-start drives the
    iteration count down to 1–2 quickly).

    Volume integrals use **one Integral per integrand**, kept alive
    across calls — re-assigning ``.fn`` on a single Integral object
    appears to invalidate UW3's compile cache, so per-call cost was
    creeping linearly with step count.  Mesh volume is computed once
    and cached.
    """
    import underworld3 as uw

    key = id(mesh)
    state = _DIAG_CACHE.get(key)
    if state is not None:
        return state

    # Mesh volume — constant; compute once.
    integrator_vol = uw.maths.Integral(mesh, 1)
    V_total = float(integrator_vol.evaluate())

    # One Integral per integrand so its compiled form stays valid.
    integrator_meanT = uw.maths.Integral(mesh, T.sym[0])
    integrator_v2 = uw.maths.Integral(mesh, v.sym.dot(v.sym))

    flux_fn = (
        -mesh.vector.gradient(T.sym[0]).dot(mesh.CoordinateSystem.unit_e_1)
        + v.sym[1] * T.sym[0]
    )

    # Heat-flux Projection — one solve per diagnostic call, warm-started.
    # Fast PETSc settings (cg + jacobi at tol=1e-4) per the bench in
    # /tmp/projection_bench.py.
    flux_var = uw.discretisation.MeshVariable(
        r"q_y", mesh, 1, degree=1,
        varsymbol=r"q_y",
    )
    flux_proj = uw.systems.Projection(mesh, flux_var)
    flux_proj.uw_function = flux_fn
    flux_proj.smoothing = 0.0
    # Settings per Issue #156: skip SNES (linear projection) and use
    # CG + bjacobi with relative tolerance.  ksponly avoids the
    # outer Newton loop entirely; bjacobi is robust on multi-rank.
    flux_proj.petsc_options["snes_type"] = "ksponly"
    flux_proj.petsc_options["ksp_type"] = "cg"
    flux_proj.petsc_options["pc_type"] = "bjacobi"
    flux_proj.petsc_options["ksp_rtol"] = 1.0e-4
    flux_proj.tolerance = 1.0e-4

    n_samples = 100
    top_pts = np.column_stack([
        np.linspace(0.0, config.aspect_ratio, n_samples),
        np.full(n_samples, 1.0),
    ])
    bot_pts = np.column_stack([
        np.linspace(0.0, config.aspect_ratio, n_samples),
        np.full(n_samples, 0.0),
    ])
    state = {
        "V_total": V_total,
        "integrator_meanT": integrator_meanT,
        "integrator_v2": integrator_v2,
        "flux_proj": flux_proj,
        "flux_var": flux_var,
        "top_pts": top_pts,
        "bot_pts": bot_pts,
        "first_call": True,
    }
    _DIAG_CACHE[key] = state
    return state


def _compute_diagnostics(
    mesh, T, v, config: ConvectionConfig,
    timings=None, skip_volume: bool = False,
) -> dict:
    """Compute Nu_top, Nu_bot, Vrms, Vmax, mean_T.

    If ``skip_volume`` is True, the volume integrals (Vrms, mean_T)
    are skipped and returned as ``float('nan')``.  Use this for cheap
    per-step logging when the volume integrals would otherwise
    dominate (their compile cache appears to grow with each call,
    inflating their cost over a long run).

    If ``timings`` is a dict, it is filled with per-stage wall times
    under keys ``vol`` (volume integrals), ``vmax`` (peak-v reduce)
    and ``flux`` (heat-flux Projection + sample).
    """
    import time as _time
    import underworld3 as uw

    state = _diag_state(mesh, T, v, config)
    V_total = state["V_total"]

    # ── Volume integrals (mean_T, Vrms) ───────────────────────────
    t0 = _time.perf_counter()
    if skip_volume:
        mean_T = float("nan")
        Vrms = float("nan")
    else:
        mean_T = float(state["integrator_meanT"].evaluate()) / V_total
        Vrms = float(np.sqrt(max(state["integrator_v2"].evaluate(), 0.0) / V_total))
    t_vol = _time.perf_counter() - t0

    # ── Vmax (cheap) ──────────────────────────────────────────────
    t0 = _time.perf_counter()
    v_dofs = np.asarray(v.array).reshape(-1, mesh.dim)
    v_mag = np.linalg.norm(v_dofs, axis=1)
    local_max = float(v_mag.max()) if v_mag.size else 0.0
    if uw.mpi.size > 1:
        import mpi4py
        Vmax = float(uw.mpi.comm.allreduce(local_max, op=mpi4py.MPI.MAX))
    else:
        Vmax = local_max
    t_vmax = _time.perf_counter() - t0

    # ── Heat-flux Projection (warm-started after first call) ─────
    # Project the (conductive + advective) flux expression onto a
    # smooth MeshVariable, then sample at the boundary points.  The
    # Projection is the same Python object every step, so its compiled
    # form is reused and the previous solution is the initial guess
    # for the next solve.
    t0 = _time.perf_counter()
    flux_proj = state["flux_proj"]
    flux_var = state["flux_var"]
    flux_proj.solve(zero_init_guess=state["first_call"])
    state["first_call"] = False
    flux_top = float(np.mean(
        uw.function.evaluate(flux_var.sym[0], state["top_pts"])
    ))
    flux_bot = float(np.mean(
        uw.function.evaluate(flux_var.sym[0], state["bot_pts"])
    ))
    t_flux = _time.perf_counter() - t0

    delta_T = config.T_bottom - config.T_top  # H = 1
    Nu_top = flux_top / max(abs(delta_T), 1e-30)
    Nu_bot = flux_bot / max(abs(delta_T), 1e-30)

    if timings is not None:
        timings["vol"] = t_vol
        timings["vmax"] = t_vmax
        timings["flux"] = t_flux

    return {
        "Nu_top": Nu_top, "Nu_bot": Nu_bot,
        "Vrms": Vrms, "Vmax": Vmax,
        "mean_T": mean_T,
    }


# ---------------------------------------------------------------------------
# Workflow steps — the DAG
# ---------------------------------------------------------------------------


@workflow_step(
    description="Predict thermal boundary-layer thickness from Ra (high-Ra 2D scaling)",
    produces=["bl_thickness"],
)
def predict_bl_thickness(config: ConvectionConfig):
    """Boundary-layer thickness from Nu ≈ 0.27·Ra^(1/3); δ ≈ 1/(2·Nu).

    Used quietly by ``set_initial_temperature`` to seed a fast-onset IC.
    """
    Nu_pred = max(1.0, 0.27 * config.rayleigh ** (1.0 / 3.0))
    return float(1.0 / (2.0 * Nu_pred))


@workflow_step(
    description="Predict mean temperature from BL theory (= 0.5 for symmetric BLs)",
    produces=["mean_T_predicted"],
)
def predict_mean_T(config: ConvectionConfig):
    return 0.5 * (config.T_top + config.T_bottom)


@workflow_step(
    description="Build (or reload) the simulation mesh; apply restart_policy on mismatch",
    produces=["mesh"],
)
def create_mesh(config: ConvectionConfig):
    """Build a 2D unstructured simplex mesh.

    Three branches:

    1. Output dir is empty (no manifest)              -> build fresh.
    2. Manifest matches current identity hash         -> reload mesh from h5.
    3. Manifest mismatches identity hash              -> archive old dir per
       ``restart_policy`` (or raise on policy ``"error"``); build fresh.

    The path of any archived directory is stashed on the returned mesh
    object as ``mesh._uw_run_archive`` for downstream steps that may
    want to seed from it.
    """
    import underworld3 as uw

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    action, archive = _check_compatibility(config, output_dir)
    mesh_h5 = output_dir / f"{RUN_NAME}.mesh.00000.h5"

    if action == "continue" and mesh_h5.exists():
        # Pass qdegree on reload so high-order T (degree=3) is fully
        # integrated — without this, Mesh defaults qdegree=2 and the
        # solvers warn about under-integration on every solve.
        mesh = uw.discretisation.Mesh(
            str(mesh_h5),
            qdegree=config.qdegree,
            coordinate_system_type=uw.coordinates.CoordinateSystemType.CARTESIAN,
        )
    else:
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0, 0.0),
            maxCoords=(config.aspect_ratio, 1.0),
            cellSize=config.cellsize,
            regular=config.regular,
            qdegree=config.qdegree,
        )

    mesh._uw_run_archive = archive
    return mesh


@workflow_step(
    description="Build Stokes + AdvDiffusion solvers and their MeshVariables",
    produces=["stokes", "adv_diff", "T", "v", "p"],
    requires=["mesh"],
)
def create_solvers(mesh, config: ConvectionConfig):
    """Set up the two coupled solvers with free-slip walls and isothermal BCs.

    Returns a dict keyed by produces.  These are live (non-persistable)
    objects; they live in the runner cache only and are rebuilt each
    session.
    """
    import underworld3 as uw

    v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=config.T_degree)

    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = config.viscosity
    stokes.tolerance = 1.0e-6
    stokes.penalty = 0.0
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Bottom")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Top")
    stokes.add_dirichlet_bc((0.0, sympy.oo), "Left")
    stokes.add_dirichlet_bc((0.0, sympy.oo), "Right")
    stokes.bodyforce = sympy.Matrix([0, config.rayleigh * T.sym[0]])

    # Multigrid PETSc options for the Stokes velocity / pressure
    # subblocks.  Lifted from the PHYS-3070 convection setup — these
    # cut Stokes-solve iterations dramatically at high Ra (where the
    # default solver clogs as the velocity field gets chaotic).
    stokes.petsc_options["snes_type"] = "newtonls"
    stokes.petsc_options["ksp_type"] = "fgmres"
    stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_type", "kaskade")
    stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_cycle_type", "w")
    stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"
    stokes.petsc_options["fieldsplit_velocity_ksp_type"] = "fcg"
    stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_type"] = "chebyshev"
    stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_max_it"] = 7
    stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_converged_maxits"] = None
    stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "gamg")
    stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "additive")
    stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")

    adv_diff = uw.systems.AdvDiffusionSLCN(mesh, u_Field=T, V_fn=v)
    adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel
    adv_diff.constitutive_model.Parameters.diffusivity = config.diffusivity
    adv_diff.theta = 0.5
    adv_diff.add_dirichlet_bc(config.T_bottom, "Bottom")
    adv_diff.add_dirichlet_bc(config.T_top, "Top")

    return {"stokes": stokes, "adv_diff": adv_diff, "T": T, "v": v, "p": p}


@workflow_step(
    description="Idempotent, resumable evolve until steady state or max_steps",
    produces=["evolution_log"],
    requires=["mesh", "stokes", "adv_diff", "T", "v", "bl_thickness"],
)
def evolve(mesh, stokes, adv_diff, T, v, bl_thickness, config: ConvectionConfig):
    """Drive the time loop, append-only, until steady state or step cap.

    Behaviour:

    * Reads ``run_summary.yaml`` first — only ``status == "steady"``
      short-circuits.  Stalled runs leave no summary, so re-invoking
      with a larger ``max_steps`` resumes and extends rather than
      treating the run as finished.
    * Otherwise discovers the latest h5 step in ``output_dir`` and either
      starts fresh (writing mesh + IC at step 0) or resumes by reading
      ``T`` and ``v`` back from the latest checkpoint.
    * Runs solver steps until the steady-state test passes or
      ``config.max_steps`` is reached.  Every ``save_every`` steps, the
      state is written to h5 *and* a timeseries row is appended to
      ``timeseries.csv``.

    Per-step diagnostics are printed (every save) so a long run can be
    interrupted with the partial chain intact.
    """
    import underworld3 as uw

    output_dir = Path(config.output_dir)
    run = Run.create(output_dir)

    # ── Short-circuit on a *steady* summary only ─────────────────────
    # ``stalled`` summaries are not written (so the user can extend with
    # a larger max_steps and resume).  Only ``status == "steady"`` is the
    # irrevocable done marker.
    summary = run.summary
    if summary is not None and summary.get("status") == "steady":
        uw.pprint(
            f"[evolve] {output_dir}: status='steady', "
            f"n_steps={summary.get('n_steps')}.  No work to do."
        )
        return run.timeseries

    timeseries = run.timeseries
    saved_steps = run.steps

    if not saved_steps:
        # ── Fresh start ─────────────────────────────────────────────
        archive = getattr(mesh, "_uw_run_archive", None)

        if archive is not None and config.restart_policy == "seed_from_old":
            old_steps = Run(archive).steps
            if old_steps:
                uw.pprint(
                    f"[evolve] Seeding T from archived run at {archive} "
                    f"(step {max(old_steps)})"
                )
                T.read_timestep(
                    data_filename=RUN_NAME, data_name="T",
                    index=max(old_steps), outputPath=str(archive),
                )
            else:
                uw.pprint(
                    f"[evolve] seed_from_old requested but no checkpoints "
                    f"found in {archive}; falling back to BL-scaled IC."
                )
                _write_bl_ic(T, mesh, bl_thickness, config)
        else:
            _write_bl_ic(T, mesh, bl_thickness, config)

        # Initial Stokes solve
        stokes.solve(zero_init_guess=True)

        run.write_manifest({
            "workflow": "rayleigh_benard",
            "config_hash": _config_hash(config),
            "config_snapshot": _identity_snapshot(config),
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "rayleigh": config.rayleigh,
            "aspect_ratio": config.aspect_ratio,
        })

        mesh.write_timestep(
            filename=RUN_NAME, index=0, outputPath=str(output_dir),
            meshVars=[v, T],
        )
        diag = _compute_diagnostics(mesh, T, v, config)
        row = {"step": 0, "t": 0.0, "dt": 0.0, **diag}
        run.append_timeseries_row(row, _TS_FIELDS)
        timeseries.append(row)
        timestep = 0
        elapsed = 0.0

        uw.pprint(
            f"[evolve] {output_dir} fresh start  Ra={config.rayleigh:.0e}  "
            f"aspect={config.aspect_ratio}  delta_BL={bl_thickness:.3f}",
            flush=True,
        )
    else:
        # ── Resume ──────────────────────────────────────────────────
        last = max(saved_steps)
        T.read_timestep(
            data_filename=RUN_NAME, data_name="T", index=last,
            outputPath=str(output_dir),
        )
        v.read_timestep(
            data_filename=RUN_NAME, data_name="U", index=last,
            outputPath=str(output_dir),
        )
        timestep = last
        elapsed = timeseries[-1]["t"] if timeseries else 0.0

        uw.pprint(
            f"[evolve] {output_dir} resume from step {last}  "
            f"({len(timeseries)} timeseries rows)",
            flush=True,
        )

    # ── Time loop ────────────────────────────────────────────────────
    # Per-step: compute lightweight diagnostics, append to timeseries.csv,
    #           print the live one-line summary.
    # Per-save_every: also write an h5 checkpoint.  Steady-state test runs
    #           after each save (no point checking more often than the
    #           CSV grows by save_every rows).
    import time as _time
    while timestep < config.max_steps:
        t0 = _time.perf_counter()
        stokes.solve(zero_init_guess=False)
        t_stokes = _time.perf_counter() - t0

        delta_t = config.dt_factor * adv_diff.estimate_dt()

        t0 = _time.perf_counter()
        adv_diff.solve(timestep=delta_t, zero_init_guess=False)
        if config.clip_T_range:
            T.array[...] = np.clip(T.array, config.T_top, config.T_bottom)
        t_adv = _time.perf_counter() - t0

        timestep += 1
        elapsed += float(delta_t)

        # Heavy volume integrals (Vrms, mean_T) cadence is controlled
        # by config.diag_every (0 = match save_every).  Nu / Vmax
        # always run every step.
        diag_period = config.diag_every if config.diag_every > 0 else config.save_every
        skip_vol = (timestep % diag_period) != 0
        diag_timings: dict = {}
        diag = _compute_diagnostics(
            mesh, T, v, config,
            timings=diag_timings, skip_volume=skip_vol,
        )
        row = {
            "step": timestep, "t": elapsed, "dt": float(delta_t), **diag,
        }
        run.append_timeseries_row(row, _TS_FIELDS)
        timeseries.append(row)

        def _fmt(val, spec):
            if not np.isfinite(val):
                return "---"
            return f"{val:{spec}}"
        uw.pprint(
            f"[step {timestep:5d}]  dt={delta_t:.2e}  "
            f"Nu_top={diag['Nu_top']:6.3f}  Nu_bot={diag['Nu_bot']:6.3f}  "
            f"Vrms={_fmt(diag['Vrms'], '8.4f'):>8}  "
            f"Vmax={diag['Vmax']:8.4f}  "
            f"<T>={_fmt(diag['mean_T'], '5.3f'):>5}  | "
            f"t_stokes={t_stokes:5.2f}s  t_adv={t_adv:5.2f}s  "
            f"t_flux={diag_timings.get('flux', 0.0):4.2f}s  "
            f"t_vol={diag_timings.get('vol', 0.0):4.2f}s",
            flush=True,
        )

        if timestep % config.save_every == 0:
            mesh.write_timestep(
                filename=RUN_NAME, index=timestep, outputPath=str(output_dir),
                meshVars=[v, T],
            )
            if _is_steady(timeseries, config):
                uw.pprint(
                    f"[evolve] Steady state reached at step {timestep}.",
                    flush=True,
                )
                break

    return timeseries


def _write_bl_ic(T, mesh, bl_thickness: float, config: ConvectionConfig):
    """Set ``T`` from a boundary-layer-scaled initial condition.

    ``T(x, y) = 0.5 - 0.5 [tanh(2y/δ) - tanh(2(1-y)/δ)] + small perturbation``

    The mean is ≈ 0.5 (= predicted mean from BL symmetry) and the
    interior is isothermal, with top and bottom thermal BLs of
    thickness ≈ δ.

    The IC δ is floored at a safe multiple of the mesh cell size so
    the tanh profile is always representable on the mesh.  Without
    this, predicted δ at high Ra can be thinner than the cell size
    (e.g. Ra=1e5 → δ≈0.04 vs cellsize=1/24≈0.042), producing a
    near-singular IC that triggers numerical blow-up before the
    system has had a chance to spread the BLs through advection.
    The thicker-than-predicted IC is no worse than starting from a
    pure conduction profile — both relax toward the same steady
    state — but it skips the conductive transient *safely*.
    """
    import underworld3 as uw

    safe_delta = max(bl_thickness, 4.0 * config.cellsize, 1.0e-3)

    x, y = mesh.X
    base = 0.5 - 0.5 * (
        sympy.tanh(2 * y / safe_delta) - sympy.tanh(2 * (1 - y) / safe_delta)
    )
    perturb = (
        0.02 * sympy.cos(0.5 * sympy.pi * x / config.aspect_ratio)
        * sympy.sin(2 * sympy.pi * y)
        + 0.005 * sympy.cos(2 * sympy.pi * x / config.aspect_ratio)
        * sympy.sin(sympy.pi * y)
    )
    init = (config.T_bottom - config.T_top) * base + config.T_top + perturb
    T.array[...] = uw.function.evaluate(init, T.coords).reshape(T.array.shape)


@workflow_step(
    description="Compute steady-state averages and write run_summary.yaml",
    produces=["run_summary"],
    requires=["evolution_log"],
)
def summarise_run(evolution_log, config: ConvectionConfig):
    """Average the steady-window of the timeseries and write the summary.

    The summary is the "done" marker: its presence with status
    ``"steady"`` or ``"stalled"`` makes ``evolve`` short-circuit.  When
    the run hasn't yet reached steady state and isn't yet at
    ``max_steps``, no summary is written (the run is just paused).
    """
    run = Run(Path(config.output_dir))

    cached = run.summary
    if cached is not None and cached.get("status") == "steady":
        return cached

    is_steady = _is_steady(evolution_log, config)
    last_step = evolution_log[-1]["step"] if evolution_log else 0

    if is_steady:
        status = "steady"
    elif last_step >= config.max_steps:
        status = "stalled"  # hit cap; bump max_steps to extend
    else:
        status = "incomplete"  # paused mid-batch; just rerun

    if not evolution_log:
        return {"status": status, "n_steps": 0}

    n_window = max(
        int(len(evolution_log) * config.steady_window),
        min(config.steady_min_window, len(evolution_log)),
    )
    window = evolution_log[-n_window:]

    nu_window = np.array([(r["Nu_top"] + r["Nu_bot"]) / 2 for r in window])
    vrms_window = np.array([r["Vrms"] for r in window])
    meanT_window = np.array([r["mean_T"] for r in window])

    # Vrms / mean_T are NaN on rows where the volume integrals were
    # skipped for cost.  Use nanmean / nanstd so the summary still has
    # meaningful values when most rows are NaN.  Nu is dense (every
    # row), so plain mean/std is fine for it.
    summary = {
        "status": status,
        "n_steps": last_step,
        "rayleigh": config.rayleigh,
        "aspect_ratio": config.aspect_ratio,
        "Nu_top_mean": float(np.mean([r["Nu_top"] for r in window])),
        "Nu_bot_mean": float(np.mean([r["Nu_bot"] for r in window])),
        "Nu_mean": float(nu_window.mean()),
        "Nu_std": float(nu_window.std()),
        "Vrms_mean": float(np.nanmean(vrms_window)),
        "Vrms_std": float(np.nanstd(vrms_window)),
        "mean_T_mean": float(np.nanmean(meanT_window)),
        "elapsed_t": float(window[-1]["t"]),
    }

    # Only persist the summary when the run reached steady state.
    # Stalled runs leave no done-marker so a re-invocation with a higher
    # ``max_steps`` extends them rather than short-circuiting.
    if status == "steady":
        run.write_summary(summary)

    return summary
