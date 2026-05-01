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

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import sympy
import yaml
from pydantic import Field

from underworld3.workflows import WorkflowConfig, workflow_step


# Filename stem used inside each run's output directory.
RUN_NAME = "run"

# Fields whose change invalidates the existing on-disk run.
_IDENTITY_FIELDS = (
    "aspect_ratio", "cellsize", "qdegree", "regular",
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

    # Identity — mesh
    cellsize: float = Field(default=1.0 / 16, gt=0)
    aspect_ratio: float = Field(default=1.0, gt=0)
    qdegree: int = Field(default=3, ge=1)
    regular: bool = False

    # Identity — physics
    rayleigh: float = Field(default=1e6, gt=0)
    viscosity: float = Field(default=1.0, gt=0)
    diffusivity: float = Field(default=1.0, gt=0)
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

    # Output
    output_dir: str = "output/convection/run"
    restart_policy: Literal["error", "fresh", "seed_from_old"] = "error"


# ---------------------------------------------------------------------------
# Identity / on-disk state
# ---------------------------------------------------------------------------


def _config_hash(config: ConvectionConfig) -> str:
    """16-char hex hash of identity fields only."""
    fields = {f: getattr(config, f) for f in _IDENTITY_FIELDS}
    s = json.dumps(fields, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def _identity_snapshot(config: ConvectionConfig) -> dict:
    return {f: getattr(config, f) for f in _IDENTITY_FIELDS}


def _read_manifest(output_dir: Path) -> Optional[dict]:
    p = output_dir / "manifest.yaml"
    if not p.exists():
        return None
    with open(p) as f:
        return yaml.safe_load(f)


def _write_manifest(output_dir: Path, data: dict) -> None:
    p = output_dir / "manifest.yaml"
    with open(p, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def _archive_run(output_dir: Path) -> Optional[Path]:
    """Move ``output_dir`` to ``output_dir.archive-<UTC-stamp>/``.

    Never deletes.  Returns the archive path, or ``None`` if there was
    nothing to archive.
    """
    if not output_dir.exists():
        return None
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = output_dir.parent / f"{output_dir.name}.archive-{ts}"
    output_dir.rename(archive)
    return archive


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
    manifest = _read_manifest(output_dir)
    if manifest is None:
        return "fresh", None

    if manifest.get("config_hash") == _config_hash(config):
        return "continue", None

    snap = manifest.get("config_snapshot", {})
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

    archive = _archive_run(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return "fresh", archive


# ---------------------------------------------------------------------------
# Timeseries on disk (append-only CSV)
# ---------------------------------------------------------------------------


_TS_FIELDS = ("step", "t", "dt", "Nu_top", "Nu_bot", "Vrms", "Vmax", "mean_T")


def _read_timeseries(path: Path) -> list[dict]:
    """Read timeseries.csv, tolerating older runs that lack Vmax."""
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for row in csv.DictReader(f):
            out.append({
                "step": int(row["step"]),
                "t": float(row["t"]),
                "dt": float(row["dt"]),
                "Nu_top": float(row["Nu_top"]),
                "Nu_bot": float(row["Nu_bot"]),
                "Vrms": float(row["Vrms"]),
                # Vmax is optional for backward-compat with older CSVs
                "Vmax": float(row.get("Vmax") or "nan"),
                "mean_T": float(row["mean_T"]),
            })
    return out


def _append_timeseries_row(path: Path, row: dict) -> None:
    """Append one row; auto-add header on first write.

    If the existing file pre-dates the current ``_TS_FIELDS`` schema
    (e.g. an older run that lacks ``Vmax``), the file is migrated in
    place by re-writing every row with the current schema (filling
    missing fields with ``nan``).
    """
    if path.exists():
        # Check existing header
        with open(path) as f:
            existing_header = f.readline().rstrip("\n")
        existing_fields = tuple(existing_header.split(","))
        if existing_fields != _TS_FIELDS:
            # Migrate: read all old rows, rewrite with the new schema.
            old_rows = _read_timeseries(path)
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=_TS_FIELDS, extrasaction="ignore",
                )
                writer.writeheader()
                for r in old_rows:
                    writer.writerow({k: r.get(k) for k in _TS_FIELDS})

    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=_TS_FIELDS, extrasaction="ignore",
        )
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in _TS_FIELDS})


def _list_saved_steps(output_dir: Path) -> list[int]:
    """Indices of completed timesteps, derived from the xdmf files."""
    if not output_dir.exists():
        return []
    steps = set()
    for p in output_dir.glob(f"{RUN_NAME}.mesh.[0-9]*.xdmf"):
        try:
            steps.add(int(p.stem.split(".")[-1]))
        except ValueError:
            continue
    return sorted(steps)


# ---------------------------------------------------------------------------
# Steady-state test
# ---------------------------------------------------------------------------


def _is_steady(timeseries: list[dict], config: ConvectionConfig) -> bool:
    """Pass when both Nu(t) and Vrms(t) are statistically stationary.

    Tested over the last ``steady_window`` fraction of points (with a
    floor of ``steady_min_window`` samples).  Both the change in mean
    between halves and the coefficient of variation must be below the
    configured tolerances.
    """
    n = len(timeseries)
    n_window = max(int(n * config.steady_window), config.steady_min_window)
    if n < n_window:
        return False

    nu = np.array([(r["Nu_top"] + r["Nu_bot"]) / 2 for r in timeseries[-n_window:]])
    vrms = np.array([r["Vrms"] for r in timeseries[-n_window:]])

    for series in (nu, vrms):
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


def _compute_diagnostics(mesh, T, v, config: ConvectionConfig) -> dict:
    import underworld3 as uw

    integrator = uw.maths.Integral(mesh, 1)
    V_total = float(integrator.evaluate())

    integrator.fn = T.sym[0]
    mean_T = float(integrator.evaluate()) / V_total

    integrator.fn = v.sym.dot(v.sym)
    Vrms = float(np.sqrt(max(integrator.evaluate(), 0.0) / V_total))

    # Vmax: peak velocity magnitude at velocity DOFs.  Fast — just an
    # in-memory max with an MPI all-reduce for parallel safety.
    v_dofs = np.asarray(v.array).reshape(-1, mesh.dim)
    v_mag = np.linalg.norm(v_dofs, axis=1)
    local_max = float(v_mag.max()) if v_mag.size else 0.0
    if uw.mpi.size > 1:
        import mpi4py
        Vmax = float(uw.mpi.comm.allreduce(local_max, op=mpi4py.MPI.MAX))
    else:
        Vmax = local_max

    # Surface flux samples — averaged at top and bottom boundaries.
    n_samples = 100
    top = np.column_stack([
        np.linspace(0.0, config.aspect_ratio, n_samples),
        np.full(n_samples, 1.0),
    ])
    bot = np.column_stack([
        np.linspace(0.0, config.aspect_ratio, n_samples),
        np.full(n_samples, 0.0),
    ])

    flux_fn = -mesh.vector.gradient(T.sym[0]).dot(mesh.CoordinateSystem.unit_e_1)
    flux_top = float(np.mean(uw.function.evaluate(flux_fn, top)))
    flux_bot = float(np.mean(uw.function.evaluate(flux_fn, bot)))

    delta_T = config.T_bottom - config.T_top  # H = 1
    Nu_top = flux_top / max(abs(delta_T), 1e-30)
    Nu_bot = flux_bot / max(abs(delta_T), 1e-30)

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
        mesh = uw.discretisation.Mesh(
            str(mesh_h5),
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
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=3)

    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = config.viscosity
    stokes.tolerance = 1.0e-3
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Bottom")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Top")
    stokes.add_dirichlet_bc((0.0, sympy.oo), "Left")
    stokes.add_dirichlet_bc((0.0, sympy.oo), "Right")
    stokes.bodyforce = sympy.Matrix([0, config.rayleigh * T.sym[0]])

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
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Short-circuit on a *steady* summary only ─────────────────────
    # ``stalled`` summaries are not written (so the user can extend with
    # a larger max_steps and resume).  Only ``status == "steady"`` is the
    # irrevocable done marker.
    summary_path = output_dir / "run_summary.yaml"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = yaml.safe_load(f) or {}
        if summary.get("status") == "steady":
            uw.pprint(
                f"[evolve] {output_dir}: status='steady', "
                f"n_steps={summary.get('n_steps')}.  No work to do."
            )
            return _read_timeseries(output_dir / "timeseries.csv")

    timeseries_path = output_dir / "timeseries.csv"
    timeseries = _read_timeseries(timeseries_path)
    saved_steps = _list_saved_steps(output_dir)

    if not saved_steps:
        # ── Fresh start ─────────────────────────────────────────────
        archive = getattr(mesh, "_uw_run_archive", None)

        if archive is not None and config.restart_policy == "seed_from_old":
            old_steps = _list_saved_steps(archive)
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

        _write_manifest(output_dir, {
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
        _append_timeseries_row(timeseries_path, row)
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
    while timestep < config.max_steps:
        stokes.solve(zero_init_guess=False)
        delta_t = config.dt_factor * adv_diff.estimate_dt()
        adv_diff.solve(timestep=delta_t, zero_init_guess=False)
        timestep += 1
        elapsed += float(delta_t)

        diag = _compute_diagnostics(mesh, T, v, config)
        row = {
            "step": timestep, "t": elapsed, "dt": float(delta_t), **diag,
        }
        _append_timeseries_row(timeseries_path, row)
        timeseries.append(row)

        uw.pprint(
            f"[step {timestep:5d}]  t={elapsed:9.4e}  dt={delta_t:.2e}  "
            f"Nu_top={diag['Nu_top']:6.3f}  Nu_bot={diag['Nu_bot']:6.3f}  "
            f"Vrms={diag['Vrms']:8.4f}  Vmax={diag['Vmax']:8.4f}  "
            f"<T>={diag['mean_T']:5.3f}",
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
    interior is isothermal, with thin top and bottom thermal BLs of
    thickness ≈ δ.  A small two-mode sinusoidal perturbation breaks
    symmetry to seed the rolls.
    """
    import underworld3 as uw

    x, y = mesh.X
    delta = max(bl_thickness, 1.0e-3)
    base = 0.5 - 0.5 * (sympy.tanh(2 * y / delta) - sympy.tanh(2 * (1 - y) / delta))
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
    output_dir = Path(config.output_dir)
    summary_path = output_dir / "run_summary.yaml"

    if summary_path.exists():
        with open(summary_path) as f:
            cached = yaml.safe_load(f)
        if cached.get("status") == "steady":
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

    summary = {
        "status": status,
        "n_steps": last_step,
        "rayleigh": config.rayleigh,
        "aspect_ratio": config.aspect_ratio,
        "Nu_top_mean": float(np.mean([r["Nu_top"] for r in window])),
        "Nu_bot_mean": float(np.mean([r["Nu_bot"] for r in window])),
        "Nu_mean": float(nu_window.mean()),
        "Nu_std": float(nu_window.std()),
        "Vrms_mean": float(vrms_window.mean()),
        "Vrms_std": float(vrms_window.std()),
        "mean_T_mean": float(meanT_window.mean()),
        "elapsed_t": float(window[-1]["t"]),
    }

    # Only persist the summary when the run reached steady state.
    # Stalled runs leave no done-marker so a re-invocation with a higher
    # ``max_steps`` extends them rather than short-circuiting.
    if status == "steady":
        with open(summary_path, "w") as f:
            yaml.dump(summary, f, default_flow_style=False, sort_keys=False)

    return summary
