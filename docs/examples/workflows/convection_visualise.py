"""Visualisation pass for the convection workflow — strict separation.

Reads h5 timesteps written by ``convection_config.evolve`` and produces:

* **PNG frames** of the temperature field per saved timestep (pyvista).
* **MP4 movies** of those frames (ffmpeg via ``imageio_ffmpeg``).
* **PNG frames + MP4** of the temperature field with passive tracers
  overlaid, integrated through the saved velocity history with midpoint
  steps.

This module knows nothing about solvers — only h5 and pyvista.  The
solver workflow can be fully complete (all ``run_summary.yaml`` files
written) before any visualisation runs, and re-running the
visualisation never touches the solver outputs.

Usage::

    import convection_visualise as viz

    cfg = viz.VisualiseConfig(
        run_dir="output/convection_sweep/aspect_1x1/Ra1e5",
        every=1,                       # use every Nth saved timestep
        n_tracers_per_dim=40,
    )
    viz.render_temperature_frames(cfg)
    viz.encode_movie(cfg, kind="temperature")
    viz.render_tracer_frames(cfg)
    viz.encode_movie(cfg, kind="tracers")
"""

import sys
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from underworld3.workflows import WorkflowConfig

from _run import RUN_NAME, Run


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TracerPopulation(BaseModel):
    """A named tracer population: how to seed it, and how to render it.

    ``region`` is one of:

    * ``"uniform"``  — fill the whole mesh with an n×n grid (the
      ``x_range`` / ``y_range`` defaults are normalised 0..1 mapped to
      the mesh extent).
    * ``"block"``    — fill a rectangular block specified by
      ``x_range`` and ``y_range`` (in mesh coordinates).

    Each population is advected independently and rendered as its own
    point cloud with its own colour.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    region: Literal["uniform", "block"] = "uniform"
    n: int = Field(default=40, ge=4, description="Tracers per dimension")
    x_range: tuple[float, float] = (0.02, 0.98)
    y_range: tuple[float, float] = (0.02, 0.98)
    colour: str = "lightgrey"
    point_size: int = Field(default=4, ge=1)


def _default_populations() -> list[TracerPopulation]:
    """Three populations: uniform, top-left block, dead-centre block.

    Mid-to-dark greys only — pure white / near-white tracers vanish
    against the hot/cold extremes of the temperature colormap (which
    has very saturated red and blue at high Ra) and against the
    yellow midtone of RdYlBu_r.
    """
    return [
        TracerPopulation(
            name="uniform",
            region="uniform",
            n=30,
            colour="#888888",
            point_size=2,
        ),
        TracerPopulation(
            name="top_left_block",
            region="block",
            n=18,
            x_range=(0.02, 0.25),
            y_range=(0.75, 0.98),
            colour="#444444",
            point_size=3,
        ),
        TracerPopulation(
            name="centre_block",
            region="block",
            n=18,
            x_range=(0.40, 0.60),
            y_range=(0.40, 0.60),
            colour="#111111",
            point_size=3,
        ),
    ]


class VisualiseConfig(WorkflowConfig):
    """Visualisation parameters; reads h5 from ``run_dir``."""

    workflow_name: str = "convection_visualise"
    description: str = "Render frames + movies from a finished convection run"

    run_dir: str = Field(default="", description="Per-run output directory to read")
    output_subdir: str = Field(default="frames", description="Frames go here under run_dir")
    movies_subdir: str = Field(default="movies", description="MP4s go here under run_dir")

    every: int = Field(default=1, ge=1, description="Use every Nth saved timestep")
    fps: int = Field(default=12, ge=1)
    window_size: tuple[int, int] = Field(default=(1000, 800))
    cmap: str = Field(default="RdBu_r")
    show_streamlines: bool = False
    show_scalar_bar: bool = True

    # Tracer populations — a list so we can render several at once with
    # distinct colours (e.g. uniform background + two tagged blocks).
    tracer_populations: list[TracerPopulation] = Field(
        default_factory=_default_populations,
    )


# ---------------------------------------------------------------------------
# H5 inventory
# ---------------------------------------------------------------------------


def _read_step_times(run_dir: Path) -> dict:
    """Map step → simulated time t from timeseries.csv (or {} if missing)."""
    return {r["step"]: r["t"] for r in Run(run_dir).timeseries}


def _format_t(t: float | None) -> str:
    """Fixed-width simulated-time label (no flicker)."""
    if t is None:
        return "t = ----.------"
    return f"t = {t:9.6f}"


def _load_run_mesh(run_dir: Path):
    import underworld3 as uw

    mesh_h5 = run_dir / f"{RUN_NAME}.mesh.00000.h5"
    if not mesh_h5.exists():
        raise FileNotFoundError(f"Mesh checkpoint not found: {mesh_h5}")
    return uw.discretisation.Mesh(
        str(mesh_h5),
        coordinate_system_type=uw.coordinates.CoordinateSystemType.CARTESIAN,
    )


def _run_T_degree(run_dir: Path) -> int:
    """Read ``T_degree`` from the run's manifest snapshot.

    Falls back to ``3`` (the legacy hardcode) when a pre-step-2
    manifest doesn't carry the field.
    """
    manifest = Run(run_dir).manifest
    if manifest is None:
        return 3
    return int(manifest.config_snapshot.get("T_degree", 3))


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------


def render_temperature_frames(cfg: VisualiseConfig) -> Optional[Path]:
    """Render one PNG per saved timestep showing the T field.

    Returns the directory of frames, or ``None`` if pyvista is
    unavailable.
    """
    import underworld3 as uw

    if uw.mpi.size > 1:
        return None

    try:
        import pyvista as pv
        import underworld3.visualisation as vis
    except ImportError:
        return None

    run_dir = Path(cfg.run_dir)
    frames_dir = run_dir / cfg.output_subdir / "temperature"
    frames_dir.mkdir(parents=True, exist_ok=True)

    mesh = _load_run_mesh(run_dir)
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=_run_T_degree(run_dir))
    # Only register the velocity MeshVariable when we actually need it —
    # adding it to the mesh DM can subtly change how T is rendered
    # (DOF layout, scalar interpolation), so for the temperature-only
    # frames we keep the mesh clean.
    v = (
        uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
        if cfg.show_streamlines else None
    )

    steps = Run(run_dir).steps
    if cfg.every > 1:
        steps = steps[:: cfg.every]
    if not steps:
        return frames_dir

    step_to_t = _read_step_times(run_dir)
    pl = pv.Plotter(window_size=list(cfg.window_size), off_screen=True)

    for step in steps:
        try:
            T.read_timestep(
                RUN_NAME, "T", step, outputPath=str(run_dir),
            )
            if cfg.show_streamlines:
                v.read_timestep(
                    RUN_NAME, "U", step, outputPath=str(run_dir),
                )
        except Exception as e:
            print(f"[viz] Skip step {step}: {e}", flush=True)
            continue

        pl.clear()
        pv_t = vis.meshVariable_to_pv_mesh_object(T)
        pv_t.point_data["T"] = vis.scalar_fn_to_pv_points(pv_t, T.sym[0])
        pl.add_mesh(
            pv_t, cmap=cfg.cmap, scalars="T",
            clim=[0.0, 1.0],
            show_edges=False,
            show_scalar_bar=cfg.show_scalar_bar,
            scalar_bar_args={"title": "T"} if cfg.show_scalar_bar else None,
        )

        if cfg.show_streamlines:
            pv_v = vis.mesh_to_pv_mesh(mesh)
            pv_v.point_data["V"] = vis.vector_fn_to_pv_points(pv_v, v.sym)
            skip = max(1, mesh._centroids.shape[0] // 200)
            pts = np.zeros((mesh._centroids[::skip].shape[0], 3))
            pts[:, 0] = mesh._centroids[::skip, 0]
            pts[:, 1] = mesh._centroids[::skip, 1]
            try:
                pvstream = pv_v.streamlines_from_source(
                    pv.PolyData(pts),
                    vectors="V",
                    integration_direction="both",
                    surface_streamlines=True,
                    max_steps=50,
                )
                pl.add_mesh(pvstream, opacity=0.4, show_scalar_bar=False)
            except Exception:
                pass

        pl.camera_position = "xy"
        pl.add_text(
            _format_t(step_to_t.get(step)), position="upper_left",
            font_size=10, color="black",
        )
        out = frames_dir / f"frame_{step:06d}.png"
        pl.screenshot(str(out), return_img=False)

    pl.close()
    return frames_dir


def _seed_tracers(
    pop: TracerPopulation,
    x_lo: float, y_lo: float, x_hi: float, y_hi: float,
) -> np.ndarray:
    """Build the initial (M, 2) tracer array for one population."""
    Lx = x_hi - x_lo
    Ly = y_hi - y_lo
    if pop.region == "uniform":
        # x_range / y_range are normalised 0..1 mapped to mesh extent.
        x0 = x_lo + pop.x_range[0] * Lx
        x1 = x_lo + pop.x_range[1] * Lx
        y0 = y_lo + pop.y_range[0] * Ly
        y1 = y_lo + pop.y_range[1] * Ly
    elif pop.region == "block":
        # x_range / y_range are absolute mesh coordinates.
        x0, x1 = pop.x_range
        y0, y1 = pop.y_range
    else:
        raise ValueError(f"Unknown region {pop.region!r} for tracer population")

    xs = np.linspace(x0, x1, pop.n)
    ys = np.linspace(y0, y1, pop.n)
    return np.array([(x, y) for y in ys for x in xs])


def render_tracer_frames(cfg: VisualiseConfig) -> Optional[Path]:
    """Render frames showing T plus one or more tracer populations.

    Each population is seeded once at step 0 (uniform across the
    domain, or in a specified rectangular block) and advected through
    the saved velocity history with midpoint integration.  Populations
    are rendered as separate point clouds, each in its own colour, so
    a uniform background population can be drawn alongside one or more
    tagged blocks.
    """
    import underworld3 as uw

    if uw.mpi.size > 1:
        return None

    try:
        import pyvista as pv
        import underworld3.visualisation as vis
    except ImportError:
        return None

    run_dir = Path(cfg.run_dir)
    frames_dir = run_dir / cfg.output_subdir / "tracers"
    frames_dir.mkdir(parents=True, exist_ok=True)

    mesh = _load_run_mesh(run_dir)
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=_run_T_degree(run_dir))
    v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)

    steps = Run(run_dir).steps
    if cfg.every > 1:
        steps = steps[:: cfg.every]
    if not steps:
        return frames_dir

    # Read mesh extent from h5-loaded coordinates (no minCoords attr after reload).
    coords = mesh.X.coords
    x_lo, y_lo = float(coords[:, 0].min()), float(coords[:, 1].min())
    x_hi, y_hi = float(coords[:, 0].max()), float(coords[:, 1].max())

    # Seed each population once.  We track them as a list aligned with
    # cfg.tracer_populations so each can be rendered in its own colour.
    populations = list(cfg.tracer_populations)
    tracers_per_pop: list[np.ndarray] = [
        _seed_tracers(p, x_lo, y_lo, x_hi, y_hi) for p in populations
    ]

    # Read timeseries to get the time at each saved step (for dt per interval).
    step_to_t = {r["step"]: r["t"] for r in Run(run_dir).timeseries}

    prev_t = None
    pl = pv.Plotter(window_size=list(cfg.window_size), off_screen=True)

    for step in steps:
        try:
            T.read_timestep(
                RUN_NAME, "T", step, outputPath=str(run_dir),
            )
            v.read_timestep(
                RUN_NAME, "U", step, outputPath=str(run_dir),
            )
        except Exception as e:
            print(f"[viz] Skip step {step}: {e}", flush=True)
            continue

        # Advance every population through the same dt using midpoint
        # integration in the velocity field at this saved step.
        # uw.function.evaluate of a 2D vector field at (n, 2) points
        # returns shape (n, 1, 2) — reshape to (n, 2).
        t_now = step_to_t.get(step)
        if prev_t is not None and t_now is not None and t_now > prev_t:
            dt = t_now - prev_t
            for i, pts in enumerate(tracers_per_pop):
                v_here = np.asarray(
                    uw.function.evaluate(v.sym, pts)
                ).reshape(-1, 2)
                half = pts + 0.5 * dt * v_here
                half = np.clip(half, [x_lo, y_lo], [x_hi, y_hi])
                v_half = np.asarray(
                    uw.function.evaluate(v.sym, half)
                ).reshape(-1, 2)
                advanced = pts + dt * v_half
                tracers_per_pop[i] = np.clip(
                    advanced, [x_lo, y_lo], [x_hi, y_hi],
                )
        prev_t = t_now

        pl.clear()
        pv_t = vis.meshVariable_to_pv_mesh_object(T)
        pv_t.point_data["T"] = vis.scalar_fn_to_pv_points(pv_t, T.sym[0])
        pl.add_mesh(
            pv_t, cmap=cfg.cmap, scalars="T",
            clim=[0.0, 1.0], show_edges=False,
            show_scalar_bar=cfg.show_scalar_bar,
            scalar_bar_args={"title": "T"} if cfg.show_scalar_bar else None,
        )

        for pop, pts in zip(populations, tracers_per_pop):
            tracer_pts3 = np.zeros((pts.shape[0], 3))
            tracer_pts3[:, :2] = pts
            pl.add_mesh(
                pv.PolyData(tracer_pts3),
                color=pop.colour,
                point_size=pop.point_size,
                render_points_as_spheres=True,
                show_scalar_bar=False,
            )

        pl.camera_position = "xy"
        pl.add_text(
            _format_t(step_to_t.get(step)), position="upper_left",
            font_size=10, color="black",
        )
        out = frames_dir / f"frame_{step:06d}.png"
        pl.screenshot(str(out), return_img=False)

    pl.close()
    return frames_dir


def render_tracer_frames_frozen(
    cfg: VisualiseConfig,
    n_frames: int = 200,
    virtual_dt: Optional[float] = None,
    source_step: Optional[int] = None,
) -> Optional[Path]:
    """Render tracer frames using a frozen steady-state velocity field.

    Useful for low-Ra runs where the actual flow is steady — looping
    the saved chain would give incorrect (artificially periodic)
    tracer trajectories.  Instead, we read the velocity at one
    chosen step (default: the final saved h5), freeze it, and advect
    every tracer population through ``virtual_dt`` for ``n_frames``
    rendered frames.  The temperature picture is the steady T from
    that step too.

    Parameters
    ----------
    n_frames : int
        Number of frames to render.
    virtual_dt : float or None
        Tracer advection step.  If None, defaults to ``L / Vrms / 4`` so
        a tracer crossing the box takes ~4 frames at peak speed.
    source_step : int or None
        Which saved h5 step to read v and T from.  Default: latest.
    """
    import time as _time
    import underworld3 as uw

    if uw.mpi.size > 1:
        return None
    try:
        import pyvista as pv
        import underworld3.visualisation as vis
    except ImportError:
        return None

    run_dir = Path(cfg.run_dir)
    frames_dir = run_dir / cfg.output_subdir / "tracers_frozen"
    frames_dir.mkdir(parents=True, exist_ok=True)

    mesh = _load_run_mesh(run_dir)
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=_run_T_degree(run_dir))
    v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)

    saved = Run(run_dir).steps
    if not saved:
        return frames_dir
    if source_step is None:
        source_step = saved[-1]
    print(f"[frozen] reading T,v from step {source_step}", flush=True)
    T.read_timestep(RUN_NAME, "T", source_step, outputPath=str(run_dir))
    v.read_timestep(RUN_NAME, "U", source_step, outputPath=str(run_dir))

    # Estimate Vmax for the dt heuristic.
    coords = mesh.X.coords
    x_lo, y_lo = float(coords[:, 0].min()), float(coords[:, 1].min())
    x_hi, y_hi = float(coords[:, 0].max()), float(coords[:, 1].max())
    L = max(x_hi - x_lo, y_hi - y_lo)
    Vmax_local = float(np.linalg.norm(np.asarray(v.array).reshape(-1, 2), axis=1).max())
    if virtual_dt is None:
        # Cross the box in ~50 frames at peak speed so streamlines are
        # legible; smaller dt → smoother but more frames needed.
        virtual_dt = L / max(Vmax_local, 1e-30) / 50.0
    print(f"[frozen] Vmax≈{Vmax_local:.4f}, virtual_dt={virtual_dt:.3e}", flush=True)

    populations = list(cfg.tracer_populations)
    tracers_per_pop: list[np.ndarray] = [
        _seed_tracers(p, x_lo, y_lo, x_hi, y_hi) for p in populations
    ]

    # Pre-build the temperature pyvista mesh (it's the same every frame).
    pl = pv.Plotter(window_size=list(cfg.window_size), off_screen=True)
    pv_t = vis.meshVariable_to_pv_mesh_object(T)
    pv_t.point_data["T"] = vis.scalar_fn_to_pv_points(pv_t, T.sym[0])

    for frame_idx in range(n_frames):
        # Advect every population by virtual_dt with midpoint integration
        # in the frozen velocity field.
        for i, pts in enumerate(tracers_per_pop):
            v_here = np.asarray(uw.function.evaluate(v.sym, pts)).reshape(-1, 2)
            half = pts + 0.5 * virtual_dt * v_here
            half = np.clip(half, [x_lo, y_lo], [x_hi, y_hi])
            v_half = np.asarray(uw.function.evaluate(v.sym, half)).reshape(-1, 2)
            tracers_per_pop[i] = np.clip(
                pts + virtual_dt * v_half, [x_lo, y_lo], [x_hi, y_hi],
            )

        pl.clear()
        pl.add_mesh(
            pv_t, cmap=cfg.cmap, scalars="T",
            clim=[0.0, 1.0], show_edges=False,
            show_scalar_bar=cfg.show_scalar_bar,
            scalar_bar_args={"title": "T"} if cfg.show_scalar_bar else None,
        )
        for pop, pts in zip(populations, tracers_per_pop):
            pts3 = np.zeros((pts.shape[0], 3))
            pts3[:, :2] = pts
            pl.add_mesh(
                pv.PolyData(pts3),
                color=pop.colour,
                point_size=pop.point_size,
                render_points_as_spheres=True,
                show_scalar_bar=False,
            )
        pl.camera_position = "xy"
        # Use a fake "virtual time" advancing by virtual_dt each frame.
        virtual_t = frame_idx * virtual_dt
        pl.add_text(
            f"frozen  {_format_t(virtual_t)}",
            position="upper_left", font_size=10, color="black", font="courier",
        )
        out = frames_dir / f"frame_{frame_idx:06d}.png"
        pl.screenshot(str(out), return_img=False)

    pl.close()
    return frames_dir


# ---------------------------------------------------------------------------
# Movie encoding
# ---------------------------------------------------------------------------


def encode_movie(
    cfg: VisualiseConfig,
    kind: Literal["temperature", "tracers", "tracers_frozen"] = "temperature",
) -> Optional[Path]:
    """Encode rendered frames into an mp4 via ``ffmpeg``.

    Uses the ``ffmpeg`` binary on ``PATH``; no Python video library
    dependency.  Returns the mp4 path, or ``None`` if no frames exist
    or ``ffmpeg`` is not available.
    """
    import shutil as _shutil
    import subprocess

    run_dir = Path(cfg.run_dir)
    frames_dir = run_dir / cfg.output_subdir / kind
    frames = sorted(frames_dir.glob("frame_*.png"))
    if not frames:
        return None

    ffmpeg = _shutil.which("ffmpeg")
    if ffmpeg is None:
        print("[viz] ffmpeg not found on PATH; skipping movie encoding.", flush=True)
        return None

    movies_dir = run_dir / cfg.movies_subdir
    movies_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = movies_dir / f"{kind}.mp4"

    # ffmpeg input pattern only works for sequentially-numbered frames.
    # Our frame_*.png files are named by save-step index, which has gaps
    # (0, 5, 10, …) so we feed them via a concat list.
    #
    # When timeseries.csv has step→t mapping, scale each frame's duration
    # to its simulated-time interval so the playback speed is proportional
    # to simulated time. This handles both adaptive-dt runs and runs with
    # mixed save cadences (e.g. every-1 then every-5 across a restart).
    # Falls back to uniform 1/fps duration if no time info is available
    # or if the time intervals are zero / non-monotonic.
    step_to_t = _read_step_times(run_dir)
    fallback_dur = 1.0 / max(cfg.fps, 1)
    durations = []
    if step_to_t:
        # Frame filename is frame_{step:06d}.png — extract the step
        steps = []
        for p in frames:
            try:
                steps.append(int(p.stem.split("_")[-1]))
            except ValueError:
                steps.append(None)
        ts = [step_to_t.get(s) if s is not None else None for s in steps]
        if all(t is not None for t in ts) and len(ts) >= 2:
            intervals = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
            if all(d > 0 for d in intervals):
                # Scale so total movie length matches len(frames) / fps
                target_total = len(frames) * fallback_dur
                scale = target_total / sum(intervals)
                durations = [d * scale for d in intervals]
                # Last frame gets a duration equal to the prior interval
                durations.append(durations[-1])
    if not durations:
        durations = [fallback_dur] * len(frames)

    list_path = frames_dir / "frames.txt"
    with open(list_path, "w") as f:
        for p, d in zip(frames, durations):
            f.write(f"file '{p.name}'\n")
            f.write(f"duration {d:.6f}\n")
        # Repeat the last frame once with no duration so concat ends cleanly.
        f.write(f"file '{frames[-1].name}'\n")

    cmd = [
        ffmpeg, "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_path),
        "-vf", "scale='trunc(iw/2)*2:trunc(ih/2)*2',format=yuv420p",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-movflags", "+faststart",
        str(mp4_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        print(f"[viz] ffmpeg failed:\n{exc.stderr.decode(errors='replace')}", flush=True)
        return None
    return mp4_path


def view():
    """Display this module's config classes and helpers."""
    from underworld3.workflows import view as _wf_view
    _wf_view(sys.modules[__name__])
