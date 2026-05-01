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
import yaml
from pydantic import Field

from underworld3.workflows import WorkflowConfig

import convection_config as _convection


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


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

    # Tracer movie
    n_tracers_per_dim: int = Field(default=40, ge=4)
    tracer_color: str = "lightgrey"
    tracer_size: int = Field(default=4, ge=1)


# ---------------------------------------------------------------------------
# H5 inventory
# ---------------------------------------------------------------------------


def _list_saved_steps(run_dir: Path) -> list[int]:
    return sorted(
        int(p.stem.split(".")[-1])
        for p in run_dir.glob(f"{_convection.RUN_NAME}.mesh.[0-9]*.xdmf")
    )


def _load_run_mesh(run_dir: Path):
    import underworld3 as uw

    mesh_h5 = run_dir / f"{_convection.RUN_NAME}.mesh.00000.h5"
    if not mesh_h5.exists():
        raise FileNotFoundError(f"Mesh checkpoint not found: {mesh_h5}")
    return uw.discretisation.Mesh(
        str(mesh_h5),
        coordinate_system_type=uw.coordinates.CoordinateSystemType.CARTESIAN,
    )


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
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=3)
    v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)

    steps = _list_saved_steps(run_dir)
    if cfg.every > 1:
        steps = steps[:: cfg.every]
    if not steps:
        return frames_dir

    pl = pv.Plotter(window_size=list(cfg.window_size), off_screen=True)

    for step in steps:
        try:
            T.read_timestep(
                _convection.RUN_NAME, "T", step, outputPath=str(run_dir),
            )
            if cfg.show_streamlines:
                v.read_timestep(
                    _convection.RUN_NAME, "U", step, outputPath=str(run_dir),
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
            show_edges=False, show_scalar_bar=False,
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
            f"step {step}", position="upper_left",
            font_size=10, color="black",
        )
        out = frames_dir / f"frame_{step:06d}.png"
        pl.screenshot(str(out), return_img=False)

    pl.close()
    return frames_dir


def render_tracer_frames(cfg: VisualiseConfig) -> Optional[Path]:
    """Render frames showing T plus passive tracers advected through history.

    Tracers are seeded on a regular ``n_tracers_per_dim x n_tracers_per_dim``
    grid in the domain at step 0 and advected with midpoint steps using
    the velocity field at each saved timestep.  This shows mixing /
    circulation patterns even when the temperature field looks steady.
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
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=3)
    v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)

    steps = _list_saved_steps(run_dir)
    if cfg.every > 1:
        steps = steps[:: cfg.every]
    if not steps:
        return frames_dir

    # Seed tracers on a regular grid inside the domain.  Read the
    # extent from the loaded mesh so we don't depend on attributes
    # that aren't preserved across the h5 round-trip.
    coords = mesh.X.coords
    x_lo, y_lo = float(coords[:, 0].min()), float(coords[:, 1].min())
    x_hi, y_hi = float(coords[:, 0].max()), float(coords[:, 1].max())
    n = cfg.n_tracers_per_dim
    xs = np.linspace(x_lo + 0.02 * (x_hi - x_lo), x_hi - 0.02 * (x_hi - x_lo), n)
    ys = np.linspace(y_lo + 0.02 * (y_hi - y_lo), y_hi - 0.02 * (y_hi - y_lo), n)
    grid = np.array([(x, y) for y in ys for x in xs])
    tracers = grid.copy()

    # Read timeseries to estimate dt for advection per saved interval
    ts_path = run_dir / "timeseries.csv"
    ts_rows = _convection._read_timeseries(ts_path)
    step_to_t = {r["step"]: r["t"] for r in ts_rows}

    pl = pv.Plotter(window_size=list(cfg.window_size), off_screen=True)
    prev_t = None

    for step in steps:
        try:
            T.read_timestep(
                _convection.RUN_NAME, "T", step, outputPath=str(run_dir),
            )
            v.read_timestep(
                _convection.RUN_NAME, "U", step, outputPath=str(run_dir),
            )
        except Exception as e:
            print(f"[viz] Skip step {step}: {e}", flush=True)
            continue

        # Advance tracers using velocity at this time, time-step = saved-step Δt.
        # uw.function.evaluate of a 2D vector field at (n, 2) points returns
        # shape (n, 1, 2) — squeeze the middle dim before arithmetic.
        t_now = step_to_t.get(step)
        if prev_t is not None and t_now is not None and t_now > prev_t:
            dt = t_now - prev_t
            v_here = np.asarray(uw.function.evaluate(v.sym, tracers)).reshape(-1, 2)
            half = tracers + 0.5 * dt * v_here
            half = np.clip(half, [x_lo, y_lo], [x_hi, y_hi])
            v_half = np.asarray(uw.function.evaluate(v.sym, half)).reshape(-1, 2)
            tracers = tracers + dt * v_half
            tracers = np.clip(tracers, [x_lo, y_lo], [x_hi, y_hi])
        prev_t = t_now

        pl.clear()
        pv_t = vis.meshVariable_to_pv_mesh_object(T)
        pv_t.point_data["T"] = vis.scalar_fn_to_pv_points(pv_t, T.sym[0])
        pl.add_mesh(
            pv_t, cmap=cfg.cmap, scalars="T",
            clim=[0.0, 1.0], show_edges=False, show_scalar_bar=False,
        )

        # Tracers as a point cloud
        tracer_pts = np.zeros((tracers.shape[0], 3))
        tracer_pts[:, :2] = tracers
        pl.add_mesh(
            pv.PolyData(tracer_pts),
            color=cfg.tracer_color,
            point_size=cfg.tracer_size,
            render_points_as_spheres=True,
            show_scalar_bar=False,
        )

        pl.camera_position = "xy"
        pl.add_text(
            f"step {step}", position="upper_left",
            font_size=10, color="black",
        )
        out = frames_dir / f"frame_{step:06d}.png"
        pl.screenshot(str(out), return_img=False)

    pl.close()
    return frames_dir


# ---------------------------------------------------------------------------
# Movie encoding
# ---------------------------------------------------------------------------


def encode_movie(
    cfg: VisualiseConfig,
    kind: Literal["temperature", "tracers"] = "temperature",
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
    list_path = frames_dir / "frames.txt"
    with open(list_path, "w") as f:
        for p in frames:
            f.write(f"file '{p.name}'\n")
            f.write(f"duration {1.0 / max(cfg.fps, 1):.6f}\n")
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
