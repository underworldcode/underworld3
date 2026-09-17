"""Animate the stress field around a tumbling fault, beside its Mohr point.

Renders from the CHECKPOINTS written by ``fault_mohr_tumble.py`` — no solving.
Each frame is one saved sample: the camera and the cut plane stay fixed while the
fault rotates, so what moves is the fault and the stress field it perturbs.

The field is the SIGNED change in the second stress invariant,

.. math:: \\Delta\\tau_{II} = \\tau_{II}(v) - \\tau_{II}^{\\infty},

against the uniform far-field state the walls impose. A welded fault barely
slips, so this perturbation is small (a fraction of a percent of the background)
and the colour range is set from the field itself rather than the background —
the point is the SHAPE: the lobes of concentration and the shadow either side.
The same renderer shows a large signal once the fault is allowed to fail.

Run (after a sweep has been checkpointed)::

    python scripts/sessions/fault_mohr_frames.py -uw_dim 2 -uw_animate 1
    python scripts/sessions/fault_mohr_frames.py -uw_dim 3 -uw_animate 1 \\
        -uw_output_dir ~/+Simulations/fault_mohr_triaxial/fmg_tol1e-3

``-uw_sample K`` renders that one sample and stops, for checking a frame.
"""
import glob
import os

import numpy as np
import sympy

import underworld3 as uw
import underworld3.visualisation as vis
import pyvista as pv

pv.OFF_SCREEN = True

params = uw.Params(
    uw_dim=2,
    uw_output_dir=os.path.expanduser("~/+Simulations/fault_mohr_triaxial"),
    uw_stride=1,
    uw_sample=-1,               # >= 0: render just this sample and stop
    uw_animate=0,
    uw_max_frames=0,            # 0 = no cap
    uw_shard=0,                 # frames are independent: run several processes
    uw_n_shards=1,
    uw_clim=0.0,                # > 0: use this colour half-range. REQUIRED when
                                # sharding, or each shard picks its own and the
                                # animation flickers
    uw_gif_only=0,              # assemble the GIF from frames already on disk
    uw_volume=1,                # 3-D: volume render (0 = one cut plane)
    uw_volume_res=160,          # samples per side of the volume grid
    uw_mask_cells=2.0,          # hide the field within this many cells of the
                                # fault: a split-node fault takes up its slip
                                # across one cell, so that layer is orders
                                # above the field around it and lumpy with the
                                # mesh. 0 draws it.
    uw_gif_stride=1,            # keep every n-th frame in the GIF
    uw_gif_scale=1.0,           # and scale it: a field panel carries a lot of
                                # colour detail, so a full-size GIF of a long
                                # sweep runs past what a page will take
)

DIM = params.uw_dim
ETA = 1.0
CENTRE = np.full(DIM, 0.5)
# The refinement size on the fault, and how far off the fault surface the
# colour range is measured (the near-fault layer is excluded from the RANGE,
# not from the picture)
H_NEAR = {2: 0.0125, 3: 0.05}[DIM]
GUARD = 2.0 * H_NEAR
RUN_DIR = os.path.join(params.uw_output_dir, f"tumble_{DIM}d")
FRAME_DIR = os.path.join(params.uw_output_dir, f"frames_{DIM}d")
PANEL = (760, 760)

INK = "#1b242c"
FAULT_RED = "#c0392b"
ACCENT = "#b03a2e"          # the failure envelope, and faults that slipped
# fixed for every frame: the fault turns, the window does not
CAMERA = [(2.6, 1.7, 2.1), tuple(np.full(3, 0.5)), (0.0, 1.0, 0.0)]


def samples():
    found = sorted(glob.glob(os.path.join(RUN_DIR, "sample_*", "probe.npz")))
    return [(int(os.path.basename(os.path.dirname(f)).split("_")[1]), f)
            for f in found]


def background_invariant(stress):
    """tau_II of the imposed far-field deviatoric stress."""
    dev = stress - np.trace(stress) / DIM * np.eye(DIM)
    return float(np.sqrt(0.5 * np.sum(dev * dev)))


def perturbation_variable(mesh, velocity, tau_background, tag):
    """Delta tau_II as a P1 field on ``mesh``, L2-PROJECTED.

    The invariant of a P2 velocity gradient is discontinuous between cells, so
    sampling it at vertices takes whichever cell the point-location happened to
    find and speckles the picture (measured: visible noise across the whole
    refined band). Projecting it is the honest reduction to P1.
    """
    x = mesh.X
    grad = sympy.Matrix(DIM, DIM,
                        lambda i, j: velocity.sym[i].diff(x[j]))
    edot = (grad + grad.T) / 2
    dev = 2 * ETA * (edot - edot.trace() / DIM * sympy.eye(DIM))
    tau_ii = sympy.sqrt(sum(dev[i, j] ** 2
                            for i in range(DIM) for j in range(DIM)) / 2)

    field = uw.discretisation.MeshVariable(f"dtau{tag}", mesh, 1, degree=1)
    projection = uw.systems.Projection(mesh, field)
    projection.uw_function = tau_ii - tau_background
    projection.smoothing = 0.0
    projection.solve()
    return field


def distance_to_fault(points, normal, radius):
    """Distance from each point to the fault SURFACE — the segment in 2-D,
    the disc in 3-D (not to its edge)."""
    rel = points[:, :DIM] - CENTRE
    across = rel @ normal
    in_plane = np.linalg.norm(rel - np.outer(across, normal), axis=1)
    return np.hypot(across, np.maximum(in_plane - radius, 0.0))


def robust_span(field, normal, radius):
    """Colour half-range, measured a couple of cells OFF the fault.

    A split-node fault is a velocity discontinuity, so the slip is taken up
    across one cell and the invariant in that layer is orders above the field
    the picture is about. Measured (2-D sample, h = 0.0125): median |dtau| is
    1.0e-1 within one cell of the fault and 2.7 at its worst, 5.5e-3 one cell
    out, 2.8e-4 in the far field — four decades. A percentile over everything
    therefore sets a range that hides the lobes entirely (+-0.52 measured).
    The near-fault layer is excluded from the RANGE only; it is still drawn,
    and saturates, which is what a discontinuity should look like.
    """
    d = distance_to_fault(field.coords, normal, radius)
    # a SHELL around the fault, not the whole exterior: the far field is orders
    # BELOW the near field just as the fault layer is orders above it, so a
    # percentile over everything outside the fault is set by the far field and
    # saturates the part worth looking at (3-D, measured: +-3.8e-4 against
    # near-field values around 1e-3).
    shell = (d > GUARD) & (d < 6.0 * H_NEAR)
    return 1.5 * float(np.percentile(np.abs(field.array[:, 0, 0])[shell], 95.0))


def fault_geometry(normal, radius):
    """The fault as PolyData: a segment in 2-D, a disc in 3-D."""
    e1 = np.cross(normal, [1.0, 0.0, 0.0]) if DIM == 3 else None
    if DIM == 2:
        tangent = np.array([-normal[1], normal[0]])
        ends = np.array([np.append(CENTRE - radius * tangent, 0.0),
                         np.append(CENTRE + radius * tangent, 0.0)])
        return pv.Line(ends[0], ends[1])
    if np.linalg.norm(e1) < 1e-8:
        e1 = np.cross(normal, [0.0, 1.0, 0.0])
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    a = np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False)
    rim = CENTRE + radius * (np.outer(np.cos(a), e1) + np.outer(np.sin(a), e2))
    pts = np.vstack([CENTRE, rim])
    faces = np.hstack([[3, 0, 1 + i, 1 + (i + 1) % len(a)]
                       for i in range(len(a))])
    return pv.PolyData(pts, faces)


def field_panel(mesh, field, normal, radius, clim, path, cut_normal):
    pv_field = vis.meshVariable_to_pv_mesh_object(field)
    values = field.array[:, 0, 0].copy()
    guard = params.uw_mask_cells * H_NEAR
    if guard > 0.0:
        # NOT drawn rather than merely out of range: the layer that takes up
        # the slip is mesh-scale, lumpy, and orders above the field it sits in,
        # so leaving it in the picture is the whole reason the lobes are hard
        # to see. The fault itself is drawn as geometry instead.
        values[distance_to_fault(field.coords, normal, radius) < guard] = np.nan
    pv_field.point_data["dtau"] = values
    edges = vis.mesh_to_pv_mesh(mesh).extract_all_edges()

    pl = pv.Plotter(off_screen=True, window_size=PANEL)
    pl.set_background("white")
    if DIM == 2:
        pl.add_mesh(pv_field, scalars="dtau", cmap="RdBu_r", clim=clim,
                    show_edges=False, lighting=False, nan_opacity=0.0,
                    scalar_bar_args=dict(title="delta tau_II", color=INK,
                                         vertical=True, position_x=0.86,
                                         width=0.05, height=0.6))
        pl.add_mesh(edges, color="#ccd4da", line_width=0.3, lighting=False)
        pl.add_mesh(fault_geometry(normal, radius), color=FAULT_RED,
                    line_width=5, lighting=False)
        pl.view_xy()
    else:
        bar = dict(title="delta tau_II", color=INK, vertical=True,
                   position_x=0.86, width=0.05, height=0.6)
        if params.uw_volume:
            # VOLUME render. The perturbation is a three-dimensional pattern of
            # lobes and a cut plane shows one slice through it. The volume
            # mapper takes regular grids only, so the P1 field is sampled onto
            # one; the opacity curve is CLEAR at zero, so undisturbed material
            # renders as nothing and what remains visible is the fault's own
            # signal. Points off the mesh sample as zero and so stay clear too.
            n = int(params.uw_volume_res)
            grid = pv.ImageData(dimensions=(n, n, n),
                                spacing=(1.0 / (n - 1),) * 3,
                                origin=(0.0, 0.0, 0.0))
            # Opacity kept LOW, so the volume reads as haze. The perturbation
            # envelops the fault, so at 0.9 (and at 0.6) the fault sat inside
            # opaque material and could not be seen at all — and where the
            # fault is is the one thing the frame has to show.
            sampled = grid.sample(pv_field)
            # CUT AWAY the half nearest the camera. The perturbation wraps the
            # fault, so with the full volume the fault sits inside the haze and
            # cannot be seen at any opacity (measured at 0.9, 0.6 and 0.32).
            # Zeroing the scalars there rather than clipping the geometry keeps
            # this a regular grid — what the volume mapper actually wants —
            # and zero is clear under the opacity curve below.
            towards_camera = np.array(CAMERA[0]) - CENTRE
            towards_camera /= np.linalg.norm(towards_camera)
            near = (sampled.points - CENTRE) @ towards_camera > 0.0
            values = np.asarray(sampled.point_data["dtau"]).copy()
            values[near] = 0.0
            guard = params.uw_mask_cells * H_NEAR
            if guard > 0.0:
                # and the slip layer itself: mesh-scale, lumpy and orders above
                # the field around it, it is what fills the volume with
                # saturated material and hides the lobes
                values[distance_to_fault(sampled.points, normal,
                                         radius) < guard] = 0.0
            sampled.point_data["dtau"] = values
            pl.add_volume(sampled, scalars="dtau", clim=clim,
                          cmap="RdBu_r", opacity=[0.55, 0.14, 0.0, 0.14, 0.55],
                          shade=False, scalar_bar_args=bar)
        else:
            # ONE fixed cut plane for every frame (the sigma1-sigma3 plane), so
            # the animation shows the fault moving through a fixed window, not
            # the window moving. Sampled onto a regular plane rather than
            # slicing cells: the mesh carries only ~8 cells across the disc and
            # a raw slice renders as blocky per-cell patches. Points off the
            # mesh are dropped by the sampler's own validity mask.
            plane = pv.Plane(center=tuple(CENTRE), direction=tuple(cut_normal),
                             i_size=1.45, j_size=1.45,
                             i_resolution=420, j_resolution=420)
            slice_ = plane.sample(pv_field).threshold(
                0.5, scalars="vtkValidPointMask")
            pl.add_mesh(slice_, scalars="dtau", cmap="RdBu_r", clim=clim,
                        show_edges=False, lighting=False, scalar_bar_args=bar)
        pl.add_mesh(vis.mesh_to_pv_mesh(mesh).outline(), color="#8c979f",
                    line_width=1.2, lighting=False)
        # the disc, and its RIM as a bold line: inside a volume render a
        # translucent surface alone cannot be picked out, and where the fault
        # IS is the one thing the frame has to show
        disc = fault_geometry(normal, radius)
        pl.add_mesh(disc, color=FAULT_RED, opacity=0.75, lighting=False,
                    show_edges=False)
        # the rim as a TUBE, not a line: a line width is a screen-space hint
        # that volume material in front of it composites away, while a tube is
        # real geometry and survives
        rim = disc.extract_feature_edges(boundary_edges=True,
                                         feature_edges=False,
                                         manifold_edges=False,
                                         non_manifold_edges=False)
        # dark, not red: the haze is red and blue, so the rim needs a colour
        # neither of them carries
        pl.add_mesh(rim.tube(radius=0.005), color="#101418", lighting=False)
        pl.camera_position = CAMERA
    pl.camera.zoom(1.25)
    pl.screenshot(path)
    pl.close()


def mohr_panel(records, k, path, sigmas):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(PANEL[0] / 170, PANEL[1] / 170), dpi=170,
                           constrained_layout=True)
    t = np.linspace(0.0, 2.0 * np.pi, 361)

    def circle(a, b, lw):
        c, r = 0.5 * (a + b), 0.5 * abs(a - b)
        ax.plot(c + r * np.cos(t), r * np.sin(t), color=INK, lw=lw)

    span = 0.5 * (sigmas[0] - sigmas[-1])
    ax.set_xlim(sigmas[-1] - 0.12 * span, sigmas[0] + 0.12 * span)
    ax.set_ylim(-1.18 * span, 1.18 * span)

    friction = max(r["friction"] for r in records)
    cohesion = max(r["cohesion"] for r in records)
    if friction > 0.0 or cohesion > 0.0:
        # The FORBIDDEN zone: beyond tau = C + mu (-sigma_n) the fault cannot
        # hold, so it slips until it is back on the envelope. No point can sit
        # above this line, which is the whole point of the figure. Cohesion
        # lifts the envelope, so fewer orientations reach it, and it carries
        # strength into mild tension (to sigma = -C/mu) instead of losing it
        # the moment the normal stress turns tensile.
        g = np.linspace(*ax.get_xlim(), 400)
        envelope = np.maximum(cohesion + friction * (-g), 0.0)
        for sign in (1, -1):
            ax.fill_between(g, sign * envelope, sign * 1.18 * span,
                            where=(sign * envelope <= sign * 1.18 * span),
                            color="#f3e2de", lw=0, zorder=0)
            ax.plot(g, sign * envelope, color=ACCENT, lw=1.2, zorder=3)

    if DIM == 3:
        circle(sigmas[0], sigmas[1], 0.8)
        circle(sigmas[1], sigmas[2], 0.8)
    circle(sigmas[0], sigmas[-1], 1.1)
    done = [r for r in records if r["sample"] <= k]
    for slipping, colour, size in ((False, "#5a6670", 9), (True, ACCENT, 13)):
        pts = [r for r in done if r["sliding"] == slipping]
        if pts:
            ax.scatter([r["sigma_n"] for r in pts], [r["tau"] for r in pts],
                       s=size, color=colour, alpha=0.65, lw=0, zorder=4)
    here = [r for r in records if r["sample"] == k][0]
    ax.scatter([here["sigma_n"]], [here["tau"]], s=90, facecolor="none",
               edgecolor=FAULT_RED, lw=2.0, zorder=5)
    ax.axhline(0.0, color="#d9dfe3", lw=0.6)
    ax.set_xlabel("normal stress $\\sigma_n$ (compression positive)")
    ax.set_ylabel("shear traction $\\tau$ (signed)")
    ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def compose(left, right, out):
    from PIL import Image

    a, b = Image.open(left), Image.open(right)
    h = max(a.height, b.height)
    canvas = Image.new("RGB", (a.width + b.width, h), "white")
    canvas.paste(a, (0, (h - a.height) // 2))
    canvas.paste(b, (a.width, (h - b.height) // 2))
    canvas.save(out)


def signed_tau(record, normal):
    """The signed shear the Mohr panel plots (as fault_mohr_plot does)."""
    axes = record["principal_axes"]
    n_p = axes.T @ normal
    return float(np.linalg.norm(record["tau"]) * np.sign(n_p[0] * n_p[-1]))


os.makedirs(FRAME_DIR, exist_ok=True)
entries = samples()
if not entries:
    raise SystemExit(f"no checkpoints under {RUN_DIR}")

records = []
for k, f in entries:
    r = np.load(f, allow_pickle=True)
    records.append(dict(sample=k, sigma_n=float(r["sigma_n"]),
                        tau=signed_tau(r, r["normal"]), normal=r["normal"],
                        stress=r["stress"], radius=float(r["radius"]),
                        friction=float(r["friction"]) if "friction" in r else 0.0,
                        cohesion=float(r["cohesion"]) if "cohesion" in r else 0.0,
                        h_near=(float(r["resolution_h_near"])
                                if "resolution_h_near" in r else 0.0),
                        sliding=bool(r["sliding"]) if "sliding" in r else False,
                        principal_axes=r["principal_axes"], path=os.path.dirname(f)))

# The mask width and the colour-range shell are both measured in CELLS, so
# they must use the cell size the sweep actually ran at — a finer run would
# otherwise be masked by the old default and scaled from the wrong shell.
if records[0]["h_near"] > 0.0:
    H_NEAR = records[0]["h_near"]
    GUARD = 2.0 * H_NEAR
    uw.pprint(f"resolution read from the run: h_near {H_NEAR}")

stress = records[0]["stress"]
sigmas = np.sort(np.linalg.eigvalsh(stress))[::-1]
cut_normal = None
if DIM == 3:
    # the sigma_2 direction: the plane holding sigma_1 and sigma_3
    order = np.argsort(np.linalg.eigvalsh(stress))[::-1]
    cut_normal = np.linalg.eigh(stress)[1][:, order[1]]

wanted = ([r for r in records if r["sample"] == params.uw_sample]
          if params.uw_sample >= 0 else records[::params.uw_stride])
if params.uw_max_frames:
    wanted = wanted[:params.uw_max_frames]
if params.uw_n_shards > 1:
    if params.uw_clim <= 0.0:
        raise SystemExit("sharded rendering needs -uw_clim: the range is taken "
                         "from the first frame a process renders, so shards "
                         "would disagree and the animation would flicker")
    wanted = wanted[params.uw_shard::params.uw_n_shards]

clim = (-params.uw_clim, params.uw_clim) if params.uw_clim > 0 else None
frames = []
for r in ([] if params.uw_gif_only else wanted):
    k = r["sample"]
    mesh = uw.discretisation.Mesh(os.path.join(r["path"], "fault.mesh.00000.h5"))
    velocity = uw.discretisation.MeshVariable(f"V{k}", mesh, DIM, degree=2)
    velocity.read_timestep("fault", f"V{k}", 0, outputPath=r["path"])
    field = perturbation_variable(mesh, velocity,
                                  background_invariant(r["stress"]), k)
    if clim is None:
        # ONE range for the whole animation, from the first frame's spread
        span = robust_span(field, r["normal"], r["radius"])
        clim = (-span, span)
    left = os.path.join(FRAME_DIR, f"_field_{k:04d}.png")
    right = os.path.join(FRAME_DIR, f"_mohr_{k:04d}.png")
    out = os.path.join(FRAME_DIR, f"frame_{k:04d}.png")
    field_panel(mesh, field, r["normal"], r["radius"], clim, left, cut_normal)
    mohr_panel(records, k, right, sigmas)
    compose(left, right, out)
    os.remove(left)
    os.remove(right)
    frames.append(out)
    uw.pprint(f"[frame {k:4d}] clim +-{clim[1]:.2e}  -> {out}", flush=True)

if params.uw_animate:
    from PIL import Image

    # from DISK, not from this process's own frames: the shards each render a
    # subset, and the animation is all of them in sample order
    on_disk = sorted(glob.glob(os.path.join(FRAME_DIR, "frame_*.png")))
    on_disk = on_disk[::max(1, int(params.uw_gif_stride))]
    images = [Image.open(f) for f in on_disk]
    if params.uw_gif_scale != 1.0:
        size = (int(images[0].width * params.uw_gif_scale),
                int(images[0].height * params.uw_gif_scale))
        images = [im.resize(size, Image.LANCZOS) for im in images]
    gif = os.path.join(params.uw_output_dir, f"stress-tumble-{DIM}d.gif")
    images[0].save(gif, save_all=True, append_images=images[1:] + [images[-1]] * 8,
                   duration=110, loop=0, optimize=True)
    uw.pprint(f"wrote {gif} ({len(images)} frames)")
