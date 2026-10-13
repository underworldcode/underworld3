"""House style for underworld3 line diagrams (matplotlib).

One module, imported by every schematic (docstring figures, notebook figures,
benchmark meshes, article figures) so that a wall, a streamline, a velocity
arrow or a control-volume boundary looks the same in all of them. The roles
and weights follow ``line-drawing-conventions.md`` in this skill; the CeTZ
block ``uwfig.typ`` carries the same roles for Typst figures.

Usage::

    from uwfig import *          # or: import uwfig as F
    fig, ax = figure(width_mm=183, aspect=0.38)
    draw_mesh(ax, mesh)                       # thin light triangles
    boundary(ax, [(0, 0), (W, 0), (W, H), (0, H), (0, 0)])
    inflow_profile(ax, x0=0.0, y0=0.0, y1=H, umax=1.5, side="left")
    label(ax, W / 2, H + 0.07, "no-slip")
    save(fig, "figures/mesh_channel")         # .pdf + .svg + .png

Everything is a thin wrapper over matplotlib: pass extra keyword arguments
through to override a role for one element, never redefine a role locally.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon

__all__ = [
    "INK", "MESH", "VELOCITY", "FORCE", "ENERGY", "DIMENSION", "TINT_FLUID", "TINT_SOLID",
    "LINE", "ARROW", "line_style", "arrow_style",
    "figure", "mesh_triangles", "draw_mesh", "boundary", "circle_body", "wall",
    "arrow", "flow_profile", "inflow_profile", "uniform_arrows", "dimension", "axes_glyph", "label", "leader", "save",
]

# --------------------------------------------------------------------------- colour tokens
# Colour is a SECONDARY channel: every role is also fixed by weight and dash pattern, and
# each arrow class has its own head and weight, so a grayscale print keeps the meaning.
INK = "#1f2a33"          # walls, bodies, text, dimensions
MESH = "#9aa3aa"         # mesh edges: light, never heavier than 0.35 pt
VELOCITY = "#1b6f6a"     # motion: velocity vectors, inflow/outflow profiles, streamlines
FORCE = "#a1522a"        # forces, pressure, tractions, applied stress
ENERGY = "#6b3d8f"       # heat, work, energy flux
DIMENSION = INK
TINT_FLUID = "#eef3f6"   # optional flat tint for a fluid region
TINT_SOLID = "#e4e0da"   # optional flat tint for a solid region (with hatching)

# --------------------------------------------------------------------------- line roles
# weights in points AT FINAL SIZE (90 mm single column / 183 mm double column); three
# weights only: 1.2 (physical boundary), 0.8 (interface, hidden, control volume), 0.5 (thin)
LINE = {
    "boundary":   dict(lw=1.2, ls="-", color=INK),                      # walls, bodies, solid surfaces
    "interface":  dict(lw=0.8, ls="-", color=INK),                      # free surface, material interface
    "streamline": dict(lw=0.5, ls="-", color=VELOCITY),                 # streamlines, contours, construction
    "hidden":     dict(lw=0.8, ls=(0, (4, 2)), color=INK),              # hidden / idealised geometry
    "cv":         dict(lw=0.8, ls=(0, (5, 2, 1, 2)), color=INK),        # control volume, computational domain
    "axis":       dict(lw=0.5, ls=(0, (8, 2, 1, 2)), color=INK),        # centre line, axis of symmetry
    "mesh":       dict(lw=0.25, ls="-", color=MESH),                    # mesh edges
    "leader":     dict(lw=0.5, ls="-", color=INK),                      # label leaders
}

# --------------------------------------------------------------------------- arrow classes
# ``mutation_scale`` is the head size in points; the head grows with the line weight.
ARROW = {
    "velocity":  dict(arrowstyle="-|>", lw=0.9, mutation_scale=7, color=VELOCITY),
    "force":     dict(arrowstyle="-|>", lw=1.3, mutation_scale=9, color=FORCE),
    "energy":    dict(arrowstyle="-|>", lw=1.1, mutation_scale=8, color=ENERGY),
    "dimension": dict(arrowstyle="<|-|>", lw=0.5, mutation_scale=5, color=DIMENSION),
    "leader":    dict(arrowstyle="-", lw=0.5, color=INK),
}


def line_style(role: str, **override) -> dict:
    """Keyword arguments for ``ax.plot`` / patches in one line role."""
    return {**LINE[role], **override}


def arrow_style(kind: str, **override) -> dict:
    """``arrowprops`` for ``ax.annotate`` in one arrow class."""
    return {**ARROW[kind], **override}


# --------------------------------------------------------------------------- rcParams
# One sans-serif face for every figure, text kept as text in PDF and SVG (editable, searchable),
# 8 pt at final size so that nothing falls below 6 pt after a modest reduction.
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "Helvetica Neue", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8,
    "mathtext.fontset": "dejavusans",
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

MM = 1 / 25.4


def figure(width_mm: float = 90.0, aspect: float = 0.6, **kw):
    """A figure drawn at the final column width: 90 mm single, 183 mm double.

    ``aspect`` is height / width. The axes fill the figure, keep equal aspect and hide the
    frame: a schematic has no axes of its own.
    """
    fig = plt.figure(figsize=(width_mm * MM, width_mm * aspect * MM), **kw)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


# --------------------------------------------------------------------------- meshes
def mesh_triangles(mesh):
    """(points[:, :2], triangles) of an underworld3 mesh, through the PyVista bridge."""
    from underworld3 import visualisation as vis
    pv = vis.mesh_to_pv_mesh(mesh)
    if hasattr(pv, "cells_dict") and 5 in pv.cells_dict:            # VTK_TRIANGLE
        return np.asarray(pv.points)[:, :2], np.asarray(pv.cells_dict[5])
    surf = pv.extract_surface().triangulate()
    return np.asarray(surf.points)[:, :2], np.asarray(surf.faces).reshape(-1, 4)[:, 1:]


def draw_mesh(ax, mesh_or_tris, **override):
    """Thin, light mesh edges. Pass a mesh or a (points, triangles) pair.

    The mesh is context, not the subject: keep it under the boundary lines and never
    heavier than the ``mesh`` role. Zoom insets may raise ``lw`` to 0.35.
    """
    pts, tris = mesh_or_tris if isinstance(mesh_or_tris, tuple) else mesh_triangles(mesh_or_tris)
    ax.triplot(pts[:, 0], pts[:, 1], tris, **line_style("mesh", **override))
    return pts, tris


# --------------------------------------------------------------------------- geometry
def boundary(ax, xy, role: str = "boundary", closed: bool = True, fill=None, **override):
    """A polyline in a line role (``boundary``, ``interface``, ``hidden``, ``cv``, ``axis``)."""
    xy = np.asarray(xy, float)
    if closed and not np.allclose(xy[0], xy[-1]):
        xy = np.vstack([xy, xy[:1]])
    st = line_style(role, **override)
    if fill is not None:
        ax.add_patch(Polygon(xy, closed=True, fc=fill, ec="none", zorder=0))
    ax.plot(xy[:, 0], xy[:, 1], **st)


def circle_body(ax, centre, radius, role: str = "boundary", fill=None, **override):
    """A circular body (cylinder cross-section) drawn in a line role."""
    st = line_style(role, **override)
    ax.add_patch(Circle(centre, radius, fill=fill is not None, fc=fill or "none",
                        ec=st["color"], lw=st["lw"], ls=st["ls"]))


def wall(ax, xy0, width, height, hatch: str = "///", **override):
    """A hatched solid wall (45 degree thin lines): the engineering-drawing convention."""
    st = line_style("boundary", **override)
    ax.add_patch(Rectangle(xy0, width, height, fc=TINT_SOLID, ec=st["color"], lw=st["lw"],
                           hatch=hatch, zorder=1))


# --------------------------------------------------------------------------- arrows
def arrow(ax, tail, head, kind: str = "velocity", **override):
    """One arrow of a class: ``velocity``, ``force``, ``energy``, ``dimension``."""
    ax.annotate("", xy=head, xytext=tail, arrowprops=arrow_style(kind, **override))


def flow_profile(ax, p0, p1, umax, inward, into: bool = True, n: int = 9, scale: float = 1.0,
                 kind: str = "velocity", **override):
    """A parabolic velocity profile on the edge p0 -> p1, drawn outside the domain.

    ``inward`` is a vector pointing from the edge into the domain. The parabola bulges
    away from the domain; the arrows follow the flow: from outside onto the edge for an
    inlet (``into=True``), from the edge outwards for an outlet (``into=False``).
    ``scale`` converts velocity to drawing units. Draw the profile instead of writing
    it: the parabola says "Poiseuille" on its own.
    """
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    nrm = -np.asarray(inward, float); nrm = nrm / (np.linalg.norm(nrm) or 1.0)   # outward unit normal
    s = np.linspace(0, 1, 200)
    u = 4 * umax * s * (1 - s) * scale
    pts = p0 + np.outer(s, p1 - p0) + np.outer(u, nrm)
    st = arrow_style(kind, **override)
    ax.plot(pts[:, 0], pts[:, 1], color=st["color"], lw=st["lw"])
    for si in np.linspace(0, 1, n + 2)[1:-1]:
        e = p0 + si * (p1 - p0); o = e + 4 * umax * si * (1 - si) * scale * nrm
        tail, head = (o, e) if into else (e, o)
        arrow(ax, tail, head, kind, lw=st["lw"] * 0.8, **override)


def inflow_profile(ax, x0, y0, y1, umax, side: str = "left", **kw):
    """``flow_profile`` on a vertical edge x = x0: ``side="left"`` is an inlet on the left
    of the domain, ``side="right"`` an outlet on the right."""
    if side == "left":
        flow_profile(ax, (x0, y0), (x0, y1), umax, inward=(1, 0), into=True, **kw)
    else:
        flow_profile(ax, (x0, y0), (x0, y1), umax, inward=(-1, 0), into=False, **kw)


def uniform_arrows(ax, x0, x1, ys, kind: str = "force", **override):
    """A row of equal arrows from x0 to x1 at heights ``ys`` (a body force, an applied stress)."""
    for y in ys:
        arrow(ax, (x0, y), (x1, y), kind, **override)


def dimension(ax, p0, p1, text: str = "", offset: float = 0.0, **override):
    """A double-headed dimension line between p0 and p1, shifted normal to itself by ``offset``."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0; nrm = np.array([-d[1], d[0]]) / (np.linalg.norm(d) or 1.0)
    q0, q1 = p0 + offset * nrm, p1 + offset * nrm
    arrow(ax, q0, q1, "dimension", **override)
    if text:
        m = (q0 + q1) / 2 + 0.02 * nrm * np.sign(offset or 1.0) * np.hypot(*d)
        label(ax, m[0], m[1], text)


def axes_glyph(ax, origin, length: float, labels=("$x$", "$y$")):
    """A small coordinate frame so the reader never has to infer it."""
    ox, oy = origin
    for (dx, dy), t, ha, va in (((length, 0), labels[0], "left", "center"),
                                ((0, length), labels[1], "center", "bottom")):
        ax.annotate("", xy=(ox + dx, oy + dy), xytext=(ox, oy),
                    arrowprops=dict(arrowstyle="-|>", lw=0.6, mutation_scale=6, color=INK))
        ax.text(ox + dx * 1.15, oy + dy * 1.15, t, ha=ha, va=va, fontsize=8, color=INK)


# --------------------------------------------------------------------------- text
def label(ax, x, y, text, ha="center", va="center", rot=0, color=INK, fontsize=8, box=True, **kw):
    """A direct label on the drawing. Symbols in mathtext (``$u$``); no caption material."""
    bbox = dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85) if box else None
    ax.text(x, y, text, ha=ha, va=va, rotation=rot, color=color, fontsize=fontsize, bbox=bbox, **kw)


def leader(ax, text_xy, target_xy, text: str = "", ha="left", color=INK):
    """A label joined to its feature by a thin leader line."""
    ax.annotate("", xy=target_xy, xytext=text_xy, arrowprops=arrow_style("leader", color=color))
    if text:
        label(ax, text_xy[0], text_xy[1], text, ha=ha, color=color)


# --------------------------------------------------------------------------- output
def save(fig, stem: str, formats=("pdf", "svg", "png"), dpi: int = 300, close: bool = True):
    """Write the figure as vector (PDF, SVG) plus a PNG at ``dpi``. ``stem`` has no extension."""
    os.makedirs(os.path.dirname(os.path.abspath(stem)), exist_ok=True)
    for ext in formats:
        fig.savefig(f"{stem}.{ext}", dpi=dpi if ext == "png" else None)
    if close:
        plt.close(fig)
    return [f"{stem}.{ext}" for ext in formats]
