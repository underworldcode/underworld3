"""Fault variant of the adaptive-convection workflow (option 2: uniform base).

Adds a static dipping low-viscosity fault to the stagnant-lid annulus
convection workflow, with **anisotropic-tensor** mesh adaptation that
refines ACROSS the fault normal. This is the *uniform-base* variant: it
measures what the mmpde mover can achieve from a uniform mesh (the
"creation cap"), re-taken on the FIXED mover base. A genuinely resolved,
maintained fault needs a gmsh line-refinement base (``refine_lines``) —
scoped as a follow-up that reuses the same metric.

What differs from the no-fault ``config.py``:

  * weak zone — isotropic Frank-Kamenetskii viscosity floored in a
    gaussian band around the fault: ``η = η_FK·(1−f) + floor·f`` with
    ``f = exp(−(d/w)²)`` the fault influence (P1, so positivity holds).
  * anisotropic SPD tensor metric — ``M = ρ·I + (Rf²−1)·exp(−(d/wₙ)²)·n nᵀ``
    (thin ACROSS the fault normal n) so a node row lands ON the
    centre-line; a scalar bump would refine a fat isotropic corridor and
    leave the centre coarse.
  * direct unsigned distance ``d`` (geometry_tools), refreshed on the
    moved nodes — NOT the signed Surface distance (which bleeds the
    metric along the fault's line extension).
  * the tensor breaks the mover's scalar skip-check, so the skip decision
    is made here from the scalar-ρ misalignment.

Diagnostic of record: the fault/bulk nearest-neighbour spacing RATIO
(``diagnostics.nn_spacing_ratios`` with the fault polyline) — < 1 means
finer than bulk. Misalignment is global and useless for a thin feature.

Run via ``fault_simulate.py`` or from Python:

    import fault_config as fc
    from underworld3.workflows import WorkflowRunner
    cfg = fc.FaultConvectionConfig(output_dir="...", max_steps=60)
    WorkflowRunner(fc, cfg).build("run_summary")
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
    RUN_NAME, Run, config_cache_key, config_snapshot, workflow_step,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diagnostics as diag  # noqa: E402
# Reuse the no-fault config base + the generic summary step + IC writer.
from config import (  # noqa: E402,F401
    AdaptiveConvectionConfig, _is_steady, _write_ic, summarise_run)


_FAULT_IDENTITY = (
    "fault_theta_deg", "fault_dip_deg", "fault_depth", "fault_dip_dir",
    "fault_profile", "fault_width", "fault_edge", "fault_floor",
    "fault_anisotropy", "fault_base_smin",
    "fault_wedge", "metric_combine",
    "fault_metric_side", "fault_rheology_side", "rheology",
    "blob_enable", "blob_theta_deg", "blob_radius", "blob_size",
    "blob_edge", "blob_floor",
)


def view():
    from underworld3.workflows import view as _wf_view
    _wf_view(sys.modules[__name__])


class FaultConvectionConfig(AdaptiveConvectionConfig):
    """No-fault config + a static dipping weak fault and its anisotropic metric."""

    workflow_name: str = "fault_convection"
    description: str = (
        "Adaptive-mesh stagnant-lid convection with a static dipping "
        "low-viscosity fault (anisotropic tensor metric, uniform base)"
    )

    # Extend the identity set with the fault geometry / strength / metric.
    _identity_fields = AdaptiveConvectionConfig._identity_fields + _FAULT_IDENTITY

    # Fault geometry
    fault_theta_deg: float = Field(default=90.0,
                                   description="Azimuth of the fault surface "
                                               "trace (90 = 12 o'clock).")
    fault_dip_deg: float = Field(default=30.0, gt=0, lt=90)
    fault_depth: float = Field(default=0.225, gt=0,
                               description="Radial depth of the fault tip "
                                           "below the outer surface.")
    fault_dip_dir: Literal["east", "west"] = "east"

    # Fault rheology
    rheology: Literal["isotropic", "ti"] = Field(
        default="isotropic",
        description="Weak-zone rheology. 'isotropic' = whole viscosity floored "
                    "in the band (linear, cheap, ksponly). 'ti' = "
                    "transverse-isotropic (weak only to fault-parallel shear, "
                    "director=fault normal) — the real fault, ~20× costlier "
                    "(intrinsic Picard), needs penalty free slip + GAMG.")

    # Fault strength (weak zone)
    fault_profile: Literal["tophat", "gaussian"] = Field(
        default="tophat",
        description="Weak-zone influence shape. 'tophat' = a coherent band "
                    "where f≈1 (so η_1 genuinely reaches the floor); 'gaussian' "
                    "= the legacy bump (f peaks at 0.5 one-sided — never reaches "
                    "the floor in a stiff lid).")
    fault_width: float = Field(default=0.06, gt=0,
                               description="Weak-zone band thickness into the "
                                           "hanging wall (tophat) / gaussian "
                                           "half-width (gaussian), model coords.")
    fault_edge: float = Field(default=0.02, gt=0,
                              description="Smoothing width of the tophat band "
                                          "edges (model coords).")
    fault_floor: float = Field(default=1.0, gt=0,
                               description="Viscosity floor in the fault zone — "
                                           "the TI fault-parallel viscosity η_1 "
                                           "genuinely reaches this where f≈1 "
                                           "(geometric blend η_1=η_FK^(1−f)·floor^f).")

    # Fault-aware refinement
    fault_anisotropy: float = Field(default=8.0, ge=0,
                                    description="Anisotropic metric ratio Rf "
                                                "(thin across the normal). "
                                                "0 = scalar fallback.")
    fault_refine_amp: float = Field(default=25.0, ge=0,
                                    description="Isotropic fault density "
                                                "amplitude. Must exceed the |∇T| "
                                                "density (~R^d) or the thermal BL "
                                                "out-competes the fault near the "
                                                "surface (refinement drifts ABOVE "
                                                "the fault). ~25 centres it at R=5.")
    fault_refine_width: float = Field(default=-1.0,
                                      description="Sharp across-fault band "
                                                  "half-width (<0 ⇒ 1.5·width).")
    fault_draw_width: float = Field(default=-1.0,
                                    description="Wide isotropic draw half-width "
                                                "(<0 ⇒ 4·refine_width).")
    metric_combine: Literal["product", "max"] = Field(
        default="max",
        description="How the fault density fuses with the |∇T| density. "
                    "'product' lets the thermal BL out-compete the fault "
                    "along its length (pulls nodes ABOVE the fault); 'max' "
                    "keeps the fault refined uniformly along its whole "
                    "length (default).")
    # One-sided fault influence. The symmetric (both-sides) metric demands
    # refinement on both flanks, but the thermal-BL + wedge pulls bias the
    # realized nodes to the UPPER (hanging-wall) side and starve the footwall.
    # A one-sided influence removes the symmetric demand (and is physically
    # faithful — fault damage zones are typically one-sided).
    fault_metric_side: Literal["both", "upper", "lower"] = Field(
        default="both",
        description="Which side of the fault PLANE the refinement metric acts "
                    "on. 'lower' biases refinement to the footwall (counter the "
                    "upward pull); 'upper' to the hanging wall.")
    fault_rheology_side: Literal["both", "upper", "lower"] = Field(
        default="both",
        description="Which side of the fault PLANE the weak zone acts on. "
                    "'upper' = a one-sided hanging-wall damage zone.")
    fault_side_width: float = Field(
        default=-1.0,
        description="Transition half-width of the one-sided gate "
                    "(<0 ⇒ fault_width).")

    # gmsh fault-base refinement: place small cells along the fault trace at
    # construction so the mover MAINTAINS (rather than has to CREATE) the
    # refinement. 0 = uniform base (mover-only, the ~1.8x creation cap).
    fault_base_smin: float = Field(default=0.0, ge=0,
                                   description="gmsh cell size on the fault "
                                               "trace (0 = uniform base).")
    fault_wedge: bool = Field(
        default=True,
        description="Also gmsh-refine the radial WEDGE between the fault and "
                    "the surface. Gives the fault pull and the surface "
                    "thermal-BL pull each their own node budget so they don't "
                    "collide in the hanging-wall sliver. Needs fault_base_smin>0.")

    # Lid-mobility weak blob — a small ISOTROPIC low-viscosity square in the
    # cold thermal boundary layer (a "ridge"): weakens the rigid lid locally so
    # it can yield/mobilise above an upwelling. Separate from the TI fault;
    # applied to BOTH viscosity components (isotropic) via the same geometric
    # blend, expressed analytically in the coordinates (crisp square, no field).
    blob_enable: bool = Field(default=False,
                              description="Add the isotropic lid-mobility blob.")
    blob_theta_deg: float = Field(default=0.0,
                                  description="Azimuth of the blob centre "
                                              "(0 = 3 o'clock).")
    blob_radius: float = Field(default=0.97, gt=0,
                               description="Radial centre of the blob. A mid-"
                                           "ocean ridge sits AT the surface, so "
                                           "the default (0.97, with size 0.05) "
                                           "reaches r_outer=1; the out-of-domain "
                                           "part is clipped.")
    blob_size: float = Field(default=0.05, gt=0,
                             description="Half-width of the (x,y) square "
                                         "(model coords).")
    blob_edge: float = Field(default=0.012, gt=0,
                             description="Edge smoothing of the square.")
    blob_floor: float = Field(default=1.0, gt=0,
                              description="Isotropic viscosity inside the blob "
                                          "(geometric blend, ~1 = fully weak).")

    output_dir: str = "output/fault_convection/run"


# ---------------------------------------------------------------------------
# Identity / on-disk state
# ---------------------------------------------------------------------------
def _config_hash(config: FaultConvectionConfig) -> str:
    return config_cache_key(config, config._identity_fields)


def _identity_snapshot(config: FaultConvectionConfig) -> dict:
    return config_snapshot(config, config._identity_fields)


def _check_compatibility(config: FaultConvectionConfig, output_dir: Path):
    run = Run(output_dir)
    manifest = run.manifest
    if manifest is None:
        return "fresh", None
    if manifest.config_hash == _config_hash(config):
        return "continue", None
    if config.restart_policy == "error":
        raise RuntimeError(
            f"Config mismatch in {output_dir}; set restart_policy='fresh' or "
            "'seed_from_old' (both archive, never delete).")
    archive = run.archive()
    output_dir.mkdir(parents=True, exist_ok=True)
    return "fresh", archive


_TS_FIELDS = ("step", "t", "dt", "wall", "Nu", "vrms", "Tmin", "Tmax",
              "adapted", "misalign", "folded", "area_ratio", "aspect",
              "min_area", "bl_ratio", "fault_ratio", "n_fault")


# ---------------------------------------------------------------------------
# Fault geometry
# ---------------------------------------------------------------------------
def _blob_center(config):
    th0 = float(np.deg2rad(config.blob_theta_deg))
    return config.blob_radius * np.cos(th0), config.blob_radius * np.sin(th0)


def _blob_points(config, spacing):
    """Grid point cloud filling the blob square — embedded for gmsh base
    refinement so the small weak square is actually resolved. Points outside
    the annulus (r ≥ r_outer) are dropped so a surface-reaching ridge does not
    embed points beyond the domain boundary."""
    x0, y0 = _blob_center(config)
    hw = config.blob_size
    g = np.arange(-hw, hw + 1e-9, spacing)
    xs, ys = np.meshgrid(x0 + g, y0 + g)
    pts = np.column_stack([xs.ravel(), ys.ravel()])
    r = np.hypot(pts[:, 0], pts[:, 1])
    return pts[r < config.r_outer - 0.5 * spacing]


def _blob_influence_sym(X, config):
    """Smooth analytic (x,y) box ≈1 inside the blob square, 0 outside (two
    sigmoids per axis — no Abs, so JIT-safe). Shared by the rheology and the
    refinement metric."""
    x0, y0 = _blob_center(config)
    hw, e = config.blob_size, config.blob_edge
    half = sympy.Rational(1, 2)
    bx = (half * (1 + sympy.tanh((X[0] - (x0 - hw)) / e))
          * half * (1 + sympy.tanh(((x0 + hw) - X[0]) / e)))
    by = (half * (1 + sympy.tanh((X[1] - (y0 - hw)) / e))
          * half * (1 + sympy.tanh(((y0 + hw) - X[1]) / e)))
    return bx * by


def _fault_geometry(config: FaultConvectionConfig):
    """Return (polyline xy (N,2), normal unit vector n, n nᵀ matrix)."""
    delta = np.deg2rad(config.fault_dip_deg)
    theta0 = np.deg2rad(config.fault_theta_deg)
    P0 = np.array([np.cos(theta0), np.sin(theta0)])
    e_hat = np.array([np.cos(theta0), np.sin(theta0)])
    t_hat = np.array([-np.sin(theta0), np.cos(theta0)])
    side = 1.0 if config.fault_dip_dir == "east" else -1.0
    dhat = side * np.cos(delta) * t_hat - np.sin(delta) * e_hat
    L = config.fault_depth / np.sin(delta)
    s = np.linspace(0.0, L, 25)[:, None]
    xy = P0[None, :] + s * dhat[None, :]
    n_vec = np.array([-dhat[1], dhat[0]])
    n_vec /= np.linalg.norm(n_vec)
    nnT = sympy.Matrix([[n_vec[0] * n_vec[0], n_vec[0] * n_vec[1]],
                        [n_vec[1] * n_vec[0], n_vec[1] * n_vec[1]]])
    return xy, n_vec, nnT


def _wedge_points(xy, r_outer, smin, eps=1.0e-3):
    """Point cloud filling the radial WEDGE between the fault and the surface.

    For each fault point, sample the radial segment from just above it up to
    just below the surface at ~``smin`` spacing. The union over the fault
    trace fills the hanging-wall sliver, so a gmsh Distance field refines the
    whole wedge — giving the fault and the surface thermal BL their own node
    budgets instead of competing for the same coarse cells between them."""
    pts = []
    for p in np.asarray(xy)[:, :2]:
        r = float(np.hypot(p[0], p[1]))
        if r <= 0 or r >= r_outer - smin:
            continue
        rad = p / r
        for rr in np.arange(r + smin, r_outer - eps, smin):
            pts.append(rad * rr)
    return np.array(pts) if pts else np.zeros((0, 2))


def _upper_sign(xy):
    """Sign of the signed distance on the radially-OUTWARD (surface / hanging-
    wall) side of the fault — so a gate ``s·_upper_sign > 0`` selects the
    'upper' side regardless of the distance tool's orientation convention."""
    from underworld3.utilities.geometry_tools import (
        signed_distance_pointcloud_polyline_2d)
    mid = np.asarray(xy)[len(xy) // 2, :2]
    r = float(np.hypot(mid[0], mid[1]))
    probe = (mid / r) * (r + 0.03)            # step toward the surface
    s = float(signed_distance_pointcloud_polyline_2d(
        probe[None, :], np.asarray(xy)[:, :2])[0])
    return 1.0 if s >= 0 else -1.0


def _side_multiplier(side, upper_sign):
    """Multiplier m for the one-sided gate ``0.5(1+tanh(m·s/w))`` (1 on the
    selected side). ``None`` for the symmetric 'both' case (no gate)."""
    if side == "upper":
        return upper_sign
    if side == "lower":
        return -upper_sign
    return None


def _refresh_fault_fields(gfac, dfac, xy, config):
    """Sample the SIGNED distance + gaussian influence at the CURRENT
    (possibly moved) P1 nodes. Direct geometric distance — NOT the signed
    Surface field (which bleeds along the line extension).

    ``dfac`` stores the SIGNED distance (the metric gaussians square it, so
    they are unaffected, but the sign lets the metric gate to one side).
    ``gfac`` is the (optionally one-sided) rheology weak-zone influence."""
    from underworld3.utilities.geometry_tools import (
        signed_distance_pointcloud_polyline_2d)
    width = config.fault_width
    edge = config.fault_edge
    coords = np.asarray(gfac.coords)[:, :2]
    d = signed_distance_pointcloud_polyline_2d(coords, xy[:, :2])
    m = _side_multiplier(config.fault_rheology_side, _upper_sign(xy))
    if config.fault_profile == "tophat":
        # A coherent band where f≈1 (so the geometric blend drives η_1 to the
        # floor genuinely), with smooth edges of width `edge`.
        if m is not None:
            # one-sided hanging-wall band: f≈1 from the fault plane (s=0) out
            # to s=width into the selected side, suppressed in the footwall.
            s = m * d
            g = (0.5 * (1.0 + np.tanh((width - s) / edge))      # outer falloff
                 * 0.5 * (1.0 + np.tanh(s / edge + 1.5)))       # footwall cut (~1 at s=0)
        else:                                  # symmetric band on both flanks
            g = 0.5 * (1.0 + np.tanh((width - np.abs(d)) / edge))
    else:                                      # gaussian — PEAKED on the fault
        # f=1 ON the drawn fault (d=0), so the geometric blend makes η_1 weakest
        # exactly there and recover away. One-sided = a hanging-wall damage zone:
        # gaussian taper of half-width `width` INTO the selected side, sharp
        # `edge` recovery on the other (footwall strong just below the fault).
        # No gate halving — the peak stays on the fault.
        if m is not None:
            s = m * d                          # >0 on the selected (hanging-wall) side
            g = np.where(s >= 0.0, np.exp(-(s / width) ** 2),
                         np.exp(-(s / edge) ** 2))
        else:
            g = np.exp(-(d / width) ** 2)      # symmetric peak on the fault
    gfac.data[:, 0] = g
    dfac.data[:, 0] = d                        # SIGNED


# ---------------------------------------------------------------------------
# Workflow steps
# ---------------------------------------------------------------------------
@workflow_step(
    description="Build (or reload) the uniform annulus mesh (fault rides as a metric, not a base)",
    produces=["mesh"],
)
def create_mesh(config: FaultConvectionConfig):
    import underworld3 as uw

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    action, archive = _check_compatibility(config, output_dir)
    mesh_h5 = output_dir / f"{RUN_NAME}.mesh.00000.h5"
    if action == "continue" and mesh_h5.exists():
        mesh = uw.discretisation.Mesh(str(mesh_h5), qdegree=config.qdegree)
    else:
        ref = config.refinement if config.refinement > 0 else None
        # gmsh fault base: place small cells along the fault trace so the
        # mover MAINTAINS the refinement (vs the ~1.8x creation cap from a
        # uniform base). The trace point sits on r_outer; sample a few
        # extra points just inside so nodes embed cleanly.
        rl = {}
        smin = (config.fault_base_smin if config.fault_base_smin > 0
                else config.cellsize / 3.0)
        lines = []
        if config.fault_base_smin > 0:
            xy, _, _ = _fault_geometry(config)
            lines.append(xy)
            if config.fault_wedge:
                wedge = _wedge_points(xy, config.r_outer, config.fault_base_smin)
                if len(wedge):
                    lines.append(wedge)
        if config.blob_enable:                        # resolve the lid blob
            lines.append(_blob_points(config, smin))
        if lines:
            rl = dict(refine_lines=lines, refine_size_min=smin,
                      refine_dist_min=0.02, refine_dist_max=0.12)
        mesh = uw.meshing.Annulus(
            radiusOuter=config.r_outer, radiusInner=config.r_inner,
            cellSize=config.cellsize, qdegree=config.qdegree, refinement=ref, **rl)
    mesh._uw_run_archive = archive
    return mesh


@workflow_step(
    description="Build solvers + fault weak zone + fault distance/influence fields",
    produces=["stokes", "adv_diff", "T", "v", "p", "gfac", "dfac"],
    requires=["mesh"],
)
def create_solvers(mesh, config: FaultConvectionConfig):
    import underworld3 as uw

    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=config.T_degree,
                                       continuous=True, varsymbol="T")
    v = uw.discretisation.MeshVariable("v", mesh, mesh.dim, degree=2,
                                       continuous=True, varsymbol=r"\mathbf{v}")
    p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1,
                                       continuous=True, varsymbol="p")
    # Fault influence f (1 on fault → 0 away) and unsigned distance d.
    # P1 (NOT P2): a viscosity-bearing factor must stay positive; P1 is a
    # convex interpolant so f ∈ [0,1] at the nodes.
    gfac = uw.discretisation.MeshVariable("eta_fac", mesh, 1, degree=1,
                                          continuous=True, varsymbol=r"f_F")
    dfac = uw.discretisation.MeshVariable("dist_fac", mesh, 1, degree=1,
                                          continuous=True, varsymbol=r"d_F")

    xy, _, _ = _fault_geometry(config)
    _refresh_fault_fields(gfac, dfac, xy, config)

    X = mesh.CoordinateSystem.X
    r_sym = sympy.sqrt(X[0] ** 2 + X[1] ** 2)
    unit_r = mesh.CoordinateSystem.unit_e_0
    T_cond = sympy.log(r_sym / config.r_outer) / sympy.log(config.r_inner / config.r_outer)
    theta_FK = float(np.log(config.delta_eta))
    eta_FK = sympy.exp(theta_FK * (1 - T.sym[0]))
    finf = gfac.sym[0]
    # Weak zone: GEOMETRIC (log-linear) blend η_1 = η_FK^(1−f)·floor^f. Unlike
    # the arithmetic blend η_FK·(1−f)+floor·f (which leaves ~η_FK·(1−f) leaking
    # through in a stiff lid — η_1≈8 at f=0.97, η_FK=250), this drives η_1 to
    # the floor GENUINELY where f→1 (η_1≈1.2 at f=0.97). Positive for f∈[0,1].
    eta_weak = eta_FK ** (1.0 - finf) * float(config.fault_floor) ** finf

    # Isotropic lid-mobility blob: a smooth (x,y) square in the cold lid that
    # pulls the viscosity to blob_floor inside it (same geometric blend, applied
    # to BOTH components so it is isotropic). Expressed analytically in the
    # coordinates — evaluated exactly at quadrature points, so a crisp square
    # with no P1 smoothing and no per-adapt refresh. Two sigmoids per axis (no
    # Abs → JIT-safe).
    def _blob_apply(eta):
        if not config.blob_enable:
            return eta
        b = _blob_influence_sym(X, config)            # ≈1 inside the square
        return eta ** (1 - b) * float(config.blob_floor) ** b

    eta_FK = _blob_apply(eta_FK)
    eta_weak = _blob_apply(eta_weak)

    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    if config.rheology == "ti":
        # Transverse-isotropic weak fault: weak only to fault-PARALLEL shear
        # (shear_viscosity_1 floored), isotropic (η_FK) otherwise. director =
        # fault-plane normal (constant for the straight fault). The
        # geophysically real fault — but ~20× costlier (intrinsic Picard
        # iterations); needs GAMG + the default newtonls (NOT ksponly, which
        # gives the wrong answer) and PENALTY free slip (Nitsche trips the
        # anisotropic-Jacobian bug).
        _, n_vec, _ = _fault_geometry(config)
        stokes.constitutive_model = uw.constitutive_models.TransverseIsotropicFlowModel
        stokes.constitutive_model.Parameters.shear_viscosity_0 = eta_FK
        stokes.constitutive_model.Parameters.shear_viscosity_1 = eta_weak
        stokes.constitutive_model.Parameters.director = sympy.Matrix(
            [float(n_vec[0]), float(n_vec[1])])
    else:
        stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
        stokes.constitutive_model.Parameters.shear_viscosity_0 = eta_weak
    stokes.tolerance = 1.0e-5
    stokes.penalty = 0.0
    if config.rheology == "isotropic":
        stokes.petsc_options["snes_type"] = "ksponly"   # linear ⇒ one KSP
    # TI: keep the default newtonls (the inexact GAMG inner solve makes the
    # linear flux iterate); ksponly would converge to the wrong answer.
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
    adv.add_dirichlet_bc(1.0, mesh.boundaries.Lower.name)
    adv.add_dirichlet_bc(0.0, mesh.boundaries.Upper.name)

    return {"stokes": stokes, "adv_diff": adv, "T": T, "v": v, "p": p,
            "gfac": gfac, "dfac": dfac}


def _seed_from_run(T, seed_run_dir, config):
    """Seed the initial T by interpolating another run's LAST checkpoint onto
    this (fresh) mesh. Cross-mesh interpolation via uw.function.evaluate (the
    seed mesh is deformed/adapted; point-location tracks the deformation on the
    fixed mover base). Use to continue a developed state at a different Ra."""
    import glob
    import re
    import underworld3 as uw
    sd = Path(seed_run_dir).expanduser()
    idxs = sorted(int(re.search(r"mesh\.(\d+)\.h5", os.path.basename(c)).group(1))
                  for c in glob.glob(str(sd / f"{RUN_NAME}.mesh.[0-9]*.h5")))
    if not idxs:
        raise RuntimeError(f"seed_run {sd}: no checkpoints found")
    last = idxs[-1]
    old_mesh = uw.discretisation.Mesh(str(sd / f"{RUN_NAME}.mesh.{last:05d}.h5"),
                                      qdegree=config.qdegree)
    old_T = uw.discretisation.MeshVariable("Tseed", old_mesh, 1,
                                           degree=config.T_degree, continuous=True)
    old_T.read_timestep(data_filename=RUN_NAME, data_name="T", index=last,
                        outputPath=str(sd))
    T.data[...] = np.asarray(
        uw.function.evaluate(old_T.sym[0], np.asarray(T.coords))).reshape(-1, 1)
    uw.pprint(f"[evolve] seeded T from {sd} step {last} "
              f"(range {float(T.data.min()):.3f}..{float(T.data.max()):.3f})",
              flush=True)


def _make_fault_adapt(mesh, stokes, adv_diff, T, v, p, gfac, dfac, config):
    """Anisotropic-tensor fault adapt closure. Returns (moved, misalign)."""
    import underworld3 as uw

    accel = None if config.mover_accel == "none" else config.mover_accel
    xy, n_vec, nnT = _fault_geometry(config)
    refine_w = (config.fault_refine_width if config.fault_refine_width > 0
                else 1.5 * config.fault_width)
    draw_w = (config.fault_draw_width if config.fault_draw_width > 0
              else 4.0 * refine_w)
    Rf = float(config.fault_anisotropy)

    def adapt():
        old_X = np.asarray(mesh.X.coords).copy()
        sk = None if config.skip_threshold > 10.0 else config.skip_threshold

        rho_T = uw.meshing.metric_density_from_gradient(
            mesh, T, refinement=float(config.resolution_ratio),
            coarsening="auto", metric_choice="front-following", name="loop")
        gauss_sharp = sympy.exp(-(dfac.sym[0] / refine_w) ** 2)
        gauss_wide = sympy.exp(-(dfac.sym[0] / draw_w) ** 2)
        # One-sided metric: gate the fault gaussians to the chosen side of the
        # fault PLANE (dfac is signed). 'lower' biases refinement to the
        # footwall to counter the upward thermal-BL/wedge pull; 'both' = the
        # symmetric default. The gaussians square dfac, so the sign only enters
        # through this gate.
        m_metric = _side_multiplier(config.fault_metric_side, _upper_sign(xy))
        if m_metric is not None:
            w_gate = (config.fault_side_width if config.fault_side_width > 0
                      else config.fault_width)
            gate = sympy.Rational(1, 2) * (
                1 + sympy.tanh(m_metric * dfac.sym[0] / w_gate))
            gauss_sharp = gauss_sharp * gate
            gauss_wide = gauss_wide * gate
        fault_rho = 1.0 + config.fault_refine_amp * gauss_wide
        # MAX (default): the fault keeps its refinement along its WHOLE length,
        # independent of the |∇T| thermal-BL density. PRODUCT lets the thermal
        # BL out-compete the fault near the surface, pulling the cluster ABOVE
        # the fault and stripping the deep fault of nodes.
        if config.metric_combine == "product":
            rho = rho_T * fault_rho
        else:
            rho = sympy.Max(rho_T, fault_rho)         # isotropic size density
        if config.blob_enable:                        # keep the lid blob refined
            blob_rho = 1.0 + config.fault_refine_amp * _blob_influence_sym(
                mesh.CoordinateSystem.X, config)
            rho = sympy.Max(rho, blob_rho)
        if Rf > 0:
            metric = rho * sympy.eye(2) + (Rf ** 2 - 1.0) * gauss_sharp * nnT
        else:
            metric = rho

        # The tensor breaks the mover's scalar skip-check: decide here from
        # the scalar-ρ misalignment, then pass skip_threshold=None.
        mm = uw.meshing.mesh_metric_mismatch(
            mesh, rho, resolution_ratio=config.resolution_ratio)
        misalign = float(mm["misalignment"])
        is_tensor = isinstance(metric, sympy.MatrixBase)
        if is_tensor and sk is not None and misalign < sk:
            return False, misalign

        uw.meshing.smooth_mesh_interior(
            mesh, metric=metric, method="mmpde",
            method_kwargs=dict(step_frac=0.2, accel=accel, momentum=0.0),
            slip_surfaces=True,
            skip_threshold=(None if is_tensor else sk), verbose=False)

        if np.allclose(np.asarray(mesh.X.coords), old_X):
            return False, misalign
        _refresh_fault_fields(gfac, dfac, xy, config)
        v.data[...] = 0.0
        p.data[...] = 0.0
        return True, misalign

    return adapt


@workflow_step(
    description="Adaptive time loop with the fault metric; measure fault/bulk refinement",
    produces=["run_directory", "evolution_log"],
    requires=["mesh", "stokes", "adv_diff", "T", "v", "p", "gfac", "dfac"],
)
def evolve(mesh, stokes, adv_diff, T, v, p, gfac, dfac, config: FaultConvectionConfig):
    import underworld3 as uw

    output_dir = Path(config.output_dir)
    run = Run.create(output_dir)
    summary = run.summary
    if summary is not None and summary.get("status") == "steady":
        uw.pprint(f"[evolve] {output_dir}: steady. No work.")
        return {"run_directory": run, "evolution_log": run.timeseries}

    timeseries = run.timeseries
    saved_steps = run.steps
    xy, _, _ = _fault_geometry(config)
    nusselt = diag.NusseltSurface(mesh, T, mesh.boundaries.Upper.name,
                                  r_inner=config.r_inner, r_outer=config.r_outer)
    adapt = _make_fault_adapt(mesh, stokes, adv_diff, T, v, p, gfac, dfac, config)
    do_adapt = config.resolution_ratio > 0

    def _row(step, t, dt, wall, did_adapt, misalign):
        T_arr = T.data[:, 0]
        q = diag.mesh_quality(mesh)
        nn = diag.nn_spacing_ratios(mesh, r_inner=config.r_inner,
                                    r_outer=config.r_outer,
                                    fault_polyline=xy, fault_width=config.fault_width)
        return dict(step=step, t=t, dt=dt, wall=wall, Nu=nusselt(),
                    vrms=diag.vrms(mesh, v), Tmin=float(T_arr.min()),
                    Tmax=float(T_arr.max()), adapted=int(did_adapt),
                    misalign=misalign, folded=q["folded"],
                    area_ratio=q["area_ratio"], aspect=q["aspect"],
                    min_area=q["min_area"], bl_ratio=nn["bl_ratio"],
                    fault_ratio=nn.get("fault_ratio", float("nan")),
                    n_fault=nn.get("n_fault", 0))

    if not saved_steps:
        if config.seed_run:
            _seed_from_run(T, config.seed_run, config)
        else:
            _write_ic(T, mesh, config)
        stokes.solve(zero_init_guess=True)
        run.write_manifest({
            "workflow": "fault_convection",
            "config_hash": _config_hash(config),
            "config_snapshot": _identity_snapshot(config),
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "rayleigh": config.rayleigh, "delta_eta": config.delta_eta,
        })
        row0 = _row(0, 0.0, 0.0, 0.0, False, float("nan"))
        run.append_step(step=0, t=0.0, dt=0.0, mesh=mesh,
                        mesh_vars=[v, T, p, gfac], diags=row0, fields=_TS_FIELDS)
        timeseries.append(row0)
        timestep, elapsed = 0, 0.0
        uw.pprint(f"[evolve] {output_dir} fresh fault start Ra={config.rayleigh:.0e} "
                  f"dip={config.fault_dip_deg} Rf={config.fault_anisotropy}", flush=True)
    else:
        last = max(saved_steps)
        T.read_timestep(data_filename=RUN_NAME, data_name="T", index=last,
                        outputPath=str(output_dir))
        v.read_timestep(data_filename=RUN_NAME, data_name="v", index=last,
                        outputPath=str(output_dir))
        _refresh_fault_fields(gfac, dfac, xy, config)
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

        row = _row(timestep, elapsed, float(dt), wall, did_adapt, misalign)
        run.append_timeseries_row(row, _TS_FIELDS)
        timeseries.append(row)
        fr = row["fault_ratio"]
        uw.pprint(
            f"[step {timestep:4d}] t={elapsed:.5f} dt={dt:.2e} {wall:5.1f}s "
            f"Nu={row['Nu']:+.3f} vrms={row['vrms']:.3e} "
            f"fold={row['folded']} arat={row['area_ratio']:.1f} "
            f"f/b={fr:.2f} n_f={row['n_fault']} "
            f"{'ADAPT' if did_adapt else ''}", flush=True)

        T_arr = T.data[:, 0]
        if not np.isfinite(T_arr).all() or T_arr.max() > 1.1 or T_arr.min() < -0.1:
            uw.pprint(f"[evolve] step {timestep}: T out of range — ABORT", flush=True)
            break
        if timestep % config.save_every == 0:
            mesh.write_timestep(filename=RUN_NAME, index=timestep,
                                outputPath=str(output_dir), meshVars=[v, T, p, gfac], meshUpdates=True)
            if _is_steady(timeseries, config):
                uw.pprint(f"[evolve] steady at step {timestep}.", flush=True)
                break
        if config.max_t > 0 and elapsed >= config.max_t:
            uw.pprint(f"[evolve] reached max_t={config.max_t} at step {timestep}.",
                      flush=True)
            if timestep % config.save_every != 0:
                mesh.write_timestep(filename=RUN_NAME, index=timestep,
                                    outputPath=str(output_dir), meshVars=[v, T, p, gfac], meshUpdates=True)
            break

    return {"run_directory": run, "evolution_log": timeseries}
