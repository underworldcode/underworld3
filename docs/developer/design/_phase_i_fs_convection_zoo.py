"""Thermal convection at low Ra with TRUE FREE SURFACE — integrator-zoo.

Third benchmark in the integrator-zoo series, alongside continent
isostasy (Case B, `_phase_i_fs_continent_fs_snapshots.py`) and Cathles
relaxation (Case A, `_phase_i_fs_relax_sl_zoo.py`). The empirical
question this benchmark answers: does the load-shortcut family
(`rk*_load_sl`, `bdf2_load_sl`) work in a regime where Δh per step is
structurally small (bounded by *thermal* CFL, not by surface
relax-CFL).

Setup
-----
- Annulus, r_inner=0.5, r_o=1.0, no buoyancy block.
- Body force = +Ra·T r̂ (positive radial → hot at bottom rises).
- Temperature: AdvDiffusionSLCN with κ=1, T=1 on Lower, T=0 on Upper.
- IC: small angular perturbation (mode 5) on conductive profile.
- Free surface on Upper boundary, no-slip on Lower.
- Default Ra = 1e4 (mildly supercritical, smooth time-evolution).

The shared `_step` kernel (SL traceback, surface-load infrastructure,
monotone γ history, Picard damping) appears byte-identically with the
continent and relaxation runners — same scheme dispatch, same dt
policy. Only `_build` and `main` differ (T field + AdvDiff drive +
thermal-CFL cap on dt).

Diagnostics per step (work_log + per-step npz):
  vrms, T_avg, Nu (Nusselt), h_max, h_rms (surface), area_uw, ΔA
"""

import os
import argparse
import glob
import numpy as np
import sympy

import pyvista as pv

import nest_asyncio
nest_asyncio.apply()


OUT_DIR = "output"
SNAP_DIR = os.path.join(OUT_DIR, "convection_zoo_snapshots")


# --- T-tracking instrumentation: monkey-patched into mesh/solvers to
# log T-data signature and mesh-quality stats around every call. Used
# to localise where the 'pepper' scatter appears inside a step.
_T_TRACK = {
    'enabled': False,
    'step': 0,                # current step (set by main loop)
    'window': (0, 0),         # (start, end) inclusive
    't_soln': None,
    'mesh': None,
    'cell_vertex_ids': None,  # cached triangle vertex indices
}


def _mesh_quality_stats(mesh):
    """Min/max signed triangle area + count of inverted triangles."""
    cv = _T_TRACK.get('cell_vertex_ids')
    if cv is None:
        dm = mesh.dm
        pStart, pEnd = dm.getDepthStratum(0)
        cStart, cEnd = dm.getHeightStratum(0)
        tris = []
        for c in range(cStart, cEnd):
            closure, _ = dm.getTransitiveClosure(c, useCone=True)
            verts = [p - pStart for p in closure
                     if pStart <= p < pEnd]
            if len(verts) == 3:
                tris.append(verts)
        cv = np.asarray(tris, dtype=np.int64)
        _T_TRACK['cell_vertex_ids'] = cv
    coords = np.asarray(mesh.X.coords)
    x = coords[:, 0][cv]
    y = coords[:, 1][cv]
    # Signed area (positive for CCW orientation).
    signed = 0.5 * (
        (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0])
        - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0]))
    n_inverted = int((signed <= 0).sum())
    return float(signed.min()), float(signed.max()), n_inverted


def _t_track(label, prev_T=None):
    if not _T_TRACK['enabled']:
        return
    step = _T_TRACK['step']
    if not (_T_TRACK['window'][0] <= step <= _T_TRACK['window'][1]):
        return
    t_soln = _T_TRACK['t_soln']
    mesh = _T_TRACK['mesh']
    T = np.asarray(t_soln.data[:, 0])
    a_min, a_max, n_inv = _mesh_quality_stats(mesh)
    h = hash(T.tobytes()) & 0xFFFF
    info = (f"T_TRACK[{label}] step={step} "
            f"T=[{T.min():+.5f},{T.max():+.5f}] "
            f"hash={h:04x} "
            f"a_signed=[{a_min:+.3e},{a_max:+.3e}] "
            f"n_inverted={n_inv}")
    if prev_T is not None:
        diff = T - prev_T
        n_changed = int((np.abs(diff) > 1.0e-12).sum())
        n_above_005 = int((np.abs(diff) > 0.05).sum())
        info += (f" Δ_from_prev: n_changed={n_changed} "
                 f"n>0.05={n_above_005} "
                 f"max|Δ|={np.abs(diff).max():.3e}")
    print("  " + info, flush=True)


def _install_t_tracking(state, args):
    """Monkey-patch mesh._deform_mesh + stokes.solve + adv_diff.solve
    to call _t_track before and after each call. Only active when
    --t-tracking-window is set."""
    if not args.t_tracking_window:
        return
    a, b = [int(x) for x in args.t_tracking_window.split(",")]
    _T_TRACK['enabled'] = True
    _T_TRACK['window'] = (a, b)
    _T_TRACK['t_soln'] = state['t_soln']
    _T_TRACK['mesh'] = state['mesh']

    mesh = state['mesh']
    stokes = state['stokes']
    adv_diff = state['adv_diff']

    orig_deform = mesh._deform_mesh
    orig_stokes = stokes.solve
    orig_advdiff = adv_diff.solve
    orig_DuDt_pre = adv_diff.DuDt.update_pre_solve
    orig_DFDt_pre = adv_diff.DFDt.update_pre_solve

    def deform_wrap(new_coords, verbose=False):
        if _T_TRACK['enabled']:
            prev = np.asarray(state['t_soln'].data[:, 0]).copy()
        orig_deform(new_coords, verbose=verbose)
        if _T_TRACK['enabled']:
            _t_track("deform", prev_T=prev)

    def stokes_wrap(*a, **kw):
        if _T_TRACK['enabled']:
            prev = np.asarray(state['t_soln'].data[:, 0]).copy()
        result = orig_stokes(*a, **kw)
        if _T_TRACK['enabled']:
            _t_track("stokes", prev_T=prev)
        return result

    def _psi_stats(label):
        """Log psi_star[0] state — the SL trace-back's output that
        feeds the SNES solve as time-history RHS."""
        if not _T_TRACK['enabled']:
            return
        step = _T_TRACK['step']
        if not (_T_TRACK['window'][0] <= step
                <= _T_TRACK['window'][1]):
            return
        ps = adv_diff.DuDt.psi_star[0]
        arr = np.asarray(ps.data[:, 0])
        h = hash(arr.tobytes()) & 0xFFFF
        n_below = int((arr < -1.0e-3).sum())
        n_above = int((arr > 1.0 + 1.0e-3).sum())
        print(f"  T_TRACK[{label}] step={step} "
              f"psi_star[0]=[{arr.min():+.5f},{arr.max():+.5f}] "
              f"hash={h:04x} n_below={n_below} n_above={n_above}",
              flush=True)

    def DuDt_pre_wrap(*a, **kw):
        if _T_TRACK['enabled']:
            t_prev = np.asarray(state['t_soln'].data[:, 0]).copy()
            ps_prev = np.asarray(
                adv_diff.DuDt.psi_star[0].data[:, 0]).copy()
        result = orig_DuDt_pre(*a, **kw)
        if _T_TRACK['enabled']:
            # Confirm t_soln unchanged by SL trace-back.
            _t_track("DuDt.pre", prev_T=t_prev)
            # Log psi_star[0] before/after diff explicitly.
            ps_after = np.asarray(
                adv_diff.DuDt.psi_star[0].data[:, 0])
            d = ps_after - ps_prev
            n_changed = int((np.abs(d) > 1.0e-12).sum())
            n_005 = int((np.abs(d) > 0.05).sum())
            print(f"  T_TRACK[DuDt.pre/psi_star] step="
                  f"{_T_TRACK['step']} "
                  f"psi_star_was=[{ps_prev.min():+.4f},"
                  f"{ps_prev.max():+.4f}] "
                  f"psi_star_now=[{ps_after.min():+.4f},"
                  f"{ps_after.max():+.4f}] "
                  f"n_changed={n_changed} n>0.05={n_005} "
                  f"max|Δpsi|={np.abs(d).max():.3e}",
                  flush=True)
        return result

    def DFDt_pre_wrap(*a, **kw):
        if _T_TRACK['enabled']:
            t_prev = np.asarray(state['t_soln'].data[:, 0]).copy()
        result = orig_DFDt_pre(*a, **kw)
        if _T_TRACK['enabled']:
            _t_track("DFDt.pre", prev_T=t_prev)
        return result

    def advdiff_wrap(*a, **kw):
        if _T_TRACK['enabled']:
            prev = np.asarray(state['t_soln'].data[:, 0]).copy()
        # NOTE: DuDt_pre_wrap and DFDt_pre_wrap fire INSIDE this call
        # via the monkey-patched .DuDt.update_pre_solve hook below.
        # After both fire, the inner super().solve() runs (the SNES
        # implicit step) — that's the gap between DFDt.pre log and
        # the advdiff log; any pepper introduced there belongs to
        # SNES, not the trace-back.
        result = orig_advdiff(*a, **kw)
        if _T_TRACK['enabled']:
            _t_track("advdiff", prev_T=prev)
        return result

    mesh._deform_mesh = deform_wrap
    stokes.solve = stokes_wrap
    adv_diff.solve = advdiff_wrap
    adv_diff.DuDt.update_pre_solve = DuDt_pre_wrap
    adv_diff.DFDt.update_pre_solve = DFDt_pre_wrap
    print(f"T-TRACK: instrumented steps {a}..{b}", flush=True)


# Cached vertex-vertex adjacency (rebuilt only when mesh topology changes).
_WINSLOW_ADJ_CACHE = {'topology_id': None, 'adj': None,
                      'is_boundary': None}


def _winslow_build_adjacency(mesh, boundary_labels=('Upper', 'Lower')):
    """Build vertex→vertex adjacency via DMPlex edge iteration.

    Far cheaper than the cell-transitive-closure approach: for a 2D
    triangular mesh, each edge's cone is its two endpoint vertices,
    so one pass over edges builds the symmetric adjacency directly.

    Returns the adjacency as a sparse CSR matrix `A` such that
    `A @ x` gives the average of each vertex's neighbour values, plus
    a boundary mask.

    Parallel-safe (local DMPlex point chart, ghost vertices included).
    """
    from scipy.sparse import csr_matrix
    dm = mesh.dm
    pStart, pEnd = dm.getDepthStratum(0)   # vertex stratum
    eStart, eEnd = dm.getDepthStratum(1)   # edge stratum (2D)
    n_verts = pEnd - pStart

    # Iterate edges; each gives two endpoint vertices → adjacency pairs.
    rows, cols = [], []
    for e in range(eStart, eEnd):
        cone = dm.getCone(e)  # length 2 for triangular mesh edges
        if len(cone) != 2:
            continue
        v0, v1 = cone[0] - pStart, cone[1] - pStart
        if 0 <= v0 < n_verts and 0 <= v1 < n_verts:
            rows.append(v0); cols.append(v1)
            rows.append(v1); cols.append(v0)

    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    data = np.ones_like(rows, dtype=np.float64)
    A_pattern = csr_matrix((data, (rows, cols)),
                           shape=(n_verts, n_verts))
    # Each row's sum = number of neighbours for that vertex.
    n_neighbours = np.asarray(A_pattern.sum(axis=1)).ravel()
    n_neighbours_safe = np.where(n_neighbours > 0, n_neighbours, 1.0)
    # Row-normalise so A @ x gives the average of neighbours.
    inv = 1.0 / n_neighbours_safe
    A = csr_matrix((data * inv[rows], (rows, cols)),
                   shape=(n_verts, n_verts))

    # Boundary mask via labels.
    is_boundary = np.zeros(n_verts, dtype=bool)
    for lname in boundary_labels:
        try:
            label = dm.getLabel(lname)
            if label is None:
                continue
            for val in label.getValueIS().getIndices():
                iset = label.getStratumIS(val)
                if iset is None:
                    continue
                for idx in iset.getIndices():
                    if pStart <= idx < pEnd:
                        is_boundary[idx - pStart] = True
        except Exception:
            pass
    return A, is_boundary


def winslow_smooth_interior(state, n_iters=1, alpha=0.5,
                            boundary_labels=('Upper', 'Lower')):
    """Discrete graph-Laplacian (Jacobi) smoothing of interior nodes.

    For each interior vertex, replace its position by a weighted blend
    of itself and the average of its mesh-edge neighbours' positions.
    Boundary vertices stay fixed (preserves deformed surface).

    Implementation: vectorised sparse-matrix Jacobi sweep
    (`new = (1-α)·old + α·(A @ old)` where A is the row-normalised
    adjacency built from DMPlex edges). Adjacency is cached per mesh
    topology.

    Parameters
    ----------
    state : dict
        Solver state from _build(). Used for mesh + adjacency cache.
    n_iters : int
        Number of Jacobi sweeps (5-10 typical).
    alpha : float in (0, 1]
        Under-relaxation. 1.0 = pure Jacobi; 0.5 = damped (safer).
    boundary_labels : tuple of str
        Boundary labels whose vertices are pinned.
    """
    mesh = state['mesh']
    dm = mesh.dm
    pStart, pEnd = dm.getDepthStratum(0)
    cStart, cEnd = dm.getHeightStratum(0)
    topo_id = (cEnd - cStart, pEnd - pStart,
               dm.getConeSize(cStart) if cEnd > cStart else 0)
    cache = _WINSLOW_ADJ_CACHE
    if cache['topology_id'] != topo_id:
        cache['adj'], cache['is_boundary'] = _winslow_build_adjacency(
            mesh, boundary_labels=boundary_labels)
        cache['topology_id'] = topo_id

    A = cache['adj']             # row-normalised neighbour-average matrix
    is_bd = cache['is_boundary']
    is_int = ~is_bd

    coords = mesh.X.coords.copy()
    for _ in range(n_iters):
        avg = A @ coords  # vectorised neighbour average
        # Update interior only; boundary stays put.
        coords[is_int] = ((1.0 - alpha) * coords[is_int]
                          + alpha * avg[is_int])
    mesh._deform_mesh(coords)


def _build(res=20, Ra=1.0e4, structured=False, v_degree=2,
           p_degree=1, stokes_tol=1.0e-5,
           fssa_theta=0.0, fssa_dt_ref=4.0e-4,
           rho_g=1.0,
           nitsche_penalty=0.0,
           t_degree=3,
           outer_refine_factor=1.0):
    """Build the thermal-convection / free-surface annulus benchmark.

    Variables
    ---------
    Vr : scalar P1 — radial-velocity diffuser field for the
         smooth-mesh-deformation Poisson kernel (interior remap).
    v  : vector P_v — velocity field.
    p  : scalar P_p — pressure field.
    t_soln : scalar P3 — temperature, advected/diffused by AdvDiffSLCN.
    delta_h_load : scalar P2 — surface load for rk*_load_sl /
                   bdf2_load_sl schemes (zero everywhere by default).
    """
    import underworld3 as uw
    r_inner, r_o = 0.5, 1.0
    cellsize = 1.0 / res
    qdeg = max(3, v_degree + 1)

    if structured:
        from _structured_annulus import AnnulusStructured
        nA = max(8, int(round(2 * np.pi * r_o / cellsize)))
        nA = nA + (nA % 2)
        nR = max(2, int(round((r_o - r_inner) / cellsize)))
        mesh = AnnulusStructured(
            radiusOuter=r_o, radiusInner=r_inner,
            nRadial=nR, nAngular=nA, qdegree=qdeg,
        )
    else:
        mesh = uw.meshing.Annulus(
            radiusOuter=r_o, radiusInner=r_inner,
            cellSize=cellsize,
            cellSizeOuter=cellsize / outer_refine_factor,
            qdegree=qdeg,
        )
    r, th = mesh.CoordinateSystem.R

    pair_tag = f"v{v_degree}p{p_degree}"
    Vr = uw.discretisation.MeshVariable(
        f"Vr_conv_{pair_tag}", mesh, vtype=uw.VarType.SCALAR,
        degree=1, continuous=True, varsymbol=r"v_r")
    v = uw.discretisation.MeshVariable(
        f"V_conv_{pair_tag}", mesh, vtype=uw.VarType.VECTOR,
        degree=v_degree, continuous=True, varsymbol=r"\mathbf{v}")
    p = uw.discretisation.MeshVariable(
        f"P_conv_{pair_tag}", mesh, vtype=uw.VarType.SCALAR,
        degree=p_degree, continuous=True, varsymbol="p")
    t_soln = uw.discretisation.MeshVariable(
        f"T_conv_{pair_tag}", mesh, vtype=uw.VarType.SCALAR,
        degree=t_degree, continuous=True, varsymbol="T")

    X_initial = mesh.X.coords.copy()
    R_initial = np.sqrt(X_initial[:, 0] ** 2 + X_initial[:, 1] ** 2)
    THETA_initial = np.arctan2(X_initial[:, 1], X_initial[:, 0])
    is_surface = np.abs(R_initial - r_o) < 0.5 * cellsize / r_o
    internal_idx = np.where(is_surface)[0]
    sort_order = np.argsort(THETA_initial[internal_idx])
    internal_idx = internal_idx[sort_order]
    internal_th = THETA_initial[internal_idx]

    diffuser = uw.systems.Poisson(mesh, Vr)
    diffuser.constitutive_model = uw.constitutive_models.DiffusionModel
    diffuser.constitutive_model.Parameters.diffusivity = 1.0
    diffuser.add_essential_bc(sympy.Matrix([0.0]),
                              mesh.boundaries.Upper.name)
    diffuser.add_essential_bc(sympy.Matrix([0.0]),
                              mesh.boundaries.Lower.name)
    diffuser.tolerance = 1.0e-3
    diffuser.solve()

    # Surface-load variable for the rk*_load_sl / bdf2_load_sl schemes.
    delta_h_load = uw.discretisation.MeshVariable(
        f"dh_load_conv_{pair_tag}", mesh, vtype=uw.VarType.SCALAR,
        degree=2, continuous=True, varsymbol=r"\delta h_{\rm load}")
    delta_h_load.data[:, 0] = 0.0

    # Per-vertex surface-displacement variable used as the diffuser's
    # Dirichlet-BC source in deform_by_inc — replaces the spectral
    # representation that required a global-gather Fourier transform.
    surface_disp_var = uw.discretisation.MeshVariable(
        f"surface_disp_{pair_tag}", mesh, vtype=uw.VarType.SCALAR,
        degree=1, continuous=True, varsymbol=r"\delta h_b")
    surface_disp_var.data[:, 0] = 0.0

    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.penalty = 0.0
    # Body force: lithostatic + buoyancy perturbation.
    #
    # 1. Lithostatic: -rho_g · r̂. The "rock weight × gravity" of the
    #    background medium. Standard Boussinesq factors this out of the
    #    momentum equation because for closed domains it's balanced by
    #    the lithostatic pressure gradient and doesn't drive flow. For
    #    free-surface problems we MUST keep it explicitly: the slab-
    #    weight restoring force at the free surface is precisely
    #    rho_g · Δh per unit area, and without rho_g in the body force
    #    the corresponding pressure gradient never establishes — there
    #    is no isostatic equilibrium for the surface to relax toward.
    #
    # 2. Perturbation: Ra · (T - T_cond) · r̂. Drives convective flow.
    #    T_cond = (r_o - r)/(r_o - r_inner) is the conductive reference;
    #    using (T - T_cond) ensures the perturbation force vanishes in
    #    the conductive state.
    #
    # rho_g defaults to 1 (matches original behaviour). For physically
    # reasonable convection setups, rho_g >> Ra is expected, giving an
    # equilibrium topography Δh_eq ~ Ra·δT·L / rho_g << 1.
    T_cond = (r_o - r) / (r_o - r_inner)
    stokes.bodyforce = (
        -rho_g * mesh.CoordinateSystem.unit_e_0
        + Ra * (t_soln.sym[0] - T_cond) * mesh.CoordinateSystem.unit_e_0
    )
    stokes.tolerance = stokes_tol
    stokes.add_essential_bc((0.0, 0.0), mesh.boundaries.Lower.name)
    # Natural BC on Upper. Two contributions, both scaled by rho_g
    # (the slab-weight coefficient — without this scaling neither term
    # represents the right physics):
    #
    #   1. -rho_g · delta_h_load · r̂ — the load-shortcut Δh-prediction
    #      slab-weight (zero by default; written by load schemes during
    #      stages).
    #
    #   2. -rho_g · θ · dt_ref · (v·r̂) · r̂ — FSSA-style stabilisation
    #      (Crameri-Tackley 2012 / Kaus et al 2010). Treats the predicted
    #      surface displacement Δh ≈ Δt·v_n implicitly through the Stokes
    #      operator. K_fssa = rho_g · θ · Δt is the stiffness that opposes
    #      surface radial velocity. Zero θ recovers no-FSSA behaviour.
    K_fssa = float(rho_g * fssa_theta * fssa_dt_ref)
    fssa_term = (
        K_fssa * v.sym.dot(mesh.CoordinateSystem.unit_e_0)
        * mesh.CoordinateSystem.unit_e_0
    )
    # Nitsche-style penalty as an updateable MeshVariable so the time
    # loop can ramp it down (release the surface gradually). Same role
    # as the Cylinder-FS example's penalty when held constant; the
    # time loop schedules its decay per --nitsche-bootstrap-steps and
    # --nitsche-release-steps.
    nitsche_K_var = uw.discretisation.MeshVariable(
        f"nitsche_K_conv_{pair_tag}", mesh, vtype=uw.VarType.SCALAR,
        degree=2, continuous=True, varsymbol=r"K_{\rm N}")
    nitsche_K_var.data[:, 0] = float(nitsche_penalty)
    nitsche_term = (
        nitsche_K_var.sym[0]
        * v.sym.dot(mesh.CoordinateSystem.unit_e_0)
        * mesh.CoordinateSystem.unit_e_0
    )
    stokes.add_natural_bc(
        -rho_g * delta_h_load.sym[0] * mesh.CoordinateSystem.unit_e_0
        - fssa_term
        + nitsche_term,
        mesh.boundaries.Upper.name)

    # Identify surface DOFs of the load variable (degree=2, may differ
    # from mesh-vertex coordinates).
    dh_coords = delta_h_load.coords
    dh_R = np.sqrt(dh_coords[:, 0] ** 2 + dh_coords[:, 1] ** 2)
    dh_TH = np.arctan2(dh_coords[:, 1], dh_coords[:, 0])
    dh_surface_mask = np.abs(dh_R - r_o) < 0.5 * cellsize / r_o
    dh_surface_idx = np.where(dh_surface_mask)[0]
    dh_surface_th = dh_TH[dh_surface_idx]
    sort = np.argsort(dh_surface_th)
    dh_surface_idx = dh_surface_idx[sort]
    dh_surface_th = dh_surface_th[sort]

    # Mesh-velocity field for ALE-corrected advection. Initialised to
    # zero; populated per-step by the time loop as (Δcoords/Δt). When
    # the time loop's --ale-correction flag is off, v_mesh stays zero
    # and V_fn = v.sym - v_mesh.sym = v.sym (recovers the original
    # non-ALE behaviour).
    v_mesh = uw.discretisation.MeshVariable(
        f"v_mesh_conv_{pair_tag}", mesh, vtype=uw.VarType.VECTOR,
        degree=v_degree, continuous=True, varsymbol=r"\mathbf{v}_{\rm mesh}")
    v_mesh.data[...] = 0.0

    # Monkey-patch the Mesh class with a geometry-agnostic
    # restore_points_to_domain method (snaps out-of-mesh coords to
    # nearest mesh DOF via kdtree). In-bounds coords pass through
    # unchanged. This is where this kind of default *should* live —
    # mesh-shape-aware out-of-domain handling is the mesh's
    # responsibility, not the solver's. Patches the class only once
    # (idempotent) so multiple meshes / multiple builds share the
    # same definition.
    if not hasattr(type(mesh), 'restore_points_to_domain'):
        def _kdtree_restore_points_to_domain(self, coords):
            # Use mesh.points_in_domain (boundary kd-tree, fast and
            # geometry-correct) to detect out-of-mesh coords. Snap them
            # to nearest mesh DOF via a per-step-cached kd-tree on the
            # DOF coordinates. In-mesh coords pass through untouched.
            in_domain = self.points_in_domain(
                coords, strict_validation=True)
            out_of_mesh = ~in_domain
            n_out = int(np.sum(out_of_mesh))
            if n_out == 0:
                return coords
            current_coords_id = id(self.X.coords)
            if getattr(self, '_restore_coords_id', None) != current_coords_id:
                self._restore_kdt = uw.kdtree.KDTree(self.X.coords)
                self._restore_coords_id = current_coords_id
            _, idxs = self._restore_kdt.query(coords[out_of_mesh], k=1)
            new_coords = coords.copy()
            new_coords[out_of_mesh] = self.X.coords[idxs]
            # Instrumentation: log restore activity at every call so we
            # can correlate floods of out-of-mesh points with T blowups.
            snap_dists = np.linalg.norm(
                coords[out_of_mesh] - self.X.coords[idxs], axis=1)
            step = globals().get('_DEBUG_STEP', '?')
            phase = globals().get('_DEBUG_PHASE', '?')
            print(f"    RESTORE step={step} phase={phase} "
                  f"n_out={n_out}/{len(coords)} ({100*n_out/len(coords):.1f}%) "
                  f"max_snap={snap_dists.max():.3e} "
                  f"mean_snap={snap_dists.mean():.3e}",
                  flush=True)
            return new_coords
        type(mesh).restore_points_to_domain = (
            _kdtree_restore_points_to_domain)

    # Advection-diffusion solver for T (semi-Lagrangian Crank–Nicolson).
    # V_fn uses (v - v_mesh) so SLCN traces back along the *relative*
    # velocity — the right ALE treatment when the mesh deforms during
    # the step. Mesh transports T DOFs implicitly (Lagrangian-by-default
    # for mesh variables); the ALE correction subtracts this transport
    # so SLCN doesn't double-count it. The restore_points_func uses the
    # mesh's monkey-patched default (snap out-of-mesh trace-back coords
    # to nearest DOF), avoiding silent extrapolation that injects
    # spurious BC-from-undeformed-surface values into the T transport.
    adv_diff = uw.systems.AdvDiffusionSLCN(
        mesh, u_Field=t_soln, V_fn=v.sym - v_mesh.sym, verbose=False,
        restore_points_func=mesh.restore_points_to_domain,
    )
    # TODO(env-flag-hack): UW_LAUNCH_CLIP — see also matching block in
    # main loop (look for `launch_clip_active`). When we decide where
    # launch-point clipping belongs (per-solver kwarg? mesh limiter?
    # AdvDiffusionSLCN API?), replace these env-var hacks with a real
    # API and remove this TODO. Search the repo for `UW_LAUNCH_CLIP`
    # and `env-flag-hack` to find all related spots.
    # Optional launch-point clip: monkey-patch DuDt.update_pre_solve so
    # that immediately after the SL trace-back values are written into
    # psi_star[0] (the snapshot used as RHS of the implicit step), we
    # clip them to [0, 1]. This catches Runge-style P3-interpolation
    # overshoots at the SOURCE rather than after CN amplification.
    if os.environ.get("UW_LAUNCH_CLIP", "0") == "1":
        _orig_ups = adv_diff.DuDt.update_pre_solve
        def _clipped_update_pre_solve(*a, **kw):
            result = _orig_ups(*a, **kw)
            ps = adv_diff.DuDt.psi_star[0]
            arr = np.asarray(ps.data[:, 0])
            n_below = int(np.sum(arr < 0.0))
            n_above = int(np.sum(arr > 1.0))
            if n_below + n_above > 0:
                step = globals().get('_DEBUG_STEP', '?')
                print(f"    LAUNCH_CLIP step={step} "
                      f"psi_star[0]=[{arr.min():+.4f},{arr.max():+.4f}] "
                      f"n_below={n_below} n_above={n_above}",
                      flush=True)
            np.clip(ps.data[:, 0], 0.0, 1.0, out=ps.data[:, 0])
            return result
        adv_diff.DuDt.update_pre_solve = _clipped_update_pre_solve
    adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel
    adv_diff.constitutive_model.Parameters.diffusivity = 1.0
    adv_diff.tolerance = 1.0e-4
    adv_diff.add_dirichlet_bc(1.0, mesh.boundaries.Lower.name)
    adv_diff.add_dirichlet_bc(0.0, mesh.boundaries.Upper.name)

    # Initial temperature: conductive profile + small mode-5 perturbation.
    # Conductive profile (r_o − r)/(r_o − r_inner) gives T=1 at Lower and
    # T=0 at Upper (matching the BCs). Perturbation seeds mode-5
    # convection without breaking the BC.
    init_t = (
        0.01 * sympy.sin(5.0 * th)
        * sympy.sin(np.pi * (r - r_inner) / (r_o - r_inner))
        + (r_o - r) / (r_o - r_inner)
    )
    t_soln.data[...] = np.asarray(uw.function.evaluate(
        init_t, t_soln.coords)).reshape(-1, 1)

    # Seed the velocity by solving Stokes once with T initialised.
    # Chicken-and-egg note: the body force depends on T but the very
    # first call sees only the initial T (no advection has happened
    # yet) so it produces a velocity field consistent with that T.
    # Subsequent steps update T from v and v from T in lockstep.
    stokes.solve()

    # Curved-element initial area for ΔA reference
    area_uw_initial = abs(float(uw.maths.Integral(mesh, 1.0).evaluate()))

    return {
        'mesh': mesh, 'r_o': r_o, 'r_inner': r_inner,
        'th_sym': th, 'r_sym': r,
        'Vr': Vr, 'v': v, 'p': p, 't_soln': t_soln, 'v_mesh': v_mesh,
        'delta_h_load': delta_h_load,
        'dh_surface_idx': dh_surface_idx,
        'dh_surface_th': dh_surface_th,
        'diffuser': diffuser, 'stokes': stokes,
        'adv_diff': adv_diff, 'Ra': Ra,
        'internal_idx': internal_idx, 'internal_th': internal_th,
        'area_uw_initial': area_uw_initial,
        'nitsche_K_var': nitsche_K_var,
        'nitsche_penalty_initial': float(nitsche_penalty),
        'X_initial': X_initial,
        'surface_disp_var': surface_disp_var,
    }


_GAMMA_HISTORY_MAX = [0.0]  # module-level retained estimate
_GAMMA_HISTORY = []         # raw γ_eff per step for diagnostics

# History buffers for multistep schemes (AB2-SL etc.). Reset between runs.
_U_N_EFF_PREV = [None]      # u_n_eff^{n-1} on the surface θ-grid
_DT_PREV = [None]           # Δt^{n-1} (used to derive variable-step AB coeffs)
# History for BDF schemes: h^n and h^{n-1} as Fourier coefficient arrays.
_H_PREV_FOURIER = [None]    # (a, b) Fourier coeffs of h^{n-1} on the start-of-step mesh
_BDF_BOOTSTRAPPED = [False] # True after one BDF1 step has produced h^n→h^{n+1}


def _reset_gamma_history():
    _GAMMA_HISTORY_MAX[0] = 0.0
    del _GAMMA_HISTORY[:]
    _U_N_EFF_PREV[0] = None
    _DT_PREV[0] = None
    _H_PREV_FOURIER[0] = None
    _BDF_BOOTSTRAPPED[0] = False


def _step(state, scheme, dt_factor, dt_cap=None,
          dt_cap_mode='fixed', dt_cap_c=0.5):
    """Take one adaptive-Δt step using `scheme` (FE/RK2/RK4/curvS/midpoint).

    Δt selection:
      dt_cap_mode='fixed'  → Δt = min(dt_factor·estimate_dt, dt_cap)
      dt_cap_mode='relax'  → Δt = min(dt_factor·estimate_dt,
                                       dt_cap_c / γ_eff)
        where γ_eff = -⟨u_n, h⟩ / ⟨h, h⟩ is the dominant surface
        relaxation rate from the current state. Amplitude-invariant.
      dt_cap_mode='none'   → Δt = dt_factor·estimate_dt (no cap)
    Returns (h_pole, dt_used)."""
    import underworld3 as uw
    mesh = state['mesh']; v = state['v']
    Vr = state['Vr']; diffuser = state['diffuser']
    stokes = state['stokes']
    internal_idx = state['internal_idx']
    internal_th = state['internal_th']
    th = state['th_sym']; r_o = state['r_o']
    unit_r = mesh.CoordinateSystem.unit_e_0

    dt = dt_factor * stokes.estimate_dt()
    if dt_cap_mode == 'fixed' and dt_cap is not None:
        dt = min(dt, dt_cap)
    elif dt_cap_mode == 'relax':
        # γ_eff from monotone history of |⟨u_n, h⟩| / ⟨h, h⟩.
        # Once a fast mode reveals itself in the surface state we
        # retain its γ as the binding constraint (damping
        # requirement). Otherwise an L² estimator will report the
        # slow dominant mode and let Δt grow large enough to
        # destabilise the fast modes — observed as wave-like step-
        # to-step oscillations in h_pole.
        upper_pos = mesh.X.coords[internal_idx]
        upper_r = np.sqrt(upper_pos[:, 0] ** 2 + upper_pos[:, 1] ** 2)
        h = upper_r - r_o
        u_n_now = np.asarray(uw.function.evaluate(
            v.sym.dot(unit_r), upper_pos)).flatten()
        num = abs(float(np.sum(u_n_now * h)))
        den = float(np.sum(h * h))
        if den > 1.0e-30 and num > 1.0e-30:
            gamma_eff = num / den
            _GAMMA_HISTORY.append(gamma_eff)
            if gamma_eff > _GAMMA_HISTORY_MAX[0]:
                _GAMMA_HISTORY_MAX[0] = gamma_eff
            gamma_used = _GAMMA_HISTORY_MAX[0]
            dt_relax = dt_cap_c / gamma_used
            dt = min(dt, dt_relax)
    n_modes = max(2, len(internal_th) // 3)

    # Trapezoid weights for Fourier on irregular θ
    def trap_w(thw):
        n = len(thw); d = np.empty(n)
        d[1:-1] = 0.5 * (thw[2:] - thw[:-2])
        d[0] = 0.5 * (thw[1] - (thw[-1] - 2 * np.pi))
        d[-1] = 0.5 * ((thw[0] + 2 * np.pi) - thw[-2])
        return d

    def fourier_decomp(values, n_modes):
        dthw = trap_w(internal_th)
        a = np.zeros(n_modes + 1); b = np.zeros(n_modes + 1)
        a[0] = float(np.sum(values * dthw) / (2 * np.pi))
        for m in range(1, n_modes + 1):
            a[m] = float(np.sum(values * np.cos(m * internal_th)
                                 * dthw) / np.pi)
            b[m] = float(np.sum(values * np.sin(m * internal_th)
                                 * dthw) / np.pi)
        return a, b

    def fourier_to_sympy(a, b, theta_sym):
        expr = sympy.Float(a[0])
        for m in range(1, len(a)):
            if abs(a[m]) > 1e-12:
                expr = expr + a[m] * sympy.cos(m * theta_sym)
            if abs(b[m]) > 1e-12:
                expr = expr + b[m] * sympy.sin(m * theta_sym)
        return expr

    def deform_by_inc(disp_internal):
        # Original Fourier→sympy → diffuser BC path. Do NOT call
        # Winslow inside this — it's invoked many times per step
        # (per Picard iter / RK stage) and Winslow side-effects on
        # T (Lagrangian DOFs ride the smoothing) cause spurious
        # mode-5 growth in the FIRST step. Winslow now lives in the
        # main loop as an end-of-step infrequent operation
        # (--winslow-every-n-steps), which is what the user wanted.
        a, b = fourier_decomp(disp_internal, n_modes)
        inc_fn = fourier_to_sympy(a, b, th)
        diffuser._reset()
        diffuser.add_essential_bc(sympy.Matrix([inc_fn]),
                                  mesh.boundaries.Upper.name)
        diffuser.add_essential_bc(sympy.Matrix([0.0]),
                                  mesh.boundaries.Lower.name)
        diffuser.solve(zero_init_guess=False)
        f = np.asarray(uw.function.evaluate(
            Vr.sym * unit_r, mesh.X.coords)).reshape(-1, 2)
        mesh._deform_mesh(mesh.X.coords + f)

    def sample_un():
        return np.asarray(uw.function.evaluate(
            v.sym.dot(unit_r),
            mesh.X.coords[internal_idx])).flatten()

    unit_t = mesh.CoordinateSystem.unit_e_1

    def sample_ut():
        return np.asarray(uw.function.evaluate(
            v.sym.dot(unit_t),
            mesh.X.coords[internal_idx])).flatten()

    def fourier_eval_numpy(a, b, theta_array):
        """Evaluate a Fourier series (a_m, b_m) at arbitrary thetas.
        Periodic by construction, so trace-back angles need no wrapping."""
        out = np.full_like(theta_array, a[0], dtype=float)
        for m in range(1, len(a)):
            if abs(a[m]) > 1e-12:
                out += a[m] * np.cos(m * theta_array)
            if abs(b[m]) > 1e-12:
                out += b[m] * np.sin(m * theta_array)
        return out

    saved_X = mesh.X.coords.copy()

    if scheme == 'fe':
        # Diffuse v·r̂; FE deformation
        diffuser._reset()
        diffuser.add_essential_bc(
            sympy.Matrix([v.sym.dot(unit_r)]),
            mesh.boundaries.Upper.name)
        diffuser.add_essential_bc(sympy.Matrix([0.0]),
                                  mesh.boundaries.Lower.name)
        diffuser.solve(zero_init_guess=False)
        disp = dt * np.asarray(uw.function.evaluate(
            Vr.sym * unit_r, mesh.X.coords)).reshape(-1, 2)
        mesh._deform_mesh(mesh.X.coords + disp)

    elif scheme == 'rk2':
        k1 = sample_un()
        deform_by_inc((dt * 0.5) * k1)
        stokes.solve(zero_init_guess=False)
        k2 = sample_un()
        mesh._deform_mesh(saved_X)
        deform_by_inc(dt * k2)

    elif scheme == 'fe_sl':
        # Forward-Euler with semi-Lagrangian-horizontal surface
        # advection — single-stage stability baseline.
        upper_pos0 = mesh.X.coords[internal_idx]
        h0 = np.sqrt(upper_pos0[:, 0] ** 2 + upper_pos0[:, 1] ** 2) - r_o
        a_h, b_h = fourier_decomp(h0, n_modes)
        u_n = sample_un()
        u_t = sample_ut()
        theta_back = internal_th - dt * u_t / r_o
        h_back = fourier_eval_numpy(a_h, b_h, theta_back)
        u_n_eff = u_n + (h_back - h0) / dt
        deform_by_inc(dt * u_n_eff)

    elif scheme == 'ab2_sl':
        # Adams–Bashforth-2 with SL-horizontal surface advection.
        # One Stokes solve per step; one history term carries u_n_eff
        # from the previous step. In the Δt → 0 limit u_n_eff is the
        # material derivative Dh/Dt, so caching it and extrapolating
        # linearly via AB2 is consistent with the underlying ODE.
        # Variable-step coefficients with ω = Δt^n / Δt^{n-1} keep
        # 2nd-order accuracy under the relaxation-CFL adaptive step.
        #
        # Robustness clamp: when ω > AB2_OMEGA_MAX (Δt grew too fast,
        # typically while γ_eff is still initialising), fall back to
        # FE-SL. The AB2 predictor coefficient (1 + ω/2) explodes
        # for large ω and has no physical justification when the
        # u_n_eff history was sampled under very different conditions.
        # Honest degradation > extrapolating into invalid regimes.
        AB2_OMEGA_MAX = 1.5
        upper_pos0 = mesh.X.coords[internal_idx]
        h0 = np.sqrt(upper_pos0[:, 0] ** 2 + upper_pos0[:, 1] ** 2) - r_o
        a_h, b_h = fourier_decomp(h0, n_modes)
        u_n = sample_un()
        u_t = sample_ut()
        theta_back = internal_th - dt * u_t / r_o
        h_back = fourier_eval_numpy(a_h, b_h, theta_back)
        u_n_eff = u_n + (h_back - h0) / dt
        history_ok = (_U_N_EFF_PREV[0] is not None
                      and _DT_PREV[0] is not None
                      and dt / _DT_PREV[0] <= AB2_OMEGA_MAX)
        if not history_ok:
            inc = dt * u_n_eff   # bootstrap or fallback
        else:
            omega = dt / _DT_PREV[0]
            inc = dt * ((1.0 + 0.5 * omega) * u_n_eff
                        - 0.5 * omega * _U_N_EFF_PREV[0])
        deform_by_inc(inc)
        _U_N_EFF_PREV[0] = u_n_eff.copy()
        _DT_PREV[0] = dt

    elif scheme == 'rk4_sl':
        # RK4 with semi-Lagrangian-horizontal surface advection.
        # Each stage produces an effective u_n that, applied over its
        # stage interval, captures both radial uplift and tangential
        # surface transport. Combined with RK4 weights for u_n.
        upper_pos0 = mesh.X.coords[internal_idx]
        h0 = np.sqrt(upper_pos0[:, 0] ** 2 + upper_pos0[:, 1] ** 2) - r_o
        a_h, b_h = fourier_decomp(h0, n_modes)

        def stage_eff(stage_dt):
            u_n_s = sample_un()
            u_t_s = sample_ut()
            theta_back_s = internal_th - stage_dt * u_t_s / r_o
            h_back_s = fourier_eval_numpy(a_h, b_h, theta_back_s)
            return u_n_s + (h_back_s - h0) / stage_dt

        # Stage 1
        k1 = stage_eff(dt * 0.5)
        mesh._deform_mesh(saved_X); deform_by_inc((dt * 0.5) * k1)
        stokes.solve(zero_init_guess=False)
        # Stage 2
        k2 = stage_eff(dt * 0.5)
        mesh._deform_mesh(saved_X); deform_by_inc((dt * 0.5) * k2)
        stokes.solve(zero_init_guess=False)
        # Stage 3
        k3 = stage_eff(dt)
        mesh._deform_mesh(saved_X); deform_by_inc(dt * k3)
        stokes.solve(zero_init_guess=False)
        # Stage 4
        k4 = stage_eff(dt)
        mesh._deform_mesh(saved_X)
        deform_by_inc((dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))

    elif scheme == 'bdf2_sl':
        # BDF2 with semi-Lagrangian-horizontal surface advection.
        # First step: BDF1 (= backward Euler), itself implicit.
        # Step n+1:  3/2 h^{n+1} − 2 h^n_back + 1/2 h^{n-1}_back = Δt u_n^{n+1}
        # where h^n_back, h^{n-1}_back are h^n, h^{n-1} traced back along
        # u_t^{n+1} (current tangential velocity, scaled by Δt^n and
        # (Δt^n + Δt^{n-1}) respectively — single-velocity approximation,
        # adequate while u_t × (2Δt) / r_o remains a small angular shift).
        #
        # Implicit solve via damped Picard iteration:
        #   h^{(k+1)} = h^{(k)} + α · (G(h^{(k)}) − h^{(k)})
        # where G(h) is the BDF formula evaluated at h. The damping factor
        #   α = 1 / (1 + γ_used · Δt)
        # under-relaxes when γΔt ≈ 1 (the regime where pure Picard would
        # oscillate). Recovers full Newton-step rate when γΔt → 0.
        BDF_PICARD_TOL = 1.0e-4
        BDF_PICARD_MAX_ITER = 12
        # Damped-Picard (= Newton with the linearised Jacobian dG/dh ≈ -γΔt).
        # γ-floor at γΔt ≥ 1 forces α ≤ 0.5 so the bootstrap step (where
        # γ_used is still 0) doesn't oscillate.
        gamma_used_local = max(_GAMMA_HISTORY_MAX[0], 1.0e-12)
        gdt = max(gamma_used_local * dt, 1.0)
        alpha_picard = 1.0 / (1.0 + gdt)

        upper_pos0 = mesh.X.coords[internal_idx]
        h0 = np.sqrt(upper_pos0[:, 0] ** 2 + upper_pos0[:, 1] ** 2) - r_o
        a_h0, b_h0 = fourier_decomp(h0, n_modes)
        # Initial guess: h_pred = h0 (zero displacement). An explicit
        # FE-SL predictor was tried and rejected — at large Δt where
        # BDF2's A-stability matters, FE overshoots equilibrium and
        # the first Picard residual is *worse* than h0. The undamped
        # h0 predictor is well-conditioned: damped Picard converges
        # geometrically with rate ~0.3 from |G−h|/|G| = 1.
        h_pred = h0.copy()
        n_iter = 0
        residual_history = []
        # Anderson acceleration state (per-step). m=0 → fall back to
        # damped Picard. m>0 keeps the last m+1 (h_pred, G_h) pairs and
        # extrapolates a Type-II Anderson combination instead of the
        # damped Picard step.
        anderson_m = int(state.get('anderson_m', 0))
        x_hist = []   # past h_pred values
        g_hist = []   # past G_h values
        while True:
            n_iter += 1
            mesh._deform_mesh(saved_X)
            deform_by_inc(h_pred - h0)
            stokes.solve(zero_init_guess=False)
            u_n_new = sample_un()
            u_t_new = sample_ut()
            theta_back_n = internal_th - dt * u_t_new / r_o
            h_n_back = fourier_eval_numpy(a_h0, b_h0, theta_back_n)
            if (not _BDF_BOOTSTRAPPED[0]) or _H_PREV_FOURIER[0] is None:
                G_h = h_n_back + dt * u_n_new
            else:
                a_h_prev, b_h_prev = _H_PREV_FOURIER[0]
                theta_back_nm1 = internal_th - (dt + _DT_PREV[0]) * u_t_new / r_o
                h_nm1_back = fourier_eval_numpy(
                    a_h_prev, b_h_prev, theta_back_nm1)
                G_h = (2.0 / 3.0) * (
                    2.0 * h_n_back - 0.5 * h_nm1_back + dt * u_n_new)
            denom = max(np.max(np.abs(G_h)), 1.0e-12)
            residual = float(np.max(np.abs(G_h - h_pred)) / denom)
            residual_history.append(residual)
            # Update step: damped Picard or Anderson-accelerated.
            using_anderson = (anderson_m > 0 and len(x_hist) >= 1)
            if using_anderson:
                m_eff = min(anderson_m, len(x_hist))
                r_k = G_h - h_pred
                # Build DR (residual differences) and DG (G differences):
                # column j = r_k - r_{k-1-j}, for j = 0..m_eff-1
                DR = np.column_stack([
                    r_k - (g_hist[-1 - j] - x_hist[-1 - j])
                    for j in range(m_eff)])
                DG = np.column_stack([
                    G_h - g_hist[-1 - j] for j in range(m_eff)])
                try:
                    gamma, *_ = np.linalg.lstsq(DR, r_k, rcond=None)
                    h_next = G_h - DG @ gamma
                    update_tag = f"anderson m={m_eff}"
                except np.linalg.LinAlgError:
                    h_next = h_pred + alpha_picard * (G_h - h_pred)
                    update_tag = "picard (anderson failed)"
            else:
                h_next = h_pred + alpha_picard * (G_h - h_pred)
                update_tag = f"picard α={alpha_picard:.3f}"
            print(f"    bdf2 iter {n_iter}: |G-h|/|G|={residual:.3e}, "
                  f"{update_tag}, γΔt={gamma_used_local*dt:.3f}",
                  flush=True)
            # Save history BEFORE overwriting h_pred
            x_hist.append(h_pred.copy())
            g_hist.append(G_h.copy())
            if len(x_hist) > anderson_m + 1:
                x_hist.pop(0)
                g_hist.pop(0)
            h_pred = h_next
            if residual < BDF_PICARD_TOL or n_iter >= BDF_PICARD_MAX_ITER:
                break
        # Final deformation to converged h_pred
        mesh._deform_mesh(saved_X)
        deform_by_inc(h_pred - h0)
        # Update history
        _H_PREV_FOURIER[0] = (a_h0.copy(), b_h0.copy())
        _DT_PREV[0] = dt
        _BDF_BOOTSTRAPPED[0] = True
        state['_bdf_n_iter'] = n_iter

    elif scheme == 'rk2_sl':
        # Semi-Lagrangian-horizontal kinematic update.
        # Surface evolves by Dh/Dt = u_n: trace each surface point
        # back along the surface by u_t·dt/r_o, sample h there via
        # Fourier interpolation, then add u_n·dt. Interior remains
        # radial-only via the existing diffuser path — mesh quality
        # is preserved exactly as in vanilla rk2.
        upper_pos0 = mesh.X.coords[internal_idx]
        upper_r0 = np.sqrt(upper_pos0[:, 0] ** 2 + upper_pos0[:, 1] ** 2)
        h0 = upper_r0 - r_o
        a_h, b_h = fourier_decomp(h0, n_modes)

        # Stage 1: half-step trace-back using current u_t
        u_n_1 = sample_un()
        u_t_1 = sample_ut()
        theta_back_h = internal_th - (dt * 0.5) * u_t_1 / r_o
        h_back_h = fourier_eval_numpy(a_h, b_h, theta_back_h)
        # Effective normal velocity such that
        # h^{n+½}(θ) - h^n(θ) = (dt/2) · u_n_eff = h_back_h - h0 + (dt/2)·u_n_1
        u_n_eff_h = u_n_1 + (h_back_h - h0) / (dt * 0.5)
        deform_by_inc((dt * 0.5) * u_n_eff_h)
        stokes.solve(zero_init_guess=False)

        # Stage 2: full-step trace-back using midpoint u_t
        u_n_2 = sample_un()
        u_t_2 = sample_ut()
        theta_back = internal_th - dt * u_t_2 / r_o
        h_back = fourier_eval_numpy(a_h, b_h, theta_back)
        u_n_eff = u_n_2 + (h_back - h0) / dt
        mesh._deform_mesh(saved_X)
        deform_by_inc(dt * u_n_eff)

    elif scheme == 'bdf2_load_sl':
        # BDF2-load: same as bdf2_sl but each Picard iteration uses
        # a surface load instead of an actual mesh deformation. Mesh
        # deforms ONCE per step at the end, after Picard converges.
        # Saves 4–11 diffuser+mesh-deform operations per step
        # (BDF2-SL uses 5–12 Picard iterations at c=0.5..4).
        BDF_PICARD_TOL = 1.0e-4
        BDF_PICARD_MAX_ITER = 12
        gamma_used_local = max(_GAMMA_HISTORY_MAX[0], 1.0e-12)
        gdt = max(gamma_used_local * dt, 1.0)
        alpha_picard = 1.0 / (1.0 + gdt)

        delta_h_load = state['delta_h_load']
        dh_surface_idx = state['dh_surface_idx']
        dh_surface_th  = state['dh_surface_th']

        upper_pos0 = mesh.X.coords[internal_idx]
        h0 = np.sqrt(upper_pos0[:, 0] ** 2 + upper_pos0[:, 1] ** 2) - r_o
        a_h0, b_h0 = fourier_decomp(h0, n_modes)
        h_pred = h0.copy()
        n_iter = 0

        def set_load_from_internal(dh_array):
            a_d, b_d = fourier_decomp(dh_array, n_modes)
            delta_h_load.data[dh_surface_idx, 0] = fourier_eval_numpy(
                a_d, b_d, dh_surface_th)

        while True:
            n_iter += 1
            # Apply (predicted h^{n+1,(k)} - h^n) as surface load.
            # Mesh stays at saved_X — no deform/restore.
            set_load_from_internal(h_pred - h0)
            stokes.solve(zero_init_guess=False)
            u_n_new = sample_un()
            u_t_new = sample_ut()
            theta_back_n = internal_th - dt * u_t_new / r_o
            h_n_back = fourier_eval_numpy(a_h0, b_h0, theta_back_n)
            if (not _BDF_BOOTSTRAPPED[0]) or _H_PREV_FOURIER[0] is None:
                G_h = h_n_back + dt * u_n_new
            else:
                a_h_prev, b_h_prev = _H_PREV_FOURIER[0]
                theta_back_nm1 = internal_th - (dt + _DT_PREV[0]) * u_t_new / r_o
                h_nm1_back = fourier_eval_numpy(
                    a_h_prev, b_h_prev, theta_back_nm1)
                G_h = (2.0 / 3.0) * (
                    2.0 * h_n_back - 0.5 * h_nm1_back + dt * u_n_new)
            h_next = h_pred + alpha_picard * (G_h - h_pred)
            denom = max(np.max(np.abs(G_h)), 1.0e-12)
            residual = float(np.max(np.abs(G_h - h_pred)) / denom)
            print(f"    bdf2-load iter {n_iter}: |G-h|/|G|={residual:.3e}, "
                  f"α={alpha_picard:.3f}, γΔt={gamma_used_local*dt:.3f}",
                  flush=True)
            h_pred = h_next
            if residual < BDF_PICARD_TOL or n_iter >= BDF_PICARD_MAX_ITER:
                break
        # Mesh deformation: drop the surface load, then deform mesh once
        delta_h_load.data[:, 0] = 0.0
        deform_by_inc(h_pred - h0)
        # Final stokes.solve at the now-deformed mesh — same reason as
        # rk*_load_sl: v needs to be in register with the actual mesh
        # geometry before adv_diff sees it, otherwise T transport smears.
        stokes.solve(zero_init_guess=False)
        _H_PREV_FOURIER[0] = (a_h0.copy(), b_h0.copy())
        _DT_PREV[0] = dt
        _BDF_BOOTSTRAPPED[0] = True
        state['_bdf_n_iter'] = n_iter

    elif scheme in ('rk2_load_sl', 'rk4_load_sl'):
        # RK2/RK4 with surface-load approximation: instead of deforming
        # the mesh between stages, treat the predicted Δh increment
        # as a Neumann surface traction on the upper boundary
        # (linearised FSSA-style: σ·n = -ρ·g·Δh r̂). This skips the
        # diffuser Poisson solve and the mesh deform/restore at every
        # intermediate stage. The mesh deforms ONCE per step at the
        # end, using the standard diffuser path.
        delta_h_load = state['delta_h_load']
        dh_surface_idx = state['dh_surface_idx']
        dh_surface_th  = state['dh_surface_th']

        upper_pos0 = mesh.X.coords[internal_idx]
        h0 = np.sqrt(upper_pos0[:, 0] ** 2 + upper_pos0[:, 1] ** 2) - r_o

        # Solution-only load-validity limiter: max|Δh|/max|h| < ratio_cap.
        # See relax_sl_zoo.py for derivation. The linearised surface
        # load is faithful only when the predicted increment per step
        # is much smaller than the surface displacement itself; this
        # check uses only the current u_n and h to enforce that.
        LOAD_VALIDITY_CAP = 0.25
        u_n_now = sample_un()
        # Floor at 0.01·r_o (1% of domain radius) so the limiter doesn't
        # trap Δt → 0 at startup when h ≈ 0 (e.g. convection from rest).
        # For cases with finite initial h (relaxation, continent isostasy),
        # h_inf is naturally larger than the floor so the floor is dormant.
        h_inf = max(float(np.max(np.abs(h0))), 0.01 * r_o)
        u_n_inf = float(np.max(np.abs(u_n_now)))
        ratio = dt * u_n_inf / h_inf
        if ratio > LOAD_VALIDITY_CAP and u_n_inf > 1e-10:
            dt_safe = LOAD_VALIDITY_CAP * h_inf / u_n_inf
            print(f"    load-validity limiter: dt {dt:.3e} → {dt_safe:.3e} "
                  f"(max|Δh|/max|h| = {ratio:.3f} > {LOAD_VALIDITY_CAP})",
                  flush=True)
            dt = dt_safe

        a_h, b_h = fourier_decomp(h0, n_modes)

        def stage_eff(stage_dt):
            u_n_s = sample_un()
            u_t_s = sample_ut()
            theta_back_s = internal_th - stage_dt * u_t_s / r_o
            h_back_s = fourier_eval_numpy(a_h, b_h, theta_back_s)
            return u_n_s + (h_back_s - h0) / stage_dt

        def set_load(dh_array_internal):
            # dh_array_internal is on internal_th; resample at
            # delta_h_load's surface DOF angles via Fourier.
            a_d, b_d = fourier_decomp(dh_array_internal, n_modes)
            delta_h_load.data[dh_surface_idx, 0] = fourier_eval_numpy(
                a_d, b_d, dh_surface_th)

        if scheme == 'rk2_load_sl':
            # Stage 1: half-step prediction with current u_n
            k1 = stage_eff(dt * 0.5)
            set_load((dt * 0.5) * k1)
            stokes.solve(zero_init_guess=False)
            # Stage 2: full-step rate at the loaded mesh
            k2 = stage_eff(dt)
            final_inc = dt * k2
        else:  # rk4_load_sl
            k1 = stage_eff(dt * 0.5)
            set_load((dt * 0.5) * k1)
            stokes.solve(zero_init_guess=False)
            k2 = stage_eff(dt * 0.5)
            set_load((dt * 0.5) * k2)
            stokes.solve(zero_init_guess=False)
            k3 = stage_eff(dt)
            set_load(dt * k3)
            stokes.solve(zero_init_guess=False)
            k4 = stage_eff(dt)
            final_inc = (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Reset surface load to zero before the actual mesh deformation
        # (mesh will be at its true position; no load needed).
        delta_h_load.data[:, 0] = 0.0
        deform_by_inc(final_inc)
        # Final stokes.solve at the now-deformed mesh so v is geometrically
        # consistent with current DOF positions. Without this, v stayed
        # attached to DOFs but its values were computed at the start mesh
        # (with load BC) — feeding adv_diff a v field out of register
        # with the mesh, smearing T transport.
        stokes.solve(zero_init_guess=False)

    elif scheme == 'rk2_full':
        # Full-velocity RK2: deform every node by dt·v_node (both
        # components). Stokes velocity is discretely div-free, so
        # this should preserve area to discretisation order — the
        # radial-only diffuser-Poisson smoothing is bypassed.
        # Mesh quality may degrade after enough steps; for the
        # benchmark we just want to see the volume-conservation
        # effect, regridding is a separate concern.
        def sample_v_full():
            return np.asarray(uw.function.evaluate(
                v.sym, mesh.X.coords)).reshape(-1, 2)
        v_full_0 = sample_v_full()
        mesh._deform_mesh(saved_X + (dt * 0.5) * v_full_0)
        stokes.solve(zero_init_guess=False)
        v_full_h = sample_v_full()
        mesh._deform_mesh(saved_X + dt * v_full_h)

    elif scheme == 'rk4':
        k1 = sample_un()
        mesh._deform_mesh(saved_X); deform_by_inc((dt * 0.5) * k1)
        stokes.solve(zero_init_guess=False); k2 = sample_un()
        mesh._deform_mesh(saved_X); deform_by_inc((dt * 0.5) * k2)
        stokes.solve(zero_init_guess=False); k3 = sample_un()
        mesh._deform_mesh(saved_X); deform_by_inc(dt * k3)
        stokes.solve(zero_init_guess=False); k4 = sample_un()
        mesh._deform_mesh(saved_X)
        deform_by_inc((dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))

    elif scheme in ('curvS', 'midpoint'):
        # Curvature-γ ETD (with optional midpoint sampling)
        eta_eff = 1.0
        upper_pos = mesh.X.coords[internal_idx]
        upper_r = np.sqrt(upper_pos[:, 0] ** 2 + upper_pos[:, 1] ** 2)
        upper_dr = upper_r - r_o
        # Windowed γ
        n = len(upper_dr); dthw = 2 * np.pi / n
        h2 = np.empty(n)
        for i in range(n):
            ip = (i + 1) % n; im = (i - 1) % n
            h2[i] = (upper_dr[ip] - 2*upper_dr[i] + upper_dr[im]) / dthw**2
        ks_sq = np.empty(n)
        W = 4
        for i in range(n):
            num = 0.0; den = 0.0
            for j in range(-W, W + 1):
                k = (i + j) % n
                num += -h2[k] * upper_dr[k]
                den += upper_dr[k] ** 2
            ks_sq[i] = num / den if den > 1e-30 else 1.0
        ks_sq = ks_sq / r_o ** 2
        ks = np.sqrt(np.maximum(np.abs(ks_sq), (1.0 / r_o) ** 2))
        gamma = 1.0 / (2.0 * eta_eff * ks)

        # Diffuse v·r̂ for u_n
        diffuser._reset()
        diffuser.add_essential_bc(
            sympy.Matrix([v.sym.dot(unit_r)]),
            mesh.boundaries.Upper.name)
        diffuser.add_essential_bc(sympy.Matrix([0.0]),
                                  mesh.boundaries.Lower.name)
        diffuser.solve(zero_init_guess=False)
        Vr_full = np.asarray(uw.function.evaluate(
            Vr.sym, mesh.X.coords)).flatten()
        u_n = Vr_full[internal_idx]

        if scheme == 'midpoint':
            half = 0.5 * dt
            alpha_h = np.exp(-half * gamma)
            phi_h = np.where(half * gamma > 1e-6,
                             (1 - alpha_h) / np.maximum(gamma, 1e-12),
                             half * (1 - 0.5 * half * gamma))
            deform_by_inc(phi_h * u_n)
            stokes.solve(zero_init_guess=False)
            diffuser._reset()
            diffuser.add_essential_bc(
                sympy.Matrix([v.sym.dot(unit_r)]),
                mesh.boundaries.Upper.name)
            diffuser.add_essential_bc(sympy.Matrix([0.0]),
                                      mesh.boundaries.Lower.name)
            diffuser.solve(zero_init_guess=False)
            upper_pos_h = mesh.X.coords[internal_idx]
            upper_r_h = np.sqrt(upper_pos_h[:, 0] ** 2
                                + upper_pos_h[:, 1] ** 2)
            upper_dr_h = upper_r_h - r_o
            for i in range(n):
                ip = (i + 1) % n; im = (i - 1) % n
                h2[i] = (upper_dr_h[ip] - 2*upper_dr_h[i] + upper_dr_h[im]) / dthw**2
            for i in range(n):
                num = 0.0; den = 0.0
                for j in range(-W, W + 1):
                    k = (i + j) % n
                    num += -h2[k] * upper_dr_h[k]
                    den += upper_dr_h[k] ** 2
                ks_sq[i] = num / den if den > 1e-30 else 1.0
            ks_sq = ks_sq / r_o ** 2
            ks_h = np.sqrt(np.maximum(np.abs(ks_sq), (1.0 / r_o) ** 2))
            gamma_h = 1.0 / (2.0 * eta_eff * ks_h)
            Vr_full_h = np.asarray(uw.function.evaluate(
                Vr.sym, mesh.X.coords)).flatten()
            u_n_h = Vr_full_h[internal_idx]
            mesh._deform_mesh(saved_X)
            alpha = np.exp(-dt * gamma_h)
            phi = np.where(dt * gamma_h > 1e-6,
                           (1 - alpha) / np.maximum(gamma_h, 1e-12),
                           dt * (1 - 0.5 * dt * gamma_h))
            deform_by_inc(phi * u_n_h)
        else:  # curvS
            alpha = np.exp(-dt * gamma)
            phi = np.where(dt * gamma > 1e-6,
                           (1 - alpha) / np.maximum(gamma, 1e-12),
                           dt * (1 - 0.5 * dt * gamma))
            deform_by_inc(phi * u_n)
    else:
        raise ValueError(scheme)

    stokes.solve(zero_init_guess=False)

    upper_pos = mesh.X.coords[internal_idx]
    upper_r = np.sqrt(upper_pos[:, 0] ** 2 + upper_pos[:, 1] ** 2)
    pole_idx_in = int(np.argmin(np.abs(internal_th)))
    h_pole = float(upper_r[pole_idx_in] - r_o)
    return h_pole, dt


def _convection_diagnostics(state):
    """Compute convection-time-series diagnostics on the deformed mesh.

    Returns a dict with vrms, T_avg, Nu, h_max, h_rms, area_uw, delta_A.

    Nu here is the unnormalised surface heat-flux integral
        Nu = -∫_Upper (∇T · n_outward) dS
    No arc-length normalisation: in deformed geometry the conventional
    dimensionless Nusselt loses meaning; the bare surface integral is
    the honest diagnostic. Sign: positive when heat flows out of the
    domain (T decreasing outward).
    """
    import underworld3 as uw
    mesh = state['mesh']; v = state['v']; t_soln = state['t_soln']
    # Volume of deformed mesh
    vol = abs(float(uw.maths.Integral(mesh, 1.0).evaluate()))
    # vrms = sqrt(integrate(v.v) / vol)
    v_dot_v = v.sym.dot(v.sym)
    vrms = float(np.sqrt(
        abs(float(uw.maths.Integral(mesh, v_dot_v).evaluate())) / vol))
    # Volume-averaged temperature
    T_avg = abs(float(
        uw.maths.Integral(mesh, t_soln.sym[0]).evaluate())) / vol
    # Surface heat-flux integral (unnormalised). ∇T · n_outward at the
    # outer boundary equals ∂T/∂r since n = unit_e_0 there.
    grad_T_dot_n = (
        t_soln.sym[0].diff(mesh.X[0]) * mesh.CoordinateSystem.unit_e_0[0]
        + t_soln.sym[0].diff(mesh.X[1]) * mesh.CoordinateSystem.unit_e_0[1]
    )
    upper_name = mesh.boundaries.Upper.name
    Nu = -float(uw.maths.BdIntegral(
        mesh, grad_T_dot_n, upper_name).evaluate())
    # Surface-deflection statistics
    internal_idx = state['internal_idx']
    upper_pos = mesh.X.coords[internal_idx]
    upper_r = np.sqrt(upper_pos[:, 0] ** 2 + upper_pos[:, 1] ** 2)
    h = upper_r - state['r_o']
    h_max = float(np.max(np.abs(h)))
    h_rms = float(np.sqrt(np.mean(h * h)))
    delta_A = (vol - state['area_uw_initial']) / state['area_uw_initial']
    return {
        'vrms': vrms, 'T_avg': T_avg, 'Nu': Nu,
        'h_max': h_max, 'h_rms': h_rms,
        'area_uw': vol, 'delta_A': delta_A,
    }


def _capture(state, scheme_name, label_short, h_pole_value, t_sim):
    """Capture the simulation state at this point.

    Writes:
      1. UW3-native checkpoint (H5 + XDMF) — deformed mesh + T + V.
      2. A pyvista VTU of the mesh with point_data T and velocity arrows.
      3. A profile npz with surface (θ, δr), diagnostics, and area.
    """
    import underworld3 as uw
    import underworld3.visualisation as vis
    mesh = state['mesh']; t_soln = state['t_soln']; v = state['v']
    os.makedirs(SNAP_DIR, exist_ok=True)

    diag = _convection_diagnostics(state)

    uw_filename = f"uw_{scheme_name}_{label_short}"
    mesh.write_timestep(
        filename=uw_filename,
        index=0,
        outputPath=SNAP_DIR,
        meshVars=[t_soln, v],
        meshUpdates=True,
        create_xdmf=True,
    )

    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, t_soln.sym)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v.sym)
    pvmesh.field_data["h_pole"] = np.array([h_pole_value])
    pvmesh.field_data["t_sim"] = np.array([t_sim])
    pvmesh.field_data["vrms"] = np.array([diag['vrms']])
    pvmesh.field_data["Nu"] = np.array([diag['Nu']])

    pv_path = os.path.join(SNAP_DIR,
                           f"pv_{scheme_name}_{label_short}.vtu")
    pvmesh.save(pv_path)

    # Surface profile + diagnostics
    internal_idx = state['internal_idx']
    internal_th = state['internal_th']
    upper_pos = mesh.X.coords[internal_idx]
    upper_r = np.sqrt(upper_pos[:, 0] ** 2 + upper_pos[:, 1] ** 2)
    upper_dr = upper_r - state['r_o']
    profile_path = os.path.join(
        SNAP_DIR, f"profile_{scheme_name}_{label_short}.npz")
    np.savez(profile_path, theta=internal_th, dr=upper_dr,
             h_pole=h_pole_value, t_sim=t_sim,
             vrms=diag['vrms'], T_avg=diag['T_avg'], Nu=diag['Nu'],
             h_max=diag['h_max'], h_rms=diag['h_rms'],
             area_uw=diag['area_uw'],
             area_uw_initial=state['area_uw_initial'],
             delta_A=diag['delta_A'])
    print(f"  [{scheme_name}/{label_short}] vrms={diag['vrms']:.4e} "
          f"Nu={diag['Nu']:+.3f} h_max={diag['h_max']:.3e} "
          f"ΔA/A={diag['delta_A']:+.4e}", flush=True)
    return pv_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--res', type=int, default=20)
    p.add_argument('--n-steps', type=int, default=200)
    p.add_argument('--dt-factor', type=float, default=1.0)
    p.add_argument('--dt-thermal-factor', type=float, default=1.0,
                   help="Scale factor applied to dt_thermal "
                   "(adv_diff.estimate_dt) before it caps the per-step "
                   "dt. <1.0 forces smaller dt than the natural thermal "
                   "CFL would set — useful for testing whether the ring "
                   "is dt-driven (large dt × λ_max gives sign-flipping "
                   "CN reflection on stiff modes) vs structural.")
    p.add_argument('--restart-suffix', type=str, default=None,
                   help="Snapshot suffix to restart from "
                   "(e.g. _rk4_full). Reads mesh/T/V from "
                   "convection_zoo_snapshots<suffix>/ at the step given "
                   "by --restart-step; internal solver state "
                   "(Anderson, KSP, DDt history) starts FRESH. Useful "
                   "for testing whether internal state contributes to "
                   "the bifurcation path.")
    p.add_argument('--restart-step', type=int, default=None,
                   help="Step number to load from the restart snapshot.")
    p.add_argument('--restart-scheme', type=str, default=None,
                   help="Scheme tag in the saved filenames "
                   "(e.g. 'rk4', 'bdf2_sl'). If omitted, uses the "
                   "first --schemes entry.")
    p.add_argument('--Ra', type=float, default=1.0e4,
                   help="Rayleigh number (default 1e4, mildly "
                   "supercritical).")
    p.add_argument('--fssa-theta', type=float, default=0.0,
                   help="FSSA stabilisation coefficient (Crameri-Tackley). "
                   "Adds -θ·Δt_ref·(v·n̂)·n̂ to the free-surface natural BC, "
                   "treating predicted Δh implicitly. 0 = no stabilisation "
                   "(default; matches original integrator-zoo behaviour). "
                   "1.0 = full stabilisation.")
    p.add_argument('--fssa-dt-ref', type=float, default=4.0e-4,
                   help="Reference Δt baked into the FSSA stabilisation "
                   "term. Default 4e-4 matches the typical thermal CFL "
                   "of the Ra=1e4 benchmark.")
    p.add_argument('--rho-g', type=float, default=1.0,
                   help="Background ρ·g (lithostatic density × gravity) "
                   "in the body force. Adds -rho_g·r̂ to the body force "
                   "and scales the slab-weight load BC + FSSA term by "
                   "rho_g. Default 1.0 (original integrator-zoo "
                   "behaviour). For physically sensible free-surface "
                   "convection, set rho_g >> Ra so the equilibrium "
                   "topography Δh ~ Ra·δT·L/rho_g is geometrically "
                   "realisable.")
    p.add_argument('--nitsche-penalty', type=float, default=0.0,
                   help="Nitsche-style boundary penalty enforcing "
                   "v·r̂ → 0 weakly at the upper surface. Useful to "
                   "stabilise the initial transient when rho_g is large "
                   "(otherwise the first Stokes solve gives unphysical "
                   "surface velocity from solver noise). Default 0 (true "
                   "free surface). Try ~rho_g/100 for moderate damping; "
                   "see --nitsche-bootstrap-steps and "
                   "--nitsche-release-steps for in-run release.")
    p.add_argument('--nitsche-bootstrap-steps', type=int, default=0,
                   help="Number of initial steps to hold Nitsche penalty "
                   "at --nitsche-penalty. After this, the penalty linearly "
                   "decays to 0 over --nitsche-release-steps additional "
                   "steps. Default 0 = no bootstrap (penalty constant).")
    p.add_argument('--nitsche-release-steps', type=int, default=0,
                   help="Number of steps over which Nitsche penalty "
                   "linearly decays from --nitsche-penalty to "
                   "--nitsche-floor, starting at step "
                   "--nitsche-bootstrap-steps. After (bootstrap+release) "
                   "steps the penalty is held at --nitsche-floor. "
                   "Default 0 = sharp release (or constant if bootstrap=0 "
                   "too).")
    p.add_argument('--nitsche-floor', type=float, default=0.0,
                   help="Floor value for the Nitsche penalty after release. "
                   "Default 0 (true free surface). Set to ρg·Δt for the "
                   "FSSA-equivalent stabilised free surface (K_FSSA "
                   "= ρg·θ·Δt with θ=1) — keeps a small physically-"
                   "motivated damping on the radial surface mode rather "
                   "than releasing all the way to bare free surface.")
    p.add_argument('--anderson-m', type=int, default=0,
                   help="Anderson acceleration history depth for BDF2 "
                   "Picard iteration. m=0 (default) → standard damped "
                   "Picard. m=3-5 typically gives super-linear "
                   "convergence at marginal extra cost (small "
                   "least-squares per iteration).")
    p.add_argument('--capture-every', type=int, default=0,
                   help="Capture VTU + npz per N steps (in addition to "
                   "final). Default 0 = final only. Useful for tracking "
                   "evolution into a blowup or other transient.")
    p.add_argument('--clip-t', action='store_true',
                   help="Clamp T to [0, 1] after each adv_diff.solve. "
                   "Defensive measure against SLCN+CN ringing on "
                   "under-resolved sharp BLs in deformed elements. "
                   "Breaks strict mass conservation locally but "
                   "prevents non-physical-overshoot positive-feedback "
                   "blowups.")
    p.add_argument('--rbf-advection', action='store_true',
                   help="Use RBF interpolation (evalf=True) for the SL "
                   "trace-back of T inside adv_diff.solve instead of FE "
                   "shape-function interpolation. RBF is monotone but "
                   "more diffusive — typically requires higher mesh "
                   "resolution to recover the same effective accuracy. "
                   "Set when SLCN+CN ringing on the FE interpolant "
                   "appears to dominate over discretisation error.")
    p.add_argument('--monotone-mode', type=str, default=None,
                   choices=[None, "clamp", "pick"],
                   help="Post-process the SL trace-back result with a "
                   "monotonicity limiter. 'clamp' (B.2) clips FE result "
                   "to [nbr_min, nbr_max] of the k=dim+1 nearest psi_star "
                   "DOFs at each upstream coord. 'pick' (B.1) keeps the "
                   "FE result where it's in-bounds, else re-evaluates "
                   "via RBF for those DOFs. Default None (pure FE, can "
                   "overshoot).")
    p.add_argument('--t-tracking-window', type=str, default=None,
                   help="Sub-step T-data + mesh-quality tracking. "
                   "Format 'start,end' (inclusive). Wraps "
                   "mesh._deform_mesh, stokes.solve, adv_diff.solve "
                   "to log T hash, T min/max, signed triangle area "
                   "min/max, n inverted, and DOF Δ from previous "
                   "checkpoint. Used to localise where 'pepper' "
                   "scatter appears within a step.")
    p.add_argument('--diffusion-theta', type=float, default=0.5,
                   help="Adams-Moulton θ for the diffusive flux at "
                   "order 1: θ=0.5 is Crank-Nicolson (A-stable, NOT "
                   "L-stable, rings on stiff modes); θ=1.0 is Backward "
                   "Euler (L-stable, monotone for diffusion, first-order "
                   "accurate). Wires to adv_diff.DuDt.theta.")
    p.add_argument('--ale-correction', action='store_true',
                   help="Compute mesh velocity per step from "
                   "Δcoords/Δt and subtract from SLCN advection "
                   "velocity. Avoids the per-stage T-transport-by-"
                   "mesh-deformation artefact that shows up as ALE "
                   "blobs in the T field. Default off (preserves "
                   "original integrator-zoo behaviour).")
    p.add_argument('--structured', action='store_true',
                   help="Use the polar-quad structured annulus mesh.")
    p.add_argument('--snap-suffix', type=str, default="",
                   help="Suffix for snapshot dir + output png "
                   "(use to keep multiple runs separated).")
    p.add_argument('--schemes', type=str, default=None,
                   help="Comma-separated subset of schemes to run "
                   "(default: SL family minus ab2_sl).")
    p.add_argument('--v-degree', type=int, default=2,
                   help="Velocity polynomial degree (default 2 → P2/Q2).")
    p.add_argument('--p-degree', type=int, default=1,
                   help="Pressure polynomial degree (default 1 → P1/Q1).")
    p.add_argument('--t-degree', type=int, default=3,
                   help="Temperature polynomial degree (default 3 → P3). "
                   "Higher gives more DOFs per element and lower per-element "
                   "gradient, but worse Runge-style ringing potential.")
    p.add_argument('--remesh-at-step', type=int, default=0,
                   help="If > 0, reset mesh to its undeformed state at "
                   "this step number, kdtree-NN-project T and v from "
                   "deformed-mesh DOFs to undeformed-mesh DOFs, and "
                   "reset BDF surface history. Tests whether reverting "
                   "element distortion rescues the SLCN+CN ringing.")
    p.add_argument('--outer-refine-factor', type=float, default=1.0,
                   help="Cell-size reduction at the outer boundary "
                   "for graded mesh refinement. cellSizeOuter = "
                   "cellSize / outer_refine_factor. Default 1.0 = "
                   "uniform mesh; 2.0 = outer cells half size; 4.0 = "
                   "quarter size. Used to test whether boundary-layer "
                   "refinement suppresses SLCN+CN ringing.")
    p.add_argument('--winslow-iters', type=int, default=0,
                   help="Number of Winslow/Laplacian Jacobi sweeps "
                   "applied to interior mesh nodes after each step's "
                   "deformation. Boundary nodes pinned, so the deformed "
                   "surface is preserved. Default 0 = off. 5-10 typical. "
                   "Use to keep refined-BL cells from being crushed by "
                   "surface deformation.")
    p.add_argument('--winslow-alpha', type=float, default=0.5,
                   help="Under-relaxation factor for Winslow smoothing "
                   "(default 0.5). 1.0 = pure Jacobi; smaller = damped.")
    p.add_argument('--winslow-every-n-steps', type=int, default=0,
                   help="Apply Winslow smoothing once every N "
                   "completed timesteps (default 0 = never). The "
                   "smoothing is INFREQUENT and lives outside the "
                   "_step inner machinery so it doesn't interfere "
                   "with Picard/RK iterations or perturb T via the "
                   "Lagrangian DOF coupling. Use as a periodic "
                   "mesh-quality maintenance pass when accumulated "
                   "deformation starts to hurt element shapes.")
    p.add_argument('--stokes-tol', type=float, default=1.0e-5,
                   help="Stokes solver tolerance (default 1e-5).")
    p.add_argument('--dt-cap-mode', type=str, default='fixed',
                   choices=['none', 'fixed', 'relax'],
                   help="Δt cap mode. 'fixed' uses the thermal-CFL Δt "
                   "(returned by adv_diff.estimate_dt) as the per-step "
                   "scalar cap. 'relax' replaces it with dt_cap_c/γ_eff "
                   "(surface-relaxation rate). 'none' disables capping.")
    p.add_argument('--dt-cap-c', type=float, default=0.5,
                   help="Safety factor c for dt_cap_mode='relax' "
                   "(unused in 'fixed' mode; default 0.5).")
    p.add_argument('--per-step-profile', action='store_true',
                   help="Save the surface profile + diagnostics at "
                   "every step into the snapshot dir as "
                   "step_<scheme>_NNNN.npz. Used for time-series "
                   "analysis.")
    p.add_argument('--work-log', type=str, default=None,
                   help="Path to a CSV file to append per-step work "
                   "log entries.")
    p.add_argument('--target-time', type=float, default=None,
                   help="If set, stop the run when cumulative simulated "
                   "time exceeds this value (in addition to --n-steps). "
                   "For example 5.0 ≈ 5 thermal diffusion times.")
    args = p.parse_args()

    if args.p_degree >= args.v_degree:
        raise SystemExit(
            f"Inf–sup violation: need v_degree > p_degree, "
            f"got v={args.v_degree}, p={args.p_degree}.")

    if args.schemes is not None:
        schemes = [s.strip() for s in args.schemes.split(',') if s.strip()]
    else:
        # Skip ab2_sl (ruled out in continent benchmarking) and the
        # legacy non-SL schemes (no direct convection counterpart).
        schemes = ['fe_sl', 'rk2_sl', 'rk4_sl', 'bdf2_sl',
                   'rk2_load_sl', 'rk4_load_sl', 'bdf2_load_sl']

    # Re-target SNAP_DIR for structured runs / suffixed runs
    global SNAP_DIR
    if args.structured:
        SNAP_DIR = os.path.join(OUT_DIR, "convection_zoo_snapshots_struct")
    if args.snap_suffix:
        SNAP_DIR = SNAP_DIR + args.snap_suffix

    import time as _time
    # Open work log once for this whole invocation (append mode)
    worklog_fh = None
    if args.work_log:
        os.makedirs(os.path.dirname(os.path.abspath(args.work_log))
                    or '.', exist_ok=True)
        new_file = not os.path.exists(args.work_log)
        worklog_fh = open(args.work_log, 'a')
        if new_file:
            worklog_fh.write(
                "scheme,step,t,dt,h_pole,vrms,T_avg,Nu,"
                "h_max,h_rms,area_uw,delta_A,wall_step_s\n")

    for scheme in schemes:
        print(f"\n=== {scheme} ===", flush=True)
        state = _build(res=args.res, Ra=args.Ra,
                       structured=args.structured,
                       v_degree=args.v_degree,
                       p_degree=args.p_degree,
                       stokes_tol=args.stokes_tol,
                       fssa_theta=args.fssa_theta,
                       fssa_dt_ref=args.fssa_dt_ref,
                       rho_g=args.rho_g,
                       nitsche_penalty=args.nitsche_penalty,
                       t_degree=args.t_degree,
                       outer_refine_factor=args.outer_refine_factor)
        adv_diff = state['adv_diff']
        _install_t_tracking(state, args)
        # Runtime-injected knobs (not build parameters)
        state['anderson_m'] = int(args.anderson_m)
        # Adams-Moulton θ for the implicit diffusive flux (CN by default,
        # BE at θ=1.0). Set on the SemiLagrangian DDt directly so it's
        # picked up on every update_pre_solve.
        adv_diff.DuDt.theta = float(args.diffusion_theta)
        adv_diff.DFDt.theta = float(args.diffusion_theta)
        # Monotonicity limiter for the SL trace-back. Set on the
        # DDt instances directly (same pattern as theta). None = pure
        # FE (legacy, can overshoot), "clamp" = B.2, "pick" = B.1.
        adv_diff.DuDt.monotone_mode = args.monotone_mode
        adv_diff.DFDt.monotone_mode = args.monotone_mode
        print(f"diffusion AM θ = {adv_diff.DuDt.theta} "
              f"({'CN' if adv_diff.DuDt.theta == 0.5 else 'BE' if adv_diff.DuDt.theta == 1.0 else 'mixed'})",
              flush=True)
        _reset_gamma_history()
        os.makedirs(SNAP_DIR, exist_ok=True)

        # Optional restart-from-checkpoint: load mesh coords + T + V
        # from a saved snapshot. Internal solver state (Anderson, KSP,
        # DDt history, gamma history) stays freshly-built — this is
        # the diagnostic for "does internal state matter?"
        if args.restart_suffix is not None and args.restart_step is not None:
            import underworld3 as uw
            restart_scheme = (args.restart_scheme
                              or args.schemes.split(',')[0])
            restart_dir = os.path.join(
                OUT_DIR,
                f"convection_zoo_snapshots{args.restart_suffix}")
            restart_root = (f"uw_{restart_scheme}_"
                            f"step{args.restart_step:04d}")
            mesh_h5 = os.path.join(restart_dir,
                                   f"{restart_root}.mesh.00000.h5")
            print(f"RESTART: loading mesh coords from {mesh_h5}",
                  flush=True)
            # Load deformed mesh coords + apply.
            saved_mesh = uw.discretisation.Mesh(mesh_h5)
            saved_X = saved_mesh.X.coords.copy()
            # Drop saved_mesh; we keep the freshly-built state['mesh']
            # and just transplant the coords onto it.
            state['mesh']._deform_mesh(saved_X)
            del saved_mesh
            # Re-read T and V onto the live MeshVariables.
            pair_tag = f"v{args.v_degree}p{args.p_degree}"
            state['t_soln'].read_timestep(
                restart_root, f"T_conv_{pair_tag}", 0,
                outputPath=restart_dir)
            state['v'].read_timestep(
                restart_root, f"V_conv_{pair_tag}", 0,
                outputPath=restart_dir)
            # Restoring h_pole/t_sim: load from per-step npz if present.
            prof_npz = os.path.join(
                restart_dir,
                f"profile_{restart_scheme}_"
                f"step{args.restart_step:04d}.npz")
            if os.path.exists(prof_npz):
                data = np.load(prof_npz)
                t_sim_start = float(data.get('t', 0.0))
                h_pole_start = float(data.get('h_pole', 0.0))
                print(f"RESTART: t_sim={t_sim_start:.6f} "
                      f"h_pole={h_pole_start:.6f}", flush=True)
            else:
                t_sim_start = 0.0; h_pole_start = 0.0
                print("RESTART: no profile.npz, t_sim=0 h_pole=0",
                      flush=True)
        else:
            t_sim_start = None
            h_pole_start = None

        h_pole = h_pole_start if h_pole_start is not None else 0.0
        t_sim = t_sim_start if t_sim_start is not None else 0.0
        for s in range(args.n_steps):
            t_step_start = _time.time()
            # Optional remesh hook: at the start of step --remesh-at-step,
            # apply UW3 adaptive remesh with a uniform metric. PRESERVES
            # the deformed boundary shape (free surface) and retessellates
            # the interior at the same target edge length, automatically
            # transferring T+v to the new mesh via PETSc's variable
            # interpolation.
            if args.remesh_at_step > 0 and (s + 1) == args.remesh_at_step:
                import underworld3 as uw
                mesh_r = state['mesh']
                old_h_max = float((np.linalg.norm(mesh_r.X.coords, axis=1)
                                   - state['r_o']).max())
                old_n_t = int(state['t_soln'].data.shape[0])
                old_n_v = int(state['v'].data.shape[0])
                # Estimate the existing target edge length from the
                # mean current edge length (so the new mesh is the same
                # resolution, just with regularised element shapes).
                X = mesh_r.X.coords
                # Approximate edge length from typical nearest-neighbour
                # vertex spacing (cheap and good enough for a uniform
                # metric).
                from scipy.spatial import cKDTree as _cKDT
                _kdt = _cKDT(X)
                _d, _ = _kdt.query(X, k=2)
                target_h = float(np.mean(_d[:, 1]))
                # Build constant metric h_values = target_h at every node.
                h_values = np.full(X.shape[0], target_h)
                metric = uw.adaptivity.create_metric(
                    mesh_r, h_values, name=f"adapt_metric_step{s+1}")
                # Apply adaptation — PRESERVES boundary shape, retessel-
                # lates interior, transfers all variables onto new mesh.
                mesh_r.adapt(metric, verbose=True)
                # Reset BDF surface history — Fourier coefs of h^{n-1}
                # were on the old mesh; safer to bootstrap with BDF1.
                _H_PREV_FOURIER[0] = None
                _BDF_BOOTSTRAPPED[0] = False
                _DT_PREV[0] = None
                _reset_gamma_history()
                # Reset adv_diff DuDt so SLCN re-snapshots T from scratch
                # on the new mesh.
                adv_diff.DuDt._history_initialised = False
                adv_diff.DuDt._n_solves_completed = 0
                # Zero v_mesh (mesh has changed; previous Δcoords/Δt is
                # no longer meaningful).
                state['v_mesh'].data[...] = 0.0
                # Recompute internal_idx / internal_th since DOF count
                # and ordering may have changed after adapt.
                # (For simplicity here, leave them — the diagnostic
                # surface profile will be slightly off until next remesh.)
                new_h_max = float((np.linalg.norm(mesh_r.X.coords, axis=1)
                                   - state['r_o']).max())
                new_n_t = int(state['t_soln'].data.shape[0])
                new_n_v = int(state['v'].data.shape[0])
                print(f"  REMESH at step {s+1}: adapt(uniform metric "
                      f"h={target_h:.3e}). "
                      f"h_max {old_h_max:.3e} → {new_h_max:.3e} "
                      f"(boundary preserved). "
                      f"T DOFs {old_n_t}→{new_n_t}, "
                      f"v DOFs {old_n_v}→{new_n_v}. "
                      f"Reset BDF/Anderson/SL history.",
                      flush=True)
            # Schedule Nitsche penalty: constant during bootstrap, then
            # linear ramp from initial to floor over the release phase,
            # then constant at floor. Floor=0 = bare free surface; floor
            # = ρg·Δt is the FSSA-equivalent stabilised free surface.
            if (args.nitsche_bootstrap_steps > 0
                    or args.nitsche_release_steps > 0):
                K0 = state['nitsche_penalty_initial']
                K_floor = float(args.nitsche_floor)
                bs = args.nitsche_bootstrap_steps
                rs = args.nitsche_release_steps
                if s < bs:
                    K_now = K0
                elif rs > 0 and s < bs + rs:
                    fraction = max(0.0, 1.0 - (s - bs) / rs)
                    K_now = K_floor + (K0 - K_floor) * fraction
                else:
                    K_now = K_floor
                state['nitsche_K_var'].data[:, 0] = K_now
            # Thermal CFL caps Δt for the convection regime — Δh per
            # step is then structurally small (free-surface dynamics
            # is "free-riding" the thermal time scale), which is
            # exactly the regime where the load-shortcut family is
            # expected to shine.
            dt_thermal = (float(adv_diff.estimate_dt())
                          * float(args.dt_thermal_factor))
            # Capture v-DOF positions BEFORE the mesh moves so we can
            # compute v_mesh = Δcoords/Δt for the ALE correction.
            v_mesh_var = state['v_mesh']
            prev_v_dof_coords = v_mesh_var.coords.copy()
            # Instrumentation: tag the step + phase for restore logging.
            globals()['_DEBUG_STEP'] = s + 1
            globals()['_DEBUG_PHASE'] = 'step'
            T_pre_step = state['t_soln'].data[:, 0].copy()
            _T_TRACK['step'] = s + 1
            _t_track("step_begin")
            h_pole, dt = _step(state, scheme, args.dt_factor,
                               dt_cap=dt_thermal,
                               dt_cap_mode=args.dt_cap_mode,
                               dt_cap_c=args.dt_cap_c)
            # Optional Winslow smoothing of interior nodes after the
            # step's mesh deformation. Boundary stays put, interior
            # re-equilibrates so high-aspect-ratio refined cells don't
            # get crushed by surface motion.
            # Infrequent end-of-step Winslow mesh smoothing. Applied
            # AFTER all the step's Stokes/Picard/SLCN work is done,
            # so the smoothing doesn't interact with iterations or
            # perturb the T evolution via Lagrangian DOF coupling.
            # Trigger when (s+1) is a multiple of every-n-steps AND
            # winslow_iters > 0.
            if (args.winslow_iters > 0
                    and args.winslow_every_n_steps > 0
                    and (s + 1) % args.winslow_every_n_steps == 0):
                winslow_smooth_interior(
                    state,
                    n_iters=args.winslow_iters,
                    alpha=args.winslow_alpha)
            # ALE correction: compute mesh velocity at v's DOFs and
            # write into v_mesh, so SLCN's V_fn = v - v_mesh traces
            # back at the right relative velocity. Off → leaves v_mesh
            # at zero (recovers original behaviour).
            if args.ale_correction and dt > 0:
                new_v_dof_coords = v_mesh_var.coords
                v_mesh_var.data[...] = (
                    new_v_dof_coords - prev_v_dof_coords) / dt
            # Operator-split: advance T after the mesh has moved (and
            # v has been re-solved at the end of _step). T's DOF
            # values are Lagrangian — they ride the mesh deformation
            # automatically — so this is geometrically consistent.
            globals()['_DEBUG_PHASE'] = 'advdiff'
            T_pre_advdiff = state['t_soln'].data[:, 0].copy()
            adv_diff.solve(timestep=dt, zero_init_guess=False,
                           _evalf=getattr(args, 'rbf_advection', False))
            T_post_raw = state['t_soln'].data[:, 0].copy()
            n_below = int(np.sum(T_post_raw < -1.0e-3))
            n_above = int(np.sum(T_post_raw > 1.0 + 1.0e-3))
            dT_max = float(np.abs(T_post_raw - T_pre_advdiff).max())
            dT_step_max = float(np.abs(T_post_raw - T_pre_step).max())
            # Defensive clip: T must lie in [T_lower_BC, T_upper_BC] = [0, 1]
            # by physical constraint. SLCN+CN can ring on under-resolved
            # sharp BLs in deformed elements, producing non-physical
            # overshoots. Clipping prevents the next step from amplifying
            # the spike via positive feedback.
            # TODO(env-flag-hack): UW_LAUNCH_CLIP — paired with the
            # monkey-patch in _build(). Remove both when launch-point
            # clipping has a proper API.
            # End-result clip is only applied when launch-point clip is
            # OFF — the two are mutually exclusive (launch-point catches
            # the overshoot at the source so end-result clipping is
            # unnecessary).
            launch_clip_active = (
                os.environ.get("UW_LAUNCH_CLIP", "0") == "1")
            if getattr(args, 'clip_t', False) and not launch_clip_active:
                np.clip(state['t_soln'].data[:, 0], 0.0, 1.0,
                        out=state['t_soln'].data[:, 0])
                clipped = (n_below + n_above) > 0
            else:
                clipped = False
            T_post = state['t_soln'].data[:, 0]
            print(f"  T_DIAG step={s+1}: pre_step=[{T_pre_step.min():+.4f},"
                  f"{T_pre_step.max():+.4f}] "
                  f"pre_ad=[{T_pre_advdiff.min():+.4f},"
                  f"{T_pre_advdiff.max():+.4f}] "
                  f"post_raw=[{T_post_raw.min():+.4f},"
                  f"{T_post_raw.max():+.4f}] "
                  f"post=[{T_post.min():+.4f},{T_post.max():+.4f}] "
                  f"n<-1e-3={n_below} n>1+1e-3={n_above} "
                  f"|ΔT_advdiff|max={dT_max:.3e} "
                  f"|ΔT_step|max={dT_step_max:.3e}"
                  f"{' CLIPPED' if clipped else ''}",
                  flush=True)
            wall_step = _time.time() - t_step_start
            t_sim += dt

            diag = None
            if args.per_step_profile or args.work_log:
                diag = _convection_diagnostics(state)

            if args.per_step_profile and diag is not None:
                mesh_p = state['mesh']
                int_idx = state['internal_idx']
                int_th = state['internal_th']
                upos = mesh_p.X.coords[int_idx]
                ur = np.sqrt(upos[:, 0] ** 2 + upos[:, 1] ** 2)
                udr = ur - state['r_o']
                np.savez(os.path.join(
                    SNAP_DIR, f"step_{scheme}_{s+1:04d}.npz"),
                    theta=int_th, dr=udr, h_pole=h_pole,
                    t=t_sim, dt=dt,
                    vrms=diag['vrms'], T_avg=diag['T_avg'],
                    Nu=diag['Nu'],
                    h_max=diag['h_max'], h_rms=diag['h_rms'],
                    area_uw=diag['area_uw'],
                    area_uw_initial=state['area_uw_initial'],
                    delta_A=diag['delta_A'])
            if worklog_fh is not None and diag is not None:
                worklog_fh.write(
                    f"{scheme},{s+1},{t_sim:.6e},{dt:.6e},"
                    f"{h_pole:+.6e},{diag['vrms']:.6e},"
                    f"{diag['T_avg']:.6e},{diag['Nu']:+.6e},"
                    f"{diag['h_max']:.6e},{diag['h_rms']:.6e},"
                    f"{diag['area_uw']:.6e},{diag['delta_A']:+.6e},"
                    f"{wall_step:.4f}\n")
                worklog_fh.flush()
            if diag is not None:
                print(f"  step {s+1}: t={t_sim:.3f} Δt={dt:.3e} "
                      f"vrms={diag['vrms']:.3e} Nu={diag['Nu']:+.3f} "
                      f"h_max={diag['h_max']:.2e} "
                      f"ΔA/A={diag['delta_A']:+.2e} "
                      f"wall={wall_step:.2f}s", flush=True)
            else:
                print(f"  step {s+1}: t={t_sim:.3f} Δt={dt:.3e} "
                      f"h_pole={h_pole:+.3e} wall={wall_step:.2f}s",
                      flush=True)

            if (args.capture_every > 0
                    and (s + 1) % args.capture_every == 0):
                _capture(state, scheme, f'step{s+1:04d}', h_pole, t_sim)

            if args.target_time is not None and t_sim >= args.target_time:
                print(f"  reached target_time={args.target_time}; "
                      f"stopping at step {s+1}", flush=True)
                break

        _capture(state, scheme, 'final', h_pole, t_sim)

    if worklog_fh is not None:
        worklog_fh.close()


if __name__ == "__main__":
    main()
