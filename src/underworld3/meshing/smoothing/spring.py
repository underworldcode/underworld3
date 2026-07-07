"""The spring-equilibrium mover (``method="spring"``, the default
metric path). See the package docstring for the module map.
"""

import warnings

import numpy as np

import underworld3 as uw

from .graph import (_edge_pairs, _tri_cells, _signed_areas,
                    _global_sum, _global_min, _global_max)


# Cached spring-smoother topology state keyed by (mesh-id,
# pinned-labels, topology): the edge vertex-index pairs and per-node
# incident-edge degree. Rebuilt automatically on a topology change
# (remesh / adapt / repartition), which produces a new cache key.
_SPRING_CACHE: dict = {}


def _spring_equilibrium_mover(mesh, metric, pinned_labels, verbose,
                    max_cg_iters=300,
                    boundary_slip=False, shape_w=1.0, size_w=8.0,
                    n_sweeps=None):
    r"""Metric-driven mesh grading by elastic-spring equilibrium.

    Every mesh edge is a linear spring whose *rest length* is set
    from the target density,

    .. math::

        L^0_{ij} \;\propto\; \rho_{\mathrm{tgt}}^{-1/d},

    scaled once so the total rest length equals the total current
    edge length (overall scale preserved — pure redistribution).
    The interior nodes are moved to the **mechanical equilibrium**
    by *minimising the truss energy*

    .. math::

        E(\mathbf{x}) \;=\; \tfrac12 \sum_{e}
        \big(\,|\mathbf{x}_i-\mathbf{x}_j| - L^0_e\,\big)^2

    over the free (non-pinned) nodes with **nonlinear conjugate
    gradients** (Polak–Ribière⁺) and an Armijo line search whose
    trial step is rejected if any cell would invert. Solving the
    equilibrium — rather than creeping with damped Jacobi sweeps,
    which stall against a per-sweep global tangle freeze — is what
    lets the absolute rest-length target actually grade the mesh
    toward spacing ``∝ ρ_tgt^{-1/d}``.

    ``ρ_tgt`` is Lagrangian (``metric = f(r0)`` with ``r0`` a frozen
    mesh variable), so the rest lengths are fixed per material node
    (computed once) and the *design* grading is restored even after
    the mesh deformed. Uniform ``ρ_tgt`` ⇒ all rest lengths equal
    the mean edge length ⇒ only a benign mild regularisation toward
    uniform spacing (no grading change).

    ``max_cg_iters`` caps the CG iterations (CG converges far faster
    than the old Jacobi sweep budget); ``n_sweeps`` is its deprecated
    former name, accepted for one cycle with a DeprecationWarning.
    The old ``relax`` / ``step_frac`` parameters were unused on the
    equilibrium path (the CG line search controls the step and the
    inversion guard) and have been removed. ``n_iters`` / ``alpha``
    do not apply.
    """
    if n_sweeps is not None:
        warnings.warn(
            "the 'n_sweeps' argument is renamed; use max_cg_iters= "
            "(it caps the nonlinear-CG iterations, not Jacobi sweeps)",
            DeprecationWarning, stacklevel=2)
        max_cg_iters = n_sweeps
    pinned_labels = tuple(pinned_labels)
    dm = mesh.dm
    pStart, pEnd = dm.getDepthStratum(0)
    cStart, cEnd = dm.getHeightStratum(0)
    cone_size = dm.getConeSize(cStart) if cEnd > cStart else 0
    n_verts = pEnd - pStart
    key = (id(mesh), pinned_labels,
           n_verts, cEnd - cStart, cone_size)

    cache = _SPRING_CACHE.get(key)
    if cache is None:
        edges = _edge_pairs(dm)
        if edges.shape[0] == 0:
            return
        deg = np.bincount(
            edges.ravel(), minlength=n_verts).astype(np.float64)
        deg[deg == 0.0] = 1.0
        _SPRING_CACHE[key] = (edges, deg)
    else:
        edges, deg = cache

    tris = _tri_cells(dm)
    cdim = mesh.cdim
    v0 = edges[:, 0]
    v1 = edges[:, 1]

    coords = np.asarray(mesh.X.coords, dtype=np.float64).copy()

    # Boundary tangential slip via the mesh-owned contract
    # (boundary-slip-strategy.md): each slip vertex slides tangentially and
    # snaps back onto its bounding surface (radial ring / plane / facet);
    # non-slip, junction, and degenerate-normal vertices pin. Replaces the
    # per-ring COM radial snap (one node/ring anchored the rotation gauge);
    # the global inversion guard below still blocks a slip node overtaking a
    # neighbour, and tangential θ-drift is a harmless re-parameterisation.
    is_pinned, _project = mesh.boundary_slip(
        boundary_slip, reference_coords=coords, boundary_labels=pinned_labels)

    free = ~is_pinned

    # ===== Volumetric spring network (shape ⟂ size, decoupled) ====
    # EQUAL edge springs (uniform rest length L̄ = current mean
    # edge) are a pure SHAPE regulariser → equant cells, resists
    # the slivers/degeneracy the graded-edge form produced. The
    # SIZE grading lives entirely in a per-CELL area ("volumetric")
    # constraint: each triangle's area is driven to a target
    # A0 ∝ 1/ρ_tgt (scaled so ΣA0 = Σ(initial area) ⇒ total area
    # conserved, pure redistribution). Both energy terms are
    # written as *relative* squared errors so the shape/size
    # weights (shape_w, size_w) are pure dimensionless knobs.
    e_vec = coords[v1] - coords[v0]
    L_cur = np.linalg.norm(e_vec, axis=1)
    sum_L = float(L_cur.sum())
    n_e = float(L_cur.size)
    sum_L = _global_sum(sum_L)
    n_e = _global_sum(n_e)
    Lbar = sum_L / max(n_e, 1.0)          # uniform edge rest length
    L0 = np.full_like(L_cur, Lbar)
    L0_mean = Lbar

    # Per-cell target area from ρ_tgt at the (initial) centroid.
    # Lagrangian metric ⇒ computed ONCE (rides material points).
    if tris is not None:
        ca = coords[tris[:, 0]]
        cb = coords[tris[:, 1]]
        cc = coords[tris[:, 2]]
        cent = (ca + cb + cc) / 3.0
        rho_c = np.asarray(
            uw.function.evaluate(metric, cent)).reshape(-1)
        rho_c = np.maximum(rho_c, 1.0e-30)
        a_init = np.abs(_signed_areas(coords, tris))
        inv = 1.0 / rho_c
        sA = float(a_init.sum())
        sI = float(inv.sum())
        sA = _global_sum(sA)
        sI = _global_sum(sI)
        A0 = (sA / max(sI, 1.0e-30)) * inv     # ΣA0 = Σa_init
        A0 = np.maximum(A0, 1.0e-30)
        ti0, ti1, ti2 = tris[:, 0], tris[:, 1], tris[:, 2]
    else:
        A0 = None

    # ---- Solve the truss EQUILIBRIUM, not Jacobi creep ----------
    # Minimise the spring energy  E(x) = ½ Σ_e (|x_i−x_j| − L0_e)²
    # over the interior nodes (boundary pinned) by nonlinear
    # conjugate gradients (Polak–Ribière⁺) with an Armijo line
    # search whose trial step is REJECTED if any cell would invert
    # — the tangle guard is inside the optimiser, so it converges to
    # the true equilibrium instead of stalling against a per-sweep
    # global freeze (the Jacobi relaxation's failure mode).
    free_idx = np.nonzero(free)[0]
    n_free = int(free_idx.size)
    # TODO(BUG): rank-LOCAL early return before the collective CG loop.
    # Under MPI, a rank whose local vertices are all pinned returns here
    # while the other ranks proceed into _global_sum/_global_min
    # collectives -> deadlock. Consistent with the documented
    # serial-exact status of this path, but a latent parallel hazard;
    # the exit decision should be reduced globally (as the OT/MA movers
    # do) before this mover is promoted to parallel-exact.
    if n_free == 0:
        mesh._deform_mesh(coords)
        return

    if tris is not None:
        orient = np.sign(np.median(_signed_areas(coords, tris)))
        orient = orient if orient != 0.0 else 1.0

    def _feasible(X):
        if tris is None:
            return True
        amin = _global_min((_signed_areas(X, tris) * orient).min())
        return amin > 0.0

    have_area = (A0 is not None) and (cdim == 2)

    def _tri_signed(X):
        a, b, c = X[ti0], X[ti1], X[ti2]
        return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                      - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))

    def _energy(X):
        ev = X[v1] - X[v0]
        L = np.sqrt((ev * ev).sum(axis=1))
        re = (L - Lbar) / Lbar               # relative edge error
        E = shape_w * _global_sum((re * re).sum())
        if have_area:
            area = orient * _tri_signed(X)
            ra = (area - A0) / A0            # relative area error
            E += size_w * _global_sum((ra * ra).sum())
        return E

    def _energy_grad(X):
        ev = X[v1] - X[v0]
        L = np.sqrt((ev * ev).sum(axis=1))
        Ls = np.maximum(L, 1.0e-30)
        re = (L - Lbar) / Lbar
        E = shape_w * _global_sum((re * re).sum())
        G = np.zeros_like(X)
        # equal-spring shape term: 2·shape_w·re/(Lbar·L)·ev
        ce = (2.0 * shape_w * re / (Lbar * Ls))[:, None]
        np.add.at(G, v1, ce * ev)
        np.add.at(G, v0, -ce * ev)
        if have_area:
            a, b, c = X[ti0], X[ti1], X[ti2]
            S = 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                       - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
            area = orient * S
            ra = (area - A0) / A0
            E += size_w * _global_sum((ra * ra).sum())
            # ∂(area)/∂· = orient · ∂S/∂· (signed-area vertex grads)
            fac = (2.0 * size_w * ra / A0 * orient)[:, None]
            gA = np.empty_like(a)
            gB = np.empty_like(a)
            gC = np.empty_like(a)
            gA[:, 0] = 0.5 * (b[:, 1] - c[:, 1])
            gA[:, 1] = 0.5 * (c[:, 0] - b[:, 0])
            gB[:, 0] = 0.5 * (c[:, 1] - a[:, 1])
            gB[:, 1] = 0.5 * (a[:, 0] - c[:, 0])
            gC[:, 0] = 0.5 * (a[:, 1] - b[:, 1])
            gC[:, 1] = 0.5 * (b[:, 0] - a[:, 0])
            np.add.at(G, ti0, fac * gA)
            np.add.at(G, ti1, fac * gB)
            np.add.at(G, ti2, fac * gC)
        G[~free] = 0.0
        return E, G

    # Jacobi (diagonal) preconditioner: the truss Hessian is
    # graph-Laplacian-structured (cond ~ (1/h)²), so plain CG crawls
    # for fine meshes. M⁻¹ = diag(1/deg) — the Laplacian diagonal
    # scale, free here since `deg` is already cached — clusters the
    # spectrum and gives the order-of-magnitude convergence speed-up
    # that turns "stuck at ~1.04" into the true graded minimum.
    invdeg = (1.0 / deg)[:, None]

    X = _project(coords.copy())
    E, G = _energy_grad(X)
    g0 = max(_global_sum((G * G).sum()) ** 0.5, 1.0e-30)
    r = -G
    s = r * invdeg
    s[~free] = 0.0
    d = s.copy()
    delta_new = _global_sum((r * s).sum())
    dmax = _global_max(max(float(np.linalg.norm(
        d[free_idx], axis=1).max()), 1.0e-30))
    t0 = 0.5 * L0_mean / dmax
    c_arm = 1.0e-4
    max_iter = int(max_cg_iters)
    for it in range(max_iter):
        gnorm = _global_sum((G * G).sum()) ** 0.5
        if gnorm <= 1.0e-8 * g0:
            break
        slope = _global_sum((G * d).sum())       # = −(r·d)
        if slope >= 0.0:                     # not descent → restart
            d = s.copy()
            slope = _global_sum((G * d).sum())
            if slope >= 0.0:
                break
        t = t0
        accepted = False
        for _ls in range(50):
            Xt = X.copy()
            Xt[free_idx] += t * d[free_idx]
            Xt = _project(Xt)                # slip nodes → boundary
            if _feasible(Xt):
                Et = _energy(Xt)
                if Et <= E + c_arm * t * slope:
                    accepted = True
                    break
            t *= 0.5
        if not accepted:
            break                            # at equilibrium / stuck
        Et, Gt = _energy_grad(Xt)
        r_new = -Gt
        s_new = r_new * invdeg
        s_new[~free] = 0.0
        delta_old = delta_new
        delta_mid = _global_sum((r_new * s).sum())
        delta_new = _global_sum((r_new * s_new).sum())
        beta = max(0.0, (delta_new - delta_mid)
                   / max(delta_old, 1.0e-30))   # preconditioned PR⁺
        X, E, G = Xt, Et, Gt
        d = s_new + beta * d
        s = s_new
        t0 = min(2.0 * t, 100.0 * t0)        # grow but stay sane

        if verbose and (it % 25 == 0 or it == max_iter - 1):
            ev = X[v1] - X[v0]
            L = np.sqrt((ev * ev).sum(axis=1))
            rms = (_global_sum(((L - L0) ** 2).sum())
                   / max(_global_sum(L0.size), 1.0)) ** 0.5
            uw.pprint(
                f"  spring PCG iter {it+1}/{max_iter}: "
                f"E={E:.4e}  rms(L-L0)/L0="
                f"{rms / max(L0_mean, 1e-30):.3e}  |g|={gnorm:.2e}")

    coords = X
    mesh._deform_mesh(coords)
