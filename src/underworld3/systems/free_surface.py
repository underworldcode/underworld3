r"""Prognostic, exponentially-relaxed free surface for viscous flow.

This module provides :class:`FreeSurface`, a manager that turns a configured
Stokes solve into a moving free surface using the three-number exponential
integrator: the surface height relaxes toward the stress-derived equilibrium
topography :math:`h_\infty` at a rate read from the instantaneous surface
velocity, with an update that is L-stable and cannot overshoot.

The physics, the benchmark evidence, and the design rationale are documented in
``docs/developer/design/FREESLIP_DYNAMIC_TOPOGRAPHY_FREESURFACE.md`` and the
consolidated ``EXPONENTIAL_FREE_SURFACE_PAPER_DRAFT.md``. This class packages the
validated recipe (rotated free-slip held lid, Consistent Boundary Flux
:math:`\sigma_{nn}` recovery, continuous pressure, material-surface advection)
so a user assembles it from one configured solver rather than by hand.
"""

import numpy as np
import sympy
from mpi4py import MPI

import underworld3 as uw
from underworld3 import function
from underworld3.coordinates import CoordinateSystemType


class FreeSurface:
    r"""Evolve a free surface by exponential relaxation toward stress equilibrium.

    The user configures ONE Stokes solve — the *free* (stress-free-top) solve that
    carries the physics: constitutive model, driving body force, wall boundary
    conditions and solver options, but **no** boundary condition on the free
    ``surface``. :class:`FreeSurface` reads that solve and derives the two companion
    solves it needs — a rotated free-slip *held* lid whose constraint reaction gives
    the equilibrium topography :math:`h_\infty`, and a *consistent* solve whose
    surface-normal velocity equals the realised relaxed rate so the surface stays a
    material boundary — then advances the surface each step:

    .. math::

        \gamma = \frac{\dot h}{h_\infty - h}, \qquad
        h \leftarrow h_\infty + (h - h_\infty)\,e^{-\gamma\,\Delta t}.

    Parameters
    ----------
    stokes : uw.systems.Stokes
        The free / stress-free-top solve, fully configured: constitutive model,
        ``bodyforce``, wall BCs (``add_rotated_freeslip_bc`` for free-slip walls,
        ``add_essential_bc`` for no-slip), and any ``petsc_options``. It must carry
        **no** boundary condition on ``surface`` — the stress-free top pins the
        pressure datum.
    surface : str
        Boundary label of the free surface (e.g. ``"Upper"`` / ``"Top"``).
    buoyancy_scale : float
        :math:`\rho g`, converting the recovered normal traction to a length
        (:math:`h_\infty = -(\sigma_{nn}-\overline{\sigma_{nn}})/\rho g`).
    normal : sympy 1×dim Matrix or None, optional
        Analytic outward surface normal for the rotated constraint and the
        normal-velocity read (e.g. ``X/|X|`` on a spherical cap). ``None`` uses the
        per-node geometric facet normal — correct as the surface deforms.
    composition : uw.discretisation.MeshVariable or None, optional
        A material field advected with the consistent surface velocity (e.g. a mesh
        level set holding a signed distance, or temperature). ``None`` evolves the
        surface alone (e.g. a topographic-relaxation benchmark).
    conserve : sympy expression or None, optional
        An integrand whose domain integral is held fixed by a uniform shift of
        ``composition`` after each advection — the enclosed-area conservation of a
        mesh level set. Requires ``composition``; the shift derivative is taken
        symbolically from this expression, so pass the same smoothed-Heaviside used
        in the body force.
    driving_buoyancy : sympy expression or None, optional
        Body force for the held solve. ``None`` reuses ``stokes.bodyforce``; because
        :math:`\sigma_{nn}` is mean-removed and a constant body force gives a uniform
        traction on the near-flat lid, the reused form and a driving-only form agree
        to leading order. Pass an explicit driving-only buoyancy for an exact match.
    smooth_length : float, optional
        Physical length for stress smoothing in the surface-velocity projection —
        mesh/order-independent, ``0`` disables.
    mass : {"lumped", "consistent"}, optional
        Boundary-mass de-smear for the :math:`\sigma_{nn}` recovery. ``"lumped"``
        (default) is monotone — no overshoot at a viscosity jump — and is the safe
        choice for driving a surface.
    max_surface_cfl : float, optional
        Surface-motion time-step cap: :meth:`estimate_dt` limits the step so the
        predicted surface displacement stays below this fraction of the local cell
        size. The exponential update is unconditionally stable, so this bounds mesh
        distortion (and keeps a multigrid hierarchy rebuild valid), not stability.
    consistent_constraint : {"strong", "penalty"}, optional
        How the consistent solve imposes :math:`\mathbf{u}\cdot\hat{\mathbf n} =
        \tilde u_n`, the condition that keeps the surface a *material* boundary.
        ``"strong"`` (default) uses the rotated per-node constraint — free-slip is
        then simply the :math:`\tilde u_n = 0` member of the same family — and
        imposes the rate exactly. ``"penalty"`` imposes it weakly; it tolerates a
        datum carrying a small net flux but leaks, both in the rate it delivers and
        in the volume it passes through the surface.
    consistent_penalty : float, optional
        Penalty magnitude used when ``consistent_constraint="penalty"``.
    background_buoyancy : None, "analytic", or sympy expression, optional
        REQUIRED (non-None) whenever the body force retains the :math:`\rho_0`
        background (full Boussinesq, :math:`\rho = \rho_0(1-\alpha\Delta T)`) — the
        formulation needed for the surface restoring force. The recovered held-lid
        reaction then contains the self-load of the current shape, which must be
        removed before the reduced-form sign conversion (otherwise
        :math:`h_\infty = -h + \mathrm{drive}/\rho g`: the surface parks at half its
        equilibrium with steady flow through it). ``"analytic"`` (recommended)
        subtracts the geometric current height — no extra solve. A sympy expression
        (the background body force, e.g. ``-rho_g * rhat``) instead runs a twin held
        solve and subtracts its recovery — the exact reference mode. ``None``
        (default) is the reduced/driving-only formulation, unchanged.
    verbose : bool, optional
        Report per-step surface diagnostics through :func:`uw.mpi.rank`-safe output.
    """

    def __init__(
        self,
        stokes,
        surface,
        buoyancy_scale=1.0,
        normal=None,
        composition=None,
        conserve=None,
        driving_buoyancy=None,
        smooth_length=0.0,
        mass="lumped",
        max_surface_cfl=0.5,
        tangent_advect=None,
        tangent_spectral_modes=0,
        surface_mask=None,
        surface_filter=0,
        consistent_constraint="strong",
        consistent_penalty=1.0e5,
        background_buoyancy=None,
        verbose=False,
    ):
        self.free = stokes
        self.mesh = stokes.mesh
        self.surface = surface
        self.buoyancy_scale = buoyancy_scale
        self.normal = normal
        self.composition = composition
        self._conserve_integrand = conserve
        self._smooth_length = smooth_length
        # sigma_nn de-smear: "lumped" is the monotone 2D default; on a 3D P2
        # trace the lumped vertex mass is identically zero, and the consistent
        # P2 mass carries the vertex-integral checkerboard (#404 hold) exactly
        # at the vertices the P1 h_inf field reads — so 3D translates the
        # default to the sound P1-PROJECTED recovery (boundary_flux mass="p1").
        if stokes.mesh.dim == 3 and mass == "lumped":
            mass = "p1"
        self._mass = mass
        self.max_surface_cfl = max_surface_cfl
        # First-pass along-surface (tangential) transport of the surface fields:
        # None (off), "shape" (transport the geometry h — the standard operator split),
        # or "fields" (transport the driving h_inf, u_n). A diagnostic to size the term.
        self.tangent_advect = tangent_advect
        # >0 uses a diffusion-free spectral (Fourier) launch-point interpolation on a
        # periodic ring instead of linear; the mode count also caps facet-scale noise.
        self._spectral_modes = int(tangent_spectral_modes)
        self._surface_mask_fn = surface_mask
        # How the consistent solve imposes u.n = ũ_n: "strong" (rotated per-node
        # constraint, exact) or "penalty" (weak natural BC). See _build_consistent.
        if consistent_constraint not in ("strong", "penalty"):
            raise ValueError(
                f"consistent_constraint must be 'strong' or 'penalty', "
                f"got {consistent_constraint!r}"
            )
        self.consistent_constraint = consistent_constraint
        self._consistent_penalty = float(consistent_penalty)
        # Full-density (Boussinesq with the rho_0 background retained) runs must pass
        # the BACKGROUND part of the body force (e.g. -rho_0*g*rhat) here. h_inf is then
        # computed from the DIFFERENCE of two held-lid reactions — full minus
        # background-only, on the same geometry — which cancels the discretely-defective
        # current-shape-load leg of the recovery and isolates the driving support.
        # None (default) = the single-reaction reduced-formulation path, unchanged.
        self.background_buoyancy = background_buoyancy
        self.verbose = verbose

        # Continuous pressure is required for the CBF reaction on a simplex free
        # surface: discontinuous pressure makes sigma_nn a node-to-node Nyquist
        # zigzag on the boundary edges (design note, Hardening 6).
        if getattr(stokes, "_p_continuous", True) is False:
            uw.pprint(
                "FreeSurface: the free solve uses discontinuous pressure; on a simplex "
                "free surface the sigma_nn recovery zigzags. Prefer continuous pressure."
            )

        # Topography direction: vertical (last axis) on a Cartesian box/slab, radial
        # on a cylindrical annulus or spherical shell. The surface height and the
        # mesh deformation follow it. `_radial` steers the geometry (any dimension);
        # `_cylindrical` additionally selects the 2D ring's angular ordering.
        ctype = self.mesh.CoordinateSystem.coordinate_type
        self._cylindrical = ctype == CoordinateSystemType.CYLINDRICAL2D
        self._radial = self._cylindrical or ctype == CoordinateSystemType.SPHERICAL
        # 3D runs on the dimension-general primitives (facet trace-mass gauge,
        # directed flux strip, unordered surface gather, nodal carrier). The two
        # 2D-ONLY features are refused per piece rather than by a blanket guard:
        if self.mesh.dim == 3:
            if tangent_advect is not None:
                raise NotImplementedError(
                    "FreeSurface: tangential surface transport is 2D-only (it runs "
                    "on the ordered surface ring; a 3D counterpart needs surface FE "
                    "advection). Construct with tangent_advect=None in 3D."
                )
            if int(surface_filter) > 0:
                raise NotImplementedError(
                    "FreeSurface: the Taubin surface filter is 2D-only (1-D ring "
                    "stencil); pass surface_filter=0 in 3D (the trace-mass gauge "
                    "and the interior carrier do not require it)."
                )
        self._walls = self._classify_walls()
        # Reference-configuration surface nodes: used ONCE to match each surface node to
        # its row in the mesh coordinate field and in the derived surface fields. Row
        # indices are identities, so they stay valid as the mesh deforms — but these
        # COORDINATES do not, and nothing at run time may sample the solution at them
        # (see the _ring_coords property).
        self._surf_coords = self._surface_node_coords()
        self._surf_rows, self._surf_x = self._field_rows(self.mesh.X, self._surf_coords)
        # Global along-surface-ordered ring (theta on an annulus, x on a box), gathered
        # across ranks. The surface filter, the tangential transport and the arc-length
        # datum weights all run on it, so all three are parallel-correct and free of the
        # x-extreme degeneracy that collapses ring nodes near theta=0/pi. The surface
        # deforms only along the normal, so the ordering is invariant and is built once.
        self._build_ring_gather()

        # Optional taper on the surface rate: h_dot -> h_dot * mask(x). A mask that
        # falls to zero near a driven wall pins the surface where a stress-free top
        # would otherwise fight an imposed wall velocity (the outflow-corner spike),
        # leaving the interior free. Evaluated once on the (Cartesian-invariant) x.
        self._surface_mask = None
        if surface_mask is not None:
            mask = np.asarray(function.evaluate(surface_mask, self._surf_coords)).flatten()
            self._surface_mask = mask[np.argsort(self._surf_coords[:, 0])]

        # Equilibrium-topography field, filled on the surface nodes by the held solve.
        self._hinf_field = uw.discretisation.MeshVariable(
            "h_inf", self.mesh, vtype=uw.VarType.SCALAR, degree=1, continuous=True
        )
        self._hinf_rows, _ = self._field_rows(self._hinf_field, self._surf_coords)

        self._build_surface_velocity_projection()
        self._build_held(driving_buoyancy)
        self._build_consistent()
        self._build_interior_diffuser()
        self._filter_iters = int(surface_filter)
        if composition is not None:
            self._build_composition_transport()

        self._h_inf = None  # recovered in solve(), consumed in advance()

    # -- geometry / boundary bookkeeping -------------------------------------

    @property
    def _ring_coords(self):
        r"""LIVE surface-node coordinates, in the fixed ring order the surface arrays use.

        Read from the mesh coordinate field every time, because the surface moves: a
        snapshot taken at construction falls behind the deforming boundary and any
        solution sampled at it is read from the *interior*, a growing fraction of a cell
        below the true surface. That corrupts :math:`\dot h` itself — and hence
        :math:`\gamma`, the tangential transport and the realised rate the consistent
        solve is asked to reproduce — with an error that grows with the deformation.
        :meth:`_current_shape` already reads the geometry live; this keeps the velocity
        sampling consistent with it.

        The ordering is fixed at construction (``_surf_rows``) and is NOT re-derived
        here: the surface moves only along the normal, so the along-surface order is
        invariant, whereas re-sorting live coordinates by ``x`` could permute neighbours
        near the annulus ``theta = 0, pi`` extremes where ``dx/dtheta -> 0``.
        """
        return np.ascontiguousarray(self.mesh.X.coords[self._surf_rows])

    def _surface_height(self, coords):
        """The coordinate along the topography direction: the last axis on a Cartesian
        box/slab, the radius on a cylindrical annulus or spherical shell."""
        if self._radial:
            return np.linalg.norm(coords, axis=1)
        return np.asarray(coords[:, -1], dtype=float)

    def _normal_direction(self, coords):
        """Per-node unit vectors along the topography direction that the surface
        increment is deformed along — vertical (Cartesian) or radial (annulus /
        spherical shell); dimension-general."""
        # TODO(BUG): this is unconditionally +radial, but since #560 the geometric
        # constraint normal (and so the recovered σ_nn and h_∞) is the DOMAIN's
        # outward normal, which on a CONCAVE surface — an inner arc / CMB free
        # surface — is −radial. The relaxation would then drive the surface away
        # from equilibrium instead of toward it. Unreachable today: every curved
        # free surface in the repo passes an explicit normal=rhat on an OUTER
        # boundary, and every Cartesian one is a flat Top. Needs the sign taken from
        # the same source as h_∞ before an inner free surface is supported.
        if self._radial:
            r = np.linalg.norm(coords, axis=1)
            r[r == 0.0] = 1.0
            return coords / r[:, None]
        n = np.zeros_like(coords)
        n[:, -1] = 1.0
        return n

    def _classify_walls(self):
        r"""Read the free solve's wall BCs as ``{boundary: ("freeslip"|"noslip", normal)}``.

        Free-slip walls are the rotated-free-slip constraints (the recommended
        primitive); essential walls carry their prescribed value verbatim — including a
        NON-zero driving velocity (a plate push), not just no-slip. The surface label is
        excluded (the free solve leaves it stress-free). Both are replayed on the derived
        solves so they see the same forcing.
        """
        walls = {}
        for boundary, wall_normal in getattr(self.free, "_rotated_freeslip_bcs", []):
            if boundary != self.surface:
                walls[boundary] = ("freeslip", wall_normal)
        for bc in self.free.essential_bcs:
            if bc.boundary != self.surface and bc.boundary not in walls:
                # bc.fn is an immutable dim x 1 matrix; add_essential_bc wants a mutable
                # sympy.Matrix in the value-first (column, oo-masked) form.
                walls[bc.boundary] = ("essential", sympy.Matrix(bc.fn))
        return walls

    def _apply_walls(self, solver):
        """Replay the classified wall BCs onto a derived solve (rotated free-slip lid
        pattern: rotate every free-slip wall so shared corners stay consistent; essential
        walls keep their prescribed value)."""
        for boundary, (kind, spec) in self._walls.items():
            if kind == "freeslip":
                solver.add_rotated_freeslip_bc(0.0, boundary, normal=spec)
            else:
                solver.add_essential_bc(spec, boundary)

    def _surface_node_coords(self):
        r"""Coordinates of the vertices on ``surface``, from the DMPlex boundary
        stratum (label-based, parallel-correct) — not a radial ``|y-H|`` heuristic.

        Each rank returns its local surface vertices; the stratum closure resolves
        the facet-to-vertex map so this is exact on deformed and curved boundaries.
        """
        from underworld3.utilities.boundary_flux import _boundary_stratum_is

        dm = self.mesh.dm
        facet_is = _boundary_stratum_is(dm, self.mesh, self.surface)
        # In parallel a rank may own NO surface facets → a NULL-handle IS; any
        # method call on it segfaults (the house guard every other stratum-IS
        # call site carries — see rotated_bc._boundary_velocity_nodes). Return
        # no local surface nodes; every consumer already handles the empty case.
        if not (facet_is and facet_is.getSize() > 0):
            return np.empty((0, self.mesh.cdim))
        v_start, v_end = dm.getDepthStratum(0)
        csec = dm.getCoordinateSection()
        cdim = self.mesh.cdim
        cvec = dm.getCoordinatesLocal().array.reshape(-1, cdim)

        vertices = set()
        for facet in facet_is.getIndices():
            closure = dm.getTransitiveClosure(facet)[0]
            vertices.update(p for p in closure if v_start <= p < v_end)
        rows = [csec.getOffset(v) // cdim for v in vertices]
        return np.ascontiguousarray(cvec[rows]) if rows else np.empty((0, cdim))

    def _field_rows(self, field, coords):
        """Local rows of ``field`` at ``coords``, matched by position (the proven
        ring-to-DOF pattern). Returns ``(rows, x)`` with ``x`` the along-surface
        coordinate used to order the ring."""
        if coords.shape[0] == 0:
            return np.empty(0, dtype=int), np.empty(0)
        tree = uw.kdtree.KDTree(np.ascontiguousarray(field.coords))
        rows = np.asarray(tree.query(coords, k=1)[1]).flatten()
        order = np.argsort(coords[:, 0])
        return rows[order], coords[order, 0]

    def _ring_weights(self):
        r"""P1 lumped trace-mass quadrature weights on the globally s-sorted surface
        ring: :math:`w_i = \oint \phi_i \,\mathrm{d}s`, accumulated per boundary FACET
        from the DMPlex (each facet contributes measure/nverts to each of its
        vertices — edge length/2 in 2D, triangle area/3 in 3D), from LIVE
        coordinates so the weights follow the deforming surface.

        The facet (chord) measure — not an arc approximation — is deliberate: the
        prescribed :math:`\tilde u_n` must be flux-free in the FINITE-ELEMENT sense
        (:math:`\oint` over the deformed polygon the discretisation actually
        integrates), or the strong datum asks the incompressible interior for a flow
        that does not exist. With trapezoid arc weights the consistent solve floored
        at rel ~2e-3 with the entire residual in the pressure (divergence) rows —
        measured, block-split; the trace-mass weights are exact for the P1 datum
        field by construction. Dimension-general: the same accumulation is the
        area-weighted gauge on a 3D boundary triangulation.
        """
        return self._ring_gather(self._surface_weights_local(), op="sum")

    def _surface_weights_local(self):
        """This rank's OWNED-facet partial trace-mass weights, aligned with the
        local ring order. Because every facet is counted exactly once globally,
        plain sums of (weight x value) over all ranks' local arrays are exact —
        no gather, no ordering, no seam bookkeeping — which is what
        :meth:`_surface_mean` reduces over, in any dimension."""
        from underworld3.utilities.boundary_flux import _boundary_stratum_is

        rc = self._ring_coords                         # local nodes, x-sorted order
        if rc.shape[0] == 0 and uw.mpi.size == 1:
            return np.empty(0)
        dm = self.mesh.dm
        cdim = self.mesh.cdim
        csec = dm.getCoordinateSection()
        cvec = dm.getCoordinatesLocal().array.reshape(-1, cdim)
        v0, v1 = dm.getDepthStratum(0)
        fS, fE = dm.getHeightStratum(1)
        # OWNED facets only: a ghost facet's contribution belongs to the owning
        # rank, and each facet must be counted exactly once globally — the seam
        # weight is then assembled by SUMMING the per-rank partial contributions
        # in the gather (op="sum"), not by averaging copies (a seam copy only
        # carries the facets its rank owns).
        ghosts = set()
        if uw.mpi.size > 1:
            nroots, ilocal, _ = dm.getPointSF().getGraph()
            if nroots > 0 and ilocal is not None:
                ghosts = set(int(i) for i in ilocal)
        acc = {}                                       # vertex coord-row -> weight
        sis = _boundary_stratum_is(dm, self.mesh, self.surface)
        if sis and sis.getSize() > 0:
            for f in sis.getIndices():
                if not (fS <= int(f) < fE) or int(f) in ghosts:
                    continue
                verts = [int(p) for p in dm.getTransitiveClosure(int(f))[0]
                         if v0 <= p < v1]
                crows = [csec.getOffset(v) // cdim for v in verts]
                xs = cvec[crows]
                if len(verts) == 2:                    # 2D: boundary edge
                    measure = float(np.linalg.norm(xs[1] - xs[0]))
                else:                                  # 3D: boundary triangle
                    measure = 0.5 * float(np.linalg.norm(
                        np.cross(xs[1] - xs[0], xs[2] - xs[0])))
                wv = measure / len(verts)
                for cr in crows:
                    acc[cr] = acc.get(cr, 0.0) + wv
        # align to the local ring order by position (both sides read the SAME live
        # plex coordinates, so the rounded keys match exactly)
        wmap = {tuple(np.round(cvec[cr], 9)): w for cr, w in acc.items()}
        w = np.array([wmap.get(tuple(np.round(c, 9)), 0.0) for c in rc])
        # Every owned-facet vertex is a local surface node, so the key matching
        # must conserve the accumulated trace mass EXACTLY — a mismatch between
        # mesh.X.coords and getCoordinatesLocal() would otherwise drop weights
        # silently (0.0 fallback) and corrupt the datum gauge invisibly.
        acc_total = float(sum(acc.values()))
        if not np.isclose(w.sum(), acc_total, rtol=1e-12, atol=1e-300):
            raise RuntimeError(
                f"surface trace-mass weights lost {acc_total - w.sum():.3e} of "
                f"{acc_total:.3e} in coordinate-key matching — the ring coords "
                "and the plex coordinates have diverged (gauge would be corrupt)."
            )
        return w

    def _surface_mean(self, values):
        r"""Area-weighted global mean of a surface-node array, identical on every rank
        (a datum gauge must be single-valued across the partition).

        Weighted by the FE trace mass (boundary length in 2D, area in 3D), NOT by node count. The distinction is not cosmetic for
        the one place it matters most: :math:`\tilde u_n`, the normal velocity the
        consistent solve is asked to reproduce, must satisfy :math:`\oint \tilde u_n
        \,\mathrm{d}s = 0` — an incompressible interior over a closed base can neither
        gain nor lose volume. Surface nodes are not equally spaced (and less so as the
        mesh deforms), so removing the *nodal* mean leaves a net flux — measured at
        0.4-1.0% of the total here. A penalty absorbs that residue quietly; a STRONG
        rotated datum cannot, because it is then asking for a flow that does not exist.
        The same weighting makes the ``h`` / ``h_inf`` datum volume-preserving rather
        than node-count-preserving.
        """
        # Reduction over OWNED partial weights: each facet's contribution is
        # counted exactly once globally, so two scalar allreduces give the exact
        # weighted mean with no gather and no ordering — dimension-general (the
        # ordered ring remains only for the 2D-only filter and transport).
        w = self._surface_weights_local()
        v = np.asarray(values, dtype=float)
        local_wv = float(np.dot(w, v)) if w.size else 0.0
        local_w = float(w.sum()) if w.size else 0.0
        comm = uw.mpi.comm
        total_wv = comm.allreduce(local_wv, op=MPI.SUM)
        total_w = comm.allreduce(local_w, op=MPI.SUM)
        return total_wv / total_w if total_w else 0.0

    def _demean(self, values):
        """Remove the global surface mean (topography datum floats)."""
        return values - self._surface_mean(values)

    # -- derived solves -------------------------------------------------------

    def _new_stokes(self, name):
        """A Stokes solve sharing the free solve's mesh, discretisation, physics and
        options — the common construction for the held and consistent lids."""
        v = uw.discretisation.MeshVariable(
            f"V_{name}", self.mesh, vtype=uw.VarType.VECTOR, degree=2, continuous=True
        )
        p = uw.discretisation.MeshVariable(
            f"P_{name}", self.mesh, vtype=uw.VarType.SCALAR, degree=1,
            continuous=getattr(self.free, "_p_continuous", True),
        )
        solver = uw.systems.Stokes(self.mesh, velocityField=v, pressureField=p)
        self._copy_constitutive_model(solver)
        solver.penalty = self.free.penalty
        solver.tolerance = self.free.tolerance
        # Carry the strategic velocity-block choice (e.g. FMG on a refined mesh); the
        # fresh Stokes already holds sensible default PETSc options.
        solver.preconditioner = self.free.preconditioner
        # Mirror the free solve's nonlinear configuration: the derived solves share its
        # rheology, so a nonlinear free solve (Newton/Picard tangent, full SNES) needs
        # nonlinear derived solves too — not a single ksponly linearisation. A linear
        # free solve keeps its ksponly one-shot. (The rotated held solve additionally
        # auto-detects nonlinearity and dispatches its own linear/Newton path.)
        solver.consistent_jacobian = self.free.consistent_jacobian
        solver.petsc_options["snes_type"] = self.free.petsc_options.getString(
            "snes_type", "newtonls"
        )
        return solver

    def _copy_constitutive_model(self, solver):
        """Give ``solver`` its own constitutive model of the free solve's class with the
        same parameters, retargeted onto ``solver``'s unknowns.

        A model instance binds to one solver's unknowns and cannot be shared, so a
        fresh instance is made and each parameter expression is copied with the free
        solve's velocity/gradient atoms rebound onto ``solver``'s own (see
        :meth:`_velocity_rebind_map`). The atoms are hidden inside a wrapped
        ``UWexpression``, so the value is unwrapped before substitution; for a linear or
        field-dependent viscosity the substitution is a no-op.
        """
        from underworld3.utilities._api_tools import ExpressionDescriptor
        from underworld3.function.expressions import unwrap

        free_model = self.free.constitutive_model
        solver.constitutive_model = type(free_model)
        rebind = self._velocity_rebind_map(solver)
        copied = set()
        for cls in type(free_model.Parameters).__mro__:
            for name, attr in cls.__dict__.items():
                if isinstance(attr, ExpressionDescriptor) and name not in copied:
                    value = getattr(free_model.Parameters, name)
                    if hasattr(value, "subs"):
                        value = unwrap(value, keep_constants=True, return_self=True).subs(rebind)
                    setattr(solver.constitutive_model.Parameters, name, value)
                    copied.add(name)

    def _velocity_rebind_map(self, solver):
        """Substitution from the free solve's velocity atoms to ``solver``'s own.

        A constitutive expression built from strain rate carries the *source* solver's
        velocity (``u``) and velocity-gradient (``Unknowns.L``) symbols — each tagged
        with that solver's variable identity. Copying the expression to another solver
        WITHOUT this substitution silently binds it to the source solution: the derived
        held/consistent solve would compute its viscosity from the free velocity rather
        than its own. (Any solver-to-solver retarget of a nonlinear constitutive model
        hits this — an adjoint operator sharing the forward rheology would too. See
        ``docs/developer/subsystems/constitutive-models.md``, "Retargeting a model".)

        Velocity and its first gradient cover strain-rate rheology; a pressure- or
        higher-gradient-dependent law would extend this map.
        """
        dim = self.mesh.dim
        rebind = {self.free.u.sym[i]: solver.u.sym[i] for i in range(dim)}
        source_L, target_L = self.free.Unknowns.L, solver.Unknowns.L
        for i in range(dim):
            for j in range(dim):
                rebind[source_L[i, j]] = target_L[i, j]
        return rebind

    def _build_held(self, driving_buoyancy):
        r"""The held free-slip lid: rotated ``u.n = 0`` on every wall and the
        surface, driving body force only. Its constraint reaction is
        :math:`\sigma_{nn}`, handed to ``dynamic_topography`` as :math:`h_\infty`."""
        self.held = self._new_stokes("held")
        self.held.bodyforce = (
            self.free.bodyforce if driving_buoyancy is None else driving_buoyancy
        )
        self._apply_walls(self.held)
        self.held.add_rotated_freeslip_bc(0.0, self.surface, normal=self.normal)
        self.held.petsc_use_pressure_nullspace = True
        if self.background_buoyancy is not None:
            # Twin of the held lid with the BACKGROUND body force only — same walls,
            # same rotated surface, same geometry — whose recovered reaction carries
            # the identical (defective) current-shape-load leg. See solve().
            self.held_bg = self._new_stokes("heldbg")
            self.held_bg.bodyforce = self.background_buoyancy
            self._apply_walls(self.held_bg)
            self.held_bg.add_rotated_freeslip_bc(0.0, self.surface, normal=self.normal)
            self.held_bg.petsc_use_pressure_nullspace = True
            self._hinf_bg_field = uw.discretisation.MeshVariable(
                "h_inf_bg", self.mesh, vtype=uw.VarType.SCALAR, degree=1, continuous=True
            )
            self._hinf_bg_rows, _ = self._field_rows(self._hinf_bg_field, self._surf_coords)

    def _build_consistent(self):
        r"""The consistent solve: walls as the free solve, plus a constraint prescribing
        :math:`\mathbf{u}\cdot\hat{\mathbf n}=\tilde u_n` (the realised relaxed rate) so
        the advection velocity keeps the surface a material boundary.

        ``consistent_constraint`` selects how that datum is imposed:

        ``"strong"``
            A rotated per-node constraint — the same primitive as the held lid, differing
            only in the constraint right-hand side, so free-slip is the ``\tilde u_n = 0``
            member of the same family. It enforces the datum to machine precision (no
            penalty leak) through the unified rotated Newton loop: an isoviscous flow
            converges in one increment (the cold-start affine lift IS the linear solve),
            and a nonlinear rheology iterates with the datum held exactly at every
            accepted iterate. It requires the datum to be discretely flux-free, which
            :meth:`_surface_mean` guarantees by weighting the demean by the FE trace mass.

        ``"penalty"``
            A weak natural BC. It tolerates a datum that carries a small net flux, at the
            cost of a leak set by the penalty magnitude.

        The datum is the ``ũ_n`` field, re-read at each solve, so refreshing
        ``_un_target`` refreshes the constraint (see #403)."""
        self.consistent = self._new_stokes("cons")
        self.consistent.bodyforce = self.free.bodyforce
        self._apply_walls(self.consistent)
        self._un_target = uw.discretisation.MeshVariable(
            "un_tgt", self.mesh, vtype=uw.VarType.SCALAR, degree=1, continuous=True
        )
        self._un_target_rows, _ = self._field_rows(self._un_target, self._surf_coords)
        self._un_target.array[...] = 0.0
        if self.consistent_constraint == "strong":
            # The datum is read along the rotated per-node normal — the same deform
            # direction the rate was measured along, so no spurious tangential-slope term
            # on a bumpy or rotating surface. Value-first: the ũ_n field read IS the
            # conds datum, re-evaluated at the boundary nodes at each solve.
            self.consistent.add_rotated_freeslip_bc(
                self._un_target.sym[0], self.surface, normal=self.normal)
            # Flux-consistent datum gauge: the divergence rows enforce
            # ∮ u·n̂_facet over the DEFORMED faceted surface, while the constraint
            # fixes u·n̂_node — the directions differ on a deformed surface, so a
            # nodal demean of the datum leaves a small net volume flux that an
            # incompressible interior cannot absorb (measured: rel ~2e-3 residual
            # floor sitting 100% in the pressure rows). Strip the datum's DIRECTED
            # mean using the same FE surface integral the residual uses:
            # Φ = ∮ ũ_n (n̂·Γ̂) ds and S = ∮ (n̂·Γ̂) ds with Γ̂ = mesh.Gamma (the
            # facet normal at quadrature points), then shift ũ_n by Φ/S so the
            # discrete net flux of the constrained field is exactly zero.
            nrm = (self.normal if self.normal is not None
                   else self.mesh.boundary_normal(self.surface))
            ncomps = sympy.flatten(sympy.Matrix(nrm))
            ndotg = sum(ncomps[k] * self.mesh.Gamma[k] for k in range(self.mesh.dim))
            self._datum_flux = uw.maths.BdIntegral(
                mesh=self.mesh, fn=self._un_target.sym[0] * ndotg,
                boundary=self.surface)
            self._datum_flux_scale = uw.maths.BdIntegral(
                mesh=self.mesh, fn=ndotg, boundary=self.surface)
        else:
            n_hat = (self.normal if self.normal is not None
                     else self.mesh.boundary_normal(self.surface))
            self.consistent.add_natural_bc(
                self._consistent_penalty
                * (n_hat.dot(self.consistent.u.sym) - self._un_target.sym[0]) * n_hat,
                self.surface,
            )
            self.consistent.petsc_options["snes_type"] = "ksponly"  # linear: no line search
        self.consistent.petsc_use_pressure_nullspace = True
        self._adv_velocity = self.consistent.u

    def _build_surface_velocity_projection(self):
        """P1 projection of the free-solve velocity — the surface node motion is
        driven from the P1-smoothed velocity, not a point-eval of the P2 field, so it
        is consistent with the P1 mesh geometry."""
        self._v_p1 = uw.discretisation.MeshVariable(
            "V_p1", self.mesh, vtype=uw.VarType.VECTOR, degree=1, continuous=True
        )
        self._v_p1_proj = uw.systems.Vector_Projection(self.mesh, self._v_p1)
        self._v_p1_proj.uw_function = self.free.u.sym
        self._v_p1_proj.smoothing_length = self._smooth_length
        # surface rows of the P1 velocity (ring order) for the direct nodal reads
        self._v_p1_rows, _ = self._field_rows(self._v_p1, self._surf_coords)

    def _build_interior_diffuser(self):
        """Laplacian carrier: a scalar Poisson solve whose surface Dirichlet value is
        the nodal surface increment and which decays it to zero at the base, giving a
        smooth minimal interior deformation (no full remesh)."""
        self._carry = uw.discretisation.MeshVariable(
            "carry", self.mesh, vtype=uw.VarType.SCALAR, degree=1, continuous=True
        )
        self._carry_bc = uw.discretisation.MeshVariable(
            "carry_bc", self.mesh, vtype=uw.VarType.SCALAR, degree=1, continuous=True
        )
        self._carry_bc_rows, _ = self._field_rows(self._carry_bc, self._surf_coords)
        self._carry_bc.array[...] = 0.0
        self._diffuser = uw.systems.Poisson(self.mesh, self._carry)
        self._diffuser.constitutive_model = uw.constitutive_models.DiffusionModel
        self._diffuser.constitutive_model.Parameters.diffusivity = 1.0
        self._diffuser.tolerance = 1.0e-3
        # Row map for the deform read: the carrier is P1 and (on our P1-geometry
        # meshes) its nodes coincide with the mesh coordinate nodes, so the
        # displacement is a DIRECT nodal read — never a point evaluation at the
        # field's own nodes (the on-node location class, O(1)-wrong on 3D cell
        # edges, #432). Row identities survive deformation; matched once here.
        tree = uw.kdtree.KDTree(np.ascontiguousarray(self._carry.coords))
        dist, rows = tree.query(np.ascontiguousarray(self.mesh.X.coords), k=1)
        self._carry_rows_at_mesh_nodes = (
            np.asarray(rows).flatten()
            if float(np.max(dist)) < 1.0e-12 else None)   # exotic geometry: fall back
        self._base = self._opposite_boundary()
        self._diffuser.add_essential_bc(self._carry_bc.sym, self.surface)
        if self._base is not None:
            self._diffuser.add_essential_bc(sympy.Matrix([0.0]), self._base)

    def _opposite_boundary(self):
        """The no-slip / far boundary that anchors the interior carrier at zero —
        the mesh's Lower/Bottom if present, else any wall that is not the surface."""
        names = [b.name for b in self.mesh.boundaries]
        for candidate in ("Lower", "Bottom"):
            if candidate in names:
                return candidate
        others = [n for n in self._walls if n != self.surface]
        return others[0] if others else None

    def _build_composition_transport(self):
        """Semi-Lagrangian transport of the material field by the consistent surface
        velocity. Diffusion is negligible; a level-set distance field carries no
        monotone clamp (it is not a bounded [0,1] field)."""
        self._comp_ddt = uw.systems.ddt.SemiLagrangian(
            self.mesh, self.composition.sym, self._adv_velocity.sym,
            vtype=uw.VarType.SCALAR, degree=self.composition.degree, continuous=True,
            varsymbol="phi", bcs=[], order=1, smoothing=0.0,
            # old_frame_traceback must be FALSE on a per-step-deforming mesh
            # (underworldcode/underworld3#423). The old-frame reach-back — introduced as
            # the fix for the earlier high-Ra blow-up — is itself an exponential
            # amplifier once the surface deformation squeezes the near-boundary cells
            # (onset ~5% of radius): the record→trace→solve loop then grows both T
            # extremes ~10% per CYCLE (worse at smaller dt), mesh-locked, until T is
            # unbounded. A minimal reproducer with no free surface at all (prescribed
            # velocity + a ±0.1%/step mesh wobble) shows the same runaway with old-frame
            # ON at any theta and is bounded with it OFF. The standard ALE path is safe
            # here because the hazards that motivated old-frame are covered by fixes
            # landed since: departure feet are restored by the deform-aware
            # return_coords_to_bounds, and the monotone clamp bounds the sample.
            # Measured on this problem (rho_g 2e5): old-frame ON is unusable beyond
            # ~5.4% deformation; OFF holds T in [0,1] to 1e-3 through 17% deformation.
            monotone_mode="clamp", theta=0.5, old_frame_traceback=False,
        )
        self._comp_adv = uw.systems.AdvDiffusionSLCN(
            self.mesh, u_Field=self.composition, V_fn=self._adv_velocity.sym,
            order=1, DuDt=self._comp_ddt,
        )
        self._comp_adv.constitutive_model = uw.constitutive_models.DiffusionModel
        self._comp_adv.constitutive_model.Parameters.diffusivity = 1.0e-7
        self._comp_adv.tolerance = 1.0e-4
        if self._conserve_integrand is not None:
            self._conserve_area = uw.maths.Integral(self.mesh, self._conserve_integrand)
            shift_rate = sympy.diff(self._conserve_integrand, self.composition.sym[0])
            self._conserve_rate = uw.maths.Integral(self.mesh, shift_rate)
            self._conserve_target = float(self._conserve_area.evaluate())

    def _build_ring_gather(self):
        """Map the local (x-sorted) surface arrays to/from a GLOBALLY sorted ring — by
        the along-surface coordinate ``s`` (``theta`` on a cylindrical annulus, ``x`` on
        a Cartesian box) — present on every rank. Both the surface filter and the
        tangential transport run on this ring, so a launch point / smoothing stencil that
        crosses a partition boundary is handled, and the x-extreme degeneracy (an x-sorted
        ring collapses the many nodes near ``theta=0`` / ``theta=pi``, where ``dx/dtheta ->
        0``) is avoided. The surface deforms only along the normal, so ``s`` and its order
        are invariant — built once."""
        comm = uw.mpi.comm
        rc = self._ring_coords                         # local nodes, x-sorted order
        if self._cylindrical:
            keys_local = np.arctan2(rc[:, 1], rc[:, 0])[:, None]
            self._ring_period = 2.0 * np.pi
        elif self.mesh.dim == 3:
            # No natural 1-D surface ordering in 3D. The ordered-ring FEATURES
            # (Taubin filter, tangential transport) are 2D-only and refused at
            # construction; the gather itself only needs a deterministic global
            # order with exact same-node grouping, which lexicographic rounded
            # coordinates provide (the surface deforms along the normal, so the
            # reference ordering is built once, like the 2D ring).
            keys_local = np.round(np.asarray(rc, dtype=float), 12)
            self._ring_period = None
        else:
            keys_local = rc[:, 0].astype(float)[:, None]
            self._ring_period = None
        self._s_local_n = int(keys_local.shape[0])
        counts = comm.allgather(self._s_local_n)
        self._ring_offset = int(np.sum(counts[: comm.rank]))
        keys_global = (np.concatenate(comm.allgather(np.ascontiguousarray(keys_local)))
                       if comm.size > 1 else keys_local.copy())
        keys_global = keys_global.reshape(self._s_local_n if comm.size == 1
                                          else -1, keys_local.shape[1])
        # lexicographic stable order over the key columns (a single column in 2D)
        self._ring_order = np.lexsort(keys_global.T[::-1])
        self._ring_inv = np.empty_like(self._ring_order)
        self._ring_inv[self._ring_order] = np.arange(self._ring_order.size)
        s_global = keys_global[:, 0] if keys_global.shape[1] == 1 else None
        # DEDUPLICATE partition-seam copies (#421). A vertex on a partition cut appears
        # once per adjacent rank in the gathered ring. Every ring operation must see each
        # PHYSICAL node exactly once: the Taubin filter's roll-stencil otherwise treats
        # the two copies as distinct neighbouring nodes (a phantom zero-length segment),
        # smooths them against DIFFERENT stencils, and hands each rank back a different
        # value — measured as a ~50% seam disagreement in h_inf after 20 iterations and
        # a few-percent net flux in the prescribed datum. Gather averages the copies
        # (identical up to round-off); scatter expands back to every copy.
        keys_srt = np.round(keys_global[self._ring_order], 12)
        if keys_srt.shape[0] == 0:
            new_group = np.empty(0, dtype=bool)
        else:
            new_group = np.r_[True, np.any(np.diff(keys_srt, axis=0) != 0.0, axis=1)]
        uniq_of_sorted = np.cumsum(new_group) - 1 if keys_srt.shape[0] else \
            np.empty(0, dtype=int)
        self._ring_uniq_of_sorted = uniq_of_sorted
        self._ring_n_uniq = int(uniq_of_sorted[-1] + 1) if keys_srt.shape[0] else 0
        self._ring_dup_count = np.bincount(uniq_of_sorted, minlength=self._ring_n_uniq)
        first_pos = np.searchsorted(uniq_of_sorted, np.arange(self._ring_n_uniq))
        # the 1-D along-surface coordinate exists only where the ordering is real
        # (2D); the 3D gather is order-agnostic and the ring features that read
        # _s_sorted are refused at construction there.
        self._s_sorted = (s_global[self._ring_order][first_pos]
                          if s_global is not None else None)

    def _ring_gather(self, local_vals, op="mean"):
        """Local (x-sorted-order) surface values -> the globally s-sorted UNIQUE ring.
        ``op="mean"`` averages partition-seam copies (field values: the copies agree
        to round-off by construction); ``op="sum"`` accumulates them (per-rank
        PARTIAL contributions such as owned-facet quadrature weights, where the
        copies deliberately each carry only their rank's share)."""
        comm = uw.mpi.comm
        v = (np.concatenate(comm.allgather(np.ascontiguousarray(local_vals)))
             if comm.size > 1 else np.asarray(local_vals, dtype=float))
        v_sorted = v[self._ring_order]
        sums = np.bincount(self._ring_uniq_of_sorted, weights=v_sorted,
                           minlength=self._ring_n_uniq)
        return sums if op == "sum" else sums / self._ring_dup_count

    def _ring_scatter(self, v_uniq):
        """Unique-ring values -> this rank's local (x-sorted-order) nodes. Every seam
        copy receives the SAME unique value, so ring fields are single-valued across
        ranks by construction."""
        v_sorted = v_uniq[self._ring_uniq_of_sorted]   # expand to seam copies
        v = v_sorted[self._ring_inv]                   # back to rank-concatenated order
        return v[self._ring_offset: self._ring_offset + self._s_local_n]

    def _ring_taubin(self, v_sorted, n_iters, lam=0.33, mu=-0.34):
        """1-D Taubin low-pass on the s-sorted ring: periodic (closed annulus) or fixed
        ends (open Cartesian surface). ``lam``/``mu`` are the standard shrink/unshrink
        pair (near volume-neutral, no long-wavelength bias)."""
        periodic = self._ring_period is not None
        v = np.asarray(v_sorted, dtype=float).copy()
        for _ in range(int(n_iters)):
            for step in (lam, mu):
                if periodic:
                    lap = 0.5 * (np.roll(v, 1) + np.roll(v, -1)) - v
                else:
                    lap = np.zeros_like(v)
                    lap[1:-1] = 0.5 * (v[:-2] + v[2:]) - v[1:-1]
                v = v + step * lap
        return v

    def _filter_surface(self, values):
        """Periodic Taubin low-pass on the globally s-sorted surface ring — parallel-
        correct, and free of the submesh-mapping collision at the along-surface-coordinate
        extremes (the theta=0 / theta=pi seam)."""
        return self._ring_scatter(
            self._ring_taubin(self._ring_gather(values), self._filter_iters))


    # -- public step ----------------------------------------------------------

    def solve(self):
        r"""Solve the free and held lids and recover the equilibrium topography.

        The free solve gives the surface velocity (the relaxation rate); the held
        rotated free-slip solve gives :math:`\sigma_{nn}` and hence :math:`h_\infty`.
        Call once per step before :meth:`estimate_dt` / :meth:`advance`.
        """
        self.free.solve(zero_init_guess=True)
        self.held.solve(zero_init_guess=True)
        self.held.dynamic_topography(
            self.surface, self._hinf_field, buoyancy_scale=self.buoyancy_scale,
            mass=self._mass,
        )
        h = np.asarray(self._hinf_field.array[self._hinf_rows, 0, 0], dtype=float)
        if isinstance(self.background_buoyancy, str) and self.background_buoyancy == "analytic":
            # SINGLE-SOLVE full-density h_inf. The recovered reaction under full density
            # is (self-load + driving) and the recovery is essentially exact (probes:
            # load leg 0.976, driving leg 0.972, exact superposition). The self-load
            # component of the returned field equals +h_current, and the reduced-form
            # negation below would flip it (h_inf = -h + drive: fixed point at HALF the
            # equilibrium, period-2 ringing). Subtracting the ANALYTIC current height --
            # read from the geometry, no extra solve -- cancels the self-load so the
            # negation applies to the driving part alone, as validated on the reduced
            # formulation. Residual error ~2.4% of h (the recovery's own accuracy).
            h = h - self._current_shape()
        elif self.background_buoyancy is not None:
            # TWO-REACTION h_inf (full-density formulation). On a deformed boundary the
            # recovered reaction mixes the CURRENT-SHAPE load rho_0*g*h with the driving
            # support sigma', and the discrete recovery of the load leg is defective
            # (measured ~0.16 of its continuum value at 5% deformation) while the
            # driving leg is exact. Running the SAME held recovery with the background
            # body force alone, on the same geometry, reproduces the identical defective
            # load leg — so the DIFFERENCE isolates sigma' exactly, and
            # h_inf = sigma'/(rho_0 g) is the true equilibrium topography.
            self.held_bg.solve(zero_init_guess=True)
            self.held_bg.dynamic_topography(
                self.surface, self._hinf_bg_field, buoyancy_scale=self.buoyancy_scale,
                mass=self._mass,
            )
            h_bg = np.asarray(
                self._hinf_bg_field.array[self._hinf_bg_rows, 0, 0], dtype=float)
            h = h - h_bg
        # dynamic_topography returns a depression over a rising load; negate for the
        # physical uplift and float the datum.
        self._h_inf = self._demean(-self._demean(h))
        if self._filter_iters:
            self._h_inf = self._demean(self._filter_surface(self._h_inf))

    def estimate_dt(self, advect_scale=1.0):
        """A step size that respects both advection and surface-motion limits.

        The advective limit comes from the material transport (or a velocity CFL if
        there is no composition); the surface limit keeps the per-step displacement
        below ``max_surface_cfl`` cell sizes so the mesh stays well-shaped.

        ``advect_scale`` multiplies the ADVECTIVE limit only: the semi-Lagrangian
        transport is unconditionally stable, so a value > 1 takes larger steps (fewer
        steps to traverse a slow spin-up transient, where a tiny advective step would
        otherwise keep the surface near-flat and the solves ill-conditioned) — while the
        surface-motion cap still binds, so the mesh cannot be over-deformed and the
        surface-flow feedback cannot run away. Scale the advective step, never the
        surface safety."""
        if self.composition is not None:
            dt_advect = float(self._comp_adv.estimate_dt())
        else:
            dt_advect = self._velocity_cfl()
        dt_advect *= float(advect_scale)
        u_n = self._surface_normal_velocity()
        speed = uw.mpi.comm.allreduce(float(np.abs(u_n).max()) if u_n.size else 0.0,
                                      op=MPI.MAX)
        h_cell = self.mesh.get_min_radius()
        dt_surface = np.inf if speed == 0.0 else self.max_surface_cfl * h_cell / speed
        return float(min(dt_advect, dt_surface))

    def _velocity_cfl(self):
        """Courant limit from the free-solve velocity magnitude (used when there is
        no material field to set the advective step).

        Reads the nodal DOF data directly rather than ``evaluate``-ing at the mesh
        vertices: a max-speed bound needs no interpolation, and on-vertex point
        location on a deformed mesh can legitimately fail and return NaN (the loud
        fallback policy), which would poison the timestep."""
        v = np.asarray(self.free.u.data)
        local_max = float(np.linalg.norm(v, axis=1).max()) if v.size else 0.0
        vmax = uw.mpi.comm.allreduce(local_max, op=MPI.MAX)
        h_cell = self.mesh.get_min_radius()
        return np.inf if vmax == 0.0 else self.max_surface_cfl * h_cell / vmax

    def advance(self, dt):
        r"""Advance the surface (and any material field) by ``dt``.

        The exponential update sets the new surface height; the consistent solve
        produces an advection velocity whose surface-normal equals the realised rate;
        the material field is transported and (optionally) volume-corrected; and the
        surface increment is carried inward and applied with :meth:`Mesh.deform`.
        """
        shape0 = self._current_shape()   # the mesh geometry the increment is applied to
        shape = shape0
        u_n = self._surface_normal_velocity()
        h_inf = self._h_inf

        if self.tangent_advect is not None:
            shape, h_inf, u_n = self._apply_tangential_transport(shape, h_inf, u_n, dt)

        displacement = h_inf - shape
        with np.errstate(divide="ignore", invalid="ignore"):
            gamma = np.where(np.abs(displacement) > 1.0e-9, u_n / displacement, 0.0)
        gamma = np.abs(gamma)  # relax toward equilibrium; L-stable for any rate
        shape_new = h_inf + (shape - h_inf) * np.exp(-gamma * dt)
        # Total height change relative to the CURRENT mesh: the tangential transport
        # (shape - shape0) plus the normal relaxation (shape_new - shape).
        increment = shape_new - shape0

        if self.composition is not None:
            self._solve_consistent(increment, dt)
            self._comp_adv.solve(timestep=dt, zero_init_guess=False)
            self._conserve_composition()

        self._carry_and_deform(increment, dt)
        if self.verbose:
            uw.pprint(
                f"FreeSurface: |increment|={np.abs(increment).max():.3e} "
                f"h_max={np.abs(shape_new).max():.3e}"
            )

    # -- step internals -------------------------------------------------------

    def _current_shape(self):
        """The surface height anomaly (topography direction), mean-removed — read from
        the deformed mesh geometry, which IS the surface state."""
        height = self._surface_height(self.mesh.X.coords[self._surf_rows])
        return self._demean(height)

    def _surface_normal_velocity(self):
        r""":math:`\dot h = \mathbf{u}\cdot\hat{\mathbf n}` on the surface nodes, from
        the P1-projected free-solve velocity, mean-removed.

        The normal here is the **deform direction** (:meth:`_normal_direction`: radial on
        an annulus, vertical in Cartesian), *not* the FE facet normal of the (bumpy)
        deformed surface. This is the correct operator split: :math:`\dot h` measures the
        height change along the direction the surface node is advected, while all
        along-surface motion is carried by the tangential transport. Using the tilted
        facet normal instead lets a large tangential flow (e.g. a co-rotating base,
        :math:`\mathbf{u}=\Omega\times\mathbf{r}`) project into :math:`\dot h` through the
        bump slope — an implicit, unstable explicit-Euler advection that double-counts the
        transport and grows the surface out of round-off. With the deform-direction normal
        a rigid rotation gives :math:`\dot h=0` exactly."""
        self._v_p1_proj.solve()
        coords = self._ring_coords                  # live, already in ring order
        n = self._normal_direction(coords)          # unit vectors, same as the deform
        # DIRECT nodal read of the P1 velocity at the surface rows — never a point
        # evaluation at the field's own nodes: on-vertex point location mis-locates at
        # partition seams (the documented ddt band-aid class), and a seam node whose
        # velocity differs across ranks breaks the global ring's single-valuedness —
        # measured as a few-percent net flux in the prescribed datum at np2 (#421).
        # The projection solve has already ghost-synced .data, so ranks agree exactly.
        v_nodes = np.asarray(self._v_p1.data)[self._v_p1_rows]
        u_n = (v_nodes * n).sum(axis=1)
        u_n = self._demean(u_n)
        if self._surface_mask is not None:
            u_n = u_n * self._surface_mask  # pin the surface rate near driven walls
        return u_n

    def _tangential_velocity(self):
        """Along-surface (tangential) velocity per LOCAL ring node, in the local
        (x-sorted) order the surface arrays use, in ``s``-units: Cartesian ``v_x`` (so
        ``s_dep = x - v_x*dt``); cylindrical the angular rate ``omega = v_theta/r`` (so
        ``s_dep = theta - omega*dt``)."""
        coords = self._ring_coords                  # live, already in ring order
        # direct nodal read, same seam rationale as _surface_normal_velocity (#421)
        v_nodes = np.asarray(self._v_p1.data)[self._v_p1_rows]
        v_x = v_nodes[:, 0]
        if not self._cylindrical:
            return v_x
        v_y = v_nodes[:, 1]
        cx, cy = coords[:, 0], coords[:, 1]
        r = np.hypot(cx, cy)
        v_theta = (-cy * v_x + cx * v_y) / r        # v . theta-hat
        return v_theta / r

    def _launch_interp(self, s, field, s_dep, period):
        """Interpolate ``field(s)`` at the launch point ``s_dep``. Open surface: linear
        (bounded), clamped ends. Closed ring: linear periodic, or — when
        ``tangent_spectral_modes>0`` — a diffusion-free spectral fit (:meth:`_fourier_interp`)."""
        order = np.argsort(s)
        s_sorted, f_sorted = s[order], field[order]
        if period is None:
            return np.interp(np.clip(s_dep, s_sorted[0], s_sorted[-1]), s_sorted, f_sorted)
        if self._spectral_modes > 0:
            return self._fourier_interp(s, field, s_dep, period, self._spectral_modes)
        lo = s_sorted[0]
        s_dep_wrapped = lo + np.mod(s_dep - lo, period)
        s_ext = np.concatenate([s_sorted, [s_sorted[0] + period]])
        f_ext = np.concatenate([f_sorted, [f_sorted[0]]])
        return np.interp(s_dep_wrapped, s_ext, f_ext)

    def _fourier_interp(self, s, field, s_dep, period, n_modes):
        """Diffusion-free interpolation on a periodic ring: a least-squares Fourier fit
        of ``field(s)`` (``n_modes`` harmonics), evaluated at the launch angles ``s_dep``.
        Exact for the resolved smooth content; the mode truncation low-passes the
        facet-scale (node) noise, so it also serves as the surface filter on the ring."""
        w = 2.0 * np.pi / period

        def design(angles):
            cols = [np.ones_like(angles)]
            for k in range(1, n_modes + 1):
                cols.append(np.cos(k * w * angles))
                cols.append(np.sin(k * w * angles))
            return np.column_stack(cols)

        coef, *_ = np.linalg.lstsq(design(s), field, rcond=None)
        return design(s_dep) @ coef

    def _apply_tangential_transport(self, shape, h_inf, u_n, dt):
        """Along-surface semi-Lagrangian transport on the GLOBALLY s-sorted ring:
        barycentric-interpolate the surface fields at the tangential launch point
        ``s - v_t*dt``. Parallel-correct — the launch point may lie on another rank's arc,
        so the fields and the tangential velocity are gathered to the full ring, the
        interpolation runs globally, and the result is scattered back to local nodes. The
        1-D interpolation is a convex combination of bracketing nodal values, so it is
        bounded (cannot overshoot); the annulus wraps periodically, the box clamps ends.
        """
        s = self._s_sorted                               # global s-sorted ring
        period = self._ring_period
        v_t = self._ring_gather(self._tangential_velocity())
        s_dep = s - v_t * dt
        shape_g, hinf_g, un_g = (self._ring_gather(shape), self._ring_gather(h_inf),
                                 self._ring_gather(u_n))
        shape_up = self._ring_scatter(self._launch_interp(s, shape_g, s_dep, period))
        h_inf_up = self._ring_scatter(self._launch_interp(s, hinf_g, s_dep, period))
        u_n_up = self._ring_scatter(self._launch_interp(s, un_g, s_dep, period))
        if self.verbose:
            comm = uw.mpi.comm
            h_scale = comm.allreduce(float(np.abs(shape).max() if shape.size else 0.0), op=MPI.MAX) + 1e-300
            u_scale = comm.allreduce(float(np.abs(u_n).max() if u_n.size else 0.0), op=MPI.MAX) + 1e-300
            dshape = comm.allreduce(float(np.abs(shape_up - shape).max() if shape.size else 0.0), op=MPI.MAX)
            dhinf = comm.allreduce(float(np.abs(h_inf_up - h_inf).max() if shape.size else 0.0), op=MPI.MAX)
            dun = comm.allreduce(float(np.abs(u_n_up - u_n).max() if u_n.size else 0.0), op=MPI.MAX)
            uw.pprint("FreeSurface tangential correction: "
                      f"shape {dshape / h_scale:.2e}, h_inf {dhinf / h_scale:.2e}, "
                      f"u_n {dun / u_scale:.2e}  (relative)")
        if self.tangent_advect == "shape":
            return shape_up, h_inf, u_n
        if self.tangent_advect == "fields":
            return shape, h_inf_up, u_n_up
        return shape, h_inf, u_n

    def _solve_consistent(self, increment, dt):
        r"""Prescribe :math:`\tilde u_n = \Delta h/\Delta t` (mean-removed by the FE trace mass,
        so the net flux is zero) on the surface and solve for the material-consistent
        velocity."""
        u_tilde = self._demean(increment / dt)
        self._un_target.array[...] = 0.0
        self._un_target.array[self._un_target_rows, 0, 0] = u_tilde
        # Exact FE-consistent flux strip (STRONG constraint only — the penalty
        # absorbs a datum flux weakly and builds no integrals): remove the
        # DIRECTED mean so the datum carries zero discrete net flux through the
        # deformed facets — the quantity the pressure rows actually enforce.
        # Collective (BdIntegral), so the shift is identical on every rank.
        if self.consistent_constraint == "strong":
            flux = float(self._datum_flux.evaluate())
            scale = float(self._datum_flux_scale.evaluate())
            if abs(scale) > 1.0e-30:
                self._un_target.array[self._un_target_rows, 0, 0] -= flux / scale
        # Warm-start from the free solve: the consistent solution IS the free
        # solution with the (small) material-boundary datum imposed, and the free
        # solve has already converged this step. Starting there keeps a power-law
        # tangent at physical strain rates — a cold start puts it at the
        # regularisation floor, where the Newton line search stalls at O(0.1)
        # relative residual (measured, power-law annulus acceptance run).
        self.consistent.u.array[...] = self.free.u.array
        self.consistent.p.array[...] = self.free.p.array
        self.consistent.solve(zero_init_guess=False)

    def _conserve_composition(self):
        r"""Hold :math:`\int` (conserve integrand) fixed by a uniform shift of the
        composition — moves the level-set contour normally without distorting the
        band (an under-relaxed Newton step on the enclosed integral)."""
        if self._conserve_integrand is None:
            return
        current = float(self._conserve_area.evaluate())
        rate = float(self._conserve_rate.evaluate())
        if abs(rate) > 1.0e-30:
            relax = 0.3  # full correction injects mesh-motion jitter (design note)
            self.composition.array[:, 0, 0] += relax * (current - self._conserve_target) / rate

    def _carry_and_deform(self, increment, dt):
        """Carry the surface increment inward with the Laplacian diffuser and apply it
        as a mesh deformation along the topography direction (vertical on a box, radial
        on an annulus)."""
        self._carry_bc.array[...] = 0.0
        self._carry_bc.array[self._carry_bc_rows, 0, 0] = increment
        self._diffuser.solve(zero_init_guess=False)
        coords = self.mesh.X.coords
        if self._carry_rows_at_mesh_nodes is not None:
            # direct nodal read (see _build_interior_diffuser: the on-node
            # evaluation class is what this avoids)
            displacement = np.asarray(
                self._carry.array[self._carry_rows_at_mesh_nodes, 0, 0]
            ).flatten()
        else:
            displacement = np.asarray(
                function.evaluate(self._carry.sym[0], coords)
            ).flatten()
        new_coords = coords + displacement[:, None] * self._normal_direction(coords)
        self.mesh.deform(new_coords, dt=dt)
