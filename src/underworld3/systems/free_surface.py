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
        self.verbose = verbose

        # Continuous pressure is required for the CBF reaction on a simplex free
        # surface: discontinuous pressure makes sigma_nn a node-to-node Nyquist
        # zigzag on the boundary edges (design note, Hardening 6).
        if getattr(stokes, "_p_continuous", True) is False:
            uw.pprint(
                "FreeSurface: the free solve uses discontinuous pressure; on a simplex "
                "free surface the sigma_nn recovery zigzags. Prefer continuous pressure."
            )

        # Topography direction: vertical (last axis) on a Cartesian box, radial on a
        # cylindrical annulus. The surface height and the mesh deformation follow it.
        self._cylindrical = (
            self.mesh.CoordinateSystem.coordinate_type == CoordinateSystemType.CYLINDRICAL2D
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
        box, the radius on a cylindrical annulus."""
        if self._cylindrical:
            return np.linalg.norm(coords, axis=1)
        return np.asarray(coords[:, -1], dtype=float)

    def _normal_direction(self, coords):
        """Per-node unit vectors along the topography direction that the surface
        increment is deformed along — vertical (Cartesian) or radial (annulus)."""
        if self._cylindrical:
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
        r"""Arc-length quadrature weights on the globally s-sorted surface ring.

        The trapezoidal weight of node :math:`i` is half the distance to each neighbour
        — :math:`\tfrac12 (s_{i+1}-s_{i-1}) r_i` on a cylindrical ring, :math:`\tfrac12
        (x_{i+1}-x_{i-1})` on an open Cartesian surface (with half-cells at the ends).
        Radii are read live, so the weights follow the deforming surface.
        """
        s = self._s_sorted
        if s.size == 0:
            return s
        if self._ring_period is not None:
            ds = 0.5 * np.mod(np.roll(s, -1) - np.roll(s, 1), self._ring_period)
            radius = self._ring_gather(np.linalg.norm(self._ring_coords, axis=1))
            return ds * radius
        ds = np.empty_like(s)
        ds[1:-1] = 0.5 * (s[2:] - s[:-2])
        ds[0] = 0.5 * (s[1] - s[0])
        ds[-1] = 0.5 * (s[-1] - s[-2])
        return ds

    def _surface_mean(self, values):
        r"""Area-weighted global mean of a surface-node array, identical on every rank
        (a datum gauge must be single-valued across the partition).

        Weighted by arc length, NOT by node count. The distinction is not cosmetic for
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
        weights = self._ring_weights()
        gathered = self._ring_gather(values)
        total = float(weights.sum())
        return float((gathered * weights).sum() / total) if total else 0.0

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

    def _build_consistent(self):
        r"""The consistent solve: walls as the free solve, plus a constraint prescribing
        :math:`\mathbf{u}\cdot\hat{\mathbf n}=\tilde u_n` (the realised relaxed rate) so
        the advection velocity keeps the surface a material boundary.

        ``consistent_constraint`` selects how that datum is imposed:

        ``"strong"``
            A rotated per-node constraint — the same primitive as the held lid, differing
            only in the constraint right-hand side, so free-slip is the ``\tilde u_n = 0``
            member of the same family. It enforces the datum to machine precision (no
            penalty leak) and routes the solve through the rotated LINEAR path (a direct
            KSP solve) rather than a Newton line search, so an isoviscous flow does not
            thrash ``newtonls``. It requires the datum to be discretely flux-free, which
            :meth:`_surface_mean` now guarantees by weighting the demean by arc length.

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
            # on a bumpy or rotating surface.
            self.consistent.add_rotated_freeslip_bc(0.0, self.surface, normal=self.normal)
            self.consistent._rotated_freeslip_datum = {self.surface: self._un_target.sym[0]}
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
            # monotone_mode="clamp" is REQUIRED here, not optional: with
            # old_frame_traceback=True the departure point is deliberately NOT clamped to
            # the domain (the foot is sampled on the OLD geometry, which covers the layer
            # a moving surface vacated), and the limiter is then the ONLY thing bounding a
            # foot that lands outside the old mesh — see SemiLagrangian._trace_back. With
            # monotone_mode=None such a foot falls through to the evaluator's RBF/Shepard
            # fallback, which averages distant DOFs: on a deforming free surface that
            # injects HOT material into the cold boundary layer at the inflow/outflow
            # stagnation points, and the field loses boundedness (T well outside [0,1]).
            monotone_mode="clamp", theta=0.5, old_frame_traceback=True,
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
            s_local = np.arctan2(rc[:, 1], rc[:, 0])
            self._ring_period = 2.0 * np.pi
        else:
            s_local = rc[:, 0].astype(float)
            self._ring_period = None
        self._s_local_n = int(s_local.size)
        counts = comm.allgather(self._s_local_n)
        self._ring_offset = int(np.sum(counts[: comm.rank]))
        s_global = (np.concatenate(comm.allgather(s_local))
                    if comm.size > 1 else s_local.copy())
        self._ring_order = np.argsort(s_global, kind="stable")     # concat -> s-sorted
        self._ring_inv = np.empty_like(self._ring_order)
        self._ring_inv[self._ring_order] = np.arange(self._ring_order.size)
        self._s_sorted = s_global[self._ring_order]

    def _ring_gather(self, local_vals):
        """Local (x-sorted-order) surface values -> the globally s-sorted ring."""
        comm = uw.mpi.comm
        v = (np.concatenate(comm.allgather(np.ascontiguousarray(local_vals)))
             if comm.size > 1 else np.asarray(local_vals, dtype=float))
        return v[self._ring_order]

    def _ring_scatter(self, v_sorted):
        """Globally s-sorted ring values -> this rank's local (x-sorted-order) nodes."""
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
        no material field to set the advective step)."""
        speed = function.evaluate(self.free.u.sym.dot(self.free.u.sym), self.mesh.X.coords)
        vmax = uw.mpi.comm.allreduce(float(np.sqrt(np.asarray(speed)).max()),
                                     op=MPI.MAX)
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
        u_n = np.zeros(coords.shape[0])
        for i in range(self.mesh.dim):
            v_i = np.asarray(function.evaluate(self._v_p1.sym[i], coords)).flatten()
            u_n += v_i * n[:, i]
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
        v_x = np.asarray(function.evaluate(self._v_p1.sym[0], coords)).flatten()
        if not self._cylindrical:
            return v_x
        v_y = np.asarray(function.evaluate(self._v_p1.sym[1], coords)).flatten()
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
        r"""Prescribe :math:`\tilde u_n = \Delta h/\Delta t` (mean-removed by arc length,
        so the net flux is zero) on the surface and solve for the material-consistent
        velocity."""
        u_tilde = self._demean(increment / dt)
        self._un_target.array[...] = 0.0
        self._un_target.array[self._un_target_rows, 0, 0] = u_tilde
        try:
            self.consistent.solve(zero_init_guess=True)
        except NotImplementedError as exc:
            # The rotated datum is implemented on the LINEAR rotated path only; the solve
            # dispatches by a measured residual probe, so this fires exactly when the
            # rheology is genuinely nonlinear. Name the knob rather than leaving the
            # caller with the primitive's message.
            raise NotImplementedError(
                "FreeSurface: the strong rotated constraint cannot prescribe u.n = ũ_n "
                "for a nonlinear rheology (the rotated datum is implemented on the linear "
                "path only). Construct the manager with consistent_constraint='penalty' "
                "to impose the material-surface rate weakly instead."
            ) from exc

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
        displacement = np.asarray(
            function.evaluate(self._carry.sym[0], coords)
        ).flatten()
        new_coords = coords + displacement[:, None] * self._normal_direction(coords)
        self.mesh.deform(new_coords, dt=dt)
