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
        self._surf_coords = self._surface_node_coords()
        self._surf_rows, self._surf_x = self._field_rows(self.mesh.X, self._surf_coords)
        # Ring coordinates in the array (x-sorted) order the surface fields are held in.
        self._ring_coords = np.ascontiguousarray(
            self._surf_coords[np.argsort(self._surf_coords[:, 0])]
        )

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
        if self._filter_iters:
            self._build_surface_filter()
        if composition is not None:
            self._build_composition_transport()

        self._h_inf = None  # recovered in solve(), consumed in advance()

    # -- geometry / boundary bookkeeping -------------------------------------

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

    def _surface_mean(self, values):
        """Global mean of a surface-node array, identical on every rank (a datum
        gauge must be single-valued across the partition)."""
        comm = uw.mpi.comm
        total = comm.allreduce(float(values.sum()))
        count = comm.allreduce(int(values.size))
        return total / count if count else 0.0

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
        r"""The consistent solve: walls as the free solve, and a STRONG rotated
        constraint prescribing :math:`\mathbf{u}\cdot\hat{\mathbf n}=\tilde u_n` (the
        realised relaxed rate) so the advection velocity keeps the surface a material
        boundary.

        The per-node rotated constraint enforces the datum to machine precision (no
        penalty leak) and, crucially, routes the solve through the rotated LINEAR solver
        (a direct KSP solve) rather than a Newton line search — so a linear (isoviscous)
        flow does not thrash ``newtonls`` the way the old penalty natural BC did. The
        datum is the ``ũ_n`` field, re-evaluated at each solve, so refreshing
        ``_un_target`` each step refreshes the constraint (see #403)."""
        self.consistent = self._new_stokes("cons")
        self.consistent.bodyforce = self.free.bodyforce
        self._apply_walls(self.consistent)
        self._un_target = uw.discretisation.MeshVariable(
            "un_tgt", self.mesh, vtype=uw.VarType.SCALAR, degree=1, continuous=True
        )
        self._un_target_rows, _ = self._field_rows(self._un_target, self._surf_coords)
        self._un_target.array[...] = 0.0
        # Strong rotated u.n = ũ_n on the surface. The datum is read along the rotated
        # per-node normal — the same deform direction the rate was measured along, so no
        # spurious tangential-slope term on a bumpy/rotating surface. `normal=None` uses
        # the FE facet normal (fine on a flat/near-flat top); pass an analytic normal for
        # a curved surface.
        self.consistent.add_rotated_freeslip_bc(0.0, self.surface, normal=self.normal)
        self.consistent._rotated_freeslip_datum = {self.surface: self._un_target.sym[0]}
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
            monotone_mode=None, theta=0.5, old_frame_traceback=True,
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

    def _build_surface_filter(self):
        """Taubin low-pass over the surface graph, to attenuate the facet-scale (CBF
        boundary) node-noise in the recovered equilibrium topography while leaving the
        long-wavelength swell. Built on the extracted surface submesh (parallel-safe);
        the ring is mapped to the submesh DOFs by coordinate."""
        self._surf_mesh = self.mesh.extract_surface(self.surface)
        self._filter_field = uw.discretisation.MeshVariable(
            "h_filter", self._surf_mesh, vtype=uw.VarType.SCALAR, degree=1, continuous=True
        )
        order = np.argsort(self._surf_coords[:, 0])
        ring_sorted = np.ascontiguousarray(self._surf_coords[order])
        tree = uw.kdtree.KDTree(np.ascontiguousarray(self._filter_field.coords))
        self._ring_to_surf = np.asarray(tree.query(ring_sorted, 1)[1]).flatten()

    def _filter_surface(self, values):
        """Taubin-smooth a ring-ordered surface array through the submesh field."""
        self._filter_field.array[self._ring_to_surf, 0, 0] = values
        uw.meshing.smooth_surface_field(self._filter_field, n_iters=self._filter_iters, taubin=True)
        return np.asarray(self._filter_field.array[self._ring_to_surf, 0, 0]).flatten()

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

    def estimate_dt(self):
        """A step size that respects both advection and surface-motion limits.

        The advective limit comes from the material transport (or a velocity CFL if
        there is no composition); the surface limit keeps the per-step displacement
        below ``max_surface_cfl`` cell sizes so the mesh stays well-shaped.
        """
        if self.composition is not None:
            dt_advect = float(self._comp_adv.estimate_dt())
        else:
            dt_advect = self._velocity_cfl()
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
        coords = self._surf_coords
        n = self._normal_direction(coords)          # unit vectors, same as the deform
        dot = np.zeros(coords.shape[0])
        for i in range(self.mesh.dim):
            v_i = np.asarray(function.evaluate(self._v_p1.sym[i], coords)).flatten()
            dot += v_i * n[:, i]
        u_n = dot[np.argsort(coords[:, 0])]
        u_n = self._demean(u_n)
        if self._surface_mask is not None:
            u_n = u_n * self._surface_mask  # pin the surface rate near driven walls
        return u_n

    def _along_surface(self):
        """The along-surface coordinate ``s``, the tangential velocity in ``s``-units,
        and the period (``None`` = open). Cartesian: ``s = x``, ``v_t = v_x``, open.
        Cylindrical annulus: ``s = theta``, ``v_t = omega = v_theta/r``, period ``2*pi``.
        All in the ring-array (x-sorted) order the surface fields are held in."""
        order = np.argsort(self._surf_coords[:, 0])
        v_x = np.asarray(function.evaluate(self._v_p1.sym[0], self._surf_coords)).flatten()[order]
        if not self._cylindrical:
            return self._surf_x, v_x, None
        v_y = np.asarray(function.evaluate(self._v_p1.sym[1], self._surf_coords)).flatten()[order]
        cx, cy = self._ring_coords[:, 0], self._ring_coords[:, 1]
        r = np.hypot(cx, cy)
        theta = np.arctan2(cy, cx)
        v_theta = (-cy * v_x + cx * v_y) / r        # v . theta-hat
        return theta, v_theta / r, 2.0 * np.pi

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
        """First-pass along-surface transport: barycentric-interpolate the surface
        fields at the tangential launch point ``s - v_t*dt`` and report the correction.

        The 1-D interpolation is a convex combination of the bracketing nodal values,
        so it is bounded (cannot overshoot) even when the launch point is crudely
        located. Geometry sets the parametrisation (:meth:`_along_surface`): the
        Cartesian ``x`` with clamped ends, or the annulus ``theta`` with periodic wrap.
        Serial first pass — the interpolation runs over this rank's local ring only (a
        parallel version gathers the ring).
        """
        s, v_t, period = self._along_surface()
        s_dep = s - v_t * dt
        shape_up = self._launch_interp(s, shape, s_dep, period)
        h_inf_up = self._launch_interp(s, h_inf, s_dep, period)
        u_n_up = self._launch_interp(s, u_n, s_dep, period)
        if self.verbose:
            h_scale = np.abs(shape).max() + 1.0e-300
            u_scale = np.abs(u_n).max() + 1.0e-300
            uw.pprint(
                "FreeSurface tangential correction: "
                f"shape {np.abs(shape_up - shape).max() / h_scale:.2e}, "
                f"h_inf {np.abs(h_inf_up - h_inf).max() / h_scale:.2e}, "
                f"u_n {np.abs(u_n_up - u_n).max() / u_scale:.2e}  (relative)"
            )
        if self.tangent_advect == "shape":
            return shape_up, h_inf, u_n
        if self.tangent_advect == "fields":
            return shape, h_inf_up, u_n_up
        return shape, h_inf, u_n

    def _solve_consistent(self, increment, dt):
        r"""Prescribe :math:`\tilde u_n = \Delta h/\Delta t` (mean-removed, net flux
        zero) on the surface and solve for the material-consistent velocity."""
        u_tilde = self._demean(increment / dt)
        self._un_target.array[...] = 0.0
        self._un_target.array[self._un_target_rows, 0, 0] = u_tilde
        self.consistent.solve(zero_init_guess=True)

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
