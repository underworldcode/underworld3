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
        self.verbose = verbose

        # Continuous pressure is required for the CBF reaction on a simplex free
        # surface: discontinuous pressure makes sigma_nn a node-to-node Nyquist
        # zigzag on the boundary edges (design note, Hardening 6).
        if getattr(stokes, "_p_continuous", True) is False:
            uw.pprint(
                "FreeSurface: the free solve uses discontinuous pressure; on a simplex "
                "free surface the sigma_nn recovery zigzags. Prefer continuous pressure."
            )

        self._yhat = self._vertical_unit()
        self._walls = self._classify_walls()
        self._surf_coords = self._surface_node_coords()
        self._surf_rows, self._surf_x = self._field_rows(self.mesh.X, self._surf_coords)

        # Equilibrium-topography field, filled on the surface nodes by the held solve.
        self._hinf_field = uw.discretisation.MeshVariable(
            "h_inf", self.mesh, vtype=uw.VarType.SCALAR, degree=1, continuous=True
        )
        self._hinf_rows, _ = self._field_rows(self._hinf_field, self._surf_coords)

        self._build_surface_velocity_projection()
        self._build_held(driving_buoyancy)
        self._build_consistent()
        self._build_interior_diffuser()
        if composition is not None:
            self._build_composition_transport()

        self._h_inf = None  # recovered in solve(), consumed in advance()

    # -- geometry / boundary bookkeeping -------------------------------------

    def _vertical_unit(self):
        """Upward unit row vector — the topography direction (Cartesian gravity)."""
        row = [[0.0] * self.mesh.dim]
        row[0][-1] = 1.0
        return sympy.Matrix(row)

    def _classify_walls(self):
        r"""Read the free solve's wall BCs as ``{boundary: ("freeslip"|"noslip", normal)}``.

        Free-slip walls are the rotated-free-slip constraints (the recommended
        primitive); no-slip walls are homogeneous essential BCs. The surface label
        is excluded — the free solve leaves it stress-free. Both are replayed onto
        the derived solves.
        """
        walls = {}
        for boundary, wall_normal in getattr(self.free, "_rotated_freeslip_bcs", []):
            if boundary != self.surface:
                walls[boundary] = ("freeslip", wall_normal)
        for bc in self.free.essential_bcs:
            if bc.boundary != self.surface and bc.boundary not in walls:
                walls[bc.boundary] = ("noslip", None)
        return walls

    def _apply_walls(self, solver):
        """Replay the classified wall BCs onto a derived solve (rotated free-slip lid
        pattern: rotate every free-slip wall so shared corners stay consistent)."""
        for boundary, (kind, wall_normal) in self._walls.items():
            if kind == "noslip":
                solver.add_essential_bc(sympy.Matrix([[0.0] * self.mesh.dim]), boundary)
            else:
                solver.add_rotated_freeslip_bc(0.0, boundary, normal=wall_normal)

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
        return solver

    def _copy_constitutive_model(self, solver):
        """Give ``solver`` its own constitutive model of the free solve's class with
        the same parameters. A model instance binds to one solver's unknowns, so it
        cannot be shared; a fresh instance is made and its parameter expressions are
        copied (linear or field-dependent viscosity — a strain-rate-dependent law
        would reference the free velocity and is out of scope)."""
        from underworld3.utilities._api_tools import ExpressionDescriptor

        free_model = self.free.constitutive_model
        solver.constitutive_model = type(free_model)
        copied = set()
        for cls in type(free_model.Parameters).__mro__:
            for name, attr in cls.__dict__.items():
                if isinstance(attr, ExpressionDescriptor) and name not in copied:
                    setattr(solver.constitutive_model.Parameters, name,
                            getattr(free_model.Parameters, name))
                    copied.add(name)

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
        self.held.petsc_options["snes_type"] = "ksponly"

    def _build_consistent(self):
        r"""The consistent solve: walls as the free solve, and a penalty natural BC
        prescribing :math:`\mathbf{u}\cdot\hat{\mathbf n}=\tilde u_n` (the realised
        relaxed rate) so the advection velocity keeps the surface a material
        boundary."""
        self.consistent = self._new_stokes("cons")
        self.consistent.bodyforce = self.free.bodyforce
        self._apply_walls(self.consistent)
        self._un_target = uw.discretisation.MeshVariable(
            "un_tgt", self.mesh, vtype=uw.VarType.SCALAR, degree=1, continuous=True
        )
        self._un_target_rows, _ = self._field_rows(self._un_target, self._surf_coords)
        self._un_target.array[...] = 0.0
        n_hat = self.mesh.boundary_normal(self.surface)
        v_sym = self.consistent.u.sym
        penalty = 1.0e5
        self.consistent.add_natural_bc(
            penalty * (n_hat.dot(v_sym) - self._un_target.sym[0]) * n_hat, self.surface
        )
        self.consistent.petsc_use_pressure_nullspace = True
        self.consistent.petsc_options["snes_type"] = "ksponly"
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
        shape = self._current_shape()
        u_n = self._surface_normal_velocity()
        displacement = self._h_inf - shape
        with np.errstate(divide="ignore", invalid="ignore"):
            gamma = np.where(np.abs(displacement) > 1.0e-9, u_n / displacement, 0.0)
        gamma = np.abs(gamma)  # relax toward equilibrium; L-stable for any rate
        shape_new = self._h_inf + (shape - self._h_inf) * np.exp(-gamma * dt)
        increment = shape_new - shape

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
        height = self.mesh.X.coords[self._surf_rows, -1]
        return self._demean(np.asarray(height, dtype=float))

    def _surface_normal_velocity(self):
        r""":math:`\dot h = \mathbf{u}\cdot\hat{\mathbf n}` on the surface nodes, from
        the P1-projected free-solve velocity and the per-node normal, mean-removed."""
        self._v_p1_proj.solve()
        coords = self._surf_coords
        n_hat = self.mesh.boundary_normal(self.surface)
        components = []
        for i in range(self.mesh.dim):
            v_i = np.asarray(function.evaluate(self._v_p1.sym[i], coords)).flatten()
            n_i = np.asarray(function.evaluate(n_hat[i], coords)).flatten()
            components.append((v_i, n_i))
        dot = sum(v_i * n_i for v_i, n_i in components)
        norm = np.sqrt(sum(n_i * n_i for _, n_i in components))
        norm[norm == 0.0] = 1.0
        u_n = (dot / norm)[np.argsort(coords[:, 0])]
        return self._demean(u_n)

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
        as a purely vertical mesh deformation."""
        self._carry_bc.array[...] = 0.0
        self._carry_bc.array[self._carry_bc_rows, 0, 0] = increment
        self._diffuser.solve(zero_init_guess=False)
        displacement = np.asarray(
            function.evaluate(self._carry.sym[0], self.mesh.X.coords)
        ).flatten()
        new_coords = self.mesh.X.coords.copy()
        new_coords[:, -1] += displacement
        self.mesh.deform(new_coords, dt=dt)
