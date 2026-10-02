r"""Conservative level-set transport: advection, reinitialisation, mass correction.

A conservative level set represents an interface by the 0.5 contour of a
smoothed indicator

.. math::
    \psi = \tfrac12\left(1 + \tanh\frac{\varphi}{2\varepsilon}\right),

with :math:`\varphi` the signed distance to the interface and
:math:`\varepsilon` the interface thickness. Transport of :math:`\psi` is
ordinary scalar advection; what makes it a level set is what happens between
steps: a reinitialisation that restores the :math:`\tanh` profile without
moving the 0.5 contour, and a global correction that restores the enclosed
volume. Both are post-step operations on the field, so any scalar transport
solver can carry it.

:class:`LevelSetSolver` takes the Eulerian SUPG solver
(:class:`~underworld3.systems.AdvDiffusion`, composed with the default
:class:`~underworld3.systems.ddt.EulerianSUPG` transport manager) by default
and the semi-Lagrangian solver on request. The reinitialisation equation is that of
Parameswaran and Mandal (2023), integrated in pseudo-time with SSP-RK3
(Gottlieb and Shu 1998); the mass correction is the uniform shift of Zhang,
Zou and Greaves (2010). The signed-distance helpers accept a polygon or a
curve (they use ``shapely``, an optional dependency) or a precomputed
distance array. :func:`material_property_field` blends material properties
across the interface, in the manner of g-adopt's ``field_interface``.

The level-set pipeline, the SUPG transport it drove and the LeVeque
swirling-flow comparison are NengLu's (issue #657, branch ``levelset``);
this module unifies the two variants of that work on one solver interface.

References
----------
Parameswaran, S. and Mandal, J. C. (2023). A stable interface-preserving
reinitialization equation for conservative level set method. European
Journal of Mechanics B/Fluids, 98, 40-63.

Zhang, Y., Zou, Q. and Greaves, D. (2010). Numerical simulation of
free-surface flow using the level-set method with global mass correction.
International Journal for Numerical Methods in Fluids, 63, 651-680.
"""

import warnings
from typing import Optional, Union

import numpy as np
import sympy

import underworld3 as uw
from underworld3 import discretisation, systems
from underworld3.materials import MaterialDistribution, _property_name


# ---------------------------------------------------------------------------
# Initial condition
# ---------------------------------------------------------------------------

def _tanh_profile(distance, epsilon):
    r"""The conservative level-set profile :math:`\tfrac12(1 + \tanh(\varphi/2\varepsilon))`."""
    return (1.0 + np.tanh(np.asarray(distance) / (2.0 * np.asarray(epsilon)))) / 2.0


def _shapely():
    """The optional ``shapely`` dependency, or a clear error."""
    try:
        from shapely import geometry
        from shapely import prepare
    except ImportError as exc:
        raise ImportError(
            "The level-set geometry helpers need the optional package "
            "'shapely' (pip install shapely). Pass signed_distance= to "
            "initialise_psi to avoid it."
        ) from exc
    return geometry, prepare


def initialise_psi(
    psi: discretisation.MeshVariable,
    epsilon,
    *,
    signed_distance=None,
    interface_geometry: Optional[str] = None,
    interface=None,
    interface_coordinates=None,
    boundary_coordinates=None,
) -> None:
    r"""Fill ``psi`` with the conservative level-set profile of an interface.

    Parameters
    ----------
    psi : MeshVariable
        Scalar field to fill; 1 inside the interface, 0 outside, 0.5 on it.
    epsilon : MeshVariable or float
        Interface thickness (see :func:`interface_thickness`).
    signed_distance : ndarray, optional
        Precomputed signed distance at ``psi``'s nodes, positive inside.
        When given, the geometry arguments are ignored and ``shapely`` is
        not needed.
    interface_geometry : {"curve", "polygon", "shapely"}, optional
        How the interface is described.
    interface : shapely LineString or Polygon, optional
        For ``interface_geometry="shapely"``.
    interface_coordinates : sequence of (x, y), optional
        Vertices of the curve or polygon.
    boundary_coordinates : sequence of (x, y), optional
        Extra vertices that close an open curve into the polygon that
        defines the inside.
    """
    eps = epsilon.array[:, 0, 0] if hasattr(epsilon, "array") else float(epsilon)
    if signed_distance is not None:
        psi.array[:, 0, 0] = _tanh_profile(signed_distance, eps)
        return
    if interface_geometry is None:
        raise ValueError("Provide either signed_distance or interface_geometry.")
    if interface_coordinates is None and interface_geometry != "shapely":
        raise ValueError(
            f"interface_coordinates is required for interface_geometry={interface_geometry!r}.")
    distance = _signed_distance_from_geometry(
        interface_geometry, interface, interface_coordinates, boundary_coordinates,
        np.asarray(psi.coords))
    psi.array[:, 0, 0] = _tanh_profile(distance, eps)


def interface_thickness(
    mesh: discretisation.Mesh,
    phi: discretisation.MeshVariable,
    *,
    scale: float = 0.35,
    use_min_edge_length: bool = False,
) -> discretisation.MeshVariable:
    r"""Interface thickness :math:`\varepsilon` from the local cell size.

    :math:`\varepsilon = \mathrm{scale}\cdot V^{1/d}/\sqrt{d}` per cell
    (or ``scale`` times the shortest edge), carried to ``phi``'s nodes from
    the nearest cell centroid. Returned as a scalar MeshVariable of the same
    degree as ``phi``.

    ``scale=0.35`` (the default, following the discontinuous-Galerkin
    setting of g-adopt) gives :math:`\varepsilon \approx h/8` on triangles,
    a band well under one cell. A continuous-Galerkin transport rings at
    that: on a rotating circle at 32 cells across, the clip of the ringing
    changed the volume by 0.84% per revolution at 0.35, 0.28% at 1.0 and
    0.18% at 2.0 (:math:`\varepsilon \approx 0.7h`, a band of two to three
    cells, ringing gone); at 3.0 the reinitialisation's own curvature error
    takes over (0.85%). For the SUPG transport, ``scale`` between 1.5 and 2
    is the sensible setting.
    """
    from scipy.spatial import cKDTree

    dm = mesh.dm
    dim = mesh.dim
    c_start, c_end = dm.getHeightStratum(0)
    n_cells = c_end - c_start
    cell_epsilon = np.empty(n_cells)
    cell_centroids = np.empty((n_cells, dim))

    if use_min_edge_length:
        if mesh.qdegree > 1:
            raise ValueError("use_min_edge_length=True needs a straight-edged mesh (qdegree=1).")
        v_start, v_end = dm.getDepthStratum(0)
        coords = np.asarray(mesh.X.coords)
        for i, cell in enumerate(range(c_start, c_end)):
            closure, _ = dm.getTransitiveClosure(cell)
            verts = [p - v_start for p in closure if v_start <= p < v_end]
            v_coords = coords[verts]
            edges = [np.linalg.norm(v_coords[a] - v_coords[b])
                     for a in range(len(verts)) for b in range(a + 1, len(verts))]
            cell_epsilon[i] = scale * min(edges)
            cell_centroids[i, :] = v_coords.mean(axis=0)
    else:
        factor = scale / np.sqrt(dim)
        for i, cell in enumerate(range(c_start, c_end)):
            vol, centroid, _ = dm.computeCellGeometryFVM(cell)
            cell_epsilon[i] = factor * float(np.asarray(vol).ravel()[0]) ** (1.0 / dim)
            cell_centroids[i, :] = np.asarray(centroid).ravel()[:dim]

    epsilon = discretisation.MeshVariable(
        r"\epsilon", mesh, 1, degree=phi.degree, continuous=phi.continuous)
    _, nearest = cKDTree(cell_centroids).query(np.asarray(phi.coords))
    epsilon.array[:, 0, 0] = cell_epsilon[nearest]
    return epsilon


def _signed_distance_closed(polygon, points):
    """Positive inside a closed polygon, negative outside."""
    geometry, prepare = _shapely()
    prepare(polygon)
    boundary = polygon.boundary
    inside = np.array([polygon.contains(geometry.Point(p)) for p in points])
    distance = np.array([boundary.distance(geometry.Point(p)) for p in points])
    return np.where(inside, distance, -distance)


def _signed_distance_open(curve, enclosed, points):
    """Distance to an open curve, signed by the polygon that defines the inside."""
    geometry, prepare = _shapely()
    prepare(enclosed)
    inside = np.array([enclosed.intersects(geometry.Point(p)) for p in points])
    distance = np.array([curve.distance(geometry.Point(p)) for p in points])
    return np.where(inside, distance, -distance)


def _signed_distance_from_geometry(interface_geometry, interface, interface_coordinates,
                                   boundary_coordinates, points):
    geometry, _prepare = _shapely()

    def closed_from(coords):
        if boundary_coordinates is not None:
            raise ValueError("boundary_coordinates is only for an open interface.")
        return _signed_distance_closed(geometry.Polygon(coords), points)

    def open_from(curve, coords):
        if boundary_coordinates is None:
            raise ValueError("boundary_coordinates must close an open interface.")
        enclosed = geometry.Polygon(np.vstack((coords, boundary_coordinates)))
        return _signed_distance_open(curve, enclosed, points)

    if interface_geometry == "curve":
        curve = geometry.LineString(interface_coordinates)
        return closed_from(interface_coordinates) if curve.is_closed else open_from(curve, interface_coordinates)
    if interface_geometry == "polygon":
        if boundary_coordinates is None:
            return _signed_distance_closed(geometry.Polygon(interface_coordinates), points)
        return open_from(geometry.LineString(interface_coordinates), interface_coordinates)
    if interface_geometry == "shapely":
        if interface is None:
            raise ValueError("interface must be given for interface_geometry='shapely'.")
        if isinstance(interface, geometry.Polygon):
            return _signed_distance_closed(interface, points)
        return open_from(interface, np.asarray(interface.coords))
    raise ValueError(
        f"Unknown interface_geometry={interface_geometry!r}; choose 'curve', 'polygon' or 'shapely'.")


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------

class LevelSetSolver:
    r"""Conservative level-set transport on a scalar transport solver.

    Each call to :meth:`solve` advects the level set by one step, applies
    the optional wall correction, reinitialises when due, and restores the
    enclosed volume.

    **Reinitialisation** integrates, in pseudo-time :math:`\tau`,

    .. math::
        \frac{\partial\psi}{\partial\tau}
            = -\psi(1-\psi)(1-2\psi) + \varepsilon(1-2\psi)|\nabla\psi|

    (Parameswaran and Mandal 2023, eq. 17) with SSP-RK3. Both terms carry the
    factor :math:`(1-2\psi)`, so :math:`\psi = 0.5` is a fixed point: the
    profile sharpens to width :math:`\varepsilon` without moving the
    interface. :math:`|\nabla\psi|` is an :math:`L_2` projection onto the
    mesh at each stage.

    **Mass correction** finds the uniform shift :math:`\delta` with
    :math:`\int \mathrm{clip}(\psi + \delta, 0, 1)\,d\Omega` equal to the
    initial enclosed volume and leaves the field in that clipped, shifted
    state (Zhang, Zou and Greaves 2010). The map is monotone and its slope is
    the area of the transition band, so a bracketed secant iteration started
    from that slope reaches the target in a few integrals (five against about
    thirty for the bisection it replaces, to the same 1e-10).

    Parameters
    ----------
    level_set : MeshVariable
        Continuous scalar field holding :math:`\psi`.
    velocity : MeshVariable or sympy Matrix
        Advecting velocity.
    epsilon : MeshVariable
        Interface thickness (:func:`interface_thickness`).
    advection : {"supg", "slcn"}, default "supg"
        The transport solver: the Eulerian SUPG solver or the semi-Lagrangian
        one. Both are pure advection here.
    order, theta : int, float
        Time scheme of the transport solver (Crank-Nicolson by default, the
        same meaning for both solvers).
    reini_dt : float, optional
        Pseudo-time step of the reinitialisation (default half the smallest
        :math:`\varepsilon`).
    reini_steps : int, default 5
        Pseudo-time steps per reinitialisation.
    reini_frequency : int, optional
        Advection steps between reinitialisations; by default from the
        domain size and :math:`\varepsilon`.
    far_field : float, optional
        Value of :math:`\psi` imposed on every mesh boundary (0 outside the
        interface, 1 inside). Set it whenever the flow crosses the domain
        boundary: a continuous-Galerkin scheme with no value on an inflow
        boundary lets mass in, measured as a 4% volume drift in twenty steps
        of a rotating circle against 8e-5 with the value imposed. Leave it
        unset only when the boundary is a streamline.
    adv_solver_opts : dict, optional
        PETSc options forwarded to the transport solver.
    adv_solver_bc : sequence of str, optional
        Box wall labels on which a zero normal gradient is imposed after
        each step by copying the neighbouring interior nodes (a box-mesh
        convenience).
    conserve_mass : bool or "auto", default "auto"
        Apply the global mass correction after each step. ``"auto"`` turns
        it on for the semi-Lagrangian transport, which loses volume through
        interpolation, and off for the Eulerian one, which conserves the
        enclosed volume to solver tolerance on its own once ``far_field`` is
        set where the flow crosses the boundary (measured: 8e-5 over twenty
        steps; the reinitialisation changes it at second order only). What
        does change it is the clip to [0, 1] of the transport's ringing at a
        thin band: 0.84% per revolution of a circle at the default thickness
        (``scale=0.35``), 0.18% at ``scale=2.0`` (see
        :func:`interface_thickness`). Turn the correction on if that
        matters; it costs about as much as the Eulerian advection step.
    mass_correction_tol, mass_correction_max_iter
        Bisection tolerance on the volume and iteration cap.

    Examples
    --------
    >>> mesh = uw.meshing.UnstructuredSimplexBox(cellSize=1 / 32)
    >>> psi = uw.discretisation.MeshVariable("psi", mesh, 1, degree=2)
    >>> eps = uw.systems.level_set.interface_thickness(mesh, psi)
    >>> uw.systems.level_set.initialise_psi(psi, eps, interface_geometry="polygon",
    ...                                     interface_coordinates=circle_points)
    >>> ls = uw.systems.LevelSetSolver(psi, velocity=v.sym, epsilon=eps)
    >>> for step in range(100):
    ...     ls.solve(dt)
    """

    def __init__(
        self,
        level_set: discretisation.MeshVariable,
        *,
        velocity,
        epsilon: discretisation.MeshVariable,
        advection: str = "supg",
        order: int = 1,
        theta: float = 0.5,
        reini_dt: Optional[float] = None,
        reini_steps: int = 5,
        reini_frequency: Optional[int] = None,
        far_field: Optional[float] = None,
        adv_solver_opts: Optional[dict] = None,
        adv_solver_bc=None,
        conserve_mass: Union[bool, str] = "auto",
        mass_correction_tol: float = 1.0e-10,
        mass_correction_max_iter: int = 40,
    ) -> None:
        if level_set.num_components != 1:
            raise ValueError("level_set must be a scalar MeshVariable.")
        if not level_set.continuous:
            raise ValueError("level_set must be a continuous MeshVariable.")
        if advection not in ("supg", "slcn"):
            raise ValueError(f"advection must be 'supg' or 'slcn', not {advection!r}.")

        self.phi = level_set
        self.mesh = level_set.mesh
        self.velocity = velocity.sym if isinstance(velocity, discretisation.MeshVariable) else velocity
        self.epsilon = epsilon
        self.advection = advection
        self.reini_dt = float(reini_dt) if reini_dt is not None else 0.5 * self._global_min_epsilon()
        self.reini_steps = int(reini_steps)
        self.step = 0

        if advection == "supg":
            self._adv_solver = systems.AdvDiffusion(
                self.mesh, self.phi, self.velocity, order=order, theta=theta)
        else:
            history = systems.ddt.SemiLagrangian(
                self.mesh, self.phi.sym, self.velocity,
                vtype=uw.VarType.SCALAR, degree=self.phi.degree, continuous=self.phi.continuous,
                varsymbol="cphi", bcs=[], order=order, smoothing=0.0,
                monotone_mode="clamp", theta=theta)
            self._adv_solver = systems.AdvDiffusionSLCN(
                self.mesh, u_Field=self.phi, V_fn=self.velocity, order=order,
                DuDt=history, theta=theta)
            self._adv_solver.constitutive_model = uw.constitutive_models.DiffusionModel
        self._adv_solver.constitutive_model.Parameters.diffusivity = 0.0
        if far_field is not None:
            for boundary in self.mesh.boundaries:
                self._adv_solver.add_dirichlet_bc(float(far_field), boundary.name)
        self._adv_solver_bc = adv_solver_bc
        for key, value in (adv_solver_opts or {}).items():
            self._adv_solver.petsc_options[key] = value

        # |grad psi| for the reinitialisation, projected onto the mesh.
        # Named from self.phi.name (not a fixed literal) -- a hardcoded
        # name here means every SECOND (and later) LevelSetSolver on the
        # same mesh collides on this auxiliary variable: PETSc/UW3 then
        # prints "Variable ... already exists - Skipping" and SILENTLY
        # hands back the FIRST instance's phi_grad, so the second
        # material's reinitialisation reads/writes the first material's
        # gradient-magnitude field with no exception anywhere -- precisely
        # the "silently shared level sets" hazard
        # MaterialDistribution._check_level_sets_are_new's own docstring
        # warns about, one level deeper (inside LevelSetSolver itself,
        # not just at the distribution's own level-set naming). Every
        # multi-material script with 2+ materials on one mesh (both
        # MultiMaterialLevelSet and MaterialCLS) hits this.
        self._grad_magnitude = sympy.sqrt(sum(g ** 2 for g in self.mesh.vector.gradient(self.phi.sym[0])))
        self.phi_grad = discretisation.MeshVariable(
            rf"|\nabla {self.phi.name}|", self.mesh, 1, degree=self.phi.degree, continuous=self.phi.continuous)
        self._grad_projector = systems.Projection(self.mesh, self.phi_grad, degree=self.phi.degree)
        self._grad_projector.uw_function = self._grad_magnitude

        self._reini_frequency = int(reini_frequency) if reini_frequency is not None else self._default_frequency()

        if conserve_mass == "auto":
            conserve_mass = advection == "slcn"
        self.conserve_mass = bool(conserve_mass)
        self._mass_correction_tol = float(mass_correction_tol)
        self._mass_correction_max_iter = int(mass_correction_max_iter)
        self._clip_volume_change = 0.0
        self._target_volume = self.interface_volume()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def advection_solver(self):
        """The transport solver carrying the level set."""
        return self._adv_solver

    @property
    def reini_frequency(self) -> int:
        """Advection steps between reinitialisations."""
        return self._reini_frequency

    def estimate_dt(self, **kwargs):
        """The transport solver's timestep estimate (see its ``estimate_dt``)."""
        return self._adv_solver.estimate_dt(**kwargs)

    def solve(self, dt: float, *, reinitialise: bool = True) -> None:
        """Advance the level set by one step of size ``dt``."""
        self._adv_solver.solve(timestep=dt)
        # The reinitialisation equation assumes psi in [0, 1]; the transport
        # can overshoot at a band a cell wide, so clip before anything reads
        # the field. Records what the clip removed, for the volume budget.
        self._clip_volume_change += self._clip_in_place()
        if self._adv_solver_bc:
            self._apply_boundary_neumann(labels=self._adv_solver_bc)
        self.step += 1

        if reinitialise and self.step % self._reini_frequency == 0:
            self.reinitialise()
            self._clip_volume_change += self._clip_in_place()   # RK stages can leave 1e-6 undershoots
            if self._adv_solver_bc:
                self._apply_boundary_neumann(labels=self._adv_solver_bc)

        if self.conserve_mass:
            self._correct_mass(self._target_volume)

    def reinitialise(self) -> None:
        """Run ``reini_steps`` SSP-RK3 pseudo-time steps of the reinitialisation equation."""
        for _ in range(self.reini_steps):
            self._reini_ssprk3_step(self.reini_dt)

    @property
    def volume_drift(self) -> float:
        """Relative change of the enclosed volume since construction."""
        return (self.interface_volume() - self._target_volume) / self._target_volume

    def interface_volume(self) -> float:
        r"""The enclosed volume :math:`\int\psi\,d\Omega`."""
        return uw.maths.Integral(self.mesh, self.phi.sym[0]).evaluate()

    def clamp(self, lo: float = 0.0, hi: float = 1.0) -> None:
        """Clip the field to ``[lo, hi]`` in place. Not mass-conserving on its own."""
        self.phi.array[:, 0, 0] = np.clip(self.phi.array[:, 0, 0], lo, hi)

    def _clip_in_place(self) -> float:
        """Clip to [0, 1]; return the volume the clip changed (a nodal estimate)."""
        values = np.asarray(self.phi.array[:, 0, 0])
        clipped = np.clip(values, 0.0, 1.0)
        if np.array_equal(values, clipped):
            return 0.0
        before = self.interface_volume()
        self.phi.array[:, 0, 0] = clipped
        return self.interface_volume() - before

    # ------------------------------------------------------------------
    # Reinitialisation
    # ------------------------------------------------------------------

    def _rhs(self, values: np.ndarray) -> np.ndarray:
        """The right-hand side of the reinitialisation equation at nodal values."""
        self.phi.array[:, 0, 0] = values
        self._grad_projector.uw_function = self._grad_magnitude
        self._grad_projector.solve()
        grad = np.asarray(self.phi_grad.array[:, 0, 0])
        eps = np.asarray(self.epsilon.array[:, 0, 0])
        sharpening = -values * (1.0 - values) * (1.0 - 2.0 * values)
        balance = eps * (1.0 - 2.0 * values) * grad
        return sharpening + balance

    def _reini_ssprk3_step(self, dtau: float) -> None:
        psi0 = np.array(self.phi.array[:, 0, 0])
        psi1 = psi0 + dtau * self._rhs(psi0)
        psi2 = 0.75 * psi0 + 0.25 * psi1 + 0.25 * dtau * self._rhs(psi1)
        self.phi.array[:, 0, 0] = psi0 / 3.0 + 2.0 * psi2 / 3.0 + 2.0 * dtau * self._rhs(psi2) / 3.0

    def _global_min_epsilon(self) -> float:
        from mpi4py import MPI
        values = np.asarray(self.epsilon.array[:, 0, 0])
        local = float(values.min()) if values.size else np.inf
        return uw.mpi.comm.allreduce(local, op=MPI.MIN)

    def _default_frequency(self) -> int:
        """Reinitialise every step on a coarse mesh, less often as it refines."""
        from mpi4py import MPI
        coords = np.asarray(self.mesh.X.coords)
        dim = coords.shape[1]
        hi = np.array([uw.mpi.comm.allreduce(float(coords[:, i].max()) if len(coords) else -np.inf, op=MPI.MAX)
                       for i in range(dim)])
        lo = np.array([uw.mpi.comm.allreduce(float(coords[:, i].min()) if len(coords) else np.inf, op=MPI.MIN)
                       for i in range(dim)])
        domain_size = float(np.sqrt(np.sum((hi - lo) ** 2)))
        return max(1, round(4.9e-3 * domain_size / self._global_min_epsilon() - 0.25))

    # ------------------------------------------------------------------
    # Wall correction (box meshes)
    # ------------------------------------------------------------------

    def _apply_boundary_neumann(self, labels=("Left", "Right", "Top", "Bottom")) -> None:
        """Zero normal gradient on box walls: copy the neighbouring interior row or column."""
        from mpi4py import MPI
        comm = uw.mpi.comm

        coords = np.asarray(self.phi.coords)
        n_local = coords.shape[0]
        axis_for_label = {"Left": 0, "Right": 0, "Top": 1, "Bottom": 1}
        is_min_side = {"Left": True, "Right": False, "Top": False, "Bottom": True}

        for label in labels:
            axis = axis_for_label[label]
            tang = 1 - axis
            op = MPI.MIN if is_min_side[label] else MPI.MAX
            if n_local:
                local_extreme = coords[:, axis].min() if is_min_side[label] else coords[:, axis].max()
            else:
                local_extreme = np.inf if is_min_side[label] else -np.inf
            wall_val = comm.allreduce(float(local_extreme), op=op)

            local_axis_vals = np.unique(coords[:, axis]) if n_local else np.empty(0)
            all_axis_vals = np.unique(np.concatenate(comm.allgather(local_axis_vals)))
            if all_axis_vals.size < 2:
                continue
            inner_val = all_axis_vals[np.argsort(np.abs(all_axis_vals - wall_val))][1]

            inner_idx = np.where(np.isclose(coords[:, axis], inner_val, atol=1e-8))[0] if n_local else np.empty(0, dtype=int)
            local_pairs = (np.column_stack((coords[inner_idx, tang], np.asarray(self.phi.array[inner_idx, 0, 0])))
                           if len(inner_idx) else np.empty((0, 2)))
            gathered = [p for p in comm.allgather(local_pairs) if p.shape[0] > 0]
            if not gathered:
                continue
            table = np.vstack(gathered)
            table = table[np.argsort(table[:, 0])]
            table = table[np.concatenate(([True], np.diff(table[:, 0]) > 1e-10))]

            wall_idx = np.where(np.isclose(coords[:, axis], wall_val, atol=1e-8))[0] if n_local else np.empty(0, dtype=int)
            if len(wall_idx) == 0:
                continue
            wall_tang = coords[wall_idx, tang]
            pos = np.clip(np.searchsorted(table[:, 0], wall_tang), 1, len(table) - 1)
            left_err = np.abs(wall_tang - table[pos - 1, 0])
            right_err = np.abs(table[pos, 0] - wall_tang)
            nearest = np.where(right_err < left_err, pos, pos - 1)
            good = np.minimum(left_err, right_err) <= 1e-6
            if not np.all(good):
                warnings.warn(
                    f"Wall correction on {label!r}: {np.count_nonzero(~good)} wall node(s) "
                    "have no interior counterpart; left unchanged.", stacklevel=2)
            self.phi.array[wall_idx[good], 0, 0] = table[nearest[good], 1]

    # ------------------------------------------------------------------
    # Mass correction
    # ------------------------------------------------------------------

    def _correct_mass(self, target: float, lo: float = 0.0, hi: float = 1.0) -> None:
        r"""Uniform shift, clipped, restoring the enclosed volume to ``target``.

        Finds :math:`\delta` with :math:`\int\mathrm{clip}(\psi+\delta, lo, hi)\,d\Omega
        = V_{\rm target}` and leaves the field in that state. The clip makes
        the map :math:`\delta \mapsto V` nonlinear but monotone: it only moves
        the transition band, so its slope is the band's area. A secant
        iteration started from that slope converges in a few evaluations;
        every evaluation is one integral over the mesh, which is what made
        the bisection this replaces cost most of a level-set step. The
        bracket is kept as a safeguard: an iterate that leaves it falls back
        to its midpoint.
        """
        data0 = np.array(self.phi.array[:, 0, 0])

        def volume_for_shift(delta: float) -> float:
            self.phi.array[:, 0, 0] = np.clip(data0 + delta, lo, hi)
            return self.interface_volume()

        v0 = volume_for_shift(0.0)
        residual = target - v0
        if abs(residual) < self._mass_correction_tol:
            return

        # First guess: only the band moves, so dV/d(delta) is about its area.
        # The nodal fraction of the field inside (lo, hi) times the domain area
        # is a fair estimate of that on a reasonably uniform mesh.
        span = hi - lo
        band = float(np.mean((data0 > lo + 1e-6 * span) & (data0 < hi - 1e-6 * span)))
        domain = self._domain_volume()
        slope = max(band * domain, 1e-12)
        d_prev, v_prev = 0.0, v0
        d_cur = residual / slope
        lo_d, hi_d = (0.0, np.inf) if residual > 0 else (-np.inf, 0.0)

        for _ in range(self._mass_correction_max_iter):
            v_cur = volume_for_shift(d_cur)
            r_cur = v_cur - target
            if abs(r_cur) < self._mass_correction_tol:
                return
            # keep the bracket [lo_d, hi_d] around the root (V is monotone)
            if r_cur < 0:
                lo_d = max(lo_d, d_cur)
            else:
                hi_d = min(hi_d, d_cur)
            dv = v_cur - v_prev
            if abs(dv) > 0:
                d_next = d_cur - r_cur * (d_cur - d_prev) / dv
            else:
                d_next = d_cur + 2.0 * (d_cur - d_prev)
            if not (lo_d < d_next < hi_d) or not np.isfinite(d_next):
                if np.isfinite(lo_d) and np.isfinite(hi_d):
                    d_next = 0.5 * (lo_d + hi_d)
                else:
                    d_next = 2.0 * d_cur
            d_prev, v_prev, d_cur = d_cur, v_cur, d_next

        warnings.warn(
            f"Mass correction did not reach the target volume {target:.6g} in "
            f"{self._mass_correction_max_iter} iterations; leaving the field at the last shift.",
            stacklevel=2)

    def _domain_volume(self) -> float:
        if not hasattr(self, "_domain_volume_value"):
            self._domain_volume_value = float(uw.maths.Integral(self.mesh, sympy.Integer(1)).evaluate())
        return self._domain_volume_value


# ---------------------------------------------------------------------------
# Material properties across the interface
# ---------------------------------------------------------------------------

def material_property_field(level_set, field_values, interface: str):
    r"""A material property blended across one or more level sets.

    Parameters
    ----------
    level_set : sympy expression or list of them
        The level-set field(s), e.g. ``psi.sym[0]``; with several, the last
        is the innermost.
    field_values : list of float
        One value per material, innermost last.
    interface : {"sharp", "sharp_adjoint", "arithmetic", "geometric", "harmonic"}
        How the property crosses the interface.
    """
    kinds = ("sharp", "sharp_adjoint", "arithmetic", "geometric", "harmonic")
    if interface not in kinds:
        raise ValueError(f"interface must be one of {kinds}, not {interface!r}.")

    level_sets = list(level_set) if isinstance(level_set, (list, tuple)) else [level_set]
    values = list(field_values)

    result = None
    while level_sets:
        ls = sympy.Max(sympy.Min(level_sets.pop(), 1), 0)
        value = values.pop()
        other = values.pop() if not level_sets else result
        if interface == "sharp":
            result = sympy.Piecewise((value, ls > sympy.Rational(1, 2)), (other, True))
        elif interface == "sharp_adjoint":
            shifted = ls - sympy.Rational(1, 2)
            heaviside = (shifted + sympy.Abs(shifted)) / 2 / shifted
            result = value * heaviside + other * (1 - heaviside)
        elif interface == "arithmetic":
            result = value * ls + other * (1 - ls)
        elif interface == "geometric":
            result = value ** ls * other ** (1 - ls)
        else:
            result = 1 / (ls / value + (1 - ls) / other)
    return result


# =============================================================================
# MaterialCLS: materials on conservative level sets, in MaterialSwarm's own
# style
# =============================================================================
#
#     materials = uw.level_set.MaterialCLS(mesh, velocity=v.sym)
#     materials.add("mantle", shear_viscosity_0=1.0,   density=3300)
#     materials.add("slab",   shear_viscosity_0=1.0e3, density=3400)
#     materials["slab"] = mesh.X[1] > 0.53
#     stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
#     stokes.materials = materials
#     stokes.bodyforce = -materials.density * mesh.CoordinateSystem.unit_e_1
#
# reads exactly like the uw.swarm.MaterialSwarm example it is modelled on --
# `.add()`, `materials["name"] = region`, `materials.<property>`,
# `stokes.materials = materials` are all inherited, unmodified, from the
# SAME shared base class MaterialSwarm and MaterialRegions already use
# (underworld3.materials.MaterialDistribution) -- see that module's
# docstring: "Both present the same face to a solver". MaterialCLS is a
# third _distribution_ (where a material is), sharing the same _definition_
# machinery (what a material is) as the other two. `stokes.materials = ...`
# therefore works identically for all three, with no special-casing.
#
# What's genuinely different from MaterialSwarm here:
#   * No particles: each material (other than the implicit default -- see
#     below) is its OWN conservative level set (LevelSetSolver), advected
#     directly by `velocity` -- no swarm, no population control, no
#     proxy-sampling choice.
#   * `materials["name"] = region` needs a DISTANCE, not just a boolean
#     mask: a level set's tanh profile is built from how FAR a point is
#     from the interface, not merely which side it's on. A bare boolean
#     half-space condition (`mesh.X[1] > 0.53`, exactly MaterialSwarm's own
#     example) is auto-converted to its EXACT signed distance when it is
#     linear/affine in the mesh coordinates -- the common case (a lid, a
#     slab dipping at a fixed angle). For curved interfaces, pass a signed-
#     distance expression directly (positive inside), e.g.
#     `r - sympy.sqrt((x-x0)**2+(y-y0)**2)` for a circular inclusion -- see
#     `_normalise_region`'s docstring for exactly what is and is not
#     accepted, and why arbitrary nonlinear conditions are refused rather
#     than silently guessed at.
#   * Exactly ONE material may be left unpainted -- the FIRST one declared
#     (materials.materials[0], index 0) -- and it becomes the IMPLICIT
#     background, `max(1 - sum(others), 0)`, with no level set or solver of
#     its own: this mirrors MaterialSwarm/MaterialRegions' own "unpainted
#     particles/cells default to index 0" convention, translated to a
#     continuous field. A SECOND unpainted material is refused (ambiguous
#     for a continuous partition, unlike a discrete index).
#   * `.solve(dt)` is the CLS-specific transport step (advects every
#     explicit material's level set, with reinitialisation/mass
#     correction) -- the direct analogue of MaterialSwarm's own
#     `.advection(v, dt)`, called explicitly by the model script each step
#     rather than implied by a solver.
#   * `mixing()`/`blend()`/`materials.<property>` add a third rule,
#     "geometric" (`exp(sum(w_i log v_i))`), alongside the base class's
#     "arithmetic" and "harmonic" -- the log-mean is the standard choice
#     for a property spanning orders of magnitude (viscosity), where
#     arithmetic under-weights a thin weak/strong inclusion and harmonic
#     assumes an iso-stress (series) geometry that does not generally hold
#     for an arbitrary, evolving multi-material arrangement. See
#     `level_set.material_property_field`, which already offers
#     "geometric" for the older two-material nested-list form this class
#     supersedes for N independent, arbitrarily-adjacent materials.


class _DerivedLevelSet:
    """The implicit default (index-0) material's "level set": never
    advected or reinitialised on its own -- it simply IS the complement of
    every explicitly-tracked material, by construction, recomputed fresh
    every time it's read. Duck-types just enough of a MeshVariable
    (`.sym[0]`) for MaterialDistribution.blend()'s `masks[i].sym[0]` to work
    unmodified."""

    def __init__(self, others):
        self._others = others  # explicit sibling level-set MeshVariables

    @property
    def sym(self):
        total = sum((sympy.Max(o.sym[0], 0) for o in self._others), sympy.Integer(0))
        return sympy.Matrix([[sympy.Max(1 - total, 0)]])


class MaterialCLS(MaterialDistribution):
    r"""Materials on conservative level sets, read the same way
    :class:`~underworld3.swarm.MaterialSwarm` is.

    Parameters
    ----------
    mesh : Mesh
        The computational mesh.
    velocity : MeshVariable or sympy expression
        Shared advecting velocity for every explicit material's level set.
    registry : MaterialRegistry, optional
        Where the materials are defined. A fresh one is made if omitted;
        pass one to share definitions with another distribution (e.g. a
        :class:`~underworld3.swarm.MaterialSwarm` on a different mesh).
    name : str, optional
        Base name for the level-set variables. Defaults to ``material``,
        then ``material_1`` and so on (shared counter with the other
        distributions), so two on one mesh do not collide.
    epsilon : MeshVariable or float, optional
        Interface thickness, shared by every explicit material. Defaults
        to :func:`interface_thickness` computed once the first explicit
        material's level set exists, at ``scale=epsilon_scale``. Pass this
        explicitly (and keep it consistent with whatever value governs
        ongoing advection) rather than relying on two independent defaults
        drifting apart.
    epsilon_scale : float, default 0.35
        Scale passed to :func:`interface_thickness` when ``epsilon`` is
        left as ``None`` -- since this class builds its own level-set
        fields internally (lazily, at first build), there is no ``phi``
        of your own to call :func:`interface_thickness` on ahead of time
        with a custom scale, hence this passthrough.
        :func:`interface_thickness`'s own docstring recommends
        ``1.5-2.0`` specifically for SUPG transport (the default here,
        ``advection="supg"``) rather than its module default of ``0.35``,
        which targets SLCN.
    degree : int, default 2
        Polynomial degree of each level-set field.
    advection, reini_steps, reini_frequency, theta, conserve_mass,
    **levelset_kwargs
        Forwarded to every explicit material's own
        :class:`LevelSetSolver` (see its docstring — pure advection only;
        no ``diffusivity`` parameter exists there). Pass a single value to
        share it across all materials, or a dict keyed by material name to
        vary it per material.

    Examples
    --------
    >>> materials = uw.level_set.MaterialCLS(mesh, velocity=v.sym)
    >>> materials.add("mantle", shear_viscosity_0=1.0,   density=3300)
    >>> materials.add("slab",   shear_viscosity_0=1.0e3, density=3400)
    >>> materials["slab"] = mesh.X[1] > 0.53          # affine -> exact distance
    >>> stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    >>> stokes.materials = materials
    >>> stokes.bodyforce = -materials.density * mesh.CoordinateSystem.unit_e_1

    A curved interface needs an explicit signed distance (positive inside),
    not a boolean condition -- there is no general closed-form distance to
    an arbitrary nonlinear condition's boundary:

    >>> x, y = mesh.X
    >>> materials.add("plume", shear_viscosity_0=1.0e-1, density=3200)
    >>> materials["plume"] = 0.05 - sympy.sqrt((x - 0.5) ** 2 + (y - 0.1) ** 2)

    Advected once per step, alongside the Stokes/FreeSurface solve:

    >>> materials.solve(dt)

    See Also
    --------
    underworld3.swarm.MaterialSwarm : the same interface, on particles.
    underworld3.materials.MaterialRegions : the same interface, on mesh regions.
    underworld3.materials.MaterialRegistry : where the definitions live.
    """

    def __init__(
        self,
        mesh,
        *,
        velocity,
        registry=None,
        name: Optional[str] = None,
        epsilon=None,
        epsilon_scale: float = 0.35,
        degree: int = 2,
        advection: str = "supg",
        reini_steps: int = 1,
        reini_frequency: Optional[int] = None,
        theta: float = 0.5,
        conserve_mass: Union[bool, str] = "auto",
        **levelset_kwargs,
    ):
        # Bookkeeping BEFORE _init_distribution touches __getattr__-visible
        # state, matching MaterialSwarm's own ordering rationale.
        self._painted = set()
        self._pending = {}
        self._phi = {}
        self._cls_solvers = {}          # material name -> LevelSetSolver
        # NB: named to not collide with MaterialDistribution's OWN
        # `self._solvers`, which is an unrelated list of ATTACHED PDE
        # solvers (stokes.materials = ... appends to that one).
        self._built = False
        self._default_name = None
        self._degree = degree
        self._epsilon = epsilon
        self._epsilon_scale = epsilon_scale
        self.mesh = mesh
        # Keep the ORIGINAL MeshVariable too (not just its .sym), if one
        # was given -- see sync_velocity_from()'s docstring for why: a
        # MeshVariable's identity never changes, only its data, so writing
        # into it updates every already-built LevelSetSolver's advection
        # in place, with no rebuild -- unlike reassigning `self.velocity`
        # itself, which only ever takes effect for solvers built AFTER the
        # reassignment.
        self._velocity_variable = velocity if isinstance(velocity, discretisation.MeshVariable) else None
        self.velocity = velocity.sym if hasattr(velocity, "sym") else velocity
        self._levelset_kwargs = dict(
            advection=advection, reini_steps=reini_steps,
            reini_frequency=reini_frequency, theta=theta,
            conserve_mass=conserve_mass,
            **levelset_kwargs,
        )
        self._init_distribution(registry=registry, name=name)

    # -- level sets (MaterialDistribution's required hooks) ---------------

    @property
    def built(self):
        """Whether the level-set fields/solvers have been created yet (on
        the first ``add()``-then-read, or an explicit ``build()``/
        ``populate()``). Public so callers taking ownership of a
        not-yet-built ``MaterialCLS`` — e.g. ``FreeSurface``, which needs
        to set its own velocity on it before that happens — can check
        without reaching into a private attribute."""
        return self._built

    def _check_can_declare(self, name):
        if self._built:
            raise RuntimeError(
                f"cannot add material {name!r}: already built. Declare every "
                "material before the first region assignment or property read."
            )

    def _per_material(self, key, value, name):
        if isinstance(value, dict):
            if name not in value:
                raise KeyError(f"{key!r} dict has no entry for material {name!r}.")
            return value[name]
        return value

    def _ensure_built(self):
        if self._built:
            return
        if len(self._registry) == 0:
            raise RuntimeError(
                "no materials have been declared — call add() before reading "
                "a material property"
            )

        materials_in_order = list(self._registry.materials)
        default = materials_in_order[0]      # index 0 -- same convention as
        explicit = materials_in_order[1:]    # MaterialSwarm/MaterialRegions

        unpainted = [m.name for m in explicit if m.name not in self._painted]
        if unpainted:
            raise RuntimeError(
                f"material(s) {unpainted} were declared but never given a "
                f"region (materials[name] = ...). Only the FIRST-declared "
                f"material ({default.name!r}) may be left unpainted -- it "
                "becomes the implicit background (whatever the others don't "
                "cover), the same index-0-is-default convention "
                "MaterialSwarm/MaterialRegions use."
            )
        if default.name in self._painted:
            # Explicitly painted too -- treat as a real, independent
            # material rather than the derived background.
            default = None

        self._phi = {
            m.name: discretisation.MeshVariable(
                rf"\phi_{{{m.name}}}", self.mesh, 1, degree=self._degree, continuous=True)
            for m in materials_in_order
            if default is None or m.name != default.name
        }
        if not self._phi:
            raise RuntimeError(
                "every material was left unpainted, including the implicit "
                "default — at least one explicit region is required."
            )

        eps = self._epsilon
        if eps is None:
            eps = interface_thickness(
                self.mesh, next(iter(self._phi.values())), scale=self._epsilon_scale)

        for name, region in self._pending.items():
            phi = self._phi[name]
            sd = (region if isinstance(region, np.ndarray)
                  else uw.function.evaluate(region, phi.coords).flatten())
            initialise_psi(phi, eps, signed_distance=sd)

        for name, phi in self._phi.items():
            kwargs = {
                k: self._per_material(k, v, name)
                for k, v in self._levelset_kwargs.items()
            }
            kwargs.setdefault("epsilon", eps)
            self._cls_solvers[name] = LevelSetSolver(phi, velocity=self.velocity, **kwargs)

        self._default_name = default.name if default is not None else None
        self._pending = {}
        self._built = True

    def _level_sets(self):
        self._ensure_built()
        order = [m.name for m in self._registry.materials]
        others = [self._phi[n] for n in order if n != self._default_name]
        return [
            _DerivedLevelSet(others) if n == self._default_name else self._phi[n]
            for n in order
        ]

    # -- painting -----------------------------------------------------------

    def __setitem__(self, name, region):
        """Give ``name`` a region -- see the class docstring for what
        ``region`` may be (an affine half-space condition, a signed-
        distance expression, or a precomputed array)."""
        if name not in {m.name for m in self._registry.materials}:
            raise KeyError(f"material {name!r} was not declared — call add({name!r}, ...) first")
        if self._built:
            raise RuntimeError(
                "cannot paint a region: already built. Region painting must "
                "happen before the first read (same rule as add())."
            )
        self._painted.add(name)
        self._pending[name] = self._normalise_region(region)

    def _normalise_region(self, region):
        r"""Accept, in order of preference:

        - a precomputed numpy signed-distance array (used as-is; must match
          the eventual ``phi.coords`` ordering for this material);
        - a sympy ``Relational`` half-space condition
          (``mesh.X[1] > 0.53``-style) that is AFFINE (linear) in the mesh
          coordinates -- converted automatically to its EXACT signed
          distance: :math:`(\ell - r)/\|\nabla(\ell-r)\|` for
          :math:`\ell > r` (sign flipped for :math:`\ell < r`), which is
          exact for any linear :math:`\ell - r` since a hyperplane's
          distance field is itself linear. This is the direct level-set
          analogue of ``materials["slab"] = mesh.X[1] > 0.53`` from
          ``MaterialSwarm`` -- same syntax, exact result for this common
          case;
        - any other sympy expression, taken to ALREADY be a signed
          distance (positive inside), e.g. ``r - sympy.sqrt((x-x0)**2 +
          (y-y0)**2)`` for a circular inclusion.

        A NON-affine boolean condition (curved or compound, e.g.
        ``(mesh.X[1] > 0.53) & (mesh.X[0] < 0.2)``) is refused rather than
        silently approximated: there is no general closed-form distance to
        an arbitrary condition's boundary, and a crude "inside=+1,
        outside=-1" substitute would give the tanh profile no graded
        transition zone at all (see the module docstring's warning about
        exactly this trap). Pass a signed-distance expression directly
        instead.
        """
        if isinstance(region, np.ndarray):
            return region
        if isinstance(region, sympy.core.relational.Relational):
            return self._affine_signed_distance(region)
        if isinstance(region, sympy.Basic):
            return region
        raise TypeError(
            "a region must be a signed-distance sympy expression (positive "
            "inside), an affine half-space condition on the mesh coordinates "
            "(e.g. mesh.X[1] > 0.53), or a precomputed numpy signed-distance "
            f"array — not {type(region).__name__}"
        )

    def _affine_signed_distance(self, condition):
        lhs, rhs = condition.lhs, condition.rhs
        if isinstance(condition, (sympy.StrictGreaterThan, sympy.GreaterThan)):
            signed = lhs - rhs
        elif isinstance(condition, (sympy.StrictLessThan, sympy.LessThan)):
            signed = rhs - lhs
        else:
            raise TypeError(
                f"unsupported condition type {type(condition).__name__} for "
                "automatic signed-distance conversion"
            )
        coords = list(self.mesh.X)
        expanded = sympy.expand(signed)
        if not expanded.is_polynomial(*coords) or sympy.Poly(expanded, *coords).total_degree() > 1:
            raise TypeError(
                "automatic signed-distance conversion only supports a LINEAR "
                "(half-space) condition in the mesh coordinates, e.g. "
                "mesh.X[1] > 0.53 — for a curved or compound interface, pass "
                "a signed-distance expression directly (positive inside), "
                "e.g. `r - sympy.sqrt((x-x0)**2+(y-y0)**2)`."
            )
        grad = [sympy.diff(signed, c) for c in coords]
        norm = sympy.sqrt(sum(g ** 2 for g in grad))
        return signed / norm

    # -- transport (the CLS-specific step; MaterialSwarm's analogue is
    # its own .advection()) -------------------------------------------------

    def solve(self, dt: float, *, reinitialise: bool = True) -> None:
        """Advance every explicit material's level set by one step of size
        ``dt``. The implicit default material needs no step of its own —
        it is always exactly the complement of the others."""
        self._ensure_built()
        for solver in self._cls_solvers.values():
            solver.solve(dt, reinitialise=reinitialise)

    def estimate_dt(self, **kwargs) -> float:
        """The most restrictive of the explicit materials' own transport
        solvers' timestep estimates.

        ``basis=`` (only meaningful for a material using
        ``advection="supg"`` — its ``LevelSetSolver`` forwards to
        ``AdvDiffusion.estimate_dt``, which has a ``basis`` concept;
        ``"slcn"``'s ``estimate_dt`` does not and would raise ``TypeError``
        if handed one) is applied ONLY to materials that are actually
        SUPG-advected — this class allows mixed per-material advection
        schemes (see ``__init__``'s per-material dict kwargs), so a
        uniform forward would break the moment any material uses "slcn".
        Every other kwarg is forwarded to all materials unchanged.
        """
        self._ensure_built()
        basis = kwargs.pop("basis", None)
        estimates = []
        for solver in self._cls_solvers.values():
            call_kwargs = dict(kwargs)
            if basis is not None and solver.advection == "supg":
                call_kwargs["basis"] = basis
            estimates.append(solver.estimate_dt(**call_kwargs))
        return min(estimates)

    def sync_velocity_from(self, other) -> None:
        r"""Copy ``other``'s CURRENT DATA into this distribution's own
        dedicated velocity ``MeshVariable`` (whatever was passed as
        ``velocity=`` at construction) — for updating the ADVECTING
        velocity's VALUES in place, without needing to rebuild any
        already-built :class:`LevelSetSolver`.

        Why this works when reassigning :attr:`velocity` itself doesn't,
        once :attr:`built` is True: each ``LevelSetSolver``'s compiled
        weak form references the velocity variable's SYMBOL — a symbol
        whose IDENTITY never changes here, only its underlying data does.
        Reassigning ``self.velocity`` to a NEW symbolic expression only
        takes effect for solvers built AFTER the reassignment (there is no
        such thing as "already built" solvers retroactively noticing a
        Python attribute changed); writing new VALUES into the SAME
        variable those solvers already reference has no such problem —
        PETSc reads a variable's current data at solve time regardless of
        when it was last written.

        Requires ``velocity=`` to have been given as a genuine
        ``MeshVariable`` at construction (not a bare sympy expression,
        e.g. ``v.sym``) — otherwise there is no fixed-identity field to
        write updated values into, and this raises. This is exactly the
        fix for the case documented on ``FreeSurface``'s ``composition``
        parameter: construct ``MaterialCLS`` with a dedicated
        ``MeshVariable`` (not ``v.sym`` directly) as ``velocity=`` so
        ``FreeSurface`` can keep it in sync with its own consistent
        surface velocity every step, even though the materials were
        necessarily already built by the time ``FreeSurface`` sees them.

        Parameters
        ----------
        other : MeshVariable
            Must share the same discretisation (mesh, degree, number of
            components) as this distribution's own velocity variable —
            typically the case when syncing from another velocity field on
            the SAME mesh (e.g. ``FreeSurface``'s
            ``self.consistent.u``). A mismatch surfaces as numpy's own
            broadcasting error on the array assignment below.
        """
        if self._velocity_variable is None:
            raise RuntimeError(
                "sync_velocity_from() needs `velocity=` to have been a "
                "genuine MeshVariable at construction (not a bare sympy "
                "expression, e.g. v.sym) -- there is no dedicated field to "
                "write updated values into. Construct this MaterialCLS with "
                "a MeshVariable instead, e.g. `velocity=v_adv` (not "
                "`velocity=v_adv.sym`)."
            )
        with self.mesh.access(self._velocity_variable):
            self._velocity_variable.array[...] = other.array

    def level_set(self, name):
        """The underlying level-set MeshVariable for an explicit material —
        the machinery, exposed for cases this interface does not cover
        (writing it to a checkpoint, plotting it directly), the same escape
        hatch ``MaterialSwarm.index`` is for the particle case. The implicit
        default material has none of its own (it's the complement of the
        others, recomputed on read — see the class docstring) and raises."""
        self._ensure_built()
        if name == self._default_name:
            raise KeyError(
                f"{name!r} is the implicit default material and has no "
                "level-set field of its own — it's always the complement of "
                "the explicit materials, not a stored field. See "
                "MaterialCLS's class docstring."
            )
        try:
            return self._phi[name]
        except KeyError:
            raise KeyError(f"no explicit material named {name!r}") from None

    def interface_volumes(self):
        """Per-explicit-material enclosed volume/area (see
        ``LevelSetSolver.interface_volume``), as a ``{name: volume}`` dict.
        Excludes the implicit default material."""
        self._ensure_built()
        return {name: s.interface_volume() for name, s in self._cls_solvers.items()}

    def total_volume_error(self) -> float:
        """``max_x |sum_i max(phi_i(x), 0) - 1|`` sampled at each explicit
        material's own level-set nodes — a diagnostic for how far the
        independently advected/reinitialised fields have drifted from
        partition-of-unity before renormalisation (`blend()`/attribute
        access always sums to exactly 1 by construction regardless)."""
        self._ensure_built()
        raw = [np.clip(np.asarray(phi.array[:, 0, 0]), 0, None)
               for phi in self._phi.values()]
        total = sum(raw)
        if self._default_name is not None:
            total = total + np.clip(1 - total, 0, None)
        return float(np.max(np.abs(total - 1)))

    # -- mixing / blend: extend the base class's arithmetic/harmonic with
    # "geometric" ------------------------------------------------------------

    _MIXING_RULES_CLS = ("arithmetic", "harmonic", "geometric")

    def mixing(self, **rules):
        """Same as ``MaterialDistribution.mixing()``, extended to also
        accept ``"geometric"`` — see the module docstring for why that
        matters for a viscosity-like property here."""
        declared = self._registry.declared_properties()
        for name, rule in rules.items():
            name = _property_name(name)
            if rule not in self._MIXING_RULES_CLS:
                raise ValueError(
                    f"mixing rule for {name!r} must be one of "
                    f"{self._MIXING_RULES_CLS}, not {rule!r}"
                )
            if name not in declared:
                raise KeyError(
                    f"no material declares {name!r}, so a mixing rule for it "
                    f"would have no effect; declared: {sorted(declared)}"
                )
            self._mixing_rules[name] = rule
        self._push_all()
        return self

    def _weights(self, clip: bool = True):
        r"""Renormalised partition-of-unity weights for every material, in
        registry order — :math:`w_i = \max(\varphi_i,0) /
        (\sum_j \max(\varphi_j,0) + \delta)`.

        NEEDED, not just a nicety: ``MaterialDistribution.blend()`` (the
        base class) does a raw weighted sum straight off ``masks[i].sym[0]``
        with no renormalisation, which is fine for ``MaterialSwarm``/
        ``MaterialRegions`` (their masks are already exactly one-hot or a
        well-formed partition) but WRONG here — independently advected/
        reinitialised level sets genuinely drift apart from summing to 1
        (see :meth:`total_volume_error`), and an un-renormalised blend
        would silently stop being a true weighted average as soon as they
        do. :meth:`blend` below is therefore a full override, not a
        geometric-only addition delegating elsewhere for the other two
        rules — it must renormalise for "arithmetic"/"harmonic" too, not
        just "geometric".
        """
        self._ensure_built()
        masks = self._level_sets()
        raw = [sympy.Max(m.sym[0], 0) if clip else m.sym[0] for m in masks]
        total = sum(raw) + 1.0e-12
        return [r / total for r in raw]

    def weight(self, name):
        """The renormalised partition-of-unity weight for ONE material
        (see :meth:`_weights`) — for when a formula needs one material's
        own fraction directly (e.g. a body-force term keyed to a single
        material), not a property blended across all of them."""
        self._ensure_built()
        order = [m.name for m in self._registry.materials]
        if name not in order:
            raise KeyError(f"no material named {name!r}")
        return self._weights()[order.index(name)]

    def blend(self, name, mixing=None):
        r"""Same as ``MaterialDistribution.blend()``, but (a) always
        renormalises the masks first (see :meth:`_weights` — REQUIRED
        here, not optional, unlike the base class) and (b) additionally
        supports ``mixing="geometric"``:
        :math:`\exp(\sum_i w_i \log v_i)` — the log-mean, the usual choice
        for a property spanning orders of magnitude.
        """
        name = _property_name(name)
        missing = [m.name for m in self._registry.materials
                   if name not in m.properties]
        if missing:
            raise KeyError(
                f"property {name!r} is not declared by {missing}. Every "
                "material must declare a property that is blended."
            )
        self._ensure_built()
        self._read_properties.add(name)
        rule = mixing or self._mixing_rules.get(name, "arithmetic")
        if rule not in self._MIXING_RULES_CLS:
            raise ValueError(
                f"mixing rule for {name!r} must be one of "
                f"{self._MIXING_RULES_CLS}, not {rule!r}"
            )

        values = [m.resolved(name) for m in self._registry.materials]
        w = self._weights()

        if rule == "arithmetic":
            return sum(wi * vi for wi, vi in zip(w, values))
        if rule == "geometric":
            return sympy.exp(sum(wi * sympy.log(vi) for wi, vi in zip(w, values)))

        # harmonic — same zero-value guard as the base class, for parity:
        # a material whose value is zero poisons the symbol at declaration
        # time (1/0 -> ComplexInfinity, a bare C-printer traceback later).
        zeros = [m.name for m, value in zip(self._registry.materials, values)
                 if getattr(value, "is_zero", value == 0)]
        if zeros:
            raise ValueError(
                f"harmonic mixing of {name!r} divides by its value in "
                f"{zeros}, which is zero. Use arithmetic mixing, or give it "
                "a small non-zero value."
            )
        return 1 / sum(wi / vi for wi, vi in zip(w, values))
