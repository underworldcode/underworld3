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
(:class:`~underworld3.systems.AdvDiffusionSUPG`) by default and the
semi-Lagrangian solver on request. The reinitialisation equation is that of
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
            self._adv_solver = systems.AdvDiffusionSUPG(
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

        # |grad psi| for the reinitialisation, projected onto the mesh
        self._grad_magnitude = sympy.sqrt(sum(g ** 2 for g in self.mesh.vector.gradient(self.phi.sym[0])))
        self.phi_grad = discretisation.MeshVariable(
            r"|\nabla\psi|", self.mesh, 1, degree=self.phi.degree, continuous=self.phi.continuous)
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
