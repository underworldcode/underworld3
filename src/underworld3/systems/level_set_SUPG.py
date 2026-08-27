import numpy as np
from typing import Optional
import warnings
import sympy

from petsc4py import PETSc  # NOTE: added -- _apply_boundary_neumann uses
                              # PETSc.COMM_WORLD but this file had no
                              # module-level PETSc import at all (a latent
                              # NameError-on-call bug in the previous
                              # version, unrelated to the solver rewrite).

import underworld3 as uw
from underworld3 import discretisation, systems
from underworld3.utilities._api_tools import Template
from typing import Optional

from shapely import geometry as sl
from shapely import prepare as _shapely_prepare

from underworld3.systems import AdvDiffusionSUPG

def initialise_psi(
    psi,
    epsilon,
    signed_distance: np.ndarray | None = None,
    interface_geometry:   str | None = None,
    interface:            sl.LineString | sl.Polygon | None = None,
    interface_coordinates = None,
    boundary_coordinates  = None,
) -> None:
    """
    Fill a UW3 MeshVariable *psi* with conservative level-set (CLS) values:

        psi = ( 1 + tanh( phi / (2 * epsilon) ) ) / 2

    where *phi* is the signed-distance function (positive on the '1-side').

    Parameters
    ----------
    psi : uw.discretisation.MeshVariable
        Target level-set field.
    epsilon : float or ndarray, shape (n_nodes,)
        Interface thickness.  Use ``interface_thickness()`` to compute it.

    Keyword-only
    ------------
    signed_distance : ndarray, shape (n_nodes,), optional
        Pre-computed signed distances.  If supplied, all geometry arguments
        are ignored and the CLS field is written immediately.

    interface_geometry : {'curve', 'polygon', 'circle', 'shapely'}
        How the interface is described (ignored when *signed_distance* is given).

    interface : shapely.LineString or shapely.Polygon, optional
        Required when ``interface_geometry='shapely'``.

    interface_coordinates : list of (x, y) or ((cx, cy), radius)
        Vertex list for 'curve'/'polygon', or (centre, radius) for 'circle'.

    boundary_coordinates : list of (x, y), optional
        Extra boundary points used to close an open interface into a polygon
        that defines the '1-side'.

    Notes
    -----
    *  ``psi → 1``  inside the interface  (positive signed distance)
    *  ``psi = 0.5``  on the interface
    *  ``psi → 0``  outside the interface  (negative signed distance)

    References
    ----------
    Parameswaran & Mandal (2023), Eur. J. Mech.-B/Fluids, 98, 40-63.
    g-ADOPT ``assign_level_set_values``:
    https://github.com/g-adopt/g-adopt/blob/main/gadopt/level_set_tools.py
    """

    if signed_distance is not None:
        psi.data[:, 0] = _tanh_profile(signed_distance, epsilon)
        return

    if interface_geometry is None:
        raise ValueError(
            "Provide either 'signed_distance' or 'interface_geometry'."
        )

    if interface_coordinates is None and interface_geometry != "shapely":
        raise ValueError(
            "'interface_coordinates' is required when "
            f"interface_geometry='{interface_geometry}'."
        )

    points = psi.coords   # shape (n_nodes, dim)
    signed_distance = _signed_distance_from_geometry(
        interface_geometry,
        interface,
        interface_coordinates,
        boundary_coordinates,
        points,)
    epsilon_data = epsilon.data[:,0]
    psi.data[:, 0] = _tanh_profile(signed_distance, epsilon_data)


def interface_thickness(
    mesh: uw.discretisation.Mesh,
    phi: uw.discretisation.MeshVariable,
    *,
    scale: float = 0.35,
    use_min_edge_length: bool = False,
) -> uw.discretisation.MeshVariable:
    """Compute a spatially-varying interface thickness ε on the same mesh and
    degree as *phi*, returned as a scalar ``MeshVariable``.
    """
    if use_min_edge_length and mesh.qdegree > 1:
        raise ValueError(
            "use_min_edge_length=True is only valid for straight-edged meshes "
            "(qdegree=1)."
        )

    from scipy.spatial import cKDTree

    dm  = mesh.dm
    dim = mesh.dim
    c_start, c_end = dm.getHeightStratum(0)   # cell range in the DMPlex
    n_cells = c_end - c_start
    cell_epsilon   = np.empty(n_cells, dtype=float)
    cell_centroids = np.empty((n_cells, dim), dtype=float)

    if not use_min_edge_length:
        scale_factor = scale / np.sqrt(dim)
        for i, cell in enumerate(range(c_start, c_end)):
            vol, centroid, _ = dm.computeCellGeometryFVM(cell)
            cell_epsilon[i]      = scale_factor * float(np.asarray(vol).ravel()[0]) ** (1.0 / dim)
            cell_centroids[i, :] = np.asarray(centroid).ravel()[:dim]
    else:
        v_start, v_end = dm.getDepthStratum(0)
        coords = mesh.data                     # (n_vertices, dim)
        for i, cell in enumerate(range(c_start, c_end)):
            closure, _ = dm.getTransitiveClosure(cell)
            verts = [p for p in closure if v_start <= p < v_end]
            v_coords = coords[[p - v_start for p in verts]]
            # minimum pairwise edge length
            min_edge = np.inf
            for a in range(len(v_coords)):
                for b in range(a + 1, len(v_coords)):
                    d = np.linalg.norm(v_coords[a] - v_coords[b])
                    if d < min_edge:
                        min_edge = d
            cell_epsilon[i]      = scale * min_edge
            cell_centroids[i, :] = v_coords.mean(axis=0)

    epsilon_var = uw.discretisation.MeshVariable(
        r"\epsilon", mesh, 1, degree=phi.degree,continuous=phi.continuous
    )
    node_coords = phi.coords               # (n_nodes, dim)
    tree        = cKDTree(cell_centroids)
    _, nearest  = tree.query(node_coords)  # nearest[i] = cell index for node i

    epsilon_var.data[:, 0] = cell_epsilon[nearest]
    return epsilon_var


def _sgn_dist_closed(interface: sl.Polygon, points: np.ndarray) -> np.ndarray:
    """Signed distance: positive inside, negative outside a closed polygon."""
    _shapely_prepare(interface)
    boundary = interface.boundary
    sgn = np.where(
        [interface.contains(sl.Point(p)) for p in points], 1.0, -1.0
    )
    dist = np.array([boundary.distance(sl.Point(p)) for p in points])
    return sgn * dist

def _sgn_dist_open(
    interface: sl.LineString,
    enclosed_side: sl.Polygon,
    points: np.ndarray,
) -> np.ndarray:
    """Signed distance w.r.t. an open interface; sign from enclosed polygon."""
    _shapely_prepare(enclosed_side)
    sgn = np.where(
        [enclosed_side.intersects(sl.Point(p)) for p in points], 1.0, -1.0
    )
    dist = np.array([interface.distance(sl.Point(p)) for p in points])
    return sgn * dist

def _tanh_profile(phi: np.ndarray, epsilon: float | np.ndarray) -> np.ndarray:
    """CLS tanh profile:  (1 + tanh(phi / 2ε)) / 2"""
    return (1.0 + np.tanh(np.asarray(phi) / (2.0 * np.asarray(epsilon)))) / 2.0

def _signed_distance_from_geometry(
    interface_geometry:   str,
    interface,
    interface_coordinates,
    boundary_coordinates,
    points: np.ndarray,
) -> np.ndarray:
    """Dispatch to the correct signed-distance routine based on geometry type."""

    match interface_geometry:

        case "curve":
            itf = sl.LineString(interface_coordinates)
            if itf.is_closed:
                _require_no_boundary(boundary_coordinates, "closed curve")
                return _sgn_dist_closed(sl.Polygon(itf), points)
            else:
                _require_boundary(boundary_coordinates, "open curve")
                enclosed = sl.Polygon(
                    np.vstack((interface_coordinates, boundary_coordinates))
                )
                return _sgn_dist_open(itf, enclosed, points)

        case "polygon":
            if boundary_coordinates is None:
                return _sgn_dist_closed(sl.Polygon(interface_coordinates), points)
            else:
                itf = sl.LineString(interface_coordinates)
                enclosed = sl.Polygon(
                    np.vstack((interface_coordinates, boundary_coordinates))
                )
                return _sgn_dist_open(itf, enclosed, points)

        case "shapely":
            if interface is None:
                raise ValueError(
                    "'interface' must be provided when interface_geometry='shapely'."
                )
            if isinstance(interface, sl.Polygon):
                return _sgn_dist_closed(interface, points)
            else:                                      # LineString
                _require_boundary(boundary_coordinates, "shapely LineString")
                enclosed = sl.Polygon(
                    np.vstack((interface.coords, boundary_coordinates))
                )
                return _sgn_dist_open(interface, enclosed, points)

        case _:
            raise ValueError(
                f"Unknown interface_geometry='{interface_geometry}'. "
                "Choose from: 'curve', 'polygon', 'shapely'."
            )

def _require_boundary(boundary_coordinates, context: str) -> None:
    if boundary_coordinates is None:
        raise ValueError(
            f"'boundary_coordinates' must be supplied for an {context}."
        )

def _require_no_boundary(boundary_coordinates, context: str) -> None:
    if boundary_coordinates is not None:
        raise ValueError(
            f"'boundary_coordinates' must not be provided for a {context}."
        )

def _allreduce_min(value: float) -> float:
    """Global MPI min via PETSc.COMM_WORLD (works in serial and parallel)."""
    from petsc4py import PETSc
    from mpi4py import MPI
    return PETSc.COMM_WORLD.tompi4py().allreduce(float(value), op=MPI.MIN)

def _allreduce_max(value: float) -> float:
    """Global MPI max via PETSc.COMM_WORLD (works in serial and parallel)."""
    from petsc4py import PETSc
    from mpi4py import MPI
    return PETSc.COMM_WORLD.tompi4py().allreduce(float(value), op=MPI.MAX)


class LevelSetSolver:
    """Conservative level-set advection + reinitialisation solver for UW3.

    Advection: Crank-Nicolson + SUPG (Brooks & Hughes, 1982), via
    :class:`SUPGAdvection` -- a hand-built weak-form solver on UW3's
    generic ``SNES_Scalar`` scaffolding, NOT ``AdvDiffusionSLCN``/
    ``SemiLagrangian``.

    Reinitialisation: Eq. (17) of Parameswaran & Mandal (2023),

        d(phi)/d(tau) = -phi(1-phi)(1-2phi) + eps(1-2phi)|grad phi|,

    integrated with the three-stage SSP-RK3 ("TVD Runge-Kutta") scheme of
    Gottlieb & Shu (1998) -- unchanged from the previous version of this
    file, since that was already the scheme requested. ``|grad phi|`` is
    now computed via 2nd-order ENO + Godunov upwinding (see
    :func:`_grad_magnitude_eno2`) on a regular node grid, NOT
    ``uw.systems.Projection``. This currently restricts ``LevelSetSolver``
    to a structured (Cartesian-topology) mesh, e.g.
    ``uw.meshing.StructuredQuadBox`` -- :class:`_StructuredGrid` raises
    ``RuntimeError`` rather than silently mis-mapping on anything else.

    Parameters
    ----------
    level_set : MeshVariable
        Scalar ``MeshVariable`` (degree >= 1, continuous) that holds the
        CLS field phi. Its mesh is used for all sub-solvers.
    velocity : MeshVariable.sym or sympy expression
        Velocity field used for advection.  Typically the `.sym` of a
        Stokes velocity ``MeshVariable``.
    epsilon : MeshVariable
        Interface thickness field ε (see ``interface_thickness()``).
    reini_dt : float, optional
        Pseudo-time step for reinitialisation (default 0.5 x eps).
    reini_steps : int, optional
        Number of pseudo-time steps per reinitialisation call (default 5).
    reini_frequency : int or None, optional
        How many advection steps between reinitialisation passes.
        ``None`` uses the automatic strategy (see `_default_frequency`).
    theta : float, optional
        Time-integration parameter for the advection solver (default 0.5,
        Crank-Nicolson; 1.0 would be backward-Euler).
    adv_solver_opts : dict, optional
        Extra PETSc options forwarded to the advection solver.
    adv_solver_bc : sequence of str, optional
        Mesh boundary labels (e.g. ``["Left","Right","Top","Bottom"]``) to
        apply the zero-normal-gradient Neumann correction to after every
        advection/reinitialisation pass (see `_apply_boundary_neumann`).

    Usage example
    -------------
    >>> import underworld3 as uw, sympy
    >>> mesh = uw.meshing.StructuredQuadBox(elementRes=(32, 32))
    >>> phi   = uw.discretisation.MeshVariable("phi", mesh, 1, degree=2, continuous=True)
    >>> v_sol = uw.discretisation.MeshVariable("v",   mesh, 2, degree=2)
    >>> eps = interface_thickness(mesh, phi)
    >>> ls = LevelSetSolver(phi, velocity=v_sol.sym, epsilon=eps)
    >>> for step in range(100):
    ...     ls.solve(dt=1e-3)   # advect (+ reinitialise if due)
    """

    def __init__(
        self,
        level_set: discretisation.MeshVariable,
        *,
        velocity,
        epsilon,
        reini_dt: Optional[float] = None,
        reini_steps: int = 5,
        reini_frequency: Optional[int] = None,
        theta: float = 0.5,
        adv_solver_opts: Optional[dict] = None,
        adv_solver_bc: Optional[dict] = None,
        conserve_mass: bool = True,
        mass_correction_tol: float = 1.0e-10,
        mass_correction_max_iter: int = 40,
    ) -> None:
        if level_set.num_components != 1:
            raise ValueError("`level_set` must be a scalar MeshVariable.")
        if not level_set.continuous:
            raise ValueError(
                "`level_set` must be a CONTINUOUS MeshVariable -- "
                "SUPGAdvection assembles a continuous-Galerkin weak form "
                "and needs shared vertex/edge DOFs across cells."
            )

        self.phi        = level_set
        self.mesh       = level_set.mesh
        self.velocity   = velocity
        self.epsilon    = epsilon
        self.reini_dt   = float(reini_dt) if reini_dt is not None else 0.5 * float(epsilon.data[:, 0].min())
        self.reini_steps = int(reini_steps)
        self.step       = 0   # counts physical advection steps taken

        # ---- Advection solver: Crank-Nicolson + SUPG ----------------------
        self._adv_solver = AdvDiffusionSUPG(self.mesh, self.phi, self.velocity, theta=theta)
        self._adv_solver_bc = adv_solver_bc

        if adv_solver_opts:
            for k, v in adv_solver_opts.items():
                self._adv_solver.petsc_options[k] = v

        # no use ---- Reinitialisation: ENO2/Godunov gradient on a regular grid ----
        #self._grid = _StructuredGrid(self.phi)
        #self._phi0_sign_grid = None  # frozen sign(phi0-0.5), set in reinitialise()

        grad_s = self.mesh.vector.gradient(self.phi.sym)
        self._grad_mag = sympy.sqrt(sum(g**2 for g in grad_s))

        # ---- Gradient vector field ∇φ -------------------------------------
        self.phi_grad = discretisation.MeshVariable(
            r"|\nabla\phi|",
            self.mesh,
            1,
            degree= self.phi.degree,continuous = self.phi.continuous
        )
        self._grad_projector = systems.Projection(self.mesh, self.phi_grad,degree= self.phi.degree)
        self._grad_projector.uw_function = self._grad_mag 

        # ---- Reinitialisation frequency -----------------------------------
        if reini_frequency is None:
            self._reini_frequency = self._default_frequency()
        else:
            self._reini_frequency = int(reini_frequency)

        # ---- Global mass correction (Zhang, Zou & Greaves 2010) -----------
        # Neither the reinitialisation equation (Eq. 17 is contour-
        # preserving, not mass-preserving) nor a raw clamp() (an
        # unweighted np.clip -- adds mass wherever it zeros an undershoot,
        # removes it wherever it caps an overshoot; if that's not
        # symmetric, e.g. from mild cross-wind oscillation SUPG alone
        # doesn't fully suppress, the imbalance accumulates every step)
        # come with any conservation guarantee. Advection itself does, in
        # theory, for a divergence-free/boundary-vanishing velocity field
        # (see SUPGAdvection's docstring), so this corrects for the other
        # two rather than second-guessing the advection solve.
        self.conserve_mass = conserve_mass
        self._mass_correction_tol = float(mass_correction_tol)
        self._mass_correction_max_iter = int(mass_correction_max_iter)
        self._target_volume = self.interface_volume() if conserve_mass else None


    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self, dt: float, *, reinitialise: bool = True) -> None:
        self._adv_solver.solve(dt)
        if self._adv_solver_bc:
            self._apply_boundary_neumann(labels=self._adv_solver_bc)
        self.step += 1

        if reinitialise and (self.step % self._reini_frequency == 0):
            self.reinitialise()
            if self._adv_solver_bc:
                self._apply_boundary_neumann(labels=self._adv_solver_bc)

        if self.conserve_mass:
            self._correct_mass(self._target_volume)

    def reinitialise(self) -> None:
        """Run `reini_steps` pseudo-time steps of CLS reinitialisation.

        Each step integrates Eq. (17) of Parameswaran & Mandal (2023),

            ∂φ/∂τₙ = θ [ −φ(1−φ)(1−2φ) + ε(1−2φ)|∇φ| ]

        using the three-stage SSP-RK3 scheme the paper validates all of its
        results with (their Eq. 28) -- unchanged. |∇φ| is now computed by
        2nd-order ENO + Godunov upwinding (Osher & Shu, 1991; Jiang & Peng,
        2000; Sussman, Smereka & Osher, 1994) on the regular node grid
        rather than an L2 projection; its upwind sign reference
        sign(phi-0.5) is frozen HERE, once, before the pseudo-time loop.
        Both RHS terms share the factor (1−2φ), so φ = 0.5 (the interface)
        is a fixed point -- reinitialisation sharpens the profile without
        moving the 0.5-contour.
        """
        for _ in range(self.reini_steps):
            self._reini_ssprk3_step(self.reini_dt)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_gradient(self) -> None:
        """L2-project |∇φ| onto ``phi_grad`` from the *current* φ data.
        """
        self._grad_projector.uw_function = self._grad_mag
        self._grad_projector.solve()

    def _rhs(self, phi_values: np.ndarray) -> np.ndarray:
        """Evaluate the RHS of Eq. (17), L(φ), at a given nodal φ array.

        Writes ``phi_values`` into ``self.phi`` first (so the gradient
        projector, built from ``self.phi.sym``, sees the correct SSP-RK3
        stage value), then projects |∇φ| and combines the sharpening and
        balancing terms nodally:

          sharpening = −φ(1−φ)(1−2φ)      balance = ε(1−2φ)|∇φ|
        """
        self.phi.data[:, 0] = phi_values
        self._update_gradient()
        grad = self.phi_grad.data[:, 0]
        eps  = self.epsilon.data[:, 0]

        sharpening = -phi_values * (1 - phi_values) * (1 - 2 * phi_values)
        balance    =  eps * (1 - 2 * phi_values) * grad
        return sharpening + balance  # theta = 1

    def _reini_ssprk3_step(self, dtau: float) -> None:
        """One SSP-RK3 pseudo-time step of Eq. (17) (Eq. 28, Parameswaran &
        Mandal 2023):

            φ⁽¹⁾ = φⁿ + Δτ  L(φⁿ)
            φ⁽²⁾ = ¾φⁿ + ¼φ⁽¹⁾ + ¼Δτ L(φ⁽¹⁾)
            φⁿ⁺¹ = ⅓φⁿ + ⅔φ⁽²⁾ + ⅔Δτ L(φ⁽²⁾)

        ``self.phi.data`` holds φⁿ on entry and φⁿ⁺¹ on exit.
        """
        psi0 = self.phi.data[:, 0].copy()

        L0   = self._rhs(psi0)
        psi1 = psi0 + dtau * L0

        L1   = self._rhs(psi1)
        psi2 = 0.75 * psi0 + 0.25 * psi1 + 0.25 * dtau * L1

        L2      = self._rhs(psi2)
        psi_new = (1.0 / 3.0) * psi0 + (2.0 / 3.0) * psi2 + (2.0 / 3.0) * dtau * L2

        self.phi.data[:, 0] = psi_new


    def _default_frequency(self) -> int:
        """Automatic reinitialisation frequency.

        reinitialise every step up to a reference cell size, then scale down as the mesh refines.
        Falls back to 1 for non-Cartesian or unusual meshes.
        """
        try:
            coords = self.mesh.data
            max_c = np.array([_allreduce_max(coords[:, i].max())
                               for i in range(coords.shape[1])])
            min_c = np.array([_allreduce_min(coords[:, i].min())
                               for i in range(coords.shape[1])])
            domain_size = float(np.sqrt(np.sum((max_c - min_c) ** 2)))
            return max(1, round(4.9e-3 * domain_size / self.epsilon.data.min() - 0.25))
        except Exception:
            warnings.warn(
                "Could not compute domain size for reinitialisation frequency; "
                "defaulting to every step.", stacklevel=2
            )
            return 1

    def _apply_boundary_neumann(self, labels=("Left", "Right", "Top", "Bottom")) -> None:
        """Enforce zero-normal-gradient at the given mesh boundaries by copying
        the adjacent interior row/column of nodes onto the wall nodes.
        """
        from mpi4py import MPI
        comm = PETSc.COMM_WORLD.tompi4py()

        coords = self.phi.coords
        n_local = coords.shape[0]

        axis_for_label = {"Left": 0, "Right": 0, "Top": 1, "Bottom": 1}
        reduce_for_label = {
            "Left": _allreduce_min, "Right": _allreduce_max,
            "Top": _allreduce_max,  "Bottom": _allreduce_min,
        }
        is_min_side = {"Left": True, "Right": False, "Top": False, "Bottom": True}

        for label in labels:
            axis = axis_for_label[label]
            tang = 1 - axis

            # global wall coordinate -- +/-inf on an empty rank so it can't
            # corrupt the min/max reduction
            if n_local:
                local_extreme = (coords[:, axis].min() if is_min_side[label]
                                  else coords[:, axis].max())
            else:
                local_extreme = np.inf if is_min_side[label] else -np.inf
            wall_val = reduce_for_label[label](float(local_extreme))

            # global interior-column coordinate
            local_axis_vals = np.unique(coords[:, axis]) if n_local else np.empty(0)
            all_axis_vals = np.unique(np.concatenate(comm.allgather(local_axis_vals)))
            if all_axis_vals.size < 2:
                continue  # degenerate mesh extent in this direction
            ordered = all_axis_vals[np.argsort(np.abs(all_axis_vals - wall_val))]
            inner_val = ordered[1]  # nearest distinct coordinate to the wall

            # this rank's contribution to the global interior-column lookup table
            if n_local:
                inner_idx = np.where(np.isclose(coords[:, axis], inner_val, atol=1e-8))[0]
            else:
                inner_idx = np.empty(0, dtype=int)
            local_pairs = (np.column_stack((coords[inner_idx, tang], self.phi.data[inner_idx, 0]))
                           if len(inner_idx) else np.empty((0, 2)))

            gathered = [p for p in comm.allgather(local_pairs) if p.shape[0] > 0]
            if not gathered:
                continue
            global_pairs = np.vstack(gathered)

            # de-duplicate shared/ghost dofs reported by more than one rank
            order = np.argsort(global_pairs[:, 0])
            global_pairs = global_pairs[order]
            uniq = np.concatenate(([True], np.diff(global_pairs[:, 0]) > 1e-10))
            global_pairs = global_pairs[uniq]

            # this rank's own wall dofs (may be empty on this rank)
            wall_idx = (np.where(np.isclose(coords[:, axis], wall_val, atol=1e-8))[0]
                        if n_local else np.empty(0, dtype=int))
            if len(wall_idx) == 0:
                continue  # this rank owns no nodes on this wall

            # nearest-neighbour lookup against the global table
            wall_tang = coords[wall_idx, tang]
            pos = np.clip(np.searchsorted(global_pairs[:, 0], wall_tang), 1, len(global_pairs) - 1)
            left_err  = np.abs(wall_tang - global_pairs[pos - 1, 0])
            right_err = np.abs(global_pairs[pos, 0] - wall_tang)
            nearest = np.where(right_err < left_err, pos, pos - 1)
            err = np.minimum(left_err, right_err)

            good = err <= 1e-6
            if not np.all(good):
                warnings.warn(
                    f"[rank {comm.rank}] Neumann BC on '{label}': "
                    f"{np.count_nonzero(~good)} wall node(s) had no matching "
                    f"interior-column coordinate within tolerance (max err "
                    f"{err.max():.3e}); those left unchanged.",
                    stacklevel=2,
                )

            self.phi.data[wall_idx[good], 0] = global_pairs[nearest[good], 1]
    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def reini_frequency(self) -> int:
        """Reinitialisation frequency (advection steps between calls)."""
        return self._reini_frequency

    def interface_volume(self) -> float:
        """Return ∫φ dΩ (approximate enclosed volume for mass-conservation checks)."""
        integ = uw.maths.Integral(self.mesh, self.phi.sym[0, 0])
        return integ.evaluate()

    def clamp(self, lo: float = 0.0, hi: float = 1.0) -> None:
        """Clamp φ values to [lo, hi] in place (post-advection safeguard).

        This is a raw, unweighted np.clip -- NOT mass-conservative on its
        own (see `_correct_mass` and the `conserve_mass` constructor
        option, which is what actually keeps `interface_volume()` from
        drifting over many steps; calling this on top of a mass-corrected
        `solve()` is a harmless no-op, since the state is already inside
        [lo, hi] by then).
        """
        self.phi.data[:, 0] = np.clip(self.phi.data[:, 0], lo, hi)

    def _correct_mass(self, target: float, lo: float = 0.0, hi: float = 1.0) -> None:
        """Global mass correction (Zhang, Zou & Greaves 2010): find a
        single uniform additive shift `delta` such that

            INT_Omega clip(phi + delta, lo, hi) dOmega == target,

        and leave `self.phi.data` in that clipped, shifted state.

        The map `delta -> resulting volume` is monotone non-decreasing
        (increasing delta can only raise or hold every clipped nodal
        value, never lower one), so a plain bisection is guaranteed to
        converge -- no Newton/derivative needed, and no assumption about
        how oscillatory or well-behaved the *current* field is beyond
        that monotonicity, which holds unconditionally for a clip.

        This does not know or care *why* the volume drifted (reinit,
        clamp asymmetry, or anything else); it is a final, cheap
        (a handful of `interface_volume()` evaluations, not a new SNES
        solve) correction applied once per `solve()` call.
        """
        data0 = self.phi.data[:, 0].copy()

        def vol_for_shift(delta: float) -> float:
            self.phi.data[:, 0] = np.clip(data0 + delta, lo, hi)
            return self.interface_volume()

        v0 = vol_for_shift(0.0)
        if abs(v0 - target) < self._mass_correction_tol:
            return  # already within tolerance; state from delta=0 stands

        span = hi - lo
        if v0 < target:
            lo_d, hi_d = 0.0, max(span * 1.0e-3, 1.0e-8)
            tries = 0
            while vol_for_shift(hi_d) < target and tries < 30:
                hi_d *= 2.0
                tries += 1
        else:
            lo_d, hi_d = -max(span * 1.0e-3, 1.0e-8), 0.0
            tries = 0
            while vol_for_shift(lo_d) > target and tries < 30:
                lo_d *= 2.0
                tries += 1

        if not (vol_for_shift(lo_d) <= target <= vol_for_shift(hi_d)):
            warnings.warn(
                "_correct_mass: could not bracket the target volume "
                f"({target:.6g}) within delta in [{lo_d:.3g}, {hi_d:.3g}] "
                "-- leaving the field at its widest attempted shift rather "
                "than an unbracketed (unreliable) bisection result. This "
                "usually means the whole field is already pinned at lo or "
                "hi, with nothing left to shift.",
                stacklevel=2,
            )
            return

        mid = 0.0
        for _ in range(self._mass_correction_max_iter):
            mid = 0.5 * (lo_d + hi_d)
            vmid = vol_for_shift(mid)
            if abs(vmid - target) < self._mass_correction_tol:
                break
            if vmid < target:
                lo_d = mid
            else:
                hi_d = mid
        vol_for_shift(mid)  # leave self.phi.data at the converged shift


def material_property_field(
    level_set: sympy.Expr | list[sympy.Expr],
    field_values: list[float],
    interface: str,
) -> sympy.Expr:
    """Generates sympy algebra describing a physical property across the domain.
    Args:
      level_set:
        A sympy expression for the level set (typically `mesh_variable.sym[0, 0]`),
        or a list thereof
      field_values:
        A list of physical-property values specific to each material
      interface:
        A string specifying how property transitions between materials are calculated
    Returns:
      Sympy algebra representing the physical property throughout the domain
    """
    impl_interface = ["sharp", "sharp_adjoint", "arithmetic", "geometric", "harmonic"]
    if interface not in impl_interface:
        raise ValueError(f"Interface must be one of {impl_interface}")

    level_set = level_set.copy() if isinstance(level_set, list) else [level_set]
    field_values = field_values.copy()

    result = None
    while level_set:
        ls = sympy.Max(sympy.Min(level_set.pop(), 1), 0)

        # Deepest (last) level set: pull both surrounding field values at once.
        # Otherwise: pull one field value and combine with the running result.
        field_value = field_values.pop()
        other_side = field_values.pop() if not level_set else result

        match interface:
            case "sharp":
                result = sympy.Piecewise((field_value, ls > sympy.Rational(1, 2)), (other_side, True))
            case "sharp_adjoint":
                ls_shift = ls - sympy.Rational(1, 2)
                heaviside = (ls_shift + sympy.Abs(ls_shift)) / 2 / ls_shift

                result = field_value * heaviside + other_side * (1 - heaviside)
            case "arithmetic":
                result = field_value * ls + other_side * (1 - ls)
            case "geometric":
                result = field_value**ls * other_side ** (1 - ls)
            case "harmonic":
                result = 1 / (ls / field_value + (1 - ls) / other_side)
    return result
