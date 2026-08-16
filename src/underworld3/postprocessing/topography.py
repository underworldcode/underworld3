r"""Dynamic-topography recovery and spherical-harmonic projection utilities."""

from __future__ import annotations

from dataclasses import dataclass

from mpi4py import MPI
import numpy as np


@dataclass(frozen=True)
class CbfLumpedBoundaryTopography:
    """CBF-lumped topography samples on a spherical boundary.

    The arrays are populated on rank 0 and set to ``None`` on other ranks. Use
    :func:`cbf_lumped_spherical_harmonic_coefficient` when only a projected
    scalar coefficient is required on all ranks.
    """

    coords: np.ndarray | None
    topography: np.ndarray | None
    harmonic_values: np.ndarray | None = None


@dataclass(frozen=True)
class SphericalShellTopographyCoefficients:
    """Surface and CMB topography coefficients plus the source used."""

    surface: float
    cmb: float
    source: str


def boundary_adjacent_cells(mesh, boundary: str) -> list[int]:
    """Return local cells adjacent to a named boundary label."""

    dm = mesh.dm
    boundary_obj = getattr(mesh.boundaries, boundary)
    label = dm.getLabel(boundary_obj.name)
    if label is None:
        raise ValueError(f"Boundary label '{boundary_obj.name}' was not found.")

    cell_start, cell_end = dm.getHeightStratum(0)
    cells = set()
    value_is = label.getValueIS()
    values = [] if value_is is None else list(value_is.getIndices())

    for value in values:
        point_is = label.getStratumIS(int(value))
        if point_is is None:
            continue
        for point in point_is.getIndices():
            found_cell = False
            for support_point in dm.getSupport(int(point)):
                if cell_start <= int(support_point) < cell_end:
                    cells.add(int(support_point))
                    found_cell = True

            if not found_cell:
                star_points, _ = dm.getTransitiveClosure(int(point), useCone=False)
                for star_point in star_points:
                    if cell_start <= int(star_point) < cell_end:
                        cells.add(int(star_point))

    return sorted(cells)


def _boundary_node_mask(coords: np.ndarray, radius: float, tolerance: float | None):
    coord_radius = np.linalg.norm(coords, axis=1)
    if tolerance is None:
        tolerance = max(1.0e-8, 1.0e-6 * max(1.0, abs(float(radius))))
    return coord_radius, np.abs(coord_radius - radius) < tolerance


def _spherical_triangle_area(a, b, c, radius: float) -> float:
    det_abc = abs(float(np.dot(a, np.cross(b, c))))
    denom_abc = float(1.0 + np.dot(a, b) + np.dot(b, c) + np.dot(c, a))
    return 2.0 * np.arctan2(det_abc, denom_abc) * radius**2


def _project_spherical_harmonic_samples(
    coords: np.ndarray,
    values: np.ndarray,
    radius: float,
    harmonic_degree: int,
    response_sign: float,
) -> float:
    """Project nodal samples over their spherical convex-hull triangulation."""

    from scipy.spatial import ConvexHull

    unit_coords = coords / np.linalg.norm(coords, axis=1)[:, None]
    hull = ConvexHull(unit_coords, qhull_options="QJ")
    projected_integral = 0.0

    for simplex in hull.simplices:
        area = _spherical_triangle_area(
            unit_coords[simplex[0]],
            unit_coords[simplex[1]],
            unit_coords[simplex[2]],
            float(radius),
        )
        centroid = unit_coords[simplex].mean(axis=0)
        centroid /= np.linalg.norm(centroid)
        centroid_harmonic = np.polynomial.legendre.Legendre.basis(int(harmonic_degree))(
            np.clip(centroid[2], -1.0, 1.0)
        )
        projected_integral += area * float(values[simplex].mean()) * centroid_harmonic

    harmonic_angular_norm = 4.0 * np.pi / (2 * int(harmonic_degree) + 1)
    return float(
        float(response_sign)
        * projected_integral
        / (float(radius) ** 2 * harmonic_angular_norm)
    )


def _gather_boundary_samples(
    coords: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Gather and average coordinate-keyed boundary samples on rank zero."""

    local_rows = np.column_stack(
        (np.asarray(coords, dtype=float), np.asarray(values, dtype=float))
    )
    gathered_rows = MPI.COMM_WORLD.gather(local_rows, root=0)
    if MPI.COMM_WORLD.rank != 0:
        return None, None

    rows = np.vstack([item for item in gathered_rows if item.size > 0])
    merged = {}
    for row in rows:
        key = tuple(np.round(row[:3], 12))
        if key not in merged:
            merged[key] = [row[:3].copy(), 0.0, 0]
        merged[key][1] += float(row[3])
        merged[key][2] += 1

    global_coords = np.array([item[0] for item in merged.values()])
    global_values = np.array([item[1] / item[2] for item in merged.values()])
    return global_coords, global_values


def cbf_lumped_boundary_topography(
    stokes,
    velocity,
    boundary: str,
    radius: float,
    normal_sign: float,
    *,
    harmonic_degree: int | None = None,
    boundary_tolerance: float | None = None,
    residual_field_id: int = 0,
    velocity_residual_key: str = "velocity",
) -> CbfLumpedBoundaryTopography:
    """Recover CBF-lumped topography samples on a spherical boundary.

    Parameters
    ----------
    stokes
        Solved Stokes system. It must provide ``compute_volume_residual_fields``.
    velocity
        Velocity ``MeshVariable`` used by ``stokes``.
    boundary
        Named mesh boundary, for example ``"Upper"`` or ``"Lower"``.
    radius
        Spherical boundary radius.
    normal_sign
        Sign of the outward unit normal relative to the radial unit vector.
        Use ``+1`` on an outer surface and ``-1`` on an inner spherical surface.
    harmonic_degree
        Optional Legendre degree used to attach ``P_l^0(cos(theta))`` values to
        the returned samples.
    boundary_tolerance
        Coordinate-radius tolerance used to identify boundary velocity nodes.
        If omitted, a small radius-relative tolerance is used.
    residual_field_id
        PETSc residual field id for the velocity block.
    velocity_residual_key
        Key used by ``compute_volume_residual_fields`` for the velocity residual.
    """

    from scipy.spatial import ConvexHull

    mesh = stokes.mesh
    adjacent_cells = boundary_adjacent_cells(mesh, boundary)
    residual_fields = stokes.compute_volume_residual_fields(
        cell_indices=adjacent_cells,
        residual_field_id=residual_field_id,
    )
    if velocity_residual_key not in residual_fields:
        available = ", ".join(sorted(residual_fields))
        raise KeyError(
            f"Residual field '{velocity_residual_key}' was not returned. "
            f"Available fields: {available}"
        )

    velocity_residual = np.asarray(residual_fields[velocity_residual_key]).reshape(
        velocity.data.shape
    )
    velocity_coords = np.asarray(velocity.coords)
    velocity_r, boundary_nodes = _boundary_node_mask(
        velocity_coords, float(radius), boundary_tolerance
    )
    if not np.any(boundary_nodes):
        raise ValueError(
            f"No velocity nodes found on boundary '{boundary}' at radius {radius}."
        )

    velocity_unit_r = velocity_coords / velocity_r[:, None]
    normal_residual = np.einsum(
        "ij,ij->i",
        velocity_residual[boundary_nodes],
        float(normal_sign) * velocity_unit_r[boundary_nodes],
    )

    if harmonic_degree is None:
        harmonic_values = np.zeros(normal_residual.shape, dtype=float)
    else:
        cos_theta = np.clip(
            velocity_coords[boundary_nodes, 2] / velocity_r[boundary_nodes],
            -1.0,
            1.0,
        )
        harmonic_values = np.polynomial.legendre.Legendre.basis(int(harmonic_degree))(
            cos_theta
        )

    local_rows = np.column_stack(
        (
            velocity_coords[boundary_nodes],
            normal_residual,
            harmonic_values,
        )
    )
    gathered_rows = MPI.COMM_WORLD.gather(local_rows, root=0)

    if MPI.COMM_WORLD.rank != 0:
        return CbfLumpedBoundaryTopography(None, None, None)

    rows = np.vstack([item for item in gathered_rows if item.size > 0])
    merged = {}
    for row in rows:
        key = tuple(np.round(row[:3], 12))
        if key not in merged:
            merged[key] = [row[:3].copy(), 0.0, 0.0, 0]
        merged[key][1] += float(row[3])
        merged[key][2] += float(row[4])
        merged[key][3] += 1

    coords = np.array([item[0] for item in merged.values()])
    nodal_residual = np.array([item[1] for item in merged.values()])
    nodal_harmonic = np.array([item[2] / item[3] for item in merged.values()])
    unit_coords = coords / np.linalg.norm(coords, axis=1)[:, None]

    hull = ConvexHull(unit_coords, qhull_options="QJ")
    nodal_mass = np.zeros(coords.shape[0])

    for simplex in hull.simplices:
        a, b, c = unit_coords[simplex]
        area = _spherical_triangle_area(a, b, c, float(radius))
        nodal_mass[simplex] += area / 3.0

    nodal_topography = np.zeros_like(nodal_residual)
    valid = nodal_mass > 0.0
    nodal_topography[valid] = nodal_residual[valid] / nodal_mass[valid]

    return CbfLumpedBoundaryTopography(coords, nodal_topography, nodal_harmonic)


def cbf_lumped_spherical_harmonic_coefficient(
    stokes,
    velocity,
    boundary: str,
    radius: float,
    harmonic_degree: int,
    normal_sign: float,
    *,
    response_sign: float = 1.0,
    boundary_tolerance: float | None = None,
    residual_field_id: int = 0,
    velocity_residual_key: str = "velocity",
) -> float:
    """Return a CBF-lumped ``P_l^0`` topography coefficient on a sphere.

    The coefficient is projected with the Zhong et al. convention
    ``4*pi / (2*l + 1)`` for the angular norm of ``P_l^0``.
    """

    samples = cbf_lumped_boundary_topography(
        stokes=stokes,
        velocity=velocity,
        boundary=boundary,
        radius=radius,
        normal_sign=normal_sign,
        harmonic_degree=harmonic_degree,
        boundary_tolerance=boundary_tolerance,
        residual_field_id=residual_field_id,
        velocity_residual_key=velocity_residual_key,
    )

    result = None
    if MPI.COMM_WORLD.rank == 0:
        result = _project_spherical_harmonic_samples(
            samples.coords,
            samples.topography,
            radius,
            harmonic_degree,
            response_sign,
        )

    return MPI.COMM_WORLD.bcast(result, root=0)


def rotated_reaction_spherical_harmonic_coefficient(
    stokes,
    boundary: str,
    radius: float,
    harmonic_degree: int,
    *,
    response_sign: float = 1.0,
    buoyancy_scale: float = 1.0,
    mass: str = "auto",
) -> float:
    """Return a spherical-harmonic topography coefficient from rotated free slip.

    The boundary normal traction is recovered directly from the strong
    free-slip constraint reaction. The returned topography follows
    ``h = -sigma_nn / buoyancy_scale`` after the traction mean has been removed
    by :meth:`stokes.boundary_normal_traction`.

    ``mass="auto"`` selects consistent surface-mass recovery for a 3D P2
    triangular trace and lumped recovery where it is valid. On faceted curved
    boundaries, raw P2 vertex values are more sensitive to geometry error than
    edge-midpoint values; use fitted harmonic coefficients such as this result,
    or midpoint samples, for convergence studies.
    """

    xs, sigma_nn = stokes.boundary_normal_traction(boundary, mass=mass)
    coords, topography = _gather_boundary_samples(
        xs,
        -np.asarray(sigma_nn, dtype=float) / float(buoyancy_scale),
    )

    result = None
    if MPI.COMM_WORLD.rank == 0:
        result = _project_spherical_harmonic_samples(
            coords,
            topography,
            radius,
            harmonic_degree,
            response_sign,
        )

    return MPI.COMM_WORLD.bcast(result, root=0)


def spherical_harmonic_boundary_coefficient(
    *,
    mesh,
    fn,
    boundary: str,
    radius: float,
    harmonic_degree: int,
    response_sign: float = 1.0,
) -> float:
    """Project a boundary scalar function onto ``P_l^0`` on a sphere."""

    import sympy
    import underworld3 as uw

    theta = mesh.CoordinateSystem.xR[1]
    cos_theta = sympy.cos(theta)
    harmonic = sympy.assoc_legendre(int(harmonic_degree), 0, cos_theta)
    harmonic_angular_norm = 4.0 * np.pi / (2 * int(harmonic_degree) + 1)
    integral = float(
        uw.maths.BdIntegral(mesh, fn=fn * harmonic, boundary=boundary).evaluate()
    )
    return float(
        float(response_sign) * integral / (float(radius) ** 2 * harmonic_angular_norm)
    )


def _has_constrained_multiplier(stokes, boundary: str) -> bool:
    if not hasattr(stokes, "multiplier"):
        return False
    try:
        return stokes.multiplier(boundary) is not None
    except (AttributeError, ValueError):
        return False


def _has_rotated_reaction(stokes, boundary: str) -> bool:
    solve_info = getattr(stokes, "_rotated_freeslip_info", None)
    if not solve_info:
        return False
    boundaries = [
        item[0] if isinstance(item, tuple) else item
        for item in solve_info.get("boundaries", ())
    ]
    return boundary in boundaries


def spherical_shell_topography_coefficients(
    *,
    stokes,
    velocity=None,
    radius_inner: float,
    radius_outer: float,
    harmonic_degree: int,
    source: str = "auto",
    boundary_tolerance: float | None = None,
    surface_boundary: str = "Upper",
    cmb_boundary: str = "Lower",
    constrained_reference=None,
) -> SphericalShellTopographyCoefficients:
    """Return surface and CMB topography coefficients from the best source.

    ``source="auto"`` uses constrained multiplier topography when the solved
    Stokes system has multipliers on both requested boundaries, rotated
    constraint reactions when both boundaries use rotated strong free slip,
    and CBF-lumped residual recovery otherwise. Explicit sources are
    ``"constrained_multiplier"``, ``"rotated_reaction"``, and
    ``"cbf_lumped_residual"``. ``velocity`` is needed only by the CBF fallback;
    when omitted, the Stokes velocity field is used.
    """

    if source == "auto":
        if _has_constrained_multiplier(
            stokes, surface_boundary
        ) and _has_constrained_multiplier(stokes, cmb_boundary):
            source = "constrained_multiplier"
        elif _has_rotated_reaction(stokes, surface_boundary) and _has_rotated_reaction(
            stokes, cmb_boundary
        ):
            source = "rotated_reaction"
        else:
            source = "cbf_lumped_residual"

    if source == "constrained_multiplier":
        surface = spherical_harmonic_boundary_coefficient(
            mesh=stokes.mesh,
            fn=stokes.topography(surface_boundary, reference=constrained_reference),
            boundary=surface_boundary,
            radius=radius_outer,
            harmonic_degree=harmonic_degree,
            response_sign=1.0,
        )
        cmb = spherical_harmonic_boundary_coefficient(
            mesh=stokes.mesh,
            fn=stokes.topography(cmb_boundary, reference=constrained_reference),
            boundary=cmb_boundary,
            radius=radius_inner,
            harmonic_degree=harmonic_degree,
            response_sign=1.0,
        )
    elif source == "rotated_reaction":
        surface = rotated_reaction_spherical_harmonic_coefficient(
            stokes=stokes,
            boundary=surface_boundary,
            radius=radius_outer,
            harmonic_degree=harmonic_degree,
            response_sign=1.0,
        )
        cmb = rotated_reaction_spherical_harmonic_coefficient(
            stokes=stokes,
            boundary=cmb_boundary,
            radius=radius_inner,
            harmonic_degree=harmonic_degree,
            response_sign=-1.0,
        )
    elif source == "cbf_lumped_residual":
        if velocity is None:
            velocity = getattr(getattr(stokes, "Unknowns", None), "u", None)
        if velocity is None:
            raise ValueError(
                "CBF residual topography requires a velocity field, either on "
                "stokes.Unknowns.u or through the velocity argument."
            )
        surface = cbf_lumped_spherical_harmonic_coefficient(
            stokes=stokes,
            velocity=velocity,
            boundary=surface_boundary,
            radius=radius_outer,
            harmonic_degree=harmonic_degree,
            normal_sign=1.0,
            response_sign=1.0,
            boundary_tolerance=boundary_tolerance,
        )
        cmb = cbf_lumped_spherical_harmonic_coefficient(
            stokes=stokes,
            velocity=velocity,
            boundary=cmb_boundary,
            radius=radius_inner,
            harmonic_degree=harmonic_degree,
            normal_sign=-1.0,
            response_sign=-1.0,
            boundary_tolerance=boundary_tolerance,
        )
    else:
        raise ValueError(
            "source must be 'auto', 'constrained_multiplier', "
            f"'rotated_reaction', or 'cbf_lumped_residual', got {source!r}"
        )

    return SphericalShellTopographyCoefficients(
        surface=float(surface),
        cmb=float(cmb),
        source=source,
    )
