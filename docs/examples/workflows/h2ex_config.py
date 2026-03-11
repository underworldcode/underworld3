"""H2Ex fault-controlled flow — configuration and workflow steps.

Demonstrates the **product-aware workflow pattern** where expensive
serial steps (mesh creation, fault loading, adaptation) produce named
products that can be saved and reloaded for parameter studies.

The full pipeline computes:

1. Geographic mesh → adapt near faults
2. Stress (TransverseIsotropic, oriented BCs) → strain rate
3. Reference stress (no faults) → background strain rate
4. Permeability from strain-rate delta
5. Darcy flow with computed permeability
6. Tracer advection → surface accumulation map

The companion notebook (``h2ex_notebook.py``) shows both the full
serial pipeline and the product-loading shortcut.

Usage::

    import h2ex_config as h2ex
    from underworld3.workflows import WorkflowProducts

    config = h2ex.H2ExConfig()
    products = WorkflowProducts(config)

    # Full build
    mesh = h2ex.create_mesh(config)
    surfaces = h2ex.load_and_build_faults(mesh, config)
    mesh = h2ex.adapt_mesh(mesh, surfaces, config)
    products.save("adapted_mesh", mesh)

    # Stress calculations
    stokes, strain_rate = h2ex.solve_stress(mesh, surfaces, config)
    strain_rate_ref = h2ex.solve_reference_stress(mesh, config)
    permeability = h2ex.compute_permeability(mesh, strain_rate, strain_rate_ref, config)

    # Flow and accumulation
    darcy, v_darcy = h2ex.solve_darcy(mesh, permeability, config)
    accumulation = h2ex.advect_tracers(mesh, v_darcy, config)

    # Inspect
    h2ex.view()
"""

import sys
from typing import Tuple

import numpy as np
import sympy
from pydantic import Field

from underworld3.workflows import WorkflowConfig, workflow_step


def view():
    """Display the workflow steps and config classes in this module."""
    from underworld3.workflows import view as _wf_view

    _wf_view(sys.modules[__name__])


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class H2ExConfig(WorkflowConfig):
    """Fault-controlled flow with geographic mesh and anisotropic rheology.

    Models fluid flow through a crustal volume where fault zones act as
    conduits with reduced viscosity.  Designed for hydrogen exploration
    workflows over the Eyre Peninsula (SA) or similar regions.
    """

    workflow_name: str = "h2ex_fault_flow"
    description: str = (
        "Fault-controlled flow: geographic mesh, anisotropic rheology"
    )

    # Reference quantities for non-dimensionalisation
    ref_length: str = "1000 km"
    ref_viscosity: str = "1e21 Pa*s"
    ref_diffusivity: str = "1e-6 m**2/s"

    # Domain
    lon_range: Tuple[float, float] = Field(
        default=(135.5, 137.5), description="Longitude range (degrees)"
    )
    lat_range: Tuple[float, float] = Field(
        default=(-34.5, -33.0), description="Latitude range (degrees)"
    )
    depth_range_km: Tuple[float, float] = Field(
        default=(0.0, 50.0), description="Depth range (km)"
    )
    ellipsoid: str = Field(default="WGS84", description="Geodetic ellipsoid")
    num_elements: Tuple[int, int, int] = Field(
        default=(16, 16, 8), description="Elements (lon, lat, depth)"
    )

    # Fault data
    fault_data_path: str = Field(
        default="Structures/faults_as_swarm_points_xyz.npz",
        description="Path to fault trace data (.npz)",
    )
    trace_resolution_km: float = Field(
        default=3.0, gt=0, description="Along-trace point spacing (km)"
    )

    # Adaptation
    adapt: bool = Field(default=True, description="Enable mesh adaptation")
    h_near_km: float = Field(
        default=2.0, gt=0, description="Target edge length near faults (km)"
    )
    h_far_km: float = Field(
        default=20.0, gt=0, description="Target edge length far from faults (km)"
    )
    transition_km: float = Field(
        default=10.0, gt=0, description="Transition distance for refinement (km)"
    )

    # Stress orientation and fault rheology
    stress_azimuth_deg: float = Field(
        default=0.0, description="Stress orientation angle from E-W (degrees)"
    )
    fault_weakness: float = Field(
        default=1e-5, gt=0, description="Fault-zone viscosity ratio (eta_1/eta_0)"
    )

    # Permeability thresholds (strain-rate delta → log10 permeability)
    delta_low: float = Field(
        default=-2.0, description="Delta below which perm is very low"
    )
    delta_mid: float = Field(
        default=0.0, description="Delta below which perm is low"
    )
    delta_high: float = Field(
        default=5.0, description="Delta above which perm is high"
    )
    delta_very_high: float = Field(
        default=10.0, description="Delta above which perm is very high"
    )

    # Darcy flow
    source_depth_km: float = Field(
        default=30.0, gt=0, description="Depth below which Darcy source is active (km)"
    )
    source_magnitude: float = Field(
        default=10.0, description="Darcy source strength"
    )
    gravity_scale: float = Field(
        default=0.01, description="Gravity term scaling for Darcy"
    )

    # Tracer advection
    tracer_grid_n: int = Field(
        default=100, gt=0, description="Tracer grid points per horizontal dimension"
    )
    tracer_n_depths: int = Field(
        default=20, gt=0, description="Number of depth levels for seeding tracers"
    )
    accumulation_depth_km: float = Field(
        default=0.0, ge=0, description="Depth at which to capture tracers (km)"
    )
    max_advection_steps: int = Field(
        default=500, gt=0, description="Maximum advection steps"
    )

    # Output
    output_dir: str = "output/h2ex"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _geographic_basis(mesh):
    """Get geographic coordinate basis vectors and coordinates.

    Returns (unit_vertical, unit_SN, unit_EW, r, th, ph).
    The SN vector is negated from the co-latitude basis vector
    to give a northward-pointing direction.
    """
    r, th, ph = mesh.CoordinateSystem.R
    unit_vertical = mesh.CoordinateSystem.unit_e_0
    unit_SN = -mesh.CoordinateSystem.unit_e_1
    unit_EW = mesh.CoordinateSystem.unit_e_2
    return unit_vertical, unit_SN, unit_EW, r, th, ph


def _build_position_function(mesh, ph):
    """Build normalized longitude position function P ∈ [-0.5, 0.5].

    Used for spatially varying boundary conditions: traction is strongest
    at the edges and zero at the center.
    """
    import underworld3 as uw

    ph_vals = uw.function.evaluate(ph, mesh.data, rbf=True)
    ph_min_val = uw.utilities.gather_data(ph_vals.min(), bcast=True).min()
    ph_max_val = uw.utilities.gather_data(ph_vals.max(), bcast=True).max()

    ph_min = uw.function.expression(r"\theta_{-}", ph_min_val, "min longitude")
    ph_max = uw.function.expression(r"\theta_{+}", ph_max_val, "max longitude")

    P = (ph - (ph_max + ph_min) / 2) / (ph_max - ph_min)
    return P


def _solve_stokes_stress(mesh, surfaces, config, fault_weakness_value):
    """Build and solve a Stokes system for stress calculation.

    Uses TransverseIsotropicFlowModel with fault normals as directors.
    Oriented velocity BCs on vertical faces and base; stress-free top.

    Parameters
    ----------
    mesh : Mesh
        The (adapted) geographic mesh.
    surfaces : SurfaceCollection or None
        Fault surfaces for anisotropy. None for reference (isotropic) solve.
    config : H2ExConfig
        Workflow configuration.
    fault_weakness_value : float
        eta_1/eta_0 ratio.  Use config.fault_weakness for fault solve,
        1.0 for reference solve.

    Returns
    -------
    stokes, v, p, strain_rate_inv2
    """
    import underworld3 as uw

    unit_vertical, unit_SN, unit_EW, r, th, ph = _geographic_basis(mesh)
    P = _build_position_function(mesh, ph)

    # Variables
    v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
    strain_rate_inv2 = uw.discretisation.MeshVariable("eps", mesh, 1, degree=1)

    # Stokes solver with TransverseIsotropicFlowModel
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.TransverseIsotropicFlowModel
    stokes.constitutive_model.Parameters.eta_0 = 1
    stokes.penalty = 0.1

    # Fault-controlled anisotropy
    if surfaces is not None and len(surfaces.surfaces) > 0:
        fault_fields = surfaces.compute_nearest_fields(mesh)
        dist_var = fault_fields["distance"]
        normal_var = fault_fields["normal"]

        stokes.constitutive_model.Parameters.eta_1 = sympy.Piecewise(
            (fault_weakness_value, dist_var.sym[0] < mesh.get_min_radius() * 5),
            (1, True),
        )
        stokes.constitutive_model.Parameters.director = normal_var.sym
    else:
        stokes.constitutive_model.Parameters.eta_1 = 1

    # Oriented velocity BCs: -P * (cos(θ)·e_EW - sin(θ)·e_SN)
    theta = np.deg2rad(config.stress_azimuth_deg)
    stress_orientation = uw.function.expression(
        r"\theta_s", theta, "Stress orientation angle"
    )
    traction = -P * (
        sympy.cos(stress_orientation) * unit_EW
        - sympy.sin(stress_orientation) * unit_SN
    )

    # Apply to vertical faces and base (stress-free top)
    for bnd_name in ["West", "North", "East", "South", "Lower"]:
        bnd = getattr(mesh.boundaries, bnd_name, None)
        if bnd is not None:
            stokes.add_essential_bc(traction, bnd.name)

    # Body force
    stokes.bodyforce = -1 * unit_vertical

    stokes.tolerance = 1.0e-3
    stokes.solve(zero_init_guess=True)

    # Project strain rate invariant
    projection = uw.systems.Projection(mesh, strain_rate_inv2)
    projection.uw_function = stokes.constitutive_model.Unknowns.Einv2
    projection.smoothing = 1.0e-6
    projection.solve()

    return stokes, v, p, strain_rate_inv2


# ---------------------------------------------------------------------------
# Workflow steps
# ---------------------------------------------------------------------------


@workflow_step(
    description="Build geographic mesh for the study region",
    produces=["mesh"],
)
def create_mesh(config: H2ExConfig):
    """Build a 3D geographic mesh over the configured region.

    Returns
    -------
    mesh
        A RegionalGeographicBox mesh.
    """
    import underworld3 as uw

    mesh = uw.meshing.RegionalGeographicBox(
        lon_range=config.lon_range,
        lat_range=config.lat_range,
        depth_range=config.depth_range_km,
        ellipsoid=config.ellipsoid,
        numElements=config.num_elements,
    )
    return mesh


@workflow_step(
    description="Load fault traces and build Surface objects",
    produces=["fault_surfaces"],
    requires=["mesh"],
)
def load_and_build_faults(mesh, config: H2ExConfig):
    """Load fault data from .npz file and build a SurfaceCollection.

    The .npz file should contain arrays keyed by fault name, each with
    shape (N, 2) or (N, 3) for trace coordinates (lon, lat[, depth]).

    Returns
    -------
    surfaces : SurfaceCollection
        Collection of fault surfaces extruded to depth.
    """
    import underworld3 as uw
    from pathlib import Path

    data_path = Path(config.fault_data_path)
    surfaces = uw.meshing.SurfaceCollection()

    if not data_path.exists():
        uw.pprint(
            0,
            f"Fault data not found at {data_path}. "
            "Returning empty SurfaceCollection.",
        )
        return surfaces

    fault_data = np.load(str(data_path), allow_pickle=True)

    for fault_name in fault_data.files:
        points = fault_data[fault_name]
        if points.ndim != 2 or points.shape[1] < 2:
            continue

        trace_xy = points[:, :2]  # (lon, lat)

        surface = uw.meshing.Surface.from_trace(
            name=fault_name,
            mesh=mesh,
            trace_points=trace_xy,
            depth_range=config.depth_range_km,
            trace_resolution=config.trace_resolution_km,
        )
        surfaces.add(surface)

    uw.pprint(0, f"Built {len(surfaces.surfaces)} fault surfaces")
    return surfaces


@workflow_step(
    description="Adapt mesh near fault surfaces",
    produces=["adapted_mesh"],
    requires=["mesh", "fault_surfaces"],
)
def adapt_mesh(mesh, surfaces, config: H2ExConfig):
    """Refine the mesh near fault surfaces using a distance-based metric.

    If ``config.adapt`` is False, returns the mesh unchanged.

    Returns
    -------
    mesh
        The (possibly adapted) mesh.
    """
    import underworld3 as uw

    if not config.adapt or len(surfaces.surfaces) == 0:
        uw.pprint(0, "Skipping adaptation (adapt=False or no faults)")
        return mesh

    # Compute distance field
    dist_var = surfaces.compute_distance_field(mesh)

    # Build a metric: small h near faults, large h far away
    h_near = config.h_near_km
    h_far = config.h_far_km
    transition = config.transition_km

    metric = uw.discretisation.MeshVariable("H", mesh, 1)

    d = dist_var.data[:, 0]
    t = np.clip(d / transition, 0.0, 1.0)
    target_h = h_near + (h_far - h_near) * t
    metric.data[:, 0] = 1.0 / target_h**2

    uw.pprint(0, "Adapting mesh...")
    mesh.adapt(metric)
    uw.pprint(0, f"Adapted mesh: {mesh.elements_count} elements")

    return mesh


@workflow_step(
    description="Solve Stokes for fault-controlled stress",
    produces=["strain_rate"],
    requires=["adapted_mesh", "fault_surfaces"],
)
def solve_stress(mesh, surfaces, config: H2ExConfig):
    """Solve Stokes with TransverseIsotropicFlowModel for fault-controlled stress.

    Uses oriented velocity BCs on vertical faces and base with the
    stress azimuth from config.  Faults appear as weak anisotropic zones
    (eta_1 = fault_weakness near faults, 1 elsewhere).

    Returns
    -------
    stokes, strain_rate_inv2
        The solved Stokes system and projected strain rate invariant.
    """
    import underworld3 as uw

    uw.pprint(
        0,
        f"Solving stress: azimuth={config.stress_azimuth_deg}°, "
        f"fault_weakness={config.fault_weakness}",
    )

    stokes, v, p, strain_rate_inv2 = _solve_stokes_stress(
        mesh, surfaces, config,
        fault_weakness_value=config.fault_weakness,
    )

    uw.pprint(0, "Stress solve complete")
    return stokes, strain_rate_inv2


@workflow_step(
    description="Solve reference stress (no faults)",
    produces=["strain_rate_ref"],
    requires=["adapted_mesh"],
)
def solve_reference_stress(mesh, config: H2ExConfig):
    """Solve Stokes with uniform viscosity (no fault weakness).

    Same BCs and setup as solve_stress but with fault_weakness=1.0,
    giving the background strain rate for comparison.

    Returns
    -------
    strain_rate_ref
        Projected strain rate invariant for the reference (no-fault) case.
    """
    import underworld3 as uw

    uw.pprint(0, "Solving reference stress (no faults)...")

    _, _, _, strain_rate_ref = _solve_stokes_stress(
        mesh, None, config,
        fault_weakness_value=1.0,
    )

    uw.pprint(0, "Reference stress solve complete")
    return strain_rate_ref


@workflow_step(
    description="Compute permeability from strain-rate delta",
    produces=["permeability"],
    requires=["strain_rate", "strain_rate_ref"],
)
def compute_permeability(mesh, strain_rate, strain_rate_ref, config: H2ExConfig):
    """Compute permeability field from difference in strain rate.

    Where faults concentrate strain (positive delta), permeability is
    enhanced.  Where strain is reduced, permeability is lowered.  The
    mapping uses piecewise thresholds on log10 scale:

    - delta < delta_low     → 10^(-2) (very low)
    - delta_low ≤ delta < 0 → 10^(-1) (low)
    - 0 ≤ delta ≤ delta_high → 10^(0)  (background)
    - delta_high < delta ≤ delta_very_high → 10^(1)  (high)
    - delta > delta_very_high → 10^(2)  (very high)

    Returns
    -------
    permeability : MeshVariable
        Permeability field on the mesh.
    """
    import underworld3 as uw

    permeability = uw.discretisation.MeshVariable(
        "Perm", mesh, 1, degree=1, varsymbol=r"\kappa"
    )

    # Strain rate delta
    delta = strain_rate.data[:, 0] - strain_rate_ref.data[:, 0]

    # Piecewise log10 permeability from thresholds
    log_perm = np.zeros_like(delta)

    log_perm[delta < config.delta_mid] = -1
    log_perm[delta < config.delta_low] = -2
    log_perm[delta > config.delta_high] = 1
    log_perm[delta > config.delta_very_high] = 2

    permeability.data[:, 0] = np.power(10.0, log_perm)

    uw.pprint(
        0,
        f"Permeability: min={permeability.data[:, 0].min():.2e}, "
        f"max={permeability.data[:, 0].max():.2e}",
    )
    return permeability


@workflow_step(
    description="Solve Darcy flow with computed permeability",
    produces=["darcy_velocity"],
    requires=["adapted_mesh", "permeability"],
)
def solve_darcy(mesh, permeability, config: H2ExConfig):
    """Solve steady-state Darcy flow with the computed permeability field.

    Boundary conditions: p = 0 at the upper surface (free drainage).
    Source term active below ``source_depth_km`` to drive upward flow.
    Gravity term points radially inward.

    Returns
    -------
    darcy, v_darcy, p_darcy
        The Darcy solver and its velocity / pressure fields.
    """
    import underworld3 as uw

    unit_vertical, unit_SN, unit_EW, r, th, ph = _geographic_basis(mesh)

    p_darcy = uw.discretisation.MeshVariable(
        "DarcyP", mesh, 1, degree=2, varsymbol=r"P_d"
    )
    v_darcy = uw.discretisation.MeshVariable(
        "DarcyU", mesh, mesh.dim, degree=1, continuous=True, varsymbol=r"V_d"
    )

    darcy = uw.systems.SteadyStateDarcy(mesh, h_Field=p_darcy, v_Field=v_darcy)
    darcy.constitutive_model = uw.constitutive_models.DarcyFlowModel
    darcy.constitutive_model.Parameters.permeability = permeability.sym[0]

    # p = 0 at upper surface
    darcy.add_essential_bc(0.0, "Upper")

    # Source at depth: drives fluid upward from below source_depth_km
    r_source = 1 - config.source_depth_km / 6370
    darcy.f = sympy.Piecewise(
        (config.source_magnitude, r < r_source),
        (0.0, True),
    )

    # Gravity term
    darcy.constitutive_model.Parameters.s = sympy.Matrix(
        -unit_vertical * config.gravity_scale
    )

    darcy.tolerance = 1.0e-3
    darcy._v_projector.tolerance = 1.0e-3
    darcy._v_projector.smoothing = 0.0

    uw.pprint(0, "Solving Darcy flow...")
    darcy.solve()
    uw.pprint(0, "Darcy solve complete")

    return darcy, v_darcy, p_darcy


@workflow_step(
    description="Advect tracers to surface and compute accumulation",
    produces=["accumulation"],
    requires=["adapted_mesh", "darcy_velocity"],
)
def advect_tracers(mesh, v_darcy, config: H2ExConfig):
    """Seed tracers at depth, advect in -v_darcy, compute surface density.

    Tracers are seeded on a regular grid in (lon, lat, depth) space,
    converted to Cartesian, and advected using midpoint integration
    in the negative Darcy velocity.  When a tracer reaches the surface
    (r > 1), it is stopped and projected to the accumulation depth.

    The density is computed via KD-tree kernel density estimation on a
    Delaunay triangulation of the upper surface.

    Returns
    -------
    accumulation : dict
        Dictionary with keys:

        - ``"grid_xyz"``: final tracer positions (N, 3)
        - ``"grid_xyz_0"``: initial tracer positions (N, 3)
        - ``"original_depths"``: initial depths in km (N,)
        - ``"density"``: surface density array (n_cells,)
        - ``"surface_mesh"``: pyvista PolyData of surface triangulation
    """
    import underworld3 as uw
    import mpi4py

    unit_vertical, _, _, r, _, _ = _geographic_basis(mesh)

    # --- Seed tracer grid ---
    lon_min, lon_max = config.lon_range
    lat_min, lat_max = config.lat_range
    d_min, d_max = 1.0, config.depth_range_km[1]

    lons = np.linspace(lon_min, lon_max, config.tracer_grid_n)
    lats = np.linspace(lat_min, lat_max, config.tracer_grid_n)
    depths = np.linspace(d_min, d_max, config.tracer_n_depths)

    grid_lons, grid_lats, grid_depths = np.meshgrid(lons, lats, depths)
    grid_llz = np.column_stack([
        grid_lons.ravel(),
        grid_lats.ravel(),
        grid_depths.ravel(),
    ])

    # Convert (lon, lat, depth_km) → Cartesian (x, y, z) on unit sphere
    grid_r = 1.0 - grid_llz[:, 2] / 6370
    grid_lat_rad = np.radians(grid_llz[:, 1])
    grid_lon_rad = np.radians(grid_llz[:, 0])

    grid_xyz = np.column_stack([
        grid_r * np.cos(grid_lat_rad) * np.cos(grid_lon_rad),
        grid_r * np.cos(grid_lat_rad) * np.sin(grid_lon_rad),
        grid_r * np.sin(grid_lat_rad),
    ])

    grid_xyz_0 = grid_xyz.copy()
    original_depths = (1.0 - grid_r) * 6370

    # --- Compute advection timestep ---
    vel = uw.function.evaluate(v_darcy.sym, mesh._centroids, rbf=True)
    max_magvel = np.linalg.norm(vel, axis=1).max()
    max_magvel_glob = uw.mpi.comm.allreduce(max_magvel, op=mpi4py.MPI.MAX)
    min_dx = mesh.get_min_radius()
    dt_adv = 10 * min_dx / max_magvel_glob

    r_cutoff = 1.0 - config.accumulation_depth_km / 6370

    # --- Midpoint advection ---
    mask = np.ones((grid_xyz.shape[0], 1), dtype=int)
    counter = 0

    while np.count_nonzero(mask) > 1 and counter < config.max_advection_steps:
        if counter % 50 == 0:
            uw.pprint(
                0,
                f"Advection step {counter}, "
                f"active tracers: {np.count_nonzero(mask)}",
            )

        v_xyz = uw.function.evaluate(-v_darcy.sym, grid_xyz, rbf=True)
        xyz_half = 0.5 * dt_adv * v_xyz + grid_xyz
        vh_xyz = uw.function.evaluate(-v_darcy.sym, xyz_half, rbf=True)
        grid_xyz = mask * dt_adv * vh_xyz + grid_xyz

        r_new = np.sqrt(np.sum(grid_xyz**2, axis=1))
        mask[r_new > 1] = 0
        hit_surface = r_new > 1
        if np.any(hit_surface):
            grid_xyz[hit_surface] *= r_cutoff / r_new[hit_surface].reshape(-1, 1)

        counter += 1

    uw.pprint(0, f"Advection complete after {counter} steps")

    # --- Surface density estimation ---
    result = {
        "grid_xyz": grid_xyz,
        "grid_xyz_0": grid_xyz_0,
        "original_depths": original_depths,
    }

    try:
        import pyvista as pv

        # Build surface triangulation from upper boundary
        upper_points = uw.discretisation.petsc_dm_find_labeled_points_local(
            mesh.dm, "UW_Boundaries", 2
        )
        upper_polydata = pv.PolyData(mesh.data[upper_points])
        upper_tri = upper_polydata.delaunay_2d(offset=0.01)
        upper_tri = upper_tri.compute_cell_sizes(length=False, volume=False)

        # KD-tree density
        kdtree = uw.kdtree.KDTree(upper_tri.cell_centers().points)
        dist_sq, nearest = kdtree.query(
            np.ascontiguousarray(grid_xyz), k=25, sqr_dists=True,
        )

        density = np.zeros(upper_tri.number_of_cells)
        for j in range(nearest.shape[1]):
            for i in range(nearest.shape[0]):
                density[nearest[i, j]] += 1.0 / np.sqrt(dist_sq[i, j])

        # Normalize by cell area
        cell_area = np.asarray(upper_tri.cell_data["Area"]).ravel()
        density /= np.maximum(cell_area, 1e-30)
        density /= np.maximum(density.max(), 1e-30)

        result["density"] = density
        result["surface_mesh"] = upper_tri

        uw.pprint(0, f"Surface density computed ({upper_tri.number_of_cells} cells)")

    except (ImportError, Exception) as e:
        uw.pprint(0, f"Surface density computation skipped: {e}")

    return result


@workflow_step(
    description="Visualize Darcy flow with fault surfaces",
    requires=["adapted_mesh", "darcy_velocity"],
)
def plot_results(mesh, v_darcy, surfaces=None, config=None, save_to=None):
    """Plot the Darcy velocity magnitude with optional fault surface overlay.

    Parameters
    ----------
    mesh : Mesh
        The simulation mesh.
    v_darcy : MeshVariable
        Darcy velocity field.
    surfaces : SurfaceCollection, optional
        Fault surfaces to overlay.
    config : H2ExConfig, optional
        Configuration for title/output.
    save_to : str or Path, optional
        If given, save screenshot instead of interactive display.

    Returns
    -------
    plotter or None
    """
    import underworld3 as uw

    if uw.mpi.size > 1:
        return None

    try:
        import pyvista as pv
        import underworld3.visualisation as vis

        pv_mesh = vis.mesh_to_pv_mesh(mesh)
        vel = vis.vector_fn_to_pv_points(pv_mesh, v_darcy.sym)
        pv_mesh.point_data["Vd_mag"] = np.linalg.norm(vel, axis=1)
        pv_mesh.point_data["Vd"] = vel

        pl = pv.Plotter(window_size=(1200, 800))

        pl.add_mesh(
            pv_mesh,
            scalars="Vd_mag",
            cmap="viridis",
            show_edges=False,
            opacity=0.3,
        )

        pl.add_arrows(
            pv_mesh.points, -vel, mag=5e-3,
            show_scalar_bar=False, opacity=0.5,
        )

        # Overlay fault surfaces
        if surfaces is not None:
            for name, surf in surfaces.surfaces.items():
                if hasattr(surf, "_pv_mesh") and surf._pv_mesh is not None:
                    pl.add_mesh(
                        surf._pv_mesh,
                        color="red",
                        opacity=0.5,
                        label=name,
                    )

        title = "H2Ex Darcy Flow"
        if config is not None:
            title += f" (azimuth={config.stress_azimuth_deg}°)"
        pl.add_title(title)

        if save_to:
            pl.screenshot(str(save_to), window_size=(1280, 960), return_img=False)
            pl.close()
        else:
            pl.show()

        return pl

    except (ImportError, Exception):
        return None
