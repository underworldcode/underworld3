"""H2Ex fault-controlled flow — configuration and workflow steps.

Demonstrates the **product-aware workflow pattern** where expensive
serial steps (mesh creation, fault loading, adaptation) produce named
products that can be saved and reloaded for parameter studies.

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
    products.save("fault_surfaces", surfaces)

    # Or reload products for a parameter study
    mesh = products.load("adapted_mesh")
    surfaces = products.load("fault_surfaces", mesh=mesh)

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

    # Rheology
    rheology: str = Field(
        default="anisotropic",
        description="Rheology type: isoviscous or anisotropic",
    )
    eta_0_Pa_s: float = Field(
        default=1e21, gt=0, description="Background viscosity (Pa s)"
    )
    eta_1_ratio: float = Field(
        default=0.1, gt=0, description="Fault-zone viscosity ratio (eta_1/eta_0)"
    )
    fault_width_km: float = Field(
        default=10.0, gt=0, description="Fault influence width (km)"
    )

    # Boundary conditions
    driving_velocity_cm_yr: float = Field(
        default=1.0, description="Driving velocity (cm/yr)"
    )
    penalty_factor: float = Field(
        default=1.0e6, gt=0, description="BC penalty factor"
    )

    # Output
    output_dir: str = "output/h2ex"


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
    # Metric is 1/h² (edge-length based)
    metric.data[:, 0] = 1.0 / target_h**2

    uw.pprint(0, "Adapting mesh...")
    mesh.adapt(metric)
    uw.pprint(0, f"Adapted mesh: {mesh.elements_count} elements")

    return mesh


@workflow_step(
    description="Create Stokes solver on the mesh",
    produces=["stokes"],
    requires=["adapted_mesh"],
)
def create_stokes(mesh, config: H2ExConfig):
    """Create a Stokes solver with velocity and pressure variables.

    Returns
    -------
    stokes, v, p
        The Stokes solver and its velocity / pressure variables.
    """
    import underworld3 as uw

    v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)

    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.tolerance = 1.0e-3

    return stokes, v, p


@workflow_step(
    description="Configure fault-controlled rheology",
    produces=["fault_fields"],
    requires=["stokes", "fault_surfaces"],
)
def setup_rheology(stokes, surfaces, mesh, config: H2ExConfig):
    """Set up isoviscous or anisotropic fault-controlled rheology.

    For anisotropic mode, creates a MeshVariable for fault influence
    and configures viscosity reduction near fault surfaces.

    Returns
    -------
    fields : dict
        Dictionary of created fields (e.g. ``{"fault_influence": var}``).
        Empty dict for isoviscous mode.
    """
    import underworld3 as uw

    fields = {}

    if config.rheology == "isoviscous" or len(surfaces.surfaces) == 0:
        stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
        uw.pprint(0, "Rheology: isoviscous")
        return fields

    # Anisotropic: reduce viscosity near faults
    dist_var = surfaces.compute_distance_field(mesh)

    # Gaussian influence function
    fault_influence = uw.discretisation.MeshVariable(
        "fault_influence", mesh, 1, degree=2
    )
    sigma = config.fault_width_km
    fault_influence.data[:, 0] = np.exp(
        -0.5 * (dist_var.data[:, 0] / sigma) ** 2
    )
    fields["fault_influence"] = fault_influence

    # Effective viscosity: eta = eta_0 * (1 - (1-ratio) * influence)
    ratio = config.eta_1_ratio
    eta_eff = 1.0 - (1.0 - ratio) * fault_influence.sym[0]
    stokes.constitutive_model.Parameters.shear_viscosity_0 = eta_eff

    uw.pprint(0, f"Rheology: anisotropic (ratio={ratio}, width={sigma} km)")
    return fields


@workflow_step(
    description="Apply boundary conditions",
    requires=["stokes"],
)
def set_boundary_conditions(stokes, mesh, config: H2ExConfig):
    """Apply geographic penalty BCs to the Stokes solver.

    Free-slip on top/bottom, driving velocity on sides.
    """
    # Free-slip top and bottom
    stokes.add_dirichlet_bc((sympy.oo, sympy.oo, 0.0), "Top")
    stokes.add_dirichlet_bc((sympy.oo, sympy.oo, 0.0), "Bottom")

    # Simple side BCs — no-normal-flow
    stokes.add_dirichlet_bc((0.0, sympy.oo, sympy.oo), "Left")
    stokes.add_dirichlet_bc((0.0, sympy.oo, sympy.oo), "Right")
    stokes.add_dirichlet_bc((sympy.oo, 0.0, sympy.oo), "Front")
    stokes.add_dirichlet_bc((sympy.oo, 0.0, sympy.oo), "Back")


@workflow_step(
    description="Visualize results with fault surfaces",
    requires=["stokes", "fault_fields"],
)
def plot_results(mesh, stokes, surfaces=None, config=None, save_to=None):
    """Plot the velocity magnitude with optional fault surface overlay.

    Parameters
    ----------
    mesh : Mesh
        The simulation mesh.
    stokes : Stokes
        Solved Stokes system.
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

        v = stokes.u
        pv_mesh = vis.mesh_to_pv_mesh(mesh)
        vel = vis.vector_fn_to_pv_points(pv_mesh, v.sym)
        pv_mesh.point_data["velocity_mag"] = np.linalg.norm(vel, axis=1)

        pl = pv.Plotter(window_size=(1200, 800))
        pl.add_mesh(
            pv_mesh,
            scalars="velocity_mag",
            cmap="viridis",
            show_edges=False,
        )

        # Overlay fault surfaces
        if surfaces is not None:
            for name, surf in surfaces.surfaces.items():
                if hasattr(surf, "pv_mesh") and surf.pv_mesh is not None:
                    pl.add_mesh(
                        surf.pv_mesh,
                        color="red",
                        opacity=0.5,
                        label=name,
                    )

        title = "H2Ex Fault Flow"
        if config is not None:
            title += f" ({config.rheology})"
        pl.add_title(title)

        if save_to:
            pl.screenshot(str(save_to), window_size=(1280, 960), return_img=False)
            pl.close()
        else:
            pl.show()

        return pl

    except (ImportError, Exception):
        return None
