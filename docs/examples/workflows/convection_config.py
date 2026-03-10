"""Thermal convection workflow — configuration and helpers.

Demonstrates the **workflow package pattern** described in
``docs/developer/guides/workflow-packages.md``.  This module defines
all the parameters and setup functions for Rayleigh–Bénard convection
in a Cartesian box.

In a standalone package this would be the pip-installable library;
the companion notebook (``convection_notebook.py``) is the clean
user-facing script.

Usage::

    import convection_config as convection

    config = convection.ConvectionConfig()
    model  = config.setup_model()
    mesh   = convection.create_mesh(config)
    stokes, v, p = convection.create_stokes(mesh, config)
    adv_diff, T  = convection.create_advdiff(mesh, v, config)
    convection.set_initial_temperature(T, mesh, config)

    # Inspect the workflow
    convection.view()              # list all steps
    convection.create_mesh.view()  # source of a single step
"""

import numpy as np
import sympy
from pydantic import Field

from underworld3.workflows import WorkflowConfig, workflow_step


def view():
    """Display the workflow steps and config classes in this module."""
    import sys
    from underworld3.workflows import view as _wf_view

    _wf_view(sys.modules[__name__])


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ConvectionConfig(WorkflowConfig):
    """Parameters for constant-viscosity Rayleigh–Bénard convection.

    Non-dimensional by default — all reference quantities are unity so
    the Rayleigh number is the single controlling parameter.  Set
    ``ref_length``, ``ref_viscosity``, etc. to switch to dimensional mode.
    """

    workflow_name: str = "rayleigh_benard"
    description: str = "Constant viscosity thermal convection in a Cartesian box"

    # Mesh
    cellsize: float = Field(default=1.0 / 16, gt=0, description="Mesh cell size")
    aspect_ratio: float = Field(default=1.0, gt=0, description="Domain width / height")
    qdegree: int = Field(default=3, ge=1, description="Quadrature degree")

    # Physics
    rayleigh: float = Field(default=1e6, gt=0, description="Rayleigh number")
    viscosity: float = Field(default=1.0, gt=0, description="Reference viscosity")
    diffusivity: float = Field(default=1.0, gt=0, description="Thermal diffusivity")

    # Boundary conditions
    T_top: float = Field(default=0.0, description="Top temperature")
    T_bottom: float = Field(default=1.0, description="Bottom temperature")

    # Time stepping
    n_steps: int = Field(default=50, ge=0, description="Number of time steps")
    dt_factor: float = Field(default=2.0, gt=0, description="CFL multiplier for dt")

    # Output
    output_dir: str = "output/convection"
    save_every: int = Field(default=10, ge=1, description="Save output every N steps")


# ---------------------------------------------------------------------------
# Helper functions — each returns standard UW3 objects
# ---------------------------------------------------------------------------


@workflow_step(
    description="Build a 2D unstructured simplex mesh",
    produces=["mesh"],
)
def create_mesh(config: ConvectionConfig):
    """Build a 2D unstructured simplex mesh.

    Returns
    -------
    mesh
    """
    import underworld3 as uw

    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(config.aspect_ratio, 1.0),
        cellSize=config.cellsize,
        regular=False,
        qdegree=config.qdegree,
    )
    return mesh


@workflow_step(
    description="Configure Stokes solver with constant viscosity",
    produces=["stokes"],
    requires=["mesh"],
)
def create_stokes(mesh, config: ConvectionConfig):
    """Configure a Stokes solver with constant viscosity and free-slip walls.

    Buoyancy is coupled to a temperature variable ``T`` which must be
    created by ``create_advdiff`` before solving.

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
    stokes.constitutive_model.Parameters.shear_viscosity_0 = config.viscosity
    stokes.tolerance = 1.0e-3

    # Free-slip on all walls
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Bottom")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Top")
    stokes.add_dirichlet_bc((0.0, sympy.oo), "Left")
    stokes.add_dirichlet_bc((0.0, sympy.oo), "Right")

    return stokes, v, p


@workflow_step(
    description="Configure advection-diffusion solver for temperature",
    produces=["adv_diff"],
    requires=["mesh", "stokes"],
)
def create_advdiff(mesh, v, config: ConvectionConfig):
    """Configure the advection–diffusion solver for temperature.

    Returns
    -------
    adv_diff, T
        The thermal solver and its temperature variable.
    """
    import underworld3 as uw

    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=3)

    adv_diff = uw.systems.AdvDiffusionSLCN(mesh, u_Field=T, V_fn=v)
    adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel
    adv_diff.constitutive_model.Parameters.diffusivity = config.diffusivity
    adv_diff.theta = 0.5

    # Dirichlet: hot bottom, cold top
    adv_diff.add_dirichlet_bc(config.T_bottom, "Bottom")
    adv_diff.add_dirichlet_bc(config.T_top, "Top")

    return adv_diff, T


@workflow_step(
    description="Wire buoyancy force into the Stokes solver",
    requires=["stokes", "adv_diff"],
)
def set_buoyancy(stokes, T, config: ConvectionConfig):
    """Wire the buoyancy force into the Stokes solver.

    Must be called after both ``create_stokes`` and ``create_advdiff``
    so that ``T`` exists.
    """
    stokes.bodyforce = sympy.Matrix([0, config.rayleigh * T.sym[0]])


@workflow_step(
    description="Set initial temperature: linear gradient + perturbation",
    requires=["mesh", "adv_diff"],
)
def set_initial_temperature(T, mesh, config: ConvectionConfig):
    """Set a linear gradient with a sinusoidal perturbation.

    T(x, y) = T_bottom + (T_top - T_bottom) * y
              + 0.2 * sin(pi * x / aspect) * sin(pi * y)
    """
    import underworld3 as uw

    x, y = mesh.X
    dT = config.T_top - config.T_bottom
    init_t = (
        config.T_bottom
        + dT * y
        + 0.2 * sympy.sin(sympy.pi * x / config.aspect_ratio) * sympy.sin(sympy.pi * y)
    )
    T.array[...] = uw.function.evaluate(init_t, T.coords)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


@workflow_step(
    description="Plot temperature field with optional streamlines",
    requires=["mesh", "adv_diff", "stokes"],
)
def plot_temperature(mesh, T, v=None, config=None, title=None, save_to=None):
    """Plot the temperature field with optional velocity streamlines.

    Parameters
    ----------
    mesh : uw mesh
        The simulation mesh.
    T : MeshVariable
        Temperature field.
    v : MeshVariable, optional
        Velocity field — adds streamlines if provided.
    config : ConvectionConfig, optional
        Used for the title if *title* is not given.
    title : str, optional
        Plot title.  Defaults to ``"Ra = ... "`` from *config*.
    save_to : str or Path, optional
        If given, save a screenshot instead of showing interactively.

    Returns
    -------
    pv.Plotter or None
        The plotter (for further customisation) or None if
        visualisation is unavailable.
    """
    import underworld3 as uw

    if uw.mpi.size > 1:
        return None

    try:
        import pyvista as pv
        import underworld3.visualisation as vis

        import os, matplotlib  # noqa: E401
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            matplotlib.use("Agg")

        # Temperature on the high-order mesh
        pv_mesh_t = vis.meshVariable_to_pv_mesh_object(T)
        pv_mesh_t.point_data["T"] = vis.scalar_fn_to_pv_points(pv_mesh_t, T.sym)

        pl = pv.Plotter(window_size=(1000, 750))
        pl.add_mesh(pv_mesh_t, cmap="coolwarm", show_edges=False, scalars="T", opacity=1)
        pl.add_mesh(pv_mesh_t.copy(), style="wireframe", color="Black", opacity=0.05)

        # Streamlines from velocity field
        if v is not None:
            pvmesh = vis.mesh_to_pv_mesh(mesh)
            pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v.sym) / 333

            cpoints = np.zeros((mesh._centroids[::4].shape[0], 3))
            cpoints[:, 0] = mesh._centroids[::4, 0]
            cpoints[:, 1] = mesh._centroids[::4, 1]

            pvstream = pvmesh.streamlines_from_source(
                pv.PolyData(cpoints),
                vectors="V",
                integrator_type=45,
                integration_direction="forward",
                compute_vorticity=False,
                max_steps=25,
                surface_streamlines=True,
            )
            pl.add_mesh(pvstream, opacity=0.666)

        if title is None and config is not None:
            title = f"Ra = {config.rayleigh:.0e}"
        if title:
            pl.add_title(title)

        if save_to:
            pl.screenshot(str(save_to), window_size=(1280, 1280), return_img=False)
            pl.close()
        else:
            pl.show(cpos="xy")

        return pl

    except (ImportError, Exception):
        return None
