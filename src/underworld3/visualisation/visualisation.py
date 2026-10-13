"""PyVista visualization helper routines for Underworld3.

This module provides functions to convert Underworld meshes, swarms,
and variables to PyVista objects for interactive 3D visualization.
"""
import os
import underworld3 as uw


def initialise(jupyter_backend):
    """Initialize PyVista with Underworld-friendly defaults.

    Sets up PyVista global theme with white background, anti-aliasing,
    and appropriate Jupyter backend.

    Parameters
    ----------
    jupyter_backend : str or None
        Jupyter backend to use ('trame', 'client', 'panel', etc.).
        If None, auto-detects based on environment.
    """
    import pyvista as pv

    pv.global_theme.background = "white"
    pv.global_theme.anti_aliasing = "msaa"
    pv.global_theme.smooth_shading = True
    pv.global_theme.camera["viewup"] = [0.0, 1.0, 0.0]
    pv.global_theme.camera["position"] = [0.0, 0.0, 5.0]

    # Check if we're in a remote Jupyter environment (binder, JupyterHub, etc.)
    is_remote = (
        "BINDER_LAUNCH_HOST" in os.environ
        or "BINDER_REPO_URL" in os.environ
        or "JUPYTERHUB_SERVICE_PREFIX" in os.environ
    )

    try:
        if jupyter_backend is not None:
            pv.global_theme.jupyter_backend = jupyter_backend
        else:
            # Use trame backend for all Jupyter environments
            pv.global_theme.jupyter_backend = "trame"

        # Configure trame for remote environments (binder, JupyterHub)
        # Enable server proxy but let PyVista auto-detect the prefix
        if is_remote:
            pv.global_theme.trame.server_proxy_enabled = True
            # Don't set prefix - let PyVista/trame auto-detect from environment

    except RuntimeError:
        pv.global_theme.jupyter_backend = "panel"

    return


def mesh_to_pv_mesh(mesh, jupyter_backend=None):
    """Convert Underworld mesh to PyVista unstructured grid.

    Parameters
    ----------
    mesh : Mesh
        Underworld mesh to convert.
    jupyter_backend : str, optional
        PyVista Jupyter backend to use.

    Returns
    -------
    pyvista.UnstructuredGrid
        PyVista mesh with unit metadata attached.
    """

    # # Required in notebooks
    # import nest_asyncio
    # nest_asyncio.apply()

    initialise(jupyter_backend)

    # import os
    # import shutil
    # import tempfile
    import pyvista as pv
    import numpy as np

    # with tempfile.TemporaryDirectory() as tmp:
    #     if type(mesh) == str:  # reading msh file directly
    #         vtk_filename = os.path.join(tmp, "tmpMsh.msh")
    #         shutil.copyfile(mesh, vtk_filename)
    #     else:  # reading mesh by creating vtk
    #         vtk_filename = os.path.join(tmp, "tmpMsh.vtk")
    #         mesh.vtk(vtk_filename)

    #     pvmesh = pv.read(vtk_filename)

    # return pvmesh

    ## Alternative - not via file / create an unstructured grid in pyvista

    from petsc4py import PETSc

    cStart, cEnd = mesh.dm.getHeightStratum(0)
    fStart, fEnd = mesh.dm.getHeightStratum(1)
    pStart, pEnd = mesh.dm.getDepthStratum(0)

    cell_num_points = mesh.element.entities[mesh.dim]
    face_num_points = mesh.element.face_entities[mesh.dim]

    cell_points_list = []
    for cell_id in range(cStart, cEnd):
        cell_points = mesh.dm.getTransitiveClosure(cell_id)[0][-cell_num_points:]
        cell_points_list.append(cell_points - pStart)

    # PETSc DMPlex uses opposite winding to VTK for 3D cells.
    # Reorder vertices to match VTK convention (right-handed orientation)
    # so that face normals, shading, clipping, and back-face culling work correctly.
    #
    # Hexahedra: PETSc winds bottom face CCW from above, VTK expects CW from above.
    #   Fix: swap vertices 1 and 3 on the bottom face → [0,3,2,1,4,5,6,7]
    # Tetrahedra: PETSc uses left-handed orientation, VTK expects right-handed.
    #   Fix: swap vertices 1 and 2 → [0,2,1,3]
    if mesh.dim == 3:
        if mesh.dm.isSimplex():
            tet_reorder = [0, 2, 1, 3]
            cell_points_list = [pts[tet_reorder] for pts in cell_points_list]
        else:
            hex_reorder = [0, 3, 2, 1, 4, 5, 6, 7]
            cell_points_list = [pts[hex_reorder] for pts in cell_points_list]

    try:
        import meshio

        match (mesh.dm.isSimplex(), mesh.dim):
            case (True, 2):
                meshio_cell_type = "triangle"
            case (True, 3):
                meshio_cell_type = "tetra"
            case (False, 2):
                meshio_cell_type = "quad"
            case (False, 3):
                meshio_cell_type = "hexahedron"

        # Use non-dimensional [0-1] coordinates for PyVista
        # PyVista only needs coordinates for spatial positioning (visualization)
        # evaluate() expects non-dimensional coords to query PETSc KDTrees
        mesh_coordinates_nd = np.asarray(mesh.X.coords, dtype=np.float64)

        # Store units metadata for labeling and axis annotation
        mesh_units = mesh.units if mesh.units is not None else uw.units.dimensionless

        mmesh = meshio.Mesh(
            points=mesh_coordinates_nd, cells=[(meshio_cell_type, np.array(cell_points_list))]
        )

        pv_mesh = pv.from_meshio(mmesh)
        pv_mesh._units = mesh_units
        # Store original coordinate array for proper evaluation
        pv_mesh._coord_array = mesh.X.coords

        return pv_mesh

    except ImportError:

        match (mesh.dm.isSimplex(), mesh.dim):
            case (True, 2):
                vtk_cell_type = pv.cell.CellType.TRIANGLE
            case (True, 3):
                vtk_cell_type = pv.cell.CellType.TETRA
            case (False, 2):
                vtk_cell_type = pv.cell.CellType.QUAD
            case (False, 3):
                vtk_cell_type = pv.cell.CellType.HEXAHEDRON

        cells_array = np.array(cell_points_list, dtype=int)
        cells_size = np.full((cells_array.shape[0], 1), cell_num_points, dtype=int)
        cells_type = np.full((cells_array.shape[0], 1), vtk_cell_type, dtype=int)

        cells_array = np.hstack((cells_size, cells_array), dtype=int)

        # Use non-dimensional [0-1] coordinates for PyVista (see meshio path above)
        mesh_coordinates_nd = np.asarray(mesh.X.coords, dtype=np.float64)
        pv_mesh = pv.UnstructuredGrid(cells_array, cells_type, coords_to_pv_coords(mesh_coordinates_nd))

        # Store units metadata for labeling
        pv_mesh._units = mesh.units if mesh.units is not None else uw.units.dimensionless
        # Store original coordinate array for proper evaluation
        pv_mesh._coord_array = mesh.X.coords

        return pv_mesh


def labelled_facets_to_pv_mesh(mesh, name):
    """The facets carrying a boundary label, as a PyVista object of their own.

    An embedded surface — a conforming fault, a material interface — is a set of
    facets *inside* the mesh, so drawing it with the mesh hides it: in 2-D it is
    a few lines among thousands, and in 3-D the surrounding elements occlude it
    entirely. Returned separately it can be drawn as a wireframe over a
    transparent or clipped mesh, and saved to ``.vtp`` for interactive viewing.

    The result is dimension-general because a labelled facet's closure gives its
    vertices whatever the dimension: two in 2-D (a line segment), three in 3-D
    (a triangle).

    Parameters
    ----------
    mesh : Mesh
        The mesh carrying the label.
    name : str
        A boundary name, normally one added by
        :meth:`~underworld3.discretisation.Mesh.add_conforming_surface`.

    Returns
    -------
    pyvista.PolyData
        Lines in 2-D, triangles in 3-D. Empty if this rank owns no part of the
        surface, which is normal in parallel.

    Examples
    --------
    >>> fault = vis.labelled_facets_to_pv_mesh(cut, "Fault")
    >>> pl.add_mesh(vis.mesh_to_pv_mesh(cut), style="wireframe",
    ...             color="lightgrey", opacity=0.3)
    >>> pl.add_mesh(fault, color="red", line_width=3)
    >>> fault.save("fault.vtp")            # open in ParaView or pv.read()
    """
    import numpy as np
    import pyvista as pv

    if name not in [b.name for b in mesh.boundaries]:
        raise ValueError(
            f"{name!r} is not a boundary of this mesh; have "
            f"{[b.name for b in mesh.boundaries]}")

    dm = mesh.dm
    vS, vE = dm.getDepthStratum(0)
    X = np.asarray(dm.getCoordinatesLocal().array).reshape(
        -1, dm.getCoordinateDim())

    value = mesh.boundaries[name].value
    label = dm.getLabel(name)
    # An empty stratum yields a null IS that segfaults in getIndices(), and a
    # rank owning no part of the surface is the normal case in parallel.
    if label is None or label.getStratumSize(value) == 0:
        return pv.PolyData()

    facets = [
        [int(p) - vS for p in dm.getTransitiveClosure(int(f))[0] if vS <= p < vE]
        for f in label.getStratumIS(value).getIndices()
    ]
    used = sorted({v for facet in facets for v in facet})
    remap = {v: i for i, v in enumerate(used)}
    points = _vector_to_pv_vector(X[used])

    cells = np.hstack([[len(f)] + [remap[v] for v in f] for f in facets])
    out = pv.PolyData(points)
    if all(len(f) == 2 for f in facets):
        out.lines = cells
    else:
        out.faces = cells
    return out


def coords_to_pv_coords(coords):
    """Convert coordinate array to PyVista-compatible 3D coordinates.

    Parameters
    ----------
    coords : numpy.ndarray
        Coordinate array of shape ``(n, 2)`` or ``(n, 3)``.

    Returns
    -------
    numpy.ndarray
        3D coordinate array of shape ``(n, 3)``.
    """
    return _vector_to_pv_vector(coords)


def _vector_to_pv_vector(vector):
    """Convert numpy coordinate array to pyvista compatible array"""

    import numpy as np

    if vector.shape[1] == 3:
        return vector
    else:
        vector3 = np.zeros((vector.shape[0], 3))
        vector3[:, 0:2] = vector[:]

        return vector3


def swarm_to_pv_cloud(swarm):
    """Convert swarm particle positions to PyVista point cloud.

    Parameters
    ----------
    swarm : Swarm
        Underworld swarm object.

    Returns
    -------
    pyvista.PolyData
        Point cloud with particle positions.
    """
    import numpy as np
    import pyvista as pv

    points = np.zeros((swarm.local_size, 3))
    points[:, 0] = swarm.data[:, 0]
    points[:, 1] = swarm.data[:, 1]
    if swarm.mesh.dim == 2:
        points[:, 2] = 0.0
    else:
        points[:, 2] = swarm.data[:, 2]

    point_cloud = pv.PolyData(points)

    return point_cloud


def meshVariable_to_pv_cloud(meshVar):
    """Convert mesh variable node positions to PyVista point cloud.

    Parameters
    ----------
    meshVar : MeshVariable
        Underworld mesh variable.

    Returns
    -------
    pyvista.PolyData
        Point cloud at mesh variable nodal locations.
    """

    import numpy as np
    import pyvista as pv
    import underworld3 as uw

    # Get coordinates from the mesh variable
    # These may be dimensional (UnitAwareArray with meters) when units are active
    # The alpha parameter in meshVariable_to_pv_mesh_object is now computed from
    # these coordinates directly, ensuring scale consistency
    coords = np.asarray(meshVar.coords, dtype=np.float64)

    points = np.zeros((coords.shape[0], 3))
    points[:, 0] = coords[:, 0]
    points[:, 1] = coords[:, 1]

    if meshVar.mesh.dim == 2:
        points[:, 2] = 0.0
    else:
        points[:, 2] = coords[:, 2]

    point_cloud = pv.PolyData(points)

    # Store units metadata for labeling (same as mesh_to_pv_mesh)
    point_cloud._units = meshVar.mesh.units if meshVar.mesh.units is not None else uw.units.dimensionless
    # Store original coordinate array for proper evaluation
    point_cloud._coord_array = meshVar.coords

    return point_cloud


def meshVariable_to_native_pv_mesh(meshVar):
    """The mesh's OWN cells, renumbered so point ``i`` is the variable's DOF ``i``.

    Returns ``None`` when the variable's degrees of freedom are not the mesh
    vertices — a higher-order or discontinuous field — in which case there is no
    native triangulation carrying it and the caller must fall back.

    Why this exists
    ---------------
    A **continuous P1** field has exactly one degree of freedom per vertex, so
    the triangulation that carries it already exists in the DM. Re-deriving it
    with ``delaunay_2d`` is not merely redundant, it is lossy on a graded mesh:
    Delaunay is a property of the point set alone, so it neither knows nor
    respects which triangles the mesh actually has, and the ``alpha`` filter —
    one length for the whole domain — **deletes** cells whose circumradius
    exceeds it. On an adapted mesh those are precisely the coarse cells. Measured
    on a fault mesh graded 8:1, 361 of 11610 cells were dropped, and they render
    as blank holes in the middle of the field.

    The points are returned in the VARIABLE's DOF order rather than the DM's
    vertex order, so the documented pattern

    >>> pvm = vis.meshVariable_to_pv_mesh_object(T)
    >>> pvm.point_data["T"] = np.asarray(T.data[:, 0])

    keeps working unchanged. Getting that backwards would draw the right mesh
    with the values shuffled, which looks like noise rather than like an error.
    """
    import numpy as np
    import pyvista as pv
    from scipy.spatial import cKDTree

    mesh = meshVar.mesh
    dim = mesh.dim
    pvm = mesh_to_pv_mesh(mesh)

    coords = np.asarray(meshVar.coords, dtype=np.float64)
    if coords.shape[0] != pvm.n_points:
        return None                      # not one DOF per vertex

    pts = np.asarray(pvm.points, dtype=np.float64)[:, :dim]
    dist, dof_of_point = cKDTree(coords[:, :dim]).query(pts)
    extent = float(np.ptp(pts)) or 1.0
    if dist.max() > 1.0e-8 * extent:
        return None                      # coincident in count but not in place

    # Renumber the connectivity into DOF order, and hand back the variable's own
    # coordinates as the points so the two are aligned by construction.
    try:
        conn = np.asarray(pvm.cell_connectivity)
        offsets = np.asarray(pvm.offset)
    except AttributeError:               # older pyvista
        return None
    sizes = np.diff(offsets)
    if not len(sizes) or (sizes != sizes[0]).any():
        return None                      # mixed cell types: not worth the risk
    cells = np.column_stack(
        [np.full(len(sizes), sizes[0]),
         dof_of_point[conn].reshape(len(sizes), sizes[0])]).ravel()

    points = np.zeros((coords.shape[0], 3))
    points[:, :dim] = coords[:, :dim]
    native = pv.UnstructuredGrid(cells, np.asarray(pvm.celltypes), points)
    for attr in ("_units", "_coord_array"):
        if hasattr(pvm, attr):
            setattr(native, attr, getattr(pvm, attr))
    native._coord_array = meshVar.coords
    return native


def meshVariable_to_pv_mesh_object(meshVar, alpha=None):
    """Convert a mesh variable to a PyVista mesh carrying its nodal points.

    Uses the mesh's **own** triangulation when the variable's degrees of freedom
    are the mesh vertices (a continuous P1 field) — see
    :func:`meshVariable_to_native_pv_mesh`, which also explains why the Delaunay
    route silently drops coarse cells on an adapted mesh.

    Otherwise the points are Delaunay-triangulated, which is what higher-order
    and discontinuous variables need: the base mesh does not carry their DOFs.

    Parameters
    ----------
    meshVar : MeshVariable
        Underworld mesh variable.
    alpha : float, optional
        Alpha parameter for Delaunay triangulation. If None, computed
        automatically from coordinate range. Ignored on the native path.

    Returns
    -------
    pyvista.UnstructuredGrid
        Mesh through the variable's nodal points, in the variable's DOF order.
    """
    import numpy as np

    mesh = meshVar.mesh
    dim = mesh.dim

    if alpha is None:
        native = meshVariable_to_native_pv_mesh(meshVar)
        if native is not None:
            return native

    point_cloud = meshVariable_to_pv_cloud(meshVar)

    if alpha is None:
        # Compute alpha from the point cloud coordinates themselves
        # This ensures consistency between coordinate scale and alpha parameter
        # mesh.get_max_radius() returns nondimensional values but coords may be
        # dimensional when units are active - causing Delaunay to fail
        points = point_cloud.points
        coord_range = points.max() - points.min()
        # Use a fraction of the coordinate range as a reasonable alpha
        # This approximates the mesh element size
        n_elements_estimate = max(10, len(points) ** (1.0 / dim))  # rough estimate
        alpha = coord_range / n_elements_estimate * 2.0  # 2x for safety margin

    if dim == 2:
        pv_mesh = point_cloud.delaunay_2d(alpha=alpha)
    else:
        pv_mesh = point_cloud.delaunay_3d(alpha=alpha)

    # Propagate metadata from point_cloud to the triangulated mesh
    # PyVista's delaunay methods return a new object that doesn't preserve custom attributes
    if hasattr(point_cloud, "_coord_array"):
        pv_mesh._coord_array = point_cloud._coord_array
    if hasattr(point_cloud, "_units"):
        pv_mesh._units = point_cloud._units

    return pv_mesh


def scalar_fn_to_pv_points(pv_mesh, uw_fn, dim=None):
    """Evaluate Underworld scalar function at PyVista mesh points.

    Parameters
    ----------
    pv_mesh : pyvista.DataSet
        PyVista mesh or point cloud to evaluate at.
    uw_fn : sympy.Expr
        Underworld scalar function to evaluate.
    dim : int, optional
        Dimensionality (2 or 3). Auto-detected if None.

    Returns
    -------
    numpy.ndarray
        Scalar values at mesh points (units stripped for PyVista).
        The units string is stored as ``pv_mesh._last_scalar_units``.
    """
    import underworld3 as uw
    import numpy as np

    if dim is None:
        if pv_mesh.points[:, 2].max() - pv_mesh.points[:, 2].min() < 1.0e-6:
            dim = 2
        else:
            dim = 3

    # Use stored coordinate array if available (preserves units and dimensional info)
    if hasattr(pv_mesh, '_coord_array'):
        coords = pv_mesh._coord_array[:, 0:dim]
    else:
        coords = pv_mesh.points[:, 0:dim]

    scalar_values = uw.function.evaluate(uw_fn, coords)

    # Capture units before stripping for colorbar labels
    scalar_units = None
    if hasattr(scalar_values, "units") and scalar_values.units is not None:
        scalar_units = str(scalar_values.units)
    elif hasattr(scalar_values, "_units") and scalar_values._units is not None:
        scalar_units = str(scalar_values._units)

    pv_mesh._last_scalar_units = scalar_units

    # Strip units for PyVista compatibility
    if hasattr(scalar_values, "magnitude"):
        scalar_values = scalar_values.magnitude
    else:
        scalar_values = np.asarray(scalar_values)

    return scalar_values


def vector_fn_to_pv_points(pv_mesh, uw_fn, dim=None):
    """Evaluate Underworld vector function at PyVista mesh points.

    Parameters
    ----------
    pv_mesh : pyvista.DataSet
        PyVista mesh or point cloud to evaluate at.
    uw_fn : sympy.Matrix
        Underworld vector function to evaluate.
    dim : int, optional
        Dimensionality (not used, derived from function shape).

    Returns
    -------
    numpy.ndarray
        Vector values at mesh points, shape ``(n_points, 3)``.
        Units string stored as ``pv_mesh._last_vector_units``.
    """
    import numpy as np
    import underworld3 as uw

    dim = uw_fn.shape[1]
    if dim != 2 and dim != 3:
        print(f"UW vector function should have dimension 2 or 3")

    if hasattr(pv_mesh, '_coord_array'):
        coords = pv_mesh._coord_array[:, 0:dim]
    else:
        coords = pv_mesh.points[:, 0:dim]

    vector_values_raw = uw.function.evaluate(uw_fn, coords)

    # Capture units before stripping
    vector_units = None
    if hasattr(vector_values_raw, "units") and vector_values_raw.units is not None:
        vector_units = str(vector_values_raw.units)
    elif hasattr(vector_values_raw, "_units") and vector_values_raw._units is not None:
        vector_units = str(vector_values_raw._units)

    pv_mesh._last_vector_units = vector_units

    if hasattr(vector_values_raw, "magnitude"):
        vector_values_raw = vector_values_raw.magnitude
    else:
        vector_values_raw = np.asarray(vector_values_raw)

    vector_values = np.zeros_like(pv_mesh.points)
    vector_values[:, 0:dim] = vector_values_raw.squeeze()

    return vector_values


# def vector_fn_to_pv_arrows(coords, uw_fn, dim=None):
#     """evaluate uw vector function on point cloud"""

#     import numpy as np

#     dim = uw_fn.shape[1]
#     if dim != 2 and dim != 3:
#         print(f"UW vector function should have dimension 2 or 3")

#     coords = pv_mesh.points[:, 0 : dim - 1]
#     vector_values = np.zeros_like(coords)

#     for i in range(0, dim):
#         vector_values[:, i] = uw.function.evaluate(uw_fn[i], coords, evalf=True)

#     return vector_values


def clip_mesh(pvmesh, clip_angle):
    """
    Clip the given mesh using planes at the specified angle.

    Parameters:
    -----------
    pvmesh : object
        The PyVista mesh object to be clipped.

    clip_angle : float
        The angle (in degrees) at which to clip the mesh.

    Returns:
    --------
    list
        A list containing the two clipped mesh parts.
    """
    import numpy as np

    # Calculate normals for clipping planes
    clip1_normal = (np.cos(np.deg2rad(clip_angle)), np.cos(np.deg2rad(clip_angle)), 0.0)
    clip2_normal = (
        np.cos(np.deg2rad(clip_angle)),
        -np.cos(np.deg2rad(clip_angle)),
        0.0,
    )

    # Perform clipping
    clip1 = pvmesh.clip(origin=(0.0, 0.0, 0.0), normal=clip1_normal, invert=False, crinkle=False)
    clip2 = pvmesh.clip(origin=(0.0, 0.0, 0.0), normal=clip2_normal, invert=False, crinkle=False)

    return [clip1, clip2]



#: Wireframe colours for a multigrid tail, coarsest first. Blues and greys, so
#: that the fault colour has the warm half of the wheel to itself — the whole
#: point of the figure is that the fault is findable at a glance.
MG_LEVEL_COLOURS = ("#9aa5ad", "#6fa8c7", "#3d86b4", "#1f5f96", "#123f6b",
                    "#0a2748")

#: The fault. Deliberately the loudest thing on the page.
FAULT_COLOUR = "#ff1408"


def plot_mesh_hierarchy(mesh, faults=(), clip=None, plotter=None,
                        window_size=(1400, 1400), background="white",
                        colours=None, fault_colour=FAULT_COLOUR,
                        line_width=None, fault_style="facets",
                        fault_line_width=3.0, opacity=1.0,
                        nodes=True, node_scale=0.30, legend=True):
    """Wireframe of a mesh, its multigrid tail, and its faults, in one figure.

    The standard way to look at a stacked-on mesh: one colour per level, coarsest
    palest, and the fault cells filled in a contrasting red. It answers the three
    questions that actually come up — did the hierarchy come out with the levels
    expected, is the refinement where the fault is, and did the fault survive the
    repair passes — without needing a separate figure for each.

    Written to carry to 3-D. Nothing here reads the dimension except the defaults:
    in 3-D the wireframes are taken from each level's SURFACE rather than every
    interior edge, because a full edge extraction of a tetrahedral hierarchy is
    an unreadable haze, and ``clip`` cuts the model open so the interior levels
    and the fault can be seen at all.

    Parameters
    ----------
    mesh : Mesh
        The finest mesh. Its ``_custom_mg_coarse_meshes`` tail is drawn beneath
        it, coarsest first; a mesh without one is simply drawn alone.
    faults : sequence of str
        Boundary label names to pick out in ``fault_colour``.
    fault_style : {"facets", "cells"}
        What "the fault" means in the picture, and the two are not the same
        thing.

        ``"facets"`` (the default) draws the LABELLED FACETS themselves, via
        :func:`labelled_facets_to_pv_mesh` — the segments in 2-D, the triangles
        in 3-D. This is the fault as the mesh actually represents it, and it is
        the honest choice: a fault one element wide is one chain of facets, and
        drawing it as such shows its width to be exactly what it is.

        ``"cells"`` fills the fault ZONE instead, via
        :meth:`~underworld3.discretisation.Mesh.cells_supporting` — every cell
        with a labelled facet, which is one element on EACH side. That is the
        right set for assigning a material property, but as a picture it makes a
        one-element fault look two or three elements thick, so it is not the
        default. Ask for it when the question is "which cells carry the weak
        viscosity", not "where is the fault".
    fault_line_width : float
        Width of the facet lines under ``fault_style="facets"``.
    nodes : bool
        Mark the vertices, with a different glyph for each kind: **circles** for
        the base level, **squares** for the levels stacked on top of it, and
        **triangles** for the fault's own nodes. Shape carries the distinction
        as well as colour, so the figure survives being printed in grey and does
        not rely on telling four blues apart.

        In 3-D the same three roles become sphere, cube and cone.
    legend : bool
        Build the key, with the RIGHT SHAPE against each entry. PyVista's
        default legend face is a triangle for everything, so a hand-rolled
        ``add_legend`` shows triangles beside wireframes and beside square
        nodes — a key that contradicts the figure it is keying. Since this
        routine chose the shapes, it is the thing that can label them.
    node_scale : float
        Glyph size as a fraction of the finest level's SMALL cells (its 5th
        percentile) — one size for every level, so shape and colour carry the
        distinction and size carries none.

        Two ways to get this wrong, both tried. Scaling each level by its own
        cell size draws a coarse level's marks at the coarse spacing, burying
        the fine mesh at exactly the zoom the figure exists for. And using the
        MEAN cell size of a graded mesh sizes the glyphs by the far field —
        measured, mean h = 0.017 against h = 0.002 at the fault, so every mark
        came out bigger than the cell it stood on.
    clip : tuple, optional
        ``(normal, origin)`` passed to PyVista's ``clip``. Mostly for 3-D, where
        an unclipped hierarchy shows only its outer skin.
    plotter : pyvista.Plotter, optional
        Draw into an existing plotter instead of making one. The plotter is
        RETURNED either way, unrendered, so the caller sets the camera and
        decides between ``show`` and ``screenshot``.
    colours : sequence of str, optional
        One per level, coarsest first; :data:`MG_LEVEL_COLOURS` by default,
        cycled if the hierarchy is deeper than the palette.
    line_width : sequence of float, optional
        One per level. By default coarse levels are drawn thicker so they read
        through the fine ones rather than being buried by them.

    Returns
    -------
    pyvista.Plotter

    Examples
    --------
    >>> pl = vis.plot_mesh_hierarchy(mesh, faults=["FaultA", "FaultB"])
    >>> pl.camera.parallel_projection = True
    >>> pl.screenshot("hierarchy.png")

    The cells carrying the weak viscosity, rather than the fault itself:

    >>> pl = vis.plot_mesh_hierarchy(mesh, faults=["FaultA"],
    ...                              fault_style="cells")
    """
    import numpy as np
    import pyvista as pv

    initialise(None)

    levels = list(getattr(mesh, "_custom_mg_coarse_meshes", None) or []) + [mesh]
    palette = list(colours) if colours else list(MG_LEVEL_COLOURS)
    widths = list(line_width) if line_width is not None else None

    if plotter is None:
        plotter = pv.Plotter(off_screen=pv.OFF_SCREEN, window_size=window_size)
    plotter.set_background(background)

    def wire(pvm):
        if clip is not None:
            pvm = pvm.clip(normal=clip[0], origin=clip[1])
        if mesh.dim == 3:
            pvm = pvm.extract_surface()
        return pvm.extract_all_edges()

    def marks(points, n_sides, size, colour, label):
        """Glyph a point set. Shape distinguishes the role, size the level."""
        pts = np.asarray(points, dtype=float)
        if not len(pts):
            return
        cloud = pv.PolyData(pts if pts.shape[1] == 3
                            else np.column_stack([pts, np.zeros(len(pts))]))
        if mesh.dim == 3:
            geom = {24: pv.Sphere(radius=0.5 * size),
                    4: pv.Cube(x_length=size, y_length=size, z_length=size),
                    3: pv.Cone(radius=0.5 * size, height=size)}[n_sides]
        else:
            geom = pv.Polygon(center=(0.0, 0.0, 0.0), radius=0.5 * size,
                              normal=(0.0, 0.0, 1.0), n_sides=n_sides)
        kw = {} if label is None else {"label": label}
        plotter.add_mesh(cloud.glyph(geom=geom, scale=False, orient=False),
                         color=colour, lighting=False, **kw)

    def fine_cell_size(level):
        """A LOW PERCENTILE of the cell size, never the mean.

        On a graded mesh the mean is set by the far field: on the fault meshes
        here the finest level averages h = 0.017 while h at the fault is 0.002,
        so glyphs sized on the mean come out several times larger than the cells
        they are meant to mark and bury the refined region completely. Same trap
        as judging a multigrid level by its mean h.
        """
        from underworld3.utilities.edge_split import cell_diameters
        d = cell_diameters(level.dm)
        return float(np.percentile(d, 5)) if len(d) else 0.0

    node_size = node_scale * fine_cell_size(levels[-1])

    # PyVista's named legend faces are only triangle / circle / rectangle /
    # none, so a WIREFRAME entry has to supply its own geometry — otherwise the
    # mesh levels and the square nodes would both key as rectangles and the
    # figure's own distinction would be lost in its key.
    line_face = pv.Line((-0.5, 0.0, 0.0), (0.5, 0.0, 0.0))

    key = []
    n = len(levels)
    for i, level in enumerate(levels):
        colour = palette[i % len(palette)]
        # Coarse thick, fine thin: without the taper the finest level's edges
        # cover every level under it and the hierarchy cannot be read.
        w = widths[i] if widths else max(0.4, 2.6 - 2.0 * i / max(n - 1, 1))
        plotter.add_mesh(wire(mesh_to_pv_mesh(level)), color=colour,
                         line_width=w, lighting=False, opacity=opacity,
                         label=f"level {i}"
                               f"{' (finest)' if i == n - 1 else ''}")
        key.append([f"level {i}{' (finest)' if i == n - 1 else ''}",
                    colour, line_face])
        if nodes:
            # Circles for the base, squares for everything stacked on it.
            # Unlabelled: the SHAPE is the key (circle = base, square = stacked
            # on, triangle = fault), and a legend line per level per glyph
            # doubles its length to say nothing the shapes do not.
            marks(np.asarray(level.X.coords), 24 if i == 0 else 4,
                  node_size, colour, None)

    for name in faults:
        if fault_style == "facets":
            pvf = labelled_facets_to_pv_mesh(mesh, name)
            if pvf.n_points == 0:
                continue                 # this rank owns none of it; normal
            if clip is not None:
                pvf = pvf.clip(normal=clip[0], origin=clip[1])
            plotter.add_mesh(pvf, color=fault_colour, lighting=False,
                             line_width=fault_line_width, label=name)
            key.append([name, fault_colour, line_face])
            if nodes:
                marks(pvf.points, 3, node_size, fault_colour, None)
        elif fault_style == "cells":
            zone = np.asarray(mesh.cells_supporting(name))
            if not zone.any():
                continue
            cells = mesh_to_pv_mesh(mesh).extract_cells(np.flatnonzero(zone))
            if clip is not None:
                cells = cells.clip(normal=clip[0], origin=clip[1])
            plotter.add_mesh(cells, color=fault_colour, lighting=False,
                             show_edges=True, edge_color=fault_colour,
                             line_width=1.0, label=name)
            key.append([f"{name} zone", fault_colour, "rectangle"])
        else:
            raise ValueError(
                f"fault_style must be 'facets' or 'cells', not {fault_style!r}")

    if nodes:
        # One entry per ROLE, not per level: the shape says which role, the
        # level colours are already keyed by the wireframe entries above.
        key.append(["base nodes", palette[0], "circle"])
        if n > 1:
            key.append(["stacked-on nodes", palette[min(n - 1,
                                                        len(palette) - 1)],
                        "rectangle"])
        if faults and fault_style == "facets":
            key.append(["fault nodes", fault_colour, "triangle"])

    # Exposed so the key can be INSPECTED rather than eyeballed: the failure
    # this guards against is a legend that disagrees with the figure, and that
    # is invisible in any check that only counts actors.
    plotter._uw_legend_key = key
    if legend and key:
        plotter.add_legend(key, bcolor="white", border=True,
                           size=(0.24, 0.030 * len(key) + 0.02),
                           loc="lower right")
    return plotter


def plot_mesh(
    mesh,
    title="",
    clip_angle=0.0,
    cpos="xy",
    window_size=(750, 750),
    show_edges=True,
    save_png=False,
    dir_fname="",
):
    """
    Plot a mesh with optional clipping, edge display, and saving functionality.

    Parameters:
    -----------
    mesh : object
        The mesh object to be plotted. This should be in a format that can be converted
        into a PyVista mesh using `vis.mesh_to_pv_mesh()`.

    title : str, optional
        The title text to be displayed on the plot. Default is an empty string, meaning no title is shown.

    clip_angle : float, optional
        The angle (in degrees) at which to clip the mesh. If set to 0.0, no clipping is applied.
        Clipping is performed using planes at the specified angle. Default is 0.0.

    cpos : str or list, optional
        The camera position for viewing the mesh. It can be a string such as 'xy', 'xz', 'yz', or
        a list specifying the exact camera position. Default is 'xy'.

    window_size : tuple of int, optional
        The size of the rendering window in pixels as (width, height). Default is (750, 750).

    show_edges : bool, optional
        Whether to display the edges of the mesh in the plot. If `True`, edges will be shown.
        Default is `True`.

    save_png : bool, optional
        Whether to save the plot as a PNG file. If `True`, the plot will be saved to the specified
        directory and filename. Default is `False`.

    dir_fname : str, optional
        The directory and filename for saving the PNG image if `save_png` is `True`.
        If left empty, no file is saved. Default is an empty string.

    Returns:
    --------
    None
        This function does not return any value. It displays the mesh plot in a PyVista window
        and optionally saves a screenshot.
    """
    import pyvista as pv

    pvmesh = mesh_to_pv_mesh(mesh)

    pl = pv.Plotter(window_size=window_size)
    if clip_angle != 0.0:
        clipped_meshes = clip_mesh(pvmesh, clip_angle)
        for clipped_mesh in clipped_meshes:
            pl.add_mesh(clipped_mesh, edge_color="k", show_edges=True, opacity=1.0)
    else:
        pl.add_mesh(
            pvmesh,
            edge_color="k",
            show_edges=show_edges,
            use_transparency=False,
            opacity=1.0,
        )

    if len(title) != 0:
        pl.add_text(title, font_size=18, position=(950, 2100))

    pl.show(cpos=cpos)

    if save_png:
        pl.camera.zoom(1.4)
        pl.screenshot(dir_fname, scale=3.5)

    return


def plot_scalar(
    mesh,
    scalar,
    scalar_name="",
    cmap="",
    clim="",
    window_size=(750, 750),
    title="",
    fmt="%10.7f",
    clip_angle=0.0,
    cpos="xy",
    show_edges=False,
    save_png=False,
    dir_fname="",
):
    """
    Plot a scalar quantity from a mesh with options for clipping, colormap, and saving.

    Parameters:
    -----------
    mesh : object
        The mesh object to be plotted. This should be in a format that can be converted
        into a PyVista mesh using `vis.mesh_to_pv_mesh()`.

    scalar : mesh variable name or sympy expression
        The scalar values associated with the mesh points. These values will be visualized
        on the mesh.

    scalar_name : str, optional
        The name of the scalar field to be used when adding it to the mesh. This name will
        also be used as the label for the scalar bar. Default is an empty string.

    cmap : str, optional
        The colormap to be used for visualizing the scalar values. This can be any colormap
        recognized by PyVista or Matplotlib. Default is an empty string, which uses the default colormap.

    clim : tuple of float, optional
        The scalar range to be used for coloring the mesh (e.g., `(min_value, max_value)`). If not
        provided, the range of the scalar values is used. Default is an empty string, which uses
        the full range of the scalar values.

    window_size : tuple of int, optional
        The size of the rendering window in pixels as (width, height). Default is (750, 750).

    title : str, optional
        The title text to be displayed on the plot. Default is an empty string, meaning no title is shown.

    fmt : str, optional
        The format string for scalar values. This is typically used when displaying values on the scalar bar.
        Default is '%10.7f'.

    clip_angle : float, optional
        The angle (in degrees) at which to clip the mesh. If set to 0.0, no clipping is applied.
        Clipping is performed using planes at the specified angle. Default is 0.0.

    cpos : str or list, optional
        The camera position for viewing the mesh. It can be a string such as 'xy', 'xz', 'yz', or
        a list specifying the exact camera position. Default is 'xy'.

    show_edges : bool, optional
        Whether to display the edges of the mesh in the plot. If `True`, edges will be shown.
        Default is `False`.

    save_png : bool, optional
        Whether to save the plot as a PNG file. If `True`, the plot will be saved to the specified
        directory and filename. Default is `False`.

    dir_fname : str, optional
        The directory and filename for saving the PNG image if `save_png` is `True`.
        If left empty, no file is saved. Default is an empty string.

    Returns:
    --------
    None
        This function does not return any value. It displays the scalar field on the mesh in a PyVista
        window and optionally saves a screenshot.
    """

    import sympy
    import numpy as np
    import pyvista as pv

    pvmesh = mesh_to_pv_mesh(mesh)
    pvmesh.point_data[scalar_name] = scalar_fn_to_pv_points(pvmesh, scalar)

    # Build scalar bar label with units if available
    scalar_bar_title = scalar_name
    if hasattr(pvmesh, '_last_scalar_units') and pvmesh._last_scalar_units:
        scalar_bar_title = f"{scalar_name} ({pvmesh._last_scalar_units})"

    pl = pv.Plotter(window_size=window_size)
    if clip_angle != 0.0:
        clipped_meshes = clip_mesh(pvmesh, clip_angle)
        for clipped_mesh in clipped_meshes:
            pl.add_mesh(
                clipped_mesh,
                cmap=cmap,
                edge_color="k",
                scalars=scalar_name,
                show_edges=show_edges,
                use_transparency=False,
                show_scalar_bar=True,
                scalar_bar_args={"title": scalar_bar_title},
                opacity=1.0,
                clim=clim,
            )
    else:
        pl.add_mesh(
            pvmesh,
            cmap=cmap,
            edge_color="k",
            scalars=scalar_name,
            show_edges=show_edges,
            use_transparency=False,
            opacity=1.0,
            clim=clim,
            show_scalar_bar=True,
            scalar_bar_args={"title": scalar_bar_title},
        )

    pl.show(cpos=cpos)

    if len(title) != 0:
        pl.add_text(title, font_size=18, position=(950, 2100))

    if save_png:
        pl.camera.zoom(1.4)
        pl.screenshot(dir_fname, scale=3.5)

    return


def plot_vector(
    mesh,
    vector,
    vector_name="",
    cmap="",
    clim="",
    vmag="",
    vfreq="",
    save_png=False,
    dir_fname="",
    title="",
    fmt="%10.7f",
    clip_angle=0.0,
    show_arrows=False,
    cpos="xy",
    show_edges=False,
    window_size=(750, 750),
    scalar=None,
    scalar_name="",
):
    """
    Plot a vector quantity from a mesh with options for clipping, colormap, vector magnitude, and saving.

    Parameters:
    -----------
    mesh : object
        The mesh object to be plotted. This should be in a format that can be converted
        into a PyVista mesh using `vis.mesh_to_pv_mesh()`.

    vector : mesh variable name or sympy expression
        The symbolic representation of the vector field associated with the mesh points.
        This vector field will be visualized on the mesh.

    vector_name : str, optional
        The name of the vector field to be used when adding it to the mesh. This name will
        also be used as the label for the vector magnitude in the scalar bar. Default is an empty string.

    cmap : str, optional
        The colormap to be used for visualizing the vector magnitudes. This can be any colormap
        recognized by PyVista or Matplotlib. Default is an empty string, which uses the default colormap.

    clim : tuple of float, optional
        The scalar range to be used for coloring the mesh based on vector magnitudes (e.g., `(min_value, max_value)`).
        If not provided, the range of the vector magnitudes is used. Default is an empty string.

    vmag : float or str, optional
        The scaling factor for the arrow magnitudes when plotting vectors as arrows.
        Default is an empty string, which uses the default scaling.

    vfreq : int, optional
        The frequency of arrows to display when `show_arrows` is `True`. For example, if set to 10, every 10th vector
        will be plotted as an arrow. Default is an empty string, which uses the default frequency.

    save_png : bool, optional
        Whether to save the plot as a PNG file. If `True`, the plot will be saved to the specified
        directory and filename. Default is `False`.

    dir_fname : str, optional
        The directory and filename for saving the PNG image if `save_png` is `True`.
        If left empty, no file is saved. Default is an empty string.

    title : str, optional
        The title text to be displayed on the plot. Default is an empty string, meaning no title is shown.

    fmt : str, optional
        The format string for scalar values, typically used in the scalar bar. Default is '%10.7f'.

    clip_angle : float, optional
        The angle (in degrees) at which to clip the mesh. If set to 0.0, no clipping is applied.
        Clipping is performed using planes at the specified angle. Default is 0.0.

    show_arrows : bool, optional
        Whether to display arrows representing the vector field on the mesh. If `True`, arrows will be shown.
        Default is `False`.

    cpos : str or list, optional
        The camera position for viewing the mesh. It can be a string such as 'xy', 'xz', 'yz', or
        a list specifying the exact camera position. Default is 'xy'.

    show_edges : bool, optional
        Whether to display the edges of the mesh in the plot. If `True`, edges will be shown.
        Default is `False`.

    window_size : tuple of int, optional
        The size of the rendering window in pixels as (width, height). Default is (750, 750).

    scalar : mesh variable name or sympy expression, optional
        An optional scalar field associated with the mesh points. If provided, this scalar field
        will be used for coloring the mesh instead of the vector magnitude. Default is `None`.

    scalar_name : str, optional
        The name of the scalar field to be used when adding it to the mesh. This name will
        be used as the label for the scalar bar if `scalar` is provided. Default is an empty string.

    Returns:
    --------
    None
        This function does not return any value. It displays the vector field on the mesh in a PyVista
        window and optionally saves a screenshot.

    Notes
    -----
    When the model uses physical units, arrows are drawn in the mesh
    coordinate space. A velocity of 1 cm/yr on a mesh in meters
    (extent ~1e6) produces arrows of length ~3e-10 in mesh units —
    effectively invisible. Adjust ``vmag`` to compensate::

        # Scale arrows to ~5% of mesh extent
        vmag = 0.05 * mesh_extent / max_velocity
    """

    import sympy
    import numpy as np
    import pyvista as pv

    pvmesh = mesh_to_pv_mesh(mesh)
    pvmesh.point_data[vector_name] = vector_fn_to_pv_points(pvmesh, vector.sym)
    if scalar is None:
        scalar_name = vector_name + "_mag"
        pvmesh.point_data[scalar_name] = scalar_fn_to_pv_points(
            pvmesh, sympy.sqrt(vector.sym.dot(vector.sym))
        )
    else:
        pvmesh.point_data[scalar_name] = scalar_fn_to_pv_points(pvmesh, scalar.sym)

    # Build scalar bar label with units if available
    scalar_bar_title = scalar_name
    if hasattr(pvmesh, '_last_scalar_units') and pvmesh._last_scalar_units:
        scalar_bar_title = f"{scalar_name} ({pvmesh._last_scalar_units})"

    velocity_points = meshVariable_to_pv_cloud(vector)
    velocity_points.point_data[vector_name] = vector_fn_to_pv_points(velocity_points, vector.sym)

    pl = pv.Plotter(window_size=window_size)
    if clip_angle != 0.0:
        clipped_meshes = clip_mesh(pvmesh, clip_angle)
        for clipped_mesh in clipped_meshes:
            pl.add_mesh(
                clipped_mesh,
                cmap=cmap,
                edge_color="k",
                scalars=scalar_name,
                show_edges=show_edges,
                use_transparency=False,
                show_scalar_bar=True,
                scalar_bar_args={"title": scalar_bar_title},
                opacity=1.0,
                clim=clim,
            )
    else:
        pl.add_mesh(
            pvmesh,
            cmap=cmap,
            edge_color="k",
            scalars=scalar_name,
            show_edges=show_edges,
            use_transparency=False,
            opacity=1.0,
            clim=clim,
            show_scalar_bar=True,
            scalar_bar_args={"title": scalar_bar_title},
        )

    if show_arrows:
        pl.add_arrows(
            velocity_points.points[::vfreq],
            velocity_points.point_data[vector_name][::vfreq],
            mag=vmag,
            color="k",
        )

    pl.show(cpos=cpos)

    if len(title) != 0:
        pl.add_text(title, font_size=18, position=(950, 1075))

    if save_png:
        pl.camera.zoom(1.4)
        pl.screenshot(dir_fname, scale=3.5)

    return


def save_colorbar(
    colormap="",
    cb_bounds=None,
    vmin=None,
    vmax=None,
    figsize_cb=(6, 1),
    primary_fs=18,
    cb_orient="vertical",
    cb_axis_label="",
    cb_label_xpos=0.5,
    cb_label_ypos=0.5,
    fformat="png",
    output_path="",
    fname="",
):
    """
    Save a colorbar separately from a plot with customizable appearance and format.

    Parameters:
    -----------
    colormap : str, optional
        The name of the colormap to be used for the colorbar. This should be a valid Matplotlib colormap name.
        Default is an empty string, which uses the default colormap.

    cb_bounds : list or array-like, optional
        The bounds to be used for the colorbar. If provided, the colorbar will be generated with these bounds.
        Default is None, which means bounds are not explicitly set.

    vmin : float, optional
        The minimum value for the colorbar. This is used to define the lower limit of the colormap.
        Default is None.

    vmax : float, optional
        The maximum value for the colorbar. This is used to define the upper limit of the colormap.
        Default is None.

    figsize_cb : tuple of float, optional
        The size of the figure for the colorbar in inches as (width, height). Default is (6, 1).

    primary_fs : int, optional
        The primary font size for the colorbar labels and title. Default is 18.

    cb_orient : str, optional
        The orientation of the colorbar, either 'vertical' or 'horizontal'. Default is 'vertical'.

    cb_axis_label : str, optional
        The label for the colorbar axis. This text will be displayed alongside the colorbar.
        Default is an empty string.

    cb_label_xpos : float, optional
        The x-position for the colorbar label. This adjusts the horizontal positioning of the label.
        Default is 0.5.

    cb_label_ypos : float, optional
        The y-position for the colorbar label. This adjusts the vertical positioning of the label.
        Default is 0.5.

    fformat : str, optional
        The format for saving the colorbar image. Supported formats are 'png' and 'pdf'.
        Default is 'png'.

    output_path : str, optional
        The directory path where the colorbar image will be saved. Default is an empty string.

    fname : str, optional
        The filename to use when saving the colorbar image. This should not include the file extension.
        Default is an empty string.

    Returns:
    --------
    None
        This function does not return any value. It saves the colorbar as a separate image file in the specified format.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    plt.figure(figsize=figsize_cb)
    plt.rc("font", size=primary_fs)  # Set font size
    if cb_bounds is not None:
        bounds_np = np.array([cb_bounds])
        img = plt.imshow(bounds_np, cmap=colormap)
    else:
        v_min_max_np = np.array([[vmin, vmax]])
        img = plt.imshow(v_min_max_np, cmap=colormap)

    plt.gca().set_visible(False)

    if cb_orient == "vertical":
        cax = plt.axes([0.1, 0.2, 0.06, 1.15])
        cb = plt.colorbar(orientation="vertical", cax=cax)
        cb.ax.set_title(
            cb_axis_label,
            fontsize=primary_fs,
            x=cb_label_xpos,
            y=cb_label_ypos,
            rotation=90,
        )
        plt.savefig(f"{output_path}{fname}_cbvert.{fformat}", dpi=150, bbox_inches="tight")

    elif cb_orient == "horizontal":
        cax = plt.axes([0.1, 0.2, 1.15, 0.06])
        cb = plt.colorbar(orientation="horizontal", cax=cax)
        cb.ax.set_title(cb_axis_label, fontsize=primary_fs, x=cb_label_xpos, y=cb_label_ypos)
        plt.savefig(f"{output_path}{fname}_cbhorz.{fformat}", dpi=150, bbox_inches="tight")

    return
