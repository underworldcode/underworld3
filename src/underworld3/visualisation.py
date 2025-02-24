## pyvista helper routines
import os


def initialise(jupyter_backend):


    import pyvista as pv

    pv.global_theme.background = "white"
    pv.global_theme.anti_aliasing = "msaa"
    pv.global_theme.smooth_shading = True
    pv.global_theme.camera["viewup"] = [0.0, 1.0, 0.0]
    pv.global_theme.camera["position"] = [0.0, 0.0, 5.0]

    try:
        if jupyter_backend is not None:
            pv.global_theme.jupyter_backend = jupyter_backend
        elif "BINDER_LAUNCH_HOST" in os.environ or "BINDER_REPO_URL" in os.environ:
            pv.global_theme.jupyter_backend = "client"
        else:
            pv.global_theme.jupyter_backend = "trame"

    except RuntimeError:
        pv.global_theme.jupyter_backend = "panel"

    return


def mesh_to_pv_mesh(mesh, jupyter_backend=None):
    """Initialise pyvista engine from existing mesh"""

    # # Required in notebooks
    # import nest_asyncio
    # nest_asyncio.apply()

    initialise(jupyter_backend)

    import os
    import shutil
    import tempfile
    import pyvista as pv

    with tempfile.TemporaryDirectory() as tmp:
        if type(mesh) == str:  # reading msh file directly
            vtk_filename = os.path.join(tmp, "tmpMsh.msh")
            shutil.copyfile(mesh, vtk_filename)
        else:  # reading mesh by creating vtk
            vtk_filename = os.path.join(tmp, "tmpMsh.vtk")
            mesh.vtk(vtk_filename)

        pvmesh = pv.read(vtk_filename)

    return pvmesh


def coords_to_pv_coords(coords):
    """For a given set of coords, return a pyvista coordinate vector"""

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
    """swarm points to pyvista PolyData  object"""

    import numpy as np
    import pyvista as pv

    with swarm.access():
        points = np.zeros((swarm.data.shape[0], 3))
        points[:, 0] = swarm.data[:, 0]
        points[:, 1] = swarm.data[:, 1]
        if swarm.mesh.dim == 2:
            points[:, 2] = 0.0
        else:
            points[:, 2] = swarm.data[:, 2]

    point_cloud = pv.PolyData(points)

    return point_cloud


def meshVariable_to_pv_cloud(meshVar):
    """meshVariable point locations to pyvista PolyData object"""

    import numpy as np
    import pyvista as pv

    points = np.zeros((meshVar.coords.shape[0], 3))
    points[:, 0] = meshVar.coords[:, 0]
    points[:, 1] = meshVar.coords[:, 1]

    if meshVar.mesh.dim == 2:
        points[:, 2] = 0.0
    else:
        points[:, 2] = meshVar.coords[:, 2]

    point_cloud = pv.PolyData(points)

    return point_cloud


def meshVariable_to_pv_mesh_object(meshVar, alpha=None):
    """Convert meshvariable to delaunay triangulated pyvista mesh object.
    This is redundant if the meshVariable degree is 1 (the original mesh exactly
    represents the data)"""

    mesh = meshVar.mesh
    dim = mesh.dim

    if alpha is None:
        alpha = mesh.get_max_radius()

    point_cloud = meshVariable_to_pv_cloud(meshVar)

    if dim == 2:
        pv_mesh = point_cloud.delaunay_2d(alpha=alpha)
    else:
        pv_mesh = point_cloud.delaunay_3d(alpha=alpha)

    return pv_mesh


def scalar_fn_to_pv_points(pv_mesh, uw_fn, dim=None, simplify=True):
    """evaluate uw scalar function at mesh/cloud points"""

    import underworld3 as uw
    import sympy

    if simplify:
        uw_fn = sympy.simplify(uw_fn)

    if dim is None:
        if pv_mesh.points[:, 2].max() - pv_mesh.points[:, 2].min() < 1.0e-6:
            dim = 2
        else:
            dim = 3

    coords = pv_mesh.points[:, 0:dim]
    scalar_values = uw.function.evalf(uw_fn, coords)

    return scalar_values


def vector_fn_to_pv_points(pv_mesh, uw_fn, dim=None, simplify=True):
    """evaluate uw vector function at mesh/cloud points"""

    import numpy as np
    import underworld3 as uw
    import sympy

    if simplify:
        uw_fn = sympy.simplify(uw_fn)
    dim = uw_fn.shape[1]
    if dim != 2 and dim != 3:
        print(f"UW vector function should have dimension 2 or 3")

    coords = pv_mesh.points[:, 0:dim]
    vector_values = np.zeros_like(pv_mesh.points)

    vector_values[:, 0:dim] = uw.function.evalf(uw_fn, coords)

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
#         vector_values[:, i] = uw.function.evalf(uw_fn[i], coords)

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
    clip1 = pvmesh.clip(
        origin=(0.0, 0.0, 0.0), normal=clip1_normal, invert=False, crinkle=False
    )
    clip2 = pvmesh.clip(
        origin=(0.0, 0.0, 0.0), normal=clip2_normal, invert=False, crinkle=False
    )

    return [clip1, clip2]


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

    print(pvmesh.point_data[scalar_name].min(), pvmesh.point_data[scalar_name].max())

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
                show_scalar_bar=False,
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
            show_scalar_bar=False,
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
        pvmesh.point_data[scalar_name] = scalar_fn_to_pv_points(
            pvmesh, scalar.sym
        )

    print(pvmesh.point_data[scalar_name].min(), pvmesh.point_data[scalar_name].max())

    velocity_points = meshVariable_to_pv_cloud(vector)
    velocity_points.point_data[vector_name] = vector_fn_to_pv_points(
        velocity_points, vector.sym
    )

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
                show_scalar_bar=False,
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
            show_scalar_bar=False,
        )

    # pl.add_scalar_bar(vector_name, vertical=False, title_font_size=25, label_font_size=20, fmt=fmt,
    #                   position_x=0.225, position_y=0.01,)

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
        plt.savefig(
            f"{output_path}{fname}_cbvert.{fformat}", dpi=150, bbox_inches="tight"
        )

    elif cb_orient == "horizontal":
        cax = plt.axes([0.1, 0.2, 1.15, 0.06])
        cb = plt.colorbar(orientation="horizontal", cax=cax)
        cb.ax.set_title(
            cb_axis_label, fontsize=primary_fs, x=cb_label_xpos, y=cb_label_ypos
        )
        plt.savefig(
            f"{output_path}{fname}_cbhorz.{fformat}", dpi=150, bbox_inches="tight"
        )

    return
