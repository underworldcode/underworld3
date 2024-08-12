## pyvista helper routines
import os

def initialise():

    import pyvista as pv

    pv.global_theme.background = "white"
    pv.global_theme.anti_aliasing = "msaa"
    pv.global_theme.smooth_shading = True
    pv.global_theme.camera["viewup"] = [0.0, 1.0, 0.0]
    pv.global_theme.camera["position"] = [0.0, 0.0, 5.0]

    try:
        if 'BINDER_LAUNCH_HOST' in os.environ or 'BINDER_REPO_URL' in os.environ:
            pv.global_theme.jupyter_backend = "client"
        else:
            pv.global_theme.jupyter_backend = "trame"
    except RuntimeError:
        pv.global_theme.jupyter_backend = "panel"

    return


def mesh_to_pv_mesh(mesh):
    """Initialise pyvista engine from existing mesh"""

    # # Required in notebooks
    # import nest_asyncio
    # nest_asyncio.apply()

    initialise()

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
    """pyvista requires 3D coordinates / vectors - fix if they are 2D"""

    return vector_to_pv_vector(coords)


def vector_to_pv_vector(vector):
    """pyvista requires 3D coordinates / vectors - fix if they are 2D"""

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


def plot_mesh(_mesh, _title='', _clip_angle=0.0, _cpos='xy', _window_size=(750, 750),
              _show_edges=True, _save_png=False, _dir_fname='',):
    
    '''
    Plot a mesh with optional clipping, edge display, and saving functionality.

    Parameters:
    -----------
    _mesh : object
        The mesh object to be plotted. This should be in a format that can be converted
        into a PyVista mesh using `vis.mesh_to_pv_mesh()`.

    _title : str, optional
        The title text to be displayed on the plot. Default is an empty string, meaning no title is shown.

    _clip_angle : float, optional
        The angle (in degrees) at which to clip the mesh. If set to 0.0, no clipping is applied. 
        Clipping is performed using planes at the specified angle. Default is 0.0.

    _cpos : str or list, optional
        The camera position for viewing the mesh. It can be a string such as 'xy', 'xz', 'yz', or 
        a list specifying the exact camera position. Default is 'xy'.

    _window_size : tuple of int, optional
        The size of the rendering window in pixels as (width, height). Default is (750, 750).

    _show_edges : bool, optional
        Whether to display the edges of the mesh in the plot. If `True`, edges will be shown.
        Default is `True`.

    _save_png : bool, optional
        Whether to save the plot as a PNG file. If `True`, the plot will be saved to the specified
        directory and filename. Default is `False`.

    _dir_fname : str, optional
        The directory and filename for saving the PNG image if `_save_png` is `True`.
        If left empty, no file is saved. Default is an empty string.

    Returns:
    --------
    None
        This function does not return any value. It displays the mesh plot in a PyVista window
        and optionally saves a screenshot.
    '''
    import sympy
    import numpy
    import pyvista as pv

    pvmesh = mesh_to_pv_mesh(_mesh)

    pl = pv.Plotter(window_size=_window_size)
    if _clip_angle!=0.0:
        clip1_normal = (np.cos(np.deg2rad(_clip_angle)), np.cos(np.deg2rad(_clip_angle)), 0.0)
        clip1 = pvmesh.clip(origin=(0.0, 0.0, 0.0), normal=clip1_normal, invert=False, crinkle=False)
        pl.add_mesh(clip1, edge_color="k", show_edges=True, opacity=1.0,)

        clip2_normal = (np.cos(np.deg2rad(_clip_angle)), -np.cos(np.deg2rad(_clip_angle)), 0.0)
        clip2 = pvmesh.clip(origin=(0.0, 0.0, 0.0), normal=clip2_normal, invert=False, crinkle=False)
        pl.add_mesh(clip2, edge_color="k", show_edges=True, opacity=1.0,)
    else:
        pl.add_mesh(pvmesh, edge_color="k", show_edges=_show_edges, use_transparency=False, opacity=1.0)

    pl.show(cpos=_cpos)

    if len(_title)!=0:
        pl.add_text(_title, font_size=18, position=(950, 2100))
    
    if _save_png:
        pl.camera.zoom(1.4)
        pl.screenshot(_dir_fname, scale=3.5,)


def plot_scalar(_mesh, _scalar, _scalar_name='', _cmap='', _clim='', _window_size=(750, 750),
                _title='', _fmt='%10.7f', _clip_angle=0.0, _cpos='xy', _show_edges=False, 
                _save_png=False, _dir_fname='',):
    
    '''
    Plot a scalar quantity from a mesh with options for clipping, colormap, and saving.

    Parameters:
    -----------
    _mesh : object
        The mesh object to be plotted. This should be in a format that can be converted
        into a PyVista mesh using `vis.mesh_to_pv_mesh()`.

    _scalar : mesh variable name (not sympy expression)
        The scalar values associated with the mesh points. These values will be visualized
        on the mesh.

    _scalar_name : str, optional
        The name of the scalar field to be used when adding it to the mesh. This name will
        also be used as the label for the scalar bar. Default is an empty string.

    _cmap : str, optional
        The colormap to be used for visualizing the scalar values. This can be any colormap
        recognized by PyVista or Matplotlib. Default is an empty string, which uses the default colormap.

    _clim : tuple of float, optional
        The scalar range to be used for coloring the mesh (e.g., `(min_value, max_value)`). If not
        provided, the range of the scalar values is used. Default is an empty string, which uses
        the full range of the scalar values.

    _window_size : tuple of int, optional
        The size of the rendering window in pixels as (width, height). Default is (750, 750).

    _title : str, optional
        The title text to be displayed on the plot. Default is an empty string, meaning no title is shown.

    _fmt : str, optional
        The format string for scalar values. This is typically used when displaying values on the scalar bar.
        Default is '%10.7f'.

    _clip_angle : float, optional
        The angle (in degrees) at which to clip the mesh. If set to 0.0, no clipping is applied.
        Clipping is performed using planes at the specified angle. Default is 0.0.

    _cpos : str or list, optional
        The camera position for viewing the mesh. It can be a string such as 'xy', 'xz', 'yz', or 
        a list specifying the exact camera position. Default is 'xy'.

    _show_edges : bool, optional
        Whether to display the edges of the mesh in the plot. If `True`, edges will be shown.
        Default is `False`.

    _save_png : bool, optional
        Whether to save the plot as a PNG file. If `True`, the plot will be saved to the specified
        directory and filename. Default is `False`.

    _dir_fname : str, optional
        The directory and filename for saving the PNG image if `_save_png` is `True`.
        If left empty, no file is saved. Default is an empty string.

    Returns:
    --------
    None
        This function does not return any value. It displays the scalar field on the mesh in a PyVista
        window and optionally saves a screenshot.
    '''
    
    import sympy
    import numpy as np
    import pyvista as pv

    pvmesh = mesh_to_pv_mesh(_mesh)
    pvmesh.point_data[_scalar_name] = scalar_fn_to_pv_points(pvmesh, _scalar.sym)

    # print(pvmesh.point_data[_scalar_name].min(), pvmesh.point_data[_scalar_name].max())
    
    pl = pv.Plotter(window_size=_window_size)
    if _clip_angle!=0.0:
        clip1_normal = (np.cos(np.deg2rad(_clip_angle)), np.cos(np.deg2rad(_clip_angle)), 0.0)
        clip1 = pvmesh.clip(origin=(0.0, 0.0, 0.0), normal=clip1_normal, invert=False, crinkle=False)
        pl.add_mesh(clip1, cmap=_cmap, edge_color="k", scalars=_scalar_name, show_edges=_show_edges, 
                    use_transparency=False, show_scalar_bar=False, opacity=1.0, clim=_clim)

        clip2_normal = (np.cos(np.deg2rad(_clip_angle)), -np.cos(np.deg2rad(_clip_angle)), 0.0)
        clip2 = pvmesh.clip(origin=(0.0, 0.0, 0.0), normal=clip2_normal, invert=False, crinkle=False)
        pl.add_mesh(clip2, cmap=_cmap, edge_color="k", scalars=_scalar_name, show_edges=_show_edges, 
                    use_transparency=False, show_scalar_bar=False, opacity=1.0, clim=_clim)
    else:
        pl.add_mesh(pvmesh, cmap=_cmap, edge_color="k", scalars=_scalar_name, show_edges=_show_edges, 
                    use_transparency=False, opacity=1.0, clim=_clim, show_scalar_bar=False)
    
    pl.show(cpos=_cpos)

    if len(_title)!=0:
        pl.add_text(_title, font_size=18, position=(950, 2100))

    if _save_png:
        pl.camera.zoom(1.4)
        pl.screenshot(_dir_fname, scale=3.5,)


def plot_vector(_mesh, _vector, _vector_name='', _cmap='', _clim='', _vmag='', _vfreq='', _save_png=False, 
                _dir_fname='', _title='', _fmt='%10.7f', _clip_angle=0.0, _show_arrows=False, _cpos='xy', 
                _show_edges=False, _window_size=(750, 750)):
    
    '''
    Plot a vector quantity from a mesh with options for clipping, colormap, vector magnitude, and saving.

    Parameters:
    -----------
    _mesh : object
        The mesh object to be plotted. This should be in a format that can be converted
        into a PyVista mesh using `vis.mesh_to_pv_mesh()`.

    _vector : mesh variable name (not sympy expression) 
        The symbolic representation of the vector field associated with the mesh points.
        This vector field will be visualized on the mesh.

    _vector_name : str, optional
        The name of the vector field to be used when adding it to the mesh. This name will
        also be used as the label for the vector magnitude in the scalar bar. Default is an empty string.

    _cmap : str, optional
        The colormap to be used for visualizing the vector magnitudes. This can be any colormap
        recognized by PyVista or Matplotlib. Default is an empty string, which uses the default colormap.

    _clim : tuple of float, optional
        The scalar range to be used for coloring the mesh based on vector magnitudes (e.g., `(min_value, max_value)`).
        If not provided, the range of the vector magnitudes is used. Default is an empty string.

    _vmag : float or str, optional
        The scaling factor for the arrow magnitudes when plotting vectors as arrows. 
        Default is an empty string, which uses the default scaling.

    _vfreq : int, optional
        The frequency of arrows to display when `_show_arrows` is `True`. For example, if set to 10, every 10th vector
        will be plotted as an arrow. Default is an empty string, which uses the default frequency.

    _save_png : bool, optional
        Whether to save the plot as a PNG file. If `True`, the plot will be saved to the specified
        directory and filename. Default is `False`.

    _dir_fname : str, optional
        The directory and filename for saving the PNG image if `_save_png` is `True`.
        If left empty, no file is saved. Default is an empty string.

    _title : str, optional
        The title text to be displayed on the plot. Default is an empty string, meaning no title is shown.

    _fmt : str, optional
        The format string for scalar values, typically used in the scalar bar. Default is '%10.7f'.

    _clip_angle : float, optional
        The angle (in degrees) at which to clip the mesh. If set to 0.0, no clipping is applied.
        Clipping is performed using planes at the specified angle. Default is 0.0.

    _show_arrows : bool, optional
        Whether to display arrows representing the vector field on the mesh. If `True`, arrows will be shown.
        Default is `False`.

    _cpos : str or list, optional
        The camera position for viewing the mesh. It can be a string such as 'xy', 'xz', 'yz', or 
        a list specifying the exact camera position. Default is 'xy'.

    _show_edges : bool, optional
        Whether to display the edges of the mesh in the plot. If `True`, edges will be shown.
        Default is `False`.

    _window_size : tuple of int, optional
        The size of the rendering window in pixels as (width, height). Default is (750, 750).

    Returns:
    --------
    None
        This function does not return any value. It displays the vector field on the mesh in a PyVista
        window and optionally saves a screenshot.
    '''
    
    import sympy
    import numpy as np
    import pyvista as pv

    pvmesh = mesh_to_pv_mesh(_mesh)
    pvmesh.point_data[_vector_name] = vector_fn_to_pv_points(pvmesh, _vector.sym)
    _vector_mag_name = _vector_name+'_mag'
    pvmesh.point_data[_vector_mag_name] = scalar_fn_to_pv_points(pvmesh, 
                                                                     sympy.sqrt(_vector.sym.dot(_vector.sym)))
    
    # print(pvmesh.point_data[_vector_mag_name].min(), pvmesh.point_data[_vector_mag_name].max())
    
    velocity_points = meshVariable_to_pv_cloud(_vector)
    velocity_points.point_data[_vector_name] = vector_fn_to_pv_points(velocity_points, _vector.sym)
    
    pl = pv.Plotter(window_size=_window_size)
    if _clip_angle!=0.0:
        clip1_normal = (np.cos(np.deg2rad(_clip_angle)), np.cos(np.deg2rad(_clip_angle)), 0.0)
        clip1 = pvmesh.clip(origin=(0.0, 0.0, 0.0), normal=clip1_normal, invert=False, crinkle=False)
        pl.add_mesh(clip1, cmap=_cmap, edge_color="k", scalars=_vector_mag_name, show_edges=_show_edges, 
                    use_transparency=False, show_scalar_bar=False, opacity=1.0, clim=_clim)

        clip2_normal = (np.cos(np.deg2rad(_clip_angle)), -np.cos(np.deg2rad(_clip_angle)), 0.0)
        clip2 = pvmesh.clip(origin=(0.0, 0.0, 0.0), normal=clip2_normal, invert=False, crinkle=False)
        pl.add_mesh(clip2, cmap=_cmap, edge_color="k", scalars=_vector_mag_name, show_edges=_show_edges, 
                    use_transparency=False, show_scalar_bar=False, opacity=1.0, clim=_clim)
    else:
        pl.add_mesh(pvmesh, cmap=_cmap, edge_color="k", scalars=_vector_mag_name, show_edges=_show_edges, 
                    use_transparency=False, opacity=1.0, clim=_clim, show_scalar_bar=False)
               
    # pl.add_scalar_bar(_vector_name, vertical=False, title_font_size=25, label_font_size=20, fmt=_fmt, 
    #                   position_x=0.225, position_y=0.01,)
    
    if _show_arrows:
        pl.add_arrows(velocity_points.points[::_vfreq], velocity_points.point_data[_vector_name][::_vfreq], 
                      mag=_vmag, color='k')

    pl.show(cpos=_cpos)

    if len(_title)!=0:
        pl.add_text(_title, font_size=18, position=(950, 1075))

    if _save_png:
        pl.camera.zoom(1.4)
        pl.screenshot(_dir_fname, scale=3.5,)


def save_colorbar(_colormap='', _cb_bounds='', _vmin='', _vmax='', _figsize_cb='', _primary_fs=18, _cb_orient='', 
                  _cb_axis_label='', _cb_label_xpos='', _cb_label_ypos='', _fformat='', _output_path='', 
                  _fname=''):
    
    '''
    Save a colorbar separately from a plot with customizable appearance and format.

    Parameters:
    -----------
    _colormap : str, optional
        The name of the colormap to be used for the colorbar. This should be a valid Matplotlib colormap name.
        Default is an empty string, which uses the default colormap.

    _cb_bounds : list or array-like, optional
        The bounds to be used for the colorbar. If provided, the colorbar will be generated with these bounds.
        Default is an empty string, which means bounds are not explicitly set.

    _vmin : float, optional
        The minimum value for the colorbar. This is used to define the lower limit of the colormap.
        Default is an empty string.

    _vmax : float, optional
        The maximum value for the colorbar. This is used to define the upper limit of the colormap.
        Default is an empty string.

    _figsize_cb : tuple of float, optional
        The size of the figure for the colorbar in inches as (width, height). Default is an empty string.

    _primary_fs : int, optional
        The primary font size for the colorbar labels and title. Default is 18.

    _cb_orient : str, optional
        The orientation of the colorbar, either 'vertical' or 'horizontal'. Default is an empty string.

    _cb_axis_label : str, optional
        The label for the colorbar axis. This text will be displayed alongside the colorbar.
        Default is an empty string.

    _cb_label_xpos : float, optional
        The x-position for the colorbar label. This adjusts the horizontal positioning of the label.
        Default is an empty string.

    _cb_label_ypos : float, optional
        The y-position for the colorbar label. This adjusts the vertical positioning of the label.
        Default is an empty string.

    _fformat : str, optional
        The format for saving the colorbar image. Supported formats are 'png' and 'pdf'. 
        Default is an empty string.

    _output_path : str, optional
        The directory path where the colorbar image will be saved. Default is an empty string.

    _fname : str, optional
        The filename to use when saving the colorbar image. This should not include the file extension.
        Default is an empty string.

    Returns:
    --------
    None
        This function does not return any value. It saves the colorbar as a separate image file in the specified format.
    '''
    
    import numpy as np
    import matplotlib.pyplot as plt

    plt.figure(figsize=_figsize_cb)
    plt.rc('font', size=_primary_fs) # font_size
    if len(_cb_bounds)!=0:
        a = np.array([bounds])
        img = plt.imshow(a, cmap=_colormap, norm=norm)
    else:
        a = np.array([[_vmin,_vmax]])
        img = plt.imshow(a, cmap=_colormap)
        
    plt.gca().set_visible(False)
    if _cb_orient=='vertical':
        cax = plt.axes([0.1, 0.2, 0.06, 1.15])
        cb = plt.colorbar(orientation='vertical', cax=cax)
        cb.ax.set_title(_cb_axis_label, fontsize=_primary_fs, x=_cb_label_xpos, y=_cb_label_ypos, rotation=90) # font_size
        if _fformat=='png':
            plt.savefig(_output_path+_fname+'_cbvert.'+_fformat, dpi=150, bbox_inches='tight')
        elif _fformat=='pdf':
            plt.savefig(_output_path+_fname+"_cbvert."+_fformat, format=_fformat, bbox_inches='tight')
    if _cb_orient=='horizontal':
        cax = plt.axes([0.1, 0.2, 1.15, 0.06])
        cb = plt.colorbar(orientation='horizontal', cax=cax)
        cb.ax.set_title(_cb_axis_label, fontsize=_primary_fs, x=_cb_label_xpos, y=_cb_label_ypos) # font_size
        if _fformat=='png':
            plt.savefig(_output_path+_fname+'_cbhorz.'+_fformat, dpi=150, bbox_inches='tight')
        elif _fformat=='pdf':
            plt.savefig(_output_path+_fname+"_cbhorz."+_fformat, format=_fformat, bbox_inches='tight')