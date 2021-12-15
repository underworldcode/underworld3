




import numpy as np
from mpi4py import MPI
COMM_WORLD = MPI.COMM_WORLD
from petsc4py import PETSc

def _from_cell_list(dim, cells, coords, comm):
    """
    Create a DMPlex from a list of cells and coords.

    :arg dim: The topological dimension of the mesh
    :arg cells: The vertices of each cell
    :arg coords: The coordinates of each vertex
    :arg comm: communicator to build the mesh on.
    """
    
    # These types are /correct/, DMPlexCreateFromCellList wants int
    # and double (not PetscInt, PetscReal).
    if comm.rank == 0:
        cells = np.asarray(cells, dtype=np.int32)
        coords = np.asarray(coords, dtype=np.double)
        comm.bcast(cells.shape, root=0)
        comm.bcast(coords.shape, root=0)
        # Provide the actual data on rank 0.
        plex = PETSc.DMPlex().createFromCellList(dim, cells, coords, comm=comm)
    else:
        cell_shape  = list(comm.bcast(None, root=0))
        coord_shape = list(comm.bcast(None, root=0))
        cell_shape[0] = 0
        coord_shape[0] = 0
        # Provide empty plex on other ranks
        # A subsequent call to plex.distribute() takes care of parallel partitioning
        plex = PETSc.DMPlex().createFromCellList(dim,
                                                 np.zeros( cell_shape, dtype=np.int32),
                                                 np.zeros(coord_shape, dtype=np.double),
                                                 comm=comm)
    return plex


def _cubedsphere_cells_and_coords(radius, refinement_level):
    """Generate vertex and face lists for cubed sphere """
    # We build the mesh out of 6 panels of the cube
    # this allows to build the gnonomic cube transformation
    # which is defined separately for each panel

    # Start by making a grid of local coordinates which we use
    # to map to each panel of the cubed sphere under the gnonomic
    # transformation
    dtheta = 2**(-refinement_level+1)*np.arctan(1.0)
    a = 3.0**(-0.5)*radius
    theta = np.arange(np.arctan(-1.0), np.arctan(1.0)+dtheta, dtheta, dtype=np.double)
    x = a*np.tan(theta)
    Nx = x.size

    # Compute panel numberings for each panel
    # We use the following "flatpack" arrangement of panels
    #   3
    #  102
    #   4
    #   5

    # 0 is the bottom of the cube, 5 is the top.
    # All panels are numbered from left to right, top to bottom
    # according to this diagram.

    panel_numbering = np.zeros((6, Nx, Nx), dtype=np.int32)

    # Numbering for panel 0
    panel_numbering[0, :, :] = np.arange(Nx**2, dtype=np.int32).reshape(Nx, Nx)
    count = panel_numbering.max()+1

    # Numbering for panel 5
    panel_numbering[5, :, :] = count + np.arange(Nx**2, dtype=np.int32).reshape(Nx, Nx)
    count = panel_numbering.max()+1

    # Numbering for panel 4 - shares top edge with 0 and bottom edge
    #                         with 5
    # interior numbering
    panel_numbering[4, 1:-1, :] = count + np.arange(Nx*(Nx-2),
                                                    dtype=np.int32).reshape(Nx-2, Nx)

    # bottom edge
    panel_numbering[4, 0, :] = panel_numbering[5, -1, :]
    # top edge
    panel_numbering[4, -1, :] = panel_numbering[0, 0, :]
    count = panel_numbering.max()+1

    # Numbering for panel 3 - shares top edge with 5 and bottom edge
    #                         with 0
    # interior numbering
    panel_numbering[3, 1:-1, :] = count + np.arange(Nx*(Nx-2),
                                                    dtype=np.int32).reshape(Nx-2, Nx)
    # bottom edge
    panel_numbering[3, 0, :] = panel_numbering[0, -1, :]
    # top edge
    panel_numbering[3, -1, :] = panel_numbering[5, 0, :]
    count = panel_numbering.max()+1

    # Numbering for panel 1
    # interior numbering
    panel_numbering[1, 1:-1, 1:-1] = count + np.arange((Nx-2)**2,
                                                       dtype=np.int32).reshape(Nx-2, Nx-2)
    # left edge of 1 is left edge of 5 (inverted)
    panel_numbering[1, :, 0] = panel_numbering[5, ::-1, 0]
    # right edge of 1 is left edge of 0
    panel_numbering[1, :, -1] = panel_numbering[0, :, 0]
    # top edge (excluding vertices) of 1 is left edge of 3 (downwards)
    panel_numbering[1, -1, 1:-1] = panel_numbering[3, -2:0:-1, 0]
    # bottom edge (excluding vertices) of 1 is left edge of 4
    panel_numbering[1, 0, 1:-1] = panel_numbering[4, 1:-1, 0]
    count = panel_numbering.max()+1

    # Numbering for panel 2
    # interior numbering
    panel_numbering[2, 1:-1, 1:-1] = count + np.arange((Nx-2)**2,
                                                       dtype=np.int32).reshape(Nx-2, Nx-2)
    # left edge of 2 is right edge of 0
    panel_numbering[2, :, 0] = panel_numbering[0, :, -1]
    # right edge of 2 is right edge of 5 (inverted)
    panel_numbering[2, :, -1] = panel_numbering[5, ::-1, -1]
    # bottom edge (excluding vertices) of 2 is right edge of 4 (downwards)
    panel_numbering[2, 0, 1:-1] = panel_numbering[4, -2:0:-1, -1]
    # top edge (excluding vertices) of 2 is right edge of 3
    panel_numbering[2, -1, 1:-1] = panel_numbering[3, 1:-1, -1]
    count = panel_numbering.max()+1

    # That's the numbering done.

    # Set up an array for all of the mesh coordinates
    Npoints = panel_numbering.max()+1
    coords = np.zeros((Npoints, 3), dtype=np.double)
    lX, lY = np.meshgrid(x, x)
    lX.shape = (Nx**2,)
    lY.shape = (Nx**2,)
    r = (a**2 + lX**2 + lY**2)**0.5

    # Now we need to compute the gnonomic transformation
    # for each of the panels
    panel_numbering.shape = (6, Nx**2)

    def coordinates_on_panel(panel_num, X, Y, Z):
        I = panel_numbering[panel_num, :]
        coords[I, 0] = radius / r * X
        coords[I, 1] = radius / r * Y
        coords[I, 2] = radius / r * Z

    coordinates_on_panel(0, lX, lY, -a)
    coordinates_on_panel(1, -a, lY, -lX)
    coordinates_on_panel(2, a, lY, lX)
    coordinates_on_panel(3, lX, a, lY)
    coordinates_on_panel(4, lX, -a, -lY)
    coordinates_on_panel(5, lX, -lY, a)

    # Now we need to build the face numbering
    # in local coordinates
    vertex_numbers = np.arange(Nx**2, dtype=np.int32).reshape(Nx, Nx)
    local_faces = np.zeros(((Nx-1)**2, 4), dtype=np.int32)
    local_faces[:, 0] = vertex_numbers[:-1, :-1].reshape(-1)
    local_faces[:, 1] = vertex_numbers[1:, :-1].reshape(-1)
    local_faces[:, 2] = vertex_numbers[1:, 1:].reshape(-1)
    local_faces[:, 3] = vertex_numbers[:-1, 1:].reshape(-1)

    cells = panel_numbering[:, local_faces].reshape(-1, 4)
    return cells, coords


# def CubedSphereMesh(radius, refinement_level=0, degree=1,
#                     reorder=None, distribution_parameters=None, comm=COMM_WORLD):
#     """Generate an cubed approximation to the surface of the
#     sphere.

#     :arg radius: The radius of the sphere to approximate.
#     :kwarg refinement_level: optional number of refinements (0 is a cube).
#     :kwarg degree: polynomial degree of coordinate space (defaults
#         to 1: bilinear quads)
#     :kwarg reorder: (optional), should the mesh be reordered?
#     :kwarg comm: Optional communicator to build the mesh on (defaults to
#         COMM_WORLD).
#     """
#     if refinement_level < 0 or refinement_level % 1:
#         raise RuntimeError("Number of refinements must be a non-negative integer")

#     if degree < 1:
#         raise ValueError("Mesh coordinate degree must be at least 1")

#     cells, coords = _cubedsphere_cells_and_coords(radius, refinement_level)
#     plex = mesh._from_cell_list(2, cells, coords, comm)

#     m = mesh.Mesh(plex, dim=3, reorder=reorder, distribution_parameters=distribution_parameters)

#     if degree > 1:
#         new_coords = function.Function(functionspace.VectorFunctionSpace(m, "Q", degree))
#         new_coords.interpolate(ufl.SpatialCoordinate(m))
#         # "push out" to sphere
#         new_coords.dat.data[:] *= (radius / np.linalg.norm(new_coords.dat.data, axis=1)).reshape(-1, 1)
#         m = mesh.Mesh(new_coords)
#     m._radius = radius
#     return m

