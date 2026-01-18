# ### Read medit (.mesh) ascii file
# #### Return mesh data type

from contextlib import contextmanager  # For creating a context manager
from ctypes import c_double, c_float  # For handling C data types in Python
import numpy as np
import re


# +
# Define custom exceptions for error handling


class ReadError(Exception):
    pass


class WriteError(Exception):
    pass


class CorruptionError(Exception):
    pass


# -


def is_buffer(obj, mode):
    """
    Function to check if an object is a buffer and
    supports the required mode (read/write)
    """
    return ("r" in mode and hasattr(obj, "read")) or ("w" in mode and hasattr(obj, "write"))


@contextmanager
def open_file(path_or_buf, mode="r"):
    """Context manager for opening a file or handling an existing buffer"""
    if is_buffer(path_or_buf, mode):
        yield path_or_buf
    else:
        with open(path_or_buf, mode) as f:
            yield f


def read_medit_ascii(filename, mesh_data_name):
    """
    Function to read a medit mesh file in ascii format.
    Return mesh data and their indices if found.
    """
    with open_file(filename) as f:
        mesh = read_ascii_buffer(f, mesh_data_name)
    return mesh


def print_medit_mesh_info(file_path):
    """print medit mesh info"""

    mesh_data_name_list = []

    with open(file_path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                # End of file
                break

            line = line.strip()

            if len(line) == 0 or line[0] == "#":
                continue

            items = line.split()

            # Check for alphabetic characters in the first item
            if items[0].isalpha():
                # Process the MeshVersionFormatted line
                if items[0] == "MeshVersionFormatted":
                    version = items[1]
                    print("Mesh Version Format:", version)

                # Process the Dimension line
                elif items[0] == "Dimension":
                    if len(items) >= 2:
                        dim = int(items[1])
                    else:
                        dim = int(f.readline().strip())
                    print("Dimension:", dim)

                # Print other lines that contain alphabetic characters
                else:
                    if line == "End":
                        pass
                    else:
                        mesh_data_name_list.append(line)
                        # print(f'"{line}"')
                        size = int(f.readline().strip())
                        print(f"Number of {line}:", size)

            else:
                # Ignore lines that do not start with an alphabetic character
                continue

    print("Mesh file contains following data types:", mesh_data_name_list)


# taken from meshio
def read_ascii_buffer(f, mesh_data_name, int_type=np.int32):

    points_in_mesh_data = {
        "Edges": ("line", 2),
        "Triangles": ("triangle", 3),
        "Quadrilaterals": ("quad", 4),
        "Tetrahedra": ("tetra", 4),
        "Prisms": ("wedge", 6),
        "Pyramids": ("pyramid", 5),
        "Hexahedra": ("hexahedron", 8),  # Frey
        "Hexaedra": ("hexahedron", 8),  # Dobrzynski
    }

    while True:
        line = f.readline()
        if not line:
            break

        line = line.strip()

        if len(line) == 0 or line[0] == "#":
            continue

        items = line.split()

        if items[0] == "MeshVersionFormatted":
            version = items[1]
            dtype = {"0": c_float, "1": c_float, "2": c_double}[version]

        elif items[0] == "Dimension":
            if len(items) >= 2:
                dim = int(items[1])
            else:
                # e.g. Dimension\n3, where the number of dimensions is on the next line
                dim = int(int(f.readline()))

        # return vertices
        elif items[0] == mesh_data_name and items[0] == "Vertices":
            if dim <= 0:
                raise ReadError()
            if dtype is None:
                raise ReadError("Expected `MeshVersionFormatted` before `Vertices`")

            num_verts = int(f.readline())
            data = np.fromfile(f, count=num_verts * (dim + 1), dtype=dtype, sep=" ").reshape(
                num_verts, dim + 1
            )
            break

        # return cells, triangles, edges
        elif items[0] == mesh_data_name and items[0] in ("Tetrahedra", "Triangles", "Edges"):
            mesh_data_type, points_per_cell = points_in_mesh_data[items[0]]

            num_cells = int(f.readline())  # The first value is the number of elements
            data = np.fromfile(
                f, count=num_cells * (points_per_cell + 1), dtype=int_type, sep=" "
            ).reshape(num_cells, points_per_cell + 1)
            break

        # return corners, RequiredVertices
        elif items[0] == mesh_data_name and items[0] in ("Corners", "RequiredVertices", "Ridges"):
            size = int(f.readline())
            data = np.fromfile(f, count=size, dtype=int_type, sep=" ")
            break

        # return normals, tangents
        elif items[0] == mesh_data_name and items[0] in ("Normals", "Tangents"):
            size = int(f.readline())
            data = np.fromfile(f, count=size * dim, dtype=dtype, sep=" ").reshape(size, dim)
            break

        # return normalatvertices, tangentatvertices
        elif items[0] == mesh_data_name and items[0] in ("NormalAtVertices", "TangentAtVertices"):
            size = int(f.readline())
            data = np.fromfile(f, count=size * 2, dtype=int_type, sep=" ").reshape(size, 2)
            break

        # checking whether file had ended or not
        else:
            if items[0] == "End":
                print("Reached end of the file. Entered mesh data is not found")
                return

    if mesh_data_name in ("Vertices", "Normals", "Tangents"):
        if data.shape[1] == dim:
            return data
        else:
            return data[:, :dim], data[:, dim].astype(int_type)
    else:  # adapt for 0-base
        if mesh_data_name in ("Tetrahedra", "Triangles", "Edges"):
            return data[:, :points_per_cell] - 1, data[:, -1]
        else:
            return data - 1
