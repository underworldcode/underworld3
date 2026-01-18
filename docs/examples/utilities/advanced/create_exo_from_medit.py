# %% [markdown]
"""
# 🎓 create exo from medit

**PHYSICS:** utilities  
**DIFFICULTY:** advanced  
**MIGRATED:** From underworld3-documentation/Notebooks

## Description
This example has been migrated from the original UW3 documentation.
Additional documentation and parameter annotations will be added.

## Migration Notes
- Original complexity preserved
- Parameters to be extracted and annotated
- Claude hints to be added in future update
"""

# %% [markdown]
"""
## Original Code
The following is the migrated code with minimal modifications.
"""

# %%
# ### Convert mesh to exo
#
# 1. I have managed to create tetra blocks from mesh data
#
# ##### Todo:
# 2. Convert triangle data into surfaces in exo file
# 3. Convert edge data into lines in exo file

import re
import numpy as np
import datetime
from netCDF4 import Dataset
import os

# +
# data types
numpy_to_exodus_dtype = {
                            'float64': 'f8',
                            'float32': 'f4',
                            'int32': 'i4',
                            'int64': 'i8'
                        }

meshio_to_exodus_type = {
                            'tetra': 'TETRA'
                        }


# +
def read_medit_mesh(input_file):
    """
    Read medit file
    Return vertices, tetrahedra, cell_ids
    """
    with open(input_file, 'r') as file:
        data = file.read()

    # Parse vertices
    vertices = []
    vertex_pattern = re.compile(r'Vertices\n(\d+)\n((?:[0-9eE\+\-\.]+\s+[0-9eE\+\-\.]+\s+[0-9eE\+\-\.]+\s+\d+\s*\n)+)', re.MULTILINE)
    vertex_match = vertex_pattern.search(data)
    if vertex_match:
        vertices = np.array([list(map(float, line.split()[:3])) for line in vertex_match.group(2).strip().split('\n')])

    # Parse tetrahedra
    tetrahedra = []
    cell_ids = []
    tetra_pattern = re.compile(r'Tetrahedra\n(\d+)\n((?:\d+\s+\d+\s+\d+\s+\d+\s+\d+\s*\n)+)', re.MULTILINE)
    tetra_match = tetra_pattern.search(data)
    if tetra_match:
        for line in tetra_match.group(2).strip().split('\n'):
            parts = list(map(int, line.split()))
            tetrahedra.append(parts[:4])
            cell_ids.append(parts[4])
        tetrahedra = np.array(tetrahedra) - 1

    return vertices, tetrahedra, cell_ids

def write_exodus_file(filename, vertices, tetrahedra, cell_ids):
    """
    Write to exo file
    """
    with Dataset(filename, "w") as rootgrp:
        # Set global data
        now = datetime.datetime.now().isoformat()
        rootgrp.title = f"Created by custom script, {now}"
        rootgrp.version = np.float32(5.1)
        rootgrp.api_version = np.float32(5.1)
        rootgrp.floating_point_word_size = 8

        # Set dimensions
        total_num_elems = tetrahedra.shape[0]
        rootgrp.createDimension("num_nodes", len(vertices))
        rootgrp.createDimension("num_dim", vertices.shape[1])
        rootgrp.createDimension("num_elem", total_num_elems)
        unique_cell_ids = np.unique(cell_ids)
        rootgrp.createDimension("num_el_blk", len(unique_cell_ids))
        rootgrp.createDimension("num_node_sets", 0)  # Adjust if you have node sets
        rootgrp.createDimension("len_string", 33)
        rootgrp.createDimension("len_line", 81)
        rootgrp.createDimension("four", 4)
        rootgrp.createDimension("time_step", None)

        # Dummy time step
        data = rootgrp.createVariable("time_whole", "f4", ("time_step"))
        data[:] = 0.0

        # Points
        coor_names = rootgrp.createVariable(
            "coor_names", "S1", ("num_dim", "len_string")
        )
        coor_names.set_auto_mask(False)
        coor_names[0, 0] = b"X"
        coor_names[1, 0] = b"Y"
        if vertices.shape[1] == 3:
            coor_names[2, 0] = b"Z"
        data = rootgrp.createVariable(
            "coord",
            numpy_to_exodus_dtype[vertices.dtype.name],
            ("num_dim", "num_nodes"))
        data[:] = vertices.T

        # Cells
        eb_prop1 = rootgrp.createVariable("eb_prop1", "i4", "num_el_blk")
        eb_names = rootgrp.createVariable("eb_names", "S1", ("num_el_blk", "len_string"))

        for i, cell_id in enumerate(unique_cell_ids):
            eb_prop1[i] = i + 1  # Unique ID for each block
            block_name = f"block_{cell_id}_{i + 1}_{now}".ljust(33)
            eb_names[i, :] = np.array(list(block_name[:33]), dtype='S1')
            block_tetrahedra = tetrahedra[np.array(cell_ids) == cell_id]
            dim1 = f"num_el_in_blk{i + 1}"
            dim2 = f"num_nod_per_el{i + 1}"
            rootgrp.createDimension(dim1, block_tetrahedra.shape[0])
            rootgrp.createDimension(dim2, block_tetrahedra.shape[1])
            dtype = numpy_to_exodus_dtype[block_tetrahedra.dtype.name]
            data = rootgrp.createVariable(f"connect{i + 1}", dtype, (dim1, dim2))
            data.elem_type = "TETRA"
            # Exodus is 1-based
            data[:] = block_tetrahedra + 1

        # Add additional variables
        ids = rootgrp.createVariable("ids", "i4", ("num_elem"))
        object_id = rootgrp.createVariable("object_id", "i4", ("num_elem"))
        vtkblockcolor = rootgrp.createVariable("vtkblockcolor", "f4", ("num_elem"))
        vtkcompositeindex = rootgrp.createVariable("vtkcompositeindex", "i4", ("num_elem"))

        ids[:] = np.arange(1, total_num_elems + 1)
        object_id[:] = np.arange(1, total_num_elems + 1)
        vtkblockcolor[:] = np.random.random(total_num_elems)
        vtkcompositeindex[:] = np.ones(total_num_elems)

def convert_mesh_to_exo(input_file, output_file):
    """
    Function to convert mesh to exo file
    """
    vertices, tetrahedra, cell_ids = read_medit_mesh(input_file)
    write_exodus_file(output_file, vertices, tetrahedra, cell_ids)
    print(f"Converted {input_file} to {output_file}")

# +
# if os.path.isfile('./meshout.exo'):
#     os.remove('./meshout.exo')
              
# # Specify input and output file paths
# input_file = '../meshout.mesh'
# output_file = './meshout.exo'

# # Convert the .mesh file to .exo file
# convert_mesh_to_exo(input_file, output_file)

# +
# # !{ncdump -h meshout.exo}

# +
# # Define the path to the ParaView executable
# paraview_executable = '/Applications/ParaView-5.11.2.app/Contents/MacOS/paraview'  # Example path for macOS

# # Define the path to the Exodus file
# exo_file_path = './meshout.exo'  # Example path

# # Command to open ParaView with the Exodus file
# command = f'{paraview_executable} --data={exo_file_path}'

# # Run the command
# # !{command}
# -


