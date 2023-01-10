# # Checking HDF5 files
#
#

import h5py


# cd /Users/lmoresi/+Underworld/underworld3/Jupyterbook/Notebooks/Examples-StokesFlow/output

file = h5py.File("stokesSphere.mesh.0.h5")

file.keys()

file = h5py.File("/Users/lmoresi/+Underworld/underworld3/Jupyterbook/Notebooks/Developers/Timing/standardMesh.h5")

file.keys()


