"""Shared distributed FE numbering and value extraction."""
import hashlib
import numpy as np
from petsc4py import PETSc
def _layout(field):
    """Independent field global numbering, including local ghost-node indices."""
    iset, dm = field.mesh.dm.createSubDM(field.field_id)
    global_vector = dm.createGlobalVec()
    mapping = dm.getLGMap()
    indices = mapping.getIndices().copy()
    size = global_vector.getSize()
    global_vector.destroy()
    mapping.destroy()
    dm.destroy()
    iset.destroy()
    return indices, size


def _scatter_values(vector, indices):
    """Fetch local/ghost field values without replicating the volume solution."""
    iset = PETSc.IS().createGeneral(
        np.asarray(indices, dtype=PETSc.IntType), comm=PETSc.COMM_SELF
    )
    local = PETSc.Vec().createSeq(len(indices), comm=PETSc.COMM_SELF)
    scatter = PETSc.Scatter().create(vector, iset, local, None)
    scatter.scatter(
        vector,
        local,
        addv=PETSc.InsertMode.INSERT_VALUES,
        mode=PETSc.ScatterMode.FORWARD,
    )
    values = local.array.copy()
    scatter.destroy()
    local.destroy()
    iset.destroy()
    return values



def array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes()).hexdigest()

