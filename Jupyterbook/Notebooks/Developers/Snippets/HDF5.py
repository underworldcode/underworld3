# # Checking HDF5 files
#
#

import h5py


# +
# m_file = h5py.File("./chkpt/test_checkpointing_np4.orig.msh.h5")
# print(m_file['topology']['order'][()])
# print(m_file['geometry']['vertices'][()])
# m_file.close()


# +
def h5_scan(filename):

    h5file = h5py.File(filename)
    entities = []
    h5file.visit(entities.append)
    return entities

display(h5_scan("./chkpt/test_checkpointing_np4.mesh.0.h5"))
display(h5_scan("./chkpt/test_checkpointing_np4.P.0.h5"))
display(h5_scan("./chkpt/test_checkpointing_np4.U.0.h5"))
# -

m_file = h5py.File("./chkpt/test_checkpointing_np4.mesh.0.h5")
print(m_file['topology']['order'][()])
print(m_file['geometry']['vertices'][()])
m_file.close()

# +
p_file = h5py.File("./chkpt/test_checkpointing_np1.P.0.h5")
P1 = p_file['fields']['P'][()][0:5].T
P1v = p_file['vertex_fields']['P_P1'][()][0:5].T
p_file.close()

p_file = h5py.File("./chkpt/test_checkpointing_np2.P.0.h5")
P2 = p_file['fields']['P'][()][0:5].T
P2v = p_file['vertex_fields']['P_P1'][()][0:5].T
p_file.close()

p_file = h5py.File("./chkpt/test_checkpointing_np4.P.0.h5")
P4 = p_file['fields']['P'][()][0:5].T
P4v = p_file['vertex_fields']['P_P1'][()][0:5].T
p_file.close()

m_file = h5py.File("./chkpt/test_checkpointing_np1.mesh.0.h5")
X1 = m_file['geometry']['vertices'][()][0:5,0].T
m_file.close()

m_file = h5py.File("./chkpt/test_checkpointing_np2.mesh.0.h5")
X2 = m_file['geometry']['vertices'][()][0:5,0].T
m_file.close()

m_file = h5py.File("./chkpt/test_checkpointing_np4.mesh.0.h5")
X4 = m_file['geometry']['vertices'][()][0:5,0].T
m_file.close()


# -

display(P1)
display(P1v)
display(X1)
display("---")
display(P2)
display(P2v)
display(X2)
display("---")
display(P4)
display(P4v)
display(X4)


# So we can see that the mesh vertices and the vector entries are saved consistently in each checkpoint but are different from the original ordering (inherent in the mesh produced / saved on a single process). 
#
