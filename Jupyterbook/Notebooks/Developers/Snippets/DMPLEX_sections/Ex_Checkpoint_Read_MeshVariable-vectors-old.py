# %%
# Enable timing (before uw imports)

import os

os.environ["UW_TIMING_ENABLE"] = "0"

from petsc4py import PETSc
import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import timing

import numpy as np
import sympy
from mpi4py import MPI

import h5py

if uw.mpi.size == 1:
    os.makedirs("output", exist_ok=True)
else:
    os.makedirs(f"output_np{uw.mpi.size}", exist_ok=True)

# %%
# Define the problem size
#      1  - ultra low res for automatic checking
#      2  - low res problem to play with this notebook
#      3  - medium resolution (be prepared to wait)
#      4  - highest resolution
#      5+ - v. high resolution (parallel only)

problem_size = uw.options.getInt("model_size", default=2)

if problem_size <= 1:
    res = 8
elif problem_size == 2:
    res = 16
elif problem_size == 3:
    res = 32
elif problem_size == 4:
    res = 48
elif problem_size == 5:
    res = 64
elif problem_size >= 6:
    res = 128

# %%
expt_name = f"test_checkpointing_np{uw.mpi.size}"

# %%
## Mesh - checkpointed and we are going to read it regardless of the number of
## processes when we restore the data

mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(-1.0, 0.0),
    maxCoords=(1.0, 1.0),
    cellSize=1.0 / res,
    regular=False,
    qdegree=3,
    filename=f"{expt_name}.orig.msh",
)

meshfilename = "test_checkpointing_np1.mesh.0.h5"
mesh = uw.discretisation.Mesh(meshfilename)

# Variables (vector, scalar) and copies
v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
vc = uw.discretisation.MeshVariable("Uc", mesh, mesh.dim, degree=2)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1, continuous=True)
pc = uw.discretisation.MeshVariable("Pc", mesh, 1, degree=1, continuous=True)
index1p = uw.discretisation.MeshVariable(
    "Index1proc", mesh, 1, degree=1, continuous=True
)
index1pc = uw.discretisation.MeshVariable(
    "Index1procc", mesh, 1, degree=1, continuous=True
)

if mesh.sf is None:
    mesh.sf = mesh.dm.getDefaultSF()


# %%

# %%
# gvec0 = mesh.dm0.createGlobalVec()
# gvec0.setName("Index1proc")

# filename="test_checkpointing_np1.Index1proc.0.h5"
# viewer = PETSc.ViewerHDF5().create(filename, "r", comm=PETSc.COMM_WORLD)
# gvec0.load(viewer)

# print(f"GVEC0 ownership: {gvec0.getOwnershipRanges()}")


# %%
iset = mesh.dm.getVertexNumbering()
indices = iset.getIndices()

with mesh.access(p, v, vc, pc, index1p, index1pc):
    p.data[:, 0] = p.coords[:, 0]
    v.data[:, :] = v.coords[:, :]
    vc.data[:, :] = 0.0
    pc.data[:, 0] = 0.0
    index1p.data[:, 0] = indices.astype(float)
    index1pc.data[:, 0] = 0.0

# %%
# with mesh.access(p,v, vc, pc, iv):
#     print(iv.data.astype(int).T)

# %%
# Do this for the serial case as a reference point

mesh.write_visualisation_xdmf(
    expt_name, meshUpdates=True, meshVars=[p, v, index1p], index=0
)


# %%
# mesh.dm.getOrdering(PETSc.Mat.OrderingType.NATURAL).getIndices()

# %%
# mesh.dm.getVertexNumbering().getIndices()


# %%
with mesh.access(vc, pc, index1pc):
    vc.data[:, :] = 0.0
    pc.data[:, 0] = 0.0
    index1pc.data[:, 0] = 0.0

pc.load_from_checkpoint(f"test_checkpointing_np1.P.0.h5", data_name="P")
index1pc.load_from_checkpoint(
    f"test_checkpointing_np1.Index1proc.0.h5", data_name="Index1proc"
)

# %%
with mesh.access(p, v, vc, pc):
    if uw.mpi.rank == 0:
        print(f"P   - {p.data[0:10].T}", flush=True)
        print(f"Pc  - {pc.data[0:10].T}", flush=True)
        print(f"Ixc - {index1pc.data[0:10].astype(int).T}", flush=True)
        print(f"Ix  - {index1p.data[0:10].astype(int).T}", flush=True)


# %%
mesh.write_visualisation_xdmf(
    expt_name + "_c", meshUpdates=True, meshVars=[p, v, index1p, index1pc], index=0
)

# %%
if mesh.sf == None:
    mesh.sf = mesh.dm.getDefaultSF()

isf = mesh.sf.createInverse()

sec = mesh.dm.getDefaultSection()
lvec = mesh.dm.getLocalVec()
gvec = mesh.dm.getGlobalVec()

# push this vec into that one

isset, subdm = mesh.dm.createSubDM(index1p.field_id)
pcvec = gvec.getSubVector(isset)
subdm.localToGlobal(index1p._lvec, pcvec)
gvec.restoreSubVector(isset, pcvec)
mesh.dm.globalToLocal(gvec, lvec)

# Do the permutation

newsec, newvec = mesh.dm.distributeField(mesh.sf, sec, lvec)
mesh.dm.localToGlobal(newvec, gvec)
pcvec = gvec.getSubVector(isset)
index1p._gvec.array[...] = pcvec.array[...]
subdm.globalToLocal(index1p._gvec, index1p._lvec)

if uw.mpi.rank == 0:
    print(f"Index Permutation - {index1p._lvec.array[0:10].T}", flush=True)


# %%
0 / 0

# %%
# newsec, newvec = mesh.dm.distributeField(mesh.sf, sec, lvec)
# mesh.dm.localToGlobal(newvec, gvec)
# pcvec = gvec.getSubVector(isset)
# index1p._gvec.array[...] = pcvec.array[...]
# subdm.globalToLocal(index1p._gvec, index1p._lvec)


# %%
# newsec.view()

# %%
if uw.mpi.rank == 0:
    print(f"Pcp - {pc._lvec.array[0:10].T}", flush=True)

# %%
if not uw.is_notebook:
    exit()
else:
    0 / 0

# %%
mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(-1.0, 0.0),
    maxCoords=(1.0, 1.0),
    cellSize=1.0 / res,
    regular=False,
    qdegree=3,
    filename=f"{expt_name}.orig.msh",
)

# Variables (vector, scalar) and copies
v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
vc = uw.discretisation.MeshVariable("Uc", mesh, mesh.dim, degree=2)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1, continuous=True)
pc = uw.discretisation.MeshVariable("Pc", mesh, 1, degree=1, continuous=True)


# %%
with mesh.access(p, v, vc, pc):
    p.data[:, 0] = p.coords[:, 0]
    v.data[:, :] = v.coords[:, :]
    vc.data[:, :] = 0.0
    pc.data[:, 0] = 0.0


# %%
mesh.write_visualisation_xdmf(expt_name, meshUpdates=True, meshVars=[p, v], index=0)

# %%
if mesh.sf == None:
    mesh.sf = mesh.dm.getDefaultSF()

# %%
# newis, subdm = mesh.dm.createSubDM(p.field_id)
# gv = subdm.createGlobalVec()
# subdm.getDefaultSF().view()

# %%
newis, subdm = mesh.dm.createSubDM(pc.field_id)
gv = subdm.createGlobalVec()

sectiondm = subdm.clone()
sectiondm.setName("P")
sf = subdm.getDefaultSF()

print(f"SF: {sf.getGraph()[0]}", flush=True)

filename = f"{expt_name}.{'P'}.1.h5"
print(f"Read section from: {filename}", flush=True)

viewer = PETSc.ViewerHDF5().create(filename, "r", comm=PETSc.COMM_WORLD)
globalsf, localsf = mesh.dm.sectionLoad(viewer, sectiondm, sf)

subdm.view()

# %%
print(f"GlobalSF: {globalsf.getGraph()[0]}", flush=True)
print(f"LocalSF: {localsf.getGraph()[0]}", flush=True)

# %%

# %%

# %%
# Now, build a mesh using the checkpointed mesh file and read
# the information back into equivalent variables on that mesh.
# In parallel, we can't guarantee the two meshes will have the same
# decomposition or data ordering but the data should be consistent.

# # ? Is this the case, is it only true for P1 variables (in a P1 mesh)

meshfilename = f"{expt_name}.mesh.0.h5"
mesh3 = uw.discretisation.Mesh(meshfilename)

# Check distribution etc

if uw.mpi.rank == 0:
    print(f"Original =================", flush=True)
mesh.dm.view()

if uw.mpi.rank == 0:
    print(f"New =================", flush=True)
mesh3.dm.view()


# %%
print(f"Original =================", flush=True)
if mesh.sf is not None:
    mesh.sf.view()

# %%
# Variables (vector, scalar) and copies
v3 = uw.discretisation.MeshVariable("U3", mesh3, mesh3.dim, degree=2)
p3 = uw.discretisation.MeshVariable("P3", mesh3, 1, degree=1, continuous=True)

# %%
v3.load_from_checkpoint(f"{expt_name}.U.0.h5", data_name="U")
p3.load_from_checkpoint(f"{expt_name}.P.0.h5", data_name="P")

vc.load_from_checkpoint(f"{expt_name}.U.0.h5", data_name="U")
pc.load_from_checkpoint(f"{expt_name}.P.0.h5", data_name="P")


# %%
with mesh3.access():
    V3 = v3.data.copy()
    P3 = p3.data.copy()

with mesh.access():
    VC = vc.data.copy()
    PC = pc.data.copy()

# %%
if uw.mpi.rank == 0:
    print(f"Original =================", flush=True)
    print(mesh.data[0:10, 0], flush=True)

# %%
if uw.mpi.rank == 0:
    print(f"New =================", flush=True)
    print(mesh3.data[0:10, 0], flush=True)

# %%
if uw.mpi.rank == 0:
    print(f"=================", flush=True)

# %%
0 / 0

# %%
mesh.write_visualisation_xdmf(
    f"viz_chpt_np{uw.mpi.size}", meshUpdates=True, meshVars=[p, v], index=0
)


# %%
import h5py

h5 = h5py.File(f"viz_chpt_np{uw.mpi.size}.U.0.h5", "r")
print(h5.keys())
U = h5["fields"]["U"][()]
h5.close()

if uw.mpi.rank == 0:
    print("U ", U[0:7, 0].T)

# %%
import h5py

h5 = h5py.File(f"viz_chpt_np{uw.mpi.size}.U.1.h5", "r")
U1 = h5["fields"]["U"][()]
h5.close()

import h5py

h5 = h5py.File(f"viz_chpt_np{uw.mpi.size}.P.1.h5", "r")
P1 = h5["fields"]["P"][()]
h5.close()

if uw.mpi.rank == 0:
    print("U1", U1[0:7, 0].T)
    print("P1", P1[0:7].T)

# %%
import h5py

h5 = h5py.File(f"viz_chpt_np{uw.mpi.size}.U.2.h5", "r")
U2 = h5["fields"]["U"][()]
h5.close()

if uw.mpi.rank == 0:
    print("U2", U2[0:7, 0].T)

# %%
vc.load_from_checkpoint(f"viz_chpt_np{uw.mpi.size}.U.0.h5", data_name="U")

# %%
with mesh.access():
    # it should be fine to test this proc-by-proc
    assert (vc.data - v.data).max() < 0.001
    if uw.mpi.rank == 0:
        print("Mesh 1 - re-load success", flush=True)

    assert (vc.data - vc.coords).max() < 0.001
    if uw.mpi.rank == 0:
        print("Mesh 1 validation success", flush=True)


# %%

# %%
# Another option, create an identical mesh and read the checkpoints back in
# This is likely to work but not if the decomposition in not deterministic
# (which is not guaranteed, obviously)

mesh2 = uw.meshing.UnstructuredSimplexBox(
    minCoords=(-1.0, 0.0),
    maxCoords=(1.0, 1.0),
    cellSize=1.0 / res,
    regular=False,
    qdegree=3,
    filename="testmesh2.msh",
)

v2 = uw.discretisation.MeshVariable("U", mesh2, mesh2.dim, degree=2)
p2 = uw.discretisation.MeshVariable("P", mesh2, 1, degree=1, continuous=True)

v2.load_from_checkpoint(f"viz_chpt_np{uw.mpi.size}.U.0.h5", data_name="U")
p2.load_from_checkpoint(f"viz_chpt_np{uw.mpi.size}.P.0.h5", data_name="P")


with mesh.access(), mesh2.access():
    # it should be fine to test this proc-by-proc
    assert (v2.data - v.data).max() < 0.001
    if uw.mpi.rank == 0:
        print("Mesh 2 re-load success")

    assert (v2.data - v2.coords).max() < 0.001
    if uw.mpi.rank == 0:
        print("Mesh 2 validation success")


# %%
# Another option, read the checkpointed mesh.
# This might give different a different ordering, but the assertion that
# the values are equivalent to the mesh coords should be fine.

meshfilename = f"viz_chpt_np{uw.mpi.size}.mesh.0.h5"
mesh3 = uw.discretisation.Mesh(meshfilename)

mesh.dm.view()
mesh3.dm.view()

v3 = uw.discretisation.MeshVariable("U", mesh3, mesh3.dim, degree=2)
p3 = uw.discretisation.MeshVariable("P", mesh3, 1, degree=1, continuous=True)

v3.load_from_checkpoint(f"viz_chpt_np{uw.mpi.size}.U.0.h5", data_name="U")
p3.load_from_checkpoint(f"viz_chpt_np{uw.mpi.size}.P.0.h5", data_name="P")

with mesh3.access(), mesh.access():
    print(f"i   - {uw.mpi.rank}: ", v3.data[0:7, 0].T)
    print(f"ii  - {uw.mpi.rank}: ", v3.coords[0:7, 0].T)
    print(f"iii - {uw.mpi.rank}: ", v.data[0:7, 0].T)
    # print((v3.data - v3.coords).max())

# %%
with mesh.access(), mesh3.access():
    assert (v3.data - v3.coords).max() < 0.001
    if uw.mpi.rank == 0:
        print("Mesh 3 validation success")

    # it should be fine to test this proc-by-proc
    assert (v3.data - v.data).max() < 0.001
    if uw.mpi.rank == 0:
        print("Mesh 3 re-load success")


# %%

# %%

# %%

# %%
0 / 0

# %%
import h5py

h5 = h5py.File(f"viz_chpt_np1.mesh.0.h5", "r")
print(h5.keys())
print(h5["geometry"]["vertices"][()])
h5.close()

# %%
import h5py

h5 = h5py.File(f"viz_chpt_np1.U.0.h5", "r")
print(h5.keys())
print(h5["vertex_fields"]["U_P2"][()])
h5.close()

# %%
import h5py

h5 = h5py.File(f"viz_chpt_np2.mesh.0.h5", "r")
print(h5.keys())
print(h5["geometry"]["vertices"][()])
h5.close()

# %%
import h5py

h5 = h5py.File(f"viz_chpt_np2.U.0.h5", "r")
print(h5.keys())
print(h5["vertex_fields"]["U_P2"][()])
h5.close()

# %%
# newis, subdm = mesh.dm.createSubDM([pc.field_id])
# subdm.setName(p.clean_name)
# sec = subdm.getDefaultGlobalSection()
# sec.setUp()
# # sec.view()

# %%
# Saved that one ...

# viewer = PETSc.ViewerHDF5().create(f"test_vec_p_save.np{uw.mpi.size}.h5", "w", comm=PETSc.COMM_WORLD)
# viewer(p._gvec)
# viewer.destroy()

# mesh.sf0.view()

# %%
# sf = mesh.dm.getDefaultSF()
# iset, subdm = mesh.dm.createSubDM(p.field_id)
# ssf = subdm.getDefaultSF()
# iset.view()


# %%
# import h5py
# h5 = h5py.File(f"test_vec_p_save.np1.h5", "r")
# print(h5.keys())
# print(h5['fields']['P'][()][0:5])
# h5.close()

# %%

# %%

# %%
import h5py

h5 = h5py.File(f"test_uw_np2.checkpoint.0.h5", "r")
print(h5.keys())
print(h5["topology"]["uw_mesh"]["dms"]["uw_mesh"]["order"][()])
doff = h5["topologies"]["uw_mesh"]["dms"]["P"]["section"]["field0"]["atlasOff"][()]
h5.close()

# %%

# %%

stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel(mesh.dim)

stokes.add_dirichlet_bc((0.0), ["Top", "Bottom"], [1])
stokes.add_dirichlet_bc((0.0), ["Left", "Right"], [0])

# %%
swarm = uw.swarm.Swarm(mesh=mesh)
material = uw.swarm.IndexSwarmVariable(
    "M", swarm, indices=2, proxy_continuous=True, proxy_degree=1
)
swarm.populate(fill_param=5)

blob = np.array([[sphereCentre[0], sphereCentre[1], sphereRadius, 1]])

with swarm.access(material):
    material.data[...] = materialLightIndex

    for i in range(blob.shape[0]):
        cx, cy, r, m = blob[i, :]
        inside = (swarm.data[:, 0] - cx) ** 2 + (swarm.data[:, 1] - cy) ** 2 < r**2
        material.data[inside] = m

# %%
mat_density = np.array([densityBG, densitySphere])
density = mat_density[0] * material.sym[0] + mat_density[1] * material.sym[1]

mat_viscosity = np.array([viscBG, viscSphere])
viscosityMat = mat_viscosity[0] * material.sym[0] + mat_viscosity[1] * material.sym[1]

viscosity = viscBG * material.sym[0] + viscSphere * material.sym[1]

# %%
stokes.constitutive_model.Parameters.viscosity = viscosity
stokes.bodyforce = sympy.Matrix([0, -1 * density])
stokes.penalty = 1.0
stokes.saddle_preconditioner = 1.0 / viscosity

snes_rtol = 1.0e-6
stokes.tolerance = snes_rtol
stokes.petsc_options["ksp_monitor"] = None

stokes._setup_terms()
stokes.solve(zero_init_guess=True)

# %%
## Now we have data we can save
mesh.write_visualisation_xdmf(
    f"test_viz_np{uw.mpi.size}", meshUpdates=True, meshVars=[p, v]
)


# %%
mesh.write_checkpoint(f"test_uw_np{uw.mpi.size}", meshUpdates=True, meshVars=[p, v])

# %%
viewer = PETSc.ViewerHDF5().create(
    f"test_uw_np{2}.section.0.h5", "w", comm=PETSc.COMM_WORLD
)
section = mesh.dm.getDefaultGlobalSection()
sectiondm = mesh.dm.clone()
sectiondm.setName("sectiondm")
sectiondm.setDefaultSection(section)
mesh.dm.sectionView(viewer, sectiondm)

# %%
import h5py

h5 = h5py.File(f"test_uw_np{2}.section.0.h5", "r")
print(h5.keys())
print(h5["topologies"]["uw_mesh"]["dms"]["sectiondm"]["order"])
h5.close()

# %%

# %%
# # Can I read back from this checkpoint ?

# viewer = PETSc.ViewerHDF5().create(f"test_uw_np{2}.checkpoint.0.h5", "r", comm=PETSc.COMM_WORLD)

# # ssectiondm = mesh.dm.clone()
# # ssectiondm.setName("uw_mesh")
# # localsf, globalsf = mesh.dm.sectionLoad(viewer, ssectiondm, mesh.sf0)

# # 0/0

# newis, subdm = mesh.dm.createSubDM([pc.field_id])
# subdm.setName(p.clean_name)
# # psectiondm = subdm.clone()
# # psectiondm.setName("uw_section")
# # mesh.dm.sectionView(viewer, psectiondm)

# # Try to read it back

# psectiondm = subdm.clone()
# psectiondm.setName(p.clean_name)
# plocalsf, pglobalsf = mesh.dm.sectionLoad(viewer, psectiondm, mesh.sf0 )
# pc._gvec.setName(p.clean_name)
# mesh.dm.globalVectorLoad(viewer, psectiondm, pglobalsf, pc._gvec)
# pc._gvec.setName(pc.clean_name)
# subdm.globalToLocal(pc._gvec, pc._lvec, addv=False)


# %%

# %%
with mesh.access():
    print(pc.data[10:20].T)
    print(p.data[10:20].T)

# %%

# %%
0 / 0

# %%
import h5py

h5 = h5py.File("test_save_load.mesh.0.h5", "r")
print(h5.keys())
h5.close()

# %%
# Mesh plus section in h5

viewer = PETSc.ViewerHDF5().create(
    "test_save_load.mesh.1.h5", "w", comm=PETSc.COMM_WORLD
)
# viewer(mesh.dm)
mesh.dm.view(viewer)
mesh.dm.sectionView(viewer, mesh.dm)

## Can you read it ?

ssectiondm = mesh.dm.clone()
ssectiondm.setName("uw_mesh")
localsf, globalsf = mesh.dm.sectionLoad(viewer, ssectiondm, mesh.sf0)

# %%


# %%

# %%

# %%

# %%
import h5py

h5 = h5py.File("test_save_load.v.1.h5", "r")
print(h5.keys())
print(h5["topologies"]["uw_mesh"]["dms"]["U"]["vecs"]["U"]["U"][()][0:10])
h5.close()

# %%
import h5py

h5 = h5py.File("test_save_load.p.1.h5", "r")
print(h5["topologies"]["uw_mesh"]["dms"]["P"]["vecs"]["P"]["P"][()][0:10])
h5.close()

# %%
import h5py

h5 = h5py.File("test_save_load.U.0.h5", "r")
print(h5.keys())
# print(h5['topologies']['uw_mesh']['dms'].keys())
h5.close()

# %%
# iset, subdm = mesh.dm.createSubDM(v.field_id)
# subdm.setName(v.clean_name)
# sect = subdm.getDefaultGlobalSection()
# sect.getFieldComponents(0)
# sect.getFieldName(0)

# %%
# Save section ?

# sectiondm = mesh.dm.clone()
# sectiondm.setName("uw_section")

# section = mesh.dm.getDefaultGlobalSection()
viewer = PETSc.ViewerHDF5().create(
    "test_save_load.section.0.h5", "w", comm=PETSc.COMM_WORLD
)
mesh.dm.sectionView(viewer, mesh.dm)

# Read it back in

# viewer = PETSc.ViewerHDF5().create("test_save_load.section.0.h5", "r", comm=PETSc.COMM_WORLD)
ssectiondm = mesh.dm.clone()
ssectiondm.setName("uw_mesh")
localsf, globalsf = mesh.dm.sectionLoad(viewer, ssectiondm, mesh.sf)


# %%

# %%
import h5py

h5 = h5py.File("test_save_load.mesh.0.h5", "r")
print(h5.keys())
print(h5["topologies"]["uw_mesh"]["dms"]["uw_mesh"].keys())
h5.close()


# %%
h5.close()

# %%
viewer = PETSc.ViewerHDF5().create("test_save_load.P3.0.h5", "w", comm=PETSc.COMM_WORLD)
newis, subdm = mesh.dm.createSubDM([p.field_id])
subdm.setName("uw_mesh")
mesh.dm.globalVectorView(viewer, subdm, p._gvec)

psectiondm = subdm.clone()
psectiondm.setName("uw_section")
mesh.dm.sectionView(viewer, psectiondm)

# Try to read it back

psectiondmr = subdm.clone()
psectiondmr.setName("uw_section")
plocalsf, pglobalsf = mesh.dm.sectionLoad(viewer, psectiondmr, mesh.sf)

mesh.dm.globalVectorLoad(viewer, psectiondm, pglobalsf, p._gvec)

# %%
sec = subdm.getDefaultGlobalSection()

# %%
sec.view()

# %%
psectiondmr.view()

# %%
0 / 0

# %%
import h5py

h5 = h5py.File("test_save_load.P3.0.h5", "r")
print(h5["topologies"]["uw_mesh"]["dms"]["uw_section"]["section"].keys())
h5.close()

# %%
viewer = PETSc.ViewerHDF5().create("test_save_load.P3.0.h5", "r", comm=PETSc.COMM_WORLD)
mesh.dm.globalVectorLoad(viewer, ssectiondm, globalsf, p._gvec)

# %%

# %%
viewer = PETSc.ViewerHDF5().create("test_save_load.P2.0.h5", "w", comm=PETSc.COMM_WORLD)
newis, subdm = mesh.dm.createSubDM([p.field_id])
subdm.setName(p.clean_name)
mesh.dm.globalVectorView(viewer, subdm, p._gvec)


# %%
# newis, subdm = mesh.dm.createSubDM([v.field_id])


# %%
viewer = PETSc.ViewerHDF5().create("test_save_load.P2.0.h5", "r", comm=PETSc.COMM_WORLD)
mesh.dm.globalVectorLoad(viewer, subdm, mesh.sf, p._gvec)

# %%

# %%
0 / 0

# %%
import h5py

h5 = h5py.File("test_save_load.P2.0.h5", "r")

# %%
h5["topologies"].keys()

# %%
h5["topologies"]["uw_mesh_topology"]["dms"]["P"]["vecs"]["P"]["P"][()]

# %%

# %%

# %%

# %%
v_stats = mesh.stats(v.sym[0])
p_stats = mesh.stats(p.sym[0])

if uw.mpi.rank == 0:
    print(v_stats, flush=True)
    print(p_stats, flush=True)


# %%
if uw.mpi.rank == 0:
    print("New mesh to match existing")

mesh2 = uw.meshing.UnstructuredSimplexBox(
    minCoords=(-1.0, 0.0),
    maxCoords=(1.0, 1.0),
    cellSize=1.0 / res,
    regular=False,
    qdegree=3,
)

v2 = uw.discretisation.MeshVariable("U2", mesh2, mesh2.dim, degree=2)
p2 = uw.discretisation.MeshVariable("P2", mesh2, 1, degree=1, continuous=True)

if uw.mpi.rank == 0:
    print("reload U,P from h5 files")

v2.load("test_save_load.U.0.h5", data_name="U")
p2.load("test_save_load.P.0.h5", data_name="P")

# %%

# %%
filename = "test_save_load.P.0.h5"
viewer = PETSc.ViewerHDF5().create(filename, "r", comm=PETSc.COMM_WORLD)

# %%
newis, subdm = mesh2.dm.createSubDM(p.field_id)
subdm.setName("P")
mesh2.dm.globalVectorLoad(viewer, subdm, mesh2.sf, p._gvec)


# %%
import h5py

h5 = h5py.File("./SampleData/Stokes_Sphere_RT_0.1_1.0s.mesh.0.h5")

# %%
v_stats = mesh.stats(v2.sym[0])
p_stats = mesh.stats(p2.sym[0])

if uw.mpi.rank == 0:
    print(v_stats, flush=True)
    print(p_stats, flush=True)


# %%
if uw.mpi.rank == 0:
    print("reload mesh from h5 file")

mesh3 = uw.discretisation.Mesh("test_save_load.mesh.0.h5")
v3 = uw.discretisation.MeshVariable("U3", mesh3, mesh3.dim, degree=2)
p3 = uw.discretisation.MeshVariable("P3", mesh3, 1, degree=1, continuous=True)

if uw.mpi.rank == 0:
    print("reload U,P from h5 files")

# v3.load("test_save_load.U.0.h5", data_name="U")
# p3.load("test_save_load.P.0.h5", data_name="P")

# %%

# %%
with mesh3.access(p3):
    p3.data[...] = 0.0

# %%
mesh3.dm.setName("uw_mesh_topology")
viewer = PETSc.ViewerHDF5().create("test_save_load.P2.0.h5", "r", comm=PETSc.COMM_WORLD)
indexset, subdm = mesh3.dm.createSubDM(p3.field_id)
subdm.setName("P")
mesh3.dm.globalVectorLoad(viewer, subdm, mesh3.sf, p3._gvec)

# %%

# %%
sectiondm = mesh3.dm.clone()
mesh3.dm.sectionLoad(viewer, sectiondm, mesh3.sf)

# %%
mesh3.dm.globalVectorLoad(viewer, mesh3.dm, mesh3.sf, p3._gvec)

# %%
viewer = PETSc.ViewerHDF5().create(
    "./SampleData/Stokes_Sphere_RT_0.1_1.0s.mesh.0.h5", "r", comm=PETSc.COMM_WORLD
)

dm = PETSc.DMPlex().create(comm=PETSc.COMM_WORLD)
sf = dm.topologyLoad(viewer)
dm.coordinatesLoad(viewer, sf)
dm.sectionLoad(viewer, dm, sf)
v = dm.getCoordinates()
v.array

# %%

# %%

# %%
v_stats = mesh.stats(v3.sym[0])
p_stats = mesh.stats(p3.sym[0])

if uw.mpi.rank == 0:
    print(v_stats, flush=True)
    print(p_stats, flush=True)


# %%
## Check the order is OK

with mesh.access():
    print(v.data[10:20, :])
with mesh2.access():
    print(v2.data[10:20, :])
with mesh3.access():
    print(v3.data[10:20, :])


# %%
with mesh.access(), mesh2.access():
    print(v.data[10:20, :] - v2.data[10:20, :])

with mesh.access(), mesh3.access():
    print(v.data[10:20, :] - v3.data[10:20, :])

# %%
