from typing import Optional, Tuple
from enum import Enum

import tempfile
import numpy as np
import petsc4py
from petsc4py import PETSc
import os

import underworld3 as uw
from underworld3.discretisation import Mesh
from underworld3 import VarType
from underworld3.coordinates import CoordinateSystemType
import underworld3.timing as timing

import sympy


# Utilities for mesh adaptation etc

# The boundary stacking is to get around the fact that, at present, the
# adaptive meshing will only interpolate one boundary label. We are using the
# gmsh workflow that generates multiple labels (with one value).


def _dm_stack_bcs(dm, boundaries, stacked_bc_label_name):
    if boundaries is None:
        return

    dm.removeLabel(stacked_bc_label_name)
    uw.mpi.barrier()

    dm.createLabel(stacked_bc_label_name)
    stacked_bc_label = dm.getLabel(stacked_bc_label_name)

    for b in boundaries:
        bc_label_name = b.name
        lab = dm.getLabel(bc_label_name)

        if not lab:
            continue

        lab_is = lab.getStratumIS(b.value)

        # Load this up on the stack
        if lab_is:
            stacked_bc_label.setStratumIS(b.value, lab_is)


def _dm_unstack_bcs(dm, boundaries, stacked_bc_label_name):
    """Unpack boundary labels to the list of names"""

    if boundaries is None:
        return

    stacked_bc_label = dm.getLabel(stacked_bc_label_name)
    vals = stacked_bc_label.getNonEmptyStratumValuesIS().getIndices()

    # Clear labels just in case
    for b in boundaries:
        dm.removeLabel(b.name)

    uw.mpi.barrier()

    for b in boundaries:
        dm.createLabel(b.name)

    uw.mpi.barrier()

    for v in vals:
        try:
            b = boundaries(v)  # ValueError if mismatch
        except ValueError:
            continue

        b_dmlabel = dm.getLabel(b.name)
        lab_is = stacked_bc_label.getStratumIS(v)
        b_dmlabel.setStratumIS(v, lab_is)

    return


# dmAdaptLabel - but first you need to call dmplex set default distribute False


def mesh_adapt_meshVar(mesh, meshVarH, metricVar, verbose=False, redistribute=False):
    # Create / use a field on the old mesh to hold the metric
    # Perhaps that should be a user-definition

    boundaries = mesh.boundaries

    # The metric
    field_id = metricVar.field_id
    hvec = meshVarH._lvec
    metric_vec = mesh.dm.metricCreateIsotropic(hvec, field_id)

    if verbose:
        print(f"{uw.mpi.rank} dm adaptation ... begin", flush=True)

    _dm_stack_bcs(mesh.dm, boundaries, "CombinedBoundaries")
    dm_a = mesh.dm.adaptMetric(metric_vec, bdLabel="CombinedBoundaries")

    icoord_vec = dm_a.getCoordinates()

    _dm_unstack_bcs(dm_a, boundaries, "CombinedBoundaries")

    if verbose:
        print(f"{uw.mpi.rank} dm adaptation ... complete", flush=True)

    meshA = uw.meshing.Mesh(
        dm_a,
        simplex=mesh.dm.isSimplex,
        coordinate_system_type=mesh.CoordinateSystem.coordinate_type,
        qdegree=mesh.qdegree,
        refinement=None,
        refinement_callback=mesh.refinement_callback,
        boundaries=mesh.boundaries,
        distribute=redistribute,
    )

    if verbose:
        print(f"{uw.mpi.rank} mesh adaptation / distribution ... complete", flush=True)

    return icoord_vec, meshA


def mesh2mesh_swarm(mesh0, mesh1, swarm0, swarmVarList, proxy=True, verbose=False):
    """Warning [NSFW] - this uses EXPLICIT message passing calls to handle the
    situation where a swarm cell_dm cannot find particles after mesh redistribution.
    This occurs when particles are moved accross non-neighbouring processes or if the
    mesh neighbours are redistricted. This should be fixed at the DMSwarm / DMPlex level
    so this code is just a placeholder. Or maybe it's just user error !

    Notes 1: This copies a swarm from one mesh to another allowing for completely incommensurate
    partitionings. Warning: this is not always a 1->1 mapping. There may be some duplication and
    particles may go missing along curved boundaries where the meshes do not necessarily overlap.
    The same is true in the shadow spaces.

    Note 2: The swarm is "adapted" to the original mesh and will need
    to be repopulated on the new one, or data can be mapped to a purpose-built swarm.

    Note 3: We pass the data around as floats for the time being. Be careful when converting back.

    Note 4: set proxy=True to automatically generate proxy variables on mesh1 but consider skipping
    if the returned swarm is ephemeral
    """

    with swarm0.access():
        swarm_data = swarm0._particle_coordinates.data.copy()
        for swarmVar in swarmVarList:
            swarm_data = np.hstack((swarm_data, np.ascontiguousarray(swarmVar.data.astype(float))))

    s_coords0 = np.ascontiguousarray(swarm_data[:, 0 : mesh0.dim])

    cell = mesh1.get_closest_local_cells(s_coords0)
    found = np.where(cell >= 0)[0]
    not_found = np.where(cell == -1)[0]

    n_found = found.shape[0]
    n_not_found = not_found.shape[0]

    # print(
    #     f"{uw.mpi.rank} - A/local found: {n_found} v. not found: {n_not_found}",
    #     flush=False,
    # )

    # Let's sync the number of missing points by rank

    mpi_size = uw.mpi.size

    missing_points = np.zeros((mpi_size))
    comm = uw.mpi.comm
    root = 0

    local_array = n_not_found
    # print(f"rank: {uw.mpi.rank}, local_array size: {n_not_found}")

    sendbuf = swarm_data[not_found]
    global_sizes = np.empty(mpi_size, dtype=int)
    local_size = np.array([sendbuf.shape[0]], dtype=int)

    comm.Allgather(local_size, global_sizes)

    global_size = global_sizes.sum()
    # print(
    #     f"{uw.mpi.rank} Sizes are: {global_sizes} Buffer size: {global_size}",
    #     flush=True,
    # )

    uw.mpi.barrier()

    recvbuf = np.empty((global_size, sendbuf.shape[1]), dtype=float)

    comm.Allgatherv(sendbuf=sendbuf, recvbuf=(recvbuf, global_sizes * sendbuf.shape[1]))

    global_unallocated_coords = recvbuf[:, 0 : mesh0.dim].copy()
    global_unallocated_data = recvbuf.copy()

    ## ====================================
    ## Now we have all the information here
    ## and we can hand it out to the new swarm
    ## =====================================

    # if swarm0.recycle_rate > 0:
    #     if swarm0.vars["DMSwarm_Xorig"] not in swarmVarList:
    #         swarmVarList.append(swarm0.vars["DMSwarm_Xorig"])

    swarm1 = uw.swarm.Swarm(mesh=mesh1, recycle_rate=swarm0.recycle_rate)

    for swarmVar in swarmVarList:
        uw.swarm.SwarmVariable(
            swarmVar.name,
            swarm1,
            size=swarmVar.shape,
            vtype=swarmVar.vtype,
            dtype=swarmVar.dtype,
            proxy_degree=swarmVar._proxy_degree,
            proxy_continuous=swarmVar._proxy_continuous,
            _proxy=proxy,
            _register=True,
            rebuild_on_cycle=swarmVar._rebuild_on_cycle,
        )

    swarm1.dm.finalizeFieldRegister()

    # First we add the found points (there are n_found of those)

    found_coords = s_coords0[found]
    adds = n_found + 1

    swarm1.dm.addNPoints(adds)

    ## Update cells etc, but don't migrate as
    ## we don't want to distrupt the data layout

    cellid = swarm1.dm.getField("DMSwarm_cellid")
    coords = swarm1.dm.getField("DMSwarmPIC_coor").reshape((-1, swarm1.dim))

    coords[...] = found_coords[...]
    if n_found > 0:
        cellid[:] = mesh1.get_closest_cells(coords)  ## found points

    swarm1.dm.restoreField("DMSwarmPIC_coor")
    swarm1.dm.restoreField("DMSwarm_cellid")

    # Add in the data fields for the found points

    offset = swarm1.dim
    for swarmVar in swarmVarList:
        varField = swarm1.dm.getField(swarmVar.clean_name)
        varCpts = swarmVar.num_components
        varData = swarm_data[:, offset : offset + varCpts]
        fieldData = varField.reshape(-1, varCpts)

        if n_found > 0:
            fieldData[:, :] = swarm_data[found, offset : offset + varCpts].astype(swarmVar.dtype)

        swarm1.dm.restoreField(swarmVar.clean_name)
        offset += varCpts

    uw.mpi.barrier()

    ## Now deal with all the points we caught
    ## from the broadcast. Again, we only own
    ## some of these. So let's add them now

    cell = mesh1.get_closest_local_cells(global_unallocated_coords)
    # found1 = np.where(cell >= 0)[0]
    # not_found1 = np.where(cell == -1)[0]s
    cell_arr = np.atleast_1d(cell)
    found1 = np.where(cell_arr >= 0)[0]
    not_found1 = np.where(cell_arr == -1)[0]

    n_found1 = found1.shape[0]
    n_not_found1 = not_found1.shape[0]

    if n_found1 > 0:
        psize = swarm1.dm.getLocalSize()
        adds = n_found1  # swarm is not blank, so only need to use N for addNPoints

        swarm1.dm.addNPoints(adds)

        cellid = swarm1.dm.getField("DMSwarm_cellid")
        coords = swarm1.dm.getField("DMSwarmPIC_coor").reshape((-1, swarm1.dim))

        coords[psize + 1 :, :] = global_unallocated_coords[found1, :]

        cellid[psize + 1 :] = cell[found1]  ## gathered points

        swarm1.dm.restoreField("DMSwarm_cellid")
        swarm1.dm.restoreField("DMSwarmPIC_coor")

    # print(
    #     f"{uw.mpi.rank}/i: {swarm1.dm.getLocalSize()} / {swarm1.dm.getSize()} cf {swarm0.dm.getSize()}",
    #     flush=True,
    # )

    uw.mpi.barrier()
    ## Now the variables from the broadcast

    offset = swarm1.dim
    for swarmVar in swarmVarList:
        varField = swarm1.dm.getField(swarmVar.clean_name)
        varCpts = swarmVar.num_components
        varData = global_unallocated_data[:, offset : offset + varCpts]
        fieldData = varField.reshape(-1, varCpts)

        if n_found1 > 0:
            fieldData[psize + 1 :, :] = global_unallocated_data[found1, offset : offset + varCpts]

        swarm1.dm.restoreField(swarmVar.clean_name)
        offset += varCpts

    ## Update proxy variables if present
    for swarm1Var in swarm1.vars.values():
        swarm1Var._update()

    # print(
    #     f"{uw.mpi.rank}-1: Swarm Size (Local) {swarm0.dm.getLocalSize()}; (Global) {swarm1.dm.getSize()}"
    # )
    uw.mpi.barrier()
    swarm1.dm.migrate(remove_sent_points=True)

    # print(
    #     f"{uw.mpi.rank}-2: Swarm Size (Local) {swarm0.dm.getLocalSize()}; (Global) {swarm1.dm.getSize()}"
    # )

    if verbose:
        size0 = swarm0.dm.getSize()
        size1 = swarm1.dm.getSize()

        print("---------", flush=True)

        uw.pprint(f"Swarm0: {size0}; Swarm1: {size1}")
        uw.pprint("---------")

        uw.mpi.barrier()

        print(
            f"{uw.mpi.rank} - Local data size {swarm_data.shape}\n"
            f"{uw.mpi.rank} - Exchanged data size {recvbuf.shape}\n"
            f"{uw.mpi.rank} - Local data found: {n_found} v. not found: {n_not_found}\n"
            f"{uw.mpi.rank} - Exchanged data : {n_found1} v. not found: {n_not_found1}",
            flush=True,
        )

        print(f"{uw.mpi.rank} - Variables defined on the new swarm:")
        for swarmVar in swarmVarList:
            print(
                f"{uw.mpi.rank} -    {swarmVar.clean_name} ({swarmVar.dtype} * {swarmVar.num_components})"
            )

        print("---------", flush=True)

        uw.mpi.barrier()

    return swarm1


## ToDo: this should take a list of vars so that the swarm migration is only done once

## ToDo: this could also be more like the SL advection: launch points from the
## new mesh, use a swarm to find values, snap back when done.
## Could also navigate there by letting the particles walk through the mesh towards
## their goal (if particle-restore works with funny mesh-holes)


def mesh2mesh_meshVariable(meshVar0, meshVar1, verbose=False):
    """Map a meshVar on mesh0 to a meshVar on mesh1 using
    an intermediary (temporary) swarm"""

    mesh0 = meshVar0.mesh
    mesh1 = meshVar1.mesh

    # 1 Create a temporary swarm with a variable that matches meshVar0

    tmp_swarm = uw.swarm.Swarm(mesh0)
    var_name = rf"{meshVar0.clean_name}"
    var_cpts = meshVar0.num_components

    tmp_varS = uw.swarm.SwarmVariable(
        var_name, tmp_swarm, size=(1, var_cpts), vtype=uw.VarType.MATRIX, _proxy=False
    )

    # Maybe 3+ if var is higher order ??
    tmp_swarm.populate(fill_param=3)

    # Set data on the swarmVar

    # print(f"Map data to swarm (rbf) - points = {tmp_swarm.dm.getSize()}", flush=True)

    with tmp_swarm.access(tmp_varS):
        tmp_varS.data[...] = meshVar0.rbf_interpolate(tmp_swarm._particle_coordinates.data)

    # print(f"Distribute swarm", flush=True)

    # Now ship this out to the other mesh via the tmp swarm
    tmp_swarm1 = mesh2mesh_swarm(mesh0, mesh1, tmp_swarm, [tmp_varS], proxy=False, verbose=verbose)

    if hasattr(tmp_swarm1, "_meshVar"):
        raise RuntimeError("Swarm should not have a proxy on it !")

    tmp_swarm1.vars[var_name]._rbf_to_meshVar(meshVar1)

    del tmp_swarm, tmp_swarm1

    return
