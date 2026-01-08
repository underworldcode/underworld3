r"""
Adaptive mesh refinement (AMR) utilities.

This module provides mesh adaptation capabilities for Underworld3, enabling
dynamic refinement and coarsening based on solution features or error
estimates. AMR is particularly useful for problems with localized features
such as boundary layers, shear zones, or phase boundaries.

Key Functions
-------------
metric_from_gradient : function
    Create adaptation metric from gradient of a scalar field. Refines
    where gradients are steep, coarsens where field is smooth.

metric_from_field : function
    Create adaptation metric from any scalar indicator field. General-purpose
    utility for creating metrics from error estimates, phase indicators, etc.

create_metric : function
    Create adaptation metric directly from target edge lengths (h-field).
    Low-level utility used by other metric functions.

mesh_adapt_meshVar : function
    Adapt mesh based on a metric MeshVariable (internal utility).

mesh2mesh_meshVariable : function
    Transfer a MeshVariable from one mesh to another using swarm intermediary.
    Useful for checkpoint/restart workflows.

Notes
-----
**Metric Tensor Mathematics**

For isotropic mesh adaptation, MMG/PETSc uses a metric tensor:

.. math::

    M = h^{-2} \cdot I

where :math:`h` is the target edge length and :math:`I` is the identity
matrix. This relationship is **dimension-independent** - the same formula
applies in 2D and 3D because the metric defines edge lengths, not areas
or volumes.

The adaptation algorithm seeks to make all edges have unit length in the
metric space (i.e., :math:`\mathbf{e}^T M \mathbf{e} = 1` for edge vector
:math:`\mathbf{e}`). Higher metric values produce smaller elements.

**Boundary Label Handling**

The boundary label stacking utilities handle the constraint that PETSc's
adaptive meshing interpolates only one boundary label at a time. The
stacking approach combines multiple gmsh-generated boundary labels into
a single composite label for adaptation, then unstacks them afterward.

This module requires PETSc to be compiled with adaptive mesh support
(pragmatic, mmg, or parmmg).

See Also
--------
underworld3.discretisation : Mesh classes that can be adapted.
underworld3.meshing : Mesh generation utilities.
underworld3.meshing.Surface.refinement_metric : Surface-based adaptation.
"""
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


# =============================================================================
# Public API: Metric Creation Functions
# =============================================================================


def create_metric(
    mesh: "Mesh",
    h_values: np.ndarray,
    name: str = None,
) -> "MeshVariable":
    r"""Create adaptation metric from target edge lengths.

    This is the core utility that converts target edge lengths (h-field) to
    the metric tensor format required by MMG/PETSc mesh adaptation.

    Parameters
    ----------
    mesh : Mesh
        The mesh to create the metric on.
    h_values : np.ndarray
        Array of target edge lengths at each mesh node. Shape should be
        (n_nodes,) or (n_nodes, 1).
    name : str, optional
        Name for the metric MeshVariable. Defaults to "adaptation_metric".

    Returns
    -------
    MeshVariable
        Scalar MeshVariable containing metric values ready for mesh.adapt().

    Notes
    -----
    **Metric Tensor Mathematics**

    For isotropic mesh adaptation, MMG/PETSc uses a metric tensor:

    .. math::

        M = h^{-2} \cdot I

    where :math:`h` is the target edge length and :math:`I` is the identity
    matrix. This relationship is **dimension-independent** - the same formula
    applies in 2D and 3D.

    Higher metric values produce smaller elements. The adaptation algorithm
    seeks to make :math:`\mathbf{e}^T M \mathbf{e} = 1` for all edges.

    Examples
    --------
    >>> # Create metric from h-field computed elsewhere
    >>> h_field = compute_error_based_h(solution)  # User function
    >>> metric = uw.adaptivity.create_metric(mesh, h_field)
    >>> mesh.adapt(metric)

    See Also
    --------
    metric_from_gradient : Create metric from scalar field gradient.
    metric_from_field : Create metric from indicator field.
    """
    if name is None:
        name = "adaptation_metric"

    # Ensure h_values is the right shape
    h_values = np.asarray(h_values).flatten()

    # Create metric MeshVariable
    metric = uw.discretisation.MeshVariable(name, mesh, 1, degree=1)

    with mesh.access(metric):
        # Convert to metric tensor: M = 1/h² × I (isotropic)
        # This is dimension-independent: same formula for 2D and 3D
        metric.data[:, 0] = 1.0 / (h_values ** 2)

    return metric


def metric_from_gradient(
    field: "MeshVariable",
    h_min: float,
    h_max: float,
    gradient_min: float = None,
    gradient_max: float = None,
    profile: str = "linear",
    name: str = None,
) -> "MeshVariable":
    r"""Create adaptation metric from gradient of a scalar field.

    Produces a metric that refines where gradients are steep (high |∇φ|)
    and coarsens where the field is smooth (low |∇φ|). This is the standard
    approach for error-driven or feature-based mesh adaptation.

    Parameters
    ----------
    field : MeshVariable
        Scalar MeshVariable whose gradient drives refinement. Must have
        num_components=1.
    h_min : float
        Target edge length where gradient is highest (finest mesh).
    h_max : float
        Target edge length where gradient is lowest (coarsest mesh).
    gradient_min : float, optional
        Gradient magnitude below this uses h_max. If None, uses 5th percentile
        of |∇φ| values.
    gradient_max : float, optional
        Gradient magnitude above this uses h_min. If None, uses 95th percentile
        of |∇φ| values.
    profile : str, optional
        Interpolation profile: "linear", "smoothstep", or "power" (default: "linear").
        - "linear": h varies linearly with gradient magnitude
        - "smoothstep": smooth S-curve transition (C¹ continuous)
        - "power": h ∝ |∇φ|^(-1/2), natural for error equidistribution
    name : str, optional
        Name for the metric MeshVariable. Defaults to "{field.name}_gradient_metric".

    Returns
    -------
    MeshVariable
        Scalar MeshVariable containing metric values ready for mesh.adapt().

    Notes
    -----
    **Gradient-Based Refinement Strategy**

    The idea is that steep gradients indicate regions where the solution is
    changing rapidly - these need finer resolution to capture accurately.
    Smooth regions can use coarser mesh without losing accuracy.

    The mapping is:
        - High |∇φ| → small h → large metric → finer mesh
        - Low |∇φ| → large h → small metric → coarser mesh

    **Choosing h_min and h_max**

    - ``h_min`` controls finest resolution (where gradients are steepest)
    - ``h_max`` controls coarsest resolution (smooth regions)
    - Ratio ``h_max/h_min`` gives refinement factor (e.g., 10 = 10× finer at peaks)

    **Auto-detection of Gradient Range**

    If ``gradient_min`` and ``gradient_max`` are not specified, they are
    computed from the actual gradient field:
        - gradient_min = 5th percentile of |∇φ|
        - gradient_max = 95th percentile of |∇φ|

    This ensures robust behavior even when gradient magnitudes span many
    orders of magnitude.

    **Implementation Note**

    Gradients are computed using the Clement interpolant via
    ``uw.function.evaluate(field.sym.diff(x), coords)``. This uses PETSc's
    ``DMPlexComputeGradientClementInterpolant`` which averages cell-wise
    gradients at vertices. The result is O(h) accurate and fast (no linear
    solve required).

    Examples
    --------
    >>> # Refine based on temperature gradient
    >>> metric = uw.adaptivity.metric_from_gradient(
    ...     T, h_min=0.005, h_max=0.05, profile="smoothstep"
    ... )
    >>> mesh.adapt(metric)

    >>> # Refine based on strain rate
    >>> # First compute strain rate magnitude as scalar field
    >>> SR = uw.discretisation.MeshVariable("SR", mesh, 1)
    >>> # ... populate SR with strain rate second invariant ...
    >>> metric = uw.adaptivity.metric_from_gradient(SR, h_min=0.01, h_max=0.1)
    >>> mesh.adapt(metric)

    See Also
    --------
    create_metric : Create metric from h-field directly.
    metric_from_field : Create metric from indicator field (not gradient).
    """
    mesh = field.mesh

    if field.num_components != 1:
        raise ValueError(
            f"metric_from_gradient requires scalar field (num_components=1), "
            f"got {field.num_components}"
        )

    # Compute gradient at mesh nodes using Clement interpolant
    # This uses PETSc's DMPlexComputeGradientClementInterpolant which
    # averages cell-wise gradients at vertices - O(h) accurate, no solve needed
    gradient_at_nodes = uw.function.compute_clement_gradient_at_nodes(field)

    # Compute gradient magnitude
    grad_mag = np.sqrt(np.sum(gradient_at_nodes ** 2, axis=1))

    # Handle gradient bounds
    if gradient_min is None:
        gradient_min = np.percentile(grad_mag, 5)
    if gradient_max is None:
        gradient_max = np.percentile(grad_mag, 95)

    # Avoid division by zero
    gradient_range = gradient_max - gradient_min
    if gradient_range < 1e-15:
        # Uniform field - use h_max everywhere
        h_values = np.full_like(grad_mag, h_max)
    else:
        # Normalize to [0, 1]
        t = np.clip((grad_mag - gradient_min) / gradient_range, 0.0, 1.0)

        # Apply profile
        if profile == "linear":
            # Linear: t directly maps to interpolation parameter
            h_values = h_max - (h_max - h_min) * t

        elif profile == "smoothstep":
            # Smoothstep: S-curve for smoother transition
            smooth_t = 3 * t**2 - 2 * t**3
            h_values = h_max - (h_max - h_min) * smooth_t

        elif profile == "power":
            # Power law: h ∝ |∇φ|^(-1/2) for error equidistribution
            # This gives optimal convergence for some error norms
            # Map t to h via: h = h_max * (1 - t*(1 - h_min/h_max))^2
            # Equivalent to h ∝ 1/sqrt(gradient) behavior
            h_ratio = h_min / h_max
            h_values = h_max * ((1.0 - t) + t * h_ratio)

        else:
            raise ValueError(
                f"Unknown profile: {profile}. Use 'linear', 'smoothstep', or 'power'"
            )

    # Create metric
    if name is None:
        name = f"{field.name}_gradient_metric"

    return create_metric(mesh, h_values, name=name)


def metric_from_field(
    indicator: "MeshVariable",
    h_min: float,
    h_max: float,
    indicator_min: float = None,
    indicator_max: float = None,
    invert: bool = False,
    profile: str = "linear",
    name: str = None,
) -> "MeshVariable":
    r"""Create adaptation metric from an indicator field.

    Maps a scalar indicator field (e.g., error estimate, phase field, distance)
    to target edge lengths. This is more general than gradient-based adaptation -
    you provide any field indicating where refinement is needed.

    Parameters
    ----------
    indicator : MeshVariable
        Scalar field indicating where refinement is needed. Higher values
        (by default) produce finer mesh.
    h_min : float
        Target edge length where indicator is highest (finest mesh).
    h_max : float
        Target edge length where indicator is lowest (coarsest mesh).
    indicator_min : float, optional
        Indicator values below this use h_max. If None, uses field minimum.
    indicator_max : float, optional
        Indicator values above this use h_min. If None, uses field maximum.
    invert : bool, optional
        If True, high indicator values → coarse mesh (swap h_min/h_max roles).
        Useful when indicator represents "smoothness" rather than "need for
        refinement". Default: False.
    profile : str, optional
        Interpolation profile: "linear" or "smoothstep". Default: "linear".
    name : str, optional
        Name for the metric MeshVariable. Defaults to "{indicator.name}_metric".

    Returns
    -------
    MeshVariable
        Scalar MeshVariable containing metric values ready for mesh.adapt().

    Notes
    -----
    **Use Cases**

    - **Error estimates**: Pass a computed error field; high error → fine mesh
    - **Phase fields**: Refine at interfaces (|φ| near transition value)
    - **Distance fields**: Refine near surfaces (use with Surface.distance)
    - **Material boundaries**: Refine near composition gradients

    **Relationship to Surface.refinement_metric()**

    This function is a general-purpose version. Surface.refinement_metric()
    is a specialized wrapper that computes the indicator from surface distance.

    Examples
    --------
    >>> # Refine based on error estimate
    >>> error = compute_error_estimate(solution)  # User function
    >>> metric = uw.adaptivity.metric_from_field(error, h_min=0.005, h_max=0.05)
    >>> mesh.adapt(metric)

    >>> # Refine at phase boundaries (φ transitions from 0 to 1)
    >>> # Want fine mesh where φ is near 0.5
    >>> phi_interface = 1 - 4 * (phi - 0.5)**2  # Peak at φ=0.5
    >>> metric = uw.adaptivity.metric_from_field(phi_interface, h_min=0.01, h_max=0.1)

    See Also
    --------
    create_metric : Create metric from h-field directly.
    metric_from_gradient : Create metric from field gradient.
    """
    mesh = indicator.mesh

    if indicator.num_components != 1:
        raise ValueError(
            f"metric_from_field requires scalar field (num_components=1), "
            f"got {indicator.num_components}"
        )

    # Get indicator values
    with mesh.access(indicator):
        ind_values = indicator.data[:, 0].copy()

    # Handle indicator bounds
    if indicator_min is None:
        indicator_min = np.min(ind_values)
    if indicator_max is None:
        indicator_max = np.max(ind_values)

    # Handle inversion
    if invert:
        h_min, h_max = h_max, h_min

    # Normalize to [0, 1]
    indicator_range = indicator_max - indicator_min
    if indicator_range < 1e-15:
        # Uniform field - use h_max everywhere
        h_values = np.full_like(ind_values, h_max)
    else:
        t = np.clip((ind_values - indicator_min) / indicator_range, 0.0, 1.0)

        # Apply profile (high indicator → small h)
        if profile == "linear":
            h_values = h_max - (h_max - h_min) * t

        elif profile == "smoothstep":
            smooth_t = 3 * t**2 - 2 * t**3
            h_values = h_max - (h_max - h_min) * smooth_t

        else:
            raise ValueError(
                f"Unknown profile: {profile}. Use 'linear' or 'smoothstep'"
            )

    # Create metric
    if name is None:
        name = f"{indicator.name}_metric"

    return create_metric(mesh, h_values, name=name)


# =============================================================================
# Internal Utilities
# =============================================================================


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
