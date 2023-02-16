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

# The boundary stacking is to


def _dm_stack_bcs(dm, boundaries, stacked_bc_label_name):

    if boundaries is None:
        return

    dm.removeLabel(stacked_bc_label_name)
    dm.createLabel(stacked_bc_label_name)
    stacked_bc_label = dm.getLabel(stacked_bc_label_name)

    for b in boundaries:
        bc_label_name = b.name
        lab = dm.getLabel(bc_label_name)

        if not lab:
            continue

        lab_is = lab.getStratumIS(b.value)

        # Load this up on the stack
        stacked_bc_label.setStratumIS(b.value, lab_is)


def _dm_unstack_bcs(dm, boundaries, stacked_bc_label_name):
    """Unpack boundary labels to the list of names"""

    if boundaries is None:
        return

    stacked_bc_label = dm.getLabel(stacked_bc_label_name)
    vals = stacked_bc_label.getNonEmptyStratumValuesIS().getIndices()

    for v in vals:
        try:
            b = boundaries(v)  # ValueError if mismatch
        except ValueError:
            continue

        dm.removeLabel(b.name)
        dm.createLabel(b.name)
        b_dmlabel = dm.getLabel(b.name)

        lab_is = stacked_bc_label.getStratumIS(v)
        b_dmlabel.setStratumIS(v, lab_is)

    return


def mesh_adapt_meshVar(mesh, meshVarH):

    # Create / use a field on the old mesh to hold the metric
    # Perhaps that should be a user-definition

    boundaries = mesh.boundaries

    field_id = None
    for i in range(mesh.dm.getNumFields()):
        f, _ = mesh.dm.getField(i)
        if f.getName() == "AdaptationMetricField":
            field_id = i

    if field_id is None:
        field_id = mesh.dm.getNumFields()

    hvec = meshVarH._lvec
    metric_vec = mesh.dm.metricCreateIsotropic(hvec, field_id)
    f, _ = mesh.dm.getField(field_id)
    f.setName("AdaptationMetricField")

    _dm_stack_bcs(mesh.dm, boundaries, "CombinedBoundaries")
    dm_a = mesh.dm.adaptMetric(metric_vec, bdLabel="CombinedBoundaries")
    _dm_stack_bcs(dm_a, boundaries, "CombinedBoundaries")

    meshA = uw.meshing.Mesh(
        dm_a,
        simplex=mesh.dm.isSimplex,
        coordinate_system_type=mesh.CoordinateSystem.coordinate_type,
        qdegree=mesh.qdegree,
        refinement=None,
        refinement_callback=mesh.refinement_callback,
        boundaries=mesh.boundaries,
    )

    return meshA
