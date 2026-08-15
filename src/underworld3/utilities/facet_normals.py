"""ONE rule for the outward normal of a boundary facet.

Three places in the tree build a nodal boundary normal by walking a boundary's facets,
taking each facet's normal from ``dm.computeCellGeometryFVM`` and accumulating it onto
the facet's closure points:

  * :func:`underworld3.utilities.rotated_bc._boundary_velocity_nodes` — the rotated
    free-slip constraint direction,
  * :meth:`underworld3.discretisation.Mesh._assemble_boundary_normal` — the P1 normal
    field behind ``mesh.boundary_normal()``, which ``add_constraint_bc`` and
    ``add_nitsche_bc`` use by default,
  * :func:`underworld3.utilities.boundary_flux._node_normals` — the projection
    direction for a vector reaction.

They were three copies of the same six lines and they drifted: #560/#561 fixed the
orientation rule and the measure weight in the first, leaving the other two on the
pre-#560 rule (which orients against the mean of THIS RANK's coordinates — rank-local,
and inward on a concave boundary). This module is that rule, once.

What the rule is
----------------
``computeCellGeometryFVM`` returns the facet's MEASURE (edge length in 2-D, face area
in 3-D) and a normal whose sign is PETSc's own convention, not the domain's. For an
EXTERIOR facet the domain's outward direction is "away from the one cell the facet
belongs to", which is local geometry — no global reference point — and is correct on a
concave boundary (an annulus or shell inner arc), where orienting away from a
coordinate mean points INTO the domain.

Only an exterior facet has "the one cell it belongs to". An INTERNAL boundary's facets
have two support cells and ``support[0]`` is whichever the DMPlex ordering lists first,
so flipping against it would orient neighbouring facets of the same surface oppositely
and they would CANCEL in a measure-weighted sum. There the raw PETSc normal is returned
unflipped and ``exterior`` is False, and the caller decides what to do about it.

The measure is returned because it is the weight a nodal normal needs to be consistent
with the assembly that integrates the boundary term facet by facet (#560); see
``_boundary_velocity_nodes``' docstring for the derivation.
"""
import numpy as np


def facet_measure_and_normal(dm, facet):
    """``(measure, unit normal, exterior)`` for a height-1 DMPlex point.

    ``measure`` is the facet's length (2-D) / area (3-D). The normal is a unit vector
    of length ``dm.getCoordinateDim()``. ``exterior`` is True when the facet has
    exactly one support cell, in which case the normal has been oriented AWAY from
    that cell — the domain's outward direction. When it is False the normal carries
    PETSc's own sign and no orientation claim is made.

    Purely rank-local: it reads geometry, takes no collective, and says nothing about
    whether this rank sees all of the node's facets. Completing a per-node SUM across
    ranks is the caller's job — a boundary facet is labelled on exactly one rank, so a
    node on a partition seam sees only some of its facets locally (#564).
    """
    measure, centroid, normal = dm.computeCellGeometryFVM(facet)
    n = np.asarray(normal, dtype=float)
    n = n / (np.linalg.norm(n) + 1.0e-30)
    exterior = dm.getSupportSize(facet) == 1
    if exterior:
        _, cell_centroid, _ = dm.computeCellGeometryFVM(int(dm.getSupport(facet)[0]))
        if np.dot(n, np.asarray(centroid) - np.asarray(cell_centroid)) < 0.0:
            n = -n
    return float(measure), n, exterior
