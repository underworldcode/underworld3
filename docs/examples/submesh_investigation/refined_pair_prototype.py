"""Prototype: coarse companion mesh from a refinement hierarchy.

A *coarsened companion* is a real ``uw.Mesh`` that wraps a coarser DM from
the fine mesh's refinement hierarchy. It carries the same submesh lineage
state as ``Mesh.extract_region`` (``parent``, registered with the parent's
``_registered_submeshes``) so the downstream pattern is identical:

    get a mesh -> create a solver -> map fields back and forth.

Transfer between the fine parent and the coarse companion uses the
PETSc-native operators that geometric multigrid relies on
(``createInterpolation`` / ``createInjection`` on the variables'
single-field sub-DMs) -- no KDTree.

This is investigation code under docs/examples/, not a merged API.
"""

from enum import Enum

import numpy as np
from petsc4py import PETSc

import underworld3 as uw

# DESIGN CONTRACT: refine-DM mode only.
#
# A level may be pulled out of a mesh ONLY when a genuine nested
# refinement hierarchy exists (the mesh was built with refinement >= 1).
# Transfer between levels uses PETSc's *nested* interpolator/injector,
# which is exact, parallel-local, and needs no geometric point location.
#
# There is deliberately NO geometric/General fallback and NO KDTree. If
# the nested path is not in effect the operators must fail loudly rather
# than silently degrade -- so we do NOT enable -dm_plex_hash_location.
# (If the General branch were ever taken it would raise "Nearest point
# location only supported with grid hashing", which is the desired
# loud failure.)


# ---------------------------------------------------------------------------
# Building the companion
# ---------------------------------------------------------------------------

def _require_refinement_hierarchy(mesh):
    """Enforce the refine-DM-only contract.

    A level can be pulled out ONLY when the mesh carries a genuine
    nested refinement hierarchy (built with ``refinement >= 1``). If
    there is no refinement relationship the operation is unavailable --
    we raise rather than offer a geometric/KDTree approximation.
    """
    hier = getattr(mesh, "dm_hierarchy", None)
    if hier is None or len(hier) < 2:
        raise ValueError(
            "Hierarchy-level extraction requires a genuine nested "
            "refinement hierarchy: build the mesh with refinement >= 1. "
            f"This mesh has {0 if hier is None else len(hier)} level(s) "
            "(no refinement relationship), so a coarse/fine companion is "
            "not available. There is deliberately no geometric or KDTree "
            "fallback."
        )
    return hier


def coarsened_companion(fine_mesh, levels=1, verbose=False):
    """Pull a coarser level out of ``fine_mesh``'s nested hierarchy.

    Analogous to ``Mesh.extract_region`` (which pulls out a subdomain):
    here we pull out a *level* of the refinement hierarchy as a
    standalone, solver-ready ``uw.Mesh``. Only available when a genuine
    refinement relationship exists (see the design contract above).

    Parameters
    ----------
    fine_mesh : uw.discretisation.Mesh
        A mesh built with ``refinement >= levels`` so ``dm_hierarchy``
        has enough levels (index 0 = coarsest, -1 = finest).
    levels : int
        How many refinement steps below the finest to take.

    Returns
    -------
    uw.discretisation.Mesh
        A solver-ready mesh whose ``.parent`` is ``fine_mesh``,
        registered with ``fine_mesh._registered_submeshes``.
    """
    hier = _require_refinement_hierarchy(fine_mesh)
    if len(hier) <= levels:
        raise ValueError(
            f"dm_hierarchy has {len(hier)} levels; need > {levels}. "
            f"Build the fine mesh with refinement >= {levels}."
        )

    coarse_dm = hier[-1 - levels]

    # Mirror extract_region: build a boundaries enum from labels that are
    # actually present (non-empty stratum) on the coarse DM. The probe
    # showed gmsh boundary labels survive every hierarchy level.
    surviving = {}
    if fine_mesh.boundaries is not None:
        for b in fine_mesh.boundaries:
            if b.name in ("Null_Boundary", "All_Boundaries"):
                continue
            lab = coarse_dm.getLabel(b.name)
            if lab:
                sis = lab.getStratumIS(b.value)
                if sis and sis.getSize() > 0:
                    surviving[b.name] = b.value
    sub_boundaries = Enum("Boundaries", surviving) if surviving else None

    # Construct the companion. refinement=None -> the constructor's
    # no-refine branch: self.dm IS coarse_dm, dm_hierarchy=[coarse_dm].
    companion = uw.discretisation.Mesh(
        coarse_dm,
        degree=fine_mesh.degree,
        qdegree=fine_mesh.qdegree,
        boundaries=sub_boundaries,
        coordinate_system_type=fine_mesh.CoordinateSystemType,
        verbose=verbose,
    )

    # Submesh lineage -- same shape as extract_region
    companion.parent = fine_mesh
    companion._parent_mesh_version = fine_mesh._mesh_version
    companion.regions = fine_mesh.regions
    companion._dof_maps = {}
    companion._interp_cache = {}          # (id(cv), id(fv)) -> (Mat, Vec)
    companion._inject_cache = {}          # (id(cv), id(fv)) -> Mat (MATSCATTER)
    companion._companion_levels = levels
    companion._is_coarsened_companion = True
    fine_mesh._registered_submeshes.add(companion)

    return companion


# ---------------------------------------------------------------------------
# Per-variable transfer operators (PETSc-native, no KDTree)
# ---------------------------------------------------------------------------

_FE_TAG = 0


def _single_field_dm(mesh, var):
    """Clone ``mesh.dm`` and install only ``var``'s FE as field 0.

    ``createSubDM(field_id)`` yields a section-only DM with no PetscFE, so
    ``createInterpolation``/``createInjection`` (FEM operators) fail with
    PETSC_ERR_SUP on it. Mirror the ``_get_coords_for_basis`` pattern
    (discretisation_mesh.py:2749-2777): clone the plex, attach the single
    PetscFE, build the DS. The clone keeps topology + coordinates so the
    FEM interpolator/injector can operate.
    """
    global _FE_TAG
    _FE_TAG += 1
    pfx = f"rpt{_FE_TAG}_"
    opts = PETSc.Options()
    opts.setValue(f"{pfx}petscspace_degree", var.degree)
    opts.setValue(f"{pfx}petscdualspace_lagrange_continuity", var.continuous)
    opts.setValue(f"{pfx}petscdualspace_lagrange_node_endpoints", False)

    fe = PETSc.FE().createDefault(
        mesh.dm.getDimension(),
        var.num_components,
        mesh.isSimplex,
        mesh.qdegree,
        pfx,
        PETSc.COMM_SELF,
    )
    dm = mesh.dm.clone()
    dm.clearDS()
    dm.setField(0, fe)
    dm.createDS()
    return dm


def _var_global_vec(var, sfdm):
    """Variable data -> single-field-DM global Vec.

    ``sfdm`` is the single-field clone (same FE, same topology as the
    variable's own field), so its local layout matches ``var.vec``.
    """
    var._set_vec(available=True)
    g = sfdm.createGlobalVector()
    loc = sfdm.getLocalVec()
    loc.array[...] = var.vec.array
    sfdm.localToGlobal(loc, g, addv=False)
    sfdm.restoreLocalVec(loc)
    return g


def _write_global_vec_to_var(var, sfdm, g):
    """Single-field-DM global Vec -> variable data."""
    loc = sfdm.getLocalVec()
    sfdm.globalToLocal(g, loc, addv=False)
    var._set_vec(available=True)
    var.vec.array[...] = loc.array
    var._lvec.array[...] = var.vec.array
    sfdm.restoreLocalVec(loc)
    if hasattr(var, "_canonical_data"):
        var._canonical_data = None


def _linked_pair(companion, coarse_var, fine_var):
    """Build a refinement-linked single-field (dm_c, dm_f) pair.

    dm_c is a single-field clone of the coarse companion DM. dm_f is
    produced by refining dm_c ``levels`` times -- so refine() sets the
    coarse/fine linkage and the regularRefinement flag, and
    createInterpolation/createInjection take the *nested* exact path
    (plex.c:10328) with no geometric point location.

    Because dm_f is the uniform refinement of the same coarse topology
    that produced the fine mesh, its DOF ordering matches the fine
    variable's storage (verified by the round-trip test).
    """
    levels = companion._companion_levels
    dm_c = _single_field_dm(companion, coarse_var)
    dm_f = dm_c
    for _ in range(levels):
        nxt = dm_f.refine()
        nxt.setCoarseDM(dm_f)
        dm_f = nxt
    # refine() carries topology; (re)build the DS so the field/section
    # exist on the refined DM for the FEM operators.
    dm_f.createDS()
    return dm_c, dm_f


def _get_interpolation(companion, coarse_var, fine_var):
    """Build & cache the nested FE prolongation coarse -> fine."""
    key = (id(coarse_var), id(fine_var))
    if key in companion._interp_cache:
        return companion._interp_cache[key]

    dm_c, dm_f = _linked_pair(companion, coarse_var, fine_var)
    matInterp, vecScale = dm_c.createInterpolation(dm_f)
    companion._interp_cache[key] = (matInterp, vecScale, dm_c, dm_f)
    return companion._interp_cache[key]


def _get_injection(companion, coarse_var, fine_var):
    """Build & cache the injection scatter (MATSCATTER) coarse <-> fine."""
    key = (id(coarse_var), id(fine_var))
    if key in companion._inject_cache:
        return companion._inject_cache[key]

    dm_c, dm_f = _linked_pair(companion, coarse_var, fine_var)
    injectMat = dm_c.createInjection(dm_f)  # MATSCATTER wrapping a VecScatter
    companion._inject_cache[key] = (injectMat, dm_c, dm_f)
    return companion._inject_cache[key]


def prolongate(companion, coarse_var, fine_var):
    """coarse -> fine, FE prolongation (fills all fine DOFs)."""
    matInterp, _, dm_c, dm_f = _get_interpolation(companion, coarse_var, fine_var)
    gc = _var_global_vec(coarse_var, dm_c)
    gf = dm_f.createGlobalVector()
    matInterp.mult(gc, gf)
    _write_global_vec_to_var(fine_var, dm_f, gf)


def restrict(companion, fine_var, coarse_var, weighted=True):
    """fine -> coarse, Galerkin restriction (transpose of prolongation)."""
    matInterp, vecScale, dm_c, dm_f = _get_interpolation(
        companion, coarse_var, fine_var
    )
    gf = _var_global_vec(fine_var, dm_f)
    gc = dm_c.createGlobalVector()
    matInterp.multTranspose(gf, gc)
    if weighted and vecScale is not None:
        gc.pointwiseMult(gc, vecScale)
    _write_global_vec_to_var(coarse_var, dm_c, gc)


def sample(companion, fine_var, coarse_var):
    """fine -> coarse, pure injection (exact at coarse-coincident DOFs)."""
    injectMat, dm_c, dm_f = _get_injection(companion, coarse_var, fine_var)
    gf = _var_global_vec(fine_var, dm_f)
    gc = dm_c.createGlobalVector()
    # MATSCATTER from createInjection maps fine -> coarse via multTranspose;
    # mult maps coarse -> fine. Verified empirically in the A/B test.
    injectMat.multTranspose(gf, gc)
    _write_global_vec_to_var(coarse_var, dm_c, gc)


def inject(companion, coarse_var, fine_var):
    """coarse -> fine, scatter onto coarse-coincident fine DOFs only.

    Fine DOFs introduced by refinement are left untouched.
    """
    injectMat, dm_c, dm_f = _get_injection(companion, coarse_var, fine_var)
    gc = _var_global_vec(coarse_var, dm_c)
    gf = dm_f.createGlobalVector()
    gf.set(0.0)
    injectMat.mult(gc, gf)
    _write_global_vec_to_var(fine_var, dm_f, gf)
