"""Mesh smoothing and fixed-topology mesh adaptation ("movers").

Every operation here moves mesh VERTICES only: topology, vertex ids,
DOF layout, and the parallel partition are preserved (adding or
removing resolution needs a remesh — ``mesh.adapt`` — not a mover).
Public entry points: :func:`smooth_mesh_interior` (dispatch over
``method=``), :func:`follow_metric` (two-knob adapter),
:func:`metric_density_from_gradient` (metric builder) and
:func:`mesh_metric_mismatch` (alignment diagnostic).

The movers:

* **Graph-Laplacian Jacobi** (``metric=None``, the no-metric default):
  each interior vertex is blended toward the mean of its edge
  neighbours over a few sweeps. Equalises connectivity → equant cells;
  use to clean up distortion left by a deformation (e.g. free-surface
  motion). Parallel-exact: the vertex-vertex adjacency is a PETSc AIJ
  Mat assembled with GLOBAL vertex indices, so partition-boundary rows
  are complete and results are bit-identical at any rank count.

* **MMPDE** (``method="mmpde"``, the metric-path default;
  :func:`_mmpde_mover`): Huang–Kamenski variational moving mesh driven
  by a full tensor (or scalar) metric — non-folding by construction,
  and it genuinely clusters and aligns to the metric. The production
  mover for metric-driven node redistribution. Currently 2D (triangle
  meshes) only; parallel-safe.

The **Taubin surface smoother** (:func:`smooth_surface_field`, in
``graph.py``) is a separate, current tool: it smooths a FIELD on a
boundary/submesh, not the mesh coordinates.

Retired movers (2026-07 maintainer ruling): the **spring-equilibrium**,
**Monge–Ampère**, **OT-improvement-step** and **anisotropic-Winslow**
interior movers were superseded by ``method="mmpde"`` — with a scalar
(isotropic) metric the MMPDE mover reaches the same equidistributed
grading (the isotropic-metric equivalence), and with a tensor metric it
does what the anisotropic mover could not (genuine clustering and
alignment). Their ``method=`` spellings now raise a ``ValueError``
naming the retirement. Git history and
``docs/developer/design/anisotropic-mmpde-mover.md`` /
``ma-newton-cofactor-exploration.md`` record the retired algorithms.

With a fixed node count no mover exceeds ≈1.3–1.8× deep/near grading
(the exact optimal-transport ~10× needs *more nodes* — a topology
change, i.e. ``mesh.adapt``). See
``docs/developer/subsystems/mesh-metric-redistribution.md`` and
``docs/developer/design/anisotropic-mmpde-mover.md``.

Package layout (READ-04 split of the former 4,500-line module, pruned
2026-07):

* ``graph.py``   — topology / masks / parallel reductions / boundary
  facet + slip primitives + :func:`smooth_surface_field`
* ``mmpde.py``   — the variational MMPDE mover
* ``metrics.py`` — strategies, metric builder, alignment diagnostic
* ``api.py``     — :func:`smooth_mesh_interior` dispatch and
  :func:`follow_metric`

Every name that is module-level in the submodules is re-exported here,
so ``from underworld3.meshing.smoothing import X`` and ``smoothing.X``
keep working for the private cross-module surface (``_edge_pairs``,
``_tri_cells``, ``_pinned_mask``, ...).
"""

import warnings

from .graph import (
    _auto_pinned_labels,
    _owned_vertex_mask,
    _pinned_mask,
    _build_scalar_dm,
    _build_adjacency_matrix,
    _build_local_to_owned_map,
    _min_incident_edge_nd,
    _owned_cell_mask,
    _tri_cells,
    _tet_cells,
    _signed_areas,
    _edge_pairs,
    _mean_edge_length,
    _global_sum,
    _global_min,
    _global_max,
    _global_mean,
    _slip_normals,
    _boundary_facets,
    _all_boundary_labels,
    _resolve_slip,
    _nearest_on_facets_2d,
    _nearest_on_facets_3d,
    smooth_surface_field,
)
from .metrics import (
    ADAPT_STRATEGIES,
    _UNSET,
    mesh_metric_mismatch,
    metric_density_from_gradient,
)
from .mmpde import _mmpde_mover
from .api import (
    smooth_mesh_interior,
    _smooth_mesh_interior_bare,
    follow_metric,
    _RETIRED_MOVER_MESSAGE,
)


# ---------------------------------------------------------------------------
# Retired-mover tombstones (2026-07 retirement ruling). The spring / MA /
# OT / anisotropic-Winslow movers were deleted outright (superseded by
# ``method="mmpde"``); these callables catch scripts that imported the old
# private names — including the READ-06 ``_winslow_*`` aliases, which were
# introduced less than one release cycle before the retirement — and raise
# the same retirement message as the ``method=`` dispatch.
# ---------------------------------------------------------------------------
def _retired_mover_stub(old_name):
    """A callable that raises the retirement ValueError for ``old_name``."""

    def _stub(*args, **kwargs):
        raise ValueError(
            f"{old_name} was retired (2026-07): {_RETIRED_MOVER_MESSAGE}")

    _stub.__name__ = old_name
    _stub.__doc__ = (f"RETIRED (2026-07). {old_name} was superseded by the "
                     f"MMPDE mover; calling it raises ValueError.")
    return _stub


_spring_equilibrium_mover = _retired_mover_stub("_spring_equilibrium_mover")
_monge_ampere_mover = _retired_mover_stub("_monge_ampere_mover")
_ot_improvement_step = _retired_mover_stub("_ot_improvement_step")
_winslow_anisotropic = _retired_mover_stub("_winslow_anisotropic")
_winslow_spring = _retired_mover_stub("_winslow_spring")
_winslow_elliptic = _retired_mover_stub("_winslow_elliptic")
_winslow_equidistribute = _retired_mover_stub("_winslow_equidistribute")


def _winslow_mmpde(*args, **kwargs):
    """One-cycle deprecated alias for :func:`_mmpde_mover` (READ-06
    rename, 2026-07): identical behaviour plus one DeprecationWarning."""
    warnings.warn(
        "_winslow_mmpde was renamed to _mmpde_mover (READ-06: the mover "
        "is not a Winslow smooth); the old name is a one-cycle deprecated "
        "alias.",
        DeprecationWarning, stacklevel=2)
    return _mmpde_mover(*args, **kwargs)
