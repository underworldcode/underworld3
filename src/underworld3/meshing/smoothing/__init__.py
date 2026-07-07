"""Mesh smoothing and fixed-topology mesh adaptation ("movers").

Every operation here moves mesh VERTICES only: topology, vertex ids,
DOF layout, and the parallel partition are preserved (adding or
removing resolution needs a remesh — ``mesh.adapt`` — not a mover).
Public entry points: :func:`smooth_mesh_interior` (dispatch over
``method=``), :func:`follow_metric` (two-knob adapter),
:func:`metric_density_from_gradient` (metric builder) and
:func:`mesh_metric_mismatch` (alignment diagnostic).

The movers, one paragraph each:

* **Graph-Laplacian Jacobi** (``metric=None``, the no-metric default):
  each interior vertex is blended toward the mean of its edge
  neighbours over a few sweeps. Equalises connectivity → equant cells;
  use to clean up distortion left by a deformation (e.g. free-surface
  motion). Parallel-exact: the vertex-vertex adjacency is a PETSc AIJ
  Mat assembled with GLOBAL vertex indices, so partition-boundary rows
  are complete and results are bit-identical at any rank count.

* **Spring equilibrium** (``method="spring"``, the default metric
  path; :func:`_spring_equilibrium_mover`): every edge is an
  equal-rest-length spring (shape regulariser, equant cells) plus a
  per-cell target-area constraint ``A0 ∝ 1/ρ_tgt`` (the size grading),
  minimised to mechanical equilibrium by preconditioned nonlinear CG
  (Polak–Ribière⁺) with a fold-rejecting Armijo line search. Fast and
  robust — the workhorse for restoring a design grading. Limitation:
  edge forces are accumulated rank-locally, so this path is
  serial-exact (partition-boundary nodes under-count forces under
  MPI; a future PR can assemble forces cross-rank like the Jacobi
  adjacency Mat).

* **Monge–Ampère** (``method="ma"``; :func:`_monge_ampere_mover`):
  Benamou–Froese–Oberman convex-branch MA equidistribution with a
  variationally-recovered Hessian. Highest-fidelity *isotropic*
  refinement, ~60× costlier than the spring; preserved for reference,
  not a production choice — see the MA section banner below for the
  2026-05-16 investigation summary (all variants cap at the same
  fixed-topology grading).

* **OT improvement step** (``method="ot"``;
  :func:`_ot_improvement_step`): one linear weighted-Poisson
  equidistribution-flow step with respect to the *current* mesh,
  freely composable. Incomplete (boundary slip is gated to radial
  geometries) and deprecated — superseded by ``"mmpde"`` with a
  scalar metric.

* **Anisotropic Winslow** (``method="anisotropic"``;
  :func:`_winslow_anisotropic`): the one genuine Winslow smooth here —
  an M-weighted Laplace solve of the coordinate map with an
  eigen-clamped, gradient-derived metric *tensor*. Reshapes cells
  (short across a feature, long along it) and removes slivers; single
  primary knob ``resolution_ratio``. Improves alignment / quality,
  not grading magnitude.

* **MMPDE** (``method="mmpde"``; :func:`_mmpde_mover`): Huang–Kamenski
  variational moving mesh driven by a full tensor (or scalar) metric —
  non-folding by construction, and it genuinely clusters and aligns to
  the metric. **Recommended for production adaptive meshing**, though
  it is not the API default (``method`` defaults to ``"spring"``).
  Currently 2D (triangle meshes) only; parallel-safe.

With a fixed node count no mover exceeds ≈1.3–1.8× deep/near grading
(the exact optimal-transport ~10× needs *more nodes* — a topology
change, i.e. ``mesh.adapt``). See
``docs/developer/subsystems/mesh-metric-redistribution.md`` and
``docs/developer/design/anisotropic-mmpde-mover.md``.

Package layout (READ-04 split of the former 4,500-line module):

* ``graph.py``   — topology / masks / parallel reductions / shared
  mover primitives + :func:`smooth_surface_field`
* ``spring.py``  — the spring-equilibrium mover
* ``monge_ampere.py`` — MA machinery, the OT step, solver wiring
* ``anisotropic.py``  — the tensor (true Winslow) mover
* ``mmpde.py``   — the variational MMPDE mover
* ``metrics.py`` — strategies, metric builder, alignment diagnostic
* ``api.py``     — :func:`smooth_mesh_interior` dispatch and
  :func:`follow_metric`

Every name that was module-level in the old ``smoothing.py`` is
re-exported here, so ``from underworld3.meshing.smoothing import X``
and ``smoothing.X`` keep working for the private cross-module surface
(``_edge_pairs``, ``_tri_cells``, ``_pinned_mask``, ...).
"""

import warnings

from .graph import (
    _auto_pinned_labels,
    _owned_vertex_mask,
    _pinned_mask,
    _build_scalar_dm,
    _build_adjacency_matrix,
    _build_local_to_owned_map,
    _min_incident_edge,
    _min_incident_edge_nd,
    _owned_cell_mask,
    _tri_cells,
    _tet_cells,
    _signed_areas,
    _edge_pairs,
    _mean_edge_length,
    _cap_step_to_edge_fraction,
    _backtracked_move,
    _reweight_displacement_radial_tangential,
    _global_sum,
    _global_min,
    _global_max,
    _global_mean,
    smooth_surface_field,
)
from .metrics import (
    ADAPT_STRATEGIES,
    _UNSET,
    mesh_metric_mismatch,
    metric_density_from_gradient,
)
from .spring import _spring_equilibrium_mover
from .monge_ampere import (
    _use_direct_solver,
    _use_iterative_solver,
    _solver_wiring,
    _warm_start_krylov,
    _patch_volumes,
    _lumped_vertex_volumes,
    _hessian_recovery_class,
    _monge_ampere_mover,
    _ot_improvement_step,
)
from .anisotropic import _winslow_anisotropic
from .mmpde import _mmpde_mover
from .api import (
    smooth_mesh_interior,
    _smooth_mesh_interior_bare,
    follow_metric,
)


# ---------------------------------------------------------------------------
# One-cycle deprecated aliases (renamed 2026-07, READ-06): the ``_winslow_``
# prefix was a misnomer on four of the five prefixed movers — only
# ``_winslow_anisotropic`` actually solves a Winslow (M-weighted Laplace)
# coordinate map. The old names are kept for one release cycle because they
# appear in exploratory scripts (scripts/) and external user scripts.
# ---------------------------------------------------------------------------
def _deprecated_mover_alias(new_func, old_name):
    """Wrap ``new_func`` so calls through ``old_name`` still work but emit
    a DeprecationWarning naming the replacement."""
    import functools

    @functools.wraps(new_func)
    def _alias(*args, **kwargs):
        warnings.warn(
            f"{old_name} was renamed to {new_func.__name__} (READ-06: the "
            f"mover is not a Winslow smooth); the old name is a one-cycle "
            f"deprecated alias.",
            DeprecationWarning, stacklevel=2)
        return new_func(*args, **kwargs)

    return _alias


_winslow_spring = _deprecated_mover_alias(
    _spring_equilibrium_mover, "_winslow_spring")
_winslow_elliptic = _deprecated_mover_alias(
    _monge_ampere_mover, "_winslow_elliptic")
_winslow_equidistribute = _deprecated_mover_alias(
    _ot_improvement_step, "_winslow_equidistribute")
_winslow_mmpde = _deprecated_mover_alias(
    _mmpde_mover, "_winslow_mmpde")
