# KDTree module — re-exports the active backend's KDTree.
#
# The backend is ckdtree (nanoflann). The original pykdtree backend was
# removed from the dependency set entirely (2026-07, WA-26/LE-22): its
# OpenMP runtime caused fatal double-init crashes on macOS when loaded
# alongside other OpenMP-using libraries (PETSc, numpy, etc.). See
# commit 16cddf5 "kdtree swap out (temporary)" for the original swap.
#
# This module exists so that `import underworld3.kdtree` doesn't break
# and any code referencing `underworld3.kdtree.KDTree` gets the working
# implementation.

from underworld3.ckdtree import KDTree  # noqa: F401
