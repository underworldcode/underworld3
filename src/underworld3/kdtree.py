# KDTree module — re-exports the active backend's KDTree.
#
# The active backend is ckdtree (set in __init__.py).  pykdtree was the
# original backend but causes OpenMP double-init crashes on macOS when
# loaded alongside other OpenMP-using libraries (PETSc, numpy, etc.).
# See commit 16cddf5 "kdtree swap out (temporary)".
#
# This module exists so that `import underworld3.kdtree` doesn't break
# and any code referencing `underworld3.kdtree.KDTree` gets the working
# implementation.

from underworld3.ckdtree import KDTree  # noqa: F401
