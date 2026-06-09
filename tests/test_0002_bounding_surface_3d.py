"""3D radial bounding-surface registration (SphericalShell).

This lives in the early (test_000x) batch on purpose. SphericalShell
construction is fragile once a long-running process has accumulated a lot of
PETSc/mesh state (a pre-existing mesh-lifecycle issue — the coordinate DM /
cdim can go stale, giving a "cannot reshape ... into shape (3)" at build time).
Built early (right after test_0001_meshes, which itself constructs spheres
cleanly) it is robust. The 2D radial-registration logic is covered by the
Annulus tests in test_0762_bounding_surfaces.py.

See docs/developer/design/boundary-slip-strategy.md.
"""
import numpy as np

import underworld3 as uw


def test_spherical_shell_registers_radial():
    m = uw.meshing.SphericalShell(radiusInner=0.5, radiusOuter=1.0, cellSize=0.4)
    bs = m.bounding_surfaces
    assert bs["Upper"].kind == "radial" and np.isclose(bs["Upper"].radius, 1.0)
    assert bs["Lower"].kind == "radial" and np.isclose(bs["Lower"].radius, 0.5)
    assert bs["Upper"].centre.shape == (3,)
    out = bs["Upper"].restore(np.array([[1.3, 0.0, 0.0], [0.0, 0.7, 0.7]]))
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)
