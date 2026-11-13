"""Every row of the 3-D nested P1 transfer is the coarse embedding (#449).

``nvb._nested_prolongation_local`` resolves a fine vertex the first round
cannot (one not at a coarse vertex or edge midpoint) as the average of two
resolved fine neighbours whose midpoint it is. In 3-D a vertex inside the
closure cascade usually has two or three such pairs, and the first one found
was sometimes not its parents: their rows lie in two different coarse cells,
and the row taken (often 1/2, 1/2 on two coarse vertices joined by no coarse
edge) is wrong by O(1) on a random field, while still reproducing every
linear field exactly. A pair is now accepted only if its two rows' supports
lie in one coarse cell, where the averaged row is the embedding whichever such
pair is taken. W1 measured 1, 4 and 9 wrong rows on passes 2 to 4 of this
fixture.

The fixture: a unit cube at cellSize 0.5 refined once, then native bisection
passes marking the cells whose centroid is within 0.2 of z = 0.5. Each pass is
checked against an independent reference: the coarse P1 value of a random
nodal field, evaluated barycentrically in the coarse tet that contains the fine
vertex (not ``uw.function.evaluate``, #432).
"""
import numpy as np
import pytest
from petsc4py import PETSc
from scipy.spatial import cKDTree

import underworld3 as uw
from underworld3.utilities import nvb
from underworld3.utilities._nvb_transform import refine

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b, pytest.mark.timeout(600)]

PASSES = 4


def _tets(dm):
    cS, cE = dm.getHeightStratum(0)
    vS, vE = dm.getDepthStratum(0)
    X = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, 3)
    T = np.array([[p - vS for p in dm.getTransitiveClosure(c)[0] if vS <= p < vE]
                  for c in range(cS, cE)], dtype=np.int64)
    return X, T


def _barycentric(points, V):
    A = np.transpose(V[:, 1:] - V[:, :1], (0, 2, 1))          # (n, 3, 3)
    lam = np.linalg.solve(A, (points - V[:, 0])[..., None])[..., 0]
    return np.column_stack([1.0 - lam.sum(axis=1), lam])


def _reference(Xc, Tc, Xf, field):
    """The coarse P1 value at each fine vertex, in the coarse tet holding it."""
    tree = cKDTree(Xc[Tc].mean(axis=1))
    k = min(24, len(Tc))
    _, cand = tree.query(Xf, k=k)
    best, lam_best, score = np.zeros(len(Xf), dtype=np.int64), None, np.full(len(Xf), -np.inf)
    lam_best = np.zeros((len(Xf), 4))
    for j in range(k):
        c = cand[:, j]
        lam = _barycentric(Xf, Xc[Tc[c]])
        m = lam.min(axis=1)
        better = m > score
        best[better], lam_best[better], score[better] = c[better], lam[better], m[better]
    assert score.min() > -1e-9, "a fine vertex outside every coarse tet"
    return np.einsum("ij,ij->i", lam_best, field[Tc[best]])


def _marked(dm):
    X, T = _tets(dm)
    zc = X[T].mean(axis=1)[:, 2]
    if dm.hasLabel("adapt"):
        dm.removeLabel("adapt")
    dm.createLabel("adapt")
    label = dm.getLabel("adapt")
    cS = dm.getHeightStratum(0)[0]
    for c in np.flatnonzero(np.abs(zc - 0.5) < 0.2):
        label.setValue(int(c) + cS, 1)
    return dm


@pytest.mark.skipif(uw.mpi.size > 1, reason="serial: the rows are compared on one rank")
def test_every_row_is_the_coarse_embedding():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0),
                                             cellSize=0.5, refinement=1)
    rng = np.random.default_rng(449)
    coarse = mesh.dm.clone()
    coarse.setCoordinatesLocal(mesh.dm.getCoordinatesLocal())
    nvb.write_tagged_state_label(coarse)            # the native refinement's seed
    wrong_by_pass = []
    for _ in range(PASSES):
        fine = refine(_marked(coarse))
        rows, cols, vals = nvb._nested_prolongation_local(coarse, fine)
        Xc, Tc = _tets(coarse)
        Xf, _ = _tets(fine)
        field = rng.standard_normal(len(Xc))
        transferred = np.zeros(len(Xf))
        np.add.at(transferred, rows, vals * field[cols])
        error = np.abs(transferred - _reference(Xc, Tc, Xf, field))
        wrong_by_pass.append(int((error > 1e-12).sum()))
        coarse = fine
    assert wrong_by_pass == [0] * PASSES, wrong_by_pass
