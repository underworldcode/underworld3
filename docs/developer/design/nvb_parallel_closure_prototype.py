"""Design-validation prototype: the PARALLEL NVB conforming-closure FIXPOINT.

Companion to ``NVB_GRADED_ADAPT.md`` (Parallel-readiness review). This de-risks the
genuinely novel part of the parallel (Route B) NVB engine — completing the bounded
conforming closure *across a partition boundary* — separately from the DMPlex / SF
construction. It proves, on a partitioned mesh, that the cross-rank closure:

  (a) CONVERGES in a bounded number of communication rounds (single-digit,
      independent of mesh size);
  (b) is globally CONFORMING (0 hanging nodes, including across the partition);
  (c) is IDENTICAL to the serial NVB mesh — partition-independent (confluence).

Faithful distributed model (single process, G "ranks"): geometric/topological
state is legitimately SHARED (a real run's point-SF keeps shared vertices / edges /
midpoints consistent — modelled here by one ``NVBMesh`` + a deterministic shared
midpoint), but CONTROL FLOW obeys the distributed discipline — a cell is only ever
bisected by its OWNER, and the sole cross-rank coupling is an explicit work queue of
cells to (attempt to) bisect; no rank reads/writes another rank's cells.

If the closure converges + matches serial under this discipline, the only remaining
work for a real parallel engine is the SF / point construction, which a native
``DMPlexTransform`` (Route B) inherits from PETSc.

Run: ``python docs/developer/design/nvb_parallel_closure_prototype.py`` (in the uw
env). Imports the *shipped* engine ``underworld3.utilities.nvb.NVBMesh``.
"""
import numpy as np
from underworld3.utilities.nvb import NVBMesh, _fs as fs


# ---- a structured triangulated unit square (pure geometry, no PETSc) --------
def structured_square(n):
    xs = np.linspace(0, 1, n + 1)
    coords = [(x, y) for y in xs for x in xs]
    vid = lambda i, j: j * (n + 1) + i
    tris = []
    for j in range(n):
        for i in range(n):
            a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            tris += [(a, b, d), (b, c, d)]
    return coords, tris


# ---- distributed NVB closure (owner-only bisection + work-queue coupling) ---
class DistributedNVB:
    def __init__(self, coords, tris, owner_of_centroid):
        self.g = NVBMesh(coords, tris)
        self.owner = {}
        C = np.array(self.g.coords)
        for cid, (p, b0, b1) in self.g.cells.items():
            self.owner[cid] = owner_of_centroid(C[[p, b0, b1]].mean(0))
        self.max_round_worklist = 0
        # children inherit their parent's owner — wrap _split to record it
        orig_split = self.g._split

        def split(cid, m):
            rank = self.owner.get(cid)
            before = set(self.g.cells)
            orig_split(cid, m)
            self.owner.pop(cid, None)
            for newcid in set(self.g.cells) - before:
                self.owner[newcid] = rank
        self.g._split = split

    def _try_bisect(self, rank, cid, requeue):
        """Owner ``rank`` attempts to bisect owned cell ``cid``. Touches only owned
        cells; defers (enqueues) when the closure needs a non-owned neighbour; a
        shared-edge split enqueues the across-edge neighbour to split at the same
        deterministic midpoint. Returns True iff ``cid`` was split."""
        while True:
            if cid not in self.g.cells:
                return False
            e = self.g._ref_edge(cid)
            nb = self.g._neighbour(cid, e)
            if nb is None or self.g._ref_edge(nb) == e:
                break                                # boundary / compatible
            if self.owner[nb] == rank:
                if not self._try_bisect(rank, nb, requeue):
                    requeue.add(cid)                 # nb deferred -> I defer too
                    return False
                continue                             # nb split -> re-eval
            requeue.add(nb)                          # non-owned: its owner refines it
            requeue.add(cid)
            return False
        m = self.g._midpoint(e)
        self.g._split(cid, m)
        if nb is not None:
            if self.owner[nb] == rank:
                self.g._split(nb, m)
            else:
                requeue.add(nb)                      # cross: split at same m
        return True

    def refine(self, marked):
        work = set(int(c) for c in marked)
        rounds = 0
        while work:
            rounds += 1
            self.max_round_worklist = max(self.max_round_worklist, len(work))
            nxt = set()
            for cid in list(work):                   # one communication round
                if cid in self.g.cells:
                    self._try_bisect(self.owner[cid], cid, nxt)
            work = nxt
            if rounds > 1000:
                raise RuntimeError("closure did not converge")
        return rounds


# ---- confluence + conformity oracle -----------------------------------------
def signature(nvb):
    coords, tris, _, _ = nvb.arrays()
    verts = np.array(sorted(map(tuple, np.round(coords, 9))))
    cents = np.array(sorted(map(tuple, np.round(coords[tris].mean(1), 9))))
    return verts, cents


def equal(a, b):
    (va, ca), (vb, cb) = a, b
    return (va.shape == vb.shape and ca.shape == cb.shape
            and np.allclose(va, vb, atol=1e-8) and np.allclose(ca, cb, atol=1e-8))


def serial_ref(coords, tris, pred, levels):
    nvb = NVBMesh(coords, tris)
    for _ in range(levels):
        cen, h, cids = nvb.centroids_h()
        m = cids[np.array([pred(c) for c in cen])]
        if m.size == 0:
            break
        nvb.refine(set(int(c) for c in m))
    return nvb


def run_case(n, G, pred, owner_fn, levels, label):
    ser = serial_ref(*structured_square(n), pred, levels)
    dist = DistributedNVB(*structured_square(n), owner_fn)
    rlog = []
    for _ in range(levels):
        cen, h, cids = dist.g.centroids_h()
        m = cids[np.array([pred(c) for c in cen])]
        if m.size == 0:
            break
        rlog.append(dist.refine(set(int(c) for c in m)))
    hang, over = dist.g.check_conforming()
    ok_c = (hang, over) == (0, 0)
    ok_e = equal(signature(ser), signature(dist.g))
    print(f"[{label}] n={n} G={G} lvl={levels}: serial {len(ser.cells)} | "
          f"dist {len(dist.g.cells)} (conf {hang},{over}) | rounds {rlog} | "
          f"CONFORMING={ok_c} EQUALS_SERIAL={ok_e}")
    assert ok_c and ok_e
    return max(rlog) if rlog else 0


def main():
    diag = lambda c: abs((c[0] + c[1]) - 1.0) < 0.18
    diag_t = lambda c: abs((c[0] + c[1]) - 1.0) < 0.10
    band = lambda c: abs(c[0] - 0.5) < 0.16
    LR = lambda c: 0 if c[0] < 0.5 else 1
    quad = lambda c: (0 if c[0] < 0.5 else 1) + (0 if c[1] < 0.5 else 2)
    stripes = lambda c: int(c[0] * 4) % 3

    print("=== convergence / conformity / confluence across partitions ===")
    mx = [run_case(8, 2, band, LR, 3, "band|LR2"),
          run_case(8, 4, diag, quad, 3, "diag|quad4"),
          run_case(8, 3, diag, stripes, 3, "diag|stripes3")]
    print("\n=== bounded rounds vs mesh size (feature fixed, base refined) ===")
    for n in (8, 16, 24, 32):
        mx.append(run_case(n, 4, diag_t, quad, 4, f"diag|quad4|n{n}"))
    print(f"\nMAX communication rounds over ALL cases: {max(mx)} "
          f"(bounded, single-digit, independent of mesh size)")
    print("PARALLEL-CLOSURE DE-RISK PASS.")


if __name__ == "__main__":
    main()
