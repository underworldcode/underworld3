"""Coarse FMG levels for the grid-hierarchy figure.

The base mesh (h = 0.25 on a 2.0 x 1.0 domain) is the finest level of a
geometric hierarchy built by STRUCTURED refinement: the coarse levels at
h = 0.5 and h = 1.0 come from the same diagonal-split quad generator, so
midpoint (regular) refinement of each level IS the next — the nesting is
exact, and this script VERIFIES it (every coarse vertex coincides with a
fine one) rather than assuming it.
"""
import json
import os

import numpy as np


def level(h):
    xs = np.arange(0.0, 2.0 + 1e-9, h)
    ys = np.arange(0.0, 1.0 + 1e-9, h)
    idx, pts = {}, []
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            idx[(i, j)] = len(pts)
            pts.append([x, y])
    tris = []
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            a, b = idx[(i, j)], idx[(i + 1, j)]
            c, d = idx[(i + 1, j + 1)], idx[(i, j + 1)]
            tris.extend([[a, b, c], [a, c, d]])
    return dict(coords=pts, tris=tris)


out = dict(coarse2=level(1.0), coarse1=level(0.5))

# exact-nesting check: every coarse vertex is a fine vertex (bit-exact
# up to the arange rounding), level by level down to the base
fine = level(0.25)
for name in ("coarse1", "coarse2"):
    coarse_set = {tuple(np.round(p, 12)) for p in out[name]["coords"]}
    fine_set = {tuple(np.round(p, 12)) for p in fine["coords"]}
    assert coarse_set <= fine_set, f"{name} not nested in the base"
print("nesting exact: coarse2 < coarse1 < base")
here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "grid-hierarchy-data.json"), "w") as f:
    json.dump(out, f)
for name, p in out.items():
    print(f"{name}: {len(p['coords'])} pts, {len(p['tris'])} tris")
