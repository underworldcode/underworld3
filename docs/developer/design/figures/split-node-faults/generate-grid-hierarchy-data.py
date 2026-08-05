"""Coarse FMG levels for the grid-hierarchy figure.

The base mesh of the stack-progression figure is itself the finest level
of a geometric (FMG) hierarchy; this generator adds the two coarser
levels below it (~2x spacing per level, endpoints exact). The refined
children and the cut come from stack-progression-data.json.
"""
import json
import os

import numpy as np
from scipy.spatial import Delaunay


def level(nx, ny):
    pts = np.array([[x, y] for x in np.linspace(0.0, 2.0, nx)
                    for y in np.linspace(0.0, 1.2, ny)])
    return dict(coords=pts.tolist(), tris=Delaunay(pts).simplices.tolist())


out = dict(coarse2=level(3, 2), coarse1=level(4, 3))
here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "grid-hierarchy-data.json"), "w") as f:
    json.dump(out, f)
for name, p in out.items():
    print(f"{name}: {len(p['coords'])} pts, {len(p['tris'])} tris")
