"""Claim (b) of UWTN 2026-022: a junction ligament closes at the coarser level.

The kissing Y of tests/test_0866 (a main strand and a splay ending one rung
from it), built as a split network on an add-on mesh of ``-uw_depth``
levels. Reports which levels of the tail carry the split (a level whose
point pairs exist) and every warning the replay raises. The claim is that
the replay stops at the first level where the one-cell ligament closes, so
that level and those beneath it stay welded.

    python probe_junction_replay.py -uw_depth 1
"""
import warnings

import numpy as np

import underworld3 as uw

params = uw.Params(uw_depth=1)

H = 0.02                       # the network's spacing
WIDTH = 0.04                   # two rungs across; the Y kisses at one rung

along = np.array([np.cos(np.deg2rad(40.0)), np.sin(np.deg2rad(40.0))])
main = 0.5 + np.linspace(-0.28, 0.28, 29)[:, None] * along
splay_dir = np.array([np.cos(np.deg2rad(62.0)), np.sin(np.deg2rad(62.0))])
splay = 0.5 + 0.03 * along + np.linspace(0.0, 0.16, 9)[:, None] * splay_dir
pieces = [("Main", main), ("Splay", splay)]

base = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=8 * H,
    regular=False, refinement=1, qdegree=2)
network = uw.meshing.FaultNetwork(pieces, hierarchy=["Main", "Splay"])
network.prepare(h=H, ligament=1.0, verbose=False)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    mesh = network.build(base=base, width=WIDTH, realisation="split",
                         max_levels=int(params.uw_depth))

tail = list(getattr(mesh, "_custom_mg_coarse_meshes", None) or [])
rows = []
for m in tail + [mesh]:
    role = ("background" if getattr(m, "_mg_static", False)
            else "finest" if m is mesh else "add-on")
    split = bool(getattr(m, "_fault_point_pairs", None))
    rows.append(f"{role} {m.dm.getHeightStratum(0)[1]} cells "
                f"{'split' if split else 'welded'}")
uw.pprint(f"PROBE depth {params.uw_depth}: " + "; ".join(rows))
for w in caught:
    text = str(w.message)
    if "replay" in text or "level" in text or "ligament" in text:
        uw.pprint(f"PROBE warning: {text[:300]}")
