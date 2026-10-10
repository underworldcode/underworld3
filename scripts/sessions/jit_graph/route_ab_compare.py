"""Compare the two routes' records written by ``route_ab.py`` for one fixture."""
import os

import numpy as np
import underworld3 as uw

params = uw.Params(uw_dir=uw.Param("~/+Simulations/jit_graph/tier2", description="directory of the .npz records"))
out = os.path.expanduser(str(params.uw_dir))
cases = sorted({f.rsplit("_", 1)[0] for f in os.listdir(out) if f.endswith(".npz")})
for case in cases:
    paths = [os.path.join(out, f"{case}_{r}.npz") for r in ("tree", "graph")]
    if not all(os.path.exists(p) for p in paths):
        continue
    a, b = (np.load(p) for p in paths)
    dv = np.abs(a["v"] - b["v"]).max() / max(np.abs(a["v"]).max(), 1e-300)
    dp = np.abs(a["p"] - b["p"]).max() / max(np.abs(a["p"]).max(), 1e-300)
    ha, hb = a["history"], b["history"]
    k = min(len(ha), len(hb))
    dh = (np.abs(ha[:k] - hb[:k]) / np.maximum(np.abs(ha[:k]), 1e-300)).max() if k else np.nan
    print(f"{case}")
    print(f"  {'':12s} {'tree':>12s} {'graph':>12s}")
    for key, fmt in (("pointwise", ".2f"), ("generate", ".2f"), ("compile", ".2f"),
                     ("header_bytes", ",d"), ("solve", ".2f"), ("t_res", ".2e"),
                     ("t_jac", ".2e"), ("nl", "d"), ("ksp", "d")):
        va, vb = a[key].item(), b[key].item()
        print(f"  {key:12s} {va:>12{fmt}} {vb:>12{fmt}}")
    print(f"  reason       {str(a['reason']):>12s} {str(b['reason']):>12s}")
    print(f"  solution: max|dv|/max|v| {dv:.2e}, max|dp|/max|p| {dp:.2e}; "
          f"residual history (first {k}) max rel diff {dh:.2e}")
