"""Compare the residuals and Jacobians ``route_assemble.py`` saved for the two routes."""
import os
import sys

import numpy as np

out = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else
                         "~/+Simulations/jit_graph/tier2/assemble")
cases = sorted({f.rsplit("_", 1)[0] for f in os.listdir(out) if f.endswith(".npz")})
for case in cases:
    paths = [os.path.join(out, f"{case}_{r}.npz") for r in ("tree", "graph")]
    if not all(os.path.exists(p) for p in paths):
        continue
    a, b = (np.load(p) for p in paths)
    fa, fb = a["F"], b["F"]
    nan = (int((~np.isfinite(fa)).sum()), int((~np.isfinite(fb)).sum()))
    df = np.abs(fa - fb).max() / max(np.abs(fa).max(), 1e-300)
    same_pattern = np.array_equal(a["ai"], b["ai"]) and np.array_equal(a["aj"], b["aj"])
    ja, jb = a["av"], b["av"]
    jnan = (int((~np.isfinite(ja)).sum()), int((~np.isfinite(jb)).sum()))
    if same_pattern:
        dj = np.abs(ja - jb)
        rel = dj.max() / max(np.abs(ja).max(), 1e-300)
        # entry by entry, against the largest entry of its row
        rows = np.repeat(np.arange(len(a["ai"]) - 1), np.diff(a["ai"]))
        rowmax = np.zeros(len(a["ai"]) - 1)
        np.maximum.at(rowmax, rows, np.abs(ja))
        rowrel = (dj / np.maximum(rowmax[rows], 1e-300)).max()
        bit = (ja == jb).mean()
        jtxt = (f"J: max|dJ|/max|J| {rel:.2e}, worst entry against its row {rowrel:.2e}, "
                f"bit-identical {100 * bit:.1f}%")
    else:
        jtxt = "J: SPARSITY PATTERNS DIFFER"
    print(f"{case}: F max|dF|/max|F| {df:.2e}, bit-identical {100 * (fa == fb).mean():.1f}%; "
          f"{jtxt}; non-finite F {nan}, J {jnan}")
