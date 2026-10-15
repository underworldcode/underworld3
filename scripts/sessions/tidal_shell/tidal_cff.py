"""Coulomb failure stress on receiver planes parallel to the tiger stripes, in the
south polar view, for a run against a reference run at the same step:

    CFF = |tau_s| + mu sigma_n        (sigma_n positive in tension: unclamping promotes slip)
    dCFF = (tau_s - tau_s_ref) sgn(tau_s_ref) + mu (sigma_n - sigma_n_ref)

Usage: tidal_cff.py <sim> <run> <reference> [mu] -> <sim>/cff_<run>_vs_<reference>_<step>.png
The receiver plane at a grid point is parallel to the stripes (the local direction
of the stripe family in the azimuthal projection) with its in-plane normal."""
import glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sim, run, ref = sys.argv[1:4]
mu = float(sys.argv[4]) if len(sys.argv) > 4 else 0.4


def receiver_frame(d):
    """Stripe-parallel tangent and in-plane normal at the polar grid points."""
    rho, psi = d["rho"].ravel(), d["psi"].ravel()
    c, e1, e2 = d["centre"], d["e1"], d["e2"]
    a, b = rho * np.cos(psi), rho * np.sin(psi)
    eps = 1e-5
    def pmap(a, b):
        r = np.maximum(np.hypot(a, b), 1e-30)
        return np.cos(r)[:, None] * c[None] + (np.sin(r) / r)[:, None] * (a[:, None] * e1[None] + b[:, None] * e2[None])
    p = pmap(a, b)
    t = (pmap(a + eps, b) - pmap(a - eps, b)) / (2 * eps)        # along the stripes
    t -= (t * p).sum(1)[:, None] * p
    t /= np.maximum(np.linalg.norm(t, axis=1), 1e-30)[:, None]
    n = np.cross(p, t)                                            # in-plane normal
    return p, t, n


def resolve(S, t, n):
    tau = np.einsum("ni,nij,nj->n", t, S, n)
    sn = np.einsum("ni,nij,nj->n", n, S, n)
    return tau, sn


files = sorted(glob.glob(os.path.join(sim, run, "polar_*.npz")))
for f in files:
    step = os.path.basename(f)[6:10]
    g = os.path.join(sim, ref, f"polar_{step}.npz")
    if not os.path.exists(g):
        continue
    A, B = np.load(f), np.load(g)
    p, t, n = receiver_frame(A)
    tau_a, sn_a = resolve(A["S"], t, n)
    tau_b, sn_b = resolve(B["S"], t, n)
    cff_a = np.abs(tau_a) + mu * sn_a
    cff_b = np.abs(tau_b) + mu * sn_b
    dcff = (tau_a - tau_b) * np.sign(tau_b + 1e-30) + mu * (sn_a - sn_b)
    shape = A["rho"].shape
    X, Y = A["rho"] * np.cos(A["psi"]), A["rho"] * np.sin(A["psi"])
    faults = A["faults"] if A["faults"].size else B["faults"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    lim = max(np.abs(cff_a).max(), np.abs(cff_b).max()); dlim = np.abs(dcff).max() + 1e-30
    for ax, fld, ttl, vmin, vmax in ((axes[0], cff_a.reshape(shape), f"CFF  {run}", -lim, lim),
                                     (axes[1], cff_b.reshape(shape), f"CFF  {ref} (reference)", -lim, lim),
                                     (axes[2], dcff.reshape(shape), f"dCFF  {run} - {ref}", -dlim, dlim)):
        pc = ax.pcolormesh(X, Y, fld, cmap="RdBu_r", vmin=vmin, vmax=vmax, shading="auto")
        for fp in faults:
            ax.plot(fp[:, 0], fp[:, 1], "k", lw=2)
        for r in (0.25, 0.5, 0.75):
            ax.add_patch(plt.Circle((0, 0), r, fill=False, color="gray", lw=0.5, alpha=0.6))
        ax.set_aspect("equal"); ax.set_xlim(-0.9, 0.9); ax.set_ylim(-0.9, 0.9); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(ttl, fontsize=10, loc="left"); fig.colorbar(pc, ax=ax, shrink=0.8)
    fig.suptitle(f"Coulomb failure stress on stripe-parallel receiver planes, mu = {mu}: step {int(step)}, t = {float(A['t']):.3f} periods",
                 fontsize=10)
    out = os.path.join(sim, f"cff_{run}_vs_{ref}_{step}.png")
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    print("wrote", out, "dCFF range", f"{dcff.min():.4f} {dcff.max():.4f}")
