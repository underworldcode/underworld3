"""A true Y-branch, bracketed: does the near-miss tributary capture it?

A genuine branch (three arms meeting at a point) cannot be split — but
it can be DECOMPOSED two ways, and the pair brackets the truth:

- A: the TRUNK is continuous (west + east arms as one fault); the
  splay abuts, stopping a ligament short.
- B: the BENT fault is continuous (west arm + splay as one deliberately
  kinked polyline — the kink response is the physics, so no smoothed
  normal); the east arm abuts.

Each decomposition welds a different pair of arms through the junction
and feeds the third across a ligament. If both give the same slip on
every arm at a small gap, the offset representation reproduces the true
branch; sweeping the gap in decomposition A shows how close "close by"
has to be. All three arms slip freely under the same drive.
"""
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

import underworld3 as uw
from underworld3.meshing.surfaces import prepare_fault_network
from underworld3.utilities import fault_contact

import common

pv.OFF_SCREEN = True
D = os.path.dirname(os.path.abspath(__file__))
H = 0.012
MU_P = 0.4
TREND = np.degrees(np.arctan2(0.10, 0.70))

J = np.array([0.50, 0.50])                       # the branch point
W_END = np.array([0.15, 0.45])
E_END = np.array([0.85, 0.55])
S_END = np.array([0.80, 0.76])
ARMS = {"west": (W_END - J) / np.linalg.norm(W_END - J),
        "east": (E_END - J) / np.linalg.norm(E_END - J),
        "splay": (S_END - J) / np.linalg.norm(S_END - J)}


def decomposition(kind, lig):
    if kind == "A":                              # trunk continuous
        faults = [("Trunk", np.array([W_END, E_END])),
                  ("Splay", np.array([J, S_END]))]
        through = ["Trunk"]
    else:                                        # bent fault continuous
        faults = [("Bent", np.array([W_END, J, S_END])),
                  ("East", np.array([J, E_END]))]
        through = ["Bent"]
    return prepare_fault_network(faults, spacing=H, ligament=lig,
                                 through=through, verbose=False)


def run(tag, prepared):
    child = common.base_mesh(H).add_fault(prepared)
    stokes = common.stokes_on(
        child, common.boundary_simple_shear(child, TREND))
    for n, _p in prepared:
        stokes.add_fault_bc(0, boundary=n)
    t0 = time.perf_counter()
    fault_contact.solve_with_fault(stokes, picard=2)
    arms = {k: ([], []) for k in ARMS}
    for n, _p in prepared:
        coords, jumps, normals = fault_contact.fault_pair_jumps(
            stokes, n, stokes._rotated_freeslip_info)
        if not len(coords):
            continue
        tang = np.column_stack([-normals[:, 1], normals[:, 0]])
        V = np.abs(np.einsum("ij,ij->i", jumps, tang))
        R = coords - J
        # classify each pair node by the arm it lies along
        proj = {k: R @ t for k, t in ARMS.items()}
        dist2 = {k: np.einsum("ij,ij->i", R - np.outer(
            np.clip(proj[k], 0, None), ARMS[k]),
            R - np.outer(np.clip(proj[k], 0, None), ARMS[k]))
            for k in ARMS}
        best = np.argmin(np.vstack([dist2[k] for k in ARMS]), axis=0)
        for i, k in enumerate(ARMS):
            sel = best == i
            arms[k][0].extend(proj[k][sel])
            arms[k][1].extend(V[sel])
    out = {}
    for k in ARMS:
        s = np.asarray(arms[k][0])
        v = np.asarray(arms[k][1])
        order = np.argsort(s)
        out[k] = (s[order], v[order])
    print(f"[{tag}] peaks " + "  ".join(
        f"{k} {out[k][1].max():.4f}" for k in ARMS)
        + f"  ({time.perf_counter() - t0:.1f} s)", flush=True)
    return out


cases = [("A, gap 1h", "A", 1.0, "#1565c0", "-"),
         ("A, gap 2h", "A", 2.0, "#64b5f6", "--"),
         ("A, gap 4h", "A", 4.0, "#b3d7f7", ":"),
         ("B, gap 1h", "B", 1.0, "#c62828", "-")]
profiles = {}
for tag, kind, lig, _c, _ls in cases:
    prepared, _rep = decomposition(kind, lig)
    profiles[tag] = run(tag, prepared)

fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), sharey=True)
for ax, arm in zip(axes, ("west", "east", "splay")):
    for tag, _k, _l, col, ls in cases:
        s, v = profiles[tag][arm]
        ax.plot(s, v, ls, lw=1.4, color=col, label=tag)
    ax.set_title(f"the {arm} arm", fontsize=10.5)
    ax.set_xlabel("distance from the branch point")
axes[0].set_ylabel("|slip|")
axes[0].legend(fontsize=8, title="decomposition, ligament")
fig.suptitle(
    "One true Y-branch, two decompositions: continuous-trunk (A) vs "
    "continuous-bend (B).\nAgreement between A and B at small gap = the "
    "near-miss tributary reproduces the true branch", fontsize=10.5)
fig.tight_layout()
out = os.path.join(D, "true-branch.png")
fig.savefig(out, dpi=200)
print("wrote", out, flush=True)
