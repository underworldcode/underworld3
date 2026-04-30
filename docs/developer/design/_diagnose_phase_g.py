"""Phase G diagnostic — point-wise σ comparison + yield-enforcement check.

Reads spatial snapshots from baseline and v5_const_alpha (the only two
that completed). For each snapshot, extracts:
  - σ_eq at fault midpoint (where yield should be most active)
  - σ_eq at bulk reference point (where no yield)
  - σ_y at the same points (for yield comparison)
  - viscosity at fault midpoint (should drop dramatically when yielding)
  - max σ_eq in domain + its location

Verifies whether each architecture actually enforces yield in the fault
zone, and whether the σ-amplitude differences are at the fault or in
the bulk.
"""

import os
import numpy as np

OUT_DIR = "output"
SNAPSHOT_STEPS = (30, 60, 90, 120)

# Fault and reference points
THETA_DEG = 15.0
W = H = 1.0
FAULT_LENGTH = 0.6
FAULT_WIDTH = 0.06
TAU_Y_FAULT = 0.05
TAU_Y_BULK = 200.0
ETA_VE = 1.0
MU_VE = 1.0
DT = 0.05


def _load(label, step):
    path = os.path.join(OUT_DIR, f"phase_g_{label}_spatial_step{step:03d}.npz")
    if not os.path.exists(path):
        return None
    return dict(np.load(path))


def _sigma_y_at(coord_x, coord_y):
    """σ_y at a point given the fault geometry (Gaussian weak zone)."""
    cx, cy = 0.5, 0.5
    theta = np.radians(THETA_DEG)
    n_x = -np.sin(theta); n_y = np.cos(theta)
    sd = abs((coord_x - cx) * n_x + (coord_y - cy) * n_y)
    al = (coord_x - cx) * n_y - (coord_y - cy) * n_x
    in_extent = abs(al) <= 0.5 * FAULT_LENGTH
    if in_extent:
        weakness = ((1.0 / TAU_Y_FAULT) * np.exp(-(sd / FAULT_WIDTH) ** 2)
                    + (1.0 / TAU_Y_BULK) * (1.0 - np.exp(-(sd / FAULT_WIDTH) ** 2)))
    else:
        weakness = 1.0 / TAU_Y_BULK
    return 1.0 / weakness


def _nearest_idx(coords, query):
    d = np.sum((coords - query) ** 2, axis=1)
    return int(np.argmin(d))


def main():
    fault_pt = np.array([0.5, 0.5])  # fault centre
    bulk_pt = np.array([0.25, 0.25])  # well inside elastic bulk

    sy_fault = _sigma_y_at(*fault_pt)
    sy_bulk = _sigma_y_at(*bulk_pt)
    print(f"σ_y at fault centre (0.5,0.5): {sy_fault:.4e}")
    print(f"σ_y at bulk point (0.25,0.25): {sy_bulk:.4e}")
    print()

    cases = [
        ("v3_baseline_const_eta", "baseline"),
        ("v5_const_alpha",        "v5_const"),
    ]

    print(f"{'step':>5}  {'V_top':>7}  {'variant':<12}  "
          f"{'σ_eq_fault':>12}  {'σ_eq_fault/σ_y':>16}  "
          f"{'σ_eq_bulk':>10}  {'visc_fault':>12}  "
          f"{'σ_eq_max':>10}  {'where_max(x,y)':>16}")
    print("-" * 130)

    for step in SNAPSHOT_STEPS:
        for label, name in cases:
            snap = _load(label, step)
            if snap is None:
                continue
            sigma_coords = snap["sigma_coords"]
            sigma_eq = snap["sigma_eq"]
            visc = snap["viscosity"]

            i_fault = _nearest_idx(sigma_coords, fault_pt)
            i_bulk = _nearest_idx(sigma_coords, bulk_pt)
            i_max = int(np.argmax(sigma_eq))
            mx, my = sigma_coords[i_max]

            print(
                f"{step:>5}  {float(snap['V_top']):>+7.3f}  {name:<12}  "
                f"{sigma_eq[i_fault]:>12.4e}  "
                f"{sigma_eq[i_fault]/sy_fault:>16.3f}  "
                f"{sigma_eq[i_bulk]:>10.4e}  "
                f"{visc[i_fault]:>12.4e}  "
                f"{sigma_eq[i_max]:>10.4f}  "
                f"({mx:.2f}, {my:.2f})"
            )
        print()


if __name__ == "__main__":
    main()
