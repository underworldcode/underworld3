"""Volume-conservation check for the free-surface continent runs.

Reads ONLY the saved checkpoints (pyvista VTU files) — no
simulation re-run. For an incompressible Stokes flow the rock
volume should be exactly conserved; any change measures numerical
compressibility error.

Initial undeformed annulus area (2D volume per unit z):
    A_0 = π · (r_o² − r_inner²) = π · (1² − 0.5²) = π·0.75
"""

import os
import glob
import numpy as np
import pyvista as pv


OUT_DIR = "output"
SNAP_DIRS = {
    "structured":   os.path.join(OUT_DIR, "continent_fs_snapshots_struct"),
    "unstructured": os.path.join(OUT_DIR, "continent_fs_snapshots"),
    "structured_capped": os.path.join(
        OUT_DIR, "continent_fs_snapshots_struct_capped"),
    "structured_capped_half": os.path.join(
        OUT_DIR, "continent_fs_snapshots_struct_capped_half"),
    "structured_v3p2_cap18": os.path.join(
        OUT_DIR, "continent_fs_snapshots_struct_v3p2"),
    "structured_v2p1_curved_cap18": os.path.join(
        OUT_DIR, "continent_fs_snapshots_struct_v2p1_curved"),
    "structured_v2p1_tol7_cap18": os.path.join(
        OUT_DIR, "continent_fs_snapshots_struct_v2p1_tol7"),
    "structured_v2p1_full_cap18": os.path.join(
        OUT_DIR, "continent_fs_snapshots_struct_v2p1_full"),
    "structured_v2p1_full_dtf05": os.path.join(
        OUT_DIR, "continent_fs_snapshots_struct_v2p1_full_dtf05"),
    "structured_v2p1_sl_cap18": os.path.join(
        OUT_DIR, "continent_fs_snapshots_struct_v2p1_sl"),
    "structured_v2p1_rk4sl_cap18": os.path.join(
        OUT_DIR, "continent_fs_snapshots_struct_v2p1_rk4sl"),
    "structured_v2p1_sl_uncap": os.path.join(
        OUT_DIR, "continent_fs_snapshots_struct_v2p1_sl_uncap"),
    "structured_v2p1_sl_dtf05": os.path.join(
        OUT_DIR, "continent_fs_snapshots_struct_v2p1_sl_dtf05"),
    "structured_v2p1_rk4sl_uncap": os.path.join(
        OUT_DIR, "continent_fs_snapshots_struct_v2p1_rk4sl_uncap"),
}


def vtu_area(path):
    """Total cell area of a pyvista VTU."""
    m = pv.read(path)
    sized = m.compute_cell_sizes(length=False, area=True, volume=False)
    areas = np.asarray(sized.cell_data["Area"])
    return float(areas.sum())


def main():
    A_0 = np.pi * (1.0 ** 2 - 0.5 ** 2)
    print(f"  Reference straight-edge annulus area A_0 = π·0.75 = "
          f"{A_0:.6f}")
    print(f"  Curved-area metric uses the per-run UW3 Integral of the "
          f"undeformed mesh\n  (stored in the profile npz as "
          f"area_uw_initial), so ΔA is independent of\n  pyvista's "
          f"straight-edge cell representation.\n")

    schemes = ['rk2', 'rk4', 'rk2_full', 'rk2_sl', 'rk4_sl',
               'fe_sl', 'curvS', 'midpoint']
    labels = ['halfway', 'final']

    for mesh_kind, snap_dir in SNAP_DIRS.items():
        print(f"=== {mesh_kind} mesh ({snap_dir}) ===")
        if not os.path.isdir(snap_dir):
            print(f"  (no snapshots)")
            continue
        print(f"  {'scheme':>10s}  {'label':>8s}  "
              f"{'pv-Area':>12s}  {'pv-ΔA/A_0':>11s}    "
              f"{'curved-Area':>12s}  {'curved-ΔA/A_init':>17s}")
        for scheme in schemes:
            for label in labels:
                vtu = os.path.join(snap_dir,
                                   f"pv_{scheme}_{label}.vtu")
                npz = os.path.join(snap_dir,
                                   f"profile_{scheme}_{label}.npz")
                if not os.path.isfile(vtu):
                    continue
                A_pv = vtu_area(vtu)
                pct_pv = 100.0 * (A_pv - A_0) / A_0
                pv_str = f"{A_pv:>12.6f}  {pct_pv:>+10.4f}%"
                cv_str = f"{'':>12s}  {'(no npz)':>17s}"
                if os.path.isfile(npz):
                    d = np.load(npz)
                    if 'area_uw' in d.files and 'area_uw_initial' in d.files:
                        A_uw = float(d['area_uw'])
                        A_uw0 = float(d['area_uw_initial'])
                        pct_uw = 100.0 * (A_uw - A_uw0) / A_uw0
                        cv_str = (f"{A_uw:>12.6f}  "
                                  f"{pct_uw:>+16.4f}%")
                print(f"  {scheme:>10s}  {label:>8s}  "
                      f"{pv_str}    {cv_str}")
        print()


if __name__ == "__main__":
    main()
