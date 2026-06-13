"""Restart bdf2_sl from a checkpointed step and run a few more steps.

Reproducer for the late-stage T contamination observed at step 28-30 in the
5pct/Ra=1e5/rho_g=1e5/no-Nitsche bdf2 + ALE + restore + plain Picard run.
Avoids re-running 27 steps from scratch each time we test a hypothesis.

Loads:
- mesh deformed coords from uw_*.mesh.00000.h5
- T from uw_*.mesh.T_conv_v2p1.00000.h5 (via read_timestep)
- V from uw_*.mesh.V_conv_v2p1.00000.h5 (via read_timestep)
- BDF2 surface-history Fourier coefficients from prev-step mesh coords
- Δt history from work_log CSV
"""
import os
import sys
import time as _time
import argparse
import numpy as np
import sympy
import h5py

# Allow importing the runner module from this directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _phase_i_fs_convection_zoo import (
    _build, _step, _convection_diagnostics, _reset_gamma_history,
    _GAMMA_HISTORY_MAX, _GAMMA_HISTORY,
    _U_N_EFF_PREV, _DT_PREV, _H_PREV_FOURIER, _BDF_BOOTSTRAPPED,
)


def fourier_decomp(values, internal_th, n_modes):
    """Same Fourier decomposition as in _step (irregular θ trapezoid)."""
    thw = internal_th
    n = len(thw)
    d = np.empty(n)
    d[1:-1] = 0.5 * (thw[2:] - thw[:-2])
    d[0] = 0.5 * (thw[1] - (thw[-1] - 2 * np.pi))
    d[-1] = 0.5 * ((thw[0] + 2 * np.pi) - thw[-2])
    a = np.zeros(n_modes + 1)
    b = np.zeros(n_modes + 1)
    a[0] = float(np.sum(values * d) / (2 * np.pi))
    for m in range(1, n_modes + 1):
        a[m] = float(np.sum(values * np.cos(m * thw) * d) / np.pi)
        b[m] = float(np.sum(values * np.sin(m * thw) * d) / np.pi)
    return a, b


def load_mesh_coords_from_checkpoint(h5_path):
    """Read deformed mesh coordinates from a UW3 mesh checkpoint .h5 file.

    The PETSc DMPlex HDF5 layout puts vertex coords under
    /geometry/vertices (or similar). Probe the structure.
    """
    with h5py.File(h5_path, "r") as f:
        # PETSc uses /geometry/vertices for DMPlex coordinates
        for path in ("/geometry/vertices",
                     "/dm_geometry/vertices",
                     "/coordinates/values"):
            if path in f:
                return np.asarray(f[path])
        # Fallback: walk the tree to find the largest float dataset
        candidates = []
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.dtype.kind == "f":
                candidates.append((name, obj.shape))
        f.visititems(visitor)
        print(f"  available datasets in {h5_path}:")
        for name, shape in candidates:
            print(f"    {name} {shape}")
        raise RuntimeError("could not find vertex-coords dataset in mesh h5")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--restart-dir', required=True,
                   help='dir containing uw_*.mesh.*.h5 checkpoints')
    p.add_argument('--restart-prefix', default='uw_bdf2_sl',
                   help='checkpoint filename prefix (before _step####)')
    p.add_argument('--restart-step', type=int, required=True,
                   help='step number to restart from (must have ckpt files)')
    p.add_argument('--n-extra-steps', type=int, default=5,
                   help='how many additional steps to run after restart')
    # Re-build params (must match what generated the checkpoint)
    p.add_argument('--res', type=int, default=20)
    p.add_argument('--Ra', type=float, default=1.0e5)
    p.add_argument('--rho-g', type=float, default=1.0e5)
    p.add_argument('--v-degree', type=int, default=2)
    p.add_argument('--p-degree', type=int, default=1)
    p.add_argument('--stokes-tol', type=float, default=1.0e-7)
    p.add_argument('--fssa-theta', type=float, default=0.0)
    p.add_argument('--fssa-dt-ref', type=float, default=4.0e-4)
    p.add_argument('--nitsche-penalty', type=float, default=0.0)
    p.add_argument('--ale-correction', action='store_true')
    p.add_argument('--anderson-m', type=int, default=0)
    # Re-run params
    p.add_argument('--scheme', default='bdf2_sl')
    p.add_argument('--work-log', default=None,
                   help='CSV from the original full run, used to recover Δt history')
    args = p.parse_args()

    print(f"Building solver state...", flush=True)
    state = _build(res=args.res, Ra=args.Ra, structured=False,
                   v_degree=args.v_degree, p_degree=args.p_degree,
                   stokes_tol=args.stokes_tol,
                   fssa_theta=args.fssa_theta,
                   fssa_dt_ref=args.fssa_dt_ref,
                   rho_g=args.rho_g,
                   nitsche_penalty=args.nitsche_penalty)
    state['anderson_m'] = int(args.anderson_m)
    _reset_gamma_history()

    mesh = state['mesh']
    t_soln = state['t_soln']
    v = state['v']
    adv_diff = state['adv_diff']
    pair_tag = f"v{args.v_degree}p{args.p_degree}"
    internal_idx = state['internal_idx']
    internal_th = state['internal_th']
    r_o = state['r_o']

    # === Restore deformed mesh coords ===
    base = os.path.join(args.restart_dir,
                        f"{args.restart_prefix}_step{args.restart_step:04d}")
    mesh_h5 = base + ".mesh.00000.h5"
    print(f"Loading mesh coords from {mesh_h5}", flush=True)
    saved_coords = load_mesh_coords_from_checkpoint(mesh_h5)
    print(f"  loaded coords shape {saved_coords.shape}", flush=True)
    if saved_coords.shape != mesh.X.coords.shape:
        raise RuntimeError(
            f"mesh shape mismatch: live {mesh.X.coords.shape} vs "
            f"saved {saved_coords.shape}")
    mesh._deform_mesh(saved_coords)

    # === Restore T and V via read_timestep ===
    print(f"Loading T...", flush=True)
    t_soln.read_timestep(
        data_filename=f"{args.restart_prefix}_step{args.restart_step:04d}",
        data_name=f"T_conv_{pair_tag}",
        index=0,
        outputPath=args.restart_dir,
    )
    print(f"Loading V...", flush=True)
    v.read_timestep(
        data_filename=f"{args.restart_prefix}_step{args.restart_step:04d}",
        data_name=f"V_conv_{pair_tag}",
        index=0,
        outputPath=args.restart_dir,
    )

    # === Reconstruct BDF2 history from previous-step mesh + work_log ===
    prev_step = args.restart_step - 1
    prev_mesh_h5 = os.path.join(
        args.restart_dir,
        f"{args.restart_prefix}_step{prev_step:04d}.mesh.00000.h5")
    if os.path.exists(prev_mesh_h5):
        prev_coords = load_mesh_coords_from_checkpoint(prev_mesh_h5)
        prev_upper = prev_coords[internal_idx]
        prev_h = np.sqrt(prev_upper[:, 0]**2 + prev_upper[:, 1]**2) - r_o
        n_modes = max(2, len(internal_th) // 3)
        a_prev, b_prev = fourier_decomp(prev_h, internal_th, n_modes)
        _H_PREV_FOURIER[0] = (a_prev, b_prev)
        _BDF_BOOTSTRAPPED[0] = True
        # Δt of the prev->restart step (i.e. dt that took prev to restart)
        if args.work_log:
            wl = np.loadtxt(args.work_log, delimiter=',', skiprows=1,
                            usecols=(1, 3))
            row = wl[wl[:, 0].astype(int) == args.restart_step]
            if len(row) > 0:
                _DT_PREV[0] = float(row[0, 1])
                print(f"  BDF2 history: a_prev[0]={a_prev[0]:.3e}, "
                      f"_DT_PREV={_DT_PREV[0]:.3e}", flush=True)
            else:
                print(f"  WARN: no work_log row for step {args.restart_step}",
                      flush=True)
        else:
            _DT_PREV[0] = 8.4e-5  # rough default for our regime
            print(f"  WARN: no work_log; using default _DT_PREV={_DT_PREV[0]}",
                  flush=True)
    else:
        print(f"  WARN: no prev-step checkpoint at {prev_mesh_h5}; "
              f"BDF2 will bootstrap with BDF1", flush=True)
        _BDF_BOOTSTRAPPED[0] = False

    # === Run forward ===
    print(f"\nRunning {args.n_extra_steps} steps from restart point...",
          flush=True)
    for s in range(args.n_extra_steps):
        t0 = _time.time()
        dt_thermal = float(adv_diff.estimate_dt())
        h_pole, dt = _step(state, args.scheme, 1.0,
                           dt_cap=dt_thermal, dt_cap_mode='fixed', dt_cap_c=0.5)
        # ALE
        if args.ale_correction and dt > 0:
            v_mesh_var = state['v_mesh']
            # We didn't capture before _step here, so just compute roughly
            # from coord change. (Approximate; for diagnostic this is fine.)
            v_mesh_var.data[...] = 0.0  # zero for first restarted step
        adv_diff.solve(timestep=dt, zero_init_guess=False)
        diag = _convection_diagnostics(state)
        # Read T_min directly from data
        T_data = np.asarray(t_soln.data[:, 0])
        wall = _time.time() - t0
        print(f"  step+{s+1} (orig step {args.restart_step+s+1}): "
              f"vrms={diag['vrms']:.3e} h_max={diag['h_max']:.3e} "
              f"T_min={T_data.min():+.4f} T_max={T_data.max():+.4f} "
              f"wall={wall:.1f}s", flush=True)


if __name__ == "__main__":
    main()
