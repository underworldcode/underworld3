"""Recompute vrms and Nu from a run's saved snapshots, all as integrals.

vrms      = sqrt(∫ v·v dA / ∫ dA)
vrms_node = the old node average, for comparison
Nu        = ∫ -∇T·n ds on the outer (cold) boundary / (2π / ln 2)
Nu_mid    = the radial flux through r = 0.75 as a line integral of the solved
            fields, sampled at 1440 points (June's definition), / (2π / ln 2)

Usage: python recompute_diagnostics.py RUN_DIR OUT.csv
"""
import glob
import os
import re
import sys

import numpy as np
import sympy
import underworld3 as uw

run, out = sys.argv[1], sys.argv[2]
labels = sorted((int(m.group(1)), os.path.basename(f).split(".mesh")[0])
                for f in glob.glob(f"{run}/step*.mesh.00000.h5")
                if (m := re.search(r"step(\d+)\.mesh\.00000\.h5$", f)))
if os.path.exists(f"{run}/init.mesh.00000.h5"):
    labels = [(0, "init")] + labels
t_of = {}
for line in open(f"{run}/run_log.txt"):
    p = line.split()
    if p and p[0].isdigit():
        t_of[int(p[0])] = float(p[1])
t_of[0] = 0.0

Q_COND = 2.0 * np.pi / np.log(2.0)
th = np.linspace(0, 2 * np.pi, 1440, endpoint=False)
pts = np.column_stack([0.75 * np.cos(th), 0.75 * np.sin(th)])
rows = []
for step, label in labels:
    mesh = uw.discretisation.Mesh(f"{run}/{label}.mesh.00000.h5")
    T = uw.discretisation.MeshVariable("T_v2p1", mesh, 1, degree=3)
    V = uw.discretisation.MeshVariable("V_v2p1", mesh, mesh.dim, degree=2)
    T.read_timestep(label, "T_v2p1", 0, outputPath=run)
    V.read_timestep(label, "V_v2p1", 0, outputPath=run)
    one = uw.discretisation.MeshVariable("one", mesh, 1, degree=1)
    one.data[...] = 1.0
    X = mesh.CoordinateSystem.X
    n = mesh.Gamma_N
    area = uw.maths.Integral(mesh, one.sym[0]).evaluate()
    vv = uw.maths.Integral(mesh, V.sym.dot(V.sym)).evaluate()
    qn = -(T.sym[0].diff(X[0]) * n[0] + T.sym[0].diff(X[1]) * n[1])
    nu = uw.maths.BdIntegral(mesh=mesh, fn=qn, boundary="Upper").evaluate() / Q_COND
    er = mesh.CoordinateSystem.unit_e_0
    qr = (V.sym[0] * er[0] + V.sym[1] * er[1]) * T.sym[0] \
        - (T.sym[0].diff(X[0]) * er[0] + T.sym[0].diff(X[1]) * er[1])
    q = np.asarray(uw.function.global_evaluate(qr, pts)).reshape(-1)
    nu_mid = q.mean() * 0.75 * 2 * np.pi / Q_COND
    vsq = np.asarray(uw.function.evaluate(V.sym.dot(V.sym), mesh.X.coords)).reshape(-1)
    s_ = uw.mpi.comm.allreduce(float(vsq.sum()))
    c_ = uw.mpi.comm.allreduce(int(vsq.size))
    rows.append((step, t_of.get(step, np.nan), np.sqrt(vv / area), np.sqrt(s_ / c_),
                 nu, nu_mid, area))
    if uw.mpi.rank == 0:
        print(f"{label:>9} t={rows[-1][1]:.5f} vrms={rows[-1][2]:.4g} node={rows[-1][3]:.4g} "
              f"Nu={nu:.4f} Nu_mid={nu_mid:.4f} area={area:.6f}", flush=True)
if uw.mpi.rank == 0:
    np.savetxt(out, np.array(rows), delimiter=",", fmt="%.6g",
               header="step,t,vrms,vrms_node,Nu_surface,Nu_mid,area", comments="")
