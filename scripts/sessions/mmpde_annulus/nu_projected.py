"""Nu from a global projection of the total heat flux q = u T - grad T, integrated over
both surfaces, against the raw-gradient surface Nu the runs log. Reads saved snapshots
(needs the refined-reload fix, #690). Usage: nu_projected.py RUN_DIR STEP [STEP ...]"""
import sys
import numpy as np
import sympy
import underworld3 as uw

run, steps = sys.argv[1], [int(s) for s in sys.argv[2:]]
Q_COND = 2.0 * np.pi / np.log(2.0)          # conductive flux of the r = 0.5..1 annulus, Delta T = 1
for step in steps:
    label = f"step{step:04d}"
    mesh = uw.discretisation.Mesh(f"{run}/{label}.mesh.00000.h5", qdegree=3)
    T = uw.discretisation.MeshVariable("T_v2p1", mesh, 1, degree=3)
    V = uw.discretisation.MeshVariable("V_v2p1", mesh, mesh.dim, degree=2)
    T.read_timestep(label, "T_v2p1", 0, outputPath=run)
    V.read_timestep(label, "V_v2p1", 0, outputPath=run)
    X = mesh.CoordinateSystem.X
    gradT = sympy.Matrix([[T.sym[0].diff(X[0]), T.sym[0].diff(X[1])]])
    q = uw.discretisation.MeshVariable("q", mesh, mesh.dim, degree=2)
    proj = uw.systems.Vector_Projection(mesh, q)
    proj.uw_function = V.sym * T.sym[0] - gradT
    proj.smoothing = 0.0
    proj.solve()
    n = mesh.Gamma_N
    qn = q.sym[0] * n[0] + q.sym[1] * n[1]
    raw = -(gradT[0] * n[0] + gradT[1] * n[1])
    out = {}
    for name in ("Upper", "Lower"):
        b = getattr(mesh.boundaries, name).name
        out[name] = (uw.maths.BdIntegral(mesh=mesh, fn=qn, boundary=b).evaluate(),
                     uw.maths.BdIntegral(mesh=mesh, fn=raw, boundary=b).evaluate())
    # Outward flux at r = 1 is positive; at r = 0.5 the heat comes in, so flip its sign.
    nu_up_p, nu_up_r = out["Upper"][0] / Q_COND, out["Upper"][1] / Q_COND
    nu_lo_p, nu_lo_r = -out["Lower"][0] / Q_COND, -out["Lower"][1] / Q_COND
    uw.pprint(f"RESULT {run.split('/')[-1]} step {step}: projected Nu outer {nu_up_p:.3f} inner {nu_lo_p:.3f} | "
              f"raw-gradient Nu outer {nu_up_r:.3f} inner {nu_lo_r:.3f}")
