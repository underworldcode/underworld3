"""Proper Nusselt: total radial heat flux q_r = v_r·T - ∂T/∂r
projected to a nodal field, integrated over SHELLS (robust on
interior shells where advection is smooth — not the
BL-resolution-sensitive boundary ∂T/∂r).

Decisive checks:
  (A) analytic conduction (v=0, T=A ln r + B): every shell flow
      must = Q_cond = 2π/ln(Ro/Ri) ⇒ Nu = 1 EXACTLY. This
      objectively validates Q_cond and the method.
  (B) a settled checkpoint: compare boundary-diffusive Nu (old
      method) vs interior-shell total-flux Nu (proper) — if the
      proper one is much larger the boundary integral was
      under-resolving the 1-element BL (user's hypothesis).
"""
import numpy as np, glob, re, os
import sympy
import underworld3 as uw

R_i, R_o = 0.5, 1.0
Q_COND = 2.0 * np.pi / np.log(R_o / R_i)          # total cond flow
D = "/tmp/metric_mesh/sat"


def shell_flow(mesh, T, v, r, n=720):
    """Total radial heat flow through the circle of radius r:
    ∮ (v_r T - ∂T/∂r) r dθ, evaluated from a projected nodal
    flux field (FE-consistent), sampled on the ring."""
    X = mesh.CoordinateSystem.X
    er = mesh.CoordinateSystem.unit_e_0
    gradT_r = T.sym[0].diff(X[0]) * er[0] + T.sym[0].diff(X[1]) * er[1]
    vr = (v.sym[0] * er[0] + v.sym[1] * er[1]) if v is not None \
        else sympy.Integer(0)
    qsym = vr * T.sym[0] - gradT_r
    # project the flux to a nodal scalar (the user's "projected to
    # the nodes") so the integrand is the FE field, not raw ∂T
    qf = uw.discretisation.MeshVariable(
        f"qr_{id(mesh)}", mesh, vtype=uw.VarType.SCALAR,
        degree=2, continuous=True)
    proj = uw.systems.Projection(mesh, qf)
    proj.uw_function = qsym
    proj.smoothing = 0.0
    proj.solve()
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.column_stack([r * np.cos(th), r * np.sin(th)])
    q = np.asarray(uw.function.evaluate(qf.sym[0], pts)).reshape(-1)
    return float(q.mean() * r * 2.0 * np.pi)       # ∮ q r dθ


def boundary_diffusive_flow(mesh, T):
    X = mesh.CoordinateSystem.X
    er = mesh.CoordinateSystem.unit_e_0
    g = T.sym[0].diff(X[0]) * er[0] + T.sym[0].diff(X[1]) * er[1]
    return -float(uw.maths.BdIntegral(
        mesh, g, mesh.boundaries.Upper.name).evaluate())


# ---- (A) analytic conduction test --------------------------------
print(f"Q_cond (analytic total) = {Q_COND:.4f}")
m = uw.meshing.Annulus(radiusOuter=R_o, radiusInner=R_i,
                        cellSize=1.0 / 24, qdegree=3)
Tc = uw.discretisation.MeshVariable("Tc", m, vtype=uw.VarType.SCALAR,
                                    degree=3, continuous=True)
A = 1.0 / np.log(R_i / R_o)
r_e = np.sqrt((np.asarray(Tc.coords) ** 2).sum(axis=1))
Tc.data[:, 0] = A * np.log(r_e) + (-A * np.log(R_o))   # =1@Ri,0@Ro
print("(A) analytic conduction (v=0): Nu per shell "
      "(must be ~1 everywhere):")
for rr in (R_i + 1e-6, 0.6, 0.7, 0.75, 0.85, R_o - 1e-6):
    fl = shell_flow(m, Tc, None, rr)
    print(f"   r={rr:5.3f}  shell-flow={fl:8.4f}  "
          f"Nu={fl / Q_COND:6.4f}")
bd = boundary_diffusive_flow(m, Tc)
print(f"   boundary-diffusive flow={bd:8.4f}  "
      f"Nu={bd / Q_COND:6.4f}  (should also be ~1)")


# ---- (B) settled checkpoints: old vs proper ----------------------
def latest(tag):
    ix = []
    for f in glob.glob(f"{D}/sat_{tag}.mesh.T.*.h5"):
        mm = re.search(r"\.mesh\.T\.(\d+)\.h5$", os.path.basename(f))
        if mm:
            ix.append(int(mm.group(1)))
    return max(ix) if ix else None


print("\n(B) settled checkpoints — Nu boundary-diffusive (old) "
      "vs interior-shell total-flux (proper):")
for tag in ("ref24", "u16", "a16s"):
    i = latest(tag)
    if i is None:
        continue
    mm = uw.discretisation.Mesh(f"{D}/sat_{tag}.mesh.{i:05}.h5")
    T = uw.discretisation.MeshVariable("T", mm,
        vtype=uw.VarType.SCALAR, degree=3, continuous=True)
    V = uw.discretisation.MeshVariable("V", mm,
        vtype=uw.VarType.VECTOR, degree=2, continuous=True)
    T.read_timestep(f"sat_{tag}", "T", i, outputPath=D)
    V.read_timestep(f"sat_{tag}", "V", i, outputPath=D)
    bd = boundary_diffusive_flow(mm, T) / Q_COND
    mid = shell_flow(mm, T, V, 0.5 * (R_i + R_o)) / Q_COND
    s2 = shell_flow(mm, T, V, R_i + 0.30 * (R_o - R_i)) / Q_COND
    s3 = shell_flow(mm, T, V, R_i + 0.70 * (R_o - R_i)) / Q_COND
    print(f"  {tag:6s} ckpt{i:4d}: Nu_bdy(old)={bd:6.3f}  "
          f"Nu_shell @0.30={s2:6.3f} @0.50={mid:6.3f} "
          f"@0.70={s3:6.3f}  (steady ⇒ shells agree)")
