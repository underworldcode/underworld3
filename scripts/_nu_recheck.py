"""Validate the fixed Nusselt (workflow BdIntegral / conductive)
on the existing settled checkpoints — no re-simulation."""
import numpy as np, glob, re, os
import underworld3 as uw

D = "/tmp/metric_mesh/sat"
QC = 2.0 * np.pi / np.log(1.0 / 0.5)   # annulus log-conduction flux


def latest(tag):
    ix = []
    for f in glob.glob(f"{D}/sat_{tag}.mesh.T.*.h5"):
        m = re.search(r"\.mesh\.T\.(\d+)\.h5$", os.path.basename(f))
        if m:
            ix.append(int(m.group(1)))
    return max(ix) if ix else None


print(f"Q_cond (annulus, logarithmic) = {QC:.4f}  "
      f"(Nu = 1 at pure conduction)")
print(f"{'run':>22} {'ckpt':>5} {'Q_meas':>9} {'Nu(fixed)':>9}  "
      f"old-stencil")
for tag, lab, old in [("ref24", "ref res-24", "~1.69"),
                      ("u16", "uniform res-16", "~1.14"),
                      ("a16p", "a16p conservative", "~1.13"),
                      ("a16s", "a16s aggressive", "~1.13")]:
    i = latest(tag)
    if i is None:
        print(f"{lab:>22}   (no ckpt)")
        continue
    m = uw.discretisation.Mesh(f"{D}/sat_{tag}.mesh.{i:05}.h5")
    T = uw.discretisation.MeshVariable(
        "T", m, vtype=uw.VarType.SCALAR, degree=3, continuous=True)
    T.read_timestep(f"sat_{tag}", "T", i, outputPath=D)
    X = m.CoordinateSystem.X
    er = m.CoordinateSystem.unit_e_0
    g = T.sym[0].diff(X[0]) * er[0] + T.sym[0].diff(X[1]) * er[1]
    Qm = -float(uw.maths.BdIntegral(
        m, g, m.boundaries.Upper.name).evaluate())
    print(f"{lab:>22} {i:5d} {Qm:9.3f} {Qm/QC:9.3f}      {old}")
