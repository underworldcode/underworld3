"""The same Taylor test, with the library's backward pass in place of the
hand-rolled AdjointMachinery: `uw.adjoint.TranscriptAdjoint` walks the run's
own transcript. The control is still the blob centre, reached through the
dual on beta_0 and the chain rule d(beta_0)/d(centre).
"""
import sys
import numpy as np
import sympy
import underworld3 as uw

from forward_sinker_transcript import build_model, solve_forward, beta0_nodal
from generate_target_transcript import (TRUE_VISCOSITY_CONTRAST, TARGET_FILENAME,
                                        TARGET_INDEX, OUTPUT_DIR as TARGET_DIR)

CANDIDATE = (uw.quantity(300, "km"), uw.quantity(350, "km"))   # true is (250, 375)
H_VALUES = [uw.quantity(5.0, "km"), uw.quantity(0.5, "km"), uw.quantity(0.05, "km")]


if __name__ == "__main__":
    hs = [uw.quantity(float(a), "km") for a in sys.argv[1:]] or H_VALUES

    model = build_model(TRUE_VISCOSITY_CONTRAST)
    mesh, beta, v = model["mesh"], model["beta"], model["v"]
    v_target = uw.discretisation.MeshVariable("v_target", mesh, 2, degree=2)
    v_target.read_timestep(data_filename=TARGET_FILENAME, data_name="v",
                           index=TARGET_INDEX, outputPath=TARGET_DIR)
    misfit = sympy.Rational(1, 2) * (v.sym - v_target.sym).dot(v.sym - v_target.sym)

    transcript, final_state = solve_forward(model, CANDIDATE, initial_on_tape=True)
    uw.pprint("the run being differentiated:")
    for entry in transcript:
        uw.pprint(f"  {entry}")
    uw.pprint("")

    back = uw.adjoint.TranscriptAdjoint(model["uwmodel"], final_state)
    result = back.gradient(misfit, fields=[beta])
    dual = result["fields"][beta][:, 0, 0]
    _, dbeta0_dc = beta0_nodal(model, CANDIDATE)
    gx, gy, J = float(dual @ dbeta0_dc[:, 0]), float(dual @ dbeta0_dc[:, 1]), result["J"]
    uw.pprint(f"J = {J:.6e}   adjoint dJ/dcx0 = {gx:.6e} /m   dJ/dcy0 = {gy:.6e} /m")

    def Jof(c):
        solve_forward(model, c, initial_on_tape=True)
        return float(uw.maths.Integral(mesh, misfit).evaluate())

    for h in hs:
        cx, cy = CANDIDATE
        two_h = 2.0 * float(h.to("m").magnitude)
        fx = (Jof((cx + h, cy)) - Jof((cx - h, cy))) / two_h
        fy = (Jof((cx, cy + h)) - Jof((cx, cy - h))) / two_h
        uw.pprint(f"h={h}:  FD dJ/dcx0 {fx:.6e}  ratio {fx/gx:8.5f}   |   "
                  f"FD dJ/dcy0 {fy:.6e}  ratio {fy/gy:8.5f}")
