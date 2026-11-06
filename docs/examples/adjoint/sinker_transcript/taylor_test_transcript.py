"""Taylor test for the transcript-driven adjoint. Central finite difference on both
control components; the ratio FD/adjoint should sit at 1 and stay there as h
shrinks.

The control is the blob centre, a pair of LENGTHS, so both the adjoint gradient
and the finite difference are dJ/d(centre) per metre.
"""
import sys
import numpy as np
import underworld3 as uw

from forward_sinker_transcript import build_model, solve_forward
from generate_target_transcript import (TRUE_VISCOSITY_CONTRAST, TARGET_FILENAME,
                                     TARGET_INDEX, OUTPUT_DIR as TARGET_DIR)
from inverse_sinker_transcript import AdjointMachinery, compute_adjoint_gradient

CANDIDATE = (uw.quantity(300, "km"), uw.quantity(350, "km"))   # true is (250, 375)
H_VALUES = [uw.quantity(5.0, "km"), uw.quantity(0.5, "km"), uw.quantity(0.05, "km")]


if __name__ == "__main__":
    hs = [uw.quantity(float(a), "km") for a in sys.argv[1:]] or H_VALUES

    model = build_model(TRUE_VISCOSITY_CONTRAST)
    mesh = model["mesh"]
    v_target = uw.discretisation.MeshVariable("v_target", mesh, 2, degree=2)
    v_target.read_timestep(data_filename=TARGET_FILENAME, data_name="v",
                           index=TARGET_INDEX, outputPath=TARGET_DIR)
    mach = AdjointMachinery(model, v_target)

    transcript, final_state = solve_forward(model, CANDIDATE)
    mach.attach(transcript, final_state)

    uw.pprint("the run being differentiated:")
    for entry in transcript:
        uw.pprint(f"  {entry}")
    uw.pprint("")

    gx, gy, J = compute_adjoint_gradient(mach, CANDIDATE)
    uw.pprint(f"J = {J:.6e}   adjoint dJ/dcx0 = {gx:.6e} /m   "
              f"dJ/dcy0 = {gy:.6e} /m")

    def Jof(c):
        """Rerun the forward model from centre `c` and evaluate the misfit."""
        solve_forward(model, c)
        return float(uw.maths.Integral(mesh, mach.misfit_integrand).evaluate())

    for h in hs:
        cx, cy = CANDIDATE
        two_h = 2.0 * float(h.to("m").magnitude)
        fx = (Jof((cx + h, cy)) - Jof((cx - h, cy))) / two_h
        fy = (Jof((cx, cy + h)) - Jof((cx, cy - h))) / two_h
        uw.pprint(f"h={h}:  FD dJ/dcx0 {fx:.6e}  ratio {fx/gx:8.5f}   |   "
                  f"FD dJ/dcy0 {fy:.6e}  ratio {fy/gy:8.5f}")
