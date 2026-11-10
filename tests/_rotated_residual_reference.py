"""Independent original-equation check shared by rotated solver regressions."""
from underworld3.utilities import rotated_bc

def _assert_original_constrained_equations(stokes, info):
    """Independently evaluate the finalized equations and their rest-state scale."""
    trial = info["U"].copy()
    residual = trial.duplicate()
    rotated = trial.duplicate()
    values = []
    try:
        for state in ("final", "rest"):
            if state == "rest":
                trial.set(0)
            stokes.snes.computeFunction(trial, residual)
            info["Q"].mult(residual, rotated)
            rotated_bc._zero_rows_local(rotated, info["normal_rows"])
            values.append(rotated.norm())
        rtol, atol, _ = stokes._rotated_tolerances()
        assert values[0] <= rtol * values[1] + atol, values
    finally:
        for vector in (rotated, residual, trial):
            vector.destroy()

