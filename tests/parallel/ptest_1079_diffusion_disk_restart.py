"""Fresh-process worker: write a symbolic snapshot or restore and continue."""

import underworld3 as uw

params = uw.Params(
    uw_phase=uw.Param("write", type=uw.ParamType.STRING,
                      description="Checkpoint phase: write or resume."),
)
assert params.uw_phase in ("write", "resume")
uw.reset_default_model()
model = uw.get_default_model()
mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0., 0.), maxCoords=(1., 1.), cellSize=0.3,
)
temperature = uw.discretisation.MeshVariable("T", mesh, 1, degree=1)
temperature.array[:, 0, 0] = temperature.coords[:, 0]
solver = uw.systems.Diffusion(mesh, temperature, order=2, theta=1.)
solver.constitutive_model = uw.constitutive_models.DiffusionModel
solver.constitutive_model.Parameters.diffusivity = 0.05
solver.tolerance = 1e-12
if params.uw_phase == "write":
    for _ in range(3):
        solver.solve(timestep=0.01, zero_init_guess=False)
    model.save_state(file="restart.h5")
else:
    model.load_state("restart.h5")
    model.save_state(file="restored.h5")

for step, dt in enumerate((0.01, 0.015)):
    solver.solve(timestep=dt, zero_init_guess=False)
    model.save_state(file=f"{params.uw_phase}_{step}.h5")
