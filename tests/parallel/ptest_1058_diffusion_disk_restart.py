"""Fresh-process worker: write a symbolic snapshot or restore and continue."""

import sys
import numpy as np
import underworld3 as uw

phase = sys.argv[1]
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
if phase == "write":
    for _ in range(3):
        solver.solve(timestep=0.01, zero_init_guess=False)
    model.save_state(file="restart.h5")
else:
    model.load_state("restart.h5")

values = {f"restored_{name}": np.array(var.array)
          for name, var in mesh.vars.items()}
values["dt_history"] = np.array(solver.DFDt._dt_history)
values["n_solves"] = solver.DFDt._n_solves_completed
values["initialized"] = solver.DFDt._history_initialised
for step, dt in enumerate((0.01, 0.015)):
    solver.solve(timestep=dt, zero_init_guess=False)
    values[f"T_{step}"] = np.array(temperature.array)
np.savez(f"{phase}_{uw.mpi.rank}.npz", **values)
