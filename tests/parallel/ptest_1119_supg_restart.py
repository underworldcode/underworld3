"""Worker for a fresh-process SUPG restart; no Stokes or A1 setup."""

from dataclasses import asdict
import h5py
import numpy as np

import underworld3 as uw


params = uw.Params(
    uw_method=uw.Param("pc2", type=uw.ParamType.STRING),
    uw_phase=uw.Param("full", type=uw.ParamType.STRING),
)
assert params.uw_method in ("pc2", "cn", "bdf2")
assert params.uw_phase in ("full", "write", "resume")
uw.reset_default_model()
orchestration_model = uw.get_default_model()
mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0),
    cellSize=0.25, qdegree=4, regular=False, filename="mesh.msh",
)
temperature = uw.discretisation.MeshVariable("T", mesh, 1, degree=1)
velocity = uw.discretisation.MeshVariable("U", mesh, 3, degree=1)
temperature.array[:, 0, 0] = np.prod(np.sin(np.pi * np.asarray(temperature.coords)), axis=1)
velocity.array[...] = 0.0
velocity.array[:, 0, 0] = 0.2
settings = ({"time_integrator": "citcoms"} if params.uw_method == "pc2"
            else {"order": 1, "theta": 0.5} if params.uw_method == "cn"
            else {"order": 2})
thermal = uw.systems.AdvDiffusionSUPG(mesh, temperature, velocity.sym, **settings)
thermal.constitutive_model.Parameters.diffusivity = 0.01
for boundary in mesh.boundaries:
    if boundary.name not in ("All_Boundaries", "Null_Boundary"):
        thermal.add_dirichlet_bc(0.0, boundary.name)
thermal.petsc_options["ksp_rtol"] = 1e-14
thermal.petsc_options["ksp_atol"] = 0.0
thermal.petsc_options["snes_rtol"] = 1e-13
thermal.petsc_options["snes_atol"] = 1e-14
orchestration_model.tracker.step = 0
orchestration_model.tracker.time = 0.0


def capture():
    """All evolving fields and numerical metadata, separately from PETSc files."""
    fields = [temperature, velocity]
    if thermal.temperature_rate is not None:
        fields.append(thermal.temperature_rate)
    else:
        fields.extend(thermal.DuDt.psi_star)
    record = {field.clean_name: np.array(field.array) for field in fields}
    record["coords"] = np.asarray(temperature.coords)
    record["step"] = orchestration_model.tracker.step
    record["time"] = orchestration_model.tracker.time
    record["estimate_dt"] = float(thermal.estimate_dt())
    for name, value in asdict(thermal.state).items():
        record["solver_" + name] = "None" if value is None else value
    if thermal.DuDt is not None:
        for name, value in asdict(thermal.DuDt.state).items():
            if name != "psi_star_var_names":
                record["history_" + name] = "None" if value is None else value
    return record


if params.uw_phase == "resume":
    orchestration_model.load_state("checkpoint.h5")
    restored = capture()
    failure = None
    try:
        with h5py.File(f"write_rank{uw.mpi.rank}.h5", "r") as saved:
            assert set(saved) == set(restored)
            for name, actual in restored.items():
                expected = saved[name][()]
                if isinstance(expected, bytes):
                    expected = expected.decode()
                np.testing.assert_array_equal(actual, expected, err_msg=name)
    except Exception as error:
        # Every rank must report a failed restore before peers enter a solve.
        failure = str(error)
    failures = uw.mpi.comm.allgather(failure)
    assert not any(failures), failures
    uw.pprint(f"SUPG_RESTORE_EXACT method={params.uw_method} ranks={uw.mpi.size}")

end_step = 5 if params.uw_phase == "write" else 12
for step in range(orchestration_model.tracker.step, end_step):
    dt = (0.002, 0.003, 0.0015, 0.0025)[step % 4]
    velocity.array[:, 0, 0] = 0.2 * (1.0 + 0.1 * np.sin(step))
    thermal.solve(timestep=dt)
    orchestration_model.tracker.step = step + 1
    orchestration_model.tracker.time += dt
    orchestration_model.tracker.dt = dt

if params.uw_phase == "write":
    orchestration_model.save_state(file="checkpoint.h5")

record = capture()
failure = None
try:
    with h5py.File(f"{params.uw_phase}_rank{uw.mpi.rank}.h5", "w") as output:
        for name, value in record.items():
            output[name] = value
except Exception as error:
    # Rank-local test output errors must not strand peers on shutdown.
    failure = str(error)
failures = uw.mpi.comm.allgather(failure)
assert not any(failures), failures
uw.pprint(f"SUPG_RESTART_STAGE phase={params.uw_phase} method={params.uw_method} step={end_step}")
