"""The generated C must not depend on the Python hash seed.

Every rank of an MPI job generates the C source independently and `getext`
allgathers a hash of it, so anything that leaks Python's per-process hash
randomisation into the emitted code aborts the run:

    RuntimeError: JIT C-source hash differs across MPI ranks: {...}

Two constants that share a display name are the way in. `Symbol.sort_key()` is
derived from the NAME, so two same-named `constants[]` placeholders sort equal
and term order inside an `Add` falls back to hash order.

**This is deliberately not an MPI test.** Whether two ranks happen to disagree
depends on their seeds, so an mpirun-based check passes or fails at random —
measured on the bug: 4 of 6 launches diverged on one occasion and 0 of 6 on
another, with the defect present and confirmed both times. A seed sweep in
subprocesses is the same property, tested deterministically. With the bug:

    seed 0 -> a6cb2e83    seed 2 -> df7e63dd
    seed 1 -> a6cb2e83    seed 3 -> 2f8aadf7

Three distinct modules from one model. Fixed, all seeds give one hash.

The in-process halves — distinct `sort_key`, distinct identity, canonical
ordering — are pinned in `test_0103_jit_rampable_constants.py`, which is where
to look first when this fails.
"""

import os
import pathlib
import subprocess
import sys
import textwrap

import pytest

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

# A two-material Stokes: both ViscousFlowModels name their viscosity \eta, so
# the residual carries two same-named constants.
_CHILD = textwrap.dedent(
    """
    import pathlib, sys
    import numpy as np, sympy, underworld3 as uw

    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.25, qdegree=2, regular=True)
    v = uw.discretisation.MeshVariable("vs", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("ps", mesh, 1, degree=1)

    swarm = uw.swarm.Swarm(mesh)
    material = uw.swarm.IndexSwarmVariable("Ms", swarm, indices=2, proxy_degree=1)
    swarm.populate(fill_param=2)
    coords = np.asarray(swarm._particle_coordinates.data)
    with uw.synchronised_array_update():
        material.data[:, 0] = (coords[:, 1] > 0.5).astype(int)

    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    lower = uw.constitutive_models.ViscousFlowModel(stokes.Unknowns, material_name="lo")
    lower.Parameters.shear_viscosity_0 = 1.0
    upper = uw.constitutive_models.ViscousFlowModel(stokes.Unknowns, material_name="up")
    upper.Parameters.shear_viscosity_0 = 1000.0
    stokes.constitutive_model = uw.MultiMaterialConstitutiveModel(
        stokes.Unknowns, material, [lower, upper])
    stokes.add_dirichlet_bc((1.0, 0.0), "Top")
    stokes.add_dirichlet_bc((0.0, 0.0), "Bottom")
    stokes._build()

    # The compiled module is named for the hash of the canonical C source.
    cache = pathlib.Path(sys.argv[1])
    names = sorted({q.name.split(".")[0] for q in cache.glob("*.so")})
    print("MODULES:" + ",".join(names))
    """
)


def _module_hash(seed, cache_dir, tmp_path):
    child = tmp_path / "child.py"
    child.write_text(_CHILD)
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(seed)
    env["UW_JIT_CACHE_DIR"] = str(cache_dir)
    result = subprocess.run(
        [sys.executable, str(child), str(cache_dir)],
        capture_output=True, text=True, env=env, timeout=900,
    )
    line = [ln for ln in result.stdout.splitlines() if ln.startswith("MODULES:")]
    assert line, (
        f"child failed under PYTHONHASHSEED={seed}\n"
        f"stdout tail:\n{result.stdout[-2000:]}\nstderr tail:\n{result.stderr[-2000:]}"
    )
    return line[0].removeprefix("MODULES:")


@pytest.mark.parametrize("seeds", [(0, 1, 2)])
def test_the_generated_module_is_the_same_under_every_hash_seed(seeds, tmp_path):
    hashes = {}
    for seed in seeds:
        cache = tmp_path / f"cache_{seed}"      # a cold cache per seed
        cache.mkdir()
        hashes[seed] = _module_hash(seed, cache, tmp_path)

    distinct = set(hashes.values())
    assert len(distinct) == 1, (
        "the emitted C depends on the Python hash seed, so MPI ranks will "
        f"disagree and getext will abort: {hashes}"
    )
