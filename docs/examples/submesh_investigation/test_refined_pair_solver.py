"""Gating test: extract a coarse companion, keep labels, solve Stokes,
map the solution back to the fine parent mesh.

This is the fork in the road for the refined-submesh-pair investigation.
If Stokes won't solve on the coarse companion, or the solution can't be
mapped back, the rest of the investigation doesn't matter.

Stages (each logged):
  1. Build fine box mesh (refinement=2) -> has dm_hierarchy
  2. coarsened_companion(levels=1) -> real uw.Mesh, parent state
  3. Assert labels intact + submesh lineage
  4. Stokes on the companion (no-slip walls + buoyancy body force)
  5. Map velocity back to the fine mesh via prolongate (nested FE)
  6. Round-trip sanity: sample fine -> coarse ~= original coarse

Run:
  pixi run -e amr-dev python -u \
      docs/examples/submesh_investigation/test_refined_pair_solver.py
"""

import numpy as np
import sympy

import underworld3 as uw
from underworld3.systems import Stokes

import refined_pair_prototype as rpp


def banner(msg):
    uw.pprint(0, f"\n{'='*70}\n{msg}\n{'='*70}")


def main():
    banner("STAGE 1: build fine box mesh, refinement=2")
    fine = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.2,
        degree=2,
        qdegree=4,
        refinement=2,
    )
    cS, cE = fine.dm.getHeightStratum(0)
    uw.pprint(0, f"fine mesh: {cE - cS} cells, hierarchy len {len(fine.dm_hierarchy)}")

    banner("STAGE 2: coarsened_companion(levels=1)")
    coarse = rpp.coarsened_companion(fine, levels=1, verbose=False)
    cS, cE = coarse.dm.getHeightStratum(0)
    uw.pprint(0, f"coarse companion: {cE - cS} cells")

    banner("STAGE 3: assert labels intact + submesh lineage")
    assert coarse.parent is fine, "companion.parent should be the fine mesh"
    assert coarse in fine._registered_submeshes, "not registered with parent"
    bnames = [b.name for b in coarse.boundaries]
    uw.pprint(0, f"companion boundaries enum: {bnames}")
    for name in ("Bottom", "Top", "Left", "Right"):
        lab = coarse.dm.getLabel(name)
        assert lab, f"boundary label {name} missing on companion DM"
        sis = lab.getStratumIS(coarse.boundaries[name].value)
        size = sis.getSize() if sis else 0
        uw.pprint(0, f"  {name}: stratum size {size}")
        assert size > 0, f"boundary {name} empty on companion"
    uw.pprint(0, "lineage + labels OK")

    banner("STAGE 4: Stokes on the coarse companion")
    v = uw.discretisation.MeshVariable("Vc", coarse, coarse.dim, degree=2)
    p = uw.discretisation.MeshVariable("Pc", coarse, 1, degree=1)

    stokes = Stokes(coarse, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0

    # Buoyancy-style body force, no-slip on all four walls.
    x, y = coarse.X
    stokes.bodyforce = sympy.Matrix([0.0, -sympy.sin(sympy.pi * x)])

    for wall in ("Bottom", "Top", "Left", "Right"):
        stokes.add_dirichlet_bc((0.0, 0.0), wall)

    stokes.petsc_options["snes_type"] = "newtonls"
    stokes.petsc_options["snes_rtol"] = 1.0e-6
    stokes.petsc_options["ksp_rtol"] = 1.0e-8
    stokes.petsc_options["pc_type"] = "lu"

    stokes.solve(verbose=False)

    vmag = np.linalg.norm(v.data, axis=1)
    uw.pprint(
        0,
        f"coarse solve done: |v| min={vmag.min():.3e} "
        f"max={vmag.max():.3e} mean={vmag.mean():.3e}",
    )
    assert np.all(np.isfinite(v.data)), "non-finite velocity on coarse solve"
    assert vmag.max() > 1.0e-6, "coarse velocity is trivially zero"

    banner("STAGE 5: map velocity back to the fine mesh (prolongate)")
    v_fine = uw.discretisation.MeshVariable("Vf", fine, fine.dim, degree=2)
    rpp.prolongate(coarse, v, v_fine)

    vf = v_fine.data
    vfmag = np.linalg.norm(vf, axis=1)
    uw.pprint(
        0,
        f"prolongated to fine: |v| min={vfmag.min():.3e} "
        f"max={vfmag.max():.3e} mean={vfmag.mean():.3e}",
    )
    assert np.all(np.isfinite(vf)), "non-finite velocity after prolongate"
    assert vfmag.max() > 1.0e-6, "prolongated velocity trivially zero"
    # Prolongated max should be close to the coarse max (nested FE does not
    # overshoot for these smooth fields).
    rel = abs(vfmag.max() - vmag.max()) / vmag.max()
    uw.pprint(0, f"max|v| coarse vs fine relative diff: {rel:.3e}")

    banner("STAGE 6: round-trip sanity (sample fine -> coarse)")
    v_back = uw.discretisation.MeshVariable("Vb", coarse, coarse.dim, degree=2)
    rpp.sample(coarse, v_fine, v_back)
    err = np.linalg.norm(v_back.data - v.data) / np.linalg.norm(v.data)
    uw.pprint(0, f"round-trip relative error (sample o prolongate): {err:.3e}")
    assert err < 1.0e-8, (
        f"round-trip via sample should recover coarse exactly, got {err:.3e}"
    )

    banner("ALL STAGES PASSED")


if __name__ == "__main__":
    main()
