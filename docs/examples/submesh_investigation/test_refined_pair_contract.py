"""Contract test: refine-DM mode only.

The coarse/fine companion is available ONLY when a genuine nested
refinement hierarchy exists. With no refinement relationship the
operation must fail loudly -- no geometric or KDTree fallback.

  1. Mesh built WITHOUT refinement -> coarsened_companion raises.
  2. Mesh built WITH refinement   -> companion works, and the
     transfer is genuinely the nested path (proved by the gating test,
     which hits machine precision with -dm_plex_hash_location absent;
     a silent General fallback would have errored on the missing hash
     option).

Run:
  pixi run -e amr-dev python -u \
      docs/examples/submesh_investigation/test_refined_pair_contract.py
"""

import underworld3 as uw

import refined_pair_prototype as rpp


def banner(msg):
    uw.pprint(0, f"\n{'='*70}\n{msg}\n{'='*70}")


def main():
    banner("CONTRACT 1: no refinement relationship -> unavailable")
    flat = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.2,
    )
    uw.pprint(0, f"flat mesh dm_hierarchy length: {len(flat.dm_hierarchy)}")
    raised = False
    try:
        rpp.coarsened_companion(flat, levels=1)
    except ValueError as e:
        raised = True
        msg = str(e)
        uw.pprint(0, f"raised ValueError as required:\n  {msg}")
        assert "refinement" in msg.lower(), "error must cite refinement"
        assert "fallback" in msg.lower(), "error must state no fallback"
    assert raised, "coarsened_companion must raise on a non-refined mesh"

    banner("CONTRACT 2: refinement present -> companion available")
    fine = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.2,
        degree=2,
        qdegree=4,
        refinement=2,
    )
    uw.pprint(0, f"fine mesh dm_hierarchy length: {len(fine.dm_hierarchy)}")
    coarse = rpp.coarsened_companion(fine, levels=1)
    assert coarse.parent is fine
    assert coarse in fine._registered_submeshes
    cS, cE = coarse.dm.getHeightStratum(0)
    uw.pprint(0, f"companion available: {cE - cS} cells, parent linked")

    # Asking for more levels than exist must also fail loudly.
    raised = False
    try:
        rpp.coarsened_companion(fine, levels=99)
    except ValueError as e:
        raised = True
        uw.pprint(0, f"deep request correctly refused: {e}")
    assert raised, "request beyond hierarchy depth must raise"

    banner("CONTRACT TEST PASSED")


if __name__ == "__main__":
    main()
