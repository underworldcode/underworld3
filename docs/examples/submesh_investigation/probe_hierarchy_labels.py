"""Probe: do boundary labels survive on the coarse DMs in dm_hierarchy?

Gating question for the refined-submesh-pair investigation. If the coarse
levels of the refinement hierarchy don't carry the named boundary labels
(with non-empty strata), we cannot build a solver-ready coarse companion
and the whole approach needs rethinking.

Run: pixi run -e amr-dev python -u docs/examples/submesh_investigation/probe_hierarchy_labels.py
"""

import underworld3 as uw


def dump_hierarchy_labels(mesh, label):
    print(f"\n=== {label} ===")
    hier = mesh.dm_hierarchy
    print(f"dm_hierarchy length: {len(hier)}  (index 0 = coarsest, -1 = finest)")
    bnames = [b.name for b in mesh.boundaries] if mesh.boundaries is not None else []
    print(f"boundaries enum: {bnames}")

    for lvl, dm in enumerate(hier):
        cStart, cEnd = dm.getHeightStratum(0)
        ncells = cEnd - cStart
        print(f"\n  level {lvl}: {ncells} cells")
        for b in mesh.boundaries:
            if b.name in ("Null_Boundary", "All_Boundaries"):
                continue
            lab = dm.getLabel(b.name)
            if not lab:
                print(f"    {b.name:<20} value={b.value:<5} LABEL ABSENT")
                continue
            sis = lab.getStratumIS(b.value)
            size = sis.getSize() if sis else 0
            flag = "" if size > 0 else "  <-- EMPTY"
            print(f"    {b.name:<20} value={b.value:<5} stratum size={size}{flag}")


def main():
    # 2D box (simplex, gmsh-generated boundary labels)
    box = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.25,
        refinement=2,
    )
    dump_hierarchy_labels(box, "UnstructuredSimplexBox 2D, refinement=2")

    # Annulus with an internal boundary (different boundary-construction path).
    # AnnulusInternalBoundary has no `refinement` kwarg, so build it and
    # re-wrap its DM through the Mesh constructor with refinement=2 — this is
    # exactly the DM-passthrough path coarsened_companion will rely on.
    annulus_base = uw.meshing.AnnulusInternalBoundary(
        radiusOuter=1.0,
        radiusInternal=0.7,
        radiusInner=0.5,
        cellSize=0.2,
    )
    annulus = uw.discretisation.Mesh(
        annulus_base.dm,
        boundaries=annulus_base.boundaries,
        coordinate_system_type=annulus_base.CoordinateSystemType,
        refinement=2,
        qdegree=annulus_base.qdegree,
    )
    dump_hierarchy_labels(
        annulus, "AnnulusInternalBoundary (re-wrapped DM), refinement=2"
    )

    # Spherical shell (3D, yet another path)
    shell = uw.meshing.SphericalShell(
        radiusOuter=1.0,
        radiusInner=0.5,
        cellSize=0.4,
        refinement=1,
    )
    dump_hierarchy_labels(shell, "SphericalShell 3D, refinement=1")


if __name__ == "__main__":
    main()
