#!/usr/bin/env python3
"""Reproducible GWB v1.1.0 to Underworld3 subduction demonstration.

Generates a four-material GWB model from the upstream cookbook,
reconstructs the slab top, initializes UW3 fields and particles,
and compares coupled and weakened interface Stokes solutions.

Fixed geometry and illustrative material properties only.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import sympy

import underworld3 as uw
from underworld3.utilities.custom_mg import set_custom_fmg

from gwb_core import (
    MATERIAL_NAMES,
    assign_interface,
    assign_materials,
    boundary_check,
    make_model,
    make_slab_top,
    query_gwb,
    strain_rms,
)
from gwb_plots import make_figures


EXAMPLE_DIR = Path(__file__).resolve().parent

# The GWB checkout and standalone executable are optional UW3 dependencies.
# The validated demonstration uses the GWB v1.1.0 cookbook and gwb-dat.
GWB_ROOT = Path(
    os.environ.get("GWB_ROOT", str(Path.home() / "WorldBuilder"))
).expanduser()
SOURCE = Path(
    os.environ.get(
        "GWB_SUBDUCTION_SOURCE",
        str(
            GWB_ROOT
            / "cookbooks"
            / "simple_subduction_2d_cartesian"
            / "simple_subduction_2d_cartesian.wb"
        ),
    )
).expanduser()
EXECUTABLE = Path(
    os.environ.get(
        "GWB_DAT", str(GWB_ROOT / "build-uw3py" / "bin" / "gwb-dat")
    )
).expanduser()
OUT = Path(
    os.environ.get(
        "GWB_UW3_OUTPUT_DIR", str(Path.cwd() / "outputs" / "gwb_subduction")
    )
).expanduser()

WIDTH = 8.0
HEIGHT = 1.6

BASE_RES = (80, 40)
FINE_RES = (160, 80)

L_REF = 1.0e6
ETA_REF = 1.0e21
RHO_REF = 3300.0
GRAVITY = 9.81
ALPHA = 3.0e-5
DELTA_T = 1300.0
T_REF = 1573.0

SLAB_DENSITY_ANOMALY = 50.0
YEAR_SECONDS = 365.25 * 86400.0

INTERFACE_WIDTH_KM = 40.0
INTERFACE_MAX_DEPTH_KM = 200.0

WEAK_VISCOSITY = 0.1
BULK_VISCOSITIES = (1.0, 5.0, 10.0, 5.0)



def solve(
    mesh,
    temperature,
    material,
    interface,
    name,
    weakened,
    density_ratio,
):
    """Solve an instantaneous two-dimensional Stokes model.

    Assign four bulk viscosities from particle material labels. The weak
    case replaces viscosity inside the separate slab-top interface mask.
    The reference case leaves the geological labels unchanged. Both use
    rotated free-slip on left, right, and top, bottom no-slip, and a
    validated custom FMG velocity-block preconditioner.

    Returns
    -------
    tuple
        Cartesian velocity-node coordinates, nondimensional nodal velocities,
        and convergence/boundary diagnostics. A failure raises immediately.
    """
    velocity = uw.discretisation.MeshVariable(
        f"U_{name}",
        mesh,
        2,
        degree=2,
    )

    pressure = uw.discretisation.MeshVariable(
        f"P_{name}",
        mesh,
        1,
        degree=1,
    )

    stokes = uw.systems.Stokes(
        mesh,
        velocityField=velocity,
        pressureField=pressure,
    )

    stokes.constitutive_model = (
        uw.constitutive_models.ViscousFlowModel
    )

    viscosity = material.createMask(
        list(BULK_VISCOSITIES)
    )

    if weakened:
        fraction = interface.sym[1]

        viscosity = (
            (1 - fraction) * viscosity
            + fraction * WEAK_VISCOSITY
        )

    stokes.constitutive_model.Parameters.shear_viscosity_0 = (
        viscosity
    )

    stokes.bodyforce = sympy.Matrix(
        [
            0,
            temperature.sym[0]
            - density_ratio * material.sym[2],
        ]
    )

    stokes.saddle_preconditioner = 1.0
    stokes.preconditioner = "fmg"
    stokes.rtol = 1.0e-5

    for boundary in (
        "Left",
        "Right",
        "Top",
    ):
        stokes.add_rotated_freeslip_bc(
            0,
            boundary,
        )

    stokes.add_dirichlet_bc(
        (0.0, 0.0),
        "Bottom",
    )

    coarse = uw.meshing.StructuredQuadBox(
        elementRes=BASE_RES,
        minCoords=(0.0, 0.0),
        maxCoords=(WIDTH, HEIGHT),
        degree=1,
    )

    set_custom_fmg(
        stokes,
        [coarse],
        field_id=0,
    )

    print(
        f"Solving {name} with rotated free-slip and custom FMG",
        flush=True,
    )

    stokes.solve(
        zero_init_guess=True
    )

    info = getattr(
        stokes,
        "_rotated_freeslip_info",
        None,
    )

    if (
        not isinstance(info, dict)
        or not info.get("converged")
    ):
        raise RuntimeError(
            f"Rotated solver did not converge: {info}"
        )

    if int(info.get("ksp_reason") or 0) <= 0:
        raise RuntimeError(
            f"Rotated KSP did not converge: {info}"
        )

    if (
        info.get("velocity_pc"),
        info.get("velocity_pc_type"),
    ) != (
        "custom-FMG",
        "mg",
    ):
        raise RuntimeError(
            f"Incorrect rotated velocity preconditioner: {info}"
        )

    if stokes.pc_fallbacks:
        raise RuntimeError(
            f"Unexpected solver fallback: {stokes.pc_fallbacks}"
        )

    coords = np.asarray(
        velocity.coords,
        dtype=float,
    ).copy()

    values = np.asarray(
        velocity.array,
        dtype=float,
    ).reshape(-1, 2).copy()

    if not np.isfinite(values).all():
        raise RuntimeError(
            "Non-finite velocity"
        )

    checks = boundary_check(
        coords, values, width=WIDTH, height=HEIGHT
    )

    return coords, values, {
        "converged": True,
        "ksp_reason": int(info["ksp_reason"]),
        "residual": float(info["rnorm"]),
        "velocity_pc": str(info["velocity_pc"]),
        "velocity_pc_type": str(info["velocity_pc_type"]),
        "boundary": checks,
    }






def main():
    """Execute and verify the full GWB v1.1.0 to Underworld3 workflow.

    The workflow is deliberately single-rank, fixed-geometry, and
    instantaneous. It is not a mesh-converged subduction calculation.
    Output is written to GWB_UW3_OUTPUT_DIR (default: outputs/gwb_subduction).
    """
    if uw.mpi.size != 1:
        raise RuntimeError(
            "Use one MPI process for this demonstration"
        )

    for path in (SOURCE, EXECUTABLE):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing GWB prerequisite: {path}. "
                "Set GWB_ROOT or the GWB_SUBDUCTION_SOURCE / GWB_DAT variables."
            )

    head_commit = subprocess.check_output(
        [
            "git",
            "-C",
            str(GWB_ROOT),
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()

    tag_commit = subprocess.check_output(
        [
            "git",
            "-C",
            str(GWB_ROOT),
            "rev-parse",
            "v1.1.0^{commit}",
        ],
        text=True,
    ).strip()

    if head_commit != tag_commit:
        raise RuntimeError(
            "GWB checkout is not at release v1.1.0"
        )

    version = "v1.1.0"

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("GWB TO UNDERWORLD3 SUBDUCTION WORKFLOW")
    print("=" * 65)
    print("GWB version:", version)

    model, local_model = make_model(
        SOURCE, OUT / "four_material_subduction.wb"
    )

    reference_model = EXAMPLE_DIR / "four_material_subduction.wb"
    if reference_model.is_file():
        reference = json.loads(reference_model.read_text(encoding="utf-8"))
        if model != reference:
            raise RuntimeError(
                "Generated GWB input differs from the validated four-material "
                "reference. Check that the original cookbook is from v1.1.0."
            )

    profile, trench, arc_length = make_slab_top(
        model, OUT / "slab_top_km.csv", INTERFACE_MAX_DEPTH_KM
    )

    print("Trench:", trench, "km")
    print(
        "Weak interface arc:",
        round(arc_length, 2),
        "km",
    )

    mesh = uw.meshing.StructuredQuadBox(
        elementRes=BASE_RES,
        minCoords=(0.0, 0.0),
        maxCoords=(WIDTH, HEIGHT),
        degree=1,
        refinement=1,
    )

    hierarchy = getattr(
        mesh,
        "dm_hierarchy",
        None,
    )

    if hierarchy is None or len(hierarchy) < 2:
        raise RuntimeError(
            "Missing geometric mesh hierarchy"
        )

    temperature = uw.discretisation.MeshVariable(
        "T_demo",
        mesh,
        1,
        degree=1,
    )

    swarm = uw.swarm.Swarm(
        mesh=mesh
    )

    material = uw.swarm.IndexSwarmVariable(
        "Material_demo",
        swarm,
        indices=4,
    )

    interface = uw.swarm.IndexSwarmVariable(
        "Weak_demo",
        swarm,
        indices=2,
    )

    swarm.populate(
        fill_param=2
    )

    mesh_coords = np.asarray(
        temperature.coords,
        dtype=float,
    ).copy()

    particle_coords = np.asarray(
        swarm.coords,
        dtype=float,
    ).copy()

    print()
    print("Querying GWB at UW3 temperature DOFs")

    new_t = query_gwb(
        mesh_coords,
        local_model,
        "temperature_four_material",
        EXECUTABLE, OUT, HEIGHT, L_REF,
    )

    old_t = query_gwb(
        mesh_coords,
        SOURCE,
        "temperature_original",
        EXECUTABLE, OUT, HEIGHT, L_REF,
    )

    print("Querying GWB at UW3 particle positions")

    fields = query_gwb(
        particle_coords,
        local_model,
        "particles_four_material",
        EXECUTABLE, OUT, HEIGHT, L_REF,
    )

    temperature_error = float(
        np.max(
            np.abs(
                new_t["T"] - old_t["T"]
            )
        )
    )

    tag_error = float(
        np.max(
            np.abs(
                new_t["tag"] - old_t["tag"]
            )
        )
    )

    combined_original = (
        old_t["c1"] > 0.5
    )

    combined_new = (
        (new_t["c2"] > 0.5)
        | (new_t["c3"] > 0.5)
    )

    mask_error = int(
        np.count_nonzero(
            combined_original != combined_new
        )
    )

    if (
        temperature_error > 1e-5
        or tag_error > 1e-8
        or mask_error
    ):
        raise RuntimeError(
            "Modified GWB model did not preserve original fields"
        )

    labels = assign_materials(
        fields
    )

    weak = assign_interface(
        particle_coords, profile, HEIGHT, INTERFACE_WIDTH_KM,
        INTERFACE_MAX_DEPTH_KM,
    )

    temperature_nd = (
        new_t["T"] - T_REF
    ) / DELTA_T

    temperature.array[:, 0, 0] = temperature_nd

    material.array[:, 0, 0] = labels
    interface.array[:, 0, 0] = weak

    material._update()
    interface._update()

    if not np.array_equal(
        np.asarray(material.array[:, 0, 0]),
        labels,
    ):
        raise RuntimeError(
            "Material swarm storage mismatch"
        )

    if not np.array_equal(
        np.asarray(interface.array[:, 0, 0]),
        weak,
    ):
        raise RuntimeError(
            "Interface swarm storage mismatch"
        )

    print()
    print("INITIAL CONDITION VALIDATION")
    print("=" * 65)

    print(
        "Temperature DOFs:",
        len(mesh_coords),
    )

    print(
        "Swarm particles:",
        len(labels),
    )

    print(
        "Temperature preservation error:",
        temperature_error,
        "K",
    )

    print(
        "GWB composition-mask mismatches:",
        mask_error,
    )

    for i, name in enumerate(MATERIAL_NAMES):
        print(
            f"{name}:",
            int(np.count_nonzero(labels == i)),
            "particles",
        )

    print(
        "Weak-zone particles:",
        int(weak.sum()),
    )

    density_ratio = (
        SLAB_DENSITY_ANOMALY
        / (RHO_REF * ALPHA * DELTA_T)
    )

    velocity_scale_mps = (
        RHO_REF
        * GRAVITY
        * ALPHA
        * DELTA_T
        * L_REF**2
        / ETA_REF
    )

    cm_per_year = (
        velocity_scale_mps
        * 100
        * YEAR_SECONDS
    )

    solutions = {}
    records = {}

    for name, softened in (
        ("coupled", False),
        ("weak_interface", True),
    ):
        coords, velocity, record = solve(
            mesh,
            temperature,
            material,
            interface,
            name,
            softened,
            density_ratio,
        )

        speed = (
            np.linalg.norm(
                velocity,
                axis=1,
            )
            * cm_per_year
        )

        record["maximum_speed_cm_year"] = float(
            speed.max()
        )

        record["strain"] = strain_rms(
            coords,
            velocity,
            profile,
            velocity_scale_mps,
            FINE_RES, WIDTH, HEIGHT, L_REF,
        )

        solutions[name] = (
            coords,
            velocity,
        )

        records[name] = record

        print()
        print(
            f"{name} maximum speed:",
            f"{speed.max():.4f} cm/year",
        )

        print(
            f"{name} band strain RMS:",
            f"{record['strain']['band_rms_per_s']:.4e} 1/s",
        )

    if not np.allclose(
        solutions["coupled"][0],
        solutions["weak_interface"][0],
    ):
        raise RuntimeError(
            "Velocity node sets do not match"
        )

    baseline = solutions["coupled"][1]
    changed = solutions["weak_interface"][1]

    relative_change = float(
        np.linalg.norm(
            changed - baseline
        )
        / np.linalg.norm(baseline)
    )

    make_figures(
        particle_coords,
        labels,
        profile,
        solutions,
        cm_per_year,
        records, OUT, WIDTH, HEIGHT,
    )

    np.savez_compressed(
        OUT / "demo_fields.npz",
        velocity_coords=solutions["coupled"][0],
        coupled_velocity=baseline,
        weak_velocity=changed,
        swarm_coords=particle_coords,
        material_labels=labels,
        weak_labels=weak,
        temperature_coords=mesh_coords,
        temperature_K=new_t["T"],
        velocity_scale_mps=velocity_scale_mps,
    )

    summary = {
        "source": str(SOURCE),
        "gwb_version": version,
        "domain_km": [8000, 1600],
        "finest_cells": list(FINE_RES),
        "temperature_dofs": len(mesh_coords),
        "swarm_particles": len(labels),
        "material_counts": {
            name: int(np.count_nonzero(labels == i))
            for i, name in enumerate(MATERIAL_NAMES)
        },
        "weak_particles": int(weak.sum()),
        "interface_width_km": INTERFACE_WIDTH_KM,
        "interface_depth_limit_km": (
            INTERFACE_MAX_DEPTH_KM
        ),
        "interface_arc_length_km": arc_length,
        "temperature_preservation_max_error_K": (
            temperature_error
        ),
        "tag_preservation_max_error": tag_error,
        "material_mask_mismatches": mask_error,
        "bulk_viscosities_Pa_s": [
            value * ETA_REF
            for value in BULK_VISCOSITIES
        ],
        "weak_viscosity_Pa_s": (
            WEAK_VISCOSITY * ETA_REF
        ),
        "slab_density_anomaly_kg_m3": (
            SLAB_DENSITY_ANOMALY
        ),
        "cases": records,
        "relative_velocity_field_change": (
            relative_change
        ),
        "scope": (
            "Instantaneous numerical proof of concept, "
            "not converged or calibrated subduction"
        ),
    }

    (
        OUT / "verification.json"
    ).write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    (
        OUT / "README.md"
    ).write_text(
        "# GWB to Underworld3 subduction workflow\n\n"
        "Standalone GWB v1.1.0 queries initialize UW3 temperature, "
        "geological material labels, and a separate weak-interface "
        "mask. A rotated free-slip Stokes solver with custom FMG "
        "compares coupled and weak-interface cases.\n\n"
        f"Temperature preservation error: {temperature_error:.3g} K.\n\n"
        f"Maximum coupled velocity: "
        f"{records['coupled']['maximum_speed_cm_year']:.3f} "
        "cm/year.\n\n"
        f"Maximum weak-interface velocity: "
        f"{records['weak_interface']['maximum_speed_cm_year']:.3f} "
        "cm/year.\n\n"
        f"Relative full-field change: {relative_change:.4f}.\n\n"
        "These are controlled fixed-geometry results, not a "
        "time-evolving or mesh-converged physical subduction model. "
        "The GWB 2D composition output offset is corrected explicitly. "
        "The Python wrapper is not required.\n"
    )

    print()
    print("FINAL WORKFLOW VERIFICATION")
    print("=" * 65)

    print(
        "Relative velocity field difference:",
        relative_change,
    )

    print(
        "PASS: GWB temperature and composition preservation"
    )

    print(
        "PASS: Four geological materials transferred"
    )

    print(
        "PASS: Separate weak interface transferred"
    )

    print(
        "PASS: Both rotated Stokes systems converged"
    )

    print(
        "PASS: Custom FMG confirmed"
    )

    print(
        "PASS: Boundary constraints verified"
    )

    print(
        "PASS: Interface kinematic diagnostics saved"
    )

    print()
    print("Output folder:", OUT)
    print("WORKFLOW DEMONSTRATION COMPLETE")


if __name__ == "__main__":
    main()
