"""
MPI tests for submesh extraction and data transfer.

Tests that extract_region, restrict, prolongate, copy_into,
and coordinate sync all work correctly in parallel.
"""

import numpy as np
import pytest
import underworld3 as uw

pytestmark = [
    pytest.mark.level_2,
    pytest.mark.tier_b,
    pytest.mark.mpi(min_size=2),
    pytest.mark.timeout(120),
]


def _make_meshes():
    full = uw.meshing.AnnulusInternalBoundary(
        radiusOuter=1.5,
        radiusInternal=1.0,
        radiusInner=0.5,
        cellSize=1.0 / 8.0,
    )
    rock = full.extract_region("Inner")
    return full, rock


# ------------------------------------------------------------------
# 1. extract_region produces a valid submesh in parallel
# ------------------------------------------------------------------

def test_extract_region_parallel():
    full, rock = _make_meshes()

    # Submesh should exist and have correct dimension
    assert rock.dm.getDimension() == 2
    assert rock.parent is full
    assert rock.subpoint_is is not None

    # All submesh vertex radii should be in [r_inner, r_internal]
    coords = rock.X.coords
    r = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)
    assert r.min() >= 0.5 - 1e-10, f"r_min={r.min()} < 0.5"
    assert r.max() <= 1.0 + 1e-10, f"r_max={r.max()} > 1.0"


# ------------------------------------------------------------------
# 2. restrict: parent → submesh data transfer
# ------------------------------------------------------------------

def test_restrict_parallel():
    full, rock = _make_meshes()

    v_full = uw.discretisation.MeshVariable("Vf", full, full.dim, degree=2)
    v_rock = uw.discretisation.MeshVariable("Vr", rock, rock.dim, degree=2)

    # Set parent to a known function of coordinates
    r_f = np.sqrt(v_full.coords[:, 0] ** 2 + v_full.coords[:, 1] ** 2)
    v_full.data[:, 0] = r_f
    v_full.data[:, 1] = -r_f

    rock.restrict(v_full, v_rock)

    # Check submesh values match the function at submesh coordinates
    r_r = np.sqrt(v_rock.coords[:, 0] ** 2 + v_rock.coords[:, 1] ** 2)
    err = np.abs(v_rock.data[:, 0] - r_r).max()

    # Gather max error across ranks
    from mpi4py import MPI

    global_err = MPI.COMM_WORLD.allreduce(err, op=MPI.MAX)
    assert global_err < 1e-10, f"restrict error: {global_err}"


# ------------------------------------------------------------------
# 3. prolongate: submesh → parent data transfer
# ------------------------------------------------------------------

def test_prolongate_parallel():
    full, rock = _make_meshes()

    v_full = uw.discretisation.MeshVariable("Vf", full, full.dim, degree=2)
    v_rock = uw.discretisation.MeshVariable("Vr", rock, rock.dim, degree=2)

    # Set submesh to known function
    r_r = np.sqrt(v_rock.coords[:, 0] ** 2 + v_rock.coords[:, 1] ** 2)
    v_rock.data[:, 0] = r_r
    v_rock.data[:, 1] = -r_r

    # Clear parent and prolongate
    v_full.data[:] = 0.0
    rock.prolongate(v_rock, v_full)

    # Check: rock-region DOFs should be set, air DOFs should be zero
    r_f = np.sqrt(v_full.coords[:, 0] ** 2 + v_full.coords[:, 1] ** 2)
    rock_mask = r_f < 1.0 + 1e-6
    air_mask = ~rock_mask

    if rock_mask.any():
        rock_err = np.abs(v_full.data[rock_mask, 0] - r_f[rock_mask]).max()
    else:
        rock_err = 0.0

    if air_mask.any():
        air_max = np.abs(v_full.data[air_mask]).max()
    else:
        air_max = 0.0

    from mpi4py import MPI

    global_rock_err = MPI.COMM_WORLD.allreduce(rock_err, op=MPI.MAX)
    global_air_max = MPI.COMM_WORLD.allreduce(air_max, op=MPI.MAX)

    assert global_rock_err < 1e-10, f"prolongate rock error: {global_rock_err}"
    assert global_air_max < 1e-10, f"prolongate air leakage: {global_air_max}"


# ------------------------------------------------------------------
# 4. copy_into works in both directions
# ------------------------------------------------------------------

def test_copy_into_parallel():
    full, rock = _make_meshes()

    v_full = uw.discretisation.MeshVariable("Vf", full, full.dim, degree=2)
    v_rock = uw.discretisation.MeshVariable("Vr", rock, rock.dim, degree=2)

    # Parent → submesh
    r_f = np.sqrt(v_full.coords[:, 0] ** 2 + v_full.coords[:, 1] ** 2)
    v_full.data[:, 0] = r_f

    v_full.copy_into(v_rock)

    r_r = np.sqrt(v_rock.coords[:, 0] ** 2 + v_rock.coords[:, 1] ** 2)
    err1 = np.abs(v_rock.data[:, 0] - r_r).max()

    # Submesh → parent
    v_full.data[:] = 0.0
    v_rock.copy_into(v_full)

    rock_mask = r_f < 1.0 + 1e-6
    if rock_mask.any():
        err2 = np.abs(v_full.data[rock_mask, 0] - r_f[rock_mask]).max()
    else:
        err2 = 0.0

    from mpi4py import MPI

    global_err1 = MPI.COMM_WORLD.allreduce(err1, op=MPI.MAX)
    global_err2 = MPI.COMM_WORLD.allreduce(err2, op=MPI.MAX)

    assert global_err1 < 1e-10, f"copy_into restrict error: {global_err1}"
    assert global_err2 < 1e-10, f"copy_into prolongate error: {global_err2}"


# ------------------------------------------------------------------
# 5. Stokes solve on extracted submesh
# ------------------------------------------------------------------

def test_stokes_on_submesh_parallel():
    import sympy

    full, rock = _make_meshes()

    v = uw.discretisation.MeshVariable("V", rock, rock.dim, degree=2)
    p = uw.discretisation.MeshVariable("P", rock, 1, degree=1, continuous=True)

    r, th = rock.CoordinateSystem.xR
    G_N = rock.Gamma_N

    stokes = uw.systems.Stokes(rock, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.bodyforce = sympy.cos(2 * th) * (-rock.CoordinateSystem.unit_e_0)
    stokes.add_natural_bc(1e4 * G_N.dot(v.sym) * G_N, "Internal")
    stokes.add_natural_bc(1e4 * G_N.dot(v.sym) * G_N, "Lower")
    stokes.tolerance = 1e-4
    stokes.petsc_options["snes_type"] = "newtonls"
    stokes.petsc_options["ksp_type"] = "fgmres"
    stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_type", "kaskade")
    stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"

    stokes.solve(verbose=False)

    vmag = np.sqrt(v.data[:, 0] ** 2 + v.data[:, 1] ** 2)

    from mpi4py import MPI

    global_max = MPI.COMM_WORLD.allreduce(vmag.max(), op=MPI.MAX)

    # Solution should be non-trivial
    assert global_max > 1e-6, f"Stokes solution is zero: max|v|={global_max}"
    # And bounded
    assert global_max < 1.0, f"Stokes solution unbounded: max|v|={global_max}"


# ------------------------------------------------------------------
# 6. Expression safety check works in parallel
# ------------------------------------------------------------------

def test_mixed_mesh_error_parallel():
    full, rock = _make_meshes()

    v_rock = uw.discretisation.MeshVariable("Vr", rock, rock.dim, degree=2)
    T_full = uw.discretisation.MeshVariable("Tf", full, 1, degree=1)

    # This should raise ValueError, not a JIT error
    with pytest.raises(ValueError, match="foreign mesh"):
        stokes = uw.systems.Stokes(
            rock,
            velocityField=v_rock,
            pressureField=uw.discretisation.MeshVariable("Pr", rock, 1, degree=1),
        )
        stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
        stokes.bodyforce = T_full.sym * rock.CoordinateSystem.unit_e_0
        stokes.solve(verbose=False)
