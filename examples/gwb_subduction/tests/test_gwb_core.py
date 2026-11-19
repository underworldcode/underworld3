"""Offline regression tests for the GWB to Underworld3 subduction example.

These tests do not import UW3 or require an installed GWB binary. They cover
model transformation, geometry, composition mapping, unit conventions, and
GWB's process/file interface using a small fake standalone executable.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

EXAMPLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE))

from gwb_core import (  # noqa: E402
    COLUMNS,
    MATERIAL_NAMES,
    assign_interface,
    assign_materials,
    boundary_check,
    make_model,
    make_slab_top,
    query_gwb,
    strain_rms,
)


@pytest.fixture
def model():
    """Load the archived four-material reference GWB model."""
    return json.loads(
        (EXAMPLE / "four_material_subduction.wb").read_text(
            encoding="utf-8"
        )
    )


def test_reference_model_structure(model):
    """Require the same four composition classes used in the demonstration."""
    features = {feature["name"]: feature for feature in model["features"]}
    assert {"mantle", "Overriding", "Subducting", "Slab"} == set(features)
    assert len(MATERIAL_NAMES) == 4
    assert features["Overriding"]["composition models"][0]["compositions"] == [3]
    assert features["Subducting"]["composition models"][0]["compositions"] == [1]
    assert features["Slab"]["composition models"][0]["compositions"] == [2]


def test_make_model_reconstructs_reference_without_temperature_changes(tmp_path, model):
    """Only intended composition assignments differ from a cookbook fixture."""
    original = copy.deepcopy(model)
    features = {feature["name"]: feature for feature in original["features"]}
    features["Subducting"]["composition models"][0]["compositions"] = [0]
    features["Slab"]["composition models"][0]["compositions"] = [0]
    del features["Overriding"]["composition models"]
    source = tmp_path / "original.wb"
    target = tmp_path / "generated.wb"
    source.write_text(json.dumps(original), encoding="utf-8")

    computed, output = make_model(source, target)
    assert computed == model
    assert output == target
    assert json.loads(target.read_text(encoding="utf-8")) == model
    assert json.loads(source.read_text(encoding="utf-8")) == original


def test_make_model_never_overwrites_input(tmp_path, model):
    """Prevent accidental destruction of the input cookbook."""
    path = tmp_path / "original.wb"
    path.write_text(json.dumps(model), encoding="utf-8")
    with pytest.raises(ValueError, match="overwrite"):
        make_model(path, path)


def test_make_model_refuses_already_modified_cookbook(tmp_path, model):
    """A previously generated model cannot silently masquerade as upstream."""
    source = tmp_path / "modified.wb"
    source.write_text(json.dumps(model), encoding="utf-8")
    with pytest.raises(RuntimeError, match="unmodified"):
        make_model(source, tmp_path / "output.wb")


def test_slab_profile_matches_archived_geometry(tmp_path, model):
    """Check the trench, cutoff depth, and integrated arc length."""
    path = tmp_path / "slab.csv"
    profile, trench, arc_length = make_slab_top(model, path)
    assert trench == pytest.approx(3500.0)
    assert arc_length == pytest.approx(410.7932503187131, abs=1e-9)
    assert profile[0, 0] == pytest.approx(3500.0)
    assert profile[0, 1] == pytest.approx(0.0)
    assert profile[-1, 1] == pytest.approx(200.0)
    assert np.all(np.diff(profile[:, 1]) >= 0)
    assert np.max(np.linalg.norm(np.diff(profile, axis=0), axis=1)) <= 0.251
    assert path.is_file()


def test_slab_profile_rejects_missing_feature(model):
    """Require the slab feature rather than guessing a slab shape."""
    model["features"] = [
        feature for feature in model["features"]
        if feature["name"] != "Slab"
    ]
    with pytest.raises(RuntimeError, match="no Slab"):
        make_slab_top(model)


def test_material_mapping_with_gwb_v110_offset():
    """The 2D GWB columns c2 to c4 map to the intended material labels."""
    fields = {
        "c2": np.array([0., 1., 0., 0.]),
        "c3": np.array([0., 0., 1., 0.]),
        "c4": np.array([0., 0., 0., 1.]),
    }
    np.testing.assert_array_equal(assign_materials(fields), [0, 1, 2, 3])


def test_material_mapping_rejects_overlap():
    """Overlapping composition masks must not silently overwrite a label."""
    fields = {
        "c2": np.array([0., 1., 0., 0.]),
        "c3": np.array([0., 1., 1., 0.]),
        "c4": np.array([0., 0., 0., 1.]),
    }
    with pytest.raises(RuntimeError, match="Overlapping"):
        assign_materials(fields)


def test_material_mapping_requires_all_four_materials():
    """An absent class is an invalid GWB transfer for this example."""
    fields = {
        "c2": np.array([0., 1., 0., 0.]),
        "c3": np.array([0., 0., 1., 0.]),
        "c4": np.zeros(4),
    }
    with pytest.raises(RuntimeError, match="material is missing"):
        assign_materials(fields)


def test_weak_interface_tags_exact_profile_points(model):
    """Particles exactly on a shallow slab top must lie in the weak band."""
    profile, _, _ = make_slab_top(model)
    sample = profile[10:30]
    coords = np.column_stack(
        (sample[:, 0] / 1000.0, 1.6 - sample[:, 1] / 1000.0)
    )
    weak = assign_interface(coords, profile)
    assert weak.dtype == np.int32
    np.testing.assert_array_equal(weak, np.ones(len(sample)))


def test_boundary_check_rejects_normal_flow():
    """A nonzero normal velocity violates the specified impermeable sides."""
    coords = np.array([[0., 0.], [8., 0.], [0., 1.6], [8., 1.6]])
    values = np.zeros((len(coords), 2))
    assert max(boundary_check(coords, values).values()) == 0.0
    values[0, 0] = 2.0e-5
    with pytest.raises(RuntimeError, match="Velocity boundary"):
        boundary_check(coords, values)


def test_linear_velocity_has_correct_rms_strain():
    """For u=(2x,-2y), the strain proxy is 2 / length_scale_m."""
    nx, ny = 2 * 8, 2 * 16
    xs = np.linspace(0, 8, nx + 1)
    ys = np.linspace(0, 1.6, ny + 1)
    xx, yy = np.meshgrid(xs, ys)
    coords = np.column_stack((xx.ravel(), yy.ravel()))
    velocities = np.column_stack((2 * coords[:, 0], -2 * coords[:, 1]))
    profile = np.column_stack((np.linspace(0, 8000, 161), np.full(161, 150.0)))
    measured = strain_rms(
        coords, velocities, profile, speed_scale_mps=1.0,
        fine_res=(8, 16),
    )
    assert measured["band_nodes"] >= 10
    assert measured["collar_nodes"] >= 10
    assert measured["band_rms_per_s"] == pytest.approx(2e-6, rel=1e-10)
    assert measured["collar_rms_per_s"] == pytest.approx(2e-6, rel=1e-10)


def test_gwb_query_file_contract_with_fake_binary(tmp_path):
    """A stub executable verifies the 12-column GWB process/file contract."""
    fake = tmp_path / "fake_gwb.py"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "import numpy as np\n"
        "points = np.loadtxt(sys.argv[2], comments='#', ndmin=2)\n"
        "print('# x z d T vx vz c0 c1 c2 c3 c4 tag')\n"
        "for x, z, d in points:\n"
        "    print(x,z,d,1573,0,0,0,0,0,0,0,0)\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    model = EXAMPLE / "four_material_subduction.wb"
    coords = np.array([[1.0, 1.6], [2.0, 1.5]])
    fields = query_gwb(
        coords, model, "test", fake, tmp_path / "out",
    )
    assert set(fields) == set(COLUMNS)
    np.testing.assert_allclose(fields["x"], [1e6, 2e6])
    np.testing.assert_allclose(fields["d"], [0.0, 1e5])
    np.testing.assert_allclose(fields["T"], [1573.0, 1573.0])


def test_gwb_query_rejects_nonfinite_coordinates(tmp_path):
    """Invalid numerical coordinates must fail before an external call."""
    with pytest.raises(ValueError, match="Non-finite"):
        query_gwb(
            np.array([[float("nan"), 0.0]]),
            tmp_path / "model.wb", "test", tmp_path / "gwb-dat", tmp_path,
        )


def test_reference_verification_is_scientifically_consistent():
    """Check archived model counts, convergence, and the comparison contract."""
    path = EXAMPLE / "reference_verification.json"
    results = json.loads(path.read_text(encoding="utf-8"))
    assert results["gwb_version"] == "v1.1.0"
    assert results["temperature_dofs"] == 13041
    assert results["swarm_particles"] == 115200
    assert sum(results["material_counts"].values()) == 115200
    assert results["temperature_preservation_max_error_K"] == 0.0
    assert results["tag_preservation_max_error"] == 0.0
    assert results["material_mask_mismatches"] == 0
    assert results["weak_particles"] > 0
    assert results["cases"]["coupled"]["converged"]
    assert results["cases"]["weak_interface"]["converged"]
    assert results["cases"]["coupled"]["velocity_pc"] == "custom-FMG"
    assert results["cases"]["weak_interface"]["velocity_pc"] == "custom-FMG"
    assert results["cases"]["weak_interface"]["maximum_speed_cm_year"] > (
        results["cases"]["coupled"]["maximum_speed_cm_year"]
    )
