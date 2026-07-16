"""Tests for the implemented mesh coordinate-units semantics.

The model owns the unit system: a mesh's coordinate units come from the
default model's reference quantities (``model.get_coordinate_unit()``),
not from the mesh constructor. The legacy ``units=`` constructor kwarg
is deprecated and ignored — passing it raises a ``DeprecationWarning``
and the model's units win regardless.

History: this file previously documented a PROPOSED mesh-units interface
(constructor-set units, ``mesh.to_units()``, unit-carrying coordinate
arrays) behind try/except-skip guards, several of which had started to
pass vacuously. That proposal was superseded by the model-owned design
(``docs/developer/design/UNITS_SIMPLIFIED_DESIGN_2025-11.md``) and the
placeholder tests were deleted (LE-10 / WA-23, units-family ruling D7,
2026-07-06).
"""

import warnings

import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


@pytest.fixture(autouse=True)
def reset_model_state():
    """Isolate each test from reference quantities set elsewhere."""
    uw.reset_default_model()
    uw.use_strict_units(False)
    yield
    uw.reset_default_model()
    uw.use_strict_units(False)


def _box(**kwargs):
    return uw.meshing.StructuredQuadBox(
        elementRes=(4, 4), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), **kwargs
    )


class TestModelOwnedMeshUnits:
    """mesh.units reflects the model, and nothing else."""

    def test_dimensionless_by_default(self):
        mesh = _box()
        assert mesh.units is None

    def test_units_follow_model_reference_quantities(self):
        model = uw.get_default_model()
        model.set_reference_quantities(domain_depth=uw.quantity(1000, "km"))
        mesh = _box()
        assert mesh.units == model.get_coordinate_unit()
        # The coordinate unit is a length unit (a Pint Unit, not a string)
        assert dict(mesh.units.dimensionality) == {"[length]": 1}

    def test_all_meshes_share_the_model_units(self):
        model = uw.get_default_model()
        model.set_reference_quantities(domain_depth=uw.quantity(500, "m"))
        mesh_a = _box()
        mesh_b = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5
        )
        assert mesh_a.units == mesh_b.units == model.get_coordinate_unit()


class TestDeprecatedUnitsKwarg:
    """The constructor ``units=`` kwarg warns and is ignored."""

    def test_units_kwarg_warns_and_is_ignored(self):
        with pytest.warns(DeprecationWarning, match="deprecated"):
            mesh = _box(units="km")
        # No model reference quantities: the mesh stays dimensionless
        # regardless of the requested constructor units.
        assert mesh.units is None

    def test_model_units_win_over_units_kwarg(self):
        model = uw.get_default_model()
        model.set_reference_quantities(domain_depth=uw.quantity(1000, "km"))
        with pytest.warns(DeprecationWarning, match="deprecated"):
            mesh = _box(units="m")
        assert mesh.units == model.get_coordinate_unit()

    def test_no_deprecation_warning_without_units_kwarg(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _box()
        assert not any(
            "'units' mesh-constructor parameter" in str(w.message) for w in caught
        )
