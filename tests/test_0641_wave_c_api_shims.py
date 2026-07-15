"""Wave C API-harmonization contract tests (WC-01..WC-13).

Every deprecation shim introduced by the 2026-07 Wave C API harmonization
carries the two-test contract here:

1. the OLD signature/spelling produces the identical result and emits exactly
   one ``DeprecationWarning`` naming the replacement;
2. the NEW canonical signature emits no ``DeprecationWarning`` at all.

Canonical conventions under test (see ``docs/developer/UW3_STYLE_CHARTER.md``
paragraph 6 and ``docs/reviews/2026-07/API-CONSISTENCY-REVIEW.md``):

- BC argument order is value-first: ``add_<kind>_bc(conds, boundary, ...)``
  (maintainer decision D2, 2026-07-04); the datum name is ``conds`` (D3).
- ``consistent_jacobian`` is a validated property (WC-03).
- ``set_custom_fmg`` is the solver-method entry point for custom MG (WC-04).
- ``uw.quantity`` is THE quantity factory (D14, WC-06).
"""

import warnings

import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _one_deprecation(record):
    """Count the DeprecationWarnings in a pytest.warns record."""
    return sum(
        1 for w in record if issubclass(w.category, DeprecationWarning)
    )


class _no_deprecation(warnings.catch_warnings):
    """Context manager: any DeprecationWarning inside is a test failure."""

    def __enter__(self):
        log = super().__enter__()
        warnings.simplefilter("error", DeprecationWarning)
        return log


@pytest.fixture(scope="module")
def mesh():
    return uw.meshing.StructuredQuadBox(
        elementRes=(2, 2), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0)
    )


@pytest.fixture()
def stokes(mesh):
    solver = uw.systems.Stokes(mesh)
    solver.constitutive_model = uw.constitutive_models.ViscousFlowModel
    return solver


# ---------------------------------------------------------------------------
# WC-01 / WC-02 - value-first BC argument order, `conds` datum name
# ---------------------------------------------------------------------------

class TestStokesNitscheValueFirst:
    def test_new_order_no_warning(self, mesh, stokes):
        with _no_deprecation():
            stokes.add_nitsche_bc(0.5, "Top")
        assert stokes.natural_bcs[-1].boundary == "Top"

    def test_legacy_positional_order_warns_once(self, mesh, stokes):
        with pytest.warns(DeprecationWarning, match=r"add_nitsche_bc\(conds, boundary") as rec:
            stokes.add_nitsche_bc("Top", 0.5)
        assert _one_deprecation(rec) == 1

    def test_legacy_g_keyword_warns_once(self, mesh, stokes):
        with pytest.warns(DeprecationWarning, match=r"add_nitsche_bc\(conds, boundary") as rec:
            stokes.add_nitsche_bc("Top", g=0.5)
        assert _one_deprecation(rec) == 1

    def test_legacy_and_new_give_identical_bc(self, mesh, stokes):
        # Register the same condition through both spellings on ONE solver so
        # every symbol (velocity, pressure, viscosity, normals) is shared -
        # the stored BC tuples must be identical.
        with pytest.warns(DeprecationWarning):
            stokes.add_nitsche_bc("Top", 0.5)
        with _no_deprecation():
            stokes.add_nitsche_bc(0.5, "Top")

        bc_old = stokes.natural_bcs[-2]
        bc_new = stokes.natural_bcs[-1]
        assert bc_old.boundary == bc_new.boundary
        assert bc_old.fn_f == bc_new.fn_f
        assert bc_old.fn_F == bc_new.fn_F
        assert bc_old.fn_p == bc_new.fn_p

    def test_alias_only_warning_names_the_keyword(self, mesh, stokes):
        # Copilot review of #334: the warning must describe the legacy form
        # the caller actually used - a keyword-only call is not "positional".
        with pytest.warns(DeprecationWarning, match=r"the 'g=' keyword of add_nitsche_bc\(\)") as rec:
            stokes.add_nitsche_bc(boundary="Top", g=0.5)
        assert _one_deprecation(rec) == 1

    def test_positional_order_warning_names_the_order(self, mesh, stokes):
        with pytest.warns(DeprecationWarning, match=r"positional order is deprecated") as rec:
            stokes.add_nitsche_bc("Top", 0.5)
        assert _one_deprecation(rec) == 1

    def test_datum_given_twice_is_an_error(self, mesh, stokes):
        with pytest.raises(TypeError, match="as 'conds' and as 'g='"):
            stokes.add_nitsche_bc(0.5, "Top", g=0.5)

    def test_missing_boundary_is_an_error(self, mesh, stokes):
        with pytest.raises(TypeError, match="boundary"):
            stokes.add_nitsche_bc(0.5)


class TestRotatedFreeslipValueFirst:
    def test_new_order_no_warning(self, mesh, stokes):
        with _no_deprecation():
            stokes.add_rotated_freeslip_bc(0, "Top")
        assert stokes._rotated_freeslip_bcs[-1] == ("Top", None)

    def test_boundary_keyword_no_warning(self, mesh, stokes):
        with _no_deprecation():
            stokes.add_rotated_freeslip_bc(boundary="Bottom")
        assert stokes._rotated_freeslip_bcs[-1] == ("Bottom", None)

    def test_legacy_boundary_first_warns_once(self, mesh, stokes):
        with pytest.warns(DeprecationWarning, match=r"add_rotated_freeslip_bc\(conds, boundary") as rec:
            stokes.add_rotated_freeslip_bc("Top")
        assert _one_deprecation(rec) == 1
        assert stokes._rotated_freeslip_bcs[-1] == ("Top", None)

    def test_legacy_positional_normal_is_preserved(self, mesh, stokes):
        n = sympy.Matrix([[0, 1]])
        with pytest.warns(DeprecationWarning) as rec:
            stokes.add_rotated_freeslip_bc("Top", n)
        assert _one_deprecation(rec) == 1
        assert stokes._rotated_freeslip_bcs[-1] == ("Top", n)

    def test_zero_datum_accepted_in_any_numeric_form(self, mesh, stokes):
        # Regression (#336): the zero guard must compare by VALUE. sympy's
        # structural == treats Float(0.0) != Integer(0), so 0.0 was rejected
        # even though the deprecation message itself recommends conds=0.
        for zero in (0.0, sympy.Float(0), sympy.S.Zero, sympy.Integer(0)):
            with _no_deprecation():
                stokes.add_rotated_freeslip_bc(zero, "Top")
            assert stokes._rotated_freeslip_bcs[-1] == ("Top", None)

    def test_nonzero_datum_not_implemented(self, mesh, stokes):
        with pytest.raises(NotImplementedError):
            stokes.add_rotated_freeslip_bc(1.0, "Top")

    def test_symbolic_possibly_nonzero_datum_not_implemented(self, mesh, stokes):
        # An expression sympy cannot prove zero must be rejected, not let through.
        with pytest.raises(NotImplementedError):
            stokes.add_rotated_freeslip_bc(sympy.Symbol("a"), "Top")


class TestConstraintBCValueFirst:
    def test_new_order_no_warning(self, mesh):
        solver = uw.systems.Stokes_Constrained(mesh)
        solver.constitutive_model = uw.constitutive_models.ViscousFlowModel
        with _no_deprecation():
            h = solver.add_constraint_bc(0.0, "Top")
        assert solver._block_constraint_bcs[-1].boundary == "Top"
        assert h is not None

    def test_legacy_order_warns_once(self, mesh):
        solver = uw.systems.Stokes_Constrained(mesh)
        solver.constitutive_model = uw.constitutive_models.ViscousFlowModel
        with pytest.warns(DeprecationWarning, match=r"add_constraint_bc\(conds, boundary") as rec:
            solver.add_constraint_bc("Top", 0.0)
        assert _one_deprecation(rec) == 1
        cbc = solver._block_constraint_bcs[-1]
        assert cbc.boundary == "Top"
        assert float(cbc.g) == 0.0

    def test_legacy_g_keyword_warns_once(self, mesh):
        solver = uw.systems.Stokes_Constrained(mesh)
        solver.constitutive_model = uw.constitutive_models.ViscousFlowModel
        with pytest.warns(DeprecationWarning) as rec:
            solver.add_constraint_bc(boundary="Top", g=0.5)
        assert _one_deprecation(rec) == 1
        assert solver._block_constraint_bcs[-1].g == sympy.Float(0.5)


class TestLegacyTrioUnchanged:
    """The original trio is already canonical - it must NOT warn."""

    def test_dirichlet_no_warning(self, mesh):
        solver = uw.systems.Poisson(mesh)
        with _no_deprecation():
            solver.add_dirichlet_bc(0.0, "Top")
            solver.add_essential_bc(1.0, "Bottom")
            solver.add_natural_bc(0.0, "Left")

    def test_components_kwarg_warns_once(self, mesh, stokes):
        with pytest.warns(DeprecationWarning, match="components") as rec:
            stokes.add_dirichlet_bc((0.0, 0.0), "Top", components=(0,))
        assert _one_deprecation(rec) == 1


# ---------------------------------------------------------------------------
# WC-03 - consistent_jacobian validated property
# ---------------------------------------------------------------------------

class TestConsistentJacobianValidation:
    def test_valid_values_round_trip(self, mesh, stokes):
        for value in (False, True, "continuation"):
            stokes.consistent_jacobian = value
            assert stokes.consistent_jacobian == value

    def test_falsy_normalizes_to_false(self, mesh, stokes):
        for value in (None, 0, 0.0, ""):
            stokes.consistent_jacobian = value
            assert stokes.consistent_jacobian is False

    def test_invalid_values_raise(self, mesh, stokes):
        for value in ("picard", "Continuation", 1, "newton", 2.0):
            with pytest.raises(ValueError, match="consistent_jacobian"):
                stokes.consistent_jacobian = value

    def test_default_is_false(self, mesh):
        solver = uw.systems.Poisson(mesh)
        assert solver.consistent_jacobian is False


# ---------------------------------------------------------------------------
# WC-04 - set_custom_fmg solver method; set_custom_mg deprecated
# ---------------------------------------------------------------------------

class TestCustomMGEntryPoint:
    def test_set_custom_fmg_method_no_warning(self, mesh):
        coarse = uw.meshing.StructuredQuadBox(
            elementRes=(1, 1), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0)
        )
        solver = uw.systems.Poisson(mesh)
        with _no_deprecation():
            solver.set_custom_fmg([coarse])
        assert solver._custom_mg["mode"] == "hierarchy"

    def test_set_custom_mg_warns_once_and_still_registers(self, mesh):
        coarse = uw.meshing.StructuredQuadBox(
            elementRes=(1, 1), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0)
        )
        solver = uw.systems.Poisson(mesh)
        with pytest.warns(DeprecationWarning, match="set_custom_fmg") as rec:
            solver.set_custom_mg([coarse])
        assert _one_deprecation(rec) == 1
        assert solver._custom_mg["kind"] == "barycentric"


# ---------------------------------------------------------------------------
# WC-05 - namespace exposure (no deep imports needed)
# ---------------------------------------------------------------------------

class TestNamespaceExposure:
    def test_utilities_modules_exposed(self):
        assert hasattr(uw.utilities, "rotated_bc")
        assert hasattr(uw.utilities, "boundary_flux")
        assert hasattr(uw.utilities, "custom_mg")
        assert callable(uw.utilities.set_custom_fmg)

    def test_meshing_bounding_surface_exposed(self):
        assert hasattr(uw.meshing, "BoundingSurface")
        for name in (
            "register_radial_surfaces",
            "register_plane_surfaces",
            "register_box_face_surfaces",
        ):
            assert callable(getattr(uw.meshing, name))
            assert name in uw.meshing.__all__
        assert "BoundingSurface" in uw.meshing.__all__


# ---------------------------------------------------------------------------
# WC-06 - uw.quantity is THE factory (D14)
# ---------------------------------------------------------------------------

class TestQuantityFactory:
    def test_quantity_no_warning_and_type_exposed(self):
        with _no_deprecation():
            q = uw.quantity(1.0, "m")
        assert isinstance(q, uw.UWQuantity)

    def test_create_quantity_warns_once(self):
        with pytest.warns(DeprecationWarning, match="uw.quantity") as rec:
            q = uw.create_quantity(10.0, "m/s")
        assert _one_deprecation(rec) == 1
        # behaviour preserved for one cycle: still the raw Pint quantity
        assert float(q.magnitude) == 10.0
        assert str(q.units) in ("meter / second", "m / s")


# ---------------------------------------------------------------------------
# WC-07 - SNES_Poisson constructor order (mesh, u_Field, degree, verbose)
# ---------------------------------------------------------------------------

class TestPoissonConstructorOrder:
    def test_positional_degree_no_warning(self, mesh):
        with _no_deprecation():
            solver = uw.systems.Poisson(mesh, None, 3)
        assert solver.u.degree == 3
        assert solver.verbose is False

    def test_legacy_positional_verbose_warns_once(self, mesh):
        with pytest.warns(DeprecationWarning, match=r"degree, verbose") as rec:
            solver = uw.systems.Poisson(mesh, None, True)
        assert _one_deprecation(rec) == 1
        assert solver.verbose is True
        assert solver.u.degree == 2  # legacy default preserved

    def test_legacy_four_positional_args(self, mesh):
        with pytest.warns(DeprecationWarning) as rec:
            solver = uw.systems.Poisson(mesh, None, True, 3)
        assert _one_deprecation(rec) == 1
        assert solver.verbose is True
        assert solver.u.degree == 3

    def test_keyword_calls_unchanged(self, mesh):
        with _no_deprecation():
            solver = uw.systems.Poisson(mesh, degree=1, verbose=False)
        assert solver.u.degree == 1


# ---------------------------------------------------------------------------
# WC-08 - SNES_Vector.add_nitsche_bc aligned with the Stokes variant
# ---------------------------------------------------------------------------

class TestVectorNitscheAlignment:
    @pytest.fixture()
    def vector_solver(self, mesh):
        v = uw.discretisation.MeshVariable(
            "v_wc08", mesh, mesh.dim, degree=2, vtype=uw.VarType.VECTOR
        )
        solver = uw.systems.Vector_Projection(mesh, v)
        solver.constitutive_model = uw.constitutive_models.ViscousFlowModel
        return solver

    def test_normal_kwarg_accepted(self, mesh, vector_solver):
        n = sympy.Matrix([[0, 1]])
        with _no_deprecation():
            vector_solver.add_nitsche_bc(0.0, "Top", normal=n)
        assert vector_solver.natural_bcs[-1].boundary == "Top"

    def test_mask_raises_not_implemented(self, mesh, vector_solver):
        with pytest.raises(NotImplementedError, match="mask"):
            vector_solver.add_nitsche_bc(0.0, "Top", mask=sympy.Integer(1))

    def test_legacy_order_warns_once(self, mesh, vector_solver):
        with pytest.warns(DeprecationWarning, match=r"add_nitsche_bc\(conds, boundary") as rec:
            vector_solver.add_nitsche_bc("Top", 0.0)
        assert _one_deprecation(rec) == 1


# ---------------------------------------------------------------------------
# WC-09 - boundary_flux_to_field renamed boundary_flux_field
# ---------------------------------------------------------------------------

class TestBoundaryFluxRename:
    def test_new_name_importable_no_warning(self):
        with _no_deprecation():
            from underworld3.utilities.boundary_flux import boundary_flux_field
        assert callable(boundary_flux_field)

    def test_old_name_warns_once(self):
        from underworld3.utilities.boundary_flux import boundary_flux_to_field

        with pytest.warns(DeprecationWarning, match="boundary_flux_field") as rec:
            # The warning is emitted before any work; the junk arguments then
            # fail inside the real implementation, which is fine here.
            with pytest.raises(Exception):
                boundary_flux_to_field(None, "Top", None)
        assert _one_deprecation(rec) == 1


# ---------------------------------------------------------------------------
# WC-12 - no-op `sync=` kwarg deprecated on swarm pack/unpack
# ---------------------------------------------------------------------------

class TestSwarmSyncDeprecated:
    @pytest.fixture(scope="class")
    def swarm_var(self, mesh):
        swarm = uw.swarm.Swarm(mesh)
        var = swarm.add_variable("s_wc12", 1)
        swarm.populate(fill_param=1)
        # yield keeps the parent swarm alive for the lifetime of the tests
        # (the variable holds only a weak link to it)
        yield var
        del swarm

    def test_unpack_no_sync_no_warning(self, swarm_var):
        with _no_deprecation():
            data = swarm_var.unpack_uw_data_from_petsc(squeeze=False)
        assert data is not None

    def test_unpack_sync_warns_once_identical_result(self, swarm_var):
        import numpy as np

        clean = swarm_var.unpack_uw_data_from_petsc(squeeze=False)
        with pytest.warns(DeprecationWarning, match="sync") as rec:
            legacy = swarm_var.unpack_uw_data_from_petsc(squeeze=False, sync=True)
        assert _one_deprecation(rec) == 1
        assert np.array_equal(np.asarray(clean), np.asarray(legacy))

    def test_pack_sync_warns_once(self, swarm_var):
        data = swarm_var.unpack_uw_data_from_petsc(squeeze=False)
        with pytest.warns(DeprecationWarning, match="sync") as rec:
            swarm_var.pack_uw_data_to_petsc(data, sync=True)
        assert _one_deprecation(rec) == 1
        with _no_deprecation():
            swarm_var.pack_uw_data_to_petsc(data)


# ---------------------------------------------------------------------------
# WC-13 - smoothing: dead params dropped, n_sweeps renamed max_cg_iters
# ---------------------------------------------------------------------------

class TestSpringMoverSignature:
    def test_dead_params_gone_new_name_present(self):
        import inspect

        from underworld3.meshing.smoothing import _spring_equilibrium_mover

        params = inspect.signature(_spring_equilibrium_mover).parameters
        assert "relax" not in params
        assert "step_frac" not in params
        assert "max_cg_iters" in params
        assert "n_sweeps" in params  # deprecated alias, one cycle

    def test_n_sweeps_alias_warns_once(self):
        import numpy as np

        from underworld3.meshing.smoothing import _spring_equilibrium_mover

        tri_mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5
        )
        pinned = ("Top", "Bottom", "Left", "Right")
        with pytest.warns(DeprecationWarning, match="max_cg_iters") as rec:
            _spring_equilibrium_mover(tri_mesh, sympy.Integer(1), pinned, False, n_sweeps=1)
        assert _one_deprecation(rec) == 1
        with _no_deprecation():
            _spring_equilibrium_mover(tri_mesh, sympy.Integer(1), pinned, False, max_cg_iters=1)
