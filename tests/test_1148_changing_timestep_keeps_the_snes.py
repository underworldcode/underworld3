"""A new timestep is a runtime constant: no setup, the same SNES, its own answer (#1026).

The timestep enters the compiled kernels as a constant, and every solve packs the
constants again, so a new one needs no rewire and no recompile. The time-dependent
solvers' ``delta_t`` setters marked a function rewire on every change all the same,
and their ``solve`` then ran its own setup (pointwise functions, discretisation,
solver) ahead of the base class's ``_build``: that setup created a new SNES without
destroying the one before. Every step with a new timestep left a whole SNES behind
(KSP, PC and multigrid hierarchy): an annulus convection run at res 128 and np 32 grew
by about 16 MB per step per rank and was killed after 1534 steps.

Each case solves three steps with three timesteps and checks that

- after the first solve, no step sets the solver up again and the JIT key stays
  the one compiled;
- the SNES is the same object on every step; and
- the result is the one a solver rebuilt from scratch on every step gives
  (``_force_setup``), and not the one a fixed timestep gives, so the new timestep
  reached the kernels.

TransientDarcy compiled its first F1 before its symbolic flux history was populated,
so the flux lacked its history term: theta times the flux, half at theta = 0.5. A
rewire put it back, and a new timestep was one; at a fixed timestep the run kept the
half flux, and one step differed from a forced setup's by 8.5 % (#1027). The last
test checks one step against a forced setup. Theta itself has no effect with that
history, which holds the current flux (#1029); the test compares two setups and does
not depend on it.

A timestep with units is reduced by the model's time scale, and a new one is a
constant like a number: TransientDarcy packed the magnitude, so 400 kyr went in as
400, and NavierStokesSLCN kept the quantity, which is not a number, and rewired on
every step. Steady Darcy's ``solve`` set itself up directly too, and a new source
left the old SNES alive.

Viscoelastic Stokes passed before the fix too; it is here because its ``solve`` had
the same setup, which now goes through ``_build``, and its timestep (``dt_elastic``)
is a constant as well.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

TIMESTEPS = (1.0e-2, 1.3e-2, 0.7e-2)


def _box():
    return uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1),
                                             cellSize=1 / 8, qdegree=3)


def _slcn(tag):
    mesh = _box()
    x, y = mesh.X
    T = uw.discretisation.MeshVariable(f"T{tag}", mesh, 1, degree=2)
    V = uw.discretisation.MeshVariable(f"V{tag}", mesh, 2, degree=2)
    T.array[:, 0, 0] = uw.function.evaluate(sympy.sin(sympy.pi * x) * sympy.sin(sympy.pi * y),
                                            T.coords).reshape(-1)
    V.array[:, 0, :] = uw.function.evaluate(sympy.Matrix([[0.5 - y, x - 0.5]]),
                                            V.coords).reshape(-1, 2)
    s = uw.systems.AdvDiffusionSLCN(mesh, u_Field=T, V_fn=V.sym)
    s.constitutive_model = uw.constitutive_models.DiffusionModel
    s.constitutive_model.Parameters.diffusivity = 0.1
    s.add_dirichlet_bc(0.0, "Bottom")
    return s, lambda: np.asarray(T.array[...]).copy()


def _diffusion(tag):
    mesh = _box()
    x, y = mesh.X
    T = uw.discretisation.MeshVariable(f"T{tag}", mesh, 1, degree=2)
    T.array[:, 0, 0] = uw.function.evaluate(sympy.sin(sympy.pi * x) * sympy.sin(sympy.pi * y),
                                            T.coords).reshape(-1)
    s = uw.systems.Diffusion(mesh, u_Field=T, order=2, theta=1.0)
    s.constitutive_model = uw.constitutive_models.DiffusionModel
    s.constitutive_model.Parameters.diffusivity = 0.1
    s.add_dirichlet_bc(0.0, "Bottom")
    return s, lambda: np.asarray(T.array[...]).copy()


def _transient_darcy(tag):
    mesh = _box()
    x, y = mesh.X
    h = uw.discretisation.MeshVariable(f"h{tag}", mesh, 1, degree=2)
    v = uw.discretisation.MeshVariable(f"v{tag}", mesh, 2, degree=1)
    h.array[:, 0, 0] = uw.function.evaluate(sympy.sin(sympy.pi * x) * sympy.sin(sympy.pi * y),
                                            h.coords).reshape(-1)
    s = uw.systems.TransientDarcy(mesh, h, v, order=1, theta=0.5)
    s.constitutive_model = uw.constitutive_models.DarcyFlowModel
    s.constitutive_model.Parameters.permeability = 1.0
    s.constitutive_model.Parameters.s = sympy.Matrix([0, 0]).T
    s.storage = 1.0
    s.f = 0.0
    s.add_dirichlet_bc(0.0, "Bottom")
    return s, lambda: np.asarray(h.array[...]).copy()


def _viscoelastic_stokes(tag):
    mesh = uw.meshing.StructuredQuadBox(elementRes=(8, 8))
    v = uw.discretisation.MeshVariable(f"U{tag}", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable(f"P{tag}", mesh, 1, degree=1)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        s.Unknowns, order=1)
    s.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    s.constitutive_model.Parameters.shear_modulus = 10.0
    s.constitutive_model.Parameters.dt_elastic = TIMESTEPS[0]
    s.add_dirichlet_bc((1.0, 0.0), "Top")
    s.add_dirichlet_bc((-1.0, 0.0), "Bottom")
    s.add_dirichlet_bc((sympy.oo, 0.0), "Left")
    s.add_dirichlet_bc((sympy.oo, 0.0), "Right")
    s.tolerance = 1.0e-8
    s._timestep_also_sets = lambda dt: setattr(s.constitutive_model.Parameters,
                                               "dt_elastic", dt)
    # the velocity is the walls' simple shear at any timestep; the stress history is not
    return s, lambda: np.asarray(s.DFDt.psi_star[0].array[...]).copy()


def _navier_stokes_slcn(tag):
    mesh = _box()
    v = uw.discretisation.MeshVariable(f"U{tag}", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable(f"P{tag}", mesh, 1, degree=1)
    s = uw.systems.NavierStokesSLCN(mesh, velocityField=v, pressureField=p, rho=1.0)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    return s, lambda: np.asarray(v.array[...]).copy()


def _march(build, tag, rebuild, timesteps=TIMESTEPS):
    """Solve at ``timesteps``; return each step's SNES, the setups and JIT keys after
    the first solve, and the final state."""
    s, state = build(tag)
    snes, setups, keys = [], [], []
    setup = s._setup_pointwise_functions

    def _counted(*args, **kwargs):
        setups.append(1)
        return setup(*args, **kwargs)

    for step, dt in enumerate(timesteps):
        also = getattr(s, "_timestep_also_sets", None)
        if also is not None:
            also(dt)
        s.solve(timestep=dt, zero_init_guess=False, _force_setup=rebuild)
        assert s.snes.getConvergedReason() > 0, s.snes.getConvergedReason()
        snes.append(s.snes)
        keys.append(getattr(s, "_current_jit_cache_key", None))
        if step == 0:
            s._setup_pointwise_functions = _counted
    return snes, len(setups), keys, state()


@pytest.mark.parametrize("build", [_slcn, _diffusion, _transient_darcy, _viscoelastic_stokes],
                         ids=["adv_diffusion_slcn", "diffusion", "transient_darcy",
                              "viscoelastic_stokes"])
def test_a_new_timestep_keeps_the_snes_and_takes_effect(build):
    tag = build.__name__
    snes, setups, keys, kept = _march(build, f"{tag}k", rebuild=False)
    # a constant: before the fix every new timestep set the solver up again
    assert setups == 0 and len(set(keys)) == 1, (
        f"{tag}: {setups} setups and {len(set(keys))} JIT keys over "
        f"{len(TIMESTEPS) - 1} new timesteps")
    # one SNES across the steps; before the fix each new timestep made another
    # and left the last one alive
    assert all(s is snes[0] for s in snes[1:]), (
        f"{tag}: {len({id(s) for s in snes})} SNES objects over {len(snes)} timesteps")
    scale = max(np.abs(kept).max(), 1.0e-30)
    *_rest, rebuilt = _march(build, f"{tag}r", rebuild=True)
    # 7e-9 (SLCN) and 9e-8 (diffusion) measured: a kept KSP and a new one stop
    # at different iterates within the solve's tolerance
    assert np.abs(kept - rebuilt).max() <= 1.0e-6 * scale, (
        f"{tag}: the kept solver's result differs from a rebuilt one's by "
        f"{np.abs(kept - rebuilt).max() / scale:.2e} (relative)")
    # and the timesteps matter: a march at the first one throughout ends elsewhere
    *_rest, fixed = _march(build, f"{tag}f", rebuild=False,
                          timesteps=(TIMESTEPS[0],) * len(TIMESTEPS))
    assert np.abs(kept - fixed).max() > 1.0e-4 * scale, (
        f"{tag}: the march ends where a fixed timestep's does")


def test_a_new_darcy_source_destroys_the_old_snes():
    mesh = _box()
    x, y = mesh.X
    h = uw.discretisation.MeshVariable("h_steady", mesh, 1, degree=2)
    v = uw.discretisation.MeshVariable("v_steady", mesh, 2, degree=1)
    s = uw.systems.SteadyStateDarcy(mesh, h, v)
    s.constitutive_model = uw.constitutive_models.DarcyFlowModel
    s.constitutive_model.Parameters.permeability = 1.0
    s.constitutive_model.Parameters.s = sympy.Matrix([0, 0]).T
    s.f = sympy.sin(sympy.pi * x)
    s.add_dirichlet_bc(0.0, "Bottom")
    s.add_dirichlet_bc(0.0, "Top")
    s.solve()
    old = s.snes
    s.f = sympy.cos(sympy.pi * x) * y
    s.solve()
    # a new source rewires the kernels in place and keeps the SNES; SNES_Darcy.solve
    # set itself up directly, made a new SNES and left the old one alive
    assert s.snes is old or old.handle == 0, "a new SNES, and the old one is still alive"


@pytest.fixture
def time_scale_20_myr():
    uw.reset_default_model()
    uw.get_default_model().set_reference_quantities(length_scale=1000 * uw.units.km,
                                                    time_scale=20 * uw.units.Myr)
    yield
    uw.reset_default_model()


@pytest.mark.parametrize("build", [_slcn, _diffusion, _transient_darcy, _navier_stokes_slcn],
                         ids=["adv_diffusion_slcn", "diffusion", "transient_darcy",
                              "navier_stokes_slcn"])
def test_a_timestep_with_units_is_the_reduced_constant(build, time_scale_20_myr):
    s, _state = build(f"{build.__name__}u")
    s.delta_t = uw.quantity(400, "kyr")
    # the first timestep replaces the solver's symbolic one, and rewires
    s._needs_function_rewire = False
    s.delta_t = uw.quantity(600, "kyr")
    # TransientDarcy packed the magnitude, 600, and NavierStokesSLCN kept the
    # quantity, which is not a number, and rewired on every step
    assert float(s.delta_t.sym) == pytest.approx(0.03, rel=1.0e-12), s.delta_t.sym
    assert not s._needs_function_rewire, f"{build.__name__}: a new timestep with units rewires"


@pytest.mark.parametrize("build", [_transient_darcy, _diffusion],
                         ids=["transient_darcy", "diffusion"])
def test_the_first_step_is_the_one_a_forced_setup_gives(build):
    tag = build.__name__
    s, state = build(f"{tag}1")
    s.solve(timestep=TIMESTEPS[0], zero_init_guess=False)
    plain = state()
    s, state = build(f"{tag}2")
    s.solve(timestep=TIMESTEPS[0], zero_init_guess=False, _force_setup=True)
    forced = state()
    scale = max(np.abs(forced).max(), 1.0e-30)
    # 8.5e-2 for TransientDarcy before the fix (#1027)
    assert np.abs(plain - forced).max() <= 1.0e-6 * scale, (
        f"{tag}: one step differs from a forced setup's by "
        f"{np.abs(plain - forced).max() / scale:.2e} (relative)")
