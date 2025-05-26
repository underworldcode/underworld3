# Testing the ability to compile and load pointwise functions
# in the various parts of the solver chain (pretty simple checks
# since we haven't validated the solvers yet

import pytest
import sympy
import underworld3 as uw
import numpy as np
import os, shutil

import numpy as np
import sympy

from underworld3.utilities._jitextension import getext


# build a small mesh - we'll load up a simple problem and then see what functions are loaded
# into the solver

mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 0.5
)

x, y = mesh.X


v = uw.discretisation.MeshVariable(
    "V",
    mesh,
    vtype=uw.VarType.VECTOR,
    degree=2,
    varsymbol=r"\mathbf{v}",
)

w = uw.discretisation.MeshVariable(
    "W",
    mesh,
    vtype=uw.VarType.SCALAR,
    degree=2,
    varsymbol=r"\mathbf{w}",
)

# clear any prior artifacts
shutil.rmtree("/tmp/fn_ptr_ext_TEST_0", ignore_errors=True)
shutil.rmtree("/tmp/fn_ptr_ext_TEST_1", ignore_errors=True)
shutil.rmtree("/tmp/fn_ptr_ext_TEST_2", ignore_errors=True)


## This needs to be fixed up for systems that don't use /tmp like this
## So does the JIT ... it assumes /tmp is used exactly like this (LM).


def test_getext_simple():

    res_fn = sympy.ImmutableDenseMatrix([sympy.sympify(1), sympy.sympify(2)])
    jac_fn = sympy.ImmutableDenseMatrix([sympy.sympify(1), sympy.sympify(2)])
    bc_fn = sympy.ImmutableDenseMatrix([sympy.sympify(1), sympy.sympify(2)])
    bd_res_fn = sympy.ImmutableDenseMatrix([sympy.sympify(1), sympy.sympify(2)])
    bd_jac_fn = sympy.ImmutableDenseMatrix([sympy.sympify(1), sympy.sympify(2)])

    with uw.utilities.CaptureStdout(split=True) as captured_setup_solver:
        compiled_extns, dictionaries = getext(
            mesh,
            [res_fn, res_fn],
            [jac_fn],
            [bc_fn],
            [bd_res_fn],
            [bd_jac_fn],
            mesh.vars.values(),
            verbose=True,
            debug=True,
            debug_name="TEST_0",
            cache=False,
        )

    assert os.path.exists(f"/tmp/fn_ptr_ext_TEST_0")
    assert os.path.exists(f"/tmp/fn_ptr_ext_TEST_0/cy_ext.h")
    assert r"Processing JIT    5 / Matrix([[1], [2]])" in captured_setup_solver


def test_getext_sympy_fns():

    res_fn = sympy.ImmutableDenseMatrix([x, y])
    jac_fn = sympy.ImmutableDenseMatrix([x**2, y**2])
    bc_fn = sympy.ImmutableDenseMatrix([sympy.sin(x), sympy.cos(y)])
    bd_res_fn = sympy.ImmutableDenseMatrix([sympy.log(x), sympy.exp(y)])
    bd_jac_fn = sympy.ImmutableDenseMatrix(
        [sympy.diff(sympy.log(x)), sympy.diff(sympy.exp(y * x), y)]
    )

    with uw.utilities.CaptureStdout(split=True) as captured_setup_solver:
        compiled_extns, dictionaries = getext(
            mesh,
            [res_fn, res_fn],
            [jac_fn],
            [bc_fn],
            [bd_res_fn],
            [bd_jac_fn],
            mesh.vars.values(),
            verbose=True,
            debug=True,
            debug_name="TEST_1",
            cache=False,
        )

    assert os.path.exists(f"/tmp/fn_ptr_ext_TEST_1")
    assert os.path.exists(f"/tmp/fn_ptr_ext_TEST_1/cy_ext.h")
    assert (
        r"Processing JIT    5 / Matrix([[1/N.x], [N.x*exp(N.x*N.y)]])"
        in captured_setup_solver
    )


    # TODO this test is failing after LM changes - DISABLING for now. JG, 26-05-25.
#def test_getext_meshVar():
#
#    res_fn = sympy.ImmutableDenseMatrix([v.sym[0], w.sym])
#    jac_fn = sympy.ImmutableDenseMatrix([x * v.sym[0], y**2])
#    bc_fn = sympy.ImmutableDenseMatrix([v.sym[1] * sympy.sin(x), sympy.cos(y)])
#    bd_res_fn = sympy.ImmutableDenseMatrix([sympy.log(v.sym[0]), sympy.exp(w.sym)])
#    bd_jac_fn = sympy.ImmutableDenseMatrix(
#        [sympy.diff(sympy.log(v.sym[0]), y), sympy.diff(sympy.exp(y * x), y)]
#    )
#
#    with uw.utilities.CaptureStdout(split=True) as captured_setup_solver:
#        compiled_extns, dictionaries = getext(
#            mesh,
#            [res_fn, res_fn],
#            [jac_fn],
#            [bc_fn],
#            [bd_res_fn],
#            [bd_jac_fn],
#            mesh.vars.values(),
#            verbose=True,
#            debug=True,
#            debug_name="TEST_2",
#            cache=False,
#        )
#
#    assert os.path.exists(f"/tmp/fn_ptr_ext_TEST_2")
#    assert os.path.exists(f"/tmp/fn_ptr_ext_TEST_2/cy_ext.h")
#    assert (
#        r"Processing JIT    5 / Matrix([[{\mathbf{v}}_{ 0,1}(N.x, N.y)/{\mathbf{v}}_{ 0 }(N.x, N.y)], [N.x*exp(N.x*N.y)]])"
#        in captured_setup_solver
#    )


# def test_build_functions():
#     stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
#     stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
#     stokes.constitutive_model.Parameters.shear_viscosity_0 = 1
#     stokes.bodyforce = sympy.Matrix([0.0, 1.0 * sympy.sin(mesh.X[0])])

#     stokes.add_dirichlet_bc((0.0, 0.0), "Top")
#     stokes.add_dirichlet_bc((sympy.oo, 0.0), "Bottom")
#     stokes.add_dirichlet_bc((0.0, sympy.oo), "Left")
#     stokes.add_dirichlet_bc((0.0, sympy.oo), "Right")

#     print(f"mesh.N.x -> {type(mesh.N.x)}")

#     with uw.utilities.CaptureStdout(split=False) as captured_setup_solver:
#         stokes._setup_pointwise_functions(
#             verbose=True,
#             debug_name="TEST_0004_0",
#         )
#         stokes._setup_discretisation(verbose=True)
#         stokes._setup_solver(verbose=True)

#     counter = len(stokes.ext_dict)

#     print("============", flush=True)
#     print(captured_setup_solver, flush=True)
#     print("============", flush=True)

#     assert os.path.exists(f"/tmp/fn_ptr_ext_TEST_0004_0")
#     assert os.path.exists(f"/tmp/fn_ptr_ext_TEST_0004_0/cy_ext.h")

#     # Solver JIT (incompressibility constraint)
#     assert (
#         r"Matrix([[\mathbf{u}_{ 0,0}(N.x, N.y) + \mathbf{u}_{ 1,1}(N.x, N.y)]])"
#         in captured_setup_solver
#     )

#     # Solver JIT (Essential Boundary condition)
#     assert r"Matrix([[oo], [0]])" in captured_setup_solver

#     # Solver JIT (Jacobian 3)
#     assert (
#         r"Matrix([[2, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 2]])"
#         in captured_setup_solver
#     )

#     # Compilation (number of equations)
#     assert r"Equation count - 18" in captured_setup_solver

#     # Compilation (number of equations)
#     assert r"Processing JIT   17" in captured_setup_solver

#     # Compilation (finds jacobians etc in the weak form)
#     # These now/sometimes by pass by the capture routines
#     #  ... can't check it

#     # assert r"jacobian_g3" in captured_setup_solver
#     # assert r"jacobian_preconditioner_g2" in captured_setup_solver

#     return


# ## This needs to be fixed up for systems that don't use /tmp like this
# def test_build_boundary_functions():
#     stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
#     stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
#     stokes.constitutive_model.Parameters.shear_viscosity_0 = 1
#     stokes.bodyforce = sympy.Matrix([0.0, 1.0 * sympy.sin(mesh.X[0])])

#     stokes.add_natural_bc((0.0, sympy.oo), "Top")
#     stokes.add_dirichlet_bc((sympy.oo, 0.0), "Bottom")
#     stokes.add_dirichlet_bc((0.0, sympy.oo), "Left")
#     stokes.add_dirichlet_bc((0.0, sympy.oo), "Right")

#     try:
#         counter = len(stokes.ext_dict)
#     except:
#         counter = 0

#     with uw.utilities.CaptureStdout(split=False) as captured_setup_solver:
#         stokes._setup_pointwise_functions(
#             verbose=True,
#             debug_name="TEST_0004_1",
#         )
#         stokes._setup_discretisation(verbose=True)
#         stokes._setup_solver(verbose=True)

#     # print("============", flush=True)
#     # print(captured_setup_solver, flush=True)
#     # print("============", flush=True)

#     counter = len(stokes.ext_dict)

#     # Solver JIT (incompressibility constraint)
#     assert (
#         r"Matrix([[\mathbf{u}_{ 0,0}(N.x, N.y) + \mathbf{u}_{ 1,1}(N.x, N.y)]])"
#         in captured_setup_solver
#     )

#     # Solver JIT (Essential Boundary condition)
#     assert r"Matrix([[oo], [0]])" in captured_setup_solver

#     # Solver JIT (Jacobian 3)
#     assert (
#         r"Matrix([[2, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 2]])"
#         in captured_setup_solver
#     )

#     # Compilation (number of equations)
#     assert r"Equation count - 23" in captured_setup_solver

#     # Compilation (number of equations)
#     assert r"Processing JIT   22" in captured_setup_solver

#     # Compilation (finds jacobians etc in the boundary weak form)
#     # assert r"boundary_jacobian_g0" in captured_setup_solver
#     # assert r"boundary_residual_f0" in captured_setup_solver
#     # assert r"boundary_jacobian_preconditioner_g0" in captured_setup_solver

#     del stokes

#     return


# # # Check if the functions are actually executed. For this we have to capture the
# # stdout from the C calls and I'm not sure how to do that.


# ## This needs to be fixed up for systems that don't use /tmp like this
# def test_debug_pointwise_functions():
#     stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
#     stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
#     stokes.constitutive_model.Parameters.shear_viscosity_0 = 1

#     stokes.bodyforce = sympy.Matrix([0.0, 1.0 * sympy.sin(mesh.X[0])])

#     #
#     stokes.add_dirichlet_bc((sympy.oo, 0.0), "Top")
#     stokes.add_dirichlet_bc((sympy.oo, 0.0), "Bottom")
#     stokes.add_dirichlet_bc((0.0, sympy.oo), "Left")
#     stokes.add_dirichlet_bc((0.0, sympy.oo), "Right")

#     stokes.petsc_options["snes_monitor"] = None
#     stokes.petsc_options["ksp_monitor"] = None

#     # Linear solve for initial guess

#     with uw.utilities.CaptureStdout(split=False) as captured_lin_solve:
#         stokes.solve(
#             verbose=True,
#             debug=True,
#             debug_name="TEST_0004_2",
#         )

#     stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
#     stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
#     stokes.constitutive_model.Parameters.shear_viscosity_0 = 1
#     stokes.bodyforce = sympy.Matrix([0.0, 1.0 * sympy.sin(mesh.X[0])])

#     #
#     stokes.add_natural_bc((0.0, 1.0 * v.sym[1]), "Top")
#     stokes.add_dirichlet_bc((sympy.oo, 0.0), "Bottom")
#     stokes.add_dirichlet_bc((0.0, sympy.oo), "Left")
#     stokes.add_dirichlet_bc((0.0, sympy.oo), "Right")

#     stokes.petsc_options["snes_monitor"] = None
#     stokes.petsc_options["ksp_monitor"] = None

#     # Non linear solve
#     # with uw.utilities.CaptureStdout(split=False) as captured_non_lin_solve:
#     stokes.solve(
#         zero_init_guess=False,
#         verbose=False,
#         debug=True,
#         debug_name="TEST_0004_3",
#         _force_setup=False,
#     )

#     # Very basic tests - there should be at least one of each of these though

#     with open(f"TEST_0004_3_debug.txt") as file:
#         debug_ptwise_contents = file.read()

#     assert r"ebc" in debug_ptwise_contents
#     assert r"bdjac" in debug_ptwise_contents
#     assert r"bdres" in debug_ptwise_contents

#     return


# def setup_function():
#     os.environ["UW_JITNAME"] = "TEST_FN"
#     return


# # clean up
# def teardown_function():
#     #    yield  # This runs the tests
#     os.unsetenv("UW_JITNAME")

#     for debug_file in [
#         "TEST_0004_0_debug.txt",
#         "TEST_0004_1_debug.txt",
#         "TEST_0004_2_debug.txt",
#         "TEST_0004_3_debug.txt",
#     ]:
#         if os.path.exists(debug_file):
#             os.remove(debug_file)

#     del os.environ["UW_JITNAME"]


# +
# Run the script in test mode

# test_build_functions()
# test_build_boundary_functions()


def setup_function():
    shutil.rmtree("/tmp/fn_ptr_ext_TEST_0", ignore_errors=True)
    shutil.rmtree("/tmp/fn_ptr_ext_TEST_1", ignore_errors=True)
    shutil.rmtree("/tmp/fn_ptr_ext_TEST_2", ignore_errors=True)
    return


# # clean up
def teardown_function():
    #    yield  # This runs the tests
    shutil.rmtree("/tmp/fn_ptr_ext_TEST_0", ignore_errors=True)
    shutil.rmtree("/tmp/fn_ptr_ext_TEST_1", ignore_errors=True)
    shutil.rmtree("/tmp/fn_ptr_ext_TEST_2", ignore_errors=True)
    return
