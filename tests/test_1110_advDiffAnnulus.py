import pytest
import sympy
import underworld3 as uw
import numpy as np


def test_adv_diff_annulus():
    mesh = uw.meshing.Annulus(
        radiusOuter=1.0, radiusInner=0.5, cellSize=0.2, refinement=1, qdegree=3
    )

    v_soln = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
    t_soln = uw.discretisation.MeshVariable("T", mesh, 1, degree=3)
    t_0 = uw.discretisation.MeshVariable("T0", mesh, 1, degree=3, varsymbol=r"T_{0}")

    # Create adv_diff object

    # Set some things
    k = 0.01
    h = 0.1
    t_i = 2.0
    t_o = 1.0
    r_i = 0.5
    r_o = 1.0
    delta_t = 0.05  ## 1/20 rotation in one step

    adv_diff = uw.systems.AdvDiffusion(
        mesh,
        u_Field=t_soln,
        V_fn=v_soln,
    )

    adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel
    adv_diff.constitutive_model.Parameters.diffusivity = k

    radius_fn = sympy.sqrt(
        mesh.rvec.dot(mesh.rvec)
    )  # normalise by outer radius if not 1.0
    unit_rvec = mesh.rvec / (1.0e-10 + radius_fn)

    # Some useful coordinate stuff

    x, y = mesh.X
    r, th = mesh.CoordinateSystem.xR

    # Rigid body rotation v_theta = constant, v_r = 0.0

    theta_dot = 2.0 * np.pi  # i.e one revolution in time 1.0
    v_x = -r * theta_dot * sympy.sin(th)
    v_y = r * theta_dot * sympy.cos(th)

    with mesh.access(v_soln):
        v_soln.data[:, 0] = uw.function.evaluate(v_x, v_soln.coords)
        v_soln.data[:, 1] = uw.function.evaluate(v_y, v_soln.coords)

    abs_r = sympy.sqrt(mesh.rvec.dot(mesh.rvec))

    init_t = sympy.exp(-30.0 * (mesh.N.x**2 + (mesh.N.y - 0.75) ** 2))

    adv_diff.add_dirichlet_bc(0.0, "Lower")
    adv_diff.add_dirichlet_bc(0.0, "Upper")

    with mesh.access(t_0, t_soln):
        t_0.data[...] = uw.function.evaluate(init_t, t_0.coords).reshape(-1, 1)
        t_soln.data[...] = t_0.data[...]

    with mesh.access(t_soln):
        t_soln.data[...] = uw.function.evaluate(init_t, t_0.coords).reshape(-1, 1)

    scalar_projection_solver = uw.systems.solvers.SNES_Projection(mesh, t_0)
    scalar_projection_solver.uw_function = t_soln.sym[0]
    scalar_projection_solver.bcs = adv_diff.bcs
    scalar_projection_solver.solve()

    scalar_projection_solver.uw_function = t_soln.sym[0]
    scalar_projection_solver.bcs = adv_diff.bcs
    scalar_projection_solver.solve()

    adv_diff.solve(timestep=delta_t)
