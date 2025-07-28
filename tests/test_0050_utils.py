# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

## %%

import underworld3 as uw
import sympy

mesh = uw.meshing.StructuredQuadBox(elementRes=(5,) * 2)
x, y = mesh.X

# %%
v = uw.discretisation.MeshVariable(
    r"mathbf{u}", mesh, mesh.dim, vtype=uw.VarType.VECTOR, degree=2
)
p = uw.discretisation.MeshVariable(
    r"mathbf{p}", mesh, 1, vtype=uw.VarType.SCALAR, degree=1
)


def bc_1(solver):
    s1 = solver
    s1.add_dirichlet_bc((0.0, 0.0), "Bottom")
    s1.add_dirichlet_bc((y, 0.0), "Top")

    s1.add_dirichlet_bc((sympy.oo, 0.0), "Left")
    s1.add_dirichlet_bc((sympy.oo, 0.0), "Right")


def bc_2(solver):
    s1 = solver
    s1.add_dirichlet_bc((0.0, sympy.oo), "Bottom")
    s1.add_dirichlet_bc((0.0, sympy.oo), "Top")

    s1.add_dirichlet_bc((0.0, 0.0), "Left")
    s1.add_dirichlet_bc((0.0, x), "Right")


# %%
def vis_model(mesh):
    import pyvista as pv
    import underworld3.visualisation as vis

    v = mesh.vars["mathbfu"]
    pl = pv.Plotter(window_size=(1000, 750))

    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v.sym)
    pvmesh.point_data["Vmag"] = vis.scalar_fn_to_pv_points(pvmesh, sympy.sqrt(v.sym.dot(v.sym)))
    pvmesh.point_data["V1"] = vis.scalar_fn_to_pv_points(pvmesh, v.sym[1])

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        scalars="Vmag",
        use_transparency=False,
        opacity=1.0,
    )

    velocity_points = vis.meshVariable_to_pv_cloud(v)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v.sym)
    arrows = pl.add_arrows(
        velocity_points.points,
        velocity_points.point_data["V"],
        mag=3e-1,
        opacity=0.5,
        show_scalar_bar=False,
        cmap="coolwarm",
    )

    pl.show(cpos="xy")


# %%
stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = 1

# %%
bc_1(stokes)

# %%
stokes.solve()

# %%
# vis_model(mesh)

# %%
s1 = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
s1.constitutive_model = uw.constitutive_models.ViscousFlowModel
s1.constitutive_model.Parameters.shear_viscosity_0 = 1
# stokes._rebuild_after_mesh_update()
bc_2(s1)

# %%
# stokes.solve()
s1.solve()

# %%
# vis_model(mesh)


def dont_test_auditor():
    # assert not values are in install data are None
    for v in uw.auditor.get_installation_data.values():
        assert v is not None

    # assert 7 uw_objects are created
    assert uw.auditor.get_runtime_data.get("uw_object_count") == 7


# %%
