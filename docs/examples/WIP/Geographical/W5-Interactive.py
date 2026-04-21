# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # H2Ex Interactive Visualiser
#
# Single interactive 3D view with:
# - **Depth slider** for horizontal cross-section
# - **Checkbox toggles** for field overlays (strain rate, permeability,
#   Darcy velocity, concentration, faults)
# - **Field selector** via checkboxes — switch between scalar fields
#
# Uses pyvista trame backend for Jupyter interactivity.

# %% [markdown]
# ## Configuration

# %%
STRESS_AZIMUTH = 0
RES = 28
CLIP_FRACTION = 0.1  # initial depth fraction

# %%
import nest_asyncio
nest_asyncio.apply()

# %%
import numpy as np
import time
from pathlib import Path
import underworld3 as uw
from underworld3.workflows import WorkflowProducts
import pyvista as pv
import underworld3.visualisation as vis

uw.reset_default_model()
model = uw.Model()
model.set_reference_quantities(
    length=uw.quantity(1000, "km"),
    viscosity=uw.quantity(1e21, "Pa.s"),
    diffusivity=uw.quantity(1e-6, "m**2/s"),
    verbose=False,
)

SHARED_DIR = Path(f"output/h2ex/res_{RES}/shared")
RUN_DIR = Path(f"output/h2ex/res_{RES}/azimuth_{STRESS_AZIMUTH:03d}")

shared = WorkflowProducts(products_dir=SHARED_DIR / "products")
run = WorkflowProducts(products_dir=RUN_DIR / "products")

print(f"H2Ex Interactive: azimuth={STRESS_AZIMUTH}, RES={RES}")

# %% [markdown]
# ## Load Data

# %%
t0 = time.time()
mesh = shared.load("adapted_mesh")
faults = shared.load("fault_surfaces", mesh=mesh)

# Load all available fields
strain_rate = uw.discretisation.MeshVariable("eps", mesh, 1, degree=1)
run.load("strain_rate", mesh_variable=strain_rate)

strain_rate_ref = uw.discretisation.MeshVariable("eps_ref", mesh, 1, degree=1)
run.load("strain_rate_ref", mesh_variable=strain_rate_ref)

permeability = uw.discretisation.MeshVariable("perm", mesh, 1, degree=1)
run.load("permeability", mesh_variable=permeability)

v_darcy = uw.discretisation.MeshVariable("v_darcy", mesh, mesh.dim, degree=1,
                                          continuous=True, units="m/s")
run.load("darcy_velocity", mesh_variable=v_darcy)

v_stokes = None
if run.exists("stokes_velocity"):
    v_stokes = uw.discretisation.MeshVariable("v", mesh, mesh.dim, degree=2, units="cm/yr")
    run.load("stokes_velocity", mesh_variable=v_stokes)

C = None
if run.exists("concentration"):
    C = uw.discretisation.MeshVariable("C", mesh, 1, degree=1)
    run.load("concentration", mesh_variable=C)

print(f"Loaded in {time.time()-t0:.1f}s: {mesh.X.coords.shape[0]} nodes, "
      f"{len(faults.surfaces)} faults, "
      f"Stokes={'yes' if v_stokes else 'no'}, C={'yes' if C else 'no'}")

# %% [markdown]
# ## Build PyVista Mesh

# %%
pvmesh = vis.mesh_to_pv_mesh(mesh)
dm_coords = mesh.dm.getCoordinatesLocal().array.reshape(-1, 3)

# Scalar fields (P1 — direct data assignment)
pvmesh.point_data["eps"] = np.asarray(strain_rate.data).ravel()
pvmesh.point_data["delta_eps"] = np.asarray(strain_rate.data).ravel() - np.asarray(strain_rate_ref.data).ravel()
pvmesh.point_data["log_perm"] = np.log10(np.maximum(np.asarray(permeability.data).ravel(), 1e-10))
pvmesh.point_data["Vd"] = np.asarray(v_darcy.data)
pvmesh.point_data["Vd_mag"] = np.linalg.norm(pvmesh.point_data["Vd"], axis=1)

if v_stokes is not None:
    vs_at_verts = np.asarray(
        uw.function.evaluate(v_stokes.sym, dm_coords, mode="fast")
    ).reshape(-1, 3)
    pvmesh.point_data["Vs"] = vs_at_verts
    pvmesh.point_data["Vs_mag"] = np.linalg.norm(vs_at_verts, axis=1)

if C is not None:
    pvmesh.point_data["C"] = np.asarray(C.data).ravel()

# Fault distance
fields = faults.compute_nearest_fields(mesh, fault_width=uw.quantity(10.0, "km"))
pvmesh.point_data["fault_dist"] = np.asarray(fields["distance"].data).ravel()

# Geometry
r = np.sqrt(np.sum(pvmesh.points**2, axis=1))
pvmesh.point_data["radius"] = r
r_surface = r.max()
r_bottom = r.min()
depth_range = r_surface - r_bottom

# Centre and normal for slicing
centre = pvmesh.center_of_mass()
normal_up = centre / np.linalg.norm(centre)

print(f"Fields: {list(pvmesh.point_data.keys())}")

# %% [markdown]
# ## Fault Surfaces (scaled to meters)

# %%
L_ref = float(model.get_fundamental_scales()["length"].magnitude)

fault_pv_meshes = []
FAULT_COLORS = ["Red", "Orange", "Blue", "Green", "Purple",
                "Cyan", "Yellow", "Magenta", "Lime", "Pink"]

for i, (name, surf) in enumerate(faults.surfaces.items()):
    if hasattr(surf, "_pv_mesh") and surf._pv_mesh is not None:
        scaled = surf._pv_mesh.copy()
        scaled.points = scaled.points * L_ref
        fault_pv_meshes.append((name, scaled, FAULT_COLORS[i % len(FAULT_COLORS)]))

print(f"{len(fault_pv_meshes)} fault surfaces prepared")

# %% [markdown]
# ## Interactive Plotter
#
# - **Depth slider**: moves the horizontal cross-section
# - **Checkboxes**: toggle fault surfaces, Darcy arrows, and scalar fields

# %%
pl = pv.Plotter(window_size=(1200, 900))
pl.enable_depth_peeling(10)

# --- Depth slice (initial) ---
r_clip = r_surface - CLIP_FRACTION * depth_range
initial_slice = pvmesh.clip_scalar(scalars="radius", value=r_clip)

# --- Field definitions: name → (colormap, clim) ---
delta = pvmesh.point_data["delta_eps"]
delta_lim = float(np.percentile(np.abs(delta), 95))
eps_data = pvmesh.point_data["eps"]

FIELD_DEFS = {
    "delta_eps":  ("RdBu_r",  [-delta_lim, delta_lim]),
    "eps":        ("inferno",  [float(eps_data.min()), float(np.percentile(eps_data, 98))]),
    "log_perm":   ("RdYlBu_r", [-1, 1]),
    "Vd_mag":     ("viridis",  [0, float(pvmesh.point_data["Vd_mag"].max())]),
    "fault_dist": ("viridis_r", None),
}
if "Vs_mag" in pvmesh.point_data:
    FIELD_DEFS["Vs_mag"] = ("plasma", [0, float(np.percentile(pvmesh.point_data["Vs_mag"], 98))])
if C is not None:
    FIELD_DEFS["C"] = ("Blues", [0, 0.5])

FIELD_NAMES = list(FIELD_DEFS.keys())

# Mutable state for the current field and depth
_state = {
    "field_idx": FIELD_NAMES.index("C") if "C" in FIELD_NAMES else 0,
    "depth_frac": CLIP_FRACTION,
}

def _redraw_slice():
    """Redraw the depth slice with the current field and depth."""
    frac = _state["depth_frac"]
    r_cut = r_surface - frac * depth_range
    sliced = pvmesh.clip_scalar(scalars="radius", value=r_cut)

    field = FIELD_NAMES[_state["field_idx"]]
    cmap, clim = FIELD_DEFS[field]

    kwargs = dict(
        scalars=field, cmap=cmap, show_edges=False, opacity=0.9,
        show_scalar_bar=True, scalar_bar_args={"title": field},
        name="depth_slice",
    )
    if clim is not None:
        kwargs["clim"] = clim

    pl.add_mesh(sliced, **kwargs)

# --- Initial slice ---
_redraw_slice()

# --- Fault surfaces ---
fault_actors = []
for name, fmesh, color in fault_pv_meshes:
    actor = pl.add_mesh(
        fmesh, style="surface", color=color,
        opacity=0.7, show_edges=True,
    )
    fault_actors.append(actor)

# --- Darcy velocity arrows (on initial slice) ---
clip0 = pvmesh.clip_scalar(scalars="radius", value=r_surface - CLIP_FRACTION * depth_range)
Vd = clip0.point_data["Vd"]
vd_max = np.linalg.norm(Vd, axis=1).max()
arrow_actor = None
if vd_max > 0:
    arrow_actor = pl.add_arrows(
        clip0.points, Vd,
        mag=10000 / vd_max,
        color="White", opacity=0.4,
        show_scalar_bar=False,
    )
    arrow_actor.SetVisibility(False)

# --- Stokes velocity arrows ---
stokes_arrow_actor = None
if "Vs" in clip0.point_data:
    Vs = clip0.point_data["Vs"]
    vs_max = np.linalg.norm(Vs, axis=1).max()
    if vs_max > 0:
        stokes_arrow_actor = pl.add_arrows(
            clip0.points, Vs,
            mag=3000 / vs_max,
            color="Cyan", opacity=0.4,
            show_scalar_bar=False,
        )
        stokes_arrow_actor.SetVisibility(False)

# =============================================
# Interactive widgets
# =============================================

# --- Field selector (text slider = dropdown-like) ---
def select_field(value):
    """Switch the displayed scalar field."""
    idx = FIELD_NAMES.index(value)
    _state["field_idx"] = idx
    _redraw_slice()

pl.add_text_slider_widget(
    select_field,
    FIELD_NAMES,
    value=FIELD_NAMES.index(FIELD_NAMES[_state["field_idx"]]),
    pointa=(0.02, 0.92),
    pointb=(0.45, 0.92),
    style="modern",
)

# --- Depth slider ---
def update_depth(value):
    _state["depth_frac"] = value
    _redraw_slice()

pl.add_slider_widget(
    update_depth,
    [0.02, 0.95],
    value=CLIP_FRACTION,
    title="Depth",
    pointa=(0.55, 0.92),
    pointb=(0.95, 0.92),
    slider_width=0.02,
    tube_width=0.002,
)

# --- Toggle checkboxes ---
def toggle_faults(flag):
    for actor in fault_actors:
        actor.SetVisibility(flag)

def toggle_darcy_arrows(flag):
    if arrow_actor is not None:
        arrow_actor.SetVisibility(flag)

def toggle_stokes_arrows(flag):
    if stokes_arrow_actor is not None:
        stokes_arrow_actor.SetVisibility(flag)

x_pos = 10
checkbox_size = 25

pl.add_checkbox_button_widget(toggle_faults, value=True, size=checkbox_size,
                               position=(x_pos, 10), color_on="red")
pl.add_text("Faults", position=(x_pos, 38), font_size=8, name="lbl_faults")
x_pos += 60

pl.add_checkbox_button_widget(toggle_darcy_arrows, value=False, size=checkbox_size,
                               position=(x_pos, 10), color_on="white")
pl.add_text("Darcy v", position=(x_pos, 38), font_size=8, name="lbl_darcy")
x_pos += 70

if stokes_arrow_actor is not None:
    pl.add_checkbox_button_widget(toggle_stokes_arrows, value=False, size=checkbox_size,
                                   position=(x_pos, 10), color_on="cyan")
    pl.add_text("Stokes v", position=(x_pos, 38), font_size=8, name="lbl_stokes")

pl.add_title(f"H2Ex: azimuth={STRESS_AZIMUTH}, RES={RES}", font_size=10)

# %%
pl.show(jupyter_backend="trame")

# %%
