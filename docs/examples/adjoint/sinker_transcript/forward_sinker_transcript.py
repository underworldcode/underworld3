# %% [markdown]
"""
# Sinking Blob — Forward Model, written in the timestepping pattern

Same physics as `../supg/forward_sinker_supg.py` and the same Eulerian SUPG
transport. What changes is the *scaffolding*, and only the scaffolding:

**The model and its reference quantities come first.** Not a wall of `REF_*`
constants and hand-divided ratios — a declaration of the three scales this
problem actually uses (a length, a viscosity, and the lithostatic stress
`rho g L`), after which every number in the script is written in the units it
is quoted in. `500 km`, `1e21 Pa s`, `1.116 Myr`. The nondimensional problem
that reaches the solver is bit-for-bit the one the old script assembled by
hand; the difference is that here the scaling is stated once and checked,
rather than spread over ten module constants.

**The timestep is a `model.step(dt)` block.** Which makes it a transaction:
the clock reads the end of the interval for the whole block (where an implicit
scheme centres its residual), the advance commits only on clean exit, and
everything the block did lands in `model.transcript` in order.

**There is no checkpoint dictionary.** The old script carried its own
`ck = {"B": [...], "V": [...], "P": [...], "dt": [...]}` — a hand-rolled
recording of exactly the arrays the adjoint happened to need, which is a thing
you can only write once you already know what the adjoint is. Here
`model.record_every = 1` asks each step to keep the state it started from, and
that snapshot is the *whole* model state: fields, transport history, clock.
The adjoint in `inverse_sinker_transcript.py` reads the transcript instead.

The three physics choices from the SUPG version are unchanged and still
load-bearing for the adjoint: Eulerian SUPG transport, backward Euler
(`theta = 1`), and a fixed timestep.
"""

# %%
import os
import numpy as np
import sympy
import underworld3 as uw

# --- the scales this problem is written in ----------------------------------
# Three quantities fix the scaling completely: a length, a viscosity, and a
# stress. Density is NOT one of them — it enters the Stokes sinker only as a
# ratio, which is why the old script's REF_DENSITY cancelled out of every
# nondimensional number it produced.
DOMAIN_DEPTH = uw.quantity(500, "km")
REF_VISCOSITY = uw.quantity(1e21, "Pa*s")
REF_DENSITY = uw.quantity(3300, "kg/m**3")
GRAVITY = uw.quantity(9.81, "m/s**2")
LITHOSTATIC_PRESSURE = REF_DENSITY * GRAVITY * DOMAIN_DEPTH

# --- geometry and materials, quoted in their own units ----------------------
RESOLUTION = 16
NSTEPS = 5

DENSITY_BACKGROUND = uw.quantity(3200, "kg/m**3")
DENSITY_BLOCK = uw.quantity(3300, "kg/m**3")

BLOB_CENTER = (uw.quantity(250, "km"), uw.quantity(375, "km"))   # 0.5, 0.75 of the box
BLOB_RADIUS = uw.quantity(50, "km")
SMOOTHING_WIDTH = 1.5 * DOMAIN_DEPTH / RESOLUTION

# Fixed timestep. `stokes.estimate_dt()` at t=0 for the true configuration
# returns about this; fixing it keeps the objective from depending on the
# control through the schedule, which was the dominant defect in the original
# adjoint. 570 dimensionless units of eta/(rho g L) is 1.1159 Myr.
DT = uw.quantity(1.1158811388500671, "Myr")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "forward")


# %%
def build_model(viscosity_contrast, resolution=RESOLUTION, stokes_tolerance=1.0e-10,
                theta=1.0):
    """Declare the model, then the mesh, then the solvers — in that order.

    Reference quantities must precede mesh creation, so the model is the first
    line of the script rather than something the mesh conjures for you.

    Returned as a dict so `inverse_sinker_transcript.py` can reuse the SAME
    objects the forward run used — the adjoint reads the transport solver's
    residual (`adv.F0`, `adv.F1`) and its assembled Jacobian, so it must be the
    very solver that produced the run, not a rebuilt copy.
    """
    uw.reset_default_model()
    uwmodel = uw.get_default_model()
    uwmodel.set_reference_quantities(
        domain_depth=DOMAIN_DEPTH,
        material_viscosity=REF_VISCOSITY,
        lithostatic_pressure=LITHOSTATIC_PRESSURE,
    )

    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=1.0 / resolution,
        regular=False,
        qdegree=3,
    )
    x, y = mesh.X

    v = uw.discretisation.MeshVariable("v", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1)
    beta = uw.discretisation.MeshVariable(
        "beta", mesh, vtype=uw.VarType.SCALAR, degree=3, continuous=True
    )

    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Top")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Bottom")
    stokes.add_dirichlet_bc((0.0, sympy.oo), "Left")
    stokes.add_dirichlet_bc((0.0, sympy.oo), "Right")

    if viscosity_contrast > 1e3:
        penalty = 100.0
    elif viscosity_contrast > 1e1:
        penalty = 10.0
    else:
        penalty = 1.0
    stokes.penalty = penalty
    stokes.tolerance = stokes_tolerance
    stokes.petsc_options.delValue("ksp_monitor")

    # The level set carries a LENGTH (it is a signed distance), so the tanh
    # smoothing width is a length too. With the model declared, that is just
    # what the expression says; without it, both were nondimensional numbers
    # whose relationship to the mesh you had to keep in your head.
    smoothing_nd = _nd(SMOOTHING_WIDTH / DOMAIN_DEPTH)
    indicator = 0.5 * (1.0 - sympy.tanh(beta.sym[0] / smoothing_nd))
    eta = sympy.exp(indicator * sympy.log(viscosity_contrast))
    density_ratio = _nd(DENSITY_BACKGROUND / REF_DENSITY) + indicator * _nd(
        (DENSITY_BLOCK - DENSITY_BACKGROUND) / REF_DENSITY)

    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = eta
    stokes.bodyforce = sympy.Matrix([0, -density_ratio])

    # Eulerian SUPG transport of the level set. AdvDiffusion's default DDt
    # plugin is EulerianSUPG; theta=1 is backward Euler (see module docstring).
    adv = uw.systems.AdvDiffusion(mesh, u_Field=beta, V_fn=v.sym)
    adv.constitutive_model = uw.constitutive_models.DiffusionModel
    adv.constitutive_model.Parameters.diffusivity = 0.0
    adv.Unknowns.DuDt.theta = theta
    adv.tolerance = stokes_tolerance
    adv.petsc_options.delValue("ksp_monitor")

    return dict(uwmodel=uwmodel, mesh=mesh, x=x, y=y, v=v, p=p, beta=beta,
                stokes=stokes, adv=adv, eta=eta, density=density_ratio,
                indicator=indicator, penalty=penalty, contrast=viscosity_contrast)


def _nd(q):
    """The plain number a dimensionless quantity stands for."""
    try:
        return float(q.to("dimensionless").magnitude)
    except AttributeError:
        return float(q)


# %%
def beta0_nodal(model, centre):
    """Initial level set at the beta nodes, and its derivative w.r.t. the centre.

    `centre` is a pair of lengths. The mesh is unit-square in model units, so
    the control is converted once, here, and the gradient this function
    returns is therefore d(beta_0)/d(centre) in the SAME units — which is what
    makes the adjoint's final dot product dimensionally honest.
    """
    scale = _length_scale()
    cx, cy = (_mag(c) / scale for c in centre)
    R = _mag(BLOB_RADIUS) / scale

    X = np.asarray(model["beta"].coords)[:, :2] / scale
    r = np.sqrt((X[:, 0] - cx) ** 2 + (X[:, 1] - cy) ** 2)
    dbeta_dc = np.stack([-(X[:, 0] - cx) / r, -(X[:, 1] - cy) / r], axis=1) / scale
    return r - R, dbeta_dc


def _length_scale():
    """Metres per model length unit."""
    return float(uw.get_default_model().get_fundamental_scales()["length"].to("m").magnitude)


def _mag(q):
    return float(q.to("m").magnitude)


# %%
def solve_forward(model, centre, nsteps=NSTEPS, dt=DT):
    """Run the forward model from `centre`.

    Returns `(transcript, final_state)`. The transcript is the record of the run:
    one entry per step, holding the interval it covered, the operators it
    applied in order, and the state it started from. `final_state` is the one
    state the transcript cannot hold — an N-step run has N+1 time levels, and the
    transcript records steps.
    """
    uwmodel = model["uwmodel"]
    beta, stokes, adv = model["beta"], model["stokes"], model["adv"]

    b0, _ = beta0_nodal(model, centre)
    beta.array[:, 0, 0] = b0
    # The Eulerian history initialises itself only on its FIRST solve, so a
    # solver reused for a second independent run silently carries the previous
    # run's psi_star. An inversion driver runs the forward model many times;
    # reset the history explicitly every time the initial condition is set.
    adv.Unknowns.DuDt.initialise_history()
    stokes.solve(zero_init_guess=True)

    # A new run gets a new transcript and a clock at zero. Without the clear, the
    # transcript would be the concatenation of every run this process has done and
    # rewind() would walk back into the previous one.
    uwmodel.clear_transcript()
    uwmodel.tracker.time = uw.quantity(0.0, "Myr")
    uwmodel.tracker.step = 0
    uwmodel.tracker.dt = None
    uwmodel.record_every = 1          # keep the state every step started from
    uwmodel.record_limit = None       # this run is short; keep all of them

    for _ in range(nsteps):
        with uwmodel.step(dt, label="sink"):
            adv.solve(timestep=dt, zero_init_guess=False)
            stokes.solve(zero_init_guess=False)

    return uwmodel.transcript, uwmodel.save_state()


# %%
if __name__ == "__main__":
    import sys

    contrast = float(sys.argv[1]) if len(sys.argv) > 1 else 1000.0
    model = build_model(contrast)
    transcript, _final = solve_forward(model, BLOB_CENTER)
    uwmodel, mesh, beta = model["uwmodel"], model["mesh"], model["beta"]

    uw.pprint(f"contrast {contrast:g}, dt {DT}, {NSTEPS} steps")
    uw.pprint(f"clock now {uwmodel.tracker.time.to('Myr')}, "
              f"step {uwmodel.tracker.step}")
    uw.pprint("")
    uw.pprint("the transcript:")
    for entry in transcript:
        uw.pprint(f"  {entry}")
    uw.pprint(f"  restorable: {len(uwmodel.restore_points)} of {len(transcript)}")
    uw.pprint("")

    # Where is the interface? Sampled on the vertical centreline. Coordinates
    # are dimensional now, so the sample line is quoted in km.
    scale = _length_scale()
    line = np.column_stack([np.full(400, 250e3), np.linspace(175e3, 475e3, 400)]) / scale

    def crossings():
        vals = np.asarray(uw.function.evaluate(beta.sym[0], line)).ravel()
        sgn = np.where(np.diff(np.sign(vals)) != 0)[0]
        return [f"{line[i, 1] * scale / 1e3:.1f} km" for i in sgn]

    uw.pprint(f"interface at t=T on x=250 km: {crossings()}")

    # And the point of recording it: put the run back one step and look again.
    uwmodel.rewind()
    uw.pprint(f"after rewind(): clock {uwmodel.tracker.time.to('Myr')}, "
              f"step {uwmodel.tracker.step}, transcript {len(uwmodel.transcript)} steps")
    uw.pprint(f"interface one step earlier : {crossings()}")
