# %% [markdown]
"""
# Sinking Blob — Adjoint driven from the model transcript

The mathematics is identical to `../supg/inverse_sinker_supg.py`: one implicit
transport step is a residual

    F(beta_new; beta_old, v, dt) = 0,     R_j = integral[ F0 phi_j + F1 . grad(phi_j) ]

so its adjoint is a transpose solve against the SAME Jacobian the SNES already
assembles, and the couplings to the previous level and to the velocity are
SymPy derivatives of `F0` and `F1`.

What changes is where the backward pass gets its states from.

## The old version carried its own recording

    ck = {"B": [...], "V": [...], "P": [...], "dt": [...]}

Four lists of numpy arrays, appended inside the forward loop, holding exactly
the quantities this adjoint turned out to need. That is a recording you can
only write once you already know the adjoint — which is the wrong way round,
and is why an adjoint is normally a rewrite of the forward model rather than an
addition to it. It is also silently incomplete: it does not hold the transport
history, so it only works because backward Euler happens to make `psi_star`
recoverable from `B[k]`.

## This version reads the run's own record

`model.record_every = 1` asks each step to keep the state it started from —
the whole state, fields and transport history and clock together — and
`model.transcript` is the ordered account of what each step did. The backward pass
walks that list:

    for k in reversed(range(len(transcript)))
        model.load_state(transcript[k].snapshot)     # the state step k started from
        adv.solve(timestep=transcript[k].dt)         # replay it: bit-for-bit
        ...

Two things are worth stating plainly about that replay. It is exact —
restoring a snapshot and re-solving reproduces the step to the last bit, where
re-running the script does not, because warm starts and preconditioner reuse
are solver history rather than model state. And it costs one extra transport
solve per step, which is what buys the recording being generic: the run does
not have to know an adjoint is coming.

## The three blocks (unchanged)

For a multiplier `mu` living on the step's unknown, with `mu_h` the finite
element function whose nodal values are `mu`:

    K   = dF/d(beta_new)        the assembled SNES Jacobian
    L^T mu = dual of   (dF0/d beta_old) mu_h + (dF1/d beta_old) . grad(mu_h)
    G^T mu = dual of   (dF0/d v_i)      mu_h + (dF1/d v_i)      . grad(mu_h)

pointwise because `theta = 1` (backward Euler) leaves `beta_old` in the
residual undifferentiated, and because the velocity never enters as `grad(v)`.

## The backward recursion

    RHS_N          = S_N
    K_l^T mu_{l+1} = -RHS_{l+1}                    l = N-1 .. 0
    RHS_l          = S_l + L_l^T mu_{l+1}
    dL/d(beta_0)   = RHS_0

then `dJ/dc = dL/d(beta_0) . d(beta_0)/dc` as a plain dot product, because
`dL/d(beta_0)` is already a dual.
"""

# %%
import numpy as np
import sympy
import underworld3 as uw
from scipy import sparse
from scipy.sparse.linalg import splu

from forward_sinker_transcript import (
    build_model, beta0_nodal, solve_forward, BLOB_CENTER, DT, NSTEPS,
)


# %%
def _jacobian_csr(solver):
    """Assembled Jacobian of a solver's residual, as scipy CSR.

    `uw.systems.Projection`'s SNES Jacobian IS the mass matrix of its space, so
    the same helper gives both the transport Jacobian and the mass matrix used
    to build duals. PETSc leaves the matrix zeroed until a Jacobian evaluation
    is forced, hence `computeJacobian`.
    """
    jac = solver.snes.getJacobian()
    A, P = jac[0], jac[1]
    solver.mesh.update_lvec()
    xv = solver.snes.getSolution().duplicate()
    xv.set(0.0)
    solver.snes.computeJacobian(xv, A, P)
    M = sparse.csr_matrix(A.getValuesCSR()[::-1])
    if M.nnz == 0 or abs(M).max() == 0.0:
        M = sparse.csr_matrix(P.getValuesCSR()[::-1])
    return M


# %%
class AdjointMachinery:
    """Everything the backward pass needs, built once for a model."""

    def __init__(self, model, target_v, stokes_tolerance=1.0e-10):
        mesh = model["mesh"]
        self.model = model
        self.uwmodel = model["uwmodel"]
        self.mesh = mesh
        beta, v = model["beta"], model["v"]
        adv, stokes = model["adv"], model["stokes"]

        # --- multiplier field, and the projection used for every dual ---
        self.mu = uw.discretisation.MeshVariable(
            "mu", mesh, vtype=uw.VarType.SCALAR, degree=3, continuous=True)
        self.scratch = uw.discretisation.MeshVariable(
            "scratch", mesh, vtype=uw.VarType.SCALAR, degree=3, continuous=True)
        self.proj = uw.systems.Projection(mesh, self.scratch)
        self.proj.smoothing = 0.0
        self.proj.tolerance = stokes_tolerance
        self.proj.uw_function = sympy.sympify(1.0)
        self.proj.solve()
        self.M3 = _jacobian_csr(self.proj)

        # --- adjoint Stokes: same operator, homogenised free-slip BCs ---
        self.u_adj = uw.discretisation.MeshVariable("u_adj", mesh, 2, degree=2)
        self.q_adj = uw.discretisation.MeshVariable("q_adj", mesh, 1, degree=1)
        self.f_adj = uw.discretisation.MeshVariable("f_adj", mesh, 2, degree=2)
        sa = uw.systems.Stokes(mesh, velocityField=self.u_adj, pressureField=self.q_adj)
        sa.constitutive_model = uw.constitutive_models.ViscousFlowModel
        sa.constitutive_model.Parameters.shear_viscosity_0 = model["eta"]
        sa.penalty = model["penalty"]
        sa.tolerance = stokes_tolerance
        sa.petsc_options.delValue("ksp_monitor")
        sa.add_essential_bc((sympy.oo, 0.0), "Top")
        sa.add_essential_bc((sympy.oo, 0.0), "Bottom")
        sa.add_essential_bc((0.0, sympy.oo), "Left")
        sa.add_essential_bc((0.0, sympy.oo), "Right")
        sa.bodyforce = self.f_adj.sym
        self.stokes_adj = sa

        # The velocity-block integrand contains grad(mu) of a P3 field, so it
        # does NOT live in the P2 velocity space. The adjoint body force must be
        # its L2 PROJECTION onto that space, not its nodal interpolant.
        self.f_proj_var = uw.discretisation.MeshVariable("f_proj", mesh, 2, degree=2)
        self.f_proj = uw.systems.Vector_Projection(mesh, self.f_proj_var)
        self.f_proj.smoothing = 0.0
        self.f_proj.tolerance = stokes_tolerance

        # --- Stokes sensitivity ---
        dSigma_dbeta = uw.function.derivative(stokes.F1, beta.sym[0])
        d_density_dbeta = uw.function.derivative(model["density"], beta.sym[0])
        self.stokes_sensitivity = (
            uw.maths.tensor.rank2_inner_product(sa.Unknowns.E, dSigma_dbeta)
            + d_density_dbeta * self.u_adj.sym[1]
        )

        # --- transport residual derivatives (built by refresh, after the run) ---
        self.psi_star = adv.Unknowns.DuDt.psi_star[0]
        self.hist_integrand = None
        self.vel_integrand = None

        # --- misfit ---
        self.v_target = target_v
        self.misfit_integrand = sympy.Rational(1, 2) * (
            (v.sym - target_v.sym).dot(v.sym - target_v.sym))

        # --- the run being differentiated (set by `attach`) ---
        self.transcript = []
        self.final_state = None

    # ------------------------------------------------------------------
    def attach(self, transcript, final_state):
        """Point the backward pass at a completed forward run.

        `transcript` is `model.transcript` — one entry per step, each holding the
        interval and the state the step started from. `final_state` is the
        snapshot taken after the last step, which is the one state the transcript
        does not hold: the transcript records STEPS, and there are N+1 levels to
        an N-step run.

        Also (re)builds the transport-residual derivatives. `adv.F0` / `adv.F1`
        are LIVE templates that re-evaluate when the solver's parameters
        change, so reading `.sym` before the solver has been configured by a
        solve snapshots a residual with the WRONG timestep.
        """
        self.transcript = list(transcript)
        self.final_state = final_state

        mesh, adv, v = self.mesh, self.model["adv"], self.model["v"]
        F0 = adv.F0.sym[0, 0]
        F1 = adv.F1.sym
        mu_s = self.mu.sym[0]
        grad_mu = mesh.vector.gradient(mu_s)

        def contract(wrt):
            """(dF0/dwrt) mu + (dF1/dwrt) . grad(mu): the integrand whose dual
            is the transposed block applied to mu."""
            d0 = uw.function.derivative(F0, wrt)
            d1 = uw.function.derivative(F1, wrt)
            out = d0 * mu_s
            for i in range(mesh.dim):
                out = out + d1[i] * grad_mu[i]
            return out

        self.hist_integrand = contract(self.psi_star.sym[0])
        self.vel_integrand = [contract(v.sym[i]) for i in range(mesh.dim)]

    # ------------------------------------------------------------------
    def dual_of(self, expr):
        """integral[ expr * phi_j ] for every P3 basis function phi_j."""
        self.proj.uw_function = expr
        self.proj.solve()
        return self.M3 @ np.asarray(self.scratch.array)[:, 0, 0]

    def state_at(self, level):
        """Restore the model to time level `level` of the recorded run.

        Level `l` for `l < N` is the state step `l` started from, which the
        transcript holds. Level `N` is the state the run finished in.

        The clock comes back with the fields, so `model.tracker.time` follows
        the backward pass — which is a small thing, but it means a diagnostic
        written during the adjoint is labelled with the time it belongs to.
        """
        if level < len(self.transcript):
            self.uwmodel.load_state(self.transcript[level].snapshot)
        else:
            self.uwmodel.load_state(self.final_state)

    def linearisation_state(self, k):
        """Restore the point step `k`'s transport residual was linearised at.

        The residual of step `k` is `F(beta_{k+1}; beta_k, V_k, dt)`, so the
        model must hold the step's OUTPUT in the unknown and the step's INPUT
        in the history slot. The transcript snapshot gives the input; replaying
        the transport solve gives the output, bit for bit. The solve's
        post-hook then shifts the history forward, so the input is put back
        where the residual reads it.
        """
        entry = self.transcript[k]
        beta, adv = self.model["beta"], self.model["adv"]

        self.state_at(k)
        beta_in = np.asarray(beta.array)[:, 0, 0].copy()
        adv.solve(timestep=entry.dt, zero_init_guess=False)
        self.psi_star.array[:, 0, 0] = beta_in

    def transport_jacobian(self, k):
        """K_k = dF/d(beta_{k+1}) for transport step k, at that step's state."""
        self.linearisation_state(k)
        return _jacobian_csr(self.model["adv"])

    def transport_duals(self, k, mu_dual):
        """The adjoint of ONE transport step, applied to multiplier `mu_dual`.

        Returns (L^T mu, G^T mu): the dual on the previous level `beta_k`, and
        the dual on the velocity `V_k` as NODAL VALUES of a body-force field,
        which is what the adjoint Stokes solve wants. This is the operation a
        `solver.adjoint(...)` method would provide.
        """
        self.linearisation_state(k)
        self.mu.array[:, 0, 0] = mu_dual
        hist_dual = self.dual_of(self.hist_integrand)
        self.f_proj.uw_function = sympy.Matrix([self.vel_integrand])
        self.f_proj.solve()
        vel_field = np.asarray(self.f_proj_var.array)[:, 0, :].copy()
        return hist_dual, vel_field

    def stokes_dual(self, level, bodyforce_field, cold=False):
        """Solve the adjoint Stokes problem at `level` with the given body-force
        NODAL VALUES, and return the dual of the resulting beta-sensitivity."""
        self.state_at(level)
        self.f_adj.array[:, 0, :] = bodyforce_field
        self.stokes_adj.solve(zero_init_guess=cold)
        return self.dual_of(self.stokes_sensitivity)

    def misfit(self):
        """J at the run's final state."""
        self.state_at(len(self.transcript))
        return float(uw.maths.Integral(self.mesh, self.misfit_integrand).evaluate())


# %%
def compute_adjoint_gradient(machinery, centre):
    """Backward pass over the recorded run. Returns (dJ/dcx0, dJ/dcy0, J)."""
    m = machinery.model
    v = m["v"]
    N = len(machinery.transcript)

    J = machinery.misfit()

    # dJ/dv_N is M_v (v_N - v_target), so the body-force FIELD is -(v_N - v_target)
    V_N = np.asarray(v.array)[:, 0, :].copy()
    VT = np.asarray(machinery.v_target.array)[:, 0, :]
    rhs = machinery.stokes_dual(N, -(V_N - VT), cold=True)

    for level in range(N - 1, -1, -1):
        K = machinery.transport_jacobian(level)
        mu = splu(K.T.tocsc()).solve(-rhs)
        hist_dual, vel_field = machinery.transport_duals(level, mu)
        rhs = machinery.stokes_dual(level, -vel_field) + hist_dual

    _, dbeta0_dc = beta0_nodal(m, centre)
    return float(rhs @ dbeta0_dc[:, 0]), float(rhs @ dbeta0_dc[:, 1]), J
