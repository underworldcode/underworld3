r"""Streamline-upwind Petrov-Galerkin scalar transport."""

from typing import Optional

import numpy as np
import sympy

import underworld3 as uw
import underworld3.timing as timing
from underworld3.function import expression
from underworld3.systems.ddt import Eulerian as Eulerian_DDt
from underworld3.systems.solvers import SNES_Diffusion, _centroid_velocities_nd


class SNES_AdvectionDiffusionSUPG(SNES_Diffusion):
    r"""Implicit scalar advection-diffusion with SUPG stabilization.

    The scalar residual is

    .. math::

       R = \frac{\mathrm{BDF}(T)}{\Delta t}
           + \mathbf{u}\cdot\nabla T - f,

    with pointwise residual terms

    .. math::

       F_0 = R, \qquad
       \mathbf{F}_1 = \boldsymbol{\kappa}\nabla T
                       + \tau\mathbf{u}R.

    The second contribution to :math:`\mathbf{F}_1` produces
    :math:`\tau(\mathbf{u}\cdot\nabla w)R` in the weak form. Diffusion
    remains standard Galerkin.

    Parameters
    ----------
    mesh : Mesh
        Computational mesh.
    u_Field : MeshVariable
        Scalar field being transported.
    V_fn : MeshVariable or sympy Matrix
        Prescribed advection velocity. It is frozen during each solve.
    order : int, default=1
        Eulerian BDF history order.
    theta : float, default=1.0
        Flux integration parameter. Only fully implicit fluxes are currently
        supported.
    tau : scalar expression, optional
        User-provided stabilization parameter. When omitted, a transient
        isotropic parameter is computed from a cell-constant streamline
        length, local velocity, diffusivity, and timestep.
    tau_model : {"generic", "citcoms"}, default="generic"
        Automatic stabilization model. ``generic`` uses the optimal 1-D
        coth(Pe)-1/Pe relation with a transient scale. ``citcoms`` uses the
        clipped steady relation on simplex streamline lengths. This option
        does not change the implicit BDF time integrator.
    DuDt, DFDt : optional
        Existing history operators. A supplied ``DuDt`` must be Eulerian and
        must not contain a velocity, because advection is represented in R.

    Notes
    -----
    Automatic tau currently supports volume simplex meshes and scalar
    isotropic diffusivity. Supply ``tau`` explicitly for other meshes or
    constitutive models. This class provides the generic implicit SUPG path;
    it is not CitcomS's row-lumped predictor-corrector time integrator.
    """

    @timing.routine_timer_decorator
    def __init__(
        self,
        mesh: uw.discretisation.Mesh,
        u_Field: uw.discretisation.MeshVariable,
        V_fn,
        order: int = 1,
        theta: float = 1.0,
        tau=None,
        tau_model: str = "generic",
        evalf: Optional[bool] = False,
        verbose: bool = False,
        DuDt: Optional[Eulerian_DDt] = None,
        DFDt=None,
    ):
        if float(theta) != 1.0:
            raise ValueError("AdvDiffusionSUPG currently requires theta=1.0.")
        if tau_model not in ("generic", "citcoms"):
            raise ValueError("tau_model must be 'generic' or 'citcoms'.")
        if DuDt is not None and not isinstance(DuDt, Eulerian_DDt):
            raise TypeError("AdvDiffusionSUPG requires an Eulerian DuDt operator.")
        if DuDt is not None and DuDt.V_fn is not None:
            raise ValueError(
                "DuDt.V_fn must be None; AdvDiffusionSUPG includes advection "
                "in its strong residual."
            )

        super().__init__(
            mesh,
            u_Field,
            order=order,
            theta=theta,
            evalf=evalf,
            verbose=verbose,
            DuDt=DuDt,
            DFDt=DFDt,
        )

        self.V_fn = V_fn
        self.tau_model = tau_model
        self._automatic_tau = tau is None
        self._supg_h = None
        self._supg_tau = None

        if self._automatic_tau:
            suffix = self.instance_number
            self._supg_h = uw.discretisation.MeshVariable(
                f"_supg_h_{suffix}", mesh, 1, degree=0, continuous=False
            )
            self._supg_tau = uw.discretisation.MeshVariable(
                f"_supg_tau_{suffix}", mesh, 1, degree=0, continuous=False
            )
            self._tau = self._supg_tau.sym[0]
        else:
            self._tau = sympy.sympify(tau)

    @property
    def V_fn(self):
        """Advection velocity expression."""
        return self._V_fn

    @V_fn.setter
    def V_fn(self, value):
        self.is_setup = False
        self._V_fn = (
            value.sym
            if isinstance(value, uw.discretisation.MeshVariable)
            else value
        )

    @property
    def tau(self):
        """SUPG stabilization parameter used in the residual."""
        return self._tau

    def _strong_transport_residual(self):
        gradient = self.mesh.vector.gradient(self.u.sym)
        advection = sympy.Matrix((self.V_fn.dot(gradient),))
        return self.DuDt.bdf() / self.delta_t + advection - self.f

    @property
    def F0(self):
        """Transient-advection-source strong residual."""
        value = expression(
            r"f_0^{SUPG}",
            self._strong_transport_residual(),
            "SUPG transient-advection-source residual",
            _unique_name_generation=True,
        )
        self._f0 = value
        return value

    @property
    def F1(self):
        """Galerkin diffusion flux plus streamline stabilization flux."""
        residual = self._strong_transport_residual()[0]
        value = expression(
            r"\mathbf{F}_1^{SUPG}",
            self.DFDt.adams_moulton_flux() + self.tau * self.V_fn * residual,
            "Diffusive and SUPG streamline flux",
            _unique_name_generation=True,
        )
        self._f1 = value
        return value

    def _update_automatic_tau(self):
        """Update local simplex streamline lengths and automatic tau values."""
        if not self._automatic_tau:
            return
        if self.constitutive_model is None:
            raise RuntimeError("Set constitutive_model before solving AdvDiffusionSUPG.")

        from underworld3.meshing.smoothing import _tet_cells, _tri_cells

        if self.mesh.dim == 2:
            cells = _tri_cells(self.mesh.dm)
        elif self.mesh.dim == 3:
            cells = _tet_cells(self.mesh.dm)
        else:
            cells = None
        if cells is None or self.mesh.dim != self.mesh.cdim:
            raise NotImplementedError(
                "Automatic SUPG tau currently requires a 2-D or 3-D volume simplex mesh."
            )

        coords = np.asarray(self.mesh.X.coords)
        cell_coords = coords[cells]
        edges = cell_coords[:, 1:, :] - cell_coords[:, :1, :]
        try:
            inverse_edges = np.linalg.inv(edges)
        except np.linalg.LinAlgError as error:
            raise RuntimeError("Cannot compute SUPG length on a singular simplex.") from error

        gradients = np.empty_like(cell_coords)
        gradients[:, 1:, :] = np.transpose(inverse_edges, (0, 2, 1))
        gradients[:, 0, :] = -gradients[:, 1:, :].sum(axis=1)

        velocity = _centroid_velocities_nd(self.V_fn, self.mesh)
        speed = np.linalg.norm(velocity, axis=1)
        directional_rate = np.abs(
            np.einsum("cad,cd->ca", gradients, velocity)
        ).sum(axis=1)
        h_stream = np.divide(
            2.0 * speed,
            directional_rate,
            out=np.zeros_like(speed),
            where=directional_rate > 0.0,
        )

        diffusivity_expr = sympy.sympify(self.constitutive_model.K)
        if isinstance(diffusivity_expr, sympy.MatrixBase):
            raise NotImplementedError(
                "Automatic SUPG tau requires scalar isotropic diffusivity; "
                "supply tau explicitly for tensor diffusivity."
            )
        diffusivity = uw.function.evaluate(diffusivity_expr, self.mesh._centroids)
        if hasattr(diffusivity, "units") and diffusivity.units is not None:
            diffusivity = uw.non_dimensionalise(diffusivity)
        elif hasattr(diffusivity, "magnitude"):
            diffusivity = diffusivity.magnitude
        diffusivity = np.asarray(diffusivity, dtype=float).reshape(-1)
        if diffusivity.size == 1:
            diffusivity = np.full_like(speed, diffusivity.item())
        if diffusivity.shape != speed.shape:
            raise ValueError("Diffusivity must evaluate to one scalar per simplex cell.")
        if np.any(diffusivity < 0.0):
            raise ValueError("SUPG diffusivity must be non-negative.")

        tau_steady = np.zeros_like(speed)
        moving = speed > np.finfo(float).eps
        diffusive = moving & (diffusivity > 0.0)
        nondiffusive = moving & ~diffusive

        if np.any(diffusive):
            pe = (
                speed[diffusive]
                * h_stream[diffusive]
                / (2.0 * diffusivity[diffusive])
            )
            xi = np.empty_like(pe)
            small = np.abs(pe) < 1.0e-3
            pe_small = pe[small]
            xi[small] = (
                pe_small / 3.0
                - pe_small**3 / 45.0
                + 2.0 * pe_small**5 / 945.0
            )
            xi[~small] = 1.0 / np.tanh(pe[~small]) - 1.0 / pe[~small]
            if self.tau_model == "generic":
                tau_steady[diffusive] = (
                    h_stream[diffusive] * xi / (2.0 * speed[diffusive])
                )
            else:
                tau_steady[diffusive] = (
                    h_stream[diffusive]
                    * np.maximum(0.0, 1.0 - 1.0 / pe)
                    / (2.0 * speed[diffusive])
                )
        tau_steady[nondiffusive] = h_stream[nondiffusive] / (2.0 * speed[nondiffusive])

        if self.tau_model == "generic":
            dt = float(self.delta_t.data)
            if dt <= 0.0:
                raise ValueError("AdvDiffusionSUPG requires a positive timestep.")
            tau_values = np.divide(
                1.0,
                np.sqrt(
                    (2.0 / dt) ** 2
                    + np.divide(
                        1.0,
                        tau_steady**2,
                        out=np.full_like(tau_steady, np.inf),
                        where=tau_steady > 0.0,
                    )
                ),
                out=np.zeros_like(tau_steady),
                where=tau_steady > 0.0,
            )
        else:
            tau_values = tau_steady

        if self._supg_h.data.shape[0] != h_stream.size:
            raise RuntimeError("SUPG P0 field and local simplex counts do not match.")
        self._supg_h.data[:, 0] = h_stream
        self._supg_tau.data[:, 0] = tau_values

    @timing.routine_timer_decorator
    def solve(
        self,
        zero_init_guess: bool = None,
        timestep: float = None,
        evalf: bool = False,
        _force_setup: bool = False,
        verbose: bool = False,
        divergence_retries: int = 0,
    ):
        """Update automatic stabilization and solve one implicit timestep."""
        if timestep is not None:
            self.delta_t = timestep
        self._update_automatic_tau()
        return super().solve(
            zero_init_guess=zero_init_guess,
            timestep=timestep,
            evalf=evalf,
            _force_setup=_force_setup,
            verbose=verbose,
            divergence_retries=divergence_retries,
        )
