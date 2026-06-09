import os
import sympy
import underworld3

# Add info for linking against the Cython compiled module which contains symbols defined below.
libdir = os.path.dirname(__file__)
libfile = os.path.basename(__file__)
# Prepend colon to force linking against filename without 'lib' prefix.
libfile = ":" + libfile  

cdef extern from "AnalyticSolNL.h" nogil:
    ctypedef struct vec2:
        double x
        double z
    ctypedef struct tensor2:
        double xx
        double zz
        double xz
    vec2   SolNL_velocity(  double eta0, unsigned n, double r, double x, double y )
    vec2   SolNL_bodyforce( double eta0, unsigned n, double r, double x, double y )
    double SolNL_viscosity( double eta0, unsigned n, double r, double x, double y )

cdef extern from "AnalyticSolCx.h" nogil:
    # vec2 / tensor2 already declared above (shared typedefs via AnalyticSolNL.h)
    vec2    SolCx_velocity(  double eta_A, double eta_B, double x_c, int n, double x, double z )
    double  SolCx_pressure(  double eta_A, double eta_B, double x_c, int n, double x, double z )
    double  SolCx_viscosity( double eta_A, double eta_B, double x_c, int n, double x, double z )
    tensor2 SolCx_stress(    double eta_A, double eta_B, double x_c, int n, double x, double z )

class sympy_function_printable(sympy.Function):
    """
    This help function simply does most of the work for c-code printing.
    Inherit from this and set `self._printstr` and `self._header` as necessary.
    See `AnalyticSolNL` for example.
    """
    _printstr = None 
    _header = None
    def _ccode(self, printer):
        # Populate linking dictionaries.
        underworld3._libfiles[libfile] = None
        underworld3._libdirs[libdir]   = None
        underworld3._incdirs[libdir]   = None
        printer.headers.add(self._header)
        param_str = ""
        for arg in self.args:
            param_str += printer._print(arg) + ","
        param_str = param_str[:-1]  # drop final comma
        if not self._printstr:
            raise RuntimeError("Trying to print unprintable function.")
        return self._printstr.format(param_str)

class AnalyticSolNL_base(sympy_function_printable):
    nargs = 5
    _header = "AnalyticSolNL.h"

class AnalyticSolNL_velocity_x(AnalyticSolNL_base):
    _printstr = "SolNL_velocity({}).x"
    def _eval_evalf(self,prec):
        from sympy import sympify
        return sympify(SolNL_velocity( self.args[0],self.args[1],self.args[2],self.args[3],self.args[4] ).x)
class AnalyticSolNL_velocity_y(AnalyticSolNL_base):
    _printstr = "SolNL_velocity({}).z"
    def _eval_evalf(self,prec):
        from sympy import sympify
        return sympify(SolNL_velocity( self.args[0],self.args[1],self.args[2],self.args[3],self.args[4] ).z)
class AnalyticSolNL_velocity(AnalyticSolNL_base):
    nargs = 5
    @classmethod
    def eval(cls, *args ):
        from sympy.vector import CoordSys3D
        N = CoordSys3D("N")
        return AnalyticSolNL_velocity_x(*args)*N.i + AnalyticSolNL_velocity_y(*args)*N.j 

class AnalyticSolNL_bodyforce_x(AnalyticSolNL_base):
    _printstr = "SolNL_bodyforce({}).x"
    def _eval_evalf(self,prec):
        from sympy import sympify
        return sympify(SolNL_bodyforce( self.args[0],self.args[1],self.args[2],self.args[3],self.args[4] ).x)
class AnalyticSolNL_bodyforce_y(AnalyticSolNL_base):
    _printstr = "SolNL_bodyforce({}).z"
    def _eval_evalf(self,prec):
        from sympy import sympify
        return sympify(SolNL_bodyforce( self.args[0],self.args[1],self.args[2],self.args[3],self.args[4] ).z)
class AnalyticSolNL_bodyforce(AnalyticSolNL_base):
    nargs = 5
    @classmethod
    def eval(cls, *args ):
        from sympy.vector import CoordSys3D
        N = CoordSys3D("N")
        return AnalyticSolNL_bodyforce_x(*args)*N.i + AnalyticSolNL_bodyforce_y(*args)*N.j 

class AnalyticSolNL_viscosity(AnalyticSolNL_base):
    _printstr = "SolNL_viscosity({})"
    def _eval_evalf(self,prec):
        from sympy import sympify
        return sympify(SolNL_viscosity( self.args[0],self.args[1],self.args[2],self.args[3],self.args[4] ))


# ----------------------------------------------------------------------------
# SolCx: viscosity step in x (eta_A for x<x_c, eta_B for x>=x_c), trigonometric
# density forcing  f = (0, +cos(pi x) sin(n pi z))  on a unit box with free-slip
# walls. The +cos sign matches UW3's momentum convention (UW2 docs quote -cos).
# args: (eta_A, eta_B, x_c, n, x, z)
# ----------------------------------------------------------------------------
class AnalyticSolCx_base(sympy_function_printable):
    nargs = 6
    _header = "AnalyticSolCx.h"

class AnalyticSolCx_velocity_x(AnalyticSolCx_base):
    _printstr = "SolCx_velocity({}).x"
    def _eval_evalf(self,prec):
        from sympy import sympify
        return sympify(SolCx_velocity( self.args[0],self.args[1],self.args[2],self.args[3],self.args[4],self.args[5] ).x)
class AnalyticSolCx_velocity_y(AnalyticSolCx_base):
    _printstr = "SolCx_velocity({}).z"
    def _eval_evalf(self,prec):
        from sympy import sympify
        return sympify(SolCx_velocity( self.args[0],self.args[1],self.args[2],self.args[3],self.args[4],self.args[5] ).z)
class AnalyticSolCx_velocity(AnalyticSolCx_base):
    nargs = 6
    @classmethod
    def eval(cls, *args ):
        from sympy.vector import CoordSys3D
        N = CoordSys3D("N")
        return AnalyticSolCx_velocity_x(*args)*N.i + AnalyticSolCx_velocity_y(*args)*N.j

class AnalyticSolCx_pressure(AnalyticSolCx_base):
    _printstr = "SolCx_pressure({})"
    def _eval_evalf(self,prec):
        from sympy import sympify
        return sympify(SolCx_pressure( self.args[0],self.args[1],self.args[2],self.args[3],self.args[4],self.args[5] ))

class AnalyticSolCx_viscosity(AnalyticSolCx_base):
    _printstr = "SolCx_viscosity({})"
    def _eval_evalf(self,prec):
        from sympy import sympify
        return sympify(SolCx_viscosity( self.args[0],self.args[1],self.args[2],self.args[3],self.args[4],self.args[5] ))

# Exact total (Cauchy) stress components sigma_ij from the Velic kernel's
# total_stress array: xx, zz (= yy in the x-z plane), xz. Dynamic topography on
# the top boundary (n = y_hat) is -n.sigma.n = -sigma_zz.
class AnalyticSolCx_stress_xx(AnalyticSolCx_base):
    _printstr = "SolCx_stress({}).xx"
    def _eval_evalf(self,prec):
        from sympy import sympify
        return sympify(SolCx_stress( self.args[0],self.args[1],self.args[2],self.args[3],self.args[4],self.args[5] ).xx)
class AnalyticSolCx_stress_yy(AnalyticSolCx_base):
    _printstr = "SolCx_stress({}).zz"
    def _eval_evalf(self,prec):
        from sympy import sympify
        return sympify(SolCx_stress( self.args[0],self.args[1],self.args[2],self.args[3],self.args[4],self.args[5] ).zz)
class AnalyticSolCx_stress_xy(AnalyticSolCx_base):
    _printstr = "SolCx_stress({}).xz"
    def _eval_evalf(self,prec):
        from sympy import sympify
        return sympify(SolCx_stress( self.args[0],self.args[1],self.args[2],self.args[3],self.args[4],self.args[5] ).xz)


class SolCx:
    r"""Velic *SolCx* analytic Stokes solution — the canonical free-slip,
    discontinuous-viscosity benchmark: a viscosity step :math:`\eta_A` (for
    :math:`x<x_c`) to :math:`\eta_B` (for :math:`x>x_c`), driven by the density
    forcing :math:`f=(0,\cos(\pi x)\sin(n\pi z))` on the unit box with free-slip
    walls. Use it to validate a UW3 Stokes solve against the exact solution::

        sol = uw.function.analytic.SolCx(mesh, eta_A=1.0, eta_B=1.0e6, x_c=0.5, n=1)
        stokes.constitutive_model.Parameters.shear_viscosity_0 = sol.fn_viscosity
        stokes.bodyforce = sol.fn_bodyforce
        # free-slip on all four walls + pressure nullspace, then solve and:
        rel = sol.velocity_error(stokes.u)     # L2 error vs the analytic

    Notes
    -----
    ``fn_velocity`` / ``fn_pressure`` wrap the compiled Velic kernel and are
    evaluated point-wise via SymPy ``evalf`` (they are not ``lambdify``-able, so
    use :meth:`evaluate_velocity` / :meth:`velocity_error` rather than
    ``uw.function.evaluate``). ``fn_bodyforce`` / ``fn_viscosity`` are elementary
    and compile through the normal JIT path when assigned to a solver.

    The body-force **sign** matches UW3's momentum convention; the original
    Underworld2 documentation quotes the opposite sign.
    """
    def __init__(self, mesh, eta_A=1.0, eta_B=1.0e6, x_c=0.5, n=1):
        import sympy as _sp
        if getattr(mesh, "dim", None) != 2:
            raise ValueError("SolCx is a 2D analytic solution; mesh.dim must be 2.")
        if not (float(eta_A) > 0.0 and float(eta_B) > 0.0):
            raise ValueError("eta_A and eta_B must be positive.")
        if not (0.0 <= float(x_c) <= 1.0):
            raise ValueError("x_c must lie in [0, 1].")
        if int(n) != n or int(n) < 1:
            raise ValueError("n (z-wavenumber) must be a positive integer.")
        self.eta_A = float(eta_A)
        self.eta_B = float(eta_B)
        self.x_c   = float(x_c)
        self.n     = int(n)
        x, y = mesh.X
        p = (self.eta_A, self.eta_B, self.x_c, self.n)
        self.fn_velocity  = _sp.Matrix([AnalyticSolCx_velocity_x(*p, x, y),
                                        AnalyticSolCx_velocity_y(*p, x, y)])
        self.fn_pressure  = AnalyticSolCx_pressure(*p, x, y)
        # exact total (Cauchy) stress components (sigma_xx, sigma_yy, sigma_xy)
        self.fn_stress_xx = AnalyticSolCx_stress_xx(*p, x, y)
        self.fn_stress_yy = AnalyticSolCx_stress_yy(*p, x, y)
        self.fn_stress_xy = AnalyticSolCx_stress_xy(*p, x, y)
        # tie-break at x==x_c matches the compiled SolCx_viscosity (eta_B for x>=x_c)
        self.fn_viscosity = _sp.Piecewise((self.eta_A, x < self.x_c), (self.eta_B, True))
        self.fn_bodyforce = _sp.Matrix([_sp.Integer(0),
                                        _sp.cos(_sp.pi * x) * _sp.sin(self.n * _sp.pi * y)])

    def evaluate_velocity(self, coords):
        """Exact velocity at ``coords`` (N×2 array), via the compiled kernel."""
        import numpy as _np
        p = (self.eta_A, self.eta_B, self.x_c, self.n)
        out = _np.empty((len(coords), 2))
        for i in range(len(coords)):
            xi = float(coords[i, 0]); yi = float(coords[i, 1])
            out[i, 0] = float(AnalyticSolCx_velocity_x(*p, xi, yi).evalf())
            out[i, 1] = float(AnalyticSolCx_velocity_y(*p, xi, yi).evalf())
        return out

    def velocity_error(self, velocity_var):
        """Relative L2 error of a velocity MeshVariable against the analytic."""
        import numpy as _np
        ua = self.evaluate_velocity(velocity_var.coords)
        return float(_np.linalg.norm(velocity_var.data - ua) / _np.linalg.norm(ua))

    def evaluate_stress(self, coords):
        """Exact total (Cauchy) stress at ``coords`` (N×2 array), via the
        compiled kernel. Returns an (N, 3) array of (sigma_xx, sigma_yy,
        sigma_xy)."""
        import numpy as _np
        p = (self.eta_A, self.eta_B, self.x_c, self.n)
        out = _np.empty((len(coords), 3))
        for i in range(len(coords)):
            xi = float(coords[i, 0]); yi = float(coords[i, 1])
            out[i, 0] = float(AnalyticSolCx_stress_xx(*p, xi, yi).evalf())
            out[i, 1] = float(AnalyticSolCx_stress_yy(*p, xi, yi).evalf())
            out[i, 2] = float(AnalyticSolCx_stress_xy(*p, xi, yi).evalf())
        return out

    def topography_top(self, coords):
        """Exact dynamic topography -n.sigma.n on the top boundary (n = y_hat),
        i.e. -sigma_yy, at ``coords`` (N×2 array). Returns a length-N array."""
        return -self.evaluate_stress(coords)[:, 1]