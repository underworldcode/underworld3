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
    vec2   SolNL_velocity(  double eta0, unsigned n, double r, double x, double y )
    vec2   SolNL_bodyforce( double eta0, unsigned n, double r, double x, double y )
    double SolNL_viscosity( double eta0, unsigned n, double r, double x, double y )

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