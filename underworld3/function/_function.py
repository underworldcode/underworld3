##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
##                                                                                   ##
##  This file forms part of the Underworld geophysics modelling application.         ##
##                                                                                   ##
##  For full license and copyright information, please refer to the LICENSE.md file  ##
##  located at the project root, or contact the authors.                             ##
##                                                                                   ##
##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
"""
Base module for the Function class. The Function class provides generic
function construction capabilities.

Note that often function objects are defined and used in different locations, 
but function consistency is only able to be tested at usage time. As such, we
record information about where a function was created to help direct the user 
back to this definition during the debugging process. As default, this information
is only recorded on the root process, although you may set the `UW_WORLD_FUNC_MESSAGES`
environment variable if you require all processes to report this information. 
Furthermore, you may disable messages altogether by setting `UW_NO_FUNC_MESSAGES`. This
may be useful for production runs on large systems where the function messages can
cause excessive filesystem chatter. 

"""
import sympy
from functools import wraps

def Function_in_Function_out(func):
    """
    This decorator converts all inputs and outputs to uw3 `Function` objects.
    This is useful for easy Sympy interchangeability. 
    """
    # first check if this is a method or function so that
    # we can handle the first arg accordingly
    # https://stackoverflow.com/questions/2435764/how-to-differentiate-between-method-and-function-in-a-decorator
    method=0
    if func.__name__ != func.__qualname__:
        method=1
    def wrapper(*args):
        # generate wrapped inputs
        newargs = []
        # handle first arg different for methods/functions
        if method:
            newargs.append(args[0])
        else:
            newargs.append(Function(args[0]))
        # now handle other args
        for arg in args[1:]:
            newargs.append(Function(arg))
        # call func with news args
        output = func(*newargs)
        # wrap and return output assuming
        # single output
        return Function(output)
    return wrapper

class Function:
    """
    Objects which inherit from this class provide user definable functions
    within Underworld.

    Functions aim to achieve a number of goals:
    * Provide a natural interface for mathematical behaviour description within python.
    * Provide a high level interface to Underworld discrete objects.
    * Allow discrete objects to be used in combination with continuous objects.
    * Handle the evaluation of discrete objects in the most efficient manner.
    * Perform all heavy calculations at the C-level for efficiency.
    * Provide an interface for users to evaluate functions directly within python, 
    utilising numpy arrays for input/output.
    """
    def __init__(self, fnobj, *args, **kwargs):
        from sympy import sympify
        # The `fnobj` might be of Function class, in which 
        # case we just grab a reference to its underlying Sympy
        # object. If it is not of Function class, then it should be
        # of Sympy or basic (float,ints,etc) class, in which case
        # we call `sympify` which will be redundant if already a 
        # Sympy object, but will otherwise generate the Sympy wrapped
        # equivalent (such as for `sympify(1.234)`).        
        if isinstance(fnobj, Function):
            self._fn = fnobj._fn
        else:
            self._fn = sympify(fnobj)
        super().__init__(*args, **kwargs)

    @property
    def sfn(self):
        """
        Returns the underlying Sympy function.
        """
        return self._fn

    def __repr__(self):
        return sympy.Function.__repr__(self.sfn)

    def __hash__(self):
        return sympy.Function.__hash__(self.sfn)

    @Function_in_Function_out
    def __add__(self,other):
        """
        Operator overloading for '+' operation:

        Fn3 = Fn1 + Fn2

        Returns
        -------
        Function: Add function.

        Examples
        --------
        >>> from underworld3.function import Function
        >>> import numpy as np
        >>> three = Function(3.)
        >>> four  = Function(4.)
        >>> np.allclose( (three + four).evaluate(0.), [[ 7.]]  )  # note we can evaluate anywhere because it's a constant
        True

        """
        return self.sfn + other.sfn

    __radd__=__add__

    @Function_in_Function_out
    def __sub__(self,other):
        """
        Operator overloading for '-' operation:

        Fn3 = Fn1 - Fn2

        Returns
        -------
        Function: Subtract function

        Examples
        --------
        >>> import underworld.function.misc as misc
        >>> import numpy as np
        >>> three = misc.constant(3.)
        >>> four  = misc.constant(4.)
        >>> np.allclose( (three - four).evaluate(0.), [[ -1.]]  )  # note we can evaluate anywhere because it's a constant
        True

        """
        return self.sfn - other.sfn

    @Function_in_Function_out
    def __rsub__(self,other):
        """
        Operator overloading for '-' operation.  Right hand version.

        Fn3 = Fn1 - Fn2

        Returns
        -------
        Function: RHS subtract function

        Examples
        --------
        >>> import underworld.function.misc as misc
        >>> import numpy as np
        >>> four  = misc.constant(4.)
        >>> np.allclose( (5. - four).evaluate(0.), [[ 1.]]  )  # note we can evaluate anywhere because it's a constant
        True

        """
        return other.sfn - self.sfn

    @Function_in_Function_out
    def  __neg__(self):
        """
        Operator overloading for unary '-'.

        FnNeg = -Fn

        Returns
        -------
        Function: Negative function

        Examples
        --------
        >>> import underworld.function.misc as misc
        >>> import numpy as np
        >>> four = misc.constant(4.)
        >>> np.allclose( (-four).evaluate(0.), [[ -4.]]  )  # note we can evaluate anywhere because it's a constant
        True

        """
        return -self.sfn

    @Function_in_Function_out
    def __mul__(self,other):
        """
        Operator overloading for '*' operation:

        Fn3 = Fn1 * Fn2

        Returns
        -------
        Function: Multiply function

        Examples
        --------
        >>> import underworld.function.misc as misc
        >>> import numpy as np
        >>> three = misc.constant(3.)
        >>> four  = misc.constant(4.)
        >>> np.allclose( (three*four).evaluate(0.), [[ 12.]]  )  # note we can evaluate anywhere because it's a constant
        True

        """
        return self.sfn * other.sfn
    __rmul__ = __mul__

    @Function_in_Function_out
    def __truediv__(self,other):
        """
        Operator overloading for '/' operation:

        Fn3 = Fn1 / Fn2

        Returns
        -------
        Function: Divide function

        Examples
        --------
        >>> import underworld.function.misc as misc
        >>> import numpy as np
        >>> two  = misc.constant(2.)
        >>> four = misc.constant(4.)
        >>> np.allclose( (four/two).evaluate(0.), [[ 2.]]  )  # note we can evaluate anywhere because it's a constant
        True

        """
        return self.sfn / other.sfn

    @Function_in_Function_out
    def __rtruediv__(self,other):
        return other.sfn / self.sfn

    @Function_in_Function_out
    def __pow__(self,other):
        """
        Operator overloading for '**' operation:

        Fn3 = Fn1 ** Fn2

        Returns
        -------
        Function: Power function

        Examples
        --------
        >>> import underworld.function.misc as misc
        >>> import numpy as np
        >>> two  = misc.constant(2.)
        >>> four = misc.constant(4.)
        >>> np.allclose( (two**four).evaluate(0.), [[ 16.]]  )  # note we can evaluate anywhere because it's a constant
        True

        """
        return self.sfn**other.sfn

    @Function_in_Function_out
    def __lt__(self,other):
        """
        Operator overloading for '<' operation:

        Fn3 = Fn1 < Fn2

        Returns
        -------
        Function: Less than function

        Examples
        --------
        >>> import underworld.function.misc as misc
        >>> import numpy as np
        >>> two  = misc.constant(2.)
        >>> four = misc.constant(4.)
        >>> (two < four).evaluate()
        array([[ True]], dtype=bool)

        """
        return self.sfn < other.sfn

    @Function_in_Function_out
    def __le__(self,other):
        """
        Operator overloading for '<=' operation:

        Fn3 = Fn1 <= Fn2

        Returns
        -------
        Function: Less than or equal to function

        Examples
        --------
        >>> import underworld.function.misc as misc
        >>> import numpy as np
        >>> two  = misc.constant(2.)
        >>> (two <= two).evaluate()
        array([[ True]], dtype=bool)

        """
        return self.sfn <= other.sfn

    @Function_in_Function_out
    def __gt__(self,other):
        """
        Operator overloading for '>' operation:

        Fn3 = Fn1 > Fn2

        Returns
        -------
        Function: Greater than function

        Examples
        --------
        >>> import underworld.function.misc as misc
        >>> import numpy as np
        >>> two  = misc.constant(2.)
        >>> four = misc.constant(4.)
        >>> (two > four).evaluate()
        array([[False]], dtype=bool)

        """
        return self.sfn > other.sfn

    @Function_in_Function_out
    def __ge__(self,other):
        """
        Operator overloading for '>=' operation:

        Fn3 = Fn1 >= Fn2

        Returns
        -------
        Function: Greater than or equal to function

        Examples
        --------
        >>> import underworld.function.misc as misc
        >>> import numpy as np
        >>> two  = misc.constant(2.)
        >>> (two >= two).evaluate()
        array([[ True]], dtype=bool)

        """
        return self.sfn >= other.sfn

    # def __and__(self,other):
    #     """
    #     Operator overloading for '&' operation:

    #     Fn3 = Fn1 & Fn2

    #     Creates a new function Fn3 which returns a bool result for the operation.

    #     Returns
    #     -------
    #     fn.logical_and: AND function

    #     Examples
    #     --------
    #     >>> import underworld.function.misc as misc
    #     >>> trueFn  = misc.constant(True)
    #     >>> falseFn = misc.constant(False)
    #     >>> (trueFn & falseFn).evaluate()
    #     array([[False]], dtype=bool)
        
    #     Notes
    #     -----
    #     The '&' operator in python is usually used for bitwise 'and' operations, with the 
    #     'and' operator used for boolean type operators. It is not possible to overload the
    #     'and' operator in python, so instead the bitwise equivalent has been utilised.

    #     """
    #     return logical_and( self, other )

    # def __or__(self,other):
    #     """
    #     Operator overloading for '|' operation:

    #     Fn3 = Fn1 | Fn2

    #     Creates a new function Fn3 which returns a bool result for the operation.

    #     Returns
    #     -------
    #     fn.logical_or: OR function

    #     Examples
    #     --------
    #     >>> import underworld.function.misc as misc
    #     >>> trueFn  = misc.constant(True)
    #     >>> falseFn = misc.constant(False)
    #     >>> (trueFn | falseFn).evaluate()
    #     array([[ True]], dtype=bool)

    #     Notes
    #     -----
    #     The '|' operator in python is usually used for bitwise 'or' operations, 
    #     with the 'or' operator used for boolean type operators. It is not possible 
    #     to overload the 'or' operator in python, so instead the bitwise equivalent 
    #     has been utilised.


    #     """

    #     return logical_or( self, other )

    # def __xor__(self,other):
    #     """
    #     Operator overloading for '^' operation:

    #     Fn3 = Fn1 ^ Fn2

    #     Creates a new function Fn3 which returns a bool result for the operation.

    #     Returns
    #     -------
    #     fn.logical_xor: XOR function

    #     Examples
    #     --------
    #     >>> import underworld.function.misc as misc
    #     >>> trueFn  = misc.constant(True)
    #     >>> falseFn = misc.constant(False)
    #     >>> (trueFn ^ falseFn).evaluate()
    #     array([[ True]], dtype=bool)
    #     >>> (trueFn ^ trueFn).evaluate()
    #     array([[False]], dtype=bool)
    #     >>> (falseFn ^ falseFn).evaluate()
    #     array([[False]], dtype=bool)

    #     Notes
    #     -----
    #     The '^' operator in python is usually used for bitwise 'xor' operations, 
    #     however here we always use the logical version, with the operation 
    #     inputs cast to their bool equivalents before the operation.  


    #     """

    #     return logical_xor( self, other )

    def __getitem__(self,index):
        """
        Operator overloading for '[]' operation:

        FnComponent = Fn[0]

        Returns
        -------
        Function: component function

        Examples
        --------
        >>> import underworld.function.misc as misc
        >>> fn  = misc.constant((2.,3.,4.))
        >>> np.allclose( fn[1].evaluate(0.), [[ 3.]]  )  # note we can evaluate anywhere because it's a constant
        True

        """
        return Function(self.sfn[index])

    # def evaluate_global(self, inputData, inputType=None):
    #     """
    #     This method attempts to evalute inputData across all processes, and 
    #     then consolide the results on the root processor. This is most useful
    #     where you wish to evalute your functions using global coordinates 
    #     which may span processes in a parallel simulation.
        
    #     Note that this method does not currently support 'FunctionInput' class
    #     input data.
        
    #     Due to the communications required for this method, a significant 
    #     performance overhead may be encountered. The standard `evaluate` method 
    #     should be used instead wherever possible.

    #     Please see `evaluate` method for parameter details.

    #     Notes
    #     -----
    #     This method must be called collectively by all processes.
        
    #     Returns
    #     -------
    #     Only the root process gets the final results array. All other processes
    #     are returned None.

    #     """
    #     from mpi4py import MPI
    #     comm = MPI.COMM_WORLD
    #     rank = comm.Get_rank()
    #     nprocs = comm.Get_size()

    #     if isinstance(inputData, FunctionInput):
    #         raise TypeError("This 'inputData' type is not currently supported for global function evaluation.")
    #     # go through the inputData and fill elements where the data is available
    #     if not isinstance(inputData, np.ndarray):
    #         inputData = self._evaluate_data_convert_to_ndarray(inputData)
    #     arrayLength = len(inputData)
    #     local =  np.zeros(arrayLength, dtype=bool)
    #     local_output = None
    #     for i in range(arrayLength):
    #         try:
    #             # get result
    #             output = self.evaluate(inputData[i:i+1], inputType)
    #             # flag as result found
    #             local[i]  = True
    #             # if not created, create
    #             if not isinstance(local_output, np.ndarray):
    #                 local_output =  np.zeros( (arrayLength, output.shape[1]), dtype=output.dtype)
    #             local_output[i] = output
    #         except ValueError:
    #             # ValueError is only raised for outside domain, which suggests that the
    #             # evaluation probably occurred on another process.
    #             pass
    #         except:
    #             # if a different error was raise, we should reraise
    #             raise

    #     # distill results down to local only
    #     local_result_count = np.count_nonzero(local)
    #     if local_result_count:
    #         local_output_distilled = np.zeros( (local_result_count, local_output.shape[1]), dtype=local_output.dtype)
    #         array_positions        = np.zeros(local_result_count, dtype=int)
    #         j=0
    #         for i,val in enumerate(local):
    #             if val:
    #                 array_positions[j]=i
    #                 local_output_distilled[j] = local_output[i]
    #                 j+=1

    #     # data sending
    #     total_output = None
    #     if(rank!=0):
    #         # send count
    #         comm.send(local_result_count, dest=0, tag=0)
    #         if local_result_count:
    #             # next send position array
    #             comm.send(array_positions, dest=0, tag=1)
    #             # finally send actual data
    #             comm.send(local_output_distilled,    dest=0, tag=2)
    #     else:
    #         # have output already from rank=0 proc; and lots of empties to fill in from others
    #         # some data IS available two multiple processors - e.g. edges
    #         for iProc in range(1,nprocs):
    #             incoming_count = comm.recv(source=iProc, tag=0)
    #             if incoming_count:
    #                 incoming_positions = comm.recv(source=iProc, tag=1)
    #                 incoming_data      = comm.recv(source=iProc, tag=2)
    #                 # create array if not done already
    #                 if not isinstance(total_output, np.ndarray):
    #                     total_output =  np.zeros( (arrayLength, incoming_data.shape[1]), dtype=incoming_data.dtype)
    #                 total_output[incoming_positions] = incoming_data

    #     # finally copy our local results into the output
    #     if (rank==0) and local_result_count:
    #         if not isinstance(total_output,np.ndarray):
    #             total_output =  np.zeros( (arrayLength,local_output_distilled.shape[1]), dtype=local_output_distilled.dtype)
    #         total_output[array_positions] = local_output_distilled

    #     if (rank==0) and (isinstance(total_output,np.ndarray)==False):
    #         # if total_output is still non-existent, no results were found
    #         raise RuntimeError("No results were found anywhere in the domain for provided input.")

    #     if rank == 0:
    #         return total_output
    #     else:
    #         # all other procs return None
    #         return None


    # def _evaluate_data_convert_to_ndarray( self, inputData ):
    #     # convert single values to tuples if necessary
    #     if isinstance( inputData, float ):
    #         inputData = (inputData,)
    #     # convert to ndarray
    #     if isinstance( inputData, (list,tuple) ):
    #         arr = np.empty( [1,len(inputData)] )
    #         ii = 0
    #         for guy in inputData:
    #             if not isinstance(guy, float):
    #                 raise TypeError("Iterable inputs must only contain python 'float' objects.")
    #             arr[0,ii] = guy
    #             ii +=1
    #         return arr
    #     else:
    #         raise TypeError("Input provided for function evaluation does not appear to be supported.")

    # # def integrate_fn( self, mesh ):
    # def integrate( self, mesh ):
    #     """
    #     Perform an integral of this underworld function over the given mesh

    #     Parameters
    #     ----------
    #     mesh : uw.mesh.FeMesh_Cartesian
    #         Domain to perform integral over.

    #     Examples
    #     --------

    #     >>> mesh = uw.mesh.FeMesh_Cartesian(minCoord=(0.0,0.0), maxCoord=(1.0,2.0))
    #     >>> fn_1 = uw.function.misc.constant(2.0)
    #     >>> np.allclose( fn_1.integrate( mesh )[0], 4 )
    #     True

    #     >>> fn_2 = uw.function.misc.constant(2.0) * (0.5, 1.0)
    #     >>> np.allclose( fn_2.integrate( mesh ), [2,4] )
    #     True

    #     """

    #     if not isinstance(mesh, uw.mesh.FeMesh_Cartesian):
    #         raise RuntimeError("Error: integrate() is only available on meshes of type 'FeMesh_Cartesian'")
    #     return mesh.integrate( fn=self )

    # def evaluate(self,inputData=None,inputType=None):
    #     """
    #     This method performs evaluate of a function at the given input(s).

    #     It accepts floats, lists, tuples, numpy arrays, or any object which is of
    #     class `FunctionInput`. lists/tuples must contain floats only.

    #     `FunctionInput` class objects are shortcuts to their underlying data, often
    #     with performance advantages, and sometimes they are the only valid input
    #     type (such as using `Swarm` objects as an inputs to `SwarmVariable`
    #     evaluation). Objects of class `FeMesh`, `Swarm`, `FeMesh_IndexSet` and
    #     `VoronoiIntegrationSwarm` are also of class `FunctionInput`. See the
    #     Function section of the user guide for more information.

    #     Results are returned as numpy array.

    #     Parameters
    #     ----------
    #     inputData: float, list, tuple, ndarray, underworld.function.FunctionInput
    #         The input to the function. The form of this input must be appropriate
    #         for the function being evaluated, or an exception will be thrown.
    #         Note that if no input is provided, function will be evaluated at `0.`
    #     inputType: str
    #         Specifies the type the provided data represents. Acceptable 
    #         values are 'scalar', 'vector', 'symmetrictensor', 'tensor',
    #         'array'.

    #     Returns
    #     -------
    #     ndarray: array of results

    #     Examples
    #     --------
    #     >>> import math as sysmath
    #     >>> import underworld.function.math as fnmath
    #     >>> sinfn = fnmath.sin()
        
    #     Single evaluation:
        
    #     >>> np.allclose( sinfn.evaluate(sysmath.pi/4.), [[ 0.5*sysmath.sqrt(2.)]]  )
    #     True
        
    #     Multiple evaluations
        
    #     >>> input = (0.,sysmath.pi/4.,2.*sysmath.pi)
    #     >>> np.allclose( sinfn.evaluate(input), [[ 0., 0.5*sysmath.sqrt(2.), 0.]]  )
    #     True
        
        
    #     Single MeshVariable evaluations
        
    #     >>> mesh = uw.mesh.FeMesh_Cartesian()
    #     >>> var = uw.mesh.MeshVariable(mesh,1)
    #     >>> import numpy as np
    #     >>> var.data[:,0] = np.linspace(0,1,len(var.data))
    #     >>> result = var.evaluate( (0.2,0.5 ) )
    #     >>> np.allclose( result, np.array([[ 0.45]]) )
    #     True
        
    #     Numpy input MeshVariable evaluation
        
    #     >>> # evaluate at a set of locations.. provide these as a numpy array.
    #     >>> count = 10
    #     >>> # create an empty array
    #     >>> locations = np.zeros( (count,2))
    #     >>> # specify evaluation coodinates
    #     >>> locations[:,0] = 0.5
    #     >>> locations[:,1] = np.linspace(0.,1.,count)
    #     >>> # evaluate
    #     >>> result = var.evaluate(locations)
    #     >>> np.allclose( result, np.array([[ 0.08333333], \
    #                                       [ 0.17592593], \
    #                                       [ 0.26851852], \
    #                                       [ 0.36111111], \
    #                                       [ 0.4537037 ], \
    #                                       [ 0.5462963 ], \
    #                                       [ 0.63888889], \
    #                                       [ 0.73148148], \
    #                                       [ 0.82407407], \
    #                                       [ 0.91666667]])  )
    #     True
        
    #     Using the mesh object as a FunctionInput
        
    #     >>> np.allclose( var.evaluate(mesh), var.evaluate(mesh.data))
    #     True

    #     Also note that if evaluating across an empty input, an empty output
    #     is returned. Note that the shape and type of the output is always fixed
    #     and may differ from the shape/type returned for an actual (non-empty)
    #     evaluation. Usually this should not be an issue.
    #     >>> var.evaluate(np.zeros((0,2)))
    #     array([], shape=(0, 1), dtype=float64)
    #     >>> var.evaluate(mesh.specialSets["Empty"])
    #     array([], shape=(0, 1), dtype=float64)

    #     """
    #     if inputData is None:
    #         inputData = 0.
    #     if inputType != None and inputType not in types.keys():
    #         raise ValueError("Provided input type does not appear to be valid.")
    #     if isinstance(inputData, FunctionInput):
    #         if inputType != None:
    #             raise ValueError("'inputType' specification not supported for this input class.")
    #         return _cfn.Query(self._fncself).query(inputData._get_iterator())
    #     elif isinstance(inputData, np.ndarray):
    #         if inputType != None:
    #             if inputType == ScalarType:
    #                 if inputData.shape[1] != 1:
    #                     raise ValueError("You have specified ScalarType input, but your input size is {}.\n".format(inputData.shape[1]) \
    #                                     +"ScalarType inputs must be of size 1.")
    #             if inputType == VectorType:
    #                 if inputData.shape[1] not in (2,3):
    #                     raise ValueError("You have specified VectorType input, but your input size is {}.\n".format(inputData.shape[1]) \
    #                                     +"VectorType inputs must be of size 2 or 3 (for 2d or 3d).")
    #             if inputType == SymmetricTensorType:
    #                 if inputData.shape[1] not in (3,6):
    #                     raise ValueError("You have specified SymmetricTensorType input, but your input size is {}.\n".format(inputData.shape[1]) \
    #                                     +"SymmetricTensorType inputs must be of size 3 or 6 (for 2d or 3d).")
    #             if inputType == TensorType:
    #                 if inputData.shape[1] not in (4,9):
    #                     raise ValueError("You have specified TensorType input, but your input size is {}.\n".format(inputData.shape[1]) \
    #                                     +"TensorType inputs must be of size 4 or 9 (for 2d or 3d).")
    #         else:
    #             inputType = ArrayType
    #         # lets check if this array owns its data.. process directly if it does, otherwise take a copy..
    #         # this is to avoid a bug in the way we parse non-trivial numpy arrays.  will fix in future.  #152
    #         # Note, we also added the check for 'F_CONTIGUOUS' as we also don't handle this correctly it seems. 
    #         if (not (inputData.base is None)) or inputData.flags['F_CONTIGUOUS']:
    #             inputData = inputData.copy()
    #         return _cfn.Query(self._fncself).query(_cfn.NumpyInput(inputData,inputType))
    #     else:
    #         # try convert and recurse
    #         return self.evaluate( self._evaluate_data_convert_to_ndarray(inputData), inputType )

    @Function_in_Function_out
    def diff( self, wrt_fn: 'Function' ):
        """
        Returns the object's derivative with respect to the provided function.
        """
        # wrap arg to our `Function` class in case sympy function provided
        return self.sfn.diff(wrt_fn.sfn)

    @Function_in_Function_out
    def dot( self, wrt_fn: 'Function' ):
        """
        Returns the dot product with the subject function. 
        This function simply wraps to the `sympy` equivalent.

        Note that this operation is only valid for vector Sympy 
        functions, and will fail otherwise.
        """
        from sympy.vector import Dot
        return Dot(self.sfn, wrt_fn.sfn)

@Function_in_Function_out
def gradient( fn: Function ):
    """
    Returns the vector gradient of the subject function. 
    This function simply wraps to the `sympy` equivalent.
    """
    import sympy.vector
    return sympy.vector.gradient( fn.sfn )