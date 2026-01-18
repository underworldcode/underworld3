"""
Variable type enumeration for mesh and swarm variables.

This module defines the ``VarType`` enum used throughout Underworld3 to
specify the tensor rank and symmetry of field variables.

The variable type determines storage layout, number of components, and
how values are interpreted in constitutive relationships.
"""
from enum import Enum as _Enum

# Note: We have to have these types **explicitly included in the JIT code** to be able to
# use variables of that type anywhere in the code. They WILL be parsed and compiled against even
# if they are not used in any expressions.
#

# Prune out unused types (comment out) so they are not used. Other will automatically error
# at the time of (JIT) compilation.


class VarType(_Enum):
    """
    Variable type specification for mesh and swarm fields.

    This enum specifies the tensor rank and symmetry properties of field
    variables, which determines storage layout and number of components.

    Attributes
    ----------
    SCALAR : int
        Scalar field (1 component).
    VECTOR : int
        Vector field (dim components).
    MATRIX : int
        General matrix field (dim × dim components).
    TENSOR : int
        Full rank-2 tensor (dim × dim), alias for MATRIX.
    SYM_TENSOR : int
        Symmetric rank-2 tensor (6 components in 3D, 3 in 2D).
    OTHER : int
        Custom type marker.

    Examples
    --------
    >>> temperature = mesh.add_variable("T", vtype=uw.VarType.SCALAR)
    >>> velocity = mesh.add_variable("V", vtype=uw.VarType.VECTOR)
    >>> stress = mesh.add_variable("tau", vtype=uw.VarType.SYM_TENSOR)
    """

    SCALAR = 1
    VECTOR = 2
    MATRIX = 3
    # COMPOSITE = 5
    TENSOR = 6  ## dim x dim tensor, otherwise use MATRIX
    # TENSOR2D = 66
    # TENSOR3D = 67
    SYM_TENSOR = 7
    # SYM_TENSOR2D = 76
    # SYM_TENSOR3D = 77
    OTHER = 99  # add as required
