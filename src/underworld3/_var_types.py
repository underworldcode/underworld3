from enum import Enum as _Enum

# Note: We have to have these types **explicitly included in the JIT code** to be able to
# use variables of that type anywhere in the code. They WILL be parsed and compiled against even
# if they are not used in any expressions.
#

# Prune out unused types (comment out) so they are not used. Other will automatically error
# at the time of (JIT) compilation.


class VarType(_Enum):
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
