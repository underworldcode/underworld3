from enum import Enum as _Enum


class VarType(_Enum):
    SCALAR = 1
    VECTOR = 2
    MATRIX = 3
    NVECTOR = 4  ## Nvector can be a MATRIX
    COMPOSITE = 5
    TENSOR = 6  ## dim x dim tensor, otherwise use MATRIX
    TENSOR2D = 66
    TENSOR3D = 67
    SYM_TENSOR = 7
    SYM_TENSOR2D = 76
    SYM_TENSOR3D = 77
    OTHER = 99  # add as required
