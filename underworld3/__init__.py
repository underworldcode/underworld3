# Should we reconsider importing these? 
# Possible it's an excessive cost at startup on HPC due to 
# parallel filesystem thrashing. 
# Is it better to defer this to import on an as-needs basis? 
# On the other hand, i'm not sure if autocomplete will play nice
# without it, and also deferring import may lead to unpredictable
# (dependent on parallel filesystem loads) import times downstream. 

import underworld3.mesh
import underworld3.maths
import underworld3.swarm
import underworld3.systems

from enum import Enum as _Enum
class VarType(_Enum):
    SCALAR=1
    VECTOR=2
    OTHER=3  # add as required 
