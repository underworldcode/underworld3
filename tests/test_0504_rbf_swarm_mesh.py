import os

# DISABLE SYMPY CACHE, AS IT GETS IN THE WAY FOR IDENTICALLY NAMED VARIABLES.
# NEED TO FIX.

import underworld3 as uw
import underworld3.function as fn
import numpy as np
import sympy
import pytest


## We need tests to
## 1) check the rbf is operating
## 2) check that the values are sane
## 3) compare values to function evaluate methods
