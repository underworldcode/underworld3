# # Test import for underworld
#
# `pytest` tests/test_000_imports.py
#

# +
import pytest

clean = False


# -

# The standard import for underworld will import all the sub-modules and dependencies 

def test_underworld3_global_import():
    import underworld3 as uw
    clean = True
    
    return


# If the standard import fails, these are the individual modules that underworld imports. 

# +
def test_underworld_mesh_import():
    import underworld3.discretisation
    
def test_underworld_meshing_import():
    import underworld3.meshing
    
def test_underworld_maths_import():
    import underworld3.maths
    
def test_underworld_swarm_import():
    import underworld3.swarm
    
def test_underworld_systems_import():
    import underworld3.systems
    
def test_underworld_tools_import():
    import underworld3.tools

def test_underworld_kdtree_import():
    import underworld3.kdtree
    
def test_underworld_mpi_import():
    import underworld3.mpi
    
