#!/usr/bin/env python

#$ python setup.py build_ext --inplace

from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

import numpy
import petsc4py

def configure():

    INCLUDE_DIRS = []
    LIBRARY_DIRS = []
    LIBRARIES    = []

    # PETSc
    import os

    if os.environ.get("CONDA_PREFIX"):
        PETSC_DIR  = os.environ['CONDA_PREFIX']
        PETSC_ARCH = os.environ.get('PETSC_ARCH', '')
    else:
        PETSC_DIR  = os.environ['PETSC_DIR']
        PETSC_ARCH = os.environ.get('PETSC_ARCH', '')
    
    from os.path import join, isdir
    if PETSC_ARCH and isdir(join(PETSC_DIR, PETSC_ARCH)):
        INCLUDE_DIRS += [join(PETSC_DIR, PETSC_ARCH, 'include'),
                         join(PETSC_DIR, 'include')]
        LIBRARY_DIRS += [join(PETSC_DIR, PETSC_ARCH, 'lib')]
    else:
        if PETSC_ARCH: pass # XXX should warn ...
        INCLUDE_DIRS += [join(PETSC_DIR, 'include')]
        LIBRARY_DIRS += [join(PETSC_DIR, 'lib')]
    LIBRARIES += ['petsc']

    # PETSc for Python
    INCLUDE_DIRS += [petsc4py.get_include()]

    # NumPy
    INCLUDE_DIRS += [numpy.get_include()]

    return dict(
        include_dirs=INCLUDE_DIRS + [os.curdir] + [os.path.join(os.curdir,'underworld3')],
        libraries=LIBRARIES,
        library_dirs=LIBRARY_DIRS,
        runtime_library_dirs=LIBRARY_DIRS,
    )

extensions = [
    Extension('underworld3.mesh',
              sources = ['underworld3/mesh.pyx',],
              extra_compile_args=['-O3', '-march=native'],
              **configure()),
    Extension('underworld3.maths',
              sources = ['underworld3/maths.pyx',],
              extra_compile_args=['-O3', '-march=native'],
              **configure()),
    Extension('underworld3.algorithms',
              sources = ['underworld3/algorithms.pyx',],
              extra_compile_args=['-O3', '-march=native'],
              language="c++",
              **configure()),
    Extension('underworld3.systems.stokes',
              sources = ['underworld3/systems/stokes.pyx',],
              extra_compile_args=['-O3', '-march=native'],
              **configure()),
    Extension('underworld3.swarm',
              sources = ['underworld3/swarm.pyx',],
              extra_compile_args=['-O3', '-march=native'],
              **configure()),
    Extension('underworld3.petsc_types',
              sources = ['underworld3/petsc_types.pyx',],
              extra_compile_args=['-O3', '-march=native'],
              **configure()),
    Extension('underworld3.systems.poisson',
              sources = ['underworld3/systems/poisson.pyx',],
              extra_compile_args=['-O3', '-march=native'],
              **configure()),
    Extension('underworld3.function._function',
              sources = ['underworld3/function/_function.pyx', 'underworld3/function/petsc_tools.c',],
              extra_compile_args=['-O3', '-march=native'],
              **configure()),
    Extension('underworld3.function.analytic',
              sources = ['underworld3/function/analytic.pyx', 'underworld3/function/AnalyticSolNL.c',],
              extra_compile_args=['-O3', '-march=native'],
            #   language="c++",
              **configure()),
]

setup(name = "underworld3", 
    packages=find_packages(),
    package_data={'underworld3':['*.pxd','*.h']},
    ext_modules = cythonize(
        extensions,
        compiler_directives={'language_level' : "3"},   # or "2" or "3str"
        build_dir="build",
        annotate=True,
        gdb_debug=True, 
        include_path=[petsc4py.get_include()]) )
