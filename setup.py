#!/usr/bin/env python

# Usage:
#  $ pip install .

from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

import os
import shutil
import numpy
import petsc4py


def _clean_stale_build_cache():
    """Remove cached Cython extensions when the PETSc target changes.

    The setuptools build directory (build/lib.*) caches compiled .so files
    and reuses them on subsequent installs — even if the PETSc installation
    has changed.  When switching between pixi environments that use different
    PETSc builds (e.g. conda-forge 3.24.2 vs custom 3.24.3), the stale .so
    files silently link against the wrong libpetsc, causing PETSc error 73
    ("Object is in wrong state") at solve time.

    This function records which PETSc was used for the last build and wipes
    the extension cache whenever the target changes.
    """
    build_dir = os.path.join(os.path.dirname(__file__) or ".", "build")
    marker = os.path.join(build_dir, ".petsc_target")

    config = petsc4py.get_config()
    current = config["PETSC_DIR"] + "/" + config.get("PETSC_ARCH", "")

    if os.path.isfile(marker):
        with open(marker) as f:
            previous = f.read().strip()
        if previous == current:
            return  # Same target — nothing to do
        print(f"\n  *** PETSc target changed ***")
        print(f"      was: {previous}")
        print(f"      now: {current}")
        print(f"      Cleaning build cache to force recompilation...\n")
    elif any(
        d.startswith("lib.") or d.startswith("temp.")
        for d in os.listdir(build_dir)
        if os.path.isdir(os.path.join(build_dir, d))
    ) if os.path.isdir(build_dir) else False:
        # No marker but cached extensions exist — stale from a previous build
        print(f"\n  *** No PETSc target marker — cleaning build cache...\n")
    else:
        # No cache at all — first build, just record the target
        os.makedirs(build_dir, exist_ok=True)
        with open(marker, "w") as f:
            f.write(current)
        return

    # Wipe cached extensions
    for name in os.listdir(build_dir):
        path = os.path.join(build_dir, name)
        if os.path.isdir(path) and (
            name.startswith("lib.") or name.startswith("temp.")
        ):
            shutil.rmtree(path)

    os.makedirs(build_dir, exist_ok=True)
    with open(marker, "w") as f:
        f.write(current)


_clean_stale_build_cache()


def configure():

    INCLUDE_DIRS = []
    LIBRARY_DIRS = []
    LIBRARIES = []

    PETSC_DIR = ""
    PETSC_ARCH = ""

    # try get PETSC_DIR from petsc pip installation
    try:
        import petsc

        PETSC_DIR = petsc.get_petsc_dir()
    except:
        pass

    # PETSc
    import os

    if not os.path.exists(PETSC_DIR):
        print(f"PETSC_INFO from petsc4py - {petsc4py.get_config()}")
        PETSC_DIR = petsc4py.get_config()["PETSC_DIR"]
        PETSC_ARCH = petsc4py.get_config()["PETSC_ARCH"]

    # It is preferable to use the petsc4py paths to the
    # petsc libraries for consistency but the pip installation
    # of PETSc sometimes points to the temporary setup up path

    if not os.path.exists(PETSC_DIR):
        print(f"PETSC_DIR {PETSC_DIR} is bad - trying another ...")

        if os.environ.get("CONDA_PREFIX") and not os.environ.get("PETSC_DIR"):
            import sys

            py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
            PETSC_DIR = os.path.join(
                os.environ["CONDA_PREFIX"],
                "lib",
                "python" + py_version,
                "site-packages",
                "petsc",
            )  # symlink to latest python
            PETSC_ARCH = os.environ.get("PETSC_ARCH", "")
        else:
            PETSC_DIR = os.environ["PETSC_DIR"]
            PETSC_ARCH = os.environ.get("PETSC_ARCH", "")

    print(f"Using PETSc:")
    print(f"PETSC_DIR: {PETSC_DIR}")
    print(f"PETSC_ARCH: {PETSC_ARCH}")

    from os.path import join, isdir

    if PETSC_ARCH and isdir(join(PETSC_DIR, PETSC_ARCH)):
        INCLUDE_DIRS += [
            join(PETSC_DIR, PETSC_ARCH, "include"),
            join(PETSC_DIR, "include"),
        ]
        LIBRARY_DIRS += [join(PETSC_DIR, PETSC_ARCH, "lib")]
        petscvars = join(
            PETSC_DIR, PETSC_ARCH, "lib", "petsc", "conf", "petscvariables"
        )
    else:
        if PETSC_ARCH:
            pass  # XXX should warn ...
        INCLUDE_DIRS += [join(PETSC_DIR, "include")]
        LIBRARY_DIRS += [join(PETSC_DIR, "lib")]
        petscvars = join(PETSC_DIR, "lib", "petsc", "conf", "petscvariables")

    LIBRARIES += ["petsc"]

    # Set C and C++ compilers to PETSc's toolchain so the UW extensions
    # build with the same compiler family and MPI wrappers as PETSc.
    cc = ""
    cxx = ""
    with open(petscvars, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("CC ="):
                cc = line.split("=", 1)[1].strip()
            elif line.startswith("CXX ="):
                cxx = line.split("=", 1)[1].strip()
    # print(f"***\n The c compiler is: {cc}\n*****")
    if cc:
        os.environ["CC"] = cc
    if cxx:
        os.environ["CXX"] = cxx

    # PETSc for Python
    INCLUDE_DIRS += [petsc4py.get_include()]

    # NumPy
    INCLUDE_DIRS += [numpy.get_include()]

    return dict(
        include_dirs=INCLUDE_DIRS
        + [os.curdir]
        + [os.path.join(os.curdir, "underworld3")]
        + [os.path.join(os.curdir, "underworld3", "petsc")],
        libraries=LIBRARIES,
        library_dirs=LIBRARY_DIRS,
        runtime_library_dirs=LIBRARY_DIRS,
    )


conf = configure()

extra_compile_args = ["-O3", "-g"]
# extra_compile_args = ['-O0', '-g']
extensions = [
    Extension(
        "underworld3.ckdtree",
        sources=[
            "src/underworld3/ckdtree.pyx",
        ],
        extra_compile_args=extra_compile_args + ["-std=c++11"],
        language="c++",
        **conf,
    ),
    Extension(
        "underworld3.cython.petsc_discretisation",
        sources=[
            "src/underworld3/cython/petsc_discretisation.pyx",
        ],
        extra_compile_args=extra_compile_args,
        **conf,
    ),
    Extension(
        "underworld3.cython.petsc_maths",
        sources=[
            "src/underworld3/cython/petsc_maths.pyx",
        ],
        extra_compile_args=extra_compile_args,
        **conf,
    ),
    Extension(
        "underworld3.cython.petsc_types",
        sources=[
            "src/underworld3/cython/petsc_types.pyx",
        ],
        extra_compile_args=extra_compile_args,
        **conf,
    ),
    Extension(
        "underworld3.cython.generic_solvers",
        sources=[
            "src/underworld3/cython/petsc_generic_snes_solvers.pyx",
        ],
        extra_compile_args=extra_compile_args,
        **conf,
    ),
    Extension(
        "underworld3.function._function",
        sources=[
            "src/underworld3/function/_function.pyx",
            "src/underworld3/function/petsc_tools.c",
        ],
        extra_compile_args=extra_compile_args,
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        **conf,
    ),
    Extension(
        "underworld3.function._dminterp_wrapper",
        sources=[
            "src/underworld3/function/_dminterp_wrapper.pyx",
            "src/underworld3/function/petsc_tools.c",  # Required for DMInterpolationEvaluate_UW
        ],
        extra_compile_args=extra_compile_args,
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        **conf,
    ),
    Extension(
        "underworld3.function.analytic",
        sources=[
            "src/underworld3/function/analytic.pyx",
            "src/underworld3/function/AnalyticSolNL.c",
        ],
        extra_compile_args=extra_compile_args,
        **conf,
    ),
]


# Create uwid if it doesn't exist
idfile = "./src/underworld3/_uwid.py"
if not os.path.isfile(idfile):
    import uuid

    with open(idfile, "w+") as f:
        f.write("uwid = '" + str(uuid.uuid4()) + "'")

setup(
    name="underworld3",
    packages=find_packages(),
    # Version is derived from git tags via setuptools_scm (configured in pyproject.toml)
    package_data={"underworld3": ["*.pxd", "*.h", "function/*.h", "cython/*.pxd"]},
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},  # or "2" or "3str"
        build_dir="build",
        annotate=True,
        # gdb_debug=True,
        include_path=[petsc4py.get_include()],
    ),
)
