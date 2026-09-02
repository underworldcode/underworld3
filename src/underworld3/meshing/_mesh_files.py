r"""Where generated mesh files are written, and how they are written.

Building a mesh hands gmsh a file to write and then reads it back through
PETSc, so the pair — ``<name>.msh`` and the ``<name>.msh.h5`` PETSc converts
it to — is scratch shared between those two steps. It is not a cache: every
construction regenerates both, and nothing checks whether they already exist.

The name is derived from the mesh PARAMETERS, so two processes building the
same geometry in one working directory choose the same name, and one can read
a file the other is still writing (issue #563). Identical geometry is exactly
the colliding case, which is why a parameter sweep or a parallel test run hits
it and ordinary use does not.

Two mechanisms make that safe, and both are needed:

* the directory is settable per process through ``UW_MESH_CACHE_DIR``, so
  independent jobs can be given somewhere of their own;
* every write lands atomically, so a reader that shares a directory anyway
  sees a complete file or no file, never a half-written one.
"""
import os
from pathlib import Path

import underworld3 as uw

DEFAULT_MESH_FILE_DIR = ".meshes"


def mesh_file_dir():
    """The directory generated mesh files are written to.

    ``UW_MESH_CACHE_DIR`` overrides the default ``.meshes``. Every rank of one
    job must agree on it, so this reads the environment — inherited identically
    by every rank — and never anything process-local such as the pid.
    """
    return Path(os.environ.get("UW_MESH_CACHE_DIR", DEFAULT_MESH_FILE_DIR))


def mesh_file_path(basename):
    """Full path for a generated mesh file, with its directory created.

    Parameters
    ----------
    basename : str
        The file's name, conventionally ``uw_<generator>_<parameters>.msh``.
    """
    directory = mesh_file_dir()
    if uw.mpi.rank == 0:
        directory.mkdir(parents=True, exist_ok=True)
    return str(directory / basename)


def _scratch_name(final):
    """A process-unique sibling of ``final`` KEEPING ITS EXTENSION.

    The extension has to survive: gmsh chooses its output format from it, so
    writing to ``mesh.msh.1234.tmp`` would silently produce something that is
    not a gmsh mesh.
    """
    final = Path(final)
    return final.with_name(f"{final.stem}.{os.getpid()}.tmp{final.suffix}")


def write_gmsh(filename):
    """``gmsh.write``, landing atomically at ``filename``.

    gmsh writes in place, so a concurrent reader can open a file that is still
    being filled. Writing under a process-unique name and renaming makes the
    appearance of the final name atomic — :func:`os.replace` is atomic within a
    filesystem — so a reader sees either the previous complete file or the new
    one.
    """
    import gmsh

    scratch = _scratch_name(filename)
    gmsh.write(str(scratch))
    os.replace(scratch, filename)
