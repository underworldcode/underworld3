"""Shared path handling for native PETSc/HDF5 I/O."""

import os
from contextlib import contextmanager


@contextmanager
def _short_io_path(path: str):
    """Expose ``path`` to native I/O as a basename from its parent directory.

    Some parallel PETSc/HDF5 stacks fail on valid absolute paths well below
    ``PATH_MAX``. Snapshot artifacts retain their normal locations, while the
    native reader or writer receives only the final path component.
    """
    absolute_path = os.path.abspath(path)
    previous_directory = os.getcwd()
    os.chdir(os.path.dirname(absolute_path))
    try:
        yield os.path.basename(absolute_path)
    finally:
        os.chdir(previous_directory)
