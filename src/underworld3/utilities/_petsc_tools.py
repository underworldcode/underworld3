from petsc4py import PETSc

## I think we should deprecate this since PETSc already does this for us

options = PETSc.Options("uw_")
"""Expose petsc options on the command line as -uw_XXX"""


def require_dirs(ListOfDirs):
    """
    List of directories required by this run
    """
    import os

    for dir in ListOfDirs:
        os.makedirs(dir, exist_ok=True)


def parse_cmd_line_options():
    """Hand the command line to PETSc's own options parser.

    UW parameters are namespaced `-uw_*` precisely so they can sit in the PETSc
    options database alongside PETSc's own without clashing, and `uw.options` is
    a `PETSc.Options("uw_")` view onto it. So there is nothing here that PETSc
    does not already do: `PetscOptionsInsertString` (exposed by petsc4py as
    `Options.insertString`) applies the same parsing rules PETSc applies to its
    own arguments.

    This used to re-implement that parsing, and got it wrong. Its test for an
    option NAME was `item[0] == "-" and item[1] != "-"`, which accepts `-2`, so
    `-uw_sense -2` stored `uw_sense` with no value and registered a stray option
    `2` -- the negative silently never arrived (#642). PETSc's own rule
    (`PetscOptionsValidKey`) requires a hyphen followed by a letter, which is
    exactly what distinguishes an option from a negative number. Deferring to it
    fixes that class of bug rather than the one instance of it.

    It exists at all because petsc4py does NOT populate the options database
    from `sys.argv` on every platform (Gadi being the case in #111), so
    something has to do the insertion explicitly. It is idempotent -- re-inserting
    the same arguments rewrites the same values -- so it is safe to call on every
    `Params` construction.
    """
    from petsc4py import PETSc
    import sys

    arguments = sys.argv[1:]
    if not arguments:
        return

    # PetscOptionsInsertString reads a single string, so an argument carrying
    # whitespace has to be quoted back up; PETSc understands double quotes.
    def requote(argument):
        return f'"{argument}"' if any(c.isspace() for c in argument) else argument

    PETSc.Options().insertString(" ".join(requote(a) for a in arguments))


import os as _os
