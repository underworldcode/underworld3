## I think we should deprecate this since PETSc already does this for us


def parse_cmd_line_options():
    """
    This function will parse all PETSc type command line options
    and pass them through via `petsc4py`.
    """
    from petsc4py import PETSc
    import sys

    options = PETSc.Options()

    def is_petsc_key(item):
        # petsc options have single hyphen prefix
        return len(item) >= 2 and item[0] == "-" and item[1] != "-"

    for index, opt in enumerate(sys.argv):
        if is_petsc_key(opt):
            key = opt[1:]
            # if it's the last item, set to None
            if len(sys.argv) == index + 1:
                options[key] = None
            # if the next item is a different key, set to None
            elif is_petsc_key(sys.argv[index + 1]):
                options[key] = None
            # else set next item to the option value
            else:
                options[key] = sys.argv[index + 1]


import os as _os

if "UW_CMD_LINE_ARGS_DISABLE" not in _os.environ:
    parse_cmd_line_options()
