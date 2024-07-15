def _is_notebook() -> bool:
    """
    Function to determine if the python environment is a Notebook or not.

    Returns 'True' if executing in a notebook, 'False' otherwise

    Script taken from https://stackoverflow.com/a/39662359/8106122
    """

    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter

def _is_interactive_vis() -> bool:
    """
        Function to determine if interactive visualisation is available.
        Returns 'True' is possible, 'False' is otherwise
    """
    try:
        import pyvista
    except:
        return False

    import underworld3 as uw
    if uw.mpi.size != 1:
        return False

    return True

is_notebook = _is_notebook()
is_interactive_vis = _is_interactive_vis()
