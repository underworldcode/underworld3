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


is_notebook = _is_notebook()
