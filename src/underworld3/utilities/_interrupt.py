"""
Underworld3 Interactive Pause

A mechanism for pausing notebook execution during interactive development.
In non-notebook environments (scripts, HPC jobs), pauses are silently skipped.

Usage:
    import underworld3 as uw

    # Pause for review in notebooks:
    uw.pause("Mesh setup complete - review before continuing")

    # With additional context:
    uw.pause(
        "Ready for expensive computation",
        hint="This will take approximately 10 minutes."
    )

    # In a script on HPC: pauses are automatically skipped
"""

import os


def is_notebook():
    """
    Detect if we're running in a Jupyter notebook.

    Returns True for:
    - Jupyter notebooks (ZMQInteractiveShell)
    - JupyterLab

    Returns False for:
    - Scripts run with `python script.py`
    - IPython terminal sessions
    - Batch jobs on HPC
    - pytest runs

    Can be overridden with environment variable:
    - UW_NOTEBOOK_EMULATION=1 forces notebook mode (pause will interrupt)
    - UW_NOTEBOOK_EMULATION=0 forces script mode (pause will be skipped)
    """
    # Environment variable override
    env_override = os.environ.get("UW_NOTEBOOK_EMULATION", "").lower()
    if env_override in ("1", "true", "yes"):
        return True
    if env_override in ("0", "false", "no"):
        return False

    # Check for Jupyter notebook
    try:
        from IPython import get_ipython
        ipy = get_ipython()
        if ipy is not None:
            # ZMQInteractiveShell = Jupyter notebook / JupyterLab
            if ipy.__class__.__name__ == "ZMQInteractiveShell":
                return True
    except (ImportError, AttributeError):
        pass

    return False


class UW_Pause(Exception):
    """
    Interactive pause exception for Underworld3 notebooks.

    This exception is raised by uw.pause() in notebook environments.
    It displays a clean, formatted message without Python tracebacks.

    In non-notebook environments (scripts, HPC), uw.pause() does nothing
    and this exception is never raised.

    Parameters
    ----------
    message : str
        The pause message to display
    explanation : str, optional
        Additional explanation for the user

    Examples
    --------
    In a Jupyter notebook:

    >>> raise UW_Pause("Mesh visualization complete")

    ─────────────────────────────────────────────────────
    Underworld3 - Paused  🛑
    ─────────────────────────────────────────────────────
    Mesh visualization complete
    ─────────────────────────────────────────────────────

    Notes
    -----
    Use uw.pause() instead of raising this directly - it handles
    the notebook/script detection automatically.
    """

    def __init__(self, message: str, explanation: str = None):
        self.message = message
        self.explanation = explanation
        super().__init__(message)

    def _render_traceback_(self):
        """
        IPython/Jupyter custom traceback rendering.

        This method is called by IPython when the exception is raised,
        displaying a clean message instead of a traceback.
        """
        lines = []
        lines.append("")
        lines.append("─" * 57)
        lines.append("Underworld3 - Paused  🛑")
        lines.append("─" * 57)
        lines.append(self.message)
        if self.explanation:
            lines.append("")
            lines.append(self.explanation)
        lines.append("─" * 57)
        lines.append("")
        return lines

    def __str__(self):
        """String representation for non-Jupyter environments."""
        parts = [self.message]
        if self.explanation:
            parts.append(f" ({self.explanation})")
        return "".join(parts)

    def __repr__(self):
        if self.explanation:
            return f"UW_Pause({self.message!r}, explanation={self.explanation!r})"
        return f"UW_Pause({self.message!r})"


def pause(message: str, explanation: str = None):
    """
    Pause notebook execution for interactive review.

    In Jupyter notebooks, this raises a UW_Pause exception that displays
    a clean, formatted message.

    In non-notebook environments (scripts, HPC jobs), this does nothing
    and execution continues - allowing the same notebook to run unattended.

    Parameters
    ----------
    message : str
        The pause message to display
    explanation : str, optional
        Additional explanation for the user

    Examples
    --------
    >>> import underworld3 as uw

    >>> # This pauses in Jupyter, continues silently in scripts:
    >>> uw.pause("Mesh ready for inspection")

    >>> # With an explanation:
    >>> uw.pause(
    ...     "About to start expensive solve",
    ...     explanation="This takes ~5 minutes on 4 cores"
    ... )

    Environment Variables
    ---------------------
    UW_NOTEBOOK_EMULATION : str
        Override automatic detection:
        - "1" or "true": Emulate notebook mode (always pause)
        - "0" or "false": Emulate script mode (never pause)

    Notes
    -----
    This is designed for development workflows where you want to:
    - Pause before expensive computations in notebooks
    - Review visualizations before continuing
    - Avoid pyvista issues when cells lack focus

    But also want the same notebook to run unmodified on HPC.
    """
    if is_notebook():
        raise UW_Pause(message, explanation)
    # In script mode: silently continue


# Export symbols
__all__ = ["UW_Pause", "pause", "is_notebook"]
