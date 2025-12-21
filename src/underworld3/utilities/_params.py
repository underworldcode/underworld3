"""
Parameter management for Underworld3 examples and scripts.

Provides a clean way to define parameters with defaults that can be:
1. Edited directly in notebooks (just assign new values)
2. Overridden from command line via PETSc options

Naming Convention:
    We encourage using 'uw_' prefix for parameter names in examples.
    This avoids collisions with PETSc solver options and makes it clear
    these are underworld example parameters.

    The CLI flag matches the Python name exactly:
    - Python: params.uw_mesh_resolution
    - CLI: -uw_mesh_resolution 0.025

Example usage:
    # Define parameters at top of notebook/script
    params = uw.Params(
        uw_mesh_resolution = 0.05,   # Cell size for mesh
        uw_diffusivity = 1.0,        # Material property
        uw_hot_temp = 100.0,         # Boundary temperature
    )

    # Use in code:
    mesh = uw.meshing.Box(cellSize=params.uw_mesh_resolution)

    # Override in notebook - just assign:
    params.uw_mesh_resolution = 0.025

    # Override from command line (flag matches Python name):
    # python script.py -uw_mesh_resolution 0.025
    # mpirun -np 4 python script.py -uw_diffusivity 2.0
"""

import underworld3 as uw


class Params:
    """
    Parameter container with PETSc command-line override support.

    Parameters are defined as keyword arguments with default values.
    Each parameter can be overridden from the command line using
    the same name as a PETSc option flag.

    Naming Convention:
        Use 'uw_' prefix for parameter names (e.g., uw_mesh_resolution).
        The CLI flag then matches: -uw_mesh_resolution 0.025

    Note:
        PETSc strips the 'uw_' prefix when storing options internally.
        This class handles that automatically - you use uw_name in Python
        and -uw_name on the command line; lookup is handled transparently.

    Attributes:
        _defaults: Original default values
        _sources: Where each value came from ('default', 'cli', 'override')

    Example:
        >>> params = Params(uw_resolution=0.1, uw_viscosity=1e21)
        >>> params.uw_resolution  # Returns 0.1 or CLI override
        0.1
        >>> params.uw_resolution = 0.05  # Notebook override
        >>> params.uw_resolution
        0.05
    """

    def __init__(self, **defaults):
        """
        Initialize parameters with defaults, checking for CLI overrides.

        Args:
            **defaults: Parameter names and default values.
                        Use uw_ prefix for collision avoidance.
        """
        # Use object.__setattr__ to avoid triggering our custom __setattr__
        object.__setattr__(self, '_defaults', dict(defaults))
        object.__setattr__(self, '_values', {})
        object.__setattr__(self, '_sources', {})

        # Use the module-level options object
        opts = uw.options

        for name, default in defaults.items():
            # PETSc strips 'uw_' prefix, so look up without it
            lookup_name = name[3:] if name.startswith('uw_') else name
            cli_value = self._get_petsc_option(opts, lookup_name, default)

            if cli_value != default:
                self._values[name] = cli_value
                self._sources[name] = 'cli'
            else:
                self._values[name] = default
                self._sources[name] = 'default'

    def _get_petsc_option(self, opts, name: str, default):
        """Get option value from PETSc, matching the type of default."""
        # PETSc stores options without the prefix
        try:
            if isinstance(default, bool):
                return opts.getBool(name, default)
            elif isinstance(default, int):
                return opts.getInt(name, default)
            elif isinstance(default, float):
                return opts.getReal(name, default)
            elif isinstance(default, str):
                return opts.getString(name, default)
            else:
                # For complex types, try string and eval
                val = opts.getString(name, None)
                if val is not None:
                    try:
                        return eval(val)
                    except:
                        return val
                return default
        except:
            return default

    def __getattr__(self, name: str):
        """Get parameter value."""
        if name.startswith('_'):
            return object.__getattribute__(self, name)
        values = object.__getattribute__(self, '_values')
        if name in values:
            return values[name]
        raise AttributeError(f"No parameter named '{name}'")

    def __setattr__(self, name: str, value):
        """Set parameter value (override)."""
        if name.startswith('_'):
            object.__setattr__(self, name, value)
            return
        values = object.__getattribute__(self, '_values')
        sources = object.__getattribute__(self, '_sources')
        if name not in values:
            raise AttributeError(
                f"Cannot add new parameter '{name}'. "
                f"Available: {list(values.keys())}"
            )
        values[name] = value
        sources[name] = 'override'

    def __repr__(self):
        """Show parameters with their sources."""
        lines = ["Params("]
        values = object.__getattribute__(self, '_values')
        sources = object.__getattribute__(self, '_sources')
        defaults = object.__getattribute__(self, '_defaults')

        for name, value in values.items():
            source = sources[name]
            default = defaults[name]

            # Format value
            if isinstance(value, float):
                val_str = f"{value:g}"
            elif isinstance(value, str):
                val_str = f"'{value}'"
            else:
                val_str = repr(value)

            # Add source indicator
            if source == 'cli':
                indicator = f"  # from -{name}"
            elif source == 'override':
                indicator = f"  # overridden (was {default!r})"
            else:
                indicator = ""

            lines.append(f"    {name} = {val_str},{indicator}")

        lines.append(")")
        return "\n".join(lines)

    def _repr_markdown_(self):
        """Rich display for Jupyter notebooks."""
        values = object.__getattribute__(self, '_values')
        sources = object.__getattribute__(self, '_sources')
        defaults = object.__getattribute__(self, '_defaults')

        lines = ["| Parameter | Value | Source | Default |",
                 "|-----------|-------|--------|---------|"]

        for name, value in values.items():
            source = sources[name]
            default = defaults[name]

            # Format for table
            if isinstance(value, float):
                val_str = f"`{value:g}`"
            else:
                val_str = f"`{value!r}`"

            if isinstance(default, float):
                def_str = f"`{default:g}`"
            else:
                def_str = f"`{default!r}`"

            # Source with CLI hint
            if source == 'default':
                src_str = "default"
            elif source == 'cli':
                src_str = f"**CLI** (`-{name}`)"
            else:
                src_str = "**override**"

            lines.append(f"| `{name}` | {val_str} | {src_str} | {def_str} |")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Return parameters as a dictionary."""
        return dict(object.__getattribute__(self, '_values'))

    def reset(self, name: str = None):
        """Reset parameter(s) to default values.

        Args:
            name: Parameter to reset, or None to reset all
        """
        values = object.__getattribute__(self, '_values')
        sources = object.__getattribute__(self, '_sources')
        defaults = object.__getattribute__(self, '_defaults')

        if name is None:
            for n in defaults:
                values[n] = defaults[n]
                sources[n] = 'default'
        elif name in defaults:
            values[name] = defaults[name]
            sources[name] = 'default'
        else:
            raise AttributeError(f"No parameter named '{name}'")

    def cli_help(self) -> str:
        """Return help text for command-line usage."""
        defaults = object.__getattribute__(self, '_defaults')

        lines = ["Command-line options (PETSc format):", ""]
        for name, default in defaults.items():
            type_name = type(default).__name__
            lines.append(f"  -{name} <{type_name}>  (default: {default!r})")

        first_name = list(defaults.keys())[0] if defaults else "param"
        lines.extend(["", "Example:",
                     f"  python script.py -{first_name} <value>"])
        return "\n".join(lines)
