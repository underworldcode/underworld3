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

Recommended pattern:
    Define default values as named constants BEFORE the Params block.
    This makes defaults easy to find and adjust in a notebook, while
    the Params block provides CLI override, units, and descriptions.

    # --- Default values (edit these in a notebook) ---
    ETA_0    = 1e21   # Pa·s – reference viscosity
    CELL_SIZE = 50.0  # km – mesh cell size
    MAX_STEPS = 100   # solver iterations

    params = uw.Params(
        uw_viscosity = uw.Param(ETA_0,     units="Pa*s", description="reference viscosity"),
        uw_cell_size = uw.Param(CELL_SIZE, units="km",   description="mesh cell size"),
        uw_max_steps = MAX_STEPS,
    )

    # Use in code:
    mesh = uw.meshing.Box(cellSize=params.uw_cell_size)

    # Override in notebook - just assign:
    params.uw_cell_size = uw.quantity(25, "km")

    # Override from command line (flag matches Python name):
    # python script.py -uw_cell_size 25km -uw_viscosity "1e22 Pa*s"
    # mpirun -np 4 python script.py -uw_cell_size 10km
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import underworld3 as uw


class ParamType(Enum):
    """Supported parameter types for uw.Param."""
    INTEGER = "int"
    FLOAT = "float"
    STRING = "str"
    BOOLEAN = "bool"
    QUANTITY = "quantity"  # Physical quantity with units
    RATIO = "ratio"        # Dimensionless ratio (float, no units)


@dataclass
class Param:
    """Rich parameter definition with type, units, and validation.

    For QUANTITY type, dimension is automatically derived from units:
      - units="km" → dimension is [length]
      - units="Pa*s" → dimension is [mass]/[length]/[time]

    This prevents any clash between units and dimension.

    Attributes:
        value: Default value (numeric fallback)
        type: Explicit type (auto-detected if None)
        units: Units string (e.g., "km", "Pa*s")
        bounds: (min, max) tuple for validation after unit conversion
        description: Help text for CLI output

    Example:
        >>> uw.Param(0.5, units="km", bounds=(0.01, 100), description="Cell size")
    """
    value: Any
    type: ParamType = None
    units: Optional[str] = None
    bounds: Optional[Tuple] = None
    description: Optional[str] = None
    _expected_dimensionality: dict = field(default=None, repr=False)

    def __post_init__(self):
        # Auto-detect type from value if not specified (Python 3.10+ match/case)
        if self.type is None:
            match self.value:
                case bool():
                    self.type = ParamType.BOOLEAN
                case int():
                    self.type = ParamType.INTEGER
                case float() if self.units is None:
                    self.type = ParamType.FLOAT
                case float():
                    self.type = ParamType.QUANTITY
                case str():
                    self.type = ParamType.STRING
                case _ if hasattr(self.value, '_pint_qty'):  # UWQuantity
                    self.type = ParamType.QUANTITY
                    if self.units is None:
                        self.units = str(self.value.units)
                case _:
                    self.type = ParamType.STRING  # Fallback

        # Derive expected dimensionality from units (for validation)
        if self.units is not None:
            from underworld3.scaling import units as ureg
            ref_qty = ureg.parse_expression(f"1 {self.units}")
            self._expected_dimensionality = dict(ref_qty.dimensionality)


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

            # Check if CLI has this option
            has_cli_value = False
            try:
                has_cli_value = opts.hasName(lookup_name)
            except Exception:
                pass

            cli_value = self._get_petsc_option(opts, lookup_name, default)

            if has_cli_value:
                self._values[name] = cli_value
                self._sources[name] = 'cli'
            else:
                self._values[name] = cli_value  # Still use processed value (e.g., UWQuantity)
                self._sources[name] = 'default'

    def _parse_cli_value(self, val_str: str, param: Param) -> Any:
        """Parse CLI string to appropriate type using Pint for quantities.

        For QUANTITY type, uses Pint's parse_expression() which handles:
          "500 m", "0.5km", "1e21 Pa*s", "500 meter", etc.

        Validates dimensionality matches the units specified in Param definition.
        """
        from underworld3.scaling import units as ureg

        match param.type:
            case ParamType.QUANTITY:
                # Use Pint's native string parsing - handles all unit formats
                try:
                    pint_qty = ureg.parse_expression(val_str)
                except Exception as e:
                    raise ValueError(
                        f"Could not parse '{val_str}' as quantity: {e}\n"
                        f"Expected format like '500 m' or '0.5 km'"
                    )

                # Check if result has units (Pint returns plain number if no units)
                if not hasattr(pint_qty, 'magnitude'):
                    # Plain number returned - no units provided
                    raise ValueError(
                        f"Units required for parameter. Got '{val_str}', "
                        f"expected something like '{pint_qty} {param.units}'"
                    )

                # Note: We don't reject dimensionless quantities here because angles
                # (degrees, radians) are dimensionless but valid units. The dimension
                # comparison below will catch mismatches (e.g., "45 degree" for a length).

                # Validate dimensionality matches expected (derived from param.units)
                if param._expected_dimensionality is not None:
                    actual_dims = dict(pint_qty.dimensionality)
                    if actual_dims != param._expected_dimensionality:
                        raise ValueError(
                            f"Dimension mismatch: expected {param._expected_dimensionality}, "
                            f"got {actual_dims} from '{val_str}'"
                        )

                # Convert to UWQuantity
                return uw.quantity(pint_qty.magnitude, str(pint_qty.units))

            case ParamType.INTEGER:
                return int(float(val_str))

            case ParamType.FLOAT | ParamType.RATIO:
                return float(val_str)

            case ParamType.BOOLEAN:
                return val_str.lower() in ('true', '1', 'yes', 'on')

            case ParamType.STRING | _:
                return val_str

    def _validate_bounds(self, value: Any, param: Param, name: str) -> Any:
        """Validate value is within bounds after unit conversion."""
        if param.bounds is None:
            return value

        min_val, max_val = param.bounds

        # For quantities, convert to non-dimensional for comparison
        if param.type == ParamType.QUANTITY and hasattr(value, '_pint_qty'):
            check_val = uw.scaling.non_dimensionalise(value)
            # Bounds should also be in same units or non-dimensional
            if hasattr(min_val, '_pint_qty'):
                min_val = uw.scaling.non_dimensionalise(min_val)
            if hasattr(max_val, '_pint_qty'):
                max_val = uw.scaling.non_dimensionalise(max_val)
        else:
            check_val = value

        if min_val is not None and check_val < min_val:
            raise ValueError(f"Parameter {name}={value} below minimum {min_val}")
        if max_val is not None and check_val > max_val:
            raise ValueError(f"Parameter {name}={value} above maximum {max_val}")

        return value

    def _get_petsc_option(self, opts, name: str, default):
        """Get option value from PETSc with type-aware parsing."""
        # Handle Param wrapper
        if isinstance(default, Param):
            # Check if option exists before trying to get it
            if opts.hasName(name):
                val_str = opts.getString(name, "")
                if val_str:
                    # Parse and validate - let errors propagate
                    value = self._parse_cli_value(val_str, default)
                    return self._validate_bounds(value, default, name)

            # Return default value (convert to UWQuantity if needed)
            match default.type:
                case ParamType.QUANTITY if default.units:
                    return uw.quantity(default.value, default.units)
                case _:
                    return default.value

        # Existing type handling for backward compatibility (match/case)
        try:
            match default:
                case bool():
                    return opts.getBool(name, default)
                case int():
                    return opts.getInt(name, default)
                case float():
                    return opts.getReal(name, default)
                case str():
                    return opts.getString(name, default)
                case _:
                    # Complex types: try string and eval
                    if opts.hasName(name):
                        val = opts.getString(name, "")
                        if val:
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

            # Format value (handle UWQuantity objects)
            if hasattr(value, '_pint_qty'):
                val_str = f"{value.magnitude:g} {value.units}"
            elif isinstance(value, float):
                val_str = f"{value:g}"
            elif isinstance(value, str):
                val_str = f"'{value}'"
            else:
                val_str = repr(value)

            # Add source indicator and units info
            if isinstance(default, Param) and default.units:
                units_hint = f" [{default.units}]"
            else:
                units_hint = ""

            if source == 'cli':
                indicator = f"  # from -{name}{units_hint}"
            elif source == 'override':
                orig_val = default.value if isinstance(default, Param) else default
                indicator = f"  # overridden (was {orig_val!r}){units_hint}"
            else:
                indicator = units_hint and f"  #{units_hint}" or ""

            lines.append(f"    {name} = {val_str},{indicator}")

        lines.append(")")
        return "\n".join(lines)

    def _repr_markdown_(self):
        """Rich display for Jupyter notebooks with type and units info."""
        values = object.__getattribute__(self, '_values')
        sources = object.__getattribute__(self, '_sources')
        defaults = object.__getattribute__(self, '_defaults')

        lines = ["| Parameter | Value | Units | Type | Source |",
                 "|-----------|-------|-------|------|--------|"]

        for name, value in values.items():
            source = sources[name]
            default = defaults[name]

            # Extract type/units info from Param
            if isinstance(default, Param):
                type_str = default.type.value if default.type else "auto"
                units_str = default.units or ""
                default_val = default.value
            else:
                type_str = type(default).__name__
                units_str = ""
                default_val = default

            # Format value (handle UWQuantity objects)
            if hasattr(value, '_pint_qty'):
                val_str = f"`{value.magnitude:g} {value.units}`"
            elif isinstance(value, float):
                val_str = f"`{value:g}`"
            else:
                val_str = f"`{value!r}`"

            # Source with CLI hint
            match source:
                case 'default':
                    src_str = "default"
                case 'cli':
                    src_str = f"**CLI** (`-{name}`)"
                case _:
                    src_str = "**override**"

            lines.append(f"| `{name}` | {val_str} | {units_str} | {type_str} | {src_str} |")

        return "\n".join(lines)

    def summary(self, title: str = "Parameters"):
        """Print a compact table of current parameter settings.

        Uses uw.pprint so only rank 0 prints in parallel.

        Args:
            title: Header line for the table.

        Example output::

            Parameters
            -----------------------------------------------
            eta_background   1e+24 Pa*s     background viscosity
            eta_base         1e+21 Pa*s     base (weak) viscosity
            convergence_rate 2.5 mm/yr      convergence velocity
            problem_size     2              mesh resolution level
            smoothing        3e-05
            -----------------------------------------------
        """
        values = object.__getattribute__(self, '_values')
        defaults = object.__getattribute__(self, '_defaults')
        sources = object.__getattribute__(self, '_sources')

        # Strip common prefix for display (e.g. "uw_" → cleaner names)
        names = list(values.keys())
        prefix = ""
        if len(names) > 1:
            candidate = names[0].split("_")[0] + "_"
            if all(n.startswith(candidate) for n in names):
                prefix = candidate

        # Build rows: (display_name, value_str, description)
        rows = []
        for name, value in values.items():
            display = name[len(prefix):] if prefix else name
            default = defaults[name]

            # Format value — use the Param's original units string if available
            # (e.g. "Pa*s") rather than Pint's verbose form ("pascal * second")
            if hasattr(value, '_pint_qty'):
                units_str = default.units if isinstance(default, Param) and default.units else str(value.units)
                val_str = f"{value.magnitude:g} {units_str}"
            elif isinstance(value, float):
                val_str = f"{value:g}"
            else:
                val_str = repr(value)

            # Source tag appended to description, not value
            source = sources[name]
            desc = default.description if isinstance(default, Param) and default.description else ""
            if source == 'cli':
                desc = f"{desc} (CLI)" if desc else "(CLI)"
            elif source == 'override':
                desc = f"{desc} (set)" if desc else "(set)"

            rows.append((display, val_str, desc))

        # Column widths
        w_name = max(len(r[0]) for r in rows)
        w_val = max(len(r[1]) for r in rows)
        table_width = max(w_name + w_val + 4 + max((len(r[2]) for r in rows), default=0),
                          len(title) + 4)

        lines = [title, "-" * table_width]
        for display, val_str, desc in rows:
            line = f"  {display:<{w_name}}  {val_str:<{w_val}}"
            if desc:
                line += f"  {desc}"
            lines.append(line)
        lines.append("-" * table_width)

        text = "\n".join(lines)
        uw.pprint(text, clean_display=False)
        return text

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
        """Return help text for command-line usage with type and units info."""
        defaults = object.__getattribute__(self, '_defaults')

        lines = ["Command-line options (PETSc format):", ""]
        example_parts = []

        for name, default in defaults.items():
            if isinstance(default, Param):
                # Rich parameter with type/units info
                type_str = default.type.value if default.type else "auto"
                default_val = default.value

                # Build the help line
                if default.units:
                    lines.append(f"  -{name} <{type_str}>   Units: {default.units}")
                    default_str = f"{default_val} {default.units}"
                    # Add example for quantity params
                    example_parts.append(f"-{name} {default_val}{default.units}")
                elif default.type == ParamType.RATIO:
                    lines.append(f"  -{name} <{type_str}>   Dimensionless ratio")
                    default_str = str(default_val)
                else:
                    lines.append(f"  -{name} <{type_str}>")
                    default_str = repr(default_val)

                # Add bounds if specified
                if default.bounds:
                    min_val, max_val = default.bounds
                    lines.append(f"                         Bounds: [{min_val}, {max_val}]")

                # Add description if specified
                if default.description:
                    lines.append(f"                         {default.description}")

                lines.append(f"                         (default: {default_str})")
                lines.append("")
            else:
                # Simple parameter (backward compatible)
                type_name = type(default).__name__
                lines.append(f"  -{name} <{type_name}>  (default: {default!r})")
                lines.append("")

        # Build example command
        if example_parts:
            lines.extend(["Example:",
                         f"  python script.py {' '.join(example_parts)}"])
        else:
            first_name = list(defaults.keys())[0] if defaults else "param"
            lines.extend(["Example:",
                         f"  python script.py -{first_name} <value>"])

        return "\n".join(lines)
