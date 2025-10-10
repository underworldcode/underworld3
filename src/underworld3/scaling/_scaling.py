"""
Utilities to convert between dimensional and non-dimensional values.
"""
from __future__ import print_function, absolute_import
import underworld3 as uw
from ._utils import TransformedDict
from pint import UnitRegistry

u = UnitRegistry()

COEFFICIENTS = None

pint_degc_labels = ['degC', 'degreeC', 'degree_Celsius', 'celsius']

def get_coefficients():
    """
    Returns the global scaling dictionary.
    """
    global COEFFICIENTS
    if COEFFICIENTS is None:
        COEFFICIENTS = TransformedDict()
        COEFFICIENTS["[length]"] = 1.0 * u.meter
        COEFFICIENTS["[mass]"] = 1.0 * u.kilogram
        COEFFICIENTS["[time]"] = 1.0 * u.year
        COEFFICIENTS["[temperature]"] = 1.0 * u.degK
        COEFFICIENTS["[substance]"] = 1.0 * u.mole
    return COEFFICIENTS

def non_dimensionalise(dimValue):
    """
    Non-dimensionalize (scale) provided quantity.

    This function uses pint to perform a dimension analysis and
    return a value scaled according to a set of scaling coefficients.

    Parameters
    ----------
    dimValue : pint.Quantity 
        A pint quantity.

    Returns
    -------
    float 
        The scaled value.

    Example
    -------

    >>> import underworld as uw
    >>> u = uw.scaling.units

    >>> # Characteristic values of the system
    >>> half_rate = 0.5 * u.centimeter / u.year
    >>> model_height = 600e3 * u.meter
    >>> refViscosity = 1e24 * u.pascal * u.second
    >>> surfaceTemp = 0. * u.kelvin
    >>> baseModelTemp = 1330. * u.kelvin
    >>> baseCrustTemp = 550. * u.kelvin

    >>> KL_meters = model_height
    >>> KT_seconds = KL_meters / half_rate
    >>> KM_kilograms = refViscosity * KL_meters * KT_seconds
    >>> Kt_degrees = (baseModelTemp - surfaceTemp)
    >>> K_substance = 1. * u.mole

    >>> scaling_coefficients = uw.scaling.get_coefficients()
    >>> scaling_coefficients["[time]"] = KT_seconds
    >>> scaling_coefficients["[length]"] = KL_meters
    >>> scaling_coefficients["[mass]"] = KM_kilograms
    >>> scaling_coefficients["[temperature]"] = Kt_degrees
    >>> scaling_coefficients["[substance]"] -= K_substance

    >>> # Get a scaled value:
    >>> gravity = uw.scaling.non_dimensionalise(9.81 * u.meter / u.second**2)
    """
    try:
        val = dimValue.unitless
        if val:
            return dimValue
    except AttributeError:
        return dimValue

    dimValue = dimValue.to_base_units()

    scaling_coefficients = get_coefficients()

    length = scaling_coefficients["[length]"]
    time = scaling_coefficients["[time]"]
    mass = scaling_coefficients["[mass]"]
    temperature = scaling_coefficients["[temperature]"]
    substance = scaling_coefficients["[substance]"]

    length = length.to_base_units()
    time = time.to_base_units()
    mass = mass.to_base_units()
    temperature = temperature.to_base_units()
    substance = substance.to_base_units()

    @u.check('[length]', '[time]', '[mass]', '[temperature]', '[substance]')
    def check(length, time, mass, temperature, substance):
        return

    check(length, time, mass, temperature, substance)

    # Get dimensionality
    dlength = dimValue.dimensionality['[length]']
    dtime = dimValue.dimensionality['[time]']
    dmass = dimValue.dimensionality['[mass]']
    dtemp = dimValue.dimensionality['[temperature]']
    dsubstance = dimValue.dimensionality['[substance]']
    factor = (length**(-dlength) *
              time**(-dtime) *
              mass**(-dmass) *
              temperature**(-dtemp) *
              substance**(-dsubstance))

    dimValue *= factor

    if dimValue.unitless:
        return dimValue.magnitude
    else:
        raise ValueError('Dimension Error')

def dimensionalise(value, units):
    """
    Dimensionalise a value.

    Parameters
    ----------
    value : float, int
        The value to be assigned units.
    units : pint units
        The units to be assigned.

    Returns
    -------
    pint quantity: dimensionalised value.

    Example
    -------
    >>> import underworld as uw
    >>> A = uw.scaling.dimensionalise(1.0, u.metre)
    """

    unit = (1.0 * units).to_base_units()

    scaling_coefficients = get_coefficients()

    length = scaling_coefficients["[length]"]
    time = scaling_coefficients["[time]"]
    mass = scaling_coefficients["[mass]"]
    temperature = scaling_coefficients["[temperature]"]
    substance = scaling_coefficients["[substance]"]

    length = length.to_base_units()
    time = time.to_base_units()
    mass = mass.to_base_units()
    temperature = temperature.to_base_units()
    substance = substance.to_base_units()

    @u.check('[length]', '[time]', '[mass]', '[temperature]', '[substance]')
    def check(length, time, mass, temperature, substance):
        return

    # Check that the scaling parameters have the correct dimensions
    check(length, time, mass, temperature, substance)

    # Get dimensionality
    dlength = unit.dimensionality['[length]']
    dtime = unit.dimensionality['[time]']
    dmass = unit.dimensionality['[mass]']
    dtemp = unit.dimensionality['[temperature]']
    dsubstance = unit.dimensionality['[substance]']
    factor = (length**(dlength) *
              time**(dtime) *
              mass**(dmass) *
              temperature**(dtemp) *
              substance**(dsubstance))

    ''' This section needs to be modified for UW3 swarm/mesh vars '''

    if (isinstance(value, uw.discretisation.MeshVariable) or
       isinstance(value, uw.swarm.SwarmVariable)):

        print('swarm/mesh objects not currently supported')

    #     tempVar = value.copy()
    #     tempVar.data[...] = (value.data[...] * factor).to(units).magnitude
    #     return tempVar
    else:
        return (value * factor).to(units)


def ndargs(f):
    """ Decorator used to non-dimensionalise the arguments of a function"""

    def convert(obj):
        if isinstance(obj, (list, tuple)):
            return type(obj)([convert(val) for val in obj])
        else:
            return non_dimensionalise(obj)

    def new_f(*args, **kwargs):
        nd_args = [convert(arg) for arg in args]
        nd_kwargs = {name:convert(val) for name, val in kwargs.items()}
        return f(*nd_args, **nd_kwargs)
    new_f.__name__ = f.__name__
    return new_f


def _units_view(registry, verbose: int = 0, show_scaling: bool = True, show_systems: bool = True):
    """
    Display a notebook-friendly summary of the units registry and scaling system.

    Parameters
    ----------
    verbose : int, default 0
        Verbosity level:
        0 = Basic units and scaling summary
        1 = Include available unit systems and contexts
        2 = Include detailed unit listings by dimension
    show_scaling : bool, default True
        Whether to show current scaling coefficients
    show_systems : bool, default True
        Whether to show available unit systems

    Example
    -------
    >>> uw.scaling.units.view()                    # Basic summary
    >>> uw.scaling.units.view(verbose=1)           # Include systems
    >>> uw.scaling.units.view(verbose=2)           # Detailed units listing
    """
    import textwrap

    # Build markdown content
    lines = []
    lines.append("# Units Registry & Scaling System")
    lines.append(f"**Backend:** {type(registry).__module__}.{type(registry).__name__}")
    lines.append("")

    # Current scaling coefficients
    if show_scaling:
        coeffs = get_coefficients()
        lines.append("## Current Scaling Coefficients")
        lines.append("*These values are used for non-dimensionalisation:*")
        lines.append("")
        for dim_name, value in coeffs.items():
            lines.append(f"- **{dim_name}**: `{value}`")
        lines.append("")

    # Available unit systems
    if show_systems and hasattr(registry, 'systems'):
        lines.append("## Available Unit Systems")
        try:
            systems = list(registry.systems.keys())
            if systems:
                if len(systems) <= 10:
                    for system in sorted(systems):
                        lines.append(f"- `{system}`")
                else:
                    for system in sorted(systems[:8]):
                        lines.append(f"- `{system}`")
                    lines.append(f"- ... and {len(systems) - 8} more systems")
            else:
                lines.append("*No unit systems available*")
        except:
            lines.append("*Unit systems information not available*")
        lines.append("")

    # Context information
    if verbose >= 1 and hasattr(registry, 'contexts'):
        try:
            contexts = list(registry.contexts.keys())
            if contexts:
                lines.append("## Available Contexts")
                lines.append("*Contexts modify unit behavior:*")
                lines.append("")
                for context in sorted(contexts[:10]):  # Limit to avoid overwhelming output
                    lines.append(f"- `{context}`")
                if len(contexts) > 10:
                    lines.append(f"- ... and {len(contexts) - 10} more contexts")
                lines.append("")
        except:
            pass

    # Dimension information
    if verbose >= 2:
        lines.append("## Base Dimensions")
        try:
            # Get base dimensions from registry
            if hasattr(registry, '_dimensions'):
                dimensions = list(registry._dimensions.keys())
                if dimensions:
                    for dim in sorted(dimensions[:15]):  # Limit output
                        lines.append(f"- `[{dim}]`")
                    if len(dimensions) > 15:
                        lines.append(f"- ... and {len(dimensions) - 15} more dimensions")
                else:
                    lines.append("*No base dimensions found*")
            else:
                lines.append("*Dimension information not available*")
        except:
            lines.append("*Could not retrieve dimension information*")
        lines.append("")

    # Common units examples
    lines.append("## Common Units")
    lines.append("*Examples of available units:*")
    lines.append("")

    # Basic unit examples
    common_units = {
        "Length": ["meter", "m", "centimeter", "cm", "kilometer", "km", "inch", "foot"],
        "Mass": ["kilogram", "kg", "gram", "g", "pound", "lb"],
        "Time": ["second", "s", "minute", "min", "hour", "h", "day", "year"],
        "Temperature": ["kelvin", "K", "celsius", "degC", "fahrenheit", "degF"],
        "Force": ["newton", "N", "dyne", "pound_force", "lbf"],
        "Pressure": ["pascal", "Pa", "bar", "atmosphere", "atm", "psi"],
        "Energy": ["joule", "J", "calorie", "cal", "kilowatt_hour", "kWh"]
    }

    for category, units_list in common_units.items():
        available_units = []
        for unit_name in units_list[:4]:  # Check first 4 units
            try:
                # Test if unit exists
                getattr(registry, unit_name)
                available_units.append(f"`{unit_name}`")
            except:
                pass

        if available_units:
            lines.append(f"- **{category}**: {', '.join(available_units)}")

    lines.append("")

    # Usage examples
    lines.append("---")
    lines.append("**Usage Examples:**")
    lines.append("```python")
    lines.append("# Create quantities")
    lines.append("length = 5.0 * uw.scaling.units.meter")
    lines.append("velocity = 10.0 * uw.scaling.units.m / uw.scaling.units.s")
    lines.append("")
    lines.append("# Non-dimensionalise for solvers")
    lines.append("nd_length = uw.scaling.non_dimensionalise(length)")
    lines.append("")
    lines.append("# Convert units")
    lines.append("length_in_cm = length.to('cm')")
    lines.append("```")
    lines.append("")

    # Advanced usage
    if verbose >= 1:
        lines.append("**Advanced Usage:**")
        lines.append("```python")
        lines.append("# Modify scaling coefficients")
        lines.append("coeffs = uw.scaling.get_coefficients()")
        lines.append("coeffs['[length]'] = 1000 * uw.scaling.units.km  # Geological scale")
        lines.append("")
        lines.append("# Unit-aware variables")
        lines.append("velocity = uw.discretisation.MeshVariable('v', mesh, 2, units='m/s')")
        lines.append("```")
        lines.append("")

    # Additional resources
    lines.append("**See Also:**")
    lines.append("- `uw.scaling.get_coefficients()` - Get/modify scaling coefficients")
    lines.append("- `uw.scaling.non_dimensionalise()` - Convert to solver units")
    lines.append("- `uw.scaling.dimensionalise()` - Convert back to dimensional")
    lines.append("- [Pint Documentation](https://pint.readthedocs.io/) - Full units library docs")

    # Display as markdown
    content = "\n".join(lines)
    try:
        from IPython.display import Markdown, display
        display(Markdown(content))
    except (ImportError, NameError):
        # Fallback to plain text if not in Jupyter
        print("=" * 80)
        # Convert markdown to plain text
        plain_text = content.replace("# ", "").replace("## ", "").replace("**", "").replace("`", "'")
        # Remove code blocks
        import re
        plain_text = re.sub(r'```[\s\S]*?```', '[Code examples available in Jupyter]', plain_text)
        print(plain_text)
        print("=" * 80)


# Add the view method to the UnitRegistry instance
u.view = lambda verbose=0, show_scaling=True, show_systems=True: _units_view(u, verbose, show_scaling, show_systems)
