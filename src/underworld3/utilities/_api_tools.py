class Stateful:
    """
    This is a mixin class for underworld objects that are stateful.
    The state of an object is incremented whenever it is modified.
    For example, heavy variables have states, and when a user modifies
    it within its `access()` context manager, its state is incremented
    at the conclusion of their modifications.
    """

    def __init__(self):
        self._state = 0
        super().__init__()

    def _increment(self):
        self._state += 1

    def _get_state(self):
        return self._state


## See this for source: https://stackoverflow.com/questions/28237955/same-name-for-classmethod-and-instancemethod
class class_or_instance_method(object):
    def __init__(self, f):
        self.f = f

    def __get__(self, instance, owner):
        if instance is not None:
            class_or_instance = instance
        else:
            class_or_instance = owner

        def newfunc(*args, **kwargs):
            return self.f(class_or_instance, *args, **kwargs)

        return newfunc


class SymbolicProperty:
    """
    Property descriptor that automatically unwraps symbolic objects.

    This descriptor provides a centralized way to handle symbolic objects in setters
    throughout Underworld3. It automatically unwraps objects that implement the
    _sympify_() protocol, eliminating the need for users to access `.sym` properties.

    Parameters
    ----------
    attr_name : str, optional
        The attribute name to store the value. If not provided, will be set
        automatically using __set_name__.
    matrix_wrap : bool, default False
        If True, automatically wraps scalar values in sympy.Matrix([value])
    allow_none : bool, default True
        If False, raises ValueError when attempting to set to None
    doc : str, optional
        Docstring for the property

    Examples
    --------
    class MySolver:
        # Simple usage - auto-unwraps symbolic objects
        uw_function = SymbolicProperty()

        # With Matrix wrapping for solver compatibility
        source_term = SymbolicProperty(matrix_wrap=True)

        # Disallow None values
        required_field = SymbolicProperty(allow_none=False)

    Notes
    -----
    Objects implementing _sympify_() include:
    - UWexpression
    - UnitAwareDerivativeMatrix (from temperature.diff(y))
    - MeshVariable and SwarmVariable
    - Any custom object with _sympify_() method
    """

    def __init__(self, attr_name=None, matrix_wrap=False, allow_none=True, doc=None):
        self.attr_name = attr_name
        self.matrix_wrap = matrix_wrap
        self.allow_none = allow_none
        self.__doc__ = doc

    def __set_name__(self, owner, name):
        """Called when descriptor is assigned to a class attribute."""
        if self.attr_name is None:
            self.attr_name = f"_{name}"

    def __get__(self, obj, objtype=None):
        """Get the stored value."""
        if obj is None:
            return self
        return getattr(obj, self.attr_name, None)

    def __set__(self, obj, value):
        """Set the value, with automatic unwrapping."""
        # Mark solver as needing setup when property changes
        if hasattr(obj, "is_setup"):
            obj.is_setup = False

        # Check None constraint
        if value is None and not self.allow_none:
            raise ValueError(f"Cannot set {self.attr_name[1:]} to None")

        # Auto-unwrap objects with _sympify_() protocol
        if value is not None and hasattr(value, "_sympify_"):
            value = value._sympify_()

        # Auto-wrap in Matrix if requested
        if self.matrix_wrap and value is not None:
            import sympy

            # Only wrap if not already a Matrix
            if not isinstance(value, sympy.matrices.MatrixBase):
                value = sympy.Matrix([value])

        # Store the value
        setattr(obj, self.attr_name, value)

    def __delete__(self, obj):
        """Delete the stored value."""
        try:
            delattr(obj, self.attr_name)
        except AttributeError:
            pass


class ExpressionDescriptor:
    """
    Unified descriptor for persistent UWexpression containers.

    Creates UWexpression objects ONCE and preserves their identity. The expression's
    .sym content references other expressions for lazy evaluation.

    Used for:
    - Solver templates (F0, F1, PF0) - read-only computed expressions
    - Solver parameters (bodyforce, penalty) - user-settable expressions
    - Constitutive model parameters (viscosity, diffusivity) - user-settable
    - Computed properties (stress, flux) - read-only

    Parameters
    ----------
    name : str or callable
        LaTeX name for the expression. If callable, called with (obj) to get name.
    value_fn : callable
        Function that returns the initial symbolic value. Called ONCE when created.
        Should return references to other expressions for lazy evaluation.
    description : str
        Description of the expression for documentation
    read_only : bool, default False
        If True, prevents user from setting .sym (for templates/computed properties)
    units : str, optional
        Expected units for validation (e.g., "Pa*s", "m/s")
    validator : callable, optional
        Custom validation function called when value is set
    category : str, optional
        Category for introspection: "parameter", "template", "computed"
    attr_name : str, optional
        Attribute name to store the expression. Auto-generated if not provided.

    Examples
    --------
    class MySolver:
        # Parameter (user can change)
        bodyforce = ExpressionDescriptor(
            r"\\mathbf{f}",
            lambda self: sympy.Matrix([[0] * self.mesh.dim]),
            "Body force",
            read_only=False,
            category="parameter"
        )

        # Template (read-only, references parameter)
        F0 = ExpressionDescriptor(
            r"f_0",
            lambda self: -self.bodyforce,  # References bodyforce expression!
            "Force term",
            read_only=True,
            category="template"
        )

    Notes
    -----
    - Expression container created ONCE on first access
    - Object identity preserved (same Python id)
    - value_fn evaluated ONCE to set .sym with expression references
    - Lazy evaluation happens automatically through expression references
    - For parameters: user can update .sym content
    - For templates: .sym is immutable, contains expression references
    """

    def __init__(
        self,
        name,
        value_fn,
        description,
        read_only=False,
        units=None,
        validator=None,
        category=None,
        attr_name=None,
    ):
        self.name = name
        self.value_fn = value_fn
        self.description = description
        self.read_only = read_only
        self.units = units
        self.validator = validator
        self.category = category
        self.attr_name = attr_name

    def __set_name__(self, owner, name):
        """Called when descriptor is assigned to a class attribute."""
        if self.attr_name is None:
            self.attr_name = f"_expr_{name}"
        # Store the public name for introspection
        self.public_name = name

    def __get__(self, obj, objtype=None):
        """Get the persistent expression, creating it if needed."""
        if obj is None:
            return self

        # Check if expression already exists
        expr = getattr(obj, self.attr_name, None)

        if expr is None:
            # Create the expression ONCE
            # Import here to avoid circular imports
            from underworld3.function import expression

            # Get the name (may be dynamic based on object state)
            if callable(self.name):
                name = self.name(obj)
            else:
                name = self.name

            # Evaluate value_fn ONCE to get initial .sym
            # This should return references to other expressions
            try:
                initial_value = self.value_fn(obj)
            except AttributeError:
                # During __init__, some attributes may not exist yet
                # Use a placeholder and will be updated on next access
                initial_value = 0

            # Create persistent expression with unique name generation
            expr = expression(
                name,
                initial_value,
                self.description,
                units=self.units,
                _unique_name_generation=True,
            )

            # Store the expression container
            setattr(obj, self.attr_name, expr)

        return expr

    def __set__(self, obj, value):
        """
        Set the expression's symbolic content.

        For parameters (read_only=False): Updates .sym or copies unit metadata
        For templates (read_only=True): Raises error

        Special handling for UWQuantity:
        - If value is a pure UWQuantity (not UWexpression), copy unit metadata
          without changing the symbolic value to preserve lazy evaluation
        - Otherwise, update .sym as normal
        """
        if self.read_only:
            raise AttributeError(
                f"Cannot set '{self.public_name}' - it is a read-only template expression. "
                f"Modify the parameters it depends on instead."
            )

        # Get or create the expression
        expr = self.__get__(obj, type(obj))

        # Import here to avoid circular dependency
        from ..function.quantities import UWQuantity
        from ..function.expressions import UWexpression

        # Special case: UWQuantity assignment (update ._sym AND copy metadata)
        # Check for pure UWQuantity (not UWexpression which is a subclass)
        if isinstance(value, UWQuantity) and not isinstance(value, UWexpression):
            # Update the symbolic value AND copy unit metadata
            # The expression object (being a SymPy Symbol) preserves identity
            # while ._sym contains the value for JIT substitution
            expr.sym = value._sym  # Update substitution value

            # Copy unit metadata
            if hasattr(value, '_pint_qty'):
                expr._pint_qty = value._pint_qty
            if hasattr(value, '_has_pint_qty'):
                expr._has_pint_qty = value._has_pint_qty
            if hasattr(value, '_dimensionality'):
                expr._dimensionality = value._dimensionality
            if hasattr(value, '_custom_units'):
                expr._custom_units = value._custom_units
            if hasattr(value, '_has_custom_units'):
                expr._has_custom_units = value._has_custom_units
            if hasattr(value, '_model_registry'):
                expr._model_registry = value._model_registry
            if hasattr(value, '_model_instance'):
                expr._model_instance = value._model_instance
            if hasattr(value, '_symbolic_with_units'):
                expr._symbolic_with_units = value._symbolic_with_units
        else:
            # Normal assignment: update symbolic value
            # Run validator if provided
            if self.validator is not None:
                value = self.validator(value)

            # Auto-unwrap if value has _sympify_
            if hasattr(value, "_sympify_"):
                value = value._sympify_()

            # Update the expression's .sym
            expr.sym = value

        # Mark solver/model as needing setup
        if hasattr(obj, "_reset"):
            # For constitutive model Parameters, call _reset() to invalidate setup
            obj._reset()
        elif hasattr(obj, "is_setup"):
            obj.is_setup = False


class Parameter(ExpressionDescriptor):
    """
    Expression descriptor for user-settable parameters.

    Thin wrapper around ExpressionDescriptor with read_only=False and category="parameter".

    Examples
    --------
    class MySolver:
        bodyforce = Parameter(
            r"\\mathbf{f}",
            lambda self: sympy.Matrix([[0] * self.mesh.dim]),
            "Body force vector"
        )

    # User can change it
    solver.bodyforce.sym = new_value
    """

    def __init__(self, name, value_fn, description, units=None, validator=None, **kwargs):
        super().__init__(
            name,
            value_fn,
            description,
            read_only=False,
            units=units,
            validator=validator,
            category="parameter",
            **kwargs,
        )


class Template(ExpressionDescriptor):
    """
    Expression descriptor for read-only template expressions.

    Thin wrapper around ExpressionDescriptor with read_only=True and category="template".

    Templates contain references to other expressions for lazy evaluation. When the owning
    object's `is_setup` flag is False (indicating parameters have changed), the template
    automatically re-evaluates its lambda and updates the expression's symbolic content.

    Examples
    --------
    class MySolver:
        bodyforce = Parameter(r"\\mathbf{f}", ..., "Body force")

        F0 = Template(
            r"f_0",
            lambda self: -self.bodyforce,  # References bodyforce expression
            "Force term"
        )

    # Cannot set directly
    solver.F0.sym = value  # Raises AttributeError

    # But when parameters change:
    solver.bodyforce.sym = new_value  # Sets is_setup = False
    f0 = solver.F0  # Automatically re-evaluates and updates .sym in-place
    """

    def __init__(self, name, value_fn, description, **kwargs):
        super().__init__(name, value_fn, description, read_only=True, category="template", **kwargs)

    def __get__(self, obj, objtype=None):
        """
        Get the persistent expression, re-evaluating if parameters have changed.

        When obj.is_setup is False, re-evaluates the lambda and updates the existing
        expression's .sym in-place, preserving object identity while updating content.
        """
        if obj is None:
            return self

        # Get existing expression (or create if first access)
        expr = super().__get__(obj, objtype)

        # Check if we need to refresh the symbolic content
        # This happens when parameters have changed (is_setup = False)
        if hasattr(obj, "is_setup") and not obj.is_setup:
            try:
                # Re-evaluate the lambda to get updated symbolic content
                updated_value = self.value_fn(obj)

                # Update the expression's .sym IN PLACE (preserves object identity)
                expr._sym = updated_value

            except AttributeError:
                # During setup, some dependencies might not be ready yet
                # This is OK - will be evaluated again on next access
                pass

        return expr


# Backward compatibility aliases
ExpressionProperty = ExpressionDescriptor
TemplateExpression = Template
SymbolicInput = SymbolicProperty


class uw_object:
    """
    The UW (mixin) class adds common functionality that we wish to provide on all uw_objects
    such as the view methods (classmethod for generic information and instance method that can be over-ridden)
    to provide instance-specific information
    """

    _obj_count = 0  # a class variable to count the number of objects

    def __init__(self):
        super().__init__

        self._uw_id = uw_object._obj_count
        uw_object._obj_count += 1

    # to order of the following decorators matters python
    # see - https://stackoverflow.com/questions/128573/using-property-on-classmethods/64738850#64738850
    @classmethod
    def uw_object_counter(cls):
        """Number of uw_object instances created"""
        return uw_object._obj_count

    @property
    def instance_number(self):
        """Unique number of the uw_object instance"""
        return self._uw_id

    def __str__(self):
        s = super().__str__()
        return f"{self.__class__.__name__} instance {self.instance_number}, {s}"

    @staticmethod
    def _reset():
        """Reset the object counter"""
        uw_object._obj_count = 0

    @class_or_instance_method
    def _ipython_display_(self_or_cls):
        from IPython.display import Latex, Markdown, display
        from textwrap import dedent
        import inspect

        ## Docstring (static / class documentation)

        if inspect.isclass(self_or_cls):

            docstring = dedent(self_or_cls.__doc__)
            # docstring = docstring.replace("$", "$").replace("$", "$")
            display(Markdown(docstring))

        else:
            display(
                Markdown(
                    f"**Class**: {self_or_cls.__class__}",
                )
            )
            self_or_cls._object_viewer()

        return

    # View is similar but we can give it arguments to force the
    # class documentation for an instance.

    @class_or_instance_method
    def view(self_or_cls, class_documentation=False):
        from IPython.display import Latex, Markdown, display
        from textwrap import dedent
        import inspect

        ## Docstring (static / class documentation)

        if inspect.isclass(self_or_cls) or class_documentation == True:

            docstring = dedent(self_or_cls.__doc__)
            display(Markdown(docstring))

            if class_documentation:
                display(Markdown("---"))

        if not inspect.isclass(self_or_cls):
            display(
                Markdown(
                    f"**Class**: {self_or_cls.__class__}",
                ),
            )

            self_or_cls._object_viewer()

    # placeholder
    def _object_viewer(self):
        from IPython.display import Latex, Markdown, display

        display(Markdown("# Details"))

        return
