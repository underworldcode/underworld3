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

class uw_object():
    """
    The UW (mixin) class adds common functionality that we wish to provide on all uw_objects
    such as the view methods (classmethod for generic information and instance method that can be over-ridden)
    to provide instance-specific information
    """

    _obj_count = 0 # a class variable to count the number of objects

    def __init__(self):
        super().__init__

        self._uw_id = uw_object._obj_count
        uw_object._obj_count += 1

    # to order of the following decorators matters python
    # see - https://stackoverflow.com/questions/128573/using-property-on-classmethods/64738850#64738850
    @classmethod
    def uw_object_counter(cls):
        """ Number of uw_object instances created """
        return uw_object._obj_count

    @property
    def instance_number(self):
        """ Unique number of the uw_object instance """
        return self._uw_id

    def __str__(self):
        s = super().__str__()
        return f"{self.__class__.__name__} instance {self.instance_number}, {s}"

    @staticmethod
    def _reset():
        """ Reset the object counter """
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
            # docstring = docstring.replace("$", "$").replace("$", "$")
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

        display(Markdown("## Details"))

        return
