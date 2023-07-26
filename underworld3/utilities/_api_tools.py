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


class uw_object:
    """
    The UW (mixin) class adds common functionality that we wish to provide on all uw_objects
    such as the view methods (classmethod for generic information and instance method that can be over-ridden)
    to provide instance-specific information
    """

    @class_or_instance_method
    def _ipython_display_(self_or_cls):
        from IPython.display import Latex, Markdown, display
        from textwrap import dedent
        import inspect

        ## Docstring (static)
        docstring = dedent(self_or_cls.__doc__)
        # docstring = docstring.replace("$", "$").replace("$", "$")
        display(Markdown(docstring))

        if inspect.isclass(self_or_cls):
            return

        # if this is an object, call the object-specific view method
        self_or_cls._object_viewer()

        return

    @class_or_instance_method
    def view(self_or_cls):
        self_or_cls._ipython_display_()
        return

    # placeholder
    def _object_viewer(self):
        from IPython.display import Latex, Markdown, display

        display(Markdown("## Details"))

        return
