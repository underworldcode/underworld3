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


class counted_metaclass(type):
    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        cls._total_instances = 0


class uw_object_counter(object, metaclass=counted_metaclass):
    def __init__(self):
        try:
            self.__class__.mro()[1]._total_instances += 1
        except AttributeError:
            pass
            # print(f"{self.__class__.mro()[1]} is not a uw_object")

        super().__init__()

        self.__class__._total_instances += 1
        self.instance_number = self.__class__._total_instances


class uw_object(uw_object_counter):
    """
    The UW (mixin) class adds common functionality that we wish to provide on all uw_objects
    such as the view methods (classmethod for generic information and instance method that can be over-ridden)
    to provide instance-specific information
    """

    def __init__(self):
        super().__init__()

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

    @classmethod
    def total_instances(cls):
        return cls._total_instances

    # placeholder
    def _object_viewer(self):
        from IPython.display import Latex, Markdown, display

        display(Markdown("## Details"))

        return
