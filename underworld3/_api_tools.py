class Stateful:
    """
    This is a mixin class for underworld objects that are stateful.
    The state of an object is incremented whenever it is modified.  
    For example, heavy variables have states, and when a user modifies
    it within its `access()` context manager, its state is incremented
    at the conclusion of their modifications.
    """
    def __init__(self,*args,**kwargs):
        self._state = 0
        super().__init__(*args,**kwargs)
    def _increment(self):
        self._state +=1
    def _get_state(self):
        return self._state