"""Model-dwelling tracker — snapshot-managed evolving state.

``Model.tracker`` is the authoritative *record* of where a run is:
simulation time, step, dt, plus any user-registered quantities. It is
deliberately NOT something solvers depend on — solvers and DDt are
untouched, and a user need not use the tracker at all. Its one
superpower: everything living in the tracker is automatically
captured by ``Model.snapshot()`` and reverted by ``Model.restore()``,
whereas a loose Python variable (``model_time = 0.0`` in a script) is
not.

Add managed quantities by plain attribute assignment::

    model.tracker.time = 0.0
    model.tracker.step = 0
    model.tracker.my_diagnostic = np.zeros(3)

Any attribute set on the tracker whose name does not start with an
underscore is a managed state variable: part of every snapshot,
restored exactly on rollback, with no special status in solvers and
no dataclass authoring required. Underscore-prefixed names are
internal and not managed.

``time``, ``step`` and ``dt`` are ordinary managed entries pre-seeded
with sensible defaults (``0.0`` / ``0`` / ``None``). They are
conventions, not privileged fields — consistent with the design
intent that user-added quantities are first-class.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from underworld3.utilities._api_tools import uw_object

from .state import SnapshottableState


@dataclass
class TrackerState(SnapshottableState):
    """Snapshot of a :class:`ModelTracker`.

    The tracker is extensible, so the State carries an open mapping
    rather than fixed fields. ``time`` / ``step`` / ``dt`` are
    ordinary entries in ``managed``.
    """

    _schema_version: int = 1
    managed: dict = field(default_factory=dict)


class ModelTracker(uw_object):
    """One per :class:`underworld3.Model`, auto-registered as a
    :class:`~underworld3.checkpoint.Snapshottable` state-bearer so the
    snapshot machinery captures and restores it with no extra
    plumbing. See the module docstring for the user-facing contract.
    """

    def __init__(self):
        # _managed must exist before any public attribute assignment
        # routes through __setattr__.
        object.__setattr__(
            self, "_managed", {"time": 0.0, "step": 0, "dt": None}
        )
        super().__init__()  # uw_object: sets self._uw_id (underscore)

    # --- attribute routing: public -> managed, underscore -> real ---

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        # Respect class-level data descriptors — notably the `state`
        # property. Without this guard, `tracker.state = ...` (done by
        # the snapshot machinery on restore) would be captured as a
        # managed quantity instead of invoking the property setter,
        # and restore would silently no-op. `state` is therefore a
        # reserved name and cannot be a user-managed quantity.
        cls_attr = getattr(type(self), name, None)
        if hasattr(cls_attr, "__set__") or hasattr(cls_attr, "__get__"):
            object.__setattr__(self, name, value)
            return
        self._managed[name] = value

    def __getattr__(self, name):
        # __getattr__ only fires when normal lookup fails, so it never
        # shadows real attributes or class properties (state,
        # instance_number, ...).
        if name.startswith("_"):
            raise AttributeError(name)
        managed = object.__getattribute__(self, "_managed")
        if name in managed:
            return managed[name]
        raise AttributeError(
            f"ModelTracker has no managed quantity {name!r}; assign "
            f"model.tracker.{name} = ... to create it"
        )

    def __delattr__(self, name):
        if name.startswith("_"):
            object.__delattr__(self, name)
        elif name in self._managed:
            del self._managed[name]
        else:
            raise AttributeError(name)

    # --- convenience ---

    def __contains__(self, name):
        return name in self._managed

    def keys(self):
        """Names of all managed quantities (including time/step/dt)."""
        return list(self._managed.keys())

    def __repr__(self):
        items = ", ".join(f"{k}={v!r}" for k, v in self._managed.items())
        return f"ModelTracker({items})"

    # --- Snapshottable contract ---

    @property
    def state(self) -> TrackerState:
        # Deep-copy on read so a held .state is isolated from later
        # mutation even if not routed through the snapshot machinery.
        return TrackerState(managed=copy.deepcopy(self._managed))

    @state.setter
    def state(self, s: TrackerState) -> None:
        if s._schema_version != TrackerState._schema_version:
            raise ValueError(
                f"TrackerState schema version mismatch: snapshot "
                f"{s._schema_version} vs current "
                f"{TrackerState._schema_version}"
            )
        # Replace wholesale: restore returns to exactly the captured
        # point, so a quantity added *after* the snapshot is dropped
        # on restore (git-stash semantics).
        object.__setattr__(self, "_managed", copy.deepcopy(s.managed))
