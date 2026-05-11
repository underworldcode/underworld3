"""State-as-dataclass contract for snapshot capture/restore.

Per the design note's ``General serialisation contract for solver-internal
state'' section, the chosen pattern for new solver-internal classes is
**option (C): state as a first-class dataclass**. A class declares a
``state`` attribute that is a dataclass; the snapshot mechanism reads it
automatically.

For existing classes being retrofitted (DDt today, parameter mutation
history next, any solver convergence-tracking state after that), the
design recommends **option (B)-style adapters**: a ``state`` property
that *derives* a dataclass from the existing private attributes, and a
matching setter that writes the dataclass values back. The dataclass is
not the authoritative store — the private attrs are — but the snapshot
mechanism only sees the dataclass, so the snapshot/restore contract is
the same as for option (C).

This module defines:

- :class:`SnapshottableState` — the dataclass base every state object
  should inherit / mimic. Carries the ``_schema_version`` field that
  drives the v1.1 on-disk migration registry.
- :class:`Snapshottable` — a structural protocol; an object is
  Snapshottable if it exposes a ``state`` attribute that is a
  :class:`SnapshottableState`. Used by ``Model.snapshot()`` discovery.

The actual State dataclasses for specific classes (``DDtState``,
``ParameterRegistryState``) live next to those classes in the systems
/ utilities packages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class SnapshottableState:
    """Base for every per-class State dataclass.

    Subclasses add their own fields. The single mandatory field is
    ``_schema_version`` — an integer that v1.1's on-disk migration
    registry uses to lift older snapshots to the current schema. In
    v1 (in-memory only) the version is checked for strict equality;
    any mismatch is a programming error since capture and restore
    happen in the same process.
    """

    _schema_version: int = 1


@runtime_checkable
class Snapshottable(Protocol):
    """Structural protocol for state-bearing objects.

    An object is Snapshottable if::

        obj.state                   # readable
        obj.state = obj.state       # writable
        isinstance(obj.state, SnapshottableState)

    The snapshot mechanism uses :attr:`state` to capture and restore.
    Implementations choose between:

    - **Option (C), authoritative dataclass.** ``state`` is a stored
      attribute holding the dataclass; every mutation site on the class
      writes ``self.state.<field>`` directly. Best for new code.

    - **Option (B), derived dataclass.** ``state`` is a property that
      builds the dataclass from existing private attrs on each read,
      and the setter writes attrs back. Best for retrofits of existing
      classes (DDt, ParameterRegistry).
    """

    @property
    def state(self) -> SnapshottableState: ...
