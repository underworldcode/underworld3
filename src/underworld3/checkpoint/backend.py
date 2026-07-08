"""Storage protocol for snapshot tokens.

The protocol is shaped from day one to support both an in-memory backend
(dict of numpy arrays, v1) and an on-disk full-state backend (HDF5,
v1.1). Per the design note, the in-memory backend is the cheapest
correctness test of the abstraction; the on-disk backend locks in the
byte-level serialisation contract.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class CheckpointBackend(Protocol):
    """Backing-store interface for :class:`underworld3.checkpoint.Snapshot`.

    Implementations
    ---------------
    - :class:`InMemoryBackend` — v1; numpy arrays held in process memory.
    - ``OnDiskFullStateBackend`` — v1.1; single monolithic HDF5 file.

    Vectors are bulk numerical data; metadata is small scalars / dicts /
    lists describing structure and provenance.
    """

    def save_vector(self, key: str, array: np.ndarray) -> None: ...

    def load_vector(self, key: str) -> np.ndarray: ...

    def save_metadata(self, key: str, value: Any) -> None: ...

    def load_metadata(self, key: str) -> Any: ...

    def list_vectors(self) -> list[str]: ...

    def list_metadata(self) -> list[str]: ...


class InMemoryBackend:
    """Snapshot storage in process memory.

    Eager-copy on both ``save_vector`` and ``load_vector`` per the v1
    scope-boundary (no lazy / copy-on-write semantics). Per-snapshot
    byte cost is the sum of captured vector sizes — expected to be
    bounded by one Stokes solve's working memory for typical setups.
    """

    def __init__(self) -> None:
        self._vectors: dict[str, np.ndarray] = {}
        self._metadata: dict[str, Any] = {}

    def save_vector(self, key: str, array: np.ndarray) -> None:
        if key in self._vectors:
            raise KeyError(f"vector key already present in snapshot: {key!r}")
        self._vectors[key] = np.asarray(array).copy()

    def load_vector(self, key: str) -> np.ndarray:
        if key not in self._vectors:
            raise KeyError(f"vector key not in snapshot: {key!r}")
        return self._vectors[key].copy()

    def save_metadata(self, key: str, value: Any) -> None:
        if key in self._metadata:
            raise KeyError(f"metadata key already present in snapshot: {key!r}")
        self._metadata[key] = value

    def load_metadata(self, key: str) -> Any:
        if key not in self._metadata:
            raise KeyError(f"metadata key not in snapshot: {key!r}")
        return self._metadata[key]

    def list_vectors(self) -> list[str]:
        return list(self._vectors.keys())

    def list_metadata(self) -> list[str]:
        return list(self._metadata.keys())
