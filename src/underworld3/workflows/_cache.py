"""Cache-key helpers for the workflow product registry.

A *cache key* is a short hex digest derived from the inputs that
produced a workflow product: the relevant Pydantic config fields, plus
(optionally) the cache keys of upstream products this one depends on.
Two products with the same cache key are guaranteed to be byte-
equivalent under deterministic producers; mismatched cache keys mean
"the inputs changed, the cached artefact is stale, rebuild".

The helpers here are intentionally workflow-agnostic.  Each workflow
declares its own *identity fields* — the subset of its config whose
change should invalidate cached products — and passes them in.

Used by both :class:`underworld3.workflows.Run` (the time-loop
run-directory) and :class:`underworld3.workflows.WorkflowProducts`
(the product-graph cache) so the freshness story is one story.
"""

from __future__ import annotations

import hashlib
import json
from typing import Iterable, Mapping, Optional

# Cache-key digests are truncated to this many hex chars.  Long enough
# to make accidental collisions vanishingly unlikely for the tens-of-
# thousands of products a long-running project will ever cache; short
# enough to keep manifests human-readable.
_DIGEST_LEN = 16


def config_cache_key(
    config,
    identity_fields: Iterable[str],
    requires: Optional[Mapping[str, str]] = None,
) -> str:
    """Return a cache key derived from *config*'s identity fields.

    Parameters
    ----------
    config : Pydantic BaseModel-like
        The configuration object.  Each name in *identity_fields* is
        read off it via ``getattr``.
    identity_fields : iterable of str
        The names of fields whose values determine the cache key.
        Other config fields (e.g. operational tolerances) are ignored.
    requires : mapping str -> str, optional
        Cache keys of upstream products this one depends on, keyed by
        product name.  Folded into the digest so an upstream
        invalidation propagates downstream.

    Returns
    -------
    digest : str
        Hex digest of length ``_DIGEST_LEN``.

    Examples
    --------
    >>> # Convection's identity hash
    >>> config_cache_key(config, ["rayleigh", "cellsize", "T_degree"])
    '32f71431391477a7'

    >>> # A downstream aggregation depending on multiple upstream products
    >>> config_cache_key(
    ...     sweep_config,
    ...     ["rayleigh_values", "aspect_ratios"],
    ...     requires={"run_summary_Ra1e4": "abc...", "run_summary_Ra1e5": "def..."},
    ... )
    """
    fields = {f: getattr(config, f) for f in identity_fields}
    if requires:
        # Wrapped payload — distinguishes "depends on upstream products"
        # from the no-deps case so a product gaining its first dep
        # doesn't silently keep the same digest.
        payload = {"config": fields, "requires": dict(requires)}
        s = json.dumps(payload, sort_keys=True, default=str)
    else:
        # No upstream deps: hash the bare field dict.  Byte-compatible
        # with the legacy convection ``_config_hash``, so pre-0.2 live
        # runs continue to resume cleanly.
        s = json.dumps(fields, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:_DIGEST_LEN]


def config_snapshot(config, identity_fields: Iterable[str]) -> dict:
    """Return ``{field: getattr(config, field)}`` for the listed fields.

    Mirror of :func:`config_cache_key`'s input handling — the snapshot
    is what the cache key was derived from, persisted alongside the
    digest so a future reader can audit what changed.
    """
    return {f: getattr(config, f) for f in identity_fields}


__all__ = ["config_cache_key", "config_snapshot"]
