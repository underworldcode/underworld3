"""Empty-safe DMLabel stratum reads (#589).

``DMLabel.getStratumIS`` hands back a null IS when the stratum is empty on
this rank — the NORMAL case in parallel (a rank owning no part of a fault),
and routine in serial too (a label whose value was never assigned here).
Calling ``getIndices()`` on that null IS is a hard SEGV, and probing
``getStratumIS(v)`` for a value outside the label's live set can abort
outright on some labels (the "Centre" pseudo-label). ``getStratumSize`` is
safe in both situations, so it is the one gate every stratum read goes
through.

This module is a LEAF — numpy only, no underworld3 imports — so both the
discretisation layer and the utilities can use it without cycles.
"""

import numpy as np

__all__ = ["label_stratum_indices"]


def label_stratum_indices(label, value):
    """The point indices of ``label``'s stratum ``value``, always an array.

    Parameters
    ----------
    label : PETSc.DMLabel or None
        The label, as returned by ``dm.getLabel(name)`` (which is ``None``
        when the DM has no label of that name — also handled here).
    value : int
        The stratum value.

    Returns
    -------
    numpy.ndarray
        The stratum's points (this rank's), dtype int64; EMPTY when the
        label is missing or the stratum has no points here. Never raises
        for an absent stratum and never touches a null IS.
    """
    if label is None or label.getStratumSize(int(value)) == 0:
        return np.empty(0, dtype=np.int64)
    iss = label.getStratumIS(int(value))
    if iss is None:
        return np.empty(0, dtype=np.int64)
    return np.asarray(iss.getIndices(), dtype=np.int64)
