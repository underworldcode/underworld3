"""Cylindrical physical gauges and solver-owned field finalization."""
import numpy as np
from underworld3.utilities._field_layout import _scatter_values
from underworld3.utilities._stokes_state import StokesSolveState

class CylindricalSolveState(StokesSolveState):
    """Borrowed annulus operator and physical pressure/rotation gauge data."""
    def restore_gauge(self, vector):
        e = self._environment
        vector.axpy(-(e['pmean'].dot(vector) + self.lift_integral) / e['pmean'].sum(), e['pressure_mode'])
        if e['components'] == 1:
            vector.axpy(-e['rotation_weight'].dot(vector) / e['rotation_weight'].dot(e['rotation_mode']), e['rotation_mode'])

    def assign_fields(self, solution):
        e = self._environment
        velocity, pressure, mesh = (e[n] for n in ('velocity', 'pressure', 'mesh'))
        umap, pmap, boundary_uids = (e[n] for n in ('umap', 'pmap', 'boundary_uids'))
        nu, npres, ncomponents, reactions = (e[n] for n in ('nu', 'npres', 'components', 'reactions'))
        velocity.array[:, 0, :] = _scatter_values(solution, umap).reshape(-1, 2)
        pressure.array[:, 0, 0] = _scatter_values(solution, pmap + nu)
        global_node_ids = umap.reshape(-1, 2)[:, 0]
        selected = np.isin(global_node_ids, boundary_uids)
        for component, reaction in enumerate(reactions):
            reaction_values = np.zeros(len(global_node_ids))
            reaction_values[selected] = _scatter_values(
                solution,
                nu
                + npres
                + ncomponents * np.searchsorted(boundary_uids, global_node_ids[selected])
                + component,
            )
            # Full canonical writes synchronize empty boundary selections too.
            reaction.array[:, 0, 0] = reaction_values
        for field in (velocity, pressure, *reactions):
            field._sync_lvec_to_gvec()
        mesh._stale_lvec = True
        return tuple(reactions)
