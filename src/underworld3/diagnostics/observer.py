"""Public observer interfaces; solver state/finalization live in utilities."""
import hashlib
from pathlib import Path
from underworld3.utilities._stokes_observer import StokesObserver, observe_stokes, solve_with_observer
from underworld3.utilities._field_layout import array_sha256
from underworld3.utilities._spherical_stokes_state import (
    SphericalSolveState, prepare_spherical_replay_fields, assign_spherical_fields,
)

class SphericalStokesObserver(StokesObserver):
    """Optional hooks; subclass only the observations an application needs.

    Source-integrity bookkeeping is owned here, so collectors need no private
    solver access. Overridden constructors should call ``super().__init__()``.
    """
    def __init__(self):
        self._capture_source()

    def _capture_source(self) -> None:
        from underworld3.utilities import _spherical_stokes_gap as native
        self._source_function = native._solve_constraints
        path = Path(native.__file__)
        self._solver_source = dict(native_source_path=str(path),
                                   native_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    @property
    def solver_source(self) -> dict[str, str]:
        return dict(self._solver_source)

    def __enter__(self):
        if not hasattr(self, '_solver_source'):
            self._capture_source()
        return self

    def __exit__(self, kind, value, traceback):
        from underworld3.utilities import _spherical_stokes_gap as native
        unchanged = (native._solve_constraints is self._source_function
                     and hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()
                         == self._solver_source['native_source_sha256'])
        if not unchanged:
            raise RuntimeError('Native spherical solver changed during observation')
        return False











def observe_spherical_stokes(stokes, observer):
    """Compatibility spelling for scoped Stokes observation on a spherical solve."""
    return observe_stokes(stokes, observer, _attribute='_spherical_gap_observer')
