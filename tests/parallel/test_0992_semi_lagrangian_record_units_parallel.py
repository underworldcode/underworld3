"""The semi-Lagrangian record of a field with units is right at any rank count
(#992).

At np > 1 the record copies the field's nodal values directly. It read the
dimensional ``var.array``, stripped the units and stored the number as
non-dimensional: T in K holding 0.5 (500 K at a 1000 K reference) recorded 500
at np 2 and 0.5 at np 1. It now copies the non-dimensional ``var.data``. With
zero velocity the record must equal the field, with or without the clamp.
"""
import numpy as np
import pytest
from mpi4py import MPI

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.timeout(240, method="thread")]


@pytest.fixture
def units_model():
    uw.get_default_model().set_reference_quantities(
        length=uw.quantity(1000, "km"), time=uw.quantity(1, "Myr"),
        mass=uw.quantity(1e24, "kg"), temperature=uw.quantity(1000, "K"),
    )
    yield
    uw.reset_default_model()


@pytest.mark.parametrize("monotone", [None, "clamp"])
def test_the_record_equals_the_field_at_rest(units_model, monotone):
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.125
    )
    T = uw.discretisation.MeshVariable("T_992", mesh, 1, degree=1, units="K")
    T.data[:, 0] = 0.5
    V = uw.discretisation.MeshVariable("V_992", mesh, 2, degree=1,
                                       vtype=uw.VarType.VECTOR, units="cm/yr")
    V.data[:] = 0.0
    kw = dict(monotone_mode=monotone) if monotone else {}
    sl = uw.systems.ddt.SemiLagrangian(mesh, T.sym, V.sym, vtype=uw.VarType.SCALAR,
                                       degree=1, continuous=True, order=1,
                                       units="K", **kw)
    sl.update_pre_solve(0.01)
    ps = np.asarray(sl.psi_star[0].data)
    lo = uw.mpi.comm.allreduce(float(ps.min(initial=np.inf)), op=MPI.MIN)
    hi = uw.mpi.comm.allreduce(float(ps.max(initial=-np.inf)), op=MPI.MAX)
    assert lo == pytest.approx(0.5) and hi == pytest.approx(0.5)
