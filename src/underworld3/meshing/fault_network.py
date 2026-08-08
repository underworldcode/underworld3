"""The user-facing 2-D fault-network toolkit.

One object carries the validated network recipe end to end:

1. **Hierarchy-respecting junction preparation** — where faults cross
   or abut, the junior trace is severed and pulled back a short
   ligament; the senior runs through (:func:`prepare_fault_network`).
2. **Network-refined meshing** — a graded mesh following every trace,
   then split-node faults cut along the prepared pieces.
3. **Contact** — the no-opening pair constraint on every piece.
4. **Damage-zone glue** — small viscoplastic plugs at the junctions
   connect the network mechanically; the stress lobes of the abutting
   tips decide how slip transfers, no reconnection geometry is ever
   prescribed.

The dials encode the measured rulings (2026-08): junction gaps of one
or two elements transmit slip within a few percent of a continuous
fault and cost nothing in solver health; the glue's strength and
regularisation dial move TOGETHER (``dial``), with zone stress
proportional to the dial down to ~100x viscosity contrast before
anything degrades — the compact plug never triggers thin-inclusion
conditioning. Oriented (TI) damage in the plug is deliberately NOT
offered: a junction must accommodate corner-turning deformation,
which a weak-plane fabric cannot relax (measured; see the study
README under ``~/+Simulations/fault_junction_rheology``).

Example
-------
>>> net = uw.meshing.FaultNetwork(
...     [("Main", main_pts), ("Splay", splay_pts)],
...     hierarchy=["Main", "Splay"])          # Main severs Splay
>>> mesh = net.prepare(h=0.006).build()
>>> stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
>>> stokes.constitutive_model = uw.constitutive_models.ViscoPlasticFlowModel
>>> stokes.constitutive_model.yield_mode = "min"
>>> stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
>>> stokes.constitutive_model.Parameters.yield_stress = \\
...     net.damage_yield(v, dial=0.05)
>>> stokes.consistent_jacobian = True
>>> net.apply_contact(stokes)
>>> # ... boundary conditions ...
>>> info = net.solve(stokes)
>>> net.slips(stokes)
{'Main': 0.14, 'Splay_1': 0.05, 'Splay_2': 0.04}
"""

import numpy as np
import sympy

from .surfaces import Surface, prepare_fault_network


class FaultNetwork:
    """A hierarchy of 2-D fault traces meshed, split, and glued.

    Parameters
    ----------
    faults : sequence of (name, points)
        The RAW traces (open polylines, ``(N, 2)`` or ``(N, 3)``),
        exactly as imported — crossings and abutments included; the
        toolkit converts them.
    hierarchy : sequence of str, optional
        Names in seniority order (most major first). Junior traces are
        severed by senior ones at crossings. Default: the input order.
    """

    def __init__(self, faults, hierarchy=None):
        self.faults = [(str(n), np.asarray(p, dtype=float)[:, :2])
                       for n, p in faults]
        names = [n for n, _ in self.faults]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate fault names: {names}")
        self.hierarchy = list(hierarchy) if hierarchy is not None \
            else list(names)
        unknown = set(self.hierarchy) - set(names)
        if unknown:
            raise ValueError(f"hierarchy names not in the network: "
                             f"{sorted(unknown)}")
        self.h_near = None
        self.prepared = None
        self.junctions = None
        self.report = None
        self.mesh = None

    # ------------------------------------------------------------------
    def prepare(self, h, ligament=2.0, through=None, verbose=True):
        """Convert crossings/abutments to offset junctions.

        ``h`` is the near-fault mesh size the network will be built at;
        ``ligament`` (in multiples of ``h``) sets the junction gap —
        the measured floor is 1, the default 2 gives one clear cell of
        intact material and slip transmission within ~5% of a
        continuous fault. ``through`` declares ABSOLUTE masters (never
        cut anywhere); the hierarchy handles everything else pairwise.
        """
        self.h_near = float(h)
        self.prepared, self.report, self.junctions = \
            prepare_fault_network(
                self.faults, spacing=self.h_near, ligament=ligament,
                through=through, hierarchy=self.hierarchy,
                verbose=verbose, return_junctions=True)
        return self

    # ------------------------------------------------------------------
    def build(self, base=None, h_far=None, band=0.03, ramp=0.08,
              max_levels=2, qdegree=2):
        """Mesh the network: graded refinement along every RAW trace,
        then split-node faults along the PREPARED pieces.

        ``base`` is an existing coarse mesh to adapt (default: a unit
        ``UnstructuredSimplexBox`` at ``h_far = 4 h``); the refinement
        holds ``h`` within ``band`` of any trace and grades to
        ``h_far`` over ``ramp``.
        """
        if self.prepared is None:
            raise RuntimeError("call prepare(h=...) first")
        from .cartesian import UnstructuredSimplexBox

        h = self.h_near
        h_far = 4.0 * h if h_far is None else float(h_far)
        if base is None:
            base = UnstructuredSimplexBox(cellSize=h_far, refinement=1,
                                          qdegree=qdegree)
        surfs = []
        for n, p in self.faults:            # RAW traces: the metric must
            s = Surface(f"FN_{n}", base,     # refine the junction gaps too
                        np.column_stack([p, np.zeros(len(p))]),
                        symbol=n[:1])
            s.discretize()
            surfs.append(s)

        def metric(pts_, _ss=surfs, _h=h, _hf=h_far, _b=band, _r=ramp):
            d = np.min([s_.unsigned_distance(pts_) for s_ in _ss], axis=0)
            hh = np.where(d < _b, _h,
                          np.minimum(_h + (_hf - _h) * (d - _b) / _r, _hf))
            return 1.0 / hh ** 2

        child = base.adapt(metric, max_levels=max_levels)
        self.mesh = child.add_fault(
            [(n, p.copy()) for n, p in self.prepared])
        return self.mesh

    # ------------------------------------------------------------------
    def apply_contact(self, solver, conds=0):
        """Register the split-node contact on every prepared piece."""
        if self.mesh is None:
            raise RuntimeError("call build() first")
        for name, _p in self.prepared:
            solver.add_fault_bc(conds, boundary=name)
        return self

    # ------------------------------------------------------------------
    def damage_yield(self, velocity, dial=0.05, radius=None,
                     tau_far=1.0e3):
        """The junction glue: a regularised viscoplastic yield stress.

        Inside a small plug at each junction the yield is
        ``dial + 2 dial * edot_II`` — strength and the xi-style
        rate regularisation share ONE dial, per the weakness-ladder
        ruling (moving them apart makes the solve harsh without making
        the zone weaker). Zone stress scales in proportion to the
        dial; 0.05 is near-invisible at unchanged solver cost, 0.01
        reaches the transmission ceiling at roughly double cost.
        Regions are sharp ``Piecewise`` — never blend a rheological
        parameter against a large sentinel through a smooth mask.

        ``velocity`` is the solver's velocity variable (the strain
        rate is formed symbolically from it). ``radius`` defaults to
        ``max(2.5 h, 1.2 * pull)`` per junction.
        """
        if self.mesh is None:
            raise RuntimeError("call build() first")
        if self.junctions is None or len(self.junctions) == 0:
            return sympy.sympify(tau_far)
        x, y = self.mesh.X[0], self.mesh.X[1]
        exx = velocity.sym[0].diff(x)
        eyy = velocity.sym[1].diff(y)
        exy = (velocity.sym[0].diff(y) + velocity.sym[1].diff(x)) / 2
        einv2 = sympy.sqrt(exx ** 2 / 2 + eyy ** 2 / 2 + exy ** 2)
        tau_plug = float(dial) * (1 + 2 * einv2)
        expr = sympy.sympify(float(tau_far))
        for j in self.junctions:
            P = np.asarray(j["point"], dtype=float)
            R = (float(radius) if radius is not None
                 else max(2.5 * self.h_near, 1.2 * float(j["pull"])))
            r2 = (x - float(P[0])) ** 2 + (y - float(P[1])) ** 2
            expr = sympy.Min(expr, sympy.Piecewise(
                (tau_plug, r2 < R ** 2), (float(tau_far), True)))
        return expr

    # ------------------------------------------------------------------
    def solve(self, solver, **kwargs):
        """Solve with the fault contact imposed (thin convenience
        wrapper over ``fault_contact.solve_with_fault``)."""
        from underworld3.utilities.fault_contact import solve_with_fault
        return solve_with_fault(solver, **kwargs)

    # ------------------------------------------------------------------
    def slips(self, solver):
        """Peak tangential slip per prepared piece (rank-local)."""
        from underworld3.utilities.fault_contact import fault_pair_jumps
        info = getattr(solver, "_rotated_freeslip_info", None)
        if info is None:
            raise RuntimeError("no fault solve on this solver yet")
        out = {}
        for name, _p in self.prepared:
            coords, jumps, normals = fault_pair_jumps(solver, name, info)
            tang = np.column_stack([-normals[:, 1], normals[:, 0]])
            out[name] = float(np.abs(
                np.einsum("ij,ij->i", jumps, tang)).max())
        return out

    # ------------------------------------------------------------------
    def __repr__(self):
        n_j = len(self.junctions) if self.junctions is not None else "?"
        n_p = len(self.prepared) if self.prepared is not None else "?"
        state = ("meshed" if self.mesh is not None else
                 "prepared" if self.prepared is not None else "raw")
        return (f"FaultNetwork({len(self.faults)} faults -> {n_p} "
                f"pieces, {n_j} junctions, {state}; "
                f"hierarchy={self.hierarchy})")
