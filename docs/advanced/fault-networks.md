# Fault networks: hierarchy, junctions, and damage-zone glue

Real fault systems cross, branch, and abut. The split-node formulation
(see [Split-node faults](split-node-faults.md)) deliberately refuses
shared vertices — a junction vertex would need a non-binary contact
pairing — so a network is represented in the **offset form**: at every
junction the junior trace stops a cell or two short of the senior one,
and a small **damage zone** of viscoplastic material connects them.
`FaultNetwork` packages that whole recipe.

```python
import underworld3 as uw

net = uw.meshing.FaultNetwork(
    [("Main", main_pts), ("Splay", splay_pts), ("Cross", cross_pts)],
    hierarchy=["Main", "Splay", "Cross"])   # seniority order

mesh = net.prepare(h=0.006).build()          # junctions -> mesh -> split

v = uw.discretisation.MeshVariable("V", mesh, 2, degree=2)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=0,
                                   continuous=False)
stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscoPlasticFlowModel
stokes.constitutive_model.yield_mode = "min"
stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
stokes.constitutive_model.Parameters.yield_stress = \
    net.damage_yield(v, dial=0.05)           # the junction glue
stokes.consistent_jacobian = True
net.apply_contact(stokes)                    # no-opening pairs, all pieces
# ... wall boundary conditions ...
info = net.solve(stokes)
print(net.slips(stokes))                     # peak slip per piece
```

## The recipe, and why each piece is the way it is

**Hierarchy.** At an X crossing the senior fault runs through and the
junior is severed and pulled back (`hierarchy`, pairwise). A trace that
*ends* on another (T abutment) is always the one trimmed — an endpoint
cannot run through. Absolute masters (`through=`) override rank.

**Small junctions.** The pull-back is `ligament * h` (default 2
elements). Measured on a collinear bridge: gaps from 1 to 6 elements
all solve at identical cost, slip transmission varies by only ~7%
(monotone — shorter is better), and one element of gap resolves at the
same answer when the junction patch is refined 2x. Make the join as
small as the mesh allows; buy fidelity with elements, not physical
size.

**The glue.** `damage_yield` places a compact viscoplastic plug at
each junction: yield `dial * (1 + 2 * edot_II)` inside, effectively
rigid outside, sharp `Piecewise` boundaries. The strength and the
rate-regularisation move together on ONE dial (separating them makes
the solve harsh without making the zone weaker). Zone stress is
proportional to the dial down to a ~100x viscosity contrast with
Newton-from-cold still converging — the compact plug conditions like a
hole, not like a thin weak layer, so the classic thin-inclusion
Schur breakdown never appears. `dial=0.05` is near-invisible in the
stress field at unchanged cost; `dial=0.01` reaches the transmission
ceiling of an inviscid plug at roughly double cost.

**No prescribed reconnection.** Nothing tells the network how to link
up: the stress lobes of the abutting tips decide. A collinear gap
grows a straight band; a branch junction feeds whichever limb the
drive favours, and switches when the drive rotates.

**Isotropic glue, deliberately.** Oriented (transversely isotropic)
weakness in the plug cannot relax the corner-turning deformation a
junction must accommodate — a weak-plane fabric only softens shear ON
the plane, and the measured junction stress stays high no matter how
weak the plane is made. Use TI damage along fault *zones* (where
deformation is plane-parallel); junctions get isotropic breakdown,
which is also what junction breccia looks like in the field.

## Baseline discipline

The stateless control for any network experiment is a plug of constant
weak viscosity — it transmits and partitions slip essentially
identically to the damage glue (measured within ~5%). What damage adds
is **state**: wear-in and healing with a slip-rate memory, which only
changes the *instantaneous* mechanics once elasticity (stored stress)
enters. Judge any refinement of the glue against that control.

## Limitations

- 2-D traces; the 3-D network (FaultSurface junction preparation) is
  not yet wired into this class.
- One damage dial per network in `damage_yield` (per-junction values:
  build the expression with `uw.meshing.damage_zone_yield` directly).
- Time-dependent damage (wear-in/healing) is study-level for now: see
  `~/+Simulations/fault_junction_rheology/damage_switch.py` for the
  pattern (per-fault scalar states driven by each fault's slip rate).
