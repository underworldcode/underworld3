# ### Passive tracer setup test
# Quick test to show how the DMSwarm handles coordinates across multiple processors

import underworld3 as uw
import numpy as np

mesh = uw.meshing.StructuredQuadBox(elementRes =(int(32),int(32)),
                                    minCoords=(0.,0.), 
                                    maxCoords=(1.,1.))

### add a tracer
if uw.mpi.rank ==0: print(f'\ninitial coord setup')
tracer = np.zeros(shape=(1,2))
tracer[:,0] = 0.2
tracer[:,1] = 0.5
print(f'rank: {uw.mpi.rank}, coord: {tracer}')

passiveSwarm = uw.swarm.Swarm(mesh)

passiveSwarm.dm.finalizeFieldRegister()

passiveSwarm.dm.addNPoints(npoints=len(tracer))

passiveSwarm.dm.setPointCoordinates(tracer)


if uw.mpi.rank ==0: print('\nadd coord via dm')
with passiveSwarm.access(passiveSwarm.particle_coordinates):
    print(f'rank: {uw.mpi.rank}, coord: {passiveSwarm.particle_coordinates.data}')

passiveSwarm1 = uw.swarm.Swarm(mesh)

passiveSwarm1.add_particles_with_coordinates(tracer)

if uw.mpi.rank ==0: print('\nadd coord UW functionality')
with passiveSwarm1.access(passiveSwarm.particle_coordinates):
    print(f'rank: {uw.mpi.rank}, coord: {passiveSwarm1.particle_coordinates.data}')


