import underworld3 as uw
import numpy as np

# Quick test to see if field access is working
from underworld3.meshing import UnstructuredSimplexBox

mesh = UnstructuredSimplexBox(
    minCoords=(0.0, 0.0),
    maxCoords=(1.0, 1.0),
    cellSize=1.0 / 8.0,
)

swarm = uw.swarm.Swarm(mesh=mesh)
s_values = uw.swarm.SwarmVariable('test', swarm, 1, proxy_degree=1)

swarm.populate(fill_param=2)

print('Testing data property access...')
try:
    # This should trigger the data property
    s_values.data[:, 0] = np.cos(np.pi * swarm._particle_coordinates.data[:, 0])
    print('✓ Data property access successful')

    # Try accessing again to test caching
    s_values.data[:, 0] = np.sin(np.pi * swarm._particle_coordinates.data[:, 1])
    print('✓ Second data property access successful')
    
    print('Field access working correctly!')
    
except Exception as e:
    print(f'✗ Error with data property: {e}')
    import traceback
    traceback.print_exc()