# Variable Statistics Expansion Plan

## Current State
- **Scalar variables**: `stats()` returns dictionary with min, max, mean, etc.
- **Vector/tensor variables**: `NotImplementedError` 
- **Swarm variables**: No stats() method at all

## Problem Statement
Users need statistics for:
1. **Vector variables**: velocity, displacement, force fields
2. **Tensor variables**: stress, strain, permeability tensors  
3. **Swarm variables**: particle properties, material tracking

## Proposed Implementation Strategy

### 1. Vector/Tensor Statistics Framework

#### Core Concept: Magnitude-Based Stats
For vectors and tensors, compute statistics on derived scalar quantities:

```python
def stats(self):
    """Extended stats for vector/tensor variables."""
    if self.num_components == 1:
        # Existing scalar implementation
        return scalar_stats()
    
    elif self.num_components > 1:
        # Vector/tensor implementation
        return vector_tensor_stats()
```

#### Vector Stats (velocity, displacement, etc.)
```python
def vector_stats(self):
    """
    Statistics for vector variables using magnitude.
    For vector v, compute stats on |v| = sqrt(v·v)
    """
    # Create temporary scalar variable for magnitude
    magnitude_var = self._create_magnitude_variable()
    
    # Compute magnitude: |v|² = v·v, then |v| = sqrt(|v|²)
    with uw.synchronised_array_update():
        magnitude_squared = 0.0
        for i in range(self.num_components):
            magnitude_squared += self.array[:, :, i] ** 2
        magnitude_var.array[:, :, 0] = np.sqrt(magnitude_squared)
    
    # Get scalar stats on magnitude
    mag_stats = magnitude_var.stats()
    
    # Add vector-specific stats
    mag_stats.update({
        'type': 'vector',
        'components': self.num_components,
        'magnitude_mean': mag_stats['mean'],
        'magnitude_max': mag_stats['max'],
        'magnitude_min': mag_stats['min']
    })
    
    # Cleanup temporary variable
    self._cleanup_temp_variable(magnitude_var)
    
    return mag_stats
```

#### Tensor Stats (stress, strain, etc.)
```python
def tensor_stats(self):
    """
    Statistics for tensor variables using multiple scalar measures.
    """
    # For tensors, provide multiple meaningful measures:
    stats = {'type': 'tensor', 'components': self.num_components}
    
    # 1. Frobenius norm: sqrt(sum of all elements squared)
    frobenius_var = self._create_scalar_temp()
    with uw.synchronised_array_update():
        sum_squares = 0.0
        for i in range(self.num_components):
            sum_squares += self.array[:, :, i] ** 2
        frobenius_var.array[:, :, 0] = np.sqrt(sum_squares)
    
    frobenius_stats = frobenius_var.stats()
    stats.update({
        'frobenius_mean': frobenius_stats['mean'],
        'frobenius_max': frobenius_stats['max'],
        'frobenius_min': frobenius_stats['min']
    })
    
    # 2. For symmetric tensors (stress/strain): eigenvalue-based measures
    if self._is_symmetric_tensor():
        # Could add trace, determinant, max eigenvalue stats
        pass
    
    self._cleanup_temp_variable(frobenius_var)
    return stats
```

### 2. Swarm Variable Statistics

#### Challenge: Distributed Particle Data
Swarm variables have particles distributed across processors with no global PETSc vector.

#### Solution 1: Global Evaluate + Broadcast
```python
def swarm_stats(self):
    """
    Statistics for swarm variables using global_evaluate approach.
    """
    # Use global_evaluate to gather all data to rank 0
    if uw.mpi.rank == 0:
        # Evaluate on a single point to trigger global gathering
        all_data = uw.function.evaluate(self.sym, [0.0] * self.swarm.mesh.dim)
        
        # Compute stats on gathered data
        stats = {
            'size': len(all_data),
            'mean': np.mean(all_data),
            'min': np.min(all_data),
            'max': np.max(all_data),
            'sum': np.sum(all_data),
            'std': np.std(all_data)
        }
    else:
        stats = None
    
    # Broadcast results to all ranks
    stats = uw.mpi.comm.bcast(stats, root=0)
    return stats
```

#### Solution 2: Proxy Variable Approach  
```python
def swarm_stats_via_proxy(self):
    """
    Statistics for swarm variables using mesh proxy.
    """
    if self.proxy_degree > 0:
        # Use the mesh proxy variable for stats
        self._update()  # Ensure proxy is current
        return self._proxy_mesh_var.stats()
    else:
        # Fall back to global evaluate approach
        return self.swarm_stats_global()
```

### 3. Implementation Architecture

#### Base Stats Method (Enhanced)
```python
@uw.collective_operation  
def stats(self):
    """
    Universal stats method for all variable types.
    
    Returns
    -------
    dict
        Statistical measures appropriate for the variable type:
        - Scalar: min, max, mean, sum, norm2, rms
        - Vector: magnitude-based stats + component info
        - Tensor: Frobenius norm + tensor-specific measures
    """
    if self.num_components == 1:
        return self._scalar_stats()
    elif self.num_components == self.mesh.dim:
        return self._vector_stats()  
    else:
        return self._tensor_stats()
```

#### Helper Methods
```python
def _create_magnitude_variable(self):
    """Create temporary scalar variable for magnitude calculations."""
    return uw.discretisation.MeshVariable(
        f"_temp_magnitude_{id(self)}", 
        self.mesh, 
        1, 
        degree=self.degree
    )

def _cleanup_temp_variable(self, temp_var):
    """Safely cleanup temporary variables."""
    # Remove from mesh variable registry
    if temp_var.name in self.mesh.vars:
        del self.mesh.vars[temp_var.name]
    
def _is_symmetric_tensor(self):
    """Check if tensor variable represents symmetric tensor."""
    # Could check metadata or naming conventions
    return self.num_components == 6  # 3D symmetric tensor
```

### 4. Swarm Variable Integration

#### Add Stats Method to SwarmVariable
```python
class SwarmVariable:
    def stats(self):
        """Statistics for swarm variables."""
        if self.proxy_degree > 0:
            # Use proxy mesh variable approach
            self._update()  # Ensure proxy is current
            return self._proxy_mesh_var.stats()
        else:
            # Use global gathering approach
            return self._global_stats()
            
    def _global_stats(self):
        """Global gather and broadcast approach."""
        # Implementation as shown above
        pass
```

### 5. Memory Management Strategy

#### Temporary Variable Lifecycle
1. **Create**: Temporary variables for derived quantities (magnitude, Frobenius norm)
2. **Compute**: Perform calculations using existing UW3 operations
3. **Extract**: Get scalar stats from temporary variable
4. **Cleanup**: Remove temporary variables from mesh registry

#### Memory Efficiency
- **Reuse patterns**: Cache temporary variables for repeated calls
- **Lazy allocation**: Only create temporaries when needed
- **Smart cleanup**: Use weak references to avoid memory leaks

### 6. API Design Principles

#### Consistent Return Format
```python
# All stats() methods return dict with common keys:
{
    'type': 'scalar'|'vector'|'tensor',
    'components': int,
    'size': int,
    'min': float,
    'max': float, 
    'mean': float,
    'sum': float,
    # Type-specific additional keys
}
```

#### Parallel Safety
- **All stats methods**: Marked with `@uw.collective_operation`
- **MPI coordination**: Proper reduction operations for distributed data
- **Consistent behavior**: Same results regardless of process count

### 7. Implementation Phases

#### Phase 1: Enhanced Scalar Stats (✅ Complete)
- [x] Return dictionary instead of tuple
- [x] Update documentation
- [x] Maintain backward compatibility

#### Phase 2: Vector Stats  
- [ ] Implement magnitude-based stats for vectors
- [ ] Add temporary variable management
- [ ] Test with velocity, displacement fields

#### Phase 3: Tensor Stats
- [ ] Implement Frobenius norm approach
- [ ] Add symmetric tensor detection
- [ ] Test with stress, strain tensors

#### Phase 4: Swarm Stats
- [ ] Implement proxy variable approach
- [ ] Add global evaluate fallback
- [ ] Test with material properties

#### Phase 5: Performance Optimization
- [ ] Cache temporary variables
- [ ] Optimize memory allocation
- [ ] Benchmark performance impact

### 8. Testing Strategy

#### Unit Tests
```python
def test_vector_stats():
    mesh = uw.meshing.StructuredQuadBox()
    velocity = uw.discretisation.MeshVariable("v", mesh, 2)
    
    # Set known velocity field
    with uw.synchronised_array_update():
        velocity.array[:, 0, 0] = 1.0  # u = 1
        velocity.array[:, 0, 1] = 0.0  # v = 0
        
    stats = velocity.stats()
    assert stats['type'] == 'vector'
    assert stats['magnitude_mean'] == 1.0
    assert stats['magnitude_max'] == 1.0
```

#### Integration Tests
- Test with realistic geophysical problems
- Verify parallel consistency across different MPI configurations
- Check memory usage patterns

### 9. Documentation Updates

#### User Guide
- Examples of vector/tensor stats usage
- Interpretation of different statistical measures
- Performance considerations

#### API Reference
- Complete documentation of return dictionary structure
- Type-specific statistical measures explanation
- Parallel safety guarantees

## Benefits

### User Experience
- **Consistent API**: Same `stats()` method works for all variable types
- **Rich information**: Appropriate statistics for each data type
- **Parallel safe**: Works correctly in all MPI configurations

### Development Benefits  
- **Extensible design**: Easy to add new statistical measures
- **Memory efficient**: Careful temporary variable management
- **Well tested**: Comprehensive test coverage for all variable types

This plan provides a systematic approach to expanding statistics capabilities while maintaining consistency with the existing UW3 architecture.