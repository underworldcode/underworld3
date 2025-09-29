# Parallel-Safe Visualization Patterns for Underworld3

## Summary

Based on testing, `global_evaluate` **can handle asymmetric calls** where some ranks provide empty arrays, making parallel visualization possible with careful patterns.

## Key Findings

### ✅ What Works
- **Asymmetric calls**: Rank 0 requests data points, other ranks call with empty arrays
- **Data gathering**: All processes participate in collective operations correctly
- **Selective visualization**: Only designated ranks create plots

### ⚠️ Known Limitations  
- **All-empty edge case**: `global_evaluate` fails when ALL ranks provide empty arrays
- **Memory constraints**: Full-resolution global data gathering is impractical for large problems
- **Distributed mesh**: Mesh topology is not available globally

## Recommended Patterns

### Pattern 1: Simple Diagnostic Plots ⭐
**Best for tutorials and quick diagnostics**

```python
def parallel_line_plot(field, sample_points, title="Field Profile"):
    \"\"\"
    Create a 1D line plot in parallel notebooks.
    Only rank 0 creates the plot, but all ranks participate in data gathering.
    \"\"\"
    if uw.mpi.rank == 0:
        # Rank 0 requests data at sample points
        data = uw.function.global_evaluate(field, sample_points)
        
        # Create the plot
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.plot(sample_points[:, 0], data.flatten())
        plt.title(title)
        plt.xlabel("Position")
        plt.ylabel("Field Value") 
        plt.show()
    else:
        # Other ranks participate with empty requests
        empty_points = np.array([]).reshape(0, field.mesh.dim)
        uw.function.global_evaluate(field, empty_points)

# Usage example:
sample_line = np.column_stack([np.linspace(0, 1, 50), np.full(50, 0.5)])
parallel_line_plot(temperature, sample_line, "Temperature Profile")
```

### Pattern 2: Multi-Field Scatter Plot

```python  
def parallel_scatter_plot(field_x, field_y, sample_points, labels=("X", "Y")):
    \"\"\"Create scatter plot of two fields at sample points\"\"\"
    if uw.mpi.rank == 0:
        # Gather both fields
        data_x = uw.function.global_evaluate(field_x, sample_points)
        data_y = uw.function.global_evaluate(field_y, sample_points)
        
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.scatter(data_x.flatten(), data_y.flatten(), alpha=0.6)
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        plt.show()
    else:
        # Other ranks participate
        empty_points = np.array([]).reshape(0, field_x.mesh.dim)
        uw.function.global_evaluate(field_x, empty_points)  
        uw.function.global_evaluate(field_y, empty_points)

# Usage:
sample_points = np.random.random((100, 2))  # Random sample points
parallel_scatter_plot(velocity[0], pressure, sample_points, ("Vx", "Pressure"))
```

### Pattern 3: Helper Function for Common Cases

```python
def parallel_evaluate_and_plot(plot_func, fields_and_points, viz_rank=0):
    \"\"\"
    Generic helper for parallel plotting with multiple fields.
    
    Args:
        plot_func: Function to create the plot (called only on viz_rank)
        fields_and_points: List of (field, points, name) tuples
        viz_rank: Which rank creates the visualization
    \"\"\"
    if uw.mpi.rank == viz_rank:
        # Gather all field data
        plot_data = {}
        for field, points, name in fields_and_points:
            plot_data[name] = uw.function.global_evaluate(field, points)
            plot_data[f"{name}_points"] = points
        
        # Create visualization
        plot_func(**plot_data)
    else:
        # Other ranks participate in data gathering
        for field, points, name in fields_and_points:
            empty_points = np.array([]).reshape(0, field.mesh.dim)
            uw.function.global_evaluate(field, empty_points)

# Usage:
def create_velocity_profile(velocity, velocity_points, temperature, temperature_points):
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(velocity_points[:, 1], velocity.flatten())
    ax1.set_title("Velocity vs Height")
    
    ax2.plot(temperature_points[:, 0], temperature.flatten()) 
    ax2.set_title("Temperature vs X")
    
    plt.show()

y_line = np.column_stack([np.full(50, 0.5), np.linspace(0, 1, 50)])
x_line = np.column_stack([np.linspace(0, 1, 50), np.full(50, 0.5)])

parallel_evaluate_and_plot(
    create_velocity_profile,
    [(velocity[1], y_line, "velocity"),
     (temperature, x_line, "temperature")]
)
```

## Technical Details

### How global_evaluate Works
1. **Swarm creation**: Creates temporary swarm with evaluation points
2. **Migration**: Particles migrate to owning processes based on spatial location
3. **Local evaluation**: Each process evaluates expressions at local particles  
4. **Reverse migration**: Particles return to requesting processes
5. **Result assembly**: Original requesting process assembles final result

### Why Asymmetric Calls Work
- Empty arrays result in zero particles added to swarm
- Migration and evaluation steps handle zero particles correctly
- Result assembly works when some processes have no returning particles

### Memory and Performance Considerations
- **Sample strategically**: Don't request excessive points (thousands, not millions)
- **Use for diagnostics**: Not suitable for full-field visualization
- **Consider alternatives**: Checkpoint-based visualization for production

## Integration with Tutorial Notebooks

### Before (Problematic):
```python
# This doesn't work in parallel - only rank 0 has visualization libraries
if uw.mpi.size == 1:
    import matplotlib.pyplot as plt
    # plotting code...
else:
    print("Visualization skipped in parallel")
```

### After (Parallel-Safe):
```python
# This works in parallel - all ranks participate in data gathering
sample_points = np.column_stack([np.linspace(0, 1, 50), np.full(50, 0.5)])
parallel_line_plot(temperature, sample_points, "Temperature Profile")
```

## Future Enhancements

1. **Fix all-empty edge case**: Patch `global_evaluate` to handle when all ranks provide empty arrays
2. **Standard library**: Add common plotting patterns to `uw.visualization.parallel`
3. **Automatic point generation**: Smart sampling strategies for different field types
4. **Integration with selective_ranks**: Combine patterns with existing parallel safety tools

## Documentation Notes

- **Checkpointing system**: Uses `mesh.write_timestep()` with PETSc HDF5 files
- **Flexible restart**: Can read back to different meshes or parallel decompositions  
- **KDTree mapping**: Uses KDTree machinery to map values to mesh nodes
- **Swarm limitations**: No down-sampling strategy but mesh-resolution independent