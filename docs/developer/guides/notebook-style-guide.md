# Underworld3 Notebook Style Guide

## Overall Approach

Notebooks should be **simple, informal, and educational** - not overly congratulatory or verbose.

## Key Principles

1. **Jupyter-Native Display**
   - Use implicit display (last line of cell) to show results
   - Prefer `object` over `print(object)` when showing a single result
   - Use markdown cells for explanations and context

2. **Minimal Print Statements**
   - Avoid excessive `uw.pprint()` - notebooks are single-process
   - Use `print()` sparingly, mainly for showing multiple related values
   - Let Jupyter's display system do the work

3. **Tone**
   - Simple and direct
   - Informal but professional
   - Not overly congratulatory (avoid "Amazing!", "Fantastic!", etc.)
   - Focus on teaching, not cheerleading

4. **Code Style**
   - Show the "uw3 way" of doing things
   - Use implicit display for single results
   - Group related output with formatted strings when needed

## Examples

### Good Style

```python
# Create a quantity
temperature = 1500 * uw.units.K

# Display it (implicit)
temperature
```

```python
# Show multiple related values
print(f"Temperature: {temperature}")
print(f"Pressure: {pressure}")
print(f"Density: {density}")
```

### Avoid

```python
# Too verbose
print("Creating temperature...")
temperature = 1500 * uw.units.K
print(f"Temperature created: {temperature}")
print("Success!")  # Unnecessary
```

## Notebook Structure

1. **Title and Introduction** (markdown)
   - What the notebook teaches
   - Brief list of topics

2. **Import Cell**
   ```python
   import nest_asyncio
   nest_asyncio.apply()

   import underworld3 as uw
   import numpy as np
   ```

3. **Concept Sections**
   - Markdown header (##) for each major concept
   - Brief explanation in markdown
   - Code cells demonstrating the concept
   - Minimal output cells (let Jupyter display)

4. **Summary** (markdown)
   - Key takeaways in bullet points
   - When to use what

5. **Try It Yourself** (markdown)
   - Optional exercises in code fences
   - Encourage exploration

## Display Patterns

### Showing a Single Value
```python
# Implicit display
temperature.units
```

### Showing Multiple Values
```python
print(f"Temperature: {T.min():.1f} to {T.max():.1f}")
print(f"Pressure: {P.min():.1f} to {P.max():.1f}")
```

### Inspecting Objects
```python
# Let Jupyter display the repr
mesh.units
```

```python
# Or use .view() for detailed info
mesh.view()
```

## What to Avoid

- ❌ Excessive congratulation ("Great job!", "Excellent!")
- ❌ Over-explanation of obvious things
- ❌ Too many print statements for simple output
- ❌ Long chains of verification cells
- ❌ Debug/exploration cells left in production notebooks

## Examples from Notebook 12 and 13

These notebooks demonstrate the preferred style:
- Simple introduction
- Concept-focused sections
- Implicit display where appropriate
- Minimal but useful output
- Practical exercises at the end
