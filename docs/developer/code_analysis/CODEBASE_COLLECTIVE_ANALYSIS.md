# Using Codebase to Identify Collective vs Non-Collective Operations

## Key Insight

**Functions called inside `if uw.mpi.rank == 0:` are NOT collective** (or they shouldn't be, or the code has a bug).

This gives us a powerful way to automatically classify operations!

## Strategy: Analyze Existing Patterns

### What Rank Conditionals Tell Us

```python
# Pattern 1: Operations inside rank conditionals are LOCAL/SAFE
if uw.mpi.rank == 0:
    print(f"Stats: {var.stats()}")  # If this works, stats() is NOT collective!
    value = var.data.max()          # data.max() is LOCAL
    plt.plot(x, y)                  # plt.plot() is LOCAL/SERIAL
```

**Conclusion**: If code runs successfully in parallel and has this pattern, the operation is safe/local.

### What's OUTSIDE Rank Conditionals

```python
# Pattern 2: Operations outside conditionals are likely COLLECTIVE
mesh.save("checkpoint.h5")  # Called by all ranks - probably collective
stokes.solve()              # Called by all ranks - definitely collective
var.stats()                 # If called by all - collective
```

## Automated Analysis

### Scan 1: Functions in Rank Conditionals (NOT Collective)

```python
import ast
import os

def find_non_collective_operations():
    """Find operations called inside rank conditionals - these are NOT collective."""
    
    non_collective = set()
    potential_bugs = set()
    
    for root, dirs, files in os.walk('src/underworld3'):
        for file in files:
            if not file.endswith('.py'):
                continue
            
            filepath = os.path.join(root, file)
            with open(filepath) as f:
                try:
                    tree = ast.parse(f.read())
                except:
                    continue
            
            for node in ast.walk(tree):
                # Find rank conditionals
                if isinstance(node, ast.If):
                    # Check if this is "if uw.mpi.rank == 0:"
                    if _is_rank_conditional(node.test):
                        # Extract all function calls in this block
                        calls = _extract_calls(node.body)
                        
                        for call in calls:
                            non_collective.add(call)
                            
                            # Check if this is a known collective - BUG!
                            if call in KNOWN_COLLECTIVE:
                                potential_bugs.add((filepath, call))
    
    return non_collective, potential_bugs

def _is_rank_conditional(test):
    """Check if test is 'uw.mpi.rank == 0' or similar."""
    if not isinstance(test, ast.Compare):
        return False
    
    # Check for uw.mpi.rank
    if isinstance(test.left, ast.Attribute):
        if (isinstance(test.left.value, ast.Attribute) and
            test.left.value.attr == 'mpi' and
            test.left.attr == 'rank'):
            return True
    
    return False

def _extract_calls(body):
    """Extract all function/method calls from AST nodes."""
    calls = []
    
    for node in ast.walk(ast.Module(body=body)):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                # Method call: obj.method()
                calls.append(node.func.attr)
            elif isinstance(node.func, ast.Name):
                # Function call: func()
                calls.append(node.func.id)
    
    return calls

# Run analysis
non_collective, bugs = find_non_collective_operations()

print("Operations found in rank conditionals (NOT collective):")
for op in sorted(non_collective):
    print(f"  - {op}")

if bugs:
    print("\n⚠️  POTENTIAL BUGS - collective ops in rank conditionals:")
    for filepath, op in bugs:
        print(f"  - {filepath}: {op}()")
```

### Scan 2: Cross-Reference with Known Collective

```python
# Known collective operations (from decorators)
KNOWN_COLLECTIVE = {
    'stats', 'solve', 'norm', 'dot', 
    'save', 'load', 'migrate',
    'rbf_interpolate', 'gather', 'barrier'
}

# Operations found in rank conditionals (NOT collective)
FOUND_IN_CONDITIONALS = {
    'print', 'plot', 'savefig', 'write',
    'max', 'min', 'shape', 'size',
    'figure', 'show', 'close'
}

# Cross-reference
def validate_operations():
    """Check for conflicts between known collective and conditional usage."""
    
    conflicts = KNOWN_COLLECTIVE & FOUND_IN_CONDITIONALS
    
    if conflicts:
        print("⚠️  CONFLICTS - These appear in both categories:")
        for op in conflicts:
            print(f"  - {op}: Marked collective but found in rank conditional")
            print(f"    Either: (1) Not actually collective, or (2) BUG in code")
    
    # Safe to mark as NON-collective
    safe_non_collective = FOUND_IN_CONDITIONALS - KNOWN_COLLECTIVE
    print("\n✅ Safe to mark as NON-collective:")
    for op in safe_non_collective:
        print(f"  - {op}")
```

### Scan 3: Functions Called By All Ranks

```python
def find_collective_operations():
    """Find operations called outside any rank conditional."""
    
    all_calls = set()
    conditional_calls = set()
    
    for filepath in find_python_files('src/underworld3'):
        tree = parse_file(filepath)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                call_name = get_call_name(node)
                all_calls.add(call_name)
                
                # Check if inside rank conditional
                if _is_inside_rank_conditional(node, tree):
                    conditional_calls.add(call_name)
    
    # Operations called OUTSIDE conditionals - likely collective
    likely_collective = all_calls - conditional_calls
    
    return likely_collective

def _is_inside_rank_conditional(node, tree):
    """Check if node is inside a rank conditional block."""
    # Walk up the AST to find parent If nodes
    for parent in ast.walk(tree):
        if isinstance(parent, ast.If):
            if _is_rank_conditional(parent.test):
                # Check if node is in this If block
                if node in ast.walk(ast.Module(body=parent.body)):
                    return True
    return False
```

## Real Example from UW3 Codebase

### Actual Pattern Analysis

```python
# Found in src/underworld3/swarm.py:972
if uw.mpi.rank == 0:
    print("No proxy mesh variable that can be saved", flush=True)

# Conclusion: print() is NOT collective ✓

# Found in examples/convection/.../Ex_Convection_1_SLCN_Cartesian.py:259
if uw.mpi.rank == 0:
    print("Timestep {}, dt {}".format(step, delta_t))

# Conclusion: print(), format() are NOT collective ✓

# Found in src/underworld3/discretisation/discretisation_mesh.py
if uw.mpi.rank == 0:
    num_cells = self.dm.getHeightStratum(0)  # DM operation!

# ⚠️  Question: Is getHeightStratum() collective?
# If this code works, then NO - it's a local query
```

## Building Operation Database

### Automatic Classification

```python
class OperationClassifier:
    def __init__(self):
        self.definitely_local = set()      # Found ONLY in conditionals
        self.definitely_collective = set()  # Marked with @collective_operation
        self.likely_local = set()          # Found mostly in conditionals
        self.likely_collective = set()     # Found mostly outside conditionals
        self.uncertain = set()             # Mixed usage
    
    def analyze_codebase(self):
        """Analyze all Python files to classify operations."""
        
        # Scan for decorated operations
        for func in find_decorated_functions('@collective_operation'):
            self.definitely_collective.add(func)
        
        # Scan for rank conditional usage
        conditional_ops = find_operations_in_rank_conditionals()
        global_ops = find_operations_outside_conditionals()
        
        # Classify
        for op in conditional_ops:
            if op not in global_ops:
                self.definitely_local.add(op)  # ONLY in conditionals
            elif global_ops[op] < conditional_ops[op]:
                self.likely_local.add(op)      # More in conditionals
            else:
                self.uncertain.add(op)         # Mixed - needs investigation
        
        for op in global_ops:
            if op not in conditional_ops:
                self.likely_collective.add(op)  # Never in conditionals
    
    def generate_markers(self):
        """Generate @collective_operation decorators."""
        
        print("# Add these decorators:")
        for op in self.definitely_collective | self.likely_collective:
            print(f"@collective_operation")
            print(f"def {op}(self, ...):")
            print(f"    ...")
            print()
        
        print("# These are safe (local operations):")
        for op in self.definitely_local:
            print(f"# {op} - found only in rank conditionals")

# Run
classifier = OperationClassifier()
classifier.analyze_codebase()
classifier.generate_markers()
```

## Validation Strategy

### Step 1: Scan Codebase
```bash
python analyze_collective_ops.py > operation_analysis.txt
```

### Step 2: Identify Conflicts
```python
# Operations in rank conditionals but marked collective - BUGS!
if uw.mpi.rank == 0:
    stats = var.stats()  # ← BUG if stats() is collective!
```

### Step 3: Build Whitelist
```python
# Safe operations (found in rank conditionals, never hang)
SAFE_LOCAL_OPS = {
    'print', 'format', 'write',  # I/O
    'plot', 'savefig', 'show',   # Matplotlib
    'max', 'min', 'sum',         # NumPy on local arrays
    'shape', 'size', 'dtype',    # Array properties
}
```

### Step 4: Build Collective List
```python
# Collective operations (never in rank conditionals, or cause bugs)
COLLECTIVE_OPS = {
    'stats', 'solve', 'migrate',     # UW3 operations
    'norm', 'dot', 'barrier',        # PETSc/MPI
    'allreduce', 'gather', 'bcast',  # MPI collectives
}
```

## Practical Tools

### Tool 1: Find Potential Bugs
```python
def find_collective_in_conditionals():
    """Find known collective ops in rank conditionals - these are BUGS."""
    
    bugs = []
    
    for filepath in find_all_python_files():
        tree = parse_file(filepath)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and _is_rank_conditional(node.test):
                # Scan this conditional block
                for call in extract_calls(node.body):
                    if call in KNOWN_COLLECTIVE:
                        bugs.append({
                            'file': filepath,
                            'line': node.lineno,
                            'operation': call,
                            'context': ast.get_source_segment(source, node)
                        })
    
    return bugs

# Report bugs
bugs = find_collective_in_conditionals()
if bugs:
    print("🐛 Found collective operations in rank conditionals (will hang!):")
    for bug in bugs:
        print(f"\n{bug['file']}:{bug['line']}")
        print(f"  Operation: {bug['operation']}()")
        print(f"  Context: {bug['context'][:100]}...")
```

### Tool 2: Auto-Generate Decorators
```python
def suggest_decorators():
    """Suggest which functions should be marked @collective_operation."""
    
    # Functions never found in rank conditionals
    always_global = find_operations_never_in_conditionals()
    
    # Functions that look like collective operations
    collective_patterns = {
        'solve', 'stats', 'gather', 'reduce', 'barrier',
        'migrate', 'interpolate', 'save', 'load'
    }
    
    suggestions = []
    
    for op in always_global:
        # Check if name suggests collective operation
        if any(pattern in op.lower() for pattern in collective_patterns):
            suggestions.append(op)
    
    print("💡 Suggest marking these as @collective_operation:")
    for op in suggestions:
        print(f"  @collective_operation")
        print(f"  def {op}(self, ...): ...")
```

## Summary

**Using rank conditionals to identify collective operations:**

### What We Learn

1. **Operations IN rank conditionals** → NOT collective (or buggy code)
2. **Operations OUTSIDE rank conditionals** → Possibly collective
3. **Operations in BOTH** → Need investigation (might be context-dependent)

### Automatic Classification

✅ **Definitely NOT collective**: Only found in `if uw.mpi.rank == 0:`
✅ **Probably collective**: Never found in rank conditionals
⚠️ **Uncertain**: Mixed usage - needs manual review
🐛 **Bugs**: Known collective in rank conditional

### Benefits

1. **Auto-discover** safe operations without manual marking
2. **Find bugs** where collective ops are in rank conditionals
3. **Build database** of operation classifications
4. **Generate decorators** automatically for likely collective ops

This is a powerful complement to the decorator approach - the codebase itself tells us what's collective!