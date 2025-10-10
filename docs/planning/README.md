# Planning Documents

This directory contains planning documents, design decisions, and implementation plans for Underworld3 development.

## Directory Structure

### `units-system/`
Planning and design documents for the universal units system:
- **PINT_NATIVE_IMPLEMENTATION_PLAN.md** - Complete plan for Pint-native approach
- **UNIVERSAL_UNITS_IMPLEMENTATION_PLAN.md** - Universal units philosophy and design
- **SYMPY_UNITS_RESPONSE.md** - Analysis of SymPy units integration (decision not to pursue)
- **VARIABLE_UNITS_IMPLEMENTATION_SUMMARY.md** - Variable-level units implementation
- **pint_prototype_summary.md** - Early prototype summary
- **UNITS_DEMO.md** - Demonstration of units capabilities

### `coordinate-scaling/`
Planning documents for coordinate scaling and transformation:
- **COORDINATE_SCALING_PROBLEMS.md** - Critical interface problems identified
- **COORDINATE_SCALING_SUMMARY.md** - Summary of coordinate scaling approach

### `development/`
General development principles and testing documentation:
- **CODING_PRINCIPLES.md** - Core coding principles for UW3 development
- **EVALUATE_FUNCTION_TEST_COVERAGE.md** - Test coverage analysis for evaluate functions

### `migration_scripts/`
Utility scripts used during major refactoring and migration:
- **update_docs_api.py** - Update documentation for new API patterns
- **update_pprint_calls.py** - Migrate to new pprint API
- **update_unwrap_calls.py** - Migrate to new unwrap API

## Purpose

These documents serve as:
- **Historical record** - Decisions made and why
- **Design reference** - Implementation approaches and architecture
- **Problem documentation** - Issues encountered and solutions
- **Developer guide** - Understanding the reasoning behind the code

## Related Documentation

- **User documentation**: See `docs/user/` and `docs/beginner/`
- **Developer documentation**: See `docs/developer/`
- **API documentation**: Auto-generated from source code
- **Examples and tutorials**: See `docs/examples/`

## Contributing

When adding new planning documents:
1. Place in the appropriate subdirectory
2. Use descriptive names (e.g., `FEATURE_NAME_IMPLEMENTATION_PLAN.md`)
3. Include date and context in the document
4. Update this README with a brief description
5. Link to related documentation where appropriate
