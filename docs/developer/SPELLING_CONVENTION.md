# Spelling Convention for Underworld3

**Date**: 2025-10-14
**Status**: Active Convention

## Preferred Spelling: UK/Australian English

Unless demonstrating that alternative spellings are available, use UK/Australian English spelling throughout the codebase.

## Common Examples

### Preferred (UK/Australian)
- **metre** (unit of length)
- **kilometre**
- **litre**
- **colour**
- **centre**
- **fibre**
- **behaviour**
- **neighbour**
- **optimise**
- **minimise**
- **maximise**
- **analyse**
- **recognise**
- **modelling** (double-l)

### Avoid (US spelling) unless showing alternatives
- meter
- kilometer
- liter
- color
- center
- fiber
- behavior
- neighbor
- optimize
- minimize
- maximize
- analyze
- recognize
- modeling (single-l)

## Context for Alternatives

When showing that alternative spellings are accepted (e.g., in unit systems), document both:

```python
# Example: Pint accepts both spellings
units="metre"      # Preferred
units="meter"      # Also accepted, documented as alternative
```

## Application

This applies to:
- Code comments
- Documentation (`.md`, `.rst`, `.qmd`)
- Docstrings
- Variable names (where appropriate)
- Example notebooks
- Test descriptions
- Error messages
- User-facing strings

## Notes

- Existing code with US spelling doesn't need bulk changes
- Apply this convention opportunistically when:
  - Writing new code
  - Updating existing documentation
  - Fixing bugs in relevant sections
  - User specifically requests spelling corrections
