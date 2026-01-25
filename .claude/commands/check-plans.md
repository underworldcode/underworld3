---
description: Check the central planning file for active tasks and bugs
---

## Planning File Location

Read the planning file at:
- **macOS**: `~/Library/CloudStorage/Box-Box/Planning/underworld.md`
- **Linux**: `~/Box/Planning/underworld.md`

---

## What to Report

After reading the planning file, briefly report:

1. **Active items** relevant to current work (from `## Active` section)
2. **Open bugs** that might affect current work (from `## Bugs` section)
3. **Periodic reviews** that are due (from `## Periodic Reviews` section)

Don't summarize the entire file - focus on actionable items.

---

## When Tasks Are Completed

Add an annotation directly below the completed item:

```markdown
<!-- PROJECT RESPONSE (YYYY-MM-DD underworld3):
Brief summary of what was done.
Reference to any files created/modified.
-->
```

---

## Adding New Items

If you discover bugs or identify new tasks:
- Add to the planning file under appropriate section (Bugs, Active, Nice to Have)
- Use project tag: `<!-- project:underworld3/subsystem -->`
- Don't create local TODO files

---

## What NOT to Do

- Don't rewrite strategic paragraphs
- Don't move items between sections (Active → Done)
- Don't restructure the document
