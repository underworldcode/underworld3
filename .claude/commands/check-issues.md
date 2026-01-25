---
description: Check open GitHub issues for underworld3
---

## Check Open Issues

Run the following commands to check open issues:

### Documentation Issues
```bash
gh issue list --repo underworldcode/underworld3 --label documentation --state open
```

### All Open Issues
```bash
gh issue list --repo underworldcode/underworld3 --state open --limit 20
```

### Bug Reports
```bash
gh issue list --repo underworldcode/underworld3 --label bug --state open
```

---

## Triage Actions

For each issue, determine:

1. **Quick fix**: Can be addressed in current session
   - Typos, broken links, minor documentation updates
   - Apply fix, commit with "Fixes #N" in message

2. **Add to planning**: Needs more work or discussion
   - Add to Active or Bugs section in planning file
   - Include issue number and brief description

3. **Needs clarification**: Insufficient information
   - Comment on issue asking for details
   - Label with `needs-info` if available

---

## View Specific Issue

To see full details of an issue:
```bash
gh issue view <number> --repo underworldcode/underworld3
```

---

## After Addressing Issues

- Issues with "Fixes #N" in commit message auto-close on merge
- For issues addressed differently, close manually:
  ```bash
  gh issue close <number> --repo underworldcode/underworld3 --comment "Fixed in <commit/PR>"
  ```
