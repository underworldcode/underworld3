# Renaming Summary: CuckooBot → HelpfulBatBot

## What Changed

All instances of "CuckooBot" and "Cuckoo" have been renamed to "HelpfulBatBot" and "HelpfulBat".

## Directory Locations

### 1. Primary Location (Version Controlled in Git)
**Path:** `/Users/lmoresi/+Underworld/underworld-pixi-2/underworld3/HelpfulBatBot/`

This is the **official** version that should be committed to the underworld3 git repository.

### 2. Testing Location (Temporary, Not Git)
**Path:** `/Users/lmoresi/+Underworld/underworld3-helpfulbat-bot/`

This is a **temporary working directory** for testing. Changes here should be copied back to the primary location.

## Files Renamed

| Old Name | New Name |
|----------|----------|
| `Cuckoo_app.py` | `HelpfulBat_app.py` |
| `Cuckoo_README.md` | `HelpfulBat_README.md` |
| `Cuckoo_refreshment.py` | `HelpfulBat_refreshment.py` |
| `Cuckoo_workflow.yml` | `HelpfulBat_workflow.yml` |
| `Cuckoo_policy.typ` | `HelpfulBat_policy.typ` |
| `CuckooBot/` directory | `HelpfulBatBot/` directory |

## Code References Updated

All Python, shell, and markdown files have been updated:
- `CuckooBot` → `HelpfulBatBot`
- `Cuckoo_app` → `HelpfulBat_app`
- `cuckoobot` → `helpfulbatbot`
- Import statements updated
- Documentation updated
- Configuration files updated

## How to Use

### From Primary Location (Recommended)
```bash
cd /Users/lmoresi/+Underworld/underworld-pixi-2/underworld3/HelpfulBatBot
./demo.sh
```

### From Testing Location
```bash
cd /Users/lmoresi/+Underworld/underworld3-helpfulbat-bot
./demo.sh
```

## Synchronization

The two locations are **mirrors** of each other. To keep them in sync:

**Copy from testing → primary:**
```bash
rsync -av --exclude='.env' \
  /Users/lmoresi/+Underworld/underworld3-helpfulbat-bot/ \
  /Users/lmoresi/+Underworld/underworld-pixi-2/underworld3/HelpfulBatBot/
```

**Copy from primary → testing:**
```bash
rsync -av --exclude='.env' \
  /Users/lmoresi/+Underworld/underworld-pixi-2/underworld3/HelpfulBatBot/ \
  /Users/lmoresi/+Underworld/underworld3-helpfulbat-bot/
```

## Git Workflow

The primary location (`underworld3/HelpfulBatBot/`) should be added to git:

```bash
cd /Users/lmoresi/+Underworld/underworld-pixi-2/underworld3
git status HelpfulBatBot/
git add HelpfulBatBot/
git commit -m "Rename CuckooBot → HelpfulBatBot and add user-focused indexing"
```

**Note:** Make sure `.env` is in `.gitignore` to avoid committing API keys!

## Quick Start (After Renaming)

```bash
cd /Users/lmoresi/+Underworld/underworld-pixi-2/underworld3/HelpfulBatBot
./start_bot.sh
python3 ask.py "How do I create a mesh?"
```

## What Wasn't Changed

- The bot functionality remains exactly the same
- Configuration in `.env` is unchanged
- All the smart path-based filtering is still active
- Index still focuses on user-facing content (86 files)

---

**Date:** November 18, 2025
**Reason:** Better name that reflects the bot's helpful nature for UW3 users
