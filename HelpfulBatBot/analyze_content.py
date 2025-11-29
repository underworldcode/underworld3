#!/usr/bin/env python3
"""Analyze UW3 content for user-facing vs internal"""

import os
from pathlib import Path

repo = Path("/Users/lmoresi/+Underworld/underworld-pixi-2/underworld3")

print("📊 Underworld3 Content Analysis")
print("=" * 70)

# User-facing content
print("\n🎓 USER-FACING CONTENT:")
print("-" * 70)

# Tutorials
tutorials = list((repo / "docs/beginner/tutorials").glob("*.ipynb"))
print(f"\n📘 Tutorials: {len(tutorials)} notebooks")
for nb in sorted(tutorials)[:5]:
    print(f"  • {nb.name}")
if len(tutorials) > 5:
    print(f"  ... and {len(tutorials)-5} more")

# Examples
examples_nb = list((repo / "examples").glob("*.ipynb")) if (repo / "examples").exists() else []
examples_py = list((repo / "examples").glob("*.py")) if (repo / "examples").exists() else []
print(f"\n📗 Examples: {len(examples_nb)} notebooks, {len(examples_py)} Python scripts")
for ex in sorted(examples_nb)[:5]:
    print(f"  • {ex.name}")

# A/B grade tests (0000-0699)
tests_simple = list((repo / "tests").glob("test_0[0-6]*.py"))
print(f"\n✅ A/B Grade Tests: {len(tests_simple)} tests")
for t in sorted(tests_simple)[:5]:
    print(f"  • {t.name}")
if len(tests_simple) > 5:
    print(f"  ... and {len(tests_simple)-5} more")

# User docs
user_docs = []
for pattern in ["docs/beginner/**/*.md", "docs/advanced/**/*.md", "README.md", "CLAUDE.md"]:
    user_docs.extend(repo.glob(pattern))
print(f"\n📄 User Documentation: {len(user_docs)} markdown files")

# INTERNAL content (to exclude)
print("\n\n🔧 INTERNAL CONTENT (Exclude from user bot):")
print("-" * 70)

# Source code
src_files = list((repo / "src").rglob("*.py"))
print(f"\n⚙️  Source Code: {len(src_files)} Python files")

# Developer docs
dev_docs = list((repo / "docs/developer").rglob("*.md"))
dev_nbs = list((repo / "docs/developer").rglob("*.ipynb"))
print(f"\n👨‍💻 Developer Docs: {len(dev_docs)} markdown, {len(dev_nbs)} notebooks")

# Planning docs
planning = list((repo / "planning").rglob("*.md"))
print(f"\n📋 Planning Docs: {len(planning)} markdown files")

# Complex tests
tests_complex = list((repo / "tests").glob("test_[1-9]*.py"))
print(f"\n🧪 Complex Tests: {len(tests_complex)} tests")

# Summary
print("\n\n💡 RECOMMENDATION FOR USER BOT:")
print("=" * 70)
print("\n✅ INDEX (User-facing):")
print(f"  • {len(tutorials)} tutorial notebooks")
print(f"  • {len(examples_nb)} example notebooks")
print(f"  • {len(tests_simple)} A/B grade test files")
print(f"  • {len([d for d in user_docs if 'developer' not in str(d)])} user docs")
print(f"  • README.md, CLAUDE.md (key context)")
print(f"\n  TOTAL: ~{len(tutorials) + len(examples_nb) + len(tests_simple) + 20} files")

print("\n❌ EXCLUDE (Internal):")
print(f"  • {len(src_files)} source code files")
print(f"  • {len(dev_docs)} developer docs")
print(f"  • {len(planning)} planning docs")
print(f"  • {len(tests_complex)} complex test files")

print("\n📝 Next step: Configure .env to index only user-facing paths")
