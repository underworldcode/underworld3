#!/usr/bin/env python3
"""
Binder Launch Wizard for Underworld3

Generates mybinder.org launch URLs and badges for any repository
that wants to use the Underworld3 environment.

Usage:
    python scripts/binder_wizard.py

Or run interactively:
    python -c "from scripts.binder_wizard import generate_badge; print(generate_badge('username/repo'))"
"""

import urllib.parse
import sys
from typing import Optional

# Configuration
DEFAULT_LAUNCHER_REPO = "underworldcode/uw3-binder-launcher"
DEFAULT_LAUNCHER_BRANCH = "uw3-release-candidate"


def url_encode(s: str) -> str:
    """URL encode a string."""
    return urllib.parse.quote(s, safe='')


def double_encode(s: str) -> str:
    """Double URL encode (needed for nested URL parameters)."""
    return urllib.parse.quote(urllib.parse.quote(s, safe=''), safe='')


def generate_binder_url(
    content_repo: str,
    content_branch: str = "main",
    notebook_path: Optional[str] = None,
    launcher_repo: str = DEFAULT_LAUNCHER_REPO,
    launcher_branch: str = DEFAULT_LAUNCHER_BRANCH,
    urlpath_type: str = "lab"
) -> str:
    """
    Generate a mybinder.org URL that launches a content repository
    using the Underworld3 base environment.

    Parameters
    ----------
    content_repo : str
        GitHub repository in "owner/repo" format (e.g., "username/my-course")
    content_branch : str
        Branch of the content repository to clone (default: "main")
    notebook_path : str, optional
        Path to a specific notebook to open (relative to repo root)
    launcher_repo : str
        The UW3 launcher repository (default: "underworldcode/uw3-launcher")
    launcher_branch : str
        Branch of the launcher repository (default: "main")
    urlpath_type : str
        JupyterLab interface type: "lab" or "tree" (default: "lab")

    Returns
    -------
    str
        Full mybinder.org launch URL

    Examples
    --------
    >>> generate_binder_url("myuser/geodynamics-course")
    'https://mybinder.org/v2/gh/underworldcode/uw3-launcher/main?urlpath=...'

    >>> generate_binder_url("myuser/course", notebook_path="tutorials/intro.ipynb")
    'https://mybinder.org/v2/gh/underworldcode/uw3-launcher/main?urlpath=...'
    """
    # Extract repo name for the target directory
    repo_name = content_repo.split('/')[-1]

    # Build the git-pull URL
    content_url = f"https://github.com/{content_repo}"

    # Build the urlpath for after cloning
    if notebook_path:
        target_path = f"{urlpath_type}/tree/{repo_name}/{notebook_path}"
    else:
        target_path = f"{urlpath_type}/tree/{repo_name}"

    # Construct the nbgitpuller URL (needs double encoding for nested params)
    gitpull_params = (
        f"repo={double_encode(content_url)}"
        f"&branch={url_encode(content_branch)}"
        f"&urlpath={double_encode(target_path)}"
    )

    # Full urlpath parameter
    urlpath = f"git-pull?{gitpull_params}"

    # Final binder URL
    binder_url = (
        f"https://mybinder.org/v2/gh/{launcher_repo}/{launcher_branch}"
        f"?urlpath={url_encode(urlpath)}"
    )

    return binder_url


def generate_badge_markdown(
    content_repo: str,
    content_branch: str = "main",
    notebook_path: Optional[str] = None,
    badge_label: str = "Launch in Binder",
    **kwargs
) -> str:
    """
    Generate a markdown badge for launching on mybinder.org.

    Parameters
    ----------
    content_repo : str
        GitHub repository in "owner/repo" format
    content_branch : str
        Branch to clone (default: "main")
    notebook_path : str, optional
        Path to a specific notebook to open
    badge_label : str
        Alt text for the badge (default: "Launch in Binder")
    **kwargs
        Additional arguments passed to generate_binder_url()

    Returns
    -------
    str
        Markdown badge string
    """
    url = generate_binder_url(content_repo, content_branch, notebook_path, **kwargs)
    return f"[![{badge_label}](https://mybinder.org/badge_logo.svg)]({url})"


def generate_badge_html(
    content_repo: str,
    content_branch: str = "main",
    notebook_path: Optional[str] = None,
    badge_label: str = "Launch in Binder",
    **kwargs
) -> str:
    """
    Generate an HTML badge for launching on mybinder.org.

    Parameters
    ----------
    content_repo : str
        GitHub repository in "owner/repo" format
    content_branch : str
        Branch to clone (default: "main")
    notebook_path : str, optional
        Path to a specific notebook to open
    badge_label : str
        Alt text for the badge (default: "Launch in Binder")
    **kwargs
        Additional arguments passed to generate_binder_url()

    Returns
    -------
    str
        HTML badge string
    """
    url = generate_binder_url(content_repo, content_branch, notebook_path, **kwargs)
    return f'<a href="{url}"><img src="https://mybinder.org/badge_logo.svg" alt="{badge_label}"></a>'


def generate_rst_badge(
    content_repo: str,
    content_branch: str = "main",
    notebook_path: Optional[str] = None,
    badge_label: str = "Launch in Binder",
    **kwargs
) -> str:
    """
    Generate a reStructuredText badge for launching on mybinder.org.

    Parameters
    ----------
    content_repo : str
        GitHub repository in "owner/repo" format
    content_branch : str
        Branch to clone (default: "main")
    notebook_path : str, optional
        Path to a specific notebook to open
    badge_label : str
        Alt text for the badge (default: "Launch in Binder")
    **kwargs
        Additional arguments passed to generate_binder_url()

    Returns
    -------
    str
        RST badge string
    """
    url = generate_binder_url(content_repo, content_branch, notebook_path, **kwargs)
    return f".. image:: https://mybinder.org/badge_logo.svg\n   :target: {url}\n   :alt: {badge_label}"


def interactive_wizard():
    """Run an interactive wizard to generate binder launch URLs."""

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     Underworld3 Binder Launch Wizard                         ║
║                                                                              ║
║  Generate mybinder.org launch URLs for your Underworld3 content              ║
║  No .binder directory needed in your repository!                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    # Get repository information
    print("Step 1: Your Content Repository")
    print("-" * 40)
    content_repo = input("  GitHub repository (e.g., username/my-course): ").strip()

    if not content_repo or '/' not in content_repo:
        print("\n  Error: Please enter a valid GitHub repository in 'owner/repo' format")
        sys.exit(1)

    content_branch = input("  Branch [main]: ").strip() or "main"

    # Optional notebook path
    print("\nStep 2: Starting Point (Optional)")
    print("-" * 40)
    notebook_path = input("  Notebook path (e.g., tutorials/intro.ipynb) [none]: ").strip() or None

    # Advanced options
    print("\nStep 3: Advanced Options")
    print("-" * 40)
    use_advanced = input("  Configure advanced options? [y/N]: ").strip().lower()

    launcher_repo = DEFAULT_LAUNCHER_REPO
    launcher_branch = DEFAULT_LAUNCHER_BRANCH

    if use_advanced == 'y':
        launcher_repo = input(f"  Launcher repository [{DEFAULT_LAUNCHER_REPO}]: ").strip() or DEFAULT_LAUNCHER_REPO
        launcher_branch = input(f"  Launcher branch [{DEFAULT_LAUNCHER_BRANCH}]: ").strip() or DEFAULT_LAUNCHER_BRANCH

    # Generate outputs
    print("\n")
    print("=" * 78)
    print("                          GENERATED LAUNCH LINKS")
    print("=" * 78)

    # Direct URL
    url = generate_binder_url(
        content_repo, content_branch, notebook_path,
        launcher_repo=launcher_repo, launcher_branch=launcher_branch
    )
    print("\n📎 Direct Launch URL:")
    print("-" * 78)
    print(url)

    # Markdown badge
    md_badge = generate_badge_markdown(
        content_repo, content_branch, notebook_path,
        launcher_repo=launcher_repo, launcher_branch=launcher_branch
    )
    print("\n📝 Markdown Badge (for README.md):")
    print("-" * 78)
    print(md_badge)

    # HTML badge
    html_badge = generate_badge_html(
        content_repo, content_branch, notebook_path,
        launcher_repo=launcher_repo, launcher_branch=launcher_branch
    )
    print("\n🌐 HTML Badge:")
    print("-" * 78)
    print(html_badge)

    # RST badge
    rst_badge = generate_rst_badge(
        content_repo, content_branch, notebook_path,
        launcher_repo=launcher_repo, launcher_branch=launcher_branch
    )
    print("\n📄 reStructuredText Badge:")
    print("-" * 78)
    print(rst_badge)

    # Instructions
    print("\n")
    print("=" * 78)
    print("                              INSTRUCTIONS")
    print("=" * 78)
    print("""
1. Copy the Markdown badge above and paste it into your README.md

2. Your repository does NOT need:
   - A .binder/ directory
   - A Dockerfile
   - Any special configuration

3. Requirements for your repository:
   - Must be public on GitHub
   - Notebooks should use the 'python3' kernel
   - Import underworld3 as: import underworld3 as uw

4. What happens when users click the badge:
   a. mybinder.org launches the Underworld3 environment (cached, fast!)
   b. nbgitpuller clones your repository into the workspace
   c. User sees your notebooks with UW3 ready to use

5. Updates to your repository are pulled automatically on each launch
   (no need to rebuild anything!)
""")

    print("=" * 78)
    print("  Happy computing with Underworld3!")
    print("=" * 78)


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        # Command-line mode
        if sys.argv[1] in ['-h', '--help']:
            print(__doc__)
            print("\nFunctions available for programmatic use:")
            print("  generate_binder_url(content_repo, content_branch, notebook_path)")
            print("  generate_badge_markdown(content_repo, ...)")
            print("  generate_badge_html(content_repo, ...)")
            print("  generate_rst_badge(content_repo, ...)")
            sys.exit(0)

        # Quick mode: just generate badge for given repo
        content_repo = sys.argv[1]
        content_branch = sys.argv[2] if len(sys.argv) > 2 else "main"
        notebook_path = sys.argv[3] if len(sys.argv) > 3 else None

        print(generate_badge_markdown(content_repo, content_branch, notebook_path))
    else:
        # Interactive mode
        interactive_wizard()


if __name__ == "__main__":
    main()
