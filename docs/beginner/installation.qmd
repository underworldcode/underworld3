---
title: "Installation Guide"
subtitle: "Get Underworld3 running on your system"
authors:
- name: Underworld Team

exports:
- format: html
---

## Quick Install (Recommended)

The fastest way to get started is with the `./uw` wrapper script:

```bash
git clone https://github.com/underworldcode/underworld3
cd underworld3
./uw setup
```

The setup wizard will:

1. Install [pixi](https://pixi.sh) if needed (a fast, modern package manager)
2. Ask two simple questions to configure your environment
3. Install all dependencies and build underworld3

**That's it!** The whole process takes about 5 minutes.

### Environment Options

The setup wizard asks two questions:

**1. Do you need adaptive mesh refinement (AMR)?**

Most users should answer **No**. This installs PETSc from conda-forge (~5 minutes).

Answer **Yes** only if you need anisotropic mesh adaptation tools (pragmatic, mmg, parmmg). This builds a custom PETSc from source (~1 hour).

**2. What features do you need?**

| Choice | Includes | Best for |
|--------|----------|----------|
| **Runtime** (default) | PyVista visualization, JupyterLab | Most users, tutorials |
| **Minimal** | Core dependencies only | HPC production, CI |
| **Developer** | + black, mypy, Claude Code | Contributing to underworld3 |

### Common Commands

After installation, use `./uw` for everything:

```bash
./uw                  # Show status and available environments
./uw jupyter lab      # Start JupyterLab
./uw test             # Run quick validation tests
./uw build            # Rebuild after source changes
./uw doctor           # Diagnose configuration issues
./uw status           # Check for updates on GitHub
./uw update           # Pull latest changes and rebuild
./uw --help           # Full command reference
```

### Staying Up to Date

Check for updates without disrupting your work:

```bash
./uw status           # See what's new on GitHub
./uw update           # Pull and rebuild when ready
```

### Troubleshooting

If something isn't working, run diagnostics:

```bash
./uw doctor
```

This checks your environment and provides specific fixes for common issues like PETSc version mismatches or missing dependencies.

---

## Alternative: Docker Container

For Windows users (without WSL) or quick trials without installation:

```bash
docker pull underworldcode/underworld3:latest
docker run -it -p 8888:8888 underworldcode/underworld3:latest
```

This launches JupyterLab with underworld3 pre-installed.

As the code is in active development, containers may lag behind the latest changes. Check the [GitHub releases](https://github.com/underworldcode/underworld3/releases) for available container versions.

---

## HPC Builds

Underworld3 is designed for high performance computing. The symbolic layer is lightweight and doesn't adversely affect launch time or execution performance.

### Using pixi on HPC

If your cluster allows user-space package managers, the `./uw` approach works well:

```bash
git clone https://github.com/underworldcode/underworld3
cd underworld3
./uw setup    # Choose "Minimal" for production runs
```

### Using System PETSc

If your HPC provides a PETSc module, you can build against it. Requirements:

- PETSc 3.21 or higher
- petsc4py matching your PETSc version
- Python 3.10+

```bash
module load petsc/3.21   # (example - check your system)

git clone https://github.com/underworldcode/underworld3
cd underworld3
pip install . --no-build-isolation
```

### Getting Help

HPC environments vary widely. Contact us through the [GitHub issue tracker](https://github.com/underworldcode/underworld3/issues) for assistance with specific systems. Include:

- System name and architecture
- Available PETSc/module versions
- Any error messages

---

## Verification

After installation, verify everything works:

```bash
./uw test
```

This runs a quick suite of validation tests (~2 minutes). For more thorough testing:

```bash
./uw test-all --isolation   # Full test suite, isolated processes
```

## Uninstalling

To remove underworld3 from the current environment:

```bash
pip uninstall underworld3
```

To clean build artifacts:

```bash
rm -rf build/ src/underworld3/*.c src/underworld3/*.so
```

To completely remove the pixi environment:

```bash
rm -rf .pixi/
```
