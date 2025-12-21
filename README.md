# underworld3

<center>
<img src="docs/user/media/SocialShareS.png", width=80%>
</center>

Welcome to `Underworld3`, a mathematically self-describing, finite-element code for geodynamic modelling. This quick-start guide has basic installation instructions and a brief introduction to some of the concepts in the `Underworld3` code.

All `Underworld3` source code is released under the LGPL-3 open source licence. This covers all files in `underworld3` constituting the Underworld3 Python module. Notebooks, stand-alone documentation and Python scripts which show how the code is used and run are licensed under the Creative Commons Attribution 4.0 International License.

## Status

main branch

[![test_uw3](https://github.com/underworldcode/underworld3/actions/workflows/build_uw3_and_test.yml/badge.svg?branch=main)](https://github.com/underworldcode/underworld3/actions/workflows/build_uw3_and_test.yml)


development branch

[![test_uw3](https://github.com/underworldcode/underworld3/actions/workflows/build_uw3_and_test.yml/badge.svg?branch=development)](https://github.com/underworldcode/underworld3/actions/workflows/build_uw3_and_test.yml)

## Documentation

Start with the online [Documentation](https://underworldcode.github.io/underworld3/development/docs/index.html) for a brief overview of the code.

The `underworld3` module (API) documentation can be found online:
  - [stable version](https://underworldcode.github.io/underworld3/main_api/underworld3/index.html)
  - [development version](https://underworldcode.github.io/underworld3/development_api/underworld3/index.html)



## Binder demonstration version

 - [Main Branch on Binder](https://mybinder.org/v2/gh/underworld-community/uw3-demo-launcher/HEAD?labpath=underworld3%2Fdocs%2Fbeginner%2Ftutorials%2FNotebook_Index.ipynb)
 - [Development Branch on Binder](https://mybinder.org/v2/gh/underworld-community/uw3-demo-launcher-dev/HEAD?labpath=underworld3%2Fdocs%2Fbeginner%2Ftutorials%2FNotebook_Index.ipynb)


## Installation Guide

The quickest option is **not to install** anything but try the binder demo above!

### Quick Install (recommended)

```bash
git clone https://github.com/underworldcode/underworld3
cd underworld3
./uw setup
```

The `./uw` wrapper handles everything:
- Installs [pixi](https://pixi.sh) if needed
- Guides you through environment selection
- Installs dependencies and builds underworld3

### Environment Options

Underworld3 uses [pixi](https://pixi.sh) to manage dependencies. The setup wizard will ask you two questions to determine which environment to install:

**1. Do you need adaptive mesh refinement (AMR)?**

Most users should answer **No**. This installs PETSc from conda-forge, which takes about 5 minutes.

Answer **Yes** only if you need anisotropic mesh adaptation tools (pragmatic, mmg, parmmg). This builds a custom PETSc from source with these tools enabled, which takes approximately one hour.

**2. What features do you need?**

- **Runtime** (recommended): Includes PyVista for 3D visualization and JupyterLab for running tutorial notebooks. This is what most users want.

- **Minimal**: Only the core dependencies needed to build and run underworld3. Use this for production runs on HPC systems or when you have your own visualization workflow.

- **Developer**: Adds code quality tools (black, mypy), documentation tools, and Claude Code support. Use this if you plan to contribute to underworld3 development.

**Summary of environments:**

| Environment | PETSc Source | Features | Use Case |
|-------------|--------------|----------|----------|
| `runtime` | conda-forge | viz, jupyter | Running tutorials and examples |
| `default` | conda-forge | minimal | HPC production runs |
| `dev` | conda-forge | viz, jupyter, dev tools | Contributing to underworld3 |
| `amr-runtime` | custom build | viz, jupyter, AMR | Adaptive mesh research |
| `amr` | custom build | minimal, AMR | AMR on HPC |
| `amr-dev` | custom build | all features | AMR development |

### Getting Started

After installation, open the tutorial index in JupyterLab:

```bash
./uw jupyter lab docs/beginner/tutorials/Notebook_Index.ipynb
```

This opens a guided introduction to underworld3 with links to all beginner tutorials.

### Common Commands

```bash
./uw                  # Show status and available environments
./uw setup            # (Re)configure environment
./uw build            # Rebuild after source changes
./uw test             # Run quick tests
./uw jupyter lab      # Start JupyterLab
./uw doctor           # Diagnose configuration issues
./uw status           # Check for updates on GitHub
./uw update           # Pull latest changes and rebuild
./uw --help           # Full documentation
```

### Troubleshooting

If you encounter build errors or import failures, run diagnostics:

```bash
./uw doctor
```

This checks your environment configuration and provides specific fixes for common issues like:
- Missing dependencies (PETSc, petsc4py)
- Library version mismatches
- Environment configuration problems

The `./uw build` command automatically handles dependency chains—if you're missing petsc4py in an AMR environment, it will build it for you before compiling underworld3.

### Alternative: Manual Installation

For more control, see the [Installation Instructions](https://underworldcode.github.io/underworld3/development/docs/beginner/installation.html)

## References and Archives 

The canonical releases of the code can be found at [zenodo.org](https://doi.org/10.5281/zenodo.16810746).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16810747.svg)](https://doi.org/10.5281/zenodo.16810746)

There is a [publication in the Journal of Open Source Software](https://joss.theoj.org/papers/4f7a1ed76bde560968c246fa8eff778d) 
([doi:10.21105/joss.07831](https://doi.org/10.21105/joss.07831)) that can be cited when using the software. 

[![status](https://joss.theoj.org/papers/4f7a1ed76bde560968c246fa8eff778d/status.svg)](https://joss.theoj.org/papers/4f7a1ed76bde560968c246fa8eff778d)
