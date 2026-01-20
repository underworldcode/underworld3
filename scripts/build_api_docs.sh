#!/bin/bash

export LOGO="https://github.com/underworldcode/underworld3/blob/main/Jupyterbook/Figures/MansoursNightmare.png?raw=true"

# echo "PYTHON: " `which python3`
# echo "PDOC: " `which pdoc`

# pdoc --math --mermaid  -o uw3_api_docs -d markdown --logo $LOGO src/underworld3 # --force

# First one must have a local build.
pip install -e . --no-build-isolation

# Build documentation
pdoc3 -o uw3_api_docs --config "latex_math=True" src/underworld3 --html --force
