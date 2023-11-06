#!/bin/zsh
alias python=python3.11
export LOGO="https://github.com/underworldcode/underworld3/blob/main/Jupyterbook/Figures/MansoursNightmare.png?raw=true"

pdoc --math --mermaid  --output-dir uw3_api_docs underworld3 -d markdown --logo $LOGO  # --force
# pdoc3 --config latex_math=True --html --output-dir uw3_api_docs underworld3 --force

