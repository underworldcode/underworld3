#!/bin/zsh
alias python=python3.11
pdoc3 --config latex_math=True --html --output-dir uw3_api_docs underworld3 --force
