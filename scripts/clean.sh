#!/bin/bash -x

## Clean-up script (removes build artifacts etc)
## NOTE: Excludes .pixi/ to avoid deleting conda environment packages!

rm -fr build
rm -fr cython_debug

# Remove .so files but EXCLUDE .pixi directory (contains conda packages)
find . -path ./.pixi -prune -o -name '*.so' -exec rm {} +

# Remove __pycache__ but EXCLUDE .pixi directory
find . -path ./.pixi -prune -o -name __pycache__ -type d -exec rm -r {} +

rm -rf underworld3.egg-info
rm -rf .pytest_cache
rm -rf uw3_api_docs
rm -f ./src/underworld3/_uwid.py
#git clean -dfX
