#!/bin/bash -x

## Clean-up script (removes build artifacts etc)

rm -fr build
rm -fr cython_debug
find . -name \*.so -exec rm {} +
find . -name __pycache__ -exec rm -r {} +
rm -rf underworld3.egg-info
rm -rf .pytest_cache
rm -rf uw3_api_docs
rm ./src/underworld3/_uwid.py
#git clean -dfX
