#!/bin/bash -x

rm -fr build
rm -fr cython_debug
find . -name \*.so -exec rm {} +
rm -rf underworld3.egg-info
rm -rf .pytest_cache
rm -rf uw3_api_docs
