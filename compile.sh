#!/usr/bin/env bash

set -x  # show commands

pip install . --no-build-isolation -v | tee compile.log
