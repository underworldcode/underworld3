#!/usr/bin/env bash

set -x  # show commands

pip install . --no-build-isolation 2>&1 | tee compile.log
