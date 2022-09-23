#!/bin/bash
rm -fr build
rm -fr cython_debug
find . -name \*.so |xargs rm
