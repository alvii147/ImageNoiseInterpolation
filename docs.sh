#!/usr/bin/bash

# remove current html docs
rm -rf docs/build/

# build html docs
sphinx-build -b html docs/source/ docs/build/html