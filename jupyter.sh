#!/bin/bash
set -e

# This script will build and run the Timeseria container for the arch in use (with Jupyter into it).

# Move to container dir
cd containers/Ubuntu_20.04

if [[ "x$BUILD" != "xFalse" ]]; then
    # Build
    echo ""
    echo "Building Timeseria Docker container. Use BUILD=False to skip."
    ./build.sh
fi
            
# Run
./run.sh
