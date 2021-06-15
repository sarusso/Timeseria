#!/bin/bash
set -e

# Move to container dir
cd containers/Ubuntu_18.04

if [[ "x$BUILD" != "xFalse" ]]; then
    # Build
    echo ""
    echo "Building Timeseria Docker container. Use BUILD=False to skip."
    ./build.sh
fi
            
# Run
./run.sh
