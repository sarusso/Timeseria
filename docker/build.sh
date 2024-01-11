#!/bin/bash
set -e

# Move to project root
cd ../

if [[ "x$CACHE" == "xFalse" ]]; then
    echo "Building without cache."
    docker build --no-cache -f docker/Dockerfile ./ -t timeseria
else
    echo "Building with cache. Use CACHE=False to disable it."
    docker build -f docker/Dockerfile ./ -t timeseria
fi
