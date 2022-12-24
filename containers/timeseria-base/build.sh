#!/bin/bash
set -e

# Move to project root
cd ../../

if [[ "x$CACHE" == "xFalse" ]]; then
    echo "Building without cache."
    docker build --no-cache -f containers/timeseria-base/Dockerfile ./ -t timeseria-base
else
    echo "Building with cache. Use CACHE=False to disable it."
    docker build -f containers/timeseria-base/Dockerfile ./ -t timeseria-base
fi
