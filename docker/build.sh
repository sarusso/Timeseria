#!/bin/bash
set -e

# Check if we are in the right place
if [ ! -f ./Dockerfile ]; then
    echo "No Dockerfile found. Are you executing this command in the 'docker' subfolder?"
    exit 1
fi

# Move to project root
cd ../

if [[ "x$CACHE" == "xFalse" ]]; then
    echo "Building without cache."
    docker build --no-cache -f docker/Dockerfile ./ -t timeseria
else
    echo "Building with cache. Use CACHE=False to disable it."
    docker build -f docker/Dockerfile ./ -t timeseria
fi
