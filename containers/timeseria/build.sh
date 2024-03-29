#!/bin/bash
set -e

# Move to project root
cd ../../

# Build Docker container
echo -e  "\n==============================="
echo -e  "|  Building Docker container  |"
echo -e  "===============================\n"

if [[ "x$CACHE" == "xFalse" ]]; then
    echo "Building without cache."
    docker build --no-cache -f containers/timeseria/Dockerfile ./ -t timeseria
else
    echo "Building with cache. Use CACHE=False to disable it."
    docker build -f containers/timeseria/Dockerfile ./ -t timeseria

fi
