#!/bin/bash
set -e

# This script will build the Timeseria container and then use it to build the docs.

# Build
if [[ "x$BUILD" != "xFalse" ]]; then

    echo -e  "\n===================================="
    echo -e  "|  Building the Docker container   |"
    echo -e  "====================================\n"

    cd ../docker
    echo "Building Timeseria Docker container. Use BUILD=False to skip."
    ./build.sh
    cd ../
else
    cd ..
fi

# Build the docs
echo -e  "\n===================================="
echo -e  "|  Building docs in the container  |"
echo -e  "====================================\n"

# Reduce verbosity and disable Python buffering
ENV_VARS="PYTHONWARNINGS=ignore TF_CPP_MIN_LOG_LEVEL=3 PYTHONUNBUFFERED=one"

# Start the build
# Note: requirements are usually already installed, the pip install here is necessary only when changing them.
docker run -v $PWD:/opt/Timeseria -it timeseria "pip install -r /opt/Timeseria/requirements.txt && cd /opt/Timeseria/docs && $ENV_VARS make clean && make html"
