#!/bin/bash
set -e

# This script will build the Timeseria container and start it with Jupyter.

# Build
if [[ "x$BUILD" != "xFalse" ]]; then

    echo -e  "\n===================================="
    echo -e  "|  Building the Docker container   |"
    echo -e  "====================================\n"

    cd docker/
    echo "Building Timeseria Docker container. Use BUILD=False to skip."
    ./build.sh
    cd ../

fi

# Run Jupyter
echo -e  "\n===================================="
echo -e  "| Running Jupyter in the container |"
echo -e  "====================================\n"

# Note: to enable image-based plots by default, use DEFAULT_PLOT_TYPE=image ./jupyter.sh
if [[ "x$NOTEBOOKS_DIR" == "x" ]]; then
    echo "Running with no notebooks dir."
    docker run -p8888:8888 -eDEFAULT_PLOT_TYPE=$DEFAULT_PLOT_TYPE -v $PWD:/opt/Timeseria -it timeseria
else
    echo "Running with notebook dir \"$NOTEBOOKS_DIR\"."
    docker run -p8888:8888 -eDEFAULT_PLOT_TYPE=$DEFAULT_PLOT_TYPE -v $PWD:/opt/Timeseria -v$NOTEBOOKS_DIR:/notebooks  -it timeseria
fi
