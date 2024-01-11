#!/bin/bash
set -e

# This script will build the Timeseria container and then run the tests into it.

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

# Run the tests
echo -e  "\n===================================="
echo -e  "|  Running tests in the container  |"
echo -e  "====================================\n"

# Reduce verbosity, disable Python buffering and set the log level
ENV_VARS="PYTHONWARNINGS=ignore TF_CPP_MIN_LOG_LEVEL=3 PYTHONUNBUFFERED=on TIMESERIA_LOGLEVEL=$TIMESERIA_LOGLEVEL"

# Note: "cd" as first command does not work, hence the "date".
# Note: requirements are usually already installed, the pip install here is necessary only when changing them.
if [ $# -eq 0 ]; then
    docker run -v $PWD:/opt/Timeseria -it timeseria "date && pip install -r /opt/Timeseria/requirements.txt && cd /opt/Timeseria && $ENV_VARS python3 -m unittest discover"
else
    docker run -v $PWD:/opt/Timeseria -it timeseria "date && pip install -r /opt/Timeseria/requirements.txt && cd /opt/Timeseria && $ENV_VARS python3 -m unittest $@"
fi
