#!/bin/bash
set -e

# This script will build the Timeseria container for the arch in use and run the tests into it.

# Move to container dir
cd containers/Ubuntu_20.04

if [[ "x$BUILD" != "xFalse" ]]; then
    # Build
    echo ""
    echo "Building Timeseria Docker container. Use BUILD=False to skip."
    ./build.sh
fi
            
        
# Start testing
cd ../../

echo -e  "\n==============================="
echo -e  "|   Running tests             |"
echo -e  "===============================\n"

# Reduce verbosity and disable Python buffering
ENV_VARS="PYTHONWARNINGS=ignore TF_CPP_MIN_LOG_LEVEL=3 PYTHONUNBUFFERED=on EXTENDED_TESTING=False TIMESERIA_LOGLEVEL=$TIMESERIA_LOGLEVEL"

# The "cd" as first command does not work, hence the "date".
if [ $# -eq 0 ]; then
    docker run -v $PWD:/opt/Timeseria -it timeseria "date && cd /opt/Timeseria && $ENV_VARS python3 -m unittest discover"
else
    docker run -v $PWD:/opt/Timeseria -it timeseria "date && cd /opt/Timeseria && $ENV_VARS python3 -m unittest $@"
fi
