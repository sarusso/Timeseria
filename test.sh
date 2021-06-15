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
            
        
# Start testing
cd ../../

echo -e  "\n==============================="
echo -e  "|   Running tests             |"
echo -e  "===============================\n"

# Reduce verbosity and disable Python buffering
ENV_VARS="PYTHONWARNINGS=ignore TF_CPP_MIN_LOG_LEVEL=3 PYTHONUNBUFFERED=on EXTENDED_TESTING=False LOGLEVEL=$LOGLEVEL"

if [ $# -eq 0 ]; then
    docker run -v $PWD:/opt/Timeseria -it timeseria "cd /opt/Timeseria && $ENV_VARS KERAS_BACKEND=tensorflow python3 -m unittest"
else
    docker run -v $PWD:/opt/Timeseria -it timeseria "cd /opt/Timeseria && $ENV_VARS KERAS_BACKEND=tensorflow python3 -m unittest $@"
fi
