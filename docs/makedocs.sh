#!/bin/bash
set -e

# Move to container dir
cd ../containers/Ubuntu_20.04

if [[ "x$BUILD" != "xFalse" ]]; then
    # Build
    echo ""
    echo "Building Timeseria Docker container. Use BUILD=False to skip."
    ./build.sh
fi
            
        
# Start building the docs
cd ../../

echo -e  "\n==============================="
echo -e  "|     Building docs           |"
echo -e  "===============================\n"

# Reduce verbosity and disable Python buffering
ENV_VARS="PYTHONWARNINGS=ignore TF_CPP_MIN_LOG_LEVEL=3 PYTHONUNBUFFERED=on EXTENDED_TESTING=False LOGLEVEL=$LOGLEVEL "

docker run -v $PWD:/opt/Timeseria -it timeseria "cd /opt/Timeseria/docs && $ENV_VARS make clean && make html"




