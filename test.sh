#!/bin/bash
set -e

if [[ "x$BUILD" != "xFalse" ]]; then

    # Build Docker container
    echo -e  "\n==============================="
    echo -e  "|  Building Docker container  |"
    echo -e  "===============================\n"

    if [[ "x$CACHE" == "xFalse" ]]; then
        docker build --no-cache -f containers/Ubuntu_18.04/Dockerfile ./ -t timeseria_test_container
    else
        docker build -f containers/Ubuntu_18.04/Dockerfile ./ -t timeseria_test_container
    fi
fi
            
# Start testing
echo -e  "\n==============================="
echo -e  "|   Running tests             |"
echo -e  "===============================\n"

# Reduce verbosity and disable Python buffering
ENV_VARS="PYTHONWARNINGS=ignore TF_CPP_MIN_LOG_LEVEL=3 PYTHONUNBUFFERED=on EXTENDED_TESTING=False LOGLEVEL=$LOGLEVEL"

if [ $# -eq 0 ]; then
    docker run -v $PWD:/opt/Timeseria -it timeseria_test_container /bin/bash -c "cd /opt/Timeseria && $ENV_VARS KERAS_BACKEND=tensorflow python3 -m unittest"
else
    docker run -v $PWD:/opt/Timeseria -it timeseria_test_container /bin/bash -c "cd /opt/Timeseria && $ENV_VARS KERAS_BACKEND=tensorflow python3 -m unittest $@"
fi
