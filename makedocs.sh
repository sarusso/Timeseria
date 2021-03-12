#!/bin/bash
set -e

if [[ "x$BUILD" != "xFalse" ]]; then

    # Build test environment
    echo -e "\n================================="
    echo -e "| Building the Docker container |"
    echo -e "=================================\n"

    if [[ "x$cache" == "xnocache" ]]; then
        docker build --no-cache -f containers/Ubuntu_18.04/Dockerfile ./ -t timeseria_test_container
    else
        docker build -f containers/Ubuntu_18.04/Dockerfile ./ -t timeseria_test_container
    fi
fi
            
# Start testing
echo -e  "\n================================="
echo -e  "|     Building the docs         |"
echo -e  "=================================\n"

# Reduce verbosity and disable Python buffering
ENV_VARS="PYTHONWARNINGS=ignore TF_CPP_MIN_LOG_LEVEL=3 PYTHONUNBUFFERED=on EXTENDED_TESTING=False LOGLEVEL=$LOGLEVEL "

docker run -v $PWD:/opt/Timeseria -it timeseria_test_container /bin/bash -c "cd /opt/Timeseria/docs && $ENV_VARS make clean && make html"


