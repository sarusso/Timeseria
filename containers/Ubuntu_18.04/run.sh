#!/bin/bash
set -e

# Move to project root
cd ../../

# Build Docker container
echo -e  "\n==============================="
echo -e  "|  Running Docker container   |"
echo -e  "===============================\n"

if [[ "x$LIVE" == "xTrue" ]]; then
    echo "Running with live code changes"
    docker run -p8888:8888 -v $PWD:/opt/Timeseria -v $PWD/notebooks:/notebooks -it timeseria
else
    echo "Running without live code changes. Use LIVE=True to enable them."
    docker run -p8888:8888 -it timeseria
fi

