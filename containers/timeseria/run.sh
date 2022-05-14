#!/bin/bash
set -e

# Move to project root
cd ../../

# Build Docker container
echo -e  "\n==============================="
echo -e  "|  Running Docker container   |"
echo -e  "===============================\n"

if [[ "x$LIVE" == "xFalse" ]]; then
    echo "Running without live code changes."
    docker run -p8888:8888 -it timeseria
else
    echo "Running with live code changes. Use LIVE=False to disable."
    #-v your_notebooks_dir:/notebooks 
    docker run -p8888:8888 -v $PWD:/opt/Timeseria -it timeseria
fi

