#!/bin/bash

# Exit on any error. More advanced approaches could be implemented in future
# (see https://stackoverflow.com/questions/4381618/exit-a-script-on-error)
set -e

if [[ "x$(uname -i)" == "xaarch64" ]] ; then
    export LD_PRELOAD=/usr/local/lib/python3.8/dist-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0 
fi

if [[ "x$@" == "x" ]] ; then

    # Reduce verbosity and disable Python buffering
    export PYTHONWARNINGS=ignore
    export TF_CPP_MIN_LOG_LEVEL=3
    export PYTHONUNBUFFERED=on
    export PYTHONPATH=\$PYTHONPATH:/opt/Timeseria

    # Set base port
	if [ "x$BASE_PORT" == "x" ]; then
	    BASE_PORT=8888
	fi

    # Set base dir
    if [ "x$BASE_DIR" == "x" ]; then
        BASE_DIR='/notebooks'
    else
        BASE_DIR="'$BASE_DIR'"
    fi

    # If base dir is empty, clone
    if [ "x$(ls -A $BASE_DIR)" == "x" ]; then
        echo "Notebooks directory is empty, downloading demo notebooks..."
        cd /tmp && git clone https://github.com/sarusso/Timeseria-notebooks
        mv Timeseria-notebooks/notebooks/* $BASE_DIR/
    fi

    # Start Jupyter as default entrypoint
    echo "Running Jupyter..."
    echo ""

    cd /opt/Timeseria
    jupyter trust $BASE_DIR/*.ipynb
    jupyter notebook --ip=0.0.0.0 --port=$BASE_PORT --NotebookApp.token='' --NotebookApp.notebook_dir=$BASE_DIR

else
    ENTRYPOINT_COMMAND=$@
    echo -n "Executing Docker entrypoint command: "
    echo $ENTRYPOINT_COMMAND
    exec /bin/bash -c "$ENTRYPOINT_COMMAND"
fi














