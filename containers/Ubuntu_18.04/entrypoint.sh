#!/bin/bash

# Exit on any error. More complex thing could be done in future
# (see https://stackoverflow.com/questions/4381618/exit-a-script-on-error)
set -e


if [[ "x$@" == "x" ]] ; then
    

# Start testing
    echo -e  "Running Jupyter"

    # Reduce verbosity and disable Python buffering
    export PYTHONWARNINGS=ignore
    export TF_CPP_MIN_LOG_LEVEL=3
    export PYTHONUNBUFFERED=on
    export EXTENDED_TESTING=False
    export PYTHONPATH=\$PYTHONPATH:/opt/Timeseria
    
    cd /opt/Timeseria && jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --NotebookApp.token='' --NotebookApp.notebook_dir='/notebooks'

else
    ENTRYPOINT_COMMAND=$@
    echo -n "Executing Docker entrypoint command: "
    echo $ENTRYPOINT_COMMAND
    exec /bin/bash -c "$ENTRYPOINT_COMMAND"
fi














