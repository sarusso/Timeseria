#!/bin/bash
if [[ "x$1" == "xbuild" ]] ; then
    cd /opt/timeseria
    python3 setup.py sdist bdist_wheel

elif [[ "x$1" == "xtestpush" ]] ; then
    pip3 install twine 
    twine upload --repository-url https://test.pypi.org/legacy/ dist/*

elif [[ "x$1" == "xpush" ]] ; then
    pip3 install twine 
    twine upload dist/*

else
    docker run -v$PWD:/opt/timeseria -it sarusso/tensorflow:2.7.0 /bin/bash -c "cd /opt/timeseria && ./pypi.sh build"
    #docker run -v$PWD:/opt/timeseria -it sarusso/tensorflow:2.7.0 /bin/bash -c "cd /opt/timeseria && ./pypi.sh testpush"
    docker run -v$PWD:/opt/timeseria -it sarusso/tensorflow:2.7.0 /bin/bash -c "cd /opt/timeseria && ./pypi.sh push"

    # Remove build artifacts
    rm -rf build dist timeseria.egg-info
fi


