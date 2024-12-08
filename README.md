# Timeseria


[![Tests status](https://github.com/sarusso/timeseria/actions/workflows/ci.yml/badge.svg)](https://github.com/sarusso/Timeseria/actions) [![Licence Apache 2](https://img.shields.io/github/license/sarusso/Timeseria)](https://github.com/sarusso/Timeseria/blob/main/LICENSE) [![Semver 2.0.0](https://img.shields.io/badge/semver-v2.0.0-blue)](https://semver.org/spec/v2.0.0.html) 

Timeseria is an object-oriented time series processing library implemented in Python, which aims at making it easier to manipulate time series data and to build statistical and machine learning models on top of it.

Unlike common numerical and data analysis frameworks, Timeseria does not make use of low level data structures as arrays and matrices to represent time series data. Instead, it builds up from well defined and reusable logical units (objects), which can be easily combined together in order to ensure an high level of consistency.

Thanks to this approach, Timeseria can address by design several non-trivial issues which are often underestimated, such as handling data losses, non-uniform sampling rates, differences between aggregated data and punctual observations, time zones, daylight saving times, and more.

Timeseria comes with a comprehensive set of base data structures, data transformations for resampling and aggregation, common data manipulation operations, and extensible models for data reconstruction, forecasting and anomaly detection. It also integrates a fully featured, interactive plotting engine capable of handling even millions of data points.


![Time series plot](docs/altogether.png?raw=true "Timeseria at work")


## Getting started

You can get started by reading the [quickstart](https://github.com/sarusso/Timeseria-notebooks/blob/main/notebooks/Quickstart.ipynb) or the [welcome](https://github.com/sarusso/Timeseria-notebooks/blob/main/notebooks/Welcome.ipynb) notebooks, or have a look at the other example notebooks provided in the [Timeseria-notebooks](https://github.com/sarusso/Timeseria-notebooks) repository. 

Also the [reference documentation](https://timeseria.readthedocs.io) might be useful.



## Installing

You can install Timeseria by just using the the [PyPI package](https://pypi.org/project/timeseria/):

    pip install timeseria

Alternatively, a Timeseria Docker image with a Jupyter Notebook server and all the requirements is ready to be played with on [Docker Hub](https://hub.docker.com/r/sarusso/timeseria):

    docker run -it -p8888:8888 -v$PWD:/notebooks sarusso/timeseria

You can also clone this repo, install the requirements from the `requirements.txt` file and add it to your `PYTHONPATH`.

## Development

To work in development mode, you can either run a Jupyter notebook:

    ./jupyter.sh

or run the unit tests:

    ./test.sh

Both commands will start a Docker container and mount the local codebase inside it as a volume to allow for live code changes. They will trigger a container build so that if any requirement is changed, it will be reflected in the container as well.

If you don't want to automatically trigger a container build, prepend a `BUILD=False`:

    BUILD=False ./test.sh

You can also run only specific tests:

    BUILD=False ./test.sh timeseria.tests.test_datastructures

To instead set a specific log level when testing (default is CRITICAL):

    TIMESERIA_LOGLEVEL=DEBUG ./test.sh


## Testing

Every push on the Timeseria codebase as well as all the pull requests are automatically tested with GitHub Actions: [check all branch statuses](https://github.com/sarusso/Timeseria/actions). Check the previous paragraph  for how to run the unit tests when in development mode.


## License
Timeseria is licensed under the Apache License version 2.0, unless otherwise specified. See [LICENSE](https://github.com/sarusso/Timeseria/blob/main/LICENSE) for the full license text.





