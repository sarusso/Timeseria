# Timeseria

Timeseria is an object-oriented time series processing library which aims at making it easy to manipulate time series data and to build statistical and machine learning models on top of it.

Unlike common numerical and data analysis frameworks, Timeseria does not make use of low level structures as arrays and matrices for representing data. Instead, it builds up from well defined and reusable logical units (objects) which can be easily composed together, ensuring a high level of consistency.

Because of this approach, Timeseria addresses by design all those annoying things which are often left as an implementation detail but that actually cause wasting massive amounts of time - as handling data losses, non-uniform sampling rates, differences between aggregated data and punctual observations, timezones, DST changes, and so on.

Timeseria  comes with a built-in set of common operations (resampling, aggregation, differencing etc.) and models (reconstruction, forecasting and anomaly detection) which can be easily extended with custom ones, and integrates a powerful plotting engine based on Dygraphs capable of plotting even millions of data points.

![Time series plot](docs/altogether.png?raw=true "Timeseria at work")





## Installing

You can install Timeseria by just using the the [PyPI package](https://pypi.org/project/timeseria/):

    pip install timeseria

Otherwise, a Docker image with a Jupyter Notebook server is ready to be played with on [Docker Hub](https://hub.docker.com/r/sarusso/timeseria):

    docker run -it -p8888:8888 -v$PWD:/notebooks sarusso/timeseria


## Getting started

You can get started by reading the [quickstart](https://github.com/sarusso/Timeseria-notebooks/blob/master/notebooks/Quickstart.ipynb) or the [welcome](https://github.com/sarusso/Timeseria-notebooks/blob/master/notebooks/Welcome.ipynb) notebooks, or have a look at the other example notebooks provided in the [Timeseria-notebooks](https://github.com/sarusso/Timeseria-notebooks) repository. 

Also the [reference documentation](https://timeseria.readthedocs.io) might be useful.

The following code (directly taken from the [quickstart](https://github.com/sarusso/Timeseria-notebooks/blob/master/notebooks/Quickstart.ipynb)) gives instead a quick idea about how Timeseria was used to generate the plot shown in the introduction.

```python
# Load some test data
from timeseria import storages
csv_storage = storages.CSVFileStorage('temperature.csv')
temperature_timeseries = csv_storage.get()
```

```python
# Resample and make data uniform: data losses are detectd and handled
temperature_timeseries = temperature_timeseries.resample('1h')  
```

```python
# Fit and apply a data reconstruction model
from timeseria.models.reconstructors import PeriodicAverageReconstructor
reconstructor = PeriodicAverageReconstructor()
reconstructor.fit(temperature_timeseries)
temperature_timeseries = reconstructor.apply(temperature_timeseries)
```

```python
# Fit and apply a forecasting model and forecast three days
from timeseria.models.forecasters import LSTMForecaster
forecaster = LSTMForecaster(window=12, neurons=64, features=['values', 'diffs', 'hours'])
forecaster.fit(temperature_timeseries)
temperature_timeseries = forecaster.apply(temperature_timeseries, steps=3*24)   
```

```python
# Fit and apply an anomaly detection model
from timeseria.models.anomaly_detectors import PeriodicAverageAnomalyDetector
anomaly_detector = PeriodicAverageAnomalyDetector()
anomaly_detector.fit(temperature_timeseries, stdevs=5)
temperature_timeseries = anomaly_detector.apply(temperature_timeseries)
```

```python
# Plot everything
temperature_timeseries.plot()    
```


## Development

To work in local development mode, you can both run a Jupyter notebook (in a Docker container):

    ./jupyter.sh

or you can run the unit tests (in a Docker container as well):

    ./test.sh

Both commands will mount the local codebase inside the container as a volume to allow for live code changes, and trigger a container build so that if any requirement is changed it will be reflected in the container as well.

If you don't want to automatically trigger a container build every time you run them, prepend a `BUILD=False`:

    BUILD=False ./test.sh

You can also run only specific tests:

    BUILD=False ./test.sh timeseria.tests.test_datastructures

To instead set a specific log level when testing (default is CRITICAL):

    TIMESERIA_LOGLEVEL=DEBUG ./test.sh


## Testing [![Tests status](https://github.com/sarusso/timeseria/actions/workflows/ci.yml/badge.svg)](https://github.com/sarusso/Timeseria/actions)

Every commit on the Timeseria codebase is automatically tested with GitHub Actions. [Check all branch statuses](https://github.com/sarusso/Timeseria/actions).


## Licensing
Timeseria is licensed under the Apache License version 2.0, unless otherwise specified. See [LICENSE](https://github.com/sarusso/Timeseria/blob/master/LICENSE) for the full license text.





