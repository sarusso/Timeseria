
# Timeseria

Timeseria is a time series processing library which aims at making it easy to handle time series data and to build statistical and machine learning models on top of it.

It comes with a built-in set of common operations (resampling, slotting, differencing etc.) and models (reconstruction, forecasting and anomaly detection). Both custom operations and models can be easily plugged in.

Timeseria also tries to address by design all those annoying things which are often left as an implementation detail but that actually cause wasting massive amounts of time - as handling data losses, non-uniform sampling rates, differences between time-slotted (aggregated) data and punctual observations, variable time units, timezones, DST changes and so on.

You can get started by reading the [quickstart](https://sarusso.github.io/Timeseria/Quickstart.html) or the [welcome](https://sarusso.github.io/Timeseria/Welcome.html) notebooks, or you can run them interactively in Binder using the button below. Also the [reference documentation](https://timeseria.readthedocs.io) might be useful.

Examples are provided in the [Timeseria-notebooks](https://github.com/sarusso/Timeseria-notebooks) repository, and a Docker image ready to be played with is available on [Docker Hub](https://hub.docker.com/r/sarusso/timeseria).


![Time series plot](docs/altogether.png?raw=true "Timeseria at work")






# Testing ![example workflow](https://github.com/sarusso/timeseria/actions/workflows/ci.yml/badge.svg)


Every commit on the Timeseria codebase is automatically tested with GitHub Actions. Travis-CI. [Check all branch statuses](https://github.com/sarusso/Timeseria/actions).


# Licensing
Timeseria is licensed under the Apache License version 2.0, unless otherwise specified. See [LICENSE](https://github.com/sarusso/Timeseria/blob/master/LICENSE) for the full license text.





