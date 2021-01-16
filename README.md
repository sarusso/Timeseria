
**WARNING:** this is still only alpha software; this means you use it at your own risk, that test coverage is still in need of expansion, and also that some modules are still in need of being optimized.

# Timeseria

A time series processing library. Provides modules for data cleaning, resampling and reconstrution, as well as forecasting and anomaly detection models.

Timeseria also tries to address and solve by design all those annoying things which are often left as a detail but that actually cause massive amounts of time wasted - as data losses, non-uniform sampling rates, differences between time-slotted data and punctual observations, variable time units, timezones, DST changes and so on.

You can get started by reading the [quickstart](https://sarusso.github.io/Timeseria/Welcome.html), or you can run it interactively in Binder using the button below (once the environment is ready, open the "Welcome" notebook).

Examples are provided in the [Timeseria-notebooks](https://github.com/sarusso/Timeseria-notebooks) repository, and are accessible in Binder as well, ready to be played with.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sarusso/Timeseria-notebooks/HEAD)

![Time series plot](docs/plot.png?raw=true "Time series data loss")


![Time series data loss](docs/data_loss.png?raw=true "Time series data loss")

![Time series reconstruction](docs/reconstructed.png?raw=true "Time series reconstruction")

![Time series forecating](docs/forecasted.png?raw=true "Time series forecating")


![Time series anomaly detection](docs/anomaly.png?raw=true "Time series anomaly detection")




# Testing ![](https://api.travis-ci.org/sarusso/Timeseria.svg?branch=master) 

Every commit on the Timeseria codebase is automatically tested with Travis-CI. [Check status on Travis](https://travis-ci.org/sarusso/Timeseria/).


# Licensing
Timeseria is licensed under the Apache License version 2.0, unless otherwise specified. See [LICENSE](https://github.com/sarusso/Timeseria/blob/master/LICENSE) for the full license text.





