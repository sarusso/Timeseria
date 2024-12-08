Welcome to Timeseria reference documentation!
=============================================


Timeseria is an object-oriented time series processing library which aims at making it easy to manipulate time series data and to build statistical and machine learning models on top of it.

Unlike common numerical and data analysis frameworks, Timeseria does not make use of low level data structures as arrays and matrices to represent time series data. Instead, it builds up from well defined and reusable logical units (objects) which can be easily combined together in order to ensure an high level of consistency.

Thanks to this approach, Timeseria can address by design several issues often left as an implementation detail but that can actually cause significant challenges - as handling data losses, non-uniform sampling rates, differences between aggregated data and punctual observations, time zones, DST changes, and so on.

Timeseria  comes with a built-in set of common operations (such as resampling, aggregations, moving averages, derivatives etc.) along with extensible models for reconstruction, forecasting and anomaly detection; and it integrates a powerful plotting engine capable of handling even millions of data points.

This is the refeerence documentations, and it is quite essential. To get started more gently, you can have a look at the
`quickstart <https://github.com/sarusso/Timeseria-notebooks/blob/master/notebooks/Quickstart.ipynb>`_ or at the `welcome <https://github.com/sarusso/Timeseria-notebooks/blob/master/notebooks/Welcome.ipynb>`_ notebooks.

Examples are provided in the `Timeseria-notebooks <https://github.com/sarusso/Timeseria-notebooks>`_ repository, and a Docker image ready to be played with is available on `Docker Hub <https://hub.docker.com/r/sarusso/timeseria>`_.


|

Main modules and submodules
---------------------------

.. automodule:: timeseria
    :members:
    :inherited-members:
    :undoc-members:


.. autosummary::
     :toctree:

     units
     datastructures
     interpolators
     operations
     transformations

     models.base
     models.forecasters
     models.reconstructors
     models.anomaly_detectors

     storages
     plots
     exceptions
     utils
     logger

|

Other resources
---------------

* :ref:`Alphabetical index <genindex>`

* :ref:`Module index <modindex>`

* `GitHub <https://github.com/sarusso/Timeseria>`_

* `License <https://github.com/sarusso/Timeseria/blob/master/LICENSE>`_
