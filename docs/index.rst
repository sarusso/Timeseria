Welcome to Timeseria reference documentation!
=============================================


Timeseria is an object-oriented time series processing library which aims at making it easy to manipulate time series data and to build statistical and machine learning models on top of it.

Unlike common numerical and data analysis frameworks, Timeseira does not make use of low level structures as arrays and matrices for representing data. Instead, it builds up from well defined and reusable logical units (objects) which can be easily composed together, ensuring a high level of consistency.

Because of this approach, Timeseria addresses by design all those annoying things which are often left as an implementation detail but that actually cause wasting massive amounts of time - as handling data losses, non-uniform sampling rates, differences between aggregated data and punctual observations, timezones, DST changes, and so on.

Timeseria  comes with a built-in set of common operations (resampling, aggregation, differencing etc.) and models (reconstruction, forecasting and anomaly detection) which can be easily extended with custom ones, and integrates a powerful plotting engine based on Dygraphs capable of plotting even millions of data points.

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
     utilities

|

Other resources
---------------

* :ref:`Alphabetical index <genindex>`

* :ref:`Module index <modindex>`

* `GitHub <https://github.com/sarusso/Timeseria>`_

* `License <https://github.com/sarusso/Timeseria/blob/master/LICENSE>`_
