Welcome to Timeseria reference documentation!
=============================================


Timeseria is an object-oriented time series processing library which aims at making it easy to manipulate time series data and to build statistical and machine learning models on top of it.

It provides a built-in set of common operations (resampling, aggregating, differencing etc.) as well as 
models (reconstruction, forecasting and anomaly detection), and both custom operations and models can be easily plugged in.

Timeseria also tries to address by design all those annoying things which are often left as an implementation detail but that actually cause wasting massive amounts of time - as handling data losses, non-uniform sampling rates, differences between time-slotted data and punctual observations, variable time units, timezones, DST changes and so on.

This is the refeerence documentations, and it is quite essential. To get started more gently, you can have a look at the
`quickstart <https://sarusso.github.io/Timeseria/Quickstart.html>`_ or at the `welcome <https://sarusso.github.io/Timeseria/Welcome.html>`_ notebooks.

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
    
     datastructures
     units
     transformations 
     storages
     time
     
     models.base
     models.forecasters
     models.reconstructors
     models.anomaly_detectors
     
     operations
     exceptions
     plots
     utilities

|

Other resources
---------------

* :ref:`Alphabetical index <genindex>`

* :ref:`Module index <modindex>`

* `GitHub <https://github.com/sarusso/Timeseria>`_

* `Travis <https://travis-ci.org/sarusso/Timeseria>`_

* `License <https://github.com/sarusso/Timeseria/blob/master/LICENSE>`_
