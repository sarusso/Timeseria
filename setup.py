#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='timeseria',
      version='0.1.3',
      description='A time series processing library',
      url="https://github.com/sarusso/timeseria",
      author='Stefano Alberto Russo',
      author_email='stefano.russo@gmail.com',
      packages=['timeseria','timeseria.tests', 'timeseria.models'],
      package_data={
          'timeseria': ['static/css/*.css', 'static/js/*.js'],
          'timeseria.tests': ['test_data/csv/*.csv']
       },
      install_requires = [
                          'Keras >=2.1.3, <3.0.0',
                          'tensorflow >=1.15.2, <3.0.0',
                          'matplotlib >=2.1.2, <4.0.0',
                          'numpy >=1.19.5, <2.0.0',
                          'scikit-learn >=0.2.2, <2.0.0',
                          'pandas >=0.23.4, <2.0.0',
                          'chardet >=3.0.4, <4.0.0',
                          'convertdate >=2.1.2, <3.0.0',
                          'lunarcalendar >=0.0.9, <1.0.0',
                          'holidays >=0.10.3, <1.0.0',
                          'cython >=0.29.17, <1.0.0',
                          'requests >=2.20.0, <3.0.0',
                          'h5py >=2.10.0, <4.0.0',
                          'scipy >=1.5.4, <2.0.0'
                          ],
      license='Apache License 2.0',
      license_files = ('LICENSE',),
    )
