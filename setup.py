#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='timeseria',
      version='0.1.0',
      description='A time series processing library',
      author='Stefano Alberto Russo',
      author_email='stefano.russo@gmail.com',
      packages=['timeseria','timeseria.tests', 'timeseria.models'],
      package_data={
          'timeseria': ['static/css/*.css', 'static/js/*.js'],
          'timeseria.tests': ['test_data/csv/*.csv']
       },      
      license='LICENSE',
    )
