#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='timeseria',
      version='2.0.2',
      description='An object-oriented time series processing library',
      long_description="""Timeseria is an object-oriented time series processing library which aims at making it easy to manipulate time series data and to build statistical 
and machine learning models on top of it.\n\nCheck out the GitHub project for more info: [Timeseria on GitHub](https://github.com/sarusso/Timeseria).\n\n
![Time series plot](https://raw.githubusercontent.com/sarusso/Timeseria/master/docs/altogether.png "Timeseria at work")""",
      long_description_content_type='text/markdown',
      url="https://github.com/sarusso/timeseria",
      author='Stefano Alberto Russo',
      author_email='stefano.russo@gmail.com',
      packages=['timeseria','timeseria.tests', 'timeseria.models'],
      package_data={
          'timeseria': ['static/css/*.css', 'static/js/*.js'],
          'timeseria.tests': ['test_data/csv/*.csv']
       },
      install_requires = [
                          'matplotlib >=2.1.2, <4.0.0',
                          'numpy >=1.19.5, <2.0.0',
                          'scikit-learn >=0.2.2, <2.0.0',
                          'pandas >=0.23.4, <2.0.0',
                          'chardet >=3.0.4, <4.0.0',
                          'convertdate >=2.1.2, <3.0.0',
                          'lunarcalendar >=0.0.9, <1.0.0',
                          'cython >=0.29.17, <1.0.0',
                          'requests >=2.20.0, <3.0.0',
                          'scipy >=1.5.4, <2.0.0',
                          'pyppeteer>=0.2.6, <1.0.0',
                          'fitter==1.7.0',
                          'propertime>=1.0.1, <2.0.0'
                          ],
      extras_require = {
                        'tensorflow': ['tensorflow >=2.0.0, <3.0.0'],
                        'tensorflow-gpu': ['tensorflow-gpu >=2.0.0, <3.0.0'],
                        'tensorflow-macos': ['tensorflow-macos >=2.0.0, <3.0.0'],
                        'tensorflow-aarch64': ['tensorflow-aarch64 >=2.0.0, <3.0.0'],
                        'prophet':['prophet >=1.1.1, <2.0.0'],
                        'arima': ['pmdarima >=1.8, <3.0.0', 'statsmodels >=0.14.0, <1.0.0']
                       },
      license='Apache License 2.0',
      license_files = ('LICENSE',),
    )
