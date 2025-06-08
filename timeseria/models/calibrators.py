# -*- coding: utf-8 -*-
"""Calibration models."""

import functools
from math import sqrt, pi

from ..utils import PFloat, _import_class_from_string
from .base import Model
from ..exceptions import NotFittedError

# Setup logging
import logging
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings as default behavior
try:
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except (ImportError,AttributeError):
    pass


class Calibrator(Model):
    """A generic calibration model."""

    @staticmethod
    def probabilize_method(probabilize_method):
        """:meta private:"""
        @functools.wraps(probabilize_method)
        def do_probabilize(self, prediction, *args, **kwargs):

            # Ensure the model is fitted if it has to.
            if self._is_fit_implemented() and not self.fitted:
                raise NotFittedError()

            # Call decorate logic
            probabilized_prediction = probabilize_method(self, prediction, *args, **kwargs)

            return probabilized_prediction

        return do_probabilize

    @Model.fit_method
    def fit(self, series, caller, error_metric, *args, **kwargs):
        """Fit the calibrator model on the given series"""
        raise NotImplementedError()

    def probabilize(self, prediction, series):
        """Make the prediction originating from a given series probabilistic, according to the calibrator.

        Args:
            prediction(dict,list): the predicted data, according to the ``predict()`` method of the calibrated model.
            series(TimeSeries): the series from which the prediction originates, as context.
        """
        raise NotImplementedError()

    def predict(self, series, *args, **kwargs):
        """Disabled. Calibrators can be used only with the ``probabilize()`` method."""
        raise NotImplementedError('Calibrators detectors can be used only with the ``probabilize()`` method.') from None

    def apply(self, series, *args, **kwargs):
        """Disabled. Calibrators can be used only with the ``probabilize()`` method."""
        raise NotImplementedError('Calibrators detectors can be used only with the ``probabilize()`` method.') from None


class ErrorDistributionCalibrator(Calibrator):
    """A calibration model based on the (global) error distribution."""

    @Model.fit_method
    def fit(self, series, caller, error_metric='AE', error_dist_type='norm'):
        """Fit the calibrator, computing the global error distribution.

        Args:
            series(TimeSeries): the series to use for calibration. which to apply the predict logic.
            caller(Model): the model being calibrated.
            error_metric(str): the error metric to calibrate for.  Defaults to ``AE``. Supported values  are: ``AE``, ``APE``, and ``ALE``.
            error_dist_type(string): the distribution used to model the error. Default to 'norm'
        """

        logger.info('Fitting calibrator using {}" error metric and "{}" error distribution'.format(error_metric, error_dist_type))

        # Right now only the Normal distribution is supported
        if error_dist_type not in ['norm']:
            raise ValueError('Distribution type "{}" is not supported'.format(error_dist_type))

        # Set predicted data labels
        predicted_data_labes = []
        for data_label in series.data_labels():
            if data_label.endswith('_pred'):
                predicted_data_labes.append(data_label.replace('_pred', ''))
        if not predicted_data_labes:
            raise ValueError('The series does not contain any predictions (_pred data labels)')

        self.data['avg_error'] = {} 

        for predicted_data_label in predicted_data_labes:
            errors = []
            for item in series:
                errors.append(item.data['{}_{}'.format(predicted_data_label, error_metric)])
            self.data['avg_error'][predicted_data_label] = sum(errors)/len(errors)

    @Calibrator.probabilize_method
    def probabilize(self, prediction, series):

        for data_label in self.data['avg_error']:

            # Compute the Normal distribution standard deviation reversing the expected absolute error formula
            std_dev = self.data['avg_error'][data_label] * sqrt(pi / 2)
            prediction[data_label] = PFloat(value = prediction[data_label],
                                            dist = {'type': 'norm',
                                                    'params': {'loc': prediction[data_label],
                                                               'scale': std_dev},
                                                    'pvalue': None })

        return prediction


class CHEMCalibrator(Calibrator):
    """A calibration model based on Calibrated Heteroscedastic Error Modeling (CHEM).

    Heteroscedasticity assumes non-constant error variance, Heteroscedasticity Error Modeling aims at capturing such variance,
    and Calibrated Heteroscedasticity Error Modeling use a calibration data set to both capture and validate such variance.

    This calibrator makes use of an error predictor, which is basically a forecaster. Its role is to predict the error
    that the model being calibrated would commit in a given context (i.e. based on the forecasting window).
    """

    @classmethod
    def load(cls, path):

        # Load the calibrator
        calibrator = super().load(path)

        # Load the error predictor
        try:
            error_predictor_class = _import_class_from_string('timeseria.models.forecasters.{}'.format(calibrator.data['error_predictor_class_name']))
        except AttributeError:
            raise ValueError('Could not find error predictor "{}" in the timeseria.models.forecasters module'.format(calibrator.data['error_predictor_class_name'])) from None

        calibrator.error_predictor = error_predictor_class.load('{}/error_predictor'.format(path))

        return calibrator

    def save(self, path):

        # Save the calibrator
        super().save(path)

        # Save the error predictor
        try:
            _import_class_from_string('timeseria.models.forecasters.{}'.format(self.data['error_predictor_class_name']))
        except AttributeError:
            raise ValueError('Cannot save a calibrator using an error predictor which is not part of the timeseria.models.forecasters module (got "{}")'.format(self.data['error_predictor_class_name'])) from None
        self.error_predictor.save('{}/error_predictor'.format(path))


    @Model.fit_method
    def fit(self, series, caller, error_metric='AE', error_predictor='default', error_dist_type='norm', **kwargs):
        """Fit the calibrator using the given error predictor.

        All the parameters starting with the ``error_predictor_`` prefix are forwarded to the error predictor initialization (without the prefix), and
        all the parameters starting with the ``error_predictor_fit_`` prefix are forwarded to the error predictor ``fit()`` method (without the prefix).

        Args:
            series(TimeSeries): the series to use for calibration. which to apply the predict logic.
            caller(Model): the model being calibrated.
            error_metric(str): the error metric to calibrate for.  Defaults to ``AE``. Supported values  are: ``SE``, 
                                ``AE``, ``APE``, ``ALE``, ``E``, ``PE`` and ``LE``.
            error_predictor:(str, Forecaster): the model to use for predicting the error. Defaults to a LSTMForecaster.
            error_dist_type(string): the distribution used to model the error. Defaults to 'norm'.
            **kwargs: extra arguments to be forwarded to the error predictor.
        """

        from .forecasters import Forecaster
        if not isinstance(caller, Forecaster):
            raise TypeError('This calibrator can be used only with a Forecaster')

        # Set default error predictor values
        if error_predictor == 'default':
            if 'error_predictor_neurons' not in kwargs:
                kwargs['error_predictor_neurons'] = 16
            if 'error_predictor_features' not in kwargs:
                kwargs['error_predictor_features'] = ['values']
            from .forecasters import LSTMForecaster 
            error_predictor_class = LSTMForecaster
        else:
            error_predictor_class = error_predictor
        self.data['error_predictor_class_name'] = error_predictor_class.__name__

        logger.info('Fitting calibrator using "{}" error predictor with "{}" metric and "{}" distribution'.format(self.data['error_predictor_class_name'], error_metric, error_dist_type))

        # Decouple error predictor (init) kwargs from error predictor fit kwargs
        error_predictor_kwargs = {}
        error_predictor_fit_kwargs = {}
        for kwarg in kwargs:
            if kwarg.startswith('error_predictor_fit_'):
                error_predictor_fit_kwargs[kwarg.replace('error_predictor_fit_', '')] = kwargs[kwarg]
            elif kwarg.startswith('error_predictor_'):
                error_predictor_kwargs[kwarg.replace('error_predictor_', '')] = kwargs[kwarg]

        if not caller.window:
            raise ValueError('CHEM calibration cannot be performed on window-less models')

        # Right now only the Normal distribution is supported
        if error_dist_type not in ['norm']:
            raise ValueError('Distribution type "{}" is not supported'.format(error_dist_type))

        # Set predicted data labels
        predicted_data_labes = []
        for data_label in series.data_labels():
            if data_label.endswith('_pred'):
                predicted_data_labes.append(data_label.replace('_pred', ''))
        if not predicted_data_labes:
            raise ValueError('The series does not contain any predictions (_pred data labels)')

        # Right now only univariate time series are supported
        if len(predicted_data_labes) > 1:
            raise NotImplementedError('Calibrating on multivariate time series is not supported yet')

        # Ok, now let's try to train a model for the error.
        for data_label in predicted_data_labes:
            error_predictor = error_predictor_class(window=caller.window, **error_predictor_kwargs)
            error_predictor.fit(series,
                                source = data_label,
                                target = '{}_{}'.format(data_label, error_metric),
                                **error_predictor_fit_kwargs)

        self.error_predictor = error_predictor
        self.data['error_predictor_metric'] = error_metric
        self.data['error_predictor_dist_type'] = error_dist_type


    @Calibrator.probabilize_method
    def probabilize(self, prediction, series):

        # Predict the prediction error
        data_label = self.error_predictor.data['data_labels'][0]
        predicted_prediction_error = self.error_predictor.predict(series)['{}_{}'.format(data_label, self.data['error_predictor_metric'])]

        # Compute the Normal distribution standard deviation reversing the expected absolute error formula
        std_dev = predicted_prediction_error * sqrt(pi / 2)
        prediction[data_label] = PFloat(value = prediction[data_label],
                                        dist = {'type': self.data['error_predictor_dist_type'],
                                                'params': {'loc': prediction[data_label],
                                                           'scale': std_dev},
                                                'pvalue': None })

        return prediction

