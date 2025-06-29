# -*- coding: utf-8 -*-
"""Calibration models."""

import functools
from math import sqrt, pi
from numpy import quantile
from scipy.stats import norm

from ..utils import PFloat, IFloat, _import_class_from_string
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
    def adjust_method(adjust_method):
        """:meta private:"""
        @functools.wraps(adjust_method)
        def do_adjust(self, prediction, *args, **kwargs):

            # Ensure the model is fitted if it has to.
            if self._is_fit_implemented() and not self.fitted:
                raise NotFittedError()

            # Call decorate logic
            adjustd_prediction = adjust_method(self, prediction, *args, **kwargs)

            return adjustd_prediction

        return do_adjust

    @Model.fit_method
    def fit(self, series, model, error_metric, *args, **kwargs):
        """Fit the calibrator model on the given series"""
        raise NotImplementedError()

    def adjust(self, prediction, series=None):
        """Adjust a probabilistic prediction interval or distribution.

        Args:
            prediction(dict,list): the predicted data, according to the ``predict()`` method of the calibrated model.
            series(TimeSeries): the series from which the prediction originates, as (optional) context.
        """
        raise NotImplementedError()

    def predict(self, series, *args, **kwargs):
        """Disabled. Calibrators can be used only with the ``wrap()`` or ``adjust()`` methods."""
        raise NotImplementedError('Calibrators can be used only with the ``wrap()`` or ``adjust()`` methods.') from None

    def apply(self, series, *args, **kwargs):
        """Disabled. Calibrators can be used only with the ``adjust()`` method."""
        raise NotImplementedError('Calibrators can be used only with the ``wrap()`` or ``adjust()`` methods.') from None

    @staticmethod
    def _warn_if_probabilistic_predictions(series):

        # If the series has probabilistic predictions, issue a warning
        warned = False
        for data_label in series.data_labels():
            if warned:
                break
            if data_label.endswith('_pred') and isinstance(series[0].data[data_label], PFloat):
                logger.warning('This model is probabilistic. Calibrating it will cause to override the probabilistic predictions with the calibrated ones.')
                warned = True


class ErrorQuantileCalibrator(Calibrator):
    """A calibration model based on the error quantiles."""

    @Calibrator.fit_method
    def fit(self, series, model, verbose, error_metric='AE', alpha=0.1):
        """Fit the calibrator, computing the global error distribution.

        Args:
            series(TimeSeries): the series to use for calibration.
            model(Model): the model being calibrated.
            error_metric(str): the error metric to calibrate for.  Defaults to ``AE``.
            alpha(float): Significance level for prediction intervals. Defaults to 0.1
        """

        if error_metric not in ['AE']:
            raise ValueError('This calibrators support only the "AE" error metric')

        from .forecasters import Forecaster
        if not isinstance(model, Forecaster):
            raise TypeError('This calibrators support only forecasting models')

        logger.info('Fitting calibrator using "{}" error metric and "{}" alpha'.format(error_metric, alpha))
        self.data['alpha'] = alpha

        # Evaluate and get the evaluation series (with all the errors)
        evaluation_series = model.evaluate(series, evaluation_series_error_metrics=[error_metric], return_evaluation_series=True, verbose=verbose)['series']
        self._warn_if_probabilistic_predictions(evaluation_series)

        # Set predicted data labels
        predicted_data_labes = []
        for data_label in evaluation_series.data_labels():
            if data_label.endswith('_pred'):
                predicted_data_labes.append(data_label.replace('_pred', ''))
        if not predicted_data_labes:
            raise ValueError('The evaluation_series does not contain any predictions (_pred data labels)?!')

        self.data['avg_error'] = {}
        self.data['q_hat'] = {}

        for predicted_data_label in predicted_data_labes:
            errors = []
            for item in evaluation_series:
                errors.append(item.data['{}_{}'.format(predicted_data_label, error_metric)])
            self.data['avg_error'][predicted_data_label] = sum(errors)/len(errors)
            self.data['q_hat'][predicted_data_label] = quantile(errors, 1-alpha)


    @Calibrator.adjust_method
    def adjust(self, prediction, series, _as_pfloat='auto'):

        for data_label in self.data['avg_error']:

            if _as_pfloat:

                # Compute z-score for two-sided interval
                z = norm.ppf(1 - self.data['alpha'] / 2)

                # Convert to Gaussian std_dev (sigma)
                std_dev = self.data['q_hat'][data_label] / z

                # Generate the Normal distribution
                prediction[data_label] = PFloat(value = prediction[data_label],
                                                dist = {'type': 'norm',
                                                        'params': {'loc': prediction[data_label],
                                                                   'scale': std_dev},
                                                        'pvalue': None })
            else:
                prediction[data_label] = IFloat(value = prediction[data_label],
                                                lower = prediction[data_label] - self.data['q_hat'][data_label],
                                                upper = prediction[data_label] + self.data['q_hat'][data_label])

        return prediction


class ErrorDistributionCalibrator(Calibrator):
    """A calibration model based on the error distribution."""

    @Calibrator.fit_method
    def fit(self, series, model, verbose, error_metric='AE', error_distribution='norm'):
        """Fit the calibrator, computing the global error distribution.

        Args:
            series(TimeSeries): the series to use for calibration.
            model(Model): the model being calibrated.
            error_metric(str): the error metric to calibrate for.
            error_distribution(string): the distribution used to model the error. Default to 'norm'
        """

        if error_metric not in ['AE']:
            raise ValueError('This calibrators support only the "AE" error metric')

        from .forecasters import Forecaster
        if not isinstance(model, Forecaster):
            raise TypeError('This calibrators support only forecasting models')

        logger.info('Fitting calibrator using "{}" error metric and "{}" error distribution'.format(error_metric, error_distribution))

        # Evaluate and get the evaluation series (with all the errors)
        evaluation_series = model.evaluate(series, evaluation_series_error_metrics=[error_metric], return_evaluation_series=True, verbose=verbose)['series']
        self._warn_if_probabilistic_predictions(evaluation_series)

        # Right now only the Normal distribution is supported
        if error_distribution not in ['norm']:
            raise ValueError('Distribution type "{}" is not supported'.format(error_distribution))

        # Set predicted data labels
        predicted_data_labes = []
        for data_label in evaluation_series.data_labels():
            if data_label.endswith('_pred'):
                predicted_data_labes.append(data_label.replace('_pred', ''))
        if not predicted_data_labes:
            raise ValueError('The evaluation_series does not contain any predictions (_pred data labels)?!')

        self.data['avg_error'] = {} 

        for predicted_data_label in predicted_data_labes:
            errors = []
            for item in evaluation_series:
                errors.append(item.data['{}_{}'.format(predicted_data_label, error_metric)])
            self.data['avg_error'][predicted_data_label] = sum(errors)/len(errors)

    @Calibrator.adjust_method
    def adjust(self, prediction, series):

        for data_label in self.data['avg_error']:

            # Compute the Normal distribution standard deviation reversing the expected absolute error formula
            std_dev = self.data['avg_error'][data_label] * sqrt(pi / 2)
            prediction[data_label] = PFloat(value = prediction[data_label],
                                            dist = {'type': 'norm',
                                                    'params': {'loc': prediction[data_label],
                                                               'scale': std_dev},
                                                    'pvalue': None })

        return prediction


class ErrorPredictionCalibrator(Calibrator):
    """A calibration model based on the error prediction.

    This calibrator makes use of an error predictor which takes as input the same
    input of the model being calibrated (i.e. the forecasting window), thus allowing
    to adapt the uncertainty estimates to the local context.

    Uncertainty is then expressed with a Normal distribution computed by reversing
    the expected absolute error formula with respect to the standard deviation.
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


    @Calibrator.fit_method
    def fit(self, series, model, verbose, error_metric='AE', error_predictor='default', **kwargs):
        """Fit the calibrator using the given error predictor.

        All the parameters starting with the ``error_predictor_`` prefix are forwarded to the error predictor initialization (without the prefix), and
        all the parameters starting with the ``error_predictor_fit_`` prefix are forwarded to the error predictor ``fit()`` method (without the prefix).

        Args:
            series(TimeSeries): the series to use for calibration.
            model(Model): the model being calibrated.
            error_metric(str): the error metric to calibrate for.  Defaults to ``AE``. Supported values  are: ``SE``, 
                                ``AE``, ``APE``, ``ALE``, ``E``, ``PE`` and ``LE``.
            error_predictor:(str, Model): the model used to predict the error. Besides specific models, any
                                          Forecaster will work. The default corresponds to a LSTMForecaster.
            **kwargs: extra arguments to be forwarded to the error predictor as introduced above.
        """

        if error_metric not in ['AE']:
            raise ValueError('This calibrators support only the "AE" error metric')

        from .forecasters import Forecaster
        if not isinstance(model, Forecaster):
            raise TypeError('This calibrators support only forecasting models')

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

        logger.info('Fitting calibrator using "{}" error predictor with "{}" metric'.format(self.data['error_predictor_class_name'], error_metric))

        # Evaluate and get the evaluation series (with all the errors)
        evaluation_series = model.evaluate(series, evaluation_series_error_metrics=[error_metric], return_evaluation_series=True, verbose=verbose)['series']
        self._warn_if_probabilistic_predictions(evaluation_series)

        # Decouple error predictor (init) kwargs from error predictor fit kwargs
        error_predictor_kwargs = {}
        error_predictor_fit_kwargs = {}
        for kwarg in kwargs:
            if kwarg.startswith('error_predictor_fit_'):
                error_predictor_fit_kwargs[kwarg.replace('error_predictor_fit_', '')] = kwargs[kwarg]
            elif kwarg.startswith('error_predictor_'):
                error_predictor_kwargs[kwarg.replace('error_predictor_', '')] = kwargs[kwarg]

        if not model.window:
            raise ValueError('CHEM calibration cannot be performed on window-less models')

        # Set predicted data labels
        predicted_data_labes = []
        for data_label in evaluation_series.data_labels():
            if data_label.endswith('_pred'):
                predicted_data_labes.append(data_label.replace('_pred', ''))
        if not predicted_data_labes:
            raise ValueError('The series does not contain any predictions (_pred data labels)')

        # Right now only univariate time series are supported
        if len(predicted_data_labes) > 1:
            raise NotImplementedError('Calibrating on multivariate time series is not supported yet')

        # Ok, now let's try to train a model for the error.
        for data_label in predicted_data_labes:
            error_predictor = error_predictor_class(window=model.window, **error_predictor_kwargs)
            error_predictor.fit(evaluation_series,
                                source = data_label,
                                target = '{}_{}'.format(data_label, error_metric),
                                **error_predictor_fit_kwargs)

        self.error_predictor = error_predictor
        self.data['error_metric'] = error_metric


    @Calibrator.adjust_method
    def adjust(self, prediction, series):

        # Predict the prediction error
        data_label = self.error_predictor.data['data_labels'][0]
        predicted_prediction_error = self.error_predictor.predict(series)['{}_{}'.format(data_label, self.data['error_metric'])]

        # Compute the Normal distribution standard deviation reversing the expected absolute error formula
        std_dev = predicted_prediction_error * sqrt(pi / 2)
        prediction[data_label] = PFloat(value = prediction[data_label],
                                        dist = {'type': 'norm',
                                                'params': {'loc': prediction[data_label],
                                                           'scale': std_dev},
                                                'pvalue': None })

        return prediction
