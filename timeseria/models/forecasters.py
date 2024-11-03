# -*- coding: utf-8 -*-
"""Forecasting models."""

import copy
import pickle
import shutil
from pandas import DataFrame
from numpy import array
from math import sqrt, log
from propertime.utils import now_s, dt_from_s, s_from_dt
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

from ..datastructures import DataTimeSlot, TimePoint, DataTimePoint, Slot, Point, TimeSeries
from ..exceptions import NonContiguityError, NotFittedError
from ..utils import detect_periodicity, _get_periodicity_index, ensure_reproducibility
from ..utils import mean_squared_error
from ..utils import mean_absolute_error, max_absolute_error
from ..utils import mean_absolute_percentage_error, max_absolute_percentage_error
from ..utils import mean_absolute_log_error, max_absolute_log_error
from .base import Model, _ProphetModel, _ARIMAModel, _KerasModel

# Setup logging
import logging
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings as default behavior
try:
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except (ImportError,AttributeError):
    pass

# Also suppress absl warnings as default behavior
# https://stackoverflow.com/questions/65697623/tensorflow-warning-found-untraced-functions-such-as-lstm-cell-6-layer-call-and
# https://github.com/tensorflow/tensorflow/issues/47554
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except ImportError:
    pass


#======================
#  Generic Forecaster
#======================

class Forecaster(Model):
    """A generic forecasting model. Besides the ``predict()`` and  ``apply()`` methods, it also has a ``forecast()``
    method which, in case of nested data structures (i.e. DataPoint or DataSlot) allows to get the full forecasted points
    or slots instead of just the raw, inner data values returned by the ``predict()``. In case of plain data structures
    (e.g. a list), the ``forecast()`` method is instead equivalent to the ``predict()``.

    Series resolution and data labels consistency are enforced between all methods and save/load operations.

    Args:
        path (str): a path from which to load a saved model. Will override all other init settings.
    """

    window = None

    def _get_actual_value(self, series, i, data_label):
        actual = series[i].data[data_label]
        return actual

    def _get_predicted_value(self, series, i, data_label, with_context, **kwargs):

        if with_context:
            prediction = self.predict(series.view(from_i=0, to_i=i), steps=1, context_data=series[i].data, **kwargs)
        else:
            prediction = self.predict(series.view(from_i=0, to_i=i), steps=1, **kwargs)

        # Handle list of dicts or dict of lists (of which we have only one value here)
        #{'value': [0.2019341593004146, 0.29462641146884005]}

        if isinstance(prediction, list):
            predicted = prediction[0][data_label]
        elif isinstance(prediction, dict):
            if isinstance(prediction[data_label], list):
                predicted = prediction[data_label][0]
            else:
                predicted = prediction[data_label]
        else:
            raise TypeError('Don\'t know how to handle a prediction with of type "{}"'.format(prediction.__class__.__name__))

        return predicted

    def forecast(self, series, steps=1, context_data=None, **kwargs):
        """Forecast n steps-ahead data points or slots"""

        # Check series
        if not isinstance(series, TimeSeries):
            raise NotImplementedError('Models work only with TimeSeries data for now (got "{}")'.format(series.__class__.__name__))

        if 'target_data_labels' in self.data and self.data['target_data_labels'] and set(self.data['target_data_labels']) != set(series.data_labels()):
            if not context_data:
                raise ValueError('Forecasting with a forecaster fit to specific target data labels requires to give context data')

        if context_data:
            predicted_data = self.predict(series, steps=steps, context_data=context_data, **kwargs)
        else:
            predicted_data = self.predict(series, steps=steps, **kwargs)

        # Handle list of predictions or single prediction
        forecast_start_item = series[-1]
        if isinstance(predicted_data,list):
            if context_data:
                raise NotImplementedError('Context with multi step-ahead predictions is not yet implemented')
            forecast = []
            last_item = forecast_start_item
            for data in predicted_data:

                if isinstance(series[0], Slot):
                    forecast.append(DataTimeSlot(start = last_item.end,
                                                 unit  = series.resolution,
                                                 data_loss = None,
                                                 #tz = series.tz,
                                                 data = data))
                else:
                    forecast.append(DataTimePoint(t = last_item.t + series.resolution,
                                                  tz = series.tz,
                                                  data = data))
                last_item = forecast[-1]
        else:
            if context_data:
                predicted_data.update(context_data)
            if isinstance(series[0], Slot):
                forecast = DataTimeSlot(start = forecast_start_item.end,
                                        unit  = series.resolution,
                                        data_loss = None,
                                        #tz = series.tz,
                                        data  = predicted_data)
            else:
                forecast = DataTimePoint(t = forecast_start_item.t + self.data['resolution'],
                                         tz = series.tz,
                                         data  = predicted_data)

        return forecast

    @Model.apply_method
    def apply(self, series, steps=1, inplace=False, context_data=None, **kwargs):
        """Apply the forecast on the given series for n steps-ahead."""

        if 'target_data_labels' in self.data and self.data['target_data_labels'] and set(self.data['target_data_labels']) != set(series.data_labels()):
            if not context_data:
                raise ValueError('Applying a forecaster fit to target specific data labels requires to give context data (target_data_labels={})'.format(self.data['target_data_labels']))

        if not inplace:
            series = series.duplicate()

        input_series_len = len(series)

        # Add the forecast index
        for item in series:
            item.data_indexes['forecast'] = 0

        # Call model forecasting logic
        try:
            if context_data:
                forecast_model_results = self.forecast(series, steps=steps, context_data=context_data, **kwargs)
            else:
                forecast_model_results = self.forecast(series, steps=steps, **kwargs)
            if not isinstance(forecast_model_results, list):
                # Add forecasted index and append
                forecast_model_results.data_indexes['forecast'] = 1
                series.append(forecast_model_results)
            else:
                for item in forecast_model_results:
                    # Add forecasted index for each item and append
                    item.data_indexes['forecast'] = 1
                    series.append(item)

        except NotImplementedError:

            for _ in range(steps):

                # Call the forecast only on the last point
                if context_data:
                    forecast_model_results = self.forecast(series, steps=1, context_data=context_data, **kwargs)
                else:
                    forecast_model_results = self.forecast(series, steps=1, **kwargs)

                # Add forecasted index
                forecast_model_results.data_indexes['forecast'] = 1

                # Add the forecast to the forecasts time series
                series.append(forecast_model_results)

        # Do we have missing forecasts?
        if input_series_len + steps != len(series):
            raise ValueError('There are missing forecasts. If your model does not support multi-step forecasting, raise a NotImplementedError if steps>1 and Timeseria will handle it for you.')

        if not inplace:
            return series
        else:
            return None

    @Model.evaluate_method
    def evaluate(self, series, steps=1, error_metrics=['RMSE', 'MAE'], return_evaluation_series=False, plot_evaluation_series=False,
                 evaluation_series_error_metrics='auto', plot_error_distribution=False, error_distribution_metrics='auto', verbose=False, **kwargs):
        """Evaluate the forecaster on a series.

        Args:
            steps (int): how many steps-ahead to evaluate the forecaster on.
            error_metrics(list): the (aggregated) error metrics to use for the evaluation. Defaults to ``RMSE`` and ``MAE``. Supported
                                 values (as list)  are: ``MSE``, ``RMSE``, ``MAE``, ``MaxAE``, ``MAPE``, ``MaxAPE``, ``MALE`` and  ``MaxALE``.
            return_evaluation_series(bool): if to return the series used for the evaluation, with the  the predicted values and the punctual errors.
            plot_evaluation_series(bool): if to plot the series used for the evaluation, with the  the predicted values and the punctual errors.
            evaluation_series_error_metrics(list): the (punctual) error metrics to be included in the evaluation series. Defaults to ``auto``.
                                                   Supported values (as list) are: ``SE``, ``AE``, ``APE``, ``ALE``, ``E``, ``PE`` and ``LE``, or ``None``.
            plot_error_distribution(bool): if to plot the error distribution.
            error_distribution_metrics(list): the (punctual) error metrics to be used for generating the error distribution(s). Defaults
                                               to ``auto``. Supported values (as list) are: ``SE``, ``AE``, ``APE``, ``ALE``, ``E``, ``PE`` and ``LE``.
            verbose(bool): if to print the evaluation progress (one dot = 10% done).
        """

        if not series:
            raise ValueError('Cannot evaluate on an empty series')

        if steps > 1:
            raise NotImplementedError('Evaluating a forecaster on more than one step ahead forecasts is not yet implemented')

        try:
            context_data_labels = self.data['context_data_labels']
        except KeyError:
            context_data_labels = None

        try:
            target_data_labels = self.data['target_data_labels']
        except KeyError:
            target_data_labels = None

        # Handle (aggregated) error metrics
        for error_metric in error_metrics:
            if error_metric not in ['MSE', 'RMSE', 'MAE', 'MaxAE', 'MAPE', 'MaxAPE', 'MALE', 'MaxALE']:
                raise ValueError('The error metric "{}" is not supported'.format(error_metric))

        # Handle evaluation series error metrics
        if evaluation_series_error_metrics is None:
            evaluation_series_error_metrics = []

        if evaluation_series_error_metrics != 'auto':
            for error_metric in evaluation_series_error_metrics:
                if error_metric not in ['SE', 'AE', 'APE', 'ALE', 'E', 'PE', 'LE']:
                    raise ValueError('The series error metric "{}" is not supported'.format(error_metric))
        else:
            evaluation_series_error_metrics = []
            for error_metric in error_metrics:
                if 'MSE' in error_metrics:
                    evaluation_series_error_metrics.append('SE')
                if 'MAE' in error_metrics:
                    evaluation_series_error_metrics.append('AE')
                if 'MAPE' in error_metrics:
                    evaluation_series_error_metrics.append('APE')
                if 'MALE' in error_metrics:
                    evaluation_series_error_metrics.append('ALE')
                if 'MaxAE' in error_metrics:
                    evaluation_series_error_metrics.append('AE')
                if 'MaxAPE' in error_metrics:
                    evaluation_series_error_metrics.append('APE')
                if 'MaxALE' in error_metrics:
                    evaluation_series_error_metrics.append('ALE')
                evaluation_series_error_metrics = list(set(evaluation_series_error_metrics))

        # Handle error distribution metrics
        if error_distribution_metrics is None:
            error_distribution_metrics = []

        if error_distribution_metrics != 'auto':
            for error_metric in error_distribution_metrics:
                if error_metric not in ['AE', 'APE', 'ALE', 'E', 'PE', 'LE']:
                    raise ValueError('The error distribution metric "{}" is not supported'.format(error_metric))
        else:
            error_distribution_metrics = []
            for error_metric in error_metrics:
                if 'MSE' in error_metrics:
                    error_distribution_metrics.append('SE')
                if 'MAE' in error_metrics:
                    error_distribution_metrics.append('AE')
                if 'MAPE' in error_metrics:
                    error_distribution_metrics.append('APE')
                if 'MALE' in error_metrics:
                    error_distribution_metrics.append('ALE')
                if 'MaxE' in error_metrics:
                    error_distribution_metrics.append('AE')
                if 'MaxPE' in error_metrics:
                    error_distribution_metrics.append('APE')
                if 'MaxLE' in error_metrics:
                    error_distribution_metrics.append('ALE')
                error_distribution_metrics = list(set(error_distribution_metrics))

        # Handle prediction series
        if plot_evaluation_series or return_evaluation_series:
            generate_evaluation_series = True
        else:
            generate_evaluation_series = False

        if generate_evaluation_series:

            if steps != 1:
                raise ValueError('Returning the evaluation series is only supported with single step ahead forecasts')

            evaluation_series = series.duplicate()
            if self.data['window']:
                evaluation_series = evaluation_series[self.data['window']:]

        # Support vars
        results = {}

        # Support vars
        actual_values = {}
        predicted_values = {}

        try:
            if not self.data['window']:
                start_i = 0
            else:
                start_i = self.data['window']
        except (KeyError, AttributeError):
            start_i = 0

        # Start evaluating
        progress_step = len(series)/10

        evaluate_data_labels = series.data_labels() if not target_data_labels else target_data_labels

        for data_label in evaluate_data_labels:

            actual_values[data_label] = []
            predicted_values[data_label] = []

            if verbose:
                print('Evaluating for "{}": '.format(data_label), end='')

            for i, _ in enumerate(series):
                if verbose:
                    if int(i%progress_step) == 0:
                        print('.', end='')

                # Skip before the window
                if i <  start_i:
                    continue

                # Predict
                actual_values[data_label].append(self._get_actual_value(series, i, data_label))
                predicted_values[data_label].append(self._get_predicted_value(series, i, data_label, with_context= True if context_data_labels else False, **kwargs))

            if verbose:
                print('')

        for data_label in evaluate_data_labels:

            # Add the prediction to the prediction series if requested
            if generate_evaluation_series:
                for i in range(len(actual_values[data_label])):
                    evaluation_series[i].data['{}_pred'.format(data_label)] = predicted_values[data_label][i]

            # Compute the error metrics
            if 'MSE' in error_metrics:
                results['{}_MSE'.format(data_label)] = mean_squared_error(actual_values[data_label], predicted_values[data_label])

            if 'RMSE' in error_metrics:
                results['{}_RMSE'.format(data_label)] = sqrt(mean_squared_error(actual_values[data_label], predicted_values[data_label]))

            if 'MAE' in error_metrics:
                results['{}_MAE'.format(data_label)] = mean_absolute_error(actual_values[data_label], predicted_values[data_label])

            if 'MaxAE' in error_metrics:
                results['{}_MaxAE'.format(data_label)] = max_absolute_error(actual_values[data_label], predicted_values[data_label])

            if 'MAPE' in error_metrics:
                results['{}_MAPE'.format(data_label)] = mean_absolute_percentage_error(actual_values[data_label], predicted_values[data_label])

            if 'MaxAPE' in error_metrics:
                results['{}_MaxAPE'.format(data_label)] = max_absolute_percentage_error(actual_values[data_label], predicted_values[data_label])

            if 'MALE' in error_metrics:
                results['{}_MALE'.format(data_label)] = mean_absolute_log_error(actual_values[data_label], predicted_values[data_label])

            if 'MaxALE' in error_metrics:
                results['{}_MaxALE'.format(data_label,)] = max_absolute_log_error(actual_values[data_label], predicted_values[data_label])

            # Compute evaluation series and distribution error metrics if requested
            if generate_evaluation_series and 'SE' in evaluation_series_error_metrics:
                for i in range(len(actual_values[data_label])):
                    evaluation_series[i].data['{}_SE'.format(data_label)] = pow((actual_values[data_label][i] - predicted_values[data_label][i]),2)
            if plot_error_distribution and 'SE' in error_distribution_metrics:
                errors = []
                for i in range(len(actual_values[data_label])):
                    errors.append(pow((actual_values[data_label][i] - predicted_values[data_label][i]),2))
                plt.hist(errors, bins=100, density=True, alpha=1, color='steelblue', label='Error distribution for "{}"'.format(data_label))
                #plt.legend(loc="upper right")
                plt.grid()
                plt.title('SE distribution for "{}"'.format(data_label))
                plt.show()

            if generate_evaluation_series and 'AE' in evaluation_series_error_metrics:
                for i in range(len(actual_values[data_label])):
                    evaluation_series[i].data['{}_AE'.format(data_label)] = abs(actual_values[data_label][i] - predicted_values[data_label][i])
            if plot_error_distribution and 'AE' in error_distribution_metrics:
                errors = []
                for i in range(len(actual_values[data_label])):
                    errors.append(abs(actual_values[data_label][i] - predicted_values[data_label][i]))
                plt.hist(errors, bins=100, density=True, alpha=1, color='steelblue', label='Error distribution for "{}"'.format(data_label))
                #plt.legend(loc="upper right")
                plt.grid()
                plt.title('AE distribution for "{}"'.format(data_label))
                plt.show()

            if generate_evaluation_series and 'APE' in evaluation_series_error_metrics:
                for i in range(len(actual_values[data_label])):
                    evaluation_series[i].data['{}_APE'.format(data_label)] = abs((actual_values[data_label][i] - predicted_values[data_label][i])/actual_values[data_label][i])
            if plot_error_distribution and 'APE' in error_distribution_metrics:
                errors = []
                for i in range(len(actual_values[data_label])):
                    errors.append(abs((actual_values[data_label][i] - predicted_values[data_label][i])/actual_values[data_label][i]))
                plt.hist(errors, bins=100, density=True, alpha=1, color='steelblue', label='Error distribution for "{}"'.format(data_label))
                #plt.legend(loc="upper right")
                plt.grid()
                plt.title('APE distribution for "{}"'.format(data_label))
                plt.show()

            if generate_evaluation_series and 'ALE' in evaluation_series_error_metrics:
                for i in range(len(actual_values[data_label])):
                    evaluation_series[i].data['{}_ALE'.format(data_label)] = abs(log(actual_values[data_label][i]/predicted_values[data_label][i]))
            if plot_error_distribution and 'ALE' in error_distribution_metrics:
                errors = []
                for i in range(len(actual_values[data_label])):
                    errors.append(abs(log(actual_values[data_label][i]/predicted_values[data_label][i])))
                plt.hist(errors, bins=100, density=True, alpha=1, color='steelblue', label='Error distribution for "{}"'.format(data_label))
                #plt.legend(loc="upper right")
                plt.grid()
                plt.title('ALE distribution for "{}"'.format(data_label))
                plt.show()

            if generate_evaluation_series and 'E' in evaluation_series_error_metrics:
                for i in range(len(actual_values[data_label])):
                    evaluation_series[i].data['{}_E'.format(data_label)] = actual_values[data_label][i] - predicted_values[data_label][i]
            if plot_error_distribution and 'E' in error_distribution_metrics:
                errors = []
                for i in range(len(actual_values[data_label])):
                    errors.append(actual_values[data_label][i] - predicted_values[data_label][i])
                plt.hist(errors, bins=100, density=True, alpha=1, color='steelblue', label='Error distribution for "{}"'.format(data_label))
                #plt.legend(loc="upper right")
                plt.grid()
                plt.title('E distribution for "{}"'.format(data_label))
                plt.show()

            if generate_evaluation_series and 'PE' in evaluation_series_error_metrics:
                for i in range(len(actual_values[data_label])):
                    evaluation_series[i].data['{}_PE'.format(data_label)] = (actual_values[data_label][i] - predicted_values[data_label][i])/actual_values[data_label][i]
            if plot_error_distribution and 'PE' in error_distribution_metrics:
                errors = []
                for i in range(len(actual_values[data_label])):
                    errors.append((actual_values[data_label][i] - predicted_values[data_label][i])/actual_values[data_label][i])
                plt.hist(errors, bins=100, density=True, alpha=1, color='steelblue', label='Error distribution for "{}"'.format(data_label))
                #plt.legend(loc="upper right")
                plt.grid()
                plt.title('PE distribution for "{}"'.format(data_label))
                plt.show()

            if generate_evaluation_series and 'LE' in evaluation_series_error_metrics:
                for i in range(len(actual_values[data_label])):
                    evaluation_series[i].data['{}_LE'.format(data_label)] = log(actual_values[data_label][i]/predicted_values[data_label][i])
            if plot_error_distribution and 'LE' in error_distribution_metrics:
                errors = []
                for i in range(len(actual_values[data_label])):
                    errors.append(log(actual_values[data_label][i]/predicted_values[data_label][i]))
                plt.hist(errors, bins=100, density=True, alpha=1, color='steelblue', label='Error distribution for "{}"'.format(data_label))
                #plt.legend(loc="upper right")
                plt.grid()
                plt.title('LE distribution for "{}"'.format(data_label))
                plt.show()

        # Plot or return prediction series if required
        if plot_evaluation_series:
            evaluation_series.plot()
        if return_evaluation_series:
            results['series'] = evaluation_series

        # Return evaluation results, if any
        if results:
            return results


#=========================
#  P. Average Forecaster
#=========================

class PeriodicAverageForecaster(Forecaster):
    """A forecasting model based on periodic averages, offsetted with respect to a given window.

    Args:
        path (str): a path from which to load a saved model. Will override all other init settings.
        window (int): the window length. If set to ``auto``, then it will be automatically handled based on the time series periodicity.
    """

    @property
    def window(self):
        return self.data['window']

    def __init__(self, window='auto'):

        # Set window
        if window != 'auto':
            try:
                int(window)
            except:
                raise ValueError('Got a non-integer window ({})'.format(window))
        self._window = window

        # Call parent init
        super(PeriodicAverageForecaster, self).__init__()

    @classmethod
    def load(cls, path):
        model = super().load(path)
        # Convert the average dict keys back to integers
        for data_label in model.data['offsets_averages']:
            model.data['offsets_averages'][data_label] = {int(key):value for key, value in model.data['offsets_averages'][data_label].items()}
        return model

    @Forecaster.fit_method
    def fit(self, series, periodicity='auto', dst_affected=False, data_loss_limit=1.0, verbose=False):
        """Fit the model on a series.

        Args:
            periodicity(int): the periodicty of the series. If set to ``auto`` then it will be automatically detected using a FFT.
            dst_affected(bool): if the model should take into account DST effects.
            data_loss_limit(float): discard from the fit elements with a data loss greater than or equal to this limit.
        """

        self.data['periodicities']  = {}
        self.data['windows'] = {}
        self.data['offsets_averages'] = {}
        self.data['dst_affected'] = dst_affected

        for data_label in series.data_labels():

            processed = 0

            # Set or detect periodicity
            if periodicity == 'auto':
                periodicity =  detect_periodicity(series[data_label])
                logger.info('Detected periodicity for "%s": %sx %s', data_label, periodicity, series.resolution)

            self.data['periodicities'][data_label]  = periodicity

            # Set window
            if self._window != 'auto':
                self.data['windows'][data_label] = self._window
            else:
                logger.info('Using a window of "{}" for "{}"'.format(periodicity, data_label))
                self.data['windows'][data_label] = periodicity

            offsets_sums = {}
            offsets_totals = {}

            for i, item in enumerate(series):
                if data_loss_limit is not None and 'data_loss' in item.data_indexes and item.data_indexes['data_loss'] >= data_loss_limit:
                    continue

                if i <= self.data['windows'][data_label]:
                    continue

                periodicity_index = _get_periodicity_index(item, series.resolution, periodicity, dst_affected)

                # Compute the average value of the window (if not of zero elements)
                if self.data['windows'][data_label] != 0:
                    sum_total = 0
                    for j in range(self.data['windows'][data_label]):
                        series_index = i - j
                        sum_total += series[series_index].data[data_label]
                    window_avg = sum_total/self.data['windows'][data_label]
                else:
                    window_avg = 0

                # Compute the offset w.r.t. the window average value. If the window has zero elements, this
                # will just be the value of the item itself and the model will use pure periodic averages.
                offset = item.data[data_label] - window_avg

                # Add to the offsets sum
                if periodicity_index not in offsets_sums:
                    offsets_sums[periodicity_index] = offset
                else:
                    offsets_sums[periodicity_index] += offset

                # Increase the offsets total
                if periodicity_index not in offsets_totals:
                    offsets_totals[periodicity_index] = 1
                else:
                    offsets_totals[periodicity_index] += 1

                # Increase the processed count
                processed += 1

            if not processed:
                raise ValueError('Too much data loss (not a single element below the limit), cannot fit!')

            # Compute the periodic average offset for each periodicity index
            offsets_averages = {}
            for periodicity_index in offsets_sums:
                offsets_averages[periodicity_index] = offsets_sums[periodicity_index]/offsets_totals[periodicity_index]
            self.data['offsets_averages'][data_label] = offsets_averages

        # Store model window as the max of the single windows
        self.data['window'] = max([self.data['windows'][data_label] for data_label in self.data['windows']])


    @Forecaster.predict_method
    def predict(self, series, steps=1, probabilistic=False):

        if probabilistic and not self.data['probabilistic']:
            raise ValueError('This model was not fitted to allow for probabilistic predictions')

        if len(series) < self.data['window']:
            raise ValueError('The series length ({}) is shorter than the model window ({})'.format(len(series), self.data['window']))

        # Get forecast start item
        predict_start_item = series[-1]

        # Support vars
        predict_timestamps = []
        predict_data = []

        # Perform the forecast
        for i in range(steps):
            step = i + 1

            # Set forecast timestamp
            if isinstance(series[0], Slot):
                try:
                    predict_timestamp = predict_timestamps[-1] + series.resolution
                    predict_timestamps.append(predict_timestamp)
                except IndexError:
                    predict_timestamp = predict_start_item.end
                    predict_timestamps.append(predict_timestamp)

            else:
                predict_timestamp = TimePoint(t = predict_start_item.t + (series.resolution.as_seconds()*step), tz = predict_start_item.tz )

            # Perform the forecast for each data label
            this_step_predict_data = {}
            for data_label in self.data['data_labels']:

                # Compute the average value of the window (if not of zero elements)
                if self.data['windows'][data_label] != 0:
                    window_sum = 0
                    for j in range(self.data['windows'][data_label]):
                        series_index = len(series) - 1 - self.data['windows'][data_label] + j
                        window_sum += series[series_index].data[data_label]
                    window_avg = window_sum/self.data['windows'][data_label]
                else:
                    window_avg = 0

                # Get the periodicity index
                periodicity_index = _get_periodicity_index(predict_timestamp,
                                                           series.resolution,
                                                           self.data['periodicities'][data_label],
                                                           dst_affected=self.data['dst_affected'])

                # Apply the periodic average offset on the window average to get the prediction
                this_step_predict_data[data_label] = window_avg + self.data['offsets_averages'][data_label][periodicity_index]

            predict_data.append(this_step_predict_data)

        # Return
        return predict_data

    def _plot_averages(self, series, **kwargs):
        averages_series = copy.deepcopy(series)
        for item in averages_series:
            value = self.data['averages'][_get_periodicity_index(item, averages_series.resolution, self.data['periodicity'], dst_affected=self.data['dst_affected'])]
            if not value:
                value = 0
            item.data['periodic_average'] =value
        averages_series.plot(**kwargs)


#=========================
#  Prophet Forecaster
#=========================

class ProphetForecaster(Forecaster, _ProphetModel):
    """A forecasting model based on Prophet. Prophet (from Facebook) implements a procedure for forecasting time series data based
    on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects.
    """

    window = None

    @Forecaster.fit_method
    def fit(self, series, verbose=False):

        if len(series.data_labels()) > 1:
            raise Exception('Multivariate time series are not yet supported')

        from prophet import Prophet

        if not verbose:
            # https://stackoverflow.com/questions/45551000/how-to-control-output-from-fbprophet
            logging.getLogger('cmdstanpy').disabled = True
            logging.getLogger('prophet').disabled = True

        data = self._from_timeseria_to_prophet(series)

        # Instantiate the Prophet model
        self.prophet_model = Prophet()

        # Fit tjhe Prophet model
        self.prophet_model.fit(data)

        # Save the series we used for the fit.
        self.fit_series = series

        # Prophet, as the ARIMA models, has no window
        self.data['window'] = 0

    @Forecaster.predict_method
    def predict(self, data, steps=1):

        series = data

        data_label = self.data['data_labels'][0]

        # Prepare a dataframe with all the timestamps to forecast
        last_item    = series[-1]
        last_item_t  = last_item.t
        last_item_dt = last_item.dt
        data_to_forecast = []

        for _ in range(steps):
            new_item_dt = last_item_dt + series.resolution
            data_to_forecast.append(self._remove_timezone(new_item_dt))
            last_item_dt = new_item_dt

        dataframe_to_forecast = DataFrame(data_to_forecast, columns = ['ds'])

        # Call Prophet predict
        forecast = self.prophet_model.predict(dataframe_to_forecast)
        #forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

        # Re arrange predict results
        forecasted_items = []
        for i in range(steps):
            forecasted_items.append({data_label: float(forecast['yhat'][i])})

        # Return
        return forecasted_items


#=========================
#  ARIMA Forecaster
#=========================

class ARIMAForecaster(Forecaster, _ARIMAModel):
    """A forecasting model based on ARIMA. AutoRegressive Integrated Moving Average models are a generalization of an
    AutoRegessive Moving Average (ARMA) model, which provide a description of a (weakly) stationary stochastic process
    in terms of two polynomials, one for the autoregression (AR) and the second for the moving average (MA). The "I"
    indicates that the data values have been replaced with the difference between their values and the previous values.

    Args:
        p(int): the order of the AR term.
        d(int): the number of differencing required to make the time series stationary.
        q(int): the order of the MA term.
    """

    window = 0

    def __init__(self, p=1,d=1,q=0): #p=5,d=2,q=5
        if (p,d,q) == (1,1,0):
            logger.info('You are using ARIMA\'s defaults of p=1, d=1, q=0. You might want to set them to more suitable values when initializing the model.')
        self.p = p
        self.d = d
        self.q = q
        # TODO: save the above in data[]?
        super(ARIMAForecaster, self).__init__()

    @Forecaster.fit_method
    def fit(self, series, verbose=False):

        import statsmodels.api as sm

        if len(series.data_labels()) > 1:
            raise Exception('Multivariate time series require to have the data_label of the prediction specified')
        data_label=series.data_labels()[0]

        data = array(series.to_df()[data_label])

        # Save model and fit
        self.model = sm.tsa.ARIMA(data, order=(self.p,self.d,self.q))
        self.model_res = self.model.fit()

        # Save the series we used for the fit.
        self.fit_series = series

        # The ARIMA models, as Prophet, have no window
        self.data['window'] = 0

    @Forecaster.predict_method
    def predict(self, data, steps=1):

        series = data

        data_label = self.data['data_labels'][0]

        # Chack that we are applying on a time series ending with the same data point where the fit series was
        if self.fit_series[-1].t != series[-1].t:
            raise NonContiguityError('Sorry, this model can be applied only on a time series ending with the same timestamp as the time series used for the fit.')

        # Return the prediction
        return [{data_label: value} for value in self.model_res.forecast(steps)]


#=========================
#  AARIMA Forecaster
#=========================

class AARIMAForecaster(Forecaster, _ARIMAModel):
    """A forecasting model based on Auto-ARIMA. Auto-ARIMA models set automatically the best values for the
    p, d and q parameters, trying different values and checking which ones perform better.
    """

    window = 0

    @Forecaster.fit_method
    def fit(self, series, verbose=False, **kwargs):

        import pmdarima as pm

        if len(series.data_labels()) > 1:
            raise Exception('Multivariate time series require to have the data_label of the prediction specified')
        data_label=series.data_labels()[0]

        data = array(series.to_df()[data_label])

        # Change some defaults # TODO: just set them and remove them as optional kwargs?
        error_action = kwargs.pop('error_action', 'ignore')
        suppress_warnings = kwargs.pop('suppress_warnings', True)
        stepwise = kwargs.pop('stepwise', True)

        # See https://alkaline-ml.com/pmdarima/_modules/pmdarima/arima/auto.html for the other defaults

        # Call the pmdarima aut_arima function
        autoarima_model = pm.auto_arima(data, error_action=error_action,
                                        suppress_warnings=suppress_warnings,
                                        stepwise=stepwise, trace=verbose, **kwargs)

        autoarima_model.summary()

        self.model = autoarima_model

        # Save the series we used for the fit.
        self.fit_series = series

        # The ARIMA models, as Prophet, have no window
        self.data['window'] = 0

    @Forecaster.predict_method
    def predict(self, series, steps=1):

        data_label = self.data['data_labels'][0]

        # Chack that we are applying on a time series ending with the same datapoint where the fit series was
        if self.fit_series[-1].t != series[-1].t:
            raise NonContiguityError('Sorry, this model can be applied only on a time series ending with the same timestamp as the time series used for the fit.')

        # Return the predicion. We need the [0] to access yhat, other indexes are erorrs etc.
        return [{data_label: value} for value in self.model.predict(steps)]


#=========================
#  LSTM Forecaster
#=========================

class LSTMForecaster(Forecaster, _KerasModel):
    """A forecasting model based on a LSTM neural network. LSTMs are artificial neutral networks particularly well suited for time series forecasting tasks.

    Args:
        window (int): the window length.
        features(list): which features to use. Supported values are:
            ``values`` (use the data values),
            ``diffs``  (use the diffs between the values), and
            ``hours``  (use the hours of the timestamp).
        neurons(int): how many neurons to use for the LSTM neural network hidden layer.
        keras_model(keras.Model) : an optional external Keras Model architecture. Will cause the ``neurons`` argument to be discarded.
    """

    @property
    def window(self):
        return self.data['window']

    def __init__(self, window=3, features=['values'], neurons=128, keras_model=None):

        # Call parent init
        super(LSTMForecaster, self).__init__()

        # Set window, neurons, features
        self.data['window'] = window
        self.data['neurons'] = neurons
        self.data['features'] = features

        # Set external model architecture if any
        self.keras_model = keras_model

    @classmethod
    def load(cls, path):
        # Override the load method to load the Keras model as well
        model = super().load(path)
        model._load_keras_model(path)
        return model

    def save(self, path):
        # Override the save method to load the Keras model as well
        super(LSTMForecaster, self).save(path)
        self._save_keras_model(path)


    def _fit(self, series, epochs=30, normalize=True, target='all', with_context=False, loss='MSE',
             data_loss_limit=1.0, reproducible=False, verbose=False, update=False):

        if reproducible:
            ensure_reproducibility()

        # Set the loss in Keras notation
        if loss == 'MSE':
            loss = 'mean_squared_error'
        elif loss == 'MAE':
            loss = 'mean_absolute_error'
        elif loss == 'MAPE':
            loss = 'mean_absolute_percentage_error'
        elif loss == 'MSLE':
            loss = 'mean_squared_logarithmic_error'
        elif loss == 'RMSE':
            from tensorflow.keras import backend as keras_backend
            def root_mean_squared_error(y_true, y_pred):
                return keras_backend.sqrt(keras_backend.mean(keras_backend.square(y_pred - y_true)))
            loss = root_mean_squared_error
        else:
            loss = loss

        # Set the targets and context data labels
        context_data_labels = None
        if target == 'all':
            target_data_labels = series.data_labels()
            if with_context:
                raise ValueError('Cannot use context with all data labels, choose which ones')
        else:
            if isinstance(target, str):
                target_data_labels = [target]
            elif isinstance(target_data_labels, list):
                target_data_labels = target
            else:
                raise TypeError('Don\'t know how to target for data labels as type "{}"'.format(target_data_labels.__class__.__name__))
            for target_data_label in target_data_labels:
                if target_data_label not in series.data_labels():
                    raise ValueError('Cannot target data label "{}" as not found in the series labels ({})'.format(target_data_label, series.data_labels()))
            if with_context:
                context_data_labels = []
                for series_data_label in series.data_labels():
                    if series_data_label not in target_data_labels:
                        context_data_labels.append(series_data_label)

        if update:
            # Check consistency
            if target_data_labels != self.data['target_data_labels']:
                raise ValueError('This model was originally fitted on target data labels "{}", cannot use target data labels "{}" (got target="{}")'.format(self.data['target_data_labels'], target_data_labels, target))
            if self.data['context_data_labels'] is not None and not with_context:
                raise ValueError('This model was originally fitted with context, cannot update the fit without (got with_context="{}")'.format(with_context))
        else:
            # Save the data labels
            self.data['data_labels'] = series.data_labels()
            self.data['target_data_labels'] = target_data_labels
            self.data['context_data_labels'] = context_data_labels

        # Set verbose switch
        if verbose:
            verbose=1
        else:
            verbose=0

        if update:
            if 'min_values' in self.data and not normalize:
                raise ValueError('This model was originally fitted with normalization enabled and must be updated in the same way')

        if normalize:

            # Compute and store, or load, min and max values (for each label)
            if update:
                min_values = self.data['min_values']
                max_values = self.data['max_values']
            else:
                min_values = series.min()
                max_values = series.max()
                self.data['min_values'] = min_values
                self.data['max_values'] = max_values

            # We need this to in order not to modify original data
            series = series.duplicate()

            # Normalize series
            for datapoint in series:
                for data_label in datapoint.data:
                    datapoint.data[data_label] = (datapoint.data[data_label] - min_values[data_label]) / (max_values[data_label] - min_values[data_label])

        # Move to matrix representation
        window_elements_matrix, target_values_vector, context_data_vector = self._to_matrix_representation(series = series,
                                                                                                           window = self.data['window'],
                                                                                                           steps = 1,
                                                                                                           context_data_labels = context_data_labels,
                                                                                                           target_data_labels = target_data_labels,
                                                                                                           data_loss_limit = data_loss_limit)

        if not window_elements_matrix:
            raise ValueError('Too much data loss (not a single element below the limit), cannot fit!')

        # Compute window (plus context) features
        window_features_matrix = []
        for i in range(len(window_elements_matrix)):
            window_features_matrix.append(self._compute_window_features(window_elements_matrix[i],
                                                                        data_labels = self.data['data_labels'],
                                                                        time_unit = series.resolution,
                                                                        features = self.data['features'],
                                                                        context_data = context_data_vector[i] if with_context else None))

        # Obtain the number of features based on _compute_window_features() output
        features_per_window_item = len(window_features_matrix[0][0])
        output_dimension = len(target_values_vector[0])

        # Set the input shape
        input_shape=(self.data['window'] + 1 if with_context else self.data['window'], features_per_window_item)

        # Create the default model architecture if not already set (either because we are updating or because it was provided in the init)
        if not self.keras_model:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Input, LSTM, Dense
            self.keras_model = Sequential()
            self.keras_model.add(Input(input_shape))
            self.keras_model.add(LSTM(self.data['neurons'], ))
            self.keras_model.add(Dense(output_dimension))
            self.keras_model.compile(loss=loss, optimizer='adam')

        # Fit
        self.keras_model.fit(array(window_features_matrix), array(target_values_vector), epochs=epochs, verbose=verbose)

    @Forecaster.fit_method
    def fit(self, series, epochs=30, normalize=True, target='all', with_context=False,
            loss='MSE', data_loss_limit=1.0, reproducible=False, verbose=False):
        """Fit the model on a series.

        Args:
            series(series): the series on which to fit the model.
            epochs(int): for how many epochs to train. Defaults to ``30``.
            normalize(bool): if to normalize the data between 0 and 1 or not. Enabled by default.
            target(str,list): what data labels to target, defaults to  ``all`` for all of them.
            with_context(bool): if to use context data when predicting. Not enabled by default.
            loss(str): the error metric to minimize while fitting (a.k.a. the loss or objective function).
                       Supported values are: ``MSE``, ``RMSE``, ``MSLE``, ``MAE`` and ``MAPE``, any other value
                       supported by Keras as loss function or any callable object. Defaults to ``MSE``.
            data_loss_limit(float): discard from the fit elements with a data loss greater than or equal to
                                    this limit. Defaults to ``1``.
            reproducible(bool): if to make the fit deterministic. Not enabled by default.
            verbose(bool): if to print the training output in the process. Not enabled by default.
        """

        # Call fit logic
        return self._fit(series, epochs=epochs, normalize=normalize, target=target, with_context=with_context,
                         loss=loss, data_loss_limit=data_loss_limit, reproducible=reproducible, verbose=verbose)

    @Forecaster.fit_update_method
    def fit_update(self, series, epochs=30, normalize=True, target='all', with_context=False,
            loss='MSE', data_loss_limit=1.0, reproducible=False, verbose=False):
        """Update the model fit on a series.

        Args:
            series(series): the series on which to fit the model.
            epochs(int): for how many epochs to train. Defaults to ``30``.
            normalize(bool): if to normalize the data between 0 and 1 or not. Enabled by default.
            target(str,list): what data labels to target, defaults to  ``all`` for all of them.
            with_context(bool): if to use context data when predicting. Not enabled by default.
            loss(str): the error metric to minimize while fitting (a.k.a. the loss or objective function).
                       Supported values are: ``MSE``, ``RMSE``, ``MSLE``, ``MAE`` and ``MAPE``, any other value
                       supported by Keras as loss function or any callable object. Defaults to ``MSE``.
            data_loss_limit(float): discard from the fit elements with a data loss greater than or equal to
                                    this limit. Defaults to ``1``.
            reproducible(bool): if to make the fit deterministic. Not enabled by default.
            verbose(bool): if to print the training output in the process. Not enabled by default.
        """

        # Call fit logic
        return self._fit(series, epochs=epochs, normalize=normalize, target=target, with_context=with_context, loss=loss,
                         data_loss_limit=data_loss_limit, reproducible=reproducible, verbose=verbose, update=True)

    @Forecaster.predict_method
    def predict(self, series, steps=1, context_data=None,  verbose=False):
        """Call the model predict logic on a series.

        Args:
            series(series): the series on which to perform the predict.
            steps(int): how may steps-haead to predict.
            context_data(dict): the data to use as context for the prediction.
            verbose(bool): if to print the predict output in the process.
        """

        if len(series) < self.data['window']:
            raise ValueError('The series length ({}) is shorter than the model window ({})'.format(len(series), self.data['window']))

        if steps>1:
            raise NotImplementedError('This forecaster does not support multi-step predictions.')

        # Set verbose switch
        if verbose:
            verbose=1
        else:
            verbose=0

        # Get the window if we were given a longer series
        window_series = series[-self.data['window']:]

        # Duplicate so that we are free to normalize in-place at the next step
        window_series = window_series.duplicate()

        # Convert to list in order to be able to handle datapoints with different labels for the contex
        window_datapoints = list(window_series)

        # Normalize window and context data if we have to do so
        try:
            self.data['min_values']
        except:
            normalize = False
        else:
            normalize = True
            for datapoint in window_datapoints:
                for data_label in datapoint.data:
                    datapoint.data[data_label] = (datapoint.data[data_label] - self.data['min_values'][data_label]) / (self.data['max_values'][data_label] - self.data['min_values'][data_label])

            if context_data:
                context_data = {key: value for key,value in context_data.items()}
                for data_label in context_data:
                    context_data[data_label] = (context_data[data_label] - self.data['min_values'][data_label]) / (self.data['max_values'][data_label] - self.data['min_values'][data_label])

        # Compute window (plus context) features
        window_features = self._compute_window_features(window_datapoints, data_labels=self.data['data_labels'], time_unit=series.resolution, features=self.data['features'], context_data=context_data)

        # Perform the predict and set prediction data
        yhat = self.keras_model.predict(array([window_features]), verbose=verbose)

        predicted_data = {}
        for i, data_label in enumerate(self.data['target_data_labels']):

            # Get the prediction
            predicted_value_normalized = yhat[0][i]

            # De-normalize if we have to
            if normalize:
                predicted_value = (predicted_value_normalized*(self.data['max_values'][data_label] - self.data['min_values'][data_label])) + self.data['min_values'][data_label]
            else:
                predicted_value = predicted_value_normalized

            # Append to prediction data
            predicted_data[data_label] = predicted_value

        # Return
        return predicted_data


#=========================
#   Linear Regression
#      Forecaster
#=========================

class LinearRegressionForecaster(Forecaster, _KerasModel):
    """A forecasting model based on linear regression.

    Args:
        window (int): the window length.
        features(list): which features to use. Supported values are:
            ``values`` (use the data values),
            ``diffs``  (use the diffs between the values), and
            ``hours``  (use the hours of the timestamp).
    """

    @property
    def window(self):
        return self.data['window']

    def __init__(self, window=3, features=['values']):

        # Call parent init
        super(LinearRegressionForecaster, self).__init__()

        # Set window and features
        self.data['window'] = window
        self.data['features'] = features

    @classmethod
    def load(cls, path):

        # Override the load method to load the Keras model as well
        model = super().load(path)

        # Load the sklearn model
        with open('{}/sklearn_model.pkl'.format(path),'rb') as f:
            model.sklearn_model = pickle.load(f)

        return model

    def save(self, path):

        # Override the save method to load the sklearn model as well
        super(LinearRegressionForecaster, self).save(path)

        # Save the sklearn model
        try:
            with open('{}/sklearn_model.pkl'.format(path),'wb') as f:
                pickle.dump(self.sklearn_model,f)
        except Exception as e:
            shutil.rmtree(path)
            raise e

    @Forecaster.fit_method
    def fit(self, series, data_loss_limit=1.0, verbose=False):
        """Fit the model on a series.

        Args:
            series(series): the series on which to fit the model.
            data_loss_limit(float): discard from the fit elements with a data loss greater than or equal to
                                    this limit. Defaults to ``1``.
            verbose(bool): not supported, has no effect.
        """

        # Data labels shortcut
        data_labels = series.data_labels()

        if len(data_labels) > 1:
            raise NotImplementedError('Multivariate time series are not supported')

        # Move to matrix representation
        window_elements_matrix, target_values_vector, _ = self._to_matrix_representation(series = series,
                                                                                         window = self.data['window'],
                                                                                         steps = 1,
                                                                                         context_data_labels = data_labels,
                                                                                         target_data_labels = data_labels,
                                                                                         data_loss_limit = data_loss_limit)

        if not window_elements_matrix:
            raise ValueError('Too much data loss (not a single element below the limit), cannot fit!')

        # Compute window (plus context) features
        window_features_matrix = []
        for i in range(len(window_elements_matrix)):
            window_features_matrix.append(self._compute_window_features(window_elements_matrix[i],
                                                                        data_labels = data_labels,
                                                                        time_unit = series.resolution,
                                                                        features = self.data['features'],
                                                                        context_data = None,
                                                                        flatten = True))

        sklearn_model = LinearRegression()
        sklearn_model.fit(array(window_features_matrix), array(target_values_vector))

        self.sklearn_model = sklearn_model

    @Forecaster.predict_method
    def predict(self, series, steps=1, verbose=False):
        """Call the model predict logic on a series.

        Args:
            series(series): the series on which to perform the predict.
            steps(int): how may steps-haead to predict.
            verbose(bool): not supported, has no effect.
        """

        if len(series) < self.data['window']:
            raise ValueError('The series length ({}) is shorter than the model window ({})'.format(len(series), self.data['window']))

        if steps>1:
            raise NotImplementedError('This forecaster does not support multi-step predictions.')

        # Get the window if we were given a longer series
        window_series = series[-self.data['window']:]

        # Convert to list in order to be able to handle datapoints with different labels for the contex
        window_datapoints = list(window_series)


        # Compute window features
        window_features = self._compute_window_features(window_datapoints,
                                                        data_labels=self.data['data_labels'],
                                                        time_unit=series.resolution,
                                                        features=self.data['features'],
                                                        context_data=None,
                                                        flatten = True)

        # Perform the predict and set prediction data
        yhat = self.sklearn_model.predict(array([window_features]))

        # Create the prediction data
        predicted_data = {self.data['data_labels'][0]: yhat[0][0]}

        # Return
        return predicted_data

