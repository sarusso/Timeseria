# -*- coding: utf-8 -*-
"""Forecasting models."""

import copy
from pandas import DataFrame
from numpy import array
from math import sqrt
from propertime.utils import now_s, dt_from_s, s_from_dt
from datetime import datetime

from ..datastructures import DataTimeSlot, TimePoint, DataTimePoint, Slot, Point, TimeSeries
from ..exceptions import NonContiguityError
from ..utilities import detect_periodicity, _get_periodicity_index, _item_is_in_range, mean_absolute_percentage_error, ensure_reproducibility
from ..units import Unit, TimeUnit
from .base import Model, _ProphetModel, _ARIMAModel, _KerasModel

# Sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
    """A generic series forecasting model. Besides the ``predict()`` and  ``apply()`` methods, it also has a ``forecast()``
    method which, in case of nested data structures (i.e. DataPoint or DataSlot) allows to get the full forecasted points
    or slots instead of just the raw, inner data values returned by the ``predict()``. In case of plain data structures
    (e.g. a list), the ``forecast()`` method is instead equivalent to the ``predict()``.

    Series resolution and data labels consistency are enforced between all methods and save/load operations.

    Args:
        path (str): a path from which to load a saved model. Will override all other init settings.
    """

    window = None

    def forecast(self, series, steps=1, forecast_start=None, context_data=None):
        """Forecast n steps-ahead data points or slots"""

        # Check series
        if not isinstance(series, TimeSeries):
            raise NotImplementedError('Models work only with TimeSeries data for now (got "{}")'.format(series.__class__.__name__))

        if 'target_data_labels' in self.data and self.data['target_data_labels'] and set(self.data['target_data_labels']) != set(series.data_labels):
            if not context_data:
                raise ValueError('Forecasting with a forecaster fit to specific target data labels requires to give context data')

        # Set forecast starting item
        if forecast_start is not None:
            forecast_start_item = series[forecast_start]
        else:
            forecast_start_item = series[-1]

        # Handle forecast start
        if forecast_start is not None:
            try:
                if context_data:
                    predicted_data = self.predict(series, steps=steps, forecast_start=forecast_start, context_data=context_data)
                else:
                    predicted_data = self.predict(series, steps=steps, forecast_start=forecast_start)
            except TypeError as e:
                if 'unexpected keyword argument' and  'forecast_start' in str(e):
                    raise NotImplementedError('The model does not support the "forecast_start" parameter, cannot proceed')
                else:
                    raise
        else:
            if context_data:
                predicted_data = self.predict(series, steps=steps, context_data=context_data)
            else:
                predicted_data = self.predict(series, steps=steps)

        # List of predictions or single prediction?
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

    @Model.apply_function
    def apply(self, series, steps=1, inplace=False, context_data=None):
        """Apply the forecast on the given series for n steps-ahead."""

        if 'target_data_labels' in self.data and self.data['target_data_labels'] and set(self.data['target_data_labels']) != set(series.data_labels):
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
                forecast_model_results = self.forecast(series, steps=steps, context_data=context_data)
            else:
                forecast_model_results = self.forecast(series, steps=steps)
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
                    forecast_model_results = self.forecast(series, steps=1, context_data=context_data)
                else:
                    forecast_model_results = self.forecast(series, steps=1)

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

    @Model.evaluate_function
    def evaluate(self, series, steps='auto', limit=None, plots=False, plot=False, metrics=['RMSE', 'MAE'], details=False, start=None, end=None, evaluation_series=False):
        """Evaluate the forecaster on a series.

        Args:
            steps (int,list): a single value or a list of values for how many steps-ahead to forecast in the evaluation. Default to automatic detection based on the model.
            limit(int): set a limit for the time data elements to use for the evaluation.
            plot(bool): if to produce an overall evaluation plot, defaulted to False. If set to True, the evaluation results are not retuned.
                        To get both the evaluation results and the overall evaluation plot, set the `evaluation_series` switch to True in
                        order to add it to the evaluation results and plot it afterwards.
            plots(bool): if to produce evaluation plots, defaulted to False. Beware that setting this option to True will cause to generate
                         a plot for each evaluation point or slot of the time data: use with caution and only on small time data. Not
                         supported with image-based plots.
            metrics(list): the error metrics to use for the evaluation.
                Supported values are:
                ``RMSE`` (Root Mean Square Error),
                ``MAE``  (Mean Absolute Error), and
                ``MAPE``  (Mean Absolute percentage Error).
            details(bool): if to add intermediate steps details to the evaluation results.
            start(float, datetime): evaluation start (epoch timestamp or datetime).
            end(float, datetime): evaluation end (epoch timestamp or datetime).
            evaluation_series(bool): if to add to the results an evaluation timeseirs containing the eror metrics. Defaulted to false.
        """

        if len(series.data_labels) > 1:
            raise NotImplementedError('Sorry, evaluating models built for multivariate time series is not supported yet')

        # Set empty list if metrics were None
        if metrics is None:
            metrics = []

        # Check supported metrics
        for metric in metrics:
            if metric not in ['RMSE', 'MAE', 'MAPE']:
                raise ValueError('The metric "{}" is not supported'.format(metric))

        # Set evaluation steps if we have to
        if steps == 'auto':
            try:
                steps = [1, self.data['periodicity']]
            except KeyError:
                try:
                    if not self.data['window']:
                        steps = [1]
                    else:
                        steps = [1, self.data['window']]
                except (KeyError, AttributeError):
                    steps = [1]
        elif isinstance(steps, list):
            if not self.data['window']:
                if steps != [1]:
                    raise ValueError('Evaluating a windowless model on a multi-step forecast does not make sense (got steps={})'.format(steps))
        else:
            if not self.data['window']:
                if steps != 1:
                    raise ValueError('Evaluating a windowless model on a multi-step forecast does not make sense (got steps={})'.format(steps))
            steps = list(range(1, steps+1))

        return_evaluation_series = evaluation_series

        if plot or evaluation_series:

            if steps != [1]:
                raise ValueError('Plotting or getting back an evaluation time series is only supported with single step ahead forecasts (steps=[1])')

            evaluation_series = series.duplicate()
            if self.data['window']:
                evaluation_series = evaluation_series[self.data['window']:]
            if limit:
                evaluation_series = evaluation_series[:limit]

        # Handle start/end
        start_t, end_t = self._handle_start_end(start, end)

        # Support vars
        results = {}
        warned = False

        # Log
        logger.info('Will evaluate model for %s steps ahead with metrics %s', steps, metrics)

        for steps_round in steps:

            # Support vars
            real_values = []
            model_values = []
            processed_samples = 0

            for data_label in series.data_labels:

                # If the model has no window, evaluate on the entire time series

                try:
                    if not self.data['window']:
                        has_window = False
                    else:
                        has_window = True
                except (KeyError, AttributeError):
                    has_window = False

                if not has_window:

                    # Note: steps_round is always equal to the entire test time series length in window-less model evaluation

                    # Create a time series where to apply the forecast, with only a point "in the past",
                    # this is done in order to use the apply function as is. Since the model is not using
                    # any window, the point data will be ignored and just used for its timestamp
                    forecast_series = series.__class__()

                    # TODO: it should not be required to check .resolution type!
                    if isinstance(series[0], Point):
                        if isinstance(series.resolution, TimeUnit):
                            forecast_series.append(series[0].__class__(dt = series[0].dt - series.resolution,
                                                                               data = series[0].data))
                        elif isinstance(series.resolution, Unit):
                            forecast_series.append(series[0].__class__(dt = dt_from_s(series[0].t - series.resolution, tz=series[0].tz),
                                                                               data = series[0].data))
                        else:
                            forecast_series.append(series[0].__class__(dt = dt_from_s(series[0].t - series.resolution, tz=series[0].tz),
                                                                               data = series[0].data))
                    elif isinstance(series[0], Slot):
                        if isinstance(series.resolution, TimeUnit):
                            forecast_series.append(series[0].__class__(dt = series[0].dt - series.resolution,
                                                                               unit = series.resolution,
                                                                               data = series[0].data))
                        elif isinstance(series.resolution, Unit):
                            forecast_series.append(series[0].__class__(dt = dt_from_s(series[0].t - series.resolution, tz=series[0].tz),
                                                                               unit = series.resolution,
                                                                               data = series[0].data))
                        else:
                            forecast_series.append(series[0].__class__(dt = dt_from_s(series[0].t - series.resolution, tz=series[0].tz),
                                                                               unit = series.resolution,
                                                                               data = series[0].data))
                    else:
                        raise TypeError('Unknown time series items type (got "{}"'.format(series[0].__class__.__name__))

                    # Set default evaluate samples
                    evaluate_samples = len(series)

                    # Do we have a limit on the evaluate sample to apply?
                    if limit:
                        if limit < evaluate_samples:
                            evaluate_samples = limit

                    # Warn if no limit given and we are over
                    if not limit and evaluate_samples > 10000:
                        logger.warning('No limit set in the evaluation with a quite long time series, this could take some time.')
                        warned=True

                    # All evaluation samples will be processed
                    processed_samples = evaluate_samples

                    # Apply the forecasting model with a length equal to the original series minus the first element
                    self.apply(forecast_series, steps=evaluate_samples, inplace=True)

                    # Save the model and the original value to be compared later on. Create the arrays by skipping the fist item
                    # and move through the forecast time series comparing with the input time series, shifted by one since in the
                    # forecast series we added an "artificial" first point to use the apply()
                    for i in range(1, evaluate_samples+1):

                        model_value = forecast_series[i].data[data_label]
                        model_values.append(model_value)

                        real_value = series[i-1].data[data_label]
                        real_values.append(real_value)


                # Else, process in streaming the series, item by item, and properly take into account the window.
                else:
                    for i in range(len(series)):

                        # Skip if needed
                        try:
                            if not _item_is_in_range(series[i], start_t, end_t):
                                continue
                        except StopIteration:
                            break

                        # Check that we can get enough data
                        if i < self.data['window']:
                            continue
                        if i > (len(series)-steps_round):
                            continue

                        # Compute the various boundaries
                        original_series_boundaries_start = i - (self.data['window'])
                        original_series_boundaries_end = i + steps_round

                        original_forecast_series_boundaries_start = original_series_boundaries_start
                        original_forecast_series_boundaries_end = original_series_boundaries_end-steps_round

                        # Create the time series where to apply the forecast
                        forecast_series = series.__class__()
                        for j in range(original_forecast_series_boundaries_start, original_forecast_series_boundaries_end):

                            if isinstance(series[0], Point):
                                forecast_series.append(series[0].__class__(t = series[j].t,
                                                                                   tz = series[j].tz,
                                                                                   data = series[j].data))
                            elif isinstance(series[0], Slot):
                                forecast_series.append(series[0].__class__(start = series[j].start,
                                                                                   end   = series[j].end,
                                                                                   unit  = series[j].unit,
                                                                                   data  = series[j].data))

                            # This would lead to add the forecasted index to the original data (and we don't want it)
                            #forecast_series.append(series[j])

                        # Apply the forecasting model
                        self.apply(forecast_series, steps=steps_round, inplace=True)

                        # Plot results time series?
                        if plots:
                            forecast_series.plot()

                        # Save the model and the original value to be compared later on
                        for step in range(steps_round):
                            original_index = original_series_boundaries_start + self.data['window'] + step

                            forecast_index = self.data['window'] + step

                            model_value = forecast_series[forecast_index].data[data_label]
                            model_values.append(model_value)

                            real_value = series[original_index].data[data_label]
                            real_values.append(real_value)

                        processed_samples+=1
                        if limit is not None and processed_samples >= limit:
                            break

                        # Warn if no limit given and we are over
                        if not limit and not warned and i > 10000:
                            logger.warning('No limit set in the evaluation with a quite long time series, this could take some time.')
                            warned=True

            if limit is not None and processed_samples < limit:
                logger.warning('The evaluation limit is set to "{}" but I have only "{}" samples for "{}" steps'.format(limit, processed_samples, steps_round))

            if not model_values:
                raise Exception('Could not evaluate model, maybe not enough data?')

            # Compute RMSE and ME, and add to the results
            if 'RMSE' in metrics:
                results['RMSE_{}_steps'.format(steps_round)] = sqrt(mean_squared_error(real_values, model_values))
            if 'MAE' in metrics:
                results['MAE_{}_steps'.format(steps_round)] = mean_absolute_error(real_values, model_values)
                if evaluation_series:
                    for i in range(len(model_values)):
                        evaluation_series[i].data['{}_AE'.format(data_label)] = abs(real_values[i] - model_values[i])
            if 'MAPE' in metrics:
                results['MAPE_{}_steps'.format(steps_round)] = mean_absolute_percentage_error(real_values, model_values)
                if evaluation_series:
                    for i in range(len(model_values)):
                        evaluation_series[i].data['{}_APE'.format(data_label)] = abs(real_values[i] - model_values[i]) / real_values[i]

            if evaluation_series:
                for i in range(len(model_values)):
                    evaluation_series[i].data['{}_pred'.format(data_label)] = model_values[i]

        # Compute overall RMSE
        if 'RMSE' in metrics:
            sum_rmse = 0
            count = 0
            for data_label in results:
                if data_label.startswith('RMSE_'):
                    sum_rmse += results[data_label]
                    count += 1
            results['RMSE'] = sum_rmse/count

        # Compute overall MAE
        if 'MAE' in metrics:
            sum_me = 0
            count = 0
            for data_label in results:
                if data_label.startswith('MAE_'):
                    sum_me += results[data_label]
                    count += 1
            results['MAE'] = sum_me/count

        # Compute overall MAPE
        if 'MAPE' in metrics:
            sum_me = 0
            count = 0
            for data_label in results:
                if data_label.startswith('MAPE_'):
                    sum_me += results[data_label]
                    count += 1
            results['MAPE'] = sum_me/count

        if not details:
            simple_results = {}
            if 'RMSE' in metrics:
                simple_results['RMSE'] = results['RMSE']
            if 'MAE' in metrics:
                simple_results['MAE'] = results['MAE']
            if 'MAPE' in metrics:
                simple_results['MAPE'] = results['MAPE']
            results = simple_results

        # Do we have to plot the evaluation series?
        if plot:
            if results:
                logger.info('Plotting the evaluation time series, not returning evaluation results (which are: {})'.format(results))
            return evaluation_series.plot()

        # Handle evaluation series if required
        if return_evaluation_series:
            results['evaluation_series'] = evaluation_series

        # Return evaluation results if any
        if results:
            return results


#=========================
#  P. Average Forecaster
#=========================

class PeriodicAverageForecaster(Forecaster):
    """A series forecasting model based on periodic averages.

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
        model.data['averages'] = {int(key):value for key, value in model.data['averages'].items()}
        return model

    @Forecaster.fit_function
    def fit(self, series, start=None, end=None, periodicity='auto', dst_affected=False, verbose=False):
        """Fit the model on a series.

        Args:
            start(float, datetime): fit start (epoch timestamp or datetime).
            end(float, datetime): fit end (epoch timestamp or datetime).
            periodicity(int): the periodicty of the series. If set to ``auto`` then it will be automatically detected using a FFT.
            dst_affected(bool): if the model should take into account DST effects.
        """

        if len(series.data_labels) > 1:
            raise NotImplementedError('Multivariate time series are not yet supported')

        start_t, end_t = self._handle_start_end(start, end)

        # Set or detect periodicity
        if periodicity == 'auto':
            periodicity =  detect_periodicity(series)
            logger.info('Detected periodicity: %sx %s', periodicity, series.resolution)

        self.data['periodicity']  = periodicity
        self.data['dst_affected'] = dst_affected

        # Set window
        if self._window != 'auto':
            self.data['window'] = self._window
        else:
            logger.info('Using a window of "{}"'.format(periodicity))
            self.data['window'] = periodicity

        for data_label in series.data_labels:
            sums   = {}
            totals = {}
            processed = 0
            for item in series:

                # Skip if needed
                try:
                    if not _item_is_in_range(item, start_t, end_t):
                        continue
                except StopIteration:
                    break

                # Process
                periodicity_index = _get_periodicity_index(item, series.resolution, periodicity, dst_affected)
                if not periodicity_index in sums:
                    sums[periodicity_index] = item.data[data_label]
                    totals[periodicity_index] = 1
                else:
                    sums[periodicity_index] += item.data[data_label]
                    totals[periodicity_index] +=1
                processed += 1

        averages={}
        for periodicity_index in sums:
            averages[periodicity_index] = sums[periodicity_index]/totals[periodicity_index]
        self.data['averages'] = averages

        logger.debug('Processed %s items', processed)

    @Forecaster.predict_function
    def predict(self, series, from_i=None, steps=1):

        if len(series) < self.data['window']:
            raise ValueError('The series length ({}) is shorter than the model window ({})'.format(len(series), self.data['window']))

        # Univariate is enforced by the fit
        data_label = self.data['data_labels'][0]

        # Set forecast starting item
        if from_i is None:
            from_i = len(series) - 1

        # Get forecast start item
        forecast_start_item = series[from_i]

        # Support vars
        forecast_timestamps = []
        forecast_data = []

        # Compute the offset (avg diff between the real values and the forecasts on the first window)
        diffs  = 0
        for j in range(self.data['window']):
            series_index = from_i - self.data['window'] + j
            real_value = series[series_index].data[data_label]
            forecast_value = self.data['averages'][_get_periodicity_index(series[series_index], series.resolution, self.data['periodicity'], dst_affected=self.data['dst_affected'])]
            diffs += (real_value - forecast_value)

        # Sum the avg diff between the real and the forecast on the window to the forecast (the offset)
        offset = diffs/j

        # Perform the forecast
        for i in range(steps):
            step = i + 1

            # Set forecast timestamp
            if isinstance(series[0], Slot):
                try:
                    forecast_timestamp = forecast_timestamps[-1] + series.resolution
                    forecast_timestamps.append(forecast_timestamp)
                except IndexError:
                    forecast_timestamp = forecast_start_item.end
                    forecast_timestamps.append(forecast_timestamp)

            else:
                forecast_timestamp = TimePoint(t = forecast_start_item.t + (series.resolution.as_seconds()*step), tz = forecast_start_item.tz )

            # Compute the real forecast data
            periodicity_index = _get_periodicity_index(forecast_timestamp, series.resolution, self.data['periodicity'], dst_affected=self.data['dst_affected'])
            forecast_data.append({data_label: self.data['averages'][periodicity_index] + (offset*1.0)})

        # Return
        return forecast_data

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
    """A series forecasting model based on Prophet. Prophet (from Facebook) implements a procedure for forecasting time series data based
    on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects.
    """

    window = None

    @Forecaster.fit_function
    def fit(self, series, start=None, end=None, verbose=False):

        if len(series.data_labels) > 1:
            raise Exception('Multivariate time series are not yet supported')

        from prophet import Prophet

        if not verbose:
            # https://stackoverflow.com/questions/45551000/how-to-control-output-from-fbprophet
            logging.getLogger('cmdstanpy').disabled = True
            logging.getLogger('prophet').disabled = True

        start_t, end_t = self._handle_start_end(start, end)

        data = self._from_timeseria_to_prophet(series, from_t=start_t, to_t=end_t)

        # Instantiate the Prophet model
        self.prophet_model = Prophet()

        # Fit tjhe Prophet model
        self.prophet_model.fit(data)

        # Save the series we used for the fit.
        self.fit_series = series

        # Prophet, as the ARIMA models, has no window
        self.data['window'] = 0

    @Forecaster.predict_function
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
    """A series forecasting model based on ARIMA. AutoRegressive Integrated Moving Average models are a generalization of an
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

    @Forecaster.fit_function
    def fit(self, series, verbose=False):

        import statsmodels.api as sm

        if len(series.data_labels) > 1:
            raise Exception('Multivariate time series require to have the data_label of the prediction specified')
        data_label=series.data_labels[0]

        data = array(series.to_df()[data_label])

        # Save model and fit
        self.model = sm.tsa.ARIMA(data, (self.p,self.d,self.q))
        self.model_res = self.model.fit(disp=verbose)

        # Save the series we used for the fit.
        self.fit_series = series

        # The ARIMA models, as Prophet, have no window
        self.data['window'] = 0

    @Forecaster.predict_function
    def predict(self, data, steps=1):

        series = data

        data_label = self.data['data_labels'][0]

        # Chack that we are applying on a time series ending with the same data point where the fit series was
        if self.fit_series[-1].t != series[-1].t:
            raise NonContiguityError('Sorry, this model can be applied only on a time series ending with the same timestamp as the time series used for the fit.')

        # Return the predicion. We need the [0] to access yhat, other indexes are erorrs etc.
        return [{data_label: value} for value in self.model_res.forecast(steps)[0]]


#=========================
#  AARIMA Forecaster
#=========================

class AARIMAForecaster(Forecaster, _ARIMAModel):
    """A series forecasting model based on Auto-ARIMA. Auto-ARIMA models set automatically the best values for the
    p, d and q parameters, trying different values and checking which ones perform better.
    """

    window = 0

    @Forecaster.fit_function
    def fit(self, series, verbose=False, **kwargs):

        import pmdarima as pm

        if len(series.data_labels) > 1:
            raise Exception('Multivariate time series require to have the data_label of the prediction specified')
        data_label=series.data_labels[0]

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

    @Forecaster.predict_function
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
    """A series forecasting model based on a LSTM neural network. LSTMs are artificial neutral networks particularly well suited for time series forecasting tasks.

    Args:
        window (int): the window length.
        features(list): which features to use. Supported values are:
            ``values`` (use the data values),
            ``diffs``  (use the diffs between the values), and
            ``hours``  (use the hours of the timestamp).
        neurons(int): how many neaurons to use for the LSTM neural nework hidden layer.
        keras_model(keras.Model) : an optional external Keras Model architecture. Will cause the ``neurons`` argument to be discarded.
    """

    @property
    def window(self):
        return self.data['window']

    def __init__(self, window=3, features=['values'], neurons=128, keras_model=None):

        if window == 3:
            logger.info('Using default window size of 3')

        if features == ['values']:
            logger.info('Using default features: values')

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

    @Forecaster.fit_function
    def fit(self, series, start=None, end=None, epochs=30, normalize=True, target='all', with_context=False, reproducible=False, verbose=False):
        """Fit the model on a series.

        Args:
            series(series): the series on whihc to fit the model.
            start(datetieme,float): the start timestamp of the fit.
            end(datetieme,float): the end timestamp of the fit.
            epochs(int): for how many epochs to train.
            normalize(bool): if to normalize the data between 0 and 1 or not.
            target(str,list): what data labels to target, 'all' for all of them.
            with_context(bool): if to use context data when predicting.
            reproducible(bool): if to make the fit deterministic.
            verbose(bool): if to print the training output in the process.
        """

        if reproducible:
            ensure_reproducibility()

        # Set and save the targets and context data labels
        context_data_labels = None
        if target == 'all':
            target_data_labels = series.data_labels
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
                if target_data_label not in series.data_labels:
                    raise ValueError('Cannot target data label "{}" as not found in the series labels ({})'.format(target_data_label, series.data_labels))
            if with_context:
                context_data_labels = []
                for series_data_label in series.data_labels:
                    if series_data_label not in target_data_labels:
                        context_data_labels.append(series_data_label)

        #logger.debug('target_data_labels: {}'.format(target_data_labels))
        #logger.series_with_forecast('context_data_labels: {}'.format(context_data_labels))

        self.data['target_data_labels'] = target_data_labels
        self.data['context_data_labels'] = context_data_labels

        start_t, end_t = self._handle_start_end(start, end)

        # Set verbose switch
        if verbose:
            verbose=1
        else:
            verbose=0

        # Data labels shortcut
        data_labels = series.data_labels

        if start is None and end is None:
            if normalize:
                series = series.duplicate()
        else:
            filtered_series = series.__class__()
            # TODO: Use _item_is_in_range()? 
            for item in series:
                valid = True
                if start is not None and item.t < start_t:
                    valid = False
                if end is not None and item.t >= end_t:
                    valid = False
                if valid:
                    if normalize:
                        filtered_series.append(copy.deepcopy(item))
                    else:
                        filtered_series.append(item)
            series = filtered_series

        if normalize:
            # Set min and max (for each label)
            min_values = series.min()
            max_values = series.max()

            # Normalize series
            for datapoint in series:
                for data_label in datapoint.data:
                    datapoint.data[data_label] = (datapoint.data[data_label] - min_values[data_label]) / (max_values[data_label] - min_values[data_label])

            # Store normalization factors
            self.data['min_values'] = min_values
            self.data['max_values'] = max_values

        # Move to "matrix" of windows plus "vector" of targets data representation. Or, in other words:
        # window_datapoints is a list of lists (matrix) where each nested list (row) is a list of window datapoints.
        window_datapoints_matrix = self._to_window_datapoints_matrix(series, window=self.data['window'], steps=1)
        if with_context:
            context_data_matrix = self._to_context_data_matrix(series, window=self.data['window'], context_data_labels=context_data_labels, steps=1)
        target_values_vector = self._to_target_values_vector(series, window=self.data['window'], steps=1, target_data_labels=target_data_labels)

        # Compute window (plus context) features
        window_features = []
        for i in range(len(window_datapoints_matrix)):
            window_features.append(self._compute_window_features(window_datapoints_matrix[i],
                                                                 data_labels = data_labels,
                                                                 time_unit = series.resolution,
                                                                 features = self.data['features'],
                                                                 context_data = context_data_matrix[i] if with_context else None))

        # Obtain the number of features based on _compute_window_features() output
        features_per_window_item = len(window_features[0][0])
        output_dimension = len(target_values_vector[0])

        # Create the default model architeture if not given in the init
        if not self.keras_model:
            from keras.models import Sequential
            from keras.layers import Dense
            from keras.layers import LSTM
            self.keras_model = Sequential()
            self.keras_model.add(LSTM(self.data['neurons'], input_shape=(self.data['window'] + 1 if with_context else self.data['window'], features_per_window_item)))
            self.keras_model.add(Dense(output_dimension))
            self.keras_model.compile(loss='mean_squared_error', optimizer='adam')

        # Fit
        self.keras_model.fit(array(window_features), array(target_values_vector), epochs=epochs, verbose=verbose)

    @Forecaster.predict_function
    def predict(self, series, from_i=None, steps=1, context_data=None,  verbose=False):
        """Fit the model on a series.

        Args:
            series(series): the series on which to fit the model.
            from_i(int): the start of the prediction as series position (index).
            steps(int): how may steps-haead to predict.
            context_data(dict): the data to use as context for the prediction.
            verbose(bool): if to print the predict output in the process.
        """

        # TODO: from_i -> start(datetieme,float,int): the start of the prediction (float for epoch and int for an index).

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
        if from_i is not None:
            window_series = series[from_i-self.data['window']:from_i]
        else:
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


