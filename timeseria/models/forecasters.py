# -*- coding: utf-8 -*-
"""Forecasting models."""

import copy
from ..datastructures import DataTimeSlot, TimePoint, DataTimePoint, Slot, Point
from ..exceptions import NonContiguityError
from ..utilities import get_periodicity, get_periodicity_index, set_from_t_and_to_t, item_is_in_range, mean_absolute_percentage_error
from ..time import dt_from_s
from ..units import Unit, TimeUnit
from pandas import DataFrame
from numpy import array
from math import sqrt

# Sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Base models and utilities
from .base import TimeSeriesParametricModel, ProphetModel, ARIMAModel, KerasModel

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

class Forecaster(TimeSeriesParametricModel):
    """A generic forecasting model. Besides the ``predict()`` and  ``apply()`` methods, also provides a ``forecast()``
    method which allows to get the forecasted n steps-ahead full data points or slots.
    
    Args:
        path (str): a path from which to load a saved model. Will override all other init settings.
    """

    def fit(self, timeseries, *args, **kwargs):
        "Fit the forecaster on a time series"
        return super(Forecaster, self).fit(timeseries, *args, **kwargs)


    def predict(self, timeseries, steps=1, *args, **kwargs):
        "Predict n steps-ahead forecast values, and return key-value data where the keys are the data labels."
 
        # Check if the input timeseries is shorter than the window, if any.
        # Note that nearly all forecasters use windows, at least of one point.
        try:
            if len(timeseries) < self.data['window']:
                raise ValueError('The timeseries length ({}) is shorter than the model window ({}), it must be at least equal.'.format(len(timeseries), self.data['window']))
        except KeyError:
            pass
        
        # Call parent predict
        return super(Forecaster, self).predict(timeseries, steps, *args, **kwargs)


    def forecast(self, timeseries, steps=1, forecast_start=None):
        """Forecast n steps-ahead full data points or slots"""

        # Set forecast starting item
        if forecast_start is not None:
            forecast_start_item = timeseries[forecast_start]
        else:
            forecast_start_item = timeseries[-1]
            
        # Handle forecast start
        if forecast_start is not None:
            try:
                predicted_data = self.predict(timeseries=timeseries, steps=steps, forecast_start=forecast_start)
            except TypeError as e:
                if 'unexpected keyword argument' and  'forecast_start' in str(e):
                    raise NotImplementedError('The model does not support the "forecast_start" parameter, cannot proceed')           
                else:
                    raise
        else:
            predicted_data = self.predict(timeseries=timeseries, steps=steps)
                
        # List of predictions or single prediction?
        if isinstance(predicted_data,list):
            forecast = []
            last_item = forecast_start_item
            for data in predicted_data:

                if isinstance(timeseries[0], Slot):
                    forecast.append(DataTimeSlot(start = last_item.end,
                                                 unit  = timeseries.resolution,
                                                 data_loss = None,
                                                 #tz = timeseries.tz,
                                                 data  = data))
                else:
                    forecast.append(DataTimePoint(t = last_item.t + timeseries.resolution,
                                                  tz = timeseries.tz,
                                                  data  = data))
                last_item = forecast[-1]
        else:
            if isinstance(timeseries[0], Slot):
                forecast = DataTimeSlot(start = forecast_start_item.end,
                                        unit  = timeseries.resolution,
                                        data_loss = None,
                                        #tz = timeseries.tz,
                                        data  = predicted_data)
            else:
                forecast = DataTimePoint(t = forecast_start_item.t + timeseries.resolution,
                                         tz = timeseries.tz,
                                         data  = predicted_data)
  
        return forecast


    def apply(self, timeseries, steps=1, *args, **kwargs):
        """Apply the forecast on a time series for n steps-ahead"""
        return super(Forecaster, self).apply(timeseries, steps, *args, **kwargs)

    def _apply(self, timeseries, steps=1, inplace=False):

        input_timeseries_len = len(timeseries)
 
        if inplace:
            forecast_timeseries = timeseries
        else:
            forecast_timeseries = timeseries.duplicate()
        
        # Add the forecast index
        for item in forecast_timeseries:
            item.data_indexes['forecast'] = 0
        
        # Call model forecasting logic
        try:
            forecast_model_results = self.forecast(timeseries = forecast_timeseries, steps=steps)
            if not isinstance(forecast_model_results, list):
                forecast_timeseries.append(forecast_model_results)
            else:
                for item in forecast_model_results:
                    item.data_indexes['forecast'] = 1
                    forecast_timeseries.append(item)

        except NotImplementedError:
            
            for _ in range(steps):
    
                # Call the forecast only on the last point
                forecast_model_results = self.forecast(timeseries = forecast_timeseries, steps=1)

                # Add forecasted index
                forecast_model_results.data_indexes['forecast'] = 1

                # Add the forecast to the forecasts time series
                forecast_timeseries.append(forecast_model_results)
    
        # Do we have missing forecasts?
        if input_timeseries_len + steps != len(forecast_timeseries):
            raise ValueError('There are missing forecasts. If your model does not support multi-step forecasting, raise a NotImplementedError if steps>1 and Timeseria will handle it for you.')
 
        if not inplace:
            return forecast_timeseries
        else:
            return None

    def evaluate(self, timeseries, steps='auto', limit=None, plots=False, plot=False, metrics=['RMSE', 'MAE'], details=False, from_t=None, to_t=None, from_dt=None, to_dt=None, evaluation_timeseries=False):
        """Evaluate the forecaster on a time series.

        Args:
            steps (int,list): a single value or a list of values for how many steps-ahead to forecast in the evaluation. Default to automatic detection based on the model.
            limit(int): set a limit for the time series elements to use for the evaluation.
            plots(bool): if to produce evaluation plots, defaulted to False. Beware that setting this option to True will cause to generate 
                         a plot for each evaluation point or slot of the time series: use with caution and only on small time series.
            plot(bool): if to produce an overall evaluation plot, defaulted to False.
            metrics(list): the error metrics to use for the evaluation.
                Supported values are:
                ``RMSE`` (Root Mean Square Error), 
                ``MAE``  (Mean Absolute Error), and 
                ``MAPE``  (Mean Absolute percentage Error).
            details(bool): if to add intermediate steps details to the evaluation results.
            from_t(float): evaluation starting epoch timestamp.
            to_t(float): evaluation ending epoch timestamp
            from_dt(datetime): evaluation starting datetime.
            to_dt(datetime) : evaluation ending datetime.
            evaluation_timeseries(bool): if to produce an evaluation timeseirs containing the eror metrics. Defaulted to false.
        """
        return super(Forecaster, self).evaluate(timeseries, steps, limit, plots, plot, metrics, details, from_t, to_t, from_dt, to_dt, evaluation_timeseries)

    def _evaluate(self, timeseries, steps='auto', limit=None, plots=False, plot=False, metrics=['RMSE', 'MAE'], details=False, from_t=None, to_t=None, from_dt=None, to_dt=None, evaluation_timeseries=False):

        # Set empty list if metrics were None
        if metrics is None:
            metrics = []

        # Set evaluation steps if we have to
        if steps == 'auto':
            try:
                steps = [1, self.data['periodicity']]
            except KeyError:
                if not self.data['window']:
                    steps = [1]
                else:
                    steps = [1, 2, 3]
        elif isinstance(steps, list):
            if not self.data['window']:
                if steps != [1]:
                    raise ValueError('Evaluating a windowless model on a multi-step forecast does not make sense (got steps={})'.format(steps))
        else:
            if not self.data['window']:
                if steps != 1:
                    raise ValueError('Evaluating a windowless model on a multi-step forecast does not make sense (got steps={})'.format(steps))
            steps = list(range(1, steps+1))
        
        return_evaluation_timeseries = evaluation_timeseries
        
        if plot or evaluation_timeseries:

            if steps != [1]:
                raise ValueError('Plotting or getting back an evaluation time series is only supported with single step ahead forecasts (steps=[1])')
            
            evaluation_timeseries = timeseries.duplicate()
            if self.data['window']:
                evaluation_timeseries = evaluation_timeseries[self.data['window']:]
            if limit:
                evaluation_timeseries = evaluation_timeseries[:limit]
        
        # Support vars
        results = {}
        from_t, to_t = set_from_t_and_to_t(from_dt, to_dt, from_t, to_t)
        warned = False

        # Log
        logger.info('Will evaluate model for %s steps ahead with metrics %s', steps, metrics)

        for steps_round in steps:
            
            # Support vars
            real_values = []
            model_values = []
            processed_samples = 0
    
            for key in timeseries.data_labels():
                
                # If the model has no window, evaluate on the entire time series
                if not self.data['window']:

                    # Note: steps_round is always equal to the entire test time series length in window-less model evaluation
     
                    # Create a time series where to apply the forecast, with only a point "in the past",
                    # this is done in order to use the apply function as is. Since the model is not using
                    # any window, the point data will be ignored and just used for its timestamp
                    forecast_timeseries = timeseries.__class__()
                    
                    # TODO: it should not be required to check .resolution type!
                    if isinstance(timeseries[0], Point):
                        if isinstance(timeseries.resolution, TimeUnit):
                            forecast_timeseries.append(timeseries[0].__class__(dt = timeseries[0].dt - timeseries.resolution,
                                                                               data = timeseries[0].data))                            
                        elif isinstance(timeseries.resolution, Unit):
                            forecast_timeseries.append(timeseries[0].__class__(dt = dt_from_s(timeseries[0].t - timeseries.resolution.value, tz=timeseries[0].tz),
                                                                               data = timeseries[0].data))
                        else:
                            forecast_timeseries.append(timeseries[0].__class__(dt = dt_from_s(timeseries[0].t - timeseries.resolution, tz=timeseries[0].tz),
                                                                               data = timeseries[0].data))                    
                    elif isinstance(timeseries[0], Slot):
                        if isinstance(timeseries.resolution, TimeUnit):
                            forecast_timeseries.append(timeseries[0].__class__(dt = timeseries[0].dt - timeseries.resolution,
                                                                               unit = timeseries.resolution,
                                                                               data = timeseries[0].data))                            
                        elif isinstance(timeseries.resolution, Unit):
                            forecast_timeseries.append(timeseries[0].__class__(dt = dt_from_s(timeseries[0].t - timeseries.resolution.value, tz=timeseries[0].tz),
                                                                               unit = timeseries.resolution,
                                                                               data = timeseries[0].data))
                        else:
                            forecast_timeseries.append(timeseries[0].__class__(dt = dt_from_s(timeseries[0].t - timeseries.resolution, tz=timeseries[0].tz),
                                                                               unit = timeseries.resolution,
                                                                               data = timeseries[0].data))
                    else:
                        raise TypeError('Unknown time series items type (got "{}"'.format(timeseries[0].__class__.__name__))

                    # Set default evaluate samples
                    evaluate_samples = len(timeseries)
                    
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
                    self._apply(forecast_timeseries, steps=evaluate_samples, inplace=True)

                    # Save the model and the original value to be compared later on. Create the arrays by skipping the fist item
                    # and move through the forecast time series comparing with the input time series, shifted by one since in the
                    # forecast timeseries we added an "artificial" first point to use the apply()
                    for i in range(1, evaluate_samples+1):
                        
                        model_value = forecast_timeseries[i].data[key]
                        model_values.append(model_value)
                        
                        real_value = timeseries[i-1].data[key]
                        real_values.append(real_value)


                # Else, process in streaming the timeseries, item by item, and properly take into account the window.
                else:
                    for i in range(len(timeseries)):
    
                        # Skip if needed
                        try:
                            if not item_is_in_range(timeseries[i], from_t, to_t):
                                continue
                        except StopIteration:
                            break  
                    
                        # Check that we can get enough data
                        if i < self.data['window']:
                            continue
                        if i > (len(timeseries)-steps_round):
                            continue
                        
                        # Compute the various boundaries
                        original_timeseries_boundaries_start = i - (self.data['window']) 
                        original_timeseries_boundaries_end = i + steps_round
                        
                        original_forecast_timeseries_boundaries_start = original_timeseries_boundaries_start
                        original_forecast_timeseries_boundaries_end = original_timeseries_boundaries_end-steps_round
                        
                        # Create the time series where to apply the forecast
                        forecast_timeseries = timeseries.__class__()
                        for j in range(original_forecast_timeseries_boundaries_start, original_forecast_timeseries_boundaries_end):

                            if isinstance(timeseries[0], Point):
                                forecast_timeseries.append(timeseries[0].__class__(t = timeseries[j].t,
                                                                                   tz = timeseries[j].tz,
                                                                                   data = timeseries[j].data))                        
                            elif isinstance(timeseries[0], Slot):
                                forecast_timeseries.append(timeseries[0].__class__(start = timeseries[j].start,
                                                                                   end   = timeseries[j].end,
                                                                                   unit  = timeseries[j].unit,
                                                                                   data  = timeseries[j].data))                           
                            
                            # This would lead to add the forecasted index to the original data (and we don't want it)
                            #forecast_timeseries.append(timeseries[j])
     
                        # Apply the forecasting model
                        self._apply(forecast_timeseries, steps=steps_round, inplace=True)
    
                        # Plot results time series?
                        if plots:
                            forecast_timeseries.plot()
                        
                        # Save the model and the original value to be compared later on
                        for step in range(steps_round):
                            original_index = original_timeseries_boundaries_start + self.data['window'] + step
    
                            forecast_index = self.data['window'] + step
    
                            model_value = forecast_timeseries[forecast_index].data[key]
                            model_values.append(model_value)
                            
                            real_value = timeseries[original_index].data[key]
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
                if evaluation_timeseries:
                    for i in range(len(model_values)):
                        evaluation_timeseries[i].data['{}_AE'.format(key)] = abs(real_values[i] - model_values[i])  
            if 'MAPE' in metrics:
                results['MAPE_{}_steps'.format(steps_round)] = mean_absolute_percentage_error(real_values, model_values)
                if evaluation_timeseries:
                    for i in range(len(model_values)):
                        evaluation_timeseries[i].data['{}_APE'.format(key)] = abs(real_values[i] - model_values[i]) / real_values[i]
            
            if evaluation_timeseries:
                for i in range(len(model_values)):
                    evaluation_timeseries[i].data['{}_pred'.format(key)] = model_values[i]
        
        # Compute overall RMSE
        if 'RMSE' in metrics:
            sum_rmse = 0
            count = 0
            for key in results:
                if key.startswith('RMSE_'):
                    sum_rmse += results[key]
                    count += 1
            results['RMSE'] = sum_rmse/count

        # Compute overall MAE
        if 'MAE' in metrics:
            sum_me = 0
            count = 0
            for key in results:
                if key.startswith('MAE_'):
                    sum_me += results[key]
                    count += 1
            results['MAE'] = sum_me/count

        # Compute overall MAPE
        if 'MAPE' in metrics:
            sum_me = 0
            count = 0
            for key in results:
                if key.startswith('MAPE_'):
                    sum_me += results[key]
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

        # Handle evaluation timeseries if required
        if return_evaluation_timeseries:
            results['evaluation_timeseries'] = evaluation_timeseries 
        if plot:
            evaluation_timeseries.plot()   

        # Return evaluation results if any
        if results:
            return results


#=========================
#  P. Average Forecaster
#=========================

class PeriodicAverageForecaster(Forecaster):
    """A forecaster based on periodic averages.
    
    Args:
        path (str): a path from which to load a saved model. Will override all other init settings.
        window (int): the window length. If set to ``auto``, then it will be automatically handled based on the time series periodicity.
    """

    def __init__(self, path=None, window='auto'):

        # Set window
        if window != 'auto':
            try:
                int(window)
            except:
                raise ValueError('Got a non-integer window ({})'.format(window)) 
        self.window = window

        # Call parent init        
        super(PeriodicAverageForecaster, self).__init__(path=path)

        # If loaded (fitted), convert the average dict keys back to integers
        if self.fitted:
            self.data['averages'] = {int(key):value for key, value in self.data['averages'].items()}
        

    def fit(self, timeseries, periodicity='auto', dst_affected=False, from_t=None, to_t=None, from_dt=None, to_dt=None):
        """Fit the forecaster on a time series.

        Args:
            periodicity(int): the periodicty of the time series. If set to ``auto`` then it will be automatically detected using a FFT.
            dst_affected(bool): if the model should take into account DST effects.
            from_t(float): fit starting epoch timestamp.
            to_t(float): fit ending epoch timestamp
            from_dt(datetime): fit starting datetime.
            to_dt(datetime) : fit ending datetime.
        """
        return super(PeriodicAverageForecaster, self).fit(timeseries, periodicity, dst_affected, from_t, to_t, from_dt, to_dt)
        
    def _fit(self, timeseries, periodicity='auto', dst_affected=False, from_t=None, to_t=None, from_dt=None, to_dt=None):

        if len(timeseries.data_labels()) > 1:
            raise NotImplementedError('Multivariate time series are not yet supported')

        from_t, to_t = set_from_t_and_to_t(from_dt, to_dt, from_t, to_t)

        # Set or detect periodicity
        if periodicity == 'auto':        
            periodicity =  get_periodicity(timeseries)
            logger.info('Detected periodicity: %sx %s', periodicity, timeseries.resolution)
                
        self.data['periodicity']  = periodicity
        self.data['dst_affected'] = dst_affected

        # Set window
        if self.window != 'auto':
            self.data['window'] = self.window
        else:
            logger.info('Using a window of "{}"'.format(periodicity))
            self.data['window'] = periodicity

        for key in timeseries.data_labels():
            sums   = {}
            totals = {}
            processed = 0
            for item in timeseries:

                # Skip if needed
                try:
                    if not item_is_in_range(item, from_t, to_t):
                        continue                  
                except StopIteration:
                    break
                
                # Process
                periodicity_index = get_periodicity_index(item, timeseries.resolution, periodicity, dst_affected)
                if not periodicity_index in sums:
                    sums[periodicity_index] = item.data[key]
                    totals[periodicity_index] = 1
                else:
                    sums[periodicity_index] += item.data[key]
                    totals[periodicity_index] +=1
                processed += 1

        averages={}
        for periodicity_index in sums:
            averages[periodicity_index] = sums[periodicity_index]/totals[periodicity_index]
        self.data['averages'] = averages
        
        logger.debug('Processed %s items', processed)


    def _predict(self, timeseries, steps=1, forecast_start=None):
        
        # TODO: remove the forecast_start or move it in the parent(s).
      
        # Univariate is enforced by the fit
        key = self.data['data_labels'][0]
      
        # Set forecast starting item
        if forecast_start is None:
            forecast_start = len(timeseries) - 1
            
        # Get forecast start item
        forecast_start_item = timeseries[forecast_start]

        # Support vars
        forecast_timestamps = []
        forecast_data = []

        # Compute the offset (avg diff between the real values and the forecasts on the first window)
        diffs  = 0                
        for j in range(self.data['window']):
            serie_index = forecast_start - self.data['window'] + j
            real_value = timeseries[serie_index].data[key]
            forecast_value = self.data['averages'][get_periodicity_index(timeseries[serie_index], timeseries.resolution, self.data['periodicity'], dst_affected=self.data['dst_affected'])]
            diffs += (real_value - forecast_value)            

        # Sum the avg diff between the real and the forecast on the window to the forecast (the offset)
        offset = diffs/j

        # Perform the forecast
        for i in range(steps):
            step = i + 1

            # Set forecast timestamp
            if isinstance(timeseries[0], Slot):
                try:
                    forecast_timestamp = forecast_timestamps[-1] + timeseries.resolution
                    forecast_timestamps.append(forecast_timestamp)
                except IndexError:
                    forecast_timestamp = forecast_start_item.end
                    forecast_timestamps.append(forecast_timestamp)

            else:
                forecast_timestamp = TimePoint(t = forecast_start_item.t + (timeseries.resolution.as_seconds()*step), tz = forecast_start_item.tz )
    
            # Compute the real forecast data
            periodicity_index = get_periodicity_index(forecast_timestamp, timeseries.resolution, self.data['periodicity'], dst_affected=self.data['dst_affected'])        
            forecast_data.append({key: self.data['averages'][periodicity_index] + (offset*1.0)})
        
        # Return
        return forecast_data

    
    def _plot_averages(self, timeseries, **kwargs):      
        averages_timeseries = copy.deepcopy(timeseries)
        for item in averages_timeseries:
            value = self.data['averages'][get_periodicity_index(item, averages_timeseries.resolution, self.data['periodicity'], dst_affected=self.data['dst_affected'])]
            if not value:
                value = 0
            item.data['periodic_average'] =value 
        averages_timeseries.plot(**kwargs)



#=========================
#  Prophet Forecaster
#=========================

class ProphetForecaster(Forecaster, ProphetModel):
    """A forecaster based on Prophet. Prophet (from Facebook) implements a procedure for forecasting time series data based
    on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. 
    
    Args:
        path (str): a path from which to load a saved model. Will override all other init settings.
    """

    def fit(self, timeseries, from_t=None, to_t=None, from_dt=None, to_dt=None):
        """Fit the forecaster on a time series.
     
            Args:
                from_t(float): fit starting epoch timestamp.
                to_t(float): fit ending epoch timestamp
                from_dt(datetime): fit starting datetime.
                to_dt(datetime) : fit ending datetime.
        """
        return super(ProphetForecaster, self).fit(timeseries, from_t, to_t, from_dt, to_dt)
 
    def _fit(self, timeseries, from_t=None, to_t=None, from_dt=None, to_dt=None):

        if len(timeseries.data_labels()) > 1:
            raise Exception('Multivariate time series are not yet supported')

        from fbprophet import Prophet

        from_t, to_t = set_from_t_and_to_t(from_dt, to_dt, from_t, to_t)

        data = self._from_timeseria_to_prophet(timeseries, from_t=from_t, to_t=to_t)

        # Instantiate the Prophet model
        self.prophet_model = Prophet()
        
        # Fit tjhe Prophet model
        self.prophet_model.fit(data)

        # Save the timeseries we used for the fit.
        self.fit_timeseries = timeseries
        
        # Prophet, as the ARIMA models, has no window
        self.data['window'] = 0


    def _predict(self, timeseries, steps=1):

        key = self.data['data_labels'][0]

        # Prepare a dataframe with all the timestamps to forecast
        last_item    = timeseries[-1]
        last_item_t  = last_item.t
        last_item_dt = last_item.dt
        data_to_forecast = []
        
        for _ in range(steps):
            new_item_dt = last_item_dt + timeseries.resolution
            data_to_forecast.append(self._remove_timezone(new_item_dt))
            last_item_dt = new_item_dt

        dataframe_to_forecast = DataFrame(data_to_forecast, columns = ['ds'])
                    
        # Call Prophet predict 
        forecast = self.prophet_model.predict(dataframe_to_forecast)
        #forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
            
        # Re arrange predict results
        forecasted_items = []
        for i in range(steps):
            forecasted_items.append({key: float(forecast['yhat'][i])})

        # Return
        return forecasted_items      



#=========================
#  ARIMA Forecaster
#=========================

class ARIMAForecaster(Forecaster, ARIMAModel):
    """A forecaster based on ARIMA. AutoRegressive Integrated Moving Average models are a generalization of an 
    AutoRegessive Moving Average (ARMA) model, whoch provide a description of a (weakly) stationary stochastic
    process in terms of two polynomials, one for the autoregression (AR) and the second for the moving average (MA).
    The I indicates that the data values have been replaced with the difference between their values and the previous values.
    
    Args:
        path (str): a path from which to load a saved model. Will override all other init settings.
        p(int): the order of the AR term.
        d(int): the number of differencing required to make the time series stationary.
        q(int): the order of the MA term.
    """

    def __init__(self, path=None, p=1,d=1,q=0): #p=5,d=2,q=5
        if (p,d,q) == (1,1,0):
            logger.info('You are using ARIMA\'s defaults of p=1, d=1, q=0. You might want to set them to more suitable values when initializing the model.')
        self.p = p
        self.d = d
        self.q = q
        # TODO: save the above in data[]?
        super(ARIMAForecaster, self).__init__(path)


    def _fit(self, timeseries):

        import statsmodels.api as sm

        if len(timeseries.data_labels()) > 1:
            raise Exception('Multivariate time series require to have the key of the prediction specified')
        key=timeseries.data_labels()[0]
                            
        data = array(timeseries.df[key])
        
        # Save model and fit
        self.model = sm.tsa.ARIMA(data, (self.p,self.d,self.q))
        self.model_res = self.model.fit()
        
        # Save the timeseries we used for the fit.
        self.fit_timeseries = timeseries
        
        # The ARIMA models, as Prophet, have no window
        self.data['window'] = 0
        
        
    def _predict(self, timeseries, steps=1):

        key = self.data['data_labels'][0]

        # Chack that we are applying on a time series ending with the same datapoint where the fit timeseries was
        if self.fit_timeseries[-1].t != timeseries[-1].t:
            raise NonContiguityError('Sorry, this model can be applied only on a time series ending with the same timestamp as the time series used for the fit.')

        # Return the predicion. We need the [0] to access yhat, other indexes are erorrs etc.
        return [{key: value} for value in self.model_res.forecast(steps)[0]] 



#=========================
#  AARIMA Forecaster
#=========================

class AARIMAForecaster(Forecaster):
    """A forecaster based on Auto-ARIMA. Auto-ARIMA mdoels set automatically the best values for the
    p, d and q parameters, trying different values and checking which ones perform better.
    
    Args:
        path (str): a path from which to load a saved model. Will override all other init settings.
    """
    def _fit(self, timeseries, **kwargs):
        
        import pmdarima as pm

        if len(timeseries.data_labels()) > 1:
            raise Exception('Multivariate time series require to have the key of the prediction specified')
        key=timeseries.data_labels()[0]
                            
        data = array(timeseries.df[key])

        # Change some defaults
        trace = kwargs.pop('trace', False)
        error_action = kwargs.pop('error_action', 'ignore')
        suppress_warnings = kwargs.pop('suppress_warnings', True)
        stepwise = kwargs.pop('stepwise', True) 

        # See https://alkaline-ml.com/pmdarima/_modules/pmdarima/arima/auto.html for the other defaults
        
        # Call the pmdarima aut_arima function
        autoarima_model = pm.auto_arima(data, error_action=error_action,  
                                        suppress_warnings=suppress_warnings, 
                                        stepwise=stepwise, trace=trace, **kwargs)

        autoarima_model.summary()

        self.model = autoarima_model
        
        # Save the timeseries we used for the fit.
        self.fit_timeseries = timeseries

        # The ARIMA models, as Prophet, have no window
        self.data['window'] = 0
        

    def _predict(self, timeseries, steps=1):

        key = self.data['data_labels'][0]

        # Chack that we are applying on a time series ending with the same datapoint where the fit timeseries was
        if self.fit_timeseries[-1].t != timeseries[-1].t:
            raise NonContiguityError('Sorry, this model can be applied only on a time series ending with the same timestamp as the time series used for the fit.')

        # Return the predicion. We need the [0] to access yhat, other indexes are erorrs etc.
        return [{key: value} for value in self.model.predict(steps)]



#=========================
#  LSTM Forecaster
#=========================

class LSTMForecaster(KerasModel, Forecaster):
    """A LSTM-based forecaster. LSTMs are artificial neutral networks particulary well suited for time series forecasting tasks.

    Args:
        path(str): a path from which to load a saved model. Will override all other init settings.
        window (int): the window length.
        features(list): which features to use. Supported values are:
            ``values`` (use the data values), 
            ``diffs``  (use the diffs between the values), and 
            ``hours``  (use the hours of the timestamp). 
        neurons(int): how many neaurons to use for the LSTM neural nework hidden layer.
        keras_model(keras.Model) : an optional external Keras Model architecture. Will cause the ``neurons`` argument to be discarded.
    """
        
    def __init__(self, path=None, window=3, features=['values'], neurons=128, keras_model=None):

        super(LSTMForecaster, self).__init__(path=path)
        
        # Did the init load a model?
        try:
            if self.fitted:
                
                # Load the kears model as well
                self._load_keras_model(path)
        
                # No need to proceed further 
                return
        except AttributeError:
            pass
        
        if not window ==3:
            logger.info('Using default window size of 3')
       
        if features == ['values']:
            logger.info('Using default features: values')
        
        # Set window, neurons, features
        self.data['window'] = window
        self.data['neurons'] = neurons
        self.data['features'] = features
        
        # Set external model architecture if any
        self.keras_model = keras_model

    def save(self, path):

        # Call parent save
        super(LSTMForecaster, self).save(path)

        # Now save the Keras model itself
        self._save_keras_model(path)

    def _fit(self, timeseries, from_t=None, to_t=None, from_dt=None, to_dt=None, verbose=False, epochs=30, normalize=True):

        # Set from and to
        from_t, to_t = set_from_t_and_to_t(from_dt, to_dt, from_t, to_t)

        # Set verbose switch
        if verbose:
            verbose=1
        else:
            verbose=0

        # Data keys shortcut
        data_labels = timeseries.data_labels()

        if normalize:
            # Set min and max
            min_values = timeseries.min()
            max_values = timeseries.max()
            
            # Fix some debeatable behaviour (which is, that min and max return different things for univariate and multivariate data)
            # TODO: fix me!
            if not isinstance(min_values, dict):
                min_values = {timeseries.data_labels()[0]:min_values}
            if not isinstance(max_values, dict):
                max_values = {timeseries.data_labels()[0]:max_values}
            
            # Normalize data
            timeseries_normalized = timeseries.duplicate()
            for datapoint in timeseries_normalized:
                for data_label in datapoint.data:
                    datapoint.data[data_label] = (datapoint.data[data_label] - min_values[data_label]) / (max_values[data_label] - min_values[data_label])
        
            # Store normalization factors
            self.data['min_values'] = min_values
            self.data['max_values'] = max_values
        
        else:
            # TODO: here the name is worn,
            timeseries_normalized = timeseries

        # Move to "matrix" of windows plus "vector" of targets data representation. Or, in other words:
        # window_datapoints is a list of lists (matrix) where each nested list (row) is a list of window datapoints.
        window_datapoints_matrix = self._to_window_datapoints_matrix(timeseries_normalized, window=self.data['window'], steps=1)
        target_values_vector = self._to_target_values_vector(timeseries_normalized, window=self.data['window'], steps=1)

        # Compute window features
        window_features = []
        for window_datapoints in window_datapoints_matrix:
            window_features.append(self._compute_window_features(window_datapoints,
                                                                data_labels = data_labels,
                                                                features=self.data['features']))

        # Obtain the number of features based on _compute_window_features() output
        features_per_window_item = len(window_features[0][0])
        output_dimension = len(target_values_vector[0])
        
        # Create the default model architeture if not given in the init
        if not self.keras_model:
            from keras.models import Sequential
            from keras.layers import Dense
            from keras.layers import LSTM
            self.keras_model = Sequential()
            self.keras_model.add(LSTM(self.data['neurons'], input_shape=(self.data['window'], features_per_window_item)))
            self.keras_model.add(Dense(output_dimension)) 
            self.keras_model.compile(loss='mean_squared_error', optimizer='adam')

        # Fit
        self.keras_model.fit(array(window_features), array(target_values_vector), epochs=epochs, verbose=verbose)


    def _predict(self, timeseries, steps=1, verbose=False):

        if steps>1:
            raise NotImplementedError('This forecaster does not support multi-step predictions.')

        # Set verbose switch
        if verbose:
            verbose=1
        else:
            verbose=0

        # Get the window if we were given a longer timeseries
        window_timeseries = timeseries[-self.data['window']:]
        
        # Duplicate so that we ae free to normalize in-place at the next step
        window_timeseries = window_timeseries.duplicate()
        
        # Normalize window data if we have to do so
        try:
            self.data['min_values']   
        except:
            normalize = False
        else:
            normalize = True
            for datapoint in window_timeseries:
                for data_label in datapoint.data:
                    datapoint.data[data_label] = (datapoint.data[data_label] - self.data['min_values'][data_label]) / (self.data['max_values'][data_label] - self.data['min_values'][data_label])

        # Compute window features
        window_features = self._compute_window_features(window_timeseries, data_labels=self.data['data_labels'], features=self.data['features'])

        # Perform the predict and set prediction data
        yhat = self.keras_model.predict(array([window_features]), verbose=verbose)

        predicted_data = {}
        for i, data_label in enumerate(self.data['data_labels']):
            
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



