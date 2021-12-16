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
except ImportError:
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

    def predict(self, timeseries, *args, **kwargs):
 
        # Check if the input timeseries is shorter than the window, if any.
        # However, nearly all forecasters use windows, at least of one point.
        try:
            if len(timeseries) < self.data['window']:
                raise ValueError('The timeseries length ({}) is shorter than the model window ({}), it must be at least equal.'.format(len(timeseries), self.data['window']))
        except KeyError:
            pass
        
        # Call parent predict
        return super(Forecaster, self).predict(timeseries, *args, **kwargs)


    def _evaluate(self, timeseries, steps='auto', limit=None, plots=False, metrics=['RMSE', 'MAE'], details=False, from_t=None, to_t=None, from_dt=None, to_dt=None):

        # Set evaluation_score steps if we have to
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
                            
        # Support vars
        evaluation_score = {}
        from_t, to_t = set_from_t_and_to_t(from_dt, to_dt, from_t, to_t)
        warned = False

        # Log
        logger.info('Will evaluate model for %s steps with metrics %s', steps, metrics)

        for steps_round in steps:
            
            # Support vars
            real_values = []
            model_values = []
            processed_samples = 0
    
            for key in timeseries.data_keys():
                
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
                    self._apply(forecast_timeseries, n=evaluate_samples, inplace=True)

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
                        if i < self.data['window']+steps_round:
                            continue
                        if i > (len(timeseries)-steps_round):
                            continue
    
                        # Compute the various boundaries
                        original_timeseries_boundaries_start = i - (self.data['window']) - steps_round
                        original_timeseries_boundaries_end = i
                        
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
                        self._apply(forecast_timeseries, n=steps_round, inplace=True)
    
                        # Plot evaluation_score time series?
                        if plots:
                            forecast_timeseries.plot(log_js=False)
                        
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

            # Compute RMSE and ME, and add to the evaluation_score
            if 'RMSE' in metrics:
                evaluation_score['RMSE_{}_steps'.format(steps_round)] = sqrt(mean_squared_error(real_values, model_values))
            if 'MAE' in metrics:
                evaluation_score['MAE_{}_steps'.format(steps_round)] = mean_absolute_error(real_values, model_values)
            if 'MAPE' in metrics:
                evaluation_score['MAPE_{}_steps'.format(steps_round)] = mean_absolute_percentage_error(real_values, model_values)

        # Compute overall RMSE
        if 'RMSE' in metrics:
            sum_rmse = 0
            count = 0
            for key in evaluation_score:
                if key.startswith('RMSE_'):
                    sum_rmse += evaluation_score[key]
                    count += 1
            evaluation_score['RMSE'] = sum_rmse/count

        # Compute overall MAE
        if 'MAE' in metrics:
            sum_me = 0
            count = 0
            for key in evaluation_score:
                if key.startswith('MAE_'):
                    sum_me += evaluation_score[key]
                    count += 1
            evaluation_score['MAE'] = sum_me/count

        # Compute overall MAPE
        if 'MAPE' in metrics:
            sum_me = 0
            count = 0
            for key in evaluation_score:
                if key.startswith('MAPE_'):
                    sum_me += evaluation_score[key]
                    count += 1
            evaluation_score['MAPE'] = sum_me/count
        
        if not details:
            simple_evaluation_score = {}
            if 'RMSE' in metrics:
                simple_evaluation_score['RMSE'] = evaluation_score['RMSE']
            if 'MAE' in metrics:
                simple_evaluation_score['MAE'] = evaluation_score['MAE']
            if 'MAPE' in metrics:
                simple_evaluation_score['MAPE'] = evaluation_score['MAPE']
            evaluation_score = simple_evaluation_score
            
        return evaluation_score


    def _apply(self, timeseries, n=1, inplace=False):


        input_timeseries_len = len(timeseries)
 
        if inplace:
            forecast_timeseries = timeseries
        else:
            forecast_timeseries = timeseries.duplicate()
        
        # Add the forecast index
        for item in forecast_timeseries:
            item.forecast = 0
        
        # Call model forecasting logic
        try:
            forecast_model_results = self.forecast(timeseries = forecast_timeseries, n=n)
            if not isinstance(forecast_model_results, list):
                forecast_timeseries.append(forecast_model_results)
            else:
                for item in forecast_model_results:
                    item.forecast = 1
                    forecast_timeseries.append(item)

        except NotImplementedError:
            
            for _ in range(n):
    
                # Call the forecast only on the last point
                forecast_model_results = self.forecast(timeseries = forecast_timeseries, n=1)

                # Add forecasted index
                forecast_model_results.forecast = 1

                # Add the forecast to the forecasts time series
                forecast_timeseries.append(forecast_model_results)
    
        # Do we have missing forecasts?
        if input_timeseries_len + n != len(forecast_timeseries):
            raise ValueError('There are missing forecasts. If your model does not support multi-step forecasting, raise a NotImplementedError if n>1 and Timeseria will handle it for you.')
 
        if not inplace:
            return forecast_timeseries
        else:
            return None


    def forecast(self, timeseries, n=1, forecast_start=None):

        # Set forecast starting item
        if forecast_start is not None:
            forecast_start_item = timeseries[forecast_start]
        else:
            forecast_start_item = timeseries[-1]
            
        # Handle forecast start
        if forecast_start is not None:
            try:
                predicted_data = self.predict(timeseries=timeseries, n=n, forecast_start=forecast_start)
            except TypeError as e:
                if 'unexpected keyword argument' and  'forecast_start' in str(e):
                    raise NotImplementedError('The model does not support the "forecast_start" parameter, cannot proceed')           
                else:
                    raise
        else:
            predicted_data = self.predict(timeseries=timeseries, n=n)
                
        # List of predictions or single prediction?
        if isinstance(predicted_data,list):
            forecast = []
            last_item = forecast_start_item
            for data in predicted_data:

                if isinstance(timeseries[0], Slot):
                    forecast.append(DataTimeSlot(start = last_item.end,
                                                 unit  = timeseries._resolution,
                                                 data_loss = None,
                                                 #tz = timeseries.tz,
                                                 data  = data))
                else:
                    forecast.append(DataTimePoint(t = last_item.t + timeseries._resolution.value,
                                                  tz = timeseries.tz,
                                                  data  = data))
                last_item = forecast[-1]
        else:
            if isinstance(timeseries[0], Slot):
                forecast = DataTimeSlot(start = forecast_start_item.end,
                                        unit  = timeseries._resolution,
                                        data_loss = None,
                                        #tz = timeseries.tz,
                                        data  = predicted_data)
            else:
                forecast = DataTimePoint(t = forecast_start_item.t + timeseries._resolution.value,
                                         tz = timeseries.tz,
                                         data  = predicted_data)
            
         
        return forecast



#=========================
#  P. Average Forecaster
#=========================

class PeriodicAverageForecaster(Forecaster):

    def __init__(self, path=None, id=None):
        
        super(PeriodicAverageForecaster, self).__init__(path=path, id=id)

        # If loaded (fitted), convert the average dict keys back to integers
        if self.fitted:
            self.data['averages'] = {int(key):value for key, value in self.data['averages'].items()}

        
    def _fit(self, timeseries, window=None, periodicity=None, dst_affected=False, from_t=None, to_t=None, from_dt=None, to_dt=None):

        if len(timeseries.data_keys()) > 1:
            raise NotImplementedError('Multivariate time series are not yet supported')

        from_t, to_t = set_from_t_and_to_t(from_dt, to_dt, from_t, to_t)

        # Set or detect periodicity
        if periodicity is None:        
            periodicity =  get_periodicity(timeseries)
            try:
                if isinstance(timeseries._resolution, TimeUnit):
                    logger.info('Detected periodicity: %sx %s', periodicity, timeseries._resolution)
                else:
                    logger.info('Detected periodicity: %sx %ss', periodicity, timeseries._resolution)
            except AttributeError:
                logger.info('Detected periodicity: %sx %ss', periodicity, timeseries._resolution)
                
        self.data['periodicity']  = periodicity
        self.data['dst_affected'] = dst_affected

        # Set or detect window
        if window:
            self.data['window'] = window
        else:
            logger.info('Using a window of "{}"'.format(periodicity))
            self.data['window'] = periodicity

        for key in timeseries.data_keys():
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
                periodicity_index = get_periodicity_index(item, timeseries._resolution, periodicity, dst_affected)
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


    def _predict(self, timeseries, n=1, forecast_start=None):
      
        # Univariate is enforced by the fit
        key = self.data['data_keys'][0]
      
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
            forecast_value = self.data['averages'][get_periodicity_index(timeseries[serie_index], timeseries._resolution, self.data['periodicity'], dst_affected=self.data['dst_affected'])]
            diffs += (real_value - forecast_value)            

        # Sum the avg diff between the real and the forecast on the window to the forecast (the offset)
        offset = diffs/j

        # Perform the forecast
        for i in range(n):
            step = i + 1

            # Set forecast timestamp
            if isinstance(timeseries[0], Slot):
                try:
                    if isinstance(timeseries._resolution, Unit):
                        forecast_timestamp = forecast_timestamps[-1] + timeseries._resolution
                    else:
                        forecast_timestamp = TimePoint(forecast_timestamps[-1].t + timeseries._resolution)
                        
                    forecast_timestamps.append(forecast_timestamp)
                except IndexError:
                    forecast_timestamp = forecast_start_item.end
                    forecast_timestamps.append(forecast_timestamp)

            else:
                forecast_timestamp = TimePoint(t = forecast_start_item.t + (timeseries._resolution*step), tz = forecast_start_item.tz )
    
            # Compute the real forecast data
            periodicity_index = get_periodicity_index(forecast_timestamp, timeseries._resolution, self.data['periodicity'], dst_affected=self.data['dst_affected'])        
            forecast_data.append({key: self.data['averages'][periodicity_index] + (offset*1.0)})
        
        # Return
        return forecast_data

    
    def _plot_averages(self, timeseries, **kwargs):      
        averages_timeseries = copy.deepcopy(timeseries)
        for item in averages_timeseries:
            value = self.data['averages'][get_periodicity_index(item, averages_timeseries._resolution, self.data['periodicity'], dst_affected=self.data['dst_affected'])]
            if not value:
                value = 0
            item.data['average'] =value 
        averages_timeseries.plot(**kwargs)



#=========================
#  Prophet Forecaster
#=========================

class ProphetForecaster(Forecaster, ProphetModel):
    '''Prophet (from Facebook) implements a procedure for forecasting time series data based on an additive 
model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects.
It works best with time series that have strong seasonal effects and several seasons of historical data.
Prophet is robust to missing data and shifts in the trend, and typically handles outliers well. 
'''

    def _fit(self, timeseries, from_t=None, to_t=None, from_dt=None, to_dt=None):

        if len(timeseries.data_keys()) > 1:
            raise Exception('Multivariate time series are not yet supported')

        from fbprophet import Prophet

        from_t, to_t = set_from_t_and_to_t(from_dt, to_dt, from_t, to_t)

        data = self.from_timeseria_to_prophet(timeseries, from_t=from_t, to_t=to_t)

        # Instantiate the Prophet model
        self.prophet_model = Prophet()
        
        # Fit tjhe Prophet model
        self.prophet_model.fit(data)

        # Save the timeseries we used for the fit.
        self.fit_timeseries = timeseries
        
        # Prophet, as the ARIMA models, has no window
        self.data['window'] = 0


    def _predict(self, timeseries, n=1):

        key = self.data['data_keys'][0]

        # Prepare a dataframe with all the timestamps to forecast
        last_item    = timeseries[-1]
        last_item_t  = last_item.t
        last_item_dt = last_item.dt
        data_to_forecast = []
        
        for _ in range(n):
            if isinstance (timeseries._resolution, TimeUnit):
                new_item_dt = last_item_dt + timeseries._resolution
                data_to_forecast.append(self.remove_timezone(new_item_dt))
                last_item_dt = new_item_dt
            else:
                new_item_t = last_item_t + timeseries._resolution
                new_item_dt = dt_from_s(new_item_t, tz=timeseries.tz)
                data_to_forecast.append(self.remove_timezone(new_item_dt))
                last_item_t = new_item_t  
        dataframe_to_forecast = DataFrame(data_to_forecast, columns = ['ds'])
                    
        # Call Prophet predict 
        forecast = self.prophet_model.predict(dataframe_to_forecast)
        #forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
            
        # Re arrange predict results
        forecasted_items = []
        for i in range(n):
            forecasted_items.append({key: float(forecast['yhat'][i])})

        # Return
        return forecasted_items      



#=========================
#  ARIMA Forecaster
#=========================

class ARIMAForecaster(Forecaster, ARIMAModel):

    def __init__(self, p=1,d=1,q=0): #p=5,d=2,q=5
        if (p,d,q) == (1,1,0):
            logger.info('You are using ARIMA\'s defaults of p=1, d=1, q=0. You might want to set them to more suitable values when initializing the model.')
        self.p = p
        self.d = d
        self.q = q
        # TODO: save the above in data[]?
        super(ARIMAForecaster, self).__init__()


    def _fit(self, timeseries):

        import statsmodels.api as sm

        if len(timeseries.data_keys()) > 1:
            raise Exception('Multivariate time series require to have the key of the prediction specified')
        key=timeseries.data_keys()[0]
                            
        data = array(timeseries.df[key])
        
        # Save model and fit
        self.model = sm.tsa.ARIMA(data, (self.p,self.d,self.q))
        self.model_res = self.model.fit()
        
        # Save the timeseries we used for the fit.
        self.fit_timeseries = timeseries
        
        # The ARIMA models, as Prophet, have no window
        self.data['window'] = 0
        
        
    def _predict(self, timeseries, n=1):

        key = self.data['data_keys'][0]

        # Chack that we are applying on a time series ending with the same datapoint where the fit timeseries was
        if self.fit_timeseries[-1].t != timeseries[-1].t:
            raise NonContiguityError('Sorry, this model can be applied only on a time series ending with the same timestamp as the time series used for the fit.')

        # Return the predicion. We need the [0] to access yhat, other indexes are erorrs etc.
        return [{key: value} for value in self.model_res.forecast(n)[0]] 



#=========================
#  AARIMA Forecaster
#=========================

class AARIMAForecaster(Forecaster):

    def _fit(self, timeseries, **kwargs):
        
        import pmdarima as pm

        if len(timeseries.data_keys()) > 1:
            raise Exception('Multivariate time series require to have the key of the prediction specified')
        key=timeseries.data_keys()[0]
                            
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
        

    def _predict(self, timeseries, n=1):

        key = self.data['data_keys'][0]

        # Chack that we are applying on a time series ending with the same datapoint where the fit timeseries was
        if self.fit_timeseries[-1].t != timeseries[-1].t:
            raise NonContiguityError('Sorry, this model can be applied only on a time series ending with the same timestamp as the time series used for the fit.')

        # Return the predicion. We need the [0] to access yhat, other indexes are erorrs etc.
        return [{key: value} for value in self.model.predict(n)]



#=========================
#  LSTM Forecaster
#=========================

class LSTMForecaster(KerasModel, Forecaster):

    def __init__(self, path=None, id=None, window=None, features=None, neurons=128, keras_model=None):

        super(LSTMForecaster, self).__init__(path=path, id=id)
        
        # Did the init load a model?
        try:
            if self.fitted:
                # If so, no need to proceed
                return
        except AttributeError:
            pass
        
        if not window:
            logger.info('Using default window size of 3')
            window = 3
       
        if not features:
            logger.info('Using default features: values')
            features = ['values']
        
        # Set window, neurons, features
        self.data['window'] = window
        self.data['neurons'] = neurons
        self.data['features'] = features
        
        # Set external model architecture if any
        self.keras_model = keras_model


    def _fit(self, timeseries, from_t=None, to_t=None, from_dt=None, to_dt=None, verbose=False, epochs=30):

        # Set from and to
        from_t, to_t = set_from_t_and_to_t(from_dt, to_dt, from_t, to_t)

        # Set verbose switch
        if verbose:
            verbose=1
        else:
            verbose=0

        # Data keys shortcut
        data_keys = timeseries.data_keys()

        # Set min and max
        min_values = timeseries.min()
        max_values = timeseries.max()
        
        # Fix some debeatable behaviour
        if not isinstance(min_values, dict):
            min_values = {timeseries.data_keys()[0]:min_values}
        if not isinstance(max_values, dict):
            max_values = {timeseries.data_keys()[0]:max_values}
        
        # Normalize data
        timeseries_normalized = timeseries.duplicate()
        for datapoint in timeseries_normalized:
            for data_key in datapoint.data:
                datapoint.data[data_key] = (datapoint.data[data_key] - min_values[data_key]) / (max_values[data_key] - min_values[data_key])

        # Move to "matrix" of windows plus "vector" of targets data representation. Or, in other words:
        # window_datapoints is a list of lists (matrix) where each nested list (row) is a list of window datapoints.
        window_datapoints_matrix = self.to_window_datapoints_matrix(timeseries_normalized, window=self.data['window'], forecast_n=1)
        target_values_vector = self.to_target_values_vector(timeseries_normalized, window=self.data['window'], forecast_n=1)

        # Compute window features
        window_features = []
        for window_datapoints in window_datapoints_matrix:
            window_features.append(self.compute_window_features(window_datapoints,
                                                                data_keys = data_keys,
                                                                features=self.data['features']))

        # Obtain the number of features based on compute_window_features() output
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
        
        # Store data
        self.data['min_values'] = min_values
        self.data['max_values'] = max_values


    def _predict(self, timeseries, n=1, verbose=False):

        if n>1:
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
        
        # Normalize window data
        for datapoint in window_timeseries:
            for data_key in datapoint.data:
                datapoint.data[data_key] = (datapoint.data[data_key] - self.data['min_values'][data_key]) / (self.data['max_values'][data_key] - self.data['min_values'][data_key])

        # Compute window features
        window_features = self.compute_window_features(window_timeseries, data_keys=self.data['data_keys'], features=self.data['features'])

        # Perform the predict and set prediction data
        yhat = self.keras_model.predict(array([window_features]), verbose=verbose)

        predicted_data = {}
        for i, data_key in enumerate(self.data['data_keys']):
            
            # Get the prediction
            predicted_value_normalized = yhat[0][i]
        
            # De-normalize
            predicted_value = (predicted_value_normalized*(self.data['max_values'][data_key] - self.data['min_values'][data_key])) + self.data['min_values'][data_key]
            
            # Append to prediction data
            predicted_data[data_key] = predicted_value

        # Return
        return predicted_data



