import os
import json
import uuid
import copy
import statistics
from ..datastructures import DataTimeSlotSeries, DataTimeSlot, TimePoint, DataTimePointSeries, DataTimePoint, Slot, Point
from ..exceptions import NotFittedError, NonContiguityError, InputException
from ..utilities import get_periodicity, is_numerical, set_from_t_and_to_t, item_is_in_range
from ..utilities import check_timeseries, check_resolution, check_data_keys
from ..time import now_t, dt_from_s, s_from_dt
from datetime import timedelta, datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
from ..units import Unit, TimeUnit
from pandas import DataFrame
from numpy import array
from math import sqrt
from copy import deepcopy
from collections import OrderedDict
import shutil

# Keras and sklearn
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras import optimizers
from keras.models import load_model as load_keras_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Setup logging
import logging
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings as default behavior
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Base models and utilities
from .base import TimeSeriesParametricModel, ProphetModel, ARIMAModel, KerasModel
from .base import get_periodicity_index, mean_absolute_percentage_error


#=========================
#  Generic Reconstructor
#=========================

class Reconstructor(TimeSeriesParametricModel):

    def _predict(self, *args, **kwargs):
        raise NotImplementedError('Reconstructors can be used only from the apply() method.') from None


    def _apply(self, timeseries, remove_data_loss=False, data_loss_threshold=1, inplace=False):
        logger.debug('Using data_loss_threshold="%s"', data_loss_threshold)

        # TODO: understand if we want the apply from/to behavior. For now it is disabled
        # (add from_t=None, to_t=None, from_dt=None, to_dt=None in the function call above)
        # from_t, to_t = set_from_t_and_to_t(from_dt, to_dt, from_t, to_t)
        # Maybe also add a timeseries.mark=[from_dt, to_dt]
         
        from_t = None
        to_t   = None
        
        if not inplace:
            timeseries = timeseries.duplicate()

        if len(timeseries.data_keys()) > 1:
            raise NotImplementedError('Multivariate time series are not yet supported')

        for key in timeseries.data_keys():
            
            gap_started = None
            
            for i, item in enumerate(timeseries):
                
                # Skip if before from_t/dt of after to_t/dt
                if from_t is not None and timeseries[i].t < from_t:
                    continue
                try:
                    # Handle slots
                    if to_t is not None and timeseries[i].end.t > to_t:
                        break
                except AttributeError:
                    # Handle points
                    if to_t is not None and timeseries[i].t > to_t:
                        break                

                if item.data_loss >= data_loss_threshold:
                    # This is the beginning of an area we want to reconstruct according to the data_loss_threshold
                    if gap_started is None:
                        gap_started = i
                else:
                    
                    if gap_started is not None:
                    
                        # Reconstruct for this gap
                        self._reconstruct(from_index=gap_started, to_index=i, timeseries=timeseries, key=key)
                        gap_started = None
                    
                    item._data_reconstructed = 0
                    
                if remove_data_loss:
                    item._data_loss = 0
            
            # Reconstruct the last gap as well if left "open"
            if gap_started is not None:
                self._reconstruct(from_index=gap_started, to_index=i+1, timeseries=timeseries, key=key)

        if not inplace:
            return timeseries
        else:
            return None


    def _evaluate(self, timeseries, steps='auto', limit=None, data_loss_threshold=1, metrics=['RMSE', 'MAE'], details=False, from_t=None, to_t=None, from_dt=None, to_dt=None):

        # Set evaluation_score steps if we have to
        if steps == 'auto':
            try:
                steps = [1, self.data['periodicity']]
            except KeyError:
                steps = [1, 2, 3]
        elif isinstance(steps, list):
            pass
        else:
            steps = list(range(1, steps+1))
         
        # Support vars
        evaluation_score = {}
        from_t, to_t = set_from_t_and_to_t(from_dt, to_dt, from_t, to_t)
        warned = False
        
        # Log
        logger.info('Will evaluate model for %s steps with metrics %s', steps, metrics)
        
        # Find areas where to evaluate the model
        for key in timeseries.data_keys():
             
            for steps_round in steps:
                
                # Support vars
                real_values = []
                reconstructed_values = []
                processed_samples = 0

                # Here we will have steps=1, steps=2 .. steps=n          
                logger.debug('Evaluating model for %s steps', steps_round)
                
                for i in range(len(timeseries)):

                    # Skip if needed
                    try:
                        if not item_is_in_range(timeseries[i], from_t, to_t):
                            continue
                    except StopIteration:
                        break                  
                
                    # Skip the first and the last ones, otherwise reconstruct the ones in the middle
                    if (i == 0) or (i >= len(timeseries)-steps_round):
                        continue

                    # Is this a "good area" where to test or do we have to stop?
                    stop = False
                    if timeseries[i-1].data_loss >= data_loss_threshold:
                        stop = True
                    for j in range(steps_round):
                        if timeseries[i+j].data_loss >= data_loss_threshold:
                            stop = True
                            break
                    if timeseries[i+steps_round].data_loss >= data_loss_threshold:
                        stop = True
                    if stop:
                        continue
                            
                    # Set prev and next
                    prev_value = timeseries[i-1].data[key]
                    next_value = timeseries[i+steps_round].data[key]
                    
                    # Compute average value
                    average_value = (prev_value+next_value)/2
                    
                    # Data to be reconstructed
                    timeseries_to_reconstruct = timeseries.__class__()
                    
                    # Append prev
                    #timeseries_to_reconstruct.append(copy.deepcopy(timeseries[i-1]))
                    
                    # Append in the middle and store real values
                    for j in range(steps_round):
                        item = copy.deepcopy(timeseries[i+j])
                        # Set the data_loss to one so the item will be reconstructed
                        item._data_loss = 1
                        item.data[key] = average_value
                        timeseries_to_reconstruct.append(item)
                        
                        real_values.append(timeseries[i+j].data[key])
              
                    # Append next
                    #timeseries_to_reconstruct.append(copy.deepcopy(timeseries[i+steps_round]))
                    
                    # Do we have a 1-point only timeseries? If so, manually set the resolution
                    # as otherwise it would be not defined. # TODO: does it make sense?
                    if len(timeseries_to_reconstruct) == 1:
                        timeseries_to_reconstruct._resolution = timeseries._resolution

                    # Apply model inplace
                    self._apply(timeseries_to_reconstruct, inplace=True)
                    processed_samples += 1

                    # Store reconstructed values
                    for j in range(steps_round):
                        reconstructed_values.append(timeseries_to_reconstruct[j].data[key])
                    
                    # Break if we have to
                    if limit is not None and processed_samples >= limit:
                        break
                    
                    # Warn if no limit given and we are over
                    if not limit and not warned and i > 10000:
                        logger.warning('No limit set in the evaluation with a quite long time series, this could take some time.')
                        warned=True
                        
                if limit and processed_samples < limit:
                    logger.warning('The evaluation limit is set to "{}" but I have only "{}" samples for "{}" steps'.format(limit, processed_samples, steps_round))

                if not reconstructed_values:
                    raise Exception('Could not evaluate model, maybe not enough data?')

                # Compute RMSE and ME, and add to the evaluation_score
                if 'RMSE' in metrics:
                    evaluation_score['RMSE_{}_steps'.format(steps_round)] = sqrt(mean_squared_error(real_values, reconstructed_values))
                if 'MAE' in metrics:
                    evaluation_score['MAE_{}_steps'.format(steps_round)] = mean_absolute_error(real_values, reconstructed_values)
                if 'MAPE' in metrics:
                    evaluation_score['MAPE_{}_steps'.format(steps_round)] = mean_absolute_percentage_error(real_values, reconstructed_values)

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


    def _reconstruct(self, *args, **krargs):
        raise NotImplementedError('Reconstruction for this model is not yet implemented')



#=========================
# P. Average Reconstructor
#=========================

class PeriodicAverageReconstructor(Reconstructor):


    def _fit(self, timeseries, data_loss_threshold=0.5, periodicity=None, dst_affected=False, from_t=None, to_t=None, from_dt=None, to_dt=None, offset_method='average'):

        if not offset_method in ['average', 'extremes']:
            raise Exception('Unknown offset method "{}"'.format(self.offset_method))
        self.offset_method = offset_method
    
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
                if item.data_loss < data_loss_threshold:
                    periodicity_index = get_periodicity_index(item, timeseries._resolution, periodicity, dst_affected=dst_affected)
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
        
        logger.debug('Processed "%s" items', processed)


    def _reconstruct(self, timeseries, key, from_index, to_index):
        logger.debug('Reconstructing between "{}" and "{}"'.format(from_index, to_index-1))

        # Compute offset (old approach)
        if self.offset_method == 'average':
            diffs=0
            for j in range(from_index, to_index):
                real_value = timeseries[j].data[key]
                periodicity_index = get_periodicity_index(timeseries[j], timeseries._resolution, self.data['periodicity'], dst_affected=self.data['dst_affected'])
                reconstructed_value = self.data['averages'][periodicity_index]
                diffs += (real_value - reconstructed_value)
            offset = diffs/(to_index-from_index)
        
        elif self.offset_method == 'extremes':
            # Compute offset (new approach)
            diffs=0
            try:
                for j in [from_index-1, to_index+1]:
                    real_value = timeseries[j].data[key]
                    periodicity_index = get_periodicity_index(timeseries[j], timeseries._resolution, self.data['periodicity'], dst_affected=self.data['dst_affected'])
                    reconstructed_value = self.data['averages'][periodicity_index]
                    diffs += (real_value - reconstructed_value)
                offset = diffs/2
            except IndexError:
                offset=0
        else:
            raise Exception('Unknown offset method "{}"'.format(self.offset_method))

        # Actually reconstruct
        for j in range(from_index, to_index):
            item_to_reconstruct = timeseries[j]
            periodicity_index = get_periodicity_index(item_to_reconstruct, timeseries._resolution, self.data['periodicity'], dst_affected=self.data['dst_affected'])
            item_to_reconstruct.data[key] = self.data['averages'][periodicity_index] + offset
            item_to_reconstruct._data_reconstructed = 1
                        

    def _plot_averages(self, timeseries, **kwargs):   
        averages_timeseries = copy.deepcopy(timeseries)
        for item in averages_timeseries:
            value = self.data['averages'][get_periodicity_index(item, averages_timeseries._resolution, self.data['periodicity'], dst_affected=self.data['dst_affected'])]
            if not value:
                value = 0
            item.data['average'] =value 
        averages_timeseries.plot(**kwargs)


#=========================
#  Prophet Reconstructor
#=========================

class ProphetReconstructor(Reconstructor, ProphetModel):
    
    def _fit(self, timeseries, from_t=None, to_t=None, from_dt=None, to_dt=None):

        from fbprophet import Prophet

        from_t, to_t = set_from_t_and_to_t(from_dt, to_dt, from_t, to_t)

        if len(timeseries.data_keys()) > 1:
            raise NotImplementedError('Multivariate time series are not yet supported')

        data = self.from_timeseria_to_prophet(timeseries, from_t, to_t)

        # Instantiate the Prophet model
        self.prophet_model = Prophet()
        
        # Fit tjhe Prophet model
        self.prophet_model.fit(data)


    def _reconstruct(self, timeseries, key, from_index, to_index):
        logger.debug('Reconstructing between "{}" and "{}"'.format(from_index, to_index-1))
    
        # Get and prepare data to reconstruct
        items_to_reconstruct = []
        for j in range(from_index, to_index):
            items_to_reconstruct.append(timeseries[j])
        data_to_reconstruct = [self.remove_timezone(dt_from_s(item.t)) for item in items_to_reconstruct]
        dataframe_to_reconstruct = DataFrame(data_to_reconstruct, columns = ['ds'])

        # Apply Prophet fit
        forecast = self.prophet_model.predict(dataframe_to_reconstruct)
        #forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

        # Ok, replace the values with the reconsturcted ones
        for i, j in enumerate(range(from_index, to_index)):
            #logger.debug('Reconstructing item #{} with reconstucted item #{}'.format(j,i))
            item_to_reconstruct = timeseries[j]
            item_to_reconstruct.data[key] = forecast['yhat'][i]
            item_to_reconstruct._data_reconstructed = 1

