import os
import json
import uuid
import copy
import statistics
from .datastructures import DataTimeSlotSeries, DataTimeSlot, TimePoint, DataTimePointSeries, DataTimePoint, Slot, Point
from .exceptions import NotFittedError, NonContiguityError, InputException
from .utilities import get_periodicity, is_numerical, set_from_t_and_to_t, item_is_in_range, check_timeseries
from .time import now_t, dt_from_s, s_from_dt
from datetime import timedelta, datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
from .units import Unit, TimeUnit
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
tf.logging.set_verbosity(tf.logging.ERROR)


HARD_DEBUG = False


#======================
#  Utility functions
#======================

def mean_absolute_percentage_error(list1, list2):
    '''Computes the MAPE, list 1 are true values, list2 arepredicted values'''
    if len(list1) != len(list2):
        raise ValueError('Lists have different lengths, cannot continue')
    p_error_sum = 0
    for i in range(len(list1)):
        p_error_sum += abs((list1[i] - list2[i])/list1[i])
    return p_error_sum/len(list1)


def get_periodicity_index(item, resolution, periodicity, dst_affected=False):

    # Handle specific cases
    if isinstance(resolution, TimeUnit):  
        resolution_s = resolution.duration_s(item.dt)
    elif isinstance(resolution, Unit):  
        if isinstance(resolution.value, list):
            raise NotImplementedError('Sorry, periodicty in multi-dimensional spaces are not defined')
        resolution_s = resolution.value
    else:
        if isinstance(resolution, list):
            raise NotImplementedError('Sorry, periodicty in multi-dimensional spaces are not defined')
        resolution_s = resolution

    # Compute periodicity index
    if not dst_affected:
    
        # Get index based on item timestamp, normalized to unit, modulus periodicity
        periodicity_index =  int(item.t / resolution_s) % periodicity
    
    else:

        # Get periodicity based on the datetime
        
        # Do we have an active DST?  
        dst_timedelta = item.dt.dst()
        
        if dst_timedelta.days == 0 and dst_timedelta.seconds == 0:
            # No DST
            periodicity_index = int(item.t / resolution_s) % periodicity
        
        else:
            # DST
            if dst_timedelta.days != 0:
                raise Exception('Don\'t know how to handle DST with days timedelta = "{}"'.format(dst_timedelta.days))

            if resolution_s > 3600:
                raise Exception('Sorry, this time series has not enough resolution to account for DST effects (resolution_s="{}", must be below 3600 seconds)'.format(resolution_s))
            
            # Get DST offset in seconds 
            dst_offset_s = dst_timedelta.seconds # 3600 usually

            # Compute the periodicity index
            periodicity_index = (int((item.t + dst_offset_s) / resolution_s) % periodicity)

    return periodicity_index



#======================
#  Base classes
#======================


class Model(object):
    '''A stateless model, or a white-box model. Exposes only predict(), apply() and evaluate() methods,
    since it is assumed that all the information is coded and nothing is learnt from the data.'''
    
    def __init__(self):
        pass

    
    def predict(self, data, *args, **kwargs):
        try:
            self._predict
        except AttributeError:
            raise NotImplementedError('Predicting from this model is not implemented')

        check_timeseries(data)
        
        return self._predict(data, *args, **kwargs)


    def apply(self, data, *args, **kwargs):
        try:
            self._apply
        except AttributeError:
            raise NotImplementedError('Applying this model is not implemented')

        check_timeseries(data)
        
        return self._apply(data, *args, **kwargs)


    def evaluate(self, data, *args, **kwargs):
        try:
            self._evaluate
        except AttributeError:
            raise NotImplementedError('Evaluating this model is not implemented')

        check_timeseries(data)
        
        return self._evaluate(data, *args, **kwargs)


class ParametricModel(Model):
    '''A stateful model with parameters, or a (partly) black-box model. Parameters can be set manually or learnt (fitted) from data.
On top of the predict(), apply() and evaluate() methods it provides a save() method to store the parameters of the model,
and optionally a fit() method if the parameters are to be learnt form data.'''
    
    def __init__(self, path=None, id=None):
        
        if path:
            with open(path+'/data.json', 'r') as f:
                self.data = json.loads(f.read())         
            self.fitted = True
            
            # Resolution as TimeUnit
            try:
                self.data['resolution'] = TimeUnit(self.data['resolution'])
            except InputException:
                # TODO: load as generic unit and make __eq__ work with both Units and TimeUnits?
                self.data['resolution'] = TimeUnit('{}s'.format(int(float(self.data['resolution']))))
                
        else:
            if not id:
                id = str(uuid.uuid4())
            self.fitted = False
            self.data = {'id': id}

        super(ParametricModel, self).__init__()

    
    def _check_resolution(self, data):
        
        # TODO: Fix this mess.. Make the .resolution behavior consistent!
        if self.data['resolution'] == data.resolution:
            return True
        try:
            if self.data['resolution'].value == data.resolution.duration_s():
                return True
        except:
            pass
        try:
            if self.data['resolution'].duration_s() == data.resolution.value:
                return True
        except:
            pass
        try:
            if self.data['resolution'].duration_s() == data.resolution:
                return True
        except:
            pass
        return False
                

    def predict(self, data, *args, **kwargs):

        try:
            self._predict
        except AttributeError:
            raise NotImplementedError('Predicting from this model is not implemented')

        if not self.fitted:
            raise NotFittedError()

        check_timeseries(data)

        if not self._check_resolution(data):
            raise ValueError('This model is fitted on "{}" resolution data, while your data has "{}" resolution.'.format(self.data['resolution'], data.resolution))
        
        return self._predict(data, *args, **kwargs)


    def apply(self, data, *args, **kwargs):

        try:
            self._apply
        except AttributeError:
            raise NotImplementedError('Applying this model is not implemented')

        if not self.fitted:
            raise NotFittedError()

        check_timeseries(data)

        if not self._check_resolution(data):
            raise ValueError('This model is fitted on "{}" resolution data, while your data has "{}" resolution.'.format(self.data['resolution'], data.resolution))
        
        return self._apply(data, *args, **kwargs)


    def evaluate(self, data, *args, **kwargs):

        try:
            self._evaluate
        except AttributeError:
            raise NotImplementedError('Evaluating this model is not implemented')

        if not self.fitted:
            raise NotFittedError()
        
        check_timeseries(data)
        
        return self._evaluate(data, *args, **kwargs)


    def fit(self, data, *args, **kwargs):

        try:
            self._fit
        except AttributeError:
            raise NotImplementedError('Fitting this model is not implemented')

        check_timeseries(data)
        
        fit_output = self._fit(data, *args, **kwargs)

        self.data['fitted_at'] = now_t()
        self.fitted = True
        self.data['resolution'] = data.resolution

            
        return fit_output

        
    def save(self, path):

        if not self.fitted:
            raise NotFittedError()

        # Dump as string representation fit data resolution, always in seconds if except if in calendar units
        try:
            if self.data['resolution'].type == TimeUnit.CALENDAR:
                self.data['resolution'] = str(self.data['resolution'])
            else:
                self.data['resolution'] = str(self.data['resolution'].duration_s())
        except AttributeError:
            self.data['resolution'] = str(self.data['resolution'])
        
        # Prepare model dir and dump data as json
        model_dir = '{}/{}'.format(path, self.data['id'])
        os.makedirs(model_dir)
        model_data_file = '{}/data.json'.format(model_dir)
        with open(model_data_file, 'w') as f:
            f.write(json.dumps(self.data))
        logger.info('Saved model with id "%s" in "%s"', self.data['id'], model_dir)
        return model_dir


    def cross_validate(self, data, *args, **kwargs):
        try:
            self._evaluate
        except AttributeError:
            raise NotImplementedError('Evaluating this model is not implemented')

        check_timeseries(data)

        # Decouple fit from validate args
        fit_kwargs = {}
        evaluate_kwargs = {}
        consumed_kwargs = []
        for kwarg in kwargs:
            if kwarg.startswith('fit_'):
                consumed_kwargs.append(kwarg)
                fit_kwargs[kwarg.replace('fit_', '')] = kwargs[kwarg]
            if kwarg.startswith('evaluate_'):
                consumed_kwargs.append(kwarg)
                evaluate_kwargs[kwarg.replace('evaluate_', '')] = kwargs[kwarg]
        for consumed_kwarg in consumed_kwargs:
            kwargs.pop(consumed_kwarg)

        # For readability
        timeseries = data
        
        # How many rounds
        rounds = kwargs.pop('rounds', 10)

        # Do we still have some kwargs?
        if kwargs:
            raise Exception('Got some unknown args: {}'.format(kwargs))
            
        # How many items per round?
        round_items = int(len(timeseries) / rounds)
        logger.debug('Items per round: {}'.format(round_items))
        
        # Start the fit / evaluate loop
        evaluations = []        
        for i in range(rounds):
            from_t = timeseries[(round_items*i)].t
            try:
                to_t = timeseries[(round_items*i) + round_items].t
            except IndexError:
                to_t = timeseries[(round_items*i) + round_items - 1].t
            from_dt = dt_from_s(from_t)
            to_dt   = dt_from_s(to_t)
            logger.info('Cross validation round #{} of {}: validate from {} ({}) to {} ({}), fit on the rest.'.format(i+1, rounds, from_t, from_dt, to_t, to_dt))
            
            # Fit
            if i == 0:            
                logger.debug('Fitting from {} ({})'.format(to_t, to_dt))
                self.fit(data, from_t=to_t, **fit_kwargs)
            else:
                logger.debug('Fitting until {} ({}) and then from {} ({}).'.format(to_t, to_dt, from_t, from_dt))
                self.fit(data, from_t=to_t, to_t=from_t, **fit_kwargs)                
            
            # Evaluate & append
            evaluations.append(self.evaluate(data, from_t=from_t, to_t=to_t, **evaluate_kwargs))
        
        # Regroup evaluations
        evaluation_metrics = list(evaluations[0].keys())
        scores_by_evaluation_metric = {}
        for evaluation in evaluations:
            for evaluation_metric in evaluation_metrics:
                try:
                    scores_by_evaluation_metric[evaluation_metric].append(evaluation[evaluation_metric])
                except KeyError:
                    try:
                        scores_by_evaluation_metric[evaluation_metric] = [evaluation[evaluation_metric]]
                    except KeyError:
                        raise Exception('Error, the model generated different evaluation metrics over the rounds, cannot compute cross validation.') from None
        
        # Prepare and return results
        results = {}
        for evaluation_metric in scores_by_evaluation_metric:
            results[evaluation_metric+'_avg'] = statistics.mean(scores_by_evaluation_metric[evaluation_metric])
            results[evaluation_metric+'_stdev'] = statistics.stdev(scores_by_evaluation_metric[evaluation_metric])         
        return results


    @property
    def id(self):
        return self.data['id']



#======================
# Data Reconstruction
#======================

class Reconstructor(ParametricModel):

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



class ProphetModel(ParametricModel):

    @classmethod
    def remove_timezone(cls, dt):
        return dt.replace(tzinfo=None)

    @classmethod
    def from_timeseria_to_prophet(cls, timeseries, from_t=None, to_t=None):

        # Create Python lists with data
        try:
            timeseries[0].data[0]
            data_keys_are_indexes = True
        except KeyError:
            timeseries[0].data.keys()
            data_keys_are_indexes = False
        
        data_as_list=[]
        for item in timeseries:
            
            # Skip if needed
            try:
                if not item_is_in_range(item, from_t, to_t):
                    continue
            except StopIteration:
                break                

            if data_keys_are_indexes:     
                data_as_list.append([cls.remove_timezone(item.dt), item.data[0]])
            else:
                data_as_list.append([cls.remove_timezone(item.dt), item.data[list(item.data.keys())[0]]])

        # Create the pandas DataFrames
        data = DataFrame(data_as_list, columns = ['ds', 'y'])

        return data



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
    


#======================
#  Forecasters
#======================

class Forecaster(ParametricModel):

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

        try:
            if len(timeseries) < self.data['window']:
                raise ValueError('The timeseries length ({}) is shorter than the model window ({}), it must be at least equal.'.format(len(timeseries), self.data['window']))
        except KeyError:
            pass

        # Set data key if only one key (for models not supporting multivariate)
        data_keys = timeseries.data_keys()
        if len(data_keys) > 1:
            key = None
        else:
            key = data_keys[0]
        
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
            forecast_model_results = self.forecast(timeseries = forecast_timeseries, key = key, n=n)
            if not isinstance(forecast_model_results, list):
                forecast_timeseries.append(forecast_model_results)
            # TODO: Do we want to silently handle custom forecast models not supporting multi-step forecasting?
            #elif len(forecast_model_results) == 1 and n >1:
            #    raise NotImplementedError('Seems like the forecaster did not implement the multi-step forecast')
            else:
                for item in forecast_model_results:
                    item.forecast = 1
                    forecast_timeseries.append(item)

        except NotImplementedError:
            
            for _ in range(n):
    
                # Call the forecast only on the last point
                forecast_model_results = self.forecast(timeseries = forecast_timeseries, key = key, n=1)

                # Add forecasted index
                forecast_model_results.forecast = 1

                # Add the forecast to the forecasts time series
                forecast_timeseries.append(forecast_model_results)
    
        # Do we have missing forecasts?
        if input_timeseries_len + n != len(forecast_timeseries):
            raise ValueError('There are missing forecasts. If your model does not support multi-step forecasting, raise a NotImplementedError if n>1 and Timeseria will handle it for you.')

        # Set serie mark for the forecast and return (old approach)
        #try:
        #    # Handle items
        #    forecast_timeseries.mark = [forecast_timeseries[-n].dt, forecast_timeseries[-1].end.dt]
        #except AttributeError:
        #    # Handle points TODO: should be dt-(resolution/2) and dt+(resolution/2)
        #    forecast_timeseries.mark = [forecast_timeseries[-n].dt, forecast_timeseries[-1].dt]
                
        if not inplace:
            return forecast_timeseries
        else:
            return None


    def forecast(self, timeseries, key=None, n=1, forecast_start=None):

        # Set forecast starting item
        if forecast_start is not None:
            forecast_start_item = timeseries[forecast_start]
        else:
            forecast_start_item = timeseries[-1]
            
        # Handle forecast start
        if forecast_start is not None:
            try:
                predicted_data = self._predict(timeseries=timeseries,
                                               key=key,
                                               n=n,
                                               forecast_start = forecast_start)
            except TypeError as e:
                if 'unexpected keyword argument' and  'forecast_start' in str(e):
                    raise NotImplementedError('The model does not support the "forecast_start" parameter, cannot proceed')           
                else:
                    raise
        else:
            predicted_data = self._predict(timeseries=timeseries,
                                           key=key,
                                           n=n)
                
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
                    forecast.append(DataTimePoint(t = last_item.t + timeseries._resolution,
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
                forecast = DataTimePoint(t = forecast_start_item.t + timeseries._resolution,
                                         tz = timeseries.tz,
                                         data  = predicted_data)
            
         
        return forecast



class PeriodicAverageForecaster(Forecaster):
        
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


    def _predict(self, timeseries, n=1, key=None, forecast_start=None):

        #if n>1:
        #    raise NotImplementedError('This forecaster does not support multi-step predictions.')

        # Set data key
        if not key:
            if len(timeseries.data_keys()) > 1:
                raise Exception('Multivariate time series require to have the key of the prediction specified')
            key=timeseries.data_keys()[0]
                    
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



class ProphetForecaster(Forecaster, ProphetModel):
    '''Prophet (from Facebook) implements a procedure for forecasting time series data based on an additive 
model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects.
It works best with time series that have strong seasonal effects and several seasons of historical data.
Prophet is robust to missing data and shifts in the trend, and typically handles outliers well. 
'''

    def _fit(self, timeseries, key=None, from_t=None, to_t=None, from_dt=None, to_dt=None):

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


    def _predict(self, timeseries, n=1, key=None):

        if not key and len(timeseries.data_keys()) > 1:
            raise Exception('Multivariate time series are not yet supported')
        if not key:
            key = timeseries.data_keys()[0]

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



class ARIMAModel():
    '''Class to wrap some common ARIMA-based models'''
    
    def get_start_end_indexes(self, timeseries, n):

        # Do the math to get the right indexes for the prediction, both out-of-sample and in-sample
        # Not used at the moment as there seems to be bugs in the statsmodel package.
        # See https://www.statsmodels.org/devel/generated/statsmodels.tsa.arima_model.ARIMAResults.predict.html
        
        # The default start_index for an arima prediction id the inxed after the fit timeseries
        # I.e. if the fit timeseries had 101 lements, the start index is the 101
        # So we need to compute the start and end indexes according to the fit timeseries,
        # we will use the "resolution" for this.
        
        # Save the requested prediction "from"
        requested_predicting_from_dt = timeseries[-1].dt + timeseries.resolution

        # Search for the index number TODO: try first with "pure" math which would work for UTC
        slider_dt = self.fit_timeseries[0].dt
        count = 0
        while True:
            if slider_dt  == requested_predicting_from_dt:
                break
             
            # To next item index
            if requested_predicting_from_dt < self.fit_timeseries[0].dt:
                slider_dt = slider_dt - self.fit_timeseries.resolution
                if slider_dt < requested_predicting_from_dt:
                    raise Exception('Miss!')
 
            else:
                slider_dt = slider_dt + self.fit_timeseries.resolution
                if slider_dt > requested_predicting_from_dt:
                    raise Exception('Miss!')
            
            count += 1
        
        # TODO: better understand the indexes in statsmodels (and potentially their bugs), as sometimes
        # setting the same start and end gave two predictions which is just nonsense in any scenario.
        start_index = count  
        end_index = count + n

        return (start_index, end_index)



class ARIMAForecaster(Forecaster, ARIMAModel):

    def __init__(self, p=1,d=1,q=0): #p=5,d=2,q=5
        if (p,d,q) == (1,1,0):
            logger.info('You are using ARIMA\'s defaults of p=1, d=1, q=0. You might want to set them to more suitable values when initializing the model.')
        self.p = p
        self.d = d
        self.q = q
        # TODO: save the above in data[]?
        super(ARIMAForecaster, self).__init__()


    def _fit(self, timeseries, key=None):

        import statsmodels.api as sm

        # Set data key
        if not key:
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
        
        
    def _predict(self, timeseries, n=1, key=None):

        # Set data key
        if not key:
            if len(timeseries.data_keys()) > 1:
                raise Exception('Multivariate time series require to have the key of the prediction specified')
            key=timeseries.data_keys()[0]

        # Chack that we are applying on a time series ending with the same datapoint where the fit timeseries was
        if self.fit_timeseries[-1].t != timeseries[-1].t:
            raise NonContiguityError('Sorry, this model can be applied only on a time series ending with the same timestamp as the time series used for the fit.')

        # Return the predicion. We need the [0] to access yhat, other indexes are erorrs etc.
        return [{key: value} for value in self.model_res.forecast(n)[0]] 



class AARIMAForecaster(Forecaster):

    def _fit(self, timeseries, key=None, **kwargs):
        
        import pmdarima as pm

        # Set data key
        if not key:
            if len(timeseries.data_keys()) > 1:
                raise Exception('Multivariate time series require to have the key of the prediction specified')
            key=timeseries.data_keys()[0]
                            
        data = array(timeseries.df[key])

        # Chenage some defaults
        trace = kwargs.pop('trace', True) # Get some output by default
        error_action = kwargs.pop('error_action', 'ignore') # Hide if an order does not work
        suppress_warnings = kwargs.pop('suppress_warnings', True) # Hide convergence warnings
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
        

    def _predict(self, timeseries, n=1, key=None):

        # Set data key
        if not key:
            if len(timeseries.data_keys()) > 1:
                raise Exception('Multivariate time series require to have the key of the prediction specified')
            key=timeseries.data_keys()[0]

        # Chack that we are applying on a time series ending with the same datapoint where the fit timeseries was
        if self.fit_timeseries[-1].t != timeseries[-1].t:
            raise NonContiguityError('Sorry, this model can be applied only on a time series ending with the same timestamp as the time series used for the fit.')

        # Return the predicion. We need the [0] to access yhat, other indexes are erorrs etc.
        return [{key: value} for value in self.model.predict(n)]



class KerasModel(Model):

    def __init__(self, path=None, id=None):
        
        super(KerasModel, self).__init__(path=path, id=id)

        if path:
            self.keras_model = load_keras_model('{}/keras_model'.format(path))


    def save(self, path):

        # Save the parameters (the "data") property
        model_dir = super(KerasModel, self).save(path)

        # Now save the Keras model itself
        try:
            self.keras_model.save('{}/keras_model'.format(model_dir))
        except Exception as e:
            shutil.rmtree(model_dir)
            raise e
        return model_dir


    @staticmethod
    def to_windows(timeseries):
        '''Compute window data values'''
        key = timeseries.data_keys()[0]
        window_data_values = []
        for item in timeseries:
            window_data_values.append(item.data[key])
        return window_data_values
            
    @staticmethod
    def to_window_datapoints_matrix(timeseries, window, forecast_n, encoder=None):
        '''Compute window datapoints matrix'''
    
        window_datapoints = []
        for i, _ in enumerate(timeseries):
            if i <  window:
                continue
            if i == len(timeseries)-forecast_n:
                break
                    
            # Add window values
            row = []
            for j in range(window):
                row.append(timeseries[i-window+j])
            window_datapoints.append(row)
                
        return window_datapoints
    
    @staticmethod
    def to_target_values_vector(timeseries, window, forecast_n):
        '''Compute target values vector'''
    
        data_keys = timeseries.data_keys()
    
        targets = []
        for i, _ in enumerate(timeseries):
            if i <  window:
                continue
            if i == len(timeseries)-forecast_n:
                break
            
            # Add forecast target value(s)
            row = []
            for j in range(forecast_n):
                for data_key in data_keys:
                    row.append(timeseries[i+j].data[data_key])
            targets.append(row)

        return targets

    @staticmethod
    def compute_window_features(window_datapoints, data_keys, features):

        available_features = ['values', 'diffs', 'hours']
        for feature in features:
            if feature not in available_features:
                raise ValueError('Unknown feature "{}"'.format(feature))
        
        window_features=[]

        for i in range(len(window_datapoints)):
            
            datapoint_features = []
            
            # 1) datapoint values (for all keys)
            if 'values' in features:
                for data_key in data_keys:
                    datapoint_features.append(window_datapoints[i].data[data_key])
                            
            # 2) Compute diffs on normalized datapoints
            if 'diffs' in features:
                for data_key in data_keys:
                    if i ==0:
                        diff = window_datapoints[1].data[data_key] - window_datapoints[0].data[data_key]
                    elif i == len(window_datapoints)-1:
                        diff = window_datapoints[-1].data[data_key] - window_datapoints[-2].data[data_key]
                    else:
                        diff = (window_datapoints[i+1].data[data_key] - window_datapoints[i-1].data[data_key]) /2
                    if diff == 0:
                        diff = 1
                    datapoint_features.append(diff)

            # 3) Hour (normlized)
            if 'hours' in features:
                datapoint_features.append(window_datapoints[i].dt.hour/24)
                
            # Now append to the window features
            window_features.append(datapoint_features)

        return window_features



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
            self.keras_model = Sequential()
            self.keras_model.add(LSTM(self.data['neurons'], input_shape=(self.data['window'], features_per_window_item)))
            self.keras_model.add(Dense(output_dimension)) 
            self.keras_model.compile(loss='mean_squared_error', optimizer='adam')

        # Fit
        self.keras_model.fit(array(window_features), array(target_values_vector), epochs=epochs, verbose=verbose)
        
        # Store data
        self.data['min_values'] = min_values
        self.data['max_values'] = max_values
        self.data['data_keys'] = data_keys


    def _predict(self, timeseries, n=1, key=None, verbose=False):

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



#======================
#  Anomaly detectors
#======================

class AnomalyDetector(ParametricModel):
    pass


class PeriodicAverageAnomalyDetector(AnomalyDetector):

    def __get_actual_and_predicted(self, timeseries, i, key, forecaster_window):

        # Call model predict logic and compare with the actual data
        actual    = timeseries[i].data[key]
        predicted = self.forecaster.predict(timeseries,
                                            n=1,
                                            key=key,
                                            forecast_start = i-1)[0][key]
        
        return (actual, predicted)


    def _fit(self, timeseries, *args, stdevs=3, **kwargs):

        if len(timeseries.data_keys()) > 1:
            raise NotImplementedError('Multivariate time series are not yet supported')
        
    
        # Fit a forecaster               
        forecaster = PeriodicAverageForecaster()
        
        # Fit and save
        forecaster.fit(timeseries, *args, **kwargs)
        self.forecaster = forecaster
        
        # Evaluate the forecaster for one step ahead and get AEs
        AEs = []
        for key in timeseries.data_keys():
            
            for i, _ in enumerate(timeseries):
                
                forecaster_window = self.forecaster.data['window']
                
                if i <=  forecaster_window:    
                    continue
                
                actual, predicted = self.__get_actual_and_predicted(timeseries, i, key, forecaster_window)
                
                AEs.append(abs(actual-predicted))

        # Compute distribution for the AEs ans set the threshold
        from scipy.stats import norm
        mean, stdev = norm.fit(AEs)
        logger.info('Using {} standard deviations as anomaly threshold: {}'.format(stdevs, stdev*stdevs))
        
        # Set AE-based threshold
        self.AE_threshold = stdev*stdevs


    def _apply(self, timeseries, inplace=False, details=False, logs=False):
        
        if inplace:
            raise Exception('Anomaly detection cannot be run inplace')

        if len(timeseries.data_keys()) > 1:
            raise NotImplementedError('Multivariate time series are not yet supported')
        
        result_timeseries = timeseries.__class__()

        for key in timeseries.data_keys():
            
            for i, item in enumerate(timeseries):
                forecaster_window = self.forecaster.data['window']
                if i <=  forecaster_window:    
                    continue
                
                actual, predicted = self.__get_actual_and_predicted(timeseries, i, key, forecaster_window)
                #if logs:
                #    logger.info('{}: {} vs {}'.format(timeseries[i].dt, actual, predicted))

                AE = abs(actual-predicted)
                
                item = deepcopy(item)
                if AE > self.AE_threshold:
                    if logs:
                        logger.info('Detected anomaly for item starting @ {} ({}) with AE="{:.3f}..."'.format(item.t, item.dt, AE))
                    item.anomaly = 1
                    if details:
                        item.data['AE_{}'.format(key)] = AE
                        item.data['predicted_{}'.format(key)] = predicted
                else:
                    item.anomaly = 0
                    if details:
                        item.data['AE_{}'.format(key)] = AE
                        item.data['predicted_{}'.format(key)] = predicted

                result_timeseries.append(item)
        
        return result_timeseries 

