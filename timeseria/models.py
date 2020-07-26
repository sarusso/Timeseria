import os
import json
import uuid
import copy
from .datastructures import DataTimeSlotSeries, DataTimeSlot, TimePoint
from .exceptions import NotFittedError
from .utilities import get_periodicity
from .time import now_t, dt_from_s, s_from_dt
from datetime import timedelta
from sklearn.metrics import mean_squared_error
from .units import TimeUnit

# Setup logging
import logging
logger = logging.getLogger(__name__)

HARD_DEBUG = False


#======================
#  Utility functions
#======================


def mean_error(list1, list2):
    if len(list1) != len(list2):
        raise ValueError('Lists have different lengths, cannot continue')
    error_sum = 0
    for i in range(len(list1)):
        error_sum += abs(list1[i] - list2[i])
    return error_sum/len(list1)


def get_periodicity_index(time_point, slot_unit, periodicity, dst_affected=False):

    # Handle specific cases
    if isinstance(slot_unit, TimeUnit):  
        if slot_unit.type == TimeUnit.LOGICAL:
            raise NotImplementedError('LOGICAL time units are not yet supported')
        elif slot_unit.type == TimeUnit.PHYSICAL:
            pass
        else:
            raise Exception('Consistency error, got slot unit type "{}" which is unknown'.format(slot_unit.type))

        slot_unit_duration = slot_unit.duration

    else:
        if isinstance(slot_unit.value, list):
            raise NotImplementedError('Sorry, periodicty in multi-dimensional spaces are not defined')
        slot_unit_duration = slot_unit.value

    # Compute periodicity index
    if not dst_affected:
    
        # Get index based on slot start, normalized to unit, modulus periodicity
        slot_start_t = time_point.t
        periodicity_index =  int(slot_start_t / slot_unit_duration) % periodicity
    
    else:

        # Get periodicity based on the datetime
        slot_start_t  = time_point.t
        slot_start_dt = time_point.dt
        
        # Do we have an active DST?  
        dst_timedelta = slot_start_dt.dst()
        
        if dst_timedelta.days == 0 and dst_timedelta.seconds == 0:
            # No DST
            periodicity_index = int(slot_start_t / slot_unit_duration) % periodicity
        
        else:
            # DST
            if dst_timedelta.days != 0:
                raise Exception('Don\'t know how to handle DST with days timedelta = "{}"'.format(dst_timedelta.days))

            if slot_unit_duration > 3600:
                raise Exception('Sorry, this time series has not enough time-resolution to account for DST effects (slot_unit_duration="{}", must be below 3600 seconds)'.format(slot_unit_duration))
            
            # Get DST offset in seconds 
            dst_offset_s = dst_timedelta.seconds # 3600 usually

            # Compute periodicity in seconds (example: 144 10-minute slots) 
            #periodicity_s = periodicity * slot_unit_duration
            
            # Get DST offset "slots"
            #dst_offset_slots = int(dst_offset_s / slot_unit_duration) # For ten-minutes slot is 6               
            #periodicity_index = (int(slot_start_t / slot_unit_duration) % periodicity) #+ dst_offset_slots
            
            periodicity_index = (int((slot_start_t + dst_offset_s) / slot_unit_duration) % periodicity)
        
        

    return periodicity_index



#======================
#  Base classes
#======================

#class EvaluationScore(object):
#    '''Class to incapsulate a model score after an evaluation_score'''
#    pass

class Model(object):
    '''A stateless model, or a white-box model. Exposes only predict, apply and evaluate methods, since it is assumed that all the
    information is coded and nothing is learnt from data. Provides also an "evaluation_score_score" property which is set by the last evaluate() call.'''
    
    def __init__(self):
        pass
    
    def predict(self):
        pass
    
    def apply(self, *args, **kwargs):
        return self._apply(self,*args, **kwargs)
    
    def evaluate(self, *args, **kwargs):
        return self._evaluate(self,*args, **kwargs)



class ParametricModel(Model):
    '''A stateful model with parameters, or a (partly) black-box model. Parameters can be set manually or learnt (fitted) from data.
    On top of the predict(), apply() and evaluate() methods it provides a save() method to store the parameters of the model,
    and optionally a fit() method if the parameters are to be learnt form data. Provides also an "evaluation_score_score" property which is set
    by the last evaluate() call or, in case of cross validation, by the average of the evaluate() functions called in the cross-validation
    process. Can optionally have a kernel incapsulating an external model (Keras, Tensorflow, Scikit-learn etc.)'''
    
    def __init__(self, path=None, id=None):
        
        if path:
            with open(path+'/data.json', 'r') as f:
                self.data = json.loads(f.read())         
            self.fitted=True
        else:
            
            if not id:
                id = str(uuid.uuid4())
            self.fitted = False
            self.data = {'id': id}

        super(ParametricModel, self).__init__()


    def fit(self, data_time_slot_series, *args, **kwargs):
        
        if not isinstance(data_time_slot_series, DataTimeSlotSeries):
            raise TypeError('DataTimeSlotSeries is required (got "{}")'.format(data_time_slot_series.__class__.__name__))
    
        if not data_time_slot_series:
            raise ValueError('A non-empty DataTimeSlotSeries is required')

        fit_output = self._fit(data_time_slot_series, *args, **kwargs)
        self.data['fitted_at'] = now_t()
        self.fitted = True
        return fit_output

        
    def save(self, path):
        # TODO: dump and enforce the TimeUnit as well
        if not self.fitted:
            raise NotFittedError()
        model_dir = '{}/{}'.format(path, self.data['id'])
        os.makedirs(model_dir)
        model_data_file = '{}/data.json'.format(model_dir)
        with open(model_data_file, 'w') as f:
            f.write(json.dumps(self.data))
        logger.info('Saved model with id "%s" in "%s"', self.data['id'], model_dir)
        return model_dir

      
    def apply(self, data_time_slot_series, *args, **kwargs):
        if not self.fitted:
            raise NotFittedError()

        if not isinstance(data_time_slot_series, DataTimeSlotSeries):
            raise TypeError('DataTimeSlotSeries is required (got "{}")'.format(data_time_slot_series.__class__.__name__))
    
        if not data_time_slot_series:
            raise ValueError('A non-empty DataTimeSlotSeries is required')

        return self._apply(data_time_slot_series, *args, **kwargs)


    def _fit(self, *args, **krargs):
        raise NotImplementedError('Fitting this model is not implemented')

    def _apply(self, *args, **krargs):
        raise NotImplementedError('Applying this model is not implemented')

    def _evaluate(self, *args, **krargs):
        raise NotImplementedError('Evaluating this model is not implemented')


    @property
    def id(self):
        return self.data['id']


    @property
    def evaluation_score(self):
        if not self.fitted:
            raise NotFittedError()
        else:
            return self.data['evaluation_score']
    


#======================
# Data Reconstruction
#======================

class Reconstructor(ParametricModel):

    def _apply(self, data_time_slot_series, remove_data_loss=False, data_loss_threshold=1, inplace=False):
        logger.debug('Using data_loss_threshold="%s"', data_loss_threshold)
        
        if inplace:
            pass # The function use the same "data_time_slot_series" variable
        
        else:
            data_time_slot_series = copy.deepcopy(data_time_slot_series)

        if len(data_time_slot_series.data_keys()) > 1:
            raise NotImplementedError('Multivariate time series are not yet supported')

        for key in data_time_slot_series.data_keys():
            
            gap_started = None
            
            for i, data_time_slot in enumerate(data_time_slot_series):
                if data_time_slot.data_loss >= data_loss_threshold:
                    # This is the beginning of an area we want to reconstruct according to the data_loss_threshold
                    if gap_started is None:
                        gap_started = i
                else:
                    
                    if gap_started is not None:
                    
                        # Reconstruct for this gap
                        self._reconstruct(from_index=gap_started, to_index=i, data_time_slot_series=data_time_slot_series, key=key)
                        gap_started = None
                    
                    data_time_slot._data_reconstructed = 0
                    
                if remove_data_loss:
                    # TOOD: move to None if we allow data_losses (coverages) to None?
                    data_time_slot._coverage = 1
            
            # Reconstruct the last gap as well if left "open"
            if gap_started is not None:
                self._reconstruct(from_index=gap_started, to_index=i+1, data_time_slot_series=data_time_slot_series, key=key)

        return data_time_slot_series


    def _evaluate(self, data_time_slot_series, steps_set='auto', samples=1000, data_loss_threshold=1):

        # Set evaluation_score steps if we have to
        if steps_set == 'auto':
            steps_set = [1, self.data['periodicity']]

        # Support var
        evaluation_score = {}
         
        # Find areas where to evaluate the model
        for key in data_time_slot_series.data_keys():
             
            for steps in steps_set:
                
                # Support vars
                real_values = []
                reconstructed_values = []
                processed_samples = 0

                # Here we will have steps=1, steps=2 .. steps=n          
                logger.debug('Evaluating model for steps %s', steps)
                
                for i in range(len(data_time_slot_series)):
                                  
                    # Skip the first and the last ones, otherwise reconstruct the ones in the middle
                    if (i == 0) or (i >= len(data_time_slot_series)-steps):
                        continue

                    # Is this a "good area" where to test or do we have to stop?
                    stop = False
                    if data_time_slot_series[i-1].data_loss >= data_loss_threshold:
                        stop = True
                    for j in range(steps):
                        if data_time_slot_series[i+j].data_loss >= data_loss_threshold:
                            stop = True
                            break
                    if data_time_slot_series[i+steps].data_loss >= data_loss_threshold:
                        stop = True
                    if stop:
                        continue
                            
                    # Set prev and next
                    prev_value = data_time_slot_series[i-1].data[key]
                    next_value = data_time_slot_series[i+steps].data[key]
                    
                    # Compute average value
                    average_value = (prev_value+next_value)/2
                    
                    # Data to be reconstructed
                    data_time_slot_series_to_reconstruct = DataTimeSlotSeries()
                    
                    # Append prev
                    #data_time_slot_series_to_reconstruct.append(copy.deepcopy(data_time_slot_series[i-1]))
                    
                    # Append in the middle and store real values
                    for j in range(steps):
                        data_time_slot = copy.deepcopy(data_time_slot_series[i+j])
                        # Set the coverage to zero so the slot will be reconstructed
                        data_time_slot._coverage = 0
                        data_time_slot.data[key] = average_value
                        data_time_slot_series_to_reconstruct.append(data_time_slot)
                        
                        real_values.append(data_time_slot_series[i+j].data[key])
              
                    # Append next
                    #data_time_slot_series_to_reconstruct.append(copy.deepcopy(data_time_slot_series[i+steps]))

                    # Apply model inplace
                    self._apply(data_time_slot_series_to_reconstruct, inplace=True)
                    processed_samples += 1

                    # Store reconstructed values
                    for j in range(steps):
                        reconstructed_values.append(data_time_slot_series_to_reconstruct[j].data[key])
                    
                    if samples is not None and processed_samples >= samples:
                        break

                if processed_samples < samples:
                    logger.warning('Could not evaluate "{}" samples for "{}" steps, processed "{}" samples only'.format(samples, steps, processed_samples))

                if not reconstructed_values:
                    raise Exception('Could not evaluate model, maybe not enough data?')

                # Compute RMSE and ME, and add to the evaluation_score
                evaluation_score['rmse_{}_steps'.format(steps)] = mean_squared_error(real_values, reconstructed_values)
                evaluation_score['me_{}_steps'.format(steps)] = mean_error(real_values, reconstructed_values)

        # Compute average RMSE
        sum_rmse = 0
        count = 0
        for key in evaluation_score:
            if key.startswith('rmse_'):
                sum_rmse += evaluation_score[key]
                count += 1
        evaluation_score['mrmse'] = sum_rmse/count

        # Compute average ME
        sum_me = 0
        count = 0
        for key in evaluation_score:
            if key.startswith('me_'):
                sum_me += evaluation_score[key]
                count += 1
        evaluation_score['mme'] = sum_me/count
        
        return evaluation_score


    def _reconstruct(self, *args, **krargs):
        raise NotImplementedError('Reconstruction for this model is not yet implemented')




class PeriodicAverageReconstructor(Reconstructor):

    def _fit(self, data_time_slot_series, data_loss_threshold=0.5, periodicity=None, evaluation_steps_set='auto', evaluation_samples=1000, dst_affected=False, timezone_affected=False):

        if len(data_time_slot_series.data_keys()) > 1:
            raise NotImplementedError('Multivariate time series are not yet supported')
        
        # Set or detect periodicity
        if periodicity is None:
            periodicity =  get_periodicity(data_time_slot_series)
            if isinstance(data_time_slot_series.slot_unit, TimeUnit):
                logger.info('Detected periodicity: %sx %s', periodicity, data_time_slot_series.slot_unit)
            else:
                logger.info('Detected periodicity: %sx %ss', periodicity, data_time_slot_series.slot_unit)
        self.data['periodicity']  = periodicity
        self.data['dst_affected'] = dst_affected 
        
        logger.info('dst_affected: {}'.format(dst_affected))
        
        for key in data_time_slot_series.data_keys():
            sums   = {}
            totals = {}
            for data_time_slot in data_time_slot_series:
                if data_time_slot.data_loss < data_loss_threshold:
                    periodicity_index = get_periodicity_index(data_time_slot.start, data_time_slot_series.slot_unit, periodicity, dst_affected=dst_affected)
                    if not periodicity_index in sums:
                        sums[periodicity_index] = data_time_slot.data[key]
                        totals[periodicity_index] = 1
                    else:
                        sums[periodicity_index] += data_time_slot.data[key]
                        totals[periodicity_index] +=1

        averages={}
        for key in sums:
            averages[key] = sums[key]/totals[key]
        self.data['averages'] = averages

        self.data['evaluation_score'] = self._evaluate(data_time_slot_series, steps_set=evaluation_steps_set, samples=evaluation_samples)
        logger.info('Model evaluation score: "{}"'.format(self.data['evaluation_score']))


    def _reconstruct(self, data_time_slot_series, key, from_index, to_index):
        logger.debug('Reconstructing between "{}" and "{}"'.format(from_index, to_index-1))
    
        # Compute offset
        diffs=0
        for j in range(from_index, to_index):
            real_value = data_time_slot_series[j].data[key]
            periodicity_index = get_periodicity_index(data_time_slot_series[j].start, data_time_slot_series.slot_unit, self.data['periodicity'], dst_affected=self.data['dst_affected'])
            reconstructed_value = self.data['averages'][periodicity_index]
            diffs += (real_value - reconstructed_value)
        offset = diffs/(to_index-from_index)

        # Actually reconstruct
        for j in range(from_index, to_index):
            data_time_slot_to_reconstruct = data_time_slot_series[j]
            periodicity_index = get_periodicity_index(data_time_slot_to_reconstruct.start, data_time_slot_series.slot_unit, self.data['periodicity'], dst_affected=self.data['dst_affected'])
            data_time_slot_to_reconstruct.data[key] = self.data['averages'][periodicity_index] + offset
            data_time_slot_to_reconstruct._data_reconstructed = 1
                        

    def _plot_averages(self, data_time_slot_series, log_js=False):      
        averages_data_time_slot_series = copy.deepcopy(data_time_slot_series)
        for data_time_slot in averages_data_time_slot_series:
            value = self.data['averages'][get_periodicity_index(data_time_slot.start, averages_data_time_slot_series.slot_unit, self.data['periodicity'], dst_affected=self.data['dst_affected'])]
            if not value:
                value = 0
            data_time_slot.data['average'] =value 
        averages_data_time_slot_series.plot(log_js=log_js)



#======================
#  Forecast
#======================

class Forecaster(ParametricModel):

    def _evaluate(self, data_time_slot_series, steps_set='auto', samples=1000, plots=False):
        
        # Set evaluation_score steps if we have to
        if steps_set == 'auto':
            steps_set = [1, self.data['periodicity']]

        # Support var
        evaluation_score = {}

        for steps in steps_set:
            
            # Support vars
            real_values = []
            model_values = []
            processed_samples = 0
    
            # For each point of the data_time_slot_series, after the window, apply the prediction and compare it with the actual value
            for key in data_time_slot_series.data_keys():
                for i in range(len(data_time_slot_series)):
                    
                    # Check that we can get enough data
                    if i < self.data['window']+steps:
                        continue
                    if i > (len(data_time_slot_series)-steps):
                        continue

                    # Compute the various boundaries
                    original_serie_boundaries_start = i - (self.data['window']) - steps
                    original_serie_boundaries_end = i
                    
                    original_forecast_serie_boundaries_start = original_serie_boundaries_start
                    original_forecast_serie_boundaries_end = original_serie_boundaries_end-steps
                    
                    # Create the time series where to apply the forecast on
                    forecast_data_time_slot_series = DataTimeSlotSeries()
                    for j in range(original_forecast_serie_boundaries_start, original_forecast_serie_boundaries_end):
                        forecast_data_time_slot_series.append(data_time_slot_series[j])
 
                    # Apply the forecasting model
                    forecast_data_time_slot_series = self._apply(forecast_data_time_slot_series, n=steps, inplace=True)

                    # Plot evaluation_score time series?
                    if plots:
                        forecast_data_time_slot_series.plot(log_js=False)
                    
                    # Compare each forecast with the original value
                    for step in range(steps):
                        original_index = original_serie_boundaries_start + self.data['window'] + step

                        forecast_index = self.data['window'] + step

                        model_value = forecast_data_time_slot_series[forecast_index].data[key]
                        model_values.append(model_value)
                        
                        real_value = data_time_slot_series[original_index].data[key]
                        real_values.append(real_value)
 
                    processed_samples+=1
                    if samples is not None and processed_samples >= samples:
                        break
                
            if processed_samples < samples:
                logger.warning('Could not evaluate "{}" samples for "{}" steps, processed "{}" samples only'.format(samples, steps, processed_samples))

            if not model_values:
                raise Exception('Could not evaluate model, maybe not enough data?')

            # Compute RMSE and ME, and add to the evaluation_score
            evaluation_score['rmse_{}_steps'.format(steps)] = mean_squared_error(real_values, model_values)
            evaluation_score['me_{}_steps'.format(steps)] = mean_error(real_values, model_values)
   
        # Compute average RMSE
        sum_rmse = 0
        count = 0
        for key in evaluation_score:
            if key.startswith('rmse_'):
                sum_rmse += evaluation_score[key]
                count += 1
        evaluation_score['mrmse'] = sum_rmse/count

        # Compute average ME
        sum_me = 0
        count = 0
        for key in evaluation_score:
            if key.startswith('me_'):
                sum_me += evaluation_score[key]
                count += 1
        evaluation_score['mme'] = sum_me/count
        
        return evaluation_score


    def _apply(self, data_time_slot_series, n=1, inplace=False):

        if len(data_time_slot_series.data_keys()) > 1:
            raise NotImplementedError('Multivariate time series are not yet supported')
 
        if len(data_time_slot_series) < self.data['window']:
            raise ValueError('The data_time_slot_series length ({}) is shorter than the model window ({}), it must be at least equal.'.format(len(data_time_slot_series), self.data['window']))
 
        if inplace:
            forecast_data_time_slot_series = data_time_slot_series
        else:
            forecast_data_time_slot_series = copy.deepcopy(data_time_slot_series)
        
        for key in forecast_data_time_slot_series.data_keys():

            # Support var
            first_call = True

            for _ in range(n):
                
                # Compute start/end for the slot to be forecasted
                if isinstance(forecast_data_time_slot_series.slot_unit, TimeUnit):
                    this_slot_start_dt = forecast_data_time_slot_series[-1].start.dt + forecast_data_time_slot_series.slot_unit
                    this_slot_end_dt   =  this_slot_start_dt + forecast_data_time_slot_series.slot_unit
                    this_slot_start_t  = s_from_dt(this_slot_start_dt) 
                    this_slot_end_t    = s_from_dt(this_slot_end_dt)
                else:
                    this_slot_start_t = forecast_data_time_slot_series[-1].start.t + forecast_data_time_slot_series.slot_unit.value
                    this_slot_end_t   = this_slot_start_t + forecast_data_time_slot_series.slot_unit.value
                    this_slot_start_dt = dt_from_s(this_slot_start_t, tz=forecast_data_time_slot_series.tz)
                    this_slot_end_dt = dt_from_s(this_slot_end_t, tz=forecast_data_time_slot_series.tz )                    

                # Set time zone
                tz = forecast_data_time_slot_series[-1].start.tz
                
                # Define TimePoints
                this_slot_start_timePoint = TimePoint(this_slot_start_t, tz=tz)
                this_slot_end_timePoint = TimePoint(this_slot_end_t, tz=tz)

                # Call model forecasting logic
                forecasted_data_time_slot = self._forecast(forecast_data_time_slot_series, data_time_slot_series.slot_unit, key, this_slot_start_timePoint, this_slot_end_timePoint, first_call)

                # Add the forecast to the forecasts time series
                forecast_data_time_slot_series.append(forecasted_data_time_slot)
                
                # Set fist call to false if this was the first call
                if first_call:
                    first_call = False 

        # Set serie mark for the forecast and return
        forecast_data_time_slot_series.mark = [forecast_data_time_slot_series[-n].start.dt, forecast_data_time_slot_series[-1].end.dt]
        return forecast_data_time_slot_series



class PeriodicAverageForecaster(Forecaster):
    
    def _fit(self, data_time_slot_series, window=None, periodicity=None, evaluation_steps_set='auto', evaluation_samples=1000, evaluation_plots=False, dst_affected=False):

        if len(data_time_slot_series.data_keys()) > 1:
            raise NotImplementedError('Multivariate time series are not yet supported')

        # Set or detect periodicity
        if periodicity is None:        
            periodicity =  get_periodicity(data_time_slot_series)
            if isinstance(data_time_slot_series.slot_unit, TimeUnit):
                logger.info('Detected periodicity: %sx %s', periodicity, data_time_slot_series.slot_unit)
            else:
                logger.info('Detected periodicity: %sx %ss', periodicity, data_time_slot_series.slot_unit)
        self.data['periodicity']  = periodicity
        self.data['dst_affected'] = dst_affected

        # Set or detect window
        if window:
            self.data['window'] = window
        else:
            logger.info('Using a window of "{}"'.format(periodicity))
            self.data['window'] = periodicity

        for key in data_time_slot_series.data_keys():
            sums   = {}
            totals = {}
            for data_time_slot in data_time_slot_series:
                periodicity_index = get_periodicity_index(data_time_slot.start, data_time_slot_series.slot_unit, periodicity, dst_affected)
                if not periodicity_index in sums:
                    sums[periodicity_index] = data_time_slot.data[key]
                    totals[periodicity_index] = 1
                else:
                    sums[periodicity_index] += data_time_slot.data[key]
                    totals[periodicity_index] +=1

        averages={}
        for key in sums:
            averages[key] = sums[key]/totals[key]
        self.data['averages'] = averages

        self.data['evaluation_score']  = self._evaluate(data_time_slot_series, steps_set=evaluation_steps_set, samples=evaluation_samples, plots=evaluation_plots)
        logger.info('Model evaluation score: "{}"'.format(self.data['evaluation_score']))


    def _forecast(self, forecast_data_time_slot_series, slot_unit, key, this_slot_start_timePoint, this_slot_end_timePoint, first_call) : #, data_time_slot_series, key, from_index, to_index):

        # Compute the offset (avg diff between the real values and the forecasts on the first window)
        try:
            self.offsets
        except AttributeError:
            self.offsets={}
            
        if key not in self.offsets or first_call:

            diffs  = 0  
            for j in range(self.data['window']):
                serie_index = -(self.data['window']-j)
                real_value = forecast_data_time_slot_series[serie_index].data[key]
                forecast_value = self.data['averages'][get_periodicity_index(forecast_data_time_slot_series[serie_index].start, forecast_data_time_slot_series.slot_unit, self.data['periodicity'], dst_affected=self.data['dst_affected'])]
                diffs += (real_value - forecast_value)
   
            # Sum the avg diff between the real and the forecast on the window to the forecast (the offset)
            offset = diffs/j
            self.offsets[key] = offset
        
        else:
            offset = self.offsets[key] 
        
        # Compute and add the real forecast data
        periodicity_index = get_periodicity_index(this_slot_start_timePoint, slot_unit, self.data['periodicity'], dst_affected=self.data['dst_affected'])        
        forecasted_data_time_slot = DataTimeSlot(start = this_slot_start_timePoint,
                                               end   = this_slot_end_timePoint,
                                               unit  = forecast_data_time_slot_series[-1].unit,
                                               coverage = 0,
                                               data  = {key: self.data['averages'][periodicity_index] + (offset*1.0)})

        return forecasted_data_time_slot
    

    def _plot_averages(self, data_time_slot_series, log_js=False):      
        averages_data_time_slot_series = copy.deepcopy(data_time_slot_series)
        for data_time_slot in averages_data_time_slot_series:
            value = self.data['averages'][get_periodicity_index(data_time_slot.start, averages_data_time_slot_series.slot_unit, self.data['periodicity'], dst_affected=self.data['dst_affected'])]
            if not value:
                value = 0
            data_time_slot.data['average'] =value 
        averages_data_time_slot_series.plot(log_js=log_js)










