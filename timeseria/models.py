import os
import json
import uuid
import copy
from .datastructures import DataTimeSlotSerie, DataTimeSlot, TimePoint
from .exceptions import NotTrainedError
from .utilities import get_periodicity
from .time import now_t, TimeSpan, dt_from_s, s_from_dt
from sklearn.metrics import mean_squared_error

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


def get_periodicity_index(timePoint, slot_span, periodicity):

    # Support var
    use_dt_instead_of_t = False

    # Handle specific cases
    if isinstance(slot_span, TimeSpan):

        if slot_span.type == TimeSpan.LOGICAL:
            use_dt_instead_of_t  = True
        elif slot_span.type == TimeSpan.PHYSICAL:
            use_dt_instead_of_t = False
            # TODO: use dt if periodicty is affected by DST (timezone).
        else:
            raise Exception('Consistency error, got slot span type "{}" which is unknown'.format(slot_span.type))
        
        if use_dt_instead_of_t:
            raise NotImplementedError('Not yet')
        else:
            slot_start_t = timePoint.t
            periodicity_index =  int(slot_start_t / slot_span.duration) % periodicity

    else:

        # Get index based on slot start, normalized to span, modulus periodicity
        slot_start_t = timePoint.t
        periodicity_index =  int(slot_start_t / slot_span) % periodicity

    return periodicity_index



#======================
#  Base classes
#======================

class Model(object):
    
    def __init__(self):
        pass
    
    def apply(self, *args, **kwargs):
        return self._apply(self,*args, **kwargs)



class TrainableModel(Model):
    
    def __init__(self, path=None, id=None):
        
        if path:
            with open(path+'/data.json', 'r') as f:
                self.data = json.loads(f.read())         
            self.trained=True
        else:
            
            if not id:
                id = str(uuid.uuid4())
            self.trained = False
            self.data = {'id': id}

        super(TrainableModel, self).__init__()


    def train(self, dataTimeSlotSerie, *args, **kwargs):
        
        if not isinstance(dataTimeSlotSerie, DataTimeSlotSerie):
            raise TypeError('DataTimeSlotSerie is required (got "{}")'.format(dataTimeSlotSerie.__class__.__name__))
    
        if not dataTimeSlotSerie:
            raise ValueError('A non-empty DataTimeSlotSerie is required')

        train_output = self._train(dataTimeSlotSerie, *args, **kwargs)
        self.data['trained_at'] = now_t()
        self.trained = True
        return train_output

        
    def save(self, path):
        # TODO: dump and enforce the TimeSpan as well
        if not self.trained:
            raise NotTrainedError()
        model_dir = '{}/{}'.format(path, self.data['id'])
        os.makedirs(model_dir)
        model_data_file = '{}/data.json'.format(model_dir)
        with open(model_data_file, 'w') as f:
            f.write(json.dumps(self.data))
        logger.info('Saved model with id "%s" in "%s"', self.data['id'], model_dir)
        return model_dir

      
    def apply(self, dataTimeSlotSerie, *args, **kwargs):
        if not self.trained:
            raise NotTrainedError()

        if not isinstance(dataTimeSlotSerie, DataTimeSlotSerie):
            raise TypeError('DataTimeSlotSerie is required (got "{}")'.format(dataTimeSlotSerie.__class__.__name__))
    
        if not dataTimeSlotSerie:
            raise ValueError('A non-empty DataTimeSlotSerie is required')

        return self._apply(dataTimeSlotSerie, *args, **kwargs)


    def _train(self, *args, **krargs):
        raise NotImplementedError('Training this model is not yet implemented')

    def _apply(self, *args, **krargs):
        raise NotImplementedError('Applying this model is not yet implemented')

    def _evaluate(self, *args, **krargs):
        raise NotImplementedError('Evaluating this model is not yet implemented')


    @property
    def id(self):
        return self.data['id']


    @property
    def evaluation(self):
        if not self.trained:
            raise NotTrainedError()
        else:
            return self.data['evaluation']
    


#======================
# Data Reconstruction
#======================

class Reconstructor(TrainableModel):

    def _apply(self, dataTimeSlotSerie, remove_data_loss=False, data_loss_threshold=1, inplace=False):
        logger.debug('Using data_loss_threshold="%s"', data_loss_threshold)
        
        if inplace:
            pass # The function use the same "dataTimeSlotSerie" variable
        
        else:
            dataTimeSlotSerie = copy.deepcopy(dataTimeSlotSerie)

        if len(dataTimeSlotSerie.data_keys()) > 1:
            raise NotImplementedError('Multivariate time series are not yet supported')

        for key in dataTimeSlotSerie.data_keys():
            
            gap_started = None
            
            for i, dataTimeSlot in enumerate(dataTimeSlotSerie):
                if dataTimeSlot.data_loss >= data_loss_threshold:
                    # This is the beginning of an area we want to reconstruct according to the data_loss_threshold
                    if gap_started is None:
                        gap_started = i
                else:
                    
                    if gap_started is not None:
                    
                        # Reconstruct for this gap
                        self._reconstruct(from_index=gap_started, to_index=i, dataTimeSlotSerie=dataTimeSlotSerie, key=key)
                        gap_started = None
                    
                    dataTimeSlot._data_reconstructed = 0
                    
                if remove_data_loss:
                    # TOOD: move to None if we allow data_losses (coverages) to None?
                    dataTimeSlot._coverage = 1
            
            # Reconstruct the last gap as well if left "open"
            if gap_started is not None:
                self._reconstruct(from_index=gap_started, to_index=i+1, dataTimeSlotSerie=dataTimeSlotSerie, key=key)

        return dataTimeSlotSerie


    def _evaluate(self, dataTimeSlotSerie, step_set='auto', samples=1000, data_loss_threshold=1):

        # Set evaluation steps if we have to
        if step_set == 'auto':
            step_set = [1, self.data['periodicity']]

        # Support var
        evaluation = {}
         
        # Find areas where to evaluate the model
        for key in dataTimeSlotSerie.data_keys():
             
            for steps in step_set:
                
                # Support vars
                real_values = []
                reconstructed_values = []
                processed_samples = 0

                # Here we will have steps=1, steps=2 .. steps=n          
                logger.debug('Evaluating model for steps %s', steps)
                
                for i in range(len(dataTimeSlotSerie)):
                                  
                    # Skip the first and the last ones, otherwise reconstruct the ones in the middle
                    if (i == 0) or (i >= len(dataTimeSlotSerie)-steps):
                        continue

                    # Is this a "good area" where to test or do we have to stop?
                    stop = False
                    if dataTimeSlotSerie[i-1].data_loss >= data_loss_threshold:
                        stop = True
                    for j in range(steps):
                        if dataTimeSlotSerie[i+j].data_loss >= data_loss_threshold:
                            stop = True
                            break
                    if dataTimeSlotSerie[i+steps].data_loss >= data_loss_threshold:
                        stop = True
                    if stop:
                        continue
                            
                    # Set prev and next
                    prev_value = dataTimeSlotSerie[i-1].data[key]
                    next_value = dataTimeSlotSerie[i+steps].data[key]
                    
                    # Compute average value
                    average_value = (prev_value+next_value)/2
                    
                    # Data to be reconstructed
                    dataTimeSlotSerie_to_reconstruct = DataTimeSlotSerie()
                    
                    # Append prev
                    #dataTimeSlotSerie_to_reconstruct.append(copy.deepcopy(dataTimeSlotSerie[i-1]))
                    
                    # Append in the middle and store real values
                    for j in range(steps):
                        dataTimeSlot = copy.deepcopy(dataTimeSlotSerie[i+j])
                        # Set the coverage to zero so the slot will be reconstructed
                        dataTimeSlot._coverage = 0
                        dataTimeSlot.data[key] = average_value
                        dataTimeSlotSerie_to_reconstruct.append(dataTimeSlot)
                        
                        real_values.append(dataTimeSlotSerie[i+j].data[key])
              
                    # Append next
                    #dataTimeSlotSerie_to_reconstruct.append(copy.deepcopy(dataTimeSlotSerie[i+steps]))

                    # Apply model inplace
                    self._apply(dataTimeSlotSerie_to_reconstruct, inplace=True)
                    processed_samples += 1

                    # Store reconstructed values
                    for j in range(steps):
                        reconstructed_values.append(dataTimeSlotSerie_to_reconstruct[j].data[key])
                    
                    if samples is not None and processed_samples >= samples:
                        break

                if processed_samples < samples:
                    logger.warning('Could not evaluate "{}" samples for "{}" steps, processed "{}" samples only'.format(samples, steps, processed_samples))

                if not reconstructed_values:
                    raise Exception('Could not evaluate model, maybe not enough data?')

                # Compute RMSE and ME, and add to the evaluation
                evaluation['rmse_{}_steps'.format(steps)] = mean_squared_error(real_values, reconstructed_values)
                evaluation['me_{}_steps'.format(steps)] = mean_error(real_values, reconstructed_values)

        # Compute average RMSE
        sum_rmse = 0
        count = 0
        for key in evaluation:
            if key.startswith('rmse_'):
                sum_rmse += evaluation[key]
                count += 1
        evaluation['mrmse'] = sum_rmse/count

        # Compute average ME
        sum_me = 0
        count = 0
        for key in evaluation:
            if key.startswith('me_'):
                sum_me += evaluation[key]
                count += 1
        evaluation['mme'] = sum_me/count
        
        return evaluation


    def _reconstruct(self, *args, **krargs):
        raise NotImplementedError('Reconstruction for this model is not yet implemented')




class PeriodicAverageReconstructor(Reconstructor):

    def _train(self, dataTimeSlotSerie, data_loss_threshold=0.5, periodicity=None, evaluation_step_set='auto', evaluation_samples=1000):

        if len(dataTimeSlotSerie.data_keys()) > 1:
            raise NotImplementedError('Multivariate time series are not yet supported')
        
        # Set or detect periodicity
        if periodicity is None:
            periodicity =  get_periodicity(dataTimeSlotSerie)
            if isinstance(dataTimeSlotSerie.slot_span, TimeSpan):
                logger.info('Detected periodicity: %sx %s', periodicity, dataTimeSlotSerie.slot_span)
            else:
                logger.info('Detected periodicity: %sx %ss', periodicity, dataTimeSlotSerie.slot_span)
        self.data['periodicity']=periodicity
        
        for key in dataTimeSlotSerie.data_keys():
            sums   = {}
            totals = {}
            for dataTimeSlot in dataTimeSlotSerie:
                if dataTimeSlot.data_loss < data_loss_threshold:
                    periodicity_index = get_periodicity_index(dataTimeSlot.start, dataTimeSlotSerie.slot_span, periodicity)
                    if not periodicity_index in sums:
                        sums[periodicity_index] = dataTimeSlot.data[key]
                        totals[periodicity_index] = 1
                    else:
                        sums[periodicity_index] += dataTimeSlot.data[key]
                        totals[periodicity_index] +=1

        averages={}
        for key in sums:
            averages[key] = sums[key]/totals[key]
        self.data['averages'] = averages

        self.data['evaluation'] = self._evaluate(dataTimeSlotSerie, step_set=evaluation_step_set, samples=evaluation_samples)
        logger.info('Model evaluation: "{}"'.format(self.data['evaluation']))


    def _reconstruct(self, dataTimeSlotSerie, key, from_index, to_index):
        logger.debug('Reconstructing between "{}" and "{}"'.format(from_index, to_index-1))
    
        # Compute offset
        diffs=0
        for j in range(from_index, to_index):
            real_value = dataTimeSlotSerie[j].data[key]
            periodicity_index = get_periodicity_index(dataTimeSlotSerie[j].start, dataTimeSlotSerie.slot_span, self.data['periodicity'])
            reconstructed_value = self.data['averages'][periodicity_index]
            diffs += (real_value - reconstructed_value)
        offset = diffs/(to_index-from_index)

        # Actually reconstruct
        for j in range(from_index, to_index):
            dataTimeSlot_to_reconstruct = dataTimeSlotSerie[j]
            periodicity_index = get_periodicity_index(dataTimeSlot_to_reconstruct.start, dataTimeSlotSerie.slot_span, self.data['periodicity'])
            dataTimeSlot_to_reconstruct.data[key] = self.data['averages'][periodicity_index] + offset
            dataTimeSlot_to_reconstruct._data_reconstructed = 1
                        


#======================
#  Forecast
#======================

class Forecaster(TrainableModel):

    def _evaluate(self, dataTimeSlotSerie, step_set='auto', samples=1000, plots=False):
        
        # Set evaluation steps if we have to
        if step_set == 'auto':
            step_set = [1, self.data['periodicity']]

        # Support var
        evaluation = {}

        for steps in step_set:
            
            # Support vars
            real_values = []
            model_values = []
            processed_samples = 0
    
            # For each point of the dataTimeSlotSerie, after the window, apply the prediction and compare it with the actual value
            for key in dataTimeSlotSerie.data_keys():
                for i in range(len(dataTimeSlotSerie)):
                    
                    # Check that we can get enough data
                    if i < self.data['window']+steps:
                        continue
                    if i > (len(dataTimeSlotSerie)-steps):
                        continue

                    # Compute the various boundaries
                    original_serie_boundaries_start = i - (self.data['window']) - steps
                    original_serie_boundaries_end = i
                    
                    original_forecast_serie_boundaries_start = original_serie_boundaries_start
                    original_forecast_serie_boundaries_end = original_serie_boundaries_end-steps
                    
                    # Create the time series where to apply the forecast on
                    forecast_dataTimeSlotSerie = DataTimeSlotSerie()
                    for j in range(original_forecast_serie_boundaries_start, original_forecast_serie_boundaries_end):
                        forecast_dataTimeSlotSerie.append(dataTimeSlotSerie[j])
 
                    # Apply the forecasting model
                    forecast_dataTimeSlotSerie = self._apply(forecast_dataTimeSlotSerie, n=steps, inplace=True)

                    # Plot evaluation time series?
                    if plots:
                        forecast_dataTimeSlotSerie.plot(log_js=False)
                    
                    # Compare each forecast with the original value
                    for step in range(steps):
                        original_index = original_serie_boundaries_start + self.data['window'] + step

                        forecast_index = self.data['window'] + step

                        model_value = forecast_dataTimeSlotSerie[forecast_index].data[key]
                        model_values.append(model_value)
                        
                        real_value = dataTimeSlotSerie[original_index].data[key]
                        real_values.append(real_value)
 
                    processed_samples+=1
                    if samples is not None and processed_samples >= samples:
                        break
                
            if processed_samples < samples:
                logger.warning('Could not evaluate "{}" samples for "{}" steps, processed "{}" samples only'.format(samples, steps, processed_samples))

            if not model_values:
                raise Exception('Could not evaluate model, maybe not enough data?')

            # Compute RMSE and ME, and add to the evaluation
            evaluation['rmse_{}_steps'.format(steps)] = mean_squared_error(real_values, model_values)
            evaluation['me_{}_steps'.format(steps)] = mean_error(real_values, model_values)
   
        # Compute average RMSE
        sum_rmse = 0
        count = 0
        for key in evaluation:
            if key.startswith('rmse_'):
                sum_rmse += evaluation[key]
                count += 1
        evaluation['mrmse'] = sum_rmse/count

        # Compute average ME
        sum_me = 0
        count = 0
        for key in evaluation:
            if key.startswith('me_'):
                sum_me += evaluation[key]
                count += 1
        evaluation['mme'] = sum_me/count
        
        return evaluation


    def _apply(self, dataTimeSlotSerie, n=1, inplace=False):

        if len(dataTimeSlotSerie.data_keys()) > 1:
            raise NotImplementedError('Multivariate time series are not yet supported')
 
        if len(dataTimeSlotSerie) < self.data['window']:
            raise ValueError('The dataTimeSlotSerie length ({}) is shorter than the model window ({}), it must be at least equal.'.format(len(dataTimeSlotSerie), self.data['window']))
 
        if inplace:
            forecast_dataTimeSlotSerie = dataTimeSlotSerie
        else:
            forecast_dataTimeSlotSerie = copy.deepcopy(dataTimeSlotSerie)
        
        for key in forecast_dataTimeSlotSerie.data_keys():

            # Support var
            first_call = True

            for _ in range(n):
                
                # Compute start/end for the slot to be forecasted
                if (isinstance(forecast_dataTimeSlotSerie.slot_span, float) or isinstance(forecast_dataTimeSlotSerie.slot_span, int)):
                    this_slot_start_t = forecast_dataTimeSlotSerie[-1].start.t + forecast_dataTimeSlotSerie.slot_span
                    this_slot_end_t   = this_slot_start_t + forecast_dataTimeSlotSerie.slot_span
                    this_slot_start_dt = dt_from_s(this_slot_start_t, tz=forecast_dataTimeSlotSerie.tz)
                    this_slot_end_dt = dt_from_s(this_slot_end_t, tz=forecast_dataTimeSlotSerie.tz )
                elif isinstance(forecast_dataTimeSlotSerie.slot_span, TimeSpan):
                    this_slot_start_dt = forecast_dataTimeSlotSerie[-1].start.dt + forecast_dataTimeSlotSerie.slot_span
                    this_slot_end_dt   =  this_slot_start_dt + forecast_dataTimeSlotSerie.slot_span
                    this_slot_start_t  = s_from_dt(this_slot_start_dt) 
                    this_slot_end_t    = s_from_dt(this_slot_end_dt)

                # Set time zone
                tz = forecast_dataTimeSlotSerie[-1].start.tz
                
                # Define TimePoints
                this_slot_start_timePoint = TimePoint(this_slot_start_t, tz=tz)
                this_slot_end_timePoint = TimePoint(this_slot_end_t, tz=tz)

                # Call model forecasting logic
                forecasted_dataTimeSlot = self._forecast(forecast_dataTimeSlotSerie, dataTimeSlotSerie.slot_span, key, this_slot_start_timePoint, this_slot_end_timePoint, first_call)

                # Add the forecast to the forecasts time series
                forecast_dataTimeSlotSerie.append(forecasted_dataTimeSlot)
                
                # Set fist call to false if this was the first call
                if first_call:
                    first_call = False 

        # Set serie mark for the forecast and return
        forecast_dataTimeSlotSerie.mark = [forecast_dataTimeSlotSerie[-n].start.dt, forecast_dataTimeSlotSerie[-1].end.dt]
        return forecast_dataTimeSlotSerie



class PeriodicAverageForecaster(Forecaster):
    
    def _train(self, dataTimeSlotSerie, window=None, periodicity=None, evaluation_step_set='auto', evaluation_samples=1000, evaluation_plots=False):

        if len(dataTimeSlotSerie.data_keys()) > 1:
            raise NotImplementedError('Multivariate time series are not yet supported')

        # Set or detect periodicity
        if periodicity is None:        
            periodicity =  get_periodicity(dataTimeSlotSerie)
            if isinstance(dataTimeSlotSerie.slot_span, TimeSpan):
                logger.info('Detected periodicity: %sx %s', periodicity, dataTimeSlotSerie.slot_span)
            else:
                logger.info('Detected periodicity: %sx %ss', periodicity, dataTimeSlotSerie.slot_span)
        self.data['periodicity'] = periodicity

        # Set or detect window
        if window:
            self.data['window'] = window
        else:
            logger.info('Using a window of "{}"'.format(periodicity))
            self.data['window'] = periodicity

        for key in dataTimeSlotSerie.data_keys():
            sums   = {}
            totals = {}
            for dataTimeSlot in dataTimeSlotSerie:
                periodicity_index = get_periodicity_index(dataTimeSlot.start, dataTimeSlotSerie.slot_span, periodicity)
                if not periodicity_index in sums:
                    sums[periodicity_index] = dataTimeSlot.data[key]
                    totals[periodicity_index] = 1
                else:
                    sums[periodicity_index] += dataTimeSlot.data[key]
                    totals[periodicity_index] +=1

        averages={}
        for key in sums:
            averages[key] = sums[key]/totals[key]
        self.data['averages'] = averages

        self.data['evaluation']  = self._evaluate(dataTimeSlotSerie, step_set=evaluation_step_set, samples=evaluation_samples, plots=evaluation_plots)
        logger.info('Model evaluation: "{}"'.format(self.data['evaluation']))


    def _forecast(self, forecast_dataTimeSlotSerie, slot_span, key, this_slot_start_timePoint, this_slot_end_timePoint, first_call) : #, dataTimeSlotSerie, key, from_index, to_index):

        # Compute the offset (avg diff between the real values and the forecasts on the first window)
        try:
            self.offsets
        except AttributeError:
            self.offsets={}
            
        if key not in self.offsets or first_call:

            diffs  = 0  
            for j in range(self.data['window']):
                serie_index = -(self.data['window']-j)
                real_value = forecast_dataTimeSlotSerie[serie_index].data[key]
                forecast_value = self.data['averages'][get_periodicity_index(forecast_dataTimeSlotSerie[serie_index].start, forecast_dataTimeSlotSerie.slot_span, self.data['periodicity'])]
                diffs += (real_value - forecast_value)
   
            # Sum the avg diff between the real and the forecast on the window to the forecast (the offset)
            offset = diffs/j
            self.offsets[key] = offset
        
        else:
            offset = self.offsets[key] 
        
        # Compute and add the real forecast data
        periodicity_index = get_periodicity_index(this_slot_start_timePoint, slot_span, self.data['periodicity'])        
        forecasted_dataTimeSlot = DataTimeSlot(start = this_slot_start_timePoint,
                                               end   = this_slot_end_timePoint,
                                               span  = forecast_dataTimeSlotSerie[-1].span,
                                               coverage = 0,
                                               data  = {key: self.data['averages'][periodicity_index] + (offset*1.0)})

        return forecasted_dataTimeSlot
    

    def _plot_averages(self, dataTimeSlotSerie, log_js=False):      
        averages_dataTimeSlotSerie = copy.deepcopy(dataTimeSlotSerie)
        for dataTimeSlot in averages_dataTimeSlotSerie:
            value = self.data['averages'][get_periodicity_index(dataTimeSlot.start, averages_dataTimeSlotSerie.slot_span, self.data['periodicity'])]
            if not value:
                value = 0
            dataTimeSlot.data['average'] =value 
        averages_dataTimeSlotSerie.plot(log_js=log_js)










