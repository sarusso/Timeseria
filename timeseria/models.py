import os
import json
import uuid
import copy
from .datastructures import DataTimeSlotSerie, DataTimeSlot, TimePoint
from .exceptions import NotTrainedError
from .utilities import get_periodicity
from .time import now_t, TimeSpan, dt_from_s, s_from_dt

# Setup logging
import logging
logger = logging.getLogger(__name__)

HARD_DEBUG = False

#======================
#  Utility functions
#======================

def get_periodicity_index(i, timePoint, dataTimeSlotSerie, periodicity):

    # Handle specific cases
    if isinstance(dataTimeSlotSerie.slot_span, TimeSpan):
        
        raise NotImplementedError(str(dataTimeSlotSerie.slot_span.type))
        # Handle cases affected by DST
        if dataTimeSlotSerie.slot_span.hours:
            periodicity_index = int(timePoint.dt.hour/dataTimeSlotSerie.slot_span.hours)
        elif dataTimeSlotSerie.slot_span.minutes:
            periodicity_index = int(timePoint.dt.hour/dataTimeSlotSerie.slot_span.minutes)
        else:
            periodicity_index = int(timePoint.t % periodicity)
        
    else:
        
        #INFO:timeseria.models:Detected periodicity: 144x 600
        #INFO:timeseria.models:Getting periodicity index for "TimePoint @ t=1546477200.0 (2019-01-03 01:00:00+00:00)" : "0" (periodicty=144)
        slot_start_t = timePoint.t
        # Get index based on slot start, normalized to span, modulus periodicity
        periodicity_index =  int(slot_start_t / dataTimeSlotSerie.slot_span) % periodicity
        #periodicity_index = i%periodicity

    #logger.info('Getting periodicity index for "{}" : "{}" (periodicty={})'.format(timePoint, periodicity_index, periodicity))
    return periodicity_index

    
    #if periodicity == 24 and (dataTimeSlotSerie.slot_span == TimeSpan('1h') or dataTimeSlotSerie.slot_span==3600):
    #    return str(timePoint.dt.hour)
    #elif periodicity == 144 and (dataTimeSlotSerie.slot_span == TimeSpan('10m') or dataTimeSlotSerie.slot_span==144):
    #    return str(int(timePoint.dt.minute/10))
    #else:
    #    return int(i % periodicity)


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

        self._train(dataTimeSlotSerie, *args, **kwargs)
        self.data['trained_at'] = now_t()
        self.trained = True

        evaluation = self._evaluate(dataTimeSlotSerie)
        logger.info('Model evaluation: "{}"'.format(evaluation))
        return evaluation
        
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

    @property
    def id(self):
        return self.data['id']

    def _train(self, *args, **krargs):
        raise NotImplementedError('Training this model is not yet implemented')

    def _apply(self, *args, **krargs):
        raise NotImplementedError('Applying this model is not yet implemented')

    def _evaluate(self, *args, **krargs):
        raise NotImplementedError('Evaluating this model is not yet implemented')



#======================
# Data Reconstruction
#======================

class PeriodicAverageReconstructor(TrainableModel):

    def _train(self, dataTimeSlotSerie, data_loss_threshold=0.5, periodicity=None):

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
            for i, dataTimeSlot in enumerate(dataTimeSlotSerie):
                if dataTimeSlot.data_loss < data_loss_threshold:
                    periodicity_index = get_periodicity_index(i, dataTimeSlot.start, dataTimeSlotSerie, periodicity)
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
                    
                        #logger.info('Reconstructing between "{}" and "{}"'.format(gap_started, i-1))
                    
                        # Compute offset
                        diffs=0
                        for j in range(gap_started, i):
                            real_value = dataTimeSlotSerie[j].data[key]
                            periodicity_index = get_periodicity_index(j, dataTimeSlotSerie[j].start, dataTimeSlotSerie, self.data['periodicity'])
                            reconstructed_value = self.data['averages'][periodicity_index]
                            diffs += (real_value - reconstructed_value)
                        offset = diffs/(i-gap_started)

                        # Actually reconstruct
                        for j in range(gap_started, i):
                            dataTimeSlot_to_reconstruct = dataTimeSlotSerie[j]
                            periodicity_index = get_periodicity_index(j, dataTimeSlot_to_reconstruct.start, dataTimeSlotSerie, self.data['periodicity'])
                            dataTimeSlot_to_reconstruct.data[key] = self.data['averages'][periodicity_index] + offset
                            dataTimeSlot_to_reconstruct._data_reconstructed = 1
                    
                    
                    gap_started=None
                    
                    dataTimeSlot._data_reconstructed = 0
                    
                if remove_data_loss:
                    # TOOD: move to None if we allow data_losses (coverages) to None?
                    dataTimeSlot._coverage = 1
        
        return dataTimeSlotSerie
        
                

    def _evaluate(self, dataTimeSlotSerie):

        
        # Find areas where to evaluate the model
        for key in dataTimeSlotSerie.data_keys():
            
            true_values          = []
            reconstructed_values = []  
            
            for i in range(len(dataTimeSlotSerie)):  
                              
                # Skip the first and the last
                if (i == 0) or (i == len(dataTimeSlotSerie)-1):
                    continue
                
                # Otherwise, for each three points reconstruct the one in the middle
                prev_value = dataTimeSlotSerie[i-1].data[key]
                this_value = dataTimeSlotSerie[i].data[key]
                next_value = dataTimeSlotSerie[i+1].data[key]                
                this_value_as_if_missing = (prev_value+next_value)/2

                # Get periodicity index
                periodicity_index = get_periodicity_index(i, dataTimeSlotSerie[i].start, dataTimeSlotSerie, self.data['periodicity'])

                # Compute offset
                reconstructed_value = self.data['averages'][periodicity_index]
                offset = this_value_as_if_missing - reconstructed_value
                # TODO: we are not evaluating the offset logic for gaps > 1 with this evaluation logic, which is basically not evaluating the model at all. Fix me!

                # Actually reconstruct
                reconstructed_value = self.data['averages'][periodicity_index] + offset

                # Log                
                #logger.debug('%s vs %s', reconstructed_value, this_value)
                
                # Add to real and reconstructed arrays
                true_values.append(this_value)
                reconstructed_values.append(reconstructed_value)

        from sklearn.metrics import mean_squared_error
        rmse = mean_squared_error(true_values, reconstructed_values)
        return {'rmse': rmse}





#======================
#  Forecast
#======================

class Forecaster(TrainableModel):
    pass


class PeriodicAverageForecaster(TrainableModel):
    '''Evaluation is done for a single data point in future'''
    
    def _train(self, dataTimeSlotSerie, window=None, periodicity=None):

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

        # Set or detect window
        if window:
            self.data['window'] = window
        else:
            logger.info('Using a window of "{}"'.format(periodicity))
            self.data['window'] = periodicity

        for key in dataTimeSlotSerie.data_keys():
            sums   = {}
            totals = {}
            for i, dataTimeSlot in enumerate(dataTimeSlotSerie):
                periodicity_index = get_periodicity_index(i, dataTimeSlot.start, dataTimeSlotSerie, periodicity)
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
            offset = None
            for i in range(n):
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

                #logger.info('-------------------------------------------')
                #logger.info('start={}'.format(this_slot_start_dt))
                #logger.info('end={}'.format(this_slot_end_dt))

                tz = forecast_dataTimeSlotSerie[-1].start.tz
                
                # Define TimePoints
                this_slot_start_timePoint = TimePoint(this_slot_start_t, tz=tz)
                this_slot_end_timePoint = TimePoint(this_slot_end_t, tz=tz)
                
                # Get periodicity index
                periodicity_index = get_periodicity_index(i+len(dataTimeSlotSerie), this_slot_start_timePoint, dataTimeSlotSerie, self.data['periodicity'])
                
                # Compute the diffs between the real and the forecast on the window
                window = self.data['window']
                if offset is None:
                    diffs  = 0  
                    for j in range(window):
                        serie_index = -(window-j)
                        real_value = forecast_dataTimeSlotSerie[serie_index].data[key]
                        forecast_value = self.data['averages'][get_periodicity_index(j, forecast_dataTimeSlotSerie[serie_index].start, forecast_dataTimeSlotSerie, self.data['periodicity'])]
                        #logger.info('{}: {} vs {}'.format(serie_index, real_value, forecast_value))
                        diffs += (real_value - forecast_value)
                    
                    # Sum the avg diff between the real and the forecast on the window to the forecast (the offset)
                    offset = diffs/j
                forecast = self.data['averages'][periodicity_index]

                # Debug
                #logger.info('forecast=%s', forecast)                
                #logger.info('offset=%s', offset)
                #logger.info('forecast+offset=%s', forecast+offset)

                forecasted_dataTimeSlot = DataTimeSlot(start = this_slot_start_timePoint,
                                                       end   = this_slot_end_timePoint,
                                                       span  = forecast_dataTimeSlotSerie[-1].span,
                                                       coverage = 0,
                                                       data  = {key: forecast+(offset*1.00)})

                forecasted_dataTimeSlot._data_reconstructed = 0
                forecast_dataTimeSlotSerie.append(forecasted_dataTimeSlot)
                 

                #logger.info('-------------------------------------------')

        # Set serie mark for the forecast        
        forecast_dataTimeSlotSerie.mark = [dataTimeSlotSerie[-1].end.dt, this_slot_end_dt]
        #logger.info('Returning {}'.format(forecast_dataTimeSlotSerie))
        return forecast_dataTimeSlotSerie



    def _plot_averages(self, dataTimeSlotSerie):
        
        
        averages_dataTimeSlotSerie = copy.deepcopy(dataTimeSlotSerie)
        
        for i, dataTimeSlot in enumerate(averages_dataTimeSlotSerie):
            dataTimeSlot.data['average']=  self.data['averages'][get_periodicity_index(i, dataTimeSlot.start, averages_dataTimeSlotSerie, self.data['periodicity'])]

        #if dataTimeSlotSerie:
        averages_dataTimeSlotSerie.plot()
        

    def _evaluate(self, dataTimeSlotSerie, until=None):
        if True:
            window = self.data['window']
            true_values = []
            model_values = []
    
            # For each point of the dataTimeSlotSerie, after the window, apply the prediction and compare it with the actual value
            for key in dataTimeSlotSerie.data_keys():
                for i in range(len(dataTimeSlotSerie)):
                    if i < window+1:
                        continue
                    if until is not None and i > until:
                        break
                    # TODO: create a "shifting" time series rather than creating it each time.
                    forecast_dataTimeSlotSerie = DataTimeSlotSerie()
                    for j in range(i-window-1, i-1):
                        forecast_dataTimeSlotSerie.append(dataTimeSlotSerie[j])
                    forecast_dataTimeSlotSerie = self._apply(forecast_dataTimeSlotSerie, n=1, inplace=True)
                    
                    model_values.append(forecast_dataTimeSlotSerie[-1].data[key])
                    true_values.append(dataTimeSlotSerie[i].data[key])
    
            #print(true_values)        
            #print(model_values)
     
            from sklearn.metrics import mean_squared_error
            rmse = mean_squared_error(true_values, model_values)
        else:
            rmse=None
        return {'rmse':rmse}

    
    






















