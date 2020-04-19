import os
import json
import uuid
import copy
from .datastructures import DataTimeSlotSerie
from .exceptions import NotTrainedError
from .utilities import get_periodicity
from .time import now_t, TimeSpan

# Setup logging
import logging
logger = logging.getLogger(__name__)

HARD_DEBUG = False


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



#======================
# Data Reconstruction
#======================

class PeriodicAverageReconstructor(TrainableModel):
    
    def get_periodicity_index(self, i, TimePoint, dataTimeSlotSerie):

        # Handle specific cases
        if dataTimeSlotSerie.timeSpan == TimeSpan('1h'):
            return str(TimePoint.dt.hour)
        else:
            return str(i % 24)
    
    def _train(self, dataTimeSlotSerie, data_loss_threshold=0.5):

        if len(dataTimeSlotSerie.data_keys()) > 1:
            raise NotImplementedError('Multivariate time series are not yet supported')
        
        # Detect periodicity
        periodicity =  get_periodicity(dataTimeSlotSerie)
        logger.info('Detected periodicity: %sx %s', periodicity, dataTimeSlotSerie.timeSpan)
        
        for key in dataTimeSlotSerie.data_keys():
            sums   = {}
            totals = {}
            for i, dataTimeSlot in enumerate(dataTimeSlotSerie):
                if dataTimeSlot.data_loss < data_loss_threshold:
                    periodicity_index = self.get_periodicity_index(i, dataTimeSlot.start, dataTimeSlotSerie)
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

    def _apply(self, dataTimeSlotSerie, remove_data_loss=False, data_loss_threshold=0.5):
        logger.debug('Using data_loss_threshold="%s"', data_loss_threshold)
        dataTimeSlotSerie = copy.deepcopy(dataTimeSlotSerie)
        
        if len(dataTimeSlotSerie.data_keys()) > 1:
            raise NotImplementedError('Multivariate time series are not yet supported')

        for key in dataTimeSlotSerie.data_keys():
            for i, dataTimeSlot in enumerate(dataTimeSlotSerie):
                if dataTimeSlot.data_loss >= data_loss_threshold:
                    periodicity_index = self.get_periodicity_index(i, dataTimeSlot.start, dataTimeSlotSerie)
                    dataTimeSlot.data[key] = self.data['averages'][periodicity_index]
                    dataTimeSlot._data_reconstructed = 1 #dataTimeSlot.data_loss
                else:
                    dataTimeSlot._data_reconstructed = 0
                if remove_data_loss:
                    # TOOD: move to None if we allow data_losses (coverages) to None?
                    dataTimeSlot._coverage = 1
        
        return dataTimeSlotSerie
        
                

    def _evaluate(self, dataTimeSlotSerie):
        dataTimeSlotSerie_reconstructed =  self.apply(dataTimeSlotSerie, data_loss_threshold=0)
        true_values          = []
        reconstructed_values = []
        for key in dataTimeSlotSerie.data_keys():
            for i in range(len(dataTimeSlotSerie)):
                true_values.append(dataTimeSlotSerie[i].data[key])
                reconstructed_values.append(dataTimeSlotSerie_reconstructed[i].data[key])

        from sklearn.metrics import mean_squared_error
        rmse = mean_squared_error(true_values, reconstructed_values)
        return {'rmse':rmse}




















