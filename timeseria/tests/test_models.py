import unittest
import os
import tempfile
from ..datastructures import DataTimeSlotSerie, DataTimeSlot, TimePoint
from ..models import Model, TrainableModel
from ..models import PeriodicAverageReconstructor
from ..exceptions import NotTrainedError
from ..storages import CSVFileStorage
from ..operators import Slotter

# Setup logging
import logging
logging.basicConfig(level=os.environ.get('LOGLEVEL') if os.environ.get('LOGLEVEL') else 'CRITICAL')

TEST_DATA_PATH = '/'.join(os.path.realpath(__file__).split('/')[0:-1]) + '/test_data/'

TEMP_MODELS_DIR = tempfile.TemporaryDirectory().name

class TestBaseModelClasses(unittest.TestCase):

    def test_Model(self):
        model = Model()
        
    def test_TrainableModel(self):
        
        
        # Define a trainable model mock
        class TrainableModelMock(TrainableModel):            
            def _train(self, *args, **kwargs):
                pass 
            def _evaluate(self, *args, **kwargs):
                pass
            def _apply(self, *args, **kwargs):
                pass 

        # Define test time series
        empty_dataTimeSlotSerie = DataTimeSlotSerie()
        dataTimeSlotSerie = DataTimeSlotSerie(DataTimeSlot(start=TimePoint(t=1), end=TimePoint(t=2), data={'metric1': 56}),
                                              DataTimeSlot(start=TimePoint(t=2), end=TimePoint(t=3), data={'metric1': 56}))

        # Instantiate a trainable model
        trainableModel = TrainableModelMock()
        trainableModel_id = trainableModel.id
        
        # Cannot apply model before training
        with self.assertRaises(NotTrainedError):
            trainableModel.apply(dataTimeSlotSerie)   

        # Cannot save model before training
        with self.assertRaises(NotTrainedError):
            trainableModel.save(TEMP_MODELS_DIR)  
        
        # Call the mock train
        with self.assertRaises(TypeError):
            trainableModel.train('hello')

        with self.assertRaises(ValueError):
            trainableModel.train(empty_dataTimeSlotSerie)
                         
        trainableModel.train(dataTimeSlotSerie)
        
        # Call the mock apply
        with self.assertRaises(TypeError):
            trainableModel.apply('hello')

        with self.assertRaises(ValueError):
            trainableModel.apply(empty_dataTimeSlotSerie)
                         
        trainableModel.apply(dataTimeSlotSerie)
        
        # And save
        model_dir = trainableModel.save(TEMP_MODELS_DIR)
        
        # Now re-load
        loaded_trainableModel = TrainableModelMock(model_dir)
        self.assertEqual(loaded_trainableModel.id, trainableModel_id)



class TestReconstructors(unittest.TestCase):

    def test_PeriodicAverageReconstructor(self):
        
        # Get test data        
        dataTimePointSerie = CSVFileStorage(TEST_DATA_PATH + '/csv/temperature.csv').get()
        dataTimeSlotSerie = Slotter('1h').process(dataTimePointSerie)
        
        # Instantiate
        periodicAverageReconstructor = PeriodicAverageReconstructor()

        # Train

        evaluation = periodicAverageReconstructor.train(dataTimeSlotSerie)
        self.assertAlmostEqual(evaluation['rmse'], 1.0499504255337846)
        dataTimeSlotSerie_reconstructed = periodicAverageReconstructor.apply(dataTimeSlotSerie, data_loss_threshold=0.3)

        self.assertEqual(len(dataTimeSlotSerie), len(dataTimeSlotSerie_reconstructed))
        for i in range(len(dataTimeSlotSerie)):
            if dataTimeSlotSerie[i].data_loss > 0.3:
                self.assertNotEqual(dataTimeSlotSerie_reconstructed[i].data, dataTimeSlotSerie[i].data, 'at position {}'.format(i))
        















