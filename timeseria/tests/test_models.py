import unittest
import os
import tempfile
from ..datastructures import DataTimeSlotSerie, DataTimeSlot, TimePoint
from ..models import Model, TrainableModel
from ..models import PeriodicAverageReconstructor, PeriodicAverageForecaster
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
        dataTimeSlotSerie = Slotter(60*60*24).process(dataTimePointSerie)
        
        # Instantiate
        periodicAverageReconstructor = PeriodicAverageReconstructor()

        # Train
        evaluation = periodicAverageReconstructor.train(dataTimeSlotSerie)
        self.assertAlmostEqual(evaluation['rmse'], 0.15155635144666973)
        dataTimeSlotSerie_reconstructed = periodicAverageReconstructor.apply(dataTimeSlotSerie, data_loss_threshold=0.3)

        self.assertEqual(len(dataTimeSlotSerie), len(dataTimeSlotSerie_reconstructed))
        for i in range(len(dataTimeSlotSerie)):
            #print('------------------------------------------')
            #print(dataTimeSlotSerie[i], dataTimeSlotSerie[i].data_loss)
            #print(dataTimeSlotSerie_reconstructed[i], dataTimeSlotSerie_reconstructed[i].data_reconstructed)
            #print('------------------------------------------')

            if dataTimeSlotSerie[i].data_loss >= 0.3:
                if (dataTimeSlotSerie[i-1].data_loss < 0.3) and (dataTimeSlotSerie[i+1].data_loss < 0.3):
                    # You need at least two "missing" slots in succession for the reconstructor to kick in. 
                    pass
                else:
                    self.assertNotEqual(dataTimeSlotSerie_reconstructed[i].data, dataTimeSlotSerie[i].data, 'at position {}'.format(i))
        


class TestForecasters(unittest.TestCase):

    def setUp(self):
        
        # Create a test dataTimeSlotSeries
        from math import sin
        self.sine_dataTimeSlotSerie = DataTimeSlotSerie()
        for i in range(1000):
            self.sine_dataTimeSlotSerie.append(DataTimeSlot(start=TimePoint(i*60), end=TimePoint((i+1)*60), data={'value':sin(i/10.0)}))

    def test_PeriodicAverageForecaster(self):
                 
        forecaster = PeriodicAverageForecaster()
        
        evaluation = forecaster.train(self.sine_dataTimeSlotSerie, periodicity=63)
        
        self.assertTrue('rmse' in evaluation)
        
        self.assertEqual(forecaster.data['periodicity'], 63)

        forecaster.apply(self.sine_dataTimeSlotSerie, n=3)








