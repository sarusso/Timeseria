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
        dataTimeSlotSerie = Slotter(60*60).process(dataTimePointSerie)
        
        # Instantiate
        periodicAverageReconstructor = PeriodicAverageReconstructor()

        # Train
        periodicAverageReconstructor.train(dataTimeSlotSerie, evaluation_samples=100)
        evaluation = periodicAverageReconstructor.evaluation
        self.assertAlmostEqual(evaluation['rmse_1_steps'], 0.019193626979166593)
        self.assertAlmostEqual(evaluation['me_1_steps'], 0.10236999999999981)
        self.assertAlmostEqual(evaluation['rmse_24_steps'], 0.3176095032385909)
        self.assertAlmostEqual(evaluation['me_24_steps'], 0.4779002102269731)
        self.assertAlmostEqual(evaluation['mrmse'], 0.16840156510887874)
        self.assertAlmostEqual(evaluation['mme'], 0.29013510511348645)

        # Train again but evaluate on specific steps:
        periodicAverageReconstructor.train(dataTimeSlotSerie, evaluation_step_set=[1,3], evaluation_samples=100)
        evaluation = periodicAverageReconstructor.evaluation
        self.assertAlmostEqual(evaluation['rmse_1_steps'], 0.019193626979166593)
        self.assertAlmostEqual(evaluation['me_1_steps'], 0.10236999999999981)
        self.assertAlmostEqual(evaluation['rmse_3_steps'], 0.05980500085942663)
        self.assertAlmostEqual(evaluation['me_3_steps'], 0.190163072234738)
        self.assertAlmostEqual(evaluation['mrmse'], 0.03949931391929661)
        self.assertAlmostEqual(evaluation['mme'], 0.1462665361173689)
        
        # Apply
        dataTimeSlotSerie_reconstructed = periodicAverageReconstructor.apply(dataTimeSlotSerie, data_loss_threshold=0.3)
        self.assertEqual(len(dataTimeSlotSerie), len(dataTimeSlotSerie_reconstructed))
        for i in range(len(dataTimeSlotSerie)):
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
        
        forecaster.train(self.sine_dataTimeSlotSerie, periodicity=63, evaluation_step_set='auto', evaluation_samples=100)
        evaluation = forecaster.evaluation
        self.assertEqual(forecaster.data['periodicity'], 63)
        self.assertAlmostEqual(evaluation['rmse_1_steps'], 0.005356297784166798)
        self.assertAlmostEqual(evaluation['me_1_steps'], 0.06622794526818285)
        self.assertAlmostEqual(evaluation['rmse_63_steps'], 0.004485993278822473)
        self.assertAlmostEqual(evaluation['me_63_steps'], 0.06016205183857482)     
        self.assertAlmostEqual(evaluation['mrmse'], 0.004921145531494635)
        self.assertAlmostEqual(evaluation['mme'], 0.06319499855337883)

        forecaster.train(self.sine_dataTimeSlotSerie, periodicity=63, evaluation_step_set=[1,3], evaluation_samples=100)
        evaluation = forecaster.evaluation
        self.assertEqual(forecaster.data['periodicity'], 63)
        self.assertAlmostEqual(evaluation['rmse_1_steps'], 0.005356297784166798)
        self.assertAlmostEqual(evaluation['me_1_steps'], 0.06622794526818285)
        self.assertAlmostEqual(evaluation['rmse_3_steps'], 0.005260627756229373)
        self.assertAlmostEqual(evaluation['me_3_steps'], 0.06567523200748912)     

 
        forecast_sine_dataTimeSlotSerie = forecaster.apply(self.sine_dataTimeSlotSerie, n=3)
        self.assertEqual(len(self.sine_dataTimeSlotSerie)+3, len(forecast_sine_dataTimeSlotSerie))








