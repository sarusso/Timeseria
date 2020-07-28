import unittest
import os
import tempfile
from ..datastructures import DataTimeSlotSeries, DataTimeSlot, TimePoint
from ..models import Model, ParametricModel
from ..models import PeriodicAverageReconstructor, PeriodicAverageForecaster
from ..exceptions import NotFittedError
from ..storages import CSVFileStorage
from ..transformations import Slotter

# Setup logging
import logging
logging.basicConfig(level=os.environ.get('LOGLEVEL') if os.environ.get('LOGLEVEL') else 'CRITICAL')

TEST_DATA_PATH = '/'.join(os.path.realpath(__file__).split('/')[0:-1]) + '/test_data/'

TEMP_MODELS_DIR = tempfile.TemporaryDirectory().name

class TestBaseModelClasses(unittest.TestCase):

    def test_Model(self):
        model = Model()
        
    def test_ParametricModel(self):
        
        
        # Define a trainable model mock
        class ParametricModelMock(ParametricModel):            
            def _fit(self, *args, **kwargs):
                pass 
            def _evaluate(self, *args, **kwargs):
                pass
            def _apply(self, *args, **kwargs):
                pass 

        # Define test time series
        empty_data_time_slot_series = DataTimeSlotSeries()
        data_time_slot_series = DataTimeSlotSeries(DataTimeSlot(start=TimePoint(t=1), end=TimePoint(t=2), data={'metric1': 56}),
                                                   DataTimeSlot(start=TimePoint(t=2), end=TimePoint(t=3), data={'metric1': 56}))

        # Instantiate a trainable model
        parametric_model = ParametricModelMock()
        parametric_model_id = parametric_model.id
        
        # Cannot apply model before training
        with self.assertRaises(NotFittedError):
            parametric_model.apply(data_time_slot_series)   

        # Cannot save model before training
        with self.assertRaises(NotFittedError):
            parametric_model.save(TEMP_MODELS_DIR)  
        
        # Call the mock train
        with self.assertRaises(TypeError):
            parametric_model.fit('hello')

        with self.assertRaises(ValueError):
            parametric_model.fit(empty_data_time_slot_series)
                         
        parametric_model.fit(data_time_slot_series)
        
        # Call the mock apply
        with self.assertRaises(TypeError):
            parametric_model.apply('hello')

        with self.assertRaises(ValueError):
            parametric_model.apply(empty_data_time_slot_series)
                         
        parametric_model.apply(data_time_slot_series)
        
        # And save
        model_dir = parametric_model.save(TEMP_MODELS_DIR)
        
        # Now re-load
        loaded_parametric_model = ParametricModelMock(model_dir)
        self.assertEqual(loaded_parametric_model.id, parametric_model_id)



class TestReconstructors(unittest.TestCase):

    def test_PeriodicAverageReconstructor(self):
        
        # Get test data        
        data_time_point_series = CSVFileStorage(TEST_DATA_PATH + '/csv/temperature.csv').get()
        data_time_slot_series = Slotter('3600s').process(data_time_point_series)
        
        # Instantiate
        periodic_average_reconstructor = PeriodicAverageReconstructor()

        # Fit
        periodic_average_reconstructor.fit(data_time_slot_series, evaluation_samples=100)
        evaluation = periodic_average_reconstructor.evaluation_score
        self.assertAlmostEqual(evaluation['rmse_1_steps'], 0.019193626979166593)
        self.assertAlmostEqual(evaluation['me_1_steps'], 0.10236999999999981)
        self.assertAlmostEqual(evaluation['rmse_24_steps'], 0.3176095032385909)
        self.assertAlmostEqual(evaluation['me_24_steps'], 0.4779002102269731)
        self.assertAlmostEqual(evaluation['mrmse'], 0.16840156510887874)
        self.assertAlmostEqual(evaluation['mme'], 0.29013510511348645)

        # Fit again but evaluate on specific steps:
        periodic_average_reconstructor.fit(data_time_slot_series, evaluation_steps_set=[1,3], evaluation_samples=100)
        evaluation = periodic_average_reconstructor.evaluation_score
        self.assertAlmostEqual(evaluation['rmse_1_steps'], 0.019193626979166593)
        self.assertAlmostEqual(evaluation['me_1_steps'], 0.10236999999999981)
        self.assertAlmostEqual(evaluation['rmse_3_steps'], 0.05980500085942663)
        self.assertAlmostEqual(evaluation['me_3_steps'], 0.190163072234738)
        self.assertAlmostEqual(evaluation['mrmse'], 0.03949931391929661)
        self.assertAlmostEqual(evaluation['mme'], 0.1462665361173689)
        
        # Apply
        data_time_slot_series_reconstructed = periodic_average_reconstructor.apply(data_time_slot_series, data_loss_threshold=0.3)
        self.assertEqual(len(data_time_slot_series), len(data_time_slot_series_reconstructed))
        for i in range(len(data_time_slot_series)):
            if data_time_slot_series[i].data_loss >= 0.3:
                if (data_time_slot_series[i-1].data_loss < 0.3) and (data_time_slot_series[i+1].data_loss < 0.3):
                    # You need at least two "missing" slots in succession for the reconstructor to kick in. 
                    pass
                else:
                    self.assertNotEqual(data_time_slot_series_reconstructed[i].data, data_time_slot_series[i].data, 'at position {}'.format(i))
        


class TestForecasters(unittest.TestCase):

    def setUp(self):
        
        # Create a test DataTimeSlotSeries
        from math import sin
        self.sine_data_time_slot_series = DataTimeSlotSeries()
        for i in range(1000):
            self.sine_data_time_slot_series.append(DataTimeSlot(start=TimePoint(i*60), end=TimePoint((i+1)*60), data={'value':sin(i/10.0)}))

    def test_PeriodicAverageForecaster(self):
                 
        forecaster = PeriodicAverageForecaster()
        
        forecaster.fit(self.sine_data_time_slot_series, periodicity=63, evaluation_steps_set='auto', evaluation_samples=100)
        evaluation = forecaster.evaluation_score
        self.assertEqual(forecaster.data['periodicity'], 63)
        self.assertAlmostEqual(evaluation['rmse_1_steps'], 0.005356297784166798)
        self.assertAlmostEqual(evaluation['me_1_steps'], 0.06622794526818285)
        self.assertAlmostEqual(evaluation['rmse_63_steps'], 0.004485993278822473)
        self.assertAlmostEqual(evaluation['me_63_steps'], 0.06016205183857482)     
        self.assertAlmostEqual(evaluation['mrmse'], 0.004921145531494635)
        self.assertAlmostEqual(evaluation['mme'], 0.06319499855337883)

        forecaster.fit(self.sine_data_time_slot_series, periodicity=63, evaluation_steps_set=[1,3], evaluation_samples=100)
        evaluation = forecaster.evaluation_score
        self.assertEqual(forecaster.data['periodicity'], 63)
        self.assertAlmostEqual(evaluation['rmse_1_steps'], 0.005356297784166798)
        self.assertAlmostEqual(evaluation['me_1_steps'], 0.06622794526818285)
        self.assertAlmostEqual(evaluation['rmse_3_steps'], 0.005260627756229373)
        self.assertAlmostEqual(evaluation['me_3_steps'], 0.06567523200748912)     

 
        forecast_sine_data_time_slot_series = forecaster.apply(self.sine_data_time_slot_series, n=3)
        self.assertEqual(len(self.sine_data_time_slot_series)+3, len(forecast_sine_data_time_slot_series))








