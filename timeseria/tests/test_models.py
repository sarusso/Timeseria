import unittest
import os
import tempfile
from math import sin, cos
from ..datastructures import TimePoint, DataTimeSlot, DataTimePoint, TimeSeries
from ..models.base import Model, SeriesModel, _KerasModel
from ..models.reconstructors import PeriodicAverageReconstructor, ProphetReconstructor
from ..models.forecasters import ProphetForecaster, PeriodicAverageForecaster, ARIMAForecaster, AARIMAForecaster, LSTMForecaster
from ..models.anomaly_detectors import PeriodicAverageAnomalyDetector
from ..exceptions import NotFittedError, NonContiguityError
from ..storages import CSVFileStorage
from ..time import dt

# Setup logging
from .. import logger
logger.setup()

# Test data and temp path 
TEST_DATA_PATH = '/'.join(os.path.realpath(__file__).split('/')[0:-1]) + '/test_data/'
TEMP_MODELS_DIR = tempfile.TemporaryDirectory().name

# Ensure reproducibility
import random
import numpy as np
random.seed(0)
np.random.seed(0)


class TestBaseModelClasses(unittest.TestCase):

    def test_Model(self):
                
        # Test non-parametric
        model = Model()
        self.assertFalse(model.is_parametric())
        self.assertEqual(model._type, 'non-parametric')
        with self.assertRaises(TypeError):
            model.save(TEMP_MODELS_DIR+'/test_model')
        with self.assertRaises(TypeError):
            model.id
        
        # Test parametric 
        class TestParametricModelOne(Model):
            def __init__(self, path=None, param1=0):
                self.data = {}
                self.data['param1'] = param1
                super(TestParametricModelOne, self).__init__(path)
        
        test_parametric_model_one = TestParametricModelOne(param1=1)
        self.assertTrue(test_parametric_model_one.is_parametric())
        self.assertEqual(test_parametric_model_one._type, 'parametric')
        self.assertEqual(test_parametric_model_one.data['param1'], 1)
        test_parametric_model_one.save(TEMP_MODELS_DIR+'/test_parametric_model_one')
        
        test_parametric_model_one = TestParametricModelOne(TEMP_MODELS_DIR+'/test_parametric_model_one')
        self.assertTrue(test_parametric_model_one.is_parametric())
        self.assertEqual(test_parametric_model_one.data['param1'], 1)

        # Test parametric with fit
        class TestParametricModelTwo(Model):
            def _fit(self, data):
                self.data['param1'] = data
            def _predict(self, data):
                return self.data['param1'] + data
            def _apply(self, data):
                return self.data['param1'] + data
            def _evaluate(self, data):
                return self.data['param1'] + data
        
        test_parametric_model_two = TestParametricModelTwo()
        self.assertTrue(test_parametric_model_two.is_parametric())
        self.assertEqual(test_parametric_model_two._type, 'parametric')        
        with self.assertRaises(NotFittedError):
            test_parametric_model_two.predict(3)
        with self.assertRaises(NotFittedError):
            test_parametric_model_two.apply(3)
        with self.assertRaises(NotFittedError):
            test_parametric_model_two.evaluate(3)
        with self.assertRaises(NotFittedError):
            test_parametric_model_two.save(TEMP_MODELS_DIR+'/test_parametric_model_two')
    
        test_parametric_model_two.fit(2)
        self.assertEqual(test_parametric_model_two.data['param1'], 2)
        self.assertEqual(test_parametric_model_two.predict(3), 5)
        self.assertEqual(test_parametric_model_two.apply(3), 5)
        self.assertEqual(test_parametric_model_two.evaluate(3), 5)
        self.assertEqual(test_parametric_model_two.apply(3), 5)
        test_parametric_model_two.save(TEMP_MODELS_DIR+'/test_parametric_model_two')        
        test_parametric_model_two = TestParametricModelTwo(TEMP_MODELS_DIR+'/test_parametric_model_two')
        self.assertTrue(test_parametric_model_two.is_parametric())
        self.assertEqual(test_parametric_model_two.data['param1'], 2)
        
  
    def test_SeriesModel(self):
        
        # Define a series model mock
        class SeriesModelMock(SeriesModel):   
            def _fit(self, *args, **kwargs):
                pass 
            def _evaluate(self, *args, **kwargs):
                pass
            def _apply(self, *args, **kwargs):
                pass 

        # Define test time series
        empty_time_series = TimeSeries()
        time_series = TimeSeries(DataTimeSlot(start=TimePoint(t=1), end=TimePoint(t=2), data={'metric1': 56}),
                                 DataTimeSlot(start=TimePoint(t=2), end=TimePoint(t=3), data={'metric1': 56}))

        # Instantiate a parametric model
        time_series_model = SeriesModelMock()
        time_series_model_id = time_series_model.id

        # Set model save path
        model_path = TEMP_MODELS_DIR+'/test_time_series_model'
        
        # Cannot apply model before fitting
        with self.assertRaises(NotFittedError):
            time_series_model.apply(time_series)   

        # Cannot save model before fitting
        with self.assertRaises(NotFittedError):
            time_series_model.save(model_path)  
        
        # Call the mock train
        with self.assertRaises(TypeError):
            time_series_model.fit('hello')

        with self.assertRaises(ValueError):
            time_series_model.fit(empty_time_series)
                         
        time_series_model.fit(time_series)
        
        # Call the mock apply
        with self.assertRaises(TypeError):
            time_series_model.apply('hello')

        with self.assertRaises(ValueError):
            time_series_model.apply(empty_time_series)
                         
        time_series_model.apply(time_series)
        
        # And save        
        time_series_model.save(model_path)
        
        # Now re-load
        loaded_time_series_model = SeriesModelMock(model_path)
        self.assertEqual(loaded_time_series_model.id, time_series_model_id)


    def test_KerasModel(self):
        
        # Define a Keras model mock
        class KerasModelMock(_KerasModel):            
            def _fit(self, *args, **kwargs):
                pass 
            def _evaluate(self, *args, **kwargs):
                pass
            def _apply(self, *args, **kwargs):
                pass 

        # Define test time series
        data_time_point_series = TimeSeries(DataTimePoint(t=1, data={'metric1': 0.1}),
                                            DataTimePoint(t=2, data={'metric1': 0.2}),
                                            DataTimePoint(t=3, data={'metric1': 0.3}),
                                            DataTimePoint(t=4, data={'metric1': 0.4}),
                                            DataTimePoint(t=5, data={'metric1': 0.5}),
                                            DataTimePoint(t=6, data={'metric1': 0.6}),)


        # Test window generation functions
        window_matrix = KerasModelMock._to_window_datapoints_matrix(data_time_point_series, window=2, steps=1, encoder=None)
        
        # What to expect (using the timestamp to represent a data point):
        # 1,2
        # 2,3
        # 3,4
        # 4,5

        self.assertEqual(len(window_matrix), 4)
        
        self.assertEqual(window_matrix[0][0].t, 1)
        self.assertEqual(window_matrix[0][1].t, 2)
        
        self.assertEqual(window_matrix[1][0].t, 2)
        self.assertEqual(window_matrix[1][1].t, 3)

        self.assertEqual(window_matrix[2][0].t, 3)
        self.assertEqual(window_matrix[2][1].t, 4)
        
        self.assertEqual(window_matrix[-1][0].t, 4)
        self.assertEqual(window_matrix[-1][1].t, 5)
        
        target_vector = KerasModelMock._to_target_values_vector(data_time_point_series, window=2, steps=1)

        # What to expect (using the data value to represent a data point):
        # [0.3], [0.4], [0.5], [0.6] Note that they are lists in order to support multi-step forecast
        self.assertEqual(target_vector, [[0.3], [0.4], [0.5], [0.6]])
        


class TestReconstructors(unittest.TestCase):

    def test_PeriodicAverageReconstructor(self):
        
        # Get test data 
        with open(TEST_DATA_PATH + '/csv/temp_slots_1h.csv') as f:
            data=f.read()
        
        time_series = TimeSeries()
        for line in data.split('\n'):
            if line:
                start_t = float(line.split(',')[0])
                start_point = TimePoint(t=start_t)
                end_point = TimePoint(t=start_t+3600)
                data_loss = 1-float(line.split(',')[2]) # from coverage to data loss
                value =  float(line.split(',')[1])
                time_series.append(DataTimeSlot(start=start_point, end=end_point, data={'temperature':value}, data_loss=data_loss))
            
        # Instantiate
        reconstructor = PeriodicAverageReconstructor()

        # Fit
        reconstructor.fit(time_series)
        
        # Evaluate
        evaluation = reconstructor.evaluate(time_series, limit=100, details=True)
        self.assertAlmostEqual(evaluation['RMSE_1_steps'], 0.138541066038798)
        self.assertAlmostEqual(evaluation['MAE_1_steps'], 0.10236999999999981)
        self.assertAlmostEqual(evaluation['RMSE_24_steps'], 0.5635685435140884)
        self.assertAlmostEqual(evaluation['MAE_24_steps'], 0.4779002102269731)
        self.assertAlmostEqual(evaluation['RMSE'], 0.3510548047764432)
        self.assertAlmostEqual(evaluation['MAE'], 0.29013510511348645)

        # Evaluate n specific steps:
        evaluation = reconstructor.evaluate(time_series, steps=[1,3], limit=100, details=True)
        self.assertAlmostEqual(evaluation['RMSE_1_steps'], 0.138541066038798)
        self.assertAlmostEqual(evaluation['MAE_1_steps'], 0.10236999999999981)
        self.assertAlmostEqual(evaluation['RMSE_3_steps'], 0.24455061001646802)
        self.assertAlmostEqual(evaluation['MAE_3_steps'], 0.190163072234738)
        self.assertAlmostEqual(evaluation['RMSE'], 0.19154583802763303)
        self.assertAlmostEqual(evaluation['MAE'], 0.1462665361173689)

        # Apply
        time_series_reconstructed = reconstructor.apply(time_series, data_loss_threshold=0.3)
        self.assertEqual(len(time_series), len(time_series_reconstructed))
        for i in range(len(time_series)):
            if time_series[i].data_loss >= 0.3:
                if (time_series[i-1].data_loss < 0.3) and (time_series[i+1].data_loss < 0.3):
                    # You need at least two "missing" slots in succession for the reconstructor to kick in. 
                    pass
                else:
                    self.assertNotEqual(time_series_reconstructed[i].data, time_series[i].data, 'at position {}'.format(i))
        
        # Fit from/to
        reconstructor = PeriodicAverageReconstructor()
        reconstructor.fit(time_series, from_dt=dt(2019,3,1), to_dt=dt(2019,4,1))
        evaluation = reconstructor.evaluate(time_series, steps=[1,3], limit=100, details=True)
        self.assertAlmostEqual(evaluation['RMSE_3_steps'], 0.28011618729276533)

        # Fit to/from
        reconstructor = PeriodicAverageReconstructor()
        reconstructor.fit(time_series, to_dt=dt(2019,3,1), from_dt=dt(2019,4,1))
        evaluation = reconstructor.evaluate(time_series, steps=[1,3], limit=100, details=True)
        self.assertAlmostEqual(evaluation['RMSE_3_steps'], 0.23978759586375123)
        
        # Cross validations
        reconstructor = PeriodicAverageReconstructor()
        cross_validation = reconstructor.cross_validate(time_series, evaluate_steps=[1,3], evaluate_limit=100, evaluate_details=True)
        self.assertAlmostEqual(cross_validation['MAE_3_steps_avg'],  0.24029106113368606)
        self.assertAlmostEqual(cross_validation['MAE_3_steps_stdev'], 0.047895485925726636)

        # Test on Points as well
        time_series = CSVFileStorage(TEST_DATA_PATH + '/csv/temperature.csv').get(limit=200)
        reconstructor = PeriodicAverageReconstructor()
        with self.assertRaises(Exception):
            reconstructor.fit(time_series)
         
        time_series = time_series.resample(600)
        reconstructor.fit(time_series)
         
        # TODO: do some actual testing.. not only that "it works"
        time_series_reconstructed  = reconstructor.apply(time_series)
        reconstructor.evaluate(time_series)
        

    def test_ProphetReconstructor(self):
        try:
            import prophet
        except ImportError:
            print('Skipping Prophet tests as no prophet module installed')
            return
            
        # Get test data        
        time_series = CSVFileStorage(TEST_DATA_PATH + '/csv/temperature.csv').get(limit=200).aggregate(3600)
        
        # Instantiate
        reconstructor = ProphetReconstructor()

        # Fit
        reconstructor.fit(time_series)

        # Apply
        time_series_reconstructed = reconstructor.apply(time_series, data_loss_threshold=0.3)
        self.assertEqual(len(time_series), len(time_series_reconstructed))
        for i in range(len(time_series)):
            if time_series[i].data_loss >= 0.3:
                if (time_series[i-1].data_loss < 0.3) and (time_series[i+1].data_loss < 0.3):
                    # You need at least two "missing" slots in succession for the reconstructor to kick in. 
                    pass
                else:
                    self.assertNotEqual(time_series_reconstructed[i].data, time_series[i].data, 'at position {}'.format(i))


    def test_LinearInterpolationReconstructor(self):
                
        # Test series
        time_series = TimeSeries()
        time_series.append(DataTimePoint(t=-3,  data={'value':3}))
        time_series.append(DataTimePoint(t=-2,  data={'value':2}))
        time_series.append(DataTimePoint(t=-1,  data={'value':1}))
        time_series.append(DataTimePoint(t=0,  data={'value':0}))
        time_series.append(DataTimePoint(t=1,  data={'value':1}))
        time_series.append(DataTimePoint(t=2,  data={'value':2}))
        time_series.append(DataTimePoint(t=3,  data={'value':3}))
        time_series.append(DataTimePoint(t=4,  data={'value':4}))
        time_series.append(DataTimePoint(t=5,  data={'value':5}))
        time_series.append(DataTimePoint(t=6,  data={'value':6}))
        time_series.append(DataTimePoint(t=7,  data={'value':7}))
        time_series.append(DataTimePoint(t=16, data={'value':16}))
        time_series.append(DataTimePoint(t=17, data={'value':17}))
        time_series.append(DataTimePoint(t=18, data={'value':18}))
        
        # Resample for 3 seconds
        resampled_time_series = time_series.resample(3)
        # DataTimePoint @ 0.0 (1970-01-01 00:00:00+00:00) with data "{'value': 0.6666666666666666}" and data_loss="0.0"
        # DataTimePoint @ 3.0 (1970-01-01 00:00:03+00:00) with data "{'value': 3.0}" and data_loss="0.0"
        # DataTimePoint @ 6.0 (1970-01-01 00:00:06+00:00) with data "{'value': 6.0}" and data_loss="0.0"
        # DataTimePoint @ 9.0 (1970-01-01 00:00:09+00:00) with data "{'value': 9.0}" and data_loss="1.0"
        # DataTimePoint @ 12.0 (1970-01-01 00:00:12+00:00) with data "{'value': 12.0}" and data_loss="1.0"
        # DataTimePoint @ 15.0 (1970-01-01 00:00:15+00:00) with data "{'value': 15.0}" and data_loss="0.6666666666666667"
        
        from ..models.reconstructors import LinearInterpolationReconstructor
        
        # Fake data to check the recostructor is working correctly
        resampled_time_series[3].data['value']=-9
        resampled_time_series[4].data['value']=-12
        
        # Instantiate the reconstructor
        reconstructor = LinearInterpolationReconstructor()
        
        # Check parametric param just in case
        self.assertFalse(reconstructor.is_parametric())
        
        # Apply the reconstructor
        reconstructed_time_series = reconstructor.apply(resampled_time_series)
        
        # Check len
        self.assertEqual(len(reconstructed_time_series), 6)
        
        # Check for t = 0
        self.assertEqual(reconstructed_time_series[0].t, 0)
        self.assertAlmostEqual(reconstructed_time_series[0].data['value'], 0.6666666666)  # Expected: 0.66..
        self.assertEqual(reconstructed_time_series[0].data_loss, 0)
        
        # Check for t = 3
        self.assertEqual(reconstructed_time_series[1].t, 3)
        self.assertEqual(reconstructed_time_series[1].data['value'], 3)  # Expected: 3
        self.assertEqual(reconstructed_time_series[1].data_loss, 0)
        
        # Check for t = 6
        self.assertEqual(reconstructed_time_series[2].t, 6)
        self.assertEqual(reconstructed_time_series[2].data['value'], 6)  # Expected: 6
        self.assertEqual(reconstructed_time_series[2].data_loss, 0)
        
        # Check for t = 9
        self.assertEqual(reconstructed_time_series[3].t, 9)
        self.assertEqual(reconstructed_time_series[3].data['value'], 9)  # Expected: 9 (fully reconstructed)
        self.assertEqual(reconstructed_time_series[3].data_loss, 1)
        
        # Check for t = 12
        self.assertEqual(reconstructed_time_series[4].t, 12)
        self.assertEqual(reconstructed_time_series[4].data['value'], 12)  # Expected: 12 (fully reconstructed)
        self.assertEqual(reconstructed_time_series[4].data_loss, 1)
        
        # Check for t = 15
        self.assertEqual(reconstructed_time_series[5].t, 15)
        self.assertEqual(reconstructed_time_series[5].data['value'], 15)  # Expected: 15
        self.assertAlmostEqual(reconstructed_time_series[5].data_loss, 0.6666666666)


class TestForecasters(unittest.TestCase):

    def setUp(self):
        
        # Create a minute-resolution test DataTimeSlotSeries
        self.sine_minute_time_series = TimeSeries()
        for i in range(1000):
            self.sine_minute_time_series.append(DataTimeSlot(start=TimePoint(i*60), end=TimePoint((i+1)*60), data={'value':sin(i/10.0)}))

        # Create a day-resolution test DataTimeSlotSeries
        self.sine_day_time_series = TimeSeries()
        for i in range(1000):
            step = 60 * 60 * 24
            self.sine_day_time_series.append(DataTimeSlot(start=TimePoint(i*step), end=TimePoint((i+1)*step), data={'value':sin(i/10.0)}))


    def test_PeriodicAverageForecaster(self):
                 
        forecaster = PeriodicAverageForecaster()
        
        # Fit
        forecaster.fit(self.sine_minute_time_series, periodicity=63)

        # Apply
        sine_minute_time_series_with_forecast = forecaster.apply(self.sine_minute_time_series, steps=3)
        self.assertEqual(len(sine_minute_time_series_with_forecast), 1003)

        # Predict
        prediction = forecaster.predict(self.sine_minute_time_series, steps=3)
        self.assertTrue(isinstance(prediction, list))
        self.assertEqual(len(prediction), 3)

        # Evaluate
        evaluation = forecaster.evaluate(self.sine_minute_time_series, steps='auto', limit=100, details=True)
        self.assertEqual(forecaster.data['periodicity'], 63)
        self.assertAlmostEqual(evaluation['RMSE_1_steps'], 0.07318673229600292)
        self.assertAlmostEqual(evaluation['MAE_1_steps'], 0.06622794526818285)
        self.assertAlmostEqual(evaluation['RMSE_63_steps'], 0.06697755802373265)
        self.assertAlmostEqual(evaluation['MAE_63_steps'], 0.06016205183857482)     
        self.assertAlmostEqual(evaluation['RMSE'], 0.07008214515986778)
        self.assertAlmostEqual(evaluation['MAE'], 0.06319499855337883)

        # Evaluate
        evaluation = forecaster.evaluate(self.sine_minute_time_series, steps=[1,3], limit=100, details=True)
        self.assertEqual(forecaster.data['periodicity'], 63)
        self.assertAlmostEqual(evaluation['RMSE_1_steps'], 0.07318673229600292)
        self.assertAlmostEqual(evaluation['MAE_1_steps'], 0.06622794526818285)
        self.assertAlmostEqual(evaluation['RMSE_3_steps'], 0.07253018513852955)
        self.assertAlmostEqual(evaluation['MAE_3_steps'], 0.06567523200748912)     

        # Fit from/to
        forecaster.fit(self.sine_minute_time_series, from_t=20000, to_t=40000)
        evaluation = forecaster.evaluate(self.sine_minute_time_series, steps=[1,3], limit=100, details=True)
        self.assertAlmostEqual(evaluation['RMSE_1_steps'], 0.37831442005531923)

        # Fit to/from
        forecaster.fit(self.sine_minute_time_series, to_t=20000, from_t=40000)
        evaluation = forecaster.evaluate(self.sine_minute_time_series, steps=[1,3], limit=100, details=True)
        self.assertAlmostEqual(evaluation['RMSE_1_steps'], 0.36033834603736264)

        # Test on Points as well
        time_series = CSVFileStorage(TEST_DATA_PATH + '/csv/temperature.csv').get(limit=200)
        forecaster = PeriodicAverageForecaster()
        with self.assertRaises(Exception):
            forecaster.fit(time_series)
          
        time_series = time_series.resample(600)
        forecaster.fit(time_series)
          
        # TODO: do some actual testing.. not only that "it works"
        forecasted_time_series  = forecaster.apply(time_series)


    def test_PeriodicAverageForecaster_save_load(self):
        
        forecaster = PeriodicAverageForecaster()
        
        forecaster.fit(self.sine_minute_time_series, periodicity=63)
        
        model_path = TEMP_MODELS_DIR + '/test_PA_model'
        
        forecaster.save(model_path)
        
        loaded_forecaster = PeriodicAverageForecaster(model_path)
        
        self.assertEqual(forecaster.data['averages'], loaded_forecaster.data['averages'])

        # TODO: do some actual testing.. not only that "it works"
        forecasted_time_series = loaded_forecaster.apply(self.sine_minute_time_series)


    def test_ProphetForecaster(self):

        try:
            import prophet
        except ImportError:
            print('Skipping Prophet tests as no prophet module installed')
            return
         
        forecaster = ProphetForecaster()
         
        forecaster.fit(self.sine_day_time_series)
        self.assertEqual(len(self.sine_day_time_series), 1000)
  
        sine_day_time_series_with_forecast = forecaster.apply(self.sine_day_time_series, steps=3)
        self.assertEqual(len(sine_day_time_series_with_forecast), 1003)

        # Test the evaluate
        evalation_results = forecaster.evaluate(self.sine_day_time_series, limit=10)
        self.assertAlmostEqual(evalation_results['RMSE'], 0.82, places=2)
        self.assertAlmostEqual(evalation_results['MAE'], 0.81, places=2)

        evalation_results = forecaster.evaluate(self.sine_day_time_series, limit=1)
        self.assertAlmostEqual(evalation_results['RMSE'], 0.54, places=2) # For one sample they must be the same
        self.assertAlmostEqual(evalation_results['MAE'], 0.54, places=2) # For one sample they must be the same
        
        # Test on Points as well
        time_series = CSVFileStorage(TEST_DATA_PATH + '/csv/temperature.csv').get(limit=200)
        forecaster = ProphetForecaster()
        with self.assertRaises(Exception):
            forecaster.fit(time_series)
           
        time_series = time_series.resample(600)
        forecaster.fit(time_series)
           
        # TODO: do some actual testing.. not only that "it works"
        forecasted_time_series = forecaster.apply(time_series)


    def test_ARIMAForecaster(self):

        try:
            import statsmodels
        except ImportError:
            print('Skipping ARIMA tests as no statsmodels module installed')
            return
         
        # Basic ARIMA 
        forecaster = ARIMAForecaster(p=1,d=1,q=0)
         
        forecaster.fit(self.sine_day_time_series)
        self.assertEqual(len(self.sine_day_time_series), 1000)

        # Cannot apply on a time series contiguous with the time series used for the fit
        with self.assertRaises(NonContiguityError):
            forecaster.apply(self.sine_day_time_series[:-1], steps=3)
    
        # Can apply on a time series contiguous with the fit one
        sine_day_time_series_with_forecast = forecaster.apply(self.sine_day_time_series, steps=3)
        self.assertEqual(len(sine_day_time_series_with_forecast), 1003)

        # Cannot evaluate on a time series not contiguous with the time series used for the fit
        with self.assertRaises(NonContiguityError):
            forecaster.evaluate(self.sine_day_time_series)

        # Can evaluate on a time series contiguous with the time series used for the fit
        forecaster = ARIMAForecaster(p=1,d=1,q=0)
        forecaster.fit(self.sine_day_time_series[0:800])                 
        evaluation_results = forecaster.evaluate(self.sine_day_time_series[800:1000])
        self.assertAlmostEqual(evaluation_results['RMSE'], 2.71, places=2)
        self.assertAlmostEqual(evaluation_results['MAE'], 2.52, places=2 )
 
        # Test on Points as well
        time_series = CSVFileStorage(TEST_DATA_PATH + '/csv/temperature.csv').get(limit=200)
        forecaster = ARIMAForecaster()
        with self.assertRaises(ValueError):
            forecaster.fit(time_series)
           
        time_series = time_series.resample(600)
        forecaster.fit(time_series)
           
        # TODO: do some actual testing.. not only that "it works"
        forecasted_time_series  = forecaster.apply(time_series)


    def test_AARIMAForecaster(self):

        try:
            import statsmodels
        except ImportError:
            print('Skipping AARIMA tests as no statsmodels module installed')
            return
         
        # Automatic ARIMA 
        forecaster = AARIMAForecaster()
         
        forecaster.fit(self.sine_day_time_series, max_p=2, max_d=1, max_q=2)
        self.assertEqual(len(self.sine_day_time_series), 1000)

        # Cannot apply on a time series contiguous with the time series used for the fit
        with self.assertRaises(NonContiguityError):
            forecaster.apply(self.sine_day_time_series[:-1], steps=3)
    
        # Can apply on a time series contiguous with the same item as the fit one
        sine_day_time_series_with_forecast = forecaster.apply(self.sine_day_time_series, steps=3)
        self.assertEqual(len(sine_day_time_series_with_forecast), 1003)

        # Cannot evaluate on a time series not contiguous with the time series used for the fit
        with self.assertRaises(NonContiguityError):
            forecaster.evaluate(self.sine_day_time_series)

        # Can evaluate on a time series contiguous with the time series used for the fit
        forecaster = AARIMAForecaster()
        forecaster.fit(self.sine_day_time_series[0:800], max_p=2, max_d=1, max_q=2)                 
        evaluation_results = forecaster.evaluate(self.sine_day_time_series[800:1000])
        self.assertTrue('RMSE' in evaluation_results)
        self.assertTrue('MAE' in evaluation_results)
        # Cannot test values, some random behavior which cannot be put under control is present somewhere
        #self.assertAlmostEqual(evaluation_results['RMSE'], 0.7179428895746799)
        #self.assertAlmostEqual(evaluation_results['MAE'], 0.6497934134525981)
 
        # Test on Points as well
        time_series = CSVFileStorage(TEST_DATA_PATH + '/csv/temperature.csv').get(limit=200)
        forecaster = AARIMAForecaster()
        with self.assertRaises(ValueError):
            forecaster.fit(time_series)
           
        time_series = time_series.resample(600)
        forecaster.fit(time_series, max_p=2, max_d=1, max_q=2)
           
        # TODO: do some actual testing.. not only that "it works"
        forecasted_time_series  = forecaster.apply(time_series)


    def test_LSTMForecaster(self):

        try:
            import tensorflow
        except ImportError:
            print('Skipping LSTM forecaster tests as no tensorflow module installed')
            return

        # Create a minute-resolution test DataTimeSlotSeries
        sine_minute_time_series = TimeSeries()
        for i in range(10):
            sine_minute_time_series.append(DataTimeSlot(start=TimePoint(i*60), end=TimePoint((i+1)*60), data={'value':sin(i/10.0)}))

        forecaster = LSTMForecaster()
        forecaster.fit(sine_minute_time_series)
        predicted_value = forecaster.predict(sine_minute_time_series)['value']
        
        # Give some tolerance
        self.assertTrue(predicted_value>0.5)
        self.assertTrue(predicted_value<1.1)
        
        # Not-existent features
        with self.assertRaises(ValueError):
            LSTMForecaster(features=['values','not_existent_feature']).fit(sine_minute_time_series)

        # Test using another feature
        LSTMForecaster(features=['values','diffs']).fit(sine_minute_time_series)


    def test_LSTMForecaster_multivariate(self):

        try:
            import tensorflow
        except ImportError:
            print('Skipping LSTM forecaster tests as no tensorflow module installed')
            return
        
        # Create a minute-resolution test DataTimeSlotSeries
        sine_minute_time_series = TimeSeries()
        for i in range(10):
            sine_minute_time_series.append(DataTimeSlot(start=TimePoint(i*60), end=TimePoint((i+1)*60), data={'sin':sin(i/10.0), 'cos':cos(i/10.0)}))

        forecaster = LSTMForecaster()
        forecaster.fit(sine_minute_time_series)
        predicted_data = forecaster.predict(sine_minute_time_series)
        
        self.assertTrue('sin' in predicted_data)
        self.assertTrue('cos' in predicted_data)


    def test_LSTMForecaster_save_load(self):

        try:
            import tensorflow
        except ImportError:
            print('Skipping LSTM forecaster tests as no tensorflow module installed')
            return
        
        # Create a minute-resolution test DataTimeSlotSeries
        sine_minute_time_series = TimeSeries()
        for i in range(10):
            sine_minute_time_series.append(DataTimeSlot(start=TimePoint(i*60), end=TimePoint((i+1)*60), data={'sin':sin(i/10.0), 'cos':cos(i/10.0)}))

        forecaster = LSTMForecaster()
        forecaster.fit(sine_minute_time_series)
        
        # Set model path
        model_path = TEMP_MODELS_DIR+ '/test_LSTM_model'
        
        # Save        
        forecaster.save(model_path)

        # Load
        loaded_forecaster = LSTMForecaster(model_path)
        
        # Predict from the loaded model 
        predicted_data = loaded_forecaster.predict(sine_minute_time_series)
        
        self.assertTrue('sin' in predicted_data)
        self.assertTrue('cos' in predicted_data)   
        


class TestAnomalyDetectors(unittest.TestCase):

    def setUp(self):
        
        # Create a minute-resolution test DataTimeSlotSeries
        self.sine_minute_time_series = TimeSeries()
        for i in range(1000):
            if i % 100 == 0:
                value = 2
            else:
                value = sin(i/10.0)

            self.sine_minute_time_series.append(DataTimeSlot(start=TimePoint(i*60), end=TimePoint((i+1)*60), data={'value':value}))


    def test_PeriodicAverageAnomalyDetector(self):
        
        anomaly_detector = PeriodicAverageAnomalyDetector()
        
        anomaly_detector.fit(self.sine_minute_time_series, periodicity=63)
        
        self.assertAlmostEqual(anomaly_detector.data['AE_threshold'], 0.5914733390853167)
        
        result_time_series = anomaly_detector.apply(self.sine_minute_time_series)

        # Count how many anomalies were detected
        anomalies_count = 0
        for slot in result_time_series:
            if slot.data_indexes['anomaly']:
                anomalies_count += 1
        self.assertEqual(anomalies_count, 9)

        # Test on Points as well
        time_series = CSVFileStorage(TEST_DATA_PATH + '/csv/temperature.csv').get(limit=200)
        anomaly_detector = PeriodicAverageAnomalyDetector()
        with self.assertRaises(ValueError):
            anomaly_detector.fit(time_series)
          
        time_series = time_series.resample(600)
        anomaly_detector.fit(time_series)
          
        # TODO: do some actual testing.. not only that "it works"
        anomaly_time_series  = anomaly_detector.apply(time_series)


    def test_PeriodicAverageAnomalyDetector_save_load(self):
        
        anomaly_detector = PeriodicAverageAnomalyDetector()
        
        anomaly_detector.fit(self.sine_minute_time_series, periodicity=63)
       
        # Set model save path
        model_path = TEMP_MODELS_DIR+'/test_anomaly_model'
        
        anomaly_detector.save(model_path)
        
        loaded_anomaly_detector = PeriodicAverageAnomalyDetector(model_path)
        
        self.assertEqual(anomaly_detector.data['AE_threshold'], loaded_anomaly_detector.data['AE_threshold'])
        self.assertEqual(anomaly_detector.forecaster.data['averages'], loaded_anomaly_detector.forecaster.data['averages'])

        result_time_series = loaded_anomaly_detector.apply(self.sine_minute_time_series)

