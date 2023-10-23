import unittest
import os
import tempfile
from math import sin, cos
from ..datastructures import TimePoint, DataTimeSlot, DataTimePoint, TimeSeries
from ..models.base import Model, _KerasModel
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
        forecaster.fit(self.sine_minute_time_series, start=20000, end=40000)
        evaluation = forecaster.evaluate(self.sine_minute_time_series, steps=[1,3], limit=100, details=True)
        self.assertAlmostEqual(evaluation['RMSE_1_steps'], 0.37831442005531923)

        # Fit to/from
        forecaster.fit(self.sine_minute_time_series, end=20000, start=40000)
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

