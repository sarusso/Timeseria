import unittest
import os
import tempfile
from math import sin, cos
from propertime.utils import dt

from ..datastructures import TimePoint, DataTimeSlot, DataTimePoint, TimeSeries
from ..models.base import Model, _KerasModel
from ..models.reconstructors import PeriodicAverageReconstructor, ProphetReconstructor
from ..models.forecasters import ProphetForecaster, PeriodicAverageForecaster, ARIMAForecaster, AARIMAForecaster, LSTMForecaster
from ..models.anomaly_detectors import PeriodicAverageAnomalyDetector
from ..exceptions import NotFittedError, NonContiguityError
from ..storages import CSVFileStorage

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


    def test_BaseForecasters(self):

        # This test is done using the Linear Interpolator Reconstructor

        time_series = TimeSeries()
        time_series.append(DataTimePoint(t=1,  data={'a':1}, data_loss=0))
        time_series.append(DataTimePoint(t=2,  data={'a':-99}, data_loss=1))
        time_series.append(DataTimePoint(t=3,  data={'a':3}, data_loss=1))
        time_series.append(DataTimePoint(t=4,  data={'a':4}, data_loss=0))

        forecaster = PeriodicAverageForecaster()
        forecaster.fitted = True
        forecaster.data['resolution'] = '1s'
        forecaster.data['data_labels'] = ['a']
        forecaster.data['window'] = 5

        # Not enough widow data for the predict, will raise
        with self.assertRaises(ValueError):
            forecaster.predict(time_series)


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
        forecaster.fit(self.sine_minute_time_series, start=20000.0, end=40000.0)
        evaluation = forecaster.evaluate(self.sine_minute_time_series, steps=[1,3], limit=100, details=True)
        self.assertAlmostEqual(evaluation['RMSE_1_steps'], 0.37831442005531923)

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

        loaded_forecaster = PeriodicAverageForecaster.load(model_path)

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
        for i in range(15):
            sine_minute_time_series.append(DataTimeSlot(start=TimePoint(i*60), end=TimePoint((i+1)*60), data={'value':sin(i/10.0)}))

        forecaster = LSTMForecaster()

        # Test the cross validation
        forecaster.cross_validate(sine_minute_time_series, rounds=3)

        # Fit and predict
        forecaster.fit(sine_minute_time_series)
        predicted_data = forecaster.predict(sine_minute_time_series)

        # Test with tolerance
        self.assertTrue(0.5 < predicted_data['value'] <1.1, 'got {}'.format(predicted_data['value']))

        # Not-existent features
        with self.assertRaises(ValueError):
            LSTMForecaster(features=['values','not_existent_feature']).fit(sine_minute_time_series)


        # Test using another feature
        LSTMForecaster(features=['values','diffs']).fit(sine_minute_time_series)


    def test_LSTMForecaster_multivariate(self):

        try:
            import tensorflow
        except ImportError:
            print('Skipping LSTM forecaster tests with multivariate series as no tensorflow module installed')
            return

        # Create a minute-resolution test DataTimeSlotSeries
        sine_cosine_minute_time_series = TimeSeries()
        for i in range(10):
            sine_cosine_minute_time_series.append(DataTimePoint(t=i*60,  data={'sin':sin(i/10.0), 'cos':cos(i/10.0)}))

        forecaster = LSTMForecaster()
        forecaster.fit(sine_cosine_minute_time_series)
        predicted_data = forecaster.predict(sine_cosine_minute_time_series)

        # Test with tolerance
        self.assertTrue(0.5 <  predicted_data['sin'] < 1.1, 'got {}'.format(predicted_data['sin']))
        self.assertTrue(0.5 <  predicted_data['cos'] < 1.1, 'got {}'.format(predicted_data['cos']))


    def test_LSTMForecaster_multivariate_with_targets(self):

        try:
            import tensorflow
        except ImportError:
            print('Skipping LSTM forecaster tests with multivariate serieswith targets as no tensorflow module installed')
            return

        # Create a minute-resolution test DataTimeSlotSeries
        sine_cosine_minute_time_series = TimeSeries()
        for i in range(10):
            sine_cosine_minute_time_series.append(DataTimePoint(t=i*60,  data={'sin':sin(i/10.0), 'cos':cos(i/10.0)}))

        # Target, no context
        forecaster = LSTMForecaster()
        forecaster.fit(sine_cosine_minute_time_series, target_data_labels=['cos'], with_context=False)

        with self.assertRaises(ValueError):
            forecaster.predict(sine_cosine_minute_time_series, context_data={'sin':0.5})

        predicted_data = forecaster.predict(sine_cosine_minute_time_series)
        self.assertEqual(list(predicted_data.keys()), ['cos'])
        self.assertTrue(0.5 <  predicted_data['cos'] < 1.1, 'got {}'.format(predicted_data['cos']))

        # All other methods will raise
        with self.assertRaises(ValueError):
            forecaster.forecast(sine_cosine_minute_time_series)

        with self.assertRaises(ValueError):
            forecaster.apply(sine_cosine_minute_time_series)

        # Target and context
        forecaster = LSTMForecaster()
        forecaster.fit(sine_cosine_minute_time_series, target_data_labels=['cos'], with_context=True)

        predicted_data = forecaster.predict(sine_cosine_minute_time_series, context_data={'sin':0.5})
        self.assertTrue(0.5 <  predicted_data['cos'] < 1.1, 'got {}'.format(predicted_data['cos']))

        forecasted_data = forecaster.forecast(sine_cosine_minute_time_series, context_data={'sin':0.5})
        self.assertEqual(forecasted_data.data['sin'], 0.5)
        self.assertTrue(0.5 <  forecasted_data.data['cos'] < 1.1, 'got {}'.format(predicted_data['cos']))

        series_with_forecast = forecaster.apply(sine_cosine_minute_time_series, context_data={'sin':0.5})
        self.assertEqual(series_with_forecast[-1].data['sin'], 0.5)
        self.assertTrue(0.5 <  series_with_forecast[-1].data['cos'] < 1.1, 'got {}'.format(predicted_data['cos']))


    def test_LSTMForecaster_save_load(self):

        try:
            import tensorflow
        except ImportError:
            print('Skipping LSTM save/load forecaster tests as no tensorflow module installed')
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
        loaded_forecaster = LSTMForecaster.load(model_path)

        # Predict from the loaded model
        predicted_data = loaded_forecaster.predict(sine_minute_time_series)

        self.assertTrue('sin' in predicted_data)
        self.assertTrue('cos' in predicted_data)


