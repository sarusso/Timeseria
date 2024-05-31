import unittest
import os
import tempfile
import random
import numpy
from math import sin, cos
from propertime.utils import dt

from ..datastructures import TimePoint, DataTimeSlot, DataTimePoint, TimeSeries
from ..models.base import Model, _KerasModel
from ..models.reconstructors import PeriodicAverageReconstructor, ProphetReconstructor
from ..models.forecasters import ProphetForecaster, PeriodicAverageForecaster, ARIMAForecaster, AARIMAForecaster, LSTMForecaster
from ..models.anomaly_detectors import PeriodicAverageAnomalyDetector
from ..exceptions import NotFittedError, NonContiguityError
from ..storages import CSVFileStorage
from .. import TEST_DATASETS_PATH

# Setup logging
from .. import logger
logger.setup()

# Test data and temp path
TEST_DATA_PATH = '/'.join(os.path.realpath(__file__).split('/')[0:-1]) + '/test_data/'
TEMP_MODELS_DIR = tempfile.TemporaryDirectory().name


class TestForecasters(unittest.TestCase):

    def setUp(self):

        # Create a minute-resolution test DataTimeSlotSeries
        self.sine_minute_timeseries = TimeSeries()
        for i in range(1000):
            self.sine_minute_timeseries.append(DataTimeSlot(start=TimePoint(i*60), end=TimePoint((i+1)*60), data={'value':sin(i/10.0)}))

        # Create a day-resolution test DataTimeSlotSeries
        self.sine_day_timeseries = TimeSeries()
        for i in range(1000):
            step = 60 * 60 * 24
            self.sine_day_timeseries.append(DataTimeSlot(start=TimePoint(i*step), end=TimePoint((i+1)*step), data={'value':sin(i/10.0)}))

    def test_BaseForecasters(self):

        # TODO: This test is done using the Periodic Average Forecaster, make a mock

        timeseries = TimeSeries()
        timeseries.append(DataTimePoint(t=1,  data={'a':1}, data_loss=0))
        timeseries.append(DataTimePoint(t=2,  data={'a':-99}, data_loss=1))
        timeseries.append(DataTimePoint(t=3,  data={'a':3}, data_loss=1))
        timeseries.append(DataTimePoint(t=4,  data={'a':4}, data_loss=0))

        forecaster = PeriodicAverageForecaster()
        forecaster.fitted = True
        forecaster.data['resolution'] = '1s'
        forecaster.data['data_labels'] = ['a']
        forecaster.data['window'] = 5

        # Not enough widow data for the predict, will raise
        with self.assertRaises(ValueError):
            forecaster.predict(timeseries)


    def test_PeriodicAverageForecaster(self):

        forecaster = PeriodicAverageForecaster()

        # Fit
        forecaster.fit(self.sine_minute_timeseries, periodicity=63)

        # Apply
        sine_minute_timeseries_with_forecast = forecaster.apply(self.sine_minute_timeseries, steps=3)
        self.assertEqual(len(sine_minute_timeseries_with_forecast), 1003)

        # Predict
        prediction = forecaster.predict(self.sine_minute_timeseries, steps=3)
        self.assertTrue(isinstance(prediction, list))
        self.assertEqual(len(prediction), 3)

        # Evaluate
        evaluation = forecaster.evaluate(self.sine_minute_timeseries[0:100], steps=1)
        self.assertEqual(forecaster.data['periodicities']['value'], 63)
        self.assertAlmostEqual(evaluation['value_RMSE'], 0.081, places=2)
        self.assertAlmostEqual(evaluation['value_MAE'], 0.074, places=2)

        # Test on Points as well
        timeseries = CSVFileStorage(TEST_DATA_PATH + '/csv/temperature.csv').get(limit=200)
        forecaster = PeriodicAverageForecaster()
        with self.assertRaises(Exception):
            forecaster.fit(timeseries)

        timeseries = timeseries.resample(600)
        forecaster.fit(timeseries)

        # TODO: do some actual testing.. not only that "it works"
        forecasted_timeseries  = forecaster.apply(timeseries)


    def test_PeriodicAverageForecaster_multivariate(self):
        try:
            import tensorflow
        except ImportError:
            print('Skipping LSTM forecaster tests with multivariate time series as no tensorflow module installed')
            return

        timeseries = TimeSeries()
        for i in range(980):
            timeseries.append(DataTimePoint(t=i*60, data={'sin': sin(i/10.0), 'cos': cos(i/10.0)}))
        forecaster = PeriodicAverageForecaster()
        forecaster.fit(timeseries)

        # TODO: do some actual testing.. not only that "it works"
        timeseries_with_forecast = forecaster.apply(timeseries[0:970], steps=5)


    def test_PeriodicAverageForecaster_save_load(self):

        forecaster = PeriodicAverageForecaster()

        forecaster.fit(self.sine_minute_timeseries, periodicity=63)

        model_path = TEMP_MODELS_DIR + '/test_PA_model'

        forecaster.save(model_path)

        loaded_forecaster = PeriodicAverageForecaster.load(model_path)

        self.assertEqual(forecaster.data['averages'], loaded_forecaster.data['averages'])

        # TODO: do some actual testing.. not only that "it works"
        forecasted_timeseries = loaded_forecaster.apply(self.sine_minute_timeseries)


    def test_ProphetForecaster(self):

        try:
            import prophet
        except ImportError:
            print('Skipping Prophet tests as no prophet module installed')
            return

        forecaster = ProphetForecaster()

        forecaster.fit(self.sine_day_timeseries)
        self.assertEqual(len(self.sine_day_timeseries), 1000)

        sine_day_timeseries_with_forecast = forecaster.apply(self.sine_day_timeseries, steps=3)
        self.assertEqual(len(sine_day_timeseries_with_forecast), 1003)

        # Test the evaluate
        evalation_results = forecaster.evaluate(self.sine_day_timeseries.view(0,10))
        self.assertAlmostEqual(evalation_results['value_RMSE'], 0.80, places=2)
        self.assertAlmostEqual(evalation_results['value_MAE'], 0.77, places=2)

        evalation_results = forecaster.evaluate(self.sine_day_timeseries.view(0,1))
        self.assertAlmostEqual(evalation_results['value_RMSE'], 0.53, places=2) # For one sample they must be the same
        self.assertAlmostEqual(evalation_results['value_MAE'], 0.53, places=2) # For one sample they must be the same

        # Test on Points as well
        timeseries = CSVFileStorage(TEST_DATA_PATH + '/csv/temperature.csv').get(limit=200)
        forecaster = ProphetForecaster()
        with self.assertRaises(Exception):
            forecaster.fit(timeseries)

        timeseries = timeseries.resample(600)
        forecaster.fit(timeseries)

        # TODO: do some actual testing.. not only that "it works"
        forecasted_timeseries = forecaster.apply(timeseries)


    def test_ARIMAForecaster(self):

        try:
            import statsmodels
        except ImportError:
            print('Skipping ARIMA tests as no statsmodels module installed')
            return

        # Basic ARIMA
        forecaster = ARIMAForecaster(p=1,d=1,q=0)

        forecaster.fit(self.sine_day_timeseries)
        self.assertEqual(len(self.sine_day_timeseries), 1000)

        # Cannot apply on a time series contiguous with the time series used for the fit
        with self.assertRaises(NonContiguityError):
            forecaster.apply(self.sine_day_timeseries[:-1], steps=3)

        # Can apply on a time series contiguous with the fit one
        sine_day_timeseries_with_forecast = forecaster.apply(self.sine_day_timeseries, steps=3)
        self.assertEqual(len(sine_day_timeseries_with_forecast), 1003)

        # # Cannot evaluate on a time series not contiguous with the time series used for the fit
        # with self.assertRaises(NonContiguityError):
        #     forecaster.evaluate(self.sine_day_timeseries)
        #
        # # Can evaluate on a time series contiguous with the time series used for the fit
        # forecaster = ARIMAForecaster(p=1,d=1,q=0)
        # forecaster.fit(self.sine_day_timeseries[0:800])
        # evaluation_results = forecaster.evaluate(self.sine_day_timeseries[800:1000])
        # self.assertAlmostEqual(evaluation_results['RMSE'], 2.71, places=2)
        # self.assertAlmostEqual(evaluation_results['MAE'], 2.52, places=2 )

        # Test on Points as well
        timeseries = CSVFileStorage(TEST_DATA_PATH + '/csv/temperature.csv').get(limit=200)
        forecaster = ARIMAForecaster()
        with self.assertRaises(ValueError):
            forecaster.fit(timeseries)

        timeseries = timeseries.resample(600)
        forecaster.fit(timeseries)

        # TODO: do some actual testing.. not only that "it works"
        forecasted_timeseries  = forecaster.apply(timeseries)


    def test_AARIMAForecaster(self):

        try:
            import statsmodels
        except ImportError:
            print('Skipping AARIMA tests as no statsmodels module installed')
            return

        # Automatic ARIMA
        forecaster = AARIMAForecaster()

        forecaster.fit(self.sine_day_timeseries, max_p=2, max_d=1, max_q=2)
        self.assertEqual(len(self.sine_day_timeseries), 1000)

        # Cannot apply on a time series contiguous with the time series used for the fit
        with self.assertRaises(NonContiguityError):
            forecaster.apply(self.sine_day_timeseries[:-1], steps=3)

        # Can apply on a time series contiguous with the same item as the fit one
        sine_day_timeseries_with_forecast = forecaster.apply(self.sine_day_timeseries, steps=3)
        self.assertEqual(len(sine_day_timeseries_with_forecast), 1003)

        # # Cannot evaluate on a time series not contiguous with the time series used for the fit
        # with self.assertRaises(NonContiguityError):
        #     forecaster.evaluate(self.sine_day_timeseries)
        #
        # # Can evaluate on a time series contiguous with the time series used for the fit
        # forecaster = AARIMAForecaster()
        # forecaster.fit(self.sine_day_timeseries[0:800], max_p=2, max_d=1, max_q=2)
        # evaluation_results = forecaster.evaluate(self.sine_day_timeseries[800:1000])
        # self.assertTrue('RMSE' in evaluation_results)
        # self.assertTrue('MAE' in evaluation_results)
        # # Cannot test values, some random behavior which cannot be put under control is present somewhere
        # #self.assertAlmostEqual(evaluation_results['RMSE'], 0.7179428895746799)
        # #self.assertAlmostEqual(evaluation_results['MAE'], 0.6497934134525981)

        # Test on Points as well
        timeseries = CSVFileStorage(TEST_DATA_PATH + '/csv/temperature.csv').get(limit=200)
        forecaster = AARIMAForecaster()
        with self.assertRaises(ValueError):
            forecaster.fit(timeseries)

        timeseries = timeseries.resample(600)
        forecaster.fit(timeseries, max_p=2, max_d=1, max_q=2)

        # TODO: do some actual testing.. not only that "it works"
        forecasted_timeseries  = forecaster.apply(timeseries)


    def test_LSTMForecaster_univariate(self):

        try:
            import tensorflow
        except ImportError:
            print('Skipping LSTM forecaster tests with univariate time series as no tensorflow module installed')
            return

        timeseries = TimeSeries()
        for i in range(980):
            timeseries.append(DataTimePoint(t=i*60, data={'sin': sin(i/10.0)}))
        forecaster = LSTMForecaster()
        forecaster.fit(timeseries, epochs=50, reproducible=True)

        # Test the evaluation
        evaluation_results = forecaster.evaluate(timeseries[0:10])
        self.assertAlmostEqual(evaluation_results['sin_RMSE'], 0.026, places=2)
        self.assertAlmostEqual(evaluation_results['sin_MAE'], 0.025, places=2)

        # Test the predict
        self.assertAlmostEqual(forecaster.predict(timeseries)['sin'], -0.53, places=2)

        # Test apply results
        timeseries_with_forecast = forecaster.apply(timeseries[0:970], steps=10)
        # for i in range(10):
        #     print('{:.2f} vs {:.2f} ({})'.format(timeseries_with_forecast[970+i].data['sin'], timeseries[970+i].data['sin'], timeseries_with_forecast[970+i].data_indexes))
        # 0.37 vs 0.38 ({'forecast': 1})
        # 0.26 vs 0.29 ({'forecast': 1})
        # 0.15 vs 0.19 ({'forecast': 1})
        # 0.04 vs 0.09 ({'forecast': 1})
        # -0.07 vs -0.01 ({'forecast': 1})
        # -0.17 vs -0.11 ({'forecast': 1})
        # -0.25 vs -0.21 ({'forecast': 1})
        # -0.32 vs -0.31 ({'forecast': 1})
        # -0.37 vs -0.40 ({'forecast': 1})
        # -0.40 vs -0.49 ({'forecast': 1})
        self.assertAlmostEqual(timeseries_with_forecast[970].data['sin'], 0.37, places=2)
        self.assertAlmostEqual(timeseries_with_forecast[973].data['sin'], 0.04, places=2)
        self.assertAlmostEqual(timeseries_with_forecast[974].data['sin'], -0.07, places=2)
        self.assertAlmostEqual(timeseries_with_forecast[979].data['sin'], -0.40, places=2)

        # Test using other parameters and features
        forecaster = LSTMForecaster(neurons=16, features=['values','diffs'])
        forecaster.fit(timeseries, reproducible=True)
        self.assertAlmostEqual(forecaster.predict(timeseries)['sin'], -0.58, places=2)

        # Test for non-existent features
        with self.assertRaises(ValueError):
            LSTMForecaster(features=['values','not_existent_feature']).fit(timeseries)


    def test_LSTMForecaster_multivariate(self):
        try:
            import tensorflow
        except ImportError:
            print('Skipping LSTM forecaster tests with multivariate time series as no tensorflow module installed')
            return

        timeseries = TimeSeries()
        for i in range(980):
            timeseries.append(DataTimePoint(t=i*60, data={'sin': sin(i/10.0), 'cos': cos(i/10.0)}))
        forecaster = LSTMForecaster()
        forecaster.fit(timeseries, epochs=10, reproducible=True)

        # Test the evaluation (TODO: this is not yet implemented!)
        #evaluation_results = forecaster.evaluate(timeseries[0:10])
        #self.assertAlmostEqual(evaluation_results['RMSE'], 0.0489, places=3)
        #self.assertAlmostEqual(evaluation_results['MAE'], 0.041, places=2)

        # Test the predict
        #self.assertAlmostEqual(forecaster.predict(timeseries)['sin'], -0.53, places=2)

        # Test apply results
        timeseries_with_forecast = forecaster.apply(timeseries[0:970], steps=5)
        #for i in range(10):
        #    print('{:.2f},{:.2f} vs {:.2f},{:.2f} ({})'.format(timeseries_with_forecast[970+i].data['sin'],
        #                                                       timeseries_with_forecast[970+i].data['cos'],
        #                                                       timeseries[970+i].data['sin'],
        #                                                       timeseries[970+i].data['cos'],
        #                                                       timeseries_with_forecast[970+i].data_indexes))
        # 0.36,-0.92 vs 0.38,-0.93 ({'forecast': 1})
        # 0.26,-0.95 vs 0.29,-0.96 ({'forecast': 1})
        # 0.15,-0.97 vs 0.19,-0.98 ({'forecast': 1})
        # 0.05,-0.97 vs 0.09,-1.00 ({'forecast': 1})
        # -0.05,-0.96 vs -0.01,-1.00 ({'forecast': 1})
        self.assertAlmostEqual(timeseries_with_forecast[970].data['sin'], 0.36, places=2)
        self.assertAlmostEqual(timeseries_with_forecast[970].data['cos'], -0.92, places=2)
        self.assertAlmostEqual(timeseries_with_forecast[973].data['sin'], 0.05, places=2)
        self.assertAlmostEqual(timeseries_with_forecast[973].data['cos'], -0.97, places=2)
        self.assertAlmostEqual(timeseries_with_forecast[974].data['sin'], -0.05, places=2)
        self.assertAlmostEqual(timeseries_with_forecast[974].data['cos'], -0.96, places=2)

        # Test using other parameters and features
        forecaster = LSTMForecaster(neurons=16, features=['values','diffs'])
        forecaster.fit(timeseries, reproducible=True)
        self.assertAlmostEqual(forecaster.predict(timeseries)['sin'], -0.541, places=2)
        self.assertAlmostEqual(forecaster.predict(timeseries)['cos'], -0.80, places=2)

        # Test for non-existent features
        with self.assertRaises(ValueError):
            LSTMForecaster(features=['values','not_existent_feature']).fit(timeseries)


    def test_LSTMForecaster_multivariate_with_targets_and_context(self):

        try:
            import tensorflow
        except ImportError:
            print('Skipping LSTM forecaster tests with multivariate time series with targets as no tensorflow module installed')
            return

        timeseries = TimeSeries()
        for i in range(980):
            timeseries.append(DataTimePoint(t=i*60, data={'sin': sin(i/10.0), 'cos': cos(i/10.0)}))

        # Target, no context
        forecaster = LSTMForecaster()
        forecaster.fit(timeseries, target='cos', with_context=False, reproducible=True)

        with self.assertRaises(ValueError):
            forecaster.predict(timeseries, context_data={'sin': -0.57})

        predicted_data = forecaster.predict(timeseries)
        self.assertEqual(list(predicted_data.keys()), ['cos'])
        self.assertAlmostEqual(predicted_data['cos'], -0.82, places=2)

        # All other methods will raise
        with self.assertRaises(ValueError):
            forecaster.forecast(timeseries)

        with self.assertRaises(ValueError):
            forecaster.apply(timeseries)

        # Target and context
        forecaster = LSTMForecaster()
        forecaster.fit(timeseries, target='cos', with_context=True, reproducible=True)

        predicted_data = forecaster.predict(timeseries, context_data={'sin': -0.57})
        self.assertAlmostEqual(predicted_data['cos'], -0.820, places=2)

        forecasted_data = forecaster.forecast(timeseries, context_data={'sin': -0.57})
        self.assertEqual(forecasted_data.data['sin'], -0.57)
        self.assertAlmostEqual(forecasted_data.data['cos'], -0.820, places=2)

        timeseries_with_forecast = forecaster.apply(timeseries, context_data={'sin': -0.57})
        self.assertEqual(timeseries_with_forecast[-1].data['sin'], -0.57)
        self.assertAlmostEqual(timeseries_with_forecast[-1].data['cos'], -0.820, places=2)


    def test_LSTMForecaster_multivariate_with_targets_and_context_humitemp(self):

        try:
            import tensorflow
        except ImportError:
            print('Skipping LSTM forecaster tests with multivariate time series on humitemp with targets as no tensorflow module installed')
            return

        # This test is here mainly te ensure there are no issues with the normalization of the context data
        timeseries = TimeSeries.from_csv(TEST_DATASETS_PATH + 'humitemp_long.csv')
        timeseries_full = timeseries.resample('1h')
        timeseries = timeseries_full[300:400] 

        forecaster = LSTMForecaster()
        forecaster.fit(timeseries, target='humidity', with_context=True, reproducible=True)

        prediction = forecaster.predict(timeseries[:-1], context_data = timeseries[-1].data)

        self.assertAlmostEqual(prediction['humidity'], 41.6734, places=2)


    def test_LSTMForecaster_cross_validation(self):

        try:
            import tensorflow
        except ImportError:
            print('Skipping LSTM forecaster cross validation tests as no tensorflow module installed')
            return

        temperature_timeseries = TimeSeries.from_csv(TEST_DATASETS_PATH + 'temperature_winter.csv').resample('1h')
        # Pretend there was no data loss at all
        for item in temperature_timeseries:
            item.data_indexes['data_loss'] = 0 
        forecaster = LSTMForecaster(window=12, neurons=64, features=['values', 'diffs', 'hours'])
        cross_validation_results = forecaster.cross_validate(temperature_timeseries[0:100], rounds=3, fit_reproducible=True)
        self.assertAlmostEqual(cross_validation_results['temperature_MAE_avg'], 0.2469, places=2)
        self.assertAlmostEqual(cross_validation_results['temperature_MAE_stdev'], 0.1316, places=2)


    def test_LSTMForecaster_save_load(self):

        try:
            import tensorflow
        except ImportError:
            print('Skipping LSTM save/load forecaster tests as no tensorflow module installed')
            return

        # Create a minute-resolution test DataTimeSlotSeries
        sine_minute_timeseries = TimeSeries()
        for i in range(10):
            sine_minute_timeseries.append(DataTimeSlot(start=TimePoint(i*60), end=TimePoint((i+1)*60), data={'sin':sin(i/10.0), 'cos':cos(i/10.0)}))

        forecaster = LSTMForecaster()
        forecaster.fit(sine_minute_timeseries, reproducible=True)

        # Set model path
        model_path = TEMP_MODELS_DIR+ '/test_LSTM_model'

        # Save
        forecaster.save(model_path)

        # Load
        loaded_forecaster = LSTMForecaster.load(model_path)

        # Predict from the loaded model
        predicted_data = loaded_forecaster.predict(sine_minute_timeseries)

        self.assertTrue('sin' in predicted_data)
        self.assertTrue('cos' in predicted_data)


