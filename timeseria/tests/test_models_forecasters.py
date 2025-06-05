import unittest
import os
import tempfile
from math import sin, cos
from propertime.utils import dt

from ..datastructures import TimePoint, DataTimeSlot, DataTimePoint, TimeSeries
from ..models.forecasters import ProphetForecaster, PeriodicAverageForecaster, ARIMAForecaster, AARIMAForecaster, LSTMForecaster
from ..storages import CSVFileStorage
from ..utils import ensure_reproducibility, PFloat
from .. import TEST_DATASETS_PATH

# Setup logging
from .. import logger
logger.setup()

# Get a temp dir for models
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

        # Ensure reproducibility
        ensure_reproducibility()


    @staticmethod
    def assertCompatible(actual, predicted, delta=0.3, item=None):
        if not abs(actual-predicted) < delta:
            raise AssertionError('Actual and predicted values are not compatible within the specified delta (actual="{}", predicted="{}", delta="{}", item="{}"'.format(actual, predicted, delta, item))


    def test_PeriodicAverageForecaster(self):

        # Note: here we test also some basic functionality of the forecaster as well,
        # given that the PeridicAVergaeForecaster is very simple and close to a mock.

        # Instantiate and fit
        forecaster = PeriodicAverageForecaster()
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
        self.assertAlmostEqual(evaluation['value_RMSE'], 0.0873, places=2)
        self.assertAlmostEqual(evaluation['value_MAE'], 0.0797, places=2)

        evaluation_series = forecaster.evaluate(self.sine_minute_timeseries[0:100], return_evaluation_series=True)['series']
        self.assertEqual(len(evaluation_series), 37)
        self.assertEqual(set(evaluation_series.data_labels()), {'value', 'value_AE', 'value_pred'})

        evaluation = forecaster.evaluate(evaluation_series, steps=1)
        self.assertEqual(forecaster.data['periodicities']['value'], 63)
        self.assertAlmostEqual(evaluation['value_RMSE'], 0.0873, places=2)
        self.assertAlmostEqual(evaluation['value_MAE'], 0.0797, places=2)

        # Test on a realistic point series, forecast horizon=1
        timeseries = CSVFileStorage(os.path.join(TEST_DATASETS_PATH, 'temperature.csv')).get(limit=200)
        forecaster = PeriodicAverageForecaster()
        with self.assertRaises(ValueError):
            forecaster.fit(timeseries)

        timeseries = timeseries.resample(600)
        forecaster.fit(timeseries)

        prediction = forecaster.predict(timeseries)
        self.assertEqual(len(prediction), 1)
        self.assertEqual(list(prediction.keys()), ['temperature'])
        self.assertAlmostEqual(prediction['temperature'], 22.891000000000005)

        forecast = forecaster.forecast(timeseries)
        self.assertEqual(forecast.t, timeseries[-1].t+600)
        self.assertEqual(list(forecast.data.keys()), ['temperature'])
        self.assertAlmostEqual(forecast.data['temperature'], 22.891000000000005)

        timeseries_with_forecast  = forecaster.apply(timeseries)
        self.assertEqual(len(timeseries_with_forecast), len(timeseries)+1)
        self.assertEqual(timeseries_with_forecast[-1].t, timeseries[-1].t+600)
        self.assertEqual(list(timeseries_with_forecast[-1].data.keys()), ['temperature'])
        self.assertAlmostEqual(timeseries_with_forecast[-1].data['temperature'], 22.891000000000005)

        # Test on a realistic point series, forecast horizon=3
        prediction = forecaster.predict(timeseries, steps=3)
        self.assertEqual(len(prediction), 3)
        self.assertEqual(list(prediction[0].keys()), ['temperature'])
        self.assertAlmostEqual(prediction[0]['temperature'], 22.891000000000005)
        self.assertEqual(list(prediction[-1].keys()), ['temperature'])
        self.assertAlmostEqual(prediction[-1]['temperature'], 22.922057142857145)

        forecast = forecaster.forecast(timeseries, steps=3)
        self.assertEqual(len(forecast), 3)
        self.assertEqual(forecast[0].t, timeseries[-1].t+600)
        self.assertEqual(list(forecast[0].data.keys()), ['temperature'])
        self.assertAlmostEqual(forecast[0].data['temperature'], 22.891000000000005)
        self.assertEqual(forecast[-1].t, timeseries[-1].t+600*3)
        self.assertEqual(list(forecast[-1].data.keys()), ['temperature'])
        self.assertAlmostEqual(forecast[-1].data['temperature'], 22.922057142857145)

        timeseries_with_forecast  = forecaster.apply(timeseries, steps=3)
        self.assertEqual(len(timeseries_with_forecast), len(timeseries)+3)
        self.assertEqual(timeseries_with_forecast[-3].t, timeseries[-1].t+600)
        self.assertEqual(list(timeseries_with_forecast[-3].data.keys()), ['temperature'])
        self.assertAlmostEqual(timeseries_with_forecast[-3].data['temperature'], 22.891)
        self.assertEqual(timeseries_with_forecast[-1].t, timeseries[-1].t+600*3)
        self.assertEqual(list(timeseries_with_forecast[-1].data.keys()), ['temperature'])
        self.assertAlmostEqual(timeseries_with_forecast[-1].data['temperature'], 22.922057142857145)


    def test_PeriodicAverageForecaster_multivariate(self):

        # Note: here we test also some basic functionality of the forecaster as well,
        # given that the PeridicAVergaeForecaster is very simple and close to a mock.

        timeseries = TimeSeries()
        for i in range(970):
            timeseries.append(DataTimePoint(t=i*60, data={'sin': sin(i/10.0), 'cos': cos(i/10.0)}))

        # Instantiate and fit
        forecaster = PeriodicAverageForecaster()
        forecaster.fit(timeseries)

        prediction = forecaster.predict(timeseries, steps=3)
        self.assertEqual(len(prediction), 3)
        self.assertEqual(list(prediction[0].keys()), ['cos', 'sin'])
        self.assertAlmostEqual(prediction[0]['cos'], 0.19600497357083713)
        self.assertAlmostEqual(prediction[0]['sin'], 0.660933759485128)
        self.assertEqual(list(prediction[-1].keys()), ['cos', 'sin'])
        self.assertAlmostEqual(prediction[-1]['cos'], 0.0642538467366801)
        self.assertAlmostEqual(prediction[-1]['sin'], 0.6923761210530861)

        forecast = forecaster.forecast(timeseries, steps=3)
        self.assertEqual(len(forecast), 3)
        self.assertEqual(forecast[0].t, timeseries[-1].t+60)
        self.assertEqual(list(forecast[0].data.keys()), ['cos', 'sin'])
        self.assertAlmostEqual(forecast[0].data['cos'], 0.19600497357083713)
        self.assertAlmostEqual(forecast[0].data['sin'], 0.660933759485128)
        self.assertEqual(forecast[-1].t, timeseries[-1].t+60*3)
        self.assertEqual(list(forecast[-1].data.keys()), ['cos', 'sin'])
        self.assertAlmostEqual(forecast[-1].data['cos'], 0.0642538467366801)
        self.assertAlmostEqual(forecast[-1].data['sin'], 0.6923761210530861)

        timeseries_with_forecast  = forecaster.apply(timeseries, steps=3)
        self.assertEqual(len(timeseries_with_forecast), len(timeseries)+3)
        self.assertEqual(timeseries_with_forecast[-3].t, timeseries[-1].t+60)
        self.assertEqual(list(timeseries_with_forecast[-3].data.keys()), ['cos', 'sin'])
        self.assertAlmostEqual(timeseries_with_forecast[-3].data['cos'], 0.19600497357083713)
        self.assertAlmostEqual(timeseries_with_forecast[-3].data['sin'], 0.660933759485128)
        self.assertEqual(timeseries_with_forecast[-1].t, timeseries[-1].t+60*3)
        self.assertEqual(list(timeseries_with_forecast[-1].data.keys()), ['cos', 'sin'])
        self.assertAlmostEqual(timeseries_with_forecast[-1].data['cos'], 0.0642538467366801)
        self.assertAlmostEqual(timeseries_with_forecast[-1].data['sin'], 0.6923761210530861)


    def test_PeriodicAverageForecaster_probabilistic(self):

        # Instantiate and fit
        forecaster = PeriodicAverageForecaster()
        forecaster.fit(self.sine_minute_timeseries, periodicity=63, probabilistic=True)

        # Check internal data
        self.assertIsInstance(forecaster.data['offsets_averages']['value'][0], PFloat)

        # Predict
        predicted = forecaster.predict(self.sine_minute_timeseries, steps=1)
        self.assertIsInstance(predicted['value'], PFloat)

        # Evaluate with confidence metrics
        evaluation = forecaster.evaluate(self.sine_minute_timeseries[0:73], confidence_interval=[0.0001,0.9999])
        self.assertAlmostEqual(evaluation['value_EC'], 0.1)

        evaluation = forecaster.evaluate(self.sine_minute_timeseries[0:73],
                                         confidence_metrics=['EC', 'ECE', 'ECPE'],
                                         confidence_interval=[0.0001,0.9999])
        self.assertAlmostEqual(evaluation['value_EC'], 0.1)
        self.assertAlmostEqual(evaluation['value_ECE'], 0.8998)
        self.assertAlmostEqual(evaluation['value_ECPE'], 0.8999799959991999)

    def test_PeriodicAverageForecaster_save_load(self):

        forecaster = PeriodicAverageForecaster()
        forecaster.fit(self.sine_minute_timeseries, periodicity=63)
        forecaster.save(os.path.join(TEMP_MODELS_DIR, 'test_PA_model'))

        loaded_forecaster = PeriodicAverageForecaster.load(os.path.join(TEMP_MODELS_DIR, 'test_PA_model'))
        self.assertEqual(forecaster.data['offsets_averages'], loaded_forecaster.data['offsets_averages'])


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

        # Test with the evaluate
        evalation_results = forecaster.evaluate(self.sine_day_timeseries.view(0,10))
        self.assertAlmostEqual(evalation_results['value_RMSE'], 0.80, places=2)
        self.assertAlmostEqual(evalation_results['value_MAE'], 0.77, places=2)

        evalation_results = forecaster.evaluate(self.sine_day_timeseries.view(0,1))
        self.assertAlmostEqual(evalation_results['value_RMSE'], 0.53, places=2) # For one sample they must be the same
        self.assertAlmostEqual(evalation_results['value_MAE'], 0.53, places=2) # For one sample they must be the same

        # Test on a realistic point series, forecast horizon=3
        timeseries = CSVFileStorage(os.path.join(TEST_DATASETS_PATH, 'temperature.csv')).get(limit=200)
        forecaster = ProphetForecaster()
        with self.assertRaises(Exception):
            forecaster.fit(timeseries)

        timeseries = timeseries.resample(600)
        forecaster.fit(timeseries)

        prediction = forecaster.predict(timeseries, steps=3)
        self.assertEqual(len(prediction), 3)
        self.assertEqual(list(prediction[0].keys()), ['temperature'])
        self.assertAlmostEqual(prediction[0]['temperature'], 23.297123620317347, places=2)
        self.assertEqual(list(prediction[-1].keys()), ['temperature'])
        self.assertAlmostEqual(prediction[-1]['temperature'], 23.336020073857974, places=2)


    def test_ARIMAForecaster(self):

        try:
            import statsmodels
        except ImportError:
            print('Skipping ARIMA tests as no statsmodels module installed')
            return

        import warnings
        from statsmodels.tools.sm_exceptions import ConvergenceWarning
        warnings.simplefilter('ignore', ConvergenceWarning)

        # Basic ARIMA
        forecaster = ARIMAForecaster(p=1,d=1,q=0)
        forecaster.fit(self.sine_day_timeseries)

        # Cannot apply on a time series ending with a different timestamp as the time series used for the fit
        with self.assertRaises(Exception):
            forecaster.apply(self.sine_day_timeseries[:-1], steps=3)

        sine_day_timeseries_with_forecast = forecaster.apply(self.sine_day_timeseries, steps=3)
        self.assertEqual(len(sine_day_timeseries_with_forecast), 1003)

        # Test on a realistic point series, forecast horizon=3
        timeseries = CSVFileStorage(os.path.join(TEST_DATASETS_PATH, 'temperature.csv')).get(limit=200)
        forecaster = ARIMAForecaster()
        with self.assertRaises(Exception):
            forecaster.fit(timeseries)

        timeseries = timeseries.resample(600)
        forecaster.fit(timeseries)

        prediction = forecaster.predict(timeseries, steps=3)
        self.assertEqual(len(prediction), 3)
        self.assertEqual(list(prediction[0].keys()), ['temperature'])
        self.assertAlmostEqual(prediction[0]['temperature'], 23.227174207689878)
        self.assertEqual(list(prediction[-1].keys()), ['temperature'])
        self.assertAlmostEqual(prediction[-1]['temperature'], 23.23067079831882)


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

        # Cannot apply on a time series ending with a different timestamp as the time series used for the fit
        with self.assertRaises(Exception ):
            forecaster.apply(self.sine_day_timeseries[:-1], steps=3)

        sine_day_timeseries_with_forecast = forecaster.apply(self.sine_day_timeseries, steps=3)
        self.assertEqual(len(sine_day_timeseries_with_forecast), 1003)

        # Test on a realistic point series, forecast horizon=3
        timeseries = CSVFileStorage(os.path.join(TEST_DATASETS_PATH, 'temperature.csv')).get(limit=200)
        forecaster = AARIMAForecaster()
        with self.assertRaises(Exception):
            forecaster.fit(timeseries)

        timeseries = timeseries.resample(600)
        forecaster.fit(timeseries)

        prediction = forecaster.predict(timeseries, steps=3)
        self.assertEqual(len(prediction), 3)
        self.assertEqual(list(prediction[0].keys()), ['temperature'])
        self.assertAlmostEqual(prediction[0]['temperature'], 23.227174207689878)
        self.assertEqual(list(prediction[-1].keys()), ['temperature'])
        self.assertAlmostEqual(prediction[-1]['temperature'], 23.23067079831882)


    def test_LSTMForecaster_univariate(self):

        try:
            import tensorflow
        except ImportError:
            print('Skipping LSTM forecaster tests with univariate time series as no tensorflow module installed')
            return

        timeseries = TimeSeries()
        for i in range(980):
            timeseries.append(DataTimePoint(t=i*60, data={'sin': sin(i/10.0)}))

        # Basic test
        forecaster = LSTMForecaster(window=10)
        forecaster.fit(timeseries[0:970], epochs=10, reproducible=True, verbose=False)
        timeseries_forecast = forecaster.apply(timeseries[0:970], steps=10)

        evaluation_results = forecaster.evaluate(timeseries[0:100], error_metrics=['MAE'])
        self.assertTrue(evaluation_results['sin_MAE'] <0.2, evaluation_results['sin_MAE'])

        for i in range(10):
            actual = timeseries[970+i].data['sin']
            predicted = timeseries_forecast[970+i].data['sin']
            self.assertCompatible(actual, predicted)

        # Test using other parameters and features
        forecaster = LSTMForecaster(window=10, neurons=64, features=['values','diffs'])
        forecaster.fit(timeseries[0:970], epochs=10, reproducible=True, verbose=False)
        timeseries_forecast = forecaster.apply(timeseries[0:970], steps=10)

        evaluation_results = forecaster.evaluate(timeseries[0:100], error_metrics=['MAE'])
        self.assertTrue(evaluation_results['sin_MAE'] <0.2, evaluation_results['sin_MAE'])

        for i in range(10):
            actual = timeseries[970+i].data['sin']
            predicted = timeseries_forecast[970+i].data['sin']
            self.assertCompatible(actual, predicted, item=i)

        # Test for non-existent features
        with self.assertRaises(ValueError):
            LSTMForecaster(features=['values','not_existent_feature']).fit(timeseries)


    def test_LSTMForecaster_univariate_with_window_mask(self):

        try:
            import tensorflow
        except ImportError:
            print('Skipping LSTM forecaster tests with univariate time series as no tensorflow module installed')
            return

        timeseries = TimeSeries()
        for i in range(980):
            timeseries.append(DataTimePoint(t=i*60, data={'sin': sin(i/10.0)}))

        # Non-ngarive mask not supported
        forecaster = LSTMForecaster(window=10, window_mask=3)
        with self.assertRaises(ValueError):
            forecaster.fit(timeseries[0:970], epochs=10, reproducible=True, verbose=False)

        # Negative mask indicates how many elements to dismiss at the end of the window
        forecaster = LSTMForecaster(window=10, window_mask=-3)
        forecaster.fit(timeseries[0:970], epochs=10, reproducible=True, verbose=False)

        evaluation_results = forecaster.evaluate(timeseries[0:100], error_metrics=['MAE'])
        self.assertAlmostEqual(evaluation_results['sin_MAE'], 0.049999925551957244, places=3)


    def test_LSTMForecaster_univariate_probabilistic(self):

        try:
            import tensorflow
        except ImportError:
            print('Skipping LSTM probabilistic forecaster tests with univariate time series as no tensorflow module installed')
            return

        timeseries = TimeSeries()
        for i in range(980):
            timeseries.append(DataTimePoint(t=i*60, data={'sin': sin(i/10.0)}))

        forecaster = LSTMForecaster(window=10)
        forecaster.fit(timeseries[0:970], epochs=3, reproducible=True, probabilistic=True, verbose=False)

        predicted = forecaster.predict(timeseries[0:970], samples=3)
        self.assertAlmostEqual(predicted['sin'], 0.593242318431153, places=3)
        self.assertIsInstance(predicted['sin'], PFloat)


    def test_LSTMForecaster_multivariate(self):
        try:
            import tensorflow
        except ImportError:
            print('Skipping LSTM forecaster tests with multivariate time series as no tensorflow module installed')
            return

        timeseries = TimeSeries()
        for i in range(980):
            timeseries.append(DataTimePoint(t=i*60, data={'sin': sin(i/10.0), 'cos': cos(i/10.0)}))

        # Basic test
        forecaster = LSTMForecaster()
        forecaster.fit(timeseries, epochs=10, reproducible=True)
        evaluation_results = forecaster.evaluate(timeseries[0:100], error_metrics=['MAE'], return_evaluation_series=True)

        self.assertTrue(evaluation_results['sin_MAE'] < 0.05, evaluation_results['sin_MAE'])
        self.assertTrue(evaluation_results['sin_MAE'] < 0.05, evaluation_results['sin_MAE'])

        timeseries_forecast = forecaster.apply(timeseries[0:970], steps=10)

        for i in range(10):
            actual = timeseries[970+i].data['sin']
            predicted = timeseries_forecast[970+i].data['sin']
            self.assertCompatible(actual, predicted)

        for i in range(10):
            actual = timeseries[970+i].data['cos']
            predicted = timeseries_forecast[970+i].data['cos']
            self.assertCompatible(actual, predicted)


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
        forecaster.fit(timeseries[0:978], epochs=10, target='cos', with_context=False, reproducible=True)

        with self.assertRaises(ValueError):
            forecaster.predict(timeseries[0:978], context_data={'sin': 'whatever'})

        predicted_data = forecaster.predict(timeseries)
        self.assertEqual(list(predicted_data.keys()), ['cos'])
        self.assertCompatible(timeseries[979].data['cos'], predicted_data['cos'])

        # Target and context
        forecaster = LSTMForecaster()
        forecaster.fit(timeseries[0:978], target='cos', with_context=True, reproducible=True)

        with self.assertRaises(ValueError):
            forecaster.predict(timeseries[0:978])

        predicted_data = forecaster.predict(timeseries[0:978], context_data={'sin': timeseries[979].data['cos']})
        self.assertEqual(list(predicted_data.keys()), ['cos'])
        self.assertCompatible(timeseries[979].data['cos'], predicted_data['cos'])

        # All other methods will raise as not suitable
        with self.assertRaises(ValueError):
            forecaster.forecast(timeseries)

        with self.assertRaises(ValueError):
            forecaster.apply(timeseries)


    def test_LSTMForecaster_cross_validation(self):

        try:
            import tensorflow
        except ImportError:
            print('Skipping LSTM forecaster cross validation tests as no tensorflow module installed')
            return

        temperature_timeseries = TimeSeries.from_csv(os.path.join(TEST_DATASETS_PATH, 'temperature_winter.csv')).resample('1h')

        # Pretend there was no data loss at all
        for item in temperature_timeseries:
            item.data_indexes['data_loss'] = 0 

        # Initialize and cross validate the forecaster
        forecaster = LSTMForecaster(window=12, neurons=64, features=['values', 'diffs', 'hours'])
        cross_validation_results = forecaster.cross_validate(temperature_timeseries[0:100], rounds=3, fit_reproducible=True,
                                                             evaluate_return_evaluation_series=True, return_full_evaluations=True)

        self.assertTrue(0.3 < cross_validation_results['temperature_RMSE_avg'] < 0.6) # Expected: 0.45243811980703913
        self.assertTrue(0.2 < cross_validation_results['temperature_MAE_avg'] < 0.5) # Expected: 0.36419170510171595
        self.assertEqual(len(cross_validation_results['evaluations']),3)
        for evaluation in cross_validation_results['evaluations']:
            self.assertTrue('temperature_RMSE' in evaluation)
            self.assertTrue('temperature_MAE' in evaluation)
            self.assertTrue('series' in evaluation)


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

        # Fit a forecaster and make a prediction
        forecaster = LSTMForecaster()
        forecaster.fit(sine_minute_timeseries, reproducible=True)
        predicted_data = forecaster.predict(sine_minute_timeseries)

        # Set model path
        model_path = os.path.join(TEMP_MODELS_DIR, 'test_LSTM_model')

        # Save
        forecaster.save(model_path)

        # Load
        forecaster_loaded = LSTMForecaster.load(model_path)

        # Predict from the loaded model
        predicted_data_loaded = forecaster_loaded.predict(sine_minute_timeseries)

        # Check predictions are the same
        self.assertEqual(predicted_data, predicted_data_loaded)


    def test_LinearRegressionForecaster(self):

        # Create a hour-resolution test DataTimeSlotSeries
        sine_hour_timeseries = TimeSeries()
        for i in range(1000):
            sine_hour_timeseries.append(DataTimePoint(t=i*3600, data={'value':sin(i/10.0)}))

        from ..models import LinearRegressionForecaster
        forecaster = LinearRegressionForecaster(features=['values','hours'])

        # Fit
        forecaster.fit(sine_hour_timeseries)

        # Predict and check
        self.assertAlmostEqual(forecaster.apply(sine_hour_timeseries)[-1].data['value'], -0.5063, places=3)

        # Save
        forecaster.save(TEMP_MODELS_DIR+'/test_lr_model')

        # Load
        loaded_forecaster = LinearRegressionForecaster.load(TEMP_MODELS_DIR+'/test_lr_model')

        # Re-predict and re-check
        self.assertAlmostEqual(loaded_forecaster.apply(sine_hour_timeseries)[-1].data['value'], -0.5063, places=3)


    def test_PeriodicAverageForecaster_ErrorDistributionCalibrator(self):

        # Note: here we test also some basic functionality of the forecaster as well,
        # given that the PeridicAVergaeForecaster is very simple and close to a mock,
        # and similarly for the ErrorDistributionCalibrator.

        # Instantiate and fit
        forecaster = PeriodicAverageForecaster()
        forecaster.fit(self.sine_minute_timeseries, periodicity=63)

        # Calibrate
        from ..models.calibrators import ErrorDistributionCalibrator
        forecaster.calibrate(self.sine_minute_timeseries, ErrorDistributionCalibrator)

        # Check prediction is PFLoat
        prediction = forecaster.predict(self.sine_minute_timeseries)['value']
        self.assertIsInstance(prediction, PFloat)

        # Check now that the apply generates PFLoats as well
        result_series = forecaster.apply(self.sine_minute_timeseries)
        self.assertIsInstance(result_series[-1].data['value'], PFloat)

        # Check a consecutive call does not raise surprises
        result_series = forecaster.apply(result_series)
        self.assertIsInstance(result_series[-1].data['value'], PFloat)

        # Check now that the apply generates PFLoats as well, for multi-step
        result_series = forecaster.apply(self.sine_minute_timeseries, steps=2)
        self.assertIsInstance(result_series[-1].data['value'], PFloat)
        self.assertIsInstance(result_series[-2].data['value'], PFloat)

        # Test save & load
        forecaster.save(os.path.join(TEMP_MODELS_DIR, 'test_PeriodicAverageForecaster_ErrorDistributionCalibrator'))
        loaded_forecaster = PeriodicAverageForecaster.load(os.path.join(TEMP_MODELS_DIR, 'test_PeriodicAverageForecaster_ErrorDistributionCalibrator'))

        # Check prediction is PFLoat
        prediction = loaded_forecaster.predict(self.sine_minute_timeseries)['value']
        self.assertIsInstance(prediction, PFloat)

        # Test the evaluation
        evaluation_series = forecaster.evaluate(self.sine_minute_timeseries[0:64], return_evaluation_series=True)['series']
        self.assertIsInstance(evaluation_series[0].data['value_pred'], PFloat)


    def test_PeriodicAverageForecaster_CHEMCalibrator(self):

        # Instantiate and fit
        forecaster = PeriodicAverageForecaster()
        forecaster.fit(self.sine_minute_timeseries, periodicity=63)

        # Calibrate
        from ..models.calibrators import CHEMCalibrator
        forecaster.calibrate(self.sine_minute_timeseries,
                             CHEMCalibrator,
                             error_predictor_neurons=8,
                             error_predictor_features=['values'],
                             error_predictor_fit_epochs=2)

        # Check prediction is PFLoat
        prediction = forecaster.predict(self.sine_minute_timeseries)['value']
        self.assertIsInstance(prediction, PFloat)

        # Check now that the apply generates PFLoats as well
        result_series = forecaster.apply(self.sine_minute_timeseries)
        self.assertIsInstance(result_series[-1].data['value'], PFloat)

        # Check a consecutive call does not raise surprises
        result_series = forecaster.apply(result_series)
        self.assertIsInstance(result_series[-1].data['value'], PFloat)

        # Check now that the apply generates PFLoats as well, for multi-step
        result_series = forecaster.apply(self.sine_minute_timeseries, steps=2)
        self.assertIsInstance(result_series[-1].data['value'], PFloat)
        self.assertIsInstance(result_series[-2].data['value'], PFloat)

        # Test save & load
        forecaster.save(os.path.join(TEMP_MODELS_DIR, 'test_PeriodicAverageForecaster_CHEMCalibrator'))
        loaded_forecaster = PeriodicAverageForecaster.load(os.path.join(TEMP_MODELS_DIR, 'test_PeriodicAverageForecaster_CHEMCalibrator'))

        # Check prediction is PFLoat
        prediction = loaded_forecaster.predict(self.sine_minute_timeseries)['value']
        self.assertIsInstance(prediction, PFloat)

        # Test the evaluation
        evaluation_series = forecaster.evaluate(self.sine_minute_timeseries[0:64], return_evaluation_series=True)['series']
        self.assertIsInstance(evaluation_series[0].data['value_pred'], PFloat)



    def test_LSTMForecaster_CHEMCalibrator(self):

        # Instantiate and fit
        forecaster = LSTMForecaster()
        forecaster.fit(self.sine_minute_timeseries, epochs=3)

        # Calibrate
        from ..models.calibrators import CHEMCalibrator
        forecaster.calibrate(self.sine_minute_timeseries,
                             CHEMCalibrator,
                             error_predictor_neurons=8,
                             error_predictor_features=['values'],
                             error_predictor_fit_epochs=2)

        # Check prediction is PFLoat
        prediction = forecaster.predict(self.sine_minute_timeseries)['value']
        self.assertIsInstance(prediction, PFloat)

        # Check now that the apply generates PFLoats as well
        result_series = forecaster.apply(self.sine_minute_timeseries)
        self.assertIsInstance(result_series[-1].data['value'], PFloat)

        # Check a consecutive call does not raise surprises
        result_series = forecaster.apply(result_series)
        self.assertIsInstance(result_series[-1].data['value'], PFloat)

        # Check now that the apply generates PFLoats as well, for multi-step
        result_series = forecaster.apply(self.sine_minute_timeseries, steps=2)
        self.assertIsInstance(result_series[-1].data['value'], PFloat)
        self.assertIsInstance(result_series[-2].data['value'], PFloat)

        # Test the evaluation
        evaluation_series = forecaster.evaluate(self.sine_minute_timeseries[:4], return_evaluation_series=True)['series']
        self.assertIsInstance(evaluation_series[0].data['value_pred'], PFloat)

