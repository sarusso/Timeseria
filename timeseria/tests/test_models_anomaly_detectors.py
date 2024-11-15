import unittest
import os
import tempfile
import random
from math import sin, cos

from ..datastructures import TimePoint, DataTimeSlot, DataTimePoint, TimeSeries
from ..models.forecasters import PeriodicAverageForecaster
from ..models.reconstructors import PeriodicAverageReconstructor
from ..models.anomaly_detectors import AnomalyDetector, ModelBasedAnomalyDetector, PeriodicAverageReconstructorAnomalyDetector, PeriodicAverageAnomalyDetector, LSTMAnomalyDetector
from ..storages import CSVFileStorage
from ..utils import ensure_reproducibility

# Setup logging
from .. import logger
logger.setup()

# Test data and temp path
TEST_DATA_PATH = '/'.join(os.path.realpath(__file__).split('/')[0:-1]) + '/test_data/'
TEMP_MODELS_DIR = tempfile.TemporaryDirectory().name

class TestAnomalyDetectors(unittest.TestCase):

    def setUp(self):

        # Create a minute-resolution test time series (1000 elements)
        self.sine_minute_timeseries = TimeSeries()
        for i in range(1000):
            if i % 100 == 0:
                value = 2
            else:
                value = sin(i/10.0)

            self.sine_minute_timeseries.append(DataTimeSlot(start=TimePoint(i*60), end=TimePoint((i+1)*60), data={'sin':value}))

        # Create a minute-resolution multivariate test time series (100 elements)
        self.sine_cosine_minute_timeseries = TimeSeries()
        for i in range(100):
            if i % 100 == 0:
                sin_value = 2
            else:
                sin_value = sin(i/10.0)
            cos_value = cos(i/10.0)

            self.sine_cosine_minute_timeseries.append(DataTimePoint(t=i*60, data={'sin':sin_value, 'cos':cos_value}))

        # Ensure reproducibility
        ensure_reproducibility()


    def test_AnomalyDetector(self):

        # Test mark_events
        test_timeseries = TimeSeries()
        test_timeseries.append(DataTimePoint(t=0, data={'value':1}, data_indexes={'anomaly': 0.0}))
        test_timeseries.append(DataTimePoint(t=1, data={'value':1}, data_indexes={'anomaly': 0.9}))
        test_timeseries.append(DataTimePoint(t=2, data={'value':1}, data_indexes={'anomaly': 0.0}))
        test_timeseries.append(DataTimePoint(t=3, data={'value':1}, data_indexes={'anomaly': 0.0}))
        test_timeseries.append(DataTimePoint(t=4, data={'value':1}, data_indexes={'anomaly': 0.0}))
        test_timeseries.append(DataTimePoint(t=5, data={'value':1}, data_indexes={'anomaly': 0.9})) # Start simple event
        test_timeseries.append(DataTimePoint(t=6, data={'value':1}, data_indexes={'anomaly': 0.9}))
        test_timeseries.append(DataTimePoint(t=7, data={'value':1}, data_indexes={'anomaly': 0.0}))
        test_timeseries.append(DataTimePoint(t=8, data={'value':1}, data_indexes={'anomaly': 0.0}))
        test_timeseries.append(DataTimePoint(t=9, data={'value':1}, data_indexes={'anomaly': 0.0}))
        test_timeseries.append(DataTimePoint(t=10, data={'value':1}, data_indexes={'anomaly': 0.9})) # Start less simple event
        test_timeseries.append(DataTimePoint(t=11, data={'value':1}, data_indexes={'anomaly': 0.6}))
        test_timeseries.append(DataTimePoint(t=12, data={'value':1}, data_indexes={'anomaly': 0.9}))
        test_timeseries.append(DataTimePoint(t=13, data={'value':1}, data_indexes={'anomaly': 0.1}))
        test_timeseries.append(DataTimePoint(t=14, data={'value':1}, data_indexes={'anomaly': 0.0}))
        test_timeseries.append(DataTimePoint(t=15, data={'value':1}, data_indexes={'anomaly': 0.0}))
        test_timeseries.append(DataTimePoint(t=16, data={'value':1}, data_indexes={'anomaly': 0.0})) 
        test_timeseries.append(DataTimePoint(t=17, data={'value':1}, data_indexes={'anomaly': 0.9})) # Start events to be merged
        test_timeseries.append(DataTimePoint(t=18, data={'value':1}, data_indexes={'anomaly': 0.0}))
        test_timeseries.append(DataTimePoint(t=19, data={'value':1}, data_indexes={'anomaly': 0.9}))
        test_timeseries.append(DataTimePoint(t=20, data={'value':1}, data_indexes={'anomaly': 0.9}))
        test_timeseries.append(DataTimePoint(t=21, data={'value':1}, data_indexes={'anomaly': 0.0}))
        test_timeseries.append(DataTimePoint(t=22, data={'value':1}, data_indexes={'anomaly': 0.6}))
        test_timeseries.append(DataTimePoint(t=23, data={'value':1}, data_indexes={'anomaly': 0.9}))
        test_timeseries.append(DataTimePoint(t=24, data={'value':1}, data_indexes={'anomaly': 0.0}))

        event_timeseries = AnomalyDetector.mark_events(test_timeseries,
                                                       index_treshold = 0.9,
                                                       min_persistence = 3,
                                                       max_gap = 2)

        events = {}
        for item in event_timeseries:
            events[item.t] = item.data_indexes['anomaly_event']

        self.assertEqual(events, {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 1, 11: 1, 12: 1, 13: 0, 14: 0, 15: 0, 16: 0, 17: 1, 18: 1, 19: 1, 20: 1, 21: 1, 22: 1, 23: 1, 24: 0})


    def test_ModelBasedAnomalyDetector(self):

        # Wrong init parameters
        with self.assertRaises(ValueError):
            ModelBasedAnomalyDetector()

        with self.assertRaises(ValueError):
            ModelBasedAnomalyDetector(model_class=PeriodicAverageForecaster, model=PeriodicAverageForecaster())

        # Forecaster-based
        anomaly_detector = ModelBasedAnomalyDetector(model_class=PeriodicAverageForecaster)
        anomaly_detector.fit(self.sine_minute_timeseries, periodicity=63, error_metric='E', error_distribution='norm')
        result_timeseries = anomaly_detector.apply(self.sine_minute_timeseries,  index_bounds=['avg_err','3_sigma'])
        self.assertEqual(len(result_timeseries),936)
        self.assertAlmostEqual(result_timeseries[0].data_indexes['anomaly'], 0.0278, places=2)
        self.assertEqual(result_timeseries[236].data_indexes['anomaly'], 1)

        # Reconstructor-based
        anomaly_detector = ModelBasedAnomalyDetector(model_class=PeriodicAverageReconstructor)
        anomaly_detector.fit(self.sine_minute_timeseries, periodicity=63, error_metric='E', error_distribution='norm')
        result_timeseries = anomaly_detector.apply(self.sine_minute_timeseries,  index_bounds=['avg_err','3_sigma'])
        self.assertEqual(len(result_timeseries),997)
        self.assertEqual(result_timeseries[0].data_indexes['anomaly'], 0)
        self.assertAlmostEqual(result_timeseries[5].data_indexes['anomaly'], 0.00186, places=2)
        self.assertEqual(result_timeseries[298].data_indexes['anomaly'], 1)

        # Pre-fitted model
        forecaster = PeriodicAverageForecaster()
        forecaster.fit(self.sine_minute_timeseries, periodicity=63)
        anomaly_detector = ModelBasedAnomalyDetector(model=forecaster)
        anomaly_detector.fit(self.sine_minute_timeseries, error_metric='E', error_distribution='norm')
        result_timeseries = anomaly_detector.apply(self.sine_minute_timeseries,  index_bounds=['avg_err','3_sigma'])
        self.assertEqual(len(result_timeseries),936)
        self.assertAlmostEqual(result_timeseries[0].data_indexes['anomaly'], 0.0278, places=2)
        self.assertEqual(result_timeseries[236].data_indexes['anomaly'], 1)

        # Save/load
        model_path = TEMP_MODELS_DIR+'/test_generic_model_based_anomaly_model'
        with self.assertRaises(NotImplementedError):
            anomaly_detector.save(model_path)


    def test_PeriodicAverageAnomalyDetector(self):
        # This model is forecaster-based

        anomaly_detector = PeriodicAverageAnomalyDetector()

        anomaly_detector.fit(self.sine_minute_timeseries, periodicity=63, error_metric='E', error_distribution='norm')
        self.assertAlmostEqual(anomaly_detector.data['error_distributions_params']['sin']['loc'], -3.73646e-05, places=6)
        self.assertAlmostEqual(anomaly_detector.data['error_distributions_params']['sin']['scale'], 0.20957, places=4)

        result_timeseries = anomaly_detector.apply(self.sine_minute_timeseries,  index_bounds=['avg_err','3_sigma'])

        # Count how many anomalies were detected
        anomalies_count = 0
        for slot in result_timeseries:
            if slot.data_indexes['anomaly'] > 0.6:
                anomalies_count += 1
        self.assertEqual(anomalies_count, 9)

        # Test on Points as well
        timeseries = CSVFileStorage(TEST_DATA_PATH + '/csv/temperature.csv').get(limit=200)
        anomaly_detector = PeriodicAverageAnomalyDetector()
        with self.assertRaises(ValueError):
            anomaly_detector.fit(timeseries, error_distribution='norm')

        timeseries = timeseries.resample(600)
        anomaly_detector.fit(timeseries, error_metric='E', error_distribution='norm')

        result_timeseries = anomaly_detector.apply(timeseries, index_bounds=['avg_err','3_sigma'])
        for point in result_timeseries:
            if 'anomaly' in  point.data_indexes:
                if point.data_indexes['anomaly'] > 0.3:
                    anomalies_count += 1
        self.assertEqual(anomalies_count, 12)


    def test_PeriodicAverageAnomalyDetector_save_load(self):

        anomaly_detector = PeriodicAverageAnomalyDetector()

        anomaly_detector.fit(self.sine_minute_timeseries, periodicity=63, error_metric='E', error_distribution='norm')

        # Set model save path
        model_path = TEMP_MODELS_DIR+'/test_periodic_average_anomaly_model'

        anomaly_detector.save(model_path)

        loaded_anomaly_detector = PeriodicAverageAnomalyDetector.load(model_path)
        self.assertEqual(set(loaded_anomaly_detector.data.keys()), set(['id', 'model_id', 'resolution', 'data_labels', 'error_metric', 'prediction_errors',
                                                                        'error_distributions', 'error_distributions_params', 'error_distributions_stats', 'index_window',
                                                                        'fitted_at', 'stdevs', 'model_window', 'with_context', 'actual_values', 'predicted_values']))

        self.assertAlmostEqual(loaded_anomaly_detector.data['error_distributions_params']['sin']['loc'], -3.73646e-05, places=5)
        self.assertAlmostEqual(loaded_anomaly_detector.data['error_distributions_params']['sin']['scale'], 0.20957, places=4)


    def test_PeriodicAverageReconstructorAnomalyDetector(self):
        # This model is reconstructor-based

        anomaly_detector = PeriodicAverageReconstructorAnomalyDetector()

        anomaly_detector.fit(self.sine_minute_timeseries, error_metric='E', periodicity=63, error_distribution='norm')

        self.assertAlmostEqual(anomaly_detector.data['error_distributions_params']['sin']['loc'], -0.0000029020, places=7)
        self.assertAlmostEqual(anomaly_detector.data['error_distributions_params']['sin']['scale'], 0.23897, places=4)

        result_timeseries = anomaly_detector.apply(self.sine_minute_timeseries, index_bounds=['avg_err','3_sigma'])

        # Count how many anomalies were detected
        anomalies_count = 0
        for slot in result_timeseries:
            if slot.data_indexes['anomaly'] > 0.6:
                anomalies_count += 1
        self.assertEqual(anomalies_count, 23)

        # Test on Points as well
        timeseries = CSVFileStorage(TEST_DATA_PATH + '/csv/temperature.csv').get(limit=200)
        anomaly_detector = PeriodicAverageReconstructorAnomalyDetector()
        with self.assertRaises(ValueError):
            anomaly_detector.fit(timeseries, error_distribution='norm')

        timeseries = timeseries.resample(600)
        anomaly_detector.fit(timeseries, error_metric='E', error_distribution='norm')

        result_timeseries = anomaly_detector.apply(timeseries, index_bounds=['avg_err','5_sigma'])
        for point in result_timeseries:
            if 'anomaly' in  point.data_indexes:
                if point.data_indexes['anomaly'] > 0.9:
                    anomalies_count += 1
        self.assertEqual(anomalies_count, 23)


    def test_LSTMAnomalyDetector_multivariate_with_context(self):

        try:
            import tensorflow
        except ImportError:
            print('Skipping LSTM anomlay detector tests with multivariate time series as no tensorflow module installed')
            return

        timeseries_clean = TimeSeries()
        for i in range(200):
            sin_value = sin(i/10.0) + random.random()/10
            cos_value = cos(i/10.0) + random.random()/10
            timeseries_clean.append(DataTimePoint(t=i*60, data={'sin':sin_value, 'cos':cos_value}))

        timeseries_with_anomalies = TimeSeries()
        for i in range(200):
            if i >0 and i % 30 == 0:
                sin_value = 2
            else:
                sin_value = sin(i/10.0) + random.random()/10
            cos_value = cos(i/10.0) + random.random()/10
            timeseries_with_anomalies.append(DataTimePoint(t=i*60, data={'sin':sin_value, 'cos':cos_value}))

        # Test only with context, as it is the only extra capability with respect to the PeriodicAverageAnomalyDetector
        anomaly_detector = LSTMAnomalyDetector(window=10)
        anomaly_detector.fit(timeseries_clean, error_metric='E', error_distribution='norm', epochs=5, with_context=True, reproducible=True)

        results_timeseries = anomaly_detector.apply(timeseries_with_anomalies, index_bounds=['max_err','10_sigma'], verbose=False, details=False)
        self.assertEqual(results_timeseries[0].data_indexes['anomaly'], 0)
        self.assertEqual(results_timeseries[19].data_indexes['anomaly'], 1)
        self.assertEqual(results_timeseries[20].data_indexes['anomaly'], 0)
        self.assertEqual(results_timeseries[49].data_indexes['anomaly'], 1)
        self.assertEqual(results_timeseries[79].data_indexes['anomaly'], 1)
        self.assertEqual(results_timeseries[109].data_indexes['anomaly'], 1)
        self.assertGreater(results_timeseries[139].data_indexes['anomaly'], 0.5)
        self.assertEqual(results_timeseries[169].data_indexes['anomaly'], 1)

        # Save & load
        model_path = TEMP_MODELS_DIR+'/test_lstm_anomaly_model_with_context'
        anomaly_detector.save(model_path)
        loaded_anomaly_detector = LSTMAnomalyDetector.load(model_path)
        results_timeseries = loaded_anomaly_detector.apply(timeseries_with_anomalies, index_bounds=['max_err','10_sigma'], verbose=False, details=False)
        self.assertEqual(results_timeseries[0].data_indexes['anomaly'], 0)
        self.assertEqual(results_timeseries[19].data_indexes['anomaly'], 1)
        self.assertEqual(results_timeseries[20].data_indexes['anomaly'], 0)
        self.assertEqual(results_timeseries[49].data_indexes['anomaly'], 1)
        self.assertEqual(results_timeseries[79].data_indexes['anomaly'], 1)
        self.assertEqual(results_timeseries[109].data_indexes['anomaly'], 1)
        self.assertGreater(results_timeseries[139].data_indexes['anomaly'], 0.5)
        self.assertEqual(results_timeseries[169].data_indexes['anomaly'], 1)


    def test_PeriodicAverageAnomalyDetector_with_index_window(self):

        anomaly_detector = PeriodicAverageAnomalyDetector(index_window=3)
        anomaly_detector.fit(self.sine_minute_timeseries, periodicity=63, error_metric='E', error_distribution='norm')

        self.assertAlmostEqual(anomaly_detector.data['error_distributions_params']['sin']['loc'], -7.83555e-05, places=6)
        self.assertAlmostEqual(anomaly_detector.data['error_distributions_params']['sin']['scale'], 0.39107, places=4)

        result_timeseries = anomaly_detector.apply(self.sine_minute_timeseries[0:70],  index_bounds=['avg_err','3_sigma'])

        self.assertAlmostEqual(result_timeseries[0].data_indexes['anomaly'], 0.0410, places=3)
        self.assertAlmostEqual(result_timeseries[1].data_indexes['anomaly'], 0.0223, places=3)
        self.assertAlmostEqual(result_timeseries[-1].data_indexes['anomaly'], 0.00998, places=4)

