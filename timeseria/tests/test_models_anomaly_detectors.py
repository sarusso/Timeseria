import unittest
import os
import tempfile
from math import sin, cos
from ..datastructures import TimePoint, DataTimeSlot, DataTimePoint, TimeSeries
from ..models.forecasters import PeriodicAverageForecaster
from ..models.reconstructors import PeriodicAverageReconstructor
from ..models.anomaly_detectors import ModelBasedAnomalyDetector, PeriodicAverageReconstructorAnomalyDetector, PeriodicAverageAnomalyDetector, LSTMAnomalyDetector
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


class TestAnomalyDetectors(unittest.TestCase):

    def setUp(self):

        # Create a minute-resolution test time series (1000 elements)
        self.sine_minute_time_series = TimeSeries()
        for i in range(1000):
            if i % 100 == 0:
                value = 2
            else:
                value = sin(i/10.0)

            self.sine_minute_time_series.append(DataTimeSlot(start=TimePoint(i*60), end=TimePoint((i+1)*60), data={'value':value}))

        # Create a minute-resolution multivariate test time series (100 elements)
        self.sine_cosine_minute_time_series = TimeSeries()
        for i in range(100):
            if i % 100 == 0:
                sin_value = 2
            else:
                sin_value = sin(i/10.0)
            cos_value = cos(i/10.0)

            self.sine_cosine_minute_time_series.append(DataTimePoint(t=i*60, data={'sin':sin_value, 'cos':cos_value}))


    def test_ModelBasedAnomalyDetector(self):

        anomaly_detector = ModelBasedAnomalyDetector(model_class=PeriodicAverageForecaster)
        anomaly_detector.fit(self.sine_minute_time_series, periodicity=63, error_distribution='norm')
        result_time_series = anomaly_detector.apply(self.sine_minute_time_series,  index_range=['avg_err','3_sigma'])
        self.assertEqual(len(result_time_series),936)
        self.assertAlmostEqual(result_time_series[0].data_indexes['anomaly'], 0.05243927385024196)
        self.assertEqual(result_time_series[236].data_indexes['anomaly'], 1)

        anomaly_detector = ModelBasedAnomalyDetector(model_class=PeriodicAverageReconstructor)
        anomaly_detector.fit(self.sine_minute_time_series, periodicity=63, error_distribution='norm')
        result_time_series = anomaly_detector.apply(self.sine_minute_time_series,  index_range=['avg_err','3_sigma'])
        self.assertEqual(len(result_time_series),997)
        self.assertEqual(result_time_series[0].data_indexes['anomaly'], 0)
        self.assertAlmostEqual(result_time_series[5].data_indexes['anomaly'], 0.001868259)
        self.assertEqual(result_time_series[298].data_indexes['anomaly'], 1)


    def test_PeriodicAverageAnomalyDetector(self):
        # This model is forecaster-based

        anomaly_detector = PeriodicAverageAnomalyDetector()

        anomaly_detector.fit(self.sine_minute_time_series, periodicity=63, error_distribution='norm')

        self.assertAlmostEqual(anomaly_detector.data['error_distribution_params']['loc'], 0.00029083304321826024, places=5)
        self.assertAlmostEqual(anomaly_detector.data['error_distribution_params']['scale'], 0.23226485795568802, places=4)

        result_time_series = anomaly_detector.apply(self.sine_minute_time_series,  index_range=['avg_err','3_sigma'])

        # Count how many anomalies were detected
        anomalies_count = 0
        for slot in result_time_series:
            if slot.data_indexes['anomaly'] > 0.6:
                anomalies_count += 1
        self.assertEqual(anomalies_count, 9)

        # Test on Points as well
        time_series = CSVFileStorage(TEST_DATA_PATH + '/csv/temperature.csv').get(limit=200)
        anomaly_detector = PeriodicAverageAnomalyDetector()
        with self.assertRaises(ValueError):
            anomaly_detector.fit(time_series, error_distribution='norm')

        time_series = time_series.resample(600)
        anomaly_detector.fit(time_series, error_distribution='norm')

        # TODO: do some actual testing.. not only that "it works"
        _  = anomaly_detector.apply(time_series, index_range=['avg_err','3_sigma'])


    def test_LSTMAnomalyDetector_multivariate(self):

        anomaly_detector = LSTMAnomalyDetector()
        anomaly_detector.fit(self.sine_cosine_minute_time_series, error_distribution='norm', epochs=30)

        # Test with tolerance
        self.assertTrue(-0.3 <  anomaly_detector.data['error_distribution_params']['loc'] < 0.3, 'got {}'.format(anomaly_detector.data['error_distribution_params']['loc']))
        self.assertTrue(0.0 <  anomaly_detector.data['error_distribution_params']['scale'] < 0.1, 'got {}'.format(anomaly_detector.data['error_distribution_params']['scale']))

        # Apply and count how many anomalies were detected
        result_time_series = anomaly_detector.apply(self.sine_cosine_minute_time_series,  index_range=['avg_err','3_sigma'])

        anomalies_count = 0
        for slot in result_time_series:
            if slot.data_indexes['anomaly'] > 0.2:
                anomalies_count += 1
        self.assertTrue(10 <  anomalies_count < 30, 'got {}'.format(anomalies_count))


    def test_PeriodicAverageAnomalyDetector_save_load(self):

        anomaly_detector = PeriodicAverageAnomalyDetector()

        anomaly_detector.fit(self.sine_minute_time_series, periodicity=63, error_distribution='norm')

        # Set model save path
        model_path = TEMP_MODELS_DIR+'/test_anomaly_model'

        anomaly_detector.save(model_path)

        loaded_anomaly_detector = PeriodicAverageAnomalyDetector.load(model_path)
        self.assertEqual(set(anomaly_detector.data.keys()), set(['id', 'model_id', 'resolution', 'data_labels', 'prediction_errors',
                                                              'error_distribution', 'error_distribution_params', 'error_distribution_stats',
                                                              'fitted_at', 'stdev']))

        self.assertAlmostEqual(anomaly_detector.data['error_distribution_params']['loc'], 0.00029083304321826024, places=5)
        self.assertAlmostEqual(anomaly_detector.data['error_distribution_params']['scale'], 0.23226485795568802, places=4)

        _ = loaded_anomaly_detector.apply(self.sine_minute_time_series, index_range=['avg_err','3_sigma'])


    def test_PeriodicAverageReconstructorAnomalyDetector(self):
        # This model is reconstructor-based

        anomaly_detector = PeriodicAverageReconstructorAnomalyDetector()

        anomaly_detector.fit(self.sine_minute_time_series, periodicity=63, error_distribution='norm')

        self.assertAlmostEqual(anomaly_detector.data['error_distribution_params']['loc'], -0.0000029020, places=7)
        self.assertAlmostEqual(anomaly_detector.data['error_distribution_params']['scale'], 0.23897, places=4)

        result_time_series = anomaly_detector.apply(self.sine_minute_time_series, index_range=['avg_err','3_sigma'])

        # Count how many anomalies were detected
        anomalies_count = 0
        for slot in result_time_series:
            if slot.data_indexes['anomaly'] > 0.6:
                anomalies_count += 1
        self.assertEqual(anomalies_count, 23)

        # Test on Points as well
        time_series = CSVFileStorage(TEST_DATA_PATH + '/csv/temperature.csv').get(limit=200)
        anomaly_detector = PeriodicAverageReconstructorAnomalyDetector()
        with self.assertRaises(ValueError):
            anomaly_detector.fit(time_series, error_distribution='norm')

        time_series = time_series.resample(600)
        anomaly_detector.fit(time_series, error_distribution='norm')

        # TODO: do some actual testing.. not only that "it works"
        _  = anomaly_detector.apply(time_series, index_range=['avg_err','3_sigma'])



