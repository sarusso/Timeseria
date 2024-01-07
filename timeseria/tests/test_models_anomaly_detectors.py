import unittest
import os
import tempfile
from math import sin
from ..datastructures import TimePoint, DataTimeSlot, DataTimePoint, TimeSeries
from ..models.anomaly_detectors import PeriodicAverageReconstructorAnomalyDetector, PeriodicAverageAnomalyDetector
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

        # Create a minute-resolution test DataTimeSlotSeries
        self.sine_minute_time_series = TimeSeries()
        for i in range(1000):
            if i % 100 == 0:
                value = 2
            else:
                value = sin(i/10.0)

            self.sine_minute_time_series.append(DataTimeSlot(start=TimePoint(i*60), end=TimePoint((i+1)*60), data={'value':value}))


    def test_PeriodicAverageAnomalyDetector(self):
        # This model is forecaster-based

        anomaly_detector = PeriodicAverageAnomalyDetector()

        anomaly_detector.fit(self.sine_minute_time_series, periodicity=63, distribution='norm')

        self.assertAlmostEqual(anomaly_detector.data['error_distribution_params']['loc'], -0.000642, places=5)
        self.assertAlmostEqual(anomaly_detector.data['error_distribution_params']['scale'], 0.21016, places=4)

        result_time_series = anomaly_detector.apply(self.sine_minute_time_series,  index_range=['avg','3_sigma'])

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
            anomaly_detector.fit(time_series, distribution='norm')

        time_series = time_series.resample(600)
        anomaly_detector.fit(time_series, distribution='norm')

        # TODO: do some actual testing.. not only that "it works"
        _  = anomaly_detector.apply(time_series, index_range=['avg','3_sigma'])


    def test_PeriodicAverageAnomalyDetector_save_load(self):

        anomaly_detector = PeriodicAverageAnomalyDetector()

        anomaly_detector.fit(self.sine_minute_time_series, periodicity=63, distribution='norm')

        # Set model save path
        model_path = TEMP_MODELS_DIR+'/test_anomaly_model'

        anomaly_detector.save(model_path)

        loaded_anomaly_detector = PeriodicAverageAnomalyDetector(model_path)
        self.assertEqual(set(anomaly_detector.data.keys()), set(['id', 'model_id', 'resolution', 'data_labels', 'prediction_errors',
                                                              'error_distribution', 'error_distribution_params', 'error_distribution_stats',
                                                              'fitted_at', 'stdev']))

        self.assertAlmostEqual(anomaly_detector.data['error_distribution_params']['loc'], -0.000642, places=5)
        self.assertAlmostEqual(anomaly_detector.data['error_distribution_params']['scale'], 0.21016, places=4)

        _ = loaded_anomaly_detector.apply(self.sine_minute_time_series, index_range=['avg','3_sigma'])


    def test_PeriodicAverageReconstructorAnomalyDetector(self):
        # This model is reconstructor-based

        anomaly_detector = PeriodicAverageReconstructorAnomalyDetector()

        anomaly_detector.fit(self.sine_minute_time_series, periodicity=63, distribution='norm')

        self.assertAlmostEqual(anomaly_detector.data['error_distribution_params']['loc'], -0.0000029020, places=7)
        self.assertAlmostEqual(anomaly_detector.data['error_distribution_params']['scale'], 0.23897, places=4)

        # The prediction errors are not normally distributed, and the default index range of ['avg', 'max'] does not work
        with self.assertRaises(ValueError):
            anomaly_detector.apply(self.sine_minute_time_series)

        result_time_series = anomaly_detector.apply(self.sine_minute_time_series, index_range=['avg','3_sigma'])

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
            anomaly_detector.fit(time_series, distribution='norm')

        time_series = time_series.resample(600)
        anomaly_detector.fit(time_series, distribution='norm')

        # TODO: do some actual testing.. not only that "it works"
        _  = anomaly_detector.apply(time_series, index_range=['avg','3_sigma'])



