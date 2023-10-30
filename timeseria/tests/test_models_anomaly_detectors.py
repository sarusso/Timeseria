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

        self.assertAlmostEqual(anomaly_detector.data['gaussian_mu'], -0.00064, places=5)
        self.assertAlmostEqual(anomaly_detector.data['gaussian_sigma'], 0.21016, places=4)
        
        result_time_series = anomaly_detector.apply(self.sine_minute_time_series)

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
            anomaly_detector.fit(time_series)
          
        time_series = time_series.resample(600)
        anomaly_detector.fit(time_series)
          
        # TODO: do some actual testing.. not only that "it works"
        _  = anomaly_detector.apply(time_series)


    def test_PeriodicAverageAnomalyDetector_save_load(self):
        
        anomaly_detector = PeriodicAverageAnomalyDetector()
        
        anomaly_detector.fit(self.sine_minute_time_series, periodicity=63)
       
        # Set model save path
        model_path = TEMP_MODELS_DIR+'/test_anomaly_model'
        
        anomaly_detector.save(model_path)
        
        loaded_anomaly_detector = PeriodicAverageAnomalyDetector(model_path)
        self.assertEqual(list(anomaly_detector.data.keys()), ['id', 'forecaster_id', 'resolution', 'data_labels', 'forecasting_errors', 'gaussian_mu', 'gaussian_sigma', 'fitted_at'])
        
        self.assertAlmostEqual(anomaly_detector.data['gaussian_mu'], -0.00064, places=5)
        self.assertAlmostEqual(anomaly_detector.data['gaussian_sigma'], 0.21016, places=4)
        
        _ = loaded_anomaly_detector.apply(self.sine_minute_time_series)

