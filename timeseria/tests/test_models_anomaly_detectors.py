import unittest
import os
import tempfile
import random
import numpy
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

class TestAnomalyDetectors(unittest.TestCase):

    def setUp(self):

        # Create a minute-resolution test time series (1000 elements)
        self.sine_minute_time_series = TimeSeries()
        for i in range(1000):
            if i % 100 == 0:
                value = 2
            else:
                value = sin(i/10.0)

            self.sine_minute_time_series.append(DataTimeSlot(start=TimePoint(i*60), end=TimePoint((i+1)*60), data={'sin':value}))

        # Create a minute-resolution multivariate test time series (100 elements)
        self.sine_cosine_minute_time_series = TimeSeries()
        for i in range(100):
            if i % 100 == 0:
                sin_value = 2
            else:
                sin_value = sin(i/10.0)
            cos_value = cos(i/10.0)

            self.sine_cosine_minute_time_series.append(DataTimePoint(t=i*60, data={'sin':sin_value, 'cos':cos_value}))

        # Ensure reproducibility
        random.seed(0)
        numpy.random.seed(0)
        try:
            import tensorflow
            import keras
        except ImportError:
            pass
        else:
            # Ensure reproducibility for Keras and Tensorflow as well
            # https://stackoverflow.com/questions/45230448/how-to-get-reproducible-result-when-running-keras-with-tensorflow-backend
            tensorflow_session_conf = tensorflow.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
            keras.backend.set_session(tensorflow.compat.v1.Session(graph=tensorflow.compat.v1.get_default_graph(), config=tensorflow_session_conf))
            tensorflow.random.set_seed(0)


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
        self.assertAlmostEqual(anomaly_detector.data['error_distributions_params']['sin']['loc'], 0.00029083304321826024, places=5)
        self.assertAlmostEqual(anomaly_detector.data['error_distributions_params']['sin']['scale'], 0.23226485795568802, places=4)

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


    def test_PeriodicAverageAnomalyDetector_save_load(self):

        anomaly_detector = PeriodicAverageAnomalyDetector()

        anomaly_detector.fit(self.sine_minute_time_series, periodicity=63, error_distribution='norm')

        # Set model save path
        model_path = TEMP_MODELS_DIR+'/test_anomaly_model'

        anomaly_detector.save(model_path)

        loaded_anomaly_detector = PeriodicAverageAnomalyDetector.load(model_path)
        self.assertEqual(set(anomaly_detector.data.keys()), set(['id', 'model_id', 'resolution', 'data_labels', 'prediction_errors',
                                                              'error_distributions', 'error_distributions_params', 'error_distributions_stats',
                                                              'fitted_at', 'stdevs', 'model_window', 'with_context']))

        self.assertAlmostEqual(anomaly_detector.data['error_distributions_params']['sin']['loc'], 0.00029083304321826024, places=5)
        self.assertAlmostEqual(anomaly_detector.data['error_distributions_params']['sin']['scale'], 0.23226485795568802, places=4)

        _ = loaded_anomaly_detector.apply(self.sine_minute_time_series, index_range=['avg_err','3_sigma'])


    def test_PeriodicAverageReconstructorAnomalyDetector(self):
        # This model is reconstructor-based

        anomaly_detector = PeriodicAverageReconstructorAnomalyDetector()

        anomaly_detector.fit(self.sine_minute_time_series, periodicity=63, error_distribution='norm')

        self.assertAlmostEqual(anomaly_detector.data['error_distributions_params']['sin']['loc'], -0.0000029020, places=7)
        self.assertAlmostEqual(anomaly_detector.data['error_distributions_params']['sin']['scale'], 0.23897, places=4)

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


    def test_LSTMAnomalyDetector_multivariate(self):

        timeseries_clean = TimeSeries()
        for i in range(100):
            sin_value = sin(i/10.0)
            cos_value = (cos(i/10.0)*2) + 1
            timeseries_clean.append(DataTimePoint(t=i*60, data={'sin':sin_value, 'cos':cos_value}))


        timeseries_with_anomalies = TimeSeries()
        for i in range(100):
            if i >0 and i % 30 == 0:
                sin_value = 2
            else:
                sin_value = sin(i/10.0)
            cos_value = (cos(i/10.0)*2) + 1
            timeseries_with_anomalies.append(DataTimePoint(t=i*60, data={'sin':sin_value, 'cos':cos_value}))

        # Semi-supervised
        anomaly_detector = LSTMAnomalyDetector()
        anomaly_detector.fit(timeseries_clean, error_distribution='norm', epochs=30, verbose=False)
        self.assertAlmostEqual(anomaly_detector.data['stdevs']['sin'], 0.060, places=2)
        self.assertAlmostEqual(anomaly_detector.data['stdevs']['cos'], 0.081, places=2)

        results_timeseries = anomaly_detector.apply(timeseries_with_anomalies, index_range=['max_err','3_sigma'])
        self.assertEqual(results_timeseries[0].data_indexes['anomaly'], 0)
        self.assertEqual(results_timeseries[26].data_indexes['anomaly'], 1)
        self.assertEqual(results_timeseries[56].data_indexes['anomaly'], 1)
        self.assertEqual(results_timeseries[86].data_indexes['anomaly'], 1)

        # Unsupervised
        anomaly_detector = LSTMAnomalyDetector()
        anomaly_detector.fit(timeseries_with_anomalies, error_distribution='norm', epochs=20, verbose=False)
        self.assertAlmostEqual(anomaly_detector.data['stdevs']['sin'], 0.41, places=2)
        self.assertAlmostEqual(anomaly_detector.data['stdevs']['cos'], 0.11, places=2)

        results_timeseries = anomaly_detector.apply(timeseries_with_anomalies)
        self.assertEqual(results_timeseries[0].data_indexes['anomaly'], 0)
        self.assertEqual(results_timeseries[26].data_indexes['anomaly'], 1)
        self.assertEqual(results_timeseries[56].data_indexes['anomaly'], 1)
        self.assertEqual(results_timeseries[86].data_indexes['anomaly'], 1)


    def test_LSTMAnomalyDetector_multivariate_with_context(self):

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

        # Semi-supervised
        anomaly_detector = LSTMAnomalyDetector(window=10)
        anomaly_detector.fit(timeseries_clean, error_distribution='norm', epochs=5, with_context=True)
        self.assertAlmostEqual(anomaly_detector.data['stdevs']['cos'], 0.104, places=2) 
        self.assertAlmostEqual(anomaly_detector.data['stdevs']['sin'], 0.070, places=2)

        results_timeseries = anomaly_detector.apply(timeseries_with_anomalies, index_range=['max_err','10_sigma'], verbose=False, details=False)
        self.assertEqual(results_timeseries[0].data_indexes['anomaly'], 0)
        self.assertEqual(results_timeseries[19].data_indexes['anomaly'], 1)
        self.assertAlmostEqual(results_timeseries[20].data_indexes['anomaly'], 0.2869, places=2)
        self.assertEqual(results_timeseries[49].data_indexes['anomaly'], 1)
        self.assertEqual(results_timeseries[79].data_indexes['anomaly'], 1)
        self.assertEqual(results_timeseries[109].data_indexes['anomaly'], 1)
        self.assertAlmostEqual(results_timeseries[139].data_indexes['anomaly'], 0.806, places=2)
        self.assertEqual(results_timeseries[169].data_indexes['anomaly'], 1)

        # Unsupervised TODO: Does not really work...
        # anomaly_detector = LSTMAnomalyDetector()
        # anomaly_detector.fit(timeseries_with_anomalies, error_distribution='norm', epochs=20, with_context=True)
        # self.assertAlmostEqual(anomaly_detector.data['stdevs']['cos'], 0.059, places=2)
        # self.assertAlmostEqual(anomaly_detector.data['stdevs']['sin'], 0.357, places=2)
        #
        # results_timeseries = anomaly_detector.apply(timeseries_with_anomalies)
        #for i, item in enumerate(results_timeseries):
        #    print(i, item.dt, item.data, item.data_indexes['anomaly'])
        # self.assertAlmostEqual(results_timeseries[0].data_indexes['anomaly'], 0.687, places=2) # TODO: Meh...
        # self.assertEqual(results_timeseries[19].data_indexes['anomaly'], 1)
        # self.assertEqual(results_timeseries[23].data_indexes['anomaly'], 0)
        # self.assertEqual(results_timeseries[49].data_indexes['anomaly'], 1)
        # self.assertEqual(results_timeseries[79].data_indexes['anomaly'], 1)
        # self.assertEqual(results_timeseries[109].data_indexes['anomaly'], 1)
        # self.assertEqual(results_timeseries[139].data_indexes['anomaly'], 1)
        # self.assertEqual(results_timeseries[169].data_indexes['anomaly'], 1)


