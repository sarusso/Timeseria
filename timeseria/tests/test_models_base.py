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



class TestBaseBaseModel(unittest.TestCase):

    def test_Model(self):

        # Test non-parametric
        model = Model()
        self.assertFalse(model._is_parametric())
        self.assertEqual(model._type, 'non-parametric')
        with self.assertRaises(TypeError):
            model.save(TEMP_MODELS_DIR+'/test_model')
        with self.assertRaises(TypeError):
            model.id

        # Test parametric
        class ParametricModelMock(Model):
            def __init__(self, param1=0):
                self.data = {}
                self.data['param1'] = param1
                super(ParametricModelMock, self).__init__()

        parametric_model = ParametricModelMock(param1=1)
        self.assertTrue(parametric_model._is_parametric())
        self.assertEqual(parametric_model._type, 'parametric')
        self.assertEqual(parametric_model.data['param1'], 1)
        parametric_model.save(TEMP_MODELS_DIR+'/parametric_model')

        parametric_model = ParametricModelMock.load(TEMP_MODELS_DIR+'/parametric_model')
        self.assertTrue(parametric_model._is_parametric())
        self.assertEqual(parametric_model.data['param1'], 1)



    def test_Model_with_TimeSeries(self):

        # Define test time series
        empty_time_series = TimeSeries()
        time_series = TimeSeries(DataTimeSlot(start=TimePoint(t=1), end=TimePoint(t=2), data={'metric1': 56}),
                                 DataTimeSlot(start=TimePoint(t=2), end=TimePoint(t=3), data={'metric1': 56}))

        # Define a fittable parametric model mock
        class FittableParametricModelMock(Model):
            @Model.fit_function
            def fit(self, series):
                self.data['param1'] = 1
            @Model.predict_function
            def predict(self, series):
                return series
            @Model.apply_function
            def apply(self, series):
                return series
            @Model.evaluate_function
            def evaluate(self, series):
                return {'grade': 'A'}

        parametric_model = FittableParametricModelMock()
        self.assertTrue(parametric_model._is_parametric())
        self.assertEqual(parametric_model._type, 'parametric')

        # Cannot predict form a model before fitting
        with self.assertRaises(NotFittedError):
            parametric_model.predict(time_series)

        # Cannot apply model before fitting
        with self.assertRaises(NotFittedError):
            parametric_model.apply(time_series)

        # Cannot evaluate model before fitting
        with self.assertRaises(NotFittedError):
            parametric_model.evaluate(time_series)

        # Cannot save model before fitting
        with self.assertRaises(NotFittedError):
            parametric_model.save(TEMP_MODELS_DIR+'/parametric_model')

        # Fit the model
        with self.assertRaises(ValueError):
            parametric_model.fit(empty_time_series)
        parametric_model.fit(time_series)
        self.assertEqual(parametric_model.data['param1'], 1)

        # Check predict
        with self.assertRaises(ValueError):
            parametric_model.predict(empty_time_series)
        with self.assertRaises(TypeError):
            parametric_model.predict([1,2,3,4,5,6])
        self.assertEqual(parametric_model.predict(time_series), time_series)

        # Check apply
        with self.assertRaises(ValueError):
            parametric_model.apply(empty_time_series)
        with self.assertRaises(TypeError):
            parametric_model.apply([1,2,3,4,5,6])
        self.assertEqual(parametric_model.apply(time_series), time_series)

        # Check evaluate
        with self.assertRaises(ValueError):
            parametric_model.evaluate(empty_time_series)
        with self.assertRaises(TypeError):
            parametric_model.evaluate([1,2,3,4,5,6])
        self.assertEqual(parametric_model.evaluate(time_series), {'grade': 'A'})

        # Save and re-load
        parametric_model.save(TEMP_MODELS_DIR+'/fittable_parametric_model')
        loaded_parametric_model = FittableParametricModelMock.load(TEMP_MODELS_DIR+'/fittable_parametric_model')
        self.assertTrue(loaded_parametric_model._is_parametric())
        self.assertEqual(loaded_parametric_model.data['param1'], 1)
        self.assertEqual(loaded_parametric_model.id, parametric_model.id)


class TestBaseKerasModel(unittest.TestCase):

    def setUp(self):

        self.test_time_series = TimeSeries(DataTimePoint(t=1, data={'label_1': 0.1}),
                                           DataTimePoint(t=2, data={'label_1': 0.2}),
                                           DataTimePoint(t=3, data={'label_1': 0.3}),
                                           DataTimePoint(t=4, data={'label_1': 0.4}),
                                           DataTimePoint(t=5, data={'label_1': 0.5}),
                                           DataTimePoint(t=6, data={'label_1': 0.6}),)

        self.test_time_series_mv = TimeSeries(DataTimePoint(t=1, data={'label_1': 0.1, 'label_2': 1.0}),
                                              DataTimePoint(t=2, data={'label_1': 0.2, 'label_2': 2.0}),
                                              DataTimePoint(t=3, data={'label_1': 0.3, 'label_2': 3.0}),
                                              DataTimePoint(t=4, data={'label_1': 0.4, 'label_2': 4.0}),
                                              DataTimePoint(t=5, data={'label_1': 0.5, 'label_2': 5.0}),
                                              DataTimePoint(t=6, data={'label_1': 0.6, 'label_2': 6.0}),)

    def test_to_window_datapoints_matrix(self):

        window_datapoints_matrix = _KerasModel._to_window_datapoints_matrix(self.test_time_series, window=2, steps=1)

        # What to expect (using the timestamp to represent a data point):
        # 1,2
        # 2,3
        # 3,4
        # 4,5

        self.assertEqual(len(window_datapoints_matrix), 4)

        self.assertEqual(window_datapoints_matrix[0][0].t, 1)
        self.assertEqual(window_datapoints_matrix[0][1].t, 2)

        self.assertEqual(window_datapoints_matrix[1][0].t, 2)
        self.assertEqual(window_datapoints_matrix[1][1].t, 3)

        self.assertEqual(window_datapoints_matrix[2][0].t, 3)
        self.assertEqual(window_datapoints_matrix[2][1].t, 4)

        self.assertEqual(window_datapoints_matrix[-1][0].t, 4)
        self.assertEqual(window_datapoints_matrix[-1][1].t, 5)


    def _to_target_values_vector(self):

        target_vector = _KerasModel._to_target_values_vector(self.test_time_series, window=2, steps=1)

        # What to expect (using the data value to represent a data point):
        # [0.3], [0.4], [0.5], [0.6] Note that they are lists in order to support multi-step forecast
        self.assertEqual(target_vector, [[0.3], [0.4], [0.5], [0.6]])


    def test_compute_window_features(self):


        window_datapoints_matrix = _KerasModel._to_window_datapoints_matrix(self.test_time_series, window=3, steps=1)


        window_features  = _KerasModel._compute_window_features(window_datapoints_matrix[0],
                                                                data_labels = self.test_time_series.data_labels(),
                                                                time_unit = self.test_time_series.resolution,
                                                                features = ['values'])

        self.assertEqual(window_features, [[0.1],[0.2],[0.3]])


        window_features  = _KerasModel._compute_window_features(window_datapoints_matrix[0],
                                                                data_labels = self.test_time_series.data_labels(),
                                                                time_unit = self.test_time_series.resolution,
                                                                features = ['values', 'diffs'])
        self.assertAlmostEqual(window_features[0][0], 0.1)
        self.assertAlmostEqual(window_features[0][1], 0.1)
        self.assertAlmostEqual(window_features[1][0], 0.2)
        self.assertAlmostEqual(window_features[1][1], 0.1)
        self.assertAlmostEqual(window_features[-1][0], 0.3)
        self.assertAlmostEqual(window_features[-1][1], 0.1)

        # Multivariate
        window_datapoints_matrix_mv = _KerasModel._to_window_datapoints_matrix(self.test_time_series_mv, window=3, steps=1)
        window_features  = _KerasModel._compute_window_features(window_datapoints_matrix_mv[0],
                                                                data_labels = self.test_time_series_mv.data_labels(),
                                                                time_unit = self.test_time_series_mv.resolution,
                                                                features = ['values'])

        self.assertEqual(window_features, [[0.1, 1.0], [0.2, 2.0], [0.3, 3.0]])

        # Multivariate with context
        window_features  = _KerasModel._compute_window_features(window_datapoints_matrix_mv[0],
                                                                data_labels = self.test_time_series_mv.data_labels(),
                                                                time_unit = self.test_time_series_mv.resolution,
                                                                features = ['values', 'diffs', 'hours'],
                                                                context_data = {'label_2':4.0})
        self.assertAlmostEqual(window_features[0][0], 0.1)
        self.assertAlmostEqual(window_features[0][1], 1.0)
        self.assertAlmostEqual(window_features[0][2], 0.1)
        self.assertAlmostEqual(window_features[0][3], 1.0)
        self.assertAlmostEqual(window_features[0][4], 0.0)

        self.assertAlmostEqual(window_features[1][0], 0.2)
        self.assertAlmostEqual(window_features[1][1], 2.0)
        self.assertAlmostEqual(window_features[1][2], 0.1)
        self.assertAlmostEqual(window_features[1][3], 1.0)
        self.assertAlmostEqual(window_features[1][4], 0.0)

        self.assertAlmostEqual(window_features[-1][0], 0.0)
        self.assertAlmostEqual(window_features[-1][1], 4.0)
        self.assertAlmostEqual(window_features[-1][2], 0.0)
        self.assertAlmostEqual(window_features[-1][3], 1.0)
        self.assertAlmostEqual(window_features[-1][4], 0.0)

