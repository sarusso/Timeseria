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


class TestBaseModelClasses(unittest.TestCase):

    def test_Model(self):
                
        # Test non-parametric
        model = Model()
        self.assertFalse(model.is_parametric())
        self.assertEqual(model._type, 'non-parametric')
        with self.assertRaises(TypeError):
            model.save(TEMP_MODELS_DIR+'/test_model')
        with self.assertRaises(TypeError):
            model.id
        
        # Test parametric 
        class ParametricModelMock(Model):
            def __init__(self, path=None, param1=0):
                self.data = {}
                self.data['param1'] = param1
                super(ParametricModelMock, self).__init__(path)
        
        parametric_model = ParametricModelMock(param1=1)
        self.assertTrue(parametric_model.is_parametric())
        self.assertEqual(parametric_model._type, 'parametric')
        self.assertEqual(parametric_model.data['param1'], 1)
        parametric_model.save(TEMP_MODELS_DIR+'/parametric_model')
        
        parametric_model = ParametricModelMock(TEMP_MODELS_DIR+'/parametric_model')
        self.assertTrue(parametric_model.is_parametric())
        self.assertEqual(parametric_model.data['param1'], 1)


      
    def test_Model_with_TimeSeries(self):

        # Define test time series
        empty_time_series = TimeSeries()
        time_series = TimeSeries(DataTimeSlot(start=TimePoint(t=1), end=TimePoint(t=2), data={'metric1': 56}),
                                 DataTimeSlot(start=TimePoint(t=2), end=TimePoint(t=3), data={'metric1': 56}))

        # Define a fittable parametric model mock
        class FittableParametricModelMock(Model):
            def _fit(self, data):
                self.data['param1'] = 1
            def _predict(self, data):
                return data
            def _apply(self, data):
                return data
            def _evaluate(self, data):
                return {'grade': 'A'}

        parametric_model = FittableParametricModelMock()
        self.assertTrue(parametric_model.is_parametric())
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
        with self.assertRaises(NotImplementedError):
            parametric_model.predict([1,2,3,4,5,6])
        self.assertEqual(parametric_model.predict(time_series), time_series)
        
        # Check apply
        with self.assertRaises(ValueError):
            parametric_model.apply(empty_time_series)
        with self.assertRaises(NotImplementedError):
            parametric_model.apply([1,2,3,4,5,6])        
        self.assertEqual(parametric_model.apply(time_series), time_series)
        
        # Check evaluate
        with self.assertRaises(ValueError):
            parametric_model.evaluate(empty_time_series)
        with self.assertRaises(NotImplementedError):
            parametric_model.evaluate([1,2,3,4,5,6])
        self.assertEqual(parametric_model.evaluate(time_series), {'grade': 'A'})

        # Save and re-load
        parametric_model.save(TEMP_MODELS_DIR+'/fittable_parametric_model')        
        loaded_parametric_model = FittableParametricModelMock(TEMP_MODELS_DIR+'/fittable_parametric_model')
        self.assertTrue(loaded_parametric_model.is_parametric())
        self.assertEqual(loaded_parametric_model.data['param1'], 1)
        self.assertEqual(loaded_parametric_model.id, parametric_model.id)


    def test_KerasModel(self):
        
        # Define a Keras model mock
        class KerasModelMock(_KerasModel):            
            def _fit(self, *args, **kwargs):
                pass 
            def _evaluate(self, *args, **kwargs):
                pass
            def _apply(self, *args, **kwargs):
                pass 

        # Define test time series
        data_time_point_series = TimeSeries(DataTimePoint(t=1, data={'metric1': 0.1}),
                                            DataTimePoint(t=2, data={'metric1': 0.2}),
                                            DataTimePoint(t=3, data={'metric1': 0.3}),
                                            DataTimePoint(t=4, data={'metric1': 0.4}),
                                            DataTimePoint(t=5, data={'metric1': 0.5}),
                                            DataTimePoint(t=6, data={'metric1': 0.6}),)


        # Test window generation functions
        window_matrix = KerasModelMock._to_window_datapoints_matrix(data_time_point_series, window=2, steps=1, encoder=None)
        
        # What to expect (using the timestamp to represent a data point):
        # 1,2
        # 2,3
        # 3,4
        # 4,5

        self.assertEqual(len(window_matrix), 4)
        
        self.assertEqual(window_matrix[0][0].t, 1)
        self.assertEqual(window_matrix[0][1].t, 2)
        
        self.assertEqual(window_matrix[1][0].t, 2)
        self.assertEqual(window_matrix[1][1].t, 3)

        self.assertEqual(window_matrix[2][0].t, 3)
        self.assertEqual(window_matrix[2][1].t, 4)
        
        self.assertEqual(window_matrix[-1][0].t, 4)
        self.assertEqual(window_matrix[-1][1].t, 5)
        
        target_vector = KerasModelMock._to_target_values_vector(data_time_point_series, window=2, steps=1)

        # What to expect (using the data value to represent a data point):
        # [0.3], [0.4], [0.5], [0.6] Note that they are lists in order to support multi-step forecast
        self.assertEqual(target_vector, [[0.3], [0.4], [0.5], [0.6]])
 
