import unittest
import os
import tempfile
from math import sin, cos
from propertime.utils import dt

from ..datastructures import TimePoint, DataTimeSlot, DataTimePoint, TimeSeries
from ..models.base import Model, _KerasModel
from ..models.reconstructors import PeriodicAverageReconstructor, ProphetReconstructor, LinearInterpolationReconstructor
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


class TestReconstructors(unittest.TestCase):


    def test_BaseReconstructor(self):

        # This test is done using the Linear Interpolator Reconstructor

        time_series = TimeSeries()
        time_series.append(DataTimePoint(t=1,  data={'a':1}, data_loss=0))
        time_series.append(DataTimePoint(t=2,  data={'a':-99}, data_loss=1))
        time_series.append(DataTimePoint(t=3,  data={'a':3}, data_loss=1))
        time_series.append(DataTimePoint(t=4,  data={'a':4}, data_loss=0))

        reconstructor = LinearInterpolationReconstructor()

        # Test the main predict
        predicted_data = reconstructor.predict(time_series, from_i=1,to_i=1)
        self.assertEqual(predicted_data, [{'a': 2.0}])

        predicted_data = reconstructor.predict(time_series, from_i=1,to_i=2)
        self.assertEqual(predicted_data, [{'a': 2.0}, {'a': 3.0}])

        # Not enough widow data for the predict, will raise
        with self.assertRaises(ValueError):
            reconstructor.predict(time_series, from_i=0,to_i=1)
        with self.assertRaises(ValueError):
            reconstructor.predict(time_series, from_i=3,to_i=4)

        # Test the apply now
        reconstructed_time_series = reconstructor.apply(time_series)
        self.assertEqual(reconstructed_time_series[1].data['a'],2)

        # Not enough widow data for the apply at the beginning, will [shrink?]
        #time_series = TimeSeries()
        #time_series.append(DataTimePoint(t=1,  data={'a':-99}, data_loss=1))
        #time_series.append(DataTimePoint(t=2,  data={'a':2}, data_loss=0))
        #reconstructor.apply(time_series)


        # Not enough widow data for the apply at the end, will [shrink?]
        #time_series = TimeSeries()
        #time_series.append(DataTimePoint(t=1,  data={'a':-99}, data_loss=1))
        #time_series.append(DataTimePoint(t=2,  data={'a':2}, data_loss=0))

        # Force the forecaster window to "2" and check that if gaps are separated
        # by a single point, all the are gets marked as to be reconstructed. What we want..?





    def test_PeriodicAverageReconstructor(self):

        # Get test data
        with open(TEST_DATA_PATH + '/csv/temp_slots_1h.csv') as f:
            data=f.read()

        time_series = TimeSeries()
        for line in data.split('\n'):
            if line:
                start_t = float(line.split(',')[0])
                start_point = TimePoint(t=start_t)
                end_point = TimePoint(t=start_t+3600)
                data_loss = 1-float(line.split(',')[2]) # from coverage to data loss
                value =  float(line.split(',')[1])
                time_series.append(DataTimeSlot(start=start_point, end=end_point, data={'temperature':value}, data_loss=data_loss))

        # # Instantiate
        # reconstructor = PeriodicAverageReconstructor()
        #
        # # Fit
        # reconstructor.fit(time_series)
        #
        # # Test the predict
        # predictions = reconstructor.predict(time_series, from_i=2,to_i=2)
        # self.assertEqual(len(predictions['temperature']), 1)
        #
        # predictions = reconstructor.predict(time_series, from_i=2,to_i=3)
        # self.assertEqual(len(predictions['temperature']), 2)
        #
        # predictions = reconstructor.predict(time_series, from_i=2,to_i=4)
        # self.assertEqual(len(predictions['temperature']), 3)
        #
        # predictions = reconstructor.predict(time_series, from_i=len(time_series)-3, to_i=len(time_series)-2)
        # self.assertEqual(len(predictions['temperature']), 2)
        #
        # # Not enough data for the window
        # with self.assertRaises(ValueError):
        #     reconstructor.predict(time_series, from_i=0,to_i=2)
        #
        # with self.assertRaises(ValueError):
        #     reconstructor.predict(time_series, from_i=len(time_series)-3, to_i=len(time_series)-1)
        #
        # # Test the apply
        # time_series_reconstructed = reconstructor.apply(time_series, data_loss_threshold=0.3)
        # self.assertEqual(len(time_series), len(time_series_reconstructed))
        # for i in range(len(time_series_reconstructed)):
        #     if time_series_reconstructed[i].data_loss >= 0.3:
        #         self.assertEqual(time_series_reconstructed[i].data_indexes['data_reconstructed'], 1)
        #     else:
        #         self.assertEqual(time_series_reconstructed[i].data_indexes['data_reconstructed'], 0)

        # Fit from/to (from_t, from_dt, from_i etc)
        reconstructor = PeriodicAverageReconstructor()
        reconstructor.fit(time_series, from_dt=dt(2019,3,1), to_dt=dt(2019,4,1))
        evaluation = reconstructor.evaluate(time_series, steps=[1,3], limit=100, details=True)
        self.assertAlmostEqual(evaluation['RMSE_3_steps'], 0.3220178189430215)

        # Fit to/from
        reconstructor = PeriodicAverageReconstructor()
        reconstructor.fit(time_series, to_dt=dt(2019,3,1), from_dt=dt(2019,4,1))
        evaluation = reconstructor.evaluate(time_series, steps=[1,3], limit=100, details=True)
        self.assertAlmostEqual(evaluation['RMSE_3_steps'], 0.23007000302456676)

        # Cross validations
        reconstructor = PeriodicAverageReconstructor()
        cross_validation = reconstructor.cross_validate(time_series, evaluate_steps=[1,3], evaluate_limit=100, evaluate_details=True)
        self.assertAlmostEqual(cross_validation['MAE_3_steps_avg'],  0.2201375926633982)
        self.assertAlmostEqual(cross_validation['MAE_3_steps_stdev'], 0.035208711688853166)

        # Test the evaluate
        evaluation = reconstructor.evaluate(time_series, limit=100, details=True)
        self.assertAlmostEqual(evaluation['RMSE_1_steps'], 0.13272865912894907)
        self.assertAlmostEqual(evaluation['MAE_1_steps'], 0.09466327970737677)
        self.assertAlmostEqual(evaluation['RMSE_24_steps'], 0.6570425995634208)
        self.assertAlmostEqual(evaluation['MAE_24_steps'], 0.5390349265995485)
        self.assertAlmostEqual(evaluation['RMSE'], 0.39488562934618493)
        self.assertAlmostEqual(evaluation['MAE'], 0.31684910315346265)

        # Test the evaluate on specific steps:
        evaluation = reconstructor.evaluate(time_series, steps=[1,3], limit=100, details=True)
        self.assertAlmostEqual(evaluation['RMSE_1_steps'], 0.13272865912894907)
        self.assertAlmostEqual(evaluation['MAE_1_steps'], 0.09466327970737677)
        self.assertAlmostEqual(evaluation['RMSE_3_steps'], 0.24206666463988694)
        self.assertAlmostEqual(evaluation['MAE_3_steps'], 0.19434430892864815)
        self.assertAlmostEqual(evaluation['RMSE'], 0.187397661884418)
        self.assertAlmostEqual(evaluation['MAE'], 0.14450379431801247)

        # Test on Points as well
        time_series = CSVFileStorage(TEST_DATA_PATH + '/csv/temperature.csv').get(limit=200)
        reconstructor = PeriodicAverageReconstructor()
        with self.assertRaises(Exception):
            reconstructor.fit(time_series)

        time_series = time_series.resample(600)
        reconstructor.fit(time_series)

        # TODO: do some actual testing.. not only that "it works"
        _  = reconstructor.apply(time_series)
        reconstructor.evaluate(time_series)


    def test_ProphetReconstructor(self):
        try:
            import prophet
        except ImportError:
            print('Skipping Prophet tests as no prophet module installed')
            return

        # Get test data
        time_series = CSVFileStorage(TEST_DATA_PATH + '/csv/temperature.csv').get(limit=200).aggregate(3600)

        # Instantiate
        reconstructor = ProphetReconstructor()

        # Fit
        reconstructor.fit(time_series)

        # Predict
        predicted_data = reconstructor.predict(time_series, from_i=2, to_i=5)
        self.assertAlmostEqual(predicted_data['temperature_avg'][0], 23.449303446403178)
        self.assertAlmostEqual(predicted_data['temperature_avg'][1], 23.45425065662797)
        self.assertAlmostEqual(predicted_data['temperature_avg'][2], 23.460265626928496)
        self.assertAlmostEqual(predicted_data['temperature_avg'][3], 23.46643974807204)

        # Predict on first and last as no window
        reconstructor.predict(time_series, from_i=0, to_i=0)
        reconstructor.predict(time_series, from_i=len(time_series)-1, to_i=len(time_series)-1)


    def test_LinearInterpolationReconstructor(self):

        # Test multivariate
        time_series = TimeSeries()
        time_series.append(DataTimePoint(t=1,  data={'a':1, 'b':4}, data_loss=0))
        time_series.append(DataTimePoint(t=2,  data={'a':-99, 'b':-99}, data_loss=1))
        time_series.append(DataTimePoint(t=3,  data={'a':3, 'b':2}, data_loss=1))
        time_series.append(DataTimePoint(t=4,  data={'a':4, 'b':1}, data_loss=0))

        reconstructor = LinearInterpolationReconstructor()

        predicted_data = reconstructor.predict(time_series, from_i=1,to_i=1)
        self.assertEqual(predicted_data, [{'a': 2.0, 'b': 3.0}])


        predicted_data = reconstructor.predict(time_series, from_i=1,to_i=2)
        self.assertEqual(predicted_data, [{'a': 2.0, 'b': 3.0}, {'a': 3.0, 'b': 2.0}])

        # TODO: move this test in the basic reconstructors
        reconstructed_time_series = reconstructor.apply(time_series)
        self.assertEqual(reconstructed_time_series[0].data, {'a': 1, 'b': 4})
        self.assertEqual(reconstructed_time_series[1].data, {'a': 2.0, 'b': 3.0})
        self.assertEqual(reconstructed_time_series[2].data, {'a': 3.0, 'b': 2.0})
        self.assertEqual(reconstructed_time_series[3].data, {'a': 4, 'b': 1})
        self.assertEqual(reconstructed_time_series[0].data_indexes, {'data_loss': 0, 'data_reconstructed': 0})
        self.assertEqual(reconstructed_time_series[1].data_indexes, {'data_loss': 1, 'data_reconstructed': 1})
        self.assertEqual(reconstructed_time_series[2].data_indexes, {'data_loss': 1, 'data_reconstructed': 1})
        self.assertEqual(reconstructed_time_series[3].data_indexes, {'data_loss': 0, 'data_reconstructed': 0})

        # Test series
        time_series = TimeSeries()
        time_series.append(DataTimePoint(t=-3,  data={'value':3}))
        time_series.append(DataTimePoint(t=-2,  data={'value':2}))
        time_series.append(DataTimePoint(t=-1,  data={'value':1}))
        time_series.append(DataTimePoint(t=0,  data={'value':0}))
        time_series.append(DataTimePoint(t=1,  data={'value':1}))
        time_series.append(DataTimePoint(t=2,  data={'value':2}))
        time_series.append(DataTimePoint(t=3,  data={'value':3}))
        time_series.append(DataTimePoint(t=4,  data={'value':4}))
        time_series.append(DataTimePoint(t=5,  data={'value':5}))
        time_series.append(DataTimePoint(t=6,  data={'value':6}))
        time_series.append(DataTimePoint(t=7,  data={'value':7}))
        time_series.append(DataTimePoint(t=16, data={'value':16}))
        time_series.append(DataTimePoint(t=17, data={'value':17}))
        time_series.append(DataTimePoint(t=18, data={'value':18}))

        # Resample for 3 seconds
        resampled_time_series = time_series.resample(3)
        # DataTimePoint @ 0.0 (1970-01-01 00:00:00+00:00) with data "{'value': 0.6666666666666666}" and data_loss="0.0"
        # DataTimePoint @ 3.0 (1970-01-01 00:00:03+00:00) with data "{'value': 3.0}" and data_loss="0.0"
        # DataTimePoint @ 6.0 (1970-01-01 00:00:06+00:00) with data "{'value': 6.0}" and data_loss="0.0"
        # DataTimePoint @ 9.0 (1970-01-01 00:00:09+00:00) with data "{'value': 9.0}" and data_loss="1.0"
        # DataTimePoint @ 12.0 (1970-01-01 00:00:12+00:00) with data "{'value': 12.0}" and data_loss="1.0"
        # DataTimePoint @ 15.0 (1970-01-01 00:00:15+00:00) with data "{'value': 15.0}" and data_loss="0.6666666666666667"



        # Fake data to check the recostructor is working correctly
        resampled_time_series[3].data['value']=-9
        resampled_time_series[4].data['value']=-12

        # Instantiate the reconstructor
        reconstructor = LinearInterpolationReconstructor()

        # Check parametric param just in case
        self.assertFalse(reconstructor.is_parametric())

        # Apply the reconstructor
        reconstructed_time_series = reconstructor.apply(resampled_time_series)

        # Check len
        self.assertEqual(len(reconstructed_time_series), 6)

        # Check for t = 0
        self.assertEqual(reconstructed_time_series[0].t, 0)
        self.assertAlmostEqual(reconstructed_time_series[0].data['value'], 0.6666666666)  # Expected: 0.66..
        self.assertEqual(reconstructed_time_series[0].data_loss, 0)

        # Check for t = 3
        self.assertEqual(reconstructed_time_series[1].t, 3)
        self.assertEqual(reconstructed_time_series[1].data['value'], 3)  # Expected: 3
        self.assertEqual(reconstructed_time_series[1].data_loss, 0)

        # Check for t = 6
        self.assertEqual(reconstructed_time_series[2].t, 6)
        self.assertEqual(reconstructed_time_series[2].data['value'], 6)  # Expected: 6
        self.assertEqual(reconstructed_time_series[2].data_loss, 0)

        # Check for t = 9
        self.assertEqual(reconstructed_time_series[3].t, 9)
        self.assertEqual(reconstructed_time_series[3].data['value'], 9)  # Expected: 9 (fully reconstructed)
        self.assertEqual(reconstructed_time_series[3].data_loss, 1)

        # Check for t = 12
        self.assertEqual(reconstructed_time_series[4].t, 12)
        self.assertEqual(reconstructed_time_series[4].data['value'], 12)  # Expected: 12 (fully reconstructed)
        self.assertEqual(reconstructed_time_series[4].data_loss, 1)

        # Check for t = 15
        self.assertEqual(reconstructed_time_series[5].t, 15)
        self.assertEqual(reconstructed_time_series[5].data['value'], 15)  # Expected: 15
        self.assertAlmostEqual(reconstructed_time_series[5].data_loss, 0.6666666666)

