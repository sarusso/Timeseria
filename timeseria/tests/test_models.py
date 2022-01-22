import unittest
import os
import tempfile
from math import sin, cos
from ..datastructures import DataTimeSlotSeries, DataTimeSlot, TimePoint, DataTimePoint, DataTimePointSeries
from ..models import Model, ParametricModel, TimeSeriesParametricModel, KerasModel
from ..models import PeriodicAverageReconstructor, PeriodicAverageForecaster, PeriodicAverageAnomalyDetector
from ..models import ProphetForecaster, ProphetReconstructor
from ..models import ARIMAForecaster, AARIMAForecaster
from ..models import LSTMForecaster
from ..exceptions import NotFittedError, NonContiguityError
from ..storages import CSVFileStorage
from ..transformations import Slotter, Resampler
from ..time import dt

# Setup logging
import logging
logging.basicConfig(level=os.environ.get('LOGLEVEL') if os.environ.get('LOGLEVEL') else 'CRITICAL')

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
        model = Model()


    def test_ParametricModel(self):
        # TODO: decouple from the TimeSeriesParametricModel test below
        parametric_model = ParametricModel()
        
  
    def test_TimeSeriesParametricModel(self):
        
        # Define a trainable model mock
        class ParametricModelMock(TimeSeriesParametricModel):            
            def _fit(self, *args, **kwargs):
                pass 
            def _evaluate(self, *args, **kwargs):
                pass
            def _apply(self, *args, **kwargs):
                pass 

        # Define test time series
        empty_data_time_slot_series = DataTimeSlotSeries()
        data_time_slot_series = DataTimeSlotSeries(DataTimeSlot(start=TimePoint(t=1), end=TimePoint(t=2), data={'metric1': 56}),
                                                   DataTimeSlot(start=TimePoint(t=2), end=TimePoint(t=3), data={'metric1': 56}))

        # Instantiate a parametric model
        parametric_model = ParametricModelMock()
        parametric_model_id = parametric_model.id
        
        # Cannot apply model before fitting
        with self.assertRaises(NotFittedError):
            parametric_model.apply(data_time_slot_series)   

        # Cannot save model before fitting
        with self.assertRaises(NotFittedError):
            parametric_model.save(TEMP_MODELS_DIR)  
        
        # Call the mock train
        with self.assertRaises(TypeError):
            parametric_model.fit('hello')

        with self.assertRaises(ValueError):
            parametric_model.fit(empty_data_time_slot_series)
                         
        parametric_model.fit(data_time_slot_series)
        
        # Call the mock apply
        with self.assertRaises(TypeError):
            parametric_model.apply('hello')

        with self.assertRaises(ValueError):
            parametric_model.apply(empty_data_time_slot_series)
                         
        parametric_model.apply(data_time_slot_series)
        
        # And save
        model_dir = parametric_model.save(TEMP_MODELS_DIR)
        
        # Now re-load
        loaded_parametric_model = ParametricModelMock(model_dir)
        self.assertEqual(loaded_parametric_model.id, parametric_model_id)


    def test_KerasModel(self):
        
        # Define a trainable model mock
        class KerasModelMock(KerasModel):            
            def _fit(self, *args, **kwargs):
                pass 
            def _evaluate(self, *args, **kwargs):
                pass
            def _apply(self, *args, **kwargs):
                pass 

        # Define test time series
        data_time_point_series = DataTimePointSeries(DataTimePoint(t=1, data={'metric1': 0.1}),
                                                     DataTimePoint(t=2, data={'metric1': 0.2}),
                                                     DataTimePoint(t=3, data={'metric1': 0.3}),
                                                     DataTimePoint(t=4, data={'metric1': 0.4}),
                                                     DataTimePoint(t=5, data={'metric1': 0.5}),
                                                     DataTimePoint(t=6, data={'metric1': 0.6}),)


        # Test window generation functions
        window_matrix = KerasModelMock.to_window_datapoints_matrix(data_time_point_series, window=2, forecast_n=1, encoder=None)
        
        # What to expect (using the timestamp to represent a datapoint):
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
        
        target_vector = KerasModelMock.to_target_values_vector(data_time_point_series, window=2, forecast_n=1)

        # What to expect (using the data value to represent a datapoint):
        # [0.3], [0.4], [0.5], [0.6] Note that they are lists in order to support multi-step forecast
        self.assertEqual(target_vector, [[0.3], [0.4], [0.5], [0.6]])
        
        
        
        


class TestReconstructors(unittest.TestCase):

    def test_PeriodicAverageReconstructor(self):
        
        # Prepare test data        
        #data_time_point_series = CSVFileStorage(TEST_DATA_PATH + '/csv/temperature.csv').get()
        #data_time_slot_series = Slotter('3600s').process(data_time_point_series)
        #for item in data_time_slot_series:
        #    print('{},{},{}'.format(item.start.t, item.data['temperature'], item.data_loss))
        
        # Get test data 
        with open(TEST_DATA_PATH + '/csv/temp_slots_1h.csv') as f:
            data=f.read()
        
        data_time_slot_series = DataTimeSlotSeries()
        for line in data.split('\n'):
            if line:
                start_t = float(line.split(',')[0])
                start_point = TimePoint(t=start_t)
                end_point = TimePoint(t=start_t+3600)
                data_loss = 1-float(line.split(',')[2]) # from coverage to data loss
                value =  float(line.split(',')[1])
                data_time_slot_series.append(DataTimeSlot(start=start_point, end=end_point, data={'temperature':value}, data_loss=data_loss))
            
        # Instantiate
        periodic_average_reconstructor = PeriodicAverageReconstructor()

        # Fit
        periodic_average_reconstructor.fit(data_time_slot_series)
        
        # Evaluate
        evaluation = periodic_average_reconstructor.evaluate(data_time_slot_series, limit=100, details=True)
        self.assertAlmostEqual(evaluation['RMSE_1_steps'], 0.138541066038798)
        self.assertAlmostEqual(evaluation['MAE_1_steps'], 0.10236999999999981)
        self.assertAlmostEqual(evaluation['RMSE_24_steps'], 0.5635685435140884)
        self.assertAlmostEqual(evaluation['MAE_24_steps'], 0.4779002102269731)
        self.assertAlmostEqual(evaluation['RMSE'], 0.3510548047764432)
        self.assertAlmostEqual(evaluation['MAE'], 0.29013510511348645)

        # Evaluatevaluate on specific steps:
        evaluation = periodic_average_reconstructor.evaluate(data_time_slot_series, steps=[1,3], limit=100, details=True)
        self.assertAlmostEqual(evaluation['RMSE_1_steps'], 0.138541066038798)
        self.assertAlmostEqual(evaluation['MAE_1_steps'], 0.10236999999999981)
        self.assertAlmostEqual(evaluation['RMSE_3_steps'], 0.24455061001646802)
        self.assertAlmostEqual(evaluation['MAE_3_steps'], 0.190163072234738)
        self.assertAlmostEqual(evaluation['RMSE'], 0.19154583802763303)
        self.assertAlmostEqual(evaluation['MAE'], 0.1462665361173689)

        # Apply
        data_time_slot_series_reconstructed = periodic_average_reconstructor.apply(data_time_slot_series, data_loss_threshold=0.3)
        self.assertEqual(len(data_time_slot_series), len(data_time_slot_series_reconstructed))
        for i in range(len(data_time_slot_series)):
            if data_time_slot_series[i].data_loss >= 0.3:
                if (data_time_slot_series[i-1].data_loss < 0.3) and (data_time_slot_series[i+1].data_loss < 0.3):
                    # You need at least two "missing" slots in succession for the reconstructor to kick in. 
                    pass
                else:
                    self.assertNotEqual(data_time_slot_series_reconstructed[i].data, data_time_slot_series[i].data, 'at position {}'.format(i))
        
        # Fit from/to
        periodic_average_reconstructor = PeriodicAverageReconstructor()
        periodic_average_reconstructor.fit(data_time_slot_series, from_dt=dt(2019,3,1), to_dt=dt(2019,4,1))
        evaluation = periodic_average_reconstructor.evaluate(data_time_slot_series, steps=[1,3], limit=100, details=True)
        self.assertAlmostEqual(evaluation['RMSE_3_steps'], 0.28011618729276533)

        # Fit to/from
        periodic_average_reconstructor = PeriodicAverageReconstructor()
        periodic_average_reconstructor.fit(data_time_slot_series, to_dt=dt(2019,3,1), from_dt=dt(2019,4,1))
        evaluation = periodic_average_reconstructor.evaluate(data_time_slot_series, steps=[1,3], limit=100, details=True)
        self.assertAlmostEqual(evaluation['RMSE_3_steps'], 0.23978759586375123)
        
        # Cross validations
        periodic_average_reconstructor = PeriodicAverageReconstructor()
        cross_validation = periodic_average_reconstructor.cross_validate(data_time_slot_series, evaluate_steps=[1,3], evaluate_limit=100, evaluate_details=True)
        self.assertAlmostEqual(cross_validation['MAE_3_steps_avg'],  0.24029106113368606)
        self.assertAlmostEqual(cross_validation['MAE_3_steps_stdev'], 0.047895485925726636)

        # Test on Points as well
        data_time_point_series = CSVFileStorage(TEST_DATA_PATH + '/csv/temperature.csv').get(limit=200)
        periodic_average_reconstructor = PeriodicAverageReconstructor()
        with self.assertRaises(Exception):
            periodic_average_reconstructor.fit(data_time_point_series)
         
        data_time_point_series = Resampler(600).process(data_time_point_series)
        periodic_average_reconstructor.fit(data_time_point_series)
         
        # TODO: do some actual testing.. not only that "it works"
        reconstructed_data_time_point_series  = periodic_average_reconstructor.apply(data_time_point_series)
        periodic_average_reconstructor.evaluate(data_time_point_series)
        

    def test_ProphetReconstructor(self):
        try:
            import fbprophet
        except ImportError:
            print('Skipping Prophet tests as no fbprophet module installed')
            return
            
        # Get test data        
        data_time_point_series = CSVFileStorage(TEST_DATA_PATH + '/csv/temperature.csv').get(limit=200)
        data_time_slot_series = Slotter('3600s').process(data_time_point_series)
        
        # Instantiate
        prophet_reconstructor = ProphetReconstructor()

        # Fit
        prophet_reconstructor.fit(data_time_slot_series)

        # Apply
        data_time_slot_series_reconstructed = prophet_reconstructor.apply(data_time_slot_series, data_loss_threshold=0.3)
        self.assertEqual(len(data_time_slot_series), len(data_time_slot_series_reconstructed))
        for i in range(len(data_time_slot_series)):
            if data_time_slot_series[i].data_loss >= 0.3:
                if (data_time_slot_series[i-1].data_loss < 0.3) and (data_time_slot_series[i+1].data_loss < 0.3):
                    # You need at least two "missing" slots in succession for the reconstructor to kick in. 
                    pass
                else:
                    self.assertNotEqual(data_time_slot_series_reconstructed[i].data, data_time_slot_series[i].data, 'at position {}'.format(i))



class TestForecasters(unittest.TestCase):

    def setUp(self):
        
        # Create a minute-resolution test DataTimeSlotSeries
        self.sine_data_time_slot_series_minute = DataTimeSlotSeries()
        for i in range(1000):
            self.sine_data_time_slot_series_minute.append(DataTimeSlot(start=TimePoint(i*60), end=TimePoint((i+1)*60), data={'value':sin(i/10.0)}))

        # Create a day-resolution test DataTimeSlotSeries
        self.sine_data_time_slot_series_day = DataTimeSlotSeries()
        for i in range(1000):
            step = 60 * 60 * 24
            self.sine_data_time_slot_series_day.append(DataTimeSlot(start=TimePoint(i*step), end=TimePoint((i+1)*step), data={'value':sin(i/10.0)}))
    
    def test_PeriodicAverageForecaster(self):
                 
        forecaster = PeriodicAverageForecaster()
        
        # Fit
        forecaster.fit(self.sine_data_time_slot_series_minute, periodicity=63)

        # Apply
        sine_data_time_slot_series_minute_with_forecast = forecaster.apply(self.sine_data_time_slot_series_minute, n=3)
        self.assertEqual(len(sine_data_time_slot_series_minute_with_forecast), 1003)

        # Predict
        prediction = forecaster.predict(self.sine_data_time_slot_series_minute, n=3)
        self.assertTrue(isinstance(prediction, list))
        self.assertEqual(len(prediction), 3)

        # Evaluate
        evaluation = forecaster.evaluate(self.sine_data_time_slot_series_minute, steps='auto', limit=100, details=True)
        self.assertEqual(forecaster.data['periodicity'], 63)
        self.assertAlmostEqual(evaluation['RMSE_1_steps'], 0.07318673229600292)
        self.assertAlmostEqual(evaluation['MAE_1_steps'], 0.06622794526818285)
        self.assertAlmostEqual(evaluation['RMSE_63_steps'], 0.06697755802373265)
        self.assertAlmostEqual(evaluation['MAE_63_steps'], 0.06016205183857482)     
        self.assertAlmostEqual(evaluation['RMSE'], 0.07008214515986778)
        self.assertAlmostEqual(evaluation['MAE'], 0.06319499855337883)

        # Evaluate
        evaluation = forecaster.evaluate(self.sine_data_time_slot_series_minute, steps=[1,3], limit=100, details=True)
        self.assertEqual(forecaster.data['periodicity'], 63)
        self.assertAlmostEqual(evaluation['RMSE_1_steps'], 0.07318673229600292)
        self.assertAlmostEqual(evaluation['MAE_1_steps'], 0.06622794526818285)
        self.assertAlmostEqual(evaluation['RMSE_3_steps'], 0.07253018513852955)
        self.assertAlmostEqual(evaluation['MAE_3_steps'], 0.06567523200748912)     

        # Fit from/to
        forecaster.fit(self.sine_data_time_slot_series_minute, from_t=20000, to_t=40000)
        evaluation = forecaster.evaluate(self.sine_data_time_slot_series_minute, steps=[1,3], limit=100, details=True)
        self.assertAlmostEqual(evaluation['RMSE_1_steps'], 0.37831442005531923)

        # Fit to/from
        forecaster.fit(self.sine_data_time_slot_series_minute, to_t=20000, from_t=40000)
        evaluation = forecaster.evaluate(self.sine_data_time_slot_series_minute, steps=[1,3], limit=100, details=True)
        self.assertAlmostEqual(evaluation['RMSE_1_steps'], 0.36033834603736264)

        # Test on Points as well
        data_time_point_series = CSVFileStorage(TEST_DATA_PATH + '/csv/temperature.csv').get(limit=200)
        forecaster = PeriodicAverageForecaster()
        with self.assertRaises(Exception):
            forecaster.fit(data_time_point_series)
          
        data_time_point_series = Resampler(600).process(data_time_point_series)
        forecaster.fit(data_time_point_series)
          
        # TODO: do some actual testing.. not only that "it works"
        forecasted_data_time_point_series  = forecaster.apply(data_time_point_series)


    def test_PeriodicAverageForecaster_save_load(self):
        
        forecaster = PeriodicAverageForecaster()
        
        forecaster.fit(self.sine_data_time_slot_series_minute, periodicity=63)
        
        model_dir = forecaster.save(TEMP_MODELS_DIR)
        
        loaded_forecaster = PeriodicAverageForecaster(model_dir)
        
        self.assertEqual(forecaster.data['averages'], loaded_forecaster.data['averages'])

        forecasted_data_time_point_series  = loaded_forecaster.apply(self.sine_data_time_slot_series_minute)



    def test_ProphetForecaster(self):

        try:
            import fbprophet
        except ImportError:
            print('Skipping Prophet tests as no fbprophet module installed')
            return
         
        forecaster = ProphetForecaster()
         
        forecaster.fit(self.sine_data_time_slot_series_day)
        self.assertEqual(len(self.sine_data_time_slot_series_day), 1000)
  
        sine_data_time_slot_series_day_with_forecast = forecaster.apply(self.sine_data_time_slot_series_day, n=3)
        self.assertEqual(len(sine_data_time_slot_series_day_with_forecast), 1003)

        # Test the evaluate
        evalation_results = forecaster.evaluate(self.sine_data_time_slot_series_day, limit=10)
        self.assertAlmostEqual(evalation_results['RMSE'], 0.8211270888684844)
        self.assertAlmostEqual(evalation_results['MAE'], 0.809400693526047)

        evalation_results = forecaster.evaluate(self.sine_data_time_slot_series_day, limit=1)
        self.assertAlmostEqual(evalation_results['RMSE'], 0.5390915558518541) # For one sample they must be the same
        self.assertAlmostEqual(evalation_results['MAE'], 0.5390915558518541) # For one sample they must be the same
        
        # Test on Points as well
        data_time_point_series = CSVFileStorage(TEST_DATA_PATH + '/csv/temperature.csv').get(limit=200)
        forecaster = ProphetForecaster()
        with self.assertRaises(Exception):
            forecaster.fit(data_time_point_series)
           
        data_time_point_series = Resampler(600).process(data_time_point_series)
        forecaster.fit(data_time_point_series)
           
        # TODO: do some actual testing.. not only that "it works"
        forecasted_data_time_point_series  = forecaster.apply(data_time_point_series)


    def test_ARIMAForecaster(self):

        try:
            import statsmodels
        except ImportError:
            print('Skipping ARIMA tests as no statsmodels module installed')
            return
         
        # Basic ARIMA 
        forecaster = ARIMAForecaster(p=1,d=1,q=0)
         
        forecaster.fit(self.sine_data_time_slot_series_day)
        self.assertEqual(len(self.sine_data_time_slot_series_day), 1000)

        # Cannot apply on a time series contiguous with the time series used for the fit
        with self.assertRaises(NonContiguityError):
            forecaster.apply(self.sine_data_time_slot_series_day[:-1], n=3)
    
        # Can apply on a time series contiguous with the fit one
        sine_data_time_slot_series_day_with_forecast = forecaster.apply(self.sine_data_time_slot_series_day, n=3)
        self.assertEqual(len(sine_data_time_slot_series_day_with_forecast), 1003)

        # Cannot evaluate on a time series not contiguous with the time series used for the fit
        with self.assertRaises(NonContiguityError):
            forecaster.evaluate(self.sine_data_time_slot_series_day)

        # Can evaluate on a time series contiguous with the time series used for the fit
        forecaster = ARIMAForecaster(p=1,d=1,q=0)
        forecaster.fit(self.sine_data_time_slot_series_day[0:800])                 
        evaluation_results = forecaster.evaluate(self.sine_data_time_slot_series_day[800:1000])
        self.assertAlmostEqual(evaluation_results['RMSE'], 2.71, places=2)
        self.assertAlmostEqual(evaluation_results['MAE'], 2.52, places=2 )
 
        # Test on Points as well
        data_time_point_series = CSVFileStorage(TEST_DATA_PATH + '/csv/temperature.csv').get(limit=200)
        forecaster = ARIMAForecaster()
        with self.assertRaises(Exception):
            forecaster.fit(data_time_point_series)
           
        data_time_point_series = Resampler(600).process(data_time_point_series)
        forecaster.fit(data_time_point_series)
           
        # TODO: do some actual testing.. not only that "it works"
        forecasted_data_time_point_series  = forecaster.apply(data_time_point_series)


    def test_AARIMAForecaster(self):

        try:
            import statsmodels
        except ImportError:
            print('Skipping AARIMA tests as no statsmodels module installed')
            return
         
        # Automatic ARIMA 
        forecaster = AARIMAForecaster()
         
        forecaster.fit(self.sine_data_time_slot_series_day, max_p=2, max_d=1, max_q=2)
        self.assertEqual(len(self.sine_data_time_slot_series_day), 1000)

        # Cannot apply on a time series contiguous with the time series used for the fit
        with self.assertRaises(NonContiguityError):
            forecaster.apply(self.sine_data_time_slot_series_day[:-1], n=3)
    
        # Can apply on a time series contiguous with the same item as the fit one
        sine_data_time_slot_series_day_with_forecast = forecaster.apply(self.sine_data_time_slot_series_day, n=3)
        self.assertEqual(len(sine_data_time_slot_series_day_with_forecast), 1003)

        # Cannot evaluate on a time series not contiguous with the time series used for the fit
        with self.assertRaises(NonContiguityError):
            forecaster.evaluate(self.sine_data_time_slot_series_day)

        # Can evaluate on a time series contiguous with the time series used for the fit
        forecaster = AARIMAForecaster()
        forecaster.fit(self.sine_data_time_slot_series_day[0:800], max_p=2, max_d=1, max_q=2)                 
        evaluation_results = forecaster.evaluate(self.sine_data_time_slot_series_day[800:1000])
        self.assertTrue('RMSE' in evaluation_results)
        self.assertTrue('MAE' in evaluation_results)
        # Cannot test values, some random behavior which cannot be put under control is present somewhere
        #self.assertAlmostEqual(evaluation_results['RMSE'], 0.7179428895746799)
        #self.assertAlmostEqual(evaluation_results['MAE'], 0.6497934134525981)
 
        # Test on Points as well
        data_time_point_series = CSVFileStorage(TEST_DATA_PATH + '/csv/temperature.csv').get(limit=200)
        forecaster = AARIMAForecaster()
        with self.assertRaises(Exception):
            forecaster.fit(data_time_point_series)
           
        data_time_point_series = Resampler(600).process(data_time_point_series)
        forecaster.fit(data_time_point_series, max_p=2, max_d=1, max_q=2)
           
        # TODO: do some actual testing.. not only that "it works"
        forecasted_data_time_point_series  = forecaster.apply(data_time_point_series)


    def test_LSTMForecaster(self):

        try:
            import tensorflow
        except ImportError:
            print('Skipping LSTM forecaster tests as no tensorflow module installed')
            return

        # Create a minute-resolution test DataTimeSlotSeries
        sine_data_time_slot_series_minute = DataTimeSlotSeries()
        for i in range(10):
            sine_data_time_slot_series_minute.append(DataTimeSlot(start=TimePoint(i*60), end=TimePoint((i+1)*60), data={'value':sin(i/10.0)}))

        forecaster = LSTMForecaster()
        forecaster.fit(sine_data_time_slot_series_minute)
        predicted_value = forecaster.predict(sine_data_time_slot_series_minute)['value']
        
        # Give some tolerance
        self.assertTrue(predicted_value>0.5)
        self.assertTrue(predicted_value<1.1)
        
        # Not-existent features
        with self.assertRaises(ValueError):
            LSTMForecaster(features=['values','not_existent_feature']).fit(sine_data_time_slot_series_minute)

        # Test using another feature
        LSTMForecaster(features=['values','diffs']).fit(sine_data_time_slot_series_minute)


    def test_LSTMForecaster_multivariate(self):

        try:
            import tensorflow
        except ImportError:
            print('Skipping LSTM forecaster tests as no tensorflow module installed')
            return
        
        # Create a minute-resolution test DataTimeSlotSeries
        sine_data_time_slot_series_minute = DataTimeSlotSeries()
        for i in range(10):
            sine_data_time_slot_series_minute.append(DataTimeSlot(start=TimePoint(i*60), end=TimePoint((i+1)*60), data={'sin':sin(i/10.0), 'cos':cos(i/10.0)}))

        forecaster = LSTMForecaster()
        forecaster.fit(sine_data_time_slot_series_minute)
        predicted_data = forecaster.predict(sine_data_time_slot_series_minute)
        
        self.assertTrue('sin' in predicted_data)
        self.assertTrue('cos' in predicted_data)


    def test_LSTMForecaster_save_load(self):
        
        try:
            import tensorflow
        except ImportError:
            print('Skipping LSTM forecaster tests as no tensorflow module installed')
            return
        
        # Create a minute-resolution test DataTimeSlotSeries
        sine_data_time_slot_series_minute = DataTimeSlotSeries()
        for i in range(10):
            sine_data_time_slot_series_minute.append(DataTimeSlot(start=TimePoint(i*60), end=TimePoint((i+1)*60), data={'sin':sin(i/10.0), 'cos':cos(i/10.0)}))

        forecaster = LSTMForecaster()
        forecaster.fit(sine_data_time_slot_series_minute)
        
        # Save
        model_dir = forecaster.save(TEMP_MODELS_DIR)

        # Load
        loaded_forecaster = LSTMForecaster(path=model_dir)
        
        # Predict from the loaded model 
        predicted_data = loaded_forecaster.predict(sine_data_time_slot_series_minute)
        
        self.assertTrue('sin' in predicted_data)
        self.assertTrue('cos' in predicted_data)   
        

class TestAnomalyDetectors(unittest.TestCase):

    def setUp(self):
        
        # Create a minute-resolution test DataTimeSlotSeries
        self.sine_data_time_slot_series_minute = DataTimeSlotSeries()
        for i in range(1000):
            if i % 100 == 0:
                value = 2
            else:
                value = sin(i/10.0)

            self.sine_data_time_slot_series_minute.append(DataTimeSlot(start=TimePoint(i*60), end=TimePoint((i+1)*60), data={'value':value}))


    def test_PeriodicAverageAnomalyDetector(self):
        
        anomaly_detector = PeriodicAverageAnomalyDetector()
        
        anomaly_detector.fit(self.sine_data_time_slot_series_minute, periodicity=63)
        
        self.assertAlmostEqual(anomaly_detector.data['AE_threshold'], 0.5914733390853167)
        
        result_time_series = anomaly_detector.apply(self.sine_data_time_slot_series_minute)

        # Count how many anomalies were detected
        anomalies_count = 0
        for slot in result_time_series:
            if slot.anomaly:
                anomalies_count += 1
        self.assertEqual(anomalies_count, 9)


        # Test on Points as well
        data_time_point_series = CSVFileStorage(TEST_DATA_PATH + '/csv/temperature.csv').get(limit=200)
        anomaly_detector = PeriodicAverageAnomalyDetector()
        with self.assertRaises(Exception):
            anomaly_detector.fit(data_time_point_series)
          
        data_time_point_series = Resampler(600).process(data_time_point_series)
        anomaly_detector.fit(data_time_point_series)
          
        # TODO: do some actual testing.. not only that "it works"
        anomaly_data_time_point_series  = anomaly_detector.apply(data_time_point_series)


    def test_PeriodicAverageAnomalyDetector_save_load(self):
        
        anomaly_detector = PeriodicAverageAnomalyDetector()
        
        anomaly_detector.fit(self.sine_data_time_slot_series_minute, periodicity=63)
        
        model_dir = anomaly_detector.save(TEMP_MODELS_DIR)
        
        loaded_anomaly_detector = PeriodicAverageAnomalyDetector(model_dir)
        
        self.assertEqual(anomaly_detector.data['AE_threshold'], loaded_anomaly_detector.data['AE_threshold'])
        self.assertEqual(anomaly_detector.forecaster.data['averages'], loaded_anomaly_detector.forecaster.data['averages'])

        result_time_series = loaded_anomaly_detector.apply(self.sine_data_time_slot_series_minute)

