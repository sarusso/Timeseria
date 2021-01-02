import unittest
import os
import tempfile
from math import sin
from ..datastructures import DataTimeSlotSeries, DataTimeSlot, TimePoint, DataTimePoint
from ..models import Model, ParametricModel
from ..models import PeriodicAverageReconstructor, PeriodicAverageForecaster, ProphetForecaster, ProphetReconstructor
from ..exceptions import NotFittedError
from ..storages import CSVFileStorage
from ..transformations import Slotter, Resampler
from ..time import dt

# Setup logging
import logging
logging.basicConfig(level=os.environ.get('LOGLEVEL') if os.environ.get('LOGLEVEL') else 'CRITICAL')

TEST_DATA_PATH = '/'.join(os.path.realpath(__file__).split('/')[0:-1]) + '/test_data/'

TEMP_MODELS_DIR = tempfile.TemporaryDirectory().name

class TestBaseModelClasses(unittest.TestCase):

    def test_Model(self):
        model = Model()
        
    def test_ParametricModel(self):
        
        
        # Define a trainable model mock
        class ParametricModelMock(ParametricModel):            
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



class TestReconstructors(unittest.TestCase):

    def test_PeriodicAverageReconstructor(self):
        
        # Prepare test data        
        #data_time_point_series = CSVFileStorage(TEST_DATA_PATH + '/csv/temperature.csv').get()
        #data_time_slot_series = Slotter('3600s').process(data_time_point_series)
        #for item in data_time_slot_series:
        #    print('{},{},{}'.format(item.start.t, item.data['temperature'], item.coverage))
        
        # Get test data 
        with open(TEST_DATA_PATH + '/csv/temp_slots_1h.csv') as f:
            data=f.read()
        
        data_time_slot_series = DataTimeSlotSeries()
        for line in data.split('\n'):
            if line:
                start_t = float(line.split(',')[0])
                start_point = TimePoint(t=start_t)
                end_point = TimePoint(t=start_t+3600)
                coverage = float(line.split(',')[2])
                value =  float(line.split(',')[1])
                data_time_slot_series.append(DataTimeSlot(start=start_point, end=end_point, data={'temperature':value}, coverage=coverage))
            
        # Instantiate
        periodic_average_reconstructor = PeriodicAverageReconstructor()

        # Fit
        periodic_average_reconstructor.fit(data_time_slot_series)
        
        # Evaluate
        evaluation = periodic_average_reconstructor.evaluate(data_time_slot_series, samples=100, details=True)
        self.assertAlmostEqual(evaluation['RMSE_1_steps'], 0.138541066038798)
        self.assertAlmostEqual(evaluation['MAE_1_steps'], 0.10236999999999981)
        self.assertAlmostEqual(evaluation['RMSE_24_steps'], 0.5635685435140884)
        self.assertAlmostEqual(evaluation['MAE_24_steps'], 0.4779002102269731)
        self.assertAlmostEqual(evaluation['RMSE'], 0.3510548047764432)
        self.assertAlmostEqual(evaluation['MAE'], 0.29013510511348645)

        # Evaluatevaluate on specific steps:
        evaluation = periodic_average_reconstructor.evaluate(data_time_slot_series, steps=[1,3], samples=100, details=True)
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
        evaluation = periodic_average_reconstructor.evaluate(data_time_slot_series, steps=[1,3], samples=100, details=True)
        self.assertAlmostEqual(evaluation['RMSE_3_steps'], 0.28011618729276533)

        # Fit to/from
        periodic_average_reconstructor = PeriodicAverageReconstructor()
        periodic_average_reconstructor.fit(data_time_slot_series, to_dt=dt(2019,3,1), from_dt=dt(2019,4,1))
        evaluation = periodic_average_reconstructor.evaluate(data_time_slot_series, steps=[1,3], samples=100, details=True)
        self.assertAlmostEqual(evaluation['RMSE_3_steps'], 0.23978759586375123)
        
        # Cross validations
        periodic_average_reconstructor = PeriodicAverageReconstructor()
        cross_validation = periodic_average_reconstructor.cross_validate(data_time_slot_series, evaluate_steps=[1,3], evaluate_samples=100, evaluate_details=True)
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
        

    def test_ProphetReconstructor(self):
        
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
        evaluation = forecaster.evaluate(self.sine_data_time_slot_series_minute, steps='auto', samples=100, details=True)
        self.assertEqual(forecaster.data['periodicity'], 63)
        self.assertAlmostEqual(evaluation['RMSE_1_steps'], 0.07318673229600292)
        self.assertAlmostEqual(evaluation['MAE_1_steps'], 0.06622794526818285)
        self.assertAlmostEqual(evaluation['RMSE_63_steps'], 0.06697755802373265)
        self.assertAlmostEqual(evaluation['MAE_63_steps'], 0.06016205183857482)     
        self.assertAlmostEqual(evaluation['RMSE'], 0.07008214515986778)
        self.assertAlmostEqual(evaluation['MAE'], 0.06319499855337883)

        # Evaluate
        evaluation = forecaster.evaluate(self.sine_data_time_slot_series_minute, steps=[1,3], samples=100, details=True)
        self.assertEqual(forecaster.data['periodicity'], 63)
        self.assertAlmostEqual(evaluation['RMSE_1_steps'], 0.07318673229600292)
        self.assertAlmostEqual(evaluation['MAE_1_steps'], 0.06622794526818285)
        self.assertAlmostEqual(evaluation['RMSE_3_steps'], 0.07253018513852955)
        self.assertAlmostEqual(evaluation['MAE_3_steps'], 0.06567523200748912)     

        # Fit from/to
        forecaster.fit(self.sine_data_time_slot_series_minute, from_t=20000, to_t=40000)
        evaluation = forecaster.evaluate(self.sine_data_time_slot_series_minute, steps=[1,3], samples=100, details=True)
        self.assertAlmostEqual(evaluation['RMSE_1_steps'], 0.37831442005531923)

        # Fit to/from
        forecaster.fit(self.sine_data_time_slot_series_minute, to_t=20000, from_t=40000)
        evaluation = forecaster.evaluate(self.sine_data_time_slot_series_minute, steps=[1,3], samples=100, details=True)
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
        


    def test_ProphetForecaster(self):
         
        forecaster = ProphetForecaster()
         
        forecaster.fit(self.sine_data_time_slot_series_day)
        self.assertEqual(len(self.sine_data_time_slot_series_day), 1000)
  
        sine_data_time_slot_series_day_with_forecast = forecaster.apply(self.sine_data_time_slot_series_day, n=3)
        self.assertEqual(len(sine_data_time_slot_series_day_with_forecast), 1003)


        # Test on Points as well
        data_time_point_series = CSVFileStorage(TEST_DATA_PATH + '/csv/temperature.csv').get(limit=200)
        forecaster = ProphetForecaster()
        with self.assertRaises(Exception):
            forecaster.fit(data_time_point_series)
          
        data_time_point_series = Resampler(600).process(data_time_point_series)
        forecaster.fit(data_time_point_series)
          
        # TODO: do some actual testing.. not only that "it works"
        forecasted_data_time_point_series  = forecaster.apply(data_time_point_series)






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

        from ..models import PeriodicAverageAnomalyDetector
        
        anomaly_detector = PeriodicAverageAnomalyDetector()
        
        anomaly_detector.fit(self.sine_data_time_slot_series_minute, periodicity=63)
        
        self.assertAlmostEqual(anomaly_detector.AE_threshold, 0.5914733390853167)
        
        result_time_series = anomaly_detector.apply(self.sine_data_time_slot_series_minute)

        # Count how many anomalies were detected
        anomalies_count = 0
        for slot in result_time_series:
            if slot.data['anomaly']:
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


    



