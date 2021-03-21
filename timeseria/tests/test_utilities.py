import unittest
import os
from ..utilities import detect_encoding, compute_coverage, compute_data_loss, get_periodicity, detect_sampling_interval
from ..datastructures import DataTimePointSeries, DataTimePoint
from ..time import dt, s_from_dt
from ..storages import CSVFileStorage
from ..transformations import Slotter
from ..units import TimeUnit

# Setup logging
import logging
logging.basicConfig(level=os.environ.get('LOGLEVEL') if os.environ.get('LOGLEVEL') else 'CRITICAL')

# Set test data path
TEST_DATA_PATH = '/'.join(os.path.realpath(__file__).split('/')[0:-1]) + '/test_data/'


class TestDetectEncoding(unittest.TestCase):

    def test_detect_encoding(self):
        
        encoding = detect_encoding('{}/csv/shampoo_sales.csv'.format(TEST_DATA_PATH), streaming=False)
        self.assertEqual(encoding, 'ascii')
        
     

class TestComputeCoverage(unittest.TestCase):

    def setUp(self):       
        
        # All the following time series have point with validity=1m
        
        # Time series from 16:58:00 to 17:32:00 (Europe/Rome)
        self.data_time_point_series_1 = DataTimePointSeries()
        start_t = 1436022000 - 120
        for i in range(35):
            data_time_point = DataTimePoint(t = start_t + (i*60),
                                            data = {'temperature': 154+i})
            self.data_time_point_series_1.append(data_time_point)
        
        # Time series from 17:00:00 to 17:30:00 (Europe/Rome)
        self.data_time_point_series_2 = DataTimePointSeries()
        start_t = 1436022000
        for i in range(34):
            data_time_point = DataTimePoint(t    = start_t + (i*60),
                                            data = {'temperature': 154+i})
            self.data_time_point_series_2.append(data_time_point)

        # Time series from 17:00:00 to 17:20:00 (Europe/Rome)
        self.data_time_point_series_3 = DataTimePointSeries()
        start_t = 1436022000 - 120
        for i in range(23):
            data_time_point = DataTimePoint(t    = start_t + (i*60),
                                            data = {'temperature': 154+i})
            self.data_time_point_series_3.append(data_time_point) 
        
        # Time series from 17:10:00 to 17:30:00 (Europe/Rome)
        self.data_time_point_series_4 = DataTimePointSeries()
        start_t = 1436022000 + 600
        for i in range(21):
            data_time_point = DataTimePoint(t    = start_t + (i*60),
                                            data = {'temperature': 154+i})
            self.data_time_point_series_4.append(data_time_point)

        # Time series from 16:58:00 to 17:32:00 (Europe/Rome)
        self.data_time_point_series_5 = DataTimePointSeries()
        start_t = 1436022000 - 120
        for i in range(35):
            if i > 10 and i <21:
                continue
            data_time_point = DataTimePoint(t    = start_t + (i*60),
                                            data = {'temperature': 154+i})
            self.data_time_point_series_5.append(data_time_point)

        # The following time series has point with validity=15m

        # Time series from 2019,10,1,1,0,0 to 2019,10,1,6,0,0 (Europe/Rome)
        from_dt  = dt(2019,10,1,1,0,0, tzinfo='Europe/Rome')
        to_dt    = dt(2019,10,1,6,0,0, tzinfo='Europe/Rome')
        time_unit = TimeUnit('15m') 
        self.data_time_point_series_6 = DataTimePointSeries()
        slider_dt = from_dt
        count = 0
        while slider_dt < to_dt:
            if count not in [1, 6, 7, 8, 9, 10]:
                data_time_point = DataTimePoint(t    = s_from_dt(slider_dt),
                                                data = {'temperature': 154+count})
                self.data_time_point_series_6.append(data_time_point)
            slider_dt = slider_dt + time_unit
            count += 1
 

    def test_compute_coverage(self):
        
        from_t = 1436022000       # 2015-07-04 17:00:00+02:00
        to_t   = 1436022000+1800  # 2015-07-04 17:30:00+02:00
        validity = 60


        # A) Full coverage (coverage=1.0) and again, to test reproducibility
        coverage = compute_coverage(data_time_point_series  = self.data_time_point_series_1,
                                    from_t = from_t, to_t = to_t, validity=validity)  
        self.assertEqual(coverage, 1.0)
        
        coverage = compute_coverage(data_time_point_series  = self.data_time_point_series_1,
                                    from_t = from_t, to_t = to_t, validity=validity)  
        self.assertEqual(coverage, 1.0)       

  
        # B) Full coverage (coverage=1.0) witjout prev/next in the timeSeries 
        coverage = compute_coverage(data_time_point_series  = self.data_time_point_series_1,
                                    from_t = from_t, to_t = to_t, validity=validity)  
        self.assertEqual(coverage, 1.0) 
        self.assertEqual(coverage, 1.0)  

 
        # C) Missing ten minutes over 30 at the end (coverage=0.683))
        coverage = compute_coverage(data_time_point_series  = self.data_time_point_series_3,
                                    from_t = from_t, to_t = to_t, validity=validity)  
        # 20 minutes plus other 30 secs validity for the 20th point over 30 minutes
        self.assertEqual(coverage, ( ((20*60.0)+30.0) / (30*60.0)) ) 
 
 
        # D) Missing ten minutes over 30 at the beginning (coverage=0.683)
        coverage = compute_coverage(data_time_point_series  = self.data_time_point_series_4,
                                    from_t = from_t, to_t = to_t, validity=validity)  
        # 20 minutes plus other 30 secs (previous) validity for the 10th point over 30 minutes
        self.assertEqual(coverage, ( ((20*60.0)+30.0) / (30*60.0)) ) 
 
 
        # E) Missing eleven minutes over 30 in the middle (coverage=0.66)
        coverage = compute_coverage(data_time_point_series  = self.data_time_point_series_5,
                                    from_t = from_t, to_t = to_t, validity=validity) 
        # 20 minutes plus other 30 secs (previous) validity for the 10th point over 30 minutes
        self.assertAlmostEqual(coverage, (2.0/3.0))
  
 
        # F) Missing half slot before slot re-start
        from_t = s_from_dt(dt=dt(2019,10,1,3,30,0, tzinfo='Europe/Rome'))
        to_t   = s_from_dt(dt=dt(2019,10,1,3,45,0, tzinfo='Europe/Rome'))
        coverage = compute_coverage(data_time_point_series  = self.data_time_point_series_6,
                                    from_t = from_t, to_t = to_t, validity=900)         
        self.assertAlmostEqual(coverage, (0.5))


    def test_compute_data_loss(self):

        # Everything is the same as the compute coverage *except* when we have data losses 
        # already present which have to be taken into account as well

        # TODO: add some better testing here...
        data_time_point_series = DataTimePointSeries()
        data_time_point_series.append(DataTimePoint(t = 20, data = {'temperature': 23}, data_loss=0))
        data_time_point_series.append(DataTimePoint(t = 30, data = {'temperature': 23}, data_loss=0))
        data_time_point_series.append(DataTimePoint(t = 40, data = {'temperature': 23}, data_loss=0.5))
        data_time_point_series.append(DataTimePoint(t = 60, data = {'temperature': 23}, data_loss=0))
        data_time_point_series.append(DataTimePoint(t = 70, data = {'temperature': 23}, data_loss=0))
        data_loss = compute_data_loss(data_time_point_series  = data_time_point_series,
                                      from_t = 30, to_t = 60, validity=10, series_resolution=data_time_point_series.resolution) 
        self.assertAlmostEqual(data_loss, 0.5)

        # Tets also from ana ctually resampled tiem series
        resampled_timeseries = self.data_time_point_series_5.resample(60)
        data_loss = compute_data_loss(data_time_point_series  = resampled_timeseries,
                                    from_t = 1436022300, to_t = 1436022600, validity=60, series_resolution=self.data_time_point_series_5) 
        self.assertAlmostEqual(data_loss, 0.4)


class TestGetPeriodicity(unittest.TestCase):

    def test_get_periodicity(self):
        
        univariate_data_time_point_series = CSVFileStorage(TEST_DATA_PATH + '/csv/temperature.csv').get()
        
        univariate_1h_data_time_slot_series = Slotter('1h').process(univariate_data_time_point_series)
        
        perdiodicity = get_periodicity(univariate_1h_data_time_slot_series)

        self.assertEqual(perdiodicity, 24)


class TestDetectSamplingInterval(unittest.TestCase):

    def test_detect_sampling_interval(self):
        
        data_time_point_series = CSVFileStorage(TEST_DATA_PATH + '/csv/humitemp_short.csv').get()
        self.assertEqual(detect_sampling_interval(data_time_point_series), 61)
        
        data_time_point_series = CSVFileStorage(TEST_DATA_PATH + '/csv/shampoo_sales.csv', time_column = 'Month', time_format = '%y-%m').get()
        self.assertEqual(detect_sampling_interval(data_time_point_series), 2678400)  # 2678400/60/60/24 = 31 Days (the most frequent)

        data_time_point_series = CSVFileStorage(TEST_DATA_PATH + '/csv/temperature.csv').get()
        self.assertEqual(detect_sampling_interval(data_time_point_series), 600)

        data_time_point_series = CSVFileStorage(TEST_DATA_PATH + '/csv/temp_short_1h.csv').get()
        self.assertEqual(detect_sampling_interval(data_time_point_series), 3600)





