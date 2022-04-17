import unittest
import os
from ..utilities import detect_encoding, get_periodicity, detect_sampling_interval
from ..utilities import compute_coverage, compute_data_loss, compute_validity_regions 

from ..datastructures import DataTimePointSeries, DataTimePoint
from ..time import dt, s_from_dt
from ..storages import CSVFileStorage
from ..transformations import Aggregator
from ..units import TimeUnit

# Setup logging
from .. import logger
logger.setup()

# Set test data path
TEST_DATA_PATH = '/'.join(os.path.realpath(__file__).split('/')[0:-1]) + '/test_data/'

# Support functions
def attach_validity_regions(series, sampling_interval=None):
    validity_regions = compute_validity_regions(series, sampling_interval=sampling_interval)
    for point in series:
        point.valid_from = validity_regions[point.t][0]
        point.valid_to = validity_regions[point.t][1]


class TestDetectEncoding(unittest.TestCase):

    def test_detect_encoding(self):
        
        encoding = detect_encoding('{}/csv/shampoo_sales.csv'.format(TEST_DATA_PATH), streaming=False)
        self.assertEqual(encoding, 'ascii')


class TestComputeValidityRegions(unittest.TestCase):
    
    def test_standard(self):
        
        series = DataTimePointSeries()
        series.append(DataTimePoint(t = 7, data = {'value': 7}))
        series.append(DataTimePoint(t = 12, data = {'value': 12}))
        series.append(DataTimePoint(t = 17, data = {'value': 17}))
        series.append(DataTimePoint(t = 22, data = {'value': 22}))
        series.append(DataTimePoint(t = 27, data = {'value': 27}))

        # All of them
        expected_results = {7: [4.5, 9.5], 12: [9.5, 14.5], 17: [14.5, 19.5], 22: [19.5, 24.5], 27: [24.5, 29.5]}
        results = compute_validity_regions(series)
        self.assertEqual(results, expected_results)
        
        # Only from-to
        expected_results = {12: [9.5, 14.5], 17: [14.5, 19.5], 22: [19.5, 24.5]}
        results = compute_validity_regions(series, from_t=10, to_t=20)
        self.assertEqual(results, expected_results)

        # Shrink them according to the from-to
        expected_results = {12: [10, 14.5], 17: [14.5, 19.5], 22: [19.5, 20]}
        results = compute_validity_regions(series, from_t=10, to_t=20, cut=True)
        self.assertEqual(results, expected_results)
        
        # Force a specific sampling intervals
        expected_results = {12: [11.5, 12.5], 17: [16.5, 17.5]}
        results = compute_validity_regions(series, from_t=10, to_t=20, sampling_interval=1)
        self.assertEqual(results, expected_results)
  
        # Single-element series
        single_element_series = DataTimePointSeries()
        single_element_series.append(DataTimePoint(t = 40, data = {'value': 4571.55}))
        expected_results = {40: [35.0, 45.0]}

        # The series has only one element and no sampling_interval is provided, no idea how to compute validity        
        with self.assertRaises(ValueError):
            compute_validity_regions(series = single_element_series, from_t = 30, to_t = 60) 

        # Call by providing the sampling_interval
        results = compute_validity_regions(series = single_element_series, from_t = 30, to_t = 60, sampling_interval=10)

        self.assertEqual(results, expected_results)


    def test_prev_next_points(self):
        series = DataTimePointSeries()
        series.append(DataTimePoint(t = 7, data = {'value': 7}))
        series.append(DataTimePoint(t = 12, data = {'value': 12}))
        series.append(DataTimePoint(t = 17, data = {'value': 17}))
        series.append(DataTimePoint(t = 22, data = {'value': 22}))
        series.append(DataTimePoint(t = 27, data = {'value': 27}))
 
        # Test prev-next
        expected_results = {7: [4.5, 9.5], 12: [9.5, 14.5], 17: [14.5, 19.5], 22: [19.5, 24.5], 27: [24.5, 29.5]}
        results = compute_validity_regions(series, from_t=9, to_t=25)
        self.assertEqual(results, expected_results)

        # Test cutted prev-next
        expected_results = {7: [9, 9.5], 12: [9.5, 14.5], 17: [14.5, 19.5], 22: [19.5, 24.5], 27: [24.5, 25]}
        results = compute_validity_regions(series, from_t=9, to_t=25, cut=True)
        self.assertEqual(results, expected_results)
 
 
    def test_overlaps(self):
         
        series = DataTimePointSeries()
        series.append(DataTimePoint(t = 7, data = {'value': 7}))
        series.append(DataTimePoint(t = 12, data = {'value': 12}))
        series.append(DataTimePoint(t = 13, data = {'value': 13}))
        series.append(DataTimePoint(t = 17, data = {'value': 17}))
        series.append(DataTimePoint(t = 22, data = {'value': 22}))
        series.append(DataTimePoint(t = 27, data = {'value': 27}))

        expected_results = {7: [4.5, 9.5], 12: [9.5, 12.5], 13: [12.5, 15], 17: [15, 19.5], 22: [19.5, 24.5], 27: [24.5, 29.5]}
        results = compute_validity_regions(series)
        self.assertEqual(results, expected_results)


    def test_major_overlaps(self):

        series = DataTimePointSeries()
        series.append(DataTimePoint(t = 7, data = {'value': 7}))
        series.append(DataTimePoint(t = 12, data = {'value': 12}))
        series.append(DataTimePoint(t = 12.1, data = {'value': 12.1}))
        series.append(DataTimePoint(t = 12.3, data = {'value': 12.3}))
        series.append(DataTimePoint(t = 13, data = {'value': 13}))
        series.append(DataTimePoint(t = 17, data = {'value': 17}))
        series.append(DataTimePoint(t = 22, data = {'value': 22}))
        series.append(DataTimePoint(t = 27, data = {'value': 27}))

        expected_results = {7: [4.5, 9.5], 12: [9.5, 12.05], 12.1: [12.05, 12.2], 12.3: [12.2, 12.65], 13: [12.65, 15.0], 17: [15.0, 19.5], 22: [19.5, 24.5], 27: [24.5, 29.5]}
        results = compute_validity_regions(series)
        self.assertEqual(results, expected_results)


class TestComputeCoverageAndDataLoss(unittest.TestCase):

    def setUp(self):       
        
        # All the following time series have point with validity=1m
        
        # Time series from 16:58:00 to 17:32:00 (Europe/Rome)
        self.data_time_point_series_1 = DataTimePointSeries()
        start_t = 1436022000 - 120
        for i in range(35):
            data_time_point = DataTimePoint(t = start_t + (i*60),
                                            data = {'temperature': 154+i})
            self.data_time_point_series_1.append(data_time_point)
        attach_validity_regions(self.data_time_point_series_1)
        
        # Time series from 17:00:00 to 17:30:00 (Europe/Rome)
        self.data_time_point_series_2 = DataTimePointSeries()
        start_t = 1436022000
        for i in range(34):
            data_time_point = DataTimePoint(t    = start_t + (i*60),
                                            data = {'temperature': 154+i})
            self.data_time_point_series_2.append(data_time_point)
        attach_validity_regions(self.data_time_point_series_2)

        # Time series from 17:00:00 to 17:20:00 (Europe/Rome)
        self.data_time_point_series_3 = DataTimePointSeries()
        start_t = 1436022000 - 120
        for i in range(23):
            data_time_point = DataTimePoint(t    = start_t + (i*60),
                                            data = {'temperature': 154+i})
            self.data_time_point_series_3.append(data_time_point) 
        attach_validity_regions(self.data_time_point_series_3)
    
        # Time series from 17:10:00 to 17:30:00 (Europe/Rome)
        self.data_time_point_series_4 = DataTimePointSeries()
        start_t = 1436022000 + 600
        for i in range(21):
            data_time_point = DataTimePoint(t    = start_t + (i*60),
                                            data = {'temperature': 154+i})
            self.data_time_point_series_4.append(data_time_point)
        attach_validity_regions(self.data_time_point_series_4)

        # Time series from 16:58:00 to 17:32:00 (Europe/Rome)
        self.data_time_point_series_5 = DataTimePointSeries()
        start_t = 1436022000 - 120
        for i in range(35):
            if i > 10 and i <21:
                continue
            data_time_point = DataTimePoint(t    = start_t + (i*60),
                                            data = {'temperature': 154+i})
            self.data_time_point_series_5.append(data_time_point)
        attach_validity_regions(self.data_time_point_series_5)


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
        attach_validity_regions(self.data_time_point_series_6)
 

    def test_compute_coverage(self):
        
        from_t = 1436022000       # 2015-07-04 17:00:00+02:00
        to_t   = 1436022000+1800  # 2015-07-04 17:30:00+02:00

        # A) Full coverage (coverage=1.0) and again, to test "reproducibility"
        coverage = compute_coverage(series = self.data_time_point_series_1, from_t = from_t, to_t = to_t)  
        self.assertEqual(coverage, 1.0)
        coverage = compute_coverage(series = self.data_time_point_series_1, from_t = from_t, to_t = to_t)  
        self.assertEqual(coverage, 1.0)       

  
        # B) Full coverage (coverage=1.0) witjout prev/next in the time series 
        # TODO: fix me
        coverage = compute_coverage(series = self.data_time_point_series_1, from_t = from_t, to_t = to_t)  
        self.assertEqual(coverage, 1.0) 
        self.assertEqual(coverage, 1.0)  

 
        # C) Missing ten minutes over 30 at the end (coverage=0.683))
        coverage = compute_coverage(series = self.data_time_point_series_3, from_t = from_t, to_t = to_t)  

        # 20 minutes plus other 30 secs validity for the 20th point over 30 minutes
        self.assertEqual(coverage, ( ((20*60.0)+30.0) / (30*60.0)) ) 
 
 
        # D) Missing ten minutes over 30 at the beginning (coverage=0.683)
        coverage = compute_coverage(series = self.data_time_point_series_4, from_t = from_t, to_t = to_t)  

        # 20 minutes plus other 30 secs (previous) validity for the 10th point over 30 minutes
        self.assertEqual(coverage, ( ((20*60.0)+30.0) / (30*60.0)) ) 
 
 
        # E) Missing eleven minutes over 30 in the middle (coverage=0.66)
        coverage = compute_coverage(series = self.data_time_point_series_5, from_t = from_t, to_t = to_t)  

        # 20 minutes plus other 30 secs (previous) validity for the 10th point over 30 minutes
        self.assertAlmostEqual(coverage, (2.0/3.0))
  
 
        # F) Missing half slot before slot re-start
        from_t = s_from_dt(dt=dt(2019,10,1,3,30,0, tzinfo='Europe/Rome'))
        to_t   = s_from_dt(dt=dt(2019,10,1,3,45,0, tzinfo='Europe/Rome'))
        coverage = compute_coverage(series = self.data_time_point_series_6, from_t = from_t, to_t = to_t)  
     
        self.assertAlmostEqual(coverage, (0.5))

        # G) Border conditions for from_t and to_t:
        data_time_point_series = DataTimePointSeries()
        data_time_point_series.append(DataTimePoint(t = 20, data = {'temperature': 23}))
        data_time_point_series.append(DataTimePoint(t = 30, data = {'temperature': 23}))
        data_time_point_series.append(DataTimePoint(t = 40, data = {'temperature': 23}))
        data_time_point_series.append(DataTimePoint(t = 60, data = {'temperature': 23}))
        data_time_point_series.append(DataTimePoint(t = 70, data = {'temperature': 23}))
        attach_validity_regions(data_time_point_series)
        
        coverage = compute_coverage(series = data_time_point_series, from_t = 30, to_t = 60) 
        self.assertAlmostEqual(coverage, (2.0/3.0))

        # H) single-element series
        single_element_series = DataTimePointSeries()
        single_element_series.append(DataTimePoint(t = 40, data = {'temperature': 23}))
        
        # TODO: the following is quite messy.
        attach_validity_regions(single_element_series, sampling_interval=10)

        #with self.assertRaises(ValueError):
        #    compute_coverage(series = single_element_series, from_t = 30, to_t = 60) 

        coverage = compute_coverage(series = single_element_series, from_t = 30, to_t = 60, sampling_interval=10)
        # 30 to 60, point is from 35 to 45 -> 10 seconds out of 30
        self.assertAlmostEqual(coverage, (1.0/3.0))


    def test_compute_data_loss(self):

        # Basic series
        series = DataTimePointSeries()
        series.append(DataTimePoint(t = 20, data = {'temperature': 23}))
        series.append(DataTimePoint(t = 30, data = {'temperature': 23}))
        series.append(DataTimePoint(t = 40, data = {'temperature': 23}))
        series.append(DataTimePoint(t = 60, data = {'temperature': 23}))
        series.append(DataTimePoint(t = 70, data = {'temperature': 23}))
        attach_validity_regions(series)
  
        # Test basic
        self.assertAlmostEqual(compute_data_loss(series, from_t = 45, to_t = 55, sampling_interval=10), 1.0)
        self.assertAlmostEqual(compute_data_loss(series, from_t = 50, to_t = 60, sampling_interval=10), 0.5)
        self.assertAlmostEqual(compute_data_loss(series, from_t = 65, to_t = 75, sampling_interval=10), 0.0)
  
        # Test out of boundaries
        data_loss = compute_data_loss(series, from_t = 100, to_t = 120, sampling_interval=10) 
        self.assertEqual(data_loss, 1)    
  
        # Test for 1-element series
        single_element_series = DataTimePointSeries()
        single_element_series.append(DataTimePoint(t = 40, data = {'temperature': 23}))
        attach_validity_regions(single_element_series, sampling_interval=10)
 
        # TODO: the following is quite messy.
 
        # The series has only one element and no sampling_interval is provided, no idea how to compute validity        
        #with self.assertRaises(ValueError):
        #    compute_data_loss(single_element_series, from_t = 30, to_t = 60) 
  
        # Only 10 seconds out of 30 (point has validity regions 35-45, interval is from 30 to 60)
        data_loss = compute_data_loss(single_element_series, from_t = 30, to_t = 60, sampling_interval=10) 
        self.assertAlmostEqual(data_loss, (2.0/3.0))
 
        # Test two elements, some border condition for sull fata losses may arise
        series_two_elements = DataTimePointSeries()
        series_two_elements.append(DataTimePoint(t = 40, data = {'temperature': 23}))
        series_two_elements.append(DataTimePoint(t = 60, data = {'temperature': 23}))
        attach_validity_regions(series_two_elements, sampling_interval=10)
         
        self.assertEqual(compute_data_loss(series_two_elements, from_t = 45, to_t = 55, sampling_interval=10), 1)
  
        # Series with pre-existent data losses, which have to be taken into account as well
        series_with_data_losses = DataTimePointSeries()
        series_with_data_losses.append(DataTimePoint(t = 20, data = {'temperature': 23}, data_loss=0))
        series_with_data_losses.append(DataTimePoint(t = 30, data = {'temperature': 23}, data_loss=0))
        series_with_data_losses.append(DataTimePoint(t = 40, data = {'temperature': 23}, data_loss=0.5))
        series_with_data_losses.append(DataTimePoint(t = 60, data = {'temperature': 23}, data_loss=0))
        series_with_data_losses.append(DataTimePoint(t = 70, data = {'temperature': 23}, data_loss=0))
        attach_validity_regions(series_with_data_losses)
 
        # Test missing + already ( 0 + 0.5 + 1 + 0 = 1.5 / 3 = 0.5)
        data_loss = compute_data_loss(series_with_data_losses, from_t = 30, to_t = 60, sampling_interval=10, force=True) 
        self.assertAlmostEqual(data_loss, 0.5)
  
        # Test also on actually resampled series
        resampled_series = series.resample(10)
        attach_validity_regions(resampled_series)
 
        data_loss = compute_data_loss(resampled_series, from_t = 30, to_t = 60)
        self.assertAlmostEqual(data_loss, 1.0/3.0)
 
        resampled_series_5 = self.data_time_point_series_5.resample(60)  
        attach_validity_regions(resampled_series_5)
       
        data_loss = compute_data_loss(resampled_series_5, from_t = 1436022300, to_t = 1436022600) 
        self.assertAlmostEqual(data_loss, 0.3)
         
        # TODO: This test come after a bug, needs to be better unrolled. the bug was
        # raising when there were overlapping points at the start or end of the slot.
    
        from timeseria import  storages
        DATASET_PATH = '/'.join(storages.__file__.split('/')[0:-1]) + '/tests/test_data/csv/'
        csv_storage = storages.CSVFileStorage(DATASET_PATH + 'temperature.csv')
        timeseries = csv_storage.get(limit=400)
        timeseries = timeseries[200:300]
        resampled_series = timeseries.resample(600)
        
        attach_validity_regions(resampled_series)
        
        data_loss = compute_data_loss(resampled_series, from_t = 1546700400, to_t = 1546711200) 
        self.assertAlmostEqual(data_loss, 1.0)


class TestGetPeriodicity(unittest.TestCase):

    def test_get_periodicity(self):
        
        univariate_data_time_point_series = CSVFileStorage(TEST_DATA_PATH + '/csv/temperature.csv').get()
        
        univariate_1h_data_time_slot_series = Aggregator('1h').process(univariate_data_time_point_series)
        
        perdiodicity = get_periodicity(univariate_1h_data_time_slot_series)

        self.assertEqual(perdiodicity, 24)


class TestDetectSamplingInterval(unittest.TestCase):

    def test_detect_sampling_interval(self):
        
        data_time_point_series = CSVFileStorage(TEST_DATA_PATH + '/csv/humitemp_short.csv').get()
        self.assertEqual(detect_sampling_interval(data_time_point_series), 61)
        
        data_time_point_series = CSVFileStorage(TEST_DATA_PATH + '/csv/shampoo_sales.csv', date_label = 'Month', date_format = '%y-%m').get()
        self.assertEqual(detect_sampling_interval(data_time_point_series), 2678400)  # 2678400/60/60/24 = 31 Days (the most frequent)

        data_time_point_series = CSVFileStorage(TEST_DATA_PATH + '/csv/temperature.csv').get()
        self.assertEqual(detect_sampling_interval(data_time_point_series), 600)

        data_time_point_series = CSVFileStorage(TEST_DATA_PATH + '/csv/temp_short_1h.csv').get()
        self.assertEqual(detect_sampling_interval(data_time_point_series), 3600)





