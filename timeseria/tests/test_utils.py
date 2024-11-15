import os
import unittest
import pandas as pd
from propertime.utils import dt, s_from_dt

from ..utils import detect_encoding, detect_periodicity, detect_sampling_interval
from ..utils import _compute_coverage, _compute_data_loss, _compute_validity_regions
from ..utils import _Gaussian
from ..utils import rescale
from ..utils import _is_index_based, _is_key_based, _has_numerical_values
from ..datastructures import DataTimePoint, TimeSeries
from ..storages import CSVFileStorage
from ..units import TimeUnit

# Setup logging
from .. import logger
logger.setup()

# Support function for attaching validity regions to a series
def attach_validity_regions(series, sampling_interval=None):
    validity_regions = _compute_validity_regions(series, sampling_interval=sampling_interval)
    for point in series:
        point.valid_from = validity_regions[point.t][0]
        point.valid_to = validity_regions[point.t][1]

# Set test data path
TEST_DATA_PATH = '/'.join(os.path.realpath(__file__).split('/')[0:-1]) + '/test_data/'


class TestComputeValidityRegions(unittest.TestCase):

    def test_standard(self):

        series = TimeSeries()
        series.append(DataTimePoint(t = 7, data = {'value': 7}))
        series.append(DataTimePoint(t = 12, data = {'value': 12}))
        series.append(DataTimePoint(t = 17, data = {'value': 17}))
        series.append(DataTimePoint(t = 22, data = {'value': 22}))
        series.append(DataTimePoint(t = 27, data = {'value': 27}))

        # All of them
        expected_results = {7: [4.5, 9.5], 12: [9.5, 14.5], 17: [14.5, 19.5], 22: [19.5, 24.5], 27: [24.5, 29.5]}
        results = _compute_validity_regions(series)
        self.assertEqual(results, expected_results)

        # Only from-to
        expected_results = {12: [9.5, 14.5], 17: [14.5, 19.5], 22: [19.5, 24.5]}
        results = _compute_validity_regions(series, from_t=10, to_t=20)
        self.assertEqual(results, expected_results)

        # Shrink them according to the from-to
        expected_results = {12: [10, 14.5], 17: [14.5, 19.5], 22: [19.5, 20]}
        results = _compute_validity_regions(series, from_t=10, to_t=20, cut=True)
        self.assertEqual(results, expected_results)

        # Force a specific sampling intervals
        expected_results = {12: [11.5, 12.5], 17: [16.5, 17.5]}
        results = _compute_validity_regions(series, from_t=10, to_t=20, sampling_interval=1)
        self.assertEqual(results, expected_results)

        # Single-element series
        single_element_series = TimeSeries()
        single_element_series.append(DataTimePoint(t = 40, data = {'value': 4571.55}))
        expected_results = {40: [35.0, 45.0]}

        # The series has only one element and no sampling_interval is provided, no idea how to compute validity
        with self.assertRaises(ValueError):
            _compute_validity_regions(series = single_element_series, from_t = 30, to_t = 60)

        # Call by providing the sampling_interval
        results = _compute_validity_regions(series = single_element_series, from_t = 30, to_t = 60, sampling_interval=10)

        self.assertEqual(results, expected_results)


    def test_prev_next_points(self):
        series = TimeSeries()
        series.append(DataTimePoint(t = 7, data = {'value': 7}))
        series.append(DataTimePoint(t = 12, data = {'value': 12}))
        series.append(DataTimePoint(t = 17, data = {'value': 17}))
        series.append(DataTimePoint(t = 22, data = {'value': 22}))
        series.append(DataTimePoint(t = 27, data = {'value': 27}))

        # Test prev-next
        expected_results = {7: [4.5, 9.5], 12: [9.5, 14.5], 17: [14.5, 19.5], 22: [19.5, 24.5], 27: [24.5, 29.5]}
        results = _compute_validity_regions(series, from_t=9, to_t=25)
        self.assertEqual(results, expected_results)

        # Test cutted prev-next
        expected_results = {7: [9, 9.5], 12: [9.5, 14.5], 17: [14.5, 19.5], 22: [19.5, 24.5], 27: [24.5, 25]}
        results = _compute_validity_regions(series, from_t=9, to_t=25, cut=True)
        self.assertEqual(results, expected_results)


    def test_overlaps(self):

        series = TimeSeries()
        series.append(DataTimePoint(t = 7, data = {'value': 7}))
        series.append(DataTimePoint(t = 12, data = {'value': 12}))
        series.append(DataTimePoint(t = 13, data = {'value': 13}))
        series.append(DataTimePoint(t = 17, data = {'value': 17}))
        series.append(DataTimePoint(t = 22, data = {'value': 22}))
        series.append(DataTimePoint(t = 27, data = {'value': 27}))

        expected_results = {7: [4.5, 9.5], 12: [9.5, 12.5], 13: [12.5, 15], 17: [15, 19.5], 22: [19.5, 24.5], 27: [24.5, 29.5]}
        results = _compute_validity_regions(series)
        self.assertEqual(results, expected_results)


    def test_major_overlaps(self):

        series = TimeSeries()
        series.append(DataTimePoint(t = 7, data = {'value': 7}))
        series.append(DataTimePoint(t = 12, data = {'value': 12}))
        series.append(DataTimePoint(t = 12.1, data = {'value': 12.1}))
        series.append(DataTimePoint(t = 12.3, data = {'value': 12.3}))
        series.append(DataTimePoint(t = 13, data = {'value': 13}))
        series.append(DataTimePoint(t = 17, data = {'value': 17}))
        series.append(DataTimePoint(t = 22, data = {'value': 22}))
        series.append(DataTimePoint(t = 27, data = {'value': 27}))

        expected_results = {7: [4.5, 9.5], 12: [9.5, 12.05], 12.1: [12.05, 12.2], 12.3: [12.2, 12.65], 13: [12.65, 15.0], 17: [15.0, 19.5], 22: [19.5, 24.5], 27: [24.5, 29.5]}
        results = _compute_validity_regions(series)
        self.assertEqual(results, expected_results)



class TestComputeCoverageAndDataLoss(unittest.TestCase):

    def setUp(self):

        # All the following time series have point with validity=1m

        # Time series from 16:58:00 to 17:32:00 (Europe/Rome)
        self.timeseries_1 = TimeSeries()
        start_t = 1436022000 - 120
        for i in range(35):
            datatimepoint = DataTimePoint(t = start_t + (i*60),
                                            data = {'temperature': 154+i})
            self.timeseries_1.append(datatimepoint)
        attach_validity_regions(self.timeseries_1)

        # Time series from 17:00:00 to 17:30:00 (Europe/Rome)
        self.timeseries_2 = TimeSeries()
        start_t = 1436022000
        for i in range(34):
            datatimepoint = DataTimePoint(t    = start_t + (i*60),
                                            data = {'temperature': 154+i})
            self.timeseries_2.append(datatimepoint)
        attach_validity_regions(self.timeseries_2)

        # Time series from 17:00:00 to 17:20:00 (Europe/Rome)
        self.timeseries_3 = TimeSeries()
        start_t = 1436022000 - 120
        for i in range(23):
            datatimepoint = DataTimePoint(t    = start_t + (i*60),
                                            data = {'temperature': 154+i})
            self.timeseries_3.append(datatimepoint)
        attach_validity_regions(self.timeseries_3)

        # Time series from 17:10:00 to 17:30:00 (Europe/Rome)
        self.timeseries_4 = TimeSeries()
        start_t = 1436022000 + 600
        for i in range(21):
            datatimepoint = DataTimePoint(t    = start_t + (i*60),
                                            data = {'temperature': 154+i})
            self.timeseries_4.append(datatimepoint)
        attach_validity_regions(self.timeseries_4)

        # Time series from 16:58:00 to 17:32:00 (Europe/Rome)
        self.timeseries_5 = TimeSeries()
        start_t = 1436022000 - 120
        for i in range(35):
            if i > 10 and i <21:
                continue
            datatimepoint = DataTimePoint(t    = start_t + (i*60),
                                            data = {'temperature': 154+i})
            self.timeseries_5.append(datatimepoint)
        attach_validity_regions(self.timeseries_5)


        # The following time series has point with validity=15m

        # Time series from 2019,10,1,1,0,0 to 2019,10,1,6,0,0 (Europe/Rome)
        from_dt  = dt(2019,10,1,1,0,0, tz='Europe/Rome')
        to_dt    = dt(2019,10,1,6,0,0, tz='Europe/Rome')
        time_unit = TimeUnit('15m')
        self.timeseries_6 = TimeSeries()
        slider_dt = from_dt
        count = 0
        while slider_dt < to_dt:
            if count not in [1, 6, 7, 8, 9, 10]:
                datatimepoint = DataTimePoint(t    = s_from_dt(slider_dt),
                                                data = {'temperature': 154+count})
                self.timeseries_6.append(datatimepoint)
            slider_dt = slider_dt + time_unit
            count += 1
        attach_validity_regions(self.timeseries_6)


    def test_compute_coverage(self):

        from_t = 1436022000       # 2015-07-04 17:00:00+02:00
        to_t   = 1436022000+1800  # 2015-07-04 17:30:00+02:00

        # A) Full coverage (coverage=1.0) and again, to test "reproducibility"
        coverage = _compute_coverage(series = self.timeseries_1, from_t = from_t, to_t = to_t)
        self.assertEqual(coverage, 1.0)
        coverage = _compute_coverage(series = self.timeseries_1, from_t = from_t, to_t = to_t)
        self.assertEqual(coverage, 1.0)


        # B) Full coverage (coverage=1.0) without prev/next in the time series
        coverage = _compute_coverage(series = self.timeseries_1, from_t = from_t, to_t = to_t)
        self.assertEqual(coverage, 1.0)
        self.assertEqual(coverage, 1.0)


        # C) Missing ten minutes over 30 at the end (coverage=0.683))
        coverage = _compute_coverage(series = self.timeseries_3, from_t = from_t, to_t = to_t)

        # 20 minutes plus other 30 secs validity for the 20th point over 30 minutes
        self.assertEqual(coverage, ( ((20*60.0)+30.0) / (30*60.0)) )


        # D) Missing ten minutes over 30 at the beginning (coverage=0.683)
        coverage = _compute_coverage(series = self.timeseries_4, from_t = from_t, to_t = to_t)

        # 20 minutes plus other 30 secs (previous) validity for the 10th point over 30 minutes
        self.assertEqual(coverage, ( ((20*60.0)+30.0) / (30*60.0)) )


        # E) Missing eleven minutes over 30 in the middle (coverage=0.66)
        coverage = _compute_coverage(series = self.timeseries_5, from_t = from_t, to_t = to_t)

        # 20 minutes plus other 30 secs (previous) validity for the 10th point over 30 minutes
        self.assertAlmostEqual(coverage, (2.0/3.0))


        # F) Missing half slot before slot re-start
        from_t = s_from_dt(dt=dt(2019,10,1,3,30,0, tz='Europe/Rome'))
        to_t   = s_from_dt(dt=dt(2019,10,1,3,45,0, tz='Europe/Rome'))
        coverage = _compute_coverage(series = self.timeseries_6, from_t = from_t, to_t = to_t)

        self.assertAlmostEqual(coverage, (0.5))

        # G) Border conditions for from_t and to_t:
        timeseries = TimeSeries()
        timeseries.append(DataTimePoint(t = 20, data = {'temperature': 23}))
        timeseries.append(DataTimePoint(t = 30, data = {'temperature': 23}))
        timeseries.append(DataTimePoint(t = 40, data = {'temperature': 23}))
        timeseries.append(DataTimePoint(t = 60, data = {'temperature': 23}))
        timeseries.append(DataTimePoint(t = 70, data = {'temperature': 23}))
        attach_validity_regions(timeseries)

        coverage = _compute_coverage(series = timeseries, from_t = 30, to_t = 60)
        self.assertAlmostEqual(coverage, (2.0/3.0))

        # H) single-element series (requires providing the sampling_interval when computing the validity regions)
        single_element_series = TimeSeries()
        single_element_series.append(DataTimePoint(t = 40, data = {'temperature': 23}))

        attach_validity_regions(single_element_series, sampling_interval=10)
        coverage = _compute_coverage(series = single_element_series, from_t = 30, to_t = 60)

        # 30 to 60, point is from 35 to 45 -> 10 seconds out of 30
        self.assertAlmostEqual(coverage, (1.0/3.0))


    def test_compute_data_loss(self):

        # Basic series
        series = TimeSeries()
        series.append(DataTimePoint(t = 20, data = {'temperature': 23}))
        series.append(DataTimePoint(t = 30, data = {'temperature': 23}))
        series.append(DataTimePoint(t = 40, data = {'temperature': 23}))
        series.append(DataTimePoint(t = 60, data = {'temperature': 23}))
        series.append(DataTimePoint(t = 70, data = {'temperature': 23}))
        attach_validity_regions(series)

        # Test basic
        self.assertAlmostEqual(_compute_data_loss(series, from_t = 45, to_t = 55, sampling_interval=10), 1.0)
        self.assertAlmostEqual(_compute_data_loss(series, from_t = 50, to_t = 60, sampling_interval=10), 0.5)
        self.assertAlmostEqual(_compute_data_loss(series, from_t = 65, to_t = 75, sampling_interval=10), 0.0)

        # Test out of boundaries
        data_loss = _compute_data_loss(series, from_t = 100, to_t = 120, sampling_interval=10)
        self.assertEqual(data_loss, 1)

        # Test for 1-element series (requires providing the sampling_interval when computing the validity regions)
        single_element_series = TimeSeries()
        single_element_series.append(DataTimePoint(t = 40, data = {'temperature': 23}))
        attach_validity_regions(single_element_series, sampling_interval=10)

        # Only 10 seconds out of 30 (point has validity regions 35-45, interval is from 30 to 60)
        data_loss = _compute_data_loss(single_element_series, from_t = 30, to_t = 60, sampling_interval=10)
        self.assertAlmostEqual(data_loss, (2.0/3.0))

        # Test two elements, some border condition for sull fata losses may arise
        series_two_elements = TimeSeries()
        series_two_elements.append(DataTimePoint(t = 40, data = {'temperature': 23}))
        series_two_elements.append(DataTimePoint(t = 60, data = {'temperature': 23}))
        attach_validity_regions(series_two_elements, sampling_interval=10)

        self.assertEqual(_compute_data_loss(series_two_elements, from_t = 45, to_t = 55, sampling_interval=10), 1)

        # Series with pre-existent data losses, which have to be taken into account as well
        series_with_data_losses = TimeSeries()
        series_with_data_losses.append(DataTimePoint(t = 20, data = {'temperature': 23}, data_loss=0))
        series_with_data_losses.append(DataTimePoint(t = 30, data = {'temperature': 23}, data_loss=0))
        series_with_data_losses.append(DataTimePoint(t = 40, data = {'temperature': 23}, data_loss=0.5))
        series_with_data_losses.append(DataTimePoint(t = 60, data = {'temperature': 23}, data_loss=0))
        series_with_data_losses.append(DataTimePoint(t = 70, data = {'temperature': 23}, data_loss=0))
        attach_validity_regions(series_with_data_losses)

        # Test missing + already ( 0 + 0.5 + 1 + 0 = 1.5 / 3 = 0.5)
        data_loss = _compute_data_loss(series_with_data_losses, from_t = 30, to_t = 60, sampling_interval=10, force=True)
        self.assertAlmostEqual(data_loss, 0.5)

        # Test also on actually resampled series
        resampled_series = series.resample(10)
        attach_validity_regions(resampled_series)

        data_loss = _compute_data_loss(resampled_series, from_t = 30, to_t = 60)
        self.assertAlmostEqual(data_loss, 1.0/3.0)

        resampled_series_5 = self.timeseries_5.resample(60)
        attach_validity_regions(resampled_series_5)

        data_loss = _compute_data_loss(resampled_series_5, from_t = 1436022300, to_t = 1436022600)
        self.assertAlmostEqual(data_loss, 0.3)

        # TODO: This test was introduced after finding a bug, and needs to be better unrolled.
        # the bug was raising when there were overlapping points at the start or end of the slot.
        from timeseria import  storages
        csv_storage = storages.CSVFileStorage(TEST_DATA_PATH + 'csv/temperature.csv')
        series = csv_storage.get(limit=400)
        series = series[200:300]
        resampled_series = series.resample(600)
        attach_validity_regions(resampled_series)
        data_loss = _compute_data_loss(resampled_series, from_t = 1546700400, to_t = 1546711200)
        self.assertAlmostEqual(data_loss, 1.0)


class TestGetPeriodicity(unittest.TestCase):

    def test_get_periodicity(self):

        univariate_timeseries = CSVFileStorage(TEST_DATA_PATH + '/csv/temperature.csv').get()

        univariate_1h_timeseries = univariate_timeseries.aggregate('1h')

        perdiodicity = detect_periodicity(univariate_1h_timeseries)

        self.assertEqual(perdiodicity, 24)


class TestDetectSamplingInterval(unittest.TestCase):

    def test_detect_sampling_interval(self):

        timeseries = CSVFileStorage(TEST_DATA_PATH + '/csv/humitemp_short.csv').get()
        self.assertEqual(detect_sampling_interval(timeseries), 61)
        self.assertAlmostEqual(detect_sampling_interval(timeseries, confidence=True)[1], 0.7368421)

        timeseries = CSVFileStorage(TEST_DATA_PATH + '/csv/shampoo_sales.csv', date_column = 'Month', date_format = '%y-%m').get()
        self.assertEqual(detect_sampling_interval(timeseries), 2678400)  # 2678400/60/60/24 = 31 Days (the most frequent)

        timeseries = CSVFileStorage(TEST_DATA_PATH + '/csv/temperature.csv').get()
        self.assertEqual(detect_sampling_interval(timeseries), 600)

        timeseries = CSVFileStorage(TEST_DATA_PATH + '/csv/temp_short_1h.csv').get()
        self.assertEqual(detect_sampling_interval(timeseries), 3600)


class TestDetectEncoding(unittest.TestCase):

    def test_detect_encoding(self):

        encoding = detect_encoding('{}/csv/shampoo_sales.csv'.format(TEST_DATA_PATH), streaming=False)
        self.assertEqual(encoding, 'ascii')


class TestGaussian(unittest.TestCase):

    def test_gaussian(self):

        test_data = [1,3,7,8,8,8,9,9,9,9,9,10,10,10,10,10,10,11,11,11,11,12,12,12,13,13,15,15,18,20]

        gaussian = _Gaussian.from_data(test_data)

        # Check parameters
        self.assertAlmostEqual(gaussian.mu, 10.466, places=2)
        self.assertAlmostEqual(gaussian.sigma, 3.639, places=2)

        # Check callable function behavior
        self.assertAlmostEqual(gaussian(gaussian.mu), 0.1096, places=3)

        self.assertAlmostEqual(gaussian(10), 0.1087, places=4)

        # Check solver (find xes):
        self.assertAlmostEqual(gaussian.find_xes(y=0.1)[0], 8.9079, places=3)
        self.assertAlmostEqual(gaussian.find_xes(y=0.1)[1], 12.0253, places=3)
        self.assertAlmostEqual(gaussian(gaussian.find_xes(y=0.05)[0]), 0.05, places=3)
        self.assertAlmostEqual(gaussian(gaussian.find_xes(y=0.05)[1]), 0.05, places=3)

        # Edge case to find back the peak
        self.assertAlmostEqual(gaussian.find_xes(y=0.1096)[0], 10.4422, places=2)
        self.assertAlmostEqual(gaussian.find_xes(y=0.1096)[1], 10.4910, places=2)

        # Test the CDF now
        self.assertEqual(gaussian.cumulative(gaussian.mu), 0.5)


class TestRescale(unittest.TestCase):

    def test_rescale(self):

        with self.assertRaises(ValueError):
            rescale(value=0.5, source_from=1, source_to=2, target_from=1, target_to=0)
        with self.assertRaises(ValueError):
            rescale(value=2.5, source_from=1, source_to=2, target_from=1, target_to=0)

        self.assertEqual(rescale(value=7.5, source_from=5, source_to=10, target_from=0, target_to=1),0.5)
        self.assertEqual(rescale(value=7.5, source_from=5, source_to=10, target_from=0, target_to=2),1)
        self.assertEqual(rescale(value=7.5, source_from=5, source_to=10, target_from=1, target_to=2),1.5)
        self.assertEqual(rescale(value=7.5, source_from=5, source_to=10, target_from=10, target_to=20),15)
        self.assertEqual(rescale(value=0.5, source_from=0, source_to=1, target_from=7, target_to=8),7.5)

        self.assertAlmostEqual(rescale(value=0.98, source_from=0.000016, source_to=1.0, target_from=0, target_to=1), 0.97999967)


class TestDetectDataTypes(unittest.TestCase):

    def test_detect_data_types(self):

        data = [1,2,3]
        self.assertEqual(_is_index_based(data), True)
        self.assertEqual(_is_key_based(data), False)
        self.assertEqual(_has_numerical_values(data), True)

        data = [1,'z',3]
        self.assertEqual(_is_index_based(data), True)
        self.assertEqual(_is_key_based(data), False)
        self.assertEqual(_has_numerical_values(data), False)

        data = [1,2,None]
        self.assertEqual(_is_index_based(data), True)
        self.assertEqual(_is_key_based(data), False)
        self.assertEqual(_has_numerical_values(data), False)


        data = (1,2,3)
        self.assertEqual(_is_index_based(data), True)
        self.assertEqual(_is_key_based(data), False)
        self.assertEqual(_has_numerical_values(data), True)

        data = 'hello'
        self.assertEqual(_is_index_based(data), True)
        self.assertEqual(_is_key_based(data), False)
        self.assertEqual(_has_numerical_values(data), False)

        data = {'a':1, 'b':2}
        self.assertEqual(_is_index_based(data), False)
        self.assertEqual(_is_key_based(data), True)
        self.assertEqual(_has_numerical_values(data), True)

        data = {'a':1, 'b':'z'}
        self.assertEqual(_is_index_based(data), False)
        self.assertEqual(_is_key_based(data), True)
        self.assertEqual(_has_numerical_values(data), False)

        data = {'a':1, 'b':None}
        self.assertEqual(_is_index_based(data), False)
        self.assertEqual(_is_key_based(data), True)
        self.assertEqual(_has_numerical_values(data), False)

        data = pd.Series([4,5,6])
        self.assertEqual(_is_index_based(data), True)
        self.assertEqual(_is_key_based(data), False)
        self.assertEqual(_has_numerical_values(data), True)

        data = pd.Series([4,5,'z'])
        self.assertEqual(_is_index_based(data), True)
        self.assertEqual(_is_key_based(data), False)
        self.assertEqual(_has_numerical_values(data), False)

        data = pd.Series([4,5,6])
        data.index = ['a','b','c']
        self.assertEqual(_is_index_based(data), True)
        self.assertEqual(_is_key_based(data), False)
        self.assertEqual(_has_numerical_values(data), True)

