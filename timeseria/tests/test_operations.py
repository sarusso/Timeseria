import unittest
from ..datastructures import TimePoint, DataTimePoint , DataTimeSlot, TimeSeries
from ..operations import derivative, integral, diff, csum, min, max, avg, filter, select, mavg, normalize, merge, offset, rescale, slice, get
from ..operations import Operation

# Setup logging
from .. import logger
logger.setup()


class TestOpertions(unittest.TestCase):

    def test_base(self):

        operation_from_callable = Operation()
        self.assertEqual(operation_from_callable.__name__, 'operation')

        def operation_as_function(data):
            pass

        self.assertEqual(operation_as_function.__name__, 'operation_as_function')


    def test_diff_csum(self):

        # Test on empty, single point or variable resolution time series
        series = TimeSeries()
        with self.assertRaises(ValueError):
            diff(series)

        series = TimeSeries()
        series.append(DataTimePoint(t=0, data={'value':10}))
        with self.assertRaises(ValueError):
            diff(series)

        series = TimeSeries()
        series.append(DataTimePoint(t=0, data={'value':10}))
        series.append(DataTimePoint(t=1, data={'value':11}))
        series.append(DataTimePoint(t=8, data={'value':12}))
        with self.assertRaises(ValueError):
            diff(series)

        # Test data for the next tests
        series = TimeSeries()
        series.append(DataTimeSlot(start=TimePoint(0), end=TimePoint(60), data={'value':10}, data_loss=0.1))
        series.append(DataTimeSlot(start=TimePoint(60), end=TimePoint(120), data={'value':12}, data_loss=0.2))
        series.append(DataTimeSlot(start=TimePoint(120), end=TimePoint(180), data={'value':15}, data_loss=0.3))
        series.append(DataTimeSlot(start=TimePoint(180), end=TimePoint(240), data={'value':16}, data_loss=0.4))

        # Test standard, from the series
        diff_series = diff(series)
        self.assertEqual(len(diff_series), 3)
        self.assertEqual(diff_series[0].data['value_diff'],2)
        self.assertEqual(diff_series[1].data['value_diff'],3)
        self.assertEqual(diff_series[2].data['value_diff'],1)

        # Test data loss correctly carried forward by the diff
        self.assertEqual(diff_series[0].data_loss, 0.2)
        self.assertEqual(diff_series[1].data_loss, 0.3)
        self.assertEqual(diff_series[2].data_loss, 0.4)

        # Test csum as well
        diff_csum_series = csum(diff_series, offset=10)
        self.assertEqual(diff_csum_series[0].data['value_diff_csum'], 10)
        self.assertEqual(diff_csum_series[1].data['value_diff_csum'], 12)
        self.assertEqual(diff_csum_series[2].data['value_diff_csum'], 15)
        self.assertEqual(diff_csum_series[3].data['value_diff_csum'], 16)

        # Test data loss correctly carried forward by the csum
        self.assertEqual(diff_csum_series[0].data_loss, 0) # This is the slot created by the offset, has no data loss
        self.assertEqual(diff_csum_series[1].data_loss, 0.2)
        self.assertEqual(diff_csum_series[2].data_loss, 0.3)
        self.assertEqual(diff_csum_series[3].data_loss, 0.4)

        # Test standalone
        self.assertEqual(len(diff(series)), 3)

        # Multi data labels Test data (on points)
        series = TimeSeries()
        series.append(DataTimePoint(t=0, data={'value':10, 'another_value': 75}))
        series.append(DataTimePoint(t=60, data={'value':12, 'another_value': 65}))
        series.append(DataTimePoint(t=120, data={'value':6, 'another_value': 32}))
        series.append(DataTimePoint(t=180, data={'value':16, 'another_value': 10}))
        diff_series = series.diff()
        self.assertEqual(diff_series[0].data['value_diff'],2)
        self.assertEqual(diff_series[0].data['another_value_diff'],-10)
        self.assertEqual(diff_series[1].data['value_diff'],-6)
        self.assertEqual(diff_series[1].data['another_value_diff'],-33)

        # Test csum as well, with dict offset (on points)
        diff_csum_series = diff_series.csum(offset={'value_diff':10, 'another_value_diff':75})
        self.assertEqual(diff_csum_series[0].data['value_diff_csum'],10)
        self.assertEqual(diff_csum_series[0].data['another_value_diff_csum'],75)
        self.assertEqual(diff_csum_series[3].data['value_diff_csum'],16)
        self.assertEqual(diff_csum_series[3].data['another_value_diff_csum'],10)

        # Test on list data as well
        series = TimeSeries(DataTimePoint(t=60, data=[1.0,-4.0]),
                            DataTimePoint(t=120, data=[2.0,-2.0]),
                            DataTimePoint(t=180, data=[4.0,-1.0]))
        self.assertEqual(len(series.diff()),2)
        self.assertEqual(series.diff()[0].data, [1.0, 2.0])
        self.assertEqual(series.diff()[1].data, [2.0, 1.0])
        self.assertEqual(len(series.csum()),3)
        self.assertEqual(series.csum()[0].data, [1.0, -4.0])
        self.assertEqual(series.csum()[1].data, [3.0, -6.0])
        self.assertEqual(series.csum()[2].data, [7.0, -7.0])


    def test_derivative_integral(self):

        # Test on empty, single point or variable resolution time serie
        with self.assertRaises(ValueError):
            series = TimeSeries()
            derivative(series)

        with self.assertRaises(ValueError):
            series = TimeSeries()
            series.append(DataTimePoint(t=0, data={'value':10}))
            derivative(series)

        # Test data
        series = TimeSeries()
        series.append(DataTimeSlot(start=TimePoint(0), end=TimePoint(1), data={'value':10}, data_loss=0.1))
        series.append(DataTimeSlot(start=TimePoint(1), end=TimePoint(2), data={'value':12}, data_loss=0.2))
        series.append(DataTimeSlot(start=TimePoint(2), end=TimePoint(3), data={'value':15}, data_loss=0.3))
        series.append(DataTimeSlot(start=TimePoint(3), end=TimePoint(4), data={'value':16}, data_loss=0.4))

        # Test standard derivative behavior, from the series
        derivative_series = derivative(series)
        self.assertEqual(len(derivative_series), 4)
        self.assertAlmostEqual(derivative_series[0].data['value_derivative'],2)     # 0
        self.assertAlmostEqual(derivative_series[1].data['value_derivative'],2.5)   # 2
        self.assertAlmostEqual(derivative_series[2].data['value_derivative'],2)     # 5
        self.assertAlmostEqual(derivative_series[3].data['value_derivative'],1.0)   # 6

        # Test data loss correctly carried forward by the derivative
        self.assertEqual(derivative_series[0].data_loss, 0.1)
        self.assertEqual(derivative_series[1].data_loss, 0.2)
        self.assertEqual(derivative_series[2].data_loss, 0.3)
        self.assertEqual(derivative_series[3].data_loss, 0.4)

        # Test integral as well
        derivative_integral_series = integral(derivative_series, c=10)
        self.assertEqual(derivative_integral_series[0].data['value_derivative_integral'], 10)
        self.assertEqual(derivative_integral_series[1].data['value_derivative_integral'], 12)
        self.assertEqual(derivative_integral_series[2].data['value_derivative_integral'], 15)
        self.assertEqual(derivative_integral_series[3].data['value_derivative_integral'], 16)

        # Test data loss correctly carried forward by the integral
        self.assertEqual(derivative_integral_series[0].data_loss, 0.1)
        self.assertEqual(derivative_integral_series[1].data_loss, 0.2)
        self.assertEqual(derivative_integral_series[2].data_loss, 0.3)
        self.assertEqual(derivative_integral_series[3].data_loss, 0.4)

        # Test data
        series = TimeSeries()
        series.append(DataTimeSlot(start=TimePoint(0), end=TimePoint(10), data={'value':10}))
        series.append(DataTimeSlot(start=TimePoint(10), end=TimePoint(20), data={'value':12}))
        series.append(DataTimeSlot(start=TimePoint(20), end=TimePoint(30), data={'value':15}))
        series.append(DataTimeSlot(start=TimePoint(30), end=TimePoint(40), data={'value':16}))

        # Test standard derivative behavior, from the series
        derivative_series = derivative(series)
        self.assertEqual(len(derivative_series), 4)
        self.assertAlmostEqual(derivative_series[0].data['value_derivative'],0.2)
        self.assertAlmostEqual(derivative_series[1].data['value_derivative'],0.25)
        self.assertAlmostEqual(derivative_series[2].data['value_derivative'],0.2)
        self.assertAlmostEqual(derivative_series[3].data['value_derivative'],0.1)

        # Multi data labels Test data
        series = TimeSeries()
        series.append(DataTimePoint(t=0, data={'value':10, 'another_value': 75}))
        series.append(DataTimePoint(t=10, data={'value':12, 'another_value': 65}))
        series.append(DataTimePoint(t=20, data={'value':6, 'another_value': 32}))
        series.append(DataTimePoint(t=30, data={'value':16, 'another_value': 10}))
        derivative_series = series.derivative()
        self.assertAlmostEqual(derivative_series[0].data['value_derivative'],0.2)
        self.assertAlmostEqual(derivative_series[0].data['another_value_derivative'],-1.0)
        self.assertAlmostEqual(derivative_series[1].data['value_derivative'],-0.2)
        self.assertAlmostEqual(derivative_series[1].data['another_value_derivative'],-2.15)

        # Test derivative-integral identity with the data
        series = TimeSeries()
        series.append(DataTimeSlot(start=TimePoint(0), end=TimePoint(1), data={'value':0}))
        series.append(DataTimeSlot(start=TimePoint(1), end=TimePoint(2), data={'value':6}))
        series.append(DataTimeSlot(start=TimePoint(2), end=TimePoint(3), data={'value':14}))
        series.append(DataTimeSlot(start=TimePoint(3), end=TimePoint(4), data={'value':16}))
        series.append(DataTimeSlot(start=TimePoint(4), end=TimePoint(5), data={'value':20}))

        derivative_series = series.derivative()
        derivative_integral_series = derivative_series.integral()

        self.assertAlmostEqual(derivative_integral_series[0].data['value_derivative_integral'],0)
        self.assertAlmostEqual(derivative_integral_series[1].data['value_derivative_integral'],6)
        self.assertAlmostEqual(derivative_integral_series[2].data['value_derivative_integral'],14)
        self.assertAlmostEqual(derivative_integral_series[3].data['value_derivative_integral'],16)
        self.assertAlmostEqual(derivative_integral_series[4].data['value_derivative_integral'],20)

        # Variable sampling rate
        series = TimeSeries()
        series.append(DataTimePoint(t=0,  data={'value':0}))
        series.append(DataTimePoint(t=1,  data={'value':0.5}))
        series.append(DataTimePoint(t=2,  data={'value':1}))
        series.append(DataTimePoint(t=4,  data={'value':2.5}))
        series.append(DataTimePoint(t=5,  data={'value':3.5}))

        derivative_series = series.derivative()
        self.assertAlmostEqual(derivative_series[0].data['value_derivative'], 0.5)
        self.assertAlmostEqual(derivative_series[1].data['value_derivative'], 0.5)
        self.assertAlmostEqual(derivative_series[2].data['value_derivative'], 0.625)
        self.assertAlmostEqual(derivative_series[3].data['value_derivative'], 0.875)
        self.assertAlmostEqual(derivative_series[4].data['value_derivative'], 1.0)

        derivative_integral_series = derivative_series.integral()

        self.assertAlmostEqual(derivative_integral_series[0].data['value_derivative_integral'],0)
        self.assertAlmostEqual(derivative_integral_series[1].data['value_derivative_integral'],0.5)
        self.assertAlmostEqual(derivative_integral_series[2].data['value_derivative_integral'],1)
        self.assertAlmostEqual(derivative_integral_series[3].data['value_derivative_integral'],2.5)
        self.assertAlmostEqual(derivative_integral_series[4].data['value_derivative_integral'],3.5)


    def test_normalize(self):
        series =  TimeSeries(DataTimePoint(t=60, data={'a':2, 'b':6}, data_loss=0.1),
                             DataTimePoint(t=120, data={'a':4, 'b':9}, data_loss=0.2),
                             DataTimePoint(t=180, data={'a':8, 'b':3}, data_loss=0.3))

        normalized_series = normalize(series)

        self.assertEqual(normalized_series[0].data['a'],0)
        self.assertEqual(normalized_series[2].data['b'],0)

        self.assertEqual(normalized_series[2].data['a'],1)
        self.assertEqual(normalized_series[1].data['b'],1)

        # Test data loss correctly carried forward by the operation
        self.assertEqual(normalized_series[0].data_loss, 0.1)
        self.assertEqual(normalized_series[1].data_loss, 0.2)
        self.assertEqual(normalized_series[2].data_loss, 0.3)

        # Test normalize with respect to another range (other than 0-1):
        custom_normalized_series = normalize(series, [0.5,1.5])

        self.assertEqual(custom_normalized_series[0].data['a'],0.5)
        self.assertAlmostEqual(custom_normalized_series[1].data['a'],0.83333333)
        self.assertEqual(custom_normalized_series[2].data['a'],1.5)

        # Test on list data as well
        series = TimeSeries(DataTimePoint(t=60, data=[1.0,-3.0]),
                            DataTimePoint(t=120, data=[2.0,-2.0]),
                            DataTimePoint(t=180, data=[3.0,-1.0]))
        self.assertEqual(series.normalize()[0].data, [0.0,0.0])
        self.assertEqual(series.normalize()[1].data, [0.5,0.5])
        self.assertEqual(series.normalize()[2].data, [1.0,1.0])


    def test_rescale(self):
        series = TimeSeries(DataTimePoint(t=60, data={'a':2, 'b':6}, data_loss=0.1),
                            DataTimePoint(t=120, data={'a':4, 'b':9}, data_loss=0.2),
                            DataTimePoint(t=180, data={'a':8, 'b':3}, data_loss=0.3))

        # Rescale with single value
        rescaled_series = rescale(series, 2)
        self.assertEqual(len(rescaled_series),3)
        self.assertEqual(rescaled_series[0].data['a'],4)
        self.assertEqual(rescaled_series[0].data['b'],12)
        self.assertEqual(rescaled_series[1].data['a'],8)
        self.assertEqual(rescaled_series[1].data['b'],18)
        self.assertEqual(rescaled_series[2].data['a'],16)
        self.assertEqual(rescaled_series[2].data['b'],6)

        # Rescale with specific value
        rescaled_series = rescale(series, {'b':2})
        self.assertEqual(len(rescaled_series),3)
        self.assertEqual(rescaled_series[0].data['a'],2)
        self.assertEqual(rescaled_series[0].data['b'],12)
        self.assertEqual(rescaled_series[1].data['a'],4)
        self.assertEqual(rescaled_series[1].data['b'],18)
        self.assertEqual(rescaled_series[2].data['a'],8)
        self.assertEqual(rescaled_series[2].data['b'],6)

        # Test data loss correctly carried forward by the operation
        self.assertEqual(rescaled_series[0].data_loss, 0.1)
        self.assertEqual(rescaled_series[1].data_loss, 0.2)
        self.assertEqual(rescaled_series[2].data_loss, 0.3)

        # Test on list data as well
        series = TimeSeries(DataTimePoint(t=60, data=[1.0,-3.0]),
                            DataTimePoint(t=120, data=[2.0,-2.0]),
                            DataTimePoint(t=180, data=[3.0,-1.0]))
        self.assertEqual(series.rescale(10)[0].data, [10.0,-30.0])
        self.assertEqual(series.rescale(10)[1].data, [20.0,-20.0])
        self.assertEqual(series.rescale(10)[2].data, [30.0,-10.0])


    def test_offset(self):
        series = TimeSeries(DataTimePoint(t=60, data={'a':2, 'b':6}, data_loss=0.1),
                            DataTimePoint(t=120, data={'a':4, 'b':9}, data_loss=0.2),
                            DataTimePoint(t=180, data={'a':8, 'b':3}, data_loss=0.3))

        # Offset with single value
        offsetted_series = offset(series, 10)
        self.assertEqual(len(offsetted_series),3)
        self.assertEqual(offsetted_series[0].data['a'],12)
        self.assertEqual(offsetted_series[0].data['b'],16)
        self.assertEqual(offsetted_series[1].data['a'],14)
        self.assertEqual(offsetted_series[1].data['b'],19)
        self.assertEqual(offsetted_series[2].data['a'],18)
        self.assertEqual(offsetted_series[2].data['b'],13)

        # Offset with specific value
        offsetted_series = offset(series, {'b':-3})
        self.assertEqual(len(offsetted_series),3)
        self.assertEqual(offsetted_series[0].data['a'],2)
        self.assertEqual(offsetted_series[0].data['b'],3)
        self.assertEqual(offsetted_series[1].data['a'],4)
        self.assertEqual(offsetted_series[1].data['b'],6)
        self.assertEqual(offsetted_series[2].data['a'],8)
        self.assertEqual(offsetted_series[2].data['b'],0)

        # Test data loss correctly carried forward by the operation
        self.assertEqual(offsetted_series[0].data_loss, 0.1)
        self.assertEqual(offsetted_series[1].data_loss, 0.2)
        self.assertEqual(offsetted_series[2].data_loss, 0.3)

        # Test on list data as well
        series = TimeSeries(DataTimePoint(t=60, data=[1.0,-3.0]),
                            DataTimePoint(t=120, data=[2.0,-2.0]),
                            DataTimePoint(t=180, data=[3.0,-1.0]))
        self.assertEqual(series.offset(10)[0].data, [11.0,7.0])
        self.assertEqual(series.offset(10)[1].data, [12.0,8.0])
        self.assertEqual(series.offset(10)[2].data, [13.0,9.0])


    def test_mavg(self):

        series = TimeSeries()
        series.append(DataTimeSlot(start=TimePoint(0), end=TimePoint(1), data={'value':2}, data_loss=0.1))
        series.append(DataTimeSlot(start=TimePoint(1), end=TimePoint(2), data={'value':4}, data_loss=0.2))
        series.append(DataTimeSlot(start=TimePoint(2), end=TimePoint(3), data={'value':8}, data_loss=0.3))
        series.append(DataTimeSlot(start=TimePoint(3), end=TimePoint(4), data={'value':16}, data_loss=0.4))

        mavg_series = mavg(series, 2)

        # Check len and timestamps (the first is skipped due to moving average window of two)
        self.assertEqual(len(mavg_series),3)
        self.assertEqual(mavg_series[0].t, 1)
        self.assertEqual(mavg_series[1].t, 2)
        self.assertEqual(mavg_series[2].t, 3)

        # Check data
        self.assertEqual(mavg_series[0].data['value_mavg_2'], 3)
        self.assertEqual(mavg_series[1].data['value_mavg_2'], 6)
        self.assertEqual(mavg_series[2].data['value_mavg_2'], 12)

        # Check data loss removed (does not make sense in a moving average logic)
        self.assertEqual(mavg_series[0].data_loss, None)
        self.assertEqual(mavg_series[1].data_loss, None)
        self.assertEqual(mavg_series[2].data_loss, None)

        # Test on list data as well
        series = TimeSeries(DataTimePoint(t=60, data=[1.0,-3.0]),
                            DataTimePoint(t=120, data=[2.0,-2.0]),
                            DataTimePoint(t=180, data=[3.0,-1.0]))
        self.assertEqual(len(series.mavg(2)),2)
        self.assertEqual(series.mavg(2)[0].data, [1.5, -2.5])
        self.assertEqual(series.mavg(2)[1].data, [2.5, -1.5])


    def test_min_max_avg(self):

        # Test data
        series = TimeSeries()
        series.append(DataTimeSlot(start=TimePoint(0), end=TimePoint(60), data={'value':10}))
        series.append(DataTimeSlot(start=TimePoint(60), end=TimePoint(120), data={'value':12}))
        series.append(DataTimeSlot(start=TimePoint(120), end=TimePoint(180), data={'value':6}))
        series.append(DataTimeSlot(start=TimePoint(180), end=TimePoint(240), data={'value':16}))

        # Test standalone
        self.assertEqual(min(series), {'value':6})
        self.assertEqual(max(series), {'value':16})
        self.assertEqual(avg(series), {'value':11})

        # Test from the series
        self.assertEqual(series.min(), {'value':6})
        self.assertEqual(series.max(), {'value':16})
        self.assertEqual(series.avg(), {'value':11})

        # Multi data labels Test data
        series = TimeSeries()
        series.append(DataTimePoint(t=0, data={'value':10, 'another_value': 75}))
        series.append(DataTimePoint(t=60, data={'value':12, 'another_value': 65}))
        series.append(DataTimePoint(t=120, data={'value':6, 'another_value': 32}))
        series.append(DataTimePoint(t=180, data={'value':16, 'another_value': 10}))

        self.assertEqual(series.min(), {'value': 6, 'another_value': 10})
        self.assertEqual(series.min(data_label='value'), 6)
        self.assertEqual(series.min(data_label='another_value'), 10)

        self.assertEqual(series.max(), {'value': 16, 'another_value': 75})
        self.assertEqual(series.max(data_label='value'), 16)
        self.assertEqual(series.max(data_label='another_value'), 75)

        self.assertEqual(series.avg(), {'value': 11, 'another_value': 45.5})
        self.assertEqual(series.avg(data_label='value'), 11)
        self.assertEqual(series.avg(data_label='another_value'), 45.5)

        # Test on list data as well
        series = TimeSeries(DataTimePoint(t=60, data=[5.8,29.6]), DataTimePoint(t=120, data=[7.8,18.6]))
        self.assertEqual(series.max(), {'0': 7.8, '1': 29.6})
        self.assertEqual(series.min(), {'0': 5.8, '1': 18.6})
        self.assertEqual(series.avg(), {'0': 6.8, '1': 24.1})


    def test_avg_weighted(self):

        series1 = TimeSeries()
        series2 = TimeSeries()

        point = DataTimePoint(t=-3, data={'value':3})
        point.weight=0
        series1.append(point)
        series2.append(point)

        point = DataTimePoint(t=-2, data={'value':2})
        point.weight=0.125
        series1.append(point)

        point = DataTimePoint(t=-1, data={'value':1})
        point.weight=0.25
        series1.append(point)
        series2.append(point)

        point = DataTimePoint(t=0, data={'value':0})
        point.weight=0.25
        series1.append(point)
        series2.append(point)

        point = DataTimePoint(t=1, data={'value':1})
        point.weight=0.25
        series1.append(point)
        series2.append(point)

        point = DataTimePoint(t=2, data={'value':2})
        point.weight=0.125
        series1.append(point)
        series2.append(point)

        # [DEBUG] timeseria.operations: Point @ 1969-12-31 23:59:57+00:00, weight: 0
        # [DEBUG] timeseria.operations: Point @ 1969-12-31 23:59:58+00:00, weight: 0.125
        # [DEBUG] timeseria.operations: Point @ 1969-12-31 23:59:59+00:00, weight: 0.25
        # [DEBUG] timeseria.operations: Point @ 1970-01-01 00:00:00+00:00, weight: 0.25
        # [DEBUG] timeseria.operations: Point @ 1970-01-01 00:00:01+00:00, weight: 0.25
        # [DEBUG] timeseria.operations: Point @ 1970-01-01 00:00:02+00:00, weight: 0.125

        self.assertEqual(avg(series1), {'value':1})
        self.assertEqual(avg(series2), {'value':0.8571428571428571})


    def test_get(self):
        series = TimeSeries()
        series.append(DataTimePoint(t=0, data={'value':0}))
        series.append(DataTimePoint(t=60, data={'value':60}))
        series.append(DataTimePoint(t=120, data={'value':120}))
        series.append(DataTimePoint(t=180, data={'value':180}))

        # By int (index)
        self.assertEqual(get(series, 0), series[0])
        self.assertEqual(get(series, 1), series[1])
        self.assertEqual(get(series, -1), series[3])

        # By float (epoch timestamp)
        self.assertEqual(get(series, at_t=float(0)), series[0])
        self.assertEqual(get(series, at_t=float(60)), series[1])

        # By datetime
        self.assertEqual(get(series, at_dt=series[0].dt), series[0])
        self.assertEqual(get(series, at_dt=series[1].dt), series[1])


    def test_filter(self):

        # Test data
        series =  TimeSeries(DataTimePoint(t=60, data={'a':1, 'b':2, 'c':3}),
                             DataTimePoint(t=120, data={'a':2, 'b':4, 'c':4}),
                             DataTimePoint(t=180, data={'a':3, 'b':8, 'c':5}))

        # Test get item by string key (filtering on data labels)
        self.assertEqual(filter(series, 'b')[0].data, {'b': 2})

        # Test get item by string key (filtering on data labels), from the series
        self.assertEqual(series.filter('a').data_labels(), ['a'])
        self.assertEqual(len(series.filter('a')), 3 )
        self.assertEqual(series.filter('a')[0].data, {'a': 1})
        self.assertEqual(series.filter('a')[1].data, {'a': 2})
        self.assertEqual(series.filter('a')[2].data, {'a': 3})


        # Test that we haven't modified the original series
        self.assertEqual(series.data_labels(), ['a', 'b', 'c'])
        self.assertEqual(series[1].data['b'], 4)

        # Test for multiple filtering data labels
        self.assertEqual(set(series.filter('a', 'c').data_labels()), {'a', 'c'})
        self.assertEqual(len(series.filter('a', 'c')), 3 )
        self.assertEqual(series.filter('a', 'c')[0].data, {'a': 1,'c':3})
        self.assertEqual(series.filter('a', 'c')[1].data, {'a': 2, 'c':4})
        self.assertEqual(series.filter('a', 'c')[2].data, {'a': 3, 'c':5})

        # Test error when filtering for non key-value  data
        series =  TimeSeries(DataTimePoint(t=60, data=[1,2]),
                             DataTimePoint(t=120, data=[3,4]),
                             DataTimePoint(t=180, data=[5,6]))
        with self.assertRaises(TypeError):
            series.filter('0')


    def test_slice(self):

        series = TimeSeries(TimePoint(t=60),TimePoint(t=120),TimePoint(t=130),TimePoint(t=140),TimePoint(t=150),TimePoint(t=160))

        # Test start/end slicing
        self.assertEqual(len(slice(series, from_t=1.0, to_t=140)),3)

        # Test start/end slicing from the series
        self.assertEqual(len(series.slice(from_t=float(1),to_t=float(140))),3)
        self.assertEqual(len(series.slice(from_t=float(61), to_t=float(140))),2)
        self.assertEqual(len(series.slice(from_t=float(61))),5)
        self.assertEqual(len(series.slice(to_t=float(61))),1)

        # Test no keyword arguments
        self.assertEqual(len(series.slice(from_t=float(61),to_t=float(140))),2)

        # Test with slots
        slot_series = TimeSeries(DataTimeSlot(start=TimePoint(t=60), end=TimePoint(t=120),data=1),
                                 DataTimeSlot(start=TimePoint(t=120), end=TimePoint(t=180),data=1),
                                 DataTimeSlot(start=TimePoint(t=180), end=TimePoint(t=240),data=1),
                                 DataTimeSlot(start=TimePoint(t=240), end=TimePoint(t=300),data=1))
        self.assertEqual(len(slice(slot_series,from_t=float(60),to_t=float(120))),1)
        self.assertEqual(len(slice(slot_series,from_t=float(60),to_t=float(121))),1)
        self.assertEqual(len(slice(slot_series,from_t=float(60),to_t=float(180))),2)

        # Check also time zone
        slot_series.change_tz('Europe/Rome')
        self.assertEqual(slot_series.slice(from_t=float(60),to_t=float(60)).tz, None)
        self.assertEqual(str(slot_series.slice(from_t=float(60),to_t=float(120)).tz), 'Europe/Rome')


    def test_select(self):

        # Test data
        series = TimeSeries()
        series.append(DataTimePoint(t=0, data={'value':10, 'another_value': 75}))
        series.append(DataTimePoint(t=60, data={'value':12, 'another_value': 65}))
        series.append(DataTimePoint(t=120, data={'value':6, 'another_value': 32}))
        series.append(DataTimePoint(t=180, data={'value':16, 'another_value': 10}))

        # Basic select test
        self.assertEqual(select(series, query='another_value=65')[0].t, 60)
        self.assertEqual(series.select('"another_value"=65')[0].t, 60)


    def test_merge(self):

        # Test data
        series1 = TimeSeries()
        series1.append(DataTimePoint(t=0, data={'value':10}, data_indexes={'data_loss':0.1, 'index_a':0.1}))
        series1.append(DataTimePoint(t=60, data={'value':12}, data_indexes={'data_loss':0.2, 'index_a':0.2}))
        series1.append(DataTimePoint(t=120, data={'value':6}, data_indexes={'data_loss':0.3, 'index_a':0.3}))
        series1.append(DataTimePoint(t=180, data={'value':16}, data_indexes={'data_loss':0.4, 'index_a':0.4}))
        series1.append(DataTimePoint(t=240, data={'value':20}, data_indexes={'data_loss':0.5, 'index_a':0.5}))

        series2 = TimeSeries()
        series2.append(DataTimePoint(t=0, data={'another_value': 75}, data_indexes={'data_loss':0.01, 'index_b':0.01}))
        series2.append(DataTimePoint(t=60, data={'another_value': 65}, data_indexes={'data_loss':0.02, 'index_b':0.02}))
        series2.append(DataTimePoint(t=120, data={'another_value': 32}, data_indexes={'data_loss':0.03, 'index_b':0.03}))
        series2.append(DataTimePoint(t=180, data={'another_value': 10}, data_indexes={'data_loss':None, 'index_b':0.04}))
        series2.append(DataTimePoint(t=240, data={'another_value': 7}, data_indexes={'data_loss':None, 'index_b':0.05}))

        series3 = TimeSeries()
        series3.append(DataTimePoint(t=0, data={'another_value': 10}, data_indexes={'data_loss':0.01, 'index_b':0.01}))
        series3.append(DataTimePoint(t=60, data={'another_value': 10}, data_indexes={'data_loss':0.02, 'index_b':0.02}))
        series3.append(DataTimePoint(t=120, data={'another_value': 10}, data_indexes={'data_loss':0.03, 'index_b':0.03}))
        series3.append(DataTimePoint(t=180, data={'another_value': 10}, data_indexes={'data_loss':None, 'index_b':0.04}))
        series3.append(DataTimePoint(t=240, data={'another_value': 10}, data_indexes={'data_loss':None, 'index_b':0.05}))

        series4 = TimeSeries()
        series4.append(DataTimePoint(t=0, data={'value': 10}))
        series4.append(DataTimePoint(t=60, data={'value': 11}))
        series4.append(DataTimePoint(t=120, data={'value': 12}))
        series4.append(DataTimePoint(t=180, data={'value': 13}))
        series4.append(DataTimePoint(t=240, data={'value': 14}))

        # Basic merge
        merged = merge(series1,series2)
        self.assertEqual(len(merged), 5)
        self.assertEqual(merged[0].t, 0)
        self.assertEqual(merged[-1].t, 240)
        self.assertEqual(merged[0].data['value'], 10)
        self.assertEqual(merged[0].data['another_value'], 75)
        self.assertEqual(merged[-1].data['value'], 20)
        self.assertEqual(merged[-1].data['another_value'], 7)

        # Subset merge
        merged = merge(series1[2:4],series2)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0].t, 120)
        self.assertEqual(merged[-1].t, 180)
        self.assertEqual(merged[0].data['value'], 6)
        self.assertEqual(merged[0].data['another_value'], 32)
        self.assertEqual(merged[-1].data['value'], 16)
        self.assertEqual(merged[-1].data['another_value'], 10)

        # Data math
        merged = merge(series2,series3)
        self.assertEqual(len(merged), 5)
        self.assertEqual(merged[0].t, 0)
        self.assertEqual(merged[-1].t, 240)
        self.assertEqual(merged[0].data['another_value'], (75+10)/2)
        self.assertEqual(merged[-1].data['another_value'], (7+10)/2)

        # Data indexes math
        merged = merge(series1,series2)
        self.assertEqual(merged[0].data_indexes, {'data_loss': (0.1+0.01)/2, 'index_a': 0.1, 'index_b': 0.01})
        self.assertEqual(merged[1].data_indexes, {'data_loss': (0.2+0.02)/2, 'index_a': 0.2, 'index_b': 0.02})
        self.assertEqual(merged[2].data_indexes, {'data_loss': (0.3+0.03)/2, 'index_a': 0.3, 'index_b': 0.03})
        self.assertEqual(merged[3].data_indexes, {'data_loss': 0.4, 'index_a': 0.4, 'index_b': 0.04})
        self.assertEqual(merged[4].data_indexes, {'data_loss': 0.5, 'index_a': 0.5, 'index_b': 0.05})

        # Merge different data types preserving them
        merged = merge(series3,series4)
        self.assertTrue(isinstance(merged[0].data['value'], int))
        self.assertTrue(isinstance(merged[1].data['value'], int))
        self.assertTrue(isinstance(merged[2].data['value'], int))
        self.assertTrue(isinstance(merged[3].data['value'], int))
        self.assertTrue(isinstance(merged[4].data['value'], int))

