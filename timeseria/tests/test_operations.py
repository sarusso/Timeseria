import unittest
import os
from ..datastructures import TimePoint, DataTimePoint, DataTimeSlotSeries, DataTimeSlot
from ..datastructures import TimePointSeries, DataTimePointSeries
from ..operations import derivative, integral, diff, csum, min, max, avg, filter, select, mavg
from ..operations import Operation

# Setup logging
import logging
logging.basicConfig(level=os.environ.get('LOGLEVEL') if os.environ.get('LOGLEVEL') else 'CRITICAL')

class TestBaseOpertions(unittest.TestCase):
    
    def test_base(self):
        
        operation_from_callable = Operation()
        self.assertEqual(operation_from_callable.__name__, 'operation')
                
        def operation_as_function(data):
            pass
        
        self.assertEqual(operation_as_function.__name__, 'operation_as_function')
        

class TestMathOperations(unittest.TestCase):
  
    def test_diff_csum(self):
  
        # Test data        
        data_time_slot_series = DataTimeSlotSeries()
        data_time_slot_series.append(DataTimeSlot(start=TimePoint(0), end=TimePoint(60), data={'value':10}))
        data_time_slot_series.append(DataTimeSlot(start=TimePoint(60), end=TimePoint(120), data={'value':12}))
        data_time_slot_series.append(DataTimeSlot(start=TimePoint(120), end=TimePoint(180), data={'value':15}))
        data_time_slot_series.append(DataTimeSlot(start=TimePoint(180), end=TimePoint(240), data={'value':16}))
           
        # Test standard, from the series
        diff_data_time_slot_series = diff(data_time_slot_series)
        self.assertEqual(len(diff_data_time_slot_series), 4)
        self.assertEqual(diff_data_time_slot_series[0].data['value_diff'],0) # 2?
        self.assertEqual(diff_data_time_slot_series[1].data['value_diff'],2)
        self.assertEqual(diff_data_time_slot_series[2].data['value_diff'],3)
        self.assertEqual(diff_data_time_slot_series[3].data['value_diff'],1)

        # Test csum as well  
        diff_csum_data_time_slot_series = csum(diff_data_time_slot_series, offset=10)
        self.assertEqual(diff_csum_data_time_slot_series[0].data['value_diff_csum'], 10)
        self.assertEqual(diff_csum_data_time_slot_series[1].data['value_diff_csum'], 12)
        self.assertEqual(diff_csum_data_time_slot_series[2].data['value_diff_csum'], 15)
        self.assertEqual(diff_csum_data_time_slot_series[3].data['value_diff_csum'], 16)
        
        # Test standalone
        self.assertEqual(len(diff(data_time_slot_series)), 4)
  
        # Test in-place  behavior
        diff(data_time_slot_series, inplace=True)
        self.assertEqual(len(data_time_slot_series), 4)
        self.assertEqual(data_time_slot_series[0].data['value'],10)
        self.assertEqual(data_time_slot_series[0].data['value_diff'],0)
        self.assertEqual(data_time_slot_series[1].data['value'],12)
        self.assertEqual(data_time_slot_series[1].data['value_diff'],2)
        self.assertEqual(data_time_slot_series[2].data['value'],15)
        self.assertEqual(data_time_slot_series[2].data['value_diff'],3)
        self.assertEqual(data_time_slot_series[3].data['value'],16)
        self.assertEqual(data_time_slot_series[3].data['value_diff'],1)

        # Multi-key Test data
        data_time_point_series = DataTimePointSeries()
        data_time_point_series.append(DataTimePoint(t=0, data={'value':10, 'another_value': 75}))
        data_time_point_series.append(DataTimePoint(t=60, data={'value':12, 'another_value': 65}))
        data_time_point_series.append(DataTimePoint(t=120, data={'value':6, 'another_value': 32}))
        data_time_point_series.append(DataTimePoint(t=180, data={'value':16, 'another_value': 10}))
        diff_data_time_point_series = data_time_point_series.diff()
        self.assertEqual(diff_data_time_point_series[0].data['value_diff'],0)
        self.assertEqual(diff_data_time_point_series[0].data['another_value_diff'],0)
        self.assertEqual(diff_data_time_point_series[1].data['value_diff'],2)
        self.assertEqual(diff_data_time_point_series[1].data['another_value_diff'],-10)

        # Test offset
        diff_csum_data_time_point_series = diff_data_time_point_series.csum(offset={'value_diff':10, 'another_value_diff':75})
        self.assertEqual(diff_csum_data_time_point_series[0].data['value_diff_csum'],10)
        self.assertEqual(diff_csum_data_time_point_series[0].data['another_value_diff_csum'],75)
        self.assertEqual(diff_csum_data_time_point_series[3].data['value_diff_csum'],16)
        self.assertEqual(diff_csum_data_time_point_series[3].data['another_value_diff_csum'],10)


    def test_derivative_integral(self):
        
        # Test data        
        data_time_slot_series = DataTimeSlotSeries()
        data_time_slot_series.append(DataTimeSlot(start=TimePoint(0), end=TimePoint(1), data={'value':10}))
        data_time_slot_series.append(DataTimeSlot(start=TimePoint(1), end=TimePoint(2), data={'value':12}))
        data_time_slot_series.append(DataTimeSlot(start=TimePoint(2), end=TimePoint(3), data={'value':15}))
        data_time_slot_series.append(DataTimeSlot(start=TimePoint(3), end=TimePoint(4), data={'value':16}))
        
        # Test standard derivativeative behavior, from the series
        derivative_data_time_slot_series = derivative(data_time_slot_series)
        self.assertEqual(len(derivative_data_time_slot_series), 4)
        self.assertAlmostEqual(derivative_data_time_slot_series[0].data['value_derivative'],2)     # 0
        self.assertAlmostEqual(derivative_data_time_slot_series[1].data['value_derivative'],2.5)   # 2
        self.assertAlmostEqual(derivative_data_time_slot_series[2].data['value_derivative'],2)     # 5
        self.assertAlmostEqual(derivative_data_time_slot_series[3].data['value_derivative'],1.0)   # 6

        # Test integral as well  
        derivative_integral_data_time_slot_series = integral(derivative_data_time_slot_series, c=10)
        self.assertEqual(derivative_integral_data_time_slot_series[0].data['value_derivative_integral'], 10)
        self.assertEqual(derivative_integral_data_time_slot_series[1].data['value_derivative_integral'], 12)
        self.assertEqual(derivative_integral_data_time_slot_series[2].data['value_derivative_integral'], 15)
        self.assertEqual(derivative_integral_data_time_slot_series[3].data['value_derivative_integral'], 16)
        
        # Test data        
        data_time_slot_series = DataTimeSlotSeries()
        data_time_slot_series.append(DataTimeSlot(start=TimePoint(0), end=TimePoint(10), data={'value':10}))
        data_time_slot_series.append(DataTimeSlot(start=TimePoint(10), end=TimePoint(20), data={'value':12}))
        data_time_slot_series.append(DataTimeSlot(start=TimePoint(20), end=TimePoint(30), data={'value':15}))
        data_time_slot_series.append(DataTimeSlot(start=TimePoint(30), end=TimePoint(40), data={'value':16}))
        
        # Test standard derivativeative behavior, from the series
        derivative_data_time_slot_series = derivative(data_time_slot_series)
        self.assertEqual(len(derivative_data_time_slot_series), 4)
        self.assertAlmostEqual(derivative_data_time_slot_series[0].data['value_derivative'],0.2)
        self.assertAlmostEqual(derivative_data_time_slot_series[1].data['value_derivative'],0.25)
        self.assertAlmostEqual(derivative_data_time_slot_series[2].data['value_derivative'],0.2)
        self.assertAlmostEqual(derivative_data_time_slot_series[3].data['value_derivative'],0.1)        

        # Multi-key Test data
        data_time_point_series = DataTimePointSeries()
        data_time_point_series.append(DataTimePoint(t=0, data={'value':10, 'another_value': 75}))
        data_time_point_series.append(DataTimePoint(t=10, data={'value':12, 'another_value': 65}))
        data_time_point_series.append(DataTimePoint(t=20, data={'value':6, 'another_value': 32}))
        data_time_point_series.append(DataTimePoint(t=30, data={'value':16, 'another_value': 10}))
        derivative_data_time_point_series = data_time_point_series.derivative()
        self.assertAlmostEqual(derivative_data_time_point_series[0].data['value_derivative'],0.2)
        self.assertAlmostEqual(derivative_data_time_point_series[0].data['another_value_derivative'],-1.0)
        self.assertAlmostEqual(derivative_data_time_point_series[1].data['value_derivative'],-0.2)
        self.assertAlmostEqual(derivative_data_time_point_series[1].data['another_value_derivative'],-2.15)

        # Test derivative-integral identity with ther data     
        data_time_slot_series = DataTimeSlotSeries()
        data_time_slot_series.append(DataTimeSlot(start=TimePoint(0), end=TimePoint(1), data={'value':0}))
        data_time_slot_series.append(DataTimeSlot(start=TimePoint(1), end=TimePoint(2), data={'value':6}))
        data_time_slot_series.append(DataTimeSlot(start=TimePoint(2), end=TimePoint(3), data={'value':14}))
        data_time_slot_series.append(DataTimeSlot(start=TimePoint(3), end=TimePoint(4), data={'value':16}))
        data_time_slot_series.append(DataTimeSlot(start=TimePoint(4), end=TimePoint(5), data={'value':20}))
        
        derivative_data_time_slot_series = data_time_slot_series.derivative()
        derivative_integral_data_time_slot_series = derivative_data_time_slot_series.integral()

        self.assertAlmostEqual(derivative_integral_data_time_slot_series[0].data['value_derivative_integral'],0)
        self.assertAlmostEqual(derivative_integral_data_time_slot_series[1].data['value_derivative_integral'],6)
        self.assertAlmostEqual(derivative_integral_data_time_slot_series[2].data['value_derivative_integral'],14)
        self.assertAlmostEqual(derivative_integral_data_time_slot_series[3].data['value_derivative_integral'],16)
        self.assertAlmostEqual(derivative_integral_data_time_slot_series[4].data['value_derivative_integral'],20)


    def test_mavg(self):

        data_time_slot_series = DataTimeSlotSeries()
        data_time_slot_series.append(DataTimeSlot(start=TimePoint(0), end=TimePoint(1), data={'value':2}))
        data_time_slot_series.append(DataTimeSlot(start=TimePoint(1), end=TimePoint(2), data={'value':4}))
        data_time_slot_series.append(DataTimeSlot(start=TimePoint(2), end=TimePoint(3), data={'value':8}))
        data_time_slot_series.append(DataTimeSlot(start=TimePoint(3), end=TimePoint(4), data={'value':16}))
  
        mavg_data_time_slot_series = mavg(data_time_slot_series, 2)
        self.assertEqual(mavg_data_time_slot_series[0].data['value_mavg_2'], 3)
        self.assertEqual(mavg_data_time_slot_series[1].data['value_mavg_2'], 6)
        self.assertEqual(mavg_data_time_slot_series[2].data['value_mavg_2'], 12)
        self.assertEqual(mavg_data_time_slot_series[0].t, 1)
        self.assertEqual(mavg_data_time_slot_series[1].t, 2)
        self.assertEqual(mavg_data_time_slot_series[2].t, 3)


    def test_min_max_avg(self):
        
        # Test Python built-in default behaviour
        self.assertEqual(str(min), 'Min operation')
        self.assertEqual(min([1,2,3]),1)
        self.assertEqual(str(max), 'Max operation')
        self.assertEqual(max([1,2,3]),3)
        
        # Test data
        data_time_slot_series = DataTimeSlotSeries()
        data_time_slot_series.append(DataTimeSlot(start=TimePoint(0), end=TimePoint(60), data={'value':10}))
        data_time_slot_series.append(DataTimeSlot(start=TimePoint(60), end=TimePoint(120), data={'value':12}))
        data_time_slot_series.append(DataTimeSlot(start=TimePoint(120), end=TimePoint(180), data={'value':6}))
        data_time_slot_series.append(DataTimeSlot(start=TimePoint(180), end=TimePoint(240), data={'value':16}))
        
        # Test standalone
        self.assertEqual(min(data_time_slot_series), 6)
        self.assertEqual(max(data_time_slot_series), 16)
        self.assertEqual(avg(data_time_slot_series), 11)
        
        # Test from the series
        self.assertEqual(data_time_slot_series.min(), 6)
        self.assertEqual(data_time_slot_series.max(), 16)
        self.assertEqual(data_time_slot_series.avg(), 11)

        # Multi-key Test data
        data_time_point_series = DataTimePointSeries()
        data_time_point_series.append(DataTimePoint(t=0, data={'value':10, 'another_value': 75}))
        data_time_point_series.append(DataTimePoint(t=60, data={'value':12, 'another_value': 65}))
        data_time_point_series.append(DataTimePoint(t=120, data={'value':6, 'another_value': 32}))
        data_time_point_series.append(DataTimePoint(t=180, data={'value':16, 'another_value': 10}))
        
        self.assertEqual(data_time_point_series.min(), {'value': 6, 'another_value': 10})
        self.assertEqual(data_time_point_series.min(data_key='value'), 6)
        self.assertEqual(data_time_point_series.min(data_key='another_value'), 10)

        self.assertEqual(data_time_point_series.max(), {'value': 16, 'another_value': 75})
        self.assertEqual(data_time_point_series.max(data_key='value'), 16)        
        self.assertEqual(data_time_point_series.max(data_key='another_value'), 75)

        self.assertEqual(data_time_point_series.avg(), {'value': 11, 'another_value': 45.5})
        self.assertEqual(data_time_point_series.avg(data_key='value'), 11)          
        self.assertEqual(data_time_point_series.avg(data_key='another_value'), 45.5)


class TestSeriesOperations(unittest.TestCase):
    
    def test_filter(self):

        test_time_point_series = TimePointSeries(TimePoint(t=60),TimePoint(t=120),TimePoint(t=130))

        # Test from/to filtering
        self.assertEqual(len(filter(test_time_point_series, from_t=1, to_t=140)),3)

        # Test from/to filtering from the series
        self.assertEqual(len(test_time_point_series.filter(from_t=1, to_t=140)),3)
        self.assertEqual(len(test_time_point_series.filter(from_t=61, to_t=140)),2)
        self.assertEqual(len(test_time_point_series.filter(from_t=61)),2)
        self.assertEqual(len(test_time_point_series.filter(to_t=61)),1)

        # Test  data
        data_time_point_series =  DataTimePointSeries(DataTimePoint(t=60, data={'a':1, 'b':2}),
                                                      DataTimePoint(t=120, data={'a':2, 'b':4}),
                                                      DataTimePoint(t=180, data={'a':3, 'b':8}))

        # Test get item by string key (filtering on data labels)
        self.assertEqual(filter(data_time_point_series, 'b')[0].data, {'b': 2})

        # Test get item by string key (filtering on data labels), from the series
        self.assertEqual(data_time_point_series.filter('a')[0].data, {'a': 1})
        self.assertEqual(data_time_point_series.filter('a')[1].data, {'a': 2})
        self.assertEqual(data_time_point_series.filter('a')[2].data, {'a': 3})
        self.assertEqual(len(data_time_point_series.filter('a')), 3 )
        
        # Test that we haven't modified the original series
        self.assertEqual(data_time_point_series.data_keys(), ['a', 'b'])
        self.assertEqual(data_time_point_series[1].data['b'], 4)


    def test_select(self):
    
        # Test data
        data_time_point_series = DataTimePointSeries()
        data_time_point_series.append(DataTimePoint(t=0, data={'value':10, 'another_value': 75}))
        data_time_point_series.append(DataTimePoint(t=60, data={'value':12, 'another_value': 65}))
        data_time_point_series.append(DataTimePoint(t=120, data={'value':6, 'another_value': 32}))
        data_time_point_series.append(DataTimePoint(t=180, data={'value':16, 'another_value': 10}))
        
        # Basic select test        
        self.assertEqual(select(data_time_point_series, query='another_value=65')[0].t, 60)
        self.assertEqual(data_time_point_series.select('"another_value"=65')[0].t, 60)



