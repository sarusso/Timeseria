import unittest
import os
from ..datastructures import Point, TimePoint, DataPoint, DataTimePoint, DataTimeSlotSeries, DataTimeSlot
from ..datastructures import Series, DataPointSeries, TimePointSeries, DataTimePointSeries
from ..storages import CSVFileStorage
#from ..operators import diff, min, slot, Slotter,
from ..operators import diff, min, derivative 
from ..time import dt, s_from_dt, dt_from_s, dt_from_str
from ..exceptions import InputException
from ..units import TimeUnit

# Setup logging
import logging
logging.basicConfig(level=os.environ.get('LOGLEVEL') if os.environ.get('LOGLEVEL') else 'CRITICAL')

# Set test data path
TEST_DATA_PATH = '/'.join(os.path.realpath(__file__).split('/')[0:-1]) + '/test_data/'


class TestDiff(unittest.TestCase):
  
    def test_call(self):
  
        # Test data        
        data_time_slot_series = DataTimeSlotSeries()
        data_time_slot_series.append(DataTimeSlot(start=TimePoint(0), end=TimePoint(60), data={'value':10}))
        data_time_slot_series.append(DataTimeSlot(start=TimePoint(60), end=TimePoint(120), data={'value':12}))
        data_time_slot_series.append(DataTimeSlot(start=TimePoint(120), end=TimePoint(180), data={'value':15}))
        data_time_slot_series.append(DataTimeSlot(start=TimePoint(180), end=TimePoint(240), data={'value':16}))
           
        # Test standard
        diff_data_time_slot_series = diff(data_time_slot_series)
        self.assertEqual(len(diff_data_time_slot_series), 4)
        self.assertEqual(diff_data_time_slot_series[0].data['value_diff'],1.0)
        self.assertEqual(diff_data_time_slot_series[1].data['value_diff'],2.5)
        self.assertEqual(diff_data_time_slot_series[2].data['value_diff'],2.0)
        self.assertEqual(diff_data_time_slot_series[3].data['value_diff'],0.5)
  
        # Test in-place  behavior
        diff(data_time_slot_series, inplace=True)
        self.assertEqual(len(data_time_slot_series), 4)
        self.assertEqual(data_time_slot_series[0].data['value'],10)
        self.assertEqual(data_time_slot_series[0].data['value_diff'],1.0)
        self.assertEqual(data_time_slot_series[1].data['value'],12)
        self.assertEqual(data_time_slot_series[1].data['value_diff'],2.5)
        self.assertEqual(data_time_slot_series[2].data['value'],15)
        self.assertEqual(data_time_slot_series[2].data['value_diff'],2.0)
        self.assertEqual(data_time_slot_series[3].data['value'],16)
        self.assertEqual(data_time_slot_series[3].data['value_diff'],0.5)
 
         
         
class TestMin(unittest.TestCase):
  
    def test_call(self):        
        self.assertEqual(str(min), 'Min operator')
        self.assertEqual(min([1,2,3]),1)
 
 
# class TestDerivative(unittest.TestCase):
#  
#  
#     def test_call(self):
#  
#         # Test data        
#         data_time_slot_series = DataTimeSlotSeries()
#         data_time_slot_series.append(DataTimeSlot(start=TimePoint(0), end=TimePoint(60), data={'value':10}))
#         data_time_slot_series.append(DataTimeSlot(start=TimePoint(60), end=TimePoint(120), data={'value':12}))
#         data_time_slot_series.append(DataTimeSlot(start=TimePoint(120), end=TimePoint(180), data={'value':15}))
#         data_time_slot_series.append(DataTimeSlot(start=TimePoint(180), end=TimePoint(240), data={'value':16}))
#            
#         # Test standard
#         derivativeivative_data_time_slot_series = derivative(data_time_slot_series)
#         self.assertEqual(len(derivativeivative_data_time_slot_series), 4)
#         self.assertEqual(derivativeivative_data_time_slot_series[0].data['value_derivative'],2.0)
#         self.assertEqual(derivativeivative_data_time_slot_series[1].data['value_derivative'],2.5)
#         self.assertEqual(derivativeivative_data_time_slot_series[2].data['value_derivative'],2.0)
#         self.assertEqual(derivativeivative_data_time_slot_series[3].data['value_derivative'],1.0)
#  
#         # Test in-place and incremental behavior
#         derivative(data_time_slot_series, inplace=True, incremental=True)
#         self.assertEqual(len(data_time_slot_series), 4)
#         self.assertEqual(data_time_slot_series[0].data['value'],10)
#         self.assertEqual(data_time_slot_series[0].data['value_derivative'],1.0)
#         self.assertEqual(data_time_slot_series[1].data['value'],12)
#         self.assertEqual(data_time_slot_series[1].data['value_derivative'],2.5)
#         self.assertEqual(data_time_slot_series[2].data['value'],15)
#         self.assertEqual(data_time_slot_series[2].data['value_derivative'],2.0)
#         self.assertEqual(data_time_slot_series[3].data['value'],16)
#         self.assertEqual(data_time_slot_series[3].data['value_derivative'],0.5)










