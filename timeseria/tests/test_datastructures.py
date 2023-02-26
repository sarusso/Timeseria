import unittest
import datetime
import os
import pandas as pd

from ..datastructures import Point, TimePoint, DataPoint, DataTimePoint
from ..datastructures import Slot, TimeSlot, DataSlot, DataTimeSlot
from ..datastructures import Series, TimeSeries, _TimeSeriesView
from ..time import UTC, dt
from ..units import Unit, TimeUnit

# Setup logging
from .. import logger
logger.setup()

# Set test data path
TEST_DATA_PATH = '/'.join(os.path.realpath(__file__).split('/')[0:-1]) + '/test_data/'


class TestPoints(unittest.TestCase):

    def test_Point(self):
        
        point = Point(1, 2)        
        self.assertEqual(point.coordinates[0],1)
        self.assertEqual(point.coordinates[1],2)
        
        # Access by index
        self.assertEqual(point[0],1)
        self.assertEqual(point[1],2)

        point_1 = Point(1, 2)
        point_2 = Point(1, 2)
        point_3 = Point(5, 3)

        self.assertEqual(point_1,point_2)
        self.assertNotEqual(point_1,point_3)


    def test_TimePoint(self):
        
        # Wrong init/casting
        with self.assertRaises(Exception):
            TimePoint()
        
        with self.assertRaises(Exception):
            TimePoint('hi')

        # Standard init (init/float)
        time_point = TimePoint(5)
        self.assertEqual(time_point.coordinates,(5,))
        self.assertEqual(time_point.coordinates[0],5)
        self.assertEqual(time_point.t,5)
        
        # Init using "t"
        time_point_init_t = TimePoint(t=5)
        self.assertEqual(time_point, time_point_init_t)
        
        # Init using naive "dt" (only in case someone uses without using timesera dt utility, i.e. from external timestamps) 
        time_point_init_dt = TimePoint(dt=datetime.datetime(1970,1,1,0,0,5))
        self.assertEqual(time_point_init_dt, time_point)
        self.assertEqual(str(time_point_init_dt.tz), 'UTC')

        # Init using "dt" with UTC timezone (timesera dt utility always add the UTC timezone)
        time_point_init_dt = TimePoint(dt=dt(1970,1,1,0,0,5))
        self.assertEqual(time_point_init_dt, time_point) 
        self.assertEqual(str(time_point_init_dt.tz), 'UTC')

        # Init using "dt" with Europe/Rome timezone (note that we create the dt at 1 AM and not at midnight for this test to pass) 
        time_point_init_dt = TimePoint(dt=dt(1970,1,1,1,0,5, tz='Europe/Rome'))
        self.assertEqual(time_point_init_dt, time_point_init_dt)         
        
        # Test standard with UTC timezone
        self.assertEqual(time_point.tz, UTC)
        self.assertEqual(type(time_point.tz), type(UTC))
        self.assertEqual(str(time_point.dt), '1970-01-01 00:00:05+00:00')

        # Test standard with Europe/Rome timezone
        time_point = TimePoint(t=1569897900, tz='Europe/Rome')
        self.assertEqual(str(time_point.tz), 'Europe/Rome')
        self.assertEqual(str(type(time_point.tz)), "<class 'pytz.tzfile.Europe/Rome'>")
        self.assertEqual(str(time_point.dt), '2019-10-01 04:45:00+02:00')

        # Cast from object extending the TimePoint
        class ExtendedTimePoint(TimePoint):
            pass
        extended_time_point = ExtendedTimePoint(10)
        time_point_catsted = TimePoint(extended_time_point)
        self.assertEqual(time_point_catsted.coordinates,(10,))


    def test_DataPoint(self):
        
        with self.assertRaises(Exception):
            DataPoint(data='hello')
        
        data_point = DataPoint(1, 2, data='hello')

        self.assertEqual(data_point.coordinates, (1,2)) 
        self.assertEqual(data_point.coordinates[0], 1)
        self.assertEqual(data_point.coordinates[1], 2)
        self.assertEqual(data_point.data,'hello')
        
        data_point_1 = DataPoint(1, 2, data='hello')
        data_point_2 = DataPoint(1, 2, data='hello')
        data_point_3 = DataPoint(1, 2, data='hola')
        
        self.assertEqual(data_point_1, data_point_2)
        self.assertNotEqual(data_point_1, data_point_3)
        
        # Test list and dict data labels
        data_point = DataPoint(1, 2, data=['hello', 'hola'])
        self.assertEqual(data_point.data_labels(), ['0','1'])
        
        data_point = DataPoint(1, 2, data={'label1':'hello', 'label2':'hola'})
        self.assertEqual(data_point.data_labels(), ['label1','label2'])

        # Test with data loss index
        data_point = DataPoint(1, 2, data='hello', data_loss=0.5)
        self.assertEqual(data_point.data_loss,0.5)
        self.assertEqual(data_point.data_indexes['data_loss'],0.5)
        
        # Test with generic indexes
        data_point = DataPoint(1, 2, data='hello', data_indexes={'data_loss':0.5, 'my_index':0.3})
        self.assertEqual(data_point.data_loss,0.5)
        self.assertEqual(data_point.data_indexes['data_loss'],0.5)
        self.assertEqual(data_point.data_indexes['my_index'],0.3)

        # Test that a data index with None value is okay
        DataPoint(1, 2, data='hello', data_indexes={'data_loss':None, 'my_index':0.3})

                 
    def test_DataTimePoint(self):
        
        with self.assertRaises(Exception):
            DataTimePoint(x=1)

        with self.assertRaises(Exception):
            DataTimePoint(data='hello')

        with self.assertRaises(Exception):
            DataTimePoint(x=1, data='hello')

        data_time_point = DataTimePoint(t=6, data='hello')

        self.assertEqual(data_time_point.coordinates, (6,)) 
        self.assertEqual(data_time_point.t,6)
        self.assertEqual(data_time_point.data,'hello')
    
    
    def test_casting(self):
        data_time_point = DataTimePoint(t=6, data='hello')
        casted_data_time_point_1 = TimePoint(data_time_point)
        self.assertEqual(casted_data_time_point_1.t, 6) 
        casted_data_time_point_2 = TimePoint(5)
        self.assertEqual(casted_data_time_point_2.t, 5) 


    def test_with_Unit(self):      
        self.assertEqual( (Point(1) + Unit(5)).coordinates[0], 6) 
        self.assertEqual( (TimePoint(1) + Unit(5)).coordinates[0], 6) 



class TestSlots(unittest.TestCase):

    def test_Slot(self):

        # A slot needs a start and an end
        with self.assertRaises(TypeError):
            Slot()
        with self.assertRaises(TypeError):
            Slot(start=Point(x=1))

        # A slot needs a start and and end of Point instance
        with self.assertRaises(TypeError):
            Slot(start=1, end=2) 

        # A slot needs a start and and end with the same dimension
        with self.assertRaises(ValueError):
            Slot(start=Point(1), end=Point(2,4))        

        # No zero-duration allowed:
        with self.assertRaises(ValueError):
            Slot(start=Point(1), end=Point(1))

        slot = Slot(start=Point(1), end=Point(2)) 
        self.assertEqual(slot.start,Point(1))
        self.assertEqual(slot.end,Point(2))
        
        slot_1 = Slot(start=Point(1), end=Point(2))
        slot_2 = Slot(start=Point(1), end=Point(2))
        slot_3 = Slot(start=Point(1), end=Point(3))
        
        self.assertEqual(slot_1,slot_2)
        self.assertNotEqual(slot_1,slot_3)
        
        # Length
        slot = Slot(start=Point(1.5), end=Point(4.7)) 
        self.assertEqual(slot.length, 3.2)

        # Unit
        slot = Slot(start=Point(1.5), unit=3.2) 
        self.assertEqual(slot.end, Point(4.7))

        slot = Slot(start=Point(1.5), unit=Unit(3.2)) 
        self.assertEqual(slot.end, Point(4.7))

        slot = Slot(start=Point(1.5), end=Point(3.0)) 
        self.assertEqual(slot.unit, Unit(1.5))

  
    def test_TimeSlot(self):

        # A time_slot needs TimePoints
        with self.assertRaises(TypeError):
            TimeSlot(start=Point(x=1), end=Point(x=2))

        # No zero-duration allowed:
        with self.assertRaises(ValueError):
            TimeSlot(start=TimePoint(t=1), end=TimePoint(t=1))

        time_slot = TimeSlot(start=TimePoint(t=1), end=TimePoint(t=2))
        self.assertEqual(time_slot.start.t,1)
        self.assertEqual(time_slot.end.t,2)

        time_slot_1 = TimeSlot(start=TimePoint(t=1), end=TimePoint(t=2))
        time_slot_2 = TimeSlot(start=TimePoint(t=2), end=TimePoint(t=3))
        time_slot_3 = TimeSlot(start=TimePoint(t=3), end=TimePoint(t=4))

        # Test succession
        self.assertTrue(time_slot_2.__succedes__(time_slot_1))
        self.assertFalse(time_slot_1.__succedes__(time_slot_2))
        self.assertFalse(time_slot_3.__succedes__(time_slot_1))
        
        # Length
        self.assertEqual(time_slot_1.length,1)

        # Timezone
        with self.assertRaises(ValueError):
            # ValueError: TimeSlot start and end must have the same timezone (got start.tz="Europe/Rome", end.tz="UTC")
            TimeSlot(start=TimePoint(t=0, tz='Europe/Rome'), end=TimePoint(t=60))
        TimeSlot(start=TimePoint(t=0, tz='Europe/Rome'), end=TimePoint(t=60, tz='Europe/Rome'))

        # Slot t and dt shortcuts
        slot = TimeSlot(start=TimePoint(60), end=TimePoint(120))
        self.assertEqual(slot.t, 60)
        self.assertEqual(slot.dt, dt(1970,1,1,0,1,0)) # 1 minute past midnight Jan 1970 UTC
        
        slot = TimeSlot(t=60, unit=60)
        self.assertEqual(slot.start, TimePoint(60))
        self.assertEqual(slot.end, TimePoint(120))
        self.assertEqual(slot.unit, 60)

        slot = TimeSlot(dt=dt(1970,1,1,0,1,0), unit=TimeUnit('1m'))
        self.assertEqual(slot.start, TimePoint(60))
        self.assertEqual(slot.end, TimePoint(120))
        self.assertEqual(slot.unit, TimeUnit('1m'))
        

    def test_DataSlot(self):
        
        with self.assertRaises(Exception):
            DataSlot(start=Point(x=1), end=Point(x=2))
 
        with self.assertRaises(TypeError):
            DataSlot(data='hello')

        data_slot = DataSlot(start=Point(1), end=Point(2), data='hello')
        self.assertEqual(data_slot.start.coordinates[0],1)
        self.assertEqual(data_slot.end.coordinates[0],2)
        self.assertEqual(data_slot.data,'hello')

        data_slot_1 = DataSlot(start=Point(1), end=Point(2), data='hello')
        data_slot_2 = DataSlot(start=Point(1), end=Point(2), data='hello')
        data_slot_3 = DataSlot(start=Point(1), end=Point(2), data='hola')

        self.assertEqual(data_slot_1, data_slot_2)
        self.assertNotEqual(data_slot_1, data_slot_3)

        data_slot_with_data_loss = DataSlot(start=Point(1), end=Point(2), data='hello', data_loss=0.98)
        self.assertEqual(data_slot_with_data_loss.data_loss,0.98)

        # Test list and dict data labels
        data_slot = DataSlot(start=Point(1), end=Point(2), data=['hello', 'hola'])
        self.assertEqual(data_slot.data_labels(), ['0','1'])
        
        data_slot = DataSlot(start=Point(1), end=Point(2), data={'label1':'hello', 'label2':'hola'})
        self.assertEqual(data_slot.data_labels(), ['label1','label2'])

        # Test with data loss index
        data_slot = DataSlot(start=Point(1), end=Point(2), data='hello', data_loss=0.5)
        self.assertEqual(data_slot.data_loss,0.5)
        self.assertEqual(data_slot.data_indexes['data_loss'],0.5)
        
        # Test with generic indexes
        data_slot = DataSlot(start=Point(1), end=Point(2), data='hello', data_indexes={'data_loss':0.5, 'my_index':0.3})
        self.assertEqual(data_slot.data_loss,0.5)
        self.assertEqual(data_slot.data_indexes['data_loss'],0.5)
        self.assertEqual(data_slot.data_indexes['my_index'],0.3)
        
        # Test that a data index with None value is okay
        DataSlot(start=Point(1), end=Point(2), data='hello', data_indexes={'data_loss':None, 'my_index':0.3})

    
    def test_DataTimeSlots(self):

        with self.assertRaises(Exception):
            DataTimeSlot(start=Point(x=1), end=Point(x=2))
 
        with self.assertRaises(TypeError):
            DataTimeSlot(data='hello')

        with self.assertRaises(TypeError):
            DataTimeSlot(start=Point(x=1), end=Point(x=2), data='hello')

        data_time_slot_ = DataTimeSlot(start=TimePoint(t=1), end=TimePoint(t=2), data='hello')
        self.assertEqual(data_time_slot_.start.t,1)
        self.assertEqual(data_time_slot_.end.t,2)
        self.assertEqual(data_time_slot_.data,'hello')

        data_time_slot_1 = DataTimeSlot(start=TimePoint(t=1), end=TimePoint(t=2), data='hello')
        data_time_slot_2 = DataTimeSlot(start=TimePoint(t=1), end=TimePoint(t=2), data='hello')
        data_time_slot_3 = DataTimeSlot(start=TimePoint(t=1), end=TimePoint(t=2), data='hola')

        self.assertEqual(data_time_slot_1, data_time_slot_2)
        self.assertNotEqual(data_time_slot_1, data_time_slot_3)


class TestSeries(unittest.TestCase):

    def test_Series(self):
        
        # Test basic behaviour
        series = Series(1,4,5)
        self.assertEqual(series, [1,4,5])
        self.assertEqual(len(series), 3)
        
        # Test append
        series.append(8)
        self.assertEqual(series, [1,4,5,8])
        self.assertEqual(len(series), 4)  

        # Test unordered append
        with self.assertRaises(ValueError):
            series.append(2)

        # Test pop
        self.assertEqual(series.pop(), 8)
        self.assertEqual(series.pop(1), 4)
        self.assertEqual(series.contents(), [1,5])
        
        # Test insert    
        series.insert(1,3)
        self.assertEqual(series.contents(), [1,3,5])

        # Test unordered insert    
        with self.assertRaises(ValueError):
            series.insert(0,3)
        with self.assertRaises(ValueError):
            series.insert(1,1)        

        # Test remove
        series.remove(3)
        self.assertEqual(series.contents(), [1,5])

        with self.assertRaises(ValueError):
            series.remove(18)

        # Demo class that implements the __succedes__ operation
        class IntegerNumber(int):
            def __succedes__(self, other):
                if other+1 != self:
                    return False
                else:
                    return True
        
        zero = IntegerNumber(0)    
        one = IntegerNumber(1)
        two = IntegerNumber(2)
        three = IntegerNumber(3)
        four = IntegerNumber(4)

        self.assertTrue(two.__succedes__(one))
        self.assertFalse(one.__succedes__(two))
        self.assertFalse(three.__succedes__(one))

        with self.assertRaises(ValueError):
            Series(one, three)

        with self.assertRaises(ValueError):
            Series(three, two)
         
        # TODO: do we want the following behavior? (cannot mix types even if they are child classes)
        with self.assertRaises(TypeError):
            Series(one, two, three, 4)

        # Define a series whose items are in succession
        succession_series = Series(one, two, three)

        # Test insert    
        succession_series.insert(0,zero)
        self.assertEqual(len(succession_series),4)
        succession_series.insert(4,four)
        self.assertEqual(len(succession_series),5)

        # Test insert not in succession  
        with self.assertRaises(IndexError):
            succession_series.insert(1,two)

        # Test remove disabled for items in a succession
        with self.assertRaises(NotImplementedError):
            succession_series.remove(one)
        
        # Test type
        series = Series(1.0,4.0,5.0)
        self.assertEqual(series, [1.0,4.0,5.0])
        self.assertEqual(len(series), 3)
        self.assertEqual(series.items_type, float)
        
        # Test head, tail & content
        series = Series(1.0,4.0,5.0)
        self.assertEqual(series.head(2),[1.0,4.0])
        self.assertEqual(series.tail(2),[4.0,5.0])
        self.assertEqual(series.contents(),[1.0,4.0,5.0])

        # TOOD: test the print with a print mock?
        #float_series.print(3)
        
        # Test with data slots
        series = Series()
        series.append(DataSlot(start=Point(1), end=Point(2), data='hello'))
        series.append(DataSlot(start=Point(2), end=Point(3), data='hola'))
        self.assertEqual(series[0].start.coordinates[0],1)
        self.assertEqual(series[0].end.coordinates[0],2)
        self.assertEqual(series[0].data,'hello') 

        # Test unrealistic case with strings just for completeness
        Series('a', 'b')
        with self.assertRaises(ValueError):
            Series('b', 'a')


class TestTimeSeries(unittest.TestCase):

    def test_TimeSeries_with_TimePoints(self):
        
        time_series = TimeSeries()
        time_series.append(TimePoint(t=60))
                
        # Test for UTC timezone (autodetect)
        time_series = TimeSeries()
        time_series.append(TimePoint(t=5))         
        self.assertEqual(time_series.tz, UTC)
        self.assertEqual(type(time_series.tz), type(UTC))
        time_series.append(TimePoint(t=10)) 
        self.assertEqual(time_series.tz, UTC)
        self.assertEqual(type(time_series.tz), type(UTC))
        time_series.append(TimePoint(t=15, tz='Europe/Rome')) # This will get ignored and timezone will stay on UTC
        self.assertEqual(time_series.tz, UTC)
        self.assertEqual(type(time_series.tz), type(UTC))

        # Test for Europe/Rome timezone
        time_series = TimeSeries() 
        time_series.append(TimePoint(t=15, tz='Europe/Rome'))
        self.assertEqual(str(time_series.tz), 'Europe/Rome')

        # Test for Europe/Rome timezone
        time_series = TimeSeries() 
        time_series.append(TimePoint(dt=dt(2015,3,5,9,27,tz='Europe/Rome')))
        self.assertEqual(str(time_series.tz), 'Europe/Rome')
            
        # Change timezone
        time_series = TimeSeries()
        time_series.append(TimePoint(t=5))
        time_series.append(TimePoint(t=10))
        time_series.change_tz('Europe/Rome')
        self.assertEqual(str(time_series.tz), 'Europe/Rome')
        self.assertEqual(str(type(time_series.tz)), "<class 'pytz.tzfile.Europe/Rome'>")
        self.assertEqual(str(time_series[0].tz), 'Europe/Rome')

        # Test for Europe/Rome timezone (set)
        time_series = TimeSeries(tz = 'Europe/Rome')
        self.assertEqual(str(time_series.tz), 'Europe/Rome')
        self.assertEqual(str(type(time_series.tz)), "<class 'pytz.tzfile.Europe/Rome'>")
        time_series.append(TimePoint(t=5))
        self.assertEqual(str(time_series.tz), 'Europe/Rome')
        self.assertEqual(str(type(time_series.tz)), "<class 'pytz.tzfile.Europe/Rome'>")        
         
        # Test for Europe/Rome timezone (autodetect)
        time_series = TimeSeries()
        time_series.append(TimePoint(t=1569897900, tz='Europe/Rome')) 
        self.assertEqual(str(time_series.tz), 'Europe/Rome')
        self.assertEqual(str(type(time_series.tz)), "<class 'pytz.tzfile.Europe/Rome'>")
        time_series.append(TimePoint(t=1569897910, tz='Europe/Rome')) 
        self.assertEqual(str(time_series.tz), 'Europe/Rome')
        self.assertEqual(str(type(time_series.tz)), "<class 'pytz.tzfile.Europe/Rome'>")
        time_series.append(TimePoint(t=1569897920))
        self.assertEqual(time_series.tz, UTC)
        self.assertEqual(type(time_series.tz), type(UTC))
        time_series.change_tz('Europe/Rome')
        self.assertEqual(str(time_series.tz), 'Europe/Rome')
        self.assertEqual(str(type(time_series.tz)), "<class 'pytz.tzfile.Europe/Rome'>")  

        # Test resolution: not defined as empty
        time_series = TimeSeries()
        self.assertEqual(time_series.resolution, None)
        
        # Test resolution: not defined as just one point
        time_series = TimeSeries(TimePoint(t=60))
        self.assertEqual(time_series.resolution, None)
        
        # Cannot guess resolution if only one point
        with self.assertRaises(ValueError):
            time_series.guess_resolution()

        # Test resolution: defined, two points, 61 seconds
        time_series = TimeSeries(TimePoint(t=60),TimePoint(t=121)) 
        self.assertTrue(isinstance(time_series.resolution,TimeUnit))
        self.assertEqual(str(time_series.resolution), '61s')
        self.assertEqual(time_series.resolution, 61) # TimeUnits support math
        self.assertEqual(time_series.resolution.value, '61s')
        self.assertEqual(time_series.resolution.as_seconds(), 61)
        
        # Test resolution: defined, two points, 1 minute
        time_series = TimeSeries(TimePoint(t=60),TimePoint(t=120))
        self.assertEqual(str(time_series.resolution), '1m')
        self.assertEqual(time_series.resolution, 60) # TimeUnits support math
        self.assertEqual(time_series.resolution.value, '1m')
        self.assertEqual(time_series.resolution.as_seconds(), 60)

        # Test resolution: variable, three points
        time_series = TimeSeries(TimePoint(t=60),TimePoint(t=120),TimePoint(t=130))
        self.assertEqual(time_series.resolution, 'variable')
        self.assertEqual(time_series.guess_resolution(), 60)
        self.assertEqual(str(time_series.guess_resolution()), '1m')
        self.assertEqual(time_series.guess_resolution().as_seconds(), 60)
        self.assertEqual(time_series.guess_resolution(confidence=True)['value'],60)
        self.assertEqual(time_series.guess_resolution(confidence=True)['confidence'],0)

        # Test resolution: variable, four points
        time_series = TimeSeries(TimePoint(t=60),TimePoint(t=120),TimePoint(t=180),TimePoint(t=190))
        self.assertEqual(time_series.resolution, 'variable')
        self.assertEqual(time_series.guess_resolution(), 60)
        self.assertEqual(str(time_series.guess_resolution()), '1m')
        self.assertEqual(time_series.guess_resolution().as_seconds(), 60)
        self.assertEqual(time_series.guess_resolution(confidence=True)['value'],60)
        self.assertEqual(time_series.guess_resolution(confidence=True)['confidence'],0.5)

        # Test resolution: defined, three points
        time_series = TimeSeries(TimePoint(t=60),TimePoint(t=120),TimePoint(t=180))
        self.assertEqual(time_series.duplicate().resolution, 60)
        self.assertEqual(time_series[0:2].resolution, 60)
        

    def test_TimeSeries_with_DataTimePoints(self):

        # Basic tests
        with self.assertRaises(TypeError):
            TimeSeries(DataTimePoint(t=60, data=23.8),
                       DataTimePoint(t=120, data=24.1),
                       DataTimePoint(t=180, data=None))
            
        with self.assertRaises(ValueError):
            TimeSeries(DataTimePoint(t=60, data=23.8),
                       DataTimePoint(t=30, data=24.1))

        with self.assertRaises(ValueError):
            TimeSeries(DataTimePoint(t=60, data=23.8),
                       DataTimePoint(t=60, data=24.1))
        
        time_series = TimeSeries(DataTimePoint(t=60, data=23.8),
                                 DataTimePoint(t=120, data=24.1),
                                 DataTimePoint(t=180, data=23.9),
                                 DataTimePoint(t=240, data=23.1),
                                 DataTimePoint(t=300, data=22.7))
                                                
        time_series.append(DataTimePoint(t=360, data=21.9))
        self.assertTrue(len(time_series), 6)
   

        time_series = TimeSeries(DataTimePoint(t=60, data=23.8),
                                 DataTimePoint(t=120, data=24.1),
                                 DataTimePoint(t=240, data=23.1),
                                 DataTimePoint(t=300, data=22.7))
        self.assertTrue(len(time_series), 5)

        # Try to append a different data type
        time_series = TimeSeries(DataTimePoint(t=60, data=[23.8]))
        with self.assertRaises(TypeError):
            time_series.append(DataTimePoint(t=120, data={'a':56}))

        # Try to append the same data type but with different cardinality
        time_series = TimeSeries(DataTimePoint(t=60, data=[23.8]))
        with self.assertRaises(ValueError):
            time_series.append(DataTimePoint(t=180, data=[23.8,31.3]))

        # Try to append the same data type but with different labels
        time_series = TimeSeries(DataTimePoint(t=60, data={'a':56, 'b':67}))
        with self.assertRaises(ValueError):
            time_series.append(DataTimePoint(t=180, data={'a':56, 'c':67}))            

        # Test accessing the time series as Pandas data frame
        time_series =  TimeSeries(DataTimePoint(t=0, data={'C': 23.8, 'RH': 57.2}),
                                  DataTimePoint(t=60, data={'C': 23.8, 'RH': 57.2}),
                                  DataTimePoint(t=120, data={'C': 23.8, 'RH': 57.2}))
        self.assertEqual(len(time_series.df),3)
        self.assertEqual(time_series.df.index[0],dt(1970,1,1,0,0,0))
        self.assertEqual(time_series.df.index[1],dt(1970,1,1,0,1,0))
        self.assertEqual(list(time_series.df.columns),['C','RH'])
        self.assertEqual(time_series.df.iloc[0][0],23.8)
        self.assertEqual(time_series.df.iloc[0][1],57.2)

        # Test resolution
        time_series = TimeSeries(DataTimePoint(dt=dt(2015,10,24,0,0,0, tzinfo='Europe/Rome'), data=23.8),
                                 DataTimePoint(dt=dt(2015,10,25,0,0,0, tzinfo='Europe/Rome'), data=24.1),
                                 DataTimePoint(dt=dt(2015,10,26,0,0,0, tzinfo='Europe/Rome'), data=23.1))
        self.assertEqual(time_series.resolution, 'variable') # DST occurred, resolution marked as undefined
        self.assertEqual(time_series.guess_resolution(), 86400)

        time_series = TimeSeries(DataTimePoint(dt=dt(2015,10,27,0,0,0, tzinfo='Europe/Rome'), data={'a':23.8}),
                                 DataTimePoint(dt=dt(2015,10,28,0,0,0, tzinfo='Europe/Rome'), data={'a':24.1}),
                                 DataTimePoint(dt=dt(2015,10,29,0,0,0, tzinfo='Europe/Rome'), data={'a':23.1}))
        self.assertEqual(time_series.resolution, 86400) # No DST occurred
        self.assertEqual(time_series.resolution, '86400s') # No DST occurred

        # Test data labels and rename a data labels
        time_series = TimeSeries(DataTimePoint(dt=dt(2015,10,27,0,0,0, tzinfo='Europe/Rome'), data={'a':23.8, 'b':1}),
                                 DataTimePoint(dt=dt(2015,10,28,0,0,0, tzinfo='Europe/Rome'), data={'a':24.1, 'b':2}),
                                 DataTimePoint(dt=dt(2015,10,29,0,0,0, tzinfo='Europe/Rome'), data={'a':23.1, 'b':3}))        
        self.assertEqual(time_series.data_labels(), ['a','b'])
        time_series.rename_data_label('b','c')
        self.assertEqual(time_series.data_labels(), ['a','c'])
        
        with self.assertRaises(KeyError):
            time_series.rename_data_label('notexistent_label','c')

        # Check timezone
        time_series = TimeSeries(DataTimePoint(dt=dt(2015,10,25,0,0,0, tz='Europe/Rome'), data={'a':23.8}),
                                 DataTimePoint(dt=dt(2015,10,26,0,0,0, tz='Europe/Rome'), data={'a':23.8}))
        self.assertEqual(str(time_series.tz), 'Europe/Rome')
            
        # Test test square brackets notation for filtering (more testing is done in the filter operation tests)
        time_series =  TimeSeries(DataTimePoint(t=60, data={'a':1, 'b':2}))
        self.assertEqual(time_series['a'][0].data, {'a': 1})        
    
        # Test square brackets notation for getting items and slicing
        time_series = TimeSeries(DataTimePoint(dt=dt(2015,10,25,6,15,0, tz='Europe/Rome'), data={'a':11, 'b':12, 'c':13}), # 1445753700
                                 DataTimePoint(dt=dt(2015,10,25,6,16,0, tz='Europe/Rome'), data={'a':21, 'b':22, 'c':23}), # 1445753760
                                 DataTimePoint(dt=dt(2015,10,25,6,17,0, tz='Europe/Rome'), data={'a':31, 'b':32, 'c':33}), # 1445753820
                                 DataTimePoint(dt=dt(2015,10,25,6,18,0, tz='Europe/Rome'), data={'a':41, 'b':42, 'c':43}), # 1445753880
                                 DataTimePoint(dt=dt(2015,10,25,6,19,0, tz='Europe/Rome'), data={'a':51, 'b':52, 'c':53}), # 1445750340
                                 DataTimePoint(dt=dt(2015,10,25,6,20,0, tz='Europe/Rome'), data={'a':61, 'b':62, 'c':63}), # 1445750400
                                 DataTimePoint(dt=dt(2015,10,25,6,21,0, tz='Europe/Rome'), data={'a':71, 'b':72, 'c':73}), # 1445750460
                                 DataTimePoint(dt=dt(2015,10,25,6,22,0, tz='Europe/Rome'), data={'a':81, 'b':82, 'c':83})) # 1445750520
        
        # Get item by t
        with self.assertRaises(ValueError):
            time_series['t=5']
        self.assertEqual(time_series['t=1445750340'], time_series[4])
        with self.assertRaises(ValueError):
            time_series[{'t':5}]
        self.assertEqual(time_series[{'t':1445750340}], time_series[4])

        # Get item by dt
        with self.assertRaises(ValueError):
            time_series['dt=2035-10-25 00:00:00+01:00']         
        self.assertEqual(time_series['dt=2015-10-25 06:19:00+01:00'], time_series[4])
        self.assertEqual(time_series['dt=2015-10-25T06:19:00+01:00'], time_series[4])
        with self.assertRaises(ValueError):
            time_series[{'dt': dt(2035,10,25,0,0,0, tz='Europe/Rome')}]         
        self.assertEqual(time_series[{'dt': dt(2015,10,25,6,19,0, tz='Europe/Rome')}], time_series[4])
        
        # Test square brackets notation for slicing
        self.assertEqual(len(time_series['t=1445750340':'t=1445750460']), 2)
        self.assertEqual(time_series['t=1445750340':'t=1445750460'][0], time_series[4])
        self.assertEqual(time_series['t=1445750340':'t=1445750460'][1], time_series[5])

        self.assertEqual(len(time_series['t=1445750340':'t=1445750461']), 3)
        self.assertEqual(time_series['t=1445750340':'t=1445750461'][0], time_series[4])
        self.assertEqual(time_series['t=1445750340':'t=1445750461'][1], time_series[5])
        self.assertEqual(time_series['t=1445750340':'t=1445750461'][2], time_series[6])

        self.assertEqual(len(time_series[{'t':1445750340}:{'t':1445750460}]), 2)
        self.assertEqual(time_series[{'t':1445750340}:{'t':1445750460}][0], time_series[4])
        self.assertEqual(time_series[{'t':1445750340}:{'t':1445750460}][1], time_series[5])

        self.assertEqual(len(time_series[{'t':1445750340}:{'t':1445750460}]), 2)
        self.assertEqual(time_series[{'t':1445750340}:{'t':1445750460}][0], time_series[4])
        self.assertEqual(time_series[{'t':1445750340}:{'t':1445750460}][1], time_series[5])

        self.assertEqual(len(time_series[{'t':1445750340}:]), 4)
        self.assertEqual(time_series[{'t':1445750340}:][0], time_series[4])
        self.assertEqual(time_series[{'t':1445750340}:][1], time_series[5])
        self.assertEqual(time_series[{'t':1445750340}:][2], time_series[6])
        self.assertEqual(time_series[{'t':1445750340}:][3], time_series[7])

        self.assertEqual(len(time_series[:{'t':1445750340}]), 4)
        self.assertEqual(time_series[:{'t':1445750340}][0], time_series[0])
        self.assertEqual(time_series[:{'t':1445750340}][1], time_series[1])
        self.assertEqual(time_series[:{'t':1445750340}][2], time_series[2])
        self.assertEqual(time_series[:{'t':1445750340}][3], time_series[3])

        # Start = end and start > end 
        self.assertEqual(len(time_series[{'t':1445750340}:{'t':1445750340}]), 0)
        self.assertEqual(len(time_series[{'t':4445750340}:{'t':1445750340}]), 0)


    def test_TimeSeries_with_TimeSlots(self):

        # Basic tests
        time_series = TimeSeries(TimeSlot(start=TimePoint(0), end=TimePoint(10)))
        with self.assertRaises(ValueError):
            # Cannot add items with different units (I have "10.0" and you tried to add "11.0")
            time_series.append(TimeSlot(start=TimePoint(10), end=TimePoint(21)))
        time_series.append(TimeSlot(start=TimePoint(10), end=TimePoint(20)))

        time_series = TimeSeries()
        time_series.append(TimeSlot(start=TimePoint(t=0), end=TimePoint(t=60)))
        with self.assertRaises(ValueError):
            time_series.append(TimeSlot(start=TimePoint(t=0), end=TimePoint(t=60)))
        with self.assertRaises(ValueError):
            time_series.append(TimeSlot(start=TimePoint(t=120), end=TimePoint(t=180)))

        time_series.append(TimeSlot(start=TimePoint(t=60), end=TimePoint(t=120)))

        self.assertEqual(len(time_series),2)
        self.assertEqual(time_series[0].start.t,0)
        self.assertEqual(time_series[0].end.t,60)
        self.assertEqual(time_series[1].start.t,60)
        self.assertEqual(time_series[1].end.t,120) 

        # Test timezone
        time_series = TimeSeries()
        time_series.append(TimeSlot(start=TimePoint(t=0, tz='Europe/Rome'), end=TimePoint(t=60, tz='Europe/Rome')))
        with self.assertRaises(ValueError):
            time_series.append(TimeSlot(start=TimePoint(t=60), end=TimePoint(t=210)))
        time_series.append(TimeSlot(start=TimePoint(t=60, tz='Europe/Rome'), end=TimePoint(t=120, tz='Europe/Rome')))

        # Test slot unit
        self.assertEqual(time_series.resolution, Unit(60.0))

        # Test resolution 
        self.assertEqual(time_series.resolution, Unit(60))


    def test_TimeSeries_with_DataTimeSlots(self):

        with self.assertRaises(TypeError):
            TimeSeries(DataTimeSlot(start=TimePoint(t=60), end=TimePoint(t=120), data=23.8),
                       DataTimeSlot(start=TimePoint(t=120), end=TimePoint(t=180), data=24.1),
                       DataTimeSlot(start=TimePoint(t=180), end=TimePoint(t=240), data=None))

        with self.assertRaises(ValueError):
            TimeSeries(DataTimeSlot(start=TimePoint(t=180), end=TimePoint(t=20), data=24.1),
                       DataTimeSlot(start=TimePoint(t=60), end=TimePoint(t=120), data=23.8))

        with self.assertRaises(ValueError):
            TimeSeries(DataTimeSlot(start=TimePoint(t=60), end=TimePoint(t=120), data=23.8),
                       DataTimeSlot(start=TimePoint(t=180), end=TimePoint(t=20), data=24.1))

        with self.assertRaises(ValueError):
            TimeSeries(DataTimeSlot(start=TimePoint(t=60), end=TimePoint(t=120), data=23.8),
                       DataTimeSlot(start=TimePoint(t=60), end=TimePoint(t=120), data=24.1))

        time_series = TimeSeries(DataTimeSlot(start=TimePoint(t=60),  end=TimePoint(t=120), data=23.8),
                                 DataTimeSlot(start=TimePoint(t=120), end=TimePoint(t=180), data=24.1),
                                 DataTimeSlot(start=TimePoint(t=180), end=TimePoint(t=240), data=23.1))

        time_series.append(DataTimeSlot(start=TimePoint(t=240), end=TimePoint(t=300), data=22.7))
        self.assertTrue(len(time_series), 4)

        # Try to append a different data type
        time_series = TimeSeries(DataTimeSlot(start=TimePoint(t=60), end=TimePoint(t=120), data=23.8))
        with self.assertRaises(TypeError):
            time_series.append(DataTimeSlot(start=TimePoint(t=120), end=TimePoint(t=180), data={'a':56}))

        # Try to append the same data type but with different cardinality
        time_series = TimeSeries(DataTimeSlot(start=TimePoint(t=60), end=TimePoint(t=120), data=[23.8]))
        with self.assertRaises(ValueError):
            time_series.append(DataTimeSlot(start=TimePoint(t=120), end=TimePoint(t=180), data=[23.8,31.3]))

        # Try to append the same data type but with different labels
        time_series = TimeSeries(DataTimeSlot(start=TimePoint(t=60), end=TimePoint(t=120), data={'a':56, 'b':67}))
        with self.assertRaises(ValueError):
            time_series.append(DataTimeSlot(start=TimePoint(t=120), end=TimePoint(t=180), data={'a':56, 'c':67}))            

        # Test with units
        self.assertEqual(time_series.resolution, Unit(60.0))
        self.assertEqual(TimeSeries(DataTimeSlot(start=TimePoint(t=60), end=TimePoint(t=120), data=23.8, unit=TimeUnit('60s'))).resolution, TimeUnit('60s'))

        time_series = TimeSeries()
        prev_t = 1595862221
        for _ in range (0, 10):
            t = prev_t + 60
            time_series.append(DataTimeSlot(start=TimePoint(t=t), unit=TimeUnit('60s'), data=[5]))    
            prev_t = t
        self.assertEqual(len(time_series),10)
        self.assertEqual(time_series[0].unit,TimeUnit('60s'))
        self.assertEqual(time_series[0].unit, Unit(60))

        # Time series with calendar time units
        time_series = TimeSeries()
        prev_t = 1595862221
        for _ in range (0, 10):
            t = prev_t + 86400
            time_series.append(DataTimeSlot(start=TimePoint(t=t), unit=TimeUnit('1D'), data=[5]))    
            prev_t = t
        self.assertEqual(len(time_series),10)
        self.assertEqual(time_series[0].unit,TimeUnit('1D'))

        # Test accessing the time series as Pandas data frame
        time_series =  TimeSeries(DataTimeSlot(start=TimePoint(t=0),  end=TimePoint(t=86400), data={'C': 23.8, 'RH': 57.2}),
                                  DataTimeSlot(start=TimePoint(t=86400), end=TimePoint(t=86400*2), data={'C': 23.8, 'RH': 57.2}),
                                  DataTimeSlot(start=TimePoint(t=86400*2), end=TimePoint(t=86400*3), data={'C': 23.8, 'RH': 57.2}))
        self.assertEqual(len(time_series.df),3)
        self.assertEqual(time_series.df.index[0],dt(1970,1,1,0,0,0))
        self.assertEqual(time_series.df.index[1],dt(1970,1,2,0,0,0))
        self.assertEqual(list(time_series.df.columns),['C','RH'])
        self.assertEqual(time_series.df.iloc[0][0],23.8)
        self.assertEqual(time_series.df.iloc[0][1],57.2)

        # Test resolution
        time_series =  TimeSeries(DataTimeSlot(start=TimePoint(t=60),  end=TimePoint(t=120), data=23.8),
                                  DataTimeSlot(start=TimePoint(t=120), end=TimePoint(t=180), data=24.1),
                                  DataTimeSlot(start=TimePoint(t=180), end=TimePoint(t=240), data=23.1))
        self.assertEqual(time_series.resolution, Unit(60))

        time_series =  TimeSeries(DataTimeSlot(start=TimePoint(t=60),  unit=TimeUnit('1m'), data=23.8),
                                  DataTimeSlot(start=TimePoint(t=120), unit=TimeUnit('1m'), data=24.1),
                                  DataTimeSlot(start=TimePoint(t=180), unit=TimeUnit('1m'), data=23.1)) 
        self.assertEqual(time_series.resolution, TimeUnit('1m'))

        # Test change timezone
        from ..time import timezonize
        time_series_UTC =  TimeSeries(DataTimeSlot(start=TimePoint(t=60),  end=TimePoint(t=120), data=23.8),
                                      DataTimeSlot(start=TimePoint(t=120), end=TimePoint(t=180), data=24.1),
                                      DataTimeSlot(start=TimePoint(t=180), end=TimePoint(t=240), data=23.1))

        time_series_UTC.change_tz('Europe/Rome')
        self.assertEqual(time_series_UTC.tz, timezonize('Europe/Rome'))
        self.assertEqual(time_series_UTC[0].tz, timezonize('Europe/Rome'))
        self.assertEqual(time_series_UTC[0].start.tz, timezonize('Europe/Rome'))
        self.assertEqual(time_series_UTC[0].end.tz, timezonize('Europe/Rome'))

        # Test get item by string key (filter on data labels). More testing is done in the operation tests
        time_series =  TimeSeries(DataTimeSlot(start=TimePoint(t=60),  end=TimePoint(t=120), data={'a':1, 'b':2}))
        self.assertEqual(time_series['a'][0].data, {'a': 1})

        # Test data labels and rename a data label
        time_series = TimeSeries(DataTimeSlot(dt=dt(2015,10,27,0,0,0), unit='1D', data={'a':23.8, 'b':1}),
                                 DataTimeSlot(dt=dt(2015,10,28,0,0,0), unit='1D', data={'a':24.1, 'b':2}),
                                 DataTimeSlot(dt=dt(2015,10,29,0,0,0), unit='1D', data={'a':23.1, 'b':3}))

        self.assertEqual(time_series.data_labels(), ['a','b'])
        time_series.rename_data_label('b','c')
        self.assertEqual(time_series.data_labels(), ['a','c'])

        with self.assertRaises(KeyError):
            time_series.rename_data_label('notexistent_label','c')


    def test_TimeSeries_alternative_inits(self):
        
        # Test init from a Pandas data frame
        df = pd.read_csv(TEST_DATA_PATH+'csv/format3.csv', header=3, parse_dates=[0], index_col=0)
        time_series = TimeSeries(df)
        self.assertEqual(len(time_series),6)
        self.assertEqual(time_series[0].dt,dt(2020,4,3,0,0,0))
        self.assertEqual(time_series[5].dt,dt(2020,4,3,5,0,0))
        self.assertEqual((time_series[0].data['C']), 21.7)
        self.assertEqual((time_series[0].data['RH']), 54.9)
 
        df = pd.read_csv(TEST_DATA_PATH+'csv/format4.csv', header=0, parse_dates=[0], index_col=0)
        time_series = TimeSeries(df, items_type=DataTimeSlot)
        self.assertEqual(len(time_series),5)
        self.assertEqual(time_series[0].start.dt,dt(2020,4,3,0,0,0))
        self.assertEqual(time_series[4].end.dt,dt(2020,4,3,10,0,0))
        self.assertEqual((time_series[0].data['C']), 21.7)
        self.assertEqual((time_series[0].data['RH']), 54.9)

        # Test init from list of dicts
        time_series = TimeSeries([{'t':60, 'data':14},
                                 {'t':120, 'data':18},
                                 {'t':128, 'data':20}])
        self.assertEqual(len(time_series), 3)
        self.assertEqual(time_series[0].t, 60)
        self.assertEqual(time_series[0].data, [14])

        time_series = TimeSeries([{'t':60, 'data':{'a': 14}},
                                 {'t':120, 'data':{'a': 18}},
                                 {'t':128, 'data':{'a': 20}}])
        self.assertEqual(len(time_series), 3)
        self.assertEqual(time_series[0].t, 60)
        self.assertEqual(time_series[0].data, {'a': 14})

        time_series = TimeSeries([{'dt':dt(1970,1,1,0,1), 'data':14},
                                 {'dt':dt(1970,1,1,0,2), 'data':18},
                                 {'dt':dt(1970,1,1,0,3), 'data':20}],
                                 slot_unit='60s')
        self.assertEqual(len(time_series), 3)
        self.assertEqual(time_series[0].start.t, 60)
        self.assertEqual(time_series[0].end.t, 120)
        self.assertEqual(time_series[0].data, [14])
        
        # Test init from CSV files
        time_series = TimeSeries(TEST_DATA_PATH + '/csv/single_value_no_labels.csv')
        self.assertEqual(len(time_series), 6)
        self.assertEqual(time_series[0].t, 946684800)
        self.assertEqual(time_series[0].data, [1000])

        time_series = TimeSeries(TEST_DATA_PATH + '/csv/only_date_no_meaningful_timestamp_label.csv',
                                 timestamp_format = '%Y-%m-%d')
        self.assertEqual(len(time_series), 100)
        self.assertEqual(time_series[0].start.t, 1197244800)
        self.assertTrue(isinstance(time_series[0], Slot))

        time_series = TimeSeries(TEST_DATA_PATH + '/csv/only_date_no_meaningful_timestamp_label.csv',
                                 timestamp_format = '%Y-%m-%d', series_type='points')
        self.assertEqual(len(time_series), 95)
        self.assertEqual(time_series[0].t, 1197244800)
        self.assertTrue(isinstance(time_series[0], Point))


class TestTimeSeriesView(unittest.TestCase):

    def test_TimeSeriesView(self):
        series = TimeSeries()
        series.append(DataTimePoint(t = 0, data = {'value': 0}))
        series.append(DataTimePoint(t = 1, data = {'value': 1}))
        series.append(DataTimePoint(t = 2, data = {'value': 2}))
        series.append(DataTimePoint(t = 3, data = {'value': 3}))
        series.append(DataTimePoint(t = 4, data = {'value': 4}))
        series.append(DataTimePoint(t = 5, data = {'value': 5}))
        series.append(DataTimePoint(t = 6, data = {'value': 6}))
        series.append(DataTimePoint(t = 7, data = {'value': 7}))
        series.append(DataTimePoint(t = 8, data = {'value': 8}))
        series.append(DataTimePoint(t = 9, data = {'value': 9}))

        from ..utilities import _compute_validity_regions
        validity_regions = _compute_validity_regions(series)
        for point in series:
            point.valid_from=validity_regions[point.t][0]
            point.valid_to=validity_regions[point.t][1]

        series_view = _TimeSeriesView(series, 2, 5)

        self.assertEqual(len(series_view), 3)

        for i, point in enumerate(series_view):
            self.assertEqual(point.t, 2+i)

        self.assertEqual(series_view[0].t, 2)
        self.assertEqual(series_view[1].t, 3)
        self.assertEqual(series_view[2].t, 4)
        self.assertEqual(series_view[-1].t, 4)

        # Test extra attributes
        self.assertEqual(str(series_view.resolution), '1s')
        self.assertEqual(series_view.data_labels(), ['value'])

        # TODO: Most operations are not supported on a TimeSeriesView and should be disabled
        #with self.assertRaises(AttributeError):
        #    series_view.diff()


    def test_dense_TimeSeriesView(self):
        series = TimeSeries()
        series.append(DataTimePoint(t = 0, data = {'value': 0}))
        series.append(DataTimePoint(t = 1, data = {'value': 1}))
        series.append(DataTimePoint(t = 2, data = {'value': 2}))
        series.append(DataTimePoint(t = 3, data = {'value': 3}))
        series.append(DataTimePoint(t = 4, data = {'value': 4}))
        series.append(DataTimePoint(t = 7, data = {'value': 7}))
        series.append(DataTimePoint(t = 8, data = {'value': 8}))
        series.append(DataTimePoint(t = 9, data = {'value': 9}))

        from ..utilities import _compute_validity_regions
        validity_regions = _compute_validity_regions(series)
        for point in series:
            point.valid_from=validity_regions[point.t][0]
            point.valid_to=validity_regions[point.t][1]

        from ..interpolators import LinearInterpolator
        series_view = _TimeSeriesView(series, 2, 8, dense=True, interpolator_class=LinearInterpolator)

        self.assertEqual(len(series_view), 7)

        series_view_materialized=[]
        for point in series_view:
            series_view_materialized.append(point)

        self.assertEqual(series_view_materialized[0].t, 2)
        self.assertEqual(series_view_materialized[1].t, 3)
        self.assertEqual(series_view_materialized[2].t, 4)
        self.assertEqual(series_view_materialized[3].t, 5.5)
        self.assertEqual(series_view_materialized[3].data['value'], 5.5)
        self.assertEqual(series_view_materialized[4].t, 7)
        self.assertEqual(series_view_materialized[5].t, 8)
        self.assertEqual(series_view_materialized[6].t, 9)

