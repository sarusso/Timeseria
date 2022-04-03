import unittest
import datetime
import os
import pandas as pd

from ..datastructures import Point, TimePoint, DataPoint, DataTimePoint
from ..datastructures import Slot, TimeSlot, DataSlot, DataTimeSlot
from ..datastructures import Series, SlotSeries, TimePointSeries, TimeSlotSeries, DataSlotSeries
from ..datastructures import DataTimePointSeries, DataTimeSlotSeries 
from ..datastructures import SeriesSlice 
from ..time import UTC, dt
from ..units import Unit, TimeUnit

# Setup logging
from .. import logger
logger.setup()

# Set test data path
TEST_DATA_PATH = '/'.join(os.path.realpath(__file__).split('/')[0:-1]) + '/test_data/'


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
        five = IntegerNumber(5)

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
        
        # Define a Series with fixed type
        class FloatSeries(Series):
            __TYPE__ = float

        with self.assertRaises(TypeError):
            FloatSeries(1,4,5, 'hello', None)

        with self.assertRaises(TypeError):
            FloatSeries(1,4,5)

        with self.assertRaises(TypeError):
            FloatSeries(1.0,4.0,None,5.0)
            
        float_series = FloatSeries(1.0,4.0,5.0)
        self.assertEqual(float_series, [1.0,4.0,5.0])
        self.assertEqual(len(float_series), 3)
        

        # Test head, tail & content
        self.assertEqual(float_series.head(2),[1.0,4.0])
        self.assertEqual(float_series.tail(2),[4.0,5.0])
        self.assertEqual(float_series.contents(),[1.0,4.0,5.0])

        # Test print with print mock?
        #print()
        #float_series.print(3)
        
        # Test edited operations
        


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
        self.assertEqual(data_point.data_labels(), [0,1])
        
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

        # Test None index
        #with self.assertRaises(ValueError):
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



class TestPointSeries(unittest.TestCase):

    def test_TimePointSeries(self):
        
        time_point_series = TimePointSeries()
        time_point_series.append(TimePoint(t=60))
                
        # Test for UTC timezone (autodetect)
        time_point_series = TimePointSeries()
        time_point_series.append(TimePoint(t=5))         
        self.assertEqual(time_point_series.tz, UTC)
        self.assertEqual(type(time_point_series.tz), type(UTC))
        time_point_series.append(TimePoint(t=10)) 
        self.assertEqual(time_point_series.tz, UTC)
        self.assertEqual(type(time_point_series.tz), type(UTC))
        time_point_series.append(TimePoint(t=15, tz='Europe/Rome')) # This will get ignored and timezone will stay on UTC
        self.assertEqual(time_point_series.tz, UTC)
        self.assertEqual(type(time_point_series.tz), type(UTC))

        # Test for Europe/Rome timezone
        time_point_series = TimePointSeries() 
        time_point_series.append(TimePoint(t=15, tz='Europe/Rome'))
        self.assertEqual(str(time_point_series.tz), 'Europe/Rome')

        # Test for Europe/Rome timezone
        time_point_series = TimePointSeries() 
        time_point_series.append(TimePoint(dt=dt(2015,3,5,9,27,tz='Europe/Rome')))
        self.assertEqual(str(time_point_series.tz), 'Europe/Rome')
        data_time_point_seriess = DataTimePointSeries(DataTimePoint(dt=dt(2015,10,25,0,0,0, tz='Europe/Rome'), data={'a':23.8}),
                                                     DataTimePoint(dt=dt(2015,10,26,0,0,0, tz='Europe/Rome'), data={'a':23.8}))
        self.assertEqual(str(data_time_point_seriess.tz), 'Europe/Rome')
               
        # Change timezone
        time_point_series = TimePointSeries()
        time_point_series.append(TimePoint(t=5))
        time_point_series.append(TimePoint(t=10))
        time_point_series.change_timezone('Europe/Rome')
        self.assertEqual(str(time_point_series.tz), 'Europe/Rome')
        self.assertEqual(str(type(time_point_series.tz)), "<class 'pytz.tzfile.Europe/Rome'>")
        self.assertEqual(str(time_point_series[0].tz), 'Europe/Rome')

        # Test for Europe/Rome timezone (set)
        time_point_series = TimePointSeries(tz = 'Europe/Rome')
        self.assertEqual(str(time_point_series.tz), 'Europe/Rome')
        self.assertEqual(str(type(time_point_series.tz)), "<class 'pytz.tzfile.Europe/Rome'>")
        time_point_series.append(TimePoint(t=5))
        self.assertEqual(str(time_point_series.tz), 'Europe/Rome')
        self.assertEqual(str(type(time_point_series.tz)), "<class 'pytz.tzfile.Europe/Rome'>")        
         
        # Test for Europe/Rome timezone  (autodetect)
        time_point_series = TimePointSeries()
        time_point_series.append(TimePoint(t=1569897900, tz='Europe/Rome')) 
        self.assertEqual(str(time_point_series.tz), 'Europe/Rome')
        self.assertEqual(str(type(time_point_series.tz)), "<class 'pytz.tzfile.Europe/Rome'>")
        time_point_series.append(TimePoint(t=1569897910, tz='Europe/Rome')) 
        self.assertEqual(str(time_point_series.tz), 'Europe/Rome')
        self.assertEqual(str(type(time_point_series.tz)), "<class 'pytz.tzfile.Europe/Rome'>")
        time_point_series.append(TimePoint(t=1569897920))
        self.assertEqual(time_point_series.tz, UTC)
        self.assertEqual(type(time_point_series.tz), type(UTC))
        time_point_series.change_timezone('Europe/Rome')
        self.assertEqual(str(time_point_series.tz), 'Europe/Rome')
        self.assertEqual(str(type(time_point_series.tz)), "<class 'pytz.tzfile.Europe/Rome'>")  
        
        # Test resolution: not defined as just one point
        time_point_series = TimePointSeries(TimePoint(t=60))
        self.assertEqual(time_point_series.resolution, None)

        # Test resolution: defined, two points       
        time_point_series = TimePointSeries(TimePoint(t=60),TimePoint(t=121))
        
        self.assertTrue(isinstance(time_point_series.resolution,TimeUnit))
        self.assertEqual(str(time_point_series.resolution), '61s')
        self.assertEqual(time_point_series.resolution, 61) # TimeUnits support math

        self.assertTrue(isinstance(time_point_series.resolution.value, str))
        self.assertEqual(time_point_series.resolution.value, '61s')

        # Test resolution: variable, threee points       
        time_point_series = TimePointSeries(TimePoint(t=60),TimePoint(t=120),TimePoint(t=130))
        self.assertEqual(time_point_series.resolution, '~1m')
        
        # Test resolution: defined, threee points               
        time_point_series = TimePointSeries(TimePoint(t=60),TimePoint(t=120),TimePoint(t=180))
        self.assertEqual(time_point_series.duplicate().resolution, 60)
        self.assertEqual(time_point_series[0:2].resolution, 60)
        

    def test_DataPointSeries(self):
        pass
        # TODO: either implement the "__suceedes__" for unidimensional DataPoints or remove it completely.
        #data_point_serie = DataPointSeries()
        #data_point_serie.append(DataPoint(x=60, data='hello'))
        #data_point_serie.append(DataPoint(x=61, data='hello'))


    def test_DataTimePointSeries(self):
        
        with self.assertRaises(TypeError):
            DataTimePointSeries(DataTimePoint(t=60, data=23.8),
                               DataTimePoint(t=120, data=24.1),
                               DataTimePoint(t=180, data=None))
            
        with self.assertRaises(ValueError):
            DataTimePointSeries(DataTimePoint(t=60, data=23.8),
                               DataTimePoint(t=30, data=24.1))

        with self.assertRaises(ValueError):
            DataTimePointSeries(DataTimePoint(t=60, data=23.8),
                               DataTimePoint(t=60, data=24.1))
        
        data_time_point_series = DataTimePointSeries(DataTimePoint(t=60, data=23.8),
                                                   DataTimePoint(t=120, data=24.1),
                                                   DataTimePoint(t=180, data=23.9),
                                                   DataTimePoint(t=240, data=23.1),
                                                   DataTimePoint(t=300, data=22.7))
                                                
        data_time_point_series.append(DataTimePoint(t=360, data=21.9))
        self.assertTrue(len(data_time_point_series), 6)
   

        data_time_point_series = DataTimePointSeries(DataTimePoint(t=60, data=23.8),
                                                   DataTimePoint(t=120, data=24.1),
                                                   DataTimePoint(t=240, data=23.1),
                                                   DataTimePoint(t=300, data=22.7))
        self.assertTrue(len(data_time_point_series), 5)

      
        # Try to append a different data type
        data_time_point_series = DataTimePointSeries(DataTimePoint(t=60, data=[23.8]))
        with self.assertRaises(TypeError):
            data_time_point_series.append(DataTimePoint(t=120, data={'a':56}))

        # Try to append the same data type but with different cardinality
        data_time_point_series = DataTimePointSeries(DataTimePoint(t=60, data=[23.8]))
        with self.assertRaises(ValueError):
            data_time_point_series.append(DataTimePoint(t=180, data=[23.8,31.3]))

        # Try to append the same data type but with different keys
        data_time_point_series = DataTimePointSeries(DataTimePoint(t=60, data={'a':56, 'b':67}))
        with self.assertRaises(ValueError):
            data_time_point_series.append(DataTimePoint(t=180, data={'a':56, 'c':67}))            

        # Test creating a time series from a Pandas data frame 
        df = pd.read_csv(TEST_DATA_PATH+'csv/format3.csv', header=3, parse_dates=[0], index_col=0)
        data_time_point_series = DataTimePointSeries(df)
        self.assertEqual(len(data_time_point_series),6)
        self.assertEqual(data_time_point_series[0].dt,dt(2020,4,3,0,0,0))
        self.assertEqual(data_time_point_series[5].dt,dt(2020,4,3,5,0,0))
        
        # Test creating a Pandas data frame from a time series
        self.assertEqual(len(data_time_point_series.df),6)
        self.assertEqual(data_time_point_series.df.index[0],dt(2020,4,3,0,0,0))
        self.assertEqual(list(data_time_point_series.df.columns),['C','RH'])
        self.assertEqual(data_time_point_series.df.iloc[0][0],21.7)
        self.assertEqual(data_time_point_series.df.iloc[0][1],54.9)

        # Test loading a Pandas data frame as a time series        
        loaded_data_time_point_series = DataTimePointSeries(data_time_point_series.df)
        self.assertEqual(loaded_data_time_point_series[0].dt,dt(2020,4,3,0,0,0))
        self.assertEqual(loaded_data_time_point_series[5].dt,dt(2020,4,3,5,0,0))
        self.assertEqual((loaded_data_time_point_series[0].data['C']), 21.7)
        self.assertEqual((loaded_data_time_point_series[0].data['RH']), 54.9)

        # Test resolution
        data_time_point_series = DataTimePointSeries(DataTimePoint(dt=dt(2015,10,24,0,0,0, tzinfo='Europe/Rome'), data=23.8),
                                                     DataTimePoint(dt=dt(2015,10,25,0,0,0, tzinfo='Europe/Rome'), data=24.1),
                                                     DataTimePoint(dt=dt(2015,10,26,0,0,0, tzinfo='Europe/Rome'), data=23.1))
        
        self.assertEqual(data_time_point_series.resolution, '~86400s') # DST occurred
        
        data_time_point_series = DataTimePointSeries(DataTimePoint(dt=dt(2015,10,27,0,0,0, tzinfo='Europe/Rome'), data={'a':23.8}),
                                                     DataTimePoint(dt=dt(2015,10,28,0,0,0, tzinfo='Europe/Rome'), data={'a':24.1}),
                                                     DataTimePoint(dt=dt(2015,10,29,0,0,0, tzinfo='Europe/Rome'), data={'a':23.1}))
        
        self.assertEqual(data_time_point_series.resolution, 86400) # No DST occured
        self.assertEqual(data_time_point_series.resolution, '86400s') # No DST occured

        # Test get item by string key (filter on data labels). More testing is done in the operation tests
        data_time_point_series =  DataTimePointSeries(DataTimePoint(t=60, data={'a':1, 'b':2}))
        self.assertEqual(data_time_point_series['a'][0].data, {'a': 1})


        # Test data keys and rename a data key
        # TODO: move this test where to test data point series ?
        data_time_point_series = DataTimePointSeries(DataTimePoint(dt=dt(2015,10,27,0,0,0, tzinfo='Europe/Rome'), data={'a':23.8, 'b':1}),
                                                     DataTimePoint(dt=dt(2015,10,28,0,0,0, tzinfo='Europe/Rome'), data={'a':24.1, 'b':2}),
                                                     DataTimePoint(dt=dt(2015,10,29,0,0,0, tzinfo='Europe/Rome'), data={'a':23.1, 'b':3}))
        
        self.assertEqual(data_time_point_series.data_labels(), ['a','b'])
        data_time_point_series.rename_data_label('b','c')
        self.assertEqual(data_time_point_series.data_labels(), ['a','c'])
        
        with self.assertRaises(KeyError):
            data_time_point_series.rename_data_label('notexistent_key','c')



class TestUnit(unittest.TestCase):

    def test_Unit(self):      

        self.assertEqual( (Point(1) + Unit(5)).coordinates[0], 6) 
        self.assertEqual( (TimePoint(1) + Unit(5)).coordinates[0], 6) 
        self.assertEqual(Unit(5)+1,6)



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
        self.assertEqual(data_slot.data_labels(), [0,1])
        
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
        
        # Test None index
        #with self.assertRaises(ValueError):
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



class TestSlotSeries(unittest.TestCase):

    def test_SlotSeries(self):

        with self.assertRaises(ValueError):
            # ValueError: Slot start and end dimensions must have the same dimension
            SlotSeries(Slot(start=Point(0), end=Point(10,20)))
            
        slot_series =  SlotSeries(Slot(start=Point(0), end=Point(10)))
        
        with self.assertRaises(ValueError):
            # Cannot add items with different units (I have "10.0" and you tried to add "11.0")
            slot_series.append(Slot(start=Point(10), end=Point(21)))
        slot_series.append(Slot(start=Point(10), end=Point(20)))
        
        # The unit is more used as a type..
        slot_series = SlotSeries(Slot(start=Point(0), end=Point(10), unit='10-ish'))
        slot_series.append(Slot(start=Point(10), end=Point(21), unit='10-ish'))


    def test_TimeSlotSeries(self):
         
        time_slot_series = TimeSlotSeries()
        time_slot_series.append(TimeSlot(start=TimePoint(t=0), end=TimePoint(t=60)))
        with self.assertRaises(ValueError):
            time_slot_series.append(TimeSlot(start=TimePoint(t=0), end=TimePoint(t=60)))
        with self.assertRaises(ValueError):
            time_slot_series.append(TimeSlot(start=TimePoint(t=120), end=TimePoint(t=180)))

        time_slot_series.append(TimeSlot(start=TimePoint(t=60), end=TimePoint(t=120)))

        self.assertEqual(len(time_slot_series),2)
        self.assertEqual(time_slot_series[0].start.t,0)
        self.assertEqual(time_slot_series[0].end.t,60)
        self.assertEqual(time_slot_series[1].start.t,60)
        self.assertEqual(time_slot_series[1].end.t,120) 
        
        # Test timezone
        time_slot_series = TimeSlotSeries()
        time_slot_series.append(TimeSlot(start=TimePoint(t=0, tz='Europe/Rome'), end=TimePoint(t=60, tz='Europe/Rome')))
        with self.assertRaises(ValueError):
            time_slot_series.append(TimeSlot(start=TimePoint(t=60), end=TimePoint(t=210)))
        time_slot_series.append(TimeSlot(start=TimePoint(t=60, tz='Europe/Rome'), end=TimePoint(t=120, tz='Europe/Rome')))

        # Test slot unit
        self.assertEqual(time_slot_series.resolution, Unit(60.0))

        # Test resolution 
        self.assertEqual(time_slot_series.resolution, Unit(60))
        

    def test_DataSlotSeries(self):
        data_slot_series = DataSlotSeries()
        data_slot_series.append(DataSlot(start=Point(1), end=Point(2), data='hello'))
        data_slot_series.append(DataSlot(start=Point(2), end=Point(3), data='hola'))
        self.assertEqual(data_slot_series[0].start.coordinates[0],1)
        self.assertEqual(data_slot_series[0].end.coordinates[0],2)
        self.assertEqual(data_slot_series[0].data,'hello')

 
    def test_DataTimeSlotSeries(self):
         
        with self.assertRaises(TypeError):
            DataTimeSlotSeries(DataTimeSlot(start=TimePoint(t=60), end=TimePoint(t=120), data=23.8),
                               DataTimeSlot(start=TimePoint(t=120), end=TimePoint(t=180), data=24.1),
                               DataTimeSlot(start=TimePoint(t=180), end=TimePoint(t=240), data=None))

        with self.assertRaises(ValueError):
            DataTimeSlotSeries(DataTimeSlot(start=TimePoint(t=180), end=TimePoint(t=20), data=24.1),
                               DataTimeSlot(start=TimePoint(t=60), end=TimePoint(t=120), data=23.8))
 
             
        with self.assertRaises(ValueError):
            DataTimeSlotSeries(DataTimeSlot(start=TimePoint(t=60), end=TimePoint(t=120), data=23.8),
                               DataTimeSlot(start=TimePoint(t=180), end=TimePoint(t=20), data=24.1))
 
        with self.assertRaises(ValueError):
            DataTimeSlotSeries(DataTimeSlot(start=TimePoint(t=60), end=TimePoint(t=120), data=23.8),
                               DataTimeSlot(start=TimePoint(t=60), end=TimePoint(t=120), data=24.1))

        data_time_slot_series =  DataTimeSlotSeries(DataTimeSlot(start=TimePoint(t=60),  end=TimePoint(t=120), data=23.8),
                                                    DataTimeSlot(start=TimePoint(t=120), end=TimePoint(t=180), data=24.1),
                                                    DataTimeSlot(start=TimePoint(t=180), end=TimePoint(t=240), data=23.1))
                                                                                                 
                                                 
        data_time_slot_series.append(DataTimeSlot(start=TimePoint(t=240), end=TimePoint(t=300), data=22.7))
        self.assertTrue(len(data_time_slot_series), 4)

        # Try to append a different data type
        data_time_slot_series = DataTimeSlotSeries(DataTimeSlot(start=TimePoint(t=60), end=TimePoint(t=120), data=23.8))
        with self.assertRaises(TypeError):
            data_time_slot_series.append(DataTimeSlot(start=TimePoint(t=120), end=TimePoint(t=180), data={'a':56}))

        # Try to append the same data type but with different cardinality
        data_time_slot_series = DataTimeSlotSeries(DataTimeSlot(start=TimePoint(t=60), end=TimePoint(t=120), data=[23.8]))
        with self.assertRaises(ValueError):
            data_time_slot_series.append(DataTimeSlot(start=TimePoint(t=120), end=TimePoint(t=180), data=[23.8,31.3]))

        # Try to append the same data type but with different keys
        data_time_slot_series = DataTimeSlotSeries(DataTimeSlot(start=TimePoint(t=60), end=TimePoint(t=120), data={'a':56, 'b':67}))
        with self.assertRaises(ValueError):
            data_time_slot_series.append(DataTimeSlot(start=TimePoint(t=120), end=TimePoint(t=180), data={'a':56, 'c':67}))            

        # Test with units
        self.assertEqual(data_time_slot_series.resolution, Unit(60.0))
        self.assertEqual(DataTimeSlotSeries(DataTimeSlot(start=TimePoint(t=60), end=TimePoint(t=120), data=23.8, unit=TimeUnit('60s'))).resolution, TimeUnit('60s'))
        
        data_time_slot_series = DataTimeSlotSeries()
        prev_t    = 1595862221
        for _ in range (0, 10):
            t    = prev_t + 60
            data_time_slot_series.append(DataTimeSlot(start=TimePoint(t=t), unit=TimeUnit('60s'), data=[5]))    
            prev_t    = t
        self.assertEqual(len(data_time_slot_series),10)
        self.assertEqual(data_time_slot_series[0].unit,TimeUnit('60s'))
        self.assertEqual(data_time_slot_series[0].unit, Unit(60))

        # Time series with calendar time units
        data_time_slot_series = DataTimeSlotSeries()
        prev_t    = 1595862221
        for _ in range (0, 10):
            t    = prev_t + 86400
            data_time_slot_series.append(DataTimeSlot(start=TimePoint(t=t), unit=TimeUnit('1D'), data=[5]))    
            prev_t    = t
        self.assertEqual(len(data_time_slot_series),10)
        self.assertEqual(data_time_slot_series[0].unit,TimeUnit('1D'))

        # Test creating a time series from a Pandas data frame 
        df = pd.read_csv(TEST_DATA_PATH+'csv/format4.csv', header=0, parse_dates=[0], index_col=0)
        data_time_slot_series = DataTimeSlotSeries(df)
        self.assertEqual(len(data_time_slot_series),5)
        self.assertEqual(data_time_slot_series[0].start.dt,dt(2020,4,3,0,0,0))
        self.assertEqual(data_time_slot_series[4].end.dt,dt(2020,4,3,10,0,0))

        # Test creating a Pandas data frame from a time series
        self.assertEqual(len(data_time_slot_series.df),5)
        self.assertEqual(data_time_slot_series.df.index[0],dt(2020,4,3,0,0,0))
        self.assertEqual(list(data_time_slot_series.df.columns),['C','RH'])
        self.assertEqual(data_time_slot_series.df.iloc[0][0],21.7)
        self.assertEqual(data_time_slot_series.df.iloc[0][1],54.9)

        # Test loading a Pandas data frame as a time series        
        loaded_data_time_slot_series = DataTimeSlotSeries(data_time_slot_series.df)
        self.assertEqual(loaded_data_time_slot_series[0].start.dt,dt(2020,4,3,0,0,0))
        self.assertEqual(loaded_data_time_slot_series[4].end.dt,dt(2020,4,3,10,0,0))
        self.assertEqual((loaded_data_time_slot_series[0].data['C']), 21.7)
        self.assertEqual((loaded_data_time_slot_series[0].data['RH']), 54.9)

        # Test resolution
        data_time_slot_series =  DataTimeSlotSeries(DataTimeSlot(start=TimePoint(t=60),  end=TimePoint(t=120), data=23.8),
                                                    DataTimeSlot(start=TimePoint(t=120), end=TimePoint(t=180), data=24.1),
                                                    DataTimeSlot(start=TimePoint(t=180), end=TimePoint(t=240), data=23.1))
        self.assertEqual(data_time_slot_series.resolution, Unit(60))
         
        data_time_slot_series =  DataTimeSlotSeries(DataTimeSlot(start=TimePoint(t=60),  unit=TimeUnit('1m'), data=23.8),
                                                    DataTimeSlot(start=TimePoint(t=120), unit=TimeUnit('1m'), data=24.1),
                                                    DataTimeSlot(start=TimePoint(t=180), unit=TimeUnit('1m'), data=23.1)) 
        self.assertEqual(data_time_slot_series.resolution, TimeUnit('1m'))
 

        # Test change timezone
        from ..time import timezonize
        data_time_slot_series_UTC =  DataTimeSlotSeries(DataTimeSlot(start=TimePoint(t=60),  end=TimePoint(t=120), data=23.8),
                                                        DataTimeSlot(start=TimePoint(t=120), end=TimePoint(t=180), data=24.1),
                                                        DataTimeSlot(start=TimePoint(t=180), end=TimePoint(t=240), data=23.1))

        data_time_slot_series_UTC.change_timezone('Europe/Rome')
        self.assertEqual(data_time_slot_series_UTC.tz, timezonize('Europe/Rome'))
        self.assertEqual(data_time_slot_series_UTC[0].tz, timezonize('Europe/Rome'))
        self.assertEqual(data_time_slot_series_UTC[0].start.tz, timezonize('Europe/Rome'))
        self.assertEqual(data_time_slot_series_UTC[0].end.tz, timezonize('Europe/Rome'))

    
        # Test get item by string key (filter on data labels). More testing is done in the operation tests
        data_time_slot_series =  DataTimeSlotSeries(DataTimeSlot(start=TimePoint(t=60),  end=TimePoint(t=120), data={'a':1, 'b':2}))
        self.assertEqual(data_time_slot_series['a'][0].data, {'a': 1})


        # Test data keys and rename a data key
        # TODO: move this test where to test data point series ?
        data_time_slot_series = DataTimeSlotSeries(DataTimeSlot(dt=dt(2015,10,27,0,0,0), unit='1D', data={'a':23.8, 'b':1}),
                                                   DataTimeSlot(dt=dt(2015,10,28,0,0,0), unit='1D', data={'a':24.1, 'b':2}),
                                                   DataTimeSlot(dt=dt(2015,10,29,0,0,0), unit='1D', data={'a':23.1, 'b':3}))
        
        self.assertEqual(data_time_slot_series.data_labels(), ['a','b'])
        data_time_slot_series.rename_data_label('b','c')
        self.assertEqual(data_time_slot_series.data_labels(), ['a','c'])
        
        with self.assertRaises(KeyError):
            data_time_slot_series.rename_data_label('notexistent_key','c')


class TestSeriesSlices(unittest.TestCase):

    def test_SeriesSlice(self):
        series = DataTimePointSeries()
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
        
        from ..utilities import compute_validity_regions
        validity_regions = compute_validity_regions(series)
        for point in series:
            point.valid_from=validity_regions[point.t][0]
            point.valid_to=validity_regions[point.t][1]
        
        series_slice = SeriesSlice(series, 2, 5)
        
        self.assertEqual(len(series_slice), 3)
        
        for i, point in enumerate(series_slice):
            self.assertEqual(point.t, 2+i)

        self.assertEqual(series_slice[0].t, 2)
        self.assertEqual(series_slice[1].t, 3)
        self.assertEqual(series_slice[2].t, 4)
        self.assertEqual(series_slice[-1].t, 4)
        
        # Test extra attributes
        self.assertEqual(str(series_slice.resolution), '1s')
        self.assertEqual(series_slice.data_labels(), ['value'])
        
        with self.assertRaises(AttributeError):
            series_slice.diff()
        with self.assertRaises(AttributeError):
            series_slice.blackmagic()


    def test_SeriesDenseSlice(self):
        series = DataTimePointSeries()
        series.append(DataTimePoint(t = 0, data = {'value': 0}))
        series.append(DataTimePoint(t = 1, data = {'value': 1}))
        series.append(DataTimePoint(t = 2, data = {'value': 2}))
        series.append(DataTimePoint(t = 3, data = {'value': 3}))
        series.append(DataTimePoint(t = 4, data = {'value': 4}))
        series.append(DataTimePoint(t = 7, data = {'value': 7}))
        series.append(DataTimePoint(t = 8, data = {'value': 8}))
        series.append(DataTimePoint(t = 9, data = {'value': 9}))
        
        from ..utilities import compute_validity_regions
        validity_regions = compute_validity_regions(series)
        for point in series:
            point.valid_from=validity_regions[point.t][0]
            point.valid_to=validity_regions[point.t][1]
        
        series_slice = SeriesSlice(series, 2, 8, dense=True)
        
        self.assertEqual(len(series_slice), 7)
        
        series_slice_materialized=[]
        for point in series_slice:
            series_slice_materialized.append(point)

        self.assertEqual(series_slice_materialized[0].t, 2)
        self.assertEqual(series_slice_materialized[1].t, 3)
        self.assertEqual(series_slice_materialized[2].t, 4)
        self.assertEqual(series_slice_materialized[3].t, 5.5)
        self.assertEqual(series_slice_materialized[3].data['value'], 5.5)
        self.assertEqual(series_slice_materialized[4].t, 7)
        self.assertEqual(series_slice_materialized[5].t, 8)
        self.assertEqual(series_slice_materialized[6].t, 9)
        




























