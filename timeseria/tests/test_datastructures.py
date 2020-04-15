import unittest
import os

from ..datastructures import Point, TimePoint, DataPoint, DataTimePoint
from ..datastructures import Slot, TimeSlot, DataSlot, DataTimeSlot
from ..datastructures import Serie
from ..datastructures import DataPointSerie, TimePointSerie, DataTimePointSerie
from ..datastructures import DataSlotSerie, TimeSlotSerie, DataTimeSlotSerie


# Setup logging
import logging
logging.basicConfig(level=os.environ.get('LOGLEVEL') if os.environ.get('LOGLEVEL') else 'CRITICAL')


class TestSeries(unittest.TestCase):

    def test_Serie(self):
        
        serie = Serie(1,4,5)
        self.assertEqual(serie, [1,4,5])
        self.assertEqual(len(serie), 3)
        
        serie.append(8)
        self.assertEqual(serie, [1,4,5,8])
        self.assertEqual(len(serie), 4)  
        
        # Demo class that implements the __succedes__ operation
        class IntegerNumber(int):
            def __succedes__(self, other):
                if other+1 != self:
                    return False
                else:
                    return True

        one = IntegerNumber(1)
        two = IntegerNumber(2)
        three = IntegerNumber(3)

        self.assertTrue(two.__succedes__(one))
        self.assertFalse(one.__succedes__(two))
        self.assertFalse(three.__succedes__(one))

        with self.assertRaises(ValueError):
            Serie(one, three)

        with self.assertRaises(ValueError):
            Serie(three, two)

        Serie(one, two, three)
        
        # TODO: do we want the following behavior? (cannot mix types  even if they are child classes)
        with self.assertRaises(TypeError):
            Serie(one, two, three, 4)

        # Define a Serie with fixed type
        class FloatSerie(Serie):
            __TYPE__ = float

        with self.assertRaises(TypeError):
            FloatSerie(1,4,5, 'hello', None)

        with self.assertRaises(TypeError):
            FloatSerie(1,4,5)

        with self.assertRaises(TypeError):
            FloatSerie(1.0,4.0,None,5.0)
            
        floatSerie = FloatSerie(1.0,4.0,5.0)
        self.assertEqual(floatSerie, [1.0,4.0,5.0])
        self.assertEqual(len(floatSerie), 3)



class TestPoints(unittest.TestCase):

    def test_Point(self):
        
        point = Point(x=1, y=2)
        self.assertEqual(point.x,1)
        self.assertEqual(point.y,2)

        point = Point(h=1, i=2)
        self.assertEqual(point.h,1)
        self.assertEqual(point.i,2)
        
        point1 = Point(x=1, y=2)
        point2 = Point(x=1, y=2)
        point3 = Point(x=5, z=3)

        self.assertEqual(point1,point2)
        self.assertNotEqual(point1,point3)


    def test_TimePoint(self):
        
        with self.assertRaises(Exception):
            TimePoint(x=1)

        with self.assertRaises(Exception):
            TimePoint(t=2, x=7)
        
        timePoint = TimePoint(t=5)
        self.assertEqual(timePoint.t,5)


    def test_DataPoints(self):
        
        with self.assertRaises(Exception):
            DataPoint(x=1)

        with self.assertRaises(Exception):
            DataPoint(data='hello')
        
        dataPoint = DataPoint(x=1, y=2, data='hello')
        self.assertEqual(dataPoint.coordinates, {'x': 1, 'y':2}) 
        self.assertEqual(dataPoint.x,1)
        self.assertEqual(dataPoint.y,2)
        self.assertEqual(dataPoint.data,'hello')
        
        dataPoint1 = DataPoint(x=1, y=2, data='hello')
        dataPoint2 = DataPoint(x=1, y=2, data='hello')
        dataPoint3 = DataPoint(x=1, y=2, data='hola')
        
        self.assertEqual(dataPoint1, dataPoint2)
        self.assertNotEqual(dataPoint1, dataPoint3)


    def test_TimeDataPoints(self):
        
        with self.assertRaises(Exception):
            DataTimePoint(x=1)

        with self.assertRaises(Exception):
            DataTimePoint(data='hello')

        with self.assertRaises(Exception):
            DataTimePoint(x=1, data='hello')
        
        dataTimePoint = DataTimePoint(t=6, data='hello')        
        self.assertEqual(dataTimePoint.coordinates, {'t': 6}) 
        self.assertEqual(dataTimePoint.t,6)
        self.assertEqual(dataTimePoint.data,'hello')
        


class TestPointSeries(unittest.TestCase):


    def test_TimePointSerie(self):
        
        timePointSerie = TimePointSerie()
        timePointSerie.append(TimePoint(t=60))


    def test_DataPointSerie(self):
        pass
        # TODO: either implement the "__suceedes__" for unidimensional DataPoints or remove it completely.
        #dataPointSerie = DataPointSerie()
        #dataPointSerie.append(DataPoint(x=60, data='hello'))
        #dataPointSerie.append(DataPoint(x=61, data='hello'))


    def test_DataTimePointSerie(self):
        
        with self.assertRaises(TypeError):
            DataTimePointSerie(DataTimePoint(t=60, data=23.8),
                               DataTimePoint(t=120, data=24.1),
                               DataTimePoint(t=180, data=None))
            
        with self.assertRaises(ValueError):
            DataTimePointSerie(DataTimePoint(t=60, data=23.8),
                               DataTimePoint(t=30, data=24.1))

        with self.assertRaises(ValueError):
            DataTimePointSerie(DataTimePoint(t=60, data=23.8),
                               DataTimePoint(t=60, data=24.1))
        
        dataTimePointSerie = DataTimePointSerie(DataTimePoint(t=60, data=23.8),
                                                DataTimePoint(t=120, data=24.1),
                                                DataTimePoint(t=180, data=23.9),
                                                DataTimePoint(t=240, data=23.1),
                                                DataTimePoint(t=300, data=22.7))
                                                
        dataTimePointSerie.append(DataTimePoint(t=360, data=21.9))
        self.assertTrue(len(dataTimePointSerie), 6)
   

        dataTimePointSerie = DataTimePointSerie(DataTimePoint(t=60, data=23.8),
                                                DataTimePoint(t=120, data=24.1),
                                                DataTimePoint(t=240, data=23.1),
                                                DataTimePoint(t=300, data=22.7))
        self.assertTrue(len(dataTimePointSerie), 5)

      
        # Try to append a different data type
        dataTimePointSerie = DataTimePointSerie(DataTimePoint(t=60, data=[23.8]))
        with self.assertRaises(TypeError):
            dataTimePointSerie.append(DataTimePoint(t=120, data={'a':56}))

        # Try to append the same data type but with different cardinality
        dataTimePointSerie = DataTimePointSerie(DataTimePoint(t=60, data=[23.8]))
        with self.assertRaises(ValueError):
            dataTimePointSerie.append(DataTimePoint(t=180, data=[23.8,31.3]))

        # Try to append the same data type but with different keys
        dataTimePointSerie = DataTimePointSerie(DataTimePoint(t=60, data={'a':56, 'b':67}))
        with self.assertRaises(ValueError):
            dataTimePointSerie.append(DataTimePoint(t=180, data={'a':56, 'c':67}))            




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

        # A slot needs a start and and end with he same coordinates
        with self.assertRaises(ValueError):
            Slot(start=Point(x=1), end=Point(y=2))        

        # No zero-duration allowed:
        with self.assertRaises(ValueError):
            Slot(start=Point(x=1), end=Point(x=1))


        slot = Slot(start=Point(x=1), end=Point(x=2)) 
        self.assertEqual(slot.start,Point(x=1))
        self.assertEqual(slot.end,Point(x=2))
        
        slot1 = Slot(start=Point(x=1), end=Point(x=2))
        slot2 = Slot(start=Point(x=1), end=Point(x=2))
        slot3 = Slot(start=Point(x=1), end=Point(x=3))
        
        self.assertEqual(slot1,slot2)
        self.assertNotEqual(slot1,slot3)

  
    def test_TimeSlot(self):

        # A timeSlot needs TimePoints
        with self.assertRaises(TypeError):
            TimeSlot(start=Point(x=1), end=Point(x=2))

        # No zero-duration allowed:
        with self.assertRaises(ValueError):
            TimeSlot(start=TimePoint(t=1), end=TimePoint(t=1))

        timeSlot = TimeSlot(start=TimePoint(t=1), end=TimePoint(t=2))
        self.assertEqual(timeSlot.start.t,1)
        self.assertEqual(timeSlot.end.t,2)

        timeSlot1 = TimeSlot(start=TimePoint(t=1), end=TimePoint(t=2))
        timeSlot2 = TimeSlot(start=TimePoint(t=2), end=TimePoint(t=3))
        timeSlot3 = TimeSlot(start=TimePoint(t=3), end=TimePoint(t=4))

        # Test succession
        self.assertTrue(timeSlot2.__succedes__(timeSlot1))
        self.assertFalse(timeSlot1.__succedes__(timeSlot2))
        self.assertFalse(timeSlot3.__succedes__(timeSlot1))
        
        # Duration
        self.assertEqual(timeSlot1.duration,1)
        
        
 
 
    def test_DataSlot(self):
        
        with self.assertRaises(Exception):
            DataSlot(start=Point(x=1), end=Point(x=2))
 
        with self.assertRaises(TypeError):
            DataSlot(data='hello')

        dataSlot = DataSlot(start=Point(x=1), end=Point(x=2), data='hello')
        self.assertEqual(dataSlot.start.x,1)
        self.assertEqual(dataSlot.end.x,2)
        self.assertEqual(dataSlot.data,'hello')

        dataSlot1 = DataSlot(start=Point(x=1), end=Point(x=2), data='hello')
        dataSlot2 = DataSlot(start=Point(x=1), end=Point(x=2), data='hello')
        dataSlot3 = DataSlot(start=Point(x=1), end=Point(x=2), data='hola')

        self.assertEqual(dataSlot1, dataSlot2)
        self.assertNotEqual(dataSlot1, dataSlot3)


  
    def test_DataTimeSlots(self):

        with self.assertRaises(Exception):
            DataTimeSlot(start=Point(x=1), end=Point(x=2))
 
        with self.assertRaises(TypeError):
            DataTimeSlot(data='hello')

        with self.assertRaises(TypeError):
            DataTimeSlot(start=Point(x=1), end=Point(x=2), data='hello')


        dataTimeSlot = DataTimeSlot(start=TimePoint(t=1), end=TimePoint(t=2), data='hello')
        self.assertEqual(dataTimeSlot.start.t,1)
        self.assertEqual(dataTimeSlot.end.t,2)
        self.assertEqual(dataTimeSlot.data,'hello')

        dataTimeSlot1 = DataTimeSlot(start=TimePoint(t=1), end=TimePoint(t=2), data='hello')
        dataTimeSlot2 = DataTimeSlot(start=TimePoint(t=1), end=TimePoint(t=2), data='hello')
        dataTimeSlot3 = DataTimeSlot(start=TimePoint(t=1), end=TimePoint(t=2), data='hola')

        self.assertEqual(dataTimeSlot1, dataTimeSlot2)
        self.assertNotEqual(dataTimeSlot1, dataTimeSlot3)


class TestSlotSeries(unittest.TestCase):
 
    def test_TimeSlotSerie(self):
         
        timeSlotSerie = TimeSlotSerie()
        timeSlotSerie.append(TimeSlot(start=TimePoint(t=0), end=TimePoint(t=60)))
        with self.assertRaises(ValueError):
            timeSlotSerie.append(TimeSlot(start=TimePoint(t=0), end=TimePoint(t=60)))
        with self.assertRaises(ValueError):
            timeSlotSerie.append(TimeSlot(start=TimePoint(t=120), end=TimePoint(t=180)))

        timeSlotSerie.append(TimeSlot(start=TimePoint(t=60), end=TimePoint(t=120)))

        self.assertEqual(len(timeSlotSerie),2)
        self.assertEqual(timeSlotSerie[0].start.t,0)
        self.assertEqual(timeSlotSerie[0].end.t,60)
        self.assertEqual(timeSlotSerie[1].start.t,60)
        self.assertEqual(timeSlotSerie[1].end.t,120) 

 
    def test_DataSlotSerie(self):
        pass
        # TODO: either implement the "__suceedes__" for unidimensional DataSlots or remove it completely.
        #dataSlotSerie = DataSlotSerie()
        #dataSlotSerie.append(DataSlot(start=Point(x=1), end=Point(x=2), data='hello'))
        #dataSlotSerie.append(DataSlot(start=Point(x=1), end=Point(x=2), data='hola'))
        #self.assertEqual(dataSlotSerie[0].start.x,1)
        #self.assertEqual(dataSlotSerie[0].end.x,2)
        #self.assertEqual(dataSlotSerie[0].data,'hello')

 
    def test_DataTimeSlotSerie(self):
         
        with self.assertRaises(TypeError):
            DataTimeSlotSerie(DataTimeSlot(start=TimePoint(t=60), end=TimePoint(t=120), data=23.8),
                              DataTimeSlot(start=TimePoint(t=120), end=TimePoint(t=180), data=24.1),
                              DataTimeSlot(start=TimePoint(t=180), end=TimePoint(t=240), data=None))

        with self.assertRaises(ValueError):
            DataTimeSlotSerie(DataTimeSlot(start=TimePoint(t=180), end=TimePoint(t=20), data=24.1),
                              DataTimeSlot(start=TimePoint(t=60), end=TimePoint(t=120), data=23.8))
 
             
        with self.assertRaises(ValueError):
            DataTimeSlotSerie(DataTimeSlot(start=TimePoint(t=60), end=TimePoint(t=120), data=23.8),
                              DataTimeSlot(start=TimePoint(t=180), end=TimePoint(t=20), data=24.1))
 
        with self.assertRaises(ValueError):
            DataTimeSlotSerie(DataTimeSlot(start=TimePoint(t=60), end=TimePoint(t=120), data=23.8),
                              DataTimeSlot(start=TimePoint(t=60), end=TimePoint(t=120), data=24.1))

        dataTimeSlotSerie =  DataTimeSlotSerie(DataTimeSlot(start=TimePoint(t=60), end=TimePoint(t=120), data=23.8),
                                               DataTimeSlot(start=TimePoint(t=120), end=TimePoint(t=180), data=24.1),
                                               DataTimeSlot(start=TimePoint(t=180), end=TimePoint(t=240), data=23.1))
                                                                                                 
                                                 
        dataTimeSlotSerie.append(DataTimeSlot(start=TimePoint(t=240), end=TimePoint(t=300), data=22.7))
        self.assertTrue(len(dataTimeSlotSerie), 4)
        return

        # Try to append a different data type
        dataTimeSlotSerie = DataTimeSlotSerie(DataTimeSlot(start=TimePoint(t=60), end=TimePoint(t=120), data=23.8))
        with self.assertRaises(TypeError):
            dataTimeSlotSerie.append(DataTimeSlot(start=TimePoint(t=120), end=TimePoint(t=180), data={'a':56}))

        # Try to append the same data type but with different cardinality
        dataTimeSlotSerie = DataTimeSlotSerie(DataTimeSlot(start=TimePoint(t=60), end=TimePoint(t=120), data=[23.8]))
        with self.assertRaises(ValueError):
            dataTimeSlotSerie.append(DataTimeSlot(start=TimePoint(t=120), end=TimePoint(t=180), data=[23.8,31.3]))

        # Try to append the same data type but with different keys
        dataTimeSlotSerie = DataTimeSlotSerie(DataTimeSlot(start=TimePoint(t=60), end=TimePoint(t=120), data={'a':56, 'b':67}))
        with self.assertRaises(ValueError):
            dataTimeSlotSerie.append(DataTimeSlot(start=TimePoint(t=120), end=TimePoint(t=180), data={'a':56, 'c':67}))            





