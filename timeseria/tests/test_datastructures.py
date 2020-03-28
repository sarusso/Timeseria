import unittest
import os
from ..datastructures import Point, TimePoint, DataPoint, DataTimePoint
from ..datastructures import Serie, DataTimePointSerie


# Setup logging
import logging
logging.basicConfig(level=os.environ.get('LOGLEVEL') if os.environ.get('LOGLEVEL') else 'CRITICAL')


class TestPoints(unittest.TestCase):

    def test_Point(self):
        
        point = Point(x=1, y=2)
        self.assertEqual(point.x,1)
        self.assertEqual(point.y,2)

        point = Point(h=1, i=2)
        self.assertEqual(point.h,1)
        self.assertEqual(point.i,2)


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
        self.assertEqual(dataPoint.x,1)
        self.assertEqual(dataPoint.y,2)
        self.assertEqual(dataPoint.data,'hello')


    def test_TimeDataPoints(self):
        
        with self.assertRaises(Exception):
            DataTimePoint(x=1)

        with self.assertRaises(Exception):
            DataTimePoint(data='hello')

        with self.assertRaises(Exception):
            DataTimePoint(x=1, data='ciao')
        
        dataTimePoint = DataTimePoint(t=6, data='hello')
        self.assertEqual(dataTimePoint.t,6)
        self.assertEqual(dataTimePoint.data,'hello')


class TestSeries(unittest.TestCase):

    def test_Serie(self):
        
        serie = Serie(1,4,5)
        self.assertEqual(serie, [1,4,5])
        self.assertEqual(len(serie), 3)
        
        serie.append(8)
        self.assertEqual(serie, [1,4,5,8])
        self.assertEqual(len(serie), 4)  
        
        
        _ = Serie(1,4,5, 'ciao', None)

        class FloatSerie(Serie):
            __TYPE__ = float

        with self.assertRaises(TypeError):
            FloatSerie(1,4,5, 'ciao', None)

        with self.assertRaises(TypeError):
            FloatSerie(1,4,5)

        with self.assertRaises(TypeError):
            FloatSerie(1.0,4.0,None,5.0)
            
        floatSerie = FloatSerie(1.0,4.0,5.0)
        self.assertEqual(floatSerie, [1.0,4.0,5.0])
        self.assertEqual(len(floatSerie), 3)

        floatSerie = FloatSerie(1.0,4.0,None,5.0, accept_None=True)
        self.assertEqual(floatSerie, [1.0,4.0,None,5.0])
        self.assertEqual(len(floatSerie), 4)


    def test_DataTimePoinSerie(self):
        
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
                                                DataTimePoint(t=180, data=None),
                                                DataTimePoint(t=240, data=23.1),
                                                DataTimePoint(t=300, data=22.7),
                                                accept_None=True)
                                                
                                                
        dataTimePointSerie.append(DataTimePoint(t=360, data=21.9))
        self.assertTrue(len(dataTimePointSerie), 6)
   

        dataTimePointSerie = DataTimePointSerie(DataTimePoint(t=60, data=23.8),
                                                DataTimePoint(t=120, data=24.1),
                                                DataTimePoint(t=240, data=23.1),
                                                DataTimePoint(t=300, data=22.7))
        self.assertTrue(len(dataTimePointSerie), 5)                  










