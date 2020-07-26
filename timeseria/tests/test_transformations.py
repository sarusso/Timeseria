import unittest
import os
from ..datastructures import Point, TimePoint, DataPoint, DataTimePoint, DataTimeSlotSeries, DataTimeSlot
from ..datastructures import Series, DataPointSeries, TimePointSeries, DataTimePointSeries
from ..storages import CSVFileStorage
#from ..operators import diff, min, slot, Slotter,
from ..transformations import Slotter
from ..time import dt, s_from_dt, dt_from_s, dt_from_str
from ..exceptions import InputException
from ..units import TimeUnit

# Setup logging
import logging
logging.basicConfig(level=os.environ.get('LOGLEVEL') if os.environ.get('LOGLEVEL') else 'CRITICAL')

# Set test data path
TEST_DATA_PATH = '/'.join(os.path.realpath(__file__).split('/')[0:-1]) + '/test_data/'


class TestSlotter(unittest.TestCase):

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
                                              data = {'temperature': 154+count},
                                              tz   = 'Europe/Rome')
                self.data_time_point_series_6.append(data_time_point)
            slider_dt = slider_dt + time_unit
            count += 1

        # The following time series has point with validity=1h

        # Time series from 2019,10,24,0,0,0 to 2019,10,31,0,0,0 (Europe/Rome), DST off -> 2 AM repeated
        from_dt   = dt(2019,10,24,0,0,0, tzinfo='Europe/Rome')
        to_dt     = dt(2019,10,31,0,0,0, tzinfo='Europe/Rome')
        time_unit = TimeUnit('1h') 
        self.data_time_point_series_7 = DataTimePointSeries()
        slider_dt = from_dt
        count = 0
        while slider_dt < to_dt:
            if count not in []:
                data_time_point = DataTimePoint(t    = s_from_dt(slider_dt),
                                                data = {'temperature': 154+count},
                                                tz   = 'Europe/Rome')
                self.data_time_point_series_7.append(data_time_point)
            slider_dt = slider_dt + time_unit
            count += 1


    def test_unit_to_TimeUnit(self):
        
        Slotter._unit_to_TimeUnit('1h')
        Slotter._unit_to_TimeUnit(TimeUnit('1h'))
        
        with self.assertRaises(InputException):
            Slotter._unit_to_TimeUnit('NO')
        
        

    def test_detect_dataPoints_validity(self):
        
        data_time_point_series = CSVFileStorage(TEST_DATA_PATH + '/csv/humitemp_short.csv').get()
        self.assertEqual(Slotter._detect_dataPoints_validity(data_time_point_series), 61)
        
        data_time_point_series = CSVFileStorage(TEST_DATA_PATH + '/csv/shampoo_sales.csv', time_column = 'Month', time_format = '%y-%m').get()
        self.assertEqual(Slotter._detect_dataPoints_validity(data_time_point_series), 2678400)
        # TODO: 2678400/60/60/24 = 31 --> detect month? same for day, week, year?

        data_time_point_series = CSVFileStorage(TEST_DATA_PATH + '/csv/temperature.csv').get()
        self.assertEqual(Slotter._detect_dataPoints_validity(data_time_point_series), 600)

        data_time_point_series = CSVFileStorage(TEST_DATA_PATH + '/csv/temp_short_1h.csv').get()
        self.assertEqual(Slotter._detect_dataPoints_validity(data_time_point_series), 3600)


    def test_slot(self):
        
        # Time series from 16:58:00 to 17:32:00 (Europe/Rome), no from/to, extremes excluded (default)
        data_time_slot_series = Slotter(unit=600).process(self.data_time_point_series_1)

        # operators.slot()
        # operations.slot .merge .min .max .sum .aggreate
        # operators.slotter .merger .min .max .sum .diff
        # slotter.apply max.apply() merge.apply()
        
        # Merge.process(time_series1, time_series2)
        # Slotter('1h').process(time_series)
        # Slotter('1h').process(time_series)
        
        
        # Slots('1h').process(time_series)
        
        # Merge().process(time_series)
        # Min().process
        # Min.process
        # Merge.merge
        
        
        self.assertEqual(len(data_time_slot_series), 3)
        self.assertEqual(data_time_slot_series[0].start.dt, dt_from_str('2015-07-04 17:00:00+02:00'))
        self.assertEqual(data_time_slot_series[1].start.dt, dt_from_str('2015-07-04 17:10:00+02:00'))
        self.assertEqual(data_time_slot_series[2].start.dt, dt_from_str('2015-07-04 17:20:00+02:00'))
        return
     
        # Time series from 16:58:00 to 17:32:00 (Europe/Rome), no from/to, extremes included.
        data_time_slot_series = Slotter('10m').slot(self.data_time_point_series_1,  include_extremes = True)
        self.assertEqual(len(data_time_slot_series), 4)
        self.assertEqual(data_time_slot_series[0].start.dt, dt_from_str('2015-07-04 16:50:00+02:00'))
        self.assertEqual(data_time_slot_series[1].start.dt, dt_from_str('2015-07-04 17:00:00+02:00'))
        self.assertEqual(data_time_slot_series[2].start.dt, dt_from_str('2015-07-04 17:10:00+02:00'))
        self.assertEqual(data_time_slot_series[3].start.dt, dt_from_str('2015-07-04 17:20:00+02:00'))
        # TODO: the following missing is a bug..
        #self.assertEqual(data_time_slot_series[4].start.dt, dt_from_str('2015-07-04 17:30:00+02:00'))
  
        # Time series from 16:58:00 to 17:32:00 (Europe/Rome), from 15:50 to 17:40
        data_time_slot_series = Slotter('10m').slot(self.data_time_point_series_1,
                                    from_t = s_from_dt(dt_from_str('2015-07-04 16:50:00+02:00')),
                                    to_t   = s_from_dt(dt_from_str('2015-07-04 17:40:00+02:00')))
  
        self.assertEqual(len(data_time_slot_series), 4)
        self.assertEqual(data_time_slot_series[0].start.dt, dt_from_str('2015-07-04 16:50:00+02:00'))
        self.assertEqual(data_time_slot_series[1].start.dt, dt_from_str('2015-07-04 17:00:00+02:00'))
        self.assertEqual(data_time_slot_series[2].start.dt, dt_from_str('2015-07-04 17:10:00+02:00'))
        self.assertEqual(data_time_slot_series[3].start.dt, dt_from_str('2015-07-04 17:20:00+02:00'))
        # TODO: the following missing is a bug..
        #self.assertEqual(data_time_slot_series[4].start.dt, dt_from_str('2015-07-04 17:30:00+02:00'))
   
        # Time series from 2019,10,1,1,0,0 to 2019,10,1,6,0,0 (Europe/Rome), validity=15m
        data_time_slot_series = Slotter('1h').slot(self.data_time_point_series_6)
        self.assertEqual(len(data_time_slot_series), 4)
        self.assertEqual(str(data_time_slot_series[0].start.dt), str('2019-10-01 01:00:00+02:00'))
        self.assertEqual(str(data_time_slot_series[1].start.dt), str('2019-10-01 02:00:00+02:00'))
        self.assertEqual(str(data_time_slot_series[2].start.dt), str('2019-10-01 03:00:00+02:00'))
        self.assertEqual(str(data_time_slot_series[3].start.dt), str('2019-10-01 04:00:00+02:00'))
 
        # Test changing timezone of the series
        self.data_time_point_series_6.tz = 'America/New_York'
        data_time_slot_series = Slotter('1h').slot(self.data_time_point_series_6)
        self.assertEqual(len(data_time_slot_series), 4)
        self.assertEqual(str(data_time_slot_series[0].start.dt), str('2019-09-30 19:00:00-04:00'))
        self.assertEqual(str(data_time_slot_series[1].start.dt), str('2019-09-30 20:00:00-04:00'))
        self.assertEqual(str(data_time_slot_series[2].start.dt), str('2019-09-30 21:00:00-04:00'))
        self.assertEqual(str(data_time_slot_series[3].start.dt), str('2019-09-30 22:00:00-04:00'))
         
        # Time series from 2019,10,24,0,0,0 to 2019,10,31,0,0,0 (Europe/Rome), DST off -> 2 AM repeated
        data_time_slot_series = Slotter('1h').slot(self.data_time_point_series_7)
 
        self.assertEqual(len(data_time_slot_series), 168)
        self.assertEqual(str(data_time_slot_series[73].start.dt), str('2019-10-27 01:00:00+02:00'))
        self.assertEqual(str(data_time_slot_series[74].start.dt), str('2019-10-27 02:00:00+02:00'))
        self.assertEqual(str(data_time_slot_series[75].start.dt), str('2019-10-27 02:00:00+01:00'))
        self.assertEqual(str(data_time_slot_series[76].start.dt), str('2019-10-27 03:00:00+01:00'))
 
 
        # Work in progress in the following
        #print('--------------------------------------------')
        #for item in self.data_time_point_series_6: print(item)
        #print('--------------------------------------------')
        #for i, item in enumerate(data_time_slot_series): print(i, item)
        #print('--------------------------------------------')
        
        # Time series from 2019,10,24,0,0,0 to 2019,10,31,0,0,0 (Europe/Rome), DST off -> 2 AM repeated
        #data_time_slot_series = slotter.process(self.data_time_point_series_7, from_t=1571868600, to_t=1571873400)
        #for item in data_time_slot_series:
        #    print(item)
        # This is a downsampling, there are coverage=None.. TOOD: fix me
  
        # 1-day sloter TODO: add all the "logical" time part, nothing works here...
        #slotter = Slotter('1D')
        # Time series from 2019,10,24,0,0,0 to 2019,10,31,0,0,0 (Europe/Rome), DST off -> 2 AM repeated
        #data_time_slot_series = slotter.process(self.data_time_point_series_7)
        #for item in data_time_slot_series:
        #    print(item)
 
 
 

 
 
 
 
 
 
 









