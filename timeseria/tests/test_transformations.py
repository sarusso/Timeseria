import unittest
import os
from ..datastructures import DataTimePoint, DataTimePointSeries
from ..storages import CSVFileStorage
from ..transformations import Resampler, Slotter, detect_dataPoints_validity, unit_to_TimeUnit 
from ..time import dt, s_from_dt, dt_from_str
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
        
        unit_to_TimeUnit('1h')
        unit_to_TimeUnit(TimeUnit('1h'))
        
        with self.assertRaises(InputException):
            unit_to_TimeUnit('NO')
        

    def test_detect_dataPoints_validity(self):
        
        data_time_point_series = CSVFileStorage(TEST_DATA_PATH + '/csv/humitemp_short.csv').get()
        self.assertEqual(detect_dataPoints_validity(data_time_point_series), 61)
        
        data_time_point_series = CSVFileStorage(TEST_DATA_PATH + '/csv/shampoo_sales.csv', time_column = 'Month', time_format = '%y-%m').get()
        self.assertEqual(detect_dataPoints_validity(data_time_point_series), 2678400)
        # TODO: 2678400/60/60/24 = 31 --> detect month? same for day, week, year?

        data_time_point_series = CSVFileStorage(TEST_DATA_PATH + '/csv/temperature.csv').get()
        self.assertEqual(detect_dataPoints_validity(data_time_point_series), 600)

        data_time_point_series = CSVFileStorage(TEST_DATA_PATH + '/csv/temp_short_1h.csv').get()
        self.assertEqual(detect_dataPoints_validity(data_time_point_series), 3600)


    def test_slot(self):
        
        # Time series from 16:58:00 to 17:32:00 (Europe/Rome), no from/to, extremes excluded (default)
        data_time_slot_series = Slotter('600s').process(self.data_time_point_series_1)        
        self.assertEqual(len(data_time_slot_series), 3)
        self.assertEqual(data_time_slot_series[0].start.dt, dt_from_str('2015-07-04 17:00:00+02:00'))
        self.assertEqual(data_time_slot_series[1].start.dt, dt_from_str('2015-07-04 17:10:00+02:00'))
        self.assertEqual(data_time_slot_series[2].start.dt, dt_from_str('2015-07-04 17:20:00+02:00'))
     
        # Time series from 16:58:00 to 17:32:00 (Europe/Rome), no from/to, extremes included.
        data_time_slot_series = Slotter('10m').process(self.data_time_point_series_1,  include_extremes = True)
        self.assertEqual(len(data_time_slot_series), 4)
        self.assertEqual(data_time_slot_series[0].start.dt, dt_from_str('2015-07-04 16:50:00+02:00'))
        self.assertEqual(data_time_slot_series[1].start.dt, dt_from_str('2015-07-04 17:00:00+02:00'))
        self.assertEqual(data_time_slot_series[2].start.dt, dt_from_str('2015-07-04 17:10:00+02:00'))
        self.assertEqual(data_time_slot_series[3].start.dt, dt_from_str('2015-07-04 17:20:00+02:00'))
        # TODO: the following missing is a bug..
        #self.assertEqual(data_time_slot_series[4].start.dt, dt_from_str('2015-07-04 17:30:00+02:00'))
  
        # Time series from 16:58:00 to 17:32:00 (Europe/Rome), from 15:50 to 17:40
        data_time_slot_series = Slotter('10m').process(self.data_time_point_series_1,
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
        data_time_slot_series = Slotter('1h').process(self.data_time_point_series_6)
        self.assertEqual(len(data_time_slot_series), 4)
        self.assertEqual(str(data_time_slot_series[0].start.dt), str('2019-10-01 01:00:00+02:00'))
        self.assertEqual(str(data_time_slot_series[1].start.dt), str('2019-10-01 02:00:00+02:00'))
        self.assertEqual(str(data_time_slot_series[2].start.dt), str('2019-10-01 03:00:00+02:00'))
        self.assertEqual(str(data_time_slot_series[3].start.dt), str('2019-10-01 04:00:00+02:00'))
 
        # Test changing timezone of the series
        self.data_time_point_series_6.tz = 'America/New_York'
        data_time_slot_series = Slotter('1h').process(self.data_time_point_series_6)
        self.assertEqual(len(data_time_slot_series), 4)
        self.assertEqual(str(data_time_slot_series[0].start.dt), str('2019-09-30 19:00:00-04:00'))
        self.assertEqual(str(data_time_slot_series[1].start.dt), str('2019-09-30 20:00:00-04:00'))
        self.assertEqual(str(data_time_slot_series[2].start.dt), str('2019-09-30 21:00:00-04:00'))
        self.assertEqual(str(data_time_slot_series[3].start.dt), str('2019-09-30 22:00:00-04:00'))
         
        # Time series from 2019,10,24,0,0,0 to 2019,10,31,0,0,0 (Europe/Rome), DST off -> 2 AM repeated
        data_time_slot_series = Slotter('1h').process(self.data_time_point_series_7)
 
        self.assertEqual(len(data_time_slot_series), 168)
        self.assertEqual(str(data_time_slot_series[73].start.dt), str('2019-10-27 01:00:00+02:00'))
        self.assertEqual(str(data_time_slot_series[74].start.dt), str('2019-10-27 02:00:00+02:00'))
        self.assertEqual(str(data_time_slot_series[75].start.dt), str('2019-10-27 02:00:00+01:00'))
        self.assertEqual(str(data_time_slot_series[76].start.dt), str('2019-10-27 03:00:00+01:00'))
        self.assertEqual(data_time_slot_series[76].data['temperature'],230.5)


        # Use the time series 7 from 2019,10,24,0,0,0 to 2019,10,31,0,0,0 (Europe/Rome), DST off -> 2 AM repeated

        # Work in progress in the following
        if False:
            print('--------------------------------------------')
            for item in self.data_time_point_series_7: print(item)
            print('--------------------------------------------')
            for i, item in enumerate(data_time_slot_series): print(i, item)
            print('--------------------------------------------')

        # This is a "downsampling", and there are data losses and strange values that should not be there.
        # TODO: fix me!
        #slotter = Slotter('10m')
        #data_time_slot_series = slotter.process(self.data_time_point_series_7, from_t=1571868600, to_t=1571873400)
        #for item in data_time_slot_series:
        #    print(item)
  
        # This is a 1-day slotting over a DST change, and slots have the wrong start time (but are  correct in legth).
        # TODO: fix me!
        #slotter = Slotter('1D')
        #data_time_slot_series = slotter.process(self.data_time_point_series_7)
        #for item in data_time_slot_series:
        #    print(item)
        
        # TODO: also test from/to with timezone and epoch.
        # i.e. 1571868000 will not be recognised as correct Europe/Rome midnight.
 
 
    def test_slot_operations(self):

        data_time_point_series = DataTimePointSeries()
        start_t = 1436022000 - 120
        for i in range(35):
            data_time_point = DataTimePoint(t = start_t + (i*60),
                                            data = {'temperature': 154+i, 'humidity': 5})
            data_time_point_series.append(data_time_point)
        
        from ..operations import min, max, avg, sum
        # Time series from 16:58:00 to 17:32:00 (Europe/Rome), no from/to, extremes excluded (default)
        
        # Add extra operations
        data_time_slot_series = Slotter('600s', extra_operations=[min,max]).process(data_time_point_series)        
        self.assertEqual(data_time_slot_series[0].data, {'temperature': 161.0, 'humidity': 5.0, 'temperature_min': 156, 'humidity_min': 5, 'temperature_max': 166, 'humidity_max': 5})

        # Change default operations
        data_time_slot_series = Slotter('600s', default_operation=sum).process(data_time_point_series)        
        self.assertEqual(data_time_slot_series[0].data, {'temperature': 1771, 'humidity': 55})
        self.assertEqual(data_time_slot_series[1].data, {'temperature': 1881, 'humidity': 55})

        # Only extra operations
        data_time_slot_series = Slotter('600s', default_operation=None, extra_operations=[avg,min,max]).process(data_time_point_series)        
        self.assertEqual(data_time_slot_series[0].data, {'temperature_avg': 161.0, 'humidity_avg': 5.0, 'temperature_min': 156, 'humidity_min': 5, 'temperature_max': 166, 'humidity_max': 5})

