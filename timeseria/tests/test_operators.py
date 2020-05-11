import unittest
import os
from ..datastructures import Point, TimePoint, DataPoint, DataTimePoint, DataTimeSlotSerie, DataTimeSlot
from ..datastructures import Serie, DataPointSerie, TimePointSerie, DataTimePointSerie
from ..storages import CSVFileStorage
from ..operators import Slotter
from ..time import TimeSpan, dt, s_from_dt, dt_from_s, dt_from_str
from ..exceptions import InputException

# Setup logging
import logging
logging.basicConfig(level=os.environ.get('LOGLEVEL') if os.environ.get('LOGLEVEL') else 'CRITICAL')

# Set test data path
TEST_DATA_PATH = '/'.join(os.path.realpath(__file__).split('/')[0:-1]) + '/test_data/'


class TestSlotter(unittest.TestCase):


    def setUp(self):       
        
        # All the following time series have point with validity=1m
        
        # Time series from 16:58:00 to 17:32:00 (Europe/Rome)
        self.dataTimePointSerie1 = DataTimePointSerie()
        start_t = 1436022000 - 120
        for i in range(35):
            dataTimePoint = DataTimePoint(t = start_t + (i*60),
                                          data = {'temperature': 154+i})
            self.dataTimePointSerie1.append(dataTimePoint)
        
        # Time series from 17:00:00 to 17:30:00 (Europe/Rome)
        self.dataTimePointSerie2 = DataTimePointSerie()
        start_t = 1436022000
        for i in range(34):
            dataTimePoint = DataTimePoint(t    = start_t + (i*60),
                                          data = {'temperature': 154+i})
            self.dataTimePointSerie2.append(dataTimePoint)

        # Time series from 17:00:00 to 17:20:00 (Europe/Rome)
        self.dataTimePointSerie3 = DataTimePointSerie()
        start_t = 1436022000 - 120
        for i in range(23):
            dataTimePoint = DataTimePoint(t    = start_t + (i*60),
                                          data = {'temperature': 154+i})
            self.dataTimePointSerie3.append(dataTimePoint) 
        
        # Time series from 17:10:00 to 17:30:00 (Europe/Rome)
        self.dataTimePointSerie4 = DataTimePointSerie()
        start_t = 1436022000 + 600
        for i in range(21):
            dataTimePoint = DataTimePoint(t    = start_t + (i*60),
                                          data = {'temperature': 154+i})
            self.dataTimePointSerie4.append(dataTimePoint)

        # Time series from 16:58:00 to 17:32:00 (Europe/Rome)
        self.dataTimePointSerie5 = DataTimePointSerie()
        start_t = 1436022000 - 120
        for i in range(35):
            if i > 10 and i <21:
                continue
            dataTimePoint = DataTimePoint(t    = start_t + (i*60),
                                          data = {'temperature': 154+i})
            self.dataTimePointSerie5.append(dataTimePoint)

        # The following time series has point with validity=15m

        # Time series from 2019,10,1,1,0,0 to 2019,10,1,6,0,0 (Europe/Rome)
        from_dt  = dt(2019,10,1,1,0,0, tzinfo='Europe/Rome')
        to_dt    = dt(2019,10,1,6,0,0, tzinfo='Europe/Rome')
        timeSpan = TimeSpan('15m') 
        self.dataTimePointSerie6 = DataTimePointSerie()
        slider_dt = from_dt
        count = 0
        while slider_dt < to_dt:
            if count not in [1, 6, 7, 8, 9, 10]:
                dataTimePoint = DataTimePoint(t    = s_from_dt(slider_dt),
                                              data = {'temperature': 154+count},
                                              tz   = 'Europe/Rome')
                self.dataTimePointSerie6.append(dataTimePoint)
            slider_dt = slider_dt + timeSpan
            count += 1

        # The following time series has point with validity=1h

        # Time series from 2019,10,24,0,0,0 to 2019,10,31,0,0,0 (Europe/Rome), DST off -> 2 AM repeated
        from_dt  = dt(2019,10,24,0,0,0, tzinfo='Europe/Rome')
        to_dt    = dt(2019,10,31,0,0,0, tzinfo='Europe/Rome')
        timeSpan = TimeSpan('1h') 
        self.dataTimePointSerie7 = DataTimePointSerie()
        slider_dt = from_dt
        count = 0
        while slider_dt < to_dt:
            if count not in []:
                dataTimePoint = DataTimePoint(t    = s_from_dt(slider_dt),
                                              data = {'temperature': 154+count},
                                              tz   = 'Europe/Rome')
                self.dataTimePointSerie7.append(dataTimePoint)
            slider_dt = slider_dt + timeSpan
            count += 1




    def test_init(self):
        
        Slotter('1h')
        Slotter(TimeSpan('1h'))
        
        with self.assertRaises(InputException):
            Slotter('NO')
        
        

    def test_detect_dataPoints_validity(self):

        dataTimePointSerie = CSVFileStorage(TEST_DATA_PATH + '/csv/humitemp_short.csv').get()
        self.assertEqual(Slotter._detect_dataPoints_validity(dataTimePointSerie), 61)
        
        dataTimePointSerie = CSVFileStorage(TEST_DATA_PATH + '/csv/shampoo_sales.csv', time_column = 'Month', time_format = '%y-%m').get()
        self.assertEqual(Slotter._detect_dataPoints_validity(dataTimePointSerie), 2678400)
        # TODO: 2678400/60/60/24 = 31 --> detect month? same for day, week, year?

        dataTimePointSerie = CSVFileStorage(TEST_DATA_PATH + '/csv/temperature.csv').get()
        self.assertEqual(Slotter._detect_dataPoints_validity(dataTimePointSerie), 600)

        dataTimePointSerie = CSVFileStorage(TEST_DATA_PATH + '/csv/temp_short_1h.csv').get()
        self.assertEqual(Slotter._detect_dataPoints_validity(dataTimePointSerie), 3600)



    def test_process(self):
        
        # Time series from 16:58:00 to 17:32:00 (Europe/Rome), no from/to, extremes excluded (default)
        slotter = Slotter('10m')
        dataTimeSlotSerie = slotter.process(self.dataTimePointSerie1)
        self.assertEqual(len(dataTimeSlotSerie), 3)
        self.assertEqual(dataTimeSlotSerie[0].start.dt, dt_from_str('2015-07-04 17:00:00+02:00'))
        self.assertEqual(dataTimeSlotSerie[1].start.dt, dt_from_str('2015-07-04 17:10:00+02:00'))
        self.assertEqual(dataTimeSlotSerie[2].start.dt, dt_from_str('2015-07-04 17:20:00+02:00'))
 
        # Time series from 16:58:00 to 17:32:00 (Europe/Rome), no from/to, extremes included.
        slotter = Slotter('10m')
        dataTimeSlotSerie = slotter.process(self.dataTimePointSerie1, include_extremes = True)
        self.assertEqual(len(dataTimeSlotSerie), 4)
        self.assertEqual(dataTimeSlotSerie[0].start.dt, dt_from_str('2015-07-04 16:50:00+02:00'))
        self.assertEqual(dataTimeSlotSerie[1].start.dt, dt_from_str('2015-07-04 17:00:00+02:00'))
        self.assertEqual(dataTimeSlotSerie[2].start.dt, dt_from_str('2015-07-04 17:10:00+02:00'))
        self.assertEqual(dataTimeSlotSerie[3].start.dt, dt_from_str('2015-07-04 17:20:00+02:00'))
        # TODO: the following missing is a bug..
        #self.assertEqual(dataTimeSlotSerie[4].start.dt, dt_from_str('2015-07-04 17:30:00+02:00'))
 
        # Time series from 16:58:00 to 17:32:00 (Europe/Rome), from 15:50 to 17:40
        slotter = Slotter('10m')
        dataTimeSlotSerie = slotter.process(self.dataTimePointSerie1,
                                            from_t = s_from_dt(dt_from_str('2015-07-04 16:50:00+02:00')),
                                            to_t   = s_from_dt(dt_from_str('2015-07-04 17:40:00+02:00')))
 
        self.assertEqual(len(dataTimeSlotSerie), 4)
        self.assertEqual(dataTimeSlotSerie[0].start.dt, dt_from_str('2015-07-04 16:50:00+02:00'))
        self.assertEqual(dataTimeSlotSerie[1].start.dt, dt_from_str('2015-07-04 17:00:00+02:00'))
        self.assertEqual(dataTimeSlotSerie[2].start.dt, dt_from_str('2015-07-04 17:10:00+02:00'))
        self.assertEqual(dataTimeSlotSerie[3].start.dt, dt_from_str('2015-07-04 17:20:00+02:00'))
        # TODO: the following missing is a bug..
        #self.assertEqual(dataTimeSlotSerie[4].start.dt, dt_from_str('2015-07-04 17:30:00+02:00'))
  
        # Time series from 2019,10,1,1,0,0 to 2019,10,1,6,0,0 (Europe/Rome), validity=15m
        slotter = Slotter('1h')
        dataTimePointSerie = self.dataTimePointSerie6
        dataTimeSlotSerie = slotter.process(dataTimePointSerie)
        self.assertEqual(len(dataTimeSlotSerie), 4)
        self.assertEqual(str(dataTimeSlotSerie[0].start.dt), str('2019-10-01 01:00:00+02:00'))
        self.assertEqual(str(dataTimeSlotSerie[1].start.dt), str('2019-10-01 02:00:00+02:00'))
        self.assertEqual(str(dataTimeSlotSerie[2].start.dt), str('2019-10-01 03:00:00+02:00'))
        self.assertEqual(str(dataTimeSlotSerie[3].start.dt), str('2019-10-01 04:00:00+02:00'))

        # Test changing timezone of the serie
        slotter = Slotter('1h')
        dataTimePointSerie = self.dataTimePointSerie6
        dataTimePointSerie.tz = 'America/New_York'
        dataTimeSlotSerie = slotter.process(dataTimePointSerie)
        self.assertEqual(len(dataTimeSlotSerie), 4)
        self.assertEqual(str(dataTimeSlotSerie[0].start.dt), str('2019-09-30 19:00:00-04:00'))
        self.assertEqual(str(dataTimeSlotSerie[1].start.dt), str('2019-09-30 20:00:00-04:00'))
        self.assertEqual(str(dataTimeSlotSerie[2].start.dt), str('2019-09-30 21:00:00-04:00'))
        self.assertEqual(str(dataTimeSlotSerie[3].start.dt), str('2019-09-30 22:00:00-04:00'))
        
        # Time series from 2019,10,24,0,0,0 to 2019,10,31,0,0,0 (Europe/Rome), DST off -> 2 AM repeated
        slotter = Slotter('1h')
        dataTimePointSerie = self.dataTimePointSerie7
        dataTimeSlotSerie = slotter.process(dataTimePointSerie)

        self.assertEqual(len(dataTimeSlotSerie), 168)
        self.assertEqual(str(dataTimeSlotSerie[73].start.dt), str('2019-10-27 01:00:00+02:00'))
        self.assertEqual(str(dataTimeSlotSerie[74].start.dt), str('2019-10-27 02:00:00+02:00'))
        self.assertEqual(str(dataTimeSlotSerie[75].start.dt), str('2019-10-27 02:00:00+01:00'))
        self.assertEqual(str(dataTimeSlotSerie[76].start.dt), str('2019-10-27 03:00:00+01:00'))


        # Work in progress in the following
        #print('--------------------------------------------')
        #for item in self.dataTimePointSerie6: print(item)
        #print('--------------------------------------------')
        #for i, item in enumerate(dataTimeSlotSerie): print(i, item)
        #print('--------------------------------------------')
       
        # Time series from 2019,10,24,0,0,0 to 2019,10,31,0,0,0 (Europe/Rome), DST off -> 2 AM repeated
        #dataTimeSlotSerie = slotter.process(self.dataTimePointSerie7, from_t=1571868600, to_t=1571873400)
        #for item in dataTimeSlotSerie:
        #    print(item)
        # This is a downsampling, there are coverage=None.. TOOD: fix me
 
        # 1-day sloter TODO: add all the "logical" time part, nothing works here...
        #slotter = Slotter('1D')
        # Time series from 2019,10,24,0,0,0 to 2019,10,31,0,0,0 (Europe/Rome), DST off -> 2 AM repeated
        #dataTimeSlotSerie = slotter.process(self.dataTimePointSerie7)
        #for item in dataTimeSlotSerie:
        #    print(item)


       
#     def test_e2e(self):
#         slotter = Slotter('1h')
#         storage2 = CSVFileStorage(TEST_DATA_PATH + '/csv/humitemp_long.csv', skip_errors=True)
#         dataTimePointSerie2 = storage2.get()
#         dataTimeSlotSerie = slotter.process(dataTimePointSerie2)
# 
#         for item in dataTimeSlotSerie:
#             print('{} - data={}'.format(item, item.data, item.coverage))



class TestDerivative(unittest.TestCase):


    def test_apply(self):

        # Test data        
        dataTimeSlotSerie = DataTimeSlotSerie()
        dataTimeSlotSerie.append(DataTimeSlot(start=TimePoint(0), end=TimePoint(60), data={'value':10}))
        dataTimeSlotSerie.append(DataTimeSlot(start=TimePoint(60), end=TimePoint(120), data={'value':12}))
        dataTimeSlotSerie.append(DataTimeSlot(start=TimePoint(120), end=TimePoint(180), data={'value':15}))
        dataTimeSlotSerie.append(DataTimeSlot(start=TimePoint(180), end=TimePoint(240), data={'value':16}))
 
        from timeseria.operators import Derivative
        
        # Test standard
        derivative_dataTimeSlotSerie = Derivative.apply(dataTimeSlotSerie)
        self.assertEqual(len(derivative_dataTimeSlotSerie), 4)
        self.assertEqual(derivative_dataTimeSlotSerie[0].data['value_der'],2.0)
        self.assertEqual(derivative_dataTimeSlotSerie[1].data['value_der'],2.5)
        self.assertEqual(derivative_dataTimeSlotSerie[2].data['value_der'],2.0)
        self.assertEqual(derivative_dataTimeSlotSerie[3].data['value_der'],1.0)

        # Test in-place and incremental behavior
        Derivative.apply(dataTimeSlotSerie, inplace=True, incremental=True)
        self.assertEqual(len(dataTimeSlotSerie), 4)
        self.assertEqual(dataTimeSlotSerie[0].data['value'],10)
        self.assertEqual(dataTimeSlotSerie[0].data['value_der'],1.0)
        self.assertEqual(dataTimeSlotSerie[1].data['value'],12)
        self.assertEqual(dataTimeSlotSerie[1].data['value_der'],2.5)
        self.assertEqual(dataTimeSlotSerie[2].data['value'],15)
        self.assertEqual(dataTimeSlotSerie[2].data['value_der'],2.0)
        self.assertEqual(dataTimeSlotSerie[3].data['value'],16)
        self.assertEqual(dataTimeSlotSerie[3].data['value_der'],0.5)































