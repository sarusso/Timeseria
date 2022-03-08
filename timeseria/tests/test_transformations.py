import unittest
import os
from ..datastructures import DataTimePoint, DataTimePointSeries
from ..transformations import Resampler, Slotter 
from ..time import dt, s_from_dt, dt_from_str
from ..units import TimeUnit

# Setup logging
from .. import logger
logger.setup()

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


    def test_slot(self):
          
        # Time series from 16:58:00 to 17:32:00 (Europe/Rome), no from/to, extremes excluded (default)
        data_time_slot_series = Slotter('600s').process(self.data_time_point_series_1)        
        self.assertEqual(len(data_time_slot_series), 3)
        self.assertEqual(data_time_slot_series[0].start.dt, dt_from_str('2015-07-04 17:00:00+02:00'))
        self.assertEqual(data_time_slot_series[1].start.dt, dt_from_str('2015-07-04 17:10:00+02:00'))
        self.assertEqual(data_time_slot_series[2].start.dt, dt_from_str('2015-07-04 17:20:00+02:00'))
         
        # Time series from 16:58:00 to 17:32:00 (Europe/Rome), no from/to, extremes included.
        data_time_slot_series = Slotter('10m').process(self.data_time_point_series_1,  include_extremes = True)
        self.assertEqual(len(data_time_slot_series), 5)
        self.assertEqual(data_time_slot_series[0].start.dt, dt_from_str('2015-07-04 16:50:00+02:00'))
        self.assertEqual(data_time_slot_series[1].start.dt, dt_from_str('2015-07-04 17:00:00+02:00'))
        self.assertEqual(data_time_slot_series[2].start.dt, dt_from_str('2015-07-04 17:10:00+02:00'))
        self.assertEqual(data_time_slot_series[3].start.dt, dt_from_str('2015-07-04 17:20:00+02:00'))
        # The following is because the include_extremes is set to True
        self.assertEqual(data_time_slot_series[4].start.dt, dt_from_str('2015-07-04 17:30:00+02:00'))
        self.assertEqual(data_time_slot_series[4].data_loss, 0.75)
 
        # Time series from 16:58:00 to 17:32:00 (Europe/Rome), from 15:50 to 17:40
        data_time_slot_series = Slotter('10m').process(self.data_time_point_series_1,
                                    from_t = s_from_dt(dt_from_str('2015-07-04 16:50:00+02:00')),
                                    to_t   = s_from_dt(dt_from_str('2015-07-04 17:40:00+02:00')))
        self.assertEqual(len(data_time_slot_series), 5)
        self.assertEqual(data_time_slot_series[0].start.dt, dt_from_str('2015-07-04 16:50:00+02:00'))
        self.assertEqual(data_time_slot_series[1].start.dt, dt_from_str('2015-07-04 17:00:00+02:00'))
        self.assertEqual(data_time_slot_series[2].start.dt, dt_from_str('2015-07-04 17:10:00+02:00'))
        self.assertEqual(data_time_slot_series[3].start.dt, dt_from_str('2015-07-04 17:20:00+02:00'))
        self.assertEqual(data_time_slot_series[4].start.dt, dt_from_str('2015-07-04 17:30:00+02:00'))

        # Time series from 16:58:00 to 17:32:00 (Europe/Rome), from 15:00 to 18:00
        data_time_slot_series = Slotter('10m').process(self.data_time_point_series_1,
                                    from_t = s_from_dt(dt_from_str('2015-07-04 15:00:00+02:00')),
                                    to_t   = s_from_dt(dt_from_str('2015-07-04 18:00:00+02:00')))
        # Here from and to are capped with the time series data points
        self.assertEqual(str(data_time_slot_series[0].start.dt), str('2015-07-04 14:50:00+00:00'))
        self.assertEqual(str(data_time_slot_series[-1].start.dt), str('2015-07-04 15:30:00+00:00'))


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
        self.assertEqual(data_time_slot_series[76].data['temperature_avg'],230.5)

        # Time series from 2019,10,24,0,0,0 to 2019,10,31,0,0,0 (Europe/Rome), DST off -> 2 AM repeated
        data_time_slot_series = Slotter('1D').process(self.data_time_point_series_7)
        self.assertEqual(len(data_time_slot_series), 6)
        self.assertEqual(str(data_time_slot_series[0].start.dt), str('2019-10-24 00:00:00+02:00'))
        self.assertEqual(str(data_time_slot_series[-1].start.dt), str('2019-10-29 00:00:00+01:00')) # Last one not included as right excluded


        # Downsampling (downslotting)
         
        # Use the time series 7 from 2019,10,24,0,0,0 to 2019,10,31,0,0,0 (Europe/Rome), DST off -> 2 AM repeated
        # Work in progress in the following
        #if False:
        #    print('--------------------------------------------')
        #    for item in self.data_time_point_series_7: print(item)
        #    print('--------------------------------------------')
        #    for i, item in enumerate(data_time_slot_series): print(i, item)
        #    print('--------------------------------------------')
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
        
        # Import (overloaded) operations
        from ..operations import min, max, avg, sum

        # Time series from 16:58:00 to 17:32:00 (Europe/Rome)
        data_time_point_series = DataTimePointSeries()
        start_t = 1436022000 - 120
        for i in range(35):
            data_time_point = DataTimePoint(t = start_t + (i*60),
                                            data = {'temperature': 154+i, 'humidity': 5})
            data_time_point_series.append(data_time_point)

        # Add extra operations
        data_time_slot_series = Slotter('600s', extra_operations=[min,max]).process(data_time_point_series)      
        self.assertEqual(data_time_slot_series[0].data, {'temperature_avg': 161.0, 'humidity_avg': 5.0, 'temperature_min': 156, 'humidity_min': 5, 'temperature_max': 166, 'humidity_max': 5})

        # Change default operations
        data_time_slot_series = Slotter('600s', default_operation=sum).process(data_time_point_series)        
        self.assertEqual(data_time_slot_series[0].data, {'temperature_sum': 1771, 'humidity_sum': 55})
        self.assertEqual(data_time_slot_series[1].data, {'temperature_sum': 1881, 'humidity_sum': 55})

        # Only extra operations
        data_time_slot_series = Slotter('600s', default_operation=None, extra_operations=[avg,min,max]).process(data_time_point_series)        
        self.assertEqual(data_time_slot_series[0].data, {'temperature_avg': 161.0, 'humidity_avg': 5.0, 'temperature_min': 156, 'humidity_min': 5, 'temperature_max': 166, 'humidity_max': 5})

        # Time series from 16:58:00 to 17:32:00 (Europe/Rome) with a short (single-slot) gap to be interpolated
        data_time_point_series = DataTimePointSeries()
        start_t = 1436022000 - 120
        for i in range(35):
            if i < 12 or i > 22:
                data_time_point = DataTimePoint(t = start_t + (i*60),
                                                data = {'temperature': 154+i, 'humidity': 5})
                data_time_point_series.append(data_time_point)

        data_time_slot_series = Slotter('600s', extra_operations=[min,max]).process(data_time_point_series)        
        self.assertEqual(data_time_slot_series[1].data['temperature_min'], 166.5)
        
        # Time series from 16:58:00 to 17:32:00 (Europe/Rome) with a long (multi-slot) gap to be interpolated
        data_time_point_series = DataTimePointSeries()
        start_t = 1436022000 - 120
        for i in range(60):
            if i < 10 or i >48:
                data_time_point = DataTimePoint(t = start_t + (i*60),
                                                data = {'temperature': 154+i, 'humidity': 5})
                data_time_point_series.append(data_time_point)

        data_time_slot_series = Slotter('600s', extra_operations=[min,max]).process(data_time_point_series)        
        self.assertEqual(data_time_slot_series[1].data['temperature_min'], 167.75)
        self.assertEqual(data_time_slot_series[2].data['temperature_min'], 179.5)
        self.assertEqual(data_time_slot_series[3].data['temperature_min'], 191.25)

        # Re-slotting with the same time unit as the original points
        data_time_slot_series = Slotter('60s').process(data_time_point_series)

        #  Upsampling (upslotting) not supported yet
        data_time_slot_series = Slotter('60s').process(data_time_point_series)
        with self.assertRaises(ValueError):    
            data_time_slot_series = Slotter('30s').process(data_time_point_series)        
        
        # TODO: directly load a day-resolution time series in this test
        data_time_slot_series = Resampler(86400).process(self.data_time_point_series_7)
        with self.assertRaises(ValueError):    
            Slotter('1h').process(data_time_slot_series)

    def test_slot_indexes(self):
        
        # Import (overloaded) operations
        from ..operations import min, max, avg, sum

        # Time series from 16:58:00 to 17:32:00 (Europe/Rome)
        data_time_point_series = DataTimePointSeries()
        start_t = 1436022000 - 120
        for i in range(35):
            if i > 10 and i < 15:
                anomaly = 0.25
                forecast = 0
                data_reconstructed = 1
                data_loss = 1
            elif i > 20:
                anomaly = None 
                forecast = 1
                data_reconstructed = 0
                data_loss = None
            else:
                anomaly = 0 
                forecast = 0
                data_reconstructed = 0
                data_loss = 0.1
                
            data_time_point = DataTimePoint(t = start_t + (i*60),
                                            data = {'temperature': 154+i, 'humidity': 5},
                                            data_loss = data_loss
                                            )
            # Add extra indexes as if something operated on the time series
            data_time_point.anomaly = anomaly
            data_time_point.forecast = forecast
            data_time_point._data_reconstructed = data_reconstructed
            
            # Append
            data_time_point_series.append(data_time_point)
                
        # This is an indirect test of the series indexes. TODO: move it away.
        self.assertEqual(data_time_point_series.indexes, ['data_reconstructed', 'data_loss', 'anomaly', 'forecast'])
        
        # Slot the time series
        slotted_data_time_point_series = Slotter('600s').process(data_time_point_series)

        # Check that we have all the indexes
        self.assertEqual(slotted_data_time_point_series.indexes, ['data_reconstructed', 'data_loss', 'anomaly', 'forecast'])

        # Check indexes math
        self.assertAlmostEqual(slotted_data_time_point_series[0].anomaly, 0.045454545454545456)
        self.assertAlmostEqual(slotted_data_time_point_series[0].data_reconstructed, 0.18181818181818182)
        self.assertAlmostEqual(slotted_data_time_point_series[0].forecast, 0)
        
        self.assertAlmostEqual(slotted_data_time_point_series[1].anomaly, 0.08333333333333333)
        self.assertAlmostEqual(slotted_data_time_point_series[1].data_reconstructed, 0.2727272727272727)
        self.assertAlmostEqual(slotted_data_time_point_series[1].forecast, 0.18181818181818182)

        self.assertEqual(slotted_data_time_point_series[2].anomaly, None)
        self.assertAlmostEqual(slotted_data_time_point_series[2].data_reconstructed, 0)
        self.assertEqual(slotted_data_time_point_series[2].forecast, 1.0)

        # The following is re-computed from missing coverage only, and not marked as None.
        # See the compute_data_loss function fore more details and some more comments.
        self.assertEqual(slotted_data_time_point_series[2].data_loss, 0) 


class TestResampler(unittest.TestCase):


    def test_resample_minimal(self):
        
        # Test series
        series = DataTimePointSeries()
        series.append(DataTimePoint(t=-4,  data={'value':4}))
        series.append(DataTimePoint(t=-3,  data={'value':3}))
        series.append(DataTimePoint(t=-2,  data={'value':2}))
        series.append(DataTimePoint(t=-1,  data={'value':1}))
        series.append(DataTimePoint(t=0,  data={'value':0}))
        series.append(DataTimePoint(t=1,  data={'value':1}))
        series.append(DataTimePoint(t=2,  data={'value':2}))
        series.append(DataTimePoint(t=3,  data={'value':3}))
        series.append(DataTimePoint(t=4,  data={'value':4}))

        # Since by default the resample does not include extremes, we onlu expect the point
        # at t=0, covering from -2 to +2. Other points would be at t=-4 and t=+4
        resampled_series = series.resample(4)
        self.assertEqual(len(resampled_series),1)
        self.assertEqual(resampled_series[0].t,0)
        self.assertEqual(resampled_series[0].data['value'],1)
        self.assertEqual(resampled_series[0].data_loss,0)   
     
        # Now include extremes as well
        resampled_series = series.resample(4, include_extremes=True)
        self.assertEqual(len(resampled_series),3)
        self.assertEqual(resampled_series[0].t,-4)
        self.assertEqual(resampled_series[0].data['value'],3.2)   
        self.assertEqual(resampled_series[0].data_loss,0.375)   
        self.assertEqual(resampled_series[1].t,0)
        self.assertEqual(resampled_series[1].data['value'],1)   
        self.assertEqual(resampled_series[1].data_loss,0)
        self.assertEqual(resampled_series[2].t,4)
        self.assertEqual(resampled_series[2].data['value'],3.2)   
        self.assertEqual(resampled_series[2].data_loss,0.375)
        
        # TODO: test resample for unit=3 here as well?


    def test_resample_basic(self):
        
        # Test series
        series = DataTimePointSeries()
        series.append(DataTimePoint(t=-3,  data={'v':3}))
        series.append(DataTimePoint(t=-2,  data={'v':2}))
        series.append(DataTimePoint(t=-1,  data={'v':1}))
        series.append(DataTimePoint(t=0,  data={'v':0}))
        series.append(DataTimePoint(t=1,  data={'v':1}))
        series.append(DataTimePoint(t=2,  data={'v':2}))
        series.append(DataTimePoint(t=3,  data={'v':3}))
        series.append(DataTimePoint(t=4,  data={'v':4}))
        series.append(DataTimePoint(t=5,  data={'v':5}))
        series.append(DataTimePoint(t=6,  data={'v':6}))
        series.append(DataTimePoint(t=7,  data={'v':7}))
        series.append(DataTimePoint(t=16, data={'v':16}))
        series.append(DataTimePoint(t=17, data={'v':17}))
        series.append(DataTimePoint(t=18, data={'v':18}))
        
        resampled_series = series.resample(4)
        for item in resampled_series:
            print(item)

        print('=====================================')

        #resampled_series = series.resample(3)
        #for item in resampled_series:
        #    print(item)


    def test_resample_edge_cases(self):
        #self.assertEqual(1, 2)
        
        #data_time_point_series = DataTimePointSeries({1:2, 2:4})
        
        data_time_point_series = DataTimePointSeries()
        #data_time_point_series.append(DataTimePoint(t = (60*35)+18, data = {'value': 4571.27}))
        #data_time_point_series.append(DataTimePoint(t = (60*40)+18, data = {'value': 4571.3}))
        #data_time_point_series.append(DataTimePoint(t = (60*45)+18, data = {'value': 4571.33}))
        #data_time_point_series.append(DataTimePoint(t = (60*55)+18, data = {'value': 4571.4}))
        #data_time_point_series.append(DataTimePoint(t = (60*60)+18, data = {'value': 4571.43}))


        data_time_point_series.append(DataTimePoint(t = (60*20), data = {'value': 4571.55}))
        data_time_point_series.append(DataTimePoint(t = (60*25), data = {'value': 4571.58}))
        data_time_point_series.append(DataTimePoint(t = (60*30), data = {'value': 4571.61}))
        data_time_point_series.append(DataTimePoint(t = (60*45), data = {'value': 4571.71}))
        data_time_point_series.append(DataTimePoint(t = (60*50), data = {'value': 4571.74}))
        
        #DataTimePoint @ 1200 (1970-01-01 00:20:00+00:00) with data "{'value': 4571.55}"
        #DataTimePoint @ 1500 (1970-01-01 00:25:00+00:00) with data "{'value': 4571.58}"
        #DataTimePoint @ 1800 (1970-01-01 00:30:00+00:00) with data "{'value': 4571.61}"
        #DataTimePoint @ 2700 (1970-01-01 00:45:00+00:00) with data "{'value': 4571.71}"
        #DataTimePoint @ 3000 (1970-01-01 00:50:00+00:00) with data "{'value': 4571.74}"

        # DataTimePoint @ 1500.0 (1970-01-01 00:25:00+00:00) with data "{'value': 4571.5650000000005}"
        # DataTimePoint @ 1800.0 (1970-01-01 00:30:00+00:00) with data "{'value': 4571.594999999999}"
        # DataTimePoint @ 2100.0 (1970-01-01 00:35:00+00:00) with data "{'value': 4571.625}"
        # DataTimePoint @ 2400.0 (1970-01-01 00:40:00+00:00) with data "{'value': 4571.66}"
        # DataTimePoint @ 2700.0 (1970-01-01 00:45:00+00:00) with data "{'value': 4571.695}"
        # DataTimePoint @ 3000.0 (1970-01-01 00:50:00+00:00) with data "{'value': 4571.725}"
        # vs 
        # DataTimePoint @ 1500.0 (1970-01-01 00:25:00+00:00) with data "{'value': 4571.5650000000005}" and data_loss="0.0"
        # DataTimePoint @ 1800.0 (1970-01-01 00:30:00+00:00) with data "{'value': 4571.594999999999}" and data_loss="0.0"
        # DataTimePoint @ 2100.0 (1970-01-01 00:35:00+00:00) with data "{'value': 4571.61}" and data_loss="0.94"
        # DataTimePoint @ 2400.0 (1970-01-01 00:40:00+00:00) with data "{'value': 4571.68}" and data_loss="0.06000000000000005"
        # DataTimePoint @ 2700.0 (1970-01-01 00:45:00+00:00) with data "{'value': 4571.695}" and data_loss="0.0"
        # DataTimePoint @ 3000.0 (1970-01-01 00:50:00+00:00) with data "{'value': 4571.725}" and data_loss="0.0"
        for item in data_time_point_series:
            print(item)

        resampled_data_time_point_series = Resampler('300s').process(data_time_point_series) 
        
        print(data_time_point_series)
        print(resampled_data_time_point_series)
        for item in resampled_data_time_point_series:
            print(item)
            
            
        print('---------------------------------')


        #self.assertEqual(1, 2)
        
        #data_time_point_series = DataTimePointSeries({1:2, 2:4})
        
        data_time_point_series = DataTimePointSeries()
        #data_time_point_series.append(DataTimePoint(t = (60*35)+18, data = {'value': 4571.27}))
        #data_time_point_series.append(DataTimePoint(t = (60*40)+18, data = {'value': 4571.3}))
        #data_time_point_series.append(DataTimePoint(t = (60*45)+18, data = {'value': 4571.33}))
        #data_time_point_series.append(DataTimePoint(t = (60*55)+18, data = {'value': 4571.4}))
        #data_time_point_series.append(DataTimePoint(t = (60*60)+18, data = {'value': 4571.43}))


        data_time_point_series.append(DataTimePoint(t = (60*20)+18, data = {'value': 4571.55}))
        data_time_point_series.append(DataTimePoint(t = (60*25)+18, data = {'value': 4571.58}))
        data_time_point_series.append(DataTimePoint(t = (60*30)+18, data = {'value': 4571.61}))
        #data_time_point_series.append(DataTimePoint(t = (60*35)+18, data = {'value': 4571.64}))
        data_time_point_series.append(DataTimePoint(t = (60*40)+18, data = {'value': 4571.68}))
        data_time_point_series.append(DataTimePoint(t = (60*45)+18, data = {'value': 4571.71}))
        data_time_point_series.append(DataTimePoint(t = (60*50)+18, data = {'value': 4571.74}))

        # DataTimePoint @ 1500.0 (1970-01-01 00:25:00+00:00) with data "{'value': 4571.5650000000005}"
        # DataTimePoint @ 1800.0 (1970-01-01 00:30:00+00:00) with data "{'value': 4571.594999999999}"
        # DataTimePoint @ 2100.0 (1970-01-01 00:35:00+00:00) with data "{'value': 4571.625}"
        # DataTimePoint @ 2400.0 (1970-01-01 00:40:00+00:00) with data "{'value': 4571.66}"
        # DataTimePoint @ 2700.0 (1970-01-01 00:45:00+00:00) with data "{'value': 4571.695}"
        # DataTimePoint @ 3000.0 (1970-01-01 00:50:00+00:00) with data "{'value': 4571.725}"
        # vs 
        # DataTimePoint @ 1500.0 (1970-01-01 00:25:00+00:00) with data "{'value': 4571.5650000000005}" and data_loss="0.0"
        # DataTimePoint @ 1800.0 (1970-01-01 00:30:00+00:00) with data "{'value': 4571.594999999999}" and data_loss="0.0"
        # DataTimePoint @ 2100.0 (1970-01-01 00:35:00+00:00) with data "{'value': 4571.61}" and data_loss="0.94"
        # DataTimePoint @ 2400.0 (1970-01-01 00:40:00+00:00) with data "{'value': 4571.68}" and data_loss="0.06000000000000005"
        # DataTimePoint @ 2700.0 (1970-01-01 00:45:00+00:00) with data "{'value': 4571.695}" and data_loss="0.0"
        # DataTimePoint @ 3000.0 (1970-01-01 00:50:00+00:00) with data "{'value': 4571.725}" and data_loss="0.0"


        resampled_data_time_point_series = Resampler('300s').process(data_time_point_series) 
        
        print(data_time_point_series)
        print(resampled_data_time_point_series)
        for item in resampled_data_time_point_series:
            print(item)












#         start_t = 1436022000 - 120
#         for i in range(35):
#             data_time_point = DataTimePoint(t = start_t + (i*60),
#                                             tz='Europe/Rome',
#                                             data = {'temperature': 154+i})
#             data_time_point_series.append(data_time_point)
#  
#         data_time_slot_series = Resampler('600s').process(data_time_point_series)        
#          
#   

    def test_resampler_indexes(self):
        
        # Import (overloaded) operations
        from ..operations import min, max, avg, sum

        # Time series from 16:58:00 to 17:32:00 (Europe/Rome)
        data_time_point_series = DataTimePointSeries()
        start_t = 1436022000 - 120
        for i in range(35):
            if i > 10 and i < 15:
                anomaly = 0.25
                forecast = 0
                data_reconstructed = 1
                data_loss = 1
            elif i > 20:
                anomaly = None 
                forecast = 1
                data_reconstructed = 0
                data_loss = None
            else:
                anomaly = 0 
                forecast = 0
                data_reconstructed = 0
                data_loss = 0.1
                
            data_time_point = DataTimePoint(t = start_t + (i*60),
                                            data = {'temperature': 154+i, 'humidity': 5},
                                            data_loss = data_loss
                                            )
            # Add extra indexes as if something operated on the time series
            data_time_point.anomaly = anomaly
            data_time_point.forecast = forecast
            data_time_point._data_reconstructed = data_reconstructed
            
            # Append
            data_time_point_series.append(data_time_point)
                
        # This is an indirect test of the series indexes. TODO: move it away.
        self.assertEqual(data_time_point_series.indexes, ['data_reconstructed', 'data_loss', 'anomaly', 'forecast'])
        
        # Slot the time series
        resampled_data_time_point_series = Resampler('600s').process(data_time_point_series)        

        # Check that we have all the indexes
        self.assertEqual(resampled_data_time_point_series.indexes, ['data_reconstructed', 'data_loss', 'anomaly', 'forecast'])

        # Check indexes math TODO: check the entire math here, there is something strange.
        self.assertAlmostEqual(resampled_data_time_point_series[0].anomaly, 0)
        self.assertAlmostEqual(resampled_data_time_point_series[0].data_reconstructed, 0)
        self.assertAlmostEqual(resampled_data_time_point_series[0].forecast, 0)
        
        self.assertAlmostEqual(resampled_data_time_point_series[1].anomaly, 0.09090909090909091)
        self.assertAlmostEqual(resampled_data_time_point_series[1].data_reconstructed, 0.36363636363636365)
        self.assertAlmostEqual(resampled_data_time_point_series[1].forecast, 0)

        self.assertEqual(resampled_data_time_point_series[2].anomaly, 0)
        self.assertAlmostEqual(resampled_data_time_point_series[2].data_reconstructed, 0)
        self.assertEqual(resampled_data_time_point_series[2].forecast, 0.6363636363636364)

        self.assertAlmostEqual(resampled_data_time_point_series[2].data_loss, 0.04) 

        # Now resample ath the same sampling interval and check indexes are still there
        same_resampled_data_time_point_series = data_time_point_series[0:5].resample(60)
        self.assertEqual(same_resampled_data_time_point_series.indexes, ['data_reconstructed', 'data_loss', 'anomaly', 'forecast'])

