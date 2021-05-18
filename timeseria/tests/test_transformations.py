import unittest
import os
from ..datastructures import DataTimePoint, DataTimePointSeries
from ..transformations import Resampler, Slotter 
from ..time import dt, s_from_dt, dt_from_str
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


    def test_resample(self):
        
        # Time series from 16:58:00 to 17:32:00 (Europe/Rome)
        data_time_point_series = DataTimePointSeries()
        start_t = 1436022000 - 120
        for i in range(35):
            data_time_point = DataTimePoint(t = start_t + (i*60),
                                            tz='Europe/Rome',
                                            data = {'temperature': 154+i})
            data_time_point_series.append(data_time_point)
 
        data_time_slot_series = Resampler('600s').process(data_time_point_series)        
 
        self.assertEqual(str(data_time_slot_series[0].dt), '2015-07-04 17:00:00+02:00')
        self.assertAlmostEqual(data_time_slot_series[0].data_loss, 0.25)
        self.assertEqual(data_time_slot_series[0].data['temperature'], 157.5)
         
        self.assertEqual(str(data_time_slot_series[1].dt), '2015-07-04 17:10:00+02:00')
        self.assertAlmostEqual(data_time_slot_series[1].data_loss, 0.0)
        self.assertEqual(data_time_slot_series[1].data['temperature'], 166.0)
         
        self.assertEqual(str(data_time_slot_series[3].dt), '2015-07-04 17:30:00+02:00')
        self.assertAlmostEqual(data_time_slot_series[3].data_loss, 0.25)
        self.assertEqual(data_time_slot_series[3].data['temperature'], 184.5)
 
     
        # Time series from 15:00:00 to 15:37:00 (UTC)
        data_time_point_series = DataTimePointSeries()
        start_t = 1436022000
        for i in range(38):
            data_time_point = DataTimePoint(t = start_t + (i*60),
                                            data = {'temperature': 154+i})
            data_time_point_series.append(data_time_point)
 
 
        data_time_slot_series = Resampler('600s').process(data_time_point_series)
         
        self.assertEqual(str(data_time_slot_series[0].dt), '2015-07-04 15:00:00+00:00')
        self.assertAlmostEqual(data_time_slot_series[0].data_loss, 0.45)
        self.assertEqual(data_time_slot_series[0].data['temperature'], 156.5)
         
        self.assertEqual(str(data_time_slot_series[1].dt), '2015-07-04 15:10:00+00:00')
        self.assertAlmostEqual(data_time_slot_series[1].data_loss, 0.0)
        self.assertEqual(data_time_slot_series[1].data['temperature'], 164.0)
         
        self.assertEqual(str(data_time_slot_series[4].dt), '2015-07-04 15:40:00+00:00')
        self.assertAlmostEqual(data_time_slot_series[4].data_loss, 0.85)
        self.assertEqual(data_time_slot_series[4].data['temperature'], 189.5)        
         
 
        # Time series from 00:00 to 00:19 (UTC) 
        data_time_point_series = DataTimePointSeries()
        start_t = 0
        for i in range(20):
            if i < 8 or i > 12:
                data_time_point = DataTimePoint(t = start_t + (i*60),
                                                data = {'temperature': 154+i})
                data_time_point_series.append(data_time_point)
 
        # Re-sample as is
        data_time_slot_series = Resampler('1m').process(data_time_point_series)        
        self.assertEqual(len(data_time_slot_series), 20)
         
        # Original points
        self.assertEqual(data_time_slot_series[0].data_loss, 0)
        self.assertEqual(data_time_slot_series[0].data['temperature'], 154)
        self.assertEqual(data_time_slot_series[1].data_loss, 0)
        self.assertEqual(data_time_slot_series[1].data['temperature'], 155)
        self.assertEqual(data_time_slot_series[19].data_loss, 0)
        self.assertEqual(data_time_slot_series[19].data['temperature'], 173)
 
        # Reconstructed (interpolated) points
        self.assertEqual(data_time_slot_series[8].data_loss, 1)
        self.assertEqual(data_time_slot_series[8].data['temperature'], 162)
        self.assertEqual(data_time_slot_series[9].data_loss, 1)
        self.assertEqual(data_time_slot_series[9].data['temperature'], 163)
        self.assertEqual(data_time_slot_series[12].data_loss, 1)
        self.assertEqual(data_time_slot_series[12].data['temperature'], 166)


        # Time series from 00:00 to 00:19 (UTC) with missing data in the middle
        data_time_point_series = DataTimePointSeries()
        start_t = 0
        for i in range(20):
            if i < 8 or i > 12:
                data_time_point = DataTimePoint(t = start_t + (i*60),
                                                data = {'temperature': 154+i})
                data_time_point_series.append(data_time_point)

        #  Upsampling not supported yet
        with self.assertRaises(ValueError):
            data_time_slot_series = Resampler('20s').process(data_time_point_series)        
        
        # When implementing it, remember about check correct linear interpolation when there is data loss vs. when there isn't,
        # and that all the prev-next math has to be correctly taken into account together with computing the right data loss.
         

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

