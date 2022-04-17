import unittest
import os
from ..datastructures import DataTimePoint, DataTimePointSeries
from ..transformations import Resampler, Aggregator 
from ..time import dt, s_from_dt, dt_from_str
from ..units import TimeUnit

# Setup logging
from .. import logger
logger.setup()

# Set test data path
TEST_DATA_PATH = '/'.join(os.path.realpath(__file__).split('/')[0:-1]) + '/test_data/'


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
        
        # Check len
        self.assertEqual(len(resampled_series), 1)
        
        # Check the only resampled point
        self.assertEqual(len(resampled_series),1)
        self.assertEqual(resampled_series[0].t,0)
        self.assertEqual(resampled_series[0].data['value'],1)
        self.assertEqual(resampled_series[0].data_loss,0)   
     
        # Now include extremes as well -> dropped support for this
        #resampled_series = series.resample(4, include_extremes=True)
        #self.assertEqual(len(resampled_series),3)
        #self.assertEqual(resampled_series[0].t,-4)
        #self.assertEqual(resampled_series[0].data['value'],3.2)   
        #self.assertEqual(resampled_series[0].data_loss,0.375)   
        #self.assertEqual(resampled_series[1].t,0)
        #self.assertEqual(resampled_series[1].data['value'],1)   
        #self.assertEqual(resampled_series[1].data_loss,0)
        #self.assertEqual(resampled_series[2].t,4)
        #self.assertEqual(resampled_series[2].data['value'],3.2)   
        #self.assertEqual(resampled_series[2].data_loss,0.375)


    def test_resample_basic(self):
        
        # Test series
        series = DataTimePointSeries()
        series.append(DataTimePoint(t=-3,  data={'value':3}))
        series.append(DataTimePoint(t=-2,  data={'value':2}))
        series.append(DataTimePoint(t=-1,  data={'value':1}))
        series.append(DataTimePoint(t=0,  data={'value':0}))
        series.append(DataTimePoint(t=1,  data={'value':1}))
        series.append(DataTimePoint(t=2,  data={'value':2}))
        series.append(DataTimePoint(t=3,  data={'value':3}))
        series.append(DataTimePoint(t=4,  data={'value':4}))
        series.append(DataTimePoint(t=5,  data={'value':5}))
        series.append(DataTimePoint(t=6,  data={'value':6}))
        series.append(DataTimePoint(t=7,  data={'value':7}))
        series.append(DataTimePoint(t=16, data={'value':16}))
        series.append(DataTimePoint(t=17, data={'value':17}))
        series.append(DataTimePoint(t=18, data={'value':18}))
        
        # Resample for 4 seconds
        resampled_series = series.resample(4)
        # DataTimePoint @ 0.0 (1970-01-01 00:00:00+00:00) with data "{'v': 1.0}" and data_loss="0.0"
        # DataTimePoint @ 4.0 (1970-01-01 00:00:04+00:00) with data "{'v': 4.0}" and data_loss="0.0"
        # DataTimePoint @ 8.0 (1970-01-01 00:00:08+00:00) with data "{'v': 7.96875}" and data_loss="0.625"
        # DataTimePoint @ 12.0 (1970-01-01 00:00:12+00:00) with data "{'v': 12.0}" and data_loss="1.0"
        # DataTimePoint @ 16.0 (1970-01-01 00:00:16+00:00) with data "{'v': 16.03125}" and data_loss="0.375"

        # Check len
        self.assertEqual(len(resampled_series), 5)

        # Check for t = 0
        self.assertEqual(resampled_series[0].t, 0)
        self.assertEqual(resampled_series[0].data['value'], 1)  # Expected: 1
        self.assertEqual(resampled_series[0].data_loss, 0)

        # Check for t = 4
        self.assertEqual(resampled_series[1].t, 4)
        self.assertEqual(resampled_series[1].data['value'], 4)  # Expected: 4
        self.assertEqual(resampled_series[1].data_loss, 0)

        # Check for t = 8
        self.assertEqual(resampled_series[2].t, 8)
        self.assertAlmostEqual(resampled_series[2].data['value'], 7.96875)  # Expected: 8
        self.assertAlmostEqual(resampled_series[2].data_loss, 0.625)

        # Check for t = 12
        self.assertEqual(resampled_series[3].t, 12)
        self.assertEqual(resampled_series[3].data['value'], 12)  # Expected: 12 (fully reconstructed)
        self.assertEqual(resampled_series[3].data_loss, 1)

        # Check for t = 16
        self.assertEqual(resampled_series[4].t, 16)
        self.assertAlmostEqual(resampled_series[4].data['value'], 16.03125)  # Expected: 16
        self.assertAlmostEqual(resampled_series[4].data_loss, 0.375)
        
        # Resample for 3 seconds
        resampled_series = series.resample(3)
        # DataTimePoint @ 0.0 (1970-01-01 00:00:00+00:00) with data "{'value': 0.6666666666666666}" and data_loss="0.0"
        # DataTimePoint @ 3.0 (1970-01-01 00:00:03+00:00) with data "{'value': 3.0}" and data_loss="0.0"
        # DataTimePoint @ 6.0 (1970-01-01 00:00:06+00:00) with data "{'value': 6.0}" and data_loss="0.0"
        # DataTimePoint @ 9.0 (1970-01-01 00:00:09+00:00) with data "{'value': 9.0}" and data_loss="1.0"
        # DataTimePoint @ 12.0 (1970-01-01 00:00:12+00:00) with data "{'value': 12.0}" and data_loss="1.0"
        # DataTimePoint @ 15.0 (1970-01-01 00:00:15+00:00) with data "{'value': 15.0}" and data_loss="0.6666666666666667"
        
        # Check len
        self.assertEqual(len(resampled_series), 6)

        # Check for t = 0
        self.assertEqual(resampled_series[0].t, 0)
        self.assertAlmostEqual(resampled_series[0].data['value'], 0.6666666666)  # Expected: 0.66..
        self.assertEqual(resampled_series[0].data_loss, 0)

        # Check for t = 3
        self.assertEqual(resampled_series[1].t, 3)
        self.assertEqual(resampled_series[1].data['value'], 3)  # Expected: 3
        self.assertEqual(resampled_series[1].data_loss, 0)

        # Check for t = 6
        self.assertEqual(resampled_series[2].t, 6)
        self.assertEqual(resampled_series[2].data['value'], 6)  # Expected: 6
        self.assertEqual(resampled_series[2].data_loss, 0)

        # Check for t = 9
        self.assertEqual(resampled_series[3].t, 9)
        self.assertEqual(resampled_series[3].data['value'], 9)  # Expected: 9 (fully reconstructed)
        self.assertEqual(resampled_series[3].data_loss, 1)

        # Check for t = 12
        self.assertEqual(resampled_series[4].t, 12)
        self.assertEqual(resampled_series[4].data['value'], 12)  # Expected: 12 (fully reconstructed)
        self.assertEqual(resampled_series[4].data_loss, 1)

        # Check for t = 15
        self.assertEqual(resampled_series[5].t, 15)
        self.assertEqual(resampled_series[5].data['value'], 15)  # Expected: 15
        self.assertAlmostEqual(resampled_series[5].data_loss, 0.6666666666)


    def test_upsample(self):
        
        # Test series
        series = DataTimePointSeries()
        series.append(DataTimePoint(t=-3,  data={'value':3}))
        series.append(DataTimePoint(t=-2,  data={'value':2}))
        series.append(DataTimePoint(t=-1,  data={'value':1}))
        series.append(DataTimePoint(t=0,  data={'value':0}))
        series.append(DataTimePoint(t=1,  data={'value':1}))
        series.append(DataTimePoint(t=2,  data={'value':2}))
        series.append(DataTimePoint(t=3,  data={'value':3}))
        series.append(DataTimePoint(t=4,  data={'value':4}))
        series.append(DataTimePoint(t=5,  data={'value':5}))
        series.append(DataTimePoint(t=6,  data={'value':6}))
        series.append(DataTimePoint(t=7,  data={'value':7}))
        series.append(DataTimePoint(t=16, data={'value':16}))
        series.append(DataTimePoint(t=17, data={'value':17}))
        series.append(DataTimePoint(t=18, data={'value':18}))
        
        # Resample (upsampling) for 4 seconds
        upsampled_series = series.resample(0.5)

        # Check start/end timestamps
        self.assertEqual(upsampled_series[0].t, -2.5)
        self.assertEqual(upsampled_series[-1].t, 17.5)
        
        # Check values & data losses
        self.assertEqual(upsampled_series[19].data['value'], 7)
        self.assertEqual(upsampled_series[19].data_loss, 0)

        self.assertEqual(upsampled_series[20].data['value'], 7.3125) # TODO: this should be 7.5.. 
        self.assertEqual(upsampled_series[20].data_loss, 0.5)

        self.assertEqual(upsampled_series[21].data['value'], 8)
        self.assertEqual(upsampled_series[21].data_loss, 1)


    def test_resample_edge_1(self):
        
        # Create 5-minute test data with 
        series = DataTimePointSeries()
        series.append(DataTimePoint(t = (60*20), data = {'value': 1.55}))
        series.append(DataTimePoint(t = (60*25), data = {'value': 1.58}))
        series.append(DataTimePoint(t = (60*30), data = {'value': 1.61}))
        #series.append(DataTimePoint(t = (60*35), data = {'value': 1.64}))
        #series.append(DataTimePoint(t = (60*40), data = {'value': 1.67}))
        series.append(DataTimePoint(t = (60*45), data = {'value': 1.70}))   # Try also 71? Will be imprecise...
        series.append(DataTimePoint(t = (60*50), data = {'value': 1.73}))   # Try also 74? Will be imprecise...
        
        # Resample
        resampled_series = Resampler('300s').process(series) 
        
        # Check len
        self.assertEqual(len(resampled_series), 5)
        
        # Do not check for t = 20 as extremes are not included 
        
        # Check for t = 25
        self.assertEqual(resampled_series[0].t, (60*25))
        self.assertAlmostEqual(resampled_series[0].data['value'], 1.58)  # Expected: 1.58 (original)
        self.assertEqual(resampled_series[0].data_loss, 0)

        # Check for t = 30
        self.assertEqual(resampled_series[1].t, (60*30))
        self.assertEqual(resampled_series[1].data['value'], 1.61)  # Expected: 1.61 (original)
        self.assertEqual(resampled_series[1].data_loss, 0)

        # Check for t = 35
        self.assertEqual(resampled_series[2].t, (60*35))
        self.assertAlmostEqual(resampled_series[2].data['value'], 1.64)  # Expected: 1.64 (reconstructed)
        self.assertEqual(resampled_series[2].data_loss, 1)

        # Check for t = 40
        self.assertEqual(resampled_series[3].t, (60*40))
        self.assertAlmostEqual(resampled_series[3].data['value'], 1.67)  # Expected: 1.67 (reconstructed)
        self.assertEqual(resampled_series[3].data_loss, 1)

        # Check for t = 45
        self.assertEqual(resampled_series[4].t, (60*45))
        self.assertAlmostEqual(resampled_series[4].data['value'], 1.70)  # Expected: 1.70 (original)
        self.assertEqual(resampled_series[4].data_loss, 0)

        # Do not check for t = 50 as extremes are not included 
            
            
    def test_resample_edge_2(self):

        # Create 5-minute test data  with ten secs ofset
        series = DataTimePointSeries()
        series.append(DataTimePoint(t = (60*20)+10, data = {'value': 1.55}))
        series.append(DataTimePoint(t = (60*25)+10, data = {'value': 1.58}))
        series.append(DataTimePoint(t = (60*30)+10, data = {'value': 1.61}))
        #series.append(DataTimePoint(t = (60*35)+10, data = {'value': 1.64}))
        #series.append(DataTimePoint(t = (60*40)+10, data = {'value': 1.68}))
        series.append(DataTimePoint(t = (60*45)+10, data = {'value': 1.71}))
        series.append(DataTimePoint(t = (60*50)+10, data = {'value': 1.74}))

        # Resample
        resampled_series = Resampler('300s').process(series) 
        
        # Check len
        self.assertEqual(len(resampled_series), 5)
        
        # Do not check for t = 20 as extremes are not included 
        
        # Check for t = 25
        self.assertEqual(resampled_series[0].t, (60*25))
        self.assertAlmostEqual(resampled_series[0].data['value'], 1.579)
        self.assertEqual(resampled_series[0].data_loss, 0)

        # Check for t = 30
        self.assertEqual(resampled_series[1].t, (60*30))
        self.assertEqual(resampled_series[1].data['value'], 1.609)
        self.assertEqual(resampled_series[1].data_loss, 0)

        # Check for t = 35
        self.assertEqual(resampled_series[2].t, (60*35))
        self.assertAlmostEqual(resampled_series[2].data['value'], 1.642, 3)
        self.assertAlmostEqual(resampled_series[2].data_loss, 290/300)

        # Check for t = 40
        self.assertEqual(resampled_series[3].t, (60*40))
        self.assertAlmostEqual(resampled_series[3].data['value'], 1.676, 3)
        self.assertEqual(resampled_series[3].data_loss, 1)

        # Check for t = 45
        self.assertEqual(resampled_series[4].t, (60*45))
        self.assertAlmostEqual(resampled_series[4].data['value'], 1.709, 3)
        self.assertAlmostEqual(resampled_series[4].data_loss, 10/300)

        # Do not check for t = 50 as extremes are not included
        # Note: a point after the right extreme here would mean
        # to have it after its validity, i.e. (60*50) + 150
        # TODO: is that what we want for the extremes?


    def test_resampler_data_indexes(self):
        
        # Time series from 16:58:00 to 17:32:00 (Europe/Rome)
        series = DataTimePointSeries()
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
                if i>10:
                    anomaly = 0.7 
                else:
                    anomaly = 0
                forecast = 0
                data_reconstructed = 0
                data_loss = 0.1
                
            point = DataTimePoint(t = start_t + (i*60),
                                  data = {'temperature': 154+i, 'humidity': 5},
                                  data_loss = data_loss)
            
            # Add extra data_indexes as if something operated on the time series
            point.data_indexes['data_reconstructed'] = data_reconstructed
            point.data_indexes['anomaly'] = anomaly
            point.data_indexes['forecast'] = forecast
            
            # Append
            series.append(point)
                
        # This is an indirect test of the series data_indexes. TODO: move it away.
        self.assertEqual(series._all_data_indexes(), ['data_reconstructed', 'data_loss', 'anomaly', 'forecast'])
        
        # Resample the time series
        resampled_series = Resampler('600s').process(series)        

        # Check that we have all the data_indexes
        self.assertEqual(resampled_series._all_data_indexes(), ['data_reconstructed', 'data_loss', 'anomaly', 'forecast'])

        #print('===========================================')
        #for i, item in enumerate(series):
        #    print('{}: dl={}, dr={}, a={}, f={}'.format(item.dt, item.data_loss, item.data_reconstructed, item.anomaly, item.forecast))
        #print('-------------------------------------------')
        #for item in resampled_series:
        #    print(item)
        #print('===========================================')

        # Check data_indexes math
        self.assertAlmostEqual(resampled_series[0].data_indexes['data_reconstructed'], 0.4)
        self.assertAlmostEqual(resampled_series[0].data_indexes['anomaly'], 0.275)
        self.assertAlmostEqual(resampled_series[0].data_indexes['forecast'], 0)
        # TODO: data loss is computed by compute_data_loss and not exact. expected 0.46 --> now corrected!
        self.assertAlmostEqual(resampled_series[0].data_loss, 0.46)
        
        self.assertAlmostEqual(resampled_series[1].data_indexes['data_reconstructed'], 0)
        self.assertAlmostEqual(resampled_series[1].data_indexes['anomaly'], 0.7)
        self.assertAlmostEqual(resampled_series[1].data_indexes['forecast'], 0.65)
        # TODO: data loss is computed by compute_data_loss. Data losses not present are treated as zero. Are we sure? 
        self.assertAlmostEqual(resampled_series[1].data_loss, 0.035)
 
        # Now resample at the same sampling interval and check data_indexes are still there
        same_resampled_series = series[0:5].resample(60)
        self.assertEqual(same_resampled_series._all_data_indexes(), ['data_reconstructed', 'data_loss', 'anomaly', 'forecast'])



class TestAggregator(unittest.TestCase):

    def setUp(self):       
                
        # Time series at 1m resolution from 14:58:00 to 15:32:00 (UTC)
        self.series_1 = DataTimePointSeries()
        start_t = 1436022000 - 120
        for i in range(35):
            point = DataTimePoint(t = start_t + (i*60),
                                  data = {'temperature': 154+i})
            self.series_1.append(point)
        
        # Time series at 1m resolution  from 14:00:00 to 15:30:00 (UTC)
        self.series_2 = DataTimePointSeries()
        start_t = 1436022000
        for i in range(34):
            point = DataTimePoint(t = start_t + (i*60),
                                  data = {'temperature': 154+i})
            self.series_2.append(point)

        # Time series at 1m resolution from 14:00:00 to 13:20:00 (UTC)
        self.series_3 = DataTimePointSeries()
        start_t = 1436022000 - 120
        for i in range(23):
            point = DataTimePoint(t = start_t + (i*60),
                                  data = {'temperature': 154+i})
            self.series_3.append(point) 
        
        # Time series at 1m resolution from 14:10:00 to 15:30:00 (UTC)
        self.series_4 = DataTimePointSeries()
        start_t = 1436022000 + 600
        for i in range(21):
            point = DataTimePoint(t = start_t + (i*60),
                                  data = {'temperature': 154+i})
            self.series_4.append(point)

        # Time series at 1m resolution from 14:58:00 to 15:32:00 (UTC)
        self.series_5 = DataTimePointSeries()
        start_t = 1436022000 - 120
        for i in range(35):
            if i > 10 and i <21:
                continue
            point = DataTimePoint(t = start_t + (i*60),
                                  data = {'temperature': 154+i})
            self.series_5.append(point)

        # Time series at 15m resolution from 01:00 to 06:00 (Europe/Rome)
        from_dt  = dt(2019,10,1,1,0,0, tzinfo='Europe/Rome')
        to_dt    = dt(2019,10,1,6,0,0, tzinfo='Europe/Rome')
        time_unit = TimeUnit('15m') 
        self.series_6 = DataTimePointSeries()
        slider_dt = from_dt
        count = 0
        while slider_dt < to_dt:
            if count not in [1, 6, 7, 8, 9, 10]:
                point = DataTimePoint(t = s_from_dt(slider_dt),
                                      data = {'temperature': 154+count},
                                      tz = 'Europe/Rome')
                self.series_6.append(point)
            slider_dt = slider_dt + time_unit
            count += 1

        # Time series at 1h resolution from 2019-10-24 00:00 to 2019-10-31 00:00:(Europe/Rome), DST off -> 2 AM repeated
        from_dt   = dt(2019,10,24,0,0,0, tzinfo='Europe/Rome')
        to_dt     = dt(2019,10,31,0,0,0, tzinfo='Europe/Rome')
        time_unit = TimeUnit('1h') 
        self.series_7 = DataTimePointSeries()
        slider_dt = from_dt
        count = 0
        while slider_dt < to_dt:
            if count not in []:
                point = DataTimePoint(t = s_from_dt(slider_dt),
                                      data = {'temperature': 154+count},
                                      tz = 'Europe/Rome')
                self.series_7.append(point)
            slider_dt = slider_dt + time_unit
            count += 1


    def test_slot(self):
          
        # Time series at 1m resolution from 14:58:00 to 15:32:00 (UTC), no from/to
        series = Aggregator('600s').process(self.series_1)        
        self.assertEqual(len(series), 3)
        self.assertEqual(series[0].start.dt, dt_from_str('2015-07-04 15:00:00+00:00'))
        self.assertEqual(series[1].start.dt, dt_from_str('2015-07-04 15:10:00+00:00'))
        self.assertEqual(series[2].start.dt, dt_from_str('2015-07-04 15:20:00+00:00'))
          
        # Time series at 1m resolution from 14:58:00 to 15:32:00 (UTC), no from/to, extremes included.
        # Disabled as extremes included is at the moment not implemented
        #series = Aggregator('10m').process(self.series_1, include_extremes=True)
        #self.assertEqual(len(series), 5)
        #self.assertEqual(series[0].start.dt, dt_from_str('2015-07-04 14:50:00+00:00'))
        #self.assertEqual(series[1].start.dt, dt_from_str('2015-07-04 15:00:00+00:00'))
        #self.assertEqual(series[2].start.dt, dt_from_str('2015-07-04 15:10:00+00:00'))
        #self.assertEqual(series[3].start.dt, dt_from_str('2015-07-04 15:20:00+00:00'))
        #self.assertEqual(series[4].start.dt, dt_from_str('2015-07-04 15:30:00+00:00'))
        #self.assertEqual(series[4].data_loss, 0.75)
 
        # Time series at 1m resolution from 14:58:00 to 17:32:00 (Europe/Rome), from 15:50 to 17:40
        # Disabled as it would require to include the extremes which is at the moment not implemented
        #series = Aggregator('10m').process(self.series_1,
        #                            from_t = s_from_dt(dt_from_str('2015-07-04 14:50:00+00:00')),
        #                            to_t   = s_from_dt(dt_from_str('2015-07-04 15:40:00+00:00')))
        #self.assertEqual(len(series), 5)
        #self.assertEqual(series[0].start.dt, dt_from_str('2015-07-04 14:50:00+00:00'))
        #self.assertEqual(series[1].start.dt, dt_from_str('2015-07-04 15:00:00+00:00'))
        #self.assertEqual(series[2].start.dt, dt_from_str('2015-07-04 15:10:00+00:00'))
        #self.assertEqual(series[3].start.dt, dt_from_str('2015-07-04 15:20:00+00:00'))
        #self.assertEqual(series[4].start.dt, dt_from_str('2015-07-04 15:30:00+00:00'))

        # Time series at 1m resolution from 14:58:00 to 15:32:00 (UTC), from 13:00 to 16:00
        # Disabled as it would require to include the extremes which is at the moment not implemented
        #series = Aggregator('10m').process(self.series_1,
        #                            from_t = s_from_dt(dt_from_str('2015-07-04 13:00:00+00:00')),
        #                            to_t   = s_from_dt(dt_from_str('2015-07-04 16:00:00+00:00')))
        # Here from and to are capped with the time series data points
        #self.assertEqual(str(series[0].start.dt), str('2015-07-04 14:50:00+00:00'))
        #self.assertEqual(str(series[-1].start.dt), str('2015-07-04 15:30:00+00:00'))

        # Time series at 15m resolution from 01:00 to 06:00 (Europe/Rome), slot in 1h
        series = Aggregator('1h').process(self.series_6)
        self.assertEqual(len(series), 4)
        self.assertEqual(str(series[0].start.dt), str('2019-10-01 01:00:00+02:00'))
        self.assertEqual(str(series[1].start.dt), str('2019-10-01 02:00:00+02:00'))
        self.assertEqual(str(series[2].start.dt), str('2019-10-01 03:00:00+02:00'))
        self.assertEqual(str(series[3].start.dt), str('2019-10-01 04:00:00+02:00'))
 
        # Test with changing timezone of the series
        self.series_6.change_timezone('America/New_York')
        series = Aggregator('1h').process(self.series_6)
        self.assertEqual(len(series), 4)
        self.assertEqual(str(series[0].start.dt), str('2019-09-30 19:00:00-04:00'))
        self.assertEqual(str(series[1].start.dt), str('2019-09-30 20:00:00-04:00'))
        self.assertEqual(str(series[2].start.dt), str('2019-09-30 21:00:00-04:00'))
        self.assertEqual(str(series[3].start.dt), str('2019-09-30 22:00:00-04:00'))
         
        # Time series from 2019,10,24,0,0,0 to 2019,10,31,0,0,0 (Europe/Rome), DST off -> 2 AM repeated
        series = Aggregator('1h').process(self.series_7)
        #print('=====================================')
        #for item in self.series_7:
        #    print(item)
        #print('-------------------------------------')
        #for item in series:
        #    print(item)
        #print('=====================================') 
        self.assertEqual(len(series), 168)
        self.assertEqual(str(series[73].start.dt), str('2019-10-27 01:00:00+02:00'))
        self.assertEqual(str(series[74].start.dt), str('2019-10-27 02:00:00+02:00'))
        self.assertEqual(str(series[75].start.dt), str('2019-10-27 02:00:00+01:00'))
        self.assertEqual(str(series[76].start.dt), str('2019-10-27 03:00:00+01:00'))
        self.assertEqual(series[76].data['temperature_avg'],230.5)

        # Time series from 2019,10,24,0,0,0 to 2019,10,31,0,0,0 (Europe/Rome), DST off -> 2 AM repeated
        series = Aggregator('1D').process(self.series_7)
        self.assertEqual(len(series), 6)
        self.assertEqual(str(series[0].start.dt), str('2019-10-24 00:00:00+02:00'))
        self.assertEqual(str(series[-1].start.dt), str('2019-10-29 00:00:00+01:00')) # Last one not included as right excluded


    def test_upslot(self):

        series = DataTimePointSeries()
        series.append(DataTimePoint(t = 0, data = {'value': 0}))
        series.append(DataTimePoint(t = 3600, data = {'value': 3600}))
        series.append(DataTimePoint(t = 7200, data = {'value': 7200}))
        
        #print('============================================')
        #for item in series: print(item)
        
        # This is a uplotting (upsampling), and there are data losses and strange values that should not be there.
        aggregator = Aggregator('10m')
        aggregated_series = aggregator.process(series)
        self.assertEqual(len(aggregated_series), 12)
        
        #print('--------------------------------------------')
        #for i, item in enumerate(aggregated_series): print(i, item)
        #print('============================================')
  
        # TODO: this is all wrong...
        # 0 DataTimeSlot @ [0.0,600.0] ([1970-01-01 00:00:00+00:00,1970-01-01 00:10:00+00:00]) with data={'value_avg': 0.0} and data_loss=0.0
        # 1 DataTimeSlot @ [600.0,1200.0] ([1970-01-01 00:10:00+00:00,1970-01-01 00:20:00+00:00]) with data={'value_avg': 0.0} and data_loss=0.0
        # 2 DataTimeSlot @ [1200.0,1800.0] ([1970-01-01 00:20:00+00:00,1970-01-01 00:30:00+00:00]) with data={'value_avg': 0.0} and data_loss=0.0
        # 3 DataTimeSlot @ [1800.0,2400.0] ([1970-01-01 00:30:00+00:00,1970-01-01 00:40:00+00:00]) with data={'value_avg': 7200.0} and data_loss=0.0
        # 4 DataTimeSlot @ [2400.0,3000.0] ([1970-01-01 00:40:00+00:00,1970-01-01 00:50:00+00:00]) with data={'value_avg': 14400.0} and data_loss=0.0
        # 5 DataTimeSlot @ [3000.0,3600.0] ([1970-01-01 00:50:00+00:00,1970-01-01 01:00:00+00:00]) with data={'value_avg': 28800.0} and data_loss=0.0
        # 6 DataTimeSlot @ [3600.0,4200.0] ([1970-01-01 01:00:00+00:00,1970-01-01 01:10:00+00:00]) with data={'value_avg': 28800.0} and data_loss=0.0
        # 7 DataTimeSlot @ [4200.0,4800.0] ([1970-01-01 01:10:00+00:00,1970-01-01 01:20:00+00:00]) with data={'value_avg': 57600.0} and data_loss=0.0
        # 8 DataTimeSlot @ [4800.0,5400.0] ([1970-01-01 01:20:00+00:00,1970-01-01 01:30:00+00:00]) with data={'value_avg': 115200.0} and data_loss=0.0
        # 9 DataTimeSlot @ [5400.0,6000.0] ([1970-01-01 01:30:00+00:00,1970-01-01 01:40:00+00:00]) with data={'value_avg': 14400.0} and data_loss=0.0
        # 10 DataTimeSlot @ [6000.0,6600.0] ([1970-01-01 01:40:00+00:00,1970-01-01 01:50:00+00:00]) with data={'value_avg': 28800.0} and data_loss=0.0
        # 11 DataTimeSlot @ [6600.0,7200.0] ([1970-01-01 01:50:00+00:00,1970-01-01 02:00:00+00:00]) with data={'value_avg': 57600.0} and data_loss=0.0


    def test_slot_operations_and_losses(self):
        
        # Import (overloaded) operations
        from ..operations import min, max, avg, sum
 
        # Time series from 16:58:00 to 17:32:00 (Europe/Rome)
        series = DataTimePointSeries()
        start_t = 1436022000 - 120
        for i in range(35):
            point = DataTimePoint(t = start_t + (i*60),
                                  data = {'temperature': 154+i, 'humidity': 5})
            series.append(point)
 
        # Add extra operations
        slotted_series = Aggregator('600s', operations=[avg, min,max]).process(series)
        self.assertAlmostEqual(slotted_series[0].data['temperature_min'], 156)
        self.assertAlmostEqual(slotted_series[0].data['temperature_max'], 165)
        self.assertAlmostEqual(slotted_series[0].data['temperature_avg'], 161)
        self.assertAlmostEqual(slotted_series[0].data['humidity_min'], 5)
        self.assertAlmostEqual(slotted_series[0].data['humidity_max'], 5)
        self.assertAlmostEqual(slotted_series[0].data['humidity_avg'], 5)
 
        # Entirely change the operation
        slotted_series = Aggregator('600s', operations=[sum]).process(series)
        # >>> (156*0.5)+157+158+159+160+161+162+163+164+165+(166*0.5)
        #1610.0
        #>>> 156+157+158+159+160+161+162+163+164+165
        #1605 
        self.assertEqual(slotted_series[0].data['temperature_sum'], 1605)
        self.assertEqual(slotted_series[0].data['humidity_sum'], 50)
        self.assertEqual(slotted_series[1].data['temperature_sum'], 1705)
        self.assertEqual(slotted_series[1].data['humidity_sum'], 50)
 
        # Operations as string
        slotted_series = Aggregator('600s', operations=['avg','min',max]).process(series)    
        self.assertAlmostEqual(slotted_series[0].data['temperature_min'], 156)
        self.assertAlmostEqual(slotted_series[0].data['temperature_max'], 165)
        self.assertAlmostEqual(slotted_series[0].data['temperature_avg'], 161)
        self.assertAlmostEqual(slotted_series[0].data['humidity_min'], 5)
        self.assertAlmostEqual(slotted_series[0].data['humidity_max'], 5)
        self.assertAlmostEqual(slotted_series[0].data['humidity_avg'], 5)


        # Time series from 16:58:00 to 17:32:00 (Europe/Rome) with a full short (single-slot) gap to be interpolated
        series = DataTimePointSeries()
        start_t = 1436022000 - 120
        for i in range(35):
            if i < 12 or i > 22:
                point = DataTimePoint(t = start_t + (i*60),
                                      data = {'temperature': 154+i, 'humidity': 5})
                series.append(point)

        slotted_series = Aggregator('600s', operations=[avg, min,max]).process(series)   
        self.assertEqual(len(slotted_series), 3)
        
        self.assertAlmostEqual(slotted_series[0].data['humidity_min'], 5)
        self.assertAlmostEqual(slotted_series[0].data['humidity_max'], 5)
        self.assertAlmostEqual(slotted_series[0].data['humidity_avg'], 5)
        self.assertAlmostEqual(slotted_series[0].data['temperature_min'], 156)
        self.assertAlmostEqual(slotted_series[0].data['temperature_max'], 165)
        self.assertAlmostEqual(slotted_series[0].data['temperature_avg'], 160.9875)
        
        self.assertAlmostEqual(slotted_series[1].data['humidity_min'], 5)
        self.assertAlmostEqual(slotted_series[1].data['humidity_max'], 5)
        self.assertAlmostEqual(slotted_series[1].data['humidity_avg'], 5)
        self.assertAlmostEqual(slotted_series[1].data['temperature_min'], 171)
        self.assertAlmostEqual(slotted_series[1].data['temperature_max'], 171)
        self.assertAlmostEqual(slotted_series[1].data['temperature_avg'], 171)

            
        # Time series from 16:58:00 to 17:32:00 (Europe/Rome) with a full long (multi-slot) gap to be interpolated
        series = DataTimePointSeries()
        start_t = 1436022000 - 120
        for i in range(60):
            if i < 10 or i >48:
                point = DataTimePoint(t = start_t + (i*60),
                                      data = {'temperature': 154+i, 'humidity': 5})
                series.append(point)

        slotted_series = Aggregator('600s', operations=[avg, min,max]).process(series)
        self.assertEqual(len(slotted_series), 5)
  
        self.assertEqual(slotted_series[0].data['temperature_min'], 156)
        self.assertEqual(slotted_series[0].data['temperature_avg'], 160.9875)
        self.assertEqual(slotted_series[1].data['temperature_min'], 171.0)
        self.assertEqual(slotted_series[1].data['temperature_avg'], 171.0)
        self.assertEqual(slotted_series[3].data['temperature_min'], 191.0)
        self.assertEqual(slotted_series[3].data['temperature_avg'], 191.0)

        # Re-slotting with the same time unit as the original points
        slotted_series = Aggregator('60s').process(series)
        # TODO: well.. tests it?

        # TODO: directly load a day-resolution time series in this test
        slotted_series = Resampler(86400).process(self.series_7)
        # TODO: well.. tests it?

        # TODO: upsampling (upslotting) not supported yet if missing data
        #slotted_series = Aggregator('60s').process(series)
        #slotted_series = Aggregator('30s').process(series)        
        

    def test_slot_data_indexes(self):
        
        # Time series from 16:58:00 to 17:32:00 (Europe/Rome)
        series = DataTimePointSeries()
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
                if i>10:
                    anomaly = 0.7 
                else:
                    anomaly = 0
                forecast = 0
                data_reconstructed = 0
                data_loss = 0.1
                
            point = DataTimePoint(t = start_t + (i*60),
                                  data = {'temperature': 154+i, 'humidity': 5},
                                  data_loss = data_loss)
            
            # Add extra data_indexes as if something operated on the time series
            point.data_indexes['data_reconstructed'] = data_reconstructed
            point.data_indexes['anomaly'] = anomaly
            point.data_indexes['forecast'] = forecast
            
            # Append
            series.append(point)
                
        # This is an indirect test of the series data_indexes. TODO: move it away.
        self.assertEqual(series._all_data_indexes(), ['data_reconstructed', 'data_loss', 'anomaly', 'forecast'])
        
        # Slot the time series
        slotted_series = Aggregator('600s').process(series)

        # Check that we have all the data_indexes
        self.assertEqual(slotted_series._all_data_indexes(), ['data_reconstructed', 'data_loss', 'anomaly', 'forecast'])

        #print('===========================================')
        #for i, item in enumerate(series):
        #    print('{}: dl={}, dr={}, a={}, f={}'.format(item.dt, item.data_loss, item.data_reconstructed, item.anomaly, item.forecast))
        #print('-------------------------------------------')
        #for item in slotted_series:
        #    print(item)
        #print('===========================================')

        # Check data_indexes math
        self.assertAlmostEqual(slotted_series[0].data_indexes['anomaly'], 0.0375)
        self.assertAlmostEqual(slotted_series[0].data_indexes['data_reconstructed'], 0.15)
        self.assertAlmostEqual(slotted_series[0].data_indexes['forecast'], 0)
         
        self.assertAlmostEqual(slotted_series[1].data_indexes['anomaly'], 0.5676, 4)
        self.assertAlmostEqual(slotted_series[1].data_indexes['data_reconstructed'], 0.25)
        self.assertAlmostEqual(slotted_series[1].data_indexes['forecast'], 0.15)
 
        self.assertEqual(slotted_series[2].data_indexes['anomaly'], None)
        self.assertAlmostEqual(slotted_series[2].data_indexes['data_reconstructed'], 0)
        self.assertEqual(slotted_series[2].data_indexes['forecast'], 1)
 
        # The following is re-computed from missing coverage only, and not marked as None.
        # See the compute_data_loss function fore more details and some more comments.
        self.assertEqual(slotted_series[2].data_loss, 0) 




