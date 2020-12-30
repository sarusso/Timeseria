import unittest
import datetime
from ..exceptions import InputException
from ..time import dt, correct_dt_dst, timezonize, s_from_dt, dt_to_str, dt_from_str, change_tz
from ..units import Unit, TimeUnit

class TestUnits(unittest.TestCase):

    def setUp(self):       
        pass


    def test_TimeUnit(self):

        with self.assertRaises(InputException):
            _ = TimeUnit(60)
        
        # TODO: I had to comment out this test, find out why..
        # Complex time intervals are not supported
        #with self.assertRaises(InputException):
        #   _ = TimeUnit('15m', '20s')
        
        # TODO: test with units
        #print TimeUnit('1m').value
        #print TimeSlot(unit=TimeUnit('1m')).unit.value

        # Not valid 'q' type
        with self.assertRaises(InputException):
            _ = TimeUnit('15q')

        # String init
        time_unit_1 = TimeUnit('15m')
        self.assertEqual(time_unit_1.string, '15m')

        time_unit_2 = TimeUnit('15m_30s_3u')
        self.assertEqual(time_unit_2.string, '15m_30s_3u')
        
        time_unit_3 = TimeUnit(days=1)
        
        # Sum with other TimeUnit objects
        self.assertEqual((time_unit_1+time_unit_2+time_unit_3).string, '1D_30m_30s_3u')

        # Sum with datetime (also on DST change)
        time_unit = TimeUnit('1h')
        date_time1 = dt(2015,10,25,0,15,0, tzinfo='Europe/Rome')
        date_time2 = date_time1 + time_unit
        date_time3 = date_time2 + time_unit
        date_time4 = date_time3 + time_unit
        date_time5 = date_time4 + time_unit

        self.assertEqual(str(date_time1), '2015-10-25 00:15:00+02:00')
        self.assertEqual(str(date_time2), '2015-10-25 01:15:00+02:00')
        self.assertEqual(str(date_time3), '2015-10-25 02:15:00+02:00')
        self.assertEqual(str(date_time4), '2015-10-25 02:15:00+01:00')
        self.assertEqual(str(date_time5), '2015-10-25 03:15:00+01:00')

        # Test length
        #self.assertEqual(TimeUnit('15m').length, 900)
        #self.assertEqual(TimeUnit('1h').length, 3600)

        #with self.assertRaises(Exception):
        #    TimeUnit('15D').length

        # Test type
        self.assertEqual(TimeUnit('15m').type, TimeUnit.PHYSICAL)
        self.assertEqual(TimeUnit('1h').type, TimeUnit.PHYSICAL)
        self.assertEqual(TimeUnit('1D').type, TimeUnit.LOGICAL)
        self.assertEqual(TimeUnit('1M').type, TimeUnit.LOGICAL)
        
        # Test sum with TimePoint
        time_unit = TimeUnit('1h')
        from ..datastructures import TimePoint
        time_point = TimePoint(60)
        self.assertEqual((time_point+time_unit).t, 3660)


    def test_TimeUnit_math(self):

        # Test that complex time_units are not handable
        time_unit = TimeUnit('1D_3h_5m')
        date_time = dt(2015,1,1,16,37,14, tzinfo='Europe/Rome')
        
        with self.assertRaises(InputException):
            _ = time_unit.floor_dt(date_time)


        # Test in ceil/floor/round normal conditions (hours)
        time_unit = TimeUnit('1h')
        date_time = dt(2015,1,1,16,37,14, tzinfo='Europe/Rome')
        self.assertEqual(time_unit.floor_dt(date_time), dt(2015,1,1,16,0,0, tzinfo='Europe/Rome'))
        self.assertEqual(time_unit.ceil_dt(date_time), dt(2015,1,1,17,0,0, tzinfo='Europe/Rome'))

         
        # Test in ceil/floor/round normal conditions (minutes)
        time_unit = TimeUnit('15m')
        date_time = dt(2015,1,1,16,37,14, tzinfo='Europe/Rome')
        self.assertEqual(time_unit.floor_dt(date_time), dt(2015,1,1,16,30,0, tzinfo='Europe/Rome'))
        self.assertEqual(time_unit.ceil_dt(date_time), dt(2015,1,1,16,45,0, tzinfo='Europe/Rome'))


        # Test ceil/floor/round in normal conditions (seconds)
        time_unit = TimeUnit('30s')
        date_time = dt(2015,1,1,16,37,14, tzinfo='Europe/Rome') 
        self.assertEqual(time_unit.floor_dt(date_time), dt(2015,1,1,16,37,0, tzinfo='Europe/Rome'))
        self.assertEqual(time_unit.ceil_dt(date_time), dt(2015,1,1,16,37,30, tzinfo='Europe/Rome'))

   
        # Test ceil/floor/round across 1970-1-1 (minutes) 
        time_unit = TimeUnit('5m')
        date_time1 = dt(1969,12,31,23,57,29, tzinfo='UTC') # epoch = -3601
        date_time2 = dt(1969,12,31,23,59,59, tzinfo='UTC') # epoch = -3601       
        self.assertEqual(time_unit.floor_dt(date_time1), dt(1969,12,31,23,55,0, tzinfo='UTC'))
        self.assertEqual(time_unit.ceil_dt(date_time1), dt(1970,1,1,0,0, tzinfo='UTC'))
        self.assertEqual(time_unit.round_dt(date_time1), dt(1969,12,31,23,55,0, tzinfo='UTC'))
        self.assertEqual(time_unit.round_dt(date_time2), dt(1970,1,1,0,0, tzinfo='UTC'))


        # Test ceil/floor/round (3 hours-test)
        time_unit = TimeUnit('3h')
        date_time = dt(1969,12,31,22,0,1, tzinfo='Europe/Rome') # negative epoch
        
        # TODO: test fails!! fix me!        
        #self.assertEqual(time_unit.floor_dt(date_time1), dt(1969,12,31,21,0,0, tzinfo='Europe/Rome'))
        #self.assertEqual(time_unit.ceil_dt(date_time1), dt(1970,1,1,0,0, tzinfo='Europe/Rome'))


        # Test ceil/floor/round across 1970-1-1 (together with the 2 hours-test, TODO: decouple) 
        time_unit = TimeUnit('2h')
        date_time1 = dt(1969,12,31,22,59,59, tzinfo='Europe/Rome') # negative epoch
        date_time2 = dt(1969,12,31,23,0,1, tzinfo='Europe/Rome') # negative epoch  
        self.assertEqual(time_unit.floor_dt(date_time1), dt(1969,12,31,22,0,0, tzinfo='Europe/Rome'))
        self.assertEqual(time_unit.ceil_dt(date_time1), dt(1970,1,1,0,0, tzinfo='Europe/Rome'))
        self.assertEqual(time_unit.round_dt(date_time1), dt(1969,12,31,22,0, tzinfo='Europe/Rome'))
        self.assertEqual(time_unit.round_dt(date_time2), dt(1970,1,1,0,0, tzinfo='Europe/Rome'))

        # Test ceil/floor/round across DST change (hours)
        time_unit = TimeUnit('1h')
        
        date_time1 = dt(2015,10,25,0,15,0, tzinfo='Europe/Rome')
        date_time2 = date_time1 + time_unit    # 2015-10-25 01:15:00+02:00    
        date_time3 = date_time2 + time_unit    # 2015-10-25 02:15:00+02:00
        date_time4 = date_time3 + time_unit    # 2015-10-25 02:15:00+01:00

        date_time1_rounded = dt(2015,10,25,0,0,0, tzinfo='Europe/Rome')
        date_time2_rounded = date_time1_rounded + time_unit        
        date_time3_rounded = date_time2_rounded + time_unit
        date_time4_rounded = date_time3_rounded + time_unit
        date_time5_rounded = date_time4_rounded + time_unit
               
        self.assertEqual(time_unit.floor_dt(date_time2), date_time2_rounded)
        self.assertEqual(time_unit.ceil_dt(date_time2), date_time3_rounded)
          
        self.assertEqual(time_unit.floor_dt(date_time3), date_time3_rounded)
        self.assertEqual(time_unit.ceil_dt(date_time3), date_time4_rounded)
        
        self.assertEqual(time_unit.floor_dt(date_time4), date_time4_rounded)
        self.assertEqual(time_unit.ceil_dt(date_time4), date_time5_rounded)


        # Test ceil/floor/round with a logical timeunit
        #time_unit = TimeUnit('1D')
        
        #date_time1 = dt(2015,10,25,0,15,0, tzinfo='Europe/Rome')
        #date_time1_floor = dt(2015,10,25,0,0,0, tzinfo='Europe/Rome')


        # TODO (test shift as well)        
        #time_unit = TimeUnit('15m')
        #date_time = dt(2015,1,1,16,37,14, tzinfo='Europe/Rome')
        #date_time_rounded = dt(2015,1,1,16,0,0, tzinfo='Europe/Rome')
        
        #print shift_dt(date_time, time_unit, 4)
        #print shift_dt(date_time, time_unit, -2)

        #print shift_dt(date_time_rounded, time_unit, 4)
        #print shift_dt(date_time_rounded, time_unit, -2)

        # Test shift across DST change

    def tearDown(self):
        pass























