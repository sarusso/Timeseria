import unittest
from ..exceptions import InputException
from ..time import dt
from ..units import Unit, TimeUnit

class TestUnits(unittest.TestCase):

    def setUp(self):       
        pass

    def test_Unit(self):
        
        with self.assertRaises(InputException):
            _ = TimeUnit('hi')        
        
        unit = Unit(60)
        self.assertEqual(unit.value, 60)

    def test_Unit_math(self):
        
        unit1 = Unit(60)
        self.assertEqual(unit1+5, 65)
        self.assertEqual(5+unit1, 65)
        self.assertEqual(unit1-5, 55)
        self.assertEqual(5-unit1, -55)        
        
        unit2 = Unit(67)
        self.assertEqual(unit1+unit2, 127)
        self.assertEqual(unit1-unit2, -7)
        self.assertEqual(unit2-unit1, 7)



    def test_TimeUnit(self):

        with self.assertRaises(InputException):
            _ = TimeUnit(60)
        
        with self.assertRaises(InputException):
            _ = TimeUnit('15m', '20s')

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

        # Test type
        self.assertEqual(TimeUnit('15m').type, TimeUnit.PHYSICAL)
        self.assertEqual(TimeUnit('1h').type, TimeUnit.PHYSICAL)
        self.assertEqual(TimeUnit('1D').type, TimeUnit.CALENDAR)
        self.assertEqual(TimeUnit('1M').type, TimeUnit.CALENDAR)
        
        # Test sum with TimePoint
        time_unit = TimeUnit('1h')
        from ..datastructures import TimePoint
        time_point = TimePoint(60)
        self.assertEqual((time_point+time_unit).t, 3660)

        # Test unit value
        with self.assertRaises(TypeError):
            TimeUnit('1D').value
        with self.assertRaises(TypeError):
            TimeUnit('2Y').value 
        self.assertEqual(TimeUnit('1m').value, 60)
        self.assertEqual(TimeUnit('15m').value, 900)
        self.assertEqual(TimeUnit('1h').value, 3600)


    def test_TimeUnit_duration(self):

        date_time1 = dt(2015,10,24,0,15,0, tzinfo='Europe/Rome')
        date_time2 = dt(2015,10,25,0,15,0, tzinfo='Europe/Rome')
        date_time3 = dt(2015,10,26,0,15,0, tzinfo='Europe/Rome')
 
        # Day unit
        time_unit = TimeUnit('1D')
        self.assertEqual(time_unit.duration_s(date_time1), 86400) # No DST, standard day
        self.assertEqual(time_unit.duration_s(date_time2), 90000) # DST, change
                 
        # Week unit
        time_unit = TimeUnit('1W')
        self.assertEqual(time_unit.duration_s(date_time1), (86400*7)+3600)
        self.assertEqual(time_unit.duration_s(date_time3), (86400*7))
         
        # Month Unit
        time_unit = TimeUnit('1M')
        self.assertEqual(time_unit.duration_s(date_time1), ((86400*31)+3600)) # October has 31 days, but here we have a DST change in the middle
        self.assertEqual(time_unit.duration_s(date_time3), (86400*31)) # October has 31 days

        # Year Unit
        time_unit = TimeUnit('1Y')
        self.assertEqual(time_unit.duration_s(dt(2014,10,24,0,15,0, tzinfo='Europe/Rome')), (86400*365)) # Standard year
        self.assertEqual(time_unit.duration_s(dt(2015,10,24,0,15,0, tzinfo='Europe/Rome')), (86400*366)) # Leap year
        

    def test_TimeUnit_shift_dt(self):

        date_time1 = dt(2015,10,24,0,15,0, tzinfo='Europe/Rome')
        date_time2 = dt(2015,10,25,0,15,0, tzinfo='Europe/Rome')
        date_time3 = dt(2015,10,26,0,15,0, tzinfo='Europe/Rome')

        # Day unit
        time_unit = TimeUnit('1D')
        self.assertEqual(time_unit.shift_dt(date_time1), dt(2015,10,25,0,15,0, tzinfo='Europe/Rome')) # No DST, standard day
        self.assertEqual(time_unit.shift_dt(date_time2), dt(2015,10,26,0,15,0, tzinfo='Europe/Rome')) # DST, change
                 
        # Week unit
        time_unit = TimeUnit('1W')
        self.assertEqual(time_unit.shift_dt(date_time1), dt(2015,10,31,0,15,0, tzinfo='Europe/Rome'))
        self.assertEqual(time_unit.shift_dt(date_time3), dt(2015,11,2,0,15,0, tzinfo='Europe/Rome'))
         
        # Month Unit
        time_unit = TimeUnit('1M')
        self.assertEqual(time_unit.shift_dt(date_time1), dt(2015,11,24,0,15,0, tzinfo='Europe/Rome'))
        self.assertEqual(time_unit.shift_dt(date_time2), dt(2015,11,25,0,15,0, tzinfo='Europe/Rome'))
        self.assertEqual(time_unit.shift_dt(date_time3), dt(2015,11,26,0,15,0, tzinfo='Europe/Rome'))
        
        # Test 12%12 must give 12 edge case
        self.assertEqual(time_unit.shift_dt(dt(2015,1,1,0,0,0, tzinfo='Europe/Rome')), dt(2015,2,1,0,0,0, tzinfo='Europe/Rome'))
        self.assertEqual(time_unit.shift_dt(dt(2015,11,1,0,0,0, tzinfo='Europe/Rome')), dt(2015,12,1,0,0,0, tzinfo='Europe/Rome'))

        # Year Unit
        time_unit = TimeUnit('1Y')
        self.assertEqual(time_unit.shift_dt(date_time1), dt(2016,10,24,0,15,0, tzinfo='Europe/Rome'))
        

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
        date_time = dt(1969,12,31,23,0,1, tzinfo='Europe/Rome') # negative epoch    
        self.assertEqual(time_unit.floor_dt(date_time), dt(1969,12,31,23,0,0, tzinfo='Europe/Rome'))
        self.assertEqual(time_unit.ceil_dt(date_time), dt(1970,1,1,2,0, tzinfo='Europe/Rome'))

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

        # Test ceil/floor/round with a calendar timeunit and across a DST change
        
        # Day unit
        time_unit = TimeUnit('1D')
        
        date_time1 = dt(2015,10,25,4,15,34, tzinfo='Europe/Rome') # DST off (+01:00)
        date_time1_floor = dt(2015,10,25,0,0,0, tzinfo='Europe/Rome') # DST on (+02:00)
        date_time1_ceil = dt(2015,10,26,0,0,0, tzinfo='Europe/Rome') # DST off (+01:00)

        self.assertEqual(time_unit.floor_dt(date_time1), date_time1_floor)
        self.assertEqual(time_unit.ceil_dt(date_time1), date_time1_ceil)

        # Month unit
        time_unit = TimeUnit('1M')
        
        date_time1 = dt(2015,10,25,4,15,34, tzinfo='Europe/Rome') # DST off (+01:00)
        date_time1_floor = dt(2015,10,1,0,0,0, tzinfo='Europe/Rome') # DST on (+02:00)
        date_time1_ceil = dt(2015,11,1,0,0,0, tzinfo='Europe/Rome') # DST off (+01:00)

        self.assertEqual(time_unit.floor_dt(date_time1), date_time1_floor)
        self.assertEqual(time_unit.ceil_dt(date_time1), date_time1_ceil)

        # Week unit
        # TODO: not implemented...

        # Year unit
        time_unit = TimeUnit('1Y')
        
        date_time1 = dt(2015,10,25,4,15,34, tzinfo='Europe/Rome')
        date_time1_floor = dt(2015,1,1,0,0,0, tzinfo='Europe/Rome')
        date_time1_ceil = dt(2016,1,1,0,0,0, tzinfo='Europe/Rome')

        self.assertEqual(time_unit.floor_dt(date_time1), date_time1_floor)
        self.assertEqual(time_unit.ceil_dt(date_time1), date_time1_ceil)


    def tearDown(self):
        pass

