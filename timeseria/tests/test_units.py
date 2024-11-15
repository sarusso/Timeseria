import unittest
from propertime.utils import dt
from ..units import Unit, TimeUnit

# Setup logging
from .. import logger
logger.setup()


class TestUnits(unittest.TestCase):

    def test_Unit(self):

        with self.assertRaises(ValueError):
            _ = TimeUnit('hi')

        unit = Unit(60)
        self.assertEqual(unit.value, 60)


    def test_Unit_math(self):

        # Sum & subtract
        unit1 = Unit(60)
        self.assertEqual(unit1+5, 65)
        self.assertEqual(5+unit1, 65)
        self.assertEqual(unit1-5, 55)
        self.assertEqual(5-unit1, -55)

        # Sum & subtract Units
        unit2 = Unit(67)
        self.assertEqual(unit1+unit2, 127)
        self.assertEqual(unit1-unit2, -7)
        self.assertEqual(unit2-unit1, 7)

        # Divide & multiply
        self.assertEqual(unit1/2, 30)
        self.assertEqual(120/unit1,2)
        self.assertEqual(unit1*2,120)
        self.assertEqual(2*unit1,120)



class TestTimeUnits(unittest.TestCase):

    def test_TimeUnit(self):

        with self.assertRaises(ValueError):
            _ = TimeUnit('15m', '20s')

        # Not valid 'q' type
        with self.assertRaises(ValueError):
            _ = TimeUnit('15q')

        # Numerical init
        time_unit_1 = TimeUnit(60)
        self.assertEqual(str(time_unit_1), '60s')

        # String init
        time_unit_1 = TimeUnit('15m')
        self.assertEqual(str(time_unit_1), '15m')

        time_unit_2 = TimeUnit('15m_30s')
        self.assertEqual(str(time_unit_2), '15m_30s')

        # Components init
        self.assertEqual(TimeUnit(days=1).days, 1)
        self.assertEqual(TimeUnit(years=2).years, 2)
        self.assertEqual(TimeUnit(minutes=1).minutes, 1)
        self.assertEqual(TimeUnit(minutes=15).minutes, 15)
        self.assertEqual(TimeUnit(hours=1).hours, 1)

        # Test various init and correct handling of time componentes
        self.assertEqual(TimeUnit('1D').days, 1)
        self.assertEqual(TimeUnit('2Y').years, 2)
        self.assertEqual(TimeUnit('1m').minutes, 1)
        self.assertEqual(TimeUnit('15m').minutes, 15)
        self.assertEqual(TimeUnit('1h').hours, 1)

        # Test floating point seconds init
        self.assertEqual(TimeUnit('1.2345s').as_seconds(), 1.2345)
        self.assertEqual(TimeUnit('1.234s').as_seconds(), 1.234)
        self.assertEqual(TimeUnit('1.02s').as_seconds(), 1.02)
        self.assertEqual(TimeUnit('1.000005s').as_seconds(), 1.000005)
        self.assertEqual(TimeUnit('67.000005s').seconds, 67.000005)

        # Test string values
        self.assertEqual(str(TimeUnit(600)), '600s') # Int converted to string representation
        self.assertEqual(str(TimeUnit(600.0)), '600s') # Float converted to string representation
        self.assertEqual(str(TimeUnit(600.45)), '600.45s') # Float converted to string representation (using microseconds)

        self.assertEqual(str(TimeUnit(days=1)), '1D')
        self.assertEqual(str(TimeUnit(years=2)), '2Y')
        self.assertEqual(str(TimeUnit(minutes=1)), '1m')
        self.assertEqual(str(TimeUnit(minutes=15)), '15m')
        self.assertEqual(str(TimeUnit(hours=1)), '1h')

        self.assertEqual(str(time_unit_1), '15m')
        self.assertEqual(str(time_unit_2), '15m_30s')
        self.assertEqual(str(TimeUnit(days=1)), '1D') # This is obtained using the unit's string representation

        # Test unit equalities
        self.assertEqual(TimeUnit(hours=1), TimeUnit(hours=1))
        self.assertEqual(TimeUnit(hours=1), '1h')
        self.assertNotEqual(TimeUnit(hours=1), TimeUnit(hours=2))
        self.assertNotEqual(TimeUnit(hours=1), 'variable')

        self.assertEqual(TimeUnit(days=1), TimeUnit(days=1))
        self.assertEqual(TimeUnit(days=1), '1D')
        self.assertNotEqual(TimeUnit(days=1), TimeUnit(days=2))
        self.assertNotEqual(TimeUnit(days=1), 'variable')

        self.assertEqual(TimeUnit(hours=1), TimeUnit(seconds=3600))
        self.assertNotEqual(TimeUnit(days=1), TimeUnit(hours=24))


    def test_TimeUnit_math(self):

        time_unit_1 = TimeUnit('15m')
        time_unit_2 = TimeUnit('15m_30s')
        time_unit_3 = TimeUnit(days=1)

        # Sum with other TimeUnit objects
        self.assertEqual(str(time_unit_1+time_unit_2+time_unit_3), '1D_30m_30s')

        # Sum with datetime (also on DST change)
        time_unit = TimeUnit('1h')
        datetime1 = dt(2015,10,25,0,15,0, tz='Europe/Rome')
        datetime2 = datetime1 + time_unit
        datetime3 = datetime2 + time_unit
        datetime4 = datetime3 + time_unit
        datetime5 = datetime4 + time_unit

        self.assertEqual(str(datetime1), '2015-10-25 00:15:00+02:00')
        self.assertEqual(str(datetime2), '2015-10-25 01:15:00+02:00')
        self.assertEqual(str(datetime3), '2015-10-25 02:15:00+02:00')
        self.assertEqual(str(datetime4), '2015-10-25 02:15:00+01:00')
        self.assertEqual(str(datetime5), '2015-10-25 03:15:00+01:00')

        # Sum with a numerical value
        time_unit = TimeUnit('1h')
        epoch1 = 3600
        self.assertEqual(epoch1 + time_unit, 7200)

        # Subtract to other TimeUnit object
        with self.assertRaises(NotImplementedError):
            time_unit_1 - time_unit_2

        # Subtract to a datetime object
        with self.assertRaises(NotImplementedError):
            time_unit_1 - datetime1

        # In general, subtracting to anything is not implemented
        with self.assertRaises(NotImplementedError):
            time_unit_1 - 'hello'

        # Subtract from a datetime (also on DST change)
        time_unit = TimeUnit('1h')
        datetime1 = dt(2015,10,25,3,15,0, tz='Europe/Rome')
        datetime2 = datetime1 - time_unit
        datetime3 = datetime2 - time_unit
        datetime4 = datetime3 - time_unit
        datetime5 = datetime4 - time_unit

        self.assertEqual(str(datetime1), '2015-10-25 03:15:00+01:00')
        self.assertEqual(str(datetime2), '2015-10-25 02:15:00+01:00')
        self.assertEqual(str(datetime3), '2015-10-25 02:15:00+02:00')
        self.assertEqual(str(datetime4), '2015-10-25 01:15:00+02:00')
        self.assertEqual(str(datetime5), '2015-10-25 00:15:00+02:00')

        # Subtract from a numerical value
        time_unit = TimeUnit('1h')
        epoch1 = 7200
        self.assertEqual(epoch1 - time_unit, 3600)

        # Test sum with TimePoint
        time_unit = TimeUnit('1h')
        from ..datastructures import TimePoint
        time_point = TimePoint(60)
        self.assertEqual((time_point+time_unit).t, 3660)

        # Test equal
        time_unit_1 = TimeUnit('15m')
        self.assertEqual(time_unit_1, 900)


    def test_TimeUnit_duration(self):

        datetime1 = dt(2015,10,24,0,15,0, tz='Europe/Rome')
        datetime2 = dt(2015,10,25,0,15,0, tz='Europe/Rome')
        datetime3 = dt(2015,10,26,0,15,0, tz='Europe/Rome')

        # Day unit
        time_unit = TimeUnit('1D')
        with self.assertRaises(ValueError):
            time_unit.as_seconds()
        self.assertEqual(time_unit.as_seconds(datetime1), 86400) # No DST, standard day
        self.assertEqual(time_unit.as_seconds(datetime2), 90000) # DST, change

        # Week unit
        time_unit = TimeUnit('1W')
        with self.assertRaises(ValueError):
            time_unit.as_seconds()
        self.assertEqual(time_unit.as_seconds(datetime1), (86400*7)+3600)
        self.assertEqual(time_unit.as_seconds(datetime3), (86400*7))

        # Month Unit
        time_unit = TimeUnit('1M')
        with self.assertRaises(ValueError):
            time_unit.as_seconds()
        self.assertEqual(time_unit.as_seconds(datetime1), ((86400*31)+3600)) # October has 31 days, but here we have a DST change in the middle
        self.assertEqual(time_unit.as_seconds(datetime3), (86400*31)) # October has 31 days

        # Year Unit
        time_unit = TimeUnit('1Y')
        with self.assertRaises(ValueError):
            time_unit.as_seconds()
        self.assertEqual(time_unit.as_seconds(dt(2014,10,24,0,15,0, tz='Europe/Rome')), (86400*365)) # Standard year
        self.assertEqual(time_unit.as_seconds(dt(2015,10,24,0,15,0, tz='Europe/Rome')), (86400*366)) # Leap year

        # Test duration with composite point seconds init
        self.assertEqual(TimeUnit(minutes=1, seconds=3).as_seconds(), 63)


    def test_TimeUnit_shift(self):

        datetime1 = dt(2015,10,24,0,15,0, tz='Europe/Rome')
        datetime2 = dt(2015,10,25,0,15,0, tz='Europe/Rome')
        datetime3 = dt(2015,10,26,0,15,0, tz='Europe/Rome')

        # Day unit
        time_unit = TimeUnit('1D')
        self.assertEqual(time_unit.shift(datetime1), dt(2015,10,25,0,15,0, tz='Europe/Rome')) # No DST, standard day
        self.assertEqual(time_unit.shift(datetime2), dt(2015,10,26,0,15,0, tz='Europe/Rome')) # DST, change

        # Week unit
        time_unit = TimeUnit('1W')
        self.assertEqual(time_unit.shift(datetime1), dt(2015,10,31,0,15,0, tz='Europe/Rome'))
        self.assertEqual(time_unit.shift(datetime3), dt(2015,11,2,0,15,0, tz='Europe/Rome'))

        # Month Unit
        time_unit = TimeUnit('1M')
        self.assertEqual(time_unit.shift(datetime1), dt(2015,11,24,0,15,0, tz='Europe/Rome'))
        self.assertEqual(time_unit.shift(datetime2), dt(2015,11,25,0,15,0, tz='Europe/Rome'))
        self.assertEqual(time_unit.shift(datetime3), dt(2015,11,26,0,15,0, tz='Europe/Rome'))

        # Test 12%12 must give 12 edge case
        self.assertEqual(time_unit.shift(dt(2015,1,1,0,0,0, tz='Europe/Rome')), dt(2015,2,1,0,0,0, tz='Europe/Rome'))
        self.assertEqual(time_unit.shift(dt(2015,11,1,0,0,0, tz='Europe/Rome')), dt(2015,12,1,0,0,0, tz='Europe/Rome'))

        # Year Unit
        time_unit = TimeUnit('1Y')
        self.assertEqual(time_unit.shift(datetime1), dt(2016,10,24,0,15,0, tz='Europe/Rome'))

        # Test on not-existent hour due to DST
        starting_dt = dt(2023,3,25,2,15, tz='Europe/Rome')
        with self.assertRaises(ValueError):
            starting_dt + TimeUnit('1D')

    def test_TimeUnit_operations(self):

        # Test that complex time_units are not handable
        time_unit = TimeUnit('1D_3h_5m')
        datetime = dt(2015,1,1,16,37,14, tz='Europe/Rome')

        with self.assertRaises(ValueError):
            _ = time_unit.floor(datetime)

        # Test in ceil/floor/round normal conditions (hours)
        time_unit = TimeUnit('1h')
        datetime = dt(2015,1,1,16,37,14, tz='Europe/Rome')
        self.assertEqual(time_unit.floor(datetime), dt(2015,1,1,16,0,0, tz='Europe/Rome'))
        self.assertEqual(time_unit.ceil(datetime), dt(2015,1,1,17,0,0, tz='Europe/Rome'))

        # Test in ceil/floor/round normal conditions (minutes)
        time_unit = TimeUnit('15m')
        datetime = dt(2015,1,1,16,37,14, tz='Europe/Rome')
        self.assertEqual(time_unit.floor(datetime), dt(2015,1,1,16,30,0, tz='Europe/Rome'))
        self.assertEqual(time_unit.ceil(datetime), dt(2015,1,1,16,45,0, tz='Europe/Rome'))

        # Test ceil/floor/round in normal conditions (seconds)
        time_unit = TimeUnit('30s')
        datetime = dt(2015,1,1,16,37,14, tz='Europe/Rome')
        self.assertEqual(time_unit.floor(datetime), dt(2015,1,1,16,37,0, tz='Europe/Rome'))
        self.assertEqual(time_unit.ceil(datetime), dt(2015,1,1,16,37,30, tz='Europe/Rome'))

        # Test ceil/floor/round across 1970-1-1 (minutes)
        time_unit = TimeUnit('5m')
        datetime1 = dt(1969,12,31,23,57,29, tz='UTC') # epoch = -3601
        datetime2 = dt(1969,12,31,23,59,59, tz='UTC') # epoch = -3601
        self.assertEqual(time_unit.floor(datetime1), dt(1969,12,31,23,55,0, tz='UTC'))
        self.assertEqual(time_unit.ceil(datetime1), dt(1970,1,1,0,0, tz='UTC'))
        self.assertEqual(time_unit.round(datetime1), dt(1969,12,31,23,55,0, tz='UTC'))
        self.assertEqual(time_unit.round(datetime2), dt(1970,1,1,0,0, tz='UTC'))

        # Test ceil/floor/round (3 hours-test)
        time_unit = TimeUnit('3h')
        datetime = dt(1969,12,31,23,0,1, tz='Europe/Rome') # negative epoch
        self.assertEqual(time_unit.floor(datetime), dt(1969,12,31,23,0,0, tz='Europe/Rome'))
        self.assertEqual(time_unit.ceil(datetime), dt(1970,1,1,2,0, tz='Europe/Rome'))

        # Test ceil/floor/round across 1970-1-1 (together with the 2 hours-test, TODO: decouple)
        time_unit = TimeUnit('2h')
        datetime1 = dt(1969,12,31,22,59,59, tz='Europe/Rome') # negative epoch
        datetime2 = dt(1969,12,31,23,0,1, tz='Europe/Rome') # negative epoch
        self.assertEqual(time_unit.floor(datetime1), dt(1969,12,31,22,0,0, tz='Europe/Rome'))
        self.assertEqual(time_unit.ceil(datetime1), dt(1970,1,1,0,0, tz='Europe/Rome'))
        self.assertEqual(time_unit.round(datetime1), dt(1969,12,31,22,0, tz='Europe/Rome'))
        self.assertEqual(time_unit.round(datetime2), dt(1970,1,1,0,0, tz='Europe/Rome'))

        # Test ceil/floor/round across DST change (hours)
        time_unit = TimeUnit('1h')

        datetime1 = dt(2015,10,25,0,15,0, tz='Europe/Rome')
        datetime2 = datetime1 + time_unit    # 2015-10-25 01:15:00+02:00
        datetime3 = datetime2 + time_unit    # 2015-10-25 02:15:00+02:00
        datetime4 = datetime3 + time_unit    # 2015-10-25 02:15:00+01:00

        datetime1_rounded = dt(2015,10,25,0,0,0, tz='Europe/Rome')
        datetime2_rounded = datetime1_rounded + time_unit
        datetime3_rounded = datetime2_rounded + time_unit
        datetime4_rounded = datetime3_rounded + time_unit
        datetime5_rounded = datetime4_rounded + time_unit

        self.assertEqual(time_unit.floor(datetime2), datetime2_rounded)
        self.assertEqual(time_unit.ceil(datetime2), datetime3_rounded)

        self.assertEqual(time_unit.floor(datetime3), datetime3_rounded)
        self.assertEqual(time_unit.ceil(datetime3), datetime4_rounded)

        self.assertEqual(time_unit.floor(datetime4), datetime4_rounded)
        self.assertEqual(time_unit.ceil(datetime4), datetime5_rounded)

        # Test ceil/floor/round with a calendar time unit and across a DST change

        # Day unit
        time_unit = TimeUnit('1D')

        datetime1 = dt(2015,10,25,4,15,34, tz='Europe/Rome') # DST off (+01:00)
        datetime1_floor = dt(2015,10,25,0,0,0, tz='Europe/Rome') # DST on (+02:00)
        datetime1_ceil = dt(2015,10,26,0,0,0, tz='Europe/Rome') # DST off (+01:00)

        self.assertEqual(time_unit.floor(datetime1), datetime1_floor)
        self.assertEqual(time_unit.ceil(datetime1), datetime1_ceil)

        # Week unit
        time_unit = TimeUnit('1W')

        datetime1 = dt(2023,10,29,15,47, tz='Europe/Rome') # DST off (+01:00)
        datetime1_floor = dt(2023,10,23,0,0, tz='Europe/Rome') # DST on (+02:00)
        datetime1_ceil = dt(2023,10,30,0,0, tz='Europe/Rome') # DST off (+01:00)

        self.assertEqual(time_unit.floor(datetime1), datetime1_floor)
        self.assertEqual(time_unit.ceil(datetime1), datetime1_ceil)

        # Month unit
        time_unit = TimeUnit('1M')

        datetime1 = dt(2015,10,25,4,15,34, tz='Europe/Rome') # DST off (+01:00)
        datetime1_floor = dt(2015,10,1,0,0,0, tz='Europe/Rome') # DST on (+02:00)
        datetime1_ceil = dt(2015,11,1,0,0,0, tz='Europe/Rome') # DST off (+01:00)

        self.assertEqual(time_unit.floor(datetime1), datetime1_floor)
        self.assertEqual(time_unit.ceil(datetime1), datetime1_ceil)

        # Year unit
        time_unit = TimeUnit('1Y')

        datetime1 = dt(2015,10,25,4,15,34, tz='Europe/Rome')
        datetime1_floor = dt(2015,1,1,0,0,0, tz='Europe/Rome')
        datetime1_ceil = dt(2016,1,1,0,0,0, tz='Europe/Rome')

        self.assertEqual(time_unit.floor(datetime1), datetime1_floor)
        self.assertEqual(time_unit.ceil(datetime1), datetime1_ceil)


