import unittest
import datetime
from ..exceptions import InputException
from ..time import dt, TimeSpan, correct_dt_dst, timezonize, s_from_dt, dt_to_str, dt_from_str, change_tz


class TestTime(unittest.TestCase):

    def setUp(self):       
        pass

    def test_correct_dt_dst(self):
        
        # 2015-03-29 01:15:00+01:00
        dateTime =  dt(2015,3,29,1,15,0, tzinfo='Europe/Rome')
        
        # 2015-03-29 02:15:00+01:00, Wrong, does not exists and cannot be corrected
        dateTime = dateTime + datetime.timedelta(hours=1)
        with self.assertRaises(InputException):
            correct_dt_dst(dateTime)

        # 2015-03-29 03:15:00+01:00, Wrong and can be corrected
        dateTime = dateTime + datetime.timedelta(hours=1)
        self.assertEqual(correct_dt_dst(dateTime), dt(2015,3,29,3,15,0, tzinfo='Europe/Rome'))


    def test_dt(self):
        
        # TODO: understand if it is fine to test with string representation. Can we use native datetime?
        
        # Test UTC
        dateTime = dt(2015,3,29,2,15,0, tzinfo='UTC')
        self.assertEqual(str(dateTime), '2015-03-29 02:15:00+00:00')
        
        # Test UTC forced
        dateTime = dt(2015,3,29,2,15,0)
        self.assertEqual(str(dateTime), '2015-03-29 02:15:00+00:00')
        
        # Test with  timezone
        dateTime = dt(2015,3,25,4,15,0, tzinfo='Europe/Rome')
        self.assertEqual(str(dateTime), '2015-03-25 04:15:00+01:00')
        dateTime = dt(2015,9,25,4,15,0, tzinfo='Europe/Rome')
        self.assertEqual(str(dateTime), '2015-09-25 04:15:00+02:00')

        #---------------------
        # Test border cases
        #---------------------
        
        # Not existent time raises
        with self.assertRaises(InputException):
            _ = dt(2015,3,29,2,15,0, tzinfo='Europe/Rome')
         
        # Not existent time does not raises   
        dateTime = dt(2015,3,29,2,15,0, tzinfo='Europe/Rome', trustme=True)
        self.assertEqual(dateTime.year, 2015)
        self.assertEqual(dateTime.month, 3)
        self.assertEqual(dateTime.day, 29)
        self.assertEqual(dateTime.hour, 2)
        self.assertEqual(dateTime.minute, 15)
        self.assertEqual(str(dateTime.tzinfo), 'Europe/Rome')

        # Very past years (no DST and messy timezones)
        dateTime = dt(1856,12,1,16,46, tzinfo='Europe/Rome')
        self.assertEqual(str(dateTime), '1856-12-01 16:46:00+00:50')

        # NOTE: with pytz releases before ~ 2015.4 it ws as follows:
        #self.assertEqual(str(dateTime), '1856-12-01 16:46:00+01:00')

        dateTime = dt(1926,12,1,16,46, tzinfo='Europe/Rome')
        self.assertEqual(str(dateTime), '1926-12-01 16:46:00+01:00')

        dateTime = dt(1926,8,1,16,46, tzinfo='Europe/Rome')
        self.assertEqual(str(dateTime), '1926-08-01 16:46:00+01:00')
        
        # Very future years (no DST
        dateTime = dt(3567,12,1,16,46, tzinfo='Europe/Rome')
        self.assertEqual(str(dateTime), '3567-12-01 16:46:00+01:00')

        dateTime = dt(3567,8,1,16,46, tzinfo='Europe/Rome')
        self.assertEqual(str(dateTime), '3567-08-01 16:46:00+01:00')


    def test_str_conversions(self):

        # Note: there is no way to reconstruct the original timezone from an ISO time format. It has to be set separately.

        # To ISO

        # To ISO on UTC, offset is 0.
        self.assertEqual(dt_to_str(dt(1986,8,1,16,46, tzinfo='UTC')), '1986-08-01T16:46:00+00:00')

        # To ISO on Europe/Rome, without DST, offset is +1. 
        self.assertEqual(dt_to_str(dt(1986,12,1,16,46, tzinfo='Europe/Rome')), '1986-12-01T16:46:00+01:00')

        # To ISO On Europe/Rome, with DST, offset is +2.        
        self.assertEqual(dt_to_str(dt(1986,8,1,16,46, tzinfo='Europe/Rome')), '1986-08-01T16:46:00+02:00')

        # From ISO
        # 2016-06-29T19:36:29.3453Z
        # 2016-06-29T19:36:29.3453-0400
        # 2016-06-29T20:56:35.450686+05:00

        # From ISO without offset is not allowed -> Not anymore, see test "From ISO assuming UTC"]
        #with self.assertRaises(InputException):
        #    dt_from_str('1986-08-01T16:46:00')
       
        # From ISO on UTC
        self.assertEqual(str(dt_from_str('1986-08-01T16:46:00Z')), '1986-08-01 16:46:00+00:00')

        # From ISO assuming UTC
        self.assertEqual(str(dt_from_str('1986-08-01T16:46:00')), '1986-08-01 16:46:00+00:00')

        # From ISO on offset +02:00
        self.assertEqual(str(dt_from_str('1986-08-01T16:46:00.362752+02:00')), '1986-08-01 16:46:00.362752+02:00')
        
        # From ISO on offset +02:00 (with microseconds)
        self.assertEqual(str(dt_from_str('1986-08-01T16:46:00+02:00')), '1986-08-01 16:46:00+02:00')

        # From ISO on offset -07:00
        self.assertEqual(str(dt_from_str('1986-08-01T16:46:00-07:00')), '1986-08-01 16:46:00-07:00')


    def test_change_tz(self):
        self.assertEqual(str(change_tz(dt_from_str('1986-08-01T16:46:00.362752+02:00'), 'UTC')), '1986-08-01 14:46:00.362752+00:00')

    def test_TimeSpan(self):
        
        # TODO: I had to comment out this test, find out why..
        # Complex time intervals are not supported
        #with self.assertRaises(InputException):
        #   _ = TimeSpan('15m', '20s')
        
        # TODO: test with spans
        #print TimeSpan('1m').value
        #print TimeSlot(span=TimeSpan('1m')).span.value

        # Not valid 'q' type
        with self.assertRaises(InputException):
            _ = TimeSpan('15q')

        # String init
        timeSpan1 = TimeSpan('15m')
        self.assertEqual(timeSpan1.string, '15m')

        timeSpan2 = TimeSpan('15m_30s_3u')
        self.assertEqual(timeSpan2.string, '15m_30s_3u')
        
        timeSpan3 = TimeSpan(days=1)
        
        # Sum with other TimeSpan objects
        self.assertEqual((timeSpan1+timeSpan2+timeSpan3).string, '1D_30m_30s_3u')

        # Sum with datetime (also on DST change)
        timeSpan = TimeSpan('1h')
        dateTime1 = dt(2015,10,25,0,15,0, tzinfo='Europe/Rome')
        dateTime2 = dateTime1 + timeSpan
        dateTime3 = dateTime2 + timeSpan
        dateTime4 = dateTime3 + timeSpan
        dateTime5 = dateTime4 + timeSpan

        self.assertEqual(str(dateTime1), '2015-10-25 00:15:00+02:00')
        self.assertEqual(str(dateTime2), '2015-10-25 01:15:00+02:00')
        self.assertEqual(str(dateTime3), '2015-10-25 02:15:00+02:00')
        self.assertEqual(str(dateTime4), '2015-10-25 02:15:00+01:00')
        self.assertEqual(str(dateTime5), '2015-10-25 03:15:00+01:00')

        # Test duration
        self.assertEqual(TimeSpan('15m').duration, 900)
        self.assertEqual(TimeSpan('1h').duration, 3600)

        with self.assertRaises(Exception):
            TimeSpan('15D').duration

        # Test type
        self.assertEqual(TimeSpan('15m').type, TimeSpan.PHYSICAL)
        self.assertEqual(TimeSpan('1h').type, TimeSpan.PHYSICAL)
        self.assertEqual(TimeSpan('1D').type, TimeSpan.LOGICAL)
        self.assertEqual(TimeSpan('1M').type, TimeSpan.LOGICAL)


    def test_dt_math(self):

        # Test that complex timeSpans are not handable
        timeSpan = TimeSpan('1D_3h_5m')
        dateTime = dt(2015,1,1,16,37,14, tzinfo='Europe/Rome')
        
        with self.assertRaises(InputException):
            _ = timeSpan.floor_dt(dateTime)


        # Test in ceil/floor/round normal conditions (hours)
        timeSpan = TimeSpan('1h')
        dateTime = dt(2015,1,1,16,37,14, tzinfo='Europe/Rome')
        self.assertEqual(timeSpan.floor_dt(dateTime), dt(2015,1,1,16,0,0, tzinfo='Europe/Rome'))
        self.assertEqual(timeSpan.ceil_dt(dateTime), dt(2015,1,1,17,0,0, tzinfo='Europe/Rome'))

         
        # Test in ceil/floor/round normal conditions (minutes)
        timeSpan = TimeSpan('15m')
        dateTime = dt(2015,1,1,16,37,14, tzinfo='Europe/Rome')
        self.assertEqual(timeSpan.floor_dt(dateTime), dt(2015,1,1,16,30,0, tzinfo='Europe/Rome'))
        self.assertEqual(timeSpan.ceil_dt(dateTime), dt(2015,1,1,16,45,0, tzinfo='Europe/Rome'))


        # Test ceil/floor/round in normal conditions (seconds)
        timeSpan = TimeSpan('30s')
        dateTime = dt(2015,1,1,16,37,14, tzinfo='Europe/Rome') 
        self.assertEqual(timeSpan.floor_dt(dateTime), dt(2015,1,1,16,37,0, tzinfo='Europe/Rome'))
        self.assertEqual(timeSpan.ceil_dt(dateTime), dt(2015,1,1,16,37,30, tzinfo='Europe/Rome'))

   
        # Test ceil/floor/round across 1970-1-1 (minutes) 
        timeSpan = TimeSpan('5m')
        dateTime1 = dt(1969,12,31,23,57,29, tzinfo='UTC') # epoch = -3601
        dateTime2 = dt(1969,12,31,23,59,59, tzinfo='UTC') # epoch = -3601       
        self.assertEqual(timeSpan.floor_dt(dateTime1), dt(1969,12,31,23,55,0, tzinfo='UTC'))
        self.assertEqual(timeSpan.ceil_dt(dateTime1), dt(1970,1,1,0,0, tzinfo='UTC'))
        self.assertEqual(timeSpan.round_dt(dateTime1), dt(1969,12,31,23,55,0, tzinfo='UTC'))
        self.assertEqual(timeSpan.round_dt(dateTime2), dt(1970,1,1,0,0, tzinfo='UTC'))


        # Test ceil/floor/round (3 hours-test)
        timeSpan = TimeSpan('3h')
        dateTime = dt(1969,12,31,22,0,1, tzinfo='Europe/Rome') # negative epoch
        
        # TODO: test fails!! fix me!        
        #self.assertEqual(timeSpan.floor_dt(dateTime1), dt(1969,12,31,21,0,0, tzinfo='Europe/Rome'))
        #self.assertEqual(timeSpan.ceil_dt(dateTime1), dt(1970,1,1,0,0, tzinfo='Europe/Rome'))


        # Test ceil/floor/round across 1970-1-1 (together with the 2 hours-test, TODO: decouple) 
        timeSpan = TimeSpan('2h')
        dateTime1 = dt(1969,12,31,22,59,59, tzinfo='Europe/Rome') # negative epoch
        dateTime2 = dt(1969,12,31,23,0,1, tzinfo='Europe/Rome') # negative epoch  
        self.assertEqual(timeSpan.floor_dt(dateTime1), dt(1969,12,31,22,0,0, tzinfo='Europe/Rome'))
        self.assertEqual(timeSpan.ceil_dt(dateTime1), dt(1970,1,1,0,0, tzinfo='Europe/Rome'))
        self.assertEqual(timeSpan.round_dt(dateTime1), dt(1969,12,31,22,0, tzinfo='Europe/Rome'))
        self.assertEqual(timeSpan.round_dt(dateTime2), dt(1970,1,1,0,0, tzinfo='Europe/Rome'))

        # Test ceil/floor/round across DST change (hours)
        timeSpan = TimeSpan('1h')
        
        dateTime1 = dt(2015,10,25,0,15,0, tzinfo='Europe/Rome')
        dateTime2 = dateTime1 + timeSpan    # 2015-10-25 01:15:00+02:00    
        dateTime3 = dateTime2 + timeSpan    # 2015-10-25 02:15:00+02:00
        dateTime4 = dateTime3 + timeSpan    # 2015-10-25 02:15:00+01:00

        dateTime1_rounded = dt(2015,10,25,0,0,0, tzinfo='Europe/Rome')
        dateTime2_rounded = dateTime1_rounded + timeSpan        
        dateTime3_rounded = dateTime2_rounded + timeSpan
        dateTime4_rounded = dateTime3_rounded + timeSpan
        dateTime5_rounded = dateTime4_rounded + timeSpan
               
        self.assertEqual(timeSpan.floor_dt(dateTime2), dateTime2_rounded)
        self.assertEqual(timeSpan.ceil_dt(dateTime2), dateTime3_rounded)
          
        self.assertEqual(timeSpan.floor_dt(dateTime3), dateTime3_rounded)
        self.assertEqual(timeSpan.ceil_dt(dateTime3), dateTime4_rounded)
        
        self.assertEqual(timeSpan.floor_dt(dateTime4), dateTime4_rounded)
        self.assertEqual(timeSpan.ceil_dt(dateTime4), dateTime5_rounded)


        # Test ceil/floor/round with a logical timespan
        #timeSpan = TimeSpan('1D')
        
        #dateTime1 = dt(2015,10,25,0,15,0, tzinfo='Europe/Rome')
        #dateTime1_floor = dt(2015,10,25,0,0,0, tzinfo='Europe/Rome')

    def test_shift_dt(self):
        
        # TODO
        pass
        
        #timeSpan = TimeSpan('15m')
        #dateTime = dt(2015,1,1,16,37,14, tzinfo='Europe/Rome')
        #dateTime_rounded = dt(2015,1,1,16,0,0, tzinfo='Europe/Rome')
        
        #print shift_dt(dateTime, timeSpan, 4)
        #print shift_dt(dateTime, timeSpan, -2)

        #print shift_dt(dateTime_rounded, timeSpan, 4)
        #print shift_dt(dateTime_rounded, timeSpan, -2)

        # Test shift across DST change

    def tearDown(self):
        pass























