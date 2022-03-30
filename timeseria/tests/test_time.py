import unittest
import datetime
import pytz
from ..time import dt, correct_dt_dst, dt_to_str, dt_from_str, s_from_dt, dt_from_s, as_timezone, timezonize
from pandas import Timestamp as PandasTimestamp

# Setup logging
from .. import logger
logger.setup()

class TestTime(unittest.TestCase):

    def setUp(self):       
        pass

    def test_correct_dt_dst(self):
        
        # 2015-03-29 01:15:00+01:00
        date_time =  dt(2015,3,29,1,15,0, tzinfo='Europe/Rome')
        
        # 2015-03-29 02:15:00+01:00, Wrong, does not exists and cannot be corrected
        date_time = date_time + datetime.timedelta(hours=1)
        with self.assertRaises(ValueError):
            correct_dt_dst(date_time)

        # 2015-03-29 03:15:00+01:00, Wrong and can be corrected
        date_time = date_time + datetime.timedelta(hours=1)
        self.assertEqual(correct_dt_dst(date_time), dt(2015,3,29,3,15,0, tzinfo='Europe/Rome'))


    def test_dt(self):
        
        # TODO: understand if it is fine to test with string representation. Can we use native datetime?
        
        # Test UTC
        date_time = dt(2015,3,29,2,15,0, tzinfo='UTC')
        self.assertEqual(str(date_time), '2015-03-29 02:15:00+00:00')
        
        # Test UTC forced
        date_time = dt(2015,3,29,2,15,0)
        self.assertEqual(str(date_time), '2015-03-29 02:15:00+00:00')
        
        # Test with  timezone
        date_time = dt(2015,3,25,4,15,0, tzinfo='Europe/Rome')
        self.assertEqual(str(date_time), '2015-03-25 04:15:00+01:00')
        date_time = dt(2015,9,25,4,15,0, tzinfo='Europe/Rome')
        self.assertEqual(str(date_time), '2015-09-25 04:15:00+02:00')

        # Not existent time raises
        with self.assertRaises(ValueError):
            _ = dt(2015,3,29,2,15,0, tzinfo='Europe/Rome')
         
        # Not existent time does not raises   
        date_time = dt(2015,3,29,2,15,0, tzinfo='Europe/Rome', trustme=True)
        self.assertEqual(date_time.year, 2015)
        self.assertEqual(date_time.month, 3)
        self.assertEqual(date_time.day, 29)
        self.assertEqual(date_time.hour, 2)
        self.assertEqual(date_time.minute, 15)
        self.assertEqual(str(date_time.tzinfo), 'Europe/Rome')

        # Very past years (no DST and messy timezones)
        date_time = dt(1856,12,1,16,46, tzinfo='Europe/Rome')
        self.assertEqual(str(date_time), '1856-12-01 16:46:00+00:50')

        # NOTE: with pytz releases before ~ 2015.4 it ws as follows:
        #self.assertEqual(str(date_time), '1856-12-01 16:46:00+01:00')

        date_time = dt(1926,12,1,16,46, tzinfo='Europe/Rome')
        self.assertEqual(str(date_time), '1926-12-01 16:46:00+01:00')

        date_time = dt(1926,8,1,16,46, tzinfo='Europe/Rome')
        self.assertEqual(str(date_time), '1926-08-01 16:46:00+01:00')
        
        # Very future years (no DST
        date_time = dt(3567,12,1,16,46, tzinfo='Europe/Rome')
        self.assertEqual(str(date_time), '3567-12-01 16:46:00+01:00')

        date_time = dt(3567,8,1,16,46, tzinfo='Europe/Rome')
        self.assertEqual(str(date_time), '3567-08-01 16:46:00+01:00')

    
    def test_s_from_dt(self):
        
        date_time = dt(2001,12,1,16,46,10,6575, tzinfo='Europe/Rome')
        self.assertEqual(s_from_dt(date_time), 1007221570.006575)
        
        # With some combinations of Python/Pands versions i.e. Pandas 0.22.0  and Python 3.4.4 this is created as
        # 2001-12-01 16:56:10.006575+01:00 (note the minutes): it is a bug. Probably because of setting the timezone.
        date_time_pandas = PandasTimestamp(year=2001,month=12,day=1,hour=16,minute=46,second=10,microsecond=6575, tzinfo=timezonize('Europe/Rome'))
        self.assertEqual(s_from_dt(date_time_pandas), 1007221570.006575)


    def test_s_from_dt_naive(self):
        
        # Create a naive timestamp without the timezone and test
        date_time = datetime.datetime.strptime('2007-12-10', '%Y-%m-%d')
        self.assertEqual(s_from_dt(date_time),1197244800)
        
        # Now set the timezone and test
        date_time = date_time.replace(tzinfo = pytz.utc)
        self.assertEqual(s_from_dt(date_time),1197244800)
        
        # Note: some versiosn of Python 3 assume a naive datetime on local time
        # when using the .timestamp() or the .strftime('%s') methods.
        # For this timestamp: 1197244800.0 (UTC) vs 1197241200.0 (Europe/Rome)

    def test_dt_from_s(self):

        self.assertEqual(dt_from_s(1197244800), dt(2007,12,10,0,0, tz='UTC'))

        self.assertEqual(dt_from_s(1197241200, tz='Europe/Rome'), dt(2007,12,10,0,0, tz='Europe/Rome'))

        
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
        #with self.assertRaises(ValueError):
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


    def test_as_timezone(self):
        self.assertEqual(str(as_timezone(dt_from_str('1986-08-01T16:46:00.362752+02:00'), 'UTC')), '1986-08-01 14:46:00.362752+00:00')


    def tearDown(self):
        pass

