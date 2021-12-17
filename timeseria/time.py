# -*- coding: utf-8 -*-
"""Provides time manipulation utilities, with a particular focus on proper timezone handling."""

import datetime, calendar, pytz

# Setup logging
import logging
logger = logging.getLogger(__name__)

UTC = pytz.UTC

def timezonize(timezone):
    """Convert a string representation of a timezone to its pytz object, or do nothing if the argument is already a pytz timezone."""
    
    # Check if timezone is a valid pytz object is hard as it seems that they are spread arount the pytz package.
    # Option 1): Try to convert if string or unicode, else try to
    # instantiate a datetiem object with the timezone to see if it is valid 
    # Option 2): Get all memebers of the pytz package and check for type, see
    # http://stackoverflow.com/questions/14570802/python-check-if-object-is-instance-of-any-class-from-a-certain-module
    # Option 3) perform a hand.made test. We go for this one, tests would fail if it gets broken
    
    if not 'pytz' in str(type(timezone)):
        timezone = pytz.timezone(timezone)
  
    return timezone

def now_s():
    """Return the current time in epoch seconds."""
    return calendar.timegm(now_dt().utctimetuple())


def now_dt(tzinfo='UTC'):
    """Return the current time in datetime format."""
    if tzinfo != 'UTC':
        raise NotImplementedError()
    return datetime.datetime.utcnow().replace(tzinfo = pytz.utc)


def dt(*args, **kwargs):
    """Initialize a datetime object with the timezone in the proper way. Using the standard datetime initilization
    leads to various problems if setting a pytz timezone. Also, it forces UTC timezone if no timezone is specified"""
    
    if 'tz' in kwargs:
        tzinfo = kwargs.pop('tz')
    else:
        tzinfo  = kwargs.pop('tzinfo', None)
        
    offset_s  = kwargs.pop('offset_s', None)   
    trustme   = kwargs.pop('trustme', None)
    
    if kwargs:
        raise Exception('Unhandled arg: "{}".'.format(kwargs))
        
    if (tzinfo is None):
        # Force UTC if None
        timezone = timezonize('UTC')
        
    else:
        timezone = timezonize(tzinfo)
    
    if offset_s:
        # Special case for the offset
        from dateutil.tz import tzoffset
        if not tzoffset:
            raise Exception('For ISO date with offset please install dateutil')
        time_dt = datetime.datetime(*args, tzinfo=tzoffset(None, offset_s))
    else:
        # Standard  timezone
        time_dt = timezone.localize(datetime.datetime(*args))

    # Check consistency    
    if not trustme and timezone != pytz.UTC:
        if not check_dt_consistency(time_dt):
            raise ValueError('Sorry, time {} does not exists on timezone {}'.format(time_dt, timezone))

    return  time_dt


def get_tz_offset_s(time_dt):
    """Get the time zone offset in seconds."""
    return s_from_dt(time_dt.replace(tzinfo=pytz.UTC)) - s_from_dt(time_dt)


def check_dt_consistency(date_dt):
    """Check that the timezone is consistent with the datetime (some conditions in Python lead to have summertime set in winter)."""

    # https://en.wikipedia.org/wiki/Tz_database
    # https://www.iana.org/time-zones
    
    if date_dt.tzinfo is None:
        return True
    else:
        
        # This check is quite heavy but there is apparently no other way to do it.
        if date_dt.utcoffset() != dt_from_s(s_from_dt(date_dt), tz=date_dt.tzinfo).utcoffset():
            return False
        else:
            return True


def correct_dt_dst(datetime_obj):
    """Check that the dst is correct and if not, change it"""

    # https://en.wikipedia.org/wiki/Tz_database
    # https://www.iana.org/time-zones

    if datetime_obj.tzinfo is None:
        return datetime_obj

    # Create and return a New datetime object. This corrects the DST if errors are present.
    return dt(datetime_obj.year,
              datetime_obj.month,
              datetime_obj.day,
              datetime_obj.hour,
              datetime_obj.minute,
              datetime_obj.second,
              datetime_obj.microsecond,
              tzinfo=datetime_obj.tzinfo)

def change_tz(dt, tz):
    """Change the timezone of a datetime object."""
    return dt.astimezone(timezonize(tz))


def dt_from_s(timestamp_s, tz=None):
    """Create a datetime object from an epoch timestamp in seconds. If no timezone is given, UTC is assumed."""

    if not tz:
        tz = "UTC"
    try:
        timestamp_dt = datetime.datetime.utcfromtimestamp(float(timestamp_s))
    except TypeError:
        raise Exception('timestamp_s argument must be string or number, got {}'.format(type(timestamp_s)))

    pytz_tz = timezonize(tz)
    timestamp_dt = timestamp_dt.replace(tzinfo=pytz.utc).astimezone(pytz_tz)
    
    return timestamp_dt


def s_from_dt(dt):
    """Return the epoch seconds from a datetime object, with floating point for milliseconds/microseconds."""
    if not (isinstance(dt, datetime.datetime)):
        raise Exception('t_from_dt function called without datetime argument, got type "{}" instead.'.format(dt.__class__.__name__))
    try:
        return dt.timestamp()
    except AttributeError:
        # Python 2.7
        microseconds_part = (dt.microsecond/1000000.0) if dt.microsecond else 0
        return (calendar.timegm(dt.utctimetuple()) + microseconds_part)


def dt_from_str(string, timezone=None):
    """Create a datetime object from a string.

    This is a basic IS08601, see https://www.w3.org/TR/NOTE-datetime

    Supported formats on UTC:
        1) YYYY-MM-DDThh:mm:ssZ
        2) YYYY-MM-DDThh:mm:ss.{u}Z

    Supported formats with offset    
        3) YYYY-MM-DDThh:mm:ss+ZZ:ZZ
        4) YYYY-MM-DDThh:mm:ss.{u}+ZZ:ZZ

    Other supported formats:
        5) YYYY-MM-DDThh:mm:ss (without the trailing Z, and assume it on UTC)
    """

    # Split and parse standard part
    if 'T' in string:
        date, time = string.split('T')
    elif ' ' in string:
        date, time = string.split(' ')
    else:
        raise ValueError('Cannot find andy date/time separator (lookign for "T" or " " in "{}"'.format(string))
        
    
    if time.endswith('Z'):
        # UTC
        offset_s = 0
        time = time[:-1]
        
    elif ('+') in time:
        # Positive offset
        time, offset = time.split('+')
        # Set time and extract positive offset
        offset_s = (int(offset.split(':')[0])*60 + int(offset.split(':')[1]) )* 60   
        
    elif ('-') in time:
        # Negative offset
        time, offset = time.split('-')
        # Set time and extract negative offset
        offset_s = -1 * (int(offset.split(':')[0])*60 + int(offset.split(':')[1])) * 60      
    
    else:
        # Assume UTC
        offset_s = 0
    
    # Handle time
    hour, minute, second = time.split(':')
    
    # Now parse date (easy)
    year, month, day = date.split('-') 

    # Convert everything to int
    year    = int(year)
    month   = int(month)
    day     = int(day)
    hour    = int(hour)
    minute  = int(minute)
    if '.' in second:
        usecond = int(second.split('.')[1])
        second  = int(second.split('.')[0])
    else:
        second  = int(second)
        usecond = 0
    
    return dt(year, month, day, hour, minute, second, usecond, offset_s=offset_s)


def dt_to_str(dt):
    """Return the IS08601 representation of a datetime."""
    return dt.isoformat()


class dt_range(object):
    """A datetime range object."""

    def __init__(self, from_dt, to_dt, time_unit):

        self.from_dt   = from_dt
        self.to_dt     = to_dt
        self.time_unit = time_unit

    def __iter__(self):
        self.current_dt = self.from_dt
        return self

    def __next__(self):

        # Iterator logic
        if self.current_dt > self.to_dt:
            raise StopIteration
        else:
            prev_current_dt = self.current_dt
            self.current_dt = self.current_dt + self.time_unit
            return prev_current_dt

