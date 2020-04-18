import datetime, calendar, pytz
from dateutil.tz import tzoffset
from .exceptions import InputException, ConsistencyException

# Setup logging
import logging
logger = logging.getLogger(__name__)

UTC = pytz.UTC

def timezonize(timezone):
    '''Convert a string representation of a timezone to its pytz object or do nothing if the argument is already a pytz timezone'''
    
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
    '''Return the current time in epoch seconds'''
    return calendar.timegm(now_dt().utctimetuple())


def now_dt(tzinfo='UTC'):
    '''Return the current time in datetime format'''
    if tzinfo != 'UTC':
        raise NotImplementedError()
    return datetime.datetime.utcnow().replace(tzinfo = pytz.utc)

def dt(*args, **kwargs):
    '''Initialize a datetime object in the proper way. Using the standard datetime leads to a lot of
     problems with the tz package. Also, it forces UTC timezone if no timezone is specified'''
    
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
        if not tzoffset:
            raise Exception('For ISO date with offset please install dateutil')
        time_dt = datetime.datetime(*args, tzinfo=tzoffset(None, offset_s))
    else:
        # Standard  timezone
        time_dt = timezone.localize(datetime.datetime(*args))

    # Check consistency    
    if not trustme and timezone != pytz.UTC:
        if not check_dt_consistency(time_dt):
            raise InputException('Sorry, time {} does not exists on timezone {}'.format(time_dt, timezone))

    return  time_dt


def get_tz_offset_s(time_dt):
    '''Get the time zone offset in seconds'''
    return s_from_dt(time_dt.replace(tzinfo=pytz.UTC)) - s_from_dt(time_dt)


def check_dt_consistency(date_dt):
    '''Check that the timezone is consistent with the datetime (some conditions in Python lead to have summertime set in winter)'''

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
    '''Check that the dst is correct and if not change it'''

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
    return dt.astimezone(timezonize(tz))


def dt_from_s(timestamp_s, tz=None):
    '''Create a datetime object from an epoch timestamp in seconds. If no timezone is given, UTC is assumed'''

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
    '''Returns seconds with floating point for milliseconds/microseconds.'''
    if not (isinstance(dt, datetime.datetime)):
        raise Exception('t_from_dt function called without datetime argument, got type "{}" instead.'.format(dt.__class__.__name__))
    microseconds_part = (dt.microsecond/1000000.0) if dt.microsecond else 0
    return  ( calendar.timegm(dt.utctimetuple()) + microseconds_part)


def dt_from_str(string, timezone=None):

    # This is a basic IS08601, see https://www.w3.org/TR/NOTE-datetime

    # Supported formats on UTC
    # 1) YYYY-MM-DDThh:mm:ssZ
    # 2) YYYY-MM-DDThh:mm:ss.{u}Z

    # Supported formats with offset    
    # 3) YYYY-MM-DDThh:mm:ss+ZZ:ZZ
    # 4) YYYY-MM-DDThh:mm:ss.{u}+ZZ:ZZ

    # Also:
    # 5) YYYY-MM-DDThh:mm:ss (without the trailing Z, and assume it on UTC)

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
    '''Return the ISO representation of the datetime as argument'''
    return dt.isoformat()


class dt_range(object):

    def __init__(self, from_dt, to_dt, timeSlotSpan):

        self.from_dt      = from_dt
        self.to_dt        = to_dt
        self.timeSlotSpan = timeSlotSpan

    def __iter__(self):
        self.current_dt = self.from_dt
        return self

    def __next__(self):

        # Iterator logic
        if self.current_dt > self.to_dt:
            raise StopIteration
        else:
            prev_current_dt = self.current_dt
            self.current_dt = self.current_dt + self.timeSlotSpan
            return prev_current_dt


class TimeSpan(object):
    ''' A time span is a duration. It can be physical (hours, minutes..),
    logical (months, for example) or defined from a start to an end'''
    
    LOGICAL  = 'Logical'
    PHYSICAL = 'Phyisical'
    
    # NOT ref to https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes :  %d, %m, %w %y - %H, %M, %S
    # Instead: M, D, Y, W - h m s
    
    mapping_table = {  'Y': 'years',
                       'M': 'months',
                       'W': 'weeks',
                       'D': 'days',
                       'h': 'hours',
                       'm': 'minutes',
                       's': 'seconds',
                       'u': 'microseconds'
                      }

    def __init__(self, string=None, years=0, weeks=0, months=0, days=0, hours=0, minutes=0, seconds=0, microseconds=0, start=None, end=None, trustme=False):

        if not trustme:

            # String OR explicit OR start/end
            if string and (years or months or days or hours or minutes or seconds or microseconds) and ((start is not None) or (end is not None)):
                raise InputException('Choose between string init and explicit setting of years,months, days, hours etc.')
    
            # Check that both start and end are set if one is set
            if start and not end:
                raise InputException('You provided the start but not the end')
            if end and not start:
                raise InputException('You provided the end but not the start')
                   
        # Set the TimeSlotSpan in seconds
        if start and end:
            seconds = s_from_dt((end-start).dt)

        self.years        = years
        self.months       = months
        self.weeks        = weeks
        self.days         = days
        self.hours        = hours
        self.minutes      = minutes 
        self.seconds      = seconds
        self.microseconds = microseconds

        if string:  
            self.strings = string.split("_")
            #if len (self.strings) > 1:
            #    raise InputException('Complex time intervals are not yet supported')

            import re
            regex = re.compile('^([0-9]+)([YMDWhmsu]{1,2})$')
            
            for string in self.strings:
                try:
                    groups   =  regex.match(string).groups()
                except AttributeError:
                    raise InputException('Cannot parse string representation for the TimeSlotSpan, unknown  format ("{}")'.format(string)) from None

                setattr(self, self.mapping_table[groups[1]], int(groups[0]))

        if not trustme:  
                     
            # If nothing set, raise error
            if not self.years and not self.weeks and not self.months and not self.days and not self.hours and not self.minutes and not self.seconds and not self.microseconds:
                raise InputException('Detected zero-length TimeSlotSpan!')
 

    # Representation..
    def __repr__(self):
        return self.string

    # Operations..
    def __add__(self, other):
        
        if isinstance(other, self.__class__):
            
            return TimeSpan(years        = self.years + other.years,
                            months       = self.months + other.months,
                            weeks        = self.weeks + other.weeks,
                            days         = self.days + other.days,
                            hours        = self.hours + other.hours,
                            minutes      = self.minutes + other.minutes,
                            seconds      = self.seconds + other.seconds,
                            microseconds = self.microseconds + other.microseconds)

        elif isinstance(other, datetime.datetime):
            
            if not other.tzinfo:
                raise InputException('Timezone of the datetime to sum with is required')

            return self.shift_dt(other, times=1)
        else:
            raise NotImplementedError

    
    def __radd__(self, other):
        return self.__add__(other)

    @property
    def string(self):

        string = ''
        if self.years: string += str(self.years)               + 'Y' + '_'
        if self.months: string += str(self.months)             + 'M' + '_'
        if self.weeks: string += str(self.weeks)               + 'W' + '_'
        if self.days: string += str(self.days)                 + 'D' + '_'
        if self.hours: string += str(self.hours)               + 'h' + '_'
        if self.minutes: string += str(self.minutes)           + 'm' + '_'
        if self.seconds: string += str(self.seconds)           + 's' + '_'
        if self.microseconds: string += str(self.microseconds) + 'u' + '_'

        string = string[:-1]
        return string

    def is_composite(self):
        
        types = 0
        for item in self.mapping_table:
            if getattr(self, self.mapping_table[item]): types +=1
        
        return True if types > 1 else False 

    @property
    def type(self):
        '''Returns the type of the TimeSlotSpan.
         - Physical if hours, minutes, seconds, micorseconds
         - Logical if Years, Months, Days, Weeks
        The difference? Years, Months, Days, Weeks have different lengths depending on the starting date.
        '''

        if self.years or self.months or self.weeks or self.days:
            return self.LOGICAL
        elif self.hours or self.minutes or self.seconds or self.microseconds:
            return self.PHYSICAL
        else:
            raise ConsistencyException('Error, TimeSlot not initialized?!')
    
    def is_physical(self):
        if self.type == self.PHYSICAL:
            return True
        else:
            return False
        
    def is_logical(self):
        if self.type == self.LOGICAL:
            return True 
        else:       
            return False        

    def round_dt(self, time_dt, how = None):
        '''Round a datetime according to this TimeSlotSpan. Only simple time intervals are supported in this operation'''

        if self.is_composite():
            raise InputException('Sorry, only simple time intervals are supported by the rebase operation')

        if not time_dt.tzinfo:
            raise InputException('Timezone of the datetime is required')    
        
        # Convert input time to seconds
        time_s = s_from_dt(time_dt)
        tz_offset_s = get_tz_offset_s(time_dt)
        

        # Handle physical time 
        if self.type == self.PHYSICAL:
            
            # Get TimeSlotSpan duration in seconds
            timeSpan_s = self.duration_s(time_dt)

            # Apply modular math (including timezone time translation trick if required (multiple hours))
            # TODO: check for correctness, the time shift should be always done...
            
            if self.hours > 1 or self.minutes > 60:
                time_floor_s = ( (time_s - tz_offset_s) - ( (time_s - tz_offset_s) % timeSpan_s) ) + tz_offset_s
            else:
                time_floor_s = time_s - (time_s % timeSpan_s)
                
            time_ceil_s   = time_floor_s + timeSpan_s
            
            if how == 'floor':
                time_rounded_s = time_floor_s
             
            elif how == 'ceil':
                time_rounded_s = time_ceil_s
            
            else:

                distance_from_time_floor_s = abs(time_s - time_floor_s) # Distance from floot
                distance_from_time_ceil_s  = abs(time_s - time_ceil_s)  # Distance from ceil

                if distance_from_time_floor_s < distance_from_time_ceil_s:
                    time_rounded_s = time_floor_s
                else:
                    time_rounded_s = time_ceil_s

        # Handle logical time 
        elif self.type == self.LOGICAL:
            
            
            # Get TimeSlotSpan duration in seconds
            timeSpan_s = self.duration_s(time_dt)
            
            logger.debug(timeSpan_s)
            
            #time_rounded_s
            
            
            
            
            
            raise NotImplementedError('Logical not yet implemented')

        # Handle other cases (Consistency error)
        else:
            raise ConsistencyException('Error, TimeSlot type not PHYSICAL nor LOGICAL?!')

        return dt_from_s(time_rounded_s, tz=time_dt.tzinfo)


    def floor_dt(self, time_dt):
        '''Floor a datetime according to this TimeSlotSpan. Only simple time intervals are supported in this operation'''       
        return self.round_dt(time_dt, how='floor')
     
    def ceil_dt(self, time_dt):
        '''Ceil a datetime according to this TimeSlotSpan. Only simple time intervals are supported in this operation'''        
        return self.round_dt(time_dt, how='ceil')

    def rebase_dt(self, time_dt):
        '''Rebase a given datetime to this TimeSlotSpan. Only simple time intervals are supported in this operation'''
        return self.round_dt(time_dt, how='floor')
              
    def shift_dt(self, time_dt, times=0):
        '''Shift a given datetime of n times of this TimeSlotSpan. Only simple time intervals are supported in this operation'''
        if self.is_composite():
            raise InputException('Sorry, only simple time intervals are supported byt he rebase operation')
 
        # Convert input time to seconds
        time_s = s_from_dt(time_dt)
         
        # Handle physical time TimeSlot
        if self.type == self.PHYSICAL:
            
            # Get TimeSlotSpan duration in seconds
            timeSpan_s = self.duration_s(time_dt)

            time_shifted_s = time_s + ( timeSpan_s * times )
            time_shifted_dt = dt_from_s(time_shifted_s, tz=time_dt.tzinfo)

        # Handle logical time TimeSlot
        elif self.type == self.LOGICAL:
            raise NotImplementedError('Shifting of Logical intervals not yet implemented')

        # Handle other cases (Consistency error)
        else:
            raise ConsistencyException('Error, TimeSlot type not PHYSICAL nor LOGICAL?!')
 
        return time_shifted_dt      

    def duration_s(self, start_dt=None):
        '''Get the duration of the interval in seconds'''
        if self.is_composite():
            raise InputException('Sorry, only simple time intervals are supported by this operation')

        if self.type == 'Logical' and not start_dt:
            raise InputException('With a logical TimeSlotSpan you can ask for duration only if you provide the starting point')
        
        if self.type == 'Logical':
            raise NotImplementedError('Computing the duration in seconds using a given start_time_dt is not yet supported')

        else:
            # Hours, Minutes, Seconds
            if self.hours:
                timeSpan_s = self.hours * 60 * 60
            if self.minutes:
                timeSpan_s = self.minutes * 60
            if self.seconds:
                timeSpan_s = self.seconds
            if self.microseconds:
                timeSpan_s = 1/1000000.0 * self.microseconds
               
        return timeSpan_s
        
    @property
    def duration(self):
        if self.type == 'Logical':
            raise InputException('Sorry, the duration of a LOGICAL time span is not defined. use duration_s() providing the starting point.')
        return self.duration_s()


    # Get start/end/center
    def get_start(self, end=None, center=None):
        new_values = []
        if end is not None:
            for i in range(len(self.value)):
                new_values.append(end.values[i] - self.value[i])
            return end.__class__(labels=end.labels, values=new_values, tz=end.tz)        
        elif center is not None:
            for i in range(len(self.value)):
                new_values.append(center.values[i] - self.value[i]/2.0)
            return center.__class__(labels=center.labels, values=new_values, tz=center.tz)
        else:
            raise InputException('get_start: Got not end nor center')        
            
    def get_end(self, start=None, center=None):
        new_values = []
        if start is not None:
            for i in range(len(self.value)):
                new_values.append(self.value[i] + start.values[i])
            return start.__class__(labels=start.labels, values=new_values, tz=start.tz)           
        elif center is not None:
            for i in range(len(self.value)):
                new_values.append(center.values[i] + self.value[i]/2.0) 
            return center.__class__(labels=center.labels, values=new_values, tz=center.tz) 
        else:
            raise InputException('get_end: Got not end nor center')

    def get_center(self, start=None, end=None):
        new_values = []
        if start is not None:
            for i in range(len(self.value)):
                new_values.append(start.values[i] + self.value[i]/2.0)
            return start.__class__(labels=start.labels, values=new_values, tz=start.tz)           
        elif end is not None:
            for i in range(len(self.value)):
                new_values.append(end.values[i] - self.value[i]/2.0) 
            return end.__class__(labels=end.labels, values=new_values, tz=end.tz) 
        else:
            raise InputException('get_center: Got not end not start') 



