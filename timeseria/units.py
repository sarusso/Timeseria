import datetime
import math
from .time import s_from_dt , dt_from_s, get_tz_offset_s
from .time import check_dt_consistency, correct_dt_dst
from .exceptions import InputException, ConsistencyException
            
# Setup logging
import logging
logger = logging.getLogger(__name__)


HARD_DEBUG = False


class Unit(object):
    
    def __init__(self, value):
        # TODO: Add support for creating from string repr of list/float?
        self.value = value
        
    def __repr__(self):   
        return str(self.value)

    def __add__(self, other):

        # Sum with int or float
        if (isinstance(self.value, int) or isinstance(self.value, float)) and (isinstance(other, int) or isinstance(other, float)):
            return self.value + other

        # Sum with another unit with value
        try:
            if (isinstance(self.value, int) or isinstance(self.value, float)) and (isinstance(other.value, int) or isinstance(other.value, float)):
                return self.value + other.value
        except:
            pass
        
        # Sum with an objects with coordinates (Points, SLots). TODO: maybe re-evaluate this part?
        try:
            if len(other.coordinates) > 1:
                raise NotImplementedError('Cannot add Units in a multidimensional space')             
            return other.__class__(self.value + other.coordinates[0])
        except:
            raise NotImplementedError('Don\'t know how to add Units with {}'.format(other.__class__.__name__)) 
    
    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):

        # Subtract with int or float
        if (isinstance(self.value, int) or isinstance(self.value, float)) and (isinstance(other, int) or isinstance(other, float)):
            return self.value - other

        # Subtract with another unit with value
        try:
            if (isinstance(self.value, int) or isinstance(self.value, float)) and (isinstance(other.value, int) or isinstance(other.value, float)):
                return self.value - other.value
        except:
            pass

        # Subtract with an objects with coordinates (Points, SLots). TODO: maybe re-evaluate this part?
        try:
            if len(other.coordinates) > 1:
                raise NotImplementedError('Cannot add Units in a multidimensional space')             
            return other.__class__(self.value - other.coordinates[0])
        except:
            raise NotImplementedError('Don\'t know how to add Units with {}'.format(other.__class__.__name__)) 

    def __rsub__(self, other):
        return -self.__sub__(other)

    def __eq__(self,other):
        if self is other:
            return True
        else:
            return self.value == other.value


class TimeUnit(Unit):
    ''' A time unit which can be both be physical (hours, minutes..),
    calendar (months, for example) or defined from a start to an end'''
    
    CALENDAR  = 'Calendar'
    PHYSICAL = 'Physical'
    
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

            if string and not isinstance(string, str):
                raise InputException('TimeUnits must be initialized with a string or explicitly setting years, months, days, hours etc. (Got "{}")'.format(string.__class__.__name__))

            # String OR explicit OR start/end
            if string and (years or months or days or hours or minutes or seconds or microseconds) and ((start is not None) or (end is not None)):
                raise InputException('Choose between string init and explicit setting of years, months, days, hours etc.')
    
            # Check types:
            if not isinstance(years, int): raise InputException('year not of type int (got "{}")'.format(years.__class__.__name__))
            if not isinstance(weeks, int): raise InputException('weeks not of type int (got "{}")'.format(years.__class__.__name__))
            if not isinstance(months, int): raise InputException('months not of type int (got "{}")'.format(years.__class__.__name__))
            if not isinstance(days, int): raise InputException('days not of type int (got "{}")'.format(years.__class__.__name__))
            if not isinstance(hours, int): raise InputException('hours not of type int (got "{}")'.format(years.__class__.__name__))
            if not isinstance(minutes, int): raise InputException('minutes not of type int (got "{}")'.format(years.__class__.__name__))
            if not isinstance(seconds, int): raise InputException('seconds not of type int (got "{}")'.format(years.__class__.__name__))
            if not isinstance(microseconds, int): raise InputException('microseconds not of type int (got "{}")'.format(years.__class__.__name__))

            # Check that both start and end are set if one is set
            if start and not end:
                raise InputException('You provided the start but not the end')
            if end and not start:
                raise InputException('You provided the end but not the start')
                   
        # Set the TimeUnit in seconds
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
                    raise InputException('Cannot parse string representation for the TimeUnit, unknown  format ("{}")'.format(string)) from None

                setattr(self, self.mapping_table[groups[1]], int(groups[0]))

        if not trustme:  
                     
            # If nothing set, raise error
            if not self.years and not self.weeks and not self.months and not self.days and not self.hours and not self.minutes and not self.seconds and not self.microseconds:
                raise InputException('Detected zero-duration TimeUnit!')
 
    # Representation..
    def __repr__(self):
        return self.string

    # Operations..
    def __add__(self, other):
        
        # TODO: This has to stay here or a circular import will start. Maybe refactor this?
        from .datastructures import TimePoint
        
        if isinstance(other, self.__class__):
            
            return TimeUnit(years        = self.years + other.years,
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
        
        
        elif isinstance(other, TimePoint):
            return TimePoint(dt=self.shift_dt(other.dt, times=1))
            
        
        else:
            raise NotImplementedError('Adding TimeUnits on top of objects other than datetimes is not yet supported')
   
    def __radd__(self, other):
        return self.__add__(other)

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        if self.years != other.years:
            return False
        if self.months != other.months:
            return False
        if self.weeks != other.weeks:
            return False
        if self.days != other.days:
            return False
        if self.hours != other.hours:
            return False
        if self.minutes != other.minutes:
            return False
        if self.seconds != other.seconds:
            return False
        if self.microseconds != other.microseconds:
            return False
        return True

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
        '''Returns the type of the TimeUnit.
         - Physical if hours, minutes, seconds, micorseconds
         - Calendar if Years, Months, Days, Weeks
        The difference? Years, Months, Days, Weeks have different durations depending on the starting date.
        '''

        if self.years or self.months or self.weeks or self.days:
            return self.CALENDAR
        elif self.hours or self.minutes or self.seconds or self.microseconds:
            return self.PHYSICAL
        else:
            raise ConsistencyException('Error, TimeSlot not initialized?!')
    
    def is_physical(self):
        if self.type == self.PHYSICAL:
            return True
        else:
            return False
        
    def is_calendar(self):
        if self.type == self.CALENDAR:
            return True 
        else:       
            return False        

    def round_dt(self, time_dt, how = None):
        '''Round a datetime according to this TimeUnit. Only simple time intervals are supported in this operation'''

        if self.is_composite():
            raise InputException('Sorry, only simple time intervals are supported by the rebase operation')

        if not time_dt.tzinfo:
            raise InputException('Timezone of the datetime is required')    
                
        # Handle physical time 
        if self.type == self.PHYSICAL:
        
            # Convert input time to seconds
            time_s = s_from_dt(time_dt)
            tz_offset_s = get_tz_offset_s(time_dt)
            
            # Get TimeUnit duration in seconds
            time_unit_s = self.duration_s(time_dt)

            # Apply modular math (including timezone time translation trick if required (multiple hours))
            # TODO: check for correctness, the time shift should be always done...
            
            if self.hours > 1 or self.minutes > 60:
                time_floor_s = ( (time_s - tz_offset_s) - ( (time_s - tz_offset_s) % time_unit_s) ) + tz_offset_s
            else:
                time_floor_s = time_s - (time_s % time_unit_s)
                
            time_ceil_s   = time_floor_s + time_unit_s
            
            if how == 'floor':
                time_rounded_s = time_floor_s
             
            elif how == 'ceil':
                time_rounded_s = time_ceil_s
            
            else:

                distance_from_time_floor_s = abs(time_s - time_floor_s) # Distance from floor
                distance_from_time_ceil_s  = abs(time_s - time_ceil_s)  # Distance from ceil

                if distance_from_time_floor_s < distance_from_time_ceil_s:
                    time_rounded_s = time_floor_s
                else:
                    time_rounded_s = time_ceil_s
                
            rounded_dt = dt_from_s(time_rounded_s, tz=time_dt.tzinfo)

        # Handle calendar time 
        elif self.type == self.CALENDAR:
            
            if self.years:
                if self.years > 1:
                    raise NotImplementedError('Cannot round based on CALENDAR TimeUnits > 1')
                rounded_dt=time_dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
                
            if self.months:
                if self.months > 1:
                    raise NotImplementedError('Cannot round based on CALENDAR TimeUnits > 1')
                rounded_dt=time_dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

            if self.weeks:
                raise NotImplementedError('Cannot round based on CALENDAR TimeUnits of week type')

            if self.days:
                if self.days > 1:
                    raise NotImplementedError('Cannot round based on CALENDAR TimeUnits > 1')
                rounded_dt=time_dt.replace(hour=0, minute=0, second=0, microsecond=0)

            # Check DST offset consistency and fix if not respected
            if not check_dt_consistency(rounded_dt):
                rounded_dt = correct_dt_dst(rounded_dt)
            
            if how == 'ceil':
                rounded_dt = self.shift_dt(rounded_dt, 1)

        # Handle other cases (Consistency error)
        else:
            raise ConsistencyException('Error, TimeSlot type not PHYSICAL nor CALENDAR?!')

        # Return
        return rounded_dt

    def floor_dt(self, time_dt):
        '''Floor a datetime according to this TimeUnit. Only simple time intervals are supported in this operation'''       
        return self.round_dt(time_dt, how='floor')
     
    def ceil_dt(self, time_dt):
        '''Ceil a datetime according to this TimeUnit. Only simple time intervals are supported in this operation'''        
        return self.round_dt(time_dt, how='ceil')

    def rebase_dt(self, time_dt):
        '''Rebase a given datetime to this TimeUnit. Only simple time intervals are supported in this operation'''
        return self.round_dt(time_dt, how='floor')
              
    def shift_dt(self, time_dt, times=0):
        '''Shift a given datetime of n times of this TimeUnit. Only simple time intervals are supported in this operation'''
        if self.is_composite():
            raise InputException('Sorry, only simple time intervals are supported byt he rebase operation')
 
        # Convert input time to seconds
        time_s = s_from_dt(time_dt)
         
        # Handle physical time TimeSlot
        if self.type == self.PHYSICAL:
            
            # Get TimeUnit duration in seconds
            time_unit_s = self.duration_s()

            time_shifted_s = time_s + ( time_unit_s * times )
            time_shifted_dt = dt_from_s(time_shifted_s, tz=time_dt.tzinfo)
            
            return time_shifted_dt   

        # Handle calendar time TimeSlot
        elif self.type == self.CALENDAR:

            # Create a TimeDelta object for everything but years and months
            delta = datetime.timedelta(weeks = self.weeks,
                                       days = self.days,
                                       hours = self.hours,
                                       minutes = self.minutes,
                                       seconds = self.seconds,
                                       microseconds = self.microseconds)
            
            # Apply the time deta for the shif
            time_shifted_dt = time_dt + delta
            
            # Handle years
            if self.years:
                time_shifted_dt = time_shifted_dt.replace(year=time_shifted_dt.year + self.years)
            
            # Handle months
            if self.months:
                
                tot_months = self.months + time_shifted_dt.month             
                years_to_add = math.floor(tot_months/12.0)
                new_month = (tot_months % 12 )
                if new_month == 0:
                    new_month=12
                    years_to_add = years_to_add -1
    
                time_shifted_dt = time_shifted_dt.replace(year=time_shifted_dt.year + years_to_add)
                time_shifted_dt = time_shifted_dt.replace(month=new_month)
            
            # Check DST offset consistency and fix if not respected
            if not check_dt_consistency(time_shifted_dt):
                time_shifted_dt = correct_dt_dst(time_shifted_dt)

        # Handle other cases (Consistency error)
        else:
            raise ConsistencyException('Error, TimeSlot type not PHYSICAL nor CALENDAR?!')
        
        # Return
        return time_shifted_dt   

    def duration_s(self, start_dt=None):
        '''Get the duration of the interval in seconds'''
        if self.is_composite():
            raise InputException('Sorry, only simple time intervals are supported by this operation')

        if self.type == 'Calendar' and not start_dt:
            raise InputException('With a calendar TimeUnit you can ask for duration only if you provide the starting point')
        
        if self.type == 'Calendar':
            
            end_dt = self.shift_dt(start_dt, 1)
            
            start_epoch = s_from_dt(start_dt)
            end_epoch = s_from_dt(end_dt)

            return (end_epoch - start_epoch)

        else:
            # Hours, Minutes, Seconds
            if self.hours:
                time_unit_s = self.hours * 60 * 60
            if self.minutes:
                time_unit_s = self.minutes * 60
            if self.seconds:
                time_unit_s = self.seconds
            if self.microseconds:
                time_unit_s = 1/1000000.0 * self.microseconds
               
        return time_unit_s

    @property
    def value(self):
        if self.type == 'Calendar':
            raise TypeError('Sorry, the value of a CALENDAR time unit is not defined. use duration_s() providing the starting point.')
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
