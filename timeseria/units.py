# -*- coding: utf-8 -*-
"""Provides Units and TimeUnits (which fully support calendar arithmetic including DST changes)."""

import re
import datetime
import math
from .time import s_from_dt , dt_from_s, get_tz_offset_s
from .time import check_dt_consistency, correct_dt_dst
from .utilities import is_numerical
from .exceptions import ConsistencyException
            
# Setup logging
import logging
logger = logging.getLogger(__name__)


HARD_DEBUG = False


class Unit(object):
    """A generic unit.
    
       Args:
           value: the unit value.    
    """
    
    def __init__(self, value):
        # TODO: check numerical or list? An perhaps support string representations?
        self.value = value
        
    def __repr__(self):   
        return str(self.value)

    def __add__(self, other):

        # Sum with numerical
        if is_numerical(self.value) and is_numerical(other):
            return self.value + other

        # Sum with another Unit
        if self.__class__ == other.__class__:
            return self.value + other.value

        # Sum with an objects with coordinates (Points, Slots)
        try:
            if len(other.coordinates) > 1:
                raise NotImplementedError('Cannot sum multidimensional objects')             
            return other.__class__(self.value + other.coordinates[0])
        except:
            raise NotImplementedError('Don\'t know how to add Units with {}'.format(other.__class__.__name__)) 
    
    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):

        # Subtract with numerical
        if is_numerical(self.value) and is_numerical(other):
            return self.value - other

        # Subtract with another Unit
        if self.__class__ == other.__class__:
            return self.value - other.value

        # Sum with an objects with coordinates (Points, Slots)
        try:
            if len(other.coordinates) > 1:
                raise NotImplementedError('Cannot subtract multidimensional objects')             
            return other.__class__(self.value - other.coordinates[0])
        except:
            raise NotImplementedError('Don\'t know how to subtract Units with {}'.format(other.__class__.__name__)) 

    def __rsub__(self, other):
        return -self.__sub__(other)


    def __truediv__(self, other):

        # Divide with numerical
        if is_numerical(self.value) and is_numerical(other):
            return self.value / other
        else:
            if is_numerical(self.value):
                raise NotImplementedError('Division with "{}" is not implemented'.format(other.__class__.__name__))
            else:
                raise NotImplementedError('Cannot divide multidimensional objects')

    def __rtruediv__(self, other):
        return ( 1 / self.__truediv__(other))


    def __mul__(self, other):

        # Multiply with numerical
        if is_numerical(self.value) and is_numerical(other):
            return self.value * other
        else:
            if is_numerical(self.value):
                raise NotImplementedError('Division with "{}" is not implemented'.format(other.__class__.__name__))
            else:
                raise NotImplementedError('Cannot divide multidimensional objects')

    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __eq__(self,other):
        if self is other:
            return True
        elif is_numerical(other):
            return self.value == other
        else:
            try:
                return self.value == other.value
            except AttributeError:
                return False


class TimeUnit(Unit):
    """A unit which can represent both physical (fixed) and calendar (variable) time units.
    Can handle precision up to the microsecond and can be summed and subtracted with numerical
    values, Python datetime objects, other TimeUnits, or TimePoints.

    Can be initialized both using a numerical value, a string representation, or by explicitly setting
    years, months, weeks, days, hours, minutes, seconds and microseconds. In the string representation,
    the mapping is as follows:
    
        * ``'Y': 'years'``
        * ``'M': 'months'``
        * ``'W': 'weeks'``
        * ``'D': 'days'``
        * ``'h': 'hours'``
        * ``'m': 'minutes'``
        * ``'s': 'seconds'``
        * ``'u': 'microseconds'``
     
    For example, to create a time unit of one hour, the following three are equivalent, where the
    first one uses the numerical value, the second the string representation, and the third explicitly
    sets the time component (hours in this case): ``TimeUnit('1h')``, ``TimeUnit(hours=1)``, or ``TimeUnit(3600)``.
    Not all time units can be initialized using the numerical value, in particular calendar time units which can
    have variable duration: a time unit of one day, or ``TimeUnit('1d')``, can last for 23, 24 or 24 hours depending
    on DST changes. On the contrary, a ``TimeUnit('24h')`` will always last 24 hours and can be initialized as
    ``TimeUnit(86400)`` as well. 
    
    Args:
        value: the time unit value, either as seconds (float) or string representation according to the mapping above.  
        years: the time unit years component.
        weeks: the time unit weeks component.
        months: the time unit weeks component.
        days: the time unit days component.
        hours: the time unit hours component.
        minutes: the time unit minutes component.
        seconds: the time unit seconds component.
        microseconds: the time unit microseconds component.
        trustme: a boolean switch to skip checks.
    """
    
    _CALENDAR = 'Calendar'
    _PHYSICAL = 'Physical'
    
    # NOT ref to https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes :  %d, %m, %w %y - %H, %M, %S
    # Instead: M, D, Y, W - h m s
    
    _mapping_table = { 
                       'Y': 'years',
                       'M': 'months',
                       'W': 'weeks',
                       'D': 'days',
                       'h': 'hours',
                       'm': 'minutes',
                       's': 'seconds',
                       'u': 'microseconds'
                      }

    def __init__(self, value=None, years=0, weeks=0, months=0, days=0, hours=0, minutes=0, seconds=0, microseconds=0, trustme=False):

        if not trustme:

            if value:
                if is_numerical(value):
                    string = '{}s'.format(value)
                else:
                    if not isinstance(value, str):
                        raise TypeError('TimeUnits must be initialized with a number, a string or explicitly setting years, months, days, hours etc. (Got "{}")'.format(string.__class__.__name__))
                    string = value
            else:
                string = None

            # Value OR explicit time components
            if value and (years or months or days or hours or minutes or seconds or microseconds):
                raise ValueError('Choose between string/numerical init and explicit setting of years, months, days, hours etc.')
    
            # Check types:
            if not isinstance(years, int): raise ValueError('year not of type int (got "{}")'.format(years.__class__.__name__))
            if not isinstance(weeks, int): raise ValueError('weeks not of type int (got "{}")'.format(weeks.__class__.__name__))
            if not isinstance(months, int): raise ValueError('months not of type int (got "{}")'.format(months.__class__.__name__))
            if not isinstance(days, int): raise ValueError('days not of type int (got "{}")'.format(days.__class__.__name__))
            if not isinstance(hours, int): raise ValueError('hours not of type int (got "{}")'.format(hours.__class__.__name__))
            if not isinstance(minutes, int): raise ValueError('minutes not of type int (got "{}")'.format(minutes.__class__.__name__))
            if not isinstance(seconds, int): raise ValueError('seconds not of type int (got "{}")'.format(seconds.__class__.__name__))
            if not isinstance(microseconds, int): raise ValueError('microseconds not of type int (got "{}")'.format(microseconds.__class__.__name__))

        # Set the time components if given
        # TODO: set them only if given?
        self.years        = years
        self.months       = months
        self.weeks        = weeks
        self.days         = days
        self.hours        = hours
        self.minutes      = minutes 
        self.seconds      = seconds
        self.microseconds = microseconds

        if string:

            # Specific case for floating point seconds (TODO: improve me, maybe inlcude in the regex?)            
            if string.endswith('s') and '.' in string:
                if '_' in string:
                    raise NotImplementedError('Composite TimeUnits with floating point seconds not yet implemented.')
                self.seconds = int(string.split('.')[0])
                
                # Get decimal seconds as string 
                decimal_seconds_str = string.split('.')[1][0:-1] # Remove the last "s"
                
                # Ensure we can handle precision
                if len(decimal_seconds_str) > 6:
                    decimal_seconds_str = decimal_seconds_str[0:6]
                    #raise ValueError('Sorry, "{}" has too many decimal seconds to be handled with a TimeUnit (which supports up to the microsecond).'.format(string))
                
                # Add missing trailing zeros
                missing_trailing_zeros = 6-len(decimal_seconds_str)                
                for _ in range(missing_trailing_zeros):
                    decimal_seconds_str += '0'
                
                # Cast to int & set
                self.microseconds = int(decimal_seconds_str)   

                return
            
            # Parse string using regex
            self.strings = string.split("_")
            regex = re.compile('^([0-9]+)([YMDWhmsu]{1,2})$')
            
            for string in self.strings:
                try:
                    groups   =  regex.match(string).groups()
                except AttributeError:
                    raise ValueError('Cannot parse string representation for the TimeUnit, unknown format ("{}")'.format(string)) from None

                setattr(self, self._mapping_table[groups[1]], int(groups[0]))

        if not trustme:  
                     
            # If nothing set, raise error
            if not self.years and not self.weeks and not self.months and not self.days and not self.hours and not self.minutes and not self.seconds and not self.microseconds:
                raise ValueError('Detected zero-duration TimeUnit!')
 
    def __repr__(self):
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
                raise ValueError('Timezone of the datetime to sum with is required')
            return self.shift_dt(other, times=1)
        
        elif isinstance(other, TimePoint):
            return TimePoint(dt=self.shift_dt(other.dt, times=1))
        
        elif is_numerical(other):
            return other + self.value
            
        else:
            raise NotImplementedError('Adding TimeUnits with objects of class "{}" is not implemented'.format(other.__class__.__name__))
   
    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        
        # TODO: This has to stay here or a circular import will start. Maybe refactor this?
        from .datastructures import TimePoint
        
        if isinstance(other, self.__class__):
            raise NotImplementedError('Subracting a TimeUnit from another TimeUnit is not implemented to prevent negative TimeUnits.')


        elif isinstance(other, datetime.datetime):
            if not other.tzinfo:
                raise ValueError('Timezone of the datetime to sum with is required')
            return self.shift_dt(other, times=-1)
        
        
        elif isinstance(other, TimePoint):
            return TimePoint(dt=self.shift_dt(other.dt, times=-1))

        elif is_numerical(other):
            return other - self.value

        else:
            raise NotImplementedError('Subracting TimeUnits with objects of class "{}" is not implemented'.format(other.__class__.__name__))
   
    def __sub__(self, other):
        raise NotImplementedError('Cannot subrtact anything from a TimeUnit. Only a TimeUnit from something else.')

    def __eq__(self, other):

        # Check for the quickest first (value)
        try:
            if self.value == other:
                return True
        except TypeError:
            # Raised if this TimeUnit is of calendar type
            pass
        
        # Check against another Unit with value
        if isinstance(other, Unit):
            try:
                if self.value == other.value:
                    return True
            except TypeError:
                # Raised if this or the other other TimeUnit is of calendar type
                pass
          
        # Check against another TimeUnit using datetime components
        if isinstance(other, TimeUnit):
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
        
        # If everything fails, return false:
        return False

    def _is_composite(self):
        types = 0
        for item in self._mapping_table:
            if getattr(self, self._mapping_table[item]): types +=1
        return True if types > 1 else False 

    @property
    def value(self):
        """The value of the TimeUnit, in seconds. Not defined for calendar time units (as they have variable duration)."""
        try:
            return self._value
        except AttributeError:
            if self.type == self._CALENDAR:
                raise TypeError('Sorry, the value of a calendar TimeUnit is not defined. use duration_s() providing the starting point.') from None
            self._value = self.duration_s()
            return self._value

    @property
    def type(self):
        """The type of the TimeUnit.
        
           - "Physical" if based on hours, minutes, seconds and  microseconds, which have fixed duration.
           - "Calendar" if based on years, months, weeks and days, which have variable duration depending on the starting date,
             and their math is not always well defined (e.g. adding a month to the 30th of January does not make sense)."""

        if self.years or self.months or self.weeks or self.days:
            return self._CALENDAR
        elif self.hours or self.minutes or self.seconds or self.microseconds:
            return self._PHYSICAL
        else:
            raise ConsistencyException('Error, TimeSlot not initialized?!')
    
    def is_physical(self):
        """Return True if the TimeUnit type is physical, False otherwise."""
        if self.type == self._PHYSICAL:
            return True
        else:
            return False
        
    def is_calendar(self):
        """Return True if the TimeUnit type is calendar, False otherwise."""
        if self.type == self._CALENDAR:
            return True 
        else:       
            return False        

    def round_dt(self, time_dt, how = None):
        """Round a datetime according to this TimeUnit."""

        if self._is_composite():
            raise ValueError('Sorry, only simple TimeUnits are supported by the rebase operation')

        if not time_dt.tzinfo:
            raise ValueError('The time zone of the datetime is required')    
                
        # Handle physical time 
        if self.type == self._PHYSICAL:
        
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
        elif self.type == self._CALENDAR:
            
            if self.years:
                if self.years > 1:
                    raise NotImplementedError('Cannot round based on calendar TimeUnits with years > 1')
                rounded_dt=time_dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
                
            if self.months:
                if self.months > 1:
                    raise NotImplementedError('Cannot round based on calendar TimeUnits with months > 1')
                rounded_dt=time_dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

            if self.weeks:
                raise NotImplementedError('Cannot round based on calendar TimeUnits with weeks')

            if self.days:
                if self.days > 1:
                    raise NotImplementedError('Cannot round based on calendar TimeUnits with days > 1')
                rounded_dt=time_dt.replace(hour=0, minute=0, second=0, microsecond=0)

            # Check DST offset consistency and fix if not respected
            if not check_dt_consistency(rounded_dt):
                rounded_dt = correct_dt_dst(rounded_dt)
            
            if how == 'ceil':
                rounded_dt = self.shift_dt(rounded_dt, 1)

        # Handle other cases (Consistency error)
        else:
            raise ConsistencyException('Error, TimeSlot type not physical nor calendar?!')

        # Return
        return rounded_dt

    def floor_dt(self, time_dt):
        """Floor a datetime according to this TimeUnit."""       
        return self.round_dt(time_dt, how='floor')
     
    def ceil_dt(self, time_dt):
        """Ceil a datetime according to this TimeUnit."""        
        return self.round_dt(time_dt, how='ceil')

    def rebase_dt(self, time_dt):
        """Rebase a given datetime to this TimeUnit."""
        return self.round_dt(time_dt, how='floor')
              
    def shift_dt(self, time_dt, times=1):
        """Shift a given datetime of n times of this TimeUnit."""
        if self._is_composite():
            raise ValueError('Sorry, only simple TimeUnits are supported by the rebase operation')
 
        # Convert input time to seconds
        time_s = s_from_dt(time_dt)
         
        # Handle physical time TimeSlot
        if self.type == self._PHYSICAL:
            
            # Get TimeUnit duration in seconds
            time_unit_s = self.duration_s()

            time_shifted_s = time_s + ( time_unit_s * times )
            time_shifted_dt = dt_from_s(time_shifted_s, tz=time_dt.tzinfo)
            
            return time_shifted_dt   

        # Handle calendar time TimeSlot
        elif self.type == self._CALENDAR:
            
            if times != 1:
                raise NotImplementedError('Cannot shift calendar TimeUnits for times than 1 (got times="{}")'.format(times))
            
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
            raise ConsistencyException('Error, TimeSlot type not physical nor calendar?!')
        
        # Return
        return time_shifted_dt   

    def duration_s(self, start_dt=None):
        """The duration of the TimeUnit in seconds."""

        if self.type == self._CALENDAR:

            if not start_dt:
                raise ValueError('With a calendar TimeUnit you can ask for duration only if you provide the starting point')

            if self._is_composite():
                raise ValueError('Sorry, only simple TimeUnits are supported by this operation')

            # Start Epoch
            start_epoch = s_from_dt(start_dt)

            # End epoch
            end_dt = self.shift_dt(start_dt, 1)
            end_epoch = s_from_dt(end_dt)

            # Get duration based on seconds
            time_unit_s = end_epoch - start_epoch

        elif self.type == 'Physical':
            time_unit_s = 0
            if self.hours:
                time_unit_s += self.hours * 60 * 60
            if self.minutes:
                time_unit_s += self.minutes * 60
            if self.seconds:
                time_unit_s += self.seconds
            if self.microseconds:
                time_unit_s += 1/1000000.0 * self.microseconds
        
        else:
            raise ConsistencyException('Unknown TimeUnit type "{}"'.format(self.type))
        
        return time_unit_s

    # Get start/end/center
    def _get_start(self, end=None, center=None):
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
            raise ValueError('get_start: Got not end nor center')        
            
    def _get_end(self, start=None, center=None):
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
            raise ValueError('get_end: Got not end nor center')

    def _get_center(self, start=None, end=None):
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
            raise ValueError('get_center: Got not end not start') 
