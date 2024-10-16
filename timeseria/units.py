# -*- coding: utf-8 -*-
"""Units, including the TimeUnit, which fully supports calendar arithmetic."""

from .utils import is_numerical
from propertime import TimeSpan

# Setup logging
import logging
logger = logging.getLogger(__name__)

# Hard debug switch
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



class TimeUnit(TimeSpan, Unit):
    """A unit that can have both fixed (physical) or variable (calendar) time length.
    It can handle precision up to the microsecond and can be added and subtracted with numerical
    values, Time and datetime objects, and other TimeUnits.

    Can be initialized both using a numerical value, a string representation, or by explicitly setting
    years, months, weeks, days, hours, minutes and seconds (including sub-second precision).
    In the string representation, the mapping is as follows:

        * ``'Y': 'years'``
        * ``'M': 'months'``
        * ``'W': 'weeks'``
        * ``'D': 'days'``
        * ``'h': 'hours'``
        * ``'m': 'minutes'``
        * ``'s': 'seconds'``

    For example, to create a time unit of one hour, the following three are equivalent, where the
    first one uses the numerical value, the second the string representation, and the third explicitly
    sets the time component (hours in this case): ``TimeUnit('1h')``, ``TimeUnit(hours=1)``, or ``TimeUnit(3600)``.
    Not all time units can be initialized using the numerical value, in particular calendar time units which can
    have variable duration: a time unit of one day, or ``TimeUnit('1d')``, can last for 23, 24 or 24 hours depending
    on DST changes. On the contrary, a ``TimeUnit('24h')`` will always last 24 hours and can be initialized as
    ``TimeUnit(86400)`` as well.

    Args:
        value: the time unit value, either as seconds (int or float) or as string representation according to the mapping above.
        years: the time unit years component (int).
        weeks: the time unit weeks component (int).
        months: the time unit weeks component (int).
        days: the time unit days component (int).
        hours: the time unit hours component (int).
        minutes: the time unit minutes component (int).
        seconds: the time unit seconds component, including sub-second precision (int, float).
    """

    # Support value-based init
    def __init__(self, *args, **kwargs):
        if args and (isinstance(args[0], int) or isinstance(args[0], float)):
            args = list(args)
            args[0] = '{}s'.format(args[0])
        if kwargs and 'value' in kwargs and (isinstance(kwargs['seconds'], int) or isinstance(kwargs['seconds'], float)) :
            kwargs['seconds'] = kwargs.pop('value')
        super(TimeUnit, self).__init__(*args, **kwargs)

    # Support for TimePoints and this TimeUnit class
    def __radd__(self, other):

        # This import has to stay here or a circular import will start. TODO: maybe refactor it?
        from .datastructures import TimePoint
        if isinstance(other, TimePoint):
            return TimePoint(dt=self.shift(other.dt, times=1))
        elif isinstance(other, self.__class__):
            return TimeUnit(years        = self.years + other.years,
                            months       = self.months + other.months,
                            weeks        = self.weeks + other.weeks,
                            days         = self.days + other.days,
                            hours        = self.hours + other.hours,
                            minutes      = self.minutes + other.minutes,
                            seconds      = self.seconds + other.seconds)
        else:
            return super().__radd__(other)

    def __rsub__(self, other):

        # This import has to stay here or a circular import will start. TODO: maybe refactor it?
        from .datastructures import TimePoint
        if isinstance(other, TimePoint):
            return TimePoint(dt=self.shift(other.dt, times=-1))
        else:
            return super().__rsub__(other)

    # For the doc strings
    def as_seconds(self, starting_at=None):
        """The length (duration) of the time unit, in seconds"""
        return super().as_seconds(starting_at)

    def ceil(self, time):
        """Ceil a time or datetime object according to this time unit."""
        return super().ceil(time)

    def floor(self, time):
        """Floor a time or datetime object according to this time unit."""
        return super().floor(time)

    def round(self, time, how='half'):
        """Round a time or datetime object according to this time unit."""
        return super().round(time, how)

    def shift(self, time, times=1):
        """Shift a given time or datetime object n times this time unit."""
        return super().shift(time, times)

