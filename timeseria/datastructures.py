# -*- coding: utf-8 -*-
"""Base data structures as Points, Slots, and Series."""

import json
from copy import deepcopy
from pandas import DataFrame, concat
from datetime import datetime
from pytz import UTC
from propertime import Time
from propertime.utils import s_from_dt , dt_from_s, timezonize, dt_from_str, str_from_dt

from .units import Unit, TimeUnit
from .utils import _is_close, _to_time_unit_string
from .utils import _is_index_based, _is_key_based, _has_numerical_values
from .exceptions import ConsistencyException

# Setup logging
import logging
logger = logging.getLogger(__name__)


#======================
#  Points
#======================

class Point():
    """A point.

       Args:
           *args (list): the coordinates.
    """

    def __init__(self, *args):
        if not args:
            raise Exception('A Point requires at least one coordinate, got none.')

        # Validate
        for arg in args:
            try:
                float(arg)
            except:
                raise Exception('Got non-numerical argument: "{}"'.format(arg))

        # Store (as tuple)
        self.coordinates = tuple(args)

    @property
    def __coordinates_repr__(self):
        if len(self.coordinates)==1:
            return str(self.coordinates[0])
        else:
            return str(self.coordinates).replace(' ', '')

    def __repr__(self):
        return 'Point @ {}'.format(self.__coordinates_repr__)

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        try:
            if not self.coordinates == other.coordinates:
                return False
            for i in range(0, len(self.coordinates)):
                if self.coordinates[i] != other.coordinates[i]:
                    return False
            return True
        except (AttributeError, IndexError):
            return False

    def __gt__(self, other):
        if len(self.coordinates) > 1:
            raise ValueError('Comparing multi-dimensional points does not make sense')
        else:
            try:
                if len(other.coordinates) > 1:
                    raise ValueError('Comparing multi-dimensional points does not make sense')
                else:
                    return self.coordinates[0] > other.coordinates[0]
            except AttributeError:
                return self.coordinates[0] > other

    def __getitem__(self,index):
        return self.coordinates[index]


class TimePoint(Point):
    """A point in the time dimension. Can be initialized using the special `t` and `dt` arguments,
    for epoch seconds and datetime objects, respectively.

       Args:
           t (float): epoch timestamp, decimals for sub-second precision.
           dt (datetime): a datetime object timestamp.
    """

    def __init__(self, *args, **kwargs):

        # Handle time zone if any (removing it from kwargs)
        tz = kwargs.pop('tz', None)
        if tz:
            self._tz = timezonize(tz)

        # Do we have to handle kwargs?
        if kwargs:
            if 't' in kwargs:

                # Ok, will create the point in the standard way, just used the "t" kwarg to set the time coordinate
                t = kwargs['t']

            elif 'dt' in kwargs:

                # Ok, will convert the datetime to epoch and then create the point in the standard way
                t = s_from_dt(kwargs['dt'])

                # If we do not have a time zone, can we use the one from the dt used to initialize this TimePoint?
                try:
                    self._tz
                except AttributeError:
                    if kwargs['dt'].tzinfo:

                        # Do not set it if it is UTC, it is the default
                        if kwargs['dt'].tzinfo == UTC:
                            pass
                        else:
                            self._tz = kwargs['dt'].tzinfo

        # Cast or create in the standard way
        elif args:
            if len(args) > 1:
                raise Exception('Don\'t know how to handle all args (got "{}")'.format(args))

            if isinstance(args[0], TimePoint):
                t   = args[0].t
                self._tz = args[0].tz
            else:
                t = args[0]

        # Call parent init
        super(TimePoint, self).__init__(t)

    @property
    def t(self):
        """The timestamp as epoch, with decimals for sub-second precision."""
        return self.coordinates[0]

    def __gt__(self, other):
        if self.t > other.t:
            return True
        else:
            return False

    @property
    def tz(self):
        """The time zone."""
        try:
            return self._tz
        except AttributeError:
            return UTC

    def change_tz(self, tz):
        """Change the time zone of the point, in-place."""
        self._tz = timezonize(tz)

    @property
    def dt(self):
        """The timestamp as datetime object."""
        return dt_from_s(self.t, tz=self.tz)

    def __repr__(self):
        return 'Time point @ {} ({})'.format(self.t, self.dt)


class DataPoint(Point):
    """A point that carries some data. Data is attached using the respective data arguments.

       Args:
           *args (list): the coordinates.
           data: the data.
           data_indexes(dict): data indexes.
           data_loss(float): the data loss index, if any.
    """

    def __init__(self, *args, **kwargs):

        # Data
        try:
            self._data = kwargs.pop('data')
        except KeyError:
            raise Exception('A DataPoint requires a special "data" argument (got only "{}")'.format(kwargs))

        # Data indexes
        try:
            data_indexes = kwargs.pop('data_indexes')
            if data_indexes is None:
                data_indexes = {}
            else:
                if not isinstance(data_indexes, dict):
                    raise ValueError('Got type "{}" for data_indexes, was expecitng a dict'.format(data_indexes.__class__.__name__))
        except KeyError:
            data_indexes = {}

        # Special data loss index
        try:
            data_loss = kwargs.pop('data_loss')
        except KeyError:
            pass
        else:
            data_indexes['data_loss'] = data_loss

        # Set data indexes
        self._data_indexes = data_indexes

        # Call parent init
        super(DataPoint, self).__init__(*args, **kwargs)

    def __repr__(self):
        if self.data_loss is not None:
            return 'Point @ {} with data "{}" and data_loss="{}"'.format(self.__coordinates_repr__, self.data, self.data_loss)
        else:
            return 'Point @ {} with data "{}"'.format(self.__coordinates_repr__, self.data)

    def __eq__(self, other):
        try:
            if self._data != other._data:
                return False
        except AttributeError:
            pass
        return super(DataPoint, self).__eq__(other)

    @property
    def data(self):
        """The data."""
        # Data is implemented using a property to enforce that it cannot be changed after being set
        # via the init, in particular with respect to the series, where data points are checked, upon
        # insertion, to carry the same data type and with the same number of elements.
        return self._data

    @property
    def data_indexes(self):
        """The data indexes."""
        return self._data_indexes

    @data_indexes.setter
    def data_indexes(self, value):
        self._data_indexes = value

    @property
    def data_loss(self):
        """The data loss index, if any. Usually computed out from resampling transformations."""
        try:
            return self.data_indexes['data_loss']
        except KeyError:
            return None

    def data_labels(self):
        """The data labels. If data is a dictionary, then these are the dictionary keys, if data is
        list-like, then these are the list indexes (as strings). Other formats are not supported.

        Returns:
            list: the data labels.
        """
        try:
            return sorted(list(self.data.keys()))
        except AttributeError:
            return [str(i) for i in range(len(self.data))]

    def _data_by_label(self, label):
        try:
            return self.data[label]
        except TypeError:
            try:
                return self.data[int(label)]
            except TypeError:
                raise TypeError('Cannot get an element by label on non-indexed data')


class DataTimePoint(DataPoint, TimePoint):
    """A point that carries some data in the time dimension. Can be initialized using the special `t` and `dt` arguments,
       for epoch seconds and datetime objects, respectively. Data is attached using the respective data arguments.

       Args:
           t(float): epoch timestamp, decimals for sub-second precision.
           dt(datetime): a datetime object timestamp.
           data: the data.
           data_indexes(dict): data indexes.
           data_loss(float): the data loss index, if any.
    """

    def __repr__(self):
        if self.data_loss is not None:
            return 'Time point @ {} ({}) with data "{}" and data_loss="{}"'.format(self.t, self.dt, self.data, self.data_loss)
        else:
            return 'Time point @ {} ({}) with data "{}"'.format(self.t, self.dt, self.data)


#======================
#  Slots
#======================

class Slot():
    """A slot. Can be initialized with start and end or start and unit.

       Args:
           start(Point): the slot starting point.
           end(Point): the slot ending point.
           unit(Unit): the slot unit.
    """
    __POINT_TYPE__ = Point
    __UNIT_TYPE__ = Unit

    def __init__(self, start, end=None, unit=None):

        if not isinstance(start, self.__POINT_TYPE__):
            raise TypeError('{} start must be a {} object (got "{}")'.format(self.__class__.__name__, self.__POINT_TYPE__.__name__, start.__class__.__name__))

        if end and not isinstance(end, self.__POINT_TYPE__):
            raise TypeError('{} end must be a {} object (got "{}")'.format(self.__class__.__name__, self.__POINT_TYPE__.__name__, end.__class__.__name__))

        if unit and not isinstance(unit, self.__UNIT_TYPE__):
            raise TypeError('{} unit must be a {} object (got "{}")'.format(self.__class__.__name__, self.__UNIT_TYPE__.__name__, unit.__class__.__name__))

        if end is None and unit is not None:
            if len(start.coordinates)>1:
                raise Exception('Sorry, setting a start and a unit only works in unidimensional spaces')
            end = start + unit
        else:
            if len(start.coordinates) != len(end.coordinates):
                raise ValueError('{} start and end dimensions must be the same (got "{}" vs "{}")'.format(self.__class__.__name__,start.coordinates, end.coordinates))
            if start == end:
                raise ValueError('{} start and end must not be the same (got start="{}", end="{}")'.format(self.__class__.__name__, start,end))

        self.start = start
        self.end   = end

        if unit is not None:
            self._unit = unit

    def __repr__(self):
        return 'Slot @ [{},{}]'.format(self.start.__coordinates_repr__, self.end.__coordinates_repr__)

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        if not self.start == other.start:
            return False
        if not self.end == other.end:
            return False
        return True

    def __succedes__(self, other):
        if other.end != self.start:
            return False
        else:
            return True

    @classmethod
    def _compute_length(cls, start, end):
        values = []
        for i in range(len(start.coordinates)):
            values.append(end.coordinates[i] - start.coordinates[i])

        return sum(values)/len(values)

    @property
    def length(self):
        """The slot length."""
        try:
            self._length
        except AttributeError:
            self._length = self._compute_length(self.start, self.end)
        return self._length

    @property
    def unit(self):
        """The slot unit."""
        try:
            return self._unit
        except AttributeError:
            if len(self.start.coordinates) == 1:
                return self.__UNIT_TYPE__(self.end.coordinates[0] - self.start.coordinates[0])
            else:
                values = []
                for i in range(len(self.start.coordinates)):
                    values.append(self.end.coordinates[i] - self.start.coordinates[i])

                return self.__UNIT_TYPE__(values)


class TimeSlot(Slot):
    """A slot in the time dimension. Can be initialized with start and end
       or start and unit.

       Args:
           start(TimePoint): the slot starting time point.
           end(TimePoint): the slot ending time point.
           unit(TimeUnit): the slot time unit."""

    __POINT_TYPE__ = TimePoint
    __UNIT_TYPE__ = TimeUnit

    def __init__(self, start=None, end=None, unit=None, **kwargs):

        # Internal-use init
        t = kwargs.get('t', None)
        dt = kwargs.get('dt', None)
        tz = kwargs.get('tz', None)

        # Handle t and dt shortcuts
        if t is not None:
            start=TimePoint(t=t, tz=tz)
        if dt is not None:
            start=TimePoint(dt=dt, tz=tz)

        # Extra time zone checks
        if start and end:
            if start.tz != end.tz:
                raise ValueError('{} start and end must have the same time zone (got start.tz="{}", end.tz="{}")'.format(self.__class__.__name__, start.tz, end.tz))

        # Call parent init
        super(TimeSlot, self).__init__(start=start, end=end, unit=unit)

        # Store time zone
        self.tz = start.tz


    # Overwrite parent succedes, this has better performance as it checks for only one dimension
    def __succedes__(self, other):
        if other.end.t != self.start.t:
            # Take into account floating point rounding errors
            if _is_close(other.end.t, self.start.t):
                return True
            return False
        else:
            return True

    def change_tz(self, tz):
        """Change the time zone of the slot, in-place."""
        self.start.change_tz(tz)
        self.end.change_tz(tz)
        self.tz = self.start.tz

    @property
    def unit(self):
        """The slot time unit"""
        try:
            return self._unit
        except AttributeError:
            # Use the string representation to handle floating point seconds
            self._unit = TimeUnit(str(self.end.t-self.start.t)+'s')
            return self._unit

    @property
    def t(self):
        """The slot epoch timestamp, intended as the starting point one."""
        return self.start.t

    @property
    def dt(self):
        """The slot datetime timestamp, intended as the starting point one."""
        return self.start.dt

    def __repr__(self):
        return 'Time slot @ [{},{}] ([{},{}])'.format(self.start.t, self.end.t, self.start.dt, self.end.dt)


class DataSlot(Slot):
    """A slot that carries some data. Can be initialized with start and end
       or start and unit, plus the data arguments.

       Args:
           start(Point): the slot starting point.
           end(Point): the slot ending point.
           unit(Unit): the slot unit.
           data: the data.
           data_indexes(dict): data indexes.
           data_loss(float): the data loss index, if any.
    """

    def __init__(self, *args, **kwargs):

        # Data
        try:
            self._data = kwargs.pop('data')
        except KeyError:
            raise Exception('A DataSlot requires a special "data" argument (got only "{}")'.format(kwargs))

        # Data indexes
        try:
            data_indexes = kwargs.pop('data_indexes')
            if data_indexes is None:
                data_indexes = {}
            else:
                if not isinstance(data_indexes, dict):
                    raise ValueError('Got type "{}" for data_indexes, was expecitng a dict'.format(data_indexes.__class__.__name__))
        except KeyError:
            data_indexes = {}

        # Special data loss index
        try:
            data_loss = kwargs.pop('data_loss')
        except KeyError:
            pass
        else:
            data_indexes['data_loss'] = data_loss

        # Set data indexes
        self._data_indexes = data_indexes

        # Call parent init
        super(DataSlot, self).__init__(*args, **kwargs)

    def __repr__(self):
        if self.data_loss is not None:
            return 'Slot @ [{},{}] with data "{}" and data_loss="{}"'.format(self.start.__coordinates_repr__, self.end.__coordinates_repr__, self.data, self.data_loss)
        else:
            return 'Slot @ [{},{}] with data "{}"'.format(self.start.__coordinates_repr__, self.end.__coordinates_repr__, self.data)

    def __eq__(self, other):
        if self._data != other._data:
            return False
        return super(DataSlot, self).__eq__(other)

    @property
    def data(self):
        """The data."""
        # Data is implemented using a property to enforce that it cannot be changed after being set
        # via the init, in particular with respect to the series, where data slots are checked, upon
        # insertion, to carry the same data type and with the same number of elements.
        return self._data

    @property
    def data_indexes(self):
        """The data indexes."""
        return self._data_indexes

    @data_indexes.setter
    def data_indexes(self, value):
        self._data_indexes = value

    @property
    def data_loss(self):
        """The data loss index, if any. Usually computed from a resampling or aggregation transformation."""
        try:
            return self.data_indexes['data_loss']
        except KeyError:
            return None

    def data_labels(self):
        """The data labels. If data is a dictionary, then these are the dictionary keys, if data is
        list-like, then these are the list indexes (as strings). Other formats are not supported.

        Returns:
            list: the data labels.
        """
        try:
            return sorted(list(self.data.keys()))
        except AttributeError:
            return [str(i) for i in range(len(self.data))]

    def _data_by_label(self, label):
        try:
            return self.data[label]
        except TypeError:
            try:
                return self.data[int(label)]
            except TypeError:
                raise TypeError('Cannot get an element by label on non-indexed data')


class DataTimeSlot(DataSlot, TimeSlot):
    """A slot that carries some data in the time dimension. Can be initialized
       with start and end or start and unit, plus the data arguments.

       Args:
           start(TimePoint): the slot starting time point.
           end(TimePoint): the slot ending time point.
           unit(TimeUnit): the slot time unit.
           data: the data.
           data_indexes(dict): data indexes.
           data_loss(float): the data loss index, if any.
    """

    def __repr__(self):
        if self.data_loss is not None:
            return 'Time slot @ [{},{}] ([{},{}]) with data={} and data_loss={}'.format(self.start.t, self.end.t, self.start.dt, self.end.dt, self.data, self.data_loss)
        else:
            return 'Time slot @ [{},{}] ([{},{}]) with data={}'.format(self.start.t, self.end.t, self.start.dt, self.end.dt, self.data)


#==============================
#  Series
#==============================

class Series(list):
    """A list of items coming one after another, where every item
       is guaranteed to be of the same type and in order or succession.

       The square brackets notation can be used for accessing series items, slicing the series
       or to filter it on a specific data label, if its elements support it:

           * ``series[3]`` will access the item in position #3;

           * ``series[3:5]`` will slice the series from item in position #3 to item in position #5 (excluded);

           * ``series['temperature']`` will filter the series keeping only the temperature data, assuming that
             in the original series there were also other data labels (e.g. humidity).

       Args:
           *args: the series items.
    """

    # 1,2,3 are in order. 1,3,8 are in order. 1,8,3 re not in order.
    # 5,6,7 are integer succession. 5.3, 5.4, 5.5 are too in a succesisons. 5,6,8 are not in a succession.
    # a group or a number of related or similar things, events, etc., arranged or occurring in temporal, spatial, or other order or succession; sequence.

    def __init__(self, *args, **kwargs):

        if kwargs:
            raise ValueError('Got an unknown argument "{}"'.format(list(kwargs.keys())[0]))

        for arg in args:
            self.append(arg)

    @property
    def item_type(self):
        """The type of the items of the series."""
        if not self:
            return None
        else:
            return self[0].__class__

    def append(self, item):
        """Append an item to the series. Accepts only items of the same
        type of the items already present in the series (unless empty)"""

        # Check type
        if self.item_type:
            if not isinstance(item, self.item_type):
                raise TypeError('Got incompatible type "{}", can only accept "{}"'.format(item.__class__.__name__, self.item_type.__name__))
        else:
            # First item appended to this series: check for compatible data type if any
            try:
                item.data
            except AttributeError:
                pass
            else:
                if _is_index_based(item.data):
                    if not _has_numerical_values(item.data):
                        logger.warning('You are using a data type that has no (or not only) numerical values. Support will be very limited.')
                elif _is_key_based(item.data):
                    if not _has_numerical_values(item.data):
                        logger.warning('You are using a data type that has no (or not only) numerical values. Support will be very limited.')
                else:
                    logger.warning('You are using a data type that is neither index-based nor in key-value format. Support will be very limited.')

        # Check order or succession if not empty
        if self:
            try:
                if not item.__succedes__(self[-1]):
                    raise ValueError('Not in succession ("{}" does not succeedes "{}")'.format(item,self[-1])) from None
            except IndexError:
                raise
            except AttributeError:
                try:
                    if not item > self[-1]:
                        raise ValueError('Not in order ("{}" does not follow "{}")'.format(item,self[-1])) from None
                except TypeError:
                    raise TypeError('Object of class "{}" does not implement a "__gt__" or a "__succedes__" method, cannot append it to a Series (which is ordered)'.format(item.__class__.__name__)) from None

        # Check data type and set if not done already
        if self and self._item_data_reference:
            if not type(self._item_data_reference) == type(item.data):
                raise TypeError('Got different data: {} vs {}'.format(self._item_data_reference.__class__.__name__, item.data.__class__.__name__))
            if isinstance(self._item_data_reference, list):
                if len(self._item_data_reference) != len(item.data):
                    raise ValueError('Got different data lengths: {} vs {}'.format(len(self._item_data_reference), len(item.data)))
            if isinstance(self._item_data_reference, dict):
                if set(self._item_data_reference.keys()) != set(item.data.keys()):
                    raise ValueError('Got different data labels: {} vs {}'.format(self._item_data_reference.keys(), item.data.keys()))
        else:
            try:
                self._item_data_reference = item.data
            except AttributeError:
                self._item_data_reference = None

        # Append
        super(Series, self).append(item)

    def __sum__(self, other):
        raise NotImplementedError

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return '{} of #{} elements'.format(self.__class__.__name__, len(self))

    def __str__(self):
        return self.__repr__()

    def _all_data_indexes(self):
        """The data_indexes of the series, to be intended as custom
        defined indicators (i.e. data_loss, anomaly_index, etc.).

        Returns:
            list: the data indexes.
        """

        data_index_names = []
        for item in self:
            for index_name in item.data_indexes:
                if index_name not in data_index_names:
                    data_index_names.append(index_name)

        # Reorder according to legacy indexes behaviour
        ordered_data_index_names = []
        legacy_indexes = ['data_reconstructed', 'data_loss', 'anomaly', 'forecast']
        for legacy_index in legacy_indexes:
            if legacy_index in data_index_names:
                ordered_data_index_names.append(legacy_index)
                data_index_names.remove(legacy_index)

        # Merge and sort
        ordered_data_index_names += sorted(data_index_names)

        return ordered_data_index_names

    # Inherited methods to be edited
    def insert(self, i, x):
        """Insert an item at a given position. The first argument is the index of the element
        before which to insert, so series.insert(0, x) inserts at the front of the series, and
        series.insert(len(series), x) is equivalent to append(x). Order or succession are enforced."""

        if len(self) > 0:

            # Check valid type
            if not isinstance(x, self.item_type):
                raise TypeError('Got incompatible type "{}", can only accept "{}"'.format(x.__class__.__name__, self.item_type.__name__))

            # Check ordering/succession
            if i == 0:
                try:
                    if not self[0].__succedes__(x):
                        raise ValueError('Cannot insert element "{}" in position "{}" as it would break the succession'.format(x,i))
                except AttributeError:
                    if not self[0] > x:
                        raise ValueError('Cannot insert element "{}" in position "{}" as not in order'.format(x,i)) from None

            elif i == len(self):
                try:
                    if not x.__succedes__(self[-1]):
                        raise ValueError('Cannot insert element "{}" in position "{}" as it would break the succession'.format(x,i))
                except AttributeError:
                    if not x > self[-1]:
                        raise ValueError('Cannot insert element "{}" in position "{}" as not in order'.format(x,i)) from None

            else:
                try:
                    self[0].__succedes__
                except AttributeError:
                    if not x > self[i-1]:
                        raise ValueError('Cannot insert element "{}" in position "{}" with element "{}" as not in order'.format(x,i, self[i])) from None
                else:
                    raise IndexError('Cannot insert an item in the middle of a series whose items follow a succession'.format(i))

            super(Series, self).insert(i,x)

        else:
            self.append(x)

    def remove(self, x):
        """Remove the first item from the list whose value is equal to x. It raises a `ValueError` if there is no such item
        and a `NotImplementedError` is the series items are in a succession as it would breake it.
        """
        try:
            self[0].__succedes__
            raise NotImplementedError('Remove is not implemented for series whose items are in a succession')
        except IndexError:
            pass
        except AttributeError:
            pass

        super(Series, self).remove(x)

    def pop(self, i=None):
        """
        Remove the item at the given position in the list, and return it. If no index is
        specified, removes and returns the last item in the series. If items are in a
        succession and the index is set, a `NotImplementerError` is raised.
        """
        if len(self) == 0:
            raise IndexError('Cannot pop from an empty series')
        if i is not None:
            if i not in [0, len(self)]:
                try:
                    self[0].__succedes__
                except AttributeError:
                    pass
                else:
                    raise IndexError('Cannot pop an item in the middle of a series whose items follow a succession')
            return super(Series, self).pop(i)
        else:
            return super(Series, self).pop()

    def clear(self):
        """Remove all items from the series."""
        super(Series, self).clear()

    def index(self, x, start=None, end=None):
        """
        Return zero-based index in the series of the item whose value is equal to x.
        Raises a ValueError if there is no such item.

        The optional arguments start and end are interpreted as in the slice notation and are
        used to limit the search to a particular subsequence of the list. The returned index is
        computed relative to the beginning of the full sequence rather than the start argument.
        """
        super(Series, self).index(x, start, end)

    def copy(self):
        """Return a shallow copy of the series."""
        return super(Series, self).copy()

    def duplicate(self):
        """ Return a deep copy of the series."""
        return deepcopy(self)

    def _item_by_i(self, i):
        return super(Series, self).__getitem__(i)

    def _slice_by_i(self, **args):
        return super(Series, self).__getitem__(**args)

    # Inherited methods to be disabled
    def extend(self):
        """Disabled (use the `merge` instead)."""
        raise NotImplementedError('Use the "merge()" instead')

    def count(self, x):
        """Disabled (there is only one item instance by design)."""
        raise NotImplementedError('There is only one item by design')

    def sort(self, key=None, reverse=False):
        """Disabled (sorting is already guaranteed)."""
        raise NotImplementedError('Sorting is already guaranteed')

    def reverse(self):
        """Disabled (reversing is not compatible with an ordering)."""
        raise NotImplementedError('Reversing is not compatible with an ordering')


    #=========================
    # Square brackets notation
    #=========================

    def __getitem__(self, key):
        if isinstance(key, slice):
            indices = range(*key.indices(len(self)))
            # Prepare the new series and return it
            series = self.__class__()
            for i in indices:
                series.append(super(Series, self).__getitem__(i))
            return series
        elif isinstance(key, str):
            # Try filtering on this data label only
            return self.filter(key)
        else:
            return super(Series, self).__getitem__(key)


    #=========================
    #  Data-related
    #=========================

    def data_labels(self):
        """The labels of the data carried by the series items. If data is a dictionary, then
        these are the dictionary keys, if data  is list-like, then these are the list indexes
        (as strings). Other formats are not supported.

        Returns:
            list: the data labels.
        """
        if len(self) > 0 and not self._item_data_reference:
            raise TypeError('Series items have no data, cannot get data labels')
        if len(self) == 0:
            return None
        else:
            return self[0].data_labels()

    def rename_data_label(self, old_data_label, new_data_label):
        """Rename a data label, in-place."""
        if len(self) > 0 and not self._item_data_reference:
            raise TypeError('Series items have no data, cannot rename a label')
        for item in self:
            item.data[new_data_label] = item.data.pop(old_data_label)

    def remove_data_label(self, data_label):
        """Remove a data label, in-place."""
        if len(self) > 0 and not self._item_data_reference:
            raise TypeError('Series items have no data, cannot rename a label')
        for item in self:
            item.data.pop(data_label, None)

    def remove_data_index(self, data_index):
        """Remove a data index, in-place."""
        if len(self) > 0 and not self._item_data_reference:
            raise TypeError('Series items have no data, cannot rename data indexes')
        for item in self:
            item.data_indexes.pop(data_index, None)

    def remove_data_loss(self):
        """Remove the ``data_loss`` index, in-place."""
        if len(self) > 0 and not self._item_data_reference:
            raise TypeError('Series items have no data, cannot remove the data loss')
        for item in self:
            item.data_indexes.pop('data_loss', None)

    #=========================
    #  Operations
    #=========================

    def min(self, data_label=None):
        """Get the minimum data value(s) of the series. A series of DataPoints or DataSlots is required.

        Args:
           data_label(string): if provided, compute the value only for this data label.
                               Defaults to None.

        Returns:
           dict or object: the computed values for each data label, or a specific value if
           providing the data_label argument.
        """
        from .operations import min as min_operation
        return min_operation(self, data_label=data_label)

    def max(self, data_label=None):
        """Get the maximum data value(s) of the series. A series of DataPoints or DataSlots is required.

        Args:
           data_label(string): if provided, compute the value only for this data label. Defaults to None.

        Returns:
           dict or object: the computed values for each data label, or a specific value if
           providing the data_label argument.
        """
        from .operations import max as max_operation
        return max_operation(self, data_label=data_label)

    def avg(self, data_label=None):
        """Get the average data value(s) of the series. A series of DataPoints or DataSlots is required.

        Args:
           data_label(string, optional): if provided, compute the value only for this data label.

        Returns:
           dict or object: the computed values for each data label, or a specific value if
           providing the data_label argument.
        """
        from .operations import avg as avg_operation
        return avg_operation(self, data_label=data_label)

    def sum(self, data_label=None):
        """Sum every data value(s) of the series. A series of DataPoints or DataSlots is required.

        Args:
           data_label(string, optional): if provided, compute the value only for this data label.

        Returns:
           dict or object: the computed values for each data label, or a specific value if
           providing the data_label argument.
        """
        from .operations import sum as sum_operation
        return sum_operation(self, data_label=data_label)

    def derivative(self, inplace=False, normalize=True, diffs=False):
        """Compute the derivative on the series. A series of DataTimePoints or DataTimeSlots is required.

        Args:
           inplace(bool): if to perform the operation in-place on the series. Defaults to False.
           normalize(bool): if to normalize the derivative w.r.t to the series resolution. Defaults to True.
           diffs(bool): if to compute the differences instead of the derivative. Defaults to False.

        Returns:
           series or None: the computed series, or None if set to perform the operation in-place.
        """
        from .operations import derivative as derivative_operation
        return derivative_operation(self, inplace=inplace, normalize=normalize, diffs=diffs)

    def integral(self, inplace=False, normalize=True, c=0, offset=0):
        """Compute the integral on the series. A series of DataTimePoints or DataTimeSlots is required.

        Args:
           inplace(bool): if to perform the operation in-place on the series. Defaults to False.
           normalize(bool): if to normalize the integral w.r.t to the series resolution. Defaults to True.
           c(float, dict): the integrative constant, as a single value or as a dictionary of values, one
                           for each data label. Defaults to zero.
           offset(float, dict): if to start the integrative process from a specific offset. Can be provided as a
                                single value or as a dictionary of values, one for each data label. Defaults to zero.

        Returns:
           series or None: the computed series, or None if set to perform the operation in-place.
        """
        from .operations import integral as integral_operation
        return integral_operation(self, inplace=inplace, normalize=normalize, c=c, offset=offset)

    def diff(self, inplace=False):
        """Compute the incremental differences on the series. Reduces the series length by one
        (removing the first element). A series of DataTimePoints or DataTimeSlots is required.

        Args:
           inplace(bool): if to perform the operation in-place on the series. Defaults to False.

        Returns:
           series or None: the computed series, or None if set to perform the operation in-place.
        """
        from .operations import diff as diff_operation
        return diff_operation(self, inplace=inplace)

    def csum(self, inplace=False, offset=None):
        """Compute the incremental sum on the series. A series of DataTimePoints or DataTimeSlots is required.

        Args:
           inplace(bool): if to perform the operation in-place on the series. Defaults to False.
           offset(float, dict): if to start computing the cumulative sum from a specific offset. Can be provided as a
                                single value or as a dictionary of values, one for each data label. Defaults to None.

        Returns:
           series or None: the computed series, or None if set to perform the operation in-place.
        """
        from .operations import csum as csum_operation
        return csum_operation(self, inplace=inplace, offset=offset)

    def normalize(self, range=[0,1], inplace=False, source_range=None):
        """Normalize the data values of the series bringing them to a given range. A series of DataTimePoints or DataTimeSlots is required.

        Args:
           range(list): the normalization target range. Defaults to [0,1].
           inplace(bool): if to perform the operation in-place on the series. Defaults to False.
           source_range(dict, optional): a custom source range, by data label, to normalize with respect to.

        Returns:
           series or None: the computed series, or None if set to perform the operation in-place.
        """
        from .operations import normalize as normalize_operation
        return normalize_operation(self, inplace=inplace, range=range, source_range=source_range)

    def rescale(self, value, inplace=False):
        """Rescale the data values of the series by a given factor. A series of DataTimePoints or DataTimeSlots is required.

        Args:
           value(float, dict): the value to use as rescaling factor. Can be provided as a single
                               value or as a dictionary of values, one for each data label.
           inplace(bool): if to perform the operation in-place on the series. Defaults to False.

        Returns:
           series or None: the computed series, or None if set to perform the operation in-place.
        """
        from .operations import rescale as rescale_operation
        return rescale_operation(self, value=value, inplace=inplace)

    def offset(self, value, inplace=False):
        """Offset the data values of the series by a given value. A series of DataTimePoints or DataTimeSlots is required.

        Args:
           value(float, dict): the value to use as offset. Can be provided as a single
                               value or as a dictionary of values, one for each data label.
           inplace(bool): if to perform the operation in-place on the series. Defaults to False.

        Returns:
           series or None: the computed series, or None if set to perform the operation in-place.
        """
        from .operations import offset as offset_operation
        return offset_operation(self, value=value, inplace=inplace)

    def mavg(self,  window, inplace=False):
        """Compute the moving average on the series. Reduces the series length by a number of values
        equal to the window size. A series of DataTimePoints or DataTimeSlots is required.

        Args:
           window(int): the length of the moving average window.
           inplace(bool): if to perform the operation in-place on the series. Defaults to False.

        Returns:
           series or None: the computed series, or None if set to perform the operation in-place.
        """
        from .operations import mavg as mavg_operation
        return mavg_operation(self, window=window, inplace=inplace)

    def merge(self, series):
        """Merge the series with one or more other series.

        Returns:
            Series: the merged series.
        """
        from .operations import merge as merge_operation
        return merge_operation(self, series)

    def get(self, at_i):
        """Get the element of the series at a given position.

        Args:
           at_i(int): the position of the item to get.

        Returns:
            object: the item in the given position or at the given time.
        """
        from .operations import get as get_operation
        return get_operation(self, at_i=at_i)

    def filter(self, *data_labels):
        """Filter the series keeping only the data labels provided as argument.

        Args:
           *data_labels(str): the data label(s) to filter against.
        """
        from .operations import filter as filter_operation
        return filter_operation(self, *data_labels)

    def slice(self, from_i=None, to_i=None):
        """Slice the series between the given positions. A series of DataPoints or DataSlots is required.

        Args:
           from_i(int): the slicing start position. Defaults to None.
           to_i(int): the slicing end position. Defaults to None.

        Returns:
            Series: the sliced series.
        """
        from .operations import slice as slice_operation
        return slice_operation(self, from_i=from_i, to_i=to_i)

    def select(self, query):
        """Select one or more items of the series given an SQL-like query. This is a preliminary
        functionality supporting only the equality. A series of DataPoints or DataSlots is required.

        Args:
           query(str): the query.

        Returns:
            list: the selected items of the series.
        """
        from .operations import select as select_operation
        return select_operation(self, query=query)


    #=========================
    #  Transformations
    #=========================

    def aggregate(self, unit, *args, **kwargs):
        """Aggregate the series in slots. A series of DataPoints or DataSlots is required.

        Args:
           unit(Unit): the target slot unit (i.e. length).

        Returns:
            Series: the aggregated series.
        """
        from .transformations import Aggregator
        aggregator = Aggregator(unit, *args, **kwargs)
        return aggregator.process(self)

    def resample(self, unit, *args, **kwargs):
        """Aggregate the series in slots. A series of DataPoints or DataSlots is required.

        Args:
           unit(Unit): the unit (i.e. length) of the target sampling interval.

        Returns:
            Series: the resampled series.
        """
        from .transformations import Resampler
        resampler = Resampler(unit, *args, **kwargs)
        return resampler.process(self)


    #=========================
    # Inspection utilities
    #=========================

    def _summary(self, limit=10):

        string = str(self)+':\n\n'
        string+='['

        if not limit or limit > len(self):

            for i, item in enumerate(self):
                if limit and i >= limit:
                    break
                else:
                    if i==0:
                        string+=str(item)+',\n'
                    elif i==len(self)-1:
                        string+=' '+str(item)
                    else:
                        string+=' '+str(item)+',\n'
        else:
            if limit==1:
                head_n=1
                tail_n=0
            else:
                if limit % 2 == 0:
                    head_n = int(limit/2)
                else:
                    head_n = int(limit/2)+1
                tail_n = int(limit/2)

            for i, item in enumerate(self.head(head_n)):
                if i==0:
                    string+=str(item)+',\n'
                else:
                    string+=' '+str(item)+',\n'
            if limit < len(self):
                string+=' ...\n'
            if tail_n != 0:
                for i, item in enumerate(self.tail(tail_n)):
                    if i==tail_n-1:
                        string+=' '+str(item)
                    else:
                        string+=' '+str(item)+',\n'

        string+=']'
        return string

    def summary(self, limit=10, newlines=False):
        """A summary of the series and its elements, limited to 10 items by default.

            Args:
                limit(int): the limit of elements to print, by default 10.
                newlines(bool): if to include the newline characters or not.

            Returns:
                str: the summary.
        """
        if newlines:
            return self._summary(limit=limit)
        else:
            return self._summary(limit=limit).replace('\n', ' ').replace('...', '...,')

    def inspect(self, limit=10):
        """Print a summary of the series and its elements, limited to 10 items by default.

            Args:
                limit(int): the limit of elements to print, by default 10.
        """
        print(self._summary(limit=limit))

    def contents(self):
        """Get all the items of the series as a list.

        Returns:
            list: all the items of the series.
        """
        return list(self)

    def head(self, n=5):
        """Get the first n items of the series as a list, 5 by default.

            Args:
                n: the number of first elements to return.

            Returns:
                list: the required first n items of the series.
        """
        return list(self[0:n])

    def tail(self, n=5):
        """Get the last n items of the series as a list, 5 by default.

            Args:
                n: the number of last elements to return .

            Returns:
                list: the required last n items of the series.
        """
        return list(self[-n:])



#==============================
#  Time Series
#==============================

class TimeSeries(Series):
    """A list of items coming one after another over time, where every item
       is guaranteed to be of the same type and in order or succession.

       Time series accept only items of type :obj:`DataTimePoint` and :obj:`DataTimeSlot`
       (or :obj:`TimePoint` and :obj:`TimeSlot` which are useful in some circumstances),
       but can be created using some shortcuts, for example:

           * providing a Pandas Dataframe with a time-based index;

           * providing a list of dictionaries in the following forms, plus an optional ``slot_unit`` argument
             if creating a slot series (e.g. ``slot_unit='1D'``):

               * ``{60: 4, 120: 6, ... }``
               * ``{dt(1970,1,1): 4, dt(1970,1,2): 6, ... }``

           * providing a string with a path to a CSV file, which will be read and parsed using a :obj:`timeseria.storages.CSVStorage`
             storage object (in this case all the key-value arguments will be forwarded to the storage).

       The square brackets notation can be used for accessing series items, slicing the series
       or to filter it on a specific data label (if the elements support it), as outlined below.

           * ``series[3]`` will access the item in position #3;

           * ``series[1446073200.7]`` will access the item for the epoch timestamp corresponding to the (floating point)
             number provided in the square brackets;

           * ``series[dt(2015,10,25,6,19,0)]`` will access the item for the corresponding datetime timestamp.

       The same three options can be used for slicing, and filtering a series on a data label can also be achieved using
       the square bracket notation, by providing the data label on which to filter the series: ``series['temperature']`` will filter
       the time series keeping only temperature data, assuming that in the original series there were also other data labels (e.g. humidity).

       For more options for accessing and selecting series items and for slicing or filtering series, see the corresponding
       methods: :func:`select()`, :func:`slice()` and :func:`filter()`.

       Args:
           *args: the time series items, or the right object for an alternative init method as described above.
    """

    def __repr__(self):
        if not self:
            return 'Empty time series'
        else:
            if issubclass(self.item_type, TimePoint):
                return 'Time series of #{} points at {}, from point @ {} ({}) to point @ {} ({})'.format(len(self), self._resolution_string, self[0].t, self[0].dt, self[-1].t, self[-1].dt)
            elif issubclass(self.item_type, TimeSlot):
                return 'Time series of #{} slots of {}, from slot starting @ {} ({}) to slot starting @ {} ({})'.format(len(self), self.resolution, self[0].start.t, self[0].start.dt, self[-1].start.t, self[-1].start.dt)
            else:
                raise ConsistencyException('Got no TimePoints nor TimeSlots in a Time Series, this is a consistency error (got {})'.format(self.item_type.__name__))


    #=========================
    #  Init
    #=========================

    def __init__(self, *args, **kwargs):


        # Handle time zone
        tz = kwargs.pop('tz', None)
        if tz:
            self._tz = timezonize(tz)

        # Call parent init
        super(TimeSeries, self).__init__(*args, **kwargs)


    #=========================
    #  Append
    #=========================

    def append(self, item):
        """Append an item to the time series. Accepts only items of type :obj:`DataTimePoint` and :obj:`DataTimeSlot`
        (or :obj:`TimePoint` and :obj:`TimeSlot`, which are useful in some particular circumstances) and in any case
        of the same type of the items already present in the time series, unless empty."""

        if isinstance(item, TimePoint):

            try:
                self.prev_t
            except AttributeError:
                pass
            else:
                # Check time ordering and handle the resolution. It is done in this way to support
                # the deepcopy, otherwise the original prev_t will be used.
                if len(self)>0:

                    if item.t < self.prev_t:
                        raise ValueError('Time t="{}" is out of order (prev t="{}")'.format(item.t, self.prev_t))

                    if item.t == self.prev_t:
                        raise ValueError('Time t="{}" is a duplicate'.format(item.t))

                    try:
                        self._resolution

                    except AttributeError:

                        # Set the resolution as seconds for a bit of performance
                        self._resolution_as_seconds = item.t - self.prev_t

                        # Set the resolution as time unit (up to the microsecond)
                        time_diff = item.t - self.prev_t
                        time_diff = int(time_diff*10000000)/10000000
                        self._resolution = TimeUnit(_to_time_unit_string(time_diff, friendlier=True))

                    else:
                        # If the resolution is constant (not variable), check that it still is
                        if self._resolution != 'variable':
                            if self._resolution_as_seconds != (item.t - self.prev_t):
                                # ...otherwise, mark it as variable
                                del self._resolution_as_seconds
                                self._resolution = 'variable'
            finally:
                # Delete the auto-detected sampling interval cache if present
                try:
                    del self._autodetected_sampling_interval
                    del self._autodetected_sampling_interval_confidence
                except:
                    pass
                # And set the prev
                self.prev_t = item.t

        elif isinstance(item, TimeSlot):

            # Slots can belong to the same series if they are in succession (checked with the __succedes__ method)
            # and if they have the same unit, which we test here instead as the __succedes__ is more general.

            # Check the time zone (only for slots, points are not affected by time zones)
            if not self.tz:
                # If no time zone set, use the item's one
                self._tz = item.tz

            else:
                # Else, check for the same time zone
                if self._tz != item.tz:
                    # Check for the zone attribute in case the default comparison fails (e.g. because they are object with different id).
                    try:
                        if self._tz.zone != item.tz.zone:
                            raise ValueError('Cannot append slots on different time zones (I have "{}" and you tried to add "{}")'.format(self.tz, item.start.tz))
                    except AttributeError:
                        raise ValueError('Cannot append slots on different time zones (I have "{}" and you tried to add "{}")'.format(self.tz, item.start.tz))

            try:
                if self._resolution != item.unit:
                    # Try for floating point precision errors
                    abort = False
                    try:
                        if not _is_close(self._resolution, item.unit):
                            abort = True
                    except (TypeError, ValueError, NotImplementedError):
                        abort = True
                    if abort:
                        raise ValueError('Cannot add slots with a different unit than the series resolution (I have "{}" and you tried to add "{}")'.format(self._resolution, item.unit))
            except AttributeError:
                self._resolution = item.unit

        else:
            raise TypeError('Adding data to a time series only accepts TimePoints o TimeSlots (got "{}")'.format(item.__class__.__name__))

        # Lastly, call the series append
        super(TimeSeries, self).append(item)

    def _item_by_t(self, t):
        # TODO: improve performance here. Bisection first, then maybe use an index-based mapping?
        for item in self:
            if item.t == t:
                return item
        raise ValueError('Cannot find any item for t={}'.format(t))


    #=========================
    # Square brackets notation
    #=========================

    def __getitem__(self, arg):
        if isinstance(arg, slice):
            if isinstance(arg.start, int) or isinstance(arg.stop, int):
                return self.slice(from_i=arg.start, to_i=arg.stop)
            elif isinstance(arg.start, float) or isinstance(arg.stop, float):
                if arg.start is not None and not (isinstance(arg.start, Time)):
                    logger.warning('Slicing in the square brackets notation with a float works (as epoch) but it can be ambiguous in the code')
                if arg.stop is not None and not (isinstance(arg.stop, Time)):
                    logger.warning('Slicing in the square brackets notation with a float works (as epoch) but it can be ambiguous in the code')
                return self.slice(from_t=arg.start, to_t=arg.stop)
            elif isinstance(arg.start, datetime) or isinstance(arg.stop, datetime):
                return self.slice(from_dt=arg.start, to_dt=arg.stop)
            else:
                raise ValueError('Don\'t know how to slice for data type "{}"'.format(arg.start.__class__.__name__))
        elif isinstance(arg, str):
            return self.filter(arg)
        else:
            if isinstance(arg, int):
                return self.get(at_i=arg)
            elif isinstance(arg, float):
                if not isinstance(arg, Time):
                    logger.warning('Getting items in the square brackets notation with a float works (as epoch) but it can be ambiguous in the code')
                return self.get(at_t=arg)
            elif isinstance(arg, datetime):
                return self.get(at_dt=arg)
            else:
                raise ValueError('Don\'t know how to slice for data type "{}"'.format(arg.start.__class__.__name__))
            return self.get(arg)

    #=========================
    #  Time zone-related
    #=========================

    @property
    def tz(self):
        """The time zone of the time series."""
        try:
            return self._tz
        except AttributeError:
            # Detect time zone on the fly. Only applies for point time series.
            # If different time zones are mixed, than fall back on UTC.
            # TODO: set the tz at append-time for point time series as well?
            detected_tz = None
            for item in self:
                if not detected_tz:
                    detected_tz = item.tz
                else:
                    # Terrible, but there seems to be no other way to compare pytz.tzfile.* classes
                    if str(item.tz) != str(detected_tz):
                        return UTC
            return detected_tz

    def change_tz(self, tz):
        """Change the time zone of the time series, in-place.

            Args:
                str or tzinfo: the time zone.
        """
        for time_point in self:
            time_point.change_tz(tz)
        self._tz = time_point.tz

    def as_tz(self, tz):
        """Get a copy of the time series on a new time zone.

            Args:
                str or tzinfo: the time zone.

            Returns:
                TimeSeries: the time series on the new time zone.
        """
        new_series = self.duplicate()
        new_series.change_tz(tz)
        return new_series


    #=========================
    #  Resolution-related
    #=========================

    @property
    def _autodetected_sampling_interval(self):
        if not issubclass(self.item_type, TimePoint):
            raise NotImplementedError('Auto-detecting the sampling rate (and its confidence) is implemented only for point series')
        try:
            return self.__autodetected_sampling_interval
        except AttributeError:
            from .utils import detect_sampling_interval
            self.__autodetected_sampling_interval, self.__autodetected_sampling_interval_confidence = detect_sampling_interval(self, confidence=True)
            return self.__autodetected_sampling_interval

    @property
    def _autodetected_sampling_interval_confidence(self):
        if not issubclass(self.item_type, TimePoint):
            raise NotImplementedError('Auto-detecting the sampling rate (and its confidence) is implemented only for point series')
        try:
            return self.__autodetected_sampling_interval_confidence
        except AttributeError:
            from .utils import detect_sampling_interval
            self.__autodetected_sampling_interval, self.__autodetected_sampling_interval_confidence = detect_sampling_interval(self, confidence=True)
            return self.__autodetected_sampling_interval_confidence

    @property
    def resolution(self):
        """The (temporal) resolution of the time series.

        Returns a :obj:`timeseria.units.TimeUnit` object, unless:

            * the resolution is not defined (returns :obj:`None`), either because the time series is empty or because it is a point time series with only one point; or
            * the resolution is variable (returns the string ``variable``), only possible for point time series, if its points are not equally spaced, for example because of
              data losses or uneven observations.

        If the time series has a variable resolution, the `guess_resolution()` method can provide an estimate. If the time series is a slot time series, then the resolutions is just
        the unit of its slots.
        """
        try:
            return self._resolution
        except AttributeError:
            # Case of an empty or 1-point series
            return None

    def guess_resolution(self, confidence=False):
        """Guess the (temporal) resolution of the time series.

            Args:
                confidence(bool): if to return, together with the guessed resolution, also its confidence (in a 0-1 range).

            Returns:
                TimeUnit: the guessed temporal resolution, as time unit.
        """
        if not issubclass(self.item_type, TimePoint):
            raise NotImplementedError('Guessing the resolution is implemented only for point series')
        if not self:
            raise ValueError('Cannot guess the resolution for an empty time series')
        if len(self) == 1:
            raise ValueError('Cannot guess the resolution for a time series with only one point')
        if self.resolution != 'variable':
            raise ValueError('The time series has a well defined resolution ({}), guessing it does not make sense'.format(self.resolution))
        else:
            try:
                self._guessed_resolution
            except AttributeError:
                self._guessed_resolution = TimeUnit(_to_time_unit_string(self._autodetected_sampling_interval, friendlier=True))
            finally:
                if confidence:
                    return {'value': self._guessed_resolution, 'confidence': self._autodetected_sampling_interval_confidence}
                else:
                    return self._guessed_resolution

    @property
    def _resolution_string(self):
        if self.resolution is None:
            resolution_string = 'undefined resolution'
        else:
            if self.resolution == 'variable':
                autodetected_sampling_interval_as_str = TimeUnit(_to_time_unit_string(self._autodetected_sampling_interval, friendlier=True))
                resolution_string = 'variable resolution (~{})'.format(autodetected_sampling_interval_as_str)
            else:
                resolution_string = '{} resolution'.format(self.resolution)
        return resolution_string


    #=========================
    #  Load/save
    #=========================

    @classmethod
    def load(cls, file_name):
        """Load a series from a file, in Timeseria CSV format.

            Args:
                file_name(str): the file name to load.
        """

        metadata = None
        with open(file_name) as f:
            for line in f:
                if line.startswith('#'):
                    if line.startswith('# Extra parameters:'):
                        metadata = json.loads(line.replace('# Extra parameters:',''))
                else:
                    break
        if not metadata:
            raise ValueError('The file provided is not a Timeseria time series data file. Perhaps you wanted to use the from_csv() method?')

        force_points = True if metadata['type'] == 'points' else False
        force_slots = True if metadata['type'] == 'slots' else False
        force_tz = metadata['tz']
        if force_slots:
            force_slot_unit = metadata['resolution']
        else:
            force_slot_unit = None

        from .storages import CSVFileStorage
        storage = CSVFileStorage(file_name)
        loaded_series = storage.get(force_points=force_points, force_slots=force_slots, force_tz=force_tz, force_slot_unit=force_slot_unit)

        if loaded_series.__class__ == cls:
            return loaded_series
        else:
            # TODO: improve performance here, the following is highly inefficient.
            series_items = loaded_series.contents()
            cls(*series_items)


    def save(self, file_name, overwrite=False, **kwargs):
        """Save the time series as a file, in Timeseria CSV format.

            Args:
                file_name(str): the file name to write.
                overwrite(bool): if to overwrite the file if already existent. Defaults to False.
        """
        from .storages import CSVFileStorage
        storage = CSVFileStorage(file_name, **kwargs)
        storage.put(self, overwrite=overwrite)


    #=========================
    #  Conversions
    #=========================

    @classmethod
    def from_dict(cls, dictionary, slot_unit=None):
        """Create a time series from a dictionary.

            Args:
                dict(bool): the dictionary containing the data, where the keys are the timestamps.

            Returns:
                TimeSeries: the created time series.
        """

        if slot_unit and not isinstance(slot_unit, Unit):
                slot_unit = TimeUnit(slot_unit)
        series = cls()

        for key in dictionary:
            item_dt = None
            item_t = None
            if isinstance(key, datetime):
                item_dt = key
            elif isinstance(key, int) or isinstance(key,float):
                item_t = key
            elif isinstance(key, str):
                try:
                    item_t = float(key)
                except ValueError:
                    try:
                        item_t = s_from_dt(dt_from_str(key))
                    except Exception as e:
                        raise e from None
            else:
                raise ValueError('Cannot handle keys other than int, float or datetime (got "{}"'.format(key.__class__.__name__))
            data = dictionary[key]
            if not (isinstance(data, list) or isinstance(data, dict)):
                data = {'value': data}

            # Create the item
            if slot_unit:
                if item_dt is not None:
                    series.append(DataTimeSlot(dt=item_dt, unit=slot_unit, data=data))
                else:
                    series.append(DataTimeSlot(t=item_t, unit=slot_unit, data=data))
            else:
                if item_dt is not None:
                    series.append((DataTimePoint(dt=item_dt, data=data)))
                else:
                    series.append((DataTimePoint(t=item_t, data=data)))

        return series 

    def to_dict(self):
        """Convert the time series to a dictionary.

            Returns:
                dict: the time series as a dictrionary.
        """
        timeseries_as_dict = {}
        for item in self:
            timeseries_as_dict[item.dt] = item.data
        return timeseries_as_dict


    @classmethod
    def from_json(cls, string, slot_unit=None):
        """Create a time series from a JSON string.

            Args:
                string(str): the string containing the JSOn data, where the keys are the timestamps.

            Returns:
                TimeSeries: the created time series.
        """
        json_data = json.loads(string)
        return cls.from_dict(json_data, slot_unit=slot_unit)


    def to_json(self):
        """Convert the time series to a JSON string.

            Returns:
                string: the time series as a JSON string.
        """

        timeseries_as_dict = {}
        for item in self:
            timeseries_as_dict[str_from_dt(item.dt)] = item.data
        return json.dumps(timeseries_as_dict)


    @classmethod
    def from_csv(cls, file_name, *args, **kwargs):
        """Create a time series from a CSV file. For the options see the storages module.

            Returns:
                TimeSeries: the created time series.
        """
        from .storages import CSVFileStorage
        storage = CSVFileStorage(file_name, *args, **kwargs)
        loaded_series = storage.get()

        if loaded_series.__class__ == cls:
            return loaded_series
        else:
            # TODO: improve performance here, the following is highly inefficient.
            series_items = loaded_series.contents()
            cls(*series_items)

    def to_csv(self, file_name, overwrite=False, **kwargs):
        """Store the time series as a CSV file. For the options see the storages module.

            Args:
                file_name(str): the file name to write.
                overwrite(bool): if to overwrite the file if already existent. Defaults to False.
        """
        from .storages import CSVFileStorage
        storage = CSVFileStorage(file_name, **kwargs)
        storage.put(self, overwrite=overwrite)


    @classmethod
    def from_df(cls, df, item_type='auto'):
        """Create a time series from a Pandas data frame.

            Args:
                df(DataFrame): the Pandas data frame.
                item_type(DataTimePoint or DataTimeSlot or str): the type of the items of the newly
                                                                 created time series. Defaults to 'auto'.

            Returns:
                TimeSeries: the created time series.
        """

        if item_type == 'auto':
            item_type = None

        # Infer if we have to create points or slots and their unit
        unit_str_pd=df.index.inferred_freq

        if not unit_str_pd:
            if not item_type:
                logger.info('Cannot infer the frequency of the dataframe, will just create points')
                item_type = DataTimePoint

        else:

            # Calendar
            unit_str_pd=unit_str_pd.replace('A', 'Y')    # Year (end) ?
            unit_str_pd=unit_str_pd.replace('Y', 'Y')    # Year (end) )
            unit_str_pd=unit_str_pd.replace('AS', 'Y')   # Year (start)
            unit_str_pd=unit_str_pd.replace('YS', 'Y')   # Year (start)
            unit_str_pd=unit_str_pd.replace('MS', 'M')   # Month (start)
            unit_str_pd=unit_str_pd.replace('M', 'M')    # Month (end)
            unit_str_pd=unit_str_pd.replace('D', 'D')    # Day

            # Physical
            unit_str_pd=unit_str_pd.replace('H', 'h')    # Hour
            unit_str_pd=unit_str_pd.replace('T', 'm')    # Minute
            unit_str_pd=unit_str_pd.replace('min', 'm')  # Minute
            unit_str_pd=unit_str_pd.replace('S', 's')    # Second

            if len(unit_str_pd) == 1:
                unit_str = '1'+unit_str_pd
            else:
                unit_str = unit_str_pd

            unit_type = ''.join([char for char in unit_str if not char.isdigit()])

            if unit_type in ['Y', 'M', 'D']:
                logger.info('Assuming slots with a slot time unit of "{}"'.format(unit_str))
                if not item_type:
                    item_type = DataTimeSlot
                else:
                    if item_type==DataTimePoint:
                        raise ValueError('Creating points with calendar time units is not supported.')
            else:
                if not item_type:
                    item_type = DataTimePoint

        # Now create the points or the slots
        if item_type==DataTimeSlot:

            # Init the unit
            unit = TimeUnit(unit_str)

            # Create data time points list
            datatimeslots = []

            # Get data frame labels
            labels = list(df.columns)

            naive_warned = False
            for row in df.iterrows():

                # Set the timestamp
                if not isinstance(row[0], datetime):
                    raise TypeError('A DataFrame with a DateTime index column is required')
                dt = row[0]

                # Prepare data
                data = {}
                for i,label in enumerate(labels):
                    data[label] = row[1][i]

                if dt.tzinfo is None:
                    if not naive_warned:
                        logger.warning('Got naive datetimes as dataframe index, assuming UTC.')
                        naive_warned = True
                    dt = UTC.localize(dt)

                datatimeslots.append(DataTimeSlot(dt=dt, unit=unit, data=data))

            # Set the list of data time points
            return cls(*datatimeslots)

        else:
            # Create data time points list
            datatimepoints = []

            # Get data frame labels
            labels = list(df.columns)

            naive_warned = False
            for row in df.iterrows():

                # Set the timestamp
                if not isinstance(row[0], datetime):
                    raise TypeError('A DataFrame with a DateTime index column is required')
                dt = row[0]

                # Prepare data
                data = {}
                for i,label in enumerate(labels):
                    data[label] = row[1][i]

                if dt.tzinfo is None:
                    if not naive_warned:
                        logger.warning('Got naive datetimes as dataframe index, assuming UTC.')
                        naive_warned = True
                    dt = UTC.localize(dt)

                datatimepoints.append(DataTimePoint(dt=dt, data=data))

            # Set the list of data time points
            return cls(*datatimepoints)

    def to_df(self):
        """Convert the time series as a Pandas data frame.

            Returns:
                DataFrame: the Pandas data frame.
        """
        data_labels = self.data_labels()

        dump_data_loss = False
        for item in self:
            if item.data_loss is not None:
                dump_data_loss = True
                break

        if dump_data_loss:
            columns = ['Timestamp'] + data_labels + ['data_loss']
        else:
            columns = ['Timestamp'] + data_labels

        df = DataFrame(columns=columns)
        for item in self:
            values = [item.data[data_label] for data_label in data_labels]
            if dump_data_loss:
                df = concat([df, DataFrame([[item.dt]+values+[item.data_loss]], columns=columns)])
            else:
                df = concat([df, DataFrame([[item.dt]+values], columns=columns)])
        df = df.set_index('Timestamp')

        return df


    #=========================
    #  Operations
    #=========================


    def get(self, at_i=None, at_t=None, at_dt=None):
        """Get the element of the series at a given position or at a given time.

        Args:
           at_i(int): the position of the item to get. Defaults to None.
           at_t(bool): the time (as epoch seconds) of the item to get. Defaults to None.
           at_dt(bool): the time (as datetime object) of the item to get. Defaults to None.

        Returns:
            object: the item in the given position or at the given time.
        """
        from .operations import get as get_operation
        return get_operation(self, at_i, at_t, at_dt)


    def slice(self, from_i=None, to_i=None, from_t=None, to_t=None, from_dt=None, to_dt=None):
        """Slice the series between the given positions or times. A series of DataPoints or DataSlots is required.

        Args:
           from_i(int): the slicing start position. Defaults to None.
           to_i(int): the slicing end position. Defaults to None.
           from_t(bool): the slicing start time (as epoch seconds). Defaults to None.
           to_t(bool): the slicing end time (as epoch seconds). Defaults to None.
           from_dt(bool): the slicing start time (as datetime object). Defaults to None.
           to_dt(bool): the slicing end time (as datetime object). Defaults to None.

        Returns:
            Series: the sliced series.
        """
        from .operations import slice as slice_operation
        return slice_operation(self, from_i=from_i, to_i=to_i,
                                     from_t=from_t, to_t=to_t,
                                     from_dt=from_dt, to_dt=to_dt)


    def view(self, from_i=None, to_i=None):
        """Get a view of the time series.

        Args:
           from_i(int): the view start position. Defaults to None.
           to_i(int): the view end position. Defaults to None.

        Returns:
            TimeSeriesView: the time series view.
        """
        return TimeSeriesView(series=self, from_i=from_i, to_i=to_i)


    #=========================
    #  Plot-related
    #=========================

    def plot(self, engine='dg', *args, **kwargs):
        """Plot the time series. The default plotting engine is Dygraphs (``engine=\'dg\'``),
           limited support for Matplotplib (``engine=\'mp\'``) is also available.
           For plotting options for Dygraphs, see :func:`~.plots.dygraphs_plot`, while for
           plotting options for Matplotlib, see :func:`~.plots.matplotlib_plot`."""
        if engine=='mp':
            from .plots import matplotlib_plot
            matplotlib_plot(self, *args, **kwargs)
        elif engine=='dg':
            from .plots import dygraphs_plot
            out = dygraphs_plot(self, *args, **kwargs)
            if out: return out
        else:
            raise ValueError('Unknown plotting engine "{}'.format(engine))



#==============================
#  Time Series view
#==============================

class TimeSeriesView(TimeSeries):
    """A time series created as a view of another one.

        Args:
           series(TimeSeries): the original time series.
           from_i(int): the view start position. Defaults to None.
           to_i(int): the view end position. Defaults to None.
    """

    def __init__(self, *items, series=None, from_i=None, to_i=None, **kwargs):
        self.series = series
        self.from_i = from_i
        self.to_i = to_i
        self.len = None
        # Call parent init in case we are using the series view as a normal series
        super().__init__(*items, **kwargs)

    def __getitem__(self, arg):

        if self.series:

            if isinstance(arg, slice):

                if isinstance(arg.start, int):
                    if arg.start>=0:
                        slice_start =  self.from_i + arg.start
                    else:
                        slice_start =  self.to_i - abs(arg.start)
                elif arg.start is None:
                    slice_start = self.from_i
                else:
                    slice_start = arg.start

                if isinstance(arg.stop, int):
                    if arg.stop>=0:
                        slice_stop =  self.from_i + arg.stop
                    else:
                        slice_stop =  self.to_i - abs(arg.stop)
                elif arg.stop is None:
                    slice_stop = self.to_i
                else:
                    slice_stop = arg.stop

                return self.series[slice_start:slice_stop]

            else:
                if isinstance(arg, int):
                    if arg>=0:
                        return self.series[self.from_i + arg]
                    else:
                        return self.series[self.to_i - abs(arg)]
                else:
                    return super().__getitem__(arg)
        else:
            return super().__getitem__(arg)

    def __iter__(self):
        if self.series:
            self.iter_count = 0
            return self
        else:
            return super().__iter__()

    def __next__(self):
        if self.series:
            this_i = self.iter_count + self.from_i

            if this_i >= self.to_i:
                # If reached the end stop
                raise StopIteration

            else:
                # Otherwise return the corresponding item
                self.iter_count += 1
                return self.series[this_i]
        else:
            return super().__next__()

    def __len__(self):
        if self.series:
            return self.to_i-self.from_i
        else:
            return super().__len__()

    def __repr__(self):
        if self.series:
            return 'TimeSeriesView from element #{} to element #{} of: {}'.format(self.from_i, self.to_i, self.series)
        else:
            return super().__repr__()

    def __bool__(self):
        if self.series is not None:
            return True if len(self.series) else False
        else:
            return True if len(self) else False

    @property
    def item_type(self):
        if self.series:
            return self.series.item_type
        else:
            return super().item_type

    @property
    def resolution(self):
        if self.series:
            return self.series.resolution
        else:
            return super().resolution

    def data_labels(self):
        if self.series:
            return self.series.data_labels()
        else:
            return super().data_labels()

    def materialize(self):
        """Materialize the time series view.

        Returns:
            TimeSeries: the time series corresponding to the materialized view.
        """
        materialized_series = self.__class__()
        for item in self:
            materialized_series.append(deepcopy(item))
        return materialized_series

    def duplicate(self):
        return self.materialize()

