# -*- coding: utf-8 -*-
"""Base data structures as Points, Slots and Series."""

from .units import Unit, TimeUnit
from .utilities import _is_close, _to_time_unit_string
from copy import deepcopy
from pandas import DataFrame
from datetime import datetime
from .exceptions import ConsistencyException
from propertime.utilities import s_from_dt , dt_from_s, timezonize, dt_from_str
from pytz import UTC

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

        # Handle timezone if any (removing it from kwargs)
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
                
                # If we do not have a timezone, can we use the one from the dt used to initialize this TimePoint?
                try:
                    self._tz
                except AttributeError:
                    if kwargs['dt'].tzinfo:
                        
                        #Do not set it if it is UTC, it is the default
                        if kwargs['dt'].tzinfo == UTC:
                            pass
                        else:
                            self._tz = kwargs['dt'].tzinfo
                            #raise NotImplementedError('Not yet tz from dt ("{}")'.format(kwargs['dt']))
                
            #else:
            #    raise Exception('Don\'t know how to handle all kwargs (got "{}")'.format(kwargs))

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
        """The timezone."""
        try:
            return self._tz
        except AttributeError:
            return UTC

    def zone(self, tz):
        """This method is deprecated in favor of change_tz().
        
        :meta private:
        """
        logger.warning('The change_timezone() method is deprecated in favor of change_tz().')
        self.change_tz(tz)

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
        #else:
        #    if None in data_indexes.values():
        #        raise ValueError('Cannot have an index set to None: do not set it at all ({})'.format(data_indexes))

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
        # Data is set like this as it cannot be set if not in the init (read: changed after created)
        # to prevent this to happend when the point is in a series where they are all supposed
        # to carry the same data type and with the same number of elements. TODO: check!
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
            #raise AttributeError('No data loss index set for this point')
            return None
        
    def data_labels(self):
        """Return the data labels. If data is a dictionary, then these are the dictionary keys, if data 
        is list-like, then these are the list indexes (as strings). Other formats are not supported."""
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
    
    def __init__(self, start, end=None, unit=None):
        
        if not isinstance(start, self.__POINT_TYPE__):
            raise TypeError('Slot start must be a Point object (got "{}")'.format(start.__class__.__name__))
        
        # Instantiate unit if not already done
        if unit and not isinstance(unit, Unit):
            unit = Unit(unit)
        
        if end is None and unit is not None:
            if len(start.coordinates)>1:
                raise Exception('Sorry, setting a start and a unit only works in unidimensional spaces')
            
            # Handle start + unit
            end = start + unit
             
        if not isinstance(end, self.__POINT_TYPE__):
            raise TypeError('Slot end must be a Point object (got "{}")'.format(end.__class__.__name__))
        
        # TODO: remove the following check, or make it optional (i.e. not used by TimeSlots)?
        if len(start.coordinates) != len(end.coordinates):
            raise ValueError('Slot start and end dimensions must be the same (got "{}" vs "{}")'.format(start.coordinates, end.coordinates))
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
                return Unit(self.end.coordinates[0] - self.start.coordinates[0])
            else:
                values = []
                for i in range(len(self.start.coordinates)):
                    values.append(self.end.coordinates[i] - self.start.coordinates[i])
                
                return Unit(values)


class TimeSlot(Slot):
    """A slot in the time dimension. Can be initialized with start and end
       or start and unit.
    
       Args:
           start(TimePoint): the slot starting time point.
           end(TimePoint): the slot ending time point.
           unit(TimeUnit): the slot time unit."""

    __POINT_TYPE__ = TimePoint

    def __init__(self, start=None, end=None, unit=None, **kwargs):
        
        # Internal-use init
        t = kwargs.get('t', None)
        dt = kwargs.get('dt', None)
        tz = kwargs.get('tz', None)
        
        # Handle t and dt shortcuts
        if t:
            start=TimePoint(t=t, tz=tz)
        if dt:
            start=TimePoint(dt=dt, tz=tz)
        
        if not isinstance(start, self.__POINT_TYPE__):
            raise TypeError('Slot start must be a {} object (got "{}")'.format(self.__POINT_TYPE__.__name__, start.__class__.__name__))

        try:
            if start.tz != end.tz:
                raise ValueError('{} start and end must have the same timezone (got start.tz="{}", end.tz="{}")'.format(self.__class__.__name__, start.tz, end.tz))
        except AttributeError:
            if end is None:
                # We are using the Unit, use the start
                self.tz = start.tz
            else:
                # If we don't have a timezone, we don't have TimePoints, the parent will make the Slot creation fail with a TypeError
                pass
        else:    
            self.tz = start.tz
        
        # Convert unit to TimeUnit
        if unit and not isinstance(unit, TimeUnit):
            unit = TimeUnit(unit)
                
        # Call parent init
        super(TimeSlot, self).__init__(start=start, end=end, unit=unit)
        
        # If we did not have the end, set its timezone now:
        if end is None:
            self.end.change_tz(self.start.tz)

    # Overwrite parent succedes, this has better performance as it checks for only one dimension
    def __succedes__(self, other):
        if other.end.t != self.start.t:
            # Take into account floating point rounding errors
            if _is_close(other.end.t, self.start.t):
                return True
            return False
        else:
            return True

    def change_timezone(self, tz):
        """This method is deprecated in favor of change_tz().
        
        :meta private:
        """
        logger.warning('The change_timezone() method is deprecated in favor of change_tz().')
        self.change_tz(tz)

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
        #else:
        #    if None in data_indexes.values():
        #        raise ValueError('Cannot have an index set to None: do not set it at all ({})'.format(data_indexes))

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
        # Data is set like this as it cannot be set if not in the init (read: changed after created)
        # to prevent this to happened when the point is in a series where they are all supposed
        # to carry the same data type and with the same number of elements. TODO: check me!
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
            #raise AttributeError('No data loss index set for this point')
            return None

    def data_labels(self):
        """Return the data labels. If data is a dictionary, then these are the dictionary keys, if data 
        is list-like, then these are the list indexes (as strings). Other formats are not supported."""
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
       
       The square brackets notation can be used both for slicing the series
       or to filter it on a specific data label, if its elements support it.

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
    def items_type(self):
        """The type of the items of the series."""
        if not self:
            return None
        else:
            return self[0].__class__

    def append(self, item):
        """Append an item to the series. Accepts only items of the same
        type of the items already present in the series (unless empty)"""
        
        # TODO: move to use the insert
        
        # logger.debug('Checking %s', item)
        
        # Check type
        if self.items_type:
            if not isinstance(item, self.items_type):
                raise TypeError('Got incompatible type "{}", can only accept "{}"'.format(item.__class__.__name__, self.items_type.__name__))
        
        # Check order or succession
        try:
            try:
                if not item.__succedes__(self[-1]):
                    raise ValueError('Not in succession ("{}" does not succeeds "{}")'.format(item,self[-1])) from None                    
            except IndexError:
                raise
            except AttributeError:
                try:
                    if not item > self[-1]:
                        raise ValueError('Not in order ("{}" does not follow "{}")'.format(item,self[-1])) from None
                except TypeError:
                    raise TypeError('Object of class "{}" does not implement a "__gt__" or a "__succedes__" method, cannot append it to a Series (which is ordered)'.format(item.__class__.__name__)) from None

        except IndexError:
            pass

        # Check data type if any
        try:
            if self._item_data_reference:
                
                if not type(self._item_data_reference) == type(item.data):
                    raise TypeError('Got different data: {} vs {}'.format(self._item_data_reference.__class__.__name__, item.data.__class__.__name__))
                if isinstance(self._item_data_reference, list):
                    if len(self._item_data_reference) != len(item.data):
                        raise ValueError('Got different data lengths: {} vs {}'.format(len(self._item_data_reference), len(item.data)))
                if isinstance(self._item_data_reference, dict):
                    if set(self._item_data_reference.keys()) != set(item.data.keys()):
                        raise ValueError('Got different data labels: {} vs {}'.format(self._item_data_reference.keys(), item.data.keys()))
        except AttributeError:
            # logger.debug('Setting data reference: %s', item.data)
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

    @property
    def title(self):
        """A title for the series, to be used for the plots."""
        try: 
            return self._title
        except AttributeError:
            return None

    @title.setter
    def title(self, title):
        self._title=title

    def __repr__(self):
        return '{} of #{} elements'.format(self.__class__.__name__, len(self))
    
    def __str__(self):
        return self.__repr__()
        
    def _all_data_indexes(self):
        """Return all the data_indexes of the series, to be intended as custom
        defined indicators (i.e. data_loss, anomaly_index, etc.)."""
        
        # TODO: move this to the Data*Series...?
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

    @property
    def mark(self):
        """A mark for the series, used for highlighting a portion of a plot.
           Required to be formatted as a list or tuple with two elements, the
           first from where the mark has to start and the second where it has
           to end.
        """
        try:
            return self._mark
        except AttributeError:
            return None

    @mark.setter
    def mark(self, value):
        if not value:
            try:
                del self._mark
            except:
                pass
        else:
            # Check valid mark
            if not isinstance(value, (list, tuple)):
                raise TypeError('Series mark must be a list or tuple')
            if not len(value) == 2:
                raise ValueError('Series mark must be a list or tuple of two elements')    
            self._mark = value

    @property
    def mark_title(self):
        """A tile for the mark, to be displayed in the plot legend."""
        try:
            return self._mark_title
        except AttributeError:
            return None

    @mark_title.setter
    def mark_title(self, value):
        if not value:
            del self._mark_title
        else: 
            self._mark_title = value

    # Inherited methods to be edited
    def insert(self, i, x):
        """Insert an item at a given position. The first argument is the index of the element 
        before which to insert, so series.insert(0, x) inserts at the front of the series, and 
        series.insert(len(series), x) is equivalent to append(x). Order or succession are enforced."""
        
        if len(self) > 0:
            
            # Check valid type
            if not isinstance(x, self.items_type):
                raise TypeError('Got incompatible type "{}", can only accept "{}"'.format(x.__class__.__name__, self.items_type.__name__))
        
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

    def duplicate(self):
        """ Return a deep copy of the series."""
        return deepcopy(self)


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
            try:
                # Preserve mark if any
                series.mark = self.mark
            except:
                pass
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
        """Returns the labels of the data carried by the series points or slots. If data is a dictionary,
        then these are the dictionary keys, if data  is list-like, then these are the list indexes
        (as strings). Other formats are not supported."""
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
            # TODO: move to the DataPoint/DataSlot?
            item.data[new_data_label] = item.data.pop(old_data_label)

    def remove_data_loss(self):
        """Remove the ``data_loss`` index, in-place."""
        if len(self) > 0 and not self._item_data_reference:
            raise TypeError('Series items have no data, cannot remove the data loss')
        for item in self:
            item.data_indexes.pop('data_loss', None)

    def remove_data_index(self, data_index):
        """Remove a data index, in-place."""
        if len(self) > 0 and not self._item_data_reference:
            raise TypeError('Series items have no data, cannot rename data indexes')
        for item in self:
            item.data_indexes.pop(data_index, None)     


    #=========================
    #  Operations
    #=========================

    def min(self, data_label=None):
        """Get the minimum data value(s) of a series. Supports an optional ``data_label`` argument.
        A series of DataPoints or DataSlots is required."""
        from .operations import min as min_operation
        return min_operation(self, data_label=data_label)

    def max(self, data_label=None):
        """Get the maximum data value(s) of a series. Supports an optional ``data_label`` argument.
        A series of DataPoints or DataSlots is required."""
        from .operations import max as max_operation
        return max_operation(self, data_label=data_label)    

    def avg(self, data_label=None):
        """Get the average data value(s) of a series. Supports an optional ``data_label`` argument.
        A series of DataPoints or DataSlots is required."""
        from .operations import avg as avg_operation
        return avg_operation(self, data_label=data_label)   

    def sum(self, data_label=None):
        """Sum every data value(s) of a series. Supports an optional ``data_label`` argument.
        A series of DataPoints or DataSlots is required."""
        from .operations import sum as sum_operation
        return sum_operation(self, data_label=data_label)  

    def derivative(self, inplace=False, normalize=True, diffs=False):
        """Compute the derivative of the series. Extra parameters: ``inplace`` (defaulted to false),
        ``normalize`` (defaulted to true) and ``diffs`` (defaulted to false) to compute differences
        instead of the derivative. A series of DataTimePoints or DataTimeSlots is required."""
        from .operations import derivative as derivative_operation
        return derivative_operation(self, inplace=inplace, normalize=normalize, diffs=diffs)   

    def integral(self, inplace=False, normalize=True, c=0, offset=0):
        """Compute the integral of the series. Extra parameters: ``inplace`` (defaulted to false),
        ``normalize`` (defaulted to true), ``c`` (defaulted to zero) for the integration constant
        and ``offset`` (defaulted to zero) to start the integration from an offset.
        A series of DataTimePoints or DataTimeSlots is required."""
        from .operations import integral as integral_operation
        return integral_operation(self, inplace=inplace, normalize=normalize, c=c, offset=offset)   

    def diff(self, inplace=False):
        """Compute the incremental differences. Reduces the series length by one (the first element).
        Extra parameters: ``inplace`` (defaulted to false). A series of DataPoints or DataSlots is required."""
        from .operations import diff as diff_operation
        return diff_operation(self, inplace=inplace)   

    def csum(self, inplace=False, offset=None):
        """Compute the incremental sum. Extra parameters: ``inplace`` (defaulted to false),
        ``offset`` (defaulted to zero) to set the starting value where to apply the sums on.
        A series of DataPoints or DataSlots is required."""
        from .operations import csum as csum_operation
        return csum_operation(self, inplace=inplace, offset=offset)

    def normalize(self, range=[0,1], inplace=False):
        """Normalize the series data values. Extra parameters: ``range`` (defaulted to [0,1]) to set the normalization
        range and ``inplace`` (defaulted to false). A series of DataPoints or DataSlots is required."""
        from .operations import normalize as normalize_operation
        return normalize_operation(self, inplace=inplace, range=range)   

    def rescale(self, value, inplace=False):
        """Rescale the series data values by a ``value``. This can be either a single number or a dictionary
        where to set rescaling factors on a per-data label basis. Extra parameters: ``inplace`` (defaulted to false)
        A series of DataPoints or DataSlots is required."""
        from .operations import rescale as rescale_operation
        return rescale_operation(self, value=value, inplace=inplace)

    def offset(self, value, inplace=False):
        """Offset the series data values by a ``value``. This can be either a single number or a dictionary
        where to set offsetting factors on a per-data label basis. Extra parameters: ``inplace`` (defaulted to false).
        A series of DataPoints or DataSlots is required."""
        from .operations import offset as offset_operation
        return offset_operation(self, value=value, inplace=inplace)  

    def mavg(self,  window, inplace=False):
        """Compute the moving average. Reduces the series length by n (the window size). Extra
        parameters: ``inplace`` (defaulted to false) and ``window``, a required parameter, for
        the size of the moving average window. A series of DataPoints or DataSlots is required."""
        from .operations import mavg as mavg_operation
        return mavg_operation(self, window=window, inplace=inplace)

    def merge(self, series):
        """Merge the series with one or more other series."""
        from .operations import merge as merge_operation
        return merge_operation(self, series)

    def filter(self, data_label, **kwargs):
        """Filter a series by a ``data_label``. A series of DataPoints or DataSlots is required."""
        # TODO: refactor this to allow generic item properties?     
        from_t = kwargs.get('from_t', None)
        to_t = kwargs.get('to_t', None)
        from_dt = kwargs.get('from_dt', None)
        to_dt = kwargs.get('to_dt', None)
        from .operations import filter as filter_operation
        return filter_operation(self, data_label=data_label, from_t=from_t, to_t=to_t, from_dt=from_dt, to_dt=to_dt) 

    def slice(self, start=None, end=None, **kwargs):
        """Slice a series from a "start" to an "end". A series of DataPoints or DataSlots is required."""
        from_t = kwargs.get('from_t', None)
        to_t = kwargs.get('to_t', None)
        from_dt = kwargs.get('from_dt', None)
        to_dt = kwargs.get('to_dt', None)
        from .operations import slice as slice_operation
        return slice_operation(self, start=start, end=end, from_t=from_t, to_t=to_t, from_dt=from_dt, to_dt=to_dt) 

    def select(self, query):
        """Select one or more items of the series given an SQL-like query. A series of DataPoints or DataSlots is required."""
        from .operations import select as select_operation
        return select_operation(self, query=query)


    #=========================
    #  Transformations
    #=========================
    
    def aggregate(self, unit, *args, **kwargs):
        """Aggregate the series in slots of length set by the ``unit`` parameter. A series of DataPoints or DataSlots is required.""" 
        from .transformations import Aggregator
        aggregator = Aggregator(unit, *args, **kwargs)
        return aggregator.process(self)  

    def resample(self, unit, *args, **kwargs):
        """Resample the series using a sampling interval of length set by the ``unit`` parameter. A series of DataPoints or DataSlots is required."""
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
        """Return a string summary of the series and its elements, limited to 10 items by default.
        
            Args:
                limit: the limit of elements to print, by default 10.
                newlines: if to include the newline characters or not.
        """
        if newlines:
            return self._summary(limit=limit)
        else:
            return self._summary(limit=limit).replace('\n', ' ').replace('...', '...,')

    def inspect(self, limit=10):
        """Print a summary of the series and its elements, limited to 10 items by default.
        
            Args:
                limit: the limit of elements to print, by default 10.
        """
        print(self._summary(limit=limit))

    def contents(self):
        """Get all the items of the series as a list."""   
        return list(self)  

    def head(self, n=5):
        """Get the first n items of the series as a list, 5 by default.
        
            Args:
                n: the number of first elements to return .
        """
        return list(self[0:n])

    def tail(self, n=5):
        """Get the last n items of the series as a list, 5 by default.
        
            Args:
                n: the number of last elements to return .
        """
        return list(self[-n:])
 


#==============================
#  Time Series 
#==============================

class TimeSeries(Series):
    """A list of items coming one after another over time, where every item
       is guaranteed to be of the same type and in order or succession.
       
       Time series accept only items of type :obj:`DataTimePoint` and :obj:`DataTimeSlot`
       (or :obj:`TimePoint` and :obj:`TimeSlot` which are useful in some particular circumstances), 
       but can be created using some shortcuts, for example:
       
           * providing a Pandas Dataframe with a time-based index;
            
           * providing a list of dictionaries in the following forms, plus an optional ``slot_unit`` argument
             if creating a slot series (e.g. ``slot_unit='1D'``):
           
               * ``[{'t':60, 'data':4}, {'t':120, 'data':6}, ... ]``
               * ``[{'dt':dt(1970,1,1), 'data':4}, {'dt':dt(1970,1,2), 'data':6}, ... ]`` 
             
           * providing a string with a CSV file name, inclding its path, that will in turn use a
             :obj:`timeseria.storages.CSVStorage`, in which case all the key-value parameters will be
             forwarded to the storage object.
       
       The square brackets notation can be used for accessing series items, slicing the series
       or to filter it on a specific data label (if its elements support it), as outlined below.
       
       Accessing time series items can be done by position, using a string with special ``t`` or ``dt``
       keywords, or using a dictionary with ``t`` or ``dt`` keys:
       
           * ``series[3]`` will access the item in position #3;
           
           * ``series['t=1446073200.7']`` and ``series[{'t': 1446073200.7}]`` will both access the item
             for the corresponding epoch timestamp;
             
           * ``series['dt=2015-10-25 06:19:00+01:00']`` and ``series[{'dt': dt(2015,10,25,6,19,0,0)}]`` will both
             access the item for the corresponding datetime timestamp. In the string notation, this can be both an
             ISO8601 timestamp or a string representation of the datetime object.

       Slicing a series works in a similar fashion, and accepts in the coulmn-separated square bracket notation ``series[start:end]``
       item positions, strings with special ``t`` or ``dt`` keywords, or dictionaries with ``t`` or ``dt`` keys, as above.

       Filtering a series on a data label can also be achieved using the square bracket notation, by providing the
       data label on which to filter the series: ``series['temperature']`` will filter the time series keeping only
       temperature data, assuming that in the original series there were also other data labels (e.g. humidity).

       For more options for accessing and selecting series items and for slicing or filtering series, see the corresponding
       methods: :func:`select()`, :func:`slice()` and :func:`filter()`.

       Args:
           *args: the time series items, or the right object for an alternative init method as described above.
    """

    def __repr__(self):
        if not self:
            return 'Empty time series'
        else:
            if issubclass(self.items_type, TimePoint):
                return 'Time series of #{} points at {}, from point @ {} ({}) to point @ {} ({})'.format(len(self), self._resolution_string, self[0].t, self[0].dt, self[-1].t, self[-1].dt)
            elif issubclass(self.items_type, TimeSlot):
                return 'Time series of #{} slots of {}, from slot starting @ {} ({}) to slot starting @ {} ({})'.format(len(self), self.resolution, self[0].start.t, self[0].start.dt, self[-1].start.t, self[-1].start.dt)            
            else:
                raise ConsistencyException('Got no TimePoints nor TimeSlots in a Time Series, this is a consistency error (got {})'.format(self.items_type.__name__))
    

    #=========================
    #  Init
    #=========================
    
    def __init__(self, *args, **kwargs):

        # Handle special df kwarg, mainly for back compatibility
        df = kwargs.pop('df', None)
              
        # Create from Data Frame
        if df or (args and (isinstance(args[0], DataFrame))):
            if not df:
                df = args[0]
            items_type = kwargs.pop('items_type', None)

            # Infer if we have to create points or slots and their unit
            unit_str_pd=df.index.inferred_freq
    
            if not unit_str_pd:
                if not items_type:
                    logger.info('Cannot infer the freqency of the dataframe, will just create points')
                    items_type = DataTimePoint
                
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
                
                unit=TimeUnit(unit_str)
                
                if unit.is_calendar():
                    logger.info('Assuming a slot time unit of "{}"'.format(unit_str))
                    if not items_type:
                        items_type = DataTimeSlot
                    else:
                        if items_type==DataTimePoint:
                            raise ValueError('Creating points with calendar time units is not supported.')
                else:
                    if not items_type:
                        items_type = DataTimePoint
            
            # Now create the points or the slots      
            if items_type==DataTimeSlot:
            
                # Create data time points list
                data_time_slots = []
    
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

                    data_time_slots.append(DataTimeSlot(dt=dt, unit=TimeUnit(unit_str), data=data))
    
                # Set the list of data time points
                super(TimeSeries, self).__init__(*data_time_slots, **kwargs)
            
            else:
                # Create data time points list
                data_time_points = []
                
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
                    
                    data_time_points.append(DataTimePoint(dt=dt, data=data))
                
                # Set the list of data time points
                super(TimeSeries, self).__init__(*data_time_points, **kwargs)
        
        # Create from list of dicts
        elif args and (isinstance(args[0], list)):
            
            slot_unit = kwargs.pop('slot_unit', None)
            if slot_unit and not isinstance(slot_unit, Unit):
                slot_unit = TimeUnit(slot_unit)
            series_items = []
            
            for item in args[0]:
                
                if not isinstance(item, dict):
                    raise TypeError('List of dicts required, got "{}"'.format(item.__class__.__name__))
                
                item_t = item.get('t', None)
                item_dt = item.get('dt', None)
                if item_t is None and item_dt is None:
                    raise ValueError('A "t" or "dt" key is required')
                
                data = item.get('data', None)
                if not data:
                    raise ValueError('A "data" key is required')
                if not (isinstance(data, list) or isinstance(data, dict) or isinstance(data, tuple)):
                    data = [data]

                # Possible alternative approach
                # value = item.get('value', None)
                # values = item.get('values', None)
                # if not value and not values:
                #     raise ValueError('A "value" or "values" key is required')
                # if values:
                #     if not (isinstance(values, list) or isinstance(values, dict) or isinstance(values, tuple)):
                #         raise ValueError('For "values", data must be list, dict or tuple (got "{}")'.format(item['values'].__class__.__name__))
                #     data = values
                # else:
                #     data = [value]
                                 
                # Create the item
                if slot_unit:
                    if item_dt is not None:
                        series_items.append(DataTimeSlot(dt=item_dt, unit=slot_unit, data=data))
                    else:
                        series_items.append(DataTimeSlot(t=item_t, unit=slot_unit, data=data))
                else:
                    if item_dt is not None:
                        series_items.append((DataTimePoint(dt=item_dt, data=data)))
                    else:
                        series_items.append((DataTimePoint(t=item_t, data=data)))

            # Set the list of data time points
            super(TimeSeries, self).__init__(*series_items, **kwargs)                        

        # Create from file
        elif args and (isinstance(args[0], str)):
            file_name = args[0]
            from .storages import CSVFileStorage
            storage = CSVFileStorage(file_name, **kwargs)
            loaded_series = storage.get()
            
            # TODO: the following might not perform well..
            series_items = loaded_series.contents()
            super(TimeSeries, self).__init__(*series_items)

        # Create from list of TimePoints or TimeSlots (just call parent init)
        else:
               
            # Handle timezone
            tz = kwargs.pop('tz', None)
            if tz:
                self._tz = timezonize(tz)
            
            # Call parent init
            super(TimeSeries, self).__init__(*args, **kwargs)

    def save(self, file_name, overwrite=False, **kwargs):
        """Save the time series as a CSV file."""
        from .storages import CSVFileStorage
        storage = CSVFileStorage(file_name, **kwargs)
        storage.put(self, overwrite=overwrite)


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
                # Check time ordering and handle the resolution.
    
                # The following if is to support the deepcopy, otherwise the original prev_t will be used
                # TODO: maybe move the above to a "hasattr" plus an "and" instead of this logic?
                if len(self)>0:
    
                    # logger.debug('Checking time ordering for t="%s" (prev_t="%s")', item.t, self.prev_t)
                    if item.t < self.prev_t:
                        raise ValueError('Time t="{}" is out of order (prev t="{}")'.format(item.t, self.prev_t))
                    
                    if item.t == self.prev_t:
                        raise ValueError('Time t="{}" is a duplicate'.format(item.t))
                    
                    try:
                        self._resolution 
                    
                    except AttributeError:
                        
                        # Set the resolution as seconds for a bit of performance
                        self._resolution_as_seconds = item.t - self.prev_t 
                        
                        # Set the resolution
                        self._resolution = TimeUnit(_to_time_unit_string(item.t - self.prev_t, friendlier=True))
                                        
                    else:
                        # If the resolution is constant (not variable), check that it still is
                        if self._resolution != 'variable':
                            if self._resolution_as_seconds != (item.t - self.prev_t):
                                # ...otherwise, mark it as variable
                                del self._resolution_as_seconds
                                self._resolution = 'variable'
            finally:
                # Delete the autodetected sampling interval cache if present
                try:
                    del self._autodetected_sampling_interval
                    del self._autodetected_sampling_interval_confidence
                except:
                    pass
                # And set the prev
                self.prev_t = item.t
        
        elif isinstance(item, TimeSlot):
            
            # Slots can belong to the same series if they are in succession (tested with the __succedes__ method)
            # and if they have the same unit, which we test here instead as the __succedes__ is more general.

            # Check the timezone (only for slots, points are not affected by timezones)
            if not self.tz:
                # If no timezone set, use the item one's
                self._tz = item.tz
    
            else:
                # Else, check for the same timezone
                if self._tz != item.tz:
                    raise ValueError('Cannot append slots on different timezones (I have "{}" and you tried to add "{}")'.format(self.tz, item.start.tz))

            try:
                if self._resolution != item.unit:
                    # Try for floating point precision errors
                    abort = False
                    try:
                        if not _is_close(self._resolution.value, item.unit.value):
                            abort = True
                    except (TypeError, ValueError):
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
        # TODO: improve performance, bisection first, then use an index?
        for item in self:
            if item.t == t:
                return item
        raise ValueError('Cannot find any item for t="{}"'.format(t))


    #=========================
    # Square brackets notation
    #=========================

    def __getitem__(self, arg):
        if isinstance(arg, slice):
            
            # Handle time-based slicing
            requested_start_t = None
            requested_stop_t = None
            requested_start_i = None
            requested_stop_i = None
            if isinstance(arg.start, str):
                if arg.start.startswith('t='):
                    requested_start_t = float(arg.start.split('=')[1])
                elif arg.start.startswith('dt='):
                    requested_start_t = s_from_dt(dt_from_str(arg.start.split('=')[1].replace(' ', 'T')))
                else:
                    raise ValueError('Don\'t know how to parse slicing start "{}"'.format(arg.start))
            elif isinstance(arg.start, dict):
                if 't' in arg.start:
                    requested_start_t = arg.start['t']
                elif 'dt' in arg.start:
                    requested_start_t = s_from_dt(arg.start['dt'])
                else:
                    raise ValueError('Getting items by dict requires a "t" or a "dt" dict key, found none (Got dict={})'.format(arg))
            else:
                requested_start_i = arg.start 
            
            if isinstance(arg.stop, str):
                if arg.start.startswith('t='):
                    requested_stop_t = float(arg.stop.split('=')[1])
                elif arg.start.startswith('dt='):
                    requested_stop_t = s_from_dt(dt_from_str(arg.stop.split('=')[1].replace(' ', 'T')))
                else:
                    raise ValueError('Don\'t know how to parse slicing stop "{}"'.format(arg.stop))
            elif isinstance(arg.stop, dict):
                if 't' in arg.stop:
                    requested_stop_t = arg.stop['t']
                elif 'dt' in arg.stop:
                    requested_stop_t = s_from_dt(arg.stop['dt'])
                else:
                    raise ValueError('Getting items by dict requires a "t" or a "dt" dict key, found none (Got dict={})'.format(arg))  
            else:
                requested_stop_i = arg.stop 

            if requested_start_t is not None or requested_stop_t is not None:
                # Slice on timestamps with the slice operations
                return self.slice(start=requested_start_t, end=requested_stop_t)
            else:
                # Slice on indexes
                # TODO: maybe use the parent list slicing somehow?
                if requested_start_i is None:
                    requested_start_i = 0
                if requested_stop_i is None:
                    requested_stop_i = len(self)
                if requested_start_i < 0:
                    requested_start_i = len(self) + requested_start_i
                if requested_stop_i < 0:
                    requested_stop_i = len(self) + requested_stop_i         
                sliced_series = self.__class__()
                sliced_series.mark = self.mark
                for i in range(requested_start_i, requested_stop_i):
                    sliced_series.append(self[i])
                return sliced_series
        
        elif isinstance(arg, str):
            
            # Are we filtering against t or dt?
            if arg.startswith('t='):
                requested_t = float(arg.split('=')[1])
                logger.debug('Will look up item for t="%s"', requested_t)
                try:
                    return self._item_by_t(requested_t)
                except ValueError:
                    raise ValueError('Cannot find any item for t="{}"'.format(requested_t))
            
            if arg.startswith('dt='):
                requested_dt = dt_from_str(arg.split('=')[1].replace(' ', 'T'))
                logger.debug('Will look up item for dt="%s"', requested_dt)
                try:
                    return self._item_by_t(s_from_dt(requested_dt))
                except ValueError:
                    raise ValueError('Cannot find any item for dt="{}"'.format(requested_dt))
        
            # Try filtering on this data label only
            return self.filter(arg)
        
        elif isinstance(arg, dict):
            
            if 't' in arg:
                requested_t = arg['t']
                logger.debug('Will look up item for t="%s"', requested_t)
                try:
                    return self._item_by_t(requested_t)
                except ValueError:
                    raise ValueError('Cannot find any item for t="{}"'.format(requested_t))
        
            elif 'dt' in arg:
                requested_dt = arg['dt']
                logger.debug('Will look up item for dt="%s"', requested_dt)
                try:
                    return self._item_by_t(s_from_dt(requested_dt))
                except ValueError:
                    raise ValueError('Cannot find any item for dt="{}"'.format(requested_dt))
        
            else:
                raise ValueError('Getting items by dict requires a "t" or a "dt" dict key, found none (Got dict={})'.format(arg))
        
        else:
            return super(Series, self).__getitem__(arg)


    #=========================
    #  Timezone-related
    #=========================

    @property
    def tz(self):
        """The timezone of the time series."""
        # Note: we compute the tz on the fly because for point time series we assume to use the tz
        # attribute way lass than the slot time series, where the tz is instead computed at append-time.
        try:
            return self._tz
        except AttributeError:
            # Detect timezone on the fly
            # TODO: this ensures that each point is on the same timezone. Do we want this?
            detected_tz = None
            for item in self:
                if not detected_tz:
                    detected_tz = item.tz
                else:
                    # Terrible but seems like no other way to compare pytz.tzfile.* classes
                    if str(item.tz) != str(detected_tz): 
                        return UTC
            return detected_tz
    
    def change_tz(self, tz):
        """Change the time zone of the time series, in-place."""
        for time_point in self:
            time_point.change_tz(tz)
        self._tz = time_point.tz

    def change_timezone(self, tz):
        """This method is deprecated in favor of change_tz().
        
        :meta private:
        """
        logger.warning('The change_timezone() method is deprecated in favor of change_tz().')
        self.change_tz(tz)

    def as_tz(self, tz):
        """Get a copy of the time series on a new time zone.""" 
        new_series = self.duplicate() 
        new_series.change_tz(tz)
        return new_series

    def as_timezone(self, tz):
        """This method is deprecated in favor of as_tz().
        
        :meta private:
        """
        logger.warning('The as_timezone() method is deprecated in favor of as_tz().')
        return self.as_tz(tz)


    #=========================
    #  Resolution-related
    #=========================

    @property
    def _autodetected_sampling_interval(self):
        if not issubclass(self.items_type, TimePoint):
            raise NotImplementedError('Auto-detecting the sampling rate (and its confidence) is implemented only for point series')
        try:
            return self.__autodetected_sampling_interval
        except AttributeError:
            from .utilities import detect_sampling_interval
            self.__autodetected_sampling_interval, self.__autodetected_sampling_interval_confidence = detect_sampling_interval(self, confidence=True)           
            return self.__autodetected_sampling_interval
    
    @property
    def _autodetected_sampling_interval_confidence(self):
        if not issubclass(self.items_type, TimePoint):
            raise NotImplementedError('Auto-detecting the sampling rate (and its confidence) is implemented only for point series')
        try:
            return self.__autodetected_sampling_interval_confidence
        except AttributeError:
            from .utilities import detect_sampling_interval
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
               confidence (bool): if to return, together with the guessed resolution, also its confidence (in a 0-1 range).
        
        """
        if not issubclass(self.items_type, TimePoint):
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
    #  Conversion-related
    #=========================

    @property
    def df(self):
        """The time series as a Pandas data frame object."""
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
                df = df.append(DataFrame([[item.dt]+values+[item.data_loss]], columns=columns))
            else:
                df = df.append(DataFrame([[item.dt]+values], columns=columns))                
        df = df.set_index('Timestamp')
        
        return df

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

class _TimeSeriesView(TimeSeries):
    """A time series view. Only used internally, maybe in future it could be made public
    as a more optimized way for performing some operations, as filtering and slicing.
    
    :meta private:
    """
    def __init__(self, series, from_i, to_i, from_t=None, to_t=None, dense=False, interpolator_class=None):
        self.series = series
        self.from_i = from_i
        self.to_i = to_i
        self.len = None
        self.new_points = {}
        self.from_t = from_t
        self.to_t=to_t
        self.dense=dense
        if self.dense:
            if not interpolator_class:
                raise ValueError('If requesting a dense slice you must provide an interpolator')
            self.interpolator = interpolator_class(series)
        else:
            self.interpolator = None
    
    def __getitem__(self, i):
        if self.dense:
            raise NotImplementedError('Getting items by index on dense slices is not supporter. Use the iterator instead.')

        if i>=0:
            return self.series[self.from_i + i]
        else:
            return self.series[self.to_i - abs(i)]
            
    def __iter__(self):
        self.count = 0
        self.prev_was_new = False
        return self
    
    def __next__(self):
                
        this_i = self.count + self.from_i
        
        if this_i >= self.to_i:
            # If reached the end stop
            raise StopIteration
        
        elif self.count == 0 or not self.dense:
            # If first point or not dense just return
            self.count += 1
            return self.series[this_i]
        
        else:
            # Otherwise check if we have to add new missing points

            if self.prev_was_new:
                # If we just created a new missing point, return
                self.prev_was_new = False
                self.count += 1
                return self.series[this_i]
            
            else:
                # Check if we have to add a new missing point: do we have a gap?
                prev_point = self.series[this_i-1]
                this_point = self.series[this_i]
                
                
                if prev_point.valid_to < this_point.valid_from:
                    
                    # yes, we do have a gap. Add a missing point by interpolation

                    # Compute new point validity
                    if self.from_t is not None and prev_point.valid_to < self.from_t:
                        new_point_valid_from = self.from_t
                    else:
                        new_point_valid_from = prev_point.valid_to
                    
                    if self.to_t is not None and this_point.valid_from > self.to_t:
                        new_point_valid_to = self.to_t
                    else:
                        new_point_valid_to = this_point.valid_from
                    
                    # Compute new point timestamp    
                    new_point_t = new_point_valid_from + (new_point_valid_to-new_point_valid_from)/2

                    # Can we use cache?
                    if new_point_t in self.new_points:
                        self.prev_was_new = True
                        return self.new_points[new_point_t]
                    
                    # Log new point creation
                    logger.debug('New point t=,%s validity: [%s,%s]',new_point_t, new_point_valid_from,new_point_valid_to)
                                            
                    # Compute the new point values using the interpolator
                    new_point_data = self.interpolator.evaluate(new_point_t, prev_i=this_i-1, next_i=this_i)
                    
                    # Create the new point    
                    new_point = this_point.__class__(t = new_point_t, data = new_point_data)
                    new_point.valid_from = new_point_valid_from
                    new_point.valid_to = new_point_valid_to
                    new_point._interpolated = True
                    
                    # Set flag
                    self.prev_was_new = True 
                    
                    # Add to cache
                    self.new_points[new_point_t] = new_point
                    
                    # ..and return it
                    return new_point
                 
                else:
                    # Return this point if no gaps
                    self.count += 1
                    return this_point
    
    def __len__(self):
        if not self.dense:
            return self.to_i-self.from_i
        else:
            if self.len is None:
                self.len=0
                for _ in self:
                    self.len+=1
            return self.len

    def __repr__(self):
        if not self.series:
            return 'Empty series slice'
        else:
            return 'Series slice'

    @property
    def items_type(self):
        for item in self:
            return item.__class__

    @property
    def resolution(self):
        return self.series.resolution

    def data_labels(self):
        return self.series.data_labels()



#=========================
#  Back-compatibility
#=========================

class DataPointSeries(Series): 
    """This class is deprecated in favor of the TimeSeries class.
    
    :meta private:
    """
    def __init__(self, *args, **kwargs):
        logger.warning('The DataPointSeries class is deprecated, please replace it with the new TimeSeries class.')
        super(DataPointSeries, self).__init__(*args, **kwargs)

class DataSlotSeries(Series):
    """This class is deprecated in favor of the TimeSeries class.
    
    :meta private:
    """
    def __init__(self, *args, **kwargs):
        logger.warning('The DataSlotSeries class is deprecated, please replace it with the new TimeSeries class.')
        super(DataSlotSeries, self).__init__(*args, **kwargs)

class TimePointSeries(TimeSeries):
    """This class is deprecated in favor of the TimeSeries class.
    
    :meta private:
    """
    def __init__(self, *args, **kwargs):
        logger.warning('The DataPointSeries class is deprecated, please replace it with the new TimeSeries class.')
        super(TimePointSeries, self).__init__(*args, **kwargs)

class DataTimePointSeries(TimeSeries):
    """This class is deprecated in favor of the TimeSeries class.
    
    :meta private:
    """
    def __init__(self, *args, **kwargs):
        logger.warning('The DataTimePointSeries class is deprecated, please replace it with the new TimeSeries class.')
        super(DataTimePointSeries, self).__init__(*args, **kwargs)

class TimeSlotSeries(TimeSeries):
    """This class is deprecated in favor of the TimeSeries class.
    
    :meta private:
    """
    def __init__(self, *args, **kwargs):
        logger.warning('The TimeSlotSeries class is deprecated, please replace it with the new TimeSeries class.')
        super(TimeSlotSeries, self).__init__(*args, **kwargs)

class DataTimeSlotSeries(TimeSeries):
    """This class is deprecated in favor of the TimeSeries class.
    
    :meta private:
    """
    def __init__(self, *args, **kwargs):
        logger.warning('The DataTimeSlotSeries class is deprecated, please replace it with the new TimeSeries class.')
        super(DataTimeSlotSeries, self).__init__(*args, **kwargs)

