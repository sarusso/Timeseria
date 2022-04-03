# -*- coding: utf-8 -*-
"""Provides base data structures: Points, Slots and Series with all their specializations."""

from .time import s_from_dt , dt_from_s, UTC, timezonize
from .units import Unit, TimeUnit
from .utilities import is_close, to_time_unit_string
from copy import deepcopy
from pandas import DataFrame
from datetime import datetime
from .exceptions import ConsistencyException

# Setup logging
import logging
logger = logging.getLogger(__name__)


#======================
#  Generic Series
#======================

class Series(list):
    """A list of items coming one after another, where every item
       is guaranteed to be of the same type and in an order or
       succession.
       
       Args:
           *args (list): the series items.    
    """

    # 1,2,3 are in order. 1,3,8 are in order. 1,8,3 re not in order.
    # 5,6,7 are integer succession. 5.3, 5.4, 5.5 are too in a succesisons. 5,6,8 are not in a succession. 
    # a group or a number of related or similar things, events, etc., arranged or occurring in temporal, spatial, or other order or succession; sequence.

    # By default the type is not defined
    __TYPE__ = None
    
    def __init__(self, *args, **kwargs):

        for arg in args:
            self.append(arg)
            
        self._title = None

    def append(self, item):
        """Append an item to the series."""
        
        # TODO: move to use the insert
        
        # logger.debug('Checking %s', item)
        
        # Set type if not already done
        if not self.__TYPE__:
            self.__TYPE__ = type(item)

        # Check type
        if not isinstance(item, self.__TYPE__):
            raise TypeError('Got incompatible type "{}", can only accept "{}"'.format(item.__class__.__name__, self.__TYPE__.__name__))
        
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
        
        # Append
        super(Series, self).append(item)
    
    def __sum__(self, other):
        raise NotImplementedError

    def __hash__(self):
        return id(self)

    @property
    def title(self):
        """A title for the series, to be used for plotting etc.""" 
        if self._title:
            return self._title
        else:
            return None

    @title.setter
    def title(self, title):
        self._title=title

    def __repr__(self):
        return '{} of #{} elements'.format(self.__class__.__name__, len(self))
    
    def __str__(self):
        return self.__repr__()
        
    # Python slice (i.e [0:35])
    def __getitem__(self, key):
        if isinstance(key, slice):   
            # TODO: improve and implement shallow copies please.         
            indices = range(*key.indices(len(self)))
            series = self.__class__()
            for i in indices:
                series.append(super(Series, self).__getitem__(i))
            try:
                series.mark = self.mark
            except:
                pass
            return series
        elif isinstance(key, str):   
            # Try filtering on this data key only
            return self.filter(key)
        else:
            return super(Series, self).__getitem__(key)

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
        """A mark for the series, useful for highlighting a portion of a plot.
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
            del self._mark
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
            if not isinstance(x, self.__TYPE__):
                raise TypeError('Got incompatible type "{}", can only accept "{}"'.format(x.__class__.__name__, self.__TYPE__.__name__))
        
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

    # Operations
    def duplicate(self):
        """ Return a deep copy of the series."""
        return deepcopy(self)

    def merge(self, *args, **kwargs):
        """Merge the series with one or more other series."""
        from .operations import merge as merge_operation
        return merge_operation(self, *args, **kwargs)

    def filter(self, *args, **kwargs):
        # TODO: refactor this to allow generic item properties? Maybe merge witht he following select?
        """Filter a series given one or more properties of its elemnents. Example filtering arguments: ``data_label``, ``from_t``, ``to_t``, ``from_dt``, ``to_dt``."""
        from .operations import filter as filter_operation
        return filter_operation(self, *args, **kwargs) 

    def select(self, *args, **kwargs):
        """Select one or more items of the series given an SQL-like query."""
        from .operations import select as select_operation
        return select_operation(self, *args, **kwargs)
    
    # Inspection utilities
    def inspect(self, limit=10):
        """Prints a summary of the series and its elements, limited to 10 items by default.
        
            Args:
                limit: the limit of elements to print, by default 10.
        """
        print(str(self)+':\n')
        print('[', end='')
        
        if not limit or limit > len(self):
        
            for i, item in enumerate(self):
                if limit and i >= limit:
                    break
                else:
                    if i==0:
                        print(str(item)+',')
                    elif i==len(self)-1:
                        print(' '+str(item), end='')                        
                    else:
                        print(' '+str(item)+',')
        else:
            
            head_n = int(limit/2)+1
            tail_n = int(limit/2)

            for i, item in enumerate(self.head(head_n)):
                if i==0:
                    print(str(item)+',')                       
                else:
                    print(' '+str(item)+',')

            print(' ...')

            for i, item in enumerate(self.tail(tail_n)):
                if i==tail_n-1:
                    print(' '+str(item), end='')                        
                else:
                    print(' '+str(item)+',')

        print(']')

    def inspect_as_str(self, limit=10):
        """Return a summary of the series and its elements, limited to 10 items by default.
        
            Args:
                limit: the limit of elements to print, by default 10.
        """
        
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
            
            head_n = int(limit/2)+1
            tail_n = int(limit/2)

            for i, item in enumerate(self.head(head_n)):
                if i==0:
                    string+=str(item)+',\n'                       
                else:
                    string+=' '+str(item)+',\n'

            string+=' ...\n'

            for i, item in enumerate(self.tail(tail_n)):
                if i==tail_n-1:
                    string+=' '+str(item)                        
                else:
                    string+=' '+str(item)+',\n'

        string+=']'
        return string

    def contents(self):
        """Get al the items of the series as a list."""   
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
        #return ','.join(str(coordinate) for coordinate in self.coordinates)
        if len(self.coordinates)==1:
            return str(self.coordinates[0])
        else:
            return str(self.coordinates)

    def __repr__(self):
        return '{} @ {}'.format(self.__class__.__name__, self.__coordinates_repr__)

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        if not self.coordinates == other.coordinates:
            return False
        for i in range(0, len(self.coordinates)):
            if self.coordinates[i] != other.coordinates[i]:
                return False
        return True
    
    def __getitem__(self,index):
        return self.coordinates[index]


class TimePoint(Point):
    """A point in the time dimension.
    
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
    
    def change_timezone(self, tz):
        """Change the timezone of the point, in-place."""
        self._tz = timezonize(tz)

    @property
    def dt(self):
        """The timestamp as datetime object."""
        return dt_from_s(self.t, tz=self.tz)

    def __repr__(self):
        return '{} @ {} ({})'.format(self.__class__.__name__, self.t, self.dt)
        # return '{} @ t={} ({})'.format(self.__class__.__name__, self.t, self.dt)
    

class DataPoint(Point):
    """A point that carries some data.
    
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
            return '{} with data "{}" and data_loss="{}"'.format(super(DataPoint, self).__repr__(), self.data, self.data_loss)            
        else:
            return '{} with data "{}"'.format(super(DataPoint, self).__repr__(), self.data)
    
    def __eq__(self, other):
        if self._data != other._data:
            return False
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
        """Return the data labels. If data is a dictionary, then these are the dictionary keys,
        if data is a list, then these are the list indexes. Other formats are not supported."""
        try:
            return sorted(list(self.data.keys()))
        except AttributeError:
            return list(range(len(self.data)))


class DataTimePoint(DataPoint, TimePoint):
    """A point that carries some data in the time dimension.
    
       Args:
           t (float): epoch timestamp, decimals for sub-second precision.
           dt (datetime): a datetime object timestamp.
           data: the data.
           data_indexes(dict): data indexes.
           data_loss(float): the data loss index, if any.
    """
    pass

    # NOTE: the __repr__ used is from the DataPoint above, which in turn uses the TimePoint one.


#======================
#  Point Series
#======================

class PointSeries(Series):
    """A series of points, where each item is guaranteed to be ordered.

       Args:
           *args (list): the series points.    
    """
    __TYPE__ = Point


class TimePointSeries(PointSeries):
    """A series of points in time, where each item is guaranteed to be ordered.

       Args:
           *args (list): the series of time points.    
    """

    __TYPE__ = TimePoint

    def __init__(self, *args, **kwargs):

        tz = kwargs.pop('tz', None)
        if tz:
            self._tz = timezonize(tz)
        
        self._resolution = None

        super(TimePointSeries, self).__init__(*args, **kwargs)
        

    # Check time ordering and resolution
    def append(self, item):
        try:
            
            # This is to support the deepcopy, otherwise the original prev_t will be used
            if len(self)>0:
                
                # logger.debug('Checking time ordering for t="%s" (prev_t="%s")', item.t, self.prev_t)
                if item.t < self.prev_t:
                    raise ValueError('Time t="{}" is out of order (prev t="{}")'.format(item.t, self.prev_t))
                
                if item.t == self.prev_t:
                    raise ValueError('Time t="{}" is a duplicate'.format(item.t))
                
                if self._resolution is None:
                    self.resolution_as_seconds = item.t - self.prev_t # For a bit of performance
                    
                    class TimeResolution(TimeUnit):
    
                        def __init__(self, *args, **kwargs):
                            variable = kwargs.pop('variable', False)
                            self.variable = variable
                            super(TimeResolution, self).__init__(*args, **kwargs)
                    
                        def __repr__(self):
                            if self.variable:
                                return "~{}".format(super(TimeResolution, self).__repr__())
                            else:
                                return "{}".format(super(TimeResolution, self).__repr__())
                        
                        def is_variable(self):
                            return self.variable

                    self._resolution = TimeResolution(to_time_unit_string(item.t - self.prev_t, friendlier=True)) 
                    #self._resolution = TimeUnit(to_time_unit_string(item.t - self.prev_t, friendlier=True))
                    
                elif not self._resolution.variable:
                    if self.resolution_as_seconds != item.t - self.prev_t:
                        # TODO: here you should set a flag and remove the resolution.
                        # Then when this is requested, set it using the autodetected sampling rate
                        self._resolution.variable = True

            self.prev_t = item.t
                
        except AttributeError:
            self.prev_t = item.t
       
        super(TimePointSeries, self).append(item)

    @property
    def tz(self):
        """The timezone of the series."""
        # Note: we compute the tz on the fly beacuse for point time series we assume to use the tz
        # attribute way lass than the slot time series, where the tz is instead computed at append-time.
        try:
            return self._tz
        except AttributeError:
            # Detect timezone on the fly
            # TODO: this ensure each ppint is on the same timezone. Do we want this?
            detected_tz = None
            for item in self:
                if not detected_tz:
                    detected_tz = item.tz
                else:
                    # Terrible but seems like no other way to compare pytz.tzfile.* classes
                    if str(item.tz) != str(detected_tz): 
                        return UTC
            return detected_tz
    
    def change_timezone(self, tz):
        """Change the timezone of the series, in-place."""
        for time_point in self:
            time_point.change_timezone(tz)
        self._tz = time_point.tz

    def as_timezone(self, tz):
        """Get a copy of the series on a new timezone.""" 
        new_series = self.duplicate() 
        new_series.change_timezone(tz)
        return new_series

    @property
    def resolution(self):
        """The (temporal) resolution of the time series."""
        try:
            return self._resolution
        except AttributeError:
            return None


class DataPointSeries(PointSeries):
    """A series of data points, where each item is guaranteed to be ordered and to carry the same data type.

       Args:
           *args (list): the series of data points.    
    """

    __TYPE__ = DataPoint

    # Check data compatibility
    def append(self, item):
        
        try:
            if not type(self._item_data_reference) == type(item.data):
                raise TypeError('Got different data: {} vs {}'.format(self._item_data_reference.__class__.__name__, item.data.__class__.__name__))
            if isinstance(self._item_data_reference, list):
                if len(self._item_data_reference) != len(item.data):
                    raise ValueError('Got different data lengths: {} vs {}'.format(len(self._item_data_reference), len(item.data)))
            if isinstance(self._item_data_reference, dict):
                if set(self._item_data_reference.keys()) != set(item.data.keys()):
                    raise ValueError('Got different data keys: {} vs {}'.format(self._item_data_reference.keys(), item.data.keys()))
            
        except AttributeError:
            # logger.debug('Setting data reference: %s', item.data)
            self._item_data_reference = item.data
            
        super(DataPointSeries, self).append(item)

    def data_labels(self):
        """Return the labels of the data carried by the DataPoints.
        If data is a dictionary, then these are the dictionary keys,
        if data is a list, then these are the list indexes. Other
        data formats are not supported."""
          
        if len(self) == 0:
            return None
        else:
            # TODO: can we optimize here? Computing them once and then serving them does not work if someone changes data keys...
            try:
                return sorted(list(self[0].data.keys()))
            except AttributeError:
                return list(range(len(self[0].data)))

    def rename_data_label(self, old_key, new_key):
        """Rename a data key, in-place."""
        for item in self:
            # TODO: move to the DataPoint/DataSlot?
            item.data[new_key] = item.data.pop(old_key)

    def remove_data_loss(self):
        """Remove the ``data_loss`` index, in-place."""
        for item in self:
            item.data_indexes.pop('data_loss', None)

    def remove_data_index(self, data_index):
        """Remove a data index, in-place."""
        for item in self:
            item.data_indexes.pop(data_index, None)        

    # Operations
    def min(self, *args, **kwargs):
        """Get the minimum data value(s) of a series. Supports an optional ``data_label`` argument."""
        from .operations import min as min_operation
        return min_operation(self, *args, **kwargs)

    def max(self, *args, **kwargs):
        """Get the maximum data value(s) of a series. Supports an optional ``data_label`` argument."""
        from .operations import max as max_operation
        return max_operation(self, *args, **kwargs)    

    def avg(self, *args, **kwargs):
        """Get the average data value(s) of a series. Supports an optional ``data_label`` argument."""
        from .operations import avg as avg_operation
        return avg_operation(self, *args, **kwargs)   

    def sum(self, *args, **kwargs):
        """Sum every data value(s) of a series. Supports an optional ``data_label`` argument."""
        from .operations import sum as sum_operation
        return sum_operation(self, *args, **kwargs)  

    def derivative(self, *args, **kwargs):
        """Compute the derivative of the series. Extra parameters: ``inplace`` (defaulted to false),
        ``normalize`` (defaulted to true), ``diffs`` (defaulted to false) to compute differences
        instead of the derivative."""
        from .operations import derivative as derivative_operation
        return derivative_operation(self, *args, **kwargs)   

    def integral(self, *args, **kwargs):
        """Compute the integral of the series. Extra parameters: ``inplace`` (defaulted to false),
        ``normalize`` (defaulted to true), ``c`` (defaulted to zero) for the integration constant."""
        from .operations import integral as integral_operation
        return integral_operation(self, *args, **kwargs)   

    def diff(self, *args, **kwargs):
        """Compute the incremental differences. Extra parameters: ``inplace`` (defaulted to false)."""
        from .operations import diff as diff_operation
        return diff_operation(self, *args, **kwargs)   

    def csum(self, *args, **kwargs):
        """Compute the incremental sum. Extra parameters: ``inplace`` (defaulted to false),
        ``offset`` (defaulted to zero) to set the starting value where to apply the sums on."""
        from .operations import csum as csum_operation
        return csum_operation(self, *args, **kwargs)

    def normalize(self, *args, **kwargs):
        """Normalize the time series data values. Extra parameters: ``inplace`` (defaulted to false)."""
        from .operations import normalize as normalize_operation
        return normalize_operation(self, *args, **kwargs)   

    def rescale(self, *args, **kwargs):
        """Rescale the time series data values. Extra parameters: ``inplace`` (defaulted to false)."""
        from .operations import rescale as rescale_operation
        return rescale_operation(self, *args, **kwargs)

    def offset(self, *args, **kwargs):
        """Offset the time series data values. Extra parameters: ``inplace`` (defaulted to false)."""
        from .operations import offset as offset_operation
        return offset_operation(self, *args, **kwargs)  

    def mavg(self, *args, **kwargs):
        """Compute the moving average. Extra parameters: ``inplace`` (defaulted to false)
        and ``window``, a required parameter, for the length of the moving average window."""
        from .operations import mavg as mavg_operation
        return mavg_operation(self, *args, **kwargs)


class DataTimePointSeries(DataPointSeries, TimePointSeries):
    """A series of data points in time, where each item is guaranteed to be ordered and to carry the same data type.
    
        Args:
           *args (list): the series of data time points.    
    """

    __TYPE__ = DataTimePoint

    def __init__(self, *args, **kwargs):

        # Handle df kwarg
        df = kwargs.pop('df', None)
        
        # Handle first argument as dataframe
        if args and (isinstance(args[0], DataFrame)):
            df = args[0]

        # Do we have to initialize the time series from a dataframe?        
        if df is not None:

            # Create data time points list
            data_time_points = []
            
            # Get data frame labels
            labels = list(df.columns) 
            
            for row in df.iterrows():
                
                # Set the timestamp
                if not isinstance(row[0], datetime):
                    raise TypeError('A DataFrame with a DateTime index column is required')
                dt = row[0]
                
                # Prepare data
                data = {}
                for i,label in enumerate(labels):
                    data[label] = row[1][i]
                
                data_time_points.append(DataTimePoint(dt=dt, data=data))
            
            # Set the list of data time points
            super(DataTimePointSeries, self).__init__(*data_time_points, **kwargs)
            return None
        
        # Original init
        super(DataTimePointSeries, self).__init__(*args, **kwargs)

    @property
    def df(self):
        """The time series as a Pandas DataFrame object."""
        data_labels = self.data_labels()
        
        if self[0].data_loss is not None:
            dump_data_loss = True
        else:
            dump_data_loss = False
        
        if dump_data_loss:
            columns = ['Timestamp'] + data_labels + ['data_loss']
        else:
            columns = ['Timestamp'] + data_labels
            
        df = DataFrame(columns=columns)
        for item in self:
            values = [item.data[key] for key in data_labels]
            if dump_data_loss:
                df = df.append(DataFrame([[item.dt]+values+[item.data_loss]], columns=columns))
            else:
                df = df.append(DataFrame([[item.dt]+values], columns=columns))                
        df = df.set_index('Timestamp')
        return df
        
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
        
    @property
    def autodetected_sampling_interval(self):
        try:
            return self._autodetected_sampling_interval
        except AttributeError:
            from .utilities import detect_sampling_interval
            self._autodetected_sampling_interval = detect_sampling_interval(self)
            return self._autodetected_sampling_interval
    
    @property
    def _resolution_string(self):
        if isinstance(self.resolution, Unit):
            # Includes TimeUnits
            _resolution_string = '{} resolution'.format(self.resolution)
        else:
            autodetected_sampling_interval_as_str = str(self.autodetected_sampling_interval)
            if autodetected_sampling_interval_as_str.endswith('.0'):
                # TODO: use a friendlier resolution here as well, as above?
                # TODO: do something like this when setting the variable resolution, roght now the .resolution
                # and ._resolution_string might return different values as they are computed differently  
                autodetected_sampling_interval_as_str = autodetected_sampling_interval_as_str[:-2] 
            _resolution_string = 'variable resolution (~{}s)'.format(autodetected_sampling_interval_as_str)
        return _resolution_string

    def __repr__(self):
        if len(self):
            return 'Time series of #{} points at {}, from point @ {} ({}) to point @ {} ({})'.format(len(self), self._resolution_string, self[0].t, self[0].dt, self[-1].t, self[-1].dt)
        else:
            return 'Time series of #0 points'

    # Transformations
    def aggregate(self, unit, *args, **kwargs):
        """Aggregate the series in slots of a length set by the ``unit`` parameter."""
        from .transformations import Aggregator
        aggregator = Aggregator(unit, *args, **kwargs)
        return aggregator.process(self)  

    def resample(self, unit, *args, **kwargs):
        """Resample the series using a sampling interval of a length set by the ``unit`` parameter."""
        from .transformations import Resampler
        resampler = Resampler(unit, *args, **kwargs)
        return resampler.process(self)  


#======================
#  Slots
#======================

class Slot():
    """A slot. Can be initialized with start and end, or start and unit.
    
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
        return '{} @ [{},{}]'.format(self.__class__.__name__, self.start.__coordinates_repr__, self.end.__coordinates_repr__)
    
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
    """A slot in the time dimension. Can be initialized with start and end,
       or start and unit. Can be also initialized using t, dt and tz, for the
       starting point but mainly for internal use and should not be relied upon.
    
       Args:
           start(TimePoint): the slot starting time point.
           end(TimePoint): the slot ending time point.
           unit(TimeUnit): the slot time unit."""

    __POINT_TYPE__ = TimePoint

    def __init__(self, start=None, end=None, unit=None, t=None, dt=None, tz=None):
        
        # Handle t and dt shortcuts
        if t:
            start=TimePoint(t=t, tz=tz)
        if dt:
            start=TimePoint(dt=dt, tz=tz)
        
        if not isinstance(start, self.__POINT_TYPE__):
            raise TypeError('Slot start must be a Point object (got "{}")'.format(start.__class__.__name__))

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
            self.end.change_timezone(self.start.tz)

    # Overwrite parent succedes, this has better performance as it checks for only one dimension
    def __succedes__(self, other):
        if other.end.t != self.start.t:
            # Take into account floating point rounding errors
            if is_close(other.end.t, self.start.t):
                return True
            return False
        else:
            return True

    def change_timezone(self, tz):
        """Change the timezone of the slot, in-place."""
        self.start.change_timezone(tz)
        self.end.change_timezone(tz)
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


class DataSlot(Slot):
    """A slot that carries some data. Can be initialized with start and end
       or start and unit, plus the data argument.
    
       Args:
           start(Point): the slot starting point.
           end(Point): the slot ending point.
           unit(Unit): the slot unit.
           data: the data.
           data_indexes(dict): data indexes.
           data_loss(float): the data loss index, if any.
    """

    def __init__(self, **kwargs):
        
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
        super(DataSlot, self).__init__(**kwargs)

    def __repr__(self):
        return '{} with start="{}" and end="{}"'.format(self.__class__.__name__, self.start, self.end)

    def __eq__(self, other):
        if self._data != other._data:
            return False
        return super(DataSlot, self).__eq__(other)

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
        """Return the data labels. If data is a dictionary, then these are the dictionary keys,
        if data is a list, then these are the list indexes. Other formats are not supported."""
        try:
            return sorted(list(self.data.keys()))
        except AttributeError:
            return list(range(len(self.data)))    


class DataTimeSlot(DataSlot, TimeSlot):
    """A slot that carries some data in the time dimension. Can be initialized
       with start and end, or start and unit, plus the data argument. Can be also
       initialized using t, dt and tz, for the starting point but mainly for internal
       use and should not be relied upon.
    
       Args:
           start(TimePoint): the slot starting time point.
           end(TimePoint): the slot ending time point.
           unit(TimeUnit): the slot time unit.
           data: the data.
           data_indexes(dict): data indexes.
           data_loss(float): the data loss index, if any.
    """


    def __repr__(self):
        #if self.data_loss is not None:
        #    return '{} @ t=[{},{}] ([{},{}]) with data={} and data_loss={}'.format(self.__class__.__name__, self.start.t, self.end.t, self.start.dt, self.end.dt, self.data, self.data_loss)
        #else:
        #    return '{} @ t=[{},{}] ([{},{}]) with data={}'.format(self.__class__.__name__, self.start.t, self.end.t, self.start.dt, self.end.dt, self.data)

        if self.data_loss is not None:
            return '{} @ [{},{}] ([{},{}]) with data={} and data_loss={}'.format(self.__class__.__name__, self.start.t, self.end.t, self.start.dt, self.end.dt, self.data, self.data_loss)
        else:
            return '{} @ [{},{}] ([{},{}]) with data={}'.format(self.__class__.__name__, self.start.t, self.end.t, self.start.dt, self.end.dt, self.data)
        


#======================
#  Slot Series
#======================

class SlotSeries(Series):
    """A series of generic slots, where each item is guaranteed to be in succession.
    
       Args:
           *args (list): the series of slots.    
    """
    
    __TYPE__ = Slot

    def append(self, item):
        
        # Slots can belong to the same series if they are in succession (tested with the __succedes__ method)
        # and if they have the same unit, which we test here instead as the __succedes__ is more general.
        try:
            if self._reference_unit != item.unit:
                # Try for floating point precision errors
                abort = False
                try:
                    if not is_close(self._reference_unit.value, item.unit.value):
                        abort = True
                except (TypeError, ValueError):
                    abort = True
                if abort:
                    raise ValueError('Cannot add items with different units (I have "{}" and you tried to add "{}")'.format(self._reference_unit, item.unit))
        except AttributeError:
            self._reference_unit = item.unit

        # Call parent append
        super(SlotSeries, self).append(item)


class TimeSlotSeries(SlotSeries):
    """A series of slots in time, where each item is guaranteed to be in succession.
    
       Args:
           *args (list): the series of time slots.    
    """

    __TYPE__ = TimeSlot

    def append(self, item):
        
        if not self.tz:
            # If no timezone set, use the item one's
            self._tz = item.tz
            
        else:
            # Else, check for the same timezone
            if self._tz != item.tz:
                raise ValueError('Cannot add items on different timezones (I have "{}" and you tried to add "{}")'.format(self.tz, item.start.tz))

        super(TimeSlotSeries, self).append(item)
 
    @property
    def tz(self):
        """The timezone of the time series."""
        try:
            return self._tz
        except AttributeError:
            return None
        
    def change_timezone(self, tz):
        """Change the timezone of the series, in-place."""
        for time_slot in self:
            time_slot.change_timezone(tz)
        self._tz = time_slot.tz

    def as_timezone(self, tz):
        """Get a copy of the series on a new timezone.""" 
        new_series = self.duplicate() 
        new_series.change_timezone(tz)
        return new_series

    @property
    def resolution(self):
        """The (temporal) resolution of the time series."""
        try:
            return self._resolution
        except AttributeError:
            try:
                
                class TimeResolution(TimeUnit):
    
                    def __init__(self, *args, **kwargs):
                        variable = kwargs.pop('variable', False)
                        self.variable = variable
                        super(TimeResolution, self).__init__(*args, **kwargs)
                
                    def __repr__(self):
                        if self.variable:
                            return "~{}".format(super(TimeResolution, self).__repr__())
                        else:
                            return "{}".format(super(TimeResolution, self).__repr__())
                    
                    def is_variable(self):
                        return self.variable
                
                self._resolution = TimeResolution(self._reference_unit.value)
                return self._resolution
            except AttributeError:
                return None


class DataSlotSeries(SlotSeries):
    """A series of data slots, where each item is guaranteed to be in succession and to carry the same data type.
    
       Args:
           *args (list): the series of data slots.    
    """

    __TYPE__ = DataSlot

    # Check data compatibility
    def append(self, item):
        
        try:
            if not type(self._item_data_reference) == type(item.data):
                raise TypeError('Got different data: {} vs {}'.format(self._item_data_reference.__class__.__name__, item.data.__class__.__name__))
            if isinstance(self._item_data_reference, list):
                if len(self._item_data_reference) != len(item.data):
                    raise ValueError('Got different data lengths: {} vs {}'.format(len(self._item_data_reference), len(item.data)))
            if isinstance(self._item_data_reference, dict):
                if set(self._item_data_reference.keys()) != set(item.data.keys()):
                    raise ValueError('Got different data keys: {} vs {}'.format(self._item_data_reference.keys(), item.data.keys()))

        except AttributeError:
            # TODO: uniform self.tz, self._resolution, self._item_data_reference
            self._item_data_reference = item.data
        
        super(DataSlotSeries, self).append(item)
    
    def data_labels(self):
        """Return the labels of the data carried by the DataSlots.
        If data is a dictionary, then these are the dictionary keys,
        if data is a list, then these are the list indexes. Other
        data formats are not supported."""
        if len(self) == 0:
            return None
        else:
            # TODO: can we optimize here? Computing them once and then serving them does not work if someone changes data keys...
            try:
                return sorted(list(self[0].data.keys()))
            except AttributeError:
                return list(range(len(self[0].data)))

    def rename_data_label(self, old_key, new_key):
        """Rename a data key, in-place."""
        for item in self:
            # TODO: move to the DataPoint/DataSlot?
            item.data[new_key] = item.data.pop(old_key)

    def remove_data_loss(self):
        """Remove the ``data_loss`` index, in-place."""
        for item in self:
            item.data_indexes.pop('data_loss', None)
    
    def remove_data_index(self, data_index):
        """Remove a data index, in-place."""
        for item in self:
            item.data_indexes.pop(data_index, None)        

    # Operations
    def min(self, *args, **kwargs):
        """Get the minimum data value(s) of a series. Supports an optional ``data_label`` argument."""
        from .operations import min as min_operation
        return min_operation(self, *args, **kwargs)

    def max(self, *args, **kwargs):
        """Get the maximum data value(s) of a series. Supports an optional ``data_label`` argument."""
        from .operations import max as max_operation
        return max_operation(self, *args, **kwargs)    

    def avg(self, *args, **kwargs):
        """Get the average data value(s) of a series. Supports an optional ``data_label`` argument."""
        from .operations import avg as avg_operation
        return avg_operation(self, *args, **kwargs)   

    def sum(self, *args, **kwargs):
        """Sum every data value(s) of a series. Supports an optional ``data_label`` argument."""
        from .operations import sum as sum_operation
        return sum_operation(self, *args, **kwargs)  

    def derivative(self, *args, **kwargs):
        """Compute the derivative of the series. Extra parameters: ``inplace`` (defaulted to false),
        ``normalize`` (defaulted to true), ``diffs`` (defaulted to false) to compute differences
        instead of the derivative."""
        from .operations import derivative as derivative_operation
        return derivative_operation(self, *args, **kwargs)   

    def integral(self, *args, **kwargs):
        """Compute the integral of the series. Extra parameters: ``inplace`` (defaulted to false),
        ``normalize`` (defaulted to true), ``c`` (defaulted to zero) for the integration constant."""
        from .operations import integral as integral_operation
        return integral_operation(self, *args, **kwargs)   

    def diff(self, *args, **kwargs):
        """Compute the incremental differences. Extra parameters: ``inplace`` (defaulted to false)."""
        from .operations import diff as diff_operation
        return diff_operation(self, *args, **kwargs)   

    def csum(self, *args, **kwargs):
        """Compute the incremental sum. Extra parameters: ``inplace`` (defaulted to false),
        ``offset`` (defaulted to zero) to set the starting value where to apply the sums on."""
        from .operations import csum as csum_operation
        return csum_operation(self, *args, **kwargs)

    def normalize(self, *args, **kwargs):
        """Normalize the time series data values. Extra parameters: ``inplace`` (defaulted to false)."""
        from .operations import normalize as normalize_operation
        return normalize_operation(self, *args, **kwargs)   

    def rescale(self, *args, **kwargs):
        """Rescale the time series data values. Extra parameters: ``inplace`` (defaulted to false)."""
        from .operations import rescale as rescale_operation
        return rescale_operation(self, *args, **kwargs)

    def offset(self, *args, **kwargs):
        """Offset the time series data values. Extra parameters: ``inplace`` (defaulted to false)."""
        from .operations import offset as offset_operation
        return offset_operation(self, *args, **kwargs)  
  
    def mavg(self, *args, **kwargs):
        """Compute the moving average. Extra parameters: ``inplace`` (defaulted to false)
        and ``window``, a required parameter, for the length of the moving average window."""
        from .operations import mavg as mavg_operation
        return mavg_operation(self, *args, **kwargs)


class DataTimeSlotSeries(DataSlotSeries, TimeSlotSeries):
    """A series of data slots in time, where each item is guaranteed to be in succession and to carry the same data type.
       
       Args:
           *args (list): the series of data time slots.    
    """

    __TYPE__ = DataTimeSlot

    def __init__(self, *args, **kwargs):

        # Handle df kwarg
        df = kwargs.pop('df', None)
        
        # Handle first argument as dataframe
        if args and (isinstance(args[0], DataFrame)):
            df = args[0]

        # Do we have to initialize the time series from a dataframe?        
        if df is not None:

            # Create data time points list
            data_time_slots = []
            
            # Get data frame labels
            labels = list(df.columns) 

            # Get TimeUnit directly using and converting the inferred frequncy on the Pandas Data Frame
            unit_str=df.index.inferred_freq
            
            if not unit_str:
                raise Exception('Cannot infer the time unit for the slots')
            
            # Calendar
            unit_str=unit_str.replace('A', 'Y')    # Year (end) ?
            unit_str=unit_str.replace('Y', 'Y')    # Year (end) )
            unit_str=unit_str.replace('AS', 'Y')    # Year (start)
            unit_str=unit_str.replace('YS', 'Y')    # Year (start)
            unit_str=unit_str.replace('MS', 'M')    # Month (start)
            unit_str=unit_str.replace('M', 'M')    # Month (end) 
            unit_str=unit_str.replace('D', 'D')    # Day
            
            # Physical
            unit_str=unit_str.replace('H', 'h')    # Hour
            unit_str=unit_str.replace('T', 'm')    # Minute
            unit_str=unit_str.replace('min', 'm')  # Minute
            unit_str=unit_str.replace('S', 's')    # Second
            
            if len(unit_str) == 1:
                unit_str = '1'+unit_str
            logger.info('Assuming a slot time unit of "{}"'.format(unit_str))
            
            for row in df.iterrows():
                
                # Set the timestamp
                if not isinstance(row[0], datetime):
                    raise TypeError('A DataFrame with a DateTime index column is required')
                dt = row[0]
                
                # Prepare data
                data = {}
                for i,label in enumerate(labels):
                    data[label] = row[1][i]
                data_time_slots.append(DataTimeSlot(dt=dt, unit=TimeUnit(unit_str), data=data))
            
            # Set the list of data time points
            super(DataTimeSlotSeries, self).__init__(*data_time_slots, **kwargs)
            return None
        
        # Original init
        super(DataTimeSlotSeries, self).__init__(*args, **kwargs)

    @property
    def df(self):
        """The time series as a Pandas DataFrame object."""
        data_labels = self.data_labels()
        
        if self[0].data_loss is not None:
            dump_data_loss = True
        else:
            dump_data_loss = False
        
        if dump_data_loss:
            columns = ['Timestamp'] + data_labels + ['data_loss']
        else:
            columns = ['Timestamp'] + data_labels
            
        df = DataFrame(columns=columns)
        for item in self:
            values = [item.data[key] for key in data_labels]
            if dump_data_loss:
                df = df.append(DataFrame([[item.dt]+values+[item.data_loss]], columns=columns))
            else:
                df = df.append(DataFrame([[item.dt]+values], columns=columns))                
        df = df.set_index('Timestamp')
        return df

    def plot(self, engine='dg', *args, **kwargs):
        """Plot the time series. The default plotting engine is Dygraphs (``engine=\'dg\'``ƒ),
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
            raise Exception('Unknown plotting engine "{}'.format(engine))

    def __repr__(self):
        if len(self):
            # TODO: "slots of unit" ?
            return 'Time series of #{} slots of {}, from slot starting @ {} ({}) to slot starting @ {} ({})'.format(len(self), self.resolution, self[0].start.t, self[0].start.dt, self[-1].start.t, self[-1].start.dt)            
        else:
            return 'Time series of #0 slots'

    # Transformations
    def slot(self, unit, *args, **kwargs):
        """Re-agregate the series in slots of a length set by the ``unit`` parameter."""
        from .transformations import Aggregator
        aggregator = Aggregator(unit, *args, **kwargs)
        return aggregator.process(self)  



#==============================
#  Series slice
#==============================

class SeriesSlice(Series):
        
    def __init__(self, series, from_i, to_i, from_t=None, to_t=None, interpolation_method='linear', dense=False):
        # TODO: move to "fill_strategy" instead of "interpolation_mode"?
        self.series = series
        self.from_i = from_i
        self.to_i = to_i
        self.interpolation_method = interpolation_method
        self.len = None
        self.new_points = {}
        self.from_t = from_t
        self.to_t=to_t
        self.dense=dense
    
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
                                            
                    # Compute the new point values with respect to the entire interpolation           
                    new_point_data = {}
                    for data_label in self.series.data_labels():
        
                        if self.interpolation_method == 'linear':
        
                            # Compute the "growth" ratio
                            diff = this_point.data[data_label] - prev_point.data[data_label]
                            delta_t = this_point.t - prev_point.t
                            ratio = diff / delta_t
                            
                            # Compute the value of the data for the new point
                            new_point_data[data_label] = prev_point.data[data_label] + ((new_point_t-prev_point.t)*ratio)
        
                        elif self.interpolation_method == 'uniform':
                            raise NotImplementedError('uniform interpolation is not implemented yet')
                            new_point_data[data_label] = (prev_point.data[data_label] + this_point.data[data_label]) /2
                       
                        else:
                            raise Exception('Unknown interpolation method "{}"'.format(self.interpolation_method))
            
                    # Create the new point    
                    new_point = this_point.__class__(t = new_point_t, data = new_point_data)
                    new_point.valid_from = new_point_valid_from
                    new_point.valid_to = new_point_valid_to
                    new_point.reconstructed = True
                    
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
    
    @property
    def resolution(self):
        return self.series.resolution

    def data_labels(self):
        return self.series.data_labels()





