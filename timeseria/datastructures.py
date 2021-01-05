from .time import s_from_dt , dt_from_s, UTC, timezonize
from .units import Unit, TimeUnit
from .utilities import is_close
from copy import deepcopy
from pandas import DataFrame
from datetime import datetime

# Setup logging
import logging
logger = logging.getLogger(__name__)


HARD_DEBUG = False

AGGREGATE_THRESHOLD = 10000

#======================
#  Generic Series
#======================

class Series(list):
    '''A list of items coming one after another where every item is guaranteed to be of the same type and in an order or succession.'''

    # 5,6,7 are integer succession. 5.3, 5.4, 5.5 are too in a succesisons. 5,6,8 are not in a succession. 
    # a group or a number of related or similar things, events, etc., arranged or occurring in temporal, spatial, or other order or succession; sequence.

    # By default the type is not defined
    __TYPE__ = None
    
    def __init__(self, *args, **kwargs):

        #if 'accept_None' in kwargs and kwargs['accept_None']:
        #    self.accept_None = True
        #else:
        #    self.accept_None = False

        for arg in args:
            self.append(arg)
            
        self._title = None

    def append(self, item):
        if HARD_DEBUG: logger.debug('Checking %s', item)
        
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
                    raise ValueError('Not in succession ("{}" vs "{}")'.format(item,self[-1])) from None                    
            except IndexError:
                raise
            except AttributeError:
                try:
                    if not item > self[-1]:
                        raise ValueError('Not in order ("{}" vs "{}")'.format(item,self[-1])) from None
                except TypeError:
                    raise TypeError('Object of class "{}" does not implement a "__gt__" or a "__succedes__" method, cannot append it to a Series (which is ordered)'.format(item.__class__.__name__)) from None
        except IndexError:
            pass
        
        # Append
        super(Series, self).append(item)
            
    def extend(self, orher):
        raise NotImplementedError

    def merge(self, orher):
        raise NotImplementedError
    
    def __sum__(self, other):
        raise NotImplementedError
    
    @property
    def title(self):
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
    
    # Duplicate
    def duplicate(self):
        return deepcopy(self)
    
    # Slice
    def __getitem__(self, key):
        if isinstance(key, slice):            
            indices = range(*key.indices(len(self)))
            series = self.__class__()
            for i in indices:
                series.append(super(Series, self).__getitem__(i))
            try:
                series.mark = self.mark
            except:
                pass
            return series
        else:
            return super(Series, self).__getitem__(key)


#======================
#  Points
#======================

class Point(object):

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
        return self.coordinates[0]
    
    def __gt__(self, other):
        if self.t > other.t:
            return True
        else:
            return False

    @property
    def tz(self):
        try:
            return self._tz
        except AttributeError:
            return UTC
    
    def change_timezone(self, new_timezone):
        self._tz = timezonize(new_timezone)

    @property
    def dt(self):
        return dt_from_s(self.t, tz=self.tz)

    def __repr__(self):
        return '{} @ {} ({})'.format(self.__class__.__name__, self.t, self.dt)
        # return '{} @ t={} ({})'.format(self.__class__.__name__, self.t, self.dt)
    
    
    # Tricks to make them behave as slots
    #@property
    #def start(self):
    #    return self
    
    #@property
    #def end(self):
    #    return self



class DataPoint(Point):
    def __init__(self, *args, **kwargs):
        try:
            self._data = kwargs.pop('data')
        except KeyError:
            raise Exception('A DataPoint requires a special "data" argument (got only "{}")'.format(kwargs))

        coverage = kwargs.pop('coverage', None)
        if coverage is not None:
            self._coverage=coverage
        
        super(DataPoint, self).__init__(*args, **kwargs)

    def __repr__(self):
        try:
            return '{} with data "{}" and data_loss="{}"'.format(super(DataPoint, self).__repr__(), self.data, 1-self._coverage)            
        except:
            return '{} with data "{}"'.format(super(DataPoint, self).__repr__(), self.data)
    
    def __eq__(self, other):
        if self._data != other._data:
            return False
        return super(DataPoint, self).__eq__(other)

    @property
    def data(self):
        return self._data

    @property
    def data_loss(self):
        try:
            return 1-self._coverage
        except AttributeError:
            return None

class DataTimePoint(DataPoint, TimePoint):
    pass

    # NOTE: the __repr__ used is from the DataPoint above, which in turn uses the TimePoint one.


#======================
#  Point Series
#======================

class PointSeries(Series):
    __TYPE__ = Point


class TimePointSeries(PointSeries):
    '''A series of TimePoints where each item is guaranteed to be ordered'''

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
                
                if HARD_DEBUG: logger.debug('Checking time ordering for t="%s" (prev_t="%s")', item.t, self.prev_t)
                if item.t < self.prev_t:
                    raise ValueError('Time t="{}" is out of order (prev t="{}")'.format(item.t, self.prev_t))
                
                if item.t == self.prev_t:
                    raise ValueError('Time t="{}" is a duplicate'.format(item.t))
                
                if self._resolution is None:
                    self._resolution = item.t - self.prev_t
                    
                elif self._resolution != 'variable':
                    if self._resolution != item.t - self.prev_t:
                        self._resolution = 'variable'

            self.prev_t = item.t
                
        except AttributeError:
            self.prev_t = item.t
       
        super(TimePointSeries, self).append(item)

    @property
    def tz(self):
        # Note: we compute the tz on the fly beacuse for point time series we assume to use the tz
        # attribute way lass than the slot time series, where the tz is instead computed at append-time.
        try:
            return self._tz
        except AttributeError:
            # Detect time zone on the fly
            detected_tz = None
            for item in self:
                if not detected_tz:
                    detected_tz = item.tz
                else:
                    # Terrible but seems like no other way to compare pytz.tzfile.* classes
                    if str(item.tz) != str(detected_tz): 
                        return UTC
            return detected_tz
    
    @tz.setter
    def tz(self, value):
        self._tz = timezonize(value) 

    def change_timezone(self, new_timezone):
        for time_point in self:
            time_point.change_timezone(new_timezone)
        self.tz = time_point.tz

    @property
    def resolution(self):
        return self._resolution
        #if isinstance(self._resolution, Unit):
        #    return self._resolution
        #else:
        #    from .transformations import unit_to_TimeUnit
        #    try:
        #        return unit_to_TimeUnit(self._resolution)
        #    except:
        #        return self._resolution

    def cut(self, from_t=None, to_t=None, from_dt=None, to_dt=None):
        if from_dt:
            if from_t is not None:
                raise Exception('Cannot set both from_t and from_dt')
            from_t = s_from_dt(from_dt)
        if to_dt:
            if to_t is not None:
                raise Exception('Cannot set both to_t and to_dt')
            to_t = s_from_dt(to_dt)
            
        if from_t is not None and to_t is not None:
            if from_t >= to_t:
                raise Exception('Got from >= to')
        
        series = self.__class__()
        for item in self:
            if from_t is not None and to_t is not None:
                if item.t >= from_t and item.t <to_t:
                    series.append(item)
            else:
                if from_t is not None:
                    if item.t >= from_t:
                        series.append(item) 
                if to_t is not None:
                    if item.t < to_t:
                        series.append(item)
        return series           
        
    

class DataPointSeries(PointSeries):
    '''A series of DataPoints where each item is guaranteed to carry the same data type'''

    __TYPE__ = DataPoint

    # Check data compatibility
    def append(self, item):
        try:
            if HARD_DEBUG: logger.debug('Checking data compatibility: %s ', item.data)
            #if item.data is None and self.accept_None:
            #    pass
            #else:
            if not type(self.item_data_reference) == type(item.data):
                raise TypeError('Got different data: {} vs {}'.format(self.item_data_reference.__class__.__name__, item.data.__class__.__name__))
            if isinstance(self.item_data_reference, list):
                if len(self.item_data_reference) != len(item.data):
                    raise ValueError('Got different data lengths: {} vs {}'.format(len(self.item_data_reference), len(item.data)))
            if isinstance(self.item_data_reference, dict):
                if set(self.item_data_reference.keys()) != set(item.data.keys()):
                    raise ValueError('Got different data keys: {} vs {}'.format(self.item_data_reference.keys(), item.data.keys()))
            
        except AttributeError:
            if HARD_DEBUG: logger.debug('Setting data reference: %s', item.data)
            self.item_data_reference = item.data
            
        super(DataPointSeries, self).append(item)

    def data_keys(self):
        if len(self) == 0:
            return None
        else:
            # TODO: can we optimize here? Computing them once and then serving them does not work if someone changes data keys...
            try:
                return list(self[0].data.keys())
            except AttributeError:
                return list(range(len(self[0].data)))


class DataTimePointSeries(DataPointSeries, TimePointSeries):
    '''A series of DataTimePoint where each item is guaranteed to carry the same data type and to be ordered'''

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
        data_keys = self.data_keys()
        
        if self[0].data_loss is not None:
            dump_data_loss = True
        else:
            dump_data_loss = False
        
        if dump_data_loss:
            columns = ['Timestamp'] + data_keys + ['data_loss']
        else:
            columns = ['Timestamp'] + data_keys
            
        df = DataFrame(columns=columns)
        for item in self:
            values = [item.data[key] for key in data_keys]
            if dump_data_loss:
                df = df.append(DataFrame([[item.dt]+values+[item.data_loss]], columns=columns))
            else:
                df = df.append(DataFrame([[item.dt]+values], columns=columns))                
        df = df.set_index('Timestamp')
        return df
        
    def plot(self, engine='dg', **kwargs):
        if 'aggregate' in kwargs:
            if kwargs['aggregate'] is False:
                aggregate_by = None
            else:
                aggregate_by = self.plot_aggregate_by
            kwargs.pop('aggregate')
        else:
            if 'aggregate_by' in kwargs:
                aggregate_by =  kwargs.pop('aggregate_by')
            else:
                aggregate_by = self.plot_aggregate_by
        if engine=='mp':
            from .plots import matplotlib_plot
            matplotlib_plot(self)
        elif engine=='dg':
            from .plots import dygraphs_plot
            dygraphs_plot(self, aggregate_by=aggregate_by, **kwargs)
        else:
            raise Exception('Unknown plotting engine "{}'.format(engine))

    @property
    def plot_aggregate_by(self):
        try:
            return self._plot_aggregate_by
        except AttributeError:
            if len(self)  > AGGREGATE_THRESHOLD:
                aggregate_by = 10**len(str(int(len(self)/float(AGGREGATE_THRESHOLD))))
            else:
                aggregate_by = None
            return aggregate_by

    def __repr__(self):
        if len(self):
            return 'Time series of #{} points, from point @ {} ({}) to point @ {} ({})'.format(len(self), self[0].t, self[0].dt, self[-1].t, self[-1].dt)
        else:
            return 'Time series of #0 points'


#======================
#  Slots
#======================

class Slot(object):
    
    __POINT_TYPE__ = Point
    
    def __init__(self, start, end=None, unit=None):
        
        if not isinstance(start, self.__POINT_TYPE__):
            raise TypeError('Slot start must be a Point object (got "{}")'.format(start.__class__.__name__))
        if end is None and unit is not None:
            if len(start.coordinates)>1:
                raise Exception('Sorry, setting a start and a unit only works in unidimensional spaces')
            
            # Handle start + unit
            if isinstance(unit, Unit):
                end = start + unit
            else:
                end = start.__class__(start.coordinates[0] + unit)    
            
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
        try:
            self._length
        except AttributeError:
            self._length = self._compute_length(self.start, self.end)
        return self._length

    @property
    def unit(self):
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

    __POINT_TYPE__ = TimePoint

   
    def __init__(self, start=None, end=None, unit=None, t=None, dt=None):
        
        # Handle t and dt shortcuts
        if t:
            start=TimePoint(t=t)
        if dt:
            start=TimePoint(dt=dt)
        
        if not isinstance(start, self.__POINT_TYPE__):
            raise TypeError('Slot start must be a Point object (got "{}")'.format(start.__class__.__name__))

        try:
            if start.tz != end.tz:
                raise ValueError('{} start and end must have the same time zone (got start.tz="{}", end.tz="{}")'.format(self.__class__.__name__, start.tz, end.tz))
        except AttributeError:
            if end is None:
                # We are using the Unit, use the start
                self.tz = start.tz
            else:
                # If we don't have a time zone, we don't have TimePoints, the parent will make the Slot creation fail with a TypeError
                pass
        else:    
            self.tz = start.tz
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

    def change_timezone(self, new_timezone):
        self.start.change_timezone(new_timezone)
        self.end.change_timezone(new_timezone)
        self.tz = self.start.tz
    
    @property
    def t(self):
        return self.start.t
    
    @property
    def dt(self):
        return self.start.dt


class DataSlot(Slot):
    def __init__(self, **kwargs):
        try:
            self._data = kwargs.pop('data')
        except KeyError:
            raise Exception('A DataSlot requires a special "data" argument (got only "{}")'.format(kwargs))

        coverage = kwargs.pop('coverage', None)
        if coverage is not None:
            self._coverage=coverage

        super(DataSlot, self).__init__(**kwargs)

    def __repr__(self):
        return '{} with start="{}" and end="{}"'.format(self.__class__.__name__, self.start, self.end)

    def __eq__(self, other):
        if self._data != other._data:
            return False
        return super(DataSlot, self).__eq__(other)

    @property
    def data(self):
        return self._data

    @property
    def coverage(self):
        try:
            return self._coverage
        except AttributeError:
            return None
    
    @property
    def data_loss(self):
        try:
            return 1-self._coverage
        except AttributeError:
            return None


    @property
    def data_reconstructed(self):
        try:
            return self._data_reconstructed
        except AttributeError:
            return None


class DataTimeSlot(DataSlot, TimeSlot):
    
    def __repr__(self):
        #if self.coverage is not None:
        #    return '{} @ t=[{},{}] ([{},{}]) with data={} and coverage={}'.format(self.__class__.__name__, self.start.t, self.end.t, self.start.dt, self.end.dt, self.data, self.coverage)
        #else:
        #    return '{} @ t=[{},{}] ([{},{}]) with data={}'.format(self.__class__.__name__, self.start.t, self.end.t, self.start.dt, self.end.dt, self.data)

        if self.coverage is not None:
            return '{} @ [{},{}] ([{},{}]) with data={} and coverage={}'.format(self.__class__.__name__, self.start.t, self.end.t, self.start.dt, self.end.dt, self.data, self.coverage)
        else:
            return '{} @ [{},{}] ([{},{}]) with data={}'.format(self.__class__.__name__, self.start.t, self.end.t, self.start.dt, self.end.dt, self.data)
        


#======================
#  Slot Series
#======================

class SlotSeries(Series):
    __TYPE__ = Slot

    def append(self, item):
        
        # Slots can belong to the same series if they are in succession (tested with the __succedes__ method)
        # and if they have the same unit, which we test here instead as the __succedes__ is more general.
        try:
            if self._resolution != item.unit:
                # Try for floatign point precision errors
                abort = False
                try:
                    if not  is_close(self._resolution, item.unit):
                        abort = True
                except (TypeError, ValueError):
                    abort = True
                if abort:
                    raise ValueError('Cannot add items with different units (I have "{}" and you tried to add "{}")'.format(self._resolution, item.unit))
        except AttributeError:
            self._resolution = item.unit

        # Call parent append
        super(SlotSeries, self).append(item)



class TimeSlotSeries(SlotSeries):
    '''A series of TimeSlots where each item is guaranteed to be ordered'''

    __TYPE__ = TimeSlot

    def append(self, item):
        
        # Check for the same time zone
        try:
            if self.tz != item.tz:
                raise ValueError('Cannot add items on different time zones (I have "{}" and you tried to add "{}")'.format(self.tz, item.start.tz))
        except AttributeError:
            self.tz = item.tz
        super(TimeSlotSeries, self).append(item)
        
    def change_timezone(self, new_timezone):
        for time_slot in self:
            time_slot.change_timezone(new_timezone)
        self.tz = time_slot.tz

    @property
    def resolution(self):
        return self._resolution
        #if isinstance(self._resolution, Unit):
        #    return self._resolution
        #else:
        #    from .transformations import unit_to_TimeUnit
        #    try:
        #        return unit_to_TimeUnit(self._resolution)
        #    except:
        #        return self._resolution


class DataSlotSeries(SlotSeries):
    '''A series of DataSlots where each item is guaranteed to carry the same data type'''

    __TYPE__ = DataSlot

    # Check data compatibility
    def append(self, item):
        
        # Check for data compatibility
        try:
            if not type(self.item_data_reference) == type(item.data):
                raise TypeError('Got different data: {} vs {}'.format(self.item_data_reference.__class__.__name__, item.data.__class__.__name__))
            if isinstance(self.item_data_reference, list):
                if len(self.item_data_reference) != len(item.data):
                    raise ValueError('Got different data lengths: {} vs {}'.format(len(self.item_data_reference), len(item.data)))
            if isinstance(self.item_data_reference, dict):
                if set(self.item_data_reference.keys()) != set(item.data.keys()):
                    raise ValueError('Got different data keys: {} vs {}'.format(self.item_data_reference.keys(), item.data.keys()))

        except AttributeError:
            # TODO: uniform self.tz, self._resolution, self.item_data_reference
            self.item_data_reference = item.data
        
        super(DataSlotSeries, self).append(item)
    
    def data_keys(self):
        if len(self) == 0:
            return None
        else:
            # TODO: can we optimize here? Computing them once and then serving them does not work if someone changes data keys...
            try:
                return list(self[0].data.keys())
            except AttributeError:
                return list(range(len(self[0].data)))


class DataTimeSlotSeries(DataSlotSeries, TimeSlotSeries):
    '''A series of DataTimeSlots where each item is guaranteed to carry the same data type and to be ordered'''

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
            
            # Human
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
        data_keys = self.data_keys()
        
        if self[0].data_loss is not None:
            dump_data_loss = True
        else:
            dump_data_loss = False
        
        if dump_data_loss:
            columns = ['Timestamp'] + data_keys + ['data_loss']
        else:
            columns = ['Timestamp'] + data_keys
            
        df = DataFrame(columns=columns)
        for item in self:
            values = [item.data[key] for key in data_keys]
            if dump_data_loss:
                df = df.append(DataFrame([[item.dt]+values+[item.data_loss]], columns=columns))
            else:
                df = df.append(DataFrame([[item.dt]+values], columns=columns))                
        df = df.set_index('Timestamp')
        return df

    def plot(self, engine='dg', **kwargs):
        if 'aggregate' in kwargs:
            if kwargs['aggregate'] is False:
                aggregate_by = None
            else:
                aggregate_by = self.plot_aggregate_by
            kwargs.pop('aggregate')
        else:
            if 'aggregate_by' in kwargs:
                aggregate_by =  kwargs.pop('aggregate_by')
            else:
                aggregate_by = self.plot_aggregate_by
        if engine=='mp':
            from .plots import matplotlib_plot
            matplotlib_plot(self)
        elif engine=='dg':
            from .plots import dygraphs_plot
            dygraphs_plot(self, aggregate_by=aggregate_by, **kwargs)
        else:
            raise Exception('Unknown plotting engine "{}'.format(engine))

    @property
    def plot_aggregate_by(self):
        # By default do not aggregate
        #return None
        
        # Aggregate if more than AGGREGATE_THRESHOLD slots
        try:
            return self._plot_aggregate_by
        except AttributeError:
            if len(self)  > AGGREGATE_THRESHOLD:
                aggregate_by = 10**len(str(int(len(self)/float(AGGREGATE_THRESHOLD))))
            else:
                aggregate_by = None
            return aggregate_by

    def __repr__(self):
        if len(self):
            if isinstance(self._resolution, TimeUnit):
                resolution_str = str(self._resolution)
            else:
                resolution_str = str(self._resolution)# + 's' 
            # TODO: "slots of unit" ?
            return 'Time series of #{} slots of {}, from slot starting @ {} ({}) to slot ending @ {} ({})'.format(len(self), resolution_str, self[0].start.t, self[0].start.dt, self[-1].end.t, self[-1].end.dt)            
        else:
            return 'Time series of #0 slots'
    
    
    