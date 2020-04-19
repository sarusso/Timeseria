from .time import s_from_dt , dt_from_s, UTC, timezonize

# Setup logging
import logging
logger = logging.getLogger(__name__)

HARD_DEBUG = False


#======================
#  Generic Serie
#======================

class Serie(list):
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
            except AttributeError: # TODO: just implement >  in the slot? or another operator?
                try:
                    if not item > self[-1]:
                        raise ValueError('Not in order ("{}" vs "{}")'.format(item,self[-1])) from None
                except TypeError:
                    raise TypeError('Object of class "{}" does not implement a "__gt__" or a "__succedes__" method, cannot append it to a Serie (which is ordered)'.format(item.__class__.__name__)) from None
        except IndexError:
            pass
        
        # Append
        super(Serie, self).append(item)
            
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


#======================
#  Points
#======================

class Point(object):
    def __init__(self, **kwargs):
        if not kwargs:
            raise Exception('A Point requires at least one coordinate, got none.')
        for kw in kwargs:
            if HARD_DEBUG: logger.debug('Setting %s to %s', kw, kwargs[kw])
            setattr(self, kw, kwargs[kw])
        #self.coordinates = kwargs.keys()
    
    @property
    def coordinates(self):
        return {k:v for k,v in self.__dict__.items() if not k.startswith('_')}
    
    def __repr__(self):
        return '{} with {}'.format(self.__class__.__name__, self.coordinates)
    
    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        if not self.coordinates == other.coordinates:
            return False
        for coordinate in self.coordinates:
            if getattr(self, coordinate) != getattr(other, coordinate):
                return False
        return True


class TimePoint(Point):
    
    def __init__(self, *args, **kwargs):

        # Handle time zone if any
        tz = kwargs.pop('tz', None)
        if tz:
            self._tz = timezonize(tz)
        
        # Cast or create
        if args:
            if isinstance(args[0], TimePoint):
                kwargs['t'] = args[0].t
                self._tz    = args[0].tz
            elif isinstance(args[0], int) or isinstance(args[0], float):
                kwargs['t'] = args[0]
            else:
                raise Exception('A TimePoint can be casted only from an int, float or by an object extending the TimePoint class itself (got "{}")'.format(args[0]))
 
        else:
            #if [*kwargs] != ['t']: # This migth speed up a bit but is for Python >= 3.5
            if list(kwargs.keys()) != ['t']:
                raise Exception('A TimePoint accepts only, and requires, a "t" coordinate (got "{}")'.format(kwargs))
            
        # Call parent init
        super(TimePoint, self).__init__(**kwargs)

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

    @ property
    def dt(self):
        return dt_from_s(self.t, tz=self.tz)

    def __repr__(self):
        return '{} @ t={} ({})'.format(self.__class__.__name__, self.t, self.dt)
    

class DataPoint(Point):
    def __init__(self, **kwargs):
        try:
            self._data = kwargs.pop('data')
        except KeyError:
            raise Exception('A DataPoint requires a special "data" argument (got only "{}")'.format(kwargs))
        super(DataPoint, self).__init__(**kwargs)

    def __repr__(self):
        return '{} with {} and data "{}"'.format(self.__class__.__name__, self.coordinates, self.data)
    
    def __eq__(self, other):
        if self._data != other._data:
            return False
        return super(DataPoint, self).__eq__(other)

    @property
    def data(self):
        return self._data


class DataTimePoint(DataPoint, TimePoint):
    
    def __repr__(self):
        return '{} @ t={} ({}) with data={}'.format(self.__class__.__name__, self.t, self.dt, self.data)
    


#======================
#  Point Series
#======================

class PointSerie(Serie):
    __TYPE__ = Point


class TimePointSerie(PointSerie):
    '''A series of TimePoints where each item is guaranteed to be ordered'''

    __TYPE__ = TimePoint

    def __init__(self, *args, **kwargs):

        tz = kwargs.pop('tz', None)
        if tz:
            self._tz = timezonize(tz)

        super(TimePointSerie, self).__init__(*args, **kwargs)

    # Check time ordering
    def append(self, item):
        try:
            if HARD_DEBUG: logger.debug('Checking time ordering for t="%s" (prev_t="%s")', item.t, self.prev_t)
            if item.t < self.prev_t:
                raise ValueError('Time t="{}" is out of order (prev t="{}")'.format(item.t, self.prev_t))
            
            if item.t == self.prev_t:
                raise ValueError('Time t="{}" is a duplicate'.format(item.t))
            
            self.prev_t = item.t
                
        except AttributeError:
            self.prev_t = item.t
       
        super(TimePointSerie, self).append(item)

    @property
    def tz(self):
        try:
            return self._tz
        except AttributeError:
            
            # Detect time zone on the fly
            detected_tz = None
            for item in self:
                if not detected_tz:
                    detected_tz = item.tz
                else:
                    if item.tz != detected_tz:
                        return UTC
            return detected_tz
    
    @tz.setter
    def tz(self, value):
        self._tz = timezonize(value) 


class DataPointSerie(PointSerie):
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
            
        super(DataPointSerie, self).append(item)


class DataTimePointSerie(DataPointSerie, TimePointSerie):
    '''A series of DataTimePoint where each item is guaranteed to carry the same data type and to be ordered'''

    __TYPE__ = DataTimePoint

    def plot(self, engine='dg', **kwargs):
        if 'aggregate_by' in kwargs:
            aggregate_by =  kwargs.pop('aggregate_by')
        else:
            aggregate_by = self.plot_aggregate_by
        if engine=='mp':
            from .plots import matplotlib_plot
            matplotlib_plot(self)
        elif engine=='dg':
            from .plots import dygraphs_plot
            dygraphs_plot(self, aggregate_by=aggregate_by)
        else:
            raise Exception('Unknown plotting engine "{}'.format(engine))

    @property
    def plot_aggregate_by(self):
        try:
            return self._plot_aggregate_by
        except AttributeError:
            if len(self)  > 10000:
                aggregate_by = 10**len(str(int(len(self)/10000.0)))
            else:
                aggregate_by = None
            return aggregate_by

    def __repr__(self):
        return '{} of #{} {}s, from {} to {}'.format(self.__class__.__name__, len(self), self.__TYPE__.__name__, TimePoint(self[0]), TimePoint(self[-1]))
    


#======================
#  Slots
#======================

class Slot(object):
    def __init__(self, start, end):
        if not isinstance(start, Point):
            raise TypeError('Slot start must be a Point object (got "{}")'.format(start.__class__.__name__))
        if not isinstance(end, Point):
            raise TypeError('Slot end must be a Point object (got "{}")'.format(end.__class__.__name__))
        if set(start.coordinates.keys()) != set(end.coordinates.keys()):
            raise ValueError('Slot start and end dimensions must be the same (got "{}" vs "{}")'.format(set(start.coordinates.keys()), set(end.coordinates.keys())))
        if start == end:
            raise ValueError('{} start and end must not be the same (got start="{}", end="{}")'.format(self.__class__.__name__, start,end))
                
        self.start = start
        self.end   = end

    def __repr__(self):
        return '{} with start="{}" and end="{}"'.format(self.__class__.__name__, self.start, self.end)
    
    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        if not self.start == other.start:
            return False
        if not self.end == other.end:
            return False
        return True


class TimeSlot(Slot):
    
    def __init__(self, start, end): 
        if not isinstance(start, TimePoint):
            raise TypeError('{} start must be a TimePoint object (got "{}")'.format(self.__class__.__name__, start.__class__.__name__))
        if not isinstance(end, TimePoint):
            raise TypeError('{} end must be a TimePoint object (got "{}")'.format(self.__class__.__name__, end.__class__.__name__))
        if start == end:
            raise ValueError('{} start and end must not be the same (got start="{}", end="{}")'.format(self.__class__.__name__, start,end))
            
        # Note: here we do not call the parent init, as it will result in performing again a number of checks already carried out in the TimePoint.
        # TODO: not nice, skip the check instead.
        self.start = start
        self.end   = end        

    def __succedes__(self, other):
        if other.end.t != self.start.t:
            return False
        else:
            return True
        
    @property
    def duration(self):
        return (self.end.t - self.start.t)
    
    #@property
    #def t(self):
    #    #return (self.start.t + (self.end.t - self.start.t))
    #    return self.start.t


class DataSlot(Slot):
    def __init__(self, **kwargs):
        try:
            self._data = kwargs.pop('data')
        except KeyError:
            raise Exception('A DataSlot requires a special "data" argument (got only "{}")'.format(kwargs))

        # Coverage. TODO: understand if we want it, if  we want it here like this.
        coverage = kwargs.pop('coverage', None)
        if coverage:
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


class DataTimeSlot(DataSlot, TimeSlot):
    
    def __repr__(self):
        return '{} @ t=[{},{}] ([{},{}]) with data={} and coverage={}'.format(self.__class__.__name__, self.start.t, self.end.t, self.start.dt, self.end.dt, self.data, self.coverage)
    



#======================
#  Slot Series
#======================

class SlotSerie(Serie):
    __TYPE__ = Slot


class TimeSlotSerie(SlotSerie):
    '''A series of TimeSlots where each item is guaranteed to be ordered'''

    __TYPE__ = TimeSlot


class DataSlotSerie(SlotSerie):
    '''A series of DataSlots where each item is guaranteed to carry the same data type'''

    __TYPE__ = DataSlot

    # Check data compatibility
    def append(self, item):
        try:
            if HARD_DEBUG: logger.debug('Checking data compatibility: %s', item.data)
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
            
        super(DataSlotSerie, self).append(item)
    
    @property
    def data_keys(self):
        if len(self) == 0:
            return None
        else:
            return set(self[0].data.keys())


class DataTimeSlotSerie(DataSlotSerie, TimeSlotSerie):
    '''A series of DataTimeSlots where each item is guaranteed to carry the same data type and to be ordered'''

    __TYPE__ = DataTimeSlot

    def plot(self, engine='dg', **kwargs):
        if 'aggregate_by' in kwargs:
            aggregate_by =  kwargs.pop('aggregate_by')
        else:
            aggregate_by = self.plot_aggregate_by
        if engine=='mp':
            from .plots import matplotlib_plot
            matplotlib_plot(self)
        elif engine=='dg':
            from .plots import dygraphs_plot
            dygraphs_plot(self, aggregate_by=aggregate_by)
        else:
            raise Exception('Unknowmn plotting engine "{}'.format(engine))

    @property
    def plot_aggregate_by(self):
        try:
            return self._plot_aggregate_by
        except AttributeError:
            if len(self)  > 10000:
                aggregate_by = 10**len(str(int(len(self)/10000.0)))
            else:
                aggregate_by = None
            return aggregate_by

    def __repr__(self):
        try:
            return '{} of #{} {} {}s, from {} to {}'.format(self.__class__.__name__, len(self), self.timeSpan, self.__TYPE__.__name__, TimePoint(self[0].start), TimePoint(self[-1].end))
        except:
            return '{} of #{} {}s, from {} to {}'.format(self.__class__.__name__, len(self), self.__TYPE__.__name__, TimePoint(self[0].start), TimePoint(self[-1].end))
            
