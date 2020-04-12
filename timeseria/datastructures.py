
from .time import s_from_dt , dt_from_s

# Setup logging
import logging
logger = logging.getLogger(__name__)

HARD_DEBUG = False


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
    
    def coordinates(self):
        return {k:v for k,v in self.__dict__.items() if not k.startswith('_')}
    
    def __repr__(self):
        return '{} with {}'.format(self.__class__.__name__, self.coordinates())
    
    def __str__(self):
        return self.__repr__()

        
class TimePoint(Point):
    
    def __init__(self, **kwargs):
        #if [*kwargs] != ['t']: # This migth speed up a bit but is for Python >= 3.5
        if list(kwargs.keys()) != ['t']:
            raise Exception('A TimePoint requires only a "t" coordinate (got "{}")'.format(kwargs))
        super(TimePoint, self).__init__(**kwargs)

    @ property
    def dt(self):
        return dt_from_s(self.t)

class DataPoint(Point):
    def __init__(self, **kwargs):
        try:
            self._data = kwargs.pop('data')
        except KeyError:
            raise Exception('A DataPoint requires a special "data" argument (got only "{}")'.format(kwargs))
        super(DataPoint, self).__init__(**kwargs)

    #def __float__(self):
    #    logger.debug('Called __float__ on {}'.format(self))
    #    try:
    #        return float(self.data)
    #    except:
    #        raise TypeError('Cannot convert "{}" to a float number'.format(self.data))

    def __repr__(self):
        return '{} with {} and data {}'.format(self.__class__.__name__, self.coordinates(), self.data)
    
    @property
    def data(self):
        return self._data


class DataTimePoint(DataPoint, TimePoint):
    pass



#======================
#  Series
#======================


class Serie(list):
    '''A list of items coming one after another where every item is guaranteed to be of the same type. Can optionally accept None types.'''

    # By default we actually accept everything
    __TYPE__ = object
    
    def __init__(self, *args, **kwargs):

        if 'accept_None' in kwargs and kwargs['accept_None']:
            self.accept_None = True
        else:
            self.accept_None = False

        for arg in args:
            self.append(arg)
            
        self._title = None

    def append(self, item):
        if HARD_DEBUG: logger.debug('Checking %s', item)
        if not isinstance(item, self.__TYPE__):
            if self.accept_None and item is None:
                pass
            else:
                raise TypeError('Got incompatible type "{}", can only accept "{}"'.format(item.__class__.__name__, self.__TYPE__.__name__))
        super(Serie, self).append(item)
            
    def extend(self, orher):
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




class PointSerie(Serie):
    __TYPE__ = Point


class TimePointSerie(PointSerie):
    '''A series where each item is guardanteed to be orderd'''

    __TYPE__ = TimePoint

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


class DataPointSerie(PointSerie):
    '''A series where each item is guardanteed to carry the same data type'''

    __TYPE__ = DataPoint

    # Check data compatibility
    def append(self, item):
        try:
            if HARD_DEBUG: logger.debug('Checking data compatibility: %s (accept_None=%s)', item.data, self.accept_None)
            if item.data is None and self.accept_None:
                pass
            else:
                if not type(self.item_data_reference) == type(item.data):
                    raise TypeError('{} vs {}'.format(self.item_data_reference.__class__.__name__, item.data.__class__.__name__))
                # TODO: if data is dict or list, check also len, key names and data types (excluding None if accept_None is set)
                
        except AttributeError:
            if HARD_DEBUG: logger.debug('Setting data reference: %s', item.data)
            self.item_data_reference = item.data
            
        super(DataPointSerie, self).append(item)


class DataTimePointSerie(DataPointSerie, TimePointSerie):
    '''A series where each item is a DataTimePoint'''

    __TYPE__ = DataTimePoint

    def plot(self, engine='dg'):
        if engine=='mp':
            from .plots import matplotlib_plot
            matplotlib_plot(self)
        elif engine=='dg':
            from .plots import dygraphs_plot
            dygraphs_plot(self)
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


