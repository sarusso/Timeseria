# -*- coding: utf-8 -*-
"""Operations on the series, returning both scalar and other series."""

from copy import copy, deepcopy
from .time import s_from_dt
from .datastructures import Series, Slot, Point, PointSeries
from .utilities import is_close
from .units import TimeUnit, Unit

# Setup logging
import logging
logger = logging.getLogger(__name__)


#=======================
#  Base Operation class
#=======================

class Operation():
    """Generic operation class (callable object)."""

    @classmethod
    def __str__(cls):
        return '{} operation'.format(cls.__name__)
    
    @classmethod
    def __repr__(cls):
        return '{}'.format(cls.__name__.lower())
    
    def __call__(self, *args, **kwargs):
        raise NotImplementedError('No operation logic implemented.')
    
    @property
    def __name__(self):
        return self.__class__.__name__.lower()


class ScalarOperation(Operation):
    """An operation operating on a series and returning a scalar (callable object)."""
    pass


class SeriesOperation(Operation):
    """An operation operating on a series and returning a series (callable object)."""
    pass


#=======================
#  Scalar Operations
#=======================

class Max(ScalarOperation):
    """Maximum operation (callable object). Comes also pre-instantiated as the ``max()``
    function in the same module (accessible as ``timeseria.operations.max``)."""
    
    def __init__(self):
        self.built_in_max = max
    
    def __call__(self, arg, data_key=None, **kwargs):
        if not isinstance(arg, Series):
            return self.built_in_max(arg)
        else:
            series = arg
            mins = {data_key: None for data_key in series.data_keys()}
            for item in series:
                for _data_key in mins:
                    if mins[_data_key] is None:
                        mins[_data_key] = item.data[_data_key]
                    else:
                        if item.data[_data_key] > mins[_data_key]:
                            mins[_data_key] = item.data[_data_key]
            
            if data_key is not None:
                return mins[data_key]
            if len(mins) == 1:
                return mins[series.data_keys()[0]]
            else:
                return mins


class Min(ScalarOperation):
    """Minimum operation (callable object). Comes also pre-instantiated as the ``min()``
    function in the same module (accessible as ``timeseria.operations.min``)."""
    
    def __init__(self):
        self.built_in_min = min
    
    def __call__(self, arg, data_key=None, **kwargs):
        if not isinstance(arg, Series):
            return self.built_in_min(arg)
        else:
            series = arg
            mins = {data_key: None for data_key in series.data_keys()}
            for item in series:
                for _data_key in mins:
                    if mins[_data_key] is None:
                        mins[_data_key] = item.data[_data_key]
                    else:
                        if item.data[_data_key] < mins[_data_key]:
                            mins[_data_key] = item.data[_data_key]
            
            if data_key is not None:
                return mins[data_key]
            if len(mins) == 1:
                return mins[series.data_keys()[0]]
            else:
                return mins


class Avg(ScalarOperation):
    """Average operation (callable object)."""
    
    def __call__(self, arg, data_key=None, **kwargs):
        if not isinstance(arg, Series):
            raise NotImplementedError('Avg implemented only on series objects')
        else:
            series = arg
            sums = {data_key: None for data_key in series.data_keys()}
            for item in series:
                for _data_key in sums:
                    if sums[_data_key] is None:
                        sums[_data_key] = item.data[_data_key]
                    else:
                        sums[_data_key] += item.data[_data_key]
            avgs={}
            for _data_key in sums:
                avgs[_data_key] = sums[_data_key] / len(series)
            if data_key is not None:
                return avgs[data_key]
            if len(avgs) == 1:
                return avgs[series.data_keys()[0]]
            else:
                return avgs


class WAvg(ScalarOperation):
    """Weighted average operation (callable object)."""

    def __call__(self, series, prev_point=None, next_point=None):
        raise NotImplementedError('Weighted Average is not yet implemented')
        #keys = None
        #avgs = {}
        #count=0
        #for data_time_point in series:
        #    count+=1              
        #    if not keys:
        #        keys=data_time_point.data.keys()
        #    for key in keys:
        #        if key not in avgs:
        #            avgs[key] = 0
        #        avgs[key] += data_time_point.data[key]
        #for key in keys:
        #    new_avgs = {key:avgs[key]/count for key in keys}
        #return new_avgs


class Sum(ScalarOperation):
    """Sum operation (callable object)."""

    def __init__(self):
        self.built_in_sum = sum
    
    def __call__(self, arg, data_key=None, **kwargs):
        if not isinstance(arg, Series):
            return self.built_in_sum(arg)
        else:
            series = arg
            sums = {data_key: 0 for data_key in series.data_keys()}
            for item in series:
                for _data_key in sums:
                    sums[_data_key] += item.data[_data_key]
            if data_key is not None:
                return sums[data_key]
            if len(sums) == 1:
                return sums[series.data_keys()[0]]
            else:
                return sums


#=======================
#  Series operations
#=======================

class Derivative(SeriesOperation):
    """Derivative operation (callable object). Must operate on a fixed resolution time series"""
    
    def __call__(self, timeseries, inplace=False, normalize=True, diffs=False):
        
        if normalize or not diffs:
            if isinstance(timeseries, PointSeries) and timeseries.resolution == 'variable':
                raise ValueError('Variable resolutions are not supported in this mode. Resample or slot the time series first.')
            postfix='derivative'
        else:
            postfix='diff'

        if not inplace:
            der_timeseries = timeseries.__class__()

        data_keys = timeseries.data_keys()
        for i, item in enumerate(timeseries):

            if not inplace:
                data = {}
            
            for key in data_keys:
                
                if diffs:
                    # Just compute the diffs
                    if i == 0:
                        diff = 0
                    else:
                        diff = timeseries[i].data[key] - timeseries[i-1].data[key]
                else:
                    # Compute the derivative
                    if i == 0:
                        # Right increment for the first item
                        diff = timeseries[i+1].data[key] - timeseries[i].data[key]
                    elif i == len(timeseries)-1:
                        # Left increment for the last item. Divide by two if not normlaizing by time resolution
                        diff = timeseries[i].data[key] - timeseries[i-1].data[key]
                    else:
                        # Both left and right increment for the items in the middle, averaged.
                        diff =  ((timeseries[i+1].data[key] - timeseries[i].data[key]) + (timeseries[i].data[key] - timeseries[i-1].data[key]))
                        diff = diff/2

                # Normalize the increment to get the actual derivative
                if normalize:
                    if isinstance(timeseries.resolution, TimeUnit):
                        diff = diff / timeseries.resolution.duration_s(item.dt)               
                    elif isinstance(timeseries.resolution, Unit):
                        diff = diff / timeseries.resolution.value
                    else:
                        diff = diff / timeseries.resolution

                # Add data
                if not inplace:
                    data['{}_{}'.format(key, postfix)] = diff
                else:
                    item.data['{}_{}'.format(key, postfix)] = diff
            
            # Create the item
            if not inplace:

                if isinstance(timeseries[0], Point):
                    der_timeseries.append(timeseries[0].__class__(t = item.t,
                                                                  tz = item.tz,
                                                                  data = data,
                                                                  data_loss = item.data_loss))         
                elif isinstance(timeseries[0], Slot):
                    der_timeseries.append(timeseries[0].__class__(start = item.start,
                                                                  unit = item.unit,
                                                                  data = data,
                                                                  data_loss = item.data_loss))                
                else:
                    raise NotImplementedError('Working on series other than slots or points not yet implemented')

        if not inplace:
            return der_timeseries


class Integral(SeriesOperation):
    """Integral operation (callable object). Must operate on a fixed resolution time series."""
    
    def __call__(self, timeseries, inplace=False, normalize=True, c=0):
        
        if normalize:
            if isinstance(timeseries, PointSeries) and timeseries.resolution == 'variable':
                raise ValueError('Variable resolutions are not supported. Resample or slot the time series first.')
            postfix='integral'
        else:
            postfix='csum'

        if not inplace:
            der_timeseries = timeseries.__class__()

        last_values={}

        data_keys = timeseries.data_keys()
        for i, item in enumerate(timeseries):

            if not inplace:
                data = {}
            
            for key in data_keys:
                
                # Get the value
                if not normalize:
                    value = timeseries[i].data[key]
                else:
                    # Compute the integral
                    if i == 0:
                        value=0
                        prev_right_component = timeseries[i].data[key]
                    else:
                        left_component = (timeseries[i-1].data[key]*2) - prev_right_component
                        # The "value" below is actually an increment, see below at  value = last_values[key] + value 
                        value = left_component 
                        prev_right_component = left_component

                # Normalize the cumulative to get the actual integral
                if normalize:
                    if isinstance(timeseries.resolution, TimeUnit):
                        value = value * timeseries.resolution.duration_s(item.dt)               
                    elif isinstance(timeseries.resolution, Unit):
                        value = value * timeseries.resolution.value
                    else:
                        value = value * timeseries.resolution

                # Sum to previous or apply offset if any
                try:
                    value = last_values[key] + value
                except:
                    try:
                        # Do we have offsets as a dict?
                        value = value + c[key]
                    except:
                        value = value + c               

                # Add data
                if not inplace:
                    data['{}_{}'.format(key, postfix)] = value
                else:
                    item.data['{}_{}'.format(key, postfix)] = value
                
                last_values[key] = value

           
            # Create the item
            if not inplace:

                if isinstance(timeseries[0], Point):
                    der_timeseries.append(timeseries[0].__class__(t = item.t,
                                                                  tz = item.tz,
                                                                  data = data,
                                                                  data_loss = item.data_loss))         
                elif isinstance(timeseries[0], Slot):
                    der_timeseries.append(timeseries[0].__class__(start = item.start,
                                                                  unit = item.unit,
                                                                  data = data,
                                                                  data_loss = item.data_loss))                
                else:
                    raise NotImplementedError('Working on series other than slots or points not yet implemented')

        if not inplace:
            return der_timeseries


class Diff(Derivative):
    """Incremental differences operation (callable object)."""
    def __call__(self, timeseries, inplace=False):
        return super(Diff, self).__call__(timeseries, inplace=inplace, normalize=False, diffs=True)


class CSum(Integral):
    """Cumulative sum operation (callable object)."""
    def __call__(self, timeseries, inplace=False, offset=0):
        return super(CSum, self).__call__(timeseries, inplace=inplace, normalize=False, c=offset)


class Normalize(SeriesOperation):
    """Normalization operation (callable object)"""
    
    def __call__(self, timeseries, inplace=False):
        
        if not inplace:
            normalized_timeseries = timeseries.__class__()

        data_keys = timeseries.data_keys()

        # Get min and max for the data keys
        for i, item in enumerate(timeseries):
            
            if i == 0:
                mins = {key: item.data[key] for key in data_keys}
                maxs = {key: item.data[key] for key in data_keys}        
            else:
                for key in data_keys:
                    if item.data[key] < mins[key]:
                        mins[key] = item.data[key]
                    if item.data[key] > maxs[key]:
                        maxs[key] = item.data[key]                
                
        
        for i, item in enumerate(timeseries):
 
            if not inplace:
                data = {}
             
            for key in data_keys:

                # Normalize data
                if not inplace:
                    data[key] = (item.data[key] - mins[key])  / (maxs[key]-mins[key])
                else:
                    item.data[key] = (item.data[key] - mins[key])  / (maxs[key]-mins[key])
             
            # Create the item
            if not inplace:
 
                if isinstance(timeseries[0], Point):
                    normalized_timeseries.append(timeseries[0].__class__(t = item.t,
                                                                         tz = item.tz,
                                                                         data = data,
                                                                         data_loss = item.data_loss))         
                elif isinstance(timeseries[0], Slot):
                    normalized_timeseries.append(timeseries[0].__class__(start = item.start,
                                                                         unit = item.unit,
                                                                         data = data,
                                                                         data_loss = item.data_loss))                
                else:
                    raise NotImplementedError('Working on series other than slots or points not yet implemented')
 
        if not inplace:
            return normalized_timeseries




class Rescale(SeriesOperation):
    """Rescaling operation (callable object)"""
    
    def __call__(self, timeseries, value, inplace=False):
        
        if not inplace:
            rescaled_timeseries = timeseries.__class__()

        data_keys = timeseries.data_keys()

        for item in timeseries:
 
            if not inplace:
                rescaled_data = {}
            
            # Rescale data
            for key in data_keys:
                if isinstance(value, dict):
                    if key in value:
                        rescaled_data[key] = item.data[key] * value[key]
                    else:
                        rescaled_data[key] = item.data[key]
                else:
                    rescaled_data[key] = item.data[key] * value
            
            # Set data or create the item
            if inplace: 
                item.data[key] = rescaled_data
                            
            else:

                if isinstance(timeseries[0], Point):
                    rescaled_timeseries.append(timeseries[0].__class__(t = item.t,
                                                                         tz = item.tz,
                                                                         data = rescaled_data,
                                                                         data_loss = item.data_loss))         
                elif isinstance(timeseries[0], Slot):
                    rescaled_timeseries.append(timeseries[0].__class__(start = item.start,
                                                                         unit = item.unit,
                                                                         data = rescaled_data,
                                                                         data_loss = item.data_loss))                
                else:
                    raise NotImplementedError('Working on series other than slots or points not yet implemented')
 
        if not inplace:
            return rescaled_timeseries


class Offset(SeriesOperation):
    """Offsetting operation (callable object)"""
    
    def __call__(self, timeseries, value, inplace=False):
        
        if not inplace:
            rescaled_timeseries = timeseries.__class__()

        data_keys = timeseries.data_keys()

        for item in timeseries:
 
            if not inplace:
                offsetted_data = {}
            
            # Offset data
            for key in data_keys:
                if isinstance(value, dict):
                    if key in value:
                        offsetted_data[key] = item.data[key] + value[key]
                    else:
                        offsetted_data[key] = item.data[key]
                else:
                    offsetted_data[key] = item.data[key] + value
            
            # Set data or create the item
            if inplace: 
                item.data[key] = offsetted_data
                            
            else:

                if isinstance(timeseries[0], Point):
                    rescaled_timeseries.append(timeseries[0].__class__(t = item.t,
                                                                         tz = item.tz,
                                                                         data = offsetted_data,
                                                                         data_loss = item.data_loss))         
                elif isinstance(timeseries[0], Slot):
                    rescaled_timeseries.append(timeseries[0].__class__(start = item.start,
                                                                         unit = item.unit,
                                                                         data = offsetted_data,
                                                                         data_loss = item.data_loss))                
                else:
                    raise NotImplementedError('Working on series other than slots or points not yet implemented')
 
        if not inplace:
            return rescaled_timeseries




class MAvg(SeriesOperation):
    """Moving average operation (callable object)."""

    def __call__(self, timeseries, window, inplace=False):

        if not window or window <1:
            raise ValueError('A integer window >0 is required (got window="{}"'.frmat(window))

        if inplace:
            raise Exception('In-place operation not supported yet (padding support required)')
        
        postfix='mavg_{}'.format(window)
        
        mavg_timeseries = timeseries.__class__()

        data_keys = timeseries.data_keys()
        for i, item in enumerate(timeseries):

            if i < window-1:
                continue

            data = {}

            for key in data_keys:
 
                # Compute the moving average
                value_sum = 0
                for j in range(i-window+1,i+1):
                    value_sum += timeseries[j].data[key]
                data['{}_{}'.format(key, postfix)] = value_sum/window

            # Create the item
            if isinstance(timeseries[0], Point):
                mavg_timeseries.append(timeseries[0].__class__(t = item.t,
                                                               tz = item.tz,
                                                               data = data))         
            elif isinstance(timeseries[0], Slot):
                mavg_timeseries.append(timeseries[0].__class__(start = item.start,
                                                               unit = item.unit,
                                                               data = data))                
            else:
                raise NotImplementedError('Working on series other than slots or points not yet implemented')

        return mavg_timeseries


class Select(SeriesOperation):
    """Select operation (callable object). Selects items of the series given SQL-like queries."""
    
    def __call__(self, series, query, *args, **kwargs):

        if (' and ' or ' AND ' or ' or ' or ' OR ') in query:
            raise NotImplementedError('Multiple conditions not yet supported')
        
        if '=' not in query:
            raise NotImplementedError('Clauses other than the equality are not yet supported')
        
        # Get key
        key = query.split('=')[0].strip()
        if key.startswith('"'):
            key=key[1:]
        if key.endswith('"'):
            key=key[:-1]
        
        # Get value
        value = float(query.split('=')[1])

        # Perform the select
        selected = None
        for item in series:
            if item.data[key] == value:
                try:
                    selected.append(item)
                except:
                    selected = [item]
        return selected 


class Filter(SeriesOperation):
    """Filter a series (callable object)."""

    def __call__(self, series, data_key=None, from_t=None, to_t=None, from_dt=None, to_dt=None):
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
        
        # Instantiate the filtered series
        filtered_series = series.__class__()
        
        # Filter based on time if from/to are set
        for item in series:
            if from_t is not None and to_t is not None:
                if item.t >= from_t and item.t <to_t:
                    filtered_series.append(item)
            else:
                if from_t is not None:
                    if item.t >= from_t:
                        filtered_series.append(item) 
                if to_t is not None:
                    if item.t < to_t:
                        filtered_series.append(item)
                if from_t is None and to_t is None:
                    # Append everything 
                    filtered_series.append(item)
        
        # Filter based on data key if set
        if data_key:
            for i, item in enumerate(filtered_series):
                filtered_series[i] = copy(item)
                filtered_series[i]._data = {data_key: item.data[data_key]}
        
        # Re-set reference data as well
        try:
            filtered_series.item_data_reference
        except AttributeError:
            pass
        else:
            filtered_series.item_data_reference = filtered_series[0].data

        return filtered_series 


class Merge(SeriesOperation):
    """Merge two or more series (callable object)."""
    
    def __call__(self, *timeseriess):
        
        # Support vars
        resolution = None
        timeseriess_starting_points_t = []
        timeseriess_ending_points_t   = []
        
        # Checks
        for i, timeseries in enumerate(timeseriess):
            #if not isinstance(arg, DataTimeSlotSeries):
            #    raise TypeError('Argument #{} is not of type DataTimeSlotSeries, got "{}"'.format(i, timeseries.__class__.__name__))
            if resolution is None:
                resolution = timeseries.resolution
            else:
                if timeseries.resolution != resolution:
                    abort = True
                    try:
                        # Handle floating point precision issues 
                        if is_close(timeseries.resolution, resolution):
                            abort = False
                    except (ValueError,TypeError):
                        pass
                    if abort:
                        raise ValueError('DataTimeSlotSeries have different units, cannot merge')
        
            # Find min and max epoch for each series (aka start-end points)
            timeseriess_starting_points_t.append(timeseries[0].t)
            timeseriess_ending_points_t.append(timeseries[-1].t)
        
        # Find max min and min max to set the subset where timeseries are all deifned. e.g.:
        # ======
        #    ======
        # we are looking for the second segment (max) starting point
        
        max_starting_point_t = None
        for starting_point_t in timeseriess_starting_points_t:
            if max_starting_point_t is None:
                max_starting_point_t = starting_point_t
            else:
                if starting_point_t > max_starting_point_t:
                    max_starting_point_t = starting_point_t 

        min_ending_point_t = None
        for ending_point_t in timeseriess_ending_points_t:
            if min_ending_point_t is None:
                min_ending_point_t = ending_point_t
            else:
                if ending_point_t < min_ending_point_t:
                    min_ending_point_t = ending_point_t

        # Find the time series with the max starting point
        for timeseries in timeseriess:
            if timeseries[0].t == max_starting_point_t:
                reference_timeseries = timeseries

        # Create the (empty) result time series 
        result_timeseries = timeseriess[0].__class__(unit=resolution)
        
        # Scroll the time series to get the offsets
        offsets = {}
        for timeseries in timeseriess:
            for i in range(len(timeseries)):
                if timeseries[i].t == max_starting_point_t:
                    offsets[timeseries] = i
                    break

        # Loop over the reference series to create the datapoints for the merge
        for i in range(len(reference_timeseries)):
            
            # Check if we gone beyond the minimum ending point
            if reference_timeseries[i].t > min_ending_point_t:
                break  
            
            # Support vars
            data = None
            data_loss = 0
            valid_data_losses = 0
            
            tz = timeseriess[0][0].tz
            if isinstance(reference_timeseries[0], Slot):
                unit = reference_timeseries[0].unit
            else:
                unit = None  
            
            # For each time series, get data in i-th position (given the offset) and merge it
            for timeseries in timeseriess:

                # Data
                if data is None:
                    data = deepcopy(timeseries[i+offsets[timeseries]].data)
                else:
                    data.update(timeseries[i+offsets[timeseries]].data)
                 
                # Data loss
                if timeseries[i+offsets[timeseries]].data_loss is not None:
                    valid_data_losses += 1
                    data_loss += timeseries[i+offsets[timeseries]].data_loss
             
            # Finalize data loss if there were valid ones
            if valid_data_losses:
                data_loss = data_loss / valid_data_losses
            else:
                data_loss = None
 
            if isinstance(reference_timeseries[0], Point):
                result_timeseries.append(timeseriess[0][0].__class__(t = reference_timeseries[i].t,
                                                                     tz = tz,
                                                                     data  = data,
                                                                     data_loss = data_loss))         
            elif isinstance(reference_timeseries[0], Slot):
                result_timeseries.append(timeseriess[0][0].__class__(start = reference_timeseries[i].start,
                                                                     unit  = unit,
                                                                     data  = data,
                                                                     data_loss = data_loss))                
            else:
                raise NotImplementedError('Working on series other than slots or points not yet implemented')

        return result_timeseries



#========================
# Instantiate operations
#========================

# Scalar operations
min = Min()
max = Max()
avg = Avg()
wavg = WAvg()
sum = Sum()

# Series operations
derivative = Derivative()
integral = Integral()

diff = Diff()
csum = CSum()
mavg = MAvg()

normalize = Normalize()
offset = Offset()
rescale = Rescale()

merge = Merge()
filter = Filter()
select = Select()

