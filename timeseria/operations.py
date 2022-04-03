# -*- coding: utf-8 -*-
"""Operations on the series, returning both scalar and other series."""

from copy import copy, deepcopy
from .time import s_from_dt
from .datastructures import Series, Slot, Point, DataTimePointSeries, DataTimeSlotSeries
from .utilities import is_close
from .units import TimeUnit, Unit
from .exceptions import ConsistencyException 

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
    supports_weights = False


class SeriesOperation(Operation):
    """An operation operating on a series and returning a series (callable object)."""


#=======================
#  Scalar Operations
#=======================

class Max(ScalarOperation):
    """Maximum operation (callable object). Comes also pre-instantiated as the ``max()``
    function in the same module (accessible as ``timeseria.operations.max``)."""
    
    def __init__(self):
        self.built_in_max = max
    
    def __call__(self, arg, data_label=None, **kwargs):
        if not isinstance(arg, Series):
            return self.built_in_max(arg)
        else:
            series = arg
            mins = {data_label: None for data_label in series.data_labels()}
            for item in series:
                for _data_label in mins:
                    if mins[_data_label] is None:
                        mins[_data_label] = item.data[_data_label]
                    else:
                        if item.data[_data_label] > mins[_data_label]:
                            mins[_data_label] = item.data[_data_label]
            
            if data_label is not None:
                return mins[data_label]
            if len(mins) == 1:
                return mins[series.data_labels()[0]]
            else:
                return mins


class Min(ScalarOperation):
    """Minimum operation (callable object). Comes also pre-instantiated as the ``min()``
    function in the same module (accessible as ``timeseria.operations.min``)."""
    
    def __init__(self):
        self.built_in_min = min
    
    def __call__(self, arg, data_label=None, **kwargs):
        if not isinstance(arg, Series):
            return self.built_in_min(arg)
        else:
            series = arg
            mins = {data_label: None for data_label in series.data_labels()}
            for item in series:
                for _data_label in mins:
                    if mins[_data_label] is None:
                        mins[_data_label] = item.data[_data_label]
                    else:
                        if item.data[_data_label] < mins[_data_label]:
                            mins[_data_label] = item.data[_data_label]
            
            if data_label is not None:
                return mins[data_label]
            if len(mins) == 1:
                return mins[series.data_labels()[0]]
            else:
                return mins


class Avg(ScalarOperation):
    """Weighted average operation (callable object)."""
    
    supports_weights = True

    def __call__(self, arg, data_label=None, **kwargs):

        # Log & checks
        logger.debug('Called avg operation')
        if not isinstance(arg, Series):
            raise NotImplementedError('Avg implemented only on series objects')
        try:
            for item in arg:
                item.weight
                weighted=True
        except AttributeError:
            weighted=False
            #raise AttributeError('Trying to apply a weightd average on non-weigthed point')     

        # Prepare
        series = arg
        sums = {data_label: None for data_label in series.data_labels()}
        
        # Compute
        if weighted:
            total_weights = 0
            for item in series:
        
                logger.debug('Point @ %s, weight: %s, data: %s', item.dt, item.weight, item.data)
        
                for _data_label in sums:
                    if sums[_data_label] is None:
                        sums[_data_label] = item.data[_data_label]*item.weight
                    else:
                        sums[_data_label] += item.data[_data_label]*item.weight
                    #logger.debug('Sums: %s',sums[_data_label])
                total_weights += item.weight
            
            # The sum are already the average as weugthed
            logger.debug('Total weights: %s', total_weights)
            if total_weights == 1:
                avgs = sums
            else:
                avgs = {}
                for _data_label in sums:
                    avgs[_data_label] = sums[_data_label] / total_weights
        else:
            for item in series:
                for _data_label in sums:
                    if sums[_data_label] is None:
                        sums[_data_label] = item.data[_data_label]
                    else:
                        sums[_data_label] += item.data[_data_label]            
            avgs = {}
            for _data_label in sums:
                avgs[_data_label] = sums[_data_label] / len(series)
            
        # FInalize and return
        if data_label is not None:
            return avgs[data_label]
        if len(avgs) == 1:
            return avgs[series.data_labels()[0]]
        else:
            return avgs


class Sum(ScalarOperation):
    """Sum operation (callable object)."""

    def __init__(self):
        self.built_in_sum = sum
    
    def __call__(self, arg, data_label=None, **kwargs):
        if not isinstance(arg, Series):
            return self.built_in_sum(arg)
        else:
            series = arg
            sums = {data_label: 0 for data_label in series.data_labels()}
            for item in series:
                for _data_label in sums:
                    sums[_data_label] += item.data[_data_label]
            if data_label is not None:
                return sums[data_label]
            if len(sums) == 1:
                return sums[series.data_labels()[0]]
            else:
                return sums


#=======================
#  Series operations
#=======================

class Derivative(SeriesOperation):
    """Derivative operation (callable object). Must operate on a fixed resolution time series"""
    
    def __call__(self, series, inplace=False, normalize=True, diffs=False):
        
        if diffs and inplace:
            raise NotImplementedError('Computing diffs in-place is not supported as it would change the series length')
        
        if normalize:
            if series.resolution.is_variable():
                variable_resolution = True
                sampling_interval = series.autodetected_sampling_interval
            else:
                variable_resolution = False
                if isinstance(series.resolution, TimeUnit):
                    sampling_interval = series.resolution.as_seconds(series[0].dt)               
                elif isinstance(series.resolution, Unit):
                    sampling_interval = series.resolution.value
                else:
                    sampling_interval = series.resolution           
            postfix='derivative'
        else:
            postfix='diff'

        if not inplace:
            der_series = series.__class__()

        data_labels = series.data_labels()
        for i, item in enumerate(series):

            # Skip the first element if computing diffs
            if diffs and i == 0:
                continue

            if not inplace:
                data = {}
            
            for key in data_labels:
                diff_left = None
                diff_right = None
                diff = None
                if diffs:
                    # Just compute the diffs
                    if i == 0:
                        raise ConsistencyException('We should never get here')
                    else:
                        diff = series[i].data[key] - series[i-1].data[key]
                else:
                    # Compute the derivative
                    if i == 0:
                        # Right increment for the first item
                        diff_right = series[i+1].data[key] - series[i].data[key]
                    elif i == len(series)-1:
                        # Left increment for the last item. Divide by two if not normlaizing by time resolution
                        diff_left = series[i].data[key] - series[i-1].data[key]
                    else:
                        # Both left and right increment for the items in the middle
                        diff_left = series[i].data[key] - series[i-1].data[key]
                        diff_right = series[i+1].data[key] - series[i].data[key]

                # Combine and normalize (if required) the increments to get the actual derivative
                if normalize:
                    if diff is not None:
                        diff = diff / sampling_interval 
                    elif diff_right is None:
                        diff = diff_left / sampling_interval
                    elif diff_left is None:
                        diff = diff_right / sampling_interval
                    else:
                        if variable_resolution:
                            diff = ((diff_left / (series[i].t - series[i-1].t)) + (diff_right / (series[i+1].t - series[i].t))) /2
                        else:
                            diff = ((diff_left + diff_right)/2)/sampling_interval
                else:
                    if diff is not None:
                        pass
                    elif diff_right is None:
                        diff = diff_left
                    elif diff_left is None:
                        diff = diff_right
                    else:
                        diff = (diff_left+diff_right)/2
                
                # Add data
                if not inplace:
                    data['{}_{}'.format(key, postfix)] = diff
                else:
                    item.data['{}_{}'.format(key, postfix)] = diff
            
            # Create the item
            if not inplace:

                if isinstance(series[0], Point):
                    der_series.append(series[0].__class__(t = item.t,
                                                          tz = item.tz,
                                                          data = data,
                                                          data_loss = item.data_loss))         
                elif isinstance(series[0], Slot):
                    der_series.append(series[0].__class__(start = item.start,
                                                          unit = item.unit,
                                                          data = data,
                                                          data_loss = item.data_loss))                
                else:
                    raise NotImplementedError('Working on series other than slots or points not yet implemented')

        if not inplace:
            return der_series


class Integral(SeriesOperation):
    """Integral operation (callable object). Must operate on a fixed resolution time series."""
    
    def __call__(self, series, inplace=False, normalize=True, c=0, offset=0):
        
        if normalize:
            if series.resolution.is_variable():
                variable_resolution = True
                sampling_interval = series.autodetected_sampling_interval
            else:
                variable_resolution = False
                if isinstance(series.resolution, TimeUnit):
                    sampling_interval = series.resolution.as_seconds(series[0].dt)               
                elif isinstance(series.resolution, Unit):
                    sampling_interval = series.resolution.value
                else:
                    sampling_interval = series.resolution     
            postfix='integral'
        else:
            postfix='csum'

        if not inplace:
            int_series = series.__class__()

        last_values={}

        data_labels = series.data_labels()
        for i, item in enumerate(series):

            # Do we have to create the previous point based on an offset?
            if i==0 and offset:
                if inplace:
                    raise ValueError('Cannot use prev_data in in.place mode, would require to add a ppint to the series')
                
                if isinstance(offset,dict):
                    data = {'{}_{}'.format(data_label,postfix): offset[data_label] for data_label in series.data_labels()} 
                else:
                    data = {'{}_{}'.format(data_label,postfix): offset for data_label in series.data_labels()} 
                
                if isinstance(item, Point):
                    int_series.append(item.__class__(t = item.t - series.resolution,
                                                     tz = item.tz,
                                                     data = data,
                                                     data_loss = 0))         
                elif isinstance(item, Slot):
                    int_series.append(item.__class__(start = item.start - item.unit,
                                                     unit = item.unit,
                                                     data = data,
                                                     data_loss = 0))
                # Now set the c accordingly:
                c = offset  


            if not inplace:
                data = {}
            
            for key in data_labels:
                
                # Get the value
                if not normalize:
                    value = series[i].data[key]
                else:
                    # Compute the integral
                    if i == 0:
                        value=0
                        prev_right_component = series[i].data[key]
                    else:
                        left_component = (series[i-1].data[key]*2) - prev_right_component
                        # The "value" below is actually an increment, see below at  value = last_values[key] + value 
                        value = left_component 
                        prev_right_component = left_component

                # Normalize the cumulative to get the actual integral
                if normalize:
                    if variable_resolution:
                        if i == 0:
                            value = value * sampling_interval
                        else:
                            value = value * (series[i].t - series[i-1].t)
                    else:
                        value = value * sampling_interval

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

                if isinstance(series[0], Point):
                    int_series.append(series[0].__class__(t = item.t,
                                                          tz = item.tz,
                                                          data = data,
                                                          data_loss = item.data_loss))         
                elif isinstance(series[0], Slot):
                    int_series.append(series[0].__class__(start = item.start,
                                                          unit = item.unit,
                                                          data = data,
                                                          data_loss = item.data_loss))                
                else:
                    raise NotImplementedError('Working on series other than slots or points not yet implemented')

        if not inplace:
            return int_series


class Diff(Derivative):
    """Incremental differences operation (callable object). Reduces the series leght by one (the firts element), same as Pandas."""
    def __call__(self, series, inplace=False):
        return super(Diff, self).__call__(series, inplace=inplace, normalize=False, diffs=True)


class CSum(Integral):
    """Cumulative sum operation (callable object)."""
    def __call__(self, series, inplace=False, offset=None):
        return super(CSum, self).__call__(series, inplace=inplace, normalize=False, offset=offset)


class Normalize(SeriesOperation):
    """Normalization operation (callable object)"""
    
    def __call__(self, series, range=[0,1], inplace=False):
        
        if range == [0,1]:
            custom_range = None
        else:
            custom_range = range
        
        if not inplace:
            normalized_series = series.__class__()

        data_labels = series.data_labels()

        # Get min and max for the data keys
        for i, item in enumerate(series):
            
            if i == 0:
                mins = {key: item.data[key] for key in data_labels}
                maxs = {key: item.data[key] for key in data_labels}        
            else:
                for key in data_labels:
                    if item.data[key] < mins[key]:
                        mins[key] = item.data[key]
                    if item.data[key] > maxs[key]:
                        maxs[key] = item.data[key]                
        
        for i, item in enumerate(series):
 
            if not inplace:
                data = {}
             
            for key in data_labels:

                # Normalize data
                if not inplace:
                    if custom_range:
                        data[key] = (((item.data[key] - mins[key])  / (maxs[key]-mins[key])) * (custom_range[1]-custom_range[0])) + custom_range[0]            
                    else:    
                        data[key] = (item.data[key] - mins[key])  / (maxs[key]-mins[key])
                else:
                    if custom_range:
                        item.data[key] = (((item.data[key] - mins[key])  / (maxs[key]-mins[key])) * (custom_range[1]-custom_range[0])) + custom_range[0]               
                    else:
                        item.data[key] = (item.data[key] - mins[key])  / (maxs[key]-mins[key])
             
            # Create the item
            if not inplace:
 
                if isinstance(series[0], Point):
                    normalized_series.append(series[0].__class__(t = item.t,
                                                                         tz = item.tz,
                                                                         data = data,
                                                                         data_loss = item.data_loss))         
                elif isinstance(series[0], Slot):
                    normalized_series.append(series[0].__class__(start = item.start,
                                                                         unit = item.unit,
                                                                         data = data,
                                                                         data_loss = item.data_loss))                
                else:
                    raise NotImplementedError('Working on series other than slots or points not yet implemented')
 
        if not inplace:
            return normalized_series




class Rescale(SeriesOperation):
    """Rescaling operation (callable object)"""
    
    def __call__(self, series, value, inplace=False):
        
        if not inplace:
            rescaled_series = series.__class__()

        data_labels = series.data_labels()

        for item in series:
 
            if not inplace:
                rescaled_data = {}
            
            # Rescale data
            for key in data_labels:
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

                if isinstance(series[0], Point):
                    rescaled_series.append(series[0].__class__(t = item.t,
                                                                         tz = item.tz,
                                                                         data = rescaled_data,
                                                                         data_loss = item.data_loss))         
                elif isinstance(series[0], Slot):
                    rescaled_series.append(series[0].__class__(start = item.start,
                                                                         unit = item.unit,
                                                                         data = rescaled_data,
                                                                         data_loss = item.data_loss))                
                else:
                    raise NotImplementedError('Working on series other than slots or points not yet implemented')
 
        if not inplace:
            return rescaled_series


class Offset(SeriesOperation):
    """Offsetting operation (callable object)"""
    
    def __call__(self, series, value, inplace=False):
        
        if not inplace:
            rescaled_series = series.__class__()

        data_labels = series.data_labels()

        for item in series:
 
            if not inplace:
                offsetted_data = {}
            
            # Offset data
            for key in data_labels:
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

                if isinstance(series[0], Point):
                    rescaled_series.append(series[0].__class__(t = item.t,
                                                                         tz = item.tz,
                                                                         data = offsetted_data,
                                                                         data_loss = item.data_loss))         
                elif isinstance(series[0], Slot):
                    rescaled_series.append(series[0].__class__(start = item.start,
                                                                         unit = item.unit,
                                                                         data = offsetted_data,
                                                                         data_loss = item.data_loss))                
                else:
                    raise NotImplementedError('Working on series other than slots or points not yet implemented')
 
        if not inplace:
            return rescaled_series




class MAvg(SeriesOperation):
    """Moving average operation (callable object). Reduces the series lenght by n (the windowd length)."""

    def __call__(self, series, window, inplace=False):

        if not window or window <1:
            raise ValueError('A integer window >0 is required (got window="{}"'.frmat(window))

        if inplace:
            raise NotImplementedError('Computing a moving average in-place is not supported as it would change the series length')

        
        postfix='mavg_{}'.format(window)
        
        mavg_series = series.__class__()

        data_labels = series.data_labels()
        for i, item in enumerate(series):

            if i < window-1:
                continue

            data = {}

            for key in data_labels:
 
                # Compute the moving average
                value_sum = 0
                for j in range(i-window+1,i+1):
                    value_sum += series[j].data[key]
                data['{}_{}'.format(key, postfix)] = value_sum/window

            # Create the item
            if isinstance(series[0], Point):
                mavg_series.append(series[0].__class__(t = item.t,
                                                               tz = item.tz,
                                                               data = data))         
            elif isinstance(series[0], Slot):
                mavg_series.append(series[0].__class__(start = item.start,
                                                               unit = item.unit,
                                                               data = data))                
            else:
                raise NotImplementedError('Working on series other than slots or points not yet implemented')

        return mavg_series


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

    def __call__(self, series, data_label=None, from_t=None, to_t=None, from_dt=None, to_dt=None):
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
        if data_label:
            for i, item in enumerate(filtered_series):
                filtered_series[i] = copy(item)
                filtered_series[i]._data = {data_label: item.data[data_label]}
        
        # Re-set reference data as well
        try:
            filtered_series._item_data_reference
        except AttributeError:
            pass
        else:
            filtered_series._item_data_reference = filtered_series[0].data

        return filtered_series 


class Merge(SeriesOperation):
    """Merge two or more series (callable object)."""
    
    def __call__(self, *seriess):
        
        # Support vars
        resolution = None
        seriess_starting_points_t = []
        seriess_ending_points_t   = []
        
        # Checks
        for i, series in enumerate(seriess):
            #if not isinstance(arg, DataTimeSlotSeries):
            #    raise TypeError('Argument #{} is not of type DataTimeSlotSeries, got "{}"'.format(i, series.__class__.__name__))
            if resolution is None:
                resolution = series.resolution
            else:
                if series.resolution != resolution:
                    abort = True
                    try:
                        # Handle floating point precision issues 
                        if is_close(series.resolution, resolution):
                            abort = False
                    except (ValueError,TypeError):
                        pass
                    if abort:
                        raise ValueError('DataTimeSlotSeries have different units, cannot merge')
        
            # Find min and max epoch for each series (aka start-end points)
            seriess_starting_points_t.append(series[0].t)
            seriess_ending_points_t.append(series[-1].t)
        
        # Find max min and min max to set the subset where series are all deifned. e.g.:
        # ======
        #    ======
        # we are looking for the second segment (max) starting point
        
        max_starting_point_t = None
        for starting_point_t in seriess_starting_points_t:
            if max_starting_point_t is None:
                max_starting_point_t = starting_point_t
            else:
                if starting_point_t > max_starting_point_t:
                    max_starting_point_t = starting_point_t 

        min_ending_point_t = None
        for ending_point_t in seriess_ending_points_t:
            if min_ending_point_t is None:
                min_ending_point_t = ending_point_t
            else:
                if ending_point_t < min_ending_point_t:
                    min_ending_point_t = ending_point_t

        # Find the time series with the max starting point
        for series in seriess:
            if series[0].t == max_starting_point_t:
                reference_series = series

        # Create the (empty) result time series 
        result_series = seriess[0].__class__(unit=resolution)
        
        # Scroll the time series to get the offsets
        offsets = {}
        for series in seriess:
            for i in range(len(series)):
                if series[i].t == max_starting_point_t:
                    offsets[series] = i
                    break

        # Loop over the reference series to create the datapoints for the merge
        for i in range(len(reference_series)):
            
            # Check if we gone beyond the minimum ending point
            if reference_series[i].t > min_ending_point_t:
                break  
            
            # Support vars
            data = None
            data_loss = 0
            valid_data_losses = 0
            
            tz = seriess[0][0].tz
            if isinstance(reference_series[0], Slot):
                unit = reference_series[0].unit
            else:
                unit = None  
            
            # For each time series, get data in i-th position (given the offset) and merge it
            for series in seriess:

                # Data
                if data is None:
                    data = deepcopy(series[i+offsets[series]].data)
                else:
                    data.update(series[i+offsets[series]].data)
                 
                # Data loss
                if series[i+offsets[series]].data_loss is not None:
                    valid_data_losses += 1
                    data_loss += series[i+offsets[series]].data_loss
             
            # Finalize data loss if there were valid ones
            if valid_data_losses:
                data_loss = data_loss / valid_data_losses
            else:
                data_loss = None
 
            if isinstance(reference_series[0], Point):
                result_series.append(seriess[0][0].__class__(t = reference_series[i].t,
                                                                     tz = tz,
                                                                     data  = data,
                                                                     data_loss = data_loss))         
            elif isinstance(reference_series[0], Slot):
                result_series.append(seriess[0][0].__class__(start = reference_series[i].start,
                                                                     unit  = unit,
                                                                     data  = data,
                                                                     data_loss = data_loss))                
            else:
                raise NotImplementedError('Working on series other than slots or points not yet implemented')

        return result_series



#========================
# Instantiate operations
#========================

# Scalar operations
min = Min()
max = Max()
avg = Avg()
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

