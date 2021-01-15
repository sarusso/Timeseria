from .time import dt_from_s, s_from_dt
from .datastructures import DataTimeSlot, DataTimeSlotSeries, TimePoint, DataTimePointSeries, Series, Slot, Point, PointSeries
from .utilities import compute_coverage, is_almost_equal, is_close
from .units import TimeUnit, Unit

# Setup logging
import logging
logger = logging.getLogger(__name__)

HARD_DEBUG = False


#=======================
#  Base Operation class
#=======================

class Operation(object):

    @classmethod
    def __str__(cls):
        return '{} operation'.format(cls.__name__)
    
    @classmethod
    def __repr__(cls):
        return cls.__str__()



#=======================
#  Scalar result
#=======================

class Max(Operation):
    
    def __init__(self):
        self.built_in_max = max
    
    def __call__(self, arg, key=None):
        if not isinstance(arg, Series):
            return self.built_in_max(arg)
        else:
            series = arg
            mins = {key: None for key in series.data_keys()}
            for item in series:
                for data_key in mins:
                    if mins[data_key] is None:
                        mins[data_key] = item.data[data_key]
                    else:
                        if item.data[data_key] > mins[data_key]:
                            mins[data_key] = item.data[data_key]
            
            if key is not None:
                return mins[key]
            if len(mins) == 1:
                return mins[series.data_keys()[0]]
            else:
                return mins


class Min(Operation):
    
    def __init__(self):
        self.built_in_min = min
    
    def __call__(self, arg, key=None):
        if not isinstance(arg, Series):
            return self.built_in_min(arg)
        else:
            series = arg
            mins = {key: None for key in series.data_keys()}
            for item in series:
                for data_key in mins:
                    if mins[data_key] is None:
                        mins[data_key] = item.data[data_key]
                    else:
                        if item.data[data_key] < mins[data_key]:
                            mins[data_key] = item.data[data_key]
            
            if key is not None:
                return mins[key]
            if len(mins) == 1:
                return mins[series.data_keys()[0]]
            else:
                return mins


class Avg(Operation):
    
    def __call__(self, arg, key=None):
        if not isinstance(arg, Series):
            raise NotImplementedError('Avg implemented only on series objects')
        else:
            series = arg
            sums = {key: None for key in series.data_keys()}
            for item in series:
                for data_key in sums:
                    if sums[data_key] is None:
                        sums[data_key] = item.data[data_key]
                    else:
                        sums[data_key] += item.data[data_key]
            avgs={}
            for data_key in sums:
                avgs[data_key] = sums[data_key] / len(series)
            if key is not None:
                return avgs[key]
            if len(avgs) == 1:
                return avgs[series.data_keys()[0]]
            else:
                return avgs

class Sum(Operation):
    
    def __init__(self):
        self.built_in_sum = sum
    
    def __call__(self, arg, key=None):
        if not isinstance(arg, Series):
            return self.built_in_sum(arg)
        else:
            series = arg
            sums = {key: 0 for key in series.data_keys()}
            for item in series:
                for data_key in sums:
                    sums[data_key] += item.data[data_key]
            if key is not None:
                return sums[key]
            if len(sums) == 1:
                return sums[series.data_keys()[0]]
            else:
                return sums


#=======================
#  Series result
#=======================

class Derivative(Operation):
    ''' Derivative operation. Must operate on a fixed resolution time series'''
    
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
                                                                  data = data))         
                elif isinstance(timeseries[0], Slot):
                    der_timeseries.append(timeseries[0].__class__(start = item.start,
                                                                  unit = item.unit,
                                                                  data = data))                
                else:
                    raise NotImplementedError('Working on series other than slots or points not yet implemented')

        if not inplace:
            return der_timeseries


class Integral(Operation):
    ''' Integral operation. Must operate on a fixed resolution time series'''
    
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
                        #prev_value = value
                        prev_right_component = timeseries[i].data[key]

                    else:

                        #left_component = timeseries[i].data[key]-prev_right_component
                        
                        left_component = (timeseries[i-1].data[key]*2) - prev_right_component #14-6
                        #right_component = (timeseries[i].data[key]*2) - left_component
                        
                        value =  left_component # This is actually a incremente, see below at  value = last_values[key] + value for fixing it
                         
                        prev_right_component = left_component
                        prev_value = value

                                  
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
                                                                  data = data))         
                elif isinstance(timeseries[0], Slot):
                    der_timeseries.append(timeseries[0].__class__(start = item.start,
                                                                  unit = item.unit,
                                                                  data = data))                
                else:
                    raise NotImplementedError('Working on series other than slots or points not yet implemented')

        if not inplace:
            return der_timeseries


class Diff(Derivative):
    def __call__(self, timeseries, inplace=False):
        return super(Diff, self).__call__(timeseries, inplace=inplace, normalize=False, diffs=True)


class CSum(Integral):
    def __call__(self, timeseries, inplace=False, offset=0):
        return super(CSum, self).__call__(timeseries, inplace=inplace, normalize=False, c=offset)

class MAvg(Operation):

    def __call__(self, timeseries, window, inplace=False):

        if not window or window <1:
            raise ValueError('A integer window >0 is required (got window="{}"'.frmat(window))

        if inplace:
            raise Exception('In-place operation not supported yet (padding support required)')
        
        postfix='mavg'
        
        mavg_timeseries = timeseries.__class__()

        data_keys = timeseries.data_keys()
        for i, item in enumerate(timeseries):

            if i < window-1:
                continue

            data = {}

            for key in data_keys:
 
                # Compute the movign averge
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


#=======================
#  Series manipulation
#=======================

class Select(Operation):
    '''Select items of the series given an SQL-like queries'''
    
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


class Slice(Operation):

    def __call__(self, series, from_t=None, to_t=None, from_dt=None, to_dt=None):
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
        
        slices_series = series.__class__()
        for item in series:
            if from_t is not None and to_t is not None:
                if item.t >= from_t and item.t <to_t:
                    slices_series.append(item)
            else:
                if from_t is not None:
                    if item.t >= from_t:
                        slices_series.append(item) 
                if to_t is not None:
                    if item.t < to_t:
                        slices_series.append(item)
        return slices_series 


class Merge(Operation):
    
    def __call__(self, *timeseriess):
        
        resolution = None
        for i, arg in enumerate(timeseriess):
            #if not isinstance(arg, DataTimeSlotSeries):
            #    raise TypeError('Argument #{} is not of type DataTimeSlotSeries, got "{}"'.format(i, arg.__class__.__name__))
            if resolution is None:
                resolution = arg.resolution
            else:
                if arg.resolution != resolution:
                    abort = True
                    try:
                        # Handle floating point precision issues 
                        if is_close(arg.resolution, resolution):
                            abort = False
                    except (ValueError,TypeError):
                        pass
                    if abort:
                        raise ValueError('DataTimeSlotSeries have different units, cannot merge')
        
        length = len(timeseriess[0])
        n_timeseriess = len(timeseriess)
        result_timeseries = timeseriess[0].__class__(unit=resolution)
        import copy
        
        for i in range(length):
            data = None
            coverage = None
            valid_coverages = 0
            for j in range(n_timeseriess):
                #logger.critical('i={}, j={}'.format(i, j))
                
                # Data
                if data is None:
                    data = copy.deepcopy(timeseriess[j][i].data)
                else:
                    data.update(timeseriess[j][i].data)
                
                # Coverage
                if coverage is None:
                    coverage = coverage
                else:
                    valid_coverages += 1
                    coverage += timeseriess[j][i].coverage
            
            # Finalize coverage if there were valid   
            if valid_coverages:
                coverage = coverage / valid_coverages
            else:
                coverage = None

            if isinstance(timeseriess[0][0], Point):
                result_timeseries.append(timeseriess[0][0].__class__(t = timeseriess[j][i].t,
                                                                     tz = timeseriess[j][i].tz,
                                                                     data  = data))         
            elif isinstance(timeseriess[0][0], Slot):
                result_timeseries.append(timeseriess[0][0].__class__(start = timeseriess[j][i].start,
                                                                     unit   = timeseriess[j][i].unit,
                                                                     data  = data))                
            else:
                raise NotImplementedError('Working on series other than slots or points not yet implemented')

        return result_timeseries






#========================
# Instantiate operations
#========================

# Scalar result
min = Min()
max = Max()
avg = Avg()
sum = Sum()

# Series result
derivative = Derivative()
integral = Integral()

diff = Diff()
csum = CSum()
mavg = MAvg()


# Series manipulation
merge = Merge()
slice = Slice()
select = Select()









