from .time import TimeUnit, dt_from_s, s_from_dt
from .datastructures import DataTimeSlot, DataTimeSlotSeries, TimePoint, DataTimePointSeries
from .utilities import compute_coverage, is_almost_equal, is_close

# Setup logging
import logging
logger = logging.getLogger(__name__)

HARD_DEBUG = False


#=======================
#  Base Operator
#=======================

class Operator(object):
    
    @classmethod
    def __str__(cls):
        return '{} operator'.format(cls.__name__)



#=======================
#  Diff Operator
#=======================

class Diff(Operator):

    def __call__(self, data_time_slot_series, inplace=False):
        
        if not inplace:
            diff_data_time_slot_series = DataTimeSlotSeries()

        data_keys = data_time_slot_series.data_keys()
        for i, data_time_slot in enumerate(data_time_slot_series):

            if not inplace:
                data = {}
            
            for key in data_keys:
                
                # Compute the diffs
                if i == 0:
                    # Right diffs for the first item
                    diff = data_time_slot_series[i+1].data[key] - data_time_slot_series[i].data[key]
                    diff = diff/2       
                elif i == len(data_time_slot_series)-1:
                    # Left diffs for the last item
                    diff = data_time_slot_series[i].data[key] - data_time_slot_series[i-1].data[key]
                    diff = diff/2
                else:
                    # Both left and right diffs for the items in the middle
                    diff =  ((data_time_slot_series[i+1].data[key] - data_time_slot_series[i].data[key]) + (data_time_slot_series[i].data[key] - data_time_slot_series[i-1].data[key])) /2
                
                # Add data
                if not inplace:
                    data['{}_diff'.format(key)] = diff
                else:
                    data_time_slot.data['{}_diff'.format(key)] = diff
            
            # Create the slot
            if not inplace:       
                diff_data_time_slot_series.append(DataTimeSlot(start = data_time_slot.start,
                                                           end   = data_time_slot.end,
                                                           data  = data))

        if not inplace:
            return diff_data_time_slot_series



#=======================
#  Derivative Operator
#=======================

class Derivative(Operator):
    
    def __call__(self, data_time_slot_series, inplace=False, incremental=False):
        
        if not inplace:
            derivative_data_time_slot_series = DataTimeSlotSeries()

        data_keys = data_time_slot_series.data_keys()
        for i, data_time_slot in enumerate(data_time_slot_series):

            if not inplace:
                data = {}
            
            for key in data_keys:
                
                # Compute the derivative
                if i == 0:
                    # Right derivative for the first item
                    der = data_time_slot_series[i+1].data[key] - data_time_slot_series[i].data[key]    
                elif i == len(data_time_slot_series)-1:
                    # Left derivative for the last item
                    der = data_time_slot_series[i].data[key] - data_time_slot_series[i-1].data[key]
                else:
                    # Both left and right derivative for the items in the middle
                    der =  ((data_time_slot_series[i+1].data[key] - data_time_slot_series[i].data[key]) + (data_time_slot_series[i].data[key] - data_time_slot_series[i-1].data[key])) /2
                    
                der = der / (data_time_slot_series.slot_unit)

                # Add data
                if not inplace:
                    data['{}_derivative'.format(key)] = der
                else:
                    data_time_slot.data['{}_derivate'.format(key)] = der
            
            # Create the slot
            if not inplace:       
                derivative_data_time_slot_series.append(DataTimeSlot(start = data_time_slot.start,
                                                                 end   = data_time_slot.end,
                                                                 data  = data))

        if not inplace:
            return derivative_data_time_slot_series



#=======================
#  Merge Operator 
#=======================

class Merge(Operator):
    

    def __call__(self, *args):
        
        slot_unit = None
        for i, arg in enumerate(args):
            if not isinstance(arg, DataTimeSlotSeries):
                raise TypeError('Argument #{} is not of type DataTimeSlotSeries, got "{}"'.format(i, arg.__class__.__name__))
            if slot_unit is None:
                slot_unit = arg.slot_unit
            else:
                if arg.slot_unit != slot_unit:
                    abort = True
                    try:
                        # Handle floating point precision issues 
                        if is_close(arg.slot_unit, slot_unit):
                            abort = False
                    except (ValueError,TypeError):
                        pass
                    if abort:
                        raise ValueError('DataTimeSlotSeries have different units, cannot merge')
        
        length = len(args[0])
        n_args = len(args)
        result_data_time_slot_series = DataTimeSlotSeries(unit=slot_unit)
        import copy
        
        for i in range(length):
            data = None
            coverage = None
            valid_coverages = 0
            for j in range(n_args):
                #logger.critical('i={}, j={}'.format(i, j))
                
                # Data
                if data is None:
                    data = copy.deepcopy(args[j][i].data)
                else:
                    data.update(args[j][i].data)
                
                # Coverage
                if coverage is None:
                    coverage = coverage
                else:
                    valid_coverages += 1
                    coverage += args[j][i].coverage
            
            # Finalize coverage if there were valid   
            if valid_coverages:
                coverage = coverage / valid_coverages
            else:
                coverage = None
                    
            data_time_slot = DataTimeSlot(start = copy.deepcopy(args[j][i].start),
                                        end   = copy.deepcopy(args[j][i].end),
                                        data  = data,
                                        coverage = coverage)
            
            result_data_time_slot_series.append(data_time_slot)

        return result_data_time_slot_series


class Min(Operator):
    
    def __init__(self):
        self.built_in_min = min
    
    def __call__(self, arg):
        return self.built_in_min(arg)



#=======================
# Instantiate operators
#=======================

min = Min()
diff = Diff()
merge = Merge()
derivative = Derivative()







