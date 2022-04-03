# -*- coding: utf-8 -*-
"""Series transformations as resampling and aggregation."""

from .time import dt_from_s, s_from_dt, as_timezone
from .datastructures import DataTimeSlot, DataTimeSlotSeries, TimePoint, DataTimePointSeries, DataTimePoint, SeriesSlice
from .utilities import compute_data_loss, compute_validity_regions
from .operations import avg
from .units import TimeUnit
from .exceptions import ConsistencyException
from . import operations as operations_module

# Setup logging
import logging
logger = logging.getLogger(__name__)


#==========================
#  Support functions
#==========================

def _compute_new(target, series, from_t, to_t, slot_first_point_i, slot_last_point_i, slot_prev_point_i, slot_next_point_i,
                 unit, point_validity, timezone, fill_with, force_data_loss, fill_gaps, series_data_indexes, series_resolution,
                 force_compute_data_loss, interpolation_method, operations=None):
    
    # Log. Note: if slot_first_point_i < slot_last_point_i, this means that the prev and next are outside the slot.
    # It is not a bug, it is how the system works. perhaps we could pass here the slot_prev_i and sòlot_next_i
    # to make it more clear to avoid some confusion.
    logger.debug('Called compute new')
    logger.debug('slot_first_point_i=%s, slot_last_point_i=%s, ', slot_first_point_i, slot_last_point_i)
    logger.debug('slot_prev_point_i=%s, slot_next_point_i=%s, ', slot_prev_point_i, slot_next_point_i)

    # Support vars
    interval_duration = to_t-from_t
    data = {}
    data_labels = series.data_labels()

    # The prev can be None as lefts are included (edge case)
    if slot_prev_point_i is None:
        slot_prev_point_i = slot_first_point_i

    # Create the slice of the series containing the slot datapoints plus the prev and next, 
    series_dense_slice_extended  = SeriesSlice(series, from_i=slot_prev_point_i, to_i=slot_next_point_i+1,  # Slicing exclude the right
                                               from_t=from_t, to_t=to_t, interpolation_method=interpolation_method, dense=True)

    # Compute the data loss for the new element. This is forced
    # by the resampler or slotter if first or last point     
    data_loss = compute_data_loss(series_dense_slice_extended,
                                  from_t = from_t,
                                  to_t = to_t,
                                  sampling_interval = point_validity,
                                  force = force_compute_data_loss)
    logger.debug('Computed data loss: "{}"'.format(data_loss))

    # For each point, attach the "weight"/"contirbution
    for point in series_dense_slice_extended:
        
        # Weight zero if completely outside the interval, which is required if in the operation
        # there is the need of knowing the next(or prev) value, even if it was far away
        if point.valid_to < from_t:
            point.weight = 0
            continue
        if point.valid_from >= to_t:
            point.weight = 0
            continue 
        
        # Set this point valid from/to accoring to the interval boundaries
        this_point_valid_from = point.valid_from if point.valid_from >= from_t else from_t
        this_point_valid_to = point.valid_to if point.valid_to < to_t else to_t
        
        # Set weigth
        point.weight = (this_point_valid_to-this_point_valid_from)/interval_duration

        # Log
        logger.debug('Series point %s data: %s, weight: %s', point.t, point.data, point.weight)

    # If creating slots, create also the slice of the series series containing only the slot datapoints    
    if target == 'slot':
        
        if slot_first_point_i is None and slot_last_point_i is None:
            # If we have no datapoints at all in the slot, we must create one to let operations not
            # supporting weights to properly work. If we have a full data loss, this can be just the
            # middle point from the series_dense_slice_extended. Otherwise, we must compute it from
            # scratch as we are upsamping with no data loss but still no points belonging to the slot.

            if data_loss==1:
                series_dense_slice = None
                for point in series_dense_slice_extended:
                    try:
                        if point.reconstructed:
                            if series_dense_slice:
                                raise ConsistencyException('Found more than one reconstructed point in a fully missing slot')
                            series_dense_slice = series.__class__(point) 
                    except AttributeError:
                        pass
                if not series_dense_slice:
                    raise ConsistencyException('Could not find any reconstructed point in a fully missing slot')
            else:
                # Just create a point based on the series_dense_slice_extended weights (TODO: are we including the reconstructed?):
                new_point_data = {data_label:0 for data_label in data_labels}
                for point in series_dense_slice_extended:
                    for data_label in data_labels:
                        point.data[data_label] += point.data[data_label] * point.weight
                new_point_t = (to_t - from_t) /2
                series_dense_slice = DataTimePointSeries(DataTimePoint(t=new_point_t, data=new_point_data, tz=series.tz))

        else:
            # Slice the original series to provide only the datapoints belonging to the slot 
            #logger.critical('Slicing dense series from {} to {}'.format(slot_first_point_i, slot_last_point_i+1))
            series_dense_slice = SeriesSlice(series, from_i=slot_first_point_i, to_i=slot_last_point_i+1, # Slicing exclude the right   
                                             from_t=from_t, to_t=to_t, interpolation_method=interpolation_method, dense=True) 



    # Compute point data
    if target=='point':

        # Compute the (weighted) average of all point contributions
        avgs = avg(series_dense_slice_extended)
        
        # ...and assign them to the data value
        if isinstance(avgs, dict):
            data = {key:avgs[key] for key in data_labels}
        else:
            data = {data_labels[0]: avgs}
             
    #  Compute slot data
    elif target=='slot':
        
        if data_loss == 1 and fill_with:
            for key in data_labels:
                for operation in operations:
                    data['{}_{}'.format(key, operation.__name__)] = fill_with

        else:
            # Handle operations                
            for operation in operations:
                try:
                    operation_supports_weights = operation.supports_weights
                except AttributeError:
                    # This is because user-defined operations can be even simple functions
                    # or based on custom classes without the supports_weights attribute
                    operation_supports_weights = False
                
                if operation_supports_weights:
                    operation_data = operation(series_dense_slice_extended)
                else:
                    operation_data = operation(series_dense_slice)
                    
                # ...and add to the data
                if isinstance(operation_data, dict):
                    for result_key in operation_data:
                        data['{}_{}'.format(result_key, operation.__name__)] = operation_data[result_key]
                else:
                    data['{}_{}'.format(data_labels[0], operation.__name__)] = operation_data
    
        if not data:
            raise Exception('No data computed at all?')
    
    else:
        raise ValueError('No idea how to compute a new "{}"'.format(target))


    # Do we have a force data_loss?
    #TODO: do not compute data_loss if fill_with not present and force_data_loss 
    if force_data_loss is not None:
        data_loss = force_data_loss
    
    # Create the new item
    if target=='point':
        new_element = DataTimePoint(t = (from_t+((interval_duration)/2)),
                                    tz = timezone,
                                    data  = data,
                                    data_loss = data_loss)
    elif target=='slot':
        # Create the DataTimeSlot
        new_element = DataTimeSlot(start = TimePoint(t=from_t, tz=timezone),
                                   end   = TimePoint(t=to_t, tz=timezone),
                                   unit  = unit,
                                   data  = data,
                                   data_loss = data_loss)
    else:
        raise ValueError('No idea how to create a new "{}"'.format(target))

    # Handle data_indexes TODO: check math and everything here
    for index in series_data_indexes:

        # Skip the data loss as it is recomputed with different logics
        if index == 'data_loss':
            continue
        
        if series_dense_slice_extended:
            index_total = 0
            index_total_weights = 0
            for point in series_dense_slice_extended:
                
                # Get index value
                try:
                    index_value = point.data_indexes[index]
                except:
                    pass
                else:
                    if index_value is not None:
                        index_total_weights += point.weight
                        index_total += index_value * point.weight
                        #logger.critical('%s@%s: %s * %s', index, point.dt, index_value, point.weight)
    
            # Compute the new index value (if there were data_indexes not None)
            if index_total_weights > 0:
                
                # Project (rescale/normalize), because we could have some points without the index at all
                new_element_index_value = index_total/index_total_weights
                #logger.debug('%s/%s', index_total, index_total_weights)

            else:
                new_element_index_value = None

        else:
            new_element_index_value = None
            
        # Set the index in the data indexes
        new_element.data_indexes[index] = new_element_index_value

    # Return
    return new_element


#==========================
#  Base Transformation
#==========================

class Transformation(object):
    """Base transformation class."""
    
    @classmethod
    def __str__(cls):
        return '{} transformation'.format(cls.__name__.replace('Operator',''))

    def process(self, *args, **kwargs):
        raise NotImplementedError('This transformation is not implemented.')


#==========================
#  Slotted transformation
#==========================
class SlottedTransformation(Transformation):

    def process(self, series, target, from_t=None, to_t=None, from_dt=None, to_dt=None, validity=None,
                include_extremes=False, fill_with=None, force_data_loss=None, fill_gaps=True, force=False):
        """Start the transformation process. If start and/or end are not set, they are set automatically
        based on first and last points of the series"""
        
        # Checks
        if include_extremes:
            raise NotImplementedError('Including the extremes is not yet implemented')        
        if not isinstance(series, DataTimePointSeries):
            raise TypeError('Can process only DataTimePointSeries, got "{}"'.format(series.__class__.__name__))
        if not series:
            raise ValueError('Cannot process empty series')
        if target not in ['points', 'slots']:
            raise ValueError('Don\'t know how to target "{}"'.format(target))           

        # Log
        logger.debug('Computing from/to with include_extremes= %s', include_extremes)

        # Set from and to. If creating points, we will move the start back of half unit
        # and the end forward of half unit as well, as the point will be in the center

        # Set "from". TODO: check given from_t/from_dt against shifted rounding if points?
        if from_t:
            from_dt = dt_from_s(from_t)
            if from_dt != self.time_unit.round_dt(from_dt):
                raise ValueError('The provided from_t is not consistent with the self.time_unit of "{}" (Got "{}")'.format(self.time_unit, from_t))        
        elif from_dt:
            from_t = s_from_dt(from_dt)
            if from_dt != self.time_unit.round_dt(from_dt):
                raise ValueError('The provided from_dt is not consistent with the self.time_unit of "{}" (Got "{}")'.format(self.time_unit, from_dt))   
        else:
            if target == 'points':
                from_t = series[0].t  - (self.time_unit.as_seconds() /2)
                from_dt = dt_from_s(from_t, tz=series.tz)
                from_dt = self.time_unit.round_dt(from_dt, how='floor' if include_extremes else 'ceil')
                from_t = s_from_dt(from_dt) + (self.time_unit.as_seconds() /2)
                from_dt = dt_from_s(from_t)
                
            elif target == 'slots':
                from_t = series[0].t
                from_dt = dt_from_s(from_t, tz=series.tz)
                if from_dt == self.time_unit.round_dt(from_dt):
                    # Only for the start, and only for the slots, if the first point is
                    # exactly equal to the first slot start, leave it as it is to include it.
                    pass
                else:
                    from_dt = self.time_unit.round_dt(from_dt, how='floor' if include_extremes else 'ceil')
                from_t = s_from_dt(from_dt)
                from_dt = dt_from_s(from_t) 
            else:
                raise ValueError('Don\'t know how to target "{}"'.format(target))           
            
        # Set "to". TODO: check given to_t/to_dt against shifted rounding if points?
        if to_t:
            to_dt = dt_from_s(to_t)
            if to_dt != self.time_unit.round_dt(to_dt):
                raise ValueError('The provided to_t is not consistent with the self.time_unit of "{}" (Got "{}")'.format(self.time_unit, to_t))        
        elif to_dt:
            to_t = s_from_dt(to_dt)
            if to_dt != self.time_unit.round_dt(to_dt):
                raise ValueError('The provided to_dt is not consistent with the self.time_unit of "{}" (Got "{}")'.format(self.time_unit, to_dt))   
        else:
            if target == 'points':
                to_t = series[-1].t  + (self.time_unit.as_seconds() /2)
                to_dt = dt_from_s(to_t, tz=series.tz)
                from_dt = self.time_unit.round_dt(from_dt, how='ceil' if include_extremes else 'floor')
                to_t = s_from_dt(to_dt) - (self.time_unit.as_seconds() /2)
                to_dt = dt_from_s(to_t)  
            elif target == 'slots':
                to_t = series[-1].t
                to_dt = dt_from_s(to_t, tz=series.tz)
                from_dt = self.time_unit.round_dt(from_dt, how='ceil' if include_extremes else 'floor')
                to_t = s_from_dt(to_dt)
                to_dt = dt_from_s(to_t)            
            else:
                raise ValueError('Don\'t know how to target "{}"'.format(target))           
        
        # Set/fix timezone
        from_dt = as_timezone(from_dt, series.tz)
        to_dt = as_timezone(to_dt, series.tz)
        
        # Log
        logger.debug('Computed from: %s', from_dt)
        logger.debug('Computed to: %s',to_dt)

        # Automatically detect validity if not set
        if validity is None:
            validity = series.autodetected_sampling_interval
            logger.info('Using auto-detected sampling interval: %ss', validity)

        # Check if upsamplimg (with some tolearance):
        if validity > (self.time_unit.as_seconds(series[0].dt) * 1.10):
            logger.warning('You are upsampling, which is not well tested yet. Expect potential issues.')

        # Compute validity regions
        validity_regions = compute_validity_regions(series, sampling_interval=validity)

        # Log
        logger.debug('Started resampling from "%s" (%s) to "%s" (%s)', from_dt, from_t, to_dt, to_t)

        # Set some support vars
        slot_start_t = None
        slot_start_dt = None
        slot_end_t = None
        slot_end_dt = None
        process_ended = False
        slot_prev_point_i = None 
        new_series = DataTimePointSeries() if target=='points' else DataTimeSlotSeries()

        # Set timezone
        timezone  = series.tz
        logger.debug('Using timezone "%s"', timezone)
       
        # Counters
        count = 0
        first = True

        # data_indexes & resolution shortcuts
        series_data_indexes = series._all_data_indexes()
        series_resolution = series.resolution
        
        # Set operations if slots
        if target == 'slots':
            operations  = self.operations
        else:
            operations = None

        # Now go trough all the data in the time series        
        for i, point in enumerate(series):

            logger.debug('Processing i=%s: %s', i, point)

            # Increase counter
            count += 1
            
            # Attach validity TODO: do we want a generic series method, or maybe to be always computed in the series?
            point.valid_from = validity_regions[point.t][0]
            point.valid_to = validity_regions[point.t][1]
            logger.debug('Attached validity to %s', point.t)
            
            # Pretend there was a slot before if we are at the beginning. TOOD: improve me.
            if slot_end_t is None:
                slot_end_t = from_t

            # First, check if we have some points to discard at the beginning       
            if point.t < from_t:
                # If we are here it means we are going data belonging to a previous slot
                logger.debug('Discarding as before start')
                # NOTE: here we set the prev index
                slot_prev_point_i = i
                continue

            # Similar concept for the end
            # TODO: what if we are in streaming mode? add if to_t is not None?
            if point.t >= to_t:
                if process_ended:
                    logger.debug('Discarding as after end')
                    continue

            # Is the current slot outdated? (are we processing a datapoint falling outside the current slot?)            
            if point.t >= slot_end_t:
                       
                # This approach is to detect if the current slot is "outdated" and spin a new one if so.
                # Works only for slots at the beginning and in the middle, but not for the last slot
                # or the missing slots at the end which need to be closed down here
                logger.debug('Detetcted outdated slot')
                
                # Keep spinning new slots until the current data point falls in one of them.
                # NOTE: Read the following "while" more as an "if" which can also lead to spin multiple
                # slot if there are empty slots between the one being closed and the point.dt.
                # TODO: leave or remove the above if for code readability?
                
                while slot_end_t <= point.t:
                    logger.debug('Checking for end %s with point %s', slot_end_t, point.t)
                    
                    if slot_start_t is None:
                        # If we are in the pre-first slot, just silently spin a new slot
                        pass
                
                    else:

                        # Set slot next point index (us, as we triggered a new slot) 
                        slot_next_point_i = i
                        
                        # The slot_prev_point_i is instead always defined in the process and correctly set.
                        slot_prev_point_i = slot_prev_point_i
        
                        logger.debug('Got slot_prev_point_i as %s',slot_prev_point_i) 
                        logger.debug('Got slot_next_point_i as %s',slot_next_point_i) 
                        
                        # Now set slot first and last point, but beware boundaries.
                        if slot_prev_point_i is not None and abs(slot_next_point_i - slot_prev_point_i) == 1:
                        
                            # No points for the slot at all
                            slot_first_point_i = None
                            slot_last_point_i = None
                        
                        else:
        
                            # Set first
                            if slot_prev_point_i is None:
                                # Edge case where there is no prev as the time series starts with a point
                                # placed exactly on the slot start, and since lefts are included we take it.
                                slot_first_point_i = 0
                            else:          
                                if series[slot_prev_point_i+1].t >= slot_start_t and series[slot_prev_point_i+1].t < slot_end_t:
                                    slot_first_point_i = slot_prev_point_i+1
                                else:
                                    slot_first_point_i = None
                            
                            # Set last
                            if series[slot_next_point_i-1].t >= slot_start_t and series[slot_next_point_i-1].t < slot_end_t:
                                slot_last_point_i = slot_next_point_i-1
                            else:
                                slot_last_point_i = None                    
                                
                                
                        logger.debug('Set slot_first_point_i to %s',slot_first_point_i) 
                        logger.debug('Set slot_last_point_i to %s',slot_last_point_i) 



                        # Log the new slot
                        # slot_first_point_i-1 is the "prev"                        
                        # slot_last_point_i+1 is the "next" (and the index where we are at the moment)
                        logger.debug('This slot is closed: start=%s (%s) and end=%s (%s). Now computing it..', slot_start_t, slot_start_dt, slot_end_t, slot_end_dt)
                        
                        # Compute the new item...
                        new_item = _compute_new(target = 'point' if target=='points' else 'slot',
                                                series = series,
                                                slot_first_point_i = slot_first_point_i,
                                                slot_last_point_i = slot_last_point_i ,                                                     
                                                slot_prev_point_i = slot_prev_point_i,
                                                slot_next_point_i = slot_next_point_i,
                                                from_t = slot_start_t,
                                                to_t = slot_end_t,
                                                unit = self.time_unit,
                                                point_validity = validity,
                                                timezone = timezone,
                                                fill_with = fill_with,
                                                force_data_loss = force_data_loss,
                                                fill_gaps = fill_gaps,
                                                series_data_indexes = series_data_indexes,
                                                series_resolution = series_resolution,
                                                force_compute_data_loss = True if first else False,
                                                interpolation_method=self.interpolation_method,
                                                operations = operations)
                        
                        # Set first to false
                        if first:
                            first = False
                        
                        # .. and append results
                        if new_item:
                            logger.debug('Computed new item: %s',new_item )
                            new_series.append(new_item)

                    # Create a new slot. This is where all the "conventional" time logic kicks-in, and where the timezone is required.
                    slot_start_t = slot_end_t
                    slot_start_dt = dt_from_s(slot_start_t, tz=timezone)
                    # TODO: if not calendar time unit, we can just use seconds and maybe speed up a bit.
                    slot_end_dt = slot_start_dt + self.time_unit
                    slot_end_t = s_from_dt(slot_end_dt)

                    # Reassign prev (if we have to)
                    if i>0 and series[i-1].t < slot_start_t:
                        slot_prev_point_i = i-1

                    # Log the newly spinned slot
                    logger.debug('Spinned a new slot, start=%s (%s), end=%s (%s)', slot_start_t, slot_start_dt, slot_end_t, slot_end_dt)

                # If last slot mark process as completed
                if point.dt >= to_dt:
                    process_ended = True
        
        if target == 'points':
            logger.info('Resampled %s DataTimePoints in %s DataTimePoints', count, len(new_series))
        else:
            if isinstance(series, DataTimePointSeries):
                logger.info('Aggregated %s DataTimePoints in %s DataTimeSlots', count, len(new_series))
            else:
                logger.info('Aggregated %s DataTimeSlots in %s DataTimeSlots', count, len(new_series))
                
        return new_series


#==========================
#   Resampler
#==========================

class Resampler(SlottedTransformation):
    """Resampling transformation."""

    def __init__(self, unit, interpolation_method='linear'):

        # Handle unit
        if isinstance(unit, TimeUnit):
            self.time_unit = unit
        else:
            self.time_unit = TimeUnit(unit)
        if self.time_unit.is_calendar():
            raise ValueError('Sorry, calendar time units are not supported by the Resampler (got "{}"). Use the Slotter instead.'.format(self.time_unit))

        # Set interpolation method
        self.interpolation_method=interpolation_method

    def process(self, *args, **kwargs):
        kwargs['target'] = 'points'
        return super(Resampler, self).process(*args, **kwargs) 


#==========================
#   Aggregator
#==========================

class Aggregator(SlottedTransformation):
    """Aggregation transformation."""

    def __init__(self, unit, operations=[avg], interpolation_method='linear'):
        
        # Handle unit
        if isinstance(unit, TimeUnit):
            self.time_unit = unit
        else:
            self.time_unit = TimeUnit(unit)
        
        if not operations:
            raise ValueError('No operations set, cannot slot.')
        
        # Handle operations
        operations_input = operations
        operations = []
        for operation in operations_input: 
            if isinstance(operation, str):
                try:
                    operation = getattr(operations_module, operation)
                except:
                    raise 
            operations.append(operation)
        self.operations = operations
        
        # Set interpolation method
        self.interpolation_method=interpolation_method

    def process(self, *args, **kwargs):
        kwargs['target'] = 'slots'
        return super(Aggregator, self).process(*args, **kwargs) 

















