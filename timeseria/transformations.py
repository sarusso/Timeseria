# -*- coding: utf-8 -*-
"""Series transformations as slotting and resampling."""

from .time import dt_from_s, s_from_dt
from .datastructures import DataTimeSlot, DataTimeSlotSeries, TimePoint, DataTimePointSeries, DataTimePoint
from .utilities import compute_data_loss
from .operations import avg
from .units import TimeUnit

# Setup logging
import logging
logger = logging.getLogger(__name__)



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
#  Resampler Transformation
#==========================

class Resampler(Transformation):
    """Resampler transformation."""

    def __init__(self, unit, interpolation_method='linear'):
        if isinstance(unit, TimeUnit):
            self.time_unit = unit
        else:
            self.time_unit = TimeUnit(unit)
        if self.time_unit.is_calendar():
            raise ValueError('Sorry, calendar time units are not supported by the Resampler (got "{}"). Use the Slotter instead.'.format(self.time_unit))
        self.interpolation_method=interpolation_method

    def _compute_resampled_point(self, data_time_point_series, unit, start_t, end_t, validity, timezone, fill_with, force_data_loss, fill_gaps, series_indexes, series_resolution, first_last):

        # Compute data_loss        
        point_data_loss = compute_data_loss(data_time_point_series,
                                            from_t   = start_t,
                                            to_t     = end_t,
                                            series_resolution = series_resolution,
                                            validity = validity,
                                            first_last = first_last)
        
        # TODO: unroll the following before the compute resampled point call
        interval_timeseries = DataTimePointSeries()
        prev_point = None
        next_point = None
        for data_time_point in data_time_point_series:
            # TODO: better check the math here. We use >= and <= because, since we are basically comparing intervals,
            #       the "right" excluded rule is excluded by comparing right with left and then left with right.
            if (data_time_point.t+(validity/2)) <= start_t:
                prev_point = data_time_point
                continue
            if (data_time_point.t-(validity/2)) >= end_t:
                next_point = data_time_point
                continue
            interval_timeseries.append(data_time_point)

        # If we have to fully reconstruct data
        if not interval_timeseries: #point_data_loss == 1: 

            # Reconstruct (fill_gaps)
            slot_data = {}
            for key in data_time_point_series[0].data.keys():
                
                # Special case with only one datapoint (i.e. beginning or end)
                if len(data_time_point_series) == 1:
                    slot_data[key] = data_time_point_series[0].data[key]
                
                else:   
                
                    prev_point = data_time_point_series[0]
                    next_point = data_time_point_series[1]
                
                    if self.interpolation_method == 'linear': 
                        diff = next_point.data[key] - prev_point.data[key]
                        delta_t = next_point.t - prev_point.t
                        ratio = diff / delta_t
                        point_t = start_t + (unit.duration_s() /2)
                        slot_data[key] = prev_point.data[key] + ((point_t - prev_point.t) * ratio)
    
                    elif self.interpolation_method == 'uniform':
                        slot_data[key] = (prev_point.data[key] + next_point.data[key]) /2
                   
                    else:
                        raise Exception('Unknown interpolation method "{}"'.format(self.interpolation_method))

        else:

            # Keys shortcut
            keys = data_time_point_series.data_keys()

            logger.debug('Slot timeseries: %s', interval_timeseries)

            # Compute sample averages data
            avgs = avg(interval_timeseries, prev_point=prev_point, next_point=next_point)
                            
            # Do we have a 100% and a fill_with?
            if fill_with is not None and point_data_loss == 1:
                slot_data = {key:fill_with for key in keys}                
            else:

                if isinstance(avgs, dict):
                    slot_data = {key:avgs[key] for key in keys}
                else:
                    slot_data = {keys[0]: avgs}
                    

        # Do we have a force data_loss? #TODO: do not compute data_loss if fill_with not present and force_data_loss 
        if force_data_loss is not None:
            point_data_loss = force_data_loss
        
#         # Create the DataTimePoint if we have data_loss
#         if not point_data_loss:
#             data_time_point = None
#         else:      
#             pass

        # Create the data time point
        data_time_point = DataTimePoint(t = (start_t+((end_t-start_t)/2)),
                                        tz = timezone,
                                        data  = slot_data,
                                        data_loss = point_data_loss)


        # Now handle indexes
        for index in series_indexes:
            if interval_timeseries:
                if index == 'data_loss':
                    continue
                index_sum = 0
                index_count = 0
                for item in interval_timeseries:
                    
                    # Get index value
                    try:
                        index_value = getattr(item, index)
                    except:
                        new_point_index_value = None
                    else:
                        if index_value is not None:
                            index_count += 1
                            index_sum += index_value
        
                # Compute the new index value (if there were indexes not None)
                if index_count > 0:
                    new_point_index_value = index_sum/index_count
                else:
                    new_point_index_value = None
    
            else:
                new_point_index_value = None
                
            # Set the index. Handle special case for data_reconstructed
            if index == 'data_reconstructed':
                setattr(data_time_point, '_data_reconstructed', new_point_index_value)
            else:
                setattr(data_time_point, index, new_point_index_value)                

        # Return
        return data_time_point


    def process(self, data_time_point_series, from_t=None, to_t=None, from_dt=None, to_dt=None,
                 validity=None, force_close_last=True, include_extremes=False, fill_with=None,
                 force_data_loss=None, fill_gaps=True, force=False):
        """Start the resampling process. If start and/or end are not set, they are set automatically
        based on first and last points of the series"""

        if not isinstance(data_time_point_series, DataTimePointSeries):
            raise TypeError('Can process only DataTimePointSeries, got "{}"'.format(data_time_point_series.__class__.__name__))

        if not data_time_point_series:
            raise ValueError('Cannot process empty data_time_point_series')

        if include_extremes:
            if from_t is not None or to_t is not None:
                raise ValueError('Setting "include_extremes" is not compatible with giving a from_t or a to_t')
            from_rounding_method = 'floor'
            to_rounding_method   = 'ceil'        
        else:
            from_rounding_method = 'ceil'
            to_rounding_method   = 'floor'

        # Move fromt_dt and to_dt to epoch to simplify the following
        if from_dt is not None:
            from_t = s_from_dt(from_dt)
        if to_dt is not None:
            to_t = s_from_dt(to_dt)
            # Also force close if we have explicitly set an end
            force_close_last = True

        # Set "from" if not set, otherwise check for consistency # TODO: move to steaming
        if from_t is None:
            from_t = data_time_point_series[0].t
            from_dt = dt_from_s(from_t, tz=data_time_point_series.tz)
            # Is the point already rounded to the time unit or do we have to round it ourselves?
            if not from_dt == self.time_unit.round_dt(from_dt):
                from_dt = self.time_unit.round_dt(from_dt, how=from_rounding_method)
                from_t  = s_from_dt(from_dt)
        else:
            from_dt = dt_from_s(from_t, tz=data_time_point_series.tz)
            if from_dt != self.time_unit.round_dt(from_dt):
                raise ValueError('Sorry, provided from_t is not consistent with the self.time_unit of "{}" (Got "{}")'.format(self.time_unit, from_t))
        
        # Set "to" if not set, otherwise check for consistency # TODO: move to streaming
        if to_t is None:
            to_t = data_time_point_series[-1].t
            to_dt = dt_from_s(to_t, tz=data_time_point_series.tz)
            # Is the point already rounded to the time unit or do we have to round it ourselves?
            if not to_dt == self.time_unit.round_dt(to_dt):
                to_dt = self.time_unit.round_dt(to_dt, how=to_rounding_method)
                to_t  = s_from_dt(to_dt)
        else:
            to_dt = dt_from_s(to_t, tz=data_time_point_series.tz)
            if to_dt != self.time_unit.round_dt(to_dt):
                raise ValueError('Sorry, provided to_t is not consistent with the self.time_unit of "{}" (Got "{}")'.format(self.time_unit, to_t))

        # Move the start back of half and the end forward of
        # half unit as well, as the point will be in the center
        from_t = from_t - (self.time_unit.duration_s() /2)
        from_dt = dt_from_s(from_t, data_time_point_series.tz)
        to_t = to_t + (self.time_unit.duration_s() /2)
        to_dt = dt_from_s(to_t, data_time_point_series.tz)

        # Automatically detect validity if not set
        if validity is None:
            validity = data_time_point_series.autodetected_sampling_interval
            logger.info('Using auto-detected sampling interval: %ss', validity)

        # Check if not upsamplimg (with some tolearance):
        if not force:
            if validity > (self.time_unit.duration_s() * 1.10):
                raise ValueError('Upsampling not supported yet (resampler unit: {}; detected time series sampling interval: {})'.format(self.time_unit, validity))

        logger.debug('Started slotter from "%s" (%s) to "%s" (%s)', from_dt, from_t, to_dt, to_t)

        if from_dt >= to_dt:
            raise ValueError('Sorry, from is >= to! (from_t={}, to_t={})'.format(from_t, to_t))

        # Set some support vars
        slot_start_t       = None
        slot_end_t         = None
        prev_data_time_point = None
        working_serie      = DataTimePointSeries()
        process_ended      = False
        resampled_data_time_point_series = DataTimePointSeries()

        # Set timezone
        timezone  = data_time_point_series.tz
        logger.debug('Using timezone "%s"', timezone)
       
        # Counters
        count = 0
        first = True

        # Indexes
        series_indexes = data_time_point_series.indexes
        series_resolution = data_time_point_series.resolution

        # Now go trough all the data in the time series        
        for data_time_point in data_time_point_series:

            logger.debug('Processing %s', data_time_point)

            # Increase counter
            count += 1
            
            # Set start_dt if not already done TODO: implement it correctly
            #if not from_dt:
            #    from_dt = self.time_unit.timeInterval.round_dt(data_time_point.dt) if rounded else data_time_point.dt
            
            # Pretend there was a slot before if we are at the beginning. TOOD: improve me.
            if slot_end_t is None:   
                slot_end_t = from_t

            # First, check if we have some points to discard at the beginning       
            if data_time_point.t < from_t:
                # If we are here it means we are going data belonging to a previous slot
                # (probably just spare data loaded to have access to the prev_datapoint)  
                prev_data_time_point = data_time_point
                continue

            # Similar concept for the end
            # TODO: what if we are in streaming mode? add if to_t is not None?
            if data_time_point.t >= to_t:
                if process_ended:
                    continue

            # The following procedure works in general for slots at the beginning and in the middle.
            # The approach is to detect if the current slot is "outdated" and spin a new one if so.

            if data_time_point.t > slot_end_t:
                # If the current slot is outdated:
                         
                # 1) Add this last point to the data_time_point_series:
                working_serie.append(data_time_point)
                 
                #2) keep spinning new slots until the current data point falls in one of them.
                
                # NOTE: Read the following "while" more as an "if" which can also lead to spin multiple
                # slot if there are empty slots between the one being closed and the data_time_point.dt.
                # TODO: leave or remove the above if for code readability?
                
                while slot_end_t < data_time_point.t:
                    logger.debug('Checking for end {} with point {}'.format(slot_end_t, data_time_point.t))
                    # If we are in the pre-first slot, just silently spin a new slot:
                    if slot_start_t is not None:
                        
                        # Append last point. Can be appended to multiple slots, this is normal since
                        # the empty slots in the middle will have only a far prev and a far next.
                        # can also be appended several times if working_serie is not reset (looping in the while)
                        if data_time_point not in working_serie:
                            working_serie.append(data_time_point)
  
                        logger.debug('This slot (start={}, end={}) is closed, now aggregating it..'.format(slot_start_t, slot_end_t))
                        
                        logger.debug('working_serie len: %s', len(working_serie))
                        logger.debug('working_serie first point dt: %s', working_serie[0].dt)
                        logger.debug('working_serie  last point dt: %s', working_serie[-1].dt)

                        
                        # Compute slot...
                        dataTimePoint = self._compute_resampled_point(working_serie,
                                                                      unit     = self.time_unit,
                                                                      start_t  = slot_start_t,
                                                                      end_t    = slot_end_t,
                                                                      validity = validity,
                                                                      timezone = timezone,
                                                                      fill_with = fill_with,
                                                                      force_data_loss = force_data_loss,
                                                                      fill_gaps = fill_gaps,
                                                                      series_indexes = series_indexes,
                                                                      series_resolution = series_resolution,
                                                                      first_last = first)
                        # Set first to false
                        if first:
                            first = False
                        
                        # .. and append results
                        if dataTimePoint:
                            logger.debug('Computed datapoint: %s',dataTimePoint )
                            resampled_data_time_point_series.append(dataTimePoint)


                    # Create a new slot. This is where all the "conventional" time logic kicks-in, and where the time zone is required.
                    slot_start_t = slot_end_t
                    slot_end_t   = s_from_dt(dt_from_s(slot_start_t, tz=timezone) + self.time_unit)

                    # Create a new working_serie as part of the "create a new slot" procedure
                    working_serie = DataTimePointSeries()
                    
                    # Append the previous prev_data_time_point to the new DataTimeSeries
                    if prev_data_time_point:
                        working_serie.append(prev_data_time_point)

                    logger.debug('Spinned a new slot (start={}, end={})'.format(slot_start_t, slot_end_t))
                    
                # If last slot mark process as completed (and aggregate last slot if necessary)
                if data_time_point.dt >= to_dt:

                    # Edge case where we would otherwise miss the last slot
                    if data_time_point.dt == to_dt:
                        
                        # Compute slot...
                        dataTimePoint = self._compute_resampled_point(working_serie,
                                                                      unit     = self.time_unit, 
                                                                      start_t  = slot_start_t,
                                                                      end_t    = slot_end_t,
                                                                      validity = validity,
                                                                      timezone = timezone,
                                                                      fill_with = fill_with,
                                                                      force_data_loss = force_data_loss,
                                                                      fill_gaps = fill_gaps,
                                                                      series_indexes = series_indexes,
                                                                      series_resolution = series_resolution,
                                                                      first_last = True)
                        
                        # .. and append results
                        if dataTimePoint:
                            resampled_data_time_point_series.append(dataTimePoint)

                    process_ended = True
                    

            # Append this point to the working serie
            working_serie.append(data_time_point)
            
            # ..and save as previous point
            prev_data_time_point =  data_time_point           


        # Last slots
        if force_close_last:

            # 1) Close the last slot and aggreagte it. You should never do it unless you knwo what you are doing
            if working_serie:
    
                logger.debug('This resampled point (start={}, end={}) is done, now computing it..'.format(slot_start_t, slot_end_t))
      
                # Compute slot...
                dataTimePoint = self._compute_resampled_point(working_serie,
                                                              unit     = self.time_unit, 
                                                              start_t  = slot_start_t,
                                                              end_t    = slot_end_t,
                                                              validity = validity,
                                                              timezone = timezone,
                                                              fill_with = fill_with,
                                                              force_data_loss = force_data_loss,
                                                              fill_gaps = fill_gaps,
                                                              series_indexes = series_indexes,
                                                              series_resolution = series_resolution,
                                                              first_last = True)
                
                # .. and append results
                if dataTimePoint:
                    resampled_data_time_point_series.append(dataTimePoint)

            # 2) Handle missing slots until the requested end (end_dt)
            # TODO: Implement it. Sure?

        logger.info('Resampled %s DataTimePoints in %s DataTimePoints', count, len(resampled_data_time_point_series))

        return resampled_data_time_point_series



#==========================
#  Slotter Transformation
#==========================

class Slotter(Transformation):
    """Slotter transformation."""

    def __init__(self, unit, default_operation=avg, extra_operations=None, interpolation_method='linear'):
        if isinstance(unit, TimeUnit):
            self.time_unit = unit
        else:
            self.time_unit = TimeUnit(unit)
        self.default_operation=default_operation
        self.extra_operations=extra_operations
        self.interpolation_method=interpolation_method


    def _compute_slot(self, data_time_point_series, unit, start_t, end_t, validity, timezone, fill_with, force_data_loss, fill_gaps, series_indexes, series_resolution, first_last):

        # Compute data_loss
        slot_data_loss = compute_data_loss(data_time_point_series,
                                           from_t   = start_t,
                                           to_t     = end_t,
                                           series_resolution = series_resolution,
                                           validity = validity,
                                           first_last = first_last)
        
        # Initialize slot data
        slot_data = {}

        # TODO: unroll the following before the compute slot call
        slot_timeseries = DataTimePointSeries()
        prev_point = None
        next_point = None
        for data_time_point in data_time_point_series:
            if (data_time_point.t+(validity/2)) < start_t:
                prev_point = data_time_point
                continue
            if (data_time_point.t-(validity/2)) >= end_t:
                next_point = data_time_point
                continue
            slot_timeseries.append(data_time_point)


        # If we have to fully reconstruct data
        # TODO: we have a huge conceputal problem here if checkong on the slot_data_loss:
        #       if data_loss is 1 but due to point data loses (maybe even reconstructed)
        #       and not entirely missing points, we should actually compute as if not a full data loss.
        
        if not slot_timeseries: # slot_data_loss == 1: 

            # Reconstruct (fill gaps)
            for key in data_time_point_series[0].data.keys():
                
                # Set data as None, will be interpolated afterwards
                slot_data['{}_{}'.format(key, self.default_operation.__name__)] = None
                
                # Handle also extra ops
                if self.extra_operations:
                    for extra_operation in self.extra_operations:
                        slot_data['{}_{}'.format(key, extra_operation.__name__)] = None

        else:
 
            # Keys shortcut
            keys = data_time_point_series.data_keys()

            # Compute the default operation (in some cases it might not be defined, hence the "if")
            if self.default_operation:
                default_operation_data = self.default_operation(slot_timeseries, prev_point=prev_point, next_point=next_point)
                                
                # Do we have a 100% and a fill_with?
                if fill_with is not None and slot_data_loss == 1:
                    for key in keys:
                        slot_data['{}_{}'.format(key, self.default_operation.__name__)] = fill_with
                   
                else:
    
                    #if isinstance(default_operation_data, dict):
                    #    slot_data = {key:default_operation_data[key] for key in keys}
                    #else:
                    #    slot_data = {keys[0]: default_operation_data}
                    
                    if isinstance(default_operation_data, dict):
                        for key in keys:
                            slot_data['{}_{}'.format(key, self.default_operation.__name__)] = default_operation_data[key]
                    else:
                        slot_data['{}_{}'.format(keys[0], self.default_operation.__name__)] = default_operation_data

            # Handle extra operations
            if self.extra_operations:
                for extra_operation in self.extra_operations:
                    extra_operation_data = extra_operation(slot_timeseries, prev_point=prev_point, next_point=next_point)
                    if isinstance(extra_operation_data, dict):
                        for result_key in extra_operation_data:
                            slot_data['{}_{}'.format(result_key, extra_operation.__name__)] = extra_operation_data[result_key]
                    else:
                        slot_data['{}_{}'.format(keys[0], extra_operation.__name__)] = extra_operation_data

        # Do we have a force data_loss? #TODO: do not compute data_loss if fill_with not present and force_data_loss 
        if force_data_loss is not None:
            slot_data_loss = force_data_loss

        # Create the DataTimeSlot
        data_time_slot = DataTimeSlot(start = TimePoint(t=start_t, tz=timezone),
                                      end   = TimePoint(t=end_t, tz=timezone),
                                      unit  = unit,
                                      data  = slot_data,
                                      data_loss = slot_data_loss)

        # Now handle indexes
        for index in series_indexes:
            if slot_timeseries:
                if index == 'data_loss':
                    continue
                index_sum = 0
                index_count = 0
                for item in slot_timeseries:
                    
                    # Get index value
                    try:
                        index_value = getattr(item, index)
                    except:
                        slotted_index_value = None
                    else:
                        if index_value is not None:
                            index_count += 1
                            index_sum += index_value
        
                # Compute the slotted index value (if there were indexes not None)
                if index_count > 0:
                    slotted_index_value = index_sum/index_count
                else:
                    slotted_index_value = None
    
            else:
                slotted_index_value = None

            # Set the index. Handle special case for data_reconstructed
            if index == 'data_reconstructed':
                setattr(data_time_slot, '_data_reconstructed', slotted_index_value)
            else:
                setattr(data_time_slot, index, slotted_index_value)                
        
                    
        # Return the slot
        return data_time_slot


    def process(self, data_time_point_series, from_t=None, from_dt=None, to_t=None, to_dt=None, validity=None, force_close_last=False,
                 include_extremes=False, fill_with=None, force_data_loss=None, fill_gaps=True, force=False):
        """Start the slotting process. If start and/or end are not set, they are set automatically based on first and last points of the series."""

        if not isinstance(data_time_point_series, DataTimePointSeries):
            raise TypeError('Can process only DataTimePointSeries, got "{}"'.format(data_time_point_series.__class__.__name__))

        if not data_time_point_series:
            raise ValueError('Cannot process empty data_time_point_series')

        if include_extremes:
            if from_t is not None or to_t is not None:
                raise ValueError('Setting "include_extremes" is not compatible with giving a from_t or a to_t')
            from_rounding_method = 'floor'
            to_rounding_method   = 'ceil' 
            force_close_last = True       
        else:
            from_rounding_method = 'ceil'
            to_rounding_method   = 'floor'
            
        # Move fromt_dt and to_dt to epoch to simplify the following
        if from_dt is not None:
            from_t = s_from_dt(from_dt)
        if to_dt is not None:
            to_t = s_from_dt(to_dt)
            # Also force close if we have explicitly set an end
            force_close_last = True

        # Set "from" if not set, otherwise check for consistency # TODO: move to steaming
        if from_t is None:
            from_t = data_time_point_series[0].t
            from_dt = dt_from_s(from_t, data_time_point_series.tz)
            # Is the point already rounded to the time unit or do we have to round it ourselves?
            if not from_dt == self.time_unit.round_dt(from_dt):
                from_dt = self.time_unit.round_dt(from_dt, how=from_rounding_method)
                from_t  = s_from_dt(from_dt)
        else:
            from_dt = dt_from_s(from_t, data_time_point_series.tz)
            if from_dt != self.time_unit.round_dt(from_dt):
                raise ValueError('Sorry, provided from_t is not consistent with the self.time_unit of "{}" (Got "{}")'.format(self.time_unit, from_t))

        # Set "to" if not set, otherwise check for consistency # TODO: move to streaming
        if to_t is None:
            to_t = data_time_point_series[-1].t
            to_dt = dt_from_s(to_t, data_time_point_series.tz)
            # Is the point already rounded to the time unit or do we have to round it ourselves?
            if not to_dt == self.time_unit.round_dt(to_dt):
                to_dt = self.time_unit.round_dt(to_dt, how=to_rounding_method)
                to_t  = s_from_dt(to_dt)
        else:
            to_dt = dt_from_s(to_t, data_time_point_series.tz)
            if to_dt != self.time_unit.round_dt(to_dt):
                raise ValueError('Sorry, provided to_t is not consistent with the self.time_unit of "{}" (Got "{}")'.format(self.time_unit, to_t))
            # Also force close if we have explicitly set an end
            force_close_last = True
            
        # Automatically detect validity if not set
        if validity is None:
            validity = data_time_point_series.autodetected_sampling_interval
            logger.info('Using auto-detected sampling interval: %ss', validity)

        # Check if not upslotting (with some tolerance)
        if not force:
            # TODO: this check is super-weak. Will fail in loads of edge cases, i.e. months slotted in 30 days.
            unit_duration_s = self.time_unit.duration_s(data_time_point_series[0].dt)
            if validity > (unit_duration_s * 1.1):
                raise ValueError('Upslotting not supported yet (slotter unit: {}; detected time series sampling interval: {})'.format(unit_duration_s, validity))
            
        # Log
        logger.debug('Started slotter from "%s" (%s) to "%s" (%s)', from_dt, from_t, to_dt, to_t)

        if from_dt >= to_dt:
            raise ValueError('Sorry, from is >= to! (from_t={}, to_t={})'.format(from_t, to_t))

        # Set some support vars
        slot_start_t       = None
        slot_end_t         = None
        prev_data_time_point = None
        working_serie      = DataTimePointSeries()
        process_ended      = False
        data_time_slot_series  = DataTimeSlotSeries()
        slots_to_be_interpolated = []
        last_no_full_data_loss_slot = None

        # Set timezone
        timezone  = data_time_point_series.tz
        logger.debug('Using timezone "%s"', timezone)
       
        # Counters
        count = 0
        first = True

        # Indexes
        series_indexes = data_time_point_series.indexes
        series_resolution = data_time_point_series.resolution

        # Now go trough all the data in the time series        
        for data_time_point in data_time_point_series:

            logger.debug('Processing %s', data_time_point)

            # Increase counter
            count += 1
            
            # Set start_dt if not already done TODO: implement it correctly
            #if not from_dt:
            #    from_dt = self.time_unit.timeInterval.round_dt(data_time_point.dt) if rounded else data_time_point.dt
            
            # Pretend there was a slot before if we are at the beginning. TOOD: improve me.
            if slot_end_t is None:   
                slot_end_t = from_t

            # First, check if we have some points to discard at the beginning       
            if data_time_point.t < from_t:
                # If we are here it means we are going data belonging to a previous slot
                # (probably just spare data loaded to have access to the prev_datapoint)  
                prev_data_time_point = data_time_point
                continue

            # Similar concept for the end
            # TODO: what if we are in streaming mode? add if to_t is not None?
            if data_time_point.t >= to_t:
                if process_ended:
                    break

            # The following procedure works in general for slots at the beginning and in the middle.
            # The approach is to detect if the current slot is "outdated" and spin a new one if so.

            if data_time_point.t > slot_end_t:
                # If the current slot is outdated:
                         
                # 1) Add this last point to the data_time_point_series:
                working_serie.append(data_time_point)
                 
                #2) keep spinning new slots until the current data point falls in one of them.
                
                # NOTE: Read the following "while" more as an "if" which can also lead to spin multiple
                # slot if there are empty slots between the one being closed and the data_time_point.dt.
                # TODO: leave or remove the above if for code readability?
                
                while slot_end_t < data_time_point.t:
                    logger.debug('Checking for end {} with point {}'.format(slot_end_t, data_time_point.t))
                    # If we are in the pre-first slot, just silently spin a new slot:
                    if slot_start_t is not None:
                        
                        # Append last point. Can be appended to multiple slots, this is normal since
                        # the empty slots in the middle will have only a far prev and a far next.
                        # can also be appended several times if working_serie is not reset (looping in the while)
                        if data_time_point not in working_serie:
                            working_serie.append(data_time_point)
  
                        logger.debug('This slot (start={}, end={}) is closed, now aggregating it..'.format(slot_start_t, slot_end_t))
                        
                        logger.debug('working_serie first point dt: %s', working_serie[0].dt)
                        logger.debug('working_serie last point dt: %s', working_serie[-1].dt)

                        # Compute slot...
                        data_time_slot = self._compute_slot(working_serie,
                                                            unit     = self.time_unit,
                                                            start_t  = slot_start_t,
                                                            end_t    = slot_end_t,
                                                            validity = validity,
                                                            timezone = timezone,
                                                            fill_with = fill_with,
                                                            force_data_loss = force_data_loss,
                                                            fill_gaps = fill_gaps,
                                                            series_indexes = series_indexes,
                                                            series_resolution = series_resolution,
                                                            first_last = first)

                        # Set first to false
                        if first:
                            first = False
                        
                        # .. and append results (unless we are before the first timeseries start point)
                        if slot_end_t > data_time_point_series[0].t:
                            if data_time_slot.data_loss == 1.0:
                                # if data loss is full, append to slot to the slots to be interpolated
                                slots_to_be_interpolated.append(data_time_slot)
                            else:
                                # If we have slots to be intepolated
                                if slots_to_be_interpolated:
                                    for i, slot_to_be_interpolated in enumerate(slots_to_be_interpolated):

                                        # Prepare for interpolated data
                                        interpolated_data = {}

                                        # Computed interpolated data
                                        if self.interpolation_method == 'linear': 
                                            for data_key in data_time_slot_series.data_keys():                                                
                                                interpolated_data[data_key] = ((((data_time_slot.data[data_key] - last_no_full_data_loss_slot.data[data_key]) /
                                                                                 (len(slots_to_be_interpolated) + 1) ) * (i+1))  + last_no_full_data_loss_slot.data[data_key])
    
                                        elif self.interpolation_method == 'uniform':
                                            for data_key in data_time_slot_series.data_keys():                                                
                                                interpolated_data[data_key] = (((data_time_slot.data[data_key] - last_no_full_data_loss_slot.data[data_key]) / 2) 
                                                                                + last_no_full_data_loss_slot.data[data_key])
                                           
                                        else:
                                            raise Exception('Unknown interpolation method "{}"'.format(self.interpolation_method))

                                        # Add interpolated data
                                        slot_to_be_interpolated._data = interpolated_data
                                        data_time_slot_series.append(slot_to_be_interpolated)
                                        
                                    # Reset the "buffer"
                                    slots_to_be_interpolated = []

                                # Append this slot to the time series
                                data_time_slot_series.append(data_time_slot)
                                
                                # ... and set this slot as the last with no full data loss
                                last_no_full_data_loss_slot = data_time_slot


                    # Create a new slot. This is where all the "calendar" time unit logic kicks-in, and where the time zone is required.
                    slot_start_t = slot_end_t
                    slot_start_dt = dt_from_s(slot_start_t, tz=timezone)
                    slot_end_t   = s_from_dt(dt_from_s(slot_start_t, tz=timezone) + self.time_unit)
                    slot_end_dt = dt_from_s(slot_end_t, tz=timezone)

                    # Create a new working_serie as part of the "create a new slot" procedure
                    working_serie = DataTimePointSeries()
                    
                    # Append the previous prev_data_time_point to the new DataTimeSeries
                    if prev_data_time_point:
                        working_serie.append(prev_data_time_point)

                    logger.debug('Spinned a new slot (start={}, end={})'.format(slot_start_dt, slot_end_dt))
                    
                # If last slot mark process as completed (and aggregate last slot if necessary)
                if data_time_point.dt >= to_dt:

                    # Edge case where we would otherwise miss the last slot
                    if data_time_point.dt == to_dt:
                        
                        # Compute slot...
                        data_time_slot = self._compute_slot(working_serie,
                                                            unit     = self.time_unit, 
                                                            start_t  = slot_start_t,
                                                            end_t    = slot_end_t,
                                                            validity = validity,
                                                            timezone = timezone,
                                                            fill_with = fill_with,
                                                            force_data_loss = force_data_loss,
                                                            fill_gaps = fill_gaps,
                                                            series_indexes = series_indexes,
                                                            series_resolution = series_resolution,
                                                            first_last = True)
                        
                        # .. and append results 
                        data_time_slot_series.append(data_time_slot)

                    process_ended = True
                    

            # Append this point to the working serie
            working_serie.append(data_time_point)
            
            # ..and save as previous point
            prev_data_time_point =  data_time_point           

        # Last slots
        if force_close_last:

            # 1) Close the last slot and aggreagte it. You should never do it unless you knwo what you are doing
            if working_serie:
    
                logger.debug('This slot (start={}, end={}) is closed, now aggregating it..'.format(slot_start_t, slot_end_t))
      
                # Compute slot...
                data_time_slot = self._compute_slot(working_serie,
                                                    unit     = self.time_unit, 
                                                    start_t  = slot_start_t,
                                                    end_t    = slot_end_t,
                                                    validity = validity,
                                                    timezone = timezone,
                                                    fill_with = fill_with,
                                                    force_data_loss = force_data_loss,
                                                    fill_gaps = fill_gaps,
                                                    series_indexes = series_indexes,
                                                    series_resolution = series_resolution,
                                                    first_last = True)
                
                # .. and append results 
                data_time_slot_series.append(data_time_slot)

            # 2) Handle missing slots until the requested end (end_dt)
            # TODO: Implement it. Sure? Clashes with the idea of reconstructors..

        logger.info('Slotted %s DataTimePoints in %s DataTimeSlots', count, len(data_time_slot_series))

        return data_time_slot_series


