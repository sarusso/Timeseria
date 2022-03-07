# -*- coding: utf-8 -*-
"""Series transformations as slotting and resampling."""

from .time import dt_from_s, s_from_dt
from .datastructures import DataTimeSlot, DataTimeSlotSeries, TimePoint, DataTimePointSeries, DataTimePoint
from .utilities import compute_data_loss, compute_validity_regions 
from .operations import avg, wavg
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
#  Support functions
#==========================

def _compute_new(type,
                 series,
                 unit,
                 start_t,
                 end_t,
                 validity,
                 timezone,
                 fill_with,
                 force_data_loss,
                 fill_gaps,
                 series_indexes,
                 series_resolution,
                 first_last,
                 interpolation_method=None,
                 default_operation=None,
                 extra_operations=None):
    
    # Support vars
    interval_duration = end_t-start_t
    
    # First, compute the validity regions for the points belonging to this interval.
    validity_regions = compute_validity_regions(series, from_t=start_t, to_t=end_t, shrink=True, sampling_interval=validity)

    # TODO: forward them to the compute_data_loss and in turn, the compute_coverage to avoid computing them again.
 
    # Second, compute the data loss for the new element. This is forced
    # by the resmapler or slotter if first or last point     
    data_loss = compute_data_loss(series,
                                  from_t   = start_t,
                                  to_t     = end_t,
                                  sampling_interval = validity,
                                  force = first_last)
    logger.debug('Computed data loss: "{}"'.format(data_loss))
    
    # The "series" now includes the previous and next data points as well.
    
    # For each point, attach the "weight"/"contirbution
    for point in series:
        try:
            point.weight = (validity_regions[point.t][1]-validity_regions[point.t][0])/interval_duration
        except KeyError:
            # If we are here it means that the point is completely outside the interval
            # This is required if in the operation there is the need of knowing the next
            # (or prev) value, even if it was far away
            point.weigth = 0
           
    # Create the series containing only the data points for the interval, 
    # and set prev and next. As always, left included, right excluded
    interval_series = DataTimePointSeries()
    prev_point = None
    next_point = None
    for point in series:
        if point.t <= start_t:
            prev_point = point
            continue
        if point.t > end_t:
            next_point = point
            continue
        interval_series.append(point)
    logger.debug('Interval series: %s', interval_series)


    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #  Compute validity regions, check if we have some data or to reconstruct it enturely, then do the stuff 
    #  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    # Prepare to compute data
    data = {}
    data_keys = series.data_keys()


    #===================================
    #  Compute point data
    #===================================
    if type=='point':
        # If we have to fully reconstruct data
        if data_loss == 1: 
    
            # Reconstruct
            for key in data_keys:
                
                # If we have to fill with a specific value, use it
                if fill_with:
                    data = {key:fill_with for key in data_keys}
                    continue                
 
                # If we only have one data point (i.e. beginning or end),
                # Then the reconstruction value is the data point value
                if len(interval_series) == 1:
                    data[key] = interval_series[0].data[key]
                
                # Otherwise, interpolate and compute this element value
                # with respect to the entire interpolation
                else:   

                    if interpolation_method == 'linear': 
                        diff = next_point.data[key] - prev_point.data[key]
                        delta_t = next_point.t - prev_point.t
                        ratio = diff / delta_t
                        point_t = start_t + (unit.duration_s() /2)
                        data[key] = prev_point.data[key] + ((point_t - prev_point.t) * ratio)
    
                    elif interpolation_method == 'uniform':
                        data[key] = (prev_point.data[key] + next_point.data[key]) /2
                   
                    else:
                        raise Exception('Unknown interpolation method "{}"'.format(interpolation_method))
    
        else:    
    
            # If we have some data to work on, compute the averages data
            avgs = avg(interval_series, prev_point=prev_point, next_point=next_point)
            
            # ...and assign them to the data value
            if isinstance(avgs, dict):
                data = {key:avgs[key] for key in data_keys}
            else:
                data = {data_keys[0]: avgs}
                        
    #===================================
    #  Compute slot data
    #===================================
    elif type=='slot':
        
        if data_loss == 1:
            
            # TODO: unroll the following, check also for the points?
        
            # If we have to fill with a specific value, use it
            if fill_with:
                for key in data_keys:
                    if default_operation:
                        data['{}_{}'.format(key, default_operation.__name__)] = fill_with
                    if extra_operations:
                        for extra_operation in extra_operations:
                            data['{}_{}'.format(key, default_operation.__name__)] = fill_with
            else:
            
                # Reconstruct (fill gaps)
                for key in data_keys:
                    
                    # Set data as None, it will be interpolated afterwards
                    data['{}_{}'.format(key, default_operation.__name__)] = None
                    
                    # Handle also extra operations
                    if extra_operations:
                        for extra_operation in extra_operations:
                            data['{}_{}'.format(key, extra_operation.__name__)] = None
            
        else:

            # Compute the default operation (in some cases it might not be defined, hence the "if")
            if default_operation:
                default_operation_data = default_operation(interval_series, prev_point=prev_point, next_point=next_point)
                                
                # ... and add to the data
                if isinstance(default_operation_data, dict):
                    for key in data_keys:
                        data['{}_{}'.format(key, default_operation.__name__)] = default_operation_data[key]
                else:
                    data['{}_{}'.format(data_keys[0], default_operation.__name__)] = default_operation_data
    
            # Handle extra operations
            if extra_operations:
                
                for extra_operation in extra_operations:
                    extra_operation_data = extra_operation(interval_series, prev_point=prev_point, next_point=next_point)
                    
                    # ...and add to the data
                    if isinstance(extra_operation_data, dict):
                        for result_key in extra_operation_data:
                            data['{}_{}'.format(result_key, extra_operation.__name__)] = extra_operation_data[result_key]
                    else:
                        data['{}_{}'.format(data_keys[0], extra_operation.__name__)] = extra_operation_data
        
        if not data:
            raise Exception('No data computed at all?')
    
    else:
        raise ValueError('No idea how to compute a new "{}"'.format(type))

    #===================================
    #  Now create the new element
    #===================================

    # Do we have a force data_loss? #TODO: do not compute data_loss if fill_with not present and force_data_loss 
    if force_data_loss is not None:
        data_loss = force_data_loss
    
    # Create the new item
    if type=='point':
        new_element = DataTimePoint(t = (start_t+((interval_duration)/2)),
                                    tz = timezone,
                                    data  = data,
                                    data_loss = data_loss)
    elif type=='slot':
        # Create the DataTimeSlot
        new_element = DataTimeSlot(start = TimePoint(t=start_t, tz=timezone),
                                   end   = TimePoint(t=end_t, tz=timezone),
                                   unit  = unit,
                                   data  = data,
                                   data_loss = data_loss)
    else:
        raise ValueError('No idea how to create a new "{}"'.format(type))

    # Now handle indexes
    for index in series_indexes:
        
        # Skip the data loss as it is recomputed with different logics
        if index == 'data_loss':
            continue
        
        if interval_series:
            index_sum = 0
            index_count = 0
            for item in interval_series:
                
                # Get index value
                try:
                    index_value = getattr(item, index)
                except:
                    new_element_index_value = None
                else:
                    if index_value is not None:
                        index_count += 1
                        index_sum += index_value
    
            # Compute the new index value (if there were indexes not None)
            if index_count > 0:
                new_element_index_value = index_sum/index_count
            else:
                new_element_index_value = None

        else:
            new_element_index_value = None
            
        # Set the index. Handle special case for data_reconstructed
        if index == 'data_reconstructed':
            setattr(new_element, '_data_reconstructed', new_element_index_value)
        else:
            setattr(new_element, index, new_element_index_value)                

    # Return
    return new_element





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

            # Is the current slot outdated? (are we processing a datapoint falling outside the current slot?)            
            if data_time_point.t >= slot_end_t:
                logger.debug('Detetcted outdated slot')
                # This approach is to detect if the current slot is "outdated" and spin a new one if so.
                # Works only for slots at the beginning and in the middle, but not for the last slot
                # or the missing slots at the end which need to be closed down here
                         
                # Add this last point to the data_time_point_series, it will be our "next"
                #working_serie.append(data_time_point)
                 
                # Then, keep spinning new slots until the current data point falls in one of them.
                # NOTE: Read the following "while" more as an "if" which can also lead to spin multiple
                # slot if there are empty slots between the one being closed and the data_time_point.dt.
                # TODO: leave or remove the above if for code readability?
                
                while slot_end_t <= data_time_point.t:
                    logger.debug('Checking for end %s with point %s', slot_end_t, data_time_point.t)
                    
                    if slot_start_t is None:
                        # If we are in the pre-first slot, just silently spin a new slot
                        pass
                
                    else:
                        
                        # Append last point, 
                        # Note: the same point can be appended to multiple slots, this is normal since
                        # the empty slots in the middle will have only a far prev and a far next.
                        # can also be appended several times if working_serie is not reset (looping in the while)
                        #if working_serie[-1].t >= slot_end_t:
                        #    if data_time_point not in working_serie:
                        logger.debug('Appending this (next) to working series: %s', data_time_point)
                        working_serie.append(data_time_point)
      
                        logger.debug('This slot (start=%s, end=%s) is closed, now computing it..', slot_start_t, slot_end_t)
                        logger.debug('working_serie first point dt: %s (t=%s)', working_serie[0].dt, working_serie[0].t)
                        logger.debug('working_serie  last point dt: %s (t=%s)', working_serie[-1].dt, working_serie[-1].t)

                        
                        # Compute slot...
                        dataTimePoint = _compute_new('point',
                                                     working_serie,
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
                                                      first_last = True if first else False,
                                                      interpolation_method=self.interpolation_method)
                        # Set first to false
                        if first:
                            first = False
                        
                        # .. and append results
                        if dataTimePoint:
                            logger.debug('Computed datapoint: %s',dataTimePoint )
                            resampled_data_time_point_series.append(dataTimePoint)


                    # Create a new slot. This is where all the "conventional" time logic kicks-in, and where the time zone is required.
                    slot_start_t = slot_end_t
                    slot_start_dt = dt_from_s(slot_start_t, tz=timezone)
                    slot_end_dt = slot_start_dt + self.time_unit
                    slot_end_t = s_from_dt(slot_end_dt)

                    # Create a new working_serie as part of the "create a new slot" procedure
                    working_serie = DataTimePointSeries()
                    
                    # Append the last prev_data_time_point to the new DataTimeSeries,
                    # that will be in turn the new prev since not yet updated outside
                    if prev_data_time_point:
                        logger.debug('Appending new prev point to working series: %s', prev_data_time_point)
                        working_serie.append(prev_data_time_point)
                    
                    # Now append this point to the new working series
                    logger.debug('Appending new this point to working series: %', data_time_point)
                    working_serie.append(data_time_point)
                
                    # ..and save it as previous point
                    logger.debug('Setting prev to %s', data_time_point.dt)
                    prev_data_time_point = data_time_point     

                    # Log the newly spinned slot
                    logger.debug('Spinned a new slot, start=%s (%s), end=%s (%s)', slot_start_t, slot_start_dt, slot_end_t, slot_end_dt)
                   
                    
                # If last slot mark process as completed (and aggregate last slot if necessary)
                if data_time_point.dt >= to_dt:

                    # Edge case where we would otherwise miss the last slot
                    if data_time_point.dt == to_dt:
                        
                        # Compute slot...
                        dataTimePoint = _compute_new('point',
                                                     working_serie,
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
                                                      first_last = True,
                                                      interpolation_method = self.interpolation_method)
                        
                        # .. and append results
                        if dataTimePoint:
                            resampled_data_time_point_series.append(dataTimePoint)

                    process_ended = True
            
            else:
                
                # Normal operation mode: append this point to the working series
                logger.debug('Appending starndard this point to working series: %s', data_time_point)
                working_serie.append(data_time_point)
                
                # ..and save it as previous point
                logger.debug('Setting prev to %s', data_time_point.dt)
                prev_data_time_point =  data_time_point           


        # Last slots
        if force_close_last:

            # 1) Close the last slot and aggreagte it. You should never do it unless you knwo what you are doing
            if working_serie:
    
                logger.debug('This resampled point (start=%s, end=%s) is done, now computing it..', slot_start_t, slot_end_t)
      
                # Compute slot...
                dataTimePoint = _compute_new('point',
                                             working_serie,
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
                                              first_last = True,
                                              interpolation_method=self.interpolation_method)
                
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
                        data_time_slot = _compute_new('slot',
                                                      working_serie,
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
                                                        first_last = first,
                                                        default_operation=self.default_operation,
                                                        extra_operations=self.extra_operations)

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
                        data_time_slot = _compute_new('slot',
                                                        working_serie,
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
                                                        first_last = True,
                                                        default_operation=self.default_operation,
                                                        extra_operations=self.extra_operations)
                        
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
                data_time_slot = _compute_new('slot',
                                                working_serie,
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
                                                first_last = True,
                                                default_operation=self.default_operation,
                                                extra_operations=self.extra_operations)
                
                # .. and append results 
                data_time_slot_series.append(data_time_slot)

            # 2) Handle missing slots until the requested end (end_dt)
            # TODO: Implement it. Sure? Clashes with the idea of reconstructors..

        logger.info('Slotted %s DataTimePoints in %s DataTimeSlots', count, len(data_time_slot_series))

        return data_time_slot_series


