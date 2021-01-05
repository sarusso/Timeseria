from .time import dt_from_s, s_from_dt
from .datastructures import DataTimeSlot, DataTimeSlotSeries, TimePoint, DataTimePointSeries, DataTimePoint
from .utilities import compute_coverage, is_almost_equal, is_close
from .units import TimeUnit

# Setup logging
import logging
logger = logging.getLogger(__name__)

HARD_DEBUG = False

#========================== 
#  Utilities
#==========================

def unit_to_TimeUnit(unit):
    if isinstance(unit, TimeUnit):
        time_unit = unit
    elif isinstance(unit, int):
        time_unit = TimeUnit(seconds=unit)
    elif isinstance(unit, float):
        if int(str(unit).split('.')[1]) != 0:
            raise ValueError('Cannot process decimal seconds yet')
        time_unit = TimeUnit(seconds=unit)
    elif isinstance(unit, str):
        time_unit = TimeUnit(unit)
        unit = time_unit
    else:
        raise ValueError('Unknown unit type "{}"'.format(unit.__class__.__name__))
    return time_unit


def detect_dataPoints_validity(data_time_pointSeries):

    diffs={}
    prev_data_time_point=None
    for data_time_point in data_time_pointSeries:
        if prev_data_time_point is not None:
            diff = data_time_point.t - prev_data_time_point.t
            if diff not in diffs:
                diffs[diff] = 1
            else:
                diffs[diff] +=1
        prev_data_time_point = data_time_point
    
    # Iterate until the diffs are not too spread, then pick the maximum.
    i=0
    while is_almost_equal(len(diffs), len(data_time_pointSeries)):
        or_diffs=diffs
        diffs={}
        for diff in or_diffs:
            diff=round(diff)
            if diff not in diffs:
                diffs[diff] = 1
            else:
                diffs[diff] +=1            
        
        if i > 10:
            raise Exception('Cannot automatically detect original resolution')
    
    most_common_diff_total = 0
    most_common_diff = None
    for diff in diffs:
        if diffs[diff] > most_common_diff_total:
            most_common_diff_total = diffs[diff]
            most_common_diff = diff
    return(most_common_diff)



#==========================
#  Base Transformation
#==========================

class Transformation(object):
    
    @classmethod
    def __str__(cls):
        return '{} transformation'.format(cls.__name__.replace('Operator',''))

    def process(self, *args, **kwargs):
        return self._process(*args, **kwargs)


#==========================
#  Slotter Transformation
#==========================

class Slotter(Transformation):


    def __init__(self, unit):
        if isinstance(unit, int):
            self.time_unit = unit_to_TimeUnit(unit)
        elif isinstance(unit, str) or isinstance(unit, TimeUnit):
            self.time_unit = unit_to_TimeUnit(unit)
        else:
            raise NotImplementedError('Sorry, only (re) resolutions as int (seconds) or TimeUnit objects (or their string representation) are supported')


    def _compute_slot(self, data_time_pointSeries, unit, start_t, end_t, validity, timezone, fill_with, force_coverage, fill_gaps, interpolation_method):

        # Compute coverage
        slot_coverage = compute_coverage(data_time_pointSeries,
                                         from_t   = start_t,
                                         to_t     = end_t,
                                         validity = validity)

        # If we have to fully reconstruct data
        if slot_coverage == 0 and len(data_time_pointSeries) == 2:

            # Reconstruct (fill_gaps)
            slot_data = {}
            for key in data_time_pointSeries[0].data.keys():
                
                if interpolation_method == 'linear': 
                    data_time_pointSeries[0].t
                    
                    diff = data_time_pointSeries[1].data[key] - data_time_pointSeries[0].data[key]
                    delta_t = data_time_pointSeries[1].t - data_time_pointSeries[0].t
                    ratio = diff/delta_t
                    mid_t = start_t + (end_t-start_t)
                    
                    slot_data[key] = data_time_pointSeries[0].data[key] + ((mid_t - data_time_pointSeries[0].t) * ratio)

                elif interpolation_method == 'uniform':
                    slot_data[key] = (data_time_pointSeries[0].data[key] + data_time_pointSeries[1].data[key]) /2
                else:
                    raise Exception('Unknown interpolation method "{}"'.format(interpolation_method))
  
        else:

            # Compute data
            keys = None
            avgs = {}
            count=0
            for data_time_point in data_time_pointSeries:
                if (data_time_point.t+(validity/2)) < start_t:
                    continue
                if (data_time_point.t-(validity/2)) >= end_t:
                    continue  
                count+=1              
                if not keys:
                    keys=data_time_point.data.keys()
                for key in keys:
                    if key not in avgs:
                        avgs[key] = 0
                    avgs[key] += data_time_point.data[key]
            
            # Do we have a 100% and a fill_with?
            if fill_with is not None and slot_coverage == 0:
                slot_data = {key:fill_with for key in keys}                
            else:
                slot_data = {key:avgs[key]/count for key in keys}

        # Do we have a force coverage? #TODO: do not compute coverage if fill_with not present and force_coverage 
        if force_coverage is not None:
            slot_coverage = force_coverage
        
        # Create the DataTimeSlot
        data_time_slot = DataTimeSlot(start = TimePoint(t=start_t, tz=timezone),
                                    end   = TimePoint(t=end_t, tz=timezone),
                                    unit  = unit,
                                    data  = slot_data,
                                    coverage = slot_coverage)
        
        return data_time_slot


    def _process(self, data_time_pointSeries, from_t=None, to_t=None, validity=None, force_close_last=False, include_extremes=False, fill_with=None, force_coverage=None, fill_gaps=True, interpolation_method='linear'):
        ''' Start the slotting process. If start and/or end are not set, they are set automatically based on first and last points of the sereis'''

        if not isinstance(data_time_pointSeries, DataTimePointSeries):
            raise TypeError('Can process only DataTimePointSeries, got "{}"'.format(data_time_pointSeries.__class__.__name__))

        if not data_time_pointSeries:
            raise ValueError('Cannot process empty data_time_pointSeries')

        if include_extremes:
            if from_t is not None or to_t is not None:
                raise ValueError('Setting "include_extremes" is not compatible with giving a from_t or a to_t')
            from_rounding_method = 'floor'
            to_rounding_method   = 'ceil'        
        else:
            from_rounding_method = 'ceil'
            to_rounding_method   = 'floor'

        # Set "from" if not set, otherwise check for consistency # TODO: move to steaming
        if from_t is None:
            from_t = data_time_pointSeries[0].t
            from_dt = dt_from_s(from_t)
            # Is the point already rounded to the time unit or do we have to round it ourselves?
            if not from_dt == self.time_unit.round_dt(from_dt):
                from_dt = self.time_unit.round_dt(from_dt, how=from_rounding_method)
                from_t  = s_from_dt(from_dt)
        else:
            from_dt = dt_from_s(from_t)
            if from_dt != self.time_unit.round_dt(from_dt):
                raise ValueError('Sorry, provided from_t is not consistent with the self.time_unit of "{}" (Got "{}")'.format(self.time_unit, from_t))

        # Set "to" if not set, otherwise check for consistency # TODO: move to streaming
        if to_t is None:
            to_t = data_time_pointSeries[-1].t
            to_dt = dt_from_s(to_t)
            # Is the point already rounded to the time unit or do we have to round it ourselves?
            if not to_dt == self.time_unit.round_dt(to_dt):
                to_dt = self.time_unit.round_dt(to_dt, how=to_rounding_method)
                to_t  = s_from_dt(to_dt)
        else:
            to_dt = dt_from_s(to_t)
            if to_dt != self.time_unit.round_dt(to_dt):
                raise ValueError('Sorry, provided to_t is not consistent with the self.time_unit of "{}" (Got "{}")'.format(self.time_unit, to_t))
        
        # Automatically detect validity if not set
        if validity is None:
            validity = detect_dataPoints_validity(data_time_pointSeries)
            logger.info('Auto-detected points validity: %ss', validity)
        

        
        force_close_last = force_close_last # TODO: set to true if inverting floor and ceil above)

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

        # Set timezone
        timezone  = data_time_pointSeries.tz
        logger.debug('Using timezone "%s"', timezone)
       
        # Counters
        count = 0

        # Now go trough all the data in the time series        
        for data_time_point in data_time_pointSeries:

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
                         
                # 1) Add this last point to the data_time_pointSeries:
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
                        dataTimeslot = self._compute_slot(working_serie,
                                                          unit     = self.time_unit,
                                                          start_t  = slot_start_t,
                                                          end_t    = slot_end_t,
                                                          validity = validity,
                                                          timezone = timezone,
                                                          fill_with = fill_with,
                                                          force_coverage = force_coverage,
                                                          fill_gaps = fill_gaps,
                                                          interpolation_method = interpolation_method)
                        
                        # .. and append results 
                        data_time_slot_series.append(dataTimeslot)


                    # Create a new slot. This is where all the "conventional" time logic kicks-in, and where the time zone is required.
                    slot_start_t = slot_end_t
                    slot_end_t   = s_from_dt(dt_from_s(slot_start_t, tz=timezone) + self.time_unit)
                    
                    # Create a new working_serie as part of the "create a new slot" procedure
                    working_serie = DataTimePointSeries()
                    
                    # Append the previous prev_data_time_point to the new DataTimeSeries
                    if prev_data_time_point:
                        working_serie.append(prev_data_time_point)

                    logger.debug('Spinned a new slot (start={}, end={})'.format(dt_from_s(slot_start_t), dt_from_s(slot_end_t)))
                    
                # If last slot mark process as completed (and aggregate last slot if necessary)
                if data_time_point.dt >= to_dt:

                    # Edge case where we would otherwise miss the last slot
                    if data_time_point.dt == to_dt:
                        
                        # Compute slot...
                        dataTimeslot = self._compute_slot(working_serie,
                                                          unit     = self.time_unit, 
                                                          start_t  = slot_start_t,
                                                          end_t    = slot_end_t,
                                                          validity = validity,
                                                          timezone = timezone,
                                                          fill_with = fill_with,
                                                          force_coverage = force_coverage,
                                                          fill_gaps = fill_gaps,
                                                          interpolation_method = interpolation_method)
                        
                        # .. and append results 
                        data_time_slot_series.append(dataTimeslot)

                    process_ended = True
                    

            # Append this point to the working serie
            working_serie.append(data_time_point)
            
            # ..and save as previous point
            prev_data_time_point =  data_time_point           


        # Last slots
        if process_ended and force_close_last: # or method = overset (vs subset that is the standard)

            # 1) Close the last slot and aggreagte it. You should never do it unless you knwo what you are doing
            if working_serie:
    
                logger.debug('This slot (start={}, end={}) is closed, now aggregating it..'.format(slot_start_t, slot_end_t))
      
                # Compute slot...
                dataTimeslot = self._compute_slot(working_serie,
                                                  unit     = self.time_unit, 
                                                  start_t  = slot_start_t,
                                                  end_t    = slot_end_t,
                                                  validity = validity,
                                                  timezone = timezone,
                                                  fill_with = fill_with,
                                                  force_coverage = force_coverage,
                                                  fill_gaps = fill_gaps,
                                                  interpolation_method = interpolation_method)
                
                # .. and append results 
                data_time_slot_series.append(dataTimeslot)

            # 2) Handle missing slots until the requested end (end_dt)
            # TODO: Implement it. Sure?

        logger.info('Slotted %s DataTimePoints in %s DataTimeSlots', count, len(data_time_slot_series))

        return data_time_slot_series













#==========================
#  Resampler Transformation
#==========================

class Resampler(Transformation):


    def __init__(self, unit):
        if isinstance(unit, int):
            self.time_unit = unit_to_TimeUnit(unit)
        elif isinstance(unit, str) or isinstance(unit, TimeUnit):
            self.time_unit = unit_to_TimeUnit(unit)
        else:
            raise NotImplementedError('Sorry, only (re) resolutions as int (seconds) or TimeUnit objects (or their string representation) are supported')
        if self.time_unit.is_human():
            raise ValueError('Sorry, human time units are not supported by the Resampler (got "{}"). Use the Slotter instead.'.format(self.time_unit))

    def _compute_resampled_point(self, data_time_pointSeries, unit, start_t, end_t, validity, timezone, fill_with, force_coverage, fill_gaps, interpolation_method):

        # Compute coverage
        point_coverage = compute_coverage(data_time_pointSeries,
                                         from_t   = start_t,
                                         to_t     = end_t,
                                         validity = validity)

        # If we have to reconstruct data
        if point_coverage == 0 and len(data_time_pointSeries) == 2:
            
            if not fill_gaps:
                return None
            
            else:
                # Reconstruct (fill_gaps)
                slot_data = {}
                for key in data_time_pointSeries[0].data.keys():
                    
                    if interpolation_method == 'linear':               
                        diff = data_time_pointSeries[1].data[key] - data_time_pointSeries[0].data[key]
                        delta_t = data_time_pointSeries[1].t - data_time_pointSeries[0].t
                        ratio = diff/delta_t
                        mid_t = start_t + (end_t-start_t)
                        
                        slot_data[key] = data_time_pointSeries[0].data[key] + ((mid_t - data_time_pointSeries[0].t) * ratio)
                    
                    elif interpolation_method == 'uniform':
                        slot_data[key] = (data_time_pointSeries[0].data[key] + data_time_pointSeries[1].data[key]) /2
                    else:
                        raise Exception('Unknown interpolation method "{}"'.format(interpolation_method))
                
        else:

            # Compute data
            keys = None
            avgs = {}
            count=0
            for data_time_point in data_time_pointSeries:
                if (data_time_point.t+(validity/2)) < start_t:
                    continue
                if (data_time_point.t-(validity/2)) >= end_t:
                    continue  
                count+=1
                if not keys:
                    keys=data_time_point.data.keys()
                for key in keys:
                    if key not in avgs:
                        avgs[key] = 0
                    avgs[key] += data_time_point.data[key]
            
            # Do we have a 100% and a fill_with?
            if fill_with is not None and point_coverage == 0:
                slot_data = {key:fill_with for key in keys}                
            else:
                slot_data = {key:avgs[key]/count for key in keys}

        # Do we have a force coverage? #TODO: do not compute coverage if fill_with not present and force_coverage 
        if force_coverage is not None:
            point_coverage = force_coverage
        
        # Create the DataTimePoint if we have coverage
        if not point_coverage:
            data_time_point = None
        else:      
            pass
        data_time_point = DataTimePoint(t = (start_t+((end_t-start_t)/2)),
                                            data  = slot_data,
                                            coverage = point_coverage)

        return data_time_point


    def _process(self, data_time_pointSeries, from_t=None, to_t=None, validity=None,
                 force_close_last=False, include_extremes=False, fill_with=None,
                 force_coverage=None, fill_gaps=True, interpolation_method='linear'):
        ''' Start the slotting process. If start and/or end are not set, they are set automatically based on first and last points of the sereis'''

        if not isinstance(data_time_pointSeries, DataTimePointSeries):
            raise TypeError('Can process only DataTimePointSeries, got "{}"'.format(data_time_pointSeries.__class__.__name__))

        if not data_time_pointSeries:
            raise ValueError('Cannot process empty data_time_pointSeries')

        if include_extremes:
            if from_t is not None or to_t is not None:
                raise ValueError('Setting "include_extremes" is not compatible with giving a from_t or a to_t')
            from_rounding_method = 'floor'
            to_rounding_method   = 'ceil'        
        else:
            from_rounding_method = 'ceil'
            to_rounding_method   = 'floor'

        # Set "from" if not set, otherwise check for consistency # TODO: move to steaming
        if from_t is None:
            from_t = data_time_pointSeries[0].t
            from_dt = dt_from_s(from_t)
            # Is the point already rounded to the time unit or do we have to round it ourselves?
            if not from_dt == self.time_unit.round_dt(from_dt):
                from_dt = self.time_unit.round_dt(from_dt, how=from_rounding_method)
                from_t  = s_from_dt(from_dt)
        else:
            from_dt = dt_from_s(from_t)
            if from_dt != self.time_unit.round_dt(from_dt):
                raise ValueError('Sorry, provided from_t is not consistent with the self.time_unit of "{}" (Got "{}")'.format(self.time_unit, from_t))

        # Set "to" if not set, otherwise check for consistency # TODO: move to streaming
        if to_t is None:
            to_t = data_time_pointSeries[-1].t
            to_dt = dt_from_s(to_t)
            # Is the point already rounded to the time unit or do we have to round it ourselves?
            if not to_dt == self.time_unit.round_dt(to_dt):
                to_dt = self.time_unit.round_dt(to_dt, how=to_rounding_method)
                to_t  = s_from_dt(to_dt)
        else:
            to_dt = dt_from_s(to_t)
            if to_dt != self.time_unit.round_dt(to_dt):
                raise ValueError('Sorry, provided to_t is not consistent with the self.time_unit of "{}" (Got "{}")'.format(self.time_unit, to_t))
        
        # Automatically detect validity if not set
        if validity is None:
            validity = detect_dataPoints_validity(data_time_pointSeries)
            logger.info('Auto-detected points validity: %ss', validity)
        

        
        force_close_last = force_close_last # TODO: set to true if inverting floor and ceil above)

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
        timezone  = data_time_pointSeries.tz
        logger.debug('Using timezone "%s"', timezone)
       
        # Counters
        count = 0

        # Now go trough all the data in the time series        
        for data_time_point in data_time_pointSeries:

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
                         
                # 1) Add this last point to the data_time_pointSeries:
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
                        logger.debug('working_serie  last point dt: %s', working_serie[-1].dt)

                        
                        # Compute slot...
                        dataTimePoint = self._compute_resampled_point(working_serie,
                                                                      unit     = self.time_unit,
                                                                      start_t  = slot_start_t,
                                                                      end_t    = slot_end_t,
                                                                      validity = validity,
                                                                      timezone = timezone,
                                                                      fill_with = fill_with,
                                                                      force_coverage = force_coverage,
                                                                      fill_gaps = fill_gaps,
                                                                      interpolation_method = interpolation_method)
                        
                        # .. and append results
                        if dataTimePoint:
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
                                                                      force_coverage = force_coverage,
                                                                      fill_gaps = fill_gaps,
                                                                      interpolation_method = interpolation_method)
                        
                        # .. and append results
                        if dataTimePoint:
                            resampled_data_time_point_series.append(dataTimePoint)

                    process_ended = True
                    

            # Append this point to the working serie
            working_serie.append(data_time_point)
            
            # ..and save as previous point
            prev_data_time_point =  data_time_point           


        # Last slots
        if process_ended and force_close_last: # or method = overset (vs subset that is the standard)

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
                                                              force_coverage = force_coverage,
                                                              fill_gaps = fill_gaps,
                                                              interpolation_method = interpolation_method)
                
                # .. and append results
                if dataTimePoint:
                    resampled_data_time_point_series.append(dataTimePoint)

            # 2) Handle missing slots until the requested end (end_dt)
            # TODO: Implement it. Sure?

        logger.info('Resampled %s DataTimePoints in %s DataTimePoints', count, len(resampled_data_time_point_series))

        return resampled_data_time_point_series















