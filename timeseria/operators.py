from .time import TimeSpan, dt_from_s, s_from_dt
from .datastructures import DataTimeSlot, DataTimeSlotSerie, TimePoint, DataTimePointSerie
from .utilities import compute_coverage

# Setup logging
import logging
logger = logging.getLogger(__name__)

HARD_DEBUG = False


#======================
#  Utilities
#======================

def almostequal(one, two):
    if 0.95 < (one / two) <= 1.05:
        return True
    else:
        return False 


class Operator(object):
    pass


#======================
#  Slotter
#======================

class Slotter(object):
    
    def __init__(self, span):
        self.span=span
        if isinstance(span, TimeSpan):
            timeSpan = span
        elif isinstance(span, int):
            timeSpan = TimeSpan(seconds=span)
        elif isinstance(span, float):
            if int(str(span).split('.')[1]) != 0:
                raise ValueError('Cannot process decimal seconds yet')
            timeSpan = TimeSpan(seconds=span)
        elif  isinstance(span, str):
            timeSpan = TimeSpan(span)
            self.span=timeSpan
        else:
            raise ValueError('Unknown span type "{}"'.format(span.__class__.__name__))

        self.timeSpan = timeSpan

    @classmethod
    def _detect_dataPoints_validity(cls, dataTimePointSerie):
    
        diffs={}
        prev_dataTimePoint=None
        for dataTimePoint in dataTimePointSerie:
            if prev_dataTimePoint is not None:
                diff = dataTimePoint.t - prev_dataTimePoint.t
                if diff not in diffs:
                    diffs[diff] = 1
                else:
                    diffs[diff] +=1
            prev_dataTimePoint = dataTimePoint
        
        # Iterate until the diffs are not too spread, then pick the maximum.
        i=0
        while almostequal(len(diffs), len(dataTimePointSerie)):
            or_diffs=diffs
            diffs={}
            for diff in or_diffs:
                diff=round(diff)
                if diff not in diffs:
                    diffs[diff] = 1
                else:
                    diffs[diff] +=1            
            
            if i > 10:
                raise Exception('Cannot automatically detect dataPoints validity')
        
        most_common_diff_total = 0
        most_common_diff = None
        for diff in diffs:
            if diffs[diff] > most_common_diff_total:
                most_common_diff_total = diffs[diff]
                most_common_diff = diff
        return(most_common_diff)


    def compute_slot(self, dataTimePointSerie, start_t, end_t, validity, timezone):

        # Compute coverage
        slot_coverage = compute_coverage(dataTimePointSerie,
                                         from_t   = start_t,
                                         to_t     = end_t,
                                         validity = validity)
        
        # Compute data
        keys = None
        avgs = {}
        for dataTimePoint in dataTimePointSerie:
            if not keys:
                keys=dataTimePoint.data.keys()
            for key in keys:
                if key not in avgs:
                    avgs[key] = 0
                avgs[key] += dataTimePoint.data[key]
        
        slot_data = {key:avgs[key]/len(dataTimePointSerie) for key in keys}

        # Create the DataTimeSlot
        dataTimeSlot = DataTimeSlot(start = TimePoint(t=start_t, tz=timezone),
                                    end   = TimePoint(t=end_t, tz=timezone),
                                    span  = self.span,
                                    data  = slot_data,
                                    coverage = slot_coverage)
        
        return dataTimeSlot



    def process(self, dataTimePointSerie, from_t=None, to_t=None, validity=None, force_close_last=False, include_extremes=False):
        ''' Start the slotting process. If start and/or end are not set, they are set automatically based on first and last points of the sereis'''

        if not isinstance(dataTimePointSerie, DataTimePointSerie):
            raise TypeError('Can process only DataTimePointSeries, got "{}"'.format(dataTimePointSerie.__class__.__name__))

        if not dataTimePointSerie:
            raise ValueError('Cannot process empty dataTimePointSerie')

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
            from_t = dataTimePointSerie[0].t
            from_dt = dt_from_s(from_t)
            # Is the point already rounded to the time span or do we have to round it ourselves?
            if not from_dt == self.timeSpan.round_dt(from_dt):
                from_dt = self.timeSpan.round_dt(from_dt, how=from_rounding_method)
                from_t  = s_from_dt(from_dt)
        else:
            from_dt = dt_from_s(from_t)
            if from_dt != self.timeSpan.round_dt(from_dt):
                raise ValueError('Sorry, provided from_t is not consistent with the timeSpan of "{}" (Got "{}")'.format(self.timeSpan, from_t))

        # Set "to" if not set, otherwise check for consistency # TODO: move to streaming
        if to_t is None:
            to_t = dataTimePointSerie[-1].t
            to_dt = dt_from_s(to_t)
            # Is the point already rounded to the time span or do we have to round it ourselves?
            if not to_dt == self.timeSpan.round_dt(to_dt):
                to_dt = self.timeSpan.round_dt(to_dt, how=to_rounding_method)
                to_t  = s_from_dt(to_dt)
        else:
            to_dt = dt_from_s(to_t)
            if to_dt != self.timeSpan.round_dt(to_dt):
                raise ValueError('Sorry, provided to_t is not consistent with the timeSpan of "{}" (Got "{}")'.format(self.timeSpan, to_t))
        
        # Automatically detect validity if not set
        if validity is None:
            validity = self._detect_dataPoints_validity(dataTimePointSerie)
            logger.info('Auto-detected dataTimePoints validity: %ss', validity)
        

        
        force_close_last = force_close_last # TODO: set to true if inverting floor and ceil above)

        logger.debug('Started slotter from "%s" (%s) to "%s" (%s)', from_dt, from_t, to_dt, to_t)

        if from_dt >= to_dt:
            raise ValueError('Sorry, from is >= to! (from_t={}, to_t={})'.format(from_t, to_t))

        # Set some support vars
        slot_start_t       = None
        slot_end_t         = None
        prev_dataTimePoint = None
        working_serie      = DataTimePointSerie()
        process_ended      = False
        dataTimeSlotSerie  = DataTimeSlotSerie()

        # Set timezone
        timezone  = dataTimePointSerie.tz
        logger.debug('Using timezone "%s"', timezone)
       
        # Counters
        count = 0

        # Now go trough all the data in the time series        
        for dataTimePoint in dataTimePointSerie:

            logger.debug('Processing %s', dataTimePoint)

            # Increase counter
            count += 1
            
            # Set start_dt if not already done TODO: implement it correctly
            #if not from_dt:
            #    from_dt = self.timeSpan.timeInterval.round_dt(dataTimePoint.dt) if rounded else dataTimePoint.dt
            
            # Pretend there was a slot before if we are at the beginning. TOOD: improve me.
            if not slot_end_t:   
                slot_end_t = from_t

            # First, check if we have some points to discard at the beginning       
            if dataTimePoint.t < from_t:
                # If we are here it means we are going data belonging to a previous slot
                # (probably just spare data loaded to have access to the prev_datapoint)  
                prev_dataTimePoint = dataTimePoint
                continue

            # Similar concept for the end
            # TODO: what if we are in streaming mode? add if to_t is not None?
            if dataTimePoint.t >= to_t:
                if process_ended:
                    continue

            # The following procedure works in general for slots at the beginning and in the middle.
            # The approach is to detect if the current slot is "outdated" and spin a new one if so.

            if dataTimePoint.t > slot_end_t:
                # If the current slot is outdated:
                         
                # 1) Add this last point to the dataTimePointSerie:
                working_serie.append(dataTimePoint)
                 
                #2) keep spinning new slots until the current data point falls in one of them.
                
                # NOTE: Read the following "while" more as an "if" which can also lead to spin multiple
                # slot if there are empty slots between the one being closed and the dataTimePoint.dt.
                # TODO: leave or remove the above if for code readability?
                
                while slot_end_t < dataTimePoint.t:
                    logger.debug('Checking for end {} with point {}'.format(slot_end_t, dataTimePoint.t))
                    # If we are in the pre-first slot, just silently spin a new slot:
                    if slot_start_t is not None:
                        
                        # Append last point. Can be appended to multiple slots, this is normal since
                        # the empty slots in the middle will have only a far prev and a far next.
                        # can also be appended several times if working_serie is not reset (looping in the while)
                        if dataTimePoint not in working_serie:
                            working_serie.append(dataTimePoint)
  
                        logger.debug('This slot (start={}, end={}) is closed, now aggregating it..'.format(slot_start_t, slot_end_t))
                        
                        logger.debug('working_serie first point dt: %s', working_serie[0].dt)
                        logger.debug('working_serie  last point dt: %s', working_serie[-1].dt)

                        
                        # Compute slot...
                        dataTimeslot = self.compute_slot(working_serie,
                                                         start_t  = slot_start_t,
                                                         end_t    = slot_end_t,
                                                         validity = validity,
                                                         timezone = timezone)
                        
                        # .. and append results 
                        dataTimeSlotSerie.append(dataTimeslot)


                    # Create a new slot. This is where all the "conventional" time logic kicks-in, and where the time zone is required.
                    slot_start_t = slot_end_t
                    slot_end_t   = s_from_dt(dt_from_s(slot_start_t, tz=timezone) + self.timeSpan)
                    
                    # Create a new working_serie as part of the "create a new slot" procedure
                    working_serie = DataTimePointSerie()
                    
                    # Append the previous prev_dataTimePoint to the new DataTimeSeries
                    if prev_dataTimePoint:
                        working_serie.append(prev_dataTimePoint)

                    logger.debug('Spinned a new slot (start={}, end={})'.format(slot_start_t, slot_end_t))
                    
                # If last slot mark process as completed (and aggregate last slot if necessary)
                if dataTimePoint.dt >= to_dt:

                    # Edge case where we would otherwise miss the last slot
                    if dataTimePoint.dt == to_dt:
                        
                        # Compute slot...
                        dataTimeslot = self.compute_slot(working_serie,
                                                         start_t  = slot_start_t,
                                                         end_t    = slot_end_t,
                                                         validity = validity,
                                                         timezone = timezone)
                        
                        # .. and append results 
                        dataTimeSlotSerie.append(dataTimeslot)

                    process_ended = True
                    

            # Append this point to the working serie
            working_serie.append(dataTimePoint)
            
            # ..and save as previous point
            prev_dataTimePoint =  dataTimePoint           


        # Last slots
        if process_ended and force_close_last: # or method = overset (vs subset that is the standard)

            # 1) Close the last slot and aggreagte it. You should never do it unless you knwo what you are doing
            if working_serie:
    
                logger.debug('This slot (start={}, end={}) is closed, now aggregating it..'.format(slot_start_t, slot_end_t))
      
                # Compute slot...
                dataTimeslot = self.compute_slot(working_serie,
                                                 start_t  = slot_start_t,
                                                 end_t    = slot_end_t,
                                                 validity = validity,
                                                 timezone = timezone)
                
                # .. and append results 
                dataTimeSlotSerie.append(dataTimeslot)

            # 2) Handle missing slots until the requested end (end_dt)
            # TODO: Implement it. Sure?

        logger.info('Slotted %s DataTimePoints in %s DataTimeSlots', count, len(dataTimeSlotSerie))

        return dataTimeSlotSerie



#======================
#  Derivative
#======================


class Derivative(Operator):
    
    @classmethod
    def apply(cls, dataTimeSlotSeries, inplace=False, incremental=False):
        
        if not inplace:
            derivative_dataTimeSlotSerie = DataTimeSlotSerie()

        data_keys = dataTimeSlotSeries.data_keys()
        for i, dataTimeSlot in enumerate(dataTimeSlotSeries):

            if not inplace:
                data = {}
            
            for key in data_keys:
                
                # Compute the derivative
                if i == 0:
                    # Right derivative for the first item
                    der = dataTimeSlotSeries[i+1].data[key] - dataTimeSlotSeries[i].data[key]
                    if incremental:
                        der = der/2       
                elif i == len(dataTimeSlotSeries)-1:
                    # Left derivative for the last item
                    der = dataTimeSlotSeries[i].data[key] - dataTimeSlotSeries[i-1].data[key]
                    if incremental:
                        der = der/2
                else:
                    # Both left and right derivative for the items in the middle
                    der =  ((dataTimeSlotSeries[i+1].data[key] - dataTimeSlotSeries[i].data[key]) + (dataTimeSlotSeries[i].data[key] - dataTimeSlotSeries[i-1].data[key])) /2
                
                # Add data
                if not inplace:
                    data['{}_der'.format(key)] = der
                else:
                    dataTimeSlot.data['{}_der'.format(key)] = der
            
            # Create the slot
            if not inplace:       
                derivative_dataTimeSlotSerie.append(DataTimeSlot(start = dataTimeSlot.start,
                                                                 end   = dataTimeSlot.end,
                                                                 data  = data))

        if not inplace:
            return derivative_dataTimeSlotSerie

















