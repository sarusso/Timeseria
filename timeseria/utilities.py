import os
import chardet
from chardet.universaldetector import UniversalDetector
from numpy import fft
from scipy.signal import find_peaks
from .time import dt_from_s
from .exceptions import InputException, ConsistencyException
from datetime import datetime
from .time import s_from_dt

# Setup logging
import logging
logger = logging.getLogger(__name__)

HARD_DEBUG = False


def is_numerical(item):
    if isinstance(item, float):
        return True
    if isinstance(item, int):
        return True
    try:
        # Handle pandas-like data types (i.e. int64)
        item + 1
        return True
    except:
        return False


def set_from_t_and_to_t(from_dt, to_dt, from_t, to_t):
    
    # Sanity chceks
    if from_t is not None and not is_numerical(from_t):
        raise Exception('from_t is not numerical')        
    if to_t is not None and not is_numerical(to_t):
        raise Exception('to_t is not numerical')
    if from_dt is not None and not isinstance(from_dt, datetime):
        raise Exception('from_dt is not datetime')    
    if to_dt is not None and not isinstance(to_dt, datetime):
        raise Exception('to_t is not datetime')

    if from_dt is not None:
        if from_t is not None:
            raise Exception('Got both from_t and from_dt, pick one')
        from_t = s_from_dt(from_dt)
    if to_dt is not None:
        if to_t is not None:
            raise Exception('Got both to_t and to_dt, pick one')
        to_t = s_from_dt(to_dt)
    return from_t, to_t


def item_is_in_range(item, from_t, to_t):
    # TODO: maybe use a custom iterator for looping over time series items? 
    # see https://stackoverflow.com/questions/6920206/sending-stopiteration-to-for-loop-from-outside-of-the-iterator
    from .datastructures import Point, Slot
    if isinstance(item, Slot):
        if from_t is not None and to_t is not None and from_t > to_t:
            if item.end.t <= to_t:
                return True
            if item.start.t >= from_t:
                return True
        else:     
            if from_t is not None and item.start.t < from_t:
                return False
            if to_t is not None and item.end.t > to_t:
                raise StopIteration
            return True
    elif isinstance(item, Point):
        if from_t is not None and to_t is not None and from_t > to_t:
            if item.t <= to_t:
                return True
            if item.t >= from_t:
                return True
        else:     
            if from_t is not None and item.t < from_t:
                return False
            if to_t is not None and item.t > to_t:
                raise StopIteration
            return True    
    else:
        raise ConsistencyException('Got unknown type "{}'.format(item.__class__.__name__))
        


def detect_encoding(filename, streaming=False):
    
    if streaming:
        detector = UniversalDetector()
        with open(filename, 'rb') as file_pointer:      
            for i, line in enumerate(file_pointer.readlines()):
                if HARD_DEBUG: logger.debug('Itearation #%s: confidence=%s',i,detector.result['confidence'])
                detector.feed(line)
                if detector.done:  
                    if HARD_DEBUG: logger.debug('Detected encoding at line "%s"', i)
                    break
        detector.close() 
        chardet_results = detector.result

    else:
        with open(filename, 'rb') as file_pointer:
            chardet_results = chardet.detect(file_pointer.read())
             
    logger.debug('Detected encoding "%s" with "%s" confidence (streaming=%s)', chardet_results['encoding'],chardet_results['confidence'], streaming)
    encoding = chardet_results['encoding']
     
    return encoding


def compute_coverage(data_time_point_series, from_t, to_t, trustme=False, validity=None, validity_placement='center'):
    '''Compute the data coverage of a data_time_point_series based on the data_time_points validity'''
    
    # TODO: The following should be implemented when computing averages as well.. put it in common?
    center = 1
    left   = 2
    right  = 3
    
    if validity_placement == 'center':
        validity_placement=center
    elif validity_placement == 'left':
        validity_placement=left
    elif validity_placement == 'right':
        validity_placement=right
    else:
        raise ValueError('Unknown value "{}" for validity_placement'.format(validity_placement))
    
    # Sanity checks
    if not trustme:
        if data_time_point_series is None:
            raise InputException('You must provide data_time_point_series, got None')
            
        if from_t is None or to_t is None:
            raise ValueError('Missing from_t or to_t')


    # Support vars
    prev_datapoint_valid_to_t = None
    empty_data_time_point_series = True
    missing_coverage = None
    next_processed = False

    logger.debug('Called compute_coverage from {} to {}'.format(from_t, to_t))


    #===========================
    #  START cycle over points
    #===========================
    
    for this_data_time_point in data_time_point_series:
        
        
        # Compute this_data_time_point validity boundaries
        if validity:
            if validity_placement==center:
                this_data_time_point_valid_from_t = this_data_time_point.t - (validity/2)
                this_data_time_point_valid_to_t   = this_data_time_point.t + (validity/2)
            else:
                raise NotImplementedError('Validity placements other than "center" are not yet supported')
        
        else:
            this_data_time_point_valid_from_t = this_data_time_point.t
            this_data_time_point_valid_to_t   = this_data_time_point.t
        
        # Hard debug
        #logger.debug('HARD DEBUG %s %s %s', this_data_time_point.Point_part, this_data_time_point.validity_region.start, this_data_time_point.validity_region.end)
        
        # If no start point has been set, just use the first one in the data
        #if start_Point is None:
        #    start_Point = data_time_point_series.Point_part
        # TODO: add support also for dynamically setting the end_Point to allow empty start_Point/end_Point input        
        
        #=====================
        #  BEFORE START
        #=====================
        
        # Are we before the start_Point? 
        if this_data_time_point.t < from_t:
            
            # Just set the previous Point valid until
            prev_datapoint_valid_to_t = this_data_time_point_valid_to_t

            # If prev point too far, skip it
            if prev_datapoint_valid_to_t <= from_t:
                prev_datapoint_valid_to_t = None

            continue


        #=====================
        #  After end
        #=====================
        # Are we after the end_Point? In this case, we can treat it as if we are in the middle-
        elif this_data_time_point.t >= to_t:

            if not next_processed: 
                next_processed = True
                
                # If "next" point too far, skip it:
                if this_data_time_point_valid_from_t > to_t:
                    continue
            else:
                continue


        #=====================
        #  In the middle
        #=====================
        
        # Otherwise, we are in the middle?
        else:
            # Normal operation mode
            pass

        

        # Okay, now we have all the values we need:
        # 1) prev_datapoint_valid_until
        # 2) this_data_time_point_valid_from
        
        # Also, if we are here it also means that we have valid data
        if empty_data_time_point_series:
            empty_data_time_point_series = False

        # Compute coverage
        # TODO: and idea could also to initialize Units and sum them
        if prev_datapoint_valid_to_t is None:
            value = this_data_time_point_valid_from_t - from_t
            
        else:
            value = this_data_time_point_valid_from_t - prev_datapoint_valid_to_t
            
        # If for whatever reason the validity regions overlap we don't want to end up in
        # invalidating the coverage calculation by summing negative numbers
        if value > 0:
            if missing_coverage is None:
                missing_coverage = value
            else:
                missing_coverage = missing_coverage + value

        # Update previous datapoint Validity:
        prev_datapoint_valid_to_t = this_data_time_point_valid_to_t
        
    #=========================
    #  END cycle over points
    #=========================

    # Compute the coverage until the end point
    if prev_datapoint_valid_to_t is not None:
        if to_t > prev_datapoint_valid_to_t:
            if missing_coverage is not None:
                missing_coverage += (to_t - prev_datapoint_valid_to_t)
            else:
                missing_coverage = (to_t - prev_datapoint_valid_to_t)
    
    # Convert missing_coverage_s_is in percentage
        
    if empty_data_time_point_series:
        coverage = 0.0 # Return zero coverage if empty
    
    elif missing_coverage is not None :
        coverage = 1.0 - float(missing_coverage) / ( to_t - from_t) 
        
        # Fix boundaries # TODO: understand better this part
        if coverage < 0:
            coverage = 0.0
            #raise ConsistencyException('Got Negative coverage!! {}'.format(coverage))
        if coverage > 1:
            coverage = 1.0
            #raise ConsistencyException('Got >1 coverage!! {}'.format(coverage))
    
    else:
        coverage = 1.0
        
    # Return
    logger.debug('compute_coverage: Returning %s (%s percent)', coverage, coverage*100.0)
    return coverage


def compute_data_loss(data_time_point_series, from_t, to_t, series_resolution, validity,
                      validity_placement='center', first_last=False, trustme=False):
    
    # Data loss from missing coverage. Computing it useless if the series has no 'variable' resolution, 
    # however, if removed, it is still to be applied for the first and last item as on the borders there
    # still may be  data losses. 
    if series_resolution == 'variable' or first_last:
        data_loss_from_missing_coverage = 1 - compute_coverage(data_time_point_series, from_t, to_t, trustme, validity, validity_placement)
    else:
        data_loss_from_missing_coverage = 0

    # Data loss from previously computed data losses
    data_loss_from_previously_computed = 0
    for this_data_time_point in data_time_point_series:

        if this_data_time_point.data_loss:

            # Skip points not to be taken dinto account
            if this_data_time_point.t + (validity/2) < from_t:
                continue
            if this_data_time_point.t - (validity/2) >= to_t:
                continue
            
            # Compute the contribution of this point data loss.
            if (this_data_time_point.t < from_t)  and this_data_time_point.t + (validity/2) >= from_t:
                this_validity = (this_data_time_point.t + (validity/2)) - from_t
            elif (this_data_time_point.t > to_t)  and this_data_time_point.t + (validity/2) < to_t:
                this_validity = to_t - (this_data_time_point.t + (validity/2))
            else:
                this_validity = validity
            
            # Now add, rescaling the data loss with respect to the validity and from/to 
            data_loss_from_previously_computed += this_data_time_point.data_loss * (this_validity/( to_t - from_t))

    # Compute total data loss    
    data_loss = data_loss_from_missing_coverage + data_loss_from_previously_computed

    # The next step is controversial, as it will cause to abuse the "None" data losses that 
    # are only used for the forecasts at the moment. TODO: what do we want to do here?
    #if series_resolution != 'variable' and not data_loss:
    #    data_loss = None

    # Return
    return data_loss


#==============================
# Floating point comparisons
#==============================
def is_close(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def is_almost_equal(one, two):
    if 0.95 < (one / two) <= 1.05:
        return True
    else:
        return False 


#==============================
# Check timeseries
#==============================

def check_timeseries(timeseries, resolution=None):
    # Import here or you will end up with cyclic imports
    from .datastructures import DataTimePointSeries, DataTimeSlotSeries 
    if isinstance(timeseries, DataTimePointSeries):
        if timeseries._resolution == 'variable':
            raise ValueError('Variable resolutions are not supported. Resample or slot the time series first.')
    elif isinstance(timeseries, DataTimeSlotSeries):
        pass
    else:
        raise TypeError('Either a DataTimePointSeries or a DataTimeSlotSeries is required (got "{}")'.format(timeseries.__class__.__name__))

    if not timeseries:
        raise ValueError('A non-empty time series is required')

def check_resolution(timeseries, resolution):
    
    def _check_resolution(timeseries, resolution):
        # TODO: Fix this mess.. Make the .resolution behavior consistent!
        if resolution == timeseries.resolution:
            return True
        try:
            if resolution.value == timeseries.resolution.duration_s():
                return True
        except:
            pass
        try:
            if resolution.duration_s() == timeseries.resolution.value:
                return True
        except:
            pass
        try:
            if resolution.duration_s() == timeseries.resolution:
                return True
        except:
            pass
        return False
            
    # Check timeseries resolution
    if not _check_resolution(timeseries, resolution):
        raise ValueError('This model is fitted on "{}" resolution data, while your data has "{}" resolution.'.format(resolution, timeseries.resolution))


def check_data_keys(timeseries, keys):
    timeseries_data_keys = timeseries.data_keys()
    if len(timeseries_data_keys) != len(keys):
        raise ValueError('This model is fitted on {} data keys, while your data has {} data keys.'.format(len(keys), len(timeseries_data_keys)))
    if timeseries_data_keys != keys:
        # TODO: logger.warning?
        raise ValueError('This model is fitted on "{}" data keys, while your data has "{}" data keys.'.format(keys, timeseries_data_keys))


#==============================
# Periodicity
#==============================

def get_periodicity(timeseries):
    
    check_timeseries(timeseries)
    
    # TODO: fix me, data_loss must not belong as key
    data_keys = timeseries.data_keys()
    
    if len(data_keys) > 1:
        raise NotImplementedError()

    # TODO: improve me, highly ineficcient
    for key in data_keys:
        
        # Get data as a vector
        y = []
        for item in timeseries:
            y.append(item.data[key])
        #y = [item.data[key] for item in timeseries]

        # Compute FFT (Fast Fourier Transform)
        yf = fft.fft(y)

        # Remove specular data        
        len_yf = len(yf)
        middle_point=round(len_yf/2)
        yf = yf[0:middle_point]
        
        # To absolute values
        yf = [abs(f) for f in yf]
            
        # Find FFT peaks
        peak_indexes, _ = find_peaks(yf, height=None)
        peaks = []
        for i in peak_indexes:
            peaks.append([i, yf[i]])
        
        # Sort by peaks intensity and compute actual frequency in base units
        # TODO: round peak frequencies to integers and/or neighbours first
        peaks = sorted(peaks, key=lambda t: t[1])
        peaks.reverse()
        
        # Compute peak frequencies:
        for i in range(len(peaks)):
            
            # Set peak frequency
            peak_frequency = (len(y) / peaks[i][0])
            peaks[i].append(peak_frequency)
        
        # Find most relevant frequency
        max_peak_frequency = None
        for i in range(len(peaks)):

            logger.debug('Peak #%s: \t index=%s,\t value=%s, freq=%s (over %s)', i, peaks[i][0], int(peaks[i][1]), peaks[i][2], len(timeseries))

            # Do not consider lower frequencies if there is a closer and higher one
            try:
                diff1=peaks[i][1]-peaks[i+1][1]
                diff2=peaks[i+1][1]-peaks[i+2][1]
                if diff1 *3 < diff2:
                    logger.debug('Peak #{} candidate to removal'.format(i))
                    if (peaks[i][2] > peaks[i+1][2]*10) and (peaks[i][2] > len(timeseries)/10):
                        logger.debug('peak #{} marked to be removed'.format(i))
                        continue
            except IndexError:
                pass
            
            if not max_peak_frequency:
                max_peak_frequency = peaks[i][2]
            if i>10:
                break
        
        # Round max peak and return
        return int(round(max_peak_frequency))
    
    
    







