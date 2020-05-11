import os
import chardet
from chardet.universaldetector import UniversalDetector
from numpy import fft
from scipy.signal import find_peaks
from .time import dt_from_s
from .exceptions import InputException

# Setup logging
import logging
logger = logging.getLogger(__name__)

HARD_DEBUG = False


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




def compute_coverage(dataTimePointSerie, from_t, to_t, trustme=False, validity=None, validity_placement='center'):
    '''Compute the data coverage of a dataTimePointSerie based on the dataTimePoints validity'''
    
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
        if dataTimePointSerie is None:
            raise InputException('You must provide dataTimePointSerie, got None')
            
        if from_t is None or to_t is None:
            raise ValueError('Missing from_t or to_t')


    # Support vars
    prev_dataPoint_valid_to_t = None
    empty_dataTimePointSerie = True
    missing_coverage = None
    next_processed = False

    logger.debug('Called compute_coverage from {} to {}'.format(from_t, to_t))


    #===========================
    #  START cycle over points
    #===========================
    
    for this_dataTimePoint in dataTimePointSerie:
        
        
        # Compute this_dataTimePoint validity boundaries
        if validity:
            if validity_placement==center:
                this_dataTimePoint_valid_from_t = this_dataTimePoint.t - (validity/2)
                this_dataTimePoint_valid_to_t   = this_dataTimePoint.t + (validity/2)
            else:
                raise NotImplementedError('Validity placements other than "center" are not yet supported')
        
        else:
            this_dataTimePoint_valid_from_t = this_dataTimePoint.t
            this_dataTimePoint_valid_to_t   = this_dataTimePoint.t
        
        # Hard debug
        #logger.debug('HARD DEBUG %s %s %s', this_dataTimePoint.Point_part, this_dataTimePoint.validity_region.start, this_dataTimePoint.validity_region.end)
        
        # If no start point has been set, just use the first one in the data
        #if start_Point is None:
        #    start_Point = dataTimePointSerie.Point_part
        # TODO: add support also for dynamically setting the end_Point to allow empty start_Point/end_Point input        
        
        #=====================
        #  BEFORE START
        #=====================
        
        # Are we before the start_Point? 
        if this_dataTimePoint.t < from_t:
            
            # Just set the previous Point valid until
            prev_dataPoint_valid_to_t = this_dataTimePoint_valid_to_t

            # If prev point too far, skip it
            if prev_dataPoint_valid_to_t <= from_t:
                prev_dataPoint_valid_to_t = None

            continue


        #=====================
        #  After end
        #=====================
        # Are we after the end_Point? In this case, we can treat it as if we are in the middle-
        elif this_dataTimePoint.t >= to_t:

            if not next_processed: 
                next_processed = True
                
                # If "next" point too far, skip it:
                if this_dataTimePoint_valid_from_t > to_t:
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
        # 1) prev_dataPoint_valid_until
        # 2) this_dataTimePoint_valid_from
        
        # Also, if we are here it also means that we have valid data
        if empty_dataTimePointSerie:
            empty_dataTimePointSerie = False

        # Compute coverage
        # TODO: and idea could also to initialize Spans and sum them
        if prev_dataPoint_valid_to_t is None:
            value = this_dataTimePoint_valid_from_t - from_t
            
        else:
            value = this_dataTimePoint_valid_from_t - prev_dataPoint_valid_to_t
            
        # If for whatever reason the validity regions overlap we don't want to end up in
        # invalidating the coverage calculation by summing negative numbers
        if value > 0:
            if missing_coverage is None:
                missing_coverage = value
            else:
                missing_coverage = missing_coverage + value

        # Update previous dataPoint Validity:
        prev_dataPoint_valid_to_t = this_dataTimePoint_valid_to_t
        
    #=========================
    #  END cycle over points
    #=========================

    # Compute the coverage until the end point
    if prev_dataPoint_valid_to_t is not None:
        if to_t > prev_dataPoint_valid_to_t:
            if missing_coverage is not None:
                missing_coverage += (to_t - prev_dataPoint_valid_to_t)
            else:
                missing_coverage = (to_t - prev_dataPoint_valid_to_t)
    
    # Convert missing_coverage_s_is in percentage
        
    if empty_dataTimePointSerie:
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
# Periodicity
#==============================

def get_periodicity(dataTimeSlotSerie):
    
    # Import here or you will end up with cyclic imports
    from .datastructures import DataTimeSlotSerie
    
    if not isinstance(dataTimeSlotSerie, DataTimeSlotSerie):
        raise TypeError('DataTimeSlotSerie is required (got"{}")'.format(dataTimeSlotSerie.__class__.__name__))

    if not dataTimeSlotSerie:
        raise ValueError('A non-empty DataTimeSlotSerie is required')
        
    # TODO: fix me, data_loss must not belong as key
    data_keys = dataTimeSlotSerie.data_keys()
    
    if len(data_keys) > 1:
        raise NotImplementedError()

    for key in data_keys:
        
        # Get data as a vector
        y = [item.data[key] for item in dataTimeSlotSerie]

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
        max_peak_frequency = None
        for i, peak in enumerate(peaks):
            peak_frequency = (len(y) / peak[0])
            if not max_peak_frequency:
                max_peak_frequency = peak_frequency
            logger.debug('Peak index \t#%s,\t value=%s, freq=%s', peak[0], int(peak[1]), peak_frequency)
            if i>10:
                break
        
        # Round max peak and return
        return int(round(max_peak_frequency))
    
    
    







 