# -*- coding: utf-8 -*-
"""Utility functions."""

import os
import re
import chardet
from chardet.universaldetector import UniversalDetector
from numpy import fft
from scipy.signal import find_peaks
from .exceptions import ConsistencyException
from datetime import datetime
from .time import s_from_dt
from .exceptions import FloatConversionError
import subprocess
from collections import namedtuple

# Setup logging
import logging
logger = logging.getLogger(__name__)

HARD_DEBUG = False


def is_numerical(item):
    """Check if item is numerical (float or int)"""
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


def compute_validity_regions(series, from_t=None, to_t=None, sampling_interval=None, cut=False):
    """
    Compute the validity regions for the series points. If from_t or to_t are given, computes them within that interval only.
    
    Args:
        series: the time series.  
        from_t: the interval start, if any.
        to_t: the interval end, if any.
        sampling_interval: the sampling interval. In not set, it will be used the series' (auto-detected) one.
        cut: if to shrink the validity according to the start/end.

    Returns:
        dict: for each point, the validity region start and end.
    """

    # If single-element series, check that the sampling_interval was given
    if len(series) == 1:
        if sampling_interval is None:
            raise ValueError('The series has only one element and no sampling_interval is provided, no idea how to compute validity') 

    # Get the series sampling interval unless it is forced to a specific value
    if not sampling_interval:
        sampling_interval = series.autodetected_sampling_interval

    # Segments dict
    validity_segments = {}
    
    # Compute validity segments for each point
    prev_point_t = None
    for point in series:

        # Get point validity boundaries
        point_valid_from_t = point.t - (sampling_interval/2)
        point_valid_to_t   = point.t + (sampling_interval/2)

        # Are we processing points not belonging to this interval, if set?
        if from_t is not None and point_valid_to_t <= from_t:
            continue
        if to_t is not None and point_valid_from_t > to_t:
            break
        
        # Shrink with respect to start-end if required
        if cut:
            if from_t is not None and point_valid_from_t < from_t:
                point_valid_from_t = from_t
            if to_t is not None and point_valid_to_t > to_t:
                point_valid_to_t = to_t
    
        # Shrink if overlaps: if the previous point validity overlap with ours, resize both
        if prev_point_t is not None:
            if point_valid_from_t < validity_segments[prev_point_t][1]:
                # Shrink both
                mid_t = (validity_segments[prev_point_t][1] + point_valid_from_t)/2                
                validity_segments[prev_point_t][1] = mid_t
                point_valid_from_t = mid_t
    
        # Set this validity segment boundaries
        validity_segments[point.t] = [point_valid_from_t,point_valid_to_t]

        # Set previous point timestamp
        prev_point_t = point.t

    return validity_segments


def compute_coverage(series, from_t, to_t, sampling_interval=None):
    """
    Compute the coverage of an interval based on the points validity regions.
    
    Args:
        series: the time series.
        from_t: the interval start.
        to_t: the interval end.
        sampling_interval: the sampling interval. In not set, it will be used the series' (auto-detected) one.
    
    Returns:
        float: the interval coverage.
    """

    logger.debug('Called compute_coverage() from {} to {}'.format(from_t, to_t))

    # Check from_t/to_t
    if from_t >= to_t:
        raise ValueError('From is higher than to (or equal): from_t="{}", to_t="{}"'.format(from_t, to_t))
    
    # If the series is empty, return zero coverage
    if not series:
        return 0.0
        
    # Compute the validity regions in the interval. if point haven't them?
    # validity_regions = compute_validity_regions(series, from_t=from_t, to_t=to_t, cut=True, sampling_interval=sampling_interval)
    # And now attach?

    # Sum all the validity regions according to the from/to:
    coverage_s = 0
    for point in series:
        try:
            # Skip at the beginning and end if any
            if point.valid_to < from_t:
                continue
            if point.valid_from >= to_t:
                break 
            
            # Skip if reconstructed as well
            try:
                if point.reconstructed:
                    continue
            except AttributeError:
                pass
            
            # Set this point valid from/to accoring to the interval boundaries
            this_point_valid_from = point.valid_from if point.valid_from >= from_t else from_t
            this_point_valid_to = point.valid_to if point.valid_to < to_t else to_t
            
            # Add this point coverage to the overall coverage
            coverage_s += this_point_valid_to-this_point_valid_from
            
            logger.debug('Point %s: from %s, to %s (current total=%s) ', point.t, this_point_valid_from, this_point_valid_to, coverage_s)
            
        except AttributeError:
            raise AttributeError('Point {} has no valid_to or valid_from'.format(point)) from None

    # Convert from coverage in seconds to ratio:
    coverage = coverage_s / (to_t - from_t)

    # Return
    logger.debug('compute_coverage: Returning %s (%s percent)', coverage, coverage*100.0)
    return coverage


def compute_data_loss(series, from_t, to_t, force=False, sampling_interval=None):
    """
    Compute the data loss  of an interval based on the points validity regions.
    
    Args:
        series: the time series.
        from_t: the interval start.
        to_t: the interval end.
        force: if to force computing even when not strictly necessary.
        sampling_interval: the sampling interval. In not set, it will be used the series' (auto-detected) one.
    
    Returns:
        float: the interval data loss.
    """

    logger.debug('Called compute_data_loss() from {} to {}'.format(from_t, to_t))


    # Get the series sampling interval unless it is forced to a specific value
    if not sampling_interval:
        sampling_interval = series.autodetected_sampling_interval
       
    #========================================
    #  Data loss from missing coverage.
    #========================================
    
    # Note: to improve performance, this could be computed only if the series is "raw" or forced,
    # i.e. at first and last items since on the borders there still may be  data losses. 
    data_loss_from_missing_coverage = 1 - compute_coverage(series, from_t, to_t, sampling_interval=sampling_interval)

    logger.debug('Data loss from missing coverage: %s', data_loss_from_missing_coverage)
    
    #========================================
    #  Data loss from previous data losses.
    #========================================

    # Computed only if the resolutions is constant, as they can be defined only if the series was already 
    # made uniform (i.e. by a resampling process), or if forced for testing or other potential reasons.
    data_loss_from_previously_computed = 0.0

    if (series.resolution is not None and not series.resolution.is_variable()) or force:

        for point in series:
            
            if point.data_loss:

                # Set validity
                point_valid_from_t = point.t - (sampling_interval/2)
                point_valid_to_t = point.t + (sampling_interval/2)
                
                # Skip points which have not to be taken into account
                if point_valid_to_t < from_t:
                    continue
                if point_valid_from_t >= to_t:
                    continue
                
                # Compute the contribution of this point data loss: overlapping
                # first point, overlapping last point, or all included?
                if point_valid_from_t < from_t:
                    point_contribution = abs(from_t - point_valid_to_t)
                elif point_valid_to_t >= to_t:
                    point_contribution = abs(to_t - point_valid_from_t)
                else:
                    point_contribution = sampling_interval
                
                #logger.debug('----------------')
                #count+=1
                #logger.debug('count: %s', count)
                #logger.debug('point_valid_from_t: %s', point_valid_from_t)
                #logger.debug('point.t: %s', point.t)
                #logger.debug('point_valid_to_t: %s', point_valid_to_t)
                #logger.debug('point_contribution: %s', point_contribution)                

                # Now rescale the data loss with respect to the contribution and from/to and sum it up
                data_loss_from_previously_computed += point.data_loss * (point_contribution/( to_t - from_t))
           
    logger.debug('Data loss from previously computed: %s', data_loss_from_previously_computed)

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
        if timeseries.resolution.is_variable():
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
            if resolution.value == timeseries.resolution.as_seconds():
                return True
        except:
            pass
        try:
            if resolution.as_seconds() == timeseries.resolution.value:
                return True
        except:
            pass
        try:
            if resolution.as_seconds() == timeseries.resolution:
                return True
        except:
            pass
        return False
            
    # Check timeseries resolution
    if not _check_resolution(timeseries, resolution):
        raise ValueError('This model is fitted on "{}" resolution data, while your data has "{}" resolution.'.format(resolution, timeseries.resolution))


def check_data_labels(timeseries, keys):
    timeseries_data_labels = timeseries.data_labels()
    if len(timeseries_data_labels) != len(keys):
        raise ValueError('This model is fitted on {} data keys, while your data has {} data keys.'.format(len(keys), len(timeseries_data_labels)))
    if timeseries_data_labels != keys:
        # TODO: logger.warning?
        raise ValueError('This model is fitted on "{}" data keys, while your data has "{}" data keys.'.format(keys, timeseries_data_labels))


#==============================
# Periodicity
#==============================

def get_periodicity(timeseries):
    
    check_timeseries(timeseries)
    
    # TODO: fix me, data_loss must not belong as key
    data_labels = timeseries.data_labels()
    
    if len(data_labels) > 1:
        raise NotImplementedError()

    # TODO: improve me, highly ineficcient
    for key in data_labels:
        
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

def mean_absolute_percentage_error(list1, list2):
    '''Computes the MAPE, list 1 are true values, list2 are predicted values'''
    if len(list1) != len(list2):
        raise ValueError('Lists have different lengths, cannot continue')
    p_error_sum = 0
    for i in range(len(list1)):
        p_error_sum += abs((list1[i] - list2[i])/list1[i])
    return p_error_sum/len(list1)


def get_periodicity_index(item, resolution, periodicity, dst_affected=False):
    from .units import Unit, TimeUnit
    # Handle specific cases
    if isinstance(resolution, TimeUnit):  
        resolution_s = resolution.as_seconds(item.dt)
    elif isinstance(resolution, Unit):  
        if isinstance(resolution.value, list):
            raise NotImplementedError('Sorry, periodocity in multi-dimensional spaces are not defined')
        resolution_s = resolution.value
    else:
        if isinstance(resolution, list):
            raise NotImplementedError('Sorry, periodocity in multi-dimensional spaces are not defined')
        resolution_s = resolution

    # Compute periodicity index
    if not dst_affected:
    
        # Get index based on item timestamp, normalized to unit, modulus periodicity
        periodicity_index =  int(item.t / resolution_s) % periodicity
    
    else:

        # Get periodicity based on the datetime
        
        # Do we have an active DST?  
        dst_timedelta = item.dt.dst()
        
        if dst_timedelta.days == 0 and dst_timedelta.seconds == 0:
            # No DST
            periodicity_index = int(item.t / resolution_s) % periodicity
        
        else:
            # DST
            if dst_timedelta.days != 0:
                raise Exception('Don\'t know how to handle DST with days timedelta = "{}"'.format(dst_timedelta.days))

            if resolution_s > 3600:
                raise Exception('Sorry, this time series has not enough resolution to account for DST effects (resolution_s="{}", must be below 3600 seconds)'.format(resolution_s))
            
            # Get DST offset in seconds 
            dst_offset_s = dst_timedelta.seconds # 3600 usually

            # Compute the periodicity index
            periodicity_index = (int((item.t + dst_offset_s) / resolution_s) % periodicity)

    return periodicity_index
    
    
#==============================
# Detetc sampling interval
#==============================

def detect_sampling_interval(data_time_point_series):

    diffs={}
    prev_data_time_point=None
    for data_time_point in data_time_point_series:
        if prev_data_time_point is not None:
            diff = data_time_point.t - prev_data_time_point.t
            if diff not in diffs:
                diffs[diff] = 1
            else:
                diffs[diff] +=1
        prev_data_time_point = data_time_point
    
    # Iterate until the diffs are not too spread, then pick the maximum.
    i=0
    while is_almost_equal(len(diffs), len(data_time_point_series)):
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



#==============================
# Storage utilities
#==============================

def sanitize_string(string, no_data_placeholders=[]):
    string = re.sub('\s+',' ',string).strip()
    if string.startswith('\'') or string.startswith('"'):
        string = string[1:]
    if string.endswith('\'') or string.endswith('"'):
        string = string[:-1]
    string = string.strip()
    if string.lower().replace('.','') in no_data_placeholders:
        return None
    return string


def is_list_of_integers(list):
    for item in list:
        if not isinstance(item, int):
            return False
    else:
        return True

def to_float(string,no_data_placeholders=[],label=None):
    sanitized_string_string = sanitize_string(string,no_data_placeholders)
    if sanitized_string_string:
        sanitized_string_string = sanitized_string_string.replace(',','.')
    try:
        return float(sanitized_string_string)
    except (ValueError, TypeError):
        # Do not raise inf converting indexes as they are allowed to be "None"
        if label and label.startswith('__'):
            return None
        raise FloatConversionError(sanitized_string_string)



def to_time_unit_string(seconds, friendlier=True):
    """Converts seconds to a (friendlier) time unit string, as 1h, 10m etc.)"""    
    seconds_str = str(seconds).replace('.0', '')
    if seconds_str == '60':
        seconds_str = '1m'
    elif seconds_str == '600':
        seconds_str = '10m'
    elif seconds_str == '3600':
        seconds_str = '1h'
    else:
        seconds_str = seconds_str+'s'
    return seconds_str


def sanitize_shell_encoding(text):
    return text.encode("utf-8", errors="ignore")


def format_shell_error(stdout, stderr, exit_code):
    
    string  = '\n#---------------------------------'
    string += '\n# Shell exited with exit code {}'.format(exit_code)
    string += '\n#---------------------------------\n'
    string += '\nStandard output: "'
    string += sanitize_shell_encoding(stdout)
    string += '"\n\nStandard error: "'
    string += sanitize_shell_encoding(stderr) +'"\n\n'
    string += '#---------------------------------\n'
    string += '# End Shell output\n'
    string += '#---------------------------------\n'

    return string


def os_shell(command, capture=False, verbose=False, interactive=False, silent=False):
    '''Execute a command in the OS shell. By default prints everything. If the capture switch is set,
    then it returns a namedtuple with stdout, stderr, and exit code.'''
    
    if capture and verbose:
        raise Exception('You cannot ask at the same time for capture and verbose, sorry')

    # Log command
    logger.debug('Shell executing command: "%s"', command)

    # Execute command in interactive mode    
    if verbose or interactive:
        exit_code = subprocess.call(command, shell=True)
        if exit_code == 0:
            return True
        else:
            return False

    # Execute command getting stdout and stderr
    # http://www.saltycrane.com/blog/2008/09/how-get-stdout-and-stderr-using-python-subprocess-module/
    
    process          = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    (stdout, stderr) = process.communicate()
    exit_code        = process.wait()

    # Convert to str (Python 3)
    stdout = stdout.decode(encoding='UTF-8')
    stderr = stderr.decode(encoding='UTF-8')

    # Formatting..
    stdout = stdout[:-1] if (stdout and stdout[-1] == '\n') else stdout
    stderr = stderr[:-1] if (stderr and stderr[-1] == '\n') else stderr

    # Output namedtuple
    Output = namedtuple('Output', 'stdout stderr exit_code')

    if exit_code != 0:
        if capture:
            return Output(stdout, stderr, exit_code)
        else:
            print(format_shell_error(stdout, stderr, exit_code))      
            return False    
    else:
        if capture:
            return Output(stdout, stderr, exit_code)
        elif not silent:
            # Just print stdout and stderr cleanly
            print(stdout)
            print(stderr)
            return True
        else:
            return True





    
    
    
