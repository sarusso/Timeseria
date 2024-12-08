# -*- coding: utf-8 -*-
"""Utility functions."""

import re
import random
import numpy
import chardet
from math import log
from chardet.universaldetector import UniversalDetector
from numpy import fft
from scipy.signal import find_peaks
from .exceptions import ConsistencyException, FloatConversionError
from datetime import datetime
from propertime.utils import s_from_dt
import subprocess
from collections import namedtuple
import numpy as np
from scipy import stats, optimize
import scipy.stats
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as sklearn_mean_squared_error
from sklearn.metrics import mean_absolute_error as sklearn_mean_absolute_error

# Setup logging
import logging
logger = logging.getLogger(__name__)


def ensure_reproducibility():
    """Ensure reproducibility by fixing seeds to zero for Random, Numpy, and Tensorflow."""

    random.seed(0)
    numpy.random.seed(0)
    try:
        import tensorflow
    except ImportError:
        pass
    else:
        # Ensure reproducibility for Tensorflow as well
        tensorflow.random.set_seed(0)


def is_numerical(item):
    """Check if an item is numerical (float or int, including Pandas data types).

        Args:
            item(obj): the item to check.
    """
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


def detect_encoding(file_name, streaming=False):
    """Detect the encoding of a file.

        Args:
            file_name(str): the file name for which to detect the encoding.
            straming(bool): if to perform the detection in streaming mode, for large files. Default to False.

        Returns:
            str: the detected encoding.
    """
    if streaming:
        detector = UniversalDetector()
        with open(file_name, 'rb') as file_pointer:
            for i, line in enumerate(file_pointer.readlines()):
                logger.debug('Itearation #%s: confidence=%s',i,detector.result['confidence'])
                detector.feed(line)
                if detector.done:
                    logger.debug('Detected encoding at line "%s"', i)
                    break
        detector.close()
        chardet_results = detector.result

    else:
        with open(file_name, 'rb') as file_pointer:
            chardet_results = chardet.detect(file_pointer.read())

    logger.debug('Detected encoding "%s" with "%s" confidence (streaming=%s)', chardet_results['encoding'],chardet_results['confidence'], streaming)
    encoding = chardet_results['encoding']

    return encoding


def detect_sampling_interval(timeseries, confidence=False):
    """Detect the sampling interval of a time series.

        Args:
            timeseries(TimeSeries): the time series for which to detect the sampling interval.
            confidence(bool): if to provide the confidence as well, in a 0-1 range.

        Returns:
            float or tuple: the detected sampling rate or the detected sampling rate with the confidence.
    """

    diffs={}
    prev_point=None
    for point in timeseries:
        if prev_point is not None:
            diff = point.t - prev_point.t
            if diff not in diffs:
                diffs[diff] = 1
            else:
                diffs[diff] +=1
        prev_point = point

    # Iterate until the diffs are not too spread, then pick the maximum.
    i=0
    while _is_almost_equal(len(diffs), len(timeseries)):
        or_diffs=diffs
        diffs={}
        for diff in or_diffs:
            diff=round(diff)
            if diff not in diffs:
                diffs[diff] = 1
            else:
                diffs[diff] +=1

        if i > 10:
            raise Exception('Cannot automatically detect the sampling interval')

    most_common_diff_total = 0
    most_common_diff = None
    for diff in diffs:
        if diffs[diff] > most_common_diff_total:
            most_common_diff_total = diffs[diff]
            most_common_diff = diff

    second_most_common_diff_total = 0
    for diff in diffs:
        if diff == most_common_diff:
            continue
        if diffs[diff] > second_most_common_diff_total:
            second_most_common_diff_total = diffs[diff]

    if confidence:
        if len(diffs) == 1:
            confidence_value = 1.0
        else:
            confidence_value = 1 - (second_most_common_diff_total / most_common_diff_total)
        return most_common_diff, confidence_value
    else:
        return most_common_diff


def detect_periodicity(timeseries):
    """Detect the periodicity of a time series.

        Args:
            timeseries(TimeSeries): the time series for which to detect the periodicity.

        Returns:
            int: the detected periodicity.
    """

    _check_timeseries(timeseries)
    data_labels = timeseries.data_labels()

    if len(data_labels) > 1:
        raise NotImplementedError()
    else:
        data_label = data_labels[0]

    # Get data as a vector
    y = [item.data[data_label] for item in timeseries]

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
    # TODO: round peak frequencies to integers and/or neighbors first?
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

    if not max_peak_frequency:
        raise ValueError('Cannot detect any periodicity!')

    # Round max peak and return
    return int(round(max_peak_frequency))


def mean_absolute_percentage_error(list1, list2):
    """Compute the MAPE.

        Args:
            list1(list): the true values.
            list2(list): the predicted values.

        Returns:
            float: the computed MAPE.
    """
    if len(list1) != len(list2):
        raise ValueError('Lists have different lengths, cannot continue')
    p_error_sum = 0
    for i in range(len(list1)):
        p_error_sum += abs((list1[i] - list2[i])/list1[i])
    return p_error_sum/len(list1)


def max_absolute_percentage_error(list1, list2):
    """Compute the MaxAPE.

        Args:
            list1(list): the true values.
            list2(list): the predicted values.

        Returns:
            float: the computed MaxAPE.
    """
    if len(list1) != len(list2):
        raise ValueError('Lists have different lengths, cannot continue')
    max_ape = None
    for i in range(len(list1)):
        ape = abs((list1[i] - list2[i])/list1[i])
        if max_ape is None:
            max_ape = ape
        else:
            if ape > max_ape:
                max_ape = ape
    return max_ape


def mean_absolute_error(list1,list2):
    """Compute the MAE.

        Args:
            list1(list): the true values.
            list2(list): the predicted values.

        Returns:
            float: the computed MAE.
    """
    return sklearn_mean_absolute_error(list1,list2)


def max_absolute_error(list1,list2):
    """Compute the MaxAE.

        Args:
            list1(list): the true values.
            list2(list): the predicted values.

        Returns:
            float: the computed MaxAE.
    """
    if len(list1) != len(list2):
        raise ValueError('Lists have different lengths, cannot continue')
    max_ae = None
    for i in range(len(list1)):
        ae = abs(list1[i] - list2[i])
        if max_ae is None:
            max_ae = ae
        else:
            if ae > max_ae:
                max_ae = ae
    return max_ae


def mean_absolute_log_error(list1, list2):
    """Compute the MALE.

        Args:
            list1(list): the true values.
            list2(list): the predicted values.

        Returns:
            float: the computed MALE.
    """
    if len(list1) != len(list2):
        raise ValueError('Lists have different lengths, cannot continue')
    ale_sum = 0
    for i in range(len(list1)):
        ale_sum += abs(log(list1[i]/list2[i]))
    return ale_sum/len(list1)


def max_absolute_log_error(list1,list2):
    """Compute the MaxALE.

        Args:
            list1(list): the true values.
            list2(list): the predicted values.

        Returns:
            float: the computed MaxALE.
    """
    if len(list1) != len(list2):
        raise ValueError('Lists have different lengths, cannot continue')
    max_ale = None
    for i in range(len(list1)):
        ale = abs(log(list1[i]/list2[i]))
        if max_ale is None:
            max_ale = ale
        else:
            if ale > max_ale:
                max_ale = ale
    return max_ale


def mean_squared_error(list1,list2):
    """Compute the MSE.

        Args:
            list1(list): the true values.
            list2(list): the predicted values.

        Returns:
            float: the computed MSE.
    """
    return sklearn_mean_squared_error(list1,list2)


def rescale(value, source_from, source_to, target_from=0, target_to=1):
    """Rescale a value from one range to another.

        Args:
            value(float, obj): the value to rescale.
            source_from(float): the source rescaling interval start.
            source_end(float): the source rescaling interval end.
            target_from(float): the target rescaling interval start. Defaults to 0.
            target_end(float): the target rescaling interval end. Defaults to 1.

        Returns:
            float or obj: the rescaled value.
    """

    if value < 0:
        raise ValueError('Cannot rescale negative value')
    if source_from < 0:
        raise ValueError('Cannot rescale using negative source_from')
    if source_to < 0:
        raise ValueError('Cannot rescale using negative source_to')
    if target_from < 0:
        raise ValueError('Cannot rescale using negative target_from')
    if target_to < 0:
        raise ValueError('Cannot rescale using negative target_to')
    if value < source_from:
        raise ValueError('Cannot rescale a value outside the source interval (value={}, source_from={}'.format(value, source_from))
    if value > source_to:
        raise ValueError('Cannot rescale a value outside the source interval (value={}, source_to={}'.format(value, source_to))

    source_total = source_to - source_from
    source_segment = value - source_from
    value_ratio = source_segment/source_total

    if target_from==0 and target_to==1:
        return value_ratio
    else:
        target_total = target_to - target_from
        return ((value_ratio*target_total)+target_from)


def os_shell(command, capture=False, verbose=False, interactive=False, silent=False):
    """Execute a command in the OS shell and print its output.

        Args:
            command(str): the command to execute.
            capture(bool): if to capture the output as a namedtuple with stdout, stderr, and exit code instead of
                           printing it. Defaults to False.
            interactive(bool): if to run the command in interactive mode. Defaults to False.
            silent(bool): if to suppress printing the output. Defaults to False.

        Returns:
            None or namedtupe: the output of the command, if any.
    """

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

    # Formatting...
    stdout = stdout[:-1] if (stdout and stdout[-1] == '\n') else stdout
    stderr = stderr[:-1] if (stderr and stderr[-1] == '\n') else stderr

    # Output namedtuple
    Output = namedtuple('Output', 'stdout stderr exit_code')

    if exit_code != 0:
        if capture:
            return Output(stdout, stderr, exit_code)
        else:
            string  = '\n#---------------------------------'
            string += '\n# Shell exited with exit code {}'.format(exit_code)
            string += '\n#---------------------------------\n'
            string += '\nStandard output: "'
            string += stdout.encode("utf-8", errors="ignore")
            string += '"\n\nStandard error: "'
            string += stderr.encode("utf-8", errors="ignore") +'"\n\n'
            string += '#---------------------------------\n'
            string += '# End Shell output\n'
            string += '#---------------------------------\n'
            print(string)
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


class DistributionFunction():
    """A class representing a statistical distribution. Implemented as a callable object, so that it can be evaluated at a given x.

        Args:
            dist(str): the name of the distirbution.
            params(dist): the parameters of the distirbution
    """

    def __init__(self, dist, params):
        self.dist = dist
        self.params = params

        # Init distirbution object
        self.dist_obj = getattr(scipy.stats, self.dist)

        # Check correct params
        self(0)

    def __call__(self, x):
        return self.dist_obj.pdf(x, **self.params)

    def plot(self, x_min=-1, x_max=1, show=True):
        """Plot the distribution.

            Args:
                x_min(float): the minimum value of the x axis.
                x_max(float): the maximum value of the x axis.
                show(bool): if to show the plot. Default to True.

            Returns:
                None or plt: the pot object, if not set to be shown.
        """

        plt.clf()

        # Populate data
        X = np.linspace(x_min, x_max, 1000)
        Y = [self(x) for x in X]
        plt.plot(X, Y, 'k', linewidth=2, label=str(self.dist), color='orange')

        # Viz
        plt.legend(loc="upper right")
        plt.grid()
        #plt.title('Distribution function: {}'.format(self.dist))

        # Show or return
        if show:
            plt.show()
        else:
            return plt

    def find_x(self, y, wideness=1000, side='right'):
        """Find the x for a given y.

            Args:
                y(float): the y to find the x for.
                wideness(float): how wide the search should be, on the x axis.
                side(str): on which side of the distribution to look.

            Returns:
                float: the x found for the given y.
        """
        # Note: "self" can be any function, we use this class as a callable here

        start = self.params['loc']
        if side == 'right':
            end = self.params['scale']*wideness
        elif side == 'left':
            end = -(self.params['scale']*wideness)
        else:
            raise ValueError('unknown side "{}"'.format(side))

        def scaled_gaussian(x):
            return self(x)-y

        x = optimize.bisect(scaled_gaussian, start, end)

        return x


#===========================
#  Private classes
#===========================

class _Gaussian():

    def __init__(self, mu, sigma, data=None):
        self.mu = mu
        self.sigma = sigma
        self.data = data

    def __call__(self, x):
        return stats.norm.pdf(x, self.mu, self.sigma)

    def cumulative(self,x):
        return stats.norm.cdf(x, self.mu, self.sigma)

    @classmethod
    def from_data(cls, data):
        mu, sigma = stats.norm.fit(data)
        return cls(mu,sigma,data)

    def plot(self, cumulative=False, bins=20):

        # Plot the histogram.
        if self.data:
            plt.hist(self.data, bins=bins, density=True, alpha=0.6, color='b')

        # Plot the gaussian itself
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)

        if cumulative:
            title = 'Gasussian (mu={:.3f}, sigma={:.3f}) - cumulative'.format(self.mu, self.sigma)
            p = stats.norm.cdf(x, self.mu, self.sigma)
        else:
            title = 'Gasussian (mu={:.3f}, sigma={:.3f})'.format(self.mu, self.sigma)
            p = stats.norm.pdf(x, self.mu, self.sigma)

        plt.plot(x, p, 'k', linewidth=2)
        plt.title(title)

        # Show the plot
        plt.show()

    def find_xes(self, y, wideness=1000):
        # Note: "self" can be any function, we use the gaussian class as a callable object here.

        start = self.mu
        end   = self.sigma*wideness

        def scaled_gaussian(x):
            return self(x)-y

        x = optimize.bisect(scaled_gaussian, start, end)

        return [ self.mu - (x-self.mu), x]


#===========================
#  Private utilities
#===========================

def _is_close(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def _is_almost_equal(one, two):
    if 0.95 < (one / two) <= 1.05:
        return True
    else:
        return False


def _set_from_t_and_to_t(from_dt, to_dt, from_t, to_t):
    """Set from_t and to_t from a (valid) combination of from_dt, to_dt, from_t, to_t."""

    # Sanity checks
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


def _item_is_in_range(item, from_t, to_t):
    """Check if an item (*TimePoint or *TimeSlot) is within the specified time range"""

    # Import here to avoid cyclic imports
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


def _compute_validity_regions(series, from_t=None, to_t=None, sampling_interval=None, cut=False):
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
        sampling_interval = series._autodetected_sampling_interval

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


def _compute_coverage(series, from_t, to_t):
    """
    Compute the coverage of an interval based on the point validity regions.

    Args:
        series: the time series.
        from_t: the interval start.
        to_t: the interval end.

    Returns:
        float: the interval coverage.
    """

    logger.debug('Called _compute_coverage() from %s to %s', from_t, to_t)

    # Check from_t/to_t
    if from_t >= to_t:
        raise ValueError('From is higher than to (or equal): from_t="{}", to_t="{}"'.format(from_t, to_t))

    # If the series is empty, return zero coverage
    if not series:
        return 0.0

    # Sum all the validity regions according to the from/to:
    coverage_s = 0
    for point in series:
        try:
            # Skip at the beginning and end if any
            if point.valid_to < from_t:
                continue
            if point.valid_from >= to_t:
                break

            # Skip if interpolated as well
            try:
                if point._interpolated:
                    continue
            except AttributeError:
                pass

            # Set this point valid from/to according to the interval boundaries
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
    logger.debug('_compute_coverage: Returning %s (%s percent)', coverage, coverage*100.0)
    return coverage


def _compute_data_loss(series, from_t, to_t, force=False, sampling_interval=None):
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

    logger.debug('Called _compute_data_loss() from %s to %s', from_t, to_t)

    # Get the series sampling interval unless it is forced to a specific value
    if not sampling_interval:
        sampling_interval = series._autodetected_sampling_interval

    # Compute the data loss from missing coverage.
    # Note: to improve performance, this could be computed only if the series is "raw" or forced,
    # i.e. at first and last items since on the borders there still may be  data losses.
    data_loss_from_missing_coverage = 1 - _compute_coverage(series, from_t, to_t)

    logger.debug('Data loss from missing coverage: %s', data_loss_from_missing_coverage)

    # Compute the data loss from previous data losses. Computed only if the resolutions is constant, as they can be defined only
    # if the series was already  made uniform (i.e. by a resampling process), or if forced for testing or other potential reasons.
    data_loss_from_previously_computed = 0.0

    if (series.resolution is not None) or force:

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

                # Now rescale the data loss with respect to the contribution and from/to and sum it up
                data_loss_from_previously_computed += point.data_loss * (point_contribution/( to_t - from_t))

    logger.debug('Data loss from previously computed: %s', data_loss_from_previously_computed)

    # Compute total data loss
    data_loss = data_loss_from_missing_coverage + data_loss_from_previously_computed

    # Return
    return data_loss


def _check_timeseries(series):
    """Check a time series for type, not emptiness and fixed resolution."""

    # Import here to avoid cyclic imports
    from .datastructures import TimeSeries

    if not isinstance(series, TimeSeries):
        raise TypeError('A TimeSeries object is required (got "{}")'.format(series.__class__.__name__))

    if not series:
        raise ValueError('A non-empty time series is required')

    if series.resolution == 'variable':
        raise ValueError('Time series with undefined (variable) resolutions are not supported. Resample or slot the time series first.')


def _check_resolution(series, resolution):
    if not series.resolution == resolution:
        raise ValueError('This model is fitted on "{}" resolution data, while your data has "{}" resolution.'.format(resolution, series.resolution))


def _check_data_labels(series, data_labels):
    series_data_labels = series.data_labels()
    if len(series_data_labels) != len(data_labels):
        raise ValueError('This model is fitted on {} data labels, while your data has {} data labels.'.format(len(data_labels), len(series_data_labels)))
    if series_data_labels != data_labels:
        raise ValueError('This model is fitted on "{}" data labels, while your data has "{}" data labels.'.format(data_labels, series_data_labels))


def _check_series_of_points_or_slots(series):
    # Import here to avoid cyclic imports
    from .datastructures import DataPoint, DataSlot
    if not (issubclass(series.item_type, DataPoint) or issubclass(series.item_type, DataSlot)):
        raise TypeError('Cannot operate on a series of "{}", only series of DataPoints or DataSlots are supported'.format(series.item_type.__name__))


def _check_indexed_data(series):
    try:
        series.data_labels()
    except TypeError:
        raise TypeError('Cannot operate on a series of "{}" with "{}" data, only series of indexed data (as lists and dicts) are supported'.format(series.item_type.__name__, series[0].data.__class__.__name__))


def _get_periodicity_index(item, resolution, periodicity, dst_affected=False):
    """Get the periodicty index."""

    # Import here to avoid cyclic imports
    from .units import Unit, TimeUnit

    # Handle specific cases
    if isinstance(resolution, TimeUnit):
        resolution_s = resolution.as_seconds(item.dt)
    elif isinstance(resolution, Unit):
        if isinstance(resolution, list):
            raise NotImplementedError('Sorry, periodocity in multi-dimensional spaces are not defined')
        resolution_s = resolution
    else:
        if isinstance(resolution, list):
            raise NotImplementedError('Sorry, periodocity in multi-dimensional spaces are not defined')
        resolution_s = resolution

    # Compute periodicity index
    if not dst_affected:

        # Get index based on item timestamp, normalized to unit, modulus periodicity
        periodicity_index =  int(item.t / resolution_s) % periodicity

    else:

        # Do we have an active DST?
        dst_timedelta = item.dt.dst()

        if dst_timedelta.days == 0 and dst_timedelta.seconds == 0:
            # No DST: get index based on item timestamp, normalized to unit, modulus periodicity
            periodicity_index = int(item.t / resolution_s) % periodicity

        else:
            # DST: get index based on item timestamp plus offset, normalized to unit, modulus periodicity 
            if dst_timedelta.days != 0:
                raise Exception('Don\'t know how to handle DST with days timedelta = "{}"'.format(dst_timedelta.days))

            if resolution_s > 3600:
                raise Exception('Sorry, this time series has not enough resolution to account for DST effects (resolution_s="{}", must be below 3600 seconds)'.format(resolution_s))

            dst_offset_s = dst_timedelta.seconds # 3600 usually
            periodicity_index = (int((item.t + dst_offset_s) / resolution_s) % periodicity)

    return periodicity_index


def _sanitize_string(string, no_data_placeholders=[]):
    """Sanitize the encoding of a string while handling "no data" placeholders"""

    string = re.sub('\s+',' ',string).strip()
    if string.startswith('\'') or string.startswith('"'):
        string = string[1:]
    if string.endswith('\'') or string.endswith('"'):
        string = string[:-1]
    string = string.strip()
    if string.lower().replace('.','') in no_data_placeholders:
        return None
    return string


def _is_list_of_integers(the_list):
    """Check if the list if made of integers"""

    for item in the_list:
        if not isinstance(item, int):
            return False
    else:
        return True


def _to_float(string,no_data_placeholders=[],label=None):
    """Convert to float while handling "no data" placeholders and discarding data indexes"""

    sanitized_string_string = _sanitize_string(string,no_data_placeholders)
    if sanitized_string_string:
        sanitized_string_string = sanitized_string_string.replace(',','.')
    try:
        return float(sanitized_string_string)
    except (ValueError, TypeError):
        # Do not raise if converting indexes as they are allowed to be "None"
        if label and label.startswith('__'):
            return None
        raise FloatConversionError(sanitized_string_string)


def _to_time_unit_string(seconds, friendlier=True):
    """Converts seconds to a (friendlier) time unit string, as 1s, 1h, 10m etc.)"""

    seconds_str = str(seconds).replace('.0', '')
    if friendlier:
        if seconds_str == '60':
            seconds_str = '1m'
        elif seconds_str == '600':
            seconds_str = '10m'
        elif seconds_str == '3600':
            seconds_str = '1h'
        else:
            seconds_str = seconds_str+'s'
    else:
        seconds_str = seconds_str+'s'
    return seconds_str


def _compute_distribution_approximation_errors(distribution_function, prediction_errors, bins=30, details=False):

    # Support vars
    max_prediction_error = max(prediction_errors)
    min_prediction_error = min(prediction_errors)
    error_unit = max_prediction_error - min_prediction_error

    # Create the bins
    error_step = error_unit / bins
    prediction_error_bins = [[] for _ in range(bins)]
    for i, prediction_error in enumerate(prediction_errors):
        for j in range(bins-1):
            if ( (min_prediction_error + (error_step*j))  <= prediction_error <= (min_prediction_error + (error_step*(j+1))) ):
                prediction_error_bins[j].append(prediction_error)

    real_distribution_values = {} # x -> y
    max_len = 0
    for i, prediction_error_bin in enumerate(prediction_error_bins):
        if prediction_error_bin:
            real_distribution_values[min_prediction_error + (error_step*i) + (error_step/2)] = len(prediction_error_bin)
            if len(prediction_error_bin) > max_len:
                max_len = len(prediction_error_bin)

    # Normalize
    for key in real_distribution_values:
        real_distribution_values[key] = real_distribution_values[key]/max_len

    # Compute the approximation errors
    approximation_errors = []
    binned_distribution_values = []
    binned_real_distribution_values = []
    for key in real_distribution_values:
        binned_distribution_values.append(distribution_function(key))
        binned_real_distribution_values.append(real_distribution_values[key])
        approximation_errors.append(abs(distribution_function(key)-real_distribution_values[key]))

    if details:
        return (approximation_errors, binned_distribution_values, binned_real_distribution_values)
    else:
        return approximation_errors


def _detect_notebook_major_version():
    stdout = subprocess.check_output(["jupyter", "--version"]).decode()
    versions={}
    for line in stdout.split('\n'):
        if ':' in line:
            line=line.strip()
            what, version = line.split(':')
            versions[what.strip()] = version.strip()
    if not versions or 'notebook' not in versions:
        notebook_major_version = None
    else:
        notebook_major_version = int(versions['notebook'].split('.')[0])
    return notebook_major_version


def _is_index_based(data):
    try:
        for i in range(len(data)):
            data[i]
    except:
        return False
    else:
        return True


def _is_key_based(data):
    if _is_index_based(data):
        return False
    try:
        for key in data:
            data[key]
    except:
        return False
    else:
        return True


def _has_numerical_values(data):

    if _is_index_based(data):
        for i in range(len(data)):
            try:
                data[i] + 1
            except:
                return False
        return True

    elif _is_key_based(data):
        for key in data:
            try:
                data[key] + 1
            except:
                return False
        return True

    else:
        raise ValueError('Don\'t know how to establish whether this data type has numerical values or not (got "{}")'.format(data.__class__.__name__))

