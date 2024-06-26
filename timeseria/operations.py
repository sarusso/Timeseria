# -*- coding: utf-8 -*-
"""Operations on the series."""

from datetime import datetime
from copy import copy, deepcopy
from propertime.utils import s_from_dt
from .datastructures import Series, Slot, Point, TimePoint
from .utilities import _is_close, _check_series_of_points_or_slots,_check_indexed_data
from .units import TimeUnit, Unit
from .exceptions import ConsistencyException

# Setup logging
import logging
logger = logging.getLogger(__name__)


#=================================
#  Base operation classes
#=================================

class Operation():
    """A generic series operation (callable object). Can return any valid data type,
    as a series, a scalar, a list of items, etc."""

    _supports_weights = False

    @classmethod
    def __str__(cls):
        return '{} operation'.format(cls.__name__)

    @classmethod
    def __repr__(cls):
        return '{}'.format(cls.__name__.lower())

    def __call__(self, series, *args, **kwargs):
        """Call self as a function.

        :meta private:
        """

        try:
            self._compute
        except AttributeError:
            raise NotImplementedError('No operation logic implemented.')
        else:
            if isinstance(series, Series):
                # Check for a single series
                if not series:
                    raise ValueError('Operations cannot work on empty series')
            else:
                # Assume list and check for multiple series
                try:
                    for item in series:
                        if isinstance(item, Series):
                            if not item:
                                raise ValueError('Operations cannot work on empty series')
                except TypeError:
                    # It was not iterable
                    raise TypeError('Operations can only work on series or lists of series')

            # Call compute logic
            return self._compute(series, *args, **kwargs)

    @property
    def __name__(self):
        return self.__class__.__name__.lower()


#=================================
#  Operations returning scalars
#=================================

class Max(Operation):
    """Maximum operation (callable object)."""

    def __init__(self):
        self.built_in_max = max

    def _compute(self, series, data_label=None):

        _check_series_of_points_or_slots(series)
        _check_indexed_data(series)

        maxs = {data_label: None for data_label in series.data_labels()}
        for item in series:
            for _data_label in maxs:

                if maxs[_data_label] is None:
                    maxs[_data_label] = item._data_by_label(_data_label)
                else:
                    if item._data_by_label(_data_label) > maxs[_data_label]:
                        maxs[_data_label] = item._data_by_label(_data_label)

        if data_label is not None:
            return maxs[data_label]
        else:
            return maxs



class Min(Operation):
    """Minimum operation (callable object)."""

    def __init__(self):
        self.built_in_min = min

    def _compute(self, series, data_label=None):

        _check_series_of_points_or_slots(series)
        _check_indexed_data(series)

        mins = {data_label: None for data_label in series.data_labels()}
        for item in series:
            for _data_label in mins:
                if mins[_data_label] is None:
                    mins[_data_label] = item._data_by_label(_data_label)
                else:
                    if item._data_by_label(_data_label) < mins[_data_label]:
                        mins[_data_label] = item._data_by_label(_data_label)

        if data_label is not None:
            return mins[data_label]
        else:
            return mins


class Avg(Operation):
    """Weighted average operation (callable object)."""

    _supports_weights = True

    def _compute(self, series, data_label=None):

        _check_series_of_points_or_slots(series)
        _check_indexed_data(series)

        # Log & weight check
        logger.debug('Called avg operation')
        try:
            for item in series:
                item.weight
                weighted=True
        except AttributeError:
            weighted=False
            #raise AttributeError('Trying to apply a weightd average on non-weigthed point')

        # Prepare
        sums = {data_label: None for data_label in series.data_labels()}

        # Compute
        if weighted:
            total_weights = 0
            for item in series:

                logger.debug('Point @ %s, weight: %s, data: %s', item.dt, item.weight, item.data)

                for _data_label in sums:
                    if sums[_data_label] is None:
                        sums[_data_label] = item._data_by_label(_data_label)*item.weight
                    else:
                        sums[_data_label] += item._data_by_label(_data_label)*item.weight
                    #logger.debug('Sums: %s',sums[_data_label])
                total_weights += item.weight

            # The sum are already the average as weugthed
            logger.debug('Total weights: %s', total_weights)
            if total_weights == 1:
                avgs = sums
            else:
                avgs = {}
                for _data_label in sums:
                    avgs[_data_label] = sums[_data_label] / total_weights
        else:
            for item in series:
                for _data_label in sums:
                    if sums[_data_label] is None:
                        sums[_data_label] = item._data_by_label(_data_label)
                    else:
                        sums[_data_label] += item._data_by_label(_data_label)
            avgs = {}
            for _data_label in sums:
                avgs[_data_label] = sums[_data_label] / len(series)

        # FInalize and return
        if data_label is not None:
            return avgs[data_label]
        else:
            return avgs


class Sum(Operation):
    """Sum operation (callable object)."""

    def _compute(self, series, data_label=None):

        _check_series_of_points_or_slots(series)
        _check_indexed_data(series)

        sums = {data_label: 0 for data_label in series.data_labels()}
        for item in series:
            for _data_label in sums:
                sums[_data_label] += item._data_by_label(_data_label)
        if data_label is not None:
            return sums[data_label]
        else:
            return sums


#=================================
#  Operations returning series
#=================================

class Derivative(Operation):
    """Derivative operation (callable object)."""

    def _compute(self, series, inplace=False, normalize=True, diffs=False):

        _check_series_of_points_or_slots(series)
        _check_indexed_data(series)

        if len(series) == 0:
            if diffs:
                raise ValueError('The differences cannot be computed on an empty series.')
            else:
                raise ValueError('The derivative cannot be computed on an empty series')

        if normalize:
            if not (issubclass(series.items_type, Point) or issubclass(series.items_type, Slot)):
                raise TypeError('Series items are not Points nor Slots, cannot compute a derivative')

        if len(series) == 1:
            if diffs:
                raise ValueError('The differences cannot be computed on a series with only one item')
            else:
                raise ValueError('The derivative cannot be computed on a series with only one item')

        if diffs and inplace:
            raise NotImplementedError('Computing differences in-place is not supported as it would change the series length')

        if normalize:
            if series.resolution == 'variable':
                variable_resolution = True
                sampling_interval = series._autodetected_sampling_interval
            else:
                variable_resolution = False
                if isinstance(series.resolution, TimeUnit):
                    sampling_interval = series.resolution.as_seconds(series[0].dt)
                elif isinstance(series.resolution, Unit):
                    sampling_interval = series.resolution
                else:
                    sampling_interval = series.resolution
            postfix='derivative'
        else:
            postfix='diff'

        if not inplace:
            der_series = series.__class__()

        try:
            # Try to access by label, if data was non key-value this raise, as labels
            # are always generated as strings even for indexed data (as lists)
            series[0].data[series.data_labels()[0]]
            data_is_keyvalue = True
        except TypeError:
            data_is_keyvalue = False

        for i, item in enumerate(series):

            # Skip the first element if computing diffs
            if diffs and i == 0:
                continue

            if not inplace:
                der_data = series[0].data.__class__()

            for data_label in series.data_labels():
                diff_left = None
                diff_right = None
                diff = None
                if diffs:
                    # Just compute the diffs
                    if i == 0:
                        raise ConsistencyException('We should never get here')
                    else:
                        diff = series[i]._data_by_label(data_label) - series[i-1]._data_by_label(data_label)
                else:
                    # Compute the derivative
                    if i == 0:
                        # Right increment for the first item
                        diff_right = series[i+1]._data_by_label(data_label) - series[i]._data_by_label(data_label)
                    elif i == len(series)-1:
                        # Left increment for the last item. Divide by two if not normalizing by time resolution
                        diff_left = series[i]._data_by_label(data_label) - series[i-1]._data_by_label(data_label)
                    else:
                        # Both left and right increment for the items in the middle
                        diff_left = series[i]._data_by_label(data_label) - series[i-1]._data_by_label(data_label)
                        diff_right = series[i+1]._data_by_label(data_label) - series[i]._data_by_label(data_label)

                # Combine and normalize (if required) the increments to get the actual derivative
                if normalize:
                    if diff is not None:
                        diff = diff / sampling_interval
                    elif diff_right is None:
                        diff = diff_left / sampling_interval
                    elif diff_left is None:
                        diff = diff_right / sampling_interval
                    else:
                        if variable_resolution:
                            diff = ((diff_left / (series[i].t - series[i-1].t)) + (diff_right / (series[i+1].t - series[i].t))) /2
                        else:
                            diff = ((diff_left + diff_right)/2)/sampling_interval
                else:
                    if diff is not None:
                        pass
                    elif diff_right is None:
                        diff = diff_left
                    elif diff_left is None:
                        diff = diff_right
                    else:
                        diff = (diff_left+diff_right)/2

                # Add data
                if not inplace:
                    if data_is_keyvalue:
                        der_data['{}_{}'.format(data_label, postfix)] = diff
                    else:
                        der_data.append(diff)
                else:
                    if data_is_keyvalue:
                        item.data['{}_{}'.format(data_label, postfix)] = diff
                    else:
                        item.data[int(data_label)] = diff


            # Create the item
            if not inplace:

                if isinstance(series[0], Point):
                    der_series.append(series[0].__class__(t = item.t,
                                                          tz = item.tz,
                                                          data = der_data,
                                                          data_loss = item.data_loss))
                elif isinstance(series[0], Slot):
                    der_series.append(series[0].__class__(start = item.start,
                                                          unit = item.unit,
                                                          data = der_data,
                                                          data_loss = item.data_loss))
                else:
                    raise NotImplementedError('Working on series other than slots or points not yet implemented')

        if not inplace:
            return der_series


class Integral(Operation):
    """Integral operation (callable object)."""

    def _compute(self, series, inplace=False, normalize=True, c=0, offset=0):

        _check_series_of_points_or_slots(series)
        _check_indexed_data(series)

        if len(series) == 0:
            if normalize:
                raise ValueError('The integral cannot be computed on an empty series.')
            else:
                raise ValueError('The cumulative sum cannot be computed on an empty series')

        if normalize:
            if not (issubclass(series.items_type, Point) or issubclass(series.items_type, Slot)):
                raise TypeError('Series items are not Points nor Slots, cannot compute an integral')

        if normalize:
            if series.resolution == 'variable':
                variable_resolution = True
                sampling_interval = series._autodetected_sampling_interval
            else:
                variable_resolution = False
                if isinstance(series.resolution, TimeUnit):
                    sampling_interval = series.resolution.as_seconds(series[0].dt)
                elif isinstance(series.resolution, Unit):
                    sampling_interval = series.resolution
                else:
                    sampling_interval = series.resolution
            postfix='integral'
        else:
            postfix='csum'

        if not inplace:
            int_series = series.__class__()

        try:
            # Try to access by label, if data was non key-value this raise, as labels
            # are always generated as strings even for indexed data (as lists)
            series[0].data[series.data_labels()[0]]
            data_is_keyvalue = True
        except TypeError:
            data_is_keyvalue = False

        last_values={}

        for i, item in enumerate(series):

            # Do we have to create the previous point based on an offset?
            if i==0 and offset:
                if inplace:
                    raise ValueError('Cannot use prev_data in in-place mode, would require to add a point to the series')

                prev_data = series[0].data.__class__()

                if isinstance(offset,dict):
                    if data_is_keyvalue:
                        for data_label in series.data_labels():
                            prev_data['{}_{}'.format(data_label,postfix)] = offset[data_label]
                    else:
                        for data_label in series.data_labels():
                            prev_data.append(offset[data_label])
                else:
                    if data_is_keyvalue:
                        for data_label in series.data_labels():
                            prev_data['{}_{}'.format(data_label,postfix)] = offset
                    else:
                        for data_label in series.data_labels():
                            prev_data.append(offset[data_label])

                if isinstance(item, Point):
                    int_series.append(item.__class__(t = item.t - series.resolution,
                                                     tz = item.tz,
                                                     data = prev_data,
                                                     data_loss = 0))
                elif isinstance(item, Slot):
                    int_series.append(item.__class__(start = item.start - item.unit,
                                                     unit = item.unit,
                                                     data = prev_data,
                                                     data_loss = 0))
                # Now set the c accordingly:
                c = offset


            if not inplace:
                data = series[0].data.__class__()

            for data_label in series.data_labels():

                # Get the value
                if not normalize:
                    value = series[i]._data_by_label(data_label)
                else:
                    # Compute the integral
                    if i == 0:
                        value=0
                        prev_right_component = series[i]._data_by_label(data_label)
                    else:
                        left_component = (series[i-1]._data_by_label(data_label)*2) - prev_right_component
                        # The "value" below is actually an increment, see below at  value = last_values[data_label] + value
                        value = left_component
                        prev_right_component = left_component

                # Normalize the cumulative to get the actual integral
                if normalize:
                    if variable_resolution:
                        if i == 0:
                            value = value * sampling_interval
                        else:
                            value = value * (series[i].t - series[i-1].t)
                    else:
                        value = value * sampling_interval

                # Sum to previous or apply offset if any
                try:
                    value = last_values[data_label] + value
                except:
                    try:
                        # Do we have offsets as a dict?
                        value = value + c[data_label]
                    except:
                        value = value + c

                # Add data
                if not inplace:
                    if data_is_keyvalue:
                        data['{}_{}'.format(data_label, postfix)] = value
                    else:
                        data.append(value)
                else:
                    if data_is_keyvalue:
                        item.data['{}_{}'.format(data_label, postfix)] = value
                    else:
                        item.data[int(data_label)] = value


                last_values[data_label] = value

            # Create the item
            if not inplace:

                if isinstance(series[0], Point):
                    int_series.append(series[0].__class__(t = item.t,
                                                          tz = item.tz,
                                                          data = data,
                                                          data_loss = item.data_loss))
                elif isinstance(series[0], Slot):
                    int_series.append(series[0].__class__(start = item.start,
                                                          unit = item.unit,
                                                          data = data,
                                                          data_loss = item.data_loss))
                else:
                    raise NotImplementedError('Working on series other than slots or points not yet implemented')

        if not inplace:
            return int_series


class Diff(Derivative):
    """Incremental differences operation (callable object)."""
    def _compute(self, series, inplace=False):
        if series.resolution == 'variable':
            raise ValueError('The differences cannot be computed on variable resolution time series, resample it or use the derivative operation.')
        return super(Diff, self)._compute(series, inplace=inplace, normalize=False, diffs=True)


class CSum(Integral):
    """Cumulative sum operation (callable object)."""
    def _compute(self, series, inplace=False, offset=None):
        if series.resolution == 'variable':
            raise ValueError('The cumulative sums cannot be computed on variable resolution time series, resample it or use the integral operation.')
        return super(CSum, self)._compute(series, inplace=inplace, normalize=False, offset=offset)


class Normalize(Operation):
    """Normalization operation (callable object)"""

    def _compute(self, series, range=[0,1], inplace=False):

        _check_series_of_points_or_slots(series)
        _check_indexed_data(series)

        if range == [0,1]:
            custom_range = None
        else:
            custom_range = range

        if not inplace:
            normalized_series = series.__class__()

        data_labels = series.data_labels()

        # Get min and max for the data labels
        for i, item in enumerate(series):

            if i == 0:
                mins = {data_label: item._data_by_label(data_label) for data_label in data_labels}
                maxs = {data_label: item._data_by_label(data_label) for data_label in data_labels}
            else:
                for data_label in data_labels:
                    if item._data_by_label(data_label) < mins[data_label]:
                        mins[data_label] = item._data_by_label(data_label)
                    if item._data_by_label(data_label) > maxs[data_label]:
                        maxs[data_label] = item._data_by_label(data_label)

        try:
            # Try to access by label, if data was non key-value this raise, as labels
            # are always generated as strings even for in-based indexed data (as lists)
            series[0].data[series.data_labels()[0]]
            data_is_keyvalue = True
        except TypeError:
            data_is_keyvalue = False

        for i, item in enumerate(series):

            if not inplace:
                normalized_data = series[0].data.__class__()

            for data_label in data_labels:

                # Normalize data
                if custom_range:
                    normalized_value = (((item._data_by_label(data_label) - mins[data_label])  / (maxs[data_label]-mins[data_label])) * (custom_range[1]-custom_range[0])) + custom_range[0]
                else:
                    normalized_value = (item._data_by_label(data_label) - mins[data_label])  / (maxs[data_label]-mins[data_label])

                if not inplace:
                    if data_is_keyvalue:
                        normalized_data[data_label] = normalized_value
                    else:
                        normalized_data.append(normalized_value)
                else:
                    if data_is_keyvalue:
                        item.data[data_label] = normalized_value
                    else:
                        item.data[int(data_label)] = normalized_value

            # Create the item
            if not inplace:

                if isinstance(series[0], Point):
                    normalized_series.append(series[0].__class__(t = item.t,
                                                                 tz = item.tz,
                                                                 data = normalized_data,
                                                                 data_loss = item.data_loss))
                elif isinstance(series[0], Slot):
                    normalized_series.append(series[0].__class__(start = item.start,
                                                                 unit = item.unit,
                                                                 data = normalized_data,
                                                                 data_loss = item.data_loss))
                else:
                    raise NotImplementedError('Working on series other than slots or points not yet implemented')

        if not inplace:
            return normalized_series


class Rescale(Operation):
    """Rescaling operation (callable object)"""

    def _compute(self, series, value, inplace=False):

        _check_series_of_points_or_slots(series)
        _check_indexed_data(series)

        if not inplace:
            rescaled_series = series.__class__()

        try:
            # Try to access by label, if data was non key-value this raise, as labels
            # are always generated as strings even for in-based indexed data (as lists)
            series[0].data[series.data_labels()[0]]
            data_is_keyvalue = True
        except TypeError:
            data_is_keyvalue = False

        for item in series:

            if not inplace:
                rescaled_data = series[0].data.__class__()

            # Rescale data
            for data_label in series.data_labels():
                if isinstance(value, dict):
                    if data_label in value:
                        rescaled_value = item._data_by_label(data_label) * value[data_label]
                    else:
                        rescaled_value = item._data_by_label(data_label)
                else:
                    rescaled_value = item._data_by_label(data_label) * value

                if not inplace:
                    if data_is_keyvalue:
                        rescaled_data[data_label] = rescaled_value
                    else:
                        rescaled_data.append(rescaled_value)
                else:
                    if data_is_keyvalue:
                        item.data[data_label] = rescaled_value
                    else:
                        item.data[int(data_label)] = rescaled_value

            # Set data or create the item
            if not inplace:
                if isinstance(series[0], Point):
                    rescaled_series.append(series[0].__class__(t = item.t,
                                                               tz = item.tz,
                                                               data = rescaled_data,
                                                               data_loss = item.data_loss))
                elif isinstance(series[0], Slot):
                    rescaled_series.append(series[0].__class__(start = item.start,
                                                               unit = item.unit,
                                                               data = rescaled_data,
                                                               data_loss = item.data_loss))
                else:
                    raise NotImplementedError('Working on series other than slots or points not yet implemented')

        if not inplace:
            return rescaled_series


class Offset(Operation):
    """Offsetting operation (callable object)"""

    def _compute(self, series, value, inplace=False):

        _check_series_of_points_or_slots(series)
        _check_indexed_data(series)

        if not inplace:
            rescaled_series = series.__class__()

        try:
            # Try to access by label, if data was non key-value this raise, as labels
            # are always generated as strings even for in-based indexed data (as lists)
            series[0].data[series.data_labels()[0]]
            data_is_keyvalue = True
        except TypeError:
            data_is_keyvalue = False

        for item in series:

            if not inplace:
                offsetted_data = series[0].data.__class__()

            # Offset data
            for data_label in series.data_labels():
                if isinstance(value, dict):
                    if data_label in value:
                        offsetted_value = item._data_by_label(data_label) + value[data_label]
                    else:
                        offsetted_value = item._data_by_label(data_label)
                else:
                    offsetted_value= item._data_by_label(data_label) + value

                if not inplace:
                    if data_is_keyvalue:
                        offsetted_data[data_label] = offsetted_value
                    else:
                        offsetted_data.append(offsetted_value)
                else:
                    if data_is_keyvalue:
                        item.data[data_label] = offsetted_value
                    else:
                        item.data[int(data_label)] = offsetted_value

            # Set data or create the item
            if not inplace:

                if isinstance(series[0], Point):
                    rescaled_series.append(series[0].__class__(t = item.t,
                                                               tz = item.tz,
                                                               data = offsetted_data,
                                                               data_loss = item.data_loss))
                elif isinstance(series[0], Slot):
                    rescaled_series.append(series[0].__class__(start = item.start,
                                                               unit = item.unit,
                                                               data = offsetted_data,
                                                               data_loss = item.data_loss))
                else:
                    raise NotImplementedError('Working on series other than slots or points not yet implemented')

        if not inplace:
            return rescaled_series


class MAvg(Operation):
    """Moving average operation (callable object)."""

    def _compute(self, series, window, inplace=False):

        _check_series_of_points_or_slots(series)
        _check_indexed_data(series)

        if not window or window <1:
            raise ValueError('A integer window >0 is required (got window="{}"'.frmat(window))

        if inplace:
            raise NotImplementedError('Computing a moving average in-place is not supported as it would change the series length')

        postfix='mavg_{}'.format(window)

        try:
            # Try to access by label, if data was non key-value this raise, as labels
            # are always generated as strings even for in-based indexed data (as lists)
            series[0].data[series.data_labels()[0]]
            data_is_keyvalue = True
        except TypeError:
            data_is_keyvalue = False

        mavg_series = series.__class__()

        for i, item in enumerate(series):

            if i < window-1:
                continue

            mavg_data = series[0].data.__class__()

            for data_label in series.data_labels():

                # Compute the moving average
                value_sum = 0
                for j in range(i-window+1,i+1):
                    value_sum += series[j]._data_by_label(data_label)
                mavg_value = value_sum/window

                if data_is_keyvalue:
                    mavg_data['{}_{}'.format(data_label, postfix)] = mavg_value
                else:
                    mavg_data.append(mavg_value)

            # Create the item
            if isinstance(series[0], Point):
                mavg_series.append(series[0].__class__(t = item.t,
                                                       tz = item.tz,
                                                       data = mavg_data))
            elif isinstance(series[0], Slot):
                mavg_series.append(series[0].__class__(start = item.start,
                                                       unit = item.unit,
                                                       data = mavg_data))
            else:
                raise NotImplementedError('Working on series other than slots or points not yet implemented')

        return mavg_series


class Get(Operation):
    """Get operation (callable object)."""

    def __call__(self, series, *args, **kwargs):
        if not series:
            raise IndexError
        else:
            return super(Get, self).__call__(series, *args, **kwargs)

    def _compute(self, series, at):

        # Detect argument type
        if isinstance(at, int):
            return series._item_by_i(at)
        if isinstance(at, float):
            return series._item_by_t(at)
        if isinstance(at, datetime):
            return series._item_by_t(s_from_dt(at))
        if  isinstance(at, TimePoint):
            return series._item_by_t(at.t)

        raise TypeError('Don\'t know how to get items from a series for type "{}"'.format(at.__class__.__name__))


class Filter(Operation):
    """Filter operation (callable object)."""

    def _compute(self, series, *data_labels):

        _check_series_of_points_or_slots(series)
        _check_indexed_data(series)

        # Instantiate the filtered series
        filtered_series = series.__class__()

        # Filter based on data label
        try:
            series[0].data[series.data_labels()[0]]
        except TypeError:
            raise TypeError('Cannot filter by data label on non key-value data (Got "{}")'.format(series[0].data.__class__.__name__))
        for item in series:
            filtered_item = copy(item)
            filtered_item._data = {data_label: item._data_by_label(data_label) for data_label in data_labels}
            filtered_series.append(filtered_item)

        # Re-set reference data as well
        try:
            filtered_series._item_data_reference = filtered_series[0].data
        except (AttributeError, IndexError):
            pass

        return filtered_series


class Slice(Operation):
    """Slice operation (callable object)."""

    def _compute(self, series, start=None, end=None):

        # TODO: check for series type to allow time-based slicing? Same in the Get..

        start_i = None
        end_i = None
        start_t = None
        end_t = None

        # Handle start
        if start is not None:
            if isinstance(start, datetime):
                start_t = s_from_dt(start)
            elif isinstance(start, float):
                start_t = start
            elif isinstance(start, int):
                if start < 0:
                    start_i = len(series) - abs(start)
                else:
                    start_i = start
            else:
                raise ValueError('Cannot use "{}" as start value for the slice as not int, float (epoch timestamp) nor datetime'.format(start))

        # Handle end
        if end is not None:
            if isinstance(end, datetime):
                end_t = s_from_dt(end)
            elif isinstance(end, float):
                end_t = end
            elif isinstance(end, int):
                if end < 0:
                    end_i = len(series) - abs(end) 
                else:
                    end_i = end
            else:
                raise ValueError('Cannot use "{}" as end value for the slice as not int, float (epoch timestamp) nor datetime'.format(end))

        # Check incompatibilities
        if start_t is not None or end_t is not None:
            if start_i is not None or end_i is not None:
                raise ValueError('Cannot slice mixing int and float (epoch timestamp) or datetime')

        # Handle inverted or equal positions (slight optimization)
        if start_t is not None and end_t is not None:
            if start_t >= end_t:
                return series.__class__()
        if start_i is not None and end_i is not None:
            if start_i >= end_i:
                return series.__class__()

        # Instantiate the new, sliced series
        sliced_series = series.__class__()

        # Preserve mark if any TODO: Check if mark still in the sliced series?
        sliced_series.mark = series.mark

        # Select sliced series items based on start end (if set) or just duplicate it
        if (start_t is not None or end_t is not None):
            for item in series:
                if start_t is not None:
                    if item.t < start_t:
                        continue
                if end_t is not None:
                    try:
                        # Slot?
                        if item.end.t > end_t:
                            break
                    except AttributeError:
                        # Point?
                        if item.t >= end_t:
                            break
                # If we are here this item has to be added to the slice
                sliced_series.append(item)

        elif (start_i is not None or end_i is not None):
            for i, item in enumerate(series):
                if start_i is not None:
                    if i < start_i:
                        continue
                if end_i is not None:
                    if i >= end_i:
                        break
                # If we are here this item has to be added to the slice
                sliced_series.append(item)
        else:
            sliced_series = series.duplicate()

        # Re-set reference data as well
        try:
            sliced_series._item_data_reference = sliced_series[0].data
        except (IndexError, AttributeError):
            pass

        return sliced_series


class Merge(Operation):
    """Merge operation (callable object)."""

    def _compute(self, *series):

        seriess = series

        # Support vars
        resolution = None
        seriess_starting_points_t = []
        seriess_ending_points_t   = []

        # Checks
        for i, series in enumerate(seriess):
            if series.resolution == 'variable':
                raise ValueError('Cannot merge variable resolution series')
            if resolution is None:
                resolution = series.resolution
            else:
                if series.resolution != resolution:
                    abort = True
                    try:
                        # Handle floating point precision issues
                        if _is_close(series.resolution, resolution):
                            abort = False
                    except (ValueError,TypeError, NotImplementedError):
                        pass
                    if abort:
                        raise ValueError('Series have different resolutions ("{}" vs "{}"), cannot merge'.format(resolution, series.resolution))

            # Find min and max epoch for each series (aka start-end points)
            seriess_starting_points_t.append(series[0].t)
            seriess_ending_points_t.append(series[-1].t)

        # Find max min and min max to set the subset where series are all deifned. e.g.:
        # ======
        #    ======
        # we are looking for the second segment (max) starting point

        max_starting_point_t = None
        for starting_point_t in seriess_starting_points_t:
            if max_starting_point_t is None:
                max_starting_point_t = starting_point_t
            else:
                if starting_point_t > max_starting_point_t:
                    max_starting_point_t = starting_point_t

        min_ending_point_t = None
        for ending_point_t in seriess_ending_points_t:
            if min_ending_point_t is None:
                min_ending_point_t = ending_point_t
            else:
                if ending_point_t < min_ending_point_t:
                    min_ending_point_t = ending_point_t

        # Find the time series with the max starting point
        for series in seriess:
            if series[0].t == max_starting_point_t:
                reference_series = series

        # Create the (empty) result time series
        result_series = seriess[0].__class__()

        # Scroll the time series to get the offsets
        offsets = {}
        for series in seriess:
            for i in range(len(series)):
                if series[i].t == max_starting_point_t:
                    offsets[series] = i
                    break

        # Loop over the reference series to create the datapoints for the merge
        for i in range(len(reference_series)):

            # Check if we gone beyond the minimum ending point
            if reference_series[i].t > min_ending_point_t:
                break

            # Support vars
            data = None
            data_loss = 0
            valid_data_losses = 0

            tz = seriess[0][0].tz
            if isinstance(reference_series[0], Slot):
                unit = reference_series[0].unit
            else:
                unit = None

            # For each time series, get data in i-th position (given the offset) and merge it
            for series in seriess:

                # Data
                if data is None:
                    data = deepcopy(series[i+offsets[series]].data)
                else:
                    data.update(series[i+offsets[series]].data)

                # Data loss
                if series[i+offsets[series]].data_loss is not None:
                    valid_data_losses += 1
                    data_loss += series[i+offsets[series]].data_loss

            # Finalize data loss if there were valid ones
            if valid_data_losses:
                data_loss = data_loss / valid_data_losses
            else:
                data_loss = None

            if isinstance(reference_series[0], Point):
                result_series.append(seriess[0][0].__class__(t = reference_series[i].t,
                                                                     tz = tz,
                                                                     data  = data,
                                                                     data_loss = data_loss))
            elif isinstance(reference_series[0], Slot):
                result_series.append(seriess[0][0].__class__(start = reference_series[i].start,
                                                                     unit  = unit,
                                                                     data  = data,
                                                                     data_loss = data_loss))
            else:
                raise NotImplementedError('Working on series other than slots or points not yet implemented')

        return result_series


#=================================
# Operations returning lists
#=================================

class Select(Operation):
    """Select operation (callable object)."""

    def _compute(self, series, query):

        _check_series_of_points_or_slots(series)
        _check_indexed_data(series)

        if (' and ' or ' AND ' or ' or ' or ' OR ') in query:
            raise NotImplementedError('Multiple conditions not yet supported')

        if '=' not in query:
            raise NotImplementedError('Clauses other than the equality are not yet supported')

        # Get data_label
        data_label = query.split('=')[0].strip()
        if data_label.startswith('"'):
            data_label=data_label[1:]
        if data_label.endswith('"'):
            data_label=data_label[:-1]

        # Get value
        value = float(query.split('=')[1])

        # Perform the select
        selected = None
        for item in series:
            if item._data_by_label(data_label) == value:
                try:
                    selected.append(item)
                except:
                    selected = [item]
        return selected


#========================
# Instantiate operations
#========================

# TODO: just abandon the callable object thing and convert them to functions?

# Scalar operations
min = Min()
max = Max()
avg = Avg()
sum = Sum()

# Series operations
derivative = Derivative()
integral = Integral()

diff = Diff()
csum = CSum()
mavg = MAvg()

normalize = Normalize()
offset = Offset()
rescale = Rescale()

merge = Merge()
filter = Filter()
slice = Slice()
get = Get()
select = Select()




