# -*- coding: utf-8 -*-
"""Operations on the series."""

from datetime import datetime
from copy import copy
from propertime.utils import s_from_dt
from .datastructures import Series, Slot, Point
from .utils import _is_close, _check_series_of_points_or_slots,_check_indexed_data
from .units import TimeUnit, Unit
from .exceptions import ConsistencyException

# Setup logging
import logging
logger = logging.getLogger(__name__)


#=================================
#  Base operation classes
#=================================

class Operation():
    """A generic series operation. Can return any data type,
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
            self._call
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
            return self._call(series, *args, **kwargs)

    @property
    def __name__(self):
        return self.__class__.__name__.lower()


#=================================
#  Operations returning scalars
#=================================

class Max(Operation):
    """Get the maximum data value(s) of a series. A series of DataPoints or DataSlots is required.

    Args:
       series(Series): the series on which to perform the operation.
       data_label(string): if provided, compute the value only for this data label. Defaults to None.

    Returns:
       dict or object: the computed values for each data label, or a specific value if
       providing the data_label argument.
    """

    def __init__(self):
        self.built_in_max = max

    def _call(self, series, data_label=None):

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
    """Get the minimum data value(s) of a series. A series of DataPoints or DataSlots is required.

    Args:
       series(Series): the series on which to perform the operation.
       data_label(string): if provided, compute the value only for this data label.
                           Defaults to None.

    Returns:
       dict or object: the computed values for each data label, or a specific value if
       providing the data_label argument.
    """

    def __init__(self):
        self.built_in_min = min

    def _call(self, series, data_label=None):

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
    """Get the average data value(s) of a series. A series of DataPoints or DataSlots is required.

    Args:
       series(Series): the series on which to perform the operation.
       data_label(string, optional): if provided, compute the value only for this data label.

    Returns:
       dict or object: the computed values for each data label, or a specific value if
       providing the data_label argument.
    """

    _supports_weights = True

    def _call(self, series, data_label=None):

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

        # Finalize and return
        if data_label is not None:
            return avgs[data_label]
        else:
            return avgs


class Sum(Operation):
    """Sum every data value(s) of a series. A series of DataPoints or DataSlots is required.

    Args:
       series(Series): the series on which to perform the operation.
       data_label(string, optional): if provided, compute the value only for this data label.

    Returns:
       dict or object: the computed values for each data label, or a specific value if
       providing the data_label argument.
    """

    def _call(self, series, data_label=None):

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
    """Compute the derivative on a series. A series of DataTimePoints or DataTimeSlots is required.

    Args:
       series(Series): the series on which to perform the operation.
       inplace(bool): if to perform the operation in-place on the series. Defaults to False.
       normalize(bool): if to normalize the derivative w.r.t to the series resolution. Defaults to True.
       diffs(bool): if to compute the differences instead of the derivative. Defaults to False.

    Returns:
       series or None: the computed series, or None if set to perform the operation in-place.
    """

    def _call(self, series, inplace=False, normalize=True, diffs=False):

        _check_series_of_points_or_slots(series)
        _check_indexed_data(series)

        if len(series) == 0:
            if diffs:
                raise ValueError('The differences cannot be computed on an empty series.')
            else:
                raise ValueError('The derivative cannot be computed on an empty series')

        if normalize:
            if not (issubclass(series.item_type, Point) or issubclass(series.item_type, Slot)):
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
                                                          data_indexes = item.data_indexes))
                elif isinstance(series[0], Slot):
                    der_series.append(series[0].__class__(start = item.start,
                                                          unit = item.unit,
                                                          data = der_data,
                                                          data_indexes = item.data_indexes))
                else:
                    raise NotImplementedError('Working on series other than slots or points not yet implemented')

        if not inplace:
            return der_series


class Integral(Operation):
    """Compute the integral on a series. A series of DataTimePoints or DataTimeSlots is required.

    Args:
       series(Series): the series on which to perform the operation.
       inplace(bool): if to perform the operation in-place on the series. Defaults to False.
       normalize(bool): if to normalize the integral w.r.t to the series resolution. Defaults to True.
       c(float, dict): the integrative constant, as a single value or as a dictionary of values, one
                       for each data label. Defaults to zero.
       offset(float, dict): if to start the integrative process from a specific offset. Can be provided as a
                            single value or as a dictionary of values, one for each data label. Defaults to zero.

    Returns:
       series or None: the computed series, or None if set to perform the operation in-place.
    """

    def _call(self, series, inplace=False, normalize=True, c=0, offset=0):

        if c and offset:
            raise ValueError('Choose between using and integrative constant or a starting offset, got both.')

        _check_series_of_points_or_slots(series)
        _check_indexed_data(series)

        if len(series) == 0:
            if normalize:
                raise ValueError('The integral cannot be computed on an empty series.')
            else:
                raise ValueError('The cumulative sum cannot be computed on an empty series')

        if normalize:
            if not (issubclass(series.item_type, Point) or issubclass(series.item_type, Slot)):
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
                    raise ValueError('Cannot use the offset in in-place mode, would require to add a point to the series')

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
                                                          data_indexes = item.data_indexes))
                elif isinstance(series[0], Slot):
                    int_series.append(series[0].__class__(start = item.start,
                                                          unit = item.unit,
                                                          data = data,
                                                          data_indexes = item.data_indexes))
                else:
                    raise NotImplementedError('Working on series other than slots or points not yet implemented')

        if not inplace:
            return int_series


class Diff(Derivative):
    """Compute the incremental differences on a series. Reduces the series length by one
    (removing the first element). A series of DataTimePoints or DataTimeSlots is required.

    Args:
       series(Series): the series on which to perform the operation.
       inplace(bool): if to perform the operation in-place on the series. Defaults to False.

    Returns:
       series or None: the computed series, or None if set to perform the operation in-place.
    """
    def _call(self, series, inplace=False):
        if series.resolution == 'variable':
            raise ValueError('The differences cannot be computed on variable resolution time series, resample it or use the derivative operation.')
        return super(Diff, self)._call(series, inplace=inplace, normalize=False, diffs=True)


class CSum(Integral):
    """Compute the incremental sum on a series. A series of DataTimePoints or DataTimeSlots is required.

    Args:
       series(Series): the series on which to perform the operation.
       inplace(bool): if to perform the operation in-place on the series. Defaults to False.
       offset(float, dict): if to start computing the cumulative sum from a specific offset. Can be provided as a
                            single value or as a dictionary of values, one for each data label. Defaults to None.

    Returns:
       series or None: the computed series, or None if set to perform the operation in-place.
    """
    def _call(self, series, inplace=False, offset=None):
        if series.resolution == 'variable':
            raise ValueError('The cumulative sums cannot be computed on variable resolution time series, resample it or use the integral operation.')
        return super(CSum, self)._call(series, inplace=inplace, normalize=False, offset=offset)


class Normalize(Operation):
    """Normalize the data values of a series bringing them to a given range. A series of DataTimePoints or DataTimeSlots is required.

    Args:
       series(Series): the series on which to perform the operation.
       range(list): the normalization target range. Defaults to [0,1].
       inplace(bool): if to perform the operation in-place on the series. Defaults to False.
       source_range(dict, optional): a custom source range, by data label, to normalize with respect to.

    Returns:
       series or None: the computed series, or None if set to perform the operation in-place.
    """
    def _call(self, series, range=[0,1], source_range=None, inplace=False):

        _check_series_of_points_or_slots(series)
        _check_indexed_data(series)

        if range == [0,1]:
            custom_range = None
        else:
            custom_range = range

        if not inplace:
            normalized_series = series.__class__()

        data_labels = series.data_labels()

        # Compute min and max for the data labels if no source range was provided
        if source_range:
            mins = {data_label: source_range[data_label][0] for data_label in data_labels}
            maxs = {data_label: source_range[data_label][1] for data_label in data_labels}
        else:
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
                                                                 data_indexes = item.data_indexes))
                elif isinstance(series[0], Slot):
                    normalized_series.append(series[0].__class__(start = item.start,
                                                                 unit = item.unit,
                                                                 data = normalized_data,
                                                                 data_indexes = item.data_indexes))
                else:
                    raise NotImplementedError('Working on series other than slots or points not yet implemented')

        if not inplace:
            return normalized_series


class Rescale(Operation):
    """Rescale the data values of a series by a given factor. A series of DataTimePoints or DataTimeSlots is required.

    Args:
       series(Series): the series on which to perform the operation.
       value(float, dict): the value to use as rescaling factor. Can be provided as a single
                           value or as a dictionary of values, one for each data label.
       inplace(bool): if to perform the operation in-place on the series. Defaults to False.

    Returns:
       series or None: the computed series, or None if set to perform the operation in-place.
    """
    def _call(self, series, value, inplace=False):

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
                                                               data_indexes = item.data_indexes))
                elif isinstance(series[0], Slot):
                    rescaled_series.append(series[0].__class__(start = item.start,
                                                               unit = item.unit,
                                                               data = rescaled_data,
                                                               data_indexes = item.data_indexes))
                else:
                    raise NotImplementedError('Working on series other than slots or points not yet implemented')

        if not inplace:
            return rescaled_series


class Offset(Operation):
    """Offset the data values of a series by a given value. A series of DataTimePoints or DataTimeSlots is required.

    Args:
       series(Series): the series on which to perform the operation.
       value(float, dict): the value to use as offset. Can be provided as a single
                           value or as a dictionary of values, one for each data label.
       inplace(bool): if to perform the operation in-place on the series. Defaults to False.

    Returns:
       series or None: the computed series, or None if set to perform the operation in-place.
    """
    def _call(self, series, value, inplace=False):

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
                                                               data_indexes = item.data_indexes))
                elif isinstance(series[0], Slot):
                    rescaled_series.append(series[0].__class__(start = item.start,
                                                               unit = item.unit,
                                                               data = offsetted_data,
                                                               data_indexes = item.data_indexes))
                else:
                    raise NotImplementedError('Working on series other than slots or points not yet implemented')

        if not inplace:
            return rescaled_series


class MAvg(Operation):
    """Compute the moving average on a series. Reduces the series length by a number of values
    equal to the window size. A series of DataTimePoints or DataTimeSlots is required.

    Args:
       series(Series): the series on which to perform the operation.
       window(int): the length of the moving average window.
       inplace(bool): if to perform the operation in-place on the series. Defaults to False.

    Returns:
       series or None: the computed series, or None if set to perform the operation in-place.
    """
    def _call(self, series, window, inplace=False):

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


class Filter(Operation):
    """Filter a series keeping only the data labels provided as argument.

    Args:
       series(Series): the series on which to perform the operation.
       *data_labels(str): the data label(s) to filter against.
    """
    def _call(self, series, *data_labels):

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
    """Slice a series between the given positions or times. A series of DataPoints or DataSlots is required.

    Args:
       series(Series): the series on which to perform the operation.
       from_i(int): the slicing start position. Defaults to None.
       to_i(int): the slicing end position. Defaults to None.
       from_t(bool): the slicing start time (as epoch seconds). Defaults to None.
       to_t(bool): the slicing end time (as epoch seconds). Defaults to None.
       from_dt(bool): the slicing start time (as datetime object). Defaults to None.
       to_dt(bool): the slicing end time (as datetime object). Defaults to None.

    Returns:
        Series: the sliced series.
    """
    def _call(self, series, from_i=None, to_i=None, from_t=None, to_t=None, from_dt=None, to_dt=None):

        if from_t is not None or to_t is not None or from_dt is not None or to_dt is not None:
            from .datastructures import TimeSeries
            if not isinstance(series, TimeSeries):
                raise TypeError('Cannot slice time-based without a time series (got "{}")'.format(series.__class__.__name__))

        froms = 0
        if from_i is not None:
            if not isinstance(from_i, int):
                raise ValueError('The argument from_i must be of type "int" (got "{}")'.format(from_i.__class__.__name__))
            froms+=1
        if from_t is not None:
            if not (isinstance(from_t, int) or isinstance(from_t, float)):
                raise ValueError('The argument from_t must be of type "int" or "float" (got "{}")'.format(from_t.__class__.__name__))
            froms+=1
        if from_dt is not None:
            if not isinstance(from_dt, datetime):
                raise ValueError('The argument from_dt must be of type "datetime" (got "{}")'.format(from_dt.__class__.__name__))
            froms+=1
        if froms > 1:
            raise ValueError('Got more than one from_i, from_t or from_dt: choose one')

        tos = 0
        if to_i is not None:
            if not isinstance(to_i, int):
                raise ValueError('The argument to_i must be of type "int" (got "{}")'.format(to_i.__class__.__name__))
            tos+=1
        if to_t is not None:
            if not (isinstance(to_t, int) or isinstance(to_t, float)):
                raise ValueError('The argument to_t must be of type "int" or "float" (got "{}")'.format(to_t.__class__.__name__))
            tos+=1
        if to_dt is not None:
            if not isinstance(to_dt, datetime):
                raise ValueError('The argument to_dt must be of type "datetime" (got "{}")'.format(to_dt.__class__.__name__))
            tos+=1
        if tos > 1:
            raise ValueError('Got more than one to_i, to_t or to_dt: choose one')

        if from_i is None:
            if from_dt is not None:
                from_t = s_from_dt(from_dt)

        if to_i is None:
            if to_dt is not None:
                to_t = s_from_dt(to_dt)

        if from_i is not None and to_t is not None:
            raise ValueError('Please use the same slicing method between indexes or time-based slicing')

        if to_i is not None and from_t is not None:
            raise ValueError('Please use the same slicing method between indexes or time-based slicing')

        # Handle negative indexes
        if from_i is not None and from_i < 0:
            from_i = len(series) - abs(from_i)
        if to_i is not None and to_i < 0:
            to_i = len(series) - abs(to_i)

        # Handle inverted or equal positions (slight optimization)
        if from_t is not None and to_t is not None:
            if from_t >= to_t:
                return series.__class__()
        if from_i is not None and to_i is not None:
            if from_i >= to_i:
                return series.__class__()

        # Instantiate the new, sliced series
        sliced_series = series.__class__()

        # A) Select sliced series items based on from_t and/or to_t
        if from_t is not None or to_t is not None:
            for item in series:
                if from_t is not None:
                    if item.t < from_t:
                        continue
                if to_t is not None:
                    try:
                        # Slot?
                        if item.end.t > to_t:
                            break
                    except AttributeError:
                        # Point?
                        if item.t >= to_t:
                            break
                # If we are here this item has to be added to the slice
                sliced_series.append(item)

        # B) Select sliced series items based on from_i and/or to_i
        elif from_i is not None or to_i is not None:
            for i, item in enumerate(series):
                if from_i is not None:
                    if i < from_i:
                        continue
                if to_i is not None:
                    if i >= to_i:
                        break
                # If we are here this item has to be added to the slice
                sliced_series.append(item)

        # C) No slicing indexes set, just re-create all the series.
        else:
            for item in series:
                sliced_series.append(item)

        # Re-set reference data as well
        try:
            sliced_series._item_data_reference = sliced_series[0].data
        except (IndexError, AttributeError):
            pass

        return sliced_series


class Merge(Operation):
    """Merge the series given as argument.

    Returns:
        Series: the merged series.
    """

    def _call(self, *series):

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

            # Set reference tz and unit
            tz = seriess[0][0].tz
            if isinstance(reference_series[0], Slot):
                unit = reference_series[0].unit
            else:
                unit = None

            # Support vars
            data = {}
            data_sum = {}
            data_count = {}
            data_indexes = {}
            data_indexes_sum ={}
            data_indexes_count = {}

            # For each time series, get the item in the i-th position given the offset
            for series in seriess:

                # Set the item according to the offset
                item = series[i + offsets[series]]

                # Data
                for data_label in item.data:
                    if data_label not in data_count:
                        data_count[data_label] = 1
                    else:
                        data_count[data_label] += 1
                    if data_label not in data_sum:
                        data_sum[data_label] = item.data[data_label]
                    else:
                        data_sum[data_label] += item.data[data_label]

                # Data indexes
                for data_index in item.data_indexes:
                    if item.data_indexes[data_index] is not None:
                        if data_index not in data_indexes_count:
                            data_indexes_count[data_index] = 1
                        else:
                            data_indexes_count[data_index] += 1
                        if data_index not in data_indexes_sum:
                            data_indexes_sum[data_index] = item.data_indexes[data_index]
                        else:
                            data_indexes_sum[data_index] += item.data_indexes[data_index]

            # Finalize data
            for data_label in data_sum:
                if data_count[data_label] == 1:
                    data[data_label]  = data_sum[data_label]
                else:
                    data[data_label]  = data_sum[data_label]  / data_count[data_label]

            # Finalize data indexes
            for data_index in data_indexes_sum:
                if data_indexes_count[data_index] > 0:
                    data_indexes[data_index]  = data_indexes_sum[data_index]  / data_indexes_count[data_index]
                else:
                    data_indexes[data_index] = None

            if isinstance(reference_series[0], Point):
                result_series.append(seriess[0][0].__class__(t = reference_series[i].t,
                                                             tz = tz,
                                                             data  = data,
                                                             data_indexes = data_indexes))
            elif isinstance(reference_series[0], Slot):
                result_series.append(seriess[0][0].__class__(start = reference_series[i].start,
                                                             unit  = unit,
                                                             data  = data,
                                                             data_indexes = data_indexes))
            else:
                raise NotImplementedError('Working on series other than slots or points not yet implemented')

        return result_series


#=================================
# Operations returning lists
#=================================

class Get(Operation):
    """Get the element of a series at a given position or at a given time.

    Args:
       series(Series): the series on which to perform the operation.
       at_i(int): the position of the item to get. Defaults to None.
       at_t(bool): the time (as epoch seconds) of the item to get. Defaults to None.
       at_dt(bool): the time (as datetime object) of the item to get. Defaults to None.

    Returns:
        object: the item in the given position or at the given time.
    """
    def __call__(self, series, *args, **kwargs):
        if not series:
            raise IndexError
        else:
            return super(Get, self).__call__(series, *args, **kwargs)

    def _call(self, series, at_i=None, at_t=None, at_dt=None):

        ats = 0
        if at_i is not None:
            if not isinstance(at_i, int):
                raise ValueError('The argument at_i must be of type "int" (got "{}")'.format(at_i.__class__.__name__))
            ats+=1
        if at_t is not None:
            if not (isinstance(at_t, int) or isinstance(at_t, float)):
                raise ValueError('The argument at_t must be of type "int" or "float" (got "{}")'.format(at_t.__class__.__name__))
            ats+=1
        if at_dt is not None:
            if not isinstance(at_dt, datetime):
                raise ValueError('The argument at_dt must be of type "datetime" (got "{}")'.format(at_dt.__class__.__name__))
            ats+=1
        if ats > 1:
            raise ValueError('Got more than one at_i, at_t or at_dt: choose one')

        if at_i is None:
            if at_dt is not None:
                at_t = s_from_dt(at_dt)

        # Handle negative index
        if at_i is not None and at_i < 0:
            at_i = len(series) - abs(at_i)

        if at_i is not None:
            return series._item_by_i(at_i)

        elif at_t is not None:
            return series._item_by_t(at_t)

        else:
            raise ValueError('No more at_i, at_t or at_dt set')


class Select(Operation):
    """Select one or more items of the series given an SQL-like query. This is a preliminary
    functionality supporting only the equality. A series of DataPoints or DataSlots is required.

    Args:
       series(Series): the series on which to perform the operation.
       query(str): the query.

    Returns:
        list: the selected items of the series.
    """
    def _call(self, series, query):

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




