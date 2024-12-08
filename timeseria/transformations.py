# -*- coding: utf-8 -*-
"""Series transformations as resampling and aggregation."""

from propertime.utils import dt_from_s, s_from_dt, as_tz
from datetime import datetime
from .datastructures import Point, Slot, DataTimeSlot, TimePoint, DataTimePoint, TimeSeries, TimeSeriesView
from .utils import _compute_data_loss, _compute_validity_regions
from .operations import avg
from .units import TimeUnit
from .exceptions import ConsistencyException
from . import operations as operations_module
from .interpolators import LinearInterpolator

# Setup logging
import logging
logger = logging.getLogger(__name__)


#==============================
# Support classes and functions
#==============================

class _TimeSeriesDenseView(TimeSeriesView):
    """A time series view which is "dense". Only used internally.

    :meta private:
    """
    def __init__(self, series, from_i, to_i, from_t=None, to_t=None, dense=False, interpolator_class=None):
        self.series = series
        self.from_i = from_i
        self.to_i = to_i
        self.len = None
        self.new_points = {}
        self.from_t = from_t
        self.to_t=to_t
        self.dense=dense
        if self.dense:
            if not interpolator_class:
                raise ValueError('If requesting a dense view, you must provide an interpolator')
            self.interpolator = interpolator_class(series)
        else:
            self.interpolator = None

    def __getitem__(self, i):
        if self.dense:
            raise NotImplementedError('Getting items by index on dense views is not supported. Use the iterator instead.')

        if i>=0:
            return self.series[self.from_i + i]
        else:
            return self.series[self.to_i - abs(i)]

    def __iter__(self):
        self.count = 0
        self.prev_was_new = False
        return self

    def __next__(self):

        this_i = self.count + self.from_i

        if this_i >= self.to_i:

            # If reached the end, stop
            raise StopIteration

        elif self.count == 0 or not self.dense:

            # If first point or not dense, just return
            self.count += 1
            return self.series[this_i]

        else:

            # Otherwise check if we have to add new missing points
            if self.prev_was_new:

                # If we just created a new missing point, simply return it
                self.prev_was_new = False
                self.count += 1
                return self.series[this_i]

            else:

                # Check if we have a gap and thus have to add a new missing point
                prev_point = self.series[this_i-1]
                this_point = self.series[this_i]

                if prev_point.valid_to < this_point.valid_from:

                    # We have a gap, compute the new point validity
                    if self.from_t is not None and prev_point.valid_to < self.from_t:
                        new_point_valid_from = self.from_t
                    else:
                        new_point_valid_from = prev_point.valid_to

                    if self.to_t is not None and this_point.valid_from > self.to_t:
                        new_point_valid_to = self.to_t
                    else:
                        new_point_valid_to = this_point.valid_from

                    # Compute the new point timestamp
                    new_point_t = new_point_valid_from + (new_point_valid_to-new_point_valid_from)/2

                    # Can we use cache?
                    if new_point_t in self.new_points:
                        self.prev_was_new = True
                        return self.new_points[new_point_t]

                    # Log new point creation
                    logger.debug('New point t=,%s validity: [%s,%s]',new_point_t, new_point_valid_from,new_point_valid_to)

                    # Compute the new point values using the interpolator
                    new_point_data = self.interpolator.evaluate(new_point_t, prev_i=this_i-1, next_i=this_i)

                    # Create the new point
                    new_point = this_point.__class__(t = new_point_t, data = new_point_data)
                    new_point.valid_from = new_point_valid_from
                    new_point.valid_to = new_point_valid_to
                    new_point._interpolated = True

                    # Set flag
                    self.prev_was_new = True

                    # Add to cache
                    self.new_points[new_point_t] = new_point

                    # ..and return it
                    return new_point

                else:
                    # Return this point if no gaps
                    self.count += 1
                    return this_point

    def __len__(self):
        if not self.dense:
            return self.to_i-self.from_i
        else:
            if self.len is None:
                self.len=0
                for _ in self:
                    self.len+=1
            return self.len

    def __repr__(self):
        if not self.series:
            return 'Empty time series view'
        else:
            return 'Time series view'

    @property
    def item_type(self):
        for item in self:
            return item.__class__

    @property
    def resolution(self):
        return self.series.resolution

    def data_labels(self):
        return self.series.data_labels()


def _compute_new(target, series, from_t, to_t, slot_first_point_i, slot_last_point_i, slot_prev_point_i, slot_next_point_i,
                 unit, point_validity, timezone, fill_with, force_data_loss, series_data_indexes, force_compute_data_loss,
                 interpolator_class, operations=None):
    """Support function for computing new items.

    :meta private:
    """

    # Note: if slot_first_point_i < slot_last_point_i, this means that the prev and next are outside the slot.
    # It is not a bug, it is how the system works. Perhaps we could pass here the slot_prev_i and sòlot_next_i
    # to make it more clear to avoid some confusion.
    logger.debug('Called compute new')
    logger.debug('slot_first_point_i=%s, slot_last_point_i=%s, ', slot_first_point_i, slot_last_point_i)
    logger.debug('slot_prev_point_i=%s, slot_next_point_i=%s, ', slot_prev_point_i, slot_next_point_i)

    # Support vars
    interval_duration = to_t-from_t
    data = {}
    data_labels = series.data_labels()

    # The prev can be None as lefts are included (edge case)
    if slot_prev_point_i is None:
        slot_prev_point_i = slot_first_point_i

    # Create a view of the series containing the slot datapoints plus the prev and next.
    # Note that the "+1" in the "to_i" argument below is because the right is excluded.
    series_dense_view_extended  = _TimeSeriesDenseView(series, from_i=slot_prev_point_i, to_i=slot_next_point_i+1,
                                                       from_t=from_t, to_t=to_t, dense=True, interpolator_class=interpolator_class)

    # Compute the data loss for the new element. This is forced if first or last point
    data_loss = _compute_data_loss(series_dense_view_extended,
                                  from_t = from_t,
                                  to_t = to_t,
                                  sampling_interval = point_validity,
                                  force = force_compute_data_loss)
    logger.debug('Computed data loss: "{}"'.format(data_loss))

    # For each point, attach the "weight": how much it contributes to the new point or slot
    for point in series_dense_view_extended:

        # Weight zero if completely outside the interval, which is required if in the operation
        # there is the need of knowing the next(or prev) value, even if it was far away
        if point.valid_to < from_t:
            point.weight = 0
            continue
        if point.valid_from >= to_t:
            point.weight = 0
            continue

        # Set this point valid from/to according to the interval boundaries
        this_point_valid_from = point.valid_from if point.valid_from >= from_t else from_t
        this_point_valid_to = point.valid_to if point.valid_to < to_t else to_t

        # Set weight
        point.weight = (this_point_valid_to-this_point_valid_from)/interval_duration

        # Log
        logger.debug('Set series point @ %s weight: %s (data: %s)', point.dt, point.weight, point.data)

    # If creating slots, create also the slice of the series containing only the slot datapoints
    if target == 'slot':

        if slot_first_point_i is None and slot_last_point_i is None:
            # If we have no datapoints at all in the slot, we must create one to let operations not
            # supporting weights to properly work. If we have a full data loss, this can be just the
            # middle point from the series_dense_view_extended. Otherwise, we must compute it from
            # scratch as we are upsamping with no data loss but still no points belonging to the slot.

            if data_loss==1:
                series_dense_view = None
                for point in series_dense_view_extended:
                    try:
                        if point._interpolated:
                            if series_dense_view:
                                raise ConsistencyException('Found more than one reconstructed point in a fully missing slot')
                            series_dense_view = series.__class__(point)
                    except AttributeError:
                        pass
                if not series_dense_view:
                    raise ConsistencyException('Could not find any reconstructed point in a fully missing slot')
            else:
                # Just create a point based on the series_dense_view_extended weights (TODO: are we including the interpolated?):
                new_point_data = {data_label:0 for data_label in data_labels}
                for point in series_dense_view_extended:
                    for data_label in data_labels:
                        point.data[data_label] += point.data[data_label] * point.weight
                new_point_t = (to_t - from_t) /2
                series_dense_view = TimeSeries(DataTimePoint(t=new_point_t, data=new_point_data, tz=series.tz))

        else:
            # Create a view of the original series to provide only the datapoints belonging to the slot.
            # Note that the "+1" in the "to_i" argument below is because the right is excluded.
            series_dense_view = _TimeSeriesDenseView(series, from_i=slot_first_point_i, to_i=slot_last_point_i+1,
                                                     from_t=from_t, to_t=to_t, dense=True, interpolator_class=interpolator_class)

    # Compute point data
    if target=='point':

        # Compute the (weighted) average of all point contributions
        avgs = avg(series_dense_view_extended)

        # ...and assign them to the data value
        if isinstance(avgs, dict):
            data = {data_label:avgs[data_label] for data_label in data_labels}
        else:
            data = {data_labels[0]: avgs}

    #  Compute slot data
    elif target=='slot':

        if data_loss == 1 and fill_with:
            for data_label in data_labels:
                for operation in operations:
                    data['{}_{}'.format(data_label, operation.__name__)] = fill_with

        else:
            # Handle operations
            for operation in operations:
                try:
                    operation_supports_weights = operation._supports_weights
                except AttributeError:
                    # This is because user-defined operations can be even simple functions
                    # or based on custom classes without the _supports_weights attribute
                    operation_supports_weights = False

                if operation_supports_weights:
                    operation_data = operation(series_dense_view_extended)
                else:
                    operation_data = operation(series_dense_view)

                # ...and add to the data
                if isinstance(operation_data, dict):
                    for result_data_label in operation_data:
                        data['{}_{}'.format(result_data_label, operation.__name__)] = operation_data[result_data_label]
                else:
                    data['{}_{}'.format(data_labels[0], operation.__name__)] = operation_data

        if not data:
            raise Exception('No data computed at all?')

    else:
        raise ValueError('No idea how to compute a new "{}"'.format(target))

    # Do we have a force data_loss?
    if force_data_loss is not None:
        data_loss = force_data_loss

    # Create the new item
    if target=='point':
        new_element = DataTimePoint(t = (from_t+((interval_duration)/2)),
                                    tz = timezone,
                                    data  = data,
                                    data_loss = data_loss)
    elif target=='slot':
        # Create the DataTimeSlot
        new_element = DataTimeSlot(start = TimePoint(t=from_t, tz=timezone),
                                   end   = TimePoint(t=to_t, tz=timezone),
                                   unit  = unit,
                                   data  = data,
                                   data_loss = data_loss)
    else:
        raise ValueError('No idea how to create a new "{}"'.format(target))

    # Handle data_indexes
    for index in series_data_indexes:

        # Skip the data loss as it is recomputed with different logics
        if index == 'data_loss':
            continue

        if series_dense_view_extended:
            index_total = 0
            index_total_weights = 0
            for point in series_dense_view_extended:

                # Get index value
                try:
                    index_value = point.data_indexes[index]
                except:
                    pass
                else:
                    if index_value is not None:
                        index_total_weights += point.weight
                        index_total += index_value * point.weight

            # Compute the new index value (if there were data_indexes not None)
            if index_total_weights > 0:

                # Rescale with respect to the total_weights (note: some points could have no index at all)
                new_element_index_value = index_total/index_total_weights

            else:
                new_element_index_value = None

        else:
            new_element_index_value = None

        # Set the index in the data indexes
        new_element.data_indexes[index] = new_element_index_value

    # Return
    return new_element


#==========================
#  Base Transformation
#==========================

class Transformation(object):
    """A generic transformation."""

    @classmethod
    def __str__(cls):
        return '{} transformation'.format(cls.__name__.replace('Transformation',''))

    def process(self, series, start=None, end=None, validity=None, include_extremes=False, fill_with=None, force_data_loss=None):
        """Start the transformation process.

            Args:
                series(TimeSeries): the time series to transform.
                start(float, datetime): the start time of the transformation process. If not set, then it is set automatically
                                        based on first item of the series. Defaults to None.
                end(float, datetime): the end time of the transformation process. If not set, then it is set automatically
                                      based on first item of the series. Defaults to None.
                validity(float): the validity (sampling) interval of the original data points. Defaults to auto-detect.
                include_extremes(bool): if to include the first and last items in the transformed time series, which might
                                        not have enough data when being created. Defaults to False.
                fill_with(): a fixed value to fill the data of the items showing a full data loss.
                force_data_loss(float): Force a specific data loss value for all the new series items.
        """

        target = self.target

        if not isinstance(series, TimeSeries):
            raise NotImplementedError('Transformations work only with TimeSeries data for now (got "{}")'.format(series.__class__.__name__))

        if not (issubclass(series.item_type, Point) or issubclass(series.item_type, Slot)):
                raise TypeError('Series items are not Points nor Slots, cannot compute any transformation')

        # Checks
        if include_extremes:
            raise NotImplementedError('Including the extremes is not yet implemented')
        if not issubclass(series.item_type, DataTimePoint):
            raise TypeError('Can process only time series of DataTimePoints, got time series of items type "{}"'.format(series.item_type.__name__))
        if not series:
            raise ValueError('Cannot process empty series')
        if target not in ['points', 'slots']:
            raise ValueError('Don\'t know how to target "{}"'.format(target))

        # Log
        logger.debug('Computing from/to with include_extremes= %s', include_extremes)

        # Handle start/end. If creating points, we will move the start back of half unit
        # and the end forward of half unit as well, as the point will be in the center
        from_t = None
        to_t = None
        from_dt = None
        to_dt = None
        if start is not None:
            if isinstance(start, datetime):
                from_dt = start
            else:
                try:
                    from_t = float(start)
                except:
                    raise ValueError('Cannot use "{}" as start value, not a datetime nor an epoch timestamp'.format(start))
        if end is not None:
            if isinstance(end, datetime):
                to_dt = end
            else:
                try:
                    to_t = float(end)
                except:
                    raise ValueError('Cannot use "{}" as end value, not a datetime nor an epoch timestamp'.format(end))

        # Set "from". TODO: check against shifted rounding if points?
        if from_t:
            from_dt = dt_from_s(from_t)
            if from_dt != self.time_unit.round(from_dt):
                raise ValueError('The provided from_t is not consistent with the self.time_unit of "{}" (Got "{}")'.format(self.time_unit, from_t))
        elif from_dt:
            from_t = s_from_dt(from_dt)
            if from_dt != self.time_unit.round(from_dt):
                raise ValueError('The provided from_dt is not consistent with the self.time_unit of "{}" (Got "{}")'.format(self.time_unit, from_dt))
        else:
            if target == 'points':
                from_t = series[0].t  - (self.time_unit.as_seconds() /2)
                from_dt = dt_from_s(from_t, tz=series.tz)
                from_dt = self.time_unit.round(from_dt, how='floor' if include_extremes else 'ceil')
                from_t = s_from_dt(from_dt) + (self.time_unit.as_seconds() /2)
                from_dt = dt_from_s(from_t)

            elif target == 'slots':
                from_t = series[0].t
                from_dt = dt_from_s(from_t, tz=series.tz)
                if from_dt == self.time_unit.round(from_dt):
                    # Only for the start, and only for the slots, if the first point is
                    # exactly equal to the first slot start, leave it as it is to include it.
                    pass
                else:
                    from_dt = self.time_unit.round(from_dt, how='floor' if include_extremes else 'ceil')
                from_t = s_from_dt(from_dt)
                from_dt = dt_from_s(from_t)
            else:
                raise ValueError('Don\'t know how to target "{}"'.format(target))

        # Set "to". TODO: check against shifted rounding if points?
        if to_t:
            to_dt = dt_from_s(to_t)
            if to_dt != self.time_unit.round(to_dt):
                raise ValueError('The provided to_t is not consistent with the self.time_unit of "{}" (Got "{}")'.format(self.time_unit, to_t))
        elif to_dt:
            to_t = s_from_dt(to_dt)
            if to_dt != self.time_unit.round(to_dt):
                raise ValueError('The provided to_dt is not consistent with the self.time_unit of "{}" (Got "{}")'.format(self.time_unit, to_dt))
        else:
            if target == 'points':
                to_t = series[-1].t  + (self.time_unit.as_seconds() /2)
                to_dt = dt_from_s(to_t, tz=series.tz)
                from_dt = self.time_unit.round(from_dt, how='ceil' if include_extremes else 'floor')
                to_t = s_from_dt(to_dt) - (self.time_unit.as_seconds() /2)
                to_dt = dt_from_s(to_t)
            elif target == 'slots':
                to_t = series[-1].t
                to_dt = dt_from_s(to_t, tz=series.tz)
                from_dt = self.time_unit.round(from_dt, how='ceil' if include_extremes else 'floor')
                to_t = s_from_dt(to_dt)
                to_dt = dt_from_s(to_t)
            else:
                raise ValueError('Don\'t know how to target "{}"'.format(target))

        # Set/fix timezone
        from_dt = as_tz(from_dt, series.tz)
        to_dt = as_tz(to_dt, series.tz)

        # Log
        logger.debug('Computed from: %s', from_dt)
        logger.debug('Computed to: %s',to_dt)

        # Automatically detect validity if not set
        if validity is None:
            validity = series._autodetected_sampling_interval
            logger.info('Using auto-detected sampling interval: %ss', validity)

        # Check if upsamplimg (with some tolerance):
        if validity > (self.time_unit.as_seconds(series[0].dt) * 1.10):
            logger.warning('You are upsampling, which is not well tested yet. Expect potential issues.')

        # Compute validity regions
        validity_regions = _compute_validity_regions(series, sampling_interval=validity)

        # Log
        logger.debug('Started resampling from "%s" (%s) to "%s" (%s)', from_dt, from_t, to_dt, to_t)

        # Set some support vars
        slot_start_t = None
        slot_start_dt = None
        slot_end_t = None
        slot_end_dt = None
        process_ended = False
        slot_prev_point_i = None
        new_series = TimeSeries()

        # Set timezone
        timezone  = series.tz
        logger.debug('Using timezone "%s"', timezone)

        # Counters
        count = 0
        first = True

        # Data_indexes shortcut
        series_data_indexes = series._all_data_indexes()

        # Set operations if slots
        if target == 'slots':
            operations  = self.operations
        else:
            operations = None

        # Now go trough all the data in the time series
        for i, point in enumerate(series):

            logger.debug('Processing i=%s: %s', i, point)

            # Increase counter
            count += 1

            # Attach validity TODO: do we want a generic series method, or maybe to be always computed in the series?
            point.valid_from = validity_regions[point.t][0]
            point.valid_to = validity_regions[point.t][1]
            logger.debug('Attached validity to %s', point.t)

            # Pretend there was a slot before if we are at the beginning.
            if slot_end_t is None:
                slot_end_t = from_t

            # First, check if we have some points to discard at the beginning
            if point.t < from_t:
                # If we are here it means we are going data belonging to a previous slot
                logger.debug('Discarding as before start')
                # NOTE: here we set the prev index
                slot_prev_point_i = i
                continue

            # Similar concept for the end
            # TODO: handle streaming mode, perhaps add and check on to_t not None?
            if point.t >= to_t:
                if process_ended:
                    logger.debug('Discarding as after end')
                    continue

            # Is the current slot outdated? (are we processing a datapoint falling outside the current slot?)
            if point.t >= slot_end_t:

                # This approach is to detect if the current slot is "outdated" and spin a new one if so.
                # Works only for slots at the beginning and in the middle, but not for the last slot
                # or the missing slots at the end which need to be closed down here
                logger.debug('Detected outdated slot')

                # Keep spinning new slots until the current data point falls in one of them.
                # NOTE: Read the following "while" more as an "if" which can also lead to spin multiple
                # slot if there are empty slots between the one being closed and the point.dt.
                while slot_end_t <= point.t:
                    logger.debug('Checking for end %s with point %s', slot_end_t, point.t)

                    if slot_start_t is None:
                        # If we are in the pre-first slot, just silently spin a new slot
                        pass

                    else:

                        # Set slot next point index (us, as we triggered a new slot)
                        slot_next_point_i = i

                        # The slot_prev_point_i is instead always defined in the process and correctly set.
                        slot_prev_point_i = slot_prev_point_i

                        logger.debug('Got slot_prev_point_i as %s',slot_prev_point_i)
                        logger.debug('Got slot_next_point_i as %s',slot_next_point_i)

                        # Now set slot first and last point, but beware boundaries
                        if slot_prev_point_i is not None and abs(slot_next_point_i - slot_prev_point_i) == 1:

                            # No points for the slot at all
                            slot_first_point_i = None
                            slot_last_point_i = None

                        else:

                            # Set first
                            if slot_prev_point_i is None:
                                # Edge case where there is no prev as the time series starts with a point
                                # placed exactly on the slot start, and since lefts are included we take it.
                                slot_first_point_i = 0
                            else:
                                if series[slot_prev_point_i+1].t >= slot_start_t and series[slot_prev_point_i+1].t < slot_end_t:
                                    slot_first_point_i = slot_prev_point_i+1
                                else:
                                    slot_first_point_i = None

                            # Set last
                            if series[slot_next_point_i-1].t >= slot_start_t and series[slot_next_point_i-1].t < slot_end_t:
                                slot_last_point_i = slot_next_point_i-1
                            else:
                                slot_last_point_i = None


                        logger.debug('Set slot_first_point_i to %s',slot_first_point_i)
                        logger.debug('Set slot_last_point_i to %s',slot_last_point_i)

                        # Log the new slot
                        # slot_first_point_i-1 is the "prev"
                        # slot_last_point_i+1 is the "next" (and the index where we are at the moment)
                        logger.debug('This slot is closed: start=%s (%s) and end=%s (%s). Now computing it..', slot_start_t, slot_start_dt, slot_end_t, slot_end_dt)

                        # Compute the new item...
                        new_item = _compute_new(target = 'point' if target=='points' else 'slot',
                                                series = series,
                                                slot_first_point_i = slot_first_point_i,
                                                slot_last_point_i = slot_last_point_i ,
                                                slot_prev_point_i = slot_prev_point_i,
                                                slot_next_point_i = slot_next_point_i,
                                                from_t = slot_start_t,
                                                to_t = slot_end_t,
                                                unit = self.time_unit,
                                                point_validity = validity,
                                                timezone = timezone,
                                                fill_with = fill_with,
                                                force_data_loss = force_data_loss,
                                                series_data_indexes = series_data_indexes,
                                                force_compute_data_loss = True if first else False,
                                                interpolator_class=self.interpolator_class,
                                                operations = operations)

                        # Set first to false
                        if first:
                            first = False

                        # .. and append results
                        if new_item:
                            logger.debug('Computed new item: %s',new_item )
                            new_series.append(new_item)

                    # Create a new slot. This is where all the "conventional" time logic kicks-in, and where the timezone is required.
                    slot_start_t = slot_end_t
                    slot_start_dt = dt_from_s(slot_start_t, tz=timezone)
                    # TODO: if not calendar time unit, we can just use seconds and maybe speed up a bit.
                    slot_end_dt = slot_start_dt + self.time_unit
                    slot_end_t = s_from_dt(slot_end_dt)

                    # Reassign prev (if we have to)
                    if i>0 and series[i-1].t < slot_start_t:
                        slot_prev_point_i = i-1

                    # Log the newly spinned slot
                    logger.debug('Spinned a new slot, start=%s (%s), end=%s (%s)', slot_start_t, slot_start_dt, slot_end_t, slot_end_dt)

                # If last slot mark process as completed
                if point.dt >= to_dt:
                    process_ended = True

        if target == 'points':
            logger.info('Resampled %s DataTimePoints in %s DataTimePoints', count, len(new_series))
        else:
            if isinstance(series[0], DataTimePoint):
                logger.info('Aggregated %s points in %s slots', count, len(new_series))
            else:
                logger.info('Aggregated %s slots in %s slots', count, len(new_series))

        return new_series


#==========================
#  Resampler
#==========================

class Resampler(Transformation):
    """Resampling transformation.

    Args:
        unit(TimeUnit. str): the time unit corresponding to the new sampling interval, or its string representation.
        interpolator_class(Interpolator): the interpolator to use for the resampling process. Defaults to :obj:`LinearInterpolator`.
    """

    target = 'points'

    def __init__(self, unit, interpolator_class=LinearInterpolator):

        # Handle unit
        if isinstance(unit, TimeUnit):
            self.time_unit = unit
        else:
            self.time_unit = TimeUnit(unit)

        unit_type = ''.join([char for char in str(self.time_unit) if not char.isdigit()])
        if unit_type in ['Y', 'M', 'D']:
            raise ValueError('Sorry, time units involving calendar components are not supported by the Resampler (got "{}"). Use the Slotter instead.'.format(self.time_unit))

        # Set the interpolator
        self.interpolator_class = interpolator_class

    def process(self, series, *args, **kwargs):
        return super(Resampler, self).process(series, *args, **kwargs)


#==========================
#   Aggregator
#==========================

class Aggregator(Transformation):
    """Aggregation transformation.

    Args:
        unit(TimeUnit. str): the time unit corresponding to the aggregation slots, or its string representation.
        operations(list): the list of operations to perform when aggregating the data. Supports any operation of the ``operations``
                          module, as well as custom ones, provided they take as input a series and return a scalar.
        interpolator_class(Interpolator): the interpolator to use to reconstruct missing samples. Defaults to :obj:`LinearInterpolator`.
    """

    target = 'slots'

    def __init__(self, unit, operations=[avg], interpolator_class=LinearInterpolator):

        # Handle unit
        if isinstance(unit, TimeUnit):
            self.time_unit = unit
        else:
            self.time_unit = TimeUnit(unit)

        if not operations:
            raise ValueError('No operations set, cannot slot.')

        # Handle operations
        operations_input = operations
        operations = []
        for operation in operations_input:
            if isinstance(operation, str):
                try:
                    operation = getattr(operations_module, operation)
                except:
                    raise
            operations.append(operation)
        self.operations = operations

        # Set the interpolator
        self.interpolator_class = interpolator_class

    def process(self, series, *args, **kwargs):
        return super(Aggregator, self).process(series, *args, **kwargs)

