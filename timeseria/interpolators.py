# -*- coding: utf-8 -*-
"""Interpolators."""

# Setup logging
import logging
logger = logging.getLogger(__name__)


class Interpolator():
    """A generic interpolator.

        Args:
            series(TimeSeries): the series on which to initialize the interpolator.
    """

    def __init__(self, series):
        logger.debug('Initialized %s interpolator on series "&s"'.format(self.__class__.__name__, series))
        self.series = series

    def evaluate(self, at, prev_i, next_i):
        """Evaluate the interpolator (compute the value).

            Args:
                at(float): the time value (as epoch seconds) at which to evaluate the interpolator.
                prev_i(int): the previous position of the item in the series.
                next_i(int): the next position of the item in the series.
        """
        raise NotImplementedError('This interpolator is not implemented')


class LinearInterpolator(Interpolator):
    """A linear interpolator.

        Args:
            series(TimeSeries): the series on which to initialize the interpolator.
    """

    def evaluate(self, at, prev_i=None, next_i=None):

        if prev_i is not None and next_i is None:
            raise ValueError('If you provide the prev_i you also need to provide the next_i')
        if next_i is not None  and prev_i is None:
            raise ValueError('If you provide the next_i you also need to provide the prev_i')

        if prev_i is None and next_i is None:
            logger.warning('You are not providing the prev_i and next_i, this will cause the interpolator to look for them and likely introduce a slow down')

            # Search for the prev_i:
            for i,item in enumerate(self.series):
                if item.t > at:
                    prev_i = i-1
                    next_i = i
                    break

            if next_i is None:
                raise ValueError('Trying to interpolate a point outside the series: prev_i="{}", next_i="{}", at="{}", series="{}"'.format(prev_i, next_i, at, self.series))

        else:
            if not self.series[prev_i].t < at < self.series[next_i].t:
                raise ValueError('Trying to interpolate a point outside the prev_i / next_i interval')

        # Shortcuts
        prev_point = self.series[prev_i]
        next_point = self.series[next_i]

        # Compute the increment with respect to the interpolation
        interpolated_data = self.series[0].data.__class__()

        # Example:
        # Indexes  0 1 2 3 4 5 6 7
        # Coords:  1 2 3 4 5 6 7 8
        # Values   1 2 x x x 6 7 8
        # prev_i = 1, next_i = 5
        # at = 3 (i=2)
        # interpolation: 4

        for label in self.series.data_labels():

            # Compute the "growth" ratio
            diff = next_point.data[label] - prev_point.data[label]
            delta_t = next_point.t - prev_point.t
            ratio = diff / delta_t

            # Compute the value of the data for the new point
            interpolated_data[label] = prev_point.data[label] + ((at-prev_point.t)*ratio)


        return interpolated_data


class UniformInterpolator(Interpolator):
    """A unform interpolator.

        Args:
            series(TimeSeries): the series on which to initialize the interpolator.
    """
    def evaluate(self, at, prev_i=None, next_i=None):

        if prev_i and next_i is None:
            raise ValueError('If you provide the prev_i you also need to provide the next_i')
        if next_i and prev_i is None:
            raise ValueError('If you provide the next_i you also need to provide the prev_i')

        if not prev_i:
            logger.warning('You are not providing the prev_i and next_i, this will cause the interpolator to look for them and likely introduce a slow down')

            # Search for the prev_i:
            for i,item in enumerate(self.series):
                if item.t > at:
                    prev_i = i-1
                    next_i = i
                break

            if next_i is None:
                raise ValueError('Trying to interpolate a point outside the series')

        else:
            if not self.series[prev_i].t < at < self.series[next_i].t:
                raise ValueError('Trying to interpolate a point outside the prev_i / next_i interval')

        # Shortcuts
        prev_point = self.series[prev_i]
        next_point = self.series[next_i]

        # Compute the increment with respect to the interpolation
        interpolated_data = self.series[0].data.__class__()

        for label in self.series.data_labels():
            interpolated_data[label] = prev_point.data[label] + ((next_point.data[label] - prev_point.data[label]) /2)

        return interpolated_data



