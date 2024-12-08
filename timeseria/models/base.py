# -*- coding: utf-8 -*-
"""Base model classes."""

import os
import json
import uuid
import functools
import statistics
from propertime.utils import now_s, dt_from_s, s_from_dt
from datetime import datetime
from pandas import DataFrame
import shutil
import copy

from ..exceptions import NotFittedError, AlreadyFittedError
from ..utils import _check_timeseries, _check_resolution, _check_data_labels, _item_is_in_range
from ..units import TimeUnit
from ..datastructures import Point, Series, TimeSeries, TimeSeriesView

# Setup logging
import logging
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings as default behavior
try:
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except (ImportError,AttributeError):
    pass


class TooMuchDataLoss(Exception):
    pass

#======================
#  Base Model
#======================

class Model():
    """A generic model. This can be either a stateless model, where all the information is coded and there are no parameters, or a
    stateful (parametric) model, where there are a number of parameters which can be both set manually or learnt (fitted) from the data.

    All models expose a ``predict()``, ``apply()`` and ``evaluate()`` methods, while parametric models also provide a ``save()`` method to
    store the parameters of the model, and an optional ``fit()`` method if the parameters are to be learnt form data. In this case also a
    ``cross_validate()`` method is available.

    Series resolution and data labels consistency are enforced between all methods and save/load operations.
    """

    def __init__(self):

        # Set type
        try:
            self.data
        except:
            # No data (parameters) for the model. Check if there is a fit implemented
            if self._is_fit_implemented():
                # Yes: the model is parameter and the parameters
                # are still to be set via the fit
                self._type = 'parametric'
            else:
                # No: the model has no parameters at all
                self._type = 'non-parametric'
        else:
            # If we have data, it means that there are parameters and
            # the model is parametric.
            self._type = 'parametric'

        # Set the model ID
        id = str(uuid.uuid4())
        self.fitted = False
        try:
            self.data['id'] = id
        except AttributeError:
            self.data = {'id': id}

    @property
    def id(self):
        """A unique identifier for the model. Only for parametric models."""
        if not self._is_parametric():
            raise TypeError('Non-parametric models have no ID') # TODO: is this the right exception?
        else:
            return self.data['id']

    @classmethod
    def load(cls, path):
        """Instantiate the model loading its parameters from a path.

            Args:
                path(str): the path from which to load the model.

            Returns:
                Model: the model instance.
        """
        # Do we have to load the model (if parametric?)
        #if not cls._is_parametric():
        #    raise ValueError('Loading a non-parametric model from a path does not make sense')

        # Create the model
        model = cls()

        # Set the data (replaces the model ID also)
        with open(path+'/data.json', 'r') as f:
            model.data = json.loads(f.read())

        # TODO: the "fitted" does not apply for models with parameters but with no fit.. fix me.
        model.fitted = True

        # Convert resolution to TimeUnit if any. TODO: right now only TimeSeries have the resolution,
        # but might change in case of adding the resolution attribute also to generic series.
        if 'resolution' in model.data:
            model.data['resolution'] = TimeUnit(model.data['resolution'])

        return model

    def save(self, path):
        """Save the model in the given path. The model is saved in "directory format",
        meaning that a new directory will be created containing the data for the model.

        Args:
            path(str): the path where to save the model.
        """

        # Is this model parametric?
        if not self._is_parametric():
            raise TypeError('Saving a non-parametric model from a path does not make sense') # TODO: is this the right exception?

        # Ensure the model is fitted if it has to
        if self._is_fit_implemented() and not self.fitted:
            raise NotFittedError()

        if not path:
            raise ValueError('Got empty path, cannot save')

        if os.path.exists(path):
            raise ValueError('The path "{}" already exists'.format(path))

        # Prepare model dir and dump data as json
        os.makedirs(path)
        model_data_file = '{}/data.json'.format(path)

        # Temporary change the resolution to its string representation (if any)
        if 'resolution' in self.data:
            resolution_obj = self.data['resolution']
            self.data['resolution'] = str(resolution_obj)

        # Write the data
        try:
            with open(model_data_file, 'w') as f:
                f.write(json.dumps(self.data))
        finally:
            # In any case revert resolution back to object (if any)
            if 'resolution' in self.data:
                self.data['resolution'] = resolution_obj

        logger.info('Saved model with id "%s" in "%s"', self.data['id'], path)


    def _is_fit_implemented(self):
        """If the model has a fit method implemented or not."""
        try:
            self.fit(series=None)
        except NotImplementedError:
            return False
        except Exception:
            return True
        else:
            return True

    def _is_parametric(self):
        """If the model is parametric or not."""
        if self._type == 'parametric':
            return True
        else:
            return False

    @staticmethod
    def fit_method(fit_method):
        """:meta private:"""
        @functools.wraps(fit_method)
        def do_fit(self, series, *args, **kwargs):

            if self.fitted:
                raise AlreadyFittedError('This model is already fitted. Use the fit_update() method if you want to update it with '
                                         + 'new data (where implemented). If you instead want to re-fit it, create a new instance.')

            # Check data
            if not isinstance(series, TimeSeries):
                raise TypeError('Models work only with TimeSeries data for now (got "{}")'.format(series.__class__.__name__))

            # If TimeSeries data, check it
            if isinstance(series, TimeSeries):
                _check_timeseries(series)

                # Set resolution
                try:
                    self.data['resolution'] = series.resolution
                except AttributeError:
                    pass

                # Set data labels
                try:
                    self.data['data_labels'] = series.data_labels()
                except AttributeError:
                    pass

            # Call fit logic
            fit_output = fit_method(self, series, *args, **kwargs)

            self.data['fitted_at'] = now_s()
            self.fitted = True

            return fit_output

        return do_fit

    def fit(self, series, verbose=False):
        """Fit the model on a series.

            Args:
                series(TimeSeries): the series on which to fit the model.
                verbose(str): if to be verbose when fitting.
        """
        raise NotImplementedError('Fitting this model is not implemented')


    @staticmethod
    def fit_update_method(fit_update_method):
        """:meta private:"""
        @functools.wraps(fit_update_method)
        def do_fit_update(self, series, *args, **kwargs):

            if not self.fitted:
                raise NotFittedError('Cannot update the fit for a non-fitted model')

            # Check data
            if not isinstance(series, TimeSeries):
                raise TypeError('Models work only with TimeSeries data for now (got "{}")'.format(series.__class__.__name__))

            # If TimeSeries data, check it
            if isinstance(series, TimeSeries):
                _check_timeseries(series)

            # Check resolution
            if 'resolution' in self.data:
                _check_resolution(series, self.data['resolution'])

            # Check data labels
            if 'data_labels' in self.data:
                _check_data_labels(series, self.data['data_labels'])

            # Call update fit logic
            fit_update_output = fit_update_method(self, series, *args, **kwargs)

            # Update fitted at and model id
            self.data['fitted_at'] = now_s()
            self.data['id'] = str(uuid.uuid4())

            return fit_update_output

        return do_fit_update

    def fit_update(self, series, verbose=False, **kwargs):
        """Update the model fit on a series.

            Args:
                series(TimeSeries): the series on which to update the fit of the model.
                verbose(str): if to be verbose when fitting.
        """
        raise NotImplementedError('Updating the fit is not implemented for this model')

    @staticmethod
    def predict_method(predict_method):
        """:meta private:"""
        @functools.wraps(predict_method)
        def do_predict(self, series, *args, **kwargs):

            # Ensure the model is fitted if it has to.
            if self._is_fit_implemented() and not self.fitted:
                raise NotFittedError()

            # Check data
            if not isinstance(series, TimeSeries):
                raise TypeError('Models work only with TimeSeries data for now (got "{}")'.format(series.__class__.__name__))

            # If TimeSeries data, check it
            if isinstance(series, TimeSeries):
                _check_timeseries(series)
                if self._is_parametric():
                    if isinstance(series.item_type, Point) and len(series) == 1:
                        # Do not check if the data is a point time series and has only one item
                        pass
                    else:
                        _check_resolution(series, self.data['resolution'])
                    _check_data_labels(series, self.data['data_labels'])

            # Check context
            if 'context_data' in kwargs and kwargs['context_data'] and ('context_data_labels' not in self.data or not self.data['context_data_labels']):
                raise ValueError('This model does not accept context data')
            if 'context_data_labels' in self.data and self.data['context_data_labels'] and ('context_data' not in kwargs or not kwargs['context_data']):
                raise ValueError('This model requires context data ({})'.format(self.data['context_data_labels']))

            # Call predict logic
            return predict_method(self, series, *args, **kwargs)

        return do_predict

    def predict(self, series):
        """Call the model predict logic on a series.

            Args:
                series(TimeSeries): the series on which to apply the predict logic.

            Returns:
                dict: the predicted data.
        """
        raise NotImplementedError('Predicting with this model is not implemented')

    @staticmethod
    def apply_method(apply_method):
        """:meta private:"""
        @functools.wraps(apply_method)
        def do_apply(self, series, *args, **kwargs):

            # Ensure the model is fitted if it has to.
            if self._is_fit_implemented() and not self.fitted:
                raise NotFittedError()

            # Check data
            if not isinstance(series, TimeSeries):
                raise TypeError('Models work only with TimeSeries data for now (got "{}")'.format(series.__class__.__name__))

            # If TimeSeries data, check it
            if isinstance(series, TimeSeries):
                _check_timeseries(series)
                if self._is_parametric():
                    if isinstance(series.item_type, Point) and len(series) == 1:
                        # Do not check if the data is a point time series and has only one item
                        pass
                    else:
                        _check_resolution(series, self.data['resolution'])
                    _check_data_labels(series, self.data['data_labels'])

            # Call apply logic
            return apply_method(self, series, *args, **kwargs)

        return do_apply

    def apply(self, series):
        """Call the model apply logic on a series.

            Args:
                series(TimeSeries): the series on which to apply the model logic.

            Returns:
                TimeSeries: the series with the results of applying the model.
        """
        raise NotImplementedError('Applying this model is not implemented')

    @staticmethod
    def evaluate_method(evaluate_method):
        """:meta private:"""
        @functools.wraps(evaluate_method)
        def do_evaluate(self, series, *args, **kwargs):

            # Ensure the model is fitted if it has to.
            if self._is_fit_implemented() and not self.fitted:
                raise NotFittedError()

            # Check data
            if not isinstance(series, TimeSeries):
                raise TypeError('Models work only with TimeSeries data for now (got "{}")'.format(series.__class__.__name__))

            # If TimeSeries data, check it
            if isinstance(series, TimeSeries):
                _check_timeseries(series)
                if self._is_parametric():
                    if isinstance(series.item_type, Point) and len(series) == 1:
                        # Do not check if the data is a point time series and has only one item
                        pass
                    else:
                        _check_resolution(series, self.data['resolution'])
                    _check_data_labels(series, self.data['data_labels'])

            # Call evaluate logic
            return evaluate_method(self, series, *args, **kwargs)

        return do_evaluate

    def evaluate(self, series):
        """Call the model evaluate logic on a series.

            Args:
                series(TimeSeries): the series on which to evaluate the model.

            Returns:
                dict: the evaluation results.
        """
        raise NotImplementedError('Evaluating this model is not implemented')

    def cross_validate(self, series, rounds=10, return_full_evaluations=False, **kwargs):
        """Cross validate the model on a series, by default with 10 fit/evaluate rounds.

        All the parameters starting with the ``fit_`` prefix are forwarded to the model ``fit()`` method (without the prefix), and
        all the parameters starting with the ``evaluate_`` prefix are forwarded to the model ``evaluate()`` method (without the prefix).

        Args:
            rounds(int): how many rounds of cross validation to run.
            return_full_evaluations(bool): if to return the full evaluations, one for each round.

        Returns:
                dict: the cross validation results.
        """

        if self.fitted:
            raise NotImplementedError('You are trying to cross-validate a model already fitted, this is not supported.')

        # Check data
        if not isinstance(series, TimeSeries):
            raise TypeError('Models work only with TimeSeries data for now (got "{}")'.format(series.__class__.__name__))

        # If TimeSeries data, check it
        if isinstance(series, TimeSeries):
            _check_timeseries(series)

        # Decouple evaluate from fit args
        fit_kwargs = {}
        evaluate_kwargs = {}
        consumed_kwargs = []
        for kwarg in kwargs:
            if kwarg.startswith('fit_'):
                consumed_kwargs.append(kwarg)
                fit_kwargs[kwarg.replace('fit_', '')] = kwargs[kwarg]
            if kwarg.startswith('evaluate_'):
                consumed_kwargs.append(kwarg)
                evaluate_kwargs[kwarg.replace('evaluate_', '')] = kwargs[kwarg]
        for consumed_kwarg in consumed_kwargs:
            kwargs.pop(consumed_kwarg)

        # Do we still have some kwargs?
        if kwargs:
            raise Exception('Got some unknown args: {}'.format(kwargs))

        # How many items per round?
        round_items = int(len(series) / rounds)
        logger.debug('Items per round: {}'.format(round_items))

        # Start the fit / evaluate loop
        evaluations = []
        fit_update_unaivalable_warned = False
        for i in range(rounds):
            validate_from_i = round_items*i
            validate_to_i =   (round_items*i)+round_items
            validate_from_t = series[validate_from_i].t
            try:
                validate_to_t = series[validate_to_i].t
            except IndexError:
                validate_to_t = series[validate_to_i - 1].t
            validate_from_dt = dt_from_s(validate_from_t)
            validate_to_dt   = dt_from_s(validate_to_t)

            # Deep-copy the model
            model = copy.deepcopy(self)

            # Perform the cross validation
            logger.info('Cross validation round {}/{}: validate from {} ({}) to {} ({}), fit on the rest.'.format(i+1, rounds, validate_from_t, validate_from_dt, validate_to_t, validate_to_dt))

            if validate_from_i == 0:
                # First chunk
                logger.debug('Fitting from {} to the end'.format(validate_to_t))
                series_view = TimeSeriesView(series=series, from_i=validate_to_i, to_i=len(series))
                model.fit(series_view, **fit_kwargs)

            elif validate_to_i >= len(series):
                # Last chunk
                logger.debug('Fitting from the beginning to {}'.format(validate_from_t))
                series_view = TimeSeriesView(series=series, from_i=0, to_i=validate_from_i)
                model.fit(series_view, **fit_kwargs)

            else:
                # Find the bigger chunk and fit on that
                if validate_from_i > len(series)-validate_to_i:
                    logger.debug('Fitting from the beginning to {}'.format(validate_from_t))
                    series_view = TimeSeriesView(series=series, from_i=0, to_i=validate_from_i)
                    model.fit(series_view, **fit_kwargs)
                    model_window = None 
                    try:
                        model_window = model.window
                    except AttributeError:
                        pass
                    if model_window and (len(series)-validate_to_i) < model_window:
                        pass
                    else:
                        # Try to fit on the other chunk as well:
                        try:
                            logger.debug('Now trying to fit also from {} to the end'.format(validate_to_t))
                            series_view = TimeSeriesView(series=series, from_i=validate_to_i, to_i=len(series))
                            model.fit_update(series_view, **fit_kwargs)
                        except NotImplementedError:
                            if not fit_update_unaivalable_warned:
                                logger.warning('This model does not support updating the fit, cross validation results will be approximate for the intermediate chunks')
                                fit_update_unaivalable_warned = True
                else:
                    logger.debug('Fitting from {} to the end'.format(validate_to_t))
                    series_view = TimeSeriesView(series=series, from_i=validate_to_i, to_i=len(series))
                    model.fit(series_view, **fit_kwargs)
                    model_window = None 
                    try:
                        model_window = model.window
                    except AttributeError:
                        pass
                    if model_window and validate_from_i < model_window:
                        pass
                    else:
                        # Try to fit on the other chunk as well:
                        try:
                            logger.debug('Now trying to fit also from the beginning to {}'.format(validate_from_t))
                            series_view = TimeSeriesView(series=series, from_i=0, to_i=validate_from_i)
                            model.fit_update(series_view, **fit_kwargs)
                        except NotImplementedError:
                            if not fit_update_unaivalable_warned:
                                logger.warning('This model does not support updating the fit, cross validation results will be approximate for the intermediate chunks')
                                fit_update_unaivalable_warned = True

            # Evaluate & append
            evaluations.append(model.evaluate(TimeSeriesView(series=series, from_i=validate_from_i, to_i=validate_to_i), **evaluate_kwargs))

        # Regroup evaluations
        evaluation_metrics = list(evaluations[0].keys())
        scores_by_evaluation_metric = {}
        for evaluation in evaluations:
            for evaluation_metric in evaluation_metrics:
                try:
                    scores_by_evaluation_metric[evaluation_metric].append(evaluation[evaluation_metric])
                except KeyError:
                    try:
                        scores_by_evaluation_metric[evaluation_metric] = [evaluation[evaluation_metric]]
                    except KeyError:
                        raise Exception('Error, the model generated different evaluation metrics over the rounds, cannot compute cross validation.') from None

        # Prepare and return results
        results = {}
        for evaluation_metric in scores_by_evaluation_metric:
            try:
                results[evaluation_metric+'_avg'] = statistics.mean(scores_by_evaluation_metric[evaluation_metric])
                results[evaluation_metric+'_stdev'] = statistics.stdev(scores_by_evaluation_metric[evaluation_metric])
            except TypeError:
                pass
        if return_full_evaluations:
            results['evaluations'] = evaluations
        return results


#=========================
#  Base Prophet model
#=========================

class _ProphetModel(Model):
    '''A model using Prophet as underlying engine, and providing some extra internal functions for common operations.'''

    @classmethod
    def _remove_timezone(cls, dt):
        return dt.replace(tzinfo=None)

    @classmethod
    def _from_timeseria_to_prophet(cls, series):

        # Create Python lists with data
        try:
            series[0].data[0]
            data_labels_are_indexes = True
        except KeyError:
            series[0].data.keys()
            data_labels_are_indexes = False

        data_as_list=[]
        for item in series:

            if data_labels_are_indexes:
                data_as_list.append([cls._remove_timezone(item.dt), item.data[0]])
            else:
                data_as_list.append([cls._remove_timezone(item.dt), item.data[list(item.data.keys())[0]]])

        # Create the pandas DataFrames
        data = DataFrame(data_as_list, columns = ['ds', 'y'])

        return data


#=========================
#  Base ARIMA model
#=========================

class _ARIMAModel(Model):
    '''A model using statsmodel's ARIMA as underlying engine, and providing some extra internal functions for common operations.'''

    def _get_start_end_indexes(self, series, n):

        # Do the math to get the right indexes for the prediction, both out-of-sample and in-sample
        # Not used at the moment as there seems to be bugs in the statsmodel package.
        # See https://www.statsmodels.org/devel/generated/statsmodels.tsa.arima_model.ARIMAResults.predict.html

        # The default start_index for an ARIMA prediction is the index after the fit series
        # I.e. if the fit series had 101 elments, the start index is the 101
        # So we need to compute the start and end indexes according to the fit series,
        # we will use the "resolution" for this.

        # Save the requested prediction "from"
        requested_predicting_from_dt = series[-1].dt + series.resolution

        # Search for the index number TODO: try first with "pure" math which would work for UTC
        slider_dt = self.fit_series[0].dt
        count = 0
        while True:
            if slider_dt  == requested_predicting_from_dt:
                break

            # To next item index
            if requested_predicting_from_dt < self.fit_series[0].dt:
                slider_dt = slider_dt - self.fit_series.resolution
                if slider_dt < requested_predicting_from_dt:
                    raise Exception('Miss!')

            else:
                slider_dt = slider_dt + self.fit_series.resolution
                if slider_dt > requested_predicting_from_dt:
                    raise Exception('Miss!')

            count += 1

        # TODO: better understand the indexes in statsmodels (and potentially their bugs), as sometimes
        # setting the same start and end gave two predictions which is just nonsense in any scenario.
        start_index = count
        end_index = count + n

        return (start_index, end_index)


#=========================
#  Base Keras model
#=========================

class _KerasModel(Model):
    '''A model using Keras as underlying engine, and providing some extra internal functions for common operations.'''

    def _load_keras_model(self, path):

        # Load the Keras model
        if path:
            from tensorflow.keras.models import load_model as load_keras_model
            self.keras_model = load_keras_model('{}/model.keras'.format(path))

    def _save_keras_model(self, path):

        # Save the Keras model
        try:
            self.keras_model.save('{}/model.keras'.format(path))
        except Exception as e:
            shutil.rmtree(path)
            raise e

    # TODO: since we are extending a generic ParametricModel, the following methods should not be here.
    # Maybe only if extending a TimeSeriesParametricModel, and still it is probably the wrong place, as for the
    # ARIMA and Prophet above also. Consider moving them in a "utility" package or directly in the models.

    @staticmethod
    def _to_matrix_representation(series, window, steps, context_data_labels, target_data_labels, data_loss_limit):

        if steps > 1:
            raise NotImplementedError('Not implemented for steps >1')

        window_elements_matrix = []
        target_values_vector = []
        context_data_vector = []

        for i, _ in enumerate(series):
            if i <  window:
                continue
            if i == len(series):
                break

            try:
                # Add window elements
                window_elements_vector = []
                for j in range(window):
                    if data_loss_limit is not None and 'data_loss' in series[i-window+j].data_indexes and series[i-window+j].data_indexes['data_loss'] >= data_loss_limit:
                        raise TooMuchDataLoss()
                    window_elements_vector.append(series[i-window+j])

                # Add target values
                target_values_sub_vector = []
                if data_loss_limit is not None and 'data_loss' in series[i].data_indexes and series[i].data_indexes['data_loss'] >= data_loss_limit:
                    raise TooMuchDataLoss()
                for target_data_label in target_data_labels:
                    target_values_sub_vector.append(series[i].data[target_data_label])
            except TooMuchDataLoss:
                continue
            else:

                # Append window elements and target data
                window_elements_matrix.append(window_elements_vector)
                target_values_vector.append(target_values_sub_vector)

                # Add context data if required
                if context_data_labels is not None:
                    context_data_vector.append({data_label: series[i].data[data_label] for data_label in context_data_labels})

        return window_elements_matrix, target_values_vector, context_data_vector


    @staticmethod
    def _compute_window_features(window_datapoints, data_labels, time_unit, features, context_data=None, flatten=False):
        """Compute features from a list of window data points (or slots).

        Args:
            window_datapoints (list): The list with the data points (or slots)
            data_labels(dict): the labels of the point (or slot) data.
            features(list): the list of the features to compute.
                Supported values are:
                ``values`` (use the data values),
                ``diffs``  (use the diffs between the values), and
                ``hours``  (use the hours of the timestamp).
            flatten(bool): if to flatten the features as a signle list.
        """

        available_features = ['values', 'diffs', 'hours']
        for feature in features:
            if feature not in available_features:
                raise ValueError('Unknown feature "{}"'.format(feature))

        # Handle context data if any (as "fake", context datapoint).
        if context_data:
            context_datapoint_dt = window_datapoints[-1].dt + time_unit
            context_datapoint = window_datapoints[-1].__class__(dt = context_datapoint_dt, data = context_data)
            window_datapoints.append(context_datapoint)

        # Compute the features
        window_features = []
        for i in range(len(window_datapoints)):

            datapoint_features = []

            # 1) Data point/slot values (for all data labels)
            if 'values' in features:
                for data_label in data_labels:
                    #if window_datapoints[i].data[data_label] is not None:
                    try:
                        datapoint_features.append(window_datapoints[i].data[data_label])
                    except KeyError:
                        datapoint_features.append(0.0)

            # 2) Compute diffs on normalized datapoints
            if 'diffs' in features:
                for data_label in data_labels:
                    try:
                        if i == 0:
                            diff = window_datapoints[1].data[data_label] - window_datapoints[0].data[data_label]
                        elif i == len(window_datapoints)-1:
                            diff = window_datapoints[-1].data[data_label] - window_datapoints[-2].data[data_label]
                        else:
                            diff = (window_datapoints[i+1].data[data_label] - window_datapoints[i-1].data[data_label]) /2
                        if diff == 0:
                            diff = 1
                        datapoint_features.append(diff)
                    except KeyError:
                        datapoint_features.append(0.0)

            # 3) Hour (normlized)
            if 'hours' in features:
                datapoint_features.append(window_datapoints[i].dt.hour/24)

            # Now add to the window features
            if flatten:
                window_features += datapoint_features
            else:
                window_features.append(datapoint_features)

        return window_features


