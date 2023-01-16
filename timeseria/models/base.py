# -*- coding: utf-8 -*-
"""Provides base model classes."""

import os
import json
import uuid
import statistics
from ..exceptions import NotFittedError
from ..utilities import _check_timeseries, _check_resolution, _check_data_labels, _item_is_in_range
from ..time import now_s, dt_from_s
from ..units import TimeUnit
from ..datastructures import Point, Series, TimeSeries
from pandas import DataFrame
import shutil

# Setup logging
import logging
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings as default behavior
try:
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except (ImportError,AttributeError):
    pass


#======================
#  Base Model
#======================

class Model():
    """A generic model. This can be either a statelsess model, where all the information is coded and there are no parameters, or a
    stateful (parametric) model, where there are a number of parameters which can be both set manually or learnt (fitted) from the data.
    
    All models expose a ``predict()``, ``apply()`` and ``evaluate()`` methods, while parametric models also provide a ``save()`` method to
    store the parameters of the model, and an optional ``fit()`` method if the parameters are to be learnt form data. In this case also a
    ``cross_validate()`` method is available.
        
    Args:
        path (optional, str): a path from which to load a saved model. Will override all other init settings. Only for parametric models.
    """
    
    def __init__(self, path=None):
        
        # Set type
        try:
            self.data
        except:
            # No data (parameters) for the model. Check if there is a fit
            try:
                self._fit
            except AttributeError:
                # If not, then the model has no parameters at all
                self._type = 'non-parametric'
            else:
                # Otherwise, the model is parameter and the parameters
                # are still to be set via the fit
                self._type = 'parametric'
        else:
            # If we have data, it means that there are parameters and
            # the model is parametric.
            self._type = 'parametric'

        # Do we have to load the model (if parametric?)    
        if self.is_parametric():
            if path:
                with open(path+'/data.json', 'r') as f:
                    self.data = json.loads(f.read())
                # TODO: the "fitted" does not apply for models with parameters but with no fit.. fix me.  
                self.fitted = True   
            else:
                id = str(uuid.uuid4())
                self.fitted = False
                try:
                    self.data['id'] = id
                except AttributeError:
                    self.data = {'id': id}
        else:
            if path:
                raise ValueError('Loading a non-parametric model from a path does not make sense')
     
    @property
    def id(self):
        """A unique identifier for the model. Only for parametric models."""
        if not self.is_parametric():
            raise TypeError('Non-parametric models have no ID') # TODO: is this the right exception?
        else:
            return self.data['id']

    def is_parametric(self):
        """If the model is parametric or not."""
        if self._type == 'parametric':
            return True
        else:
            return False

    def fit(self, *args, **kwargs):
        """Fit the model."""

        try:
            self._fit
        except AttributeError:
            raise NotImplementedError('Fitting this model is not implemented')

        fit_output = self._fit(*args, **kwargs)

        self.data['fitted_at'] = now_s()
        self.fitted = True

        return fit_output

    def predict(self, *args, **kwargs):
        """Call the model predict logic."""
        try:
            self._predict
        except AttributeError:
            raise NotImplementedError('Predicting from this model is not implemented')
        
        # Ensure the model is fitted if it has to
        try:
            self._fit
        except AttributeError:
            pass
        else:
            if not self.fitted:
                raise NotFittedError()

        return self._predict(*args, **kwargs)

    def apply(self, *args, **kwargs):
        """Apply the model."""
        try:
            self._apply
        except AttributeError:
            raise NotImplementedError('Applying this model is not implemented')

        # Ensure the model is fitted if it has to
        try:
            self._fit
        except AttributeError:
            pass
        else:
            if not self.fitted:
                raise NotFittedError()

        return self._apply(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        """Evaluate the model."""
        try:
            self._evaluate
        except AttributeError:
            raise NotImplementedError('Evaluating this model is not implemented')

        # Ensure the model is fitted if it has to
        try:
            self._fit
        except AttributeError:
            pass
        else:
            if not self.fitted:
                raise NotFittedError()

        return self._evaluate(*args, **kwargs)

    def cross_validate(self, *args, rounds=10, **kwargs):
        """Cross validate the model, by default with 10 fit/evaluate rounds.
        
        All the parameters starting with the ``fit_`` prefix are forwarded to the model ``fit()`` method (without the prefix), and
        all the parameters starting with the ``evaluate_`` prefix are forwarded to the model ``evaluate()`` method (without the prefix).
        
        Args:
            rounds(int): how many rounds of cross validation to run.
        """
        try:
            self._cross_validate
        except AttributeError:
            raise NotImplementedError('Cross validating this model is not yet implemented')
        return self._cross_validate(*args, rounds=rounds, **kwargs)

    def save(self, path):
        """Save the model in the given path. The model is saved in "directory format", 
         meaning that a new directory will be created containing the data for the model."""

        # Is this model parametric?
        if not self.is_parametric():
            raise TypeError('Saving a non-parametric model from a path does not make sense') # TODO: is this the right exception?

        # Ensure the model is fitted if it has to
        try:
            self._fit
        except AttributeError:
            pass
        else:
            if not self.fitted:
                raise NotFittedError()

        if not path:
            raise ValueError('Got empty path, cannot save')

        if os.path.exists(path):
            raise ValueError('The path "{}" already exists'.format(path)) 

        # Prepare model dir and dump data as json
        os.makedirs(path)
        model_data_file = '{}/data.json'.format(path)
        
        with open(model_data_file, 'w') as f:
            f.write(json.dumps(self.data))
        
        logger.info('Saved model with id "%s" in "%s"', self.data['id'], path)


#======================
#  Base Series Model
#======================

class SeriesModel(Model):
    """A model specifically designed to work with series data. In particular, it enforces
    resolution and data labels consistency between methods and save/load operations.

    Args:
        path (optional, str): a path from which to load a saved model. Will override all other init settings. Only for parametric models.
    """ 
    
    def __init__(self, path=None):

        # Call parent init
        super(SeriesModel, self).__init__(path)

        # If the model has been loaded, convert resolution as TimeUnit
        try:
            self.fitted
        except AttributeError:
            pass
        else:
            if self.fitted:
                # Convert resolution to TimeUnit. TODO: right now only TimeSeries have the resolution, but might change if added to the Series as well.
                if 'resolution' in self.data:
                    self.data['resolution'] = TimeUnit(self.data['resolution'])

    def fit(self, series, *args, **kwargs):
        """Fit the model on a series."""

        if not isinstance(series, Series):
            raise TypeError('A Series-like object is required, got "{}"'.format(series.__class__.__name__))

        # If TimeSeries data, check the timeseries
        if isinstance(series, TimeSeries): 
            _check_timeseries(series)

        # Call parent fit
        fit_output = super(SeriesModel, self).fit(series, *args, **kwargs)
        
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

        # Return output
        return fit_output

    def predict(self, series, *args, **kwargs):
        """Call the model predict logic on a series."""
        
        if not isinstance(series, Series):
            raise TypeError('A Series-like object is required, got "{}"'.format(series.__class__.__name__))

        # If fitted, check resolution and keys. If not fitted, the parent init will raise.
        try:
            self.fitted
        except AttributeError:
            pass
        else:
            if self.fitted:
                # If TimeSeries data, check resolution and data labels consistency
                if self.is_parametric():
                    if isinstance(series.items_type, Point) and len(series) == 1:
                        # Do not check if the series is a point series and has only one item
                        pass
                    else:
                        _check_resolution(series, self.data['resolution'])
                    _check_data_labels(series, self.data['data_labels'])
        finally:
            # Check time series in any case
            if isinstance(series, TimeSeries):
                _check_timeseries(series)
                        
        # Call parent predict and return output
        return super(SeriesModel, self).predict(series, *args, **kwargs)
 
    def apply(self, series, *args, **kwargs):
        """Apply the model on a series."""
        
        if not isinstance(series, Series):
            raise TypeError('A Series-like object is required, got "{}"'.format(series.__class__.__name__))

        # If fitted, check resolution and keys. If not fitted, the parent init will raise.
        try:
            self.fitted
        except AttributeError:
            pass
        else:
            if self.fitted:
                # If TimeSeries data, check resolution and data labels consistency
                if isinstance(series, TimeSeries): 
                    if isinstance(series.items_type, Point) and len(series) == 1:
                        # Do not check if the series is a point series and has only one item
                        pass
                    else:
                        _check_resolution(series, self.data['resolution'])
                    _check_data_labels(series, self.data['data_labels'])
        finally:
            # Check time series in any case
            if isinstance(series, TimeSeries):
                _check_timeseries(series)
            
        # Call parent apply and return output
        return super(SeriesModel, self).apply(series, *args, **kwargs)

    def evaluate(self, series, *args, **kwargs):
        """Evaluate the model on a series."""
        
        if not isinstance(series, Series):
            raise TypeError('A Series-like object is required, got "{}"'.format(series.__class__.__name__))

        # If fitted, check resolution and keys. If not fitted, the parent init will raise.
        try:
            self.fitted
        except AttributeError:
            pass
        else:
            if self.fitted:
                # If TimeSeries data, check resolution and data labels consistency
                if isinstance(series, TimeSeries): 
                    if isinstance(series.items_type, Point) and len(series) == 1:
                        # Do not check if the series is a point series and has only one item
                        pass
                    else:
                        _check_resolution(series, self.data['resolution'])
                    _check_data_labels(series, self.data['data_labels'])
        finally:
            # Check time series in any case
            if isinstance(series, TimeSeries):
                _check_timeseries(series)

        # Call parent evaluate and return output
        return super(SeriesModel, self).evaluate(series, *args, **kwargs)

    def cross_validate(self, series, *args, **kwargs):
        return super(SeriesModel, self).cross_validate(series, *args, **kwargs)

    def _cross_validate(self, series, rounds, *args, **kwargs):
        """The cross validate logic for series models."""
        
        if not isinstance(series, Series):
            raise TypeError('A Series-like object is required, got "{}"'.format(series.__class__.__name__))

        if isinstance(series, TimeSeries):
            timeseries = series  

            # If fitted, check resolution. If not fitted, the parent init will raise.
            try:
                self.fitted
            except AttributeError:
                pass
            else:
                if self.fitted:
                    # If TimeSeries data, check resolution and data labels consistency
                    if isinstance(series, TimeSeries): 
                        if isinstance(series.items_type, Point) and len(series) == 1:
                            # Do not check if the series is a point series and has only one item
                            pass
                        else:
                            _check_resolution(series, self.data['resolution'])
                        _check_data_labels(series, self.data['data_labels'])
            finally:
                # Check time series in any case
                if isinstance(series, TimeSeries):
                    _check_timeseries(series)
                    
            # Decouple fit from validate args
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
            round_items = int(len(timeseries) / rounds)
            logger.debug('Items per round: {}'.format(round_items))
            
            # Start the fit / evaluate loop
            evaluations = []        
            for i in range(rounds):
                from_t = timeseries[(round_items*i)].t
                try:
                    to_t = timeseries[(round_items*i) + round_items].t
                except IndexError:
                    to_t = timeseries[(round_items*i) + round_items - 1].t
                from_dt = dt_from_s(from_t)
                to_dt   = dt_from_s(to_t)
                logger.info('Cross validation round #{} of {}: validate from {} ({}) to {} ({}), fit on the rest.'.format(i+1, rounds, from_t, from_dt, to_t, to_dt))
                
                # Fit
                if i == 0:            
                    logger.debug('Fitting from {} ({})'.format(to_t, to_dt))
                    self.fit(timeseries, from_t=to_t, **fit_kwargs)
                else:
                    logger.debug('Fitting until {} ({}) and then from {} ({}).'.format(to_t, to_dt, from_t, from_dt))
                    self.fit(timeseries, from_t=to_t, to_t=from_t, **fit_kwargs)                
                
                # Evaluate & append
                evaluations.append(self.evaluate(timeseries, from_t=from_t, to_t=to_t, **evaluate_kwargs))
            
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
                results[evaluation_metric+'_avg'] = statistics.mean(scores_by_evaluation_metric[evaluation_metric])
                results[evaluation_metric+'_stdev'] = statistics.stdev(scores_by_evaluation_metric[evaluation_metric])         
            return results
        
        else:
            raise NotImplementedError('Cross validating on data types other than TimeSeries in not yet implemented')

    def save(self, path):

        # Temporary change the resolution to its string representation (if any)
        if 'resolution' in self.data:
            resolution_obj = self.data['resolution']
            self.data['resolution'] = str(resolution_obj)
        
        # Call parent save
        try:
            super(SeriesModel, self).save(path)
        finally:
            # In any case revert resolution back to object (if any)
            if 'resolution' in self.data:  
                self.data['resolution'] = resolution_obj


#=========================
#  Base Prophet model
#=========================

class _ProphetModel(Model):
    '''A model using Prophet as underlying engine, and providing some extra internal functions for common operations.'''

    @classmethod
    def _remove_timezone(cls, dt):
        return dt.replace(tzinfo=None)

    @classmethod
    def _from_timeseria_to_prophet(cls, timeseries, from_t=None, to_t=None):

        # Create Python lists with data
        try:
            timeseries[0].data[0]
            data_labels_are_indexes = True
        except KeyError:
            timeseries[0].data.keys()
            data_labels_are_indexes = False
        
        data_as_list=[]
        for item in timeseries:
            
            # Skip if needed
            try:
                if not _item_is_in_range(item, from_t, to_t):
                    continue
            except StopIteration:
                break                

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
    
    def _get_start_end_indexes(self, timeseries, n):

        # Do the math to get the right indexes for the prediction, both out-of-sample and in-sample
        # Not used at the moment as there seems to be bugs in the statsmodel package.
        # See https://www.statsmodels.org/devel/generated/statsmodels.tsa.arima_model.ARIMAResults.predict.html
        
        # The default start_index for an arima prediction id the inxed after the fit timeseries
        # I.e. if the fit timeseries had 101 lements, the start index is the 101
        # So we need to compute the start and end indexes according to the fit timeseries,
        # we will use the "resolution" for this.
        
        # Save the requested prediction "from"
        requested_predicting_from_dt = timeseries[-1].dt + timeseries.resolution

        # Search for the index number TODO: try first with "pure" math which would work for UTC
        slider_dt = self.fit_timeseries[0].dt
        count = 0
        while True:
            if slider_dt  == requested_predicting_from_dt:
                break
             
            # To next item index
            if requested_predicting_from_dt < self.fit_timeseries[0].dt:
                slider_dt = slider_dt - self.fit_timeseries.resolution
                if slider_dt < requested_predicting_from_dt:
                    raise Exception('Miss!')
 
            else:
                slider_dt = slider_dt + self.fit_timeseries.resolution
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
            from keras.models import load_model as load_keras_model
            self.keras_model = load_keras_model('{}/keras_model'.format(path))

    def _save_keras_model(self, path):

        # Save the Keras model
        try:
            self.keras_model.save('{}/keras_model'.format(path))
        except Exception as e:
            shutil.rmtree(path)
            raise e
    
    # TODO: since we are extending a generic ParametricModel, the following methods should not be here.
    # Maybe only if extending a TimeSeriesParametricModel, and still it is probably the wrong place, as for the
    # ARIMA and Prophet above also. Consider moving them in a "utility" package or directly in the models.
    
    @staticmethod
    def _to_window_datapoints_matrix(timeseries, window, steps, encoder=None):
        '''Compute window datapoints matrix from a time series.'''
        # steps to be intended as steps ahead (for the forecaster)
        window_datapoints = []
        for i, _ in enumerate(timeseries):
            if i <  window:
                continue
            if i == len(timeseries) + 1 - steps:
                break
                    
            # Add window values
            row = []
            for j in range(window):
                row.append(timeseries[i-window+j])
            window_datapoints.append(row)
                
        return window_datapoints

    @staticmethod
    def _to_target_values_vector(timeseries, window, steps):
        '''Compute target values vector from a time series.'''
        # steps to be intended as steps ahead (for the forecaster)
    
        data_labels = timeseries.data_labels()
    
        targets = []
        for i, _ in enumerate(timeseries):
            if i <  window:
                continue
            if i == len(timeseries) + 1 - steps:
                break
            
            # Add forecast target value(s)
            row = []
            for j in range(steps):
                for data_label in data_labels:
                    row.append(timeseries[i+j].data[data_label])
            targets.append(row)

        return targets

    @staticmethod
    def _compute_window_features(window_datapoints, data_labels, features):
        """Compute features from a list of window data points (or slots).
        
        Args:
            window_datapoints (list): The list with the data points (or slots)
            data_labels(dict): the keys of the point (or slot) data.
            features(list): the list of the features to compute.
                Supported values are:
                ``values`` (use the data values), 
                ``diffs``  (use the diffs between the values), and 
                ``hours``  (use the hours of the timestamp).
        """

        available_features = ['values', 'diffs', 'hours']
        for feature in features:
            if feature not in available_features:
                raise ValueError('Unknown feature "{}"'.format(feature))
        
        window_features=[]

        for i in range(len(window_datapoints)):
            
            datapoint_features = []
            
            # 1) datapoint values (for all keys)
            if 'values' in features:
                for data_label in data_labels:
                    datapoint_features.append(window_datapoints[i].data[data_label])
                            
            # 2) Compute diffs on normalized datapoints
            if 'diffs' in features:
                for data_label in data_labels:
                    if i == 0:
                        diff = window_datapoints[1].data[data_label] - window_datapoints[0].data[data_label]
                    elif i == len(window_datapoints)-1:
                        diff = window_datapoints[-1].data[data_label] - window_datapoints[-2].data[data_label]
                    else:
                        diff = (window_datapoints[i+1].data[data_label] - window_datapoints[i-1].data[data_label]) /2
                    if diff == 0:
                        diff = 1
                    datapoint_features.append(diff)

            # 3) Hour (normlized)
            if 'hours' in features:
                datapoint_features.append(window_datapoints[i].dt.hour/24)
                
            # Now append to the window features
            window_features.append(datapoint_features)

        return window_features

