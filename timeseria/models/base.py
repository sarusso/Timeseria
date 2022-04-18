# -*- coding: utf-8 -*-
"""Provides base model classes."""

import os
import json
import uuid
import statistics
from ..exceptions import NotFittedError
from ..utilities import check_timeseries, check_resolution, check_data_labels, item_is_in_range
from ..time import now_s, dt_from_s
from ..units import TimeUnit
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

class Model(object):
    """A stateless model, or a white-box model. Exposes only ``predict()``, ``apply()`` and ``evaluate()``
     methods, since it is assumed that all the information is coded and nothing is learnt from the data."""
    
    def __init__(self):
        pass


    def predict(self, data, *args, **kwargs):
        """Call the model predict logic on some data"""
        try:
            self._predict
        except AttributeError:
            raise NotImplementedError('Predicting from this model is not implemented')
        
        return self._predict(data, *args, **kwargs)


    def apply(self, data, *args, **kwargs):
        """Apply the model on some data"""
        try:
            self._apply
        except AttributeError:
            raise NotImplementedError('Applying this model is not implemented')
        
        return self._apply(data, *args, **kwargs)


    def evaluate(self, data, *args, **kwargs):
        """Evaluate the model on some data"""
        try:
            self._evaluate
        except AttributeError:
            raise NotImplementedError('Evaluating this model is not implemented')
        
        return self._evaluate(data, *args, **kwargs)



#=========================
#  Base parametric models
#=========================

class ParametricModel(Model):
    """A stateful model with parameters, or a (partly) black-box model. Parameters can be set manually or
    learnt (fitted) from data. On top of the ``predict()``, ``apply()`` and ``evaluate()`` methods it provides
    a ``save()`` method to store the parameters of the model, and optionally a ``fit()`` method if the parameters
    are to be learnt form data.
    
    Args:
        path (str): a path from which to load a saved model. Will override all other init settings.
 
    """
    
    def __init__(self, path=None):
        
        if path:
            with open(path+'/data.json', 'r') as f:
                self.data = json.loads(f.read())         
            self.fitted = True
            
        else:
            id = str(uuid.uuid4())
            self.fitted = False
            self.data = {'id': id}

        super(ParametricModel, self).__init__()


    @property
    def id(self):
        """A unique identifier for the model"""
        return self.data['id']


    def fit(self, data, *args, **kwargs):
        """Fit the model on some data"""

        try:
            self._fit
        except AttributeError:
            raise NotImplementedError('Fitting this model is not implemented')
        
        fit_output = self._fit(data, *args, **kwargs)

        self.data['fitted_at'] = now_s()
        self.fitted = True

        return fit_output


    def predict(self, data, *args, **kwargs):

        try:
            self._predict
        except AttributeError:
            raise NotImplementedError('Predicting from this model is not implemented')

        if not self.fitted:
            raise NotFittedError()

        return self._predict(data, *args, **kwargs)


    def apply(self, data, *args, **kwargs):

        try:
            self._apply
        except AttributeError:
            raise NotImplementedError('Applying this model is not implemented')

        if not self.fitted:
            raise NotFittedError()

        return self._apply(data, *args, **kwargs)


    def evaluate(self, data, *args, **kwargs):

        try:
            self._evaluate
        except AttributeError:
            raise NotImplementedError('Evaluating this model is not implemented')

        if not self.fitted:
            raise NotFittedError('This model is not fitted, cannot evaluate')
                
        return self._evaluate(data, *args, **kwargs)


    def cross_validate(self, data, *args, **kwargs):
        """Cross validate the model on some data"""

        try:
            self._cross_validate
        except AttributeError:
            raise NotImplementedError('Cross-validating this model is not implemented')

        return self._cross_validate(data, *args, **kwargs)


    def save(self, path):
        """Save the model in the given path. The model is saved in "directory format", 
         meaning that a new directory will be created containing the data for the model."""

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



class TimeSeriesParametricModel(ParametricModel):
    """A parametric model specifically designed to work with time series data. In particular,
    it ensures resolution and data labels consistency between methods and save/load operations.

    Args:
        path (str): a path from which to load a saved model. Will override all other init settings.
    """ 
    
    def __init__(self, path=None):

        # Call parent init
        super(TimeSeriesParametricModel, self).__init__(path)

        # If the model has been loaded, convert resolution as TimeUnit
        if self.fitted:
            self.data['resolution'] = TimeUnit(self.data['resolution'])


    def fit(self, timeseries, *args, **kwargs):
        """Fit the model on a time series"""
  
        # Check timeseries
        check_timeseries(timeseries)
        
        # Call parent fit
        fit_output = super(TimeSeriesParametricModel, self).fit(timeseries, *args, **kwargs)
        
        # Set timeseries resolution
        self.data['resolution'] = timeseries.resolution
        
        # Set timeseries data keys
        self.data['data_labels'] = timeseries.data_labels()

        # Return output
        return fit_output


    def predict(self, timeseries, *args, **kwargs):
        """Call the model predict logic on a time series"""
        
        # Check timeseries and its resolution
        check_timeseries(timeseries)
        
        # If fitted, check resolution and keys. If not fitted, the parent init will raise.
        if self.fitted:
            check_resolution(timeseries, self.data['resolution'])
            check_data_labels(timeseries, self.data['data_labels'])
                
        # Call parent predict and return output
        return super(TimeSeriesParametricModel, self).predict(timeseries, *args, **kwargs)

    
    def apply(self, timeseries, *args, **kwargs):
        """Apply the model on a time series"""
        
        # Check timeseries
        check_timeseries(timeseries)
        
        # If fitted, check resolution. If not fitted, the parent init will raise.
        if self.fitted:
            check_resolution(timeseries, self.data['resolution'])
        
        # Call parent apply and return output
        return super(TimeSeriesParametricModel, self).apply(timeseries, *args, **kwargs)


    def evaluate(self, timeseries, *args, **kwargs):
        """Evaluate the model on a time series"""
        
        # Check timeseries and its resolution
        check_timeseries(timeseries)
        
        # If fitted, check resolution. If not fitted, the parent init will raise.
        if self.fitted:
            check_resolution(timeseries, self.data['resolution'])
                
        # Call parent evaluate and return output
        return super(TimeSeriesParametricModel, self).evaluate(timeseries, *args, **kwargs)


    def cross_validate(self, timeseries, rounds=10, **kwargs):
        """Cross validate the model on a time series.
        
        All the parameters starting with the ``fit_`` prefix are forwareded to the model ``fit()`` method (without the prefix), and
        all the parameters starting with the ``evaluate_`` prefix are forwarded to the model ``evaluate()`` method (without the prefix).
        
        Args:
            rounds(int): how many rounds of cross validation to run.
        """
        return super(TimeSeriesParametricModel, self).cross_validate(timeseries, rounds, **kwargs)


    def _cross_validate(self, timeseries, rounds=10, **kwargs):

        try:
            self._evaluate
        except AttributeError:
            raise NotImplementedError('Evaluating this model is not implemented, cannot cross-validate')

        # Check timeseries
        check_timeseries(timeseries)
        
        # If fitted, check resolution. If not fitted, the parent init will raise.
        if self.fitted:
            check_resolution(timeseries, self.data['resolution'])
        
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


    def save(self, path):

        # If fitted, handle resolution. If not fitted, the parent save will raise.
        if self.fitted:
        
            # Save original data resolution (Unit or TimeUnit Object)
            orresolution = self.data['resolution']
    
            # Temporarily change the model resolution unit as string representation
            self.data['resolution'] = str(self.data['resolution'])
            
        # Call parent save
        super(TimeSeriesParametricModel, self).save(path)
        
        # Set back original data resolution
        self.data['resolution'] = orresolution



#=========================
#  Base Prophet model
#=========================

class ProphetModel():
    '''Class to wrap some internal common logic for Prophet-based models.'''

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
                if not item_is_in_range(item, from_t, to_t):
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

class ARIMAModel():
    """Class to wrap some internal common logic for ARIMA-based models."""
    
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

class KerasModel():
    """Class to wrap some internal common logic for Keras-based models."""

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
    
    # TODO: since we are extenting a generic ParametricModel, the following methods should not be here.
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

