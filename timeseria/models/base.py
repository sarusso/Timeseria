# -*- coding: utf-8 -*-
"""Provides base model classes."""

import os
import json
import uuid
import statistics
from ..exceptions import NotFittedError
from ..utilities import check_timeseries, check_resolution, check_data_keys, item_is_in_range
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
except ImportError:
    pass

#======================
#  Base Model
#======================

class Model(object):
    '''A stateless model, or a white-box model. Exposes only predict(), apply() and evaluate() methods,
    since it is assumed that all the information is coded and nothing is learnt from the data.'''
    
    def __init__(self):
        pass

    
    def predict(self, data, *args, **kwargs):
        """Call te model predict logic"""
        try:
            self._predict
        except AttributeError:
            raise NotImplementedError('Predicting from this model is not implemented')
        
        return self._predict(data, *args, **kwargs)


    def apply(self, data, *args, **kwargs):
        try:
            self._apply
        except AttributeError:
            raise NotImplementedError('Applying this model is not implemented')
        
        return self._apply(data, *args, **kwargs)


    def evaluate(self, data, *args, **kwargs):
        try:
            self._evaluate
        except AttributeError:
            raise NotImplementedError('Evaluating this model is not implemented')
        
        return self._evaluate(data, *args, **kwargs)



#=========================
#  Base parametric models
#=========================

class ParametricModel(Model):
    """A stateful model with parameters, or a (partly) black-box model. Parameters can be set manually
    or learnt (fitted) from data. On top of the predict(), apply() and evaluate() methods it provides
    a save() method to store the parameters of the model, and optionally a fit() method if the parameters
    are to be learnt form data."""
    
    def __init__(self, path=None, id=None):
        
        if path:
            with open(path+'/data.json', 'r') as f:
                self.data = json.loads(f.read())         
            self.fitted = True
            
        else:
            if not id:
                id = str(uuid.uuid4())
            self.fitted = False
            self.data = {'id': id}

        super(ParametricModel, self).__init__()


    def save(self, path):

        if not self.fitted:
            raise NotFittedError()

        # Prepare model dir and dump data as json
        model_dir = '{}/{}'.format(path, self.data['id'])
        os.makedirs(model_dir)
        model_data_file = '{}/data.json'.format(model_dir)
        with open(model_data_file, 'w') as f:
            f.write(json.dumps(self.data))
        logger.info('Saved model with id "%s" in "%s"', self.data['id'], model_dir)
        return model_dir

    
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
            raise NotFittedError()
                
        return self._evaluate(data, *args, **kwargs)


    def fit(self, *args, **kwargs):

        try:
            self._fit
        except AttributeError:
            raise NotImplementedError('Fitting this model is not implemented')
        
        fit_output = self._fit(*args, **kwargs)

        self.data['fitted_at'] = now_s()
        self.fitted = True

        return fit_output


    def cross_validate(self, *args, **kwargs):

        try:
            self._cross_validate
        except AttributeError:
            raise NotImplementedError('Cross-validating this model is not implemented')

        return self._cross_validate(*args, **kwargs)


    @property
    def id(self):
        return self.data['id']



class TimeSeriesParametricModel(ParametricModel):
    """A parametric model specifically designed to work with time series data. In particular,
    it ensures resolution and data keys consistency between methods and save/load operations.""" 
    
    def __init__(self, path=None, id=None):

        # Call parent init
        super(TimeSeriesParametricModel, self).__init__(path, id)

        # If the model has been loaded, convert resolution as TimeUnit
        if self.fitted:
            self.data['resolution'] = TimeUnit(self.data['resolution'])

    def save(self, *args, **kwargs):

        # If fitted, handle resolution. If not fitted, the parent save will raise.
        if self.fitted:
        
            # Save original data resolution (Unit or TimeUnit Object)
            or_resolution = self.data['resolution']
    
            # Temporarily change the model resolution unit as string representation
            self.data['resolution'] = str(self.data['resolution'])
            
        # Call parent save
        save_output  = super(TimeSeriesParametricModel, self).save(*args, **kwargs)
        
        # Set back original data resolution
        self.data['resolution'] = or_resolution
        
        # Return output
        return save_output


    def fit(self, timeseries, *args, **kwargs):
        
        # Check timeseries
        check_timeseries(timeseries)
        
        # Call parent fit
        fit_output = super(TimeSeriesParametricModel, self).fit(timeseries, *args, **kwargs)
        
        # Set timeseries resolution
        self.data['resolution'] = timeseries.resolution
        
        # Set timeseries data keys
        self.data['data_keys'] = timeseries.data_keys()

        # Return output
        return fit_output


    def predict(self, timeseries, *args, **kwargs):
        
        # Check timeseries and its resolution
        check_timeseries(timeseries)
        
        # If fitted, check resolution and keys. If not fitted, the parent init will raise.
        if self.fitted:
            check_resolution(timeseries, self.data['resolution'])
            check_data_keys(timeseries, self.data['data_keys'])
                
        # Call parent predict and return output
        return super(TimeSeriesParametricModel, self).predict(timeseries, *args, **kwargs)


    def evaluate(self, timeseries, *args, **kwargs):
        
        # Check timeseries and its resolution
        check_timeseries(timeseries)
        
        # If fitted, check resolution. If not fitted, the parent init will raise.
        if self.fitted:
            check_resolution(timeseries, self.data['resolution'])
                
        # Call parent evaluate and return output
        return super(TimeSeriesParametricModel, self).evaluate(timeseries, *args, **kwargs)


    def apply(self, timeseries, *args, **kwargs):
        
        # Check timeseries
        check_timeseries(timeseries)
        
        # If fitted, check resolution. If not fitted, the parent init will raise.
        if self.fitted:
            check_resolution(timeseries, self.data['resolution'])
        
        # Call parent apply and return output
        return super(TimeSeriesParametricModel, self).apply(timeseries, *args, **kwargs)


    def _cross_validate(self, data, *args, **kwargs):

        try:
            self._evaluate
        except AttributeError:
            raise NotImplementedError('Evaluating this model is not implemented, cannot cross-validate')

        # Check timeseries
        check_timeseries(data)
        
        # If fitted, check resolution. If not fitted, the parent init will raise.
        if self.fitted:
            check_resolution(data, self.data['resolution'])
        
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

        # For readability
        timeseries = data
        
        # How many rounds
        rounds = kwargs.pop('rounds', 10)

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
                self.fit(data, from_t=to_t, **fit_kwargs)
            else:
                logger.debug('Fitting until {} ({}) and then from {} ({}).'.format(to_t, to_dt, from_t, from_dt))
                self.fit(data, from_t=to_t, to_t=from_t, **fit_kwargs)                
            
            # Evaluate & append
            evaluations.append(self.evaluate(data, from_t=from_t, to_t=to_t, **evaluate_kwargs))
        
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



#=========================
#  Base Prophet model
#=========================

class ProphetModel(TimeSeriesParametricModel):
    '''Class to wrap some common logic for Prophet-based models.'''


    @classmethod
    def remove_timezone(cls, dt):
        return dt.replace(tzinfo=None)

    @classmethod
    def from_timeseria_to_prophet(cls, timeseries, from_t=None, to_t=None):

        # Create Python lists with data
        try:
            timeseries[0].data[0]
            data_keys_are_indexes = True
        except KeyError:
            timeseries[0].data.keys()
            data_keys_are_indexes = False
        
        data_as_list=[]
        for item in timeseries:
            
            # Skip if needed
            try:
                if not item_is_in_range(item, from_t, to_t):
                    continue
            except StopIteration:
                break                

            if data_keys_are_indexes:     
                data_as_list.append([cls.remove_timezone(item.dt), item.data[0]])
            else:
                data_as_list.append([cls.remove_timezone(item.dt), item.data[list(item.data.keys())[0]]])

        # Create the pandas DataFrames
        data = DataFrame(data_as_list, columns = ['ds', 'y'])

        return data



#=========================
#  Base ARIMA model
#=========================

class ARIMAModel():
    '''Class to wrap some common logic for ARIMA-based models.'''
    
    def get_start_end_indexes(self, timeseries, n):

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

class KerasModel(ParametricModel):
    '''Class to wrap some common logic for Keras-based models.'''


    def __init__(self, path=None, id=None):
        
        super(KerasModel, self).__init__(path=path, id=id)

        # If loaded, load the Keras model as well
        if path:
            from keras.models import load_model as load_keras_model
            self.keras_model = load_keras_model('{}/keras_model'.format(path))


    def save(self, path):

        # Save the parameters (the "data") property
        model_dir = super(KerasModel, self).save(path)

        # Now save the Keras model itself
        try:
            self.keras_model.save('{}/keras_model'.format(model_dir))
        except Exception as e:
            shutil.rmtree(model_dir)
            raise e
        return model_dir

    
    # TODO: this following stuff should not go here. Maybe only if extending a TimeSeriesParametricModel, and still it is probably wrong.
    
    #@staticmethod
    #def to_windows(timeseries):
    #    '''Compute window data values from a time series.'''
    #    key = timeseries.data_keys()[0]
    #    window_data_values = []
    #    for item in timeseries:
    #        window_data_values.append(item.data[key])
    #    return window_data_values

        
    @staticmethod
    def to_window_datapoints_matrix(timeseries, window, forecast_n, encoder=None):
        '''Compute window datapoints matrix from a time series.'''
    
        window_datapoints = []
        for i, _ in enumerate(timeseries):
            if i <  window:
                continue
            if i == len(timeseries) + 1 - forecast_n:
                break
                    
            # Add window values
            row = []
            for j in range(window):
                row.append(timeseries[i-window+j])
            window_datapoints.append(row)
                
        return window_datapoints

    
    @staticmethod
    def to_target_values_vector(timeseries, window, forecast_n):
        '''Compute target values vector from a time series.'''
    
        data_keys = timeseries.data_keys()
    
        targets = []
        for i, _ in enumerate(timeseries):
            if i <  window:
                continue
            if i == len(timeseries) + 1 - forecast_n:
                break
            
            # Add forecast target value(s)
            row = []
            for j in range(forecast_n):
                for data_key in data_keys:
                    row.append(timeseries[i+j].data[data_key])
            targets.append(row)

        return targets


    @staticmethod
    def compute_window_features(window_datapoints, data_keys, features):
        """Compute features from a list of window data points (or slots).
        
        Args:
            window_datapoints (list): The list with the data points (or slots)
            data_keys(dict): the keys of the point (or slot) data.
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
                for data_key in data_keys:
                    datapoint_features.append(window_datapoints[i].data[data_key])
                            
            # 2) Compute diffs on normalized datapoints
            if 'diffs' in features:
                for data_key in data_keys:
                    if i == 0:
                        diff = window_datapoints[1].data[data_key] - window_datapoints[0].data[data_key]
                    elif i == len(window_datapoints)-1:
                        diff = window_datapoints[-1].data[data_key] - window_datapoints[-2].data[data_key]
                    else:
                        diff = (window_datapoints[i+1].data[data_key] - window_datapoints[i-1].data[data_key]) /2
                    if diff == 0:
                        diff = 1
                    datapoint_features.append(diff)

            # 3) Hour (normlized)
            if 'hours' in features:
                datapoint_features.append(window_datapoints[i].dt.hour/24)
                
            # Now append to the window features
            window_features.append(datapoint_features)

        return window_features

