# -*- coding: utf-8 -*-
"""Anomaly detection models."""

from copy import deepcopy
from .forecasters import PeriodicAverageForecaster
from .base import Model

# Setup logging
import logging
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings as default behavior
try:
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except (ImportError,AttributeError):
    pass


#==================================
#  Generic Anomaly Detector
#==================================

class AnomalyDetector(Model):
    """An anomaly detector specifically designed to work with series data.
 
    Args:
        path (str): a path from which to load a saved model. Will override all other init settings.
    """

    def predict(self, input_data, *args, **kwargs):
        """Disabled. Anomaly detectors can be used only with the ``apply()`` method."""
        raise NotImplementedError('Anomaly detectors can be used only with the apply() method') from None

    def _predict(self, input_data, *args, **kwargs):
        raise NotImplementedError('Anomaly detectors can be used only with the apply() method') from None

    def evaluate(self, input_data, *args, **kwargs):
        """Disabled. Anomaly detectors cannot be evaluated yet."""
        raise NotImplementedError('Anomaly detectors cannot be evaluated yet.') from None

    def _evaluate(self, input_data, *args, **kwargs):
        raise NotImplementedError('Anomaly detectors cannot be evaluated yet.') from None

    def cross_validate(self, series, *args, **kwargs):
        """Disabled. Anomaly detectors cannot be evaluated yet."""
        raise NotImplementedError('Anomaly detectors cannot be evaluated yet.') from None

    def _cross_validate(self, input_data, *args, **kwargs):
        raise NotImplementedError('Anomaly detectors cannot be evaluated yet.') from None


#==================================
#  Forecast Anomaly detector
#==================================

class ForecasterAnomalyDetector(AnomalyDetector):
    """An anomaly detection model based on a forecaster. The concept is simple: for each element of the time series where
    the anomaly detection model is applied, the forecaster is asked to make a prediction. If the actual value is too "far"
    from the prediction, then that is an anomaly. The "far" concept is expressed in terms of error standard deviations.
    
    Args:
        path (str): a path from which to load a saved model. Will override all other init settings.
        forecaster_class(Forecaster): the forecaster to be used for the anomaly detection model, if not already set by extending the model class.
    """

    @property
    def forecaster_class(self):
        try:
            return self._forecaster_class
        except AttributeError:  
            raise NotImplementedError('No forecaster set for this model')

    def __init__(self, path=None, forecaster_class=None, *args, **kwargs):
        
        # Handle the forecaster_class
        try:
            self.forecaster_class
        except NotImplementedError:
            if not forecaster_class:
                raise ValueError('The forecaster_class is not set in the model nor given in the init')
        else:
            if forecaster_class:
                raise ValueError('The forecaster_class is given in the init but it is already set in the model')
            else:
                self._forecaster_class = forecaster_class            
        
        # Call parent init
        super(ForecasterAnomalyDetector, self).__init__(path=path)

        # Load the forecaster as nested model if we have loaded the model
        if self.fitted:
            # Note: the forecaster_id is the nested folder where the forecaster is saved
            forecaster_dir = path+'/'+self.data['forecaster_id']
            self.forecaster = self.forecaster_class(forecaster_dir)
        else:
            # Initialize the forecaster              
            self.forecaster = self.forecaster_class(*args, **kwargs)
            
        # Finally, set the id of the forecaster in the data
        self.data['forecaster_id'] = self.forecaster.data['id']
            
    def save(self, path):

        # Save the anomaly detection model
        super(ForecasterAnomalyDetector, self).save(path)

        # ..and save the forecaster as nested model
        self.forecaster.save(path+'/'+str(self.forecaster.id))

    def __get_actual_and_predicted(self, series, i, key, forecaster_window):

        # Call model predict logic and compare with the actual data
        actual = series[i].data[key]
        
        try:
            # Try the optimized predict call (which just use the data as-is)
            prediction = self.forecaster.predict(series, steps=1, forecast_start = i-1)
            
        except TypeError as e:
            if '_predict() got an unexpected keyword argument \'forecast_start\'' in str(e):
                # Otherwise, create on the fly a slice of the series for the window.
                # If items are Slots or Points, there are only linked, not copied,
                # so this is just a minor overhead. # TODO: series.view() ? Using an adapted _TimeSeriesView?
                window_series = series[i-forecaster_window:i]
                prediction = self.forecaster.predict(window_series, steps=1)
            else:
                raise e

        # TODO: this is because of forecasters not supporting multi-step forecasts.
        if not isinstance(prediction, list):
            predicted = prediction[key]
        else:
            predicted = prediction[0][key]
        
        return (actual, predicted)

    def fit(self, input_data, *args, stdevs=3, **kwargs):
        """Fit the anomaly detection model.
        
        All the parameters except ``stdevs`` are forwareded to the forecaster ``fit()`` method.
        
        Args:
            stdevs(float): how many standard deviations must there be between the actual and the predicted value
                           by the forecaster in order to mark a point or slot as anomalous.
        """
        return super(ForecasterAnomalyDetector, self).fit(input_data, *args, stdevs=stdevs, **kwargs)
    
    def _fit(self, input_data, *args, stdevs=3, **kwargs):

        series = input_data

        if len(series.data_labels()) > 1:
            raise NotImplementedError('Multivariate time series are not yet supported')

        # Fit the forecaster
        self.forecaster.fit(series, *args, **kwargs)
        
        # Evaluate the forecaster for one step ahead and get AEs
        AEs = []
        for key in series.data_labels():
            
            for i, _ in enumerate(series):
                
                forecaster_window = self.forecaster.data['window']
                
                if i <=  forecaster_window:    
                    continue
                
                actual, predicted = self.__get_actual_and_predicted(series, i, key, forecaster_window)
                
                AEs.append(abs(actual-predicted))

        # Compute distribution for the AEs ans set the threshold
        from scipy.stats import norm
        mean, stdev = norm.fit(AEs)
        logger.info('Using {} standard deviations as anomaly threshold: {}'.format(stdevs, stdev*stdevs))
        
        # Set AE-based threshold
        self.data['stdev'] = stdev
        self.data['stdevs'] = stdevs
        self.data['AE_threshold'] = stdev*stdevs

    def _apply(self, input_data, inplace=False, details=False, logs=False, stdevs=None):
        
        series = input_data
        
        if inplace:
            raise Exception('Anomaly detection cannot be run inplace')

        if len(series.data_labels()) > 1:
            raise NotImplementedError('Multivariate time series are not yet supported')
        
        result_series = series.__class__()

        for key in series.data_labels():
            
            for i, item in enumerate(series):
                forecaster_window = self.forecaster.data['window']
                if i <=  forecaster_window:    
                    continue
                
                actual, predicted = self.__get_actual_and_predicted(series, i, key, forecaster_window)
                #if logs:
                #    logger.info('{}: {} vs {}'.format(series[i].dt, actual, predicted))

                AE = abs(actual-predicted)
                
                item = deepcopy(item)
                
                if stdevs:
                    AE_threshold =  self.data['stdev'] * stdevs 
                else:
                    AE_threshold =  self.data['stdev'] * self.data['stdevs'] 
                    
                if AE > AE_threshold: 
                    if logs:
                        logger.info('Detected anomaly for item starting @ {} ({}) with AE="{:.3f}..."'.format(item.t, item.dt, AE))
                    item.data_indexes['anomaly'] = 1
                else:
                    item.data_indexes['anomaly'] = 0
                
                # Add details?
                if details:
                    if isinstance(details, list):
                        if 'AE' in details:
                            item.data['AE_{}'.format(key)] = AE
                        if 'predicted' in details:
                            item.data['{}_predicted'.format(key)] = predicted
                    else:
                        item.data['AE_{}'.format(key)] = AE
                        item.data['{}_predicted'.format(key)] = predicted

                result_series.append(item)
        
        return result_series 


#==================================
# PAvg Forecaster Anomaly detector
#==================================

class PeriodicAverageAnomalyDetector(ForecasterAnomalyDetector):
    """An anomaly detection model based on a periodic average forecaster. The concept is simple: for each element of the time series
    where the anomaly detection model is applied, the forecaster is asked to make a prediction. If the actual value is too "far" 
    from the prediction, then that is an anomaly. The "far" concept is expressed in terms of error standard deviations.
    
    Args:
        path (str): a path from which to load a saved model. Will override all other init settings.
        forecaster_class(Forecaster): the forecaster to be used for the anomaly detection model, if not already set by extending the model class.
    """
    
    forecaster_class = PeriodicAverageForecaster
