# -*- coding: utf-8 -*-
"""Anomaly detection models."""

from copy import deepcopy

# Base models and utilities
from .base import TimeSeriesParametricModel
from .forecasters import PeriodicAverageForecaster

# Setup logging
import logging
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings as default behavior
try:
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except ImportError:
    pass

#=============================
#  Generic Anomaly Detector
#=============================

class AnomalyDetector(TimeSeriesParametricModel):
    
    def _predict(self, *args, **kwargs):
        raise NotImplementedError('Anomaly detectors can be used only from the apply() method.') from None



#=============================
# Forecaster Anomaly detector
#=============================

class ForecasterAnomalyDetector(AnomalyDetector):

    @property
    def forecaster_class(self):
        raise NotImplementedError('No forecaster set for this model. Please review your code.')


    def __init__(self, path=None, id=None, **kwargs):
        
        super(ForecasterAnomalyDetector, self).__init__(path=path, id=id)

        # Load the forecaster as nested model if we have loaded the model
        if self.fitted:
            # Note: the forecaster_id is the nested folder where the forecaster is saved
            forecaster_dir = path+'/'+self.data['forecaster_id']
            self.forecaster = self.forecaster_class(forecaster_dir)
        else:
            # Initialize the forecaster              
            self.forecaster = self.forecaster_class(**kwargs)
            
        # Finally, set the id of the forecaster in the data
        self.data['forecaster_id'] = self.forecaster.data['id']
            
    def save(self, path):

        # Save the anomaly detection model
        model_dir = super(ForecasterAnomalyDetector, self).save(path)

        # ..and save the forecaster as nested model
        self.forecaster.save(model_dir)

        # Return the model dir
        return model_dir


    def __get_actual_and_predicted(self, timeseries, i, key, forecaster_window):

        # Call model predict logic and compare with the actual data
        actual    = timeseries[i].data[key]
        
        try:
            # Try the optimized predict call (which just use the data as-is)
            prediction = self.forecaster.predict(timeseries, n=1, forecast_start = i-1)
            
        except TypeError as e:
            if '_predict() got an unexpected keyword argument \'forecast_start\'' in str(e):
                # Otherwise, create on the fly a slice of the time series for the window.
                # Datapoints are only linked, not copied - so this is just a minor overhead.
                window_timeseries = timeseries[i-forecaster_window:i]
                prediction = self.forecaster.predict(window_timeseries, n=1)
            else:
                raise e

        # TODO: this is because of forecasters not supporting multi-step forecasts.
        if not isinstance(prediction, list):
            predicted = prediction[key]
        else:
            predicted = prediction[0][key]
        
        return (actual, predicted)

    def _fit(self, timeseries, *args, stdevs=3, **kwargs):

        if len(timeseries.data_keys()) > 1:
            raise NotImplementedError('Multivariate time series are not yet supported')

        # Fit the forecaster
        self.forecaster.fit(timeseries, *args, **kwargs)
        
        # Evaluate the forecaster for one step ahead and get AEs
        AEs = []
        for key in timeseries.data_keys():
            
            for i, _ in enumerate(timeseries):
                
                forecaster_window = self.forecaster.data['window']
                
                if i <=  forecaster_window:    
                    continue
                
                actual, predicted = self.__get_actual_and_predicted(timeseries, i, key, forecaster_window)
                
                AEs.append(abs(actual-predicted))

        # Compute distribution for the AEs ans set the threshold
        from scipy.stats import norm
        mean, stdev = norm.fit(AEs)
        logger.info('Using {} standard deviations as anomaly threshold: {}'.format(stdevs, stdev*stdevs))
        
        # Set AE-based threshold
        self.data['stdev'] = stdev
        self.data['stdevs'] = stdevs
        self.data['AE_threshold'] = stdev*stdevs


    def _apply(self, timeseries, inplace=False, details=False, logs=False, stdevs=None):
        
        if inplace:
            raise Exception('Anomaly detection cannot be run inplace')

        if len(timeseries.data_keys()) > 1:
            raise NotImplementedError('Multivariate time series are not yet supported')
        
        result_timeseries = timeseries.__class__()

        for key in timeseries.data_keys():
            
            for i, item in enumerate(timeseries):
                forecaster_window = self.forecaster.data['window']
                if i <=  forecaster_window:    
                    continue
                
                actual, predicted = self.__get_actual_and_predicted(timeseries, i, key, forecaster_window)
                #if logs:
                #    logger.info('{}: {} vs {}'.format(timeseries[i].dt, actual, predicted))

                AE = abs(actual-predicted)
                
                item = deepcopy(item)
                
                if stdevs:
                    AE_threshold =  self.data['stdev'] * stdevs 
                else:
                    AE_threshold =  self.data['stdev'] * self.data['stdevs'] 
                    
                if AE > AE_threshold: 
                    if logs:
                        logger.info('Detected anomaly for item starting @ {} ({}) with AE="{:.3f}..."'.format(item.t, item.dt, AE))
                    item.anomaly = 1
                else:
                    item.anomaly = 0
                
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

                result_timeseries.append(item)
        
        return result_timeseries 



#=============================
# P. Average Anomaly detector
#=============================

class PeriodicAverageAnomalyDetector(ForecasterAnomalyDetector):

    forecaster_class = PeriodicAverageForecaster


