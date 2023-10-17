# -*- coding: utf-8 -*-
"""Anomaly detection models."""

from copy import deepcopy
from ..utilities import _Gaussian
from .forecasters import PeriodicAverageForecaster
from .base import Model
from math import log

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
    """A generic series anomaly detection model.
 
    Args:
        path (str): a path from which to load a saved model. Will override all other init settings.
    """

    def predict(self, series, *args, **kwargs):
        """Disabled. Anomaly detectors can be used only with the ``apply()`` method."""
        raise NotImplementedError('Anomaly detectors can be used only with the apply() method') from None

    def _predict(self, series, *args, **kwargs):
        raise NotImplementedError('Anomaly detectors can be used only with the apply() method') from None

    def evaluate(self, series, *args, **kwargs):
        """Disabled. Anomaly detectors cannot be evaluated yet."""
        raise NotImplementedError('Anomaly detectors cannot be evaluated yet.') from None

    def _evaluate(self, series, *args, **kwargs):
        raise NotImplementedError('Anomaly detectors cannot be evaluated yet.') from None

    def cross_validate(self, series, *args, **kwargs):
        """Disabled. Anomaly detectors cannot be evaluated yet."""
        raise NotImplementedError('Anomaly detectors cannot be evaluated yet.') from None

    def _cross_validate(self, series, *args, **kwargs):
        raise NotImplementedError('Anomaly detectors cannot be evaluated yet.') from None


#==================================
#  Forecast Anomaly detector
#==================================

class ForecasterAnomalyDetector(AnomalyDetector):
    """A series anomaly detection model based on a forecaster. The concept is simple: for each element of the series where
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

    def __get_actual_and_predicted(self, series, i, data_label, forecaster_window):

        # Call model predict logic and compare with the actual data
        actual = series[i].data[data_label]
        
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
            predicted = prediction[data_label]
        else:
            predicted = prediction[0][data_label]
        
        return (actual, predicted)

    def fit(self, series, *args, **kwargs):
        """Fit the anomaly detection model on a series.
        
        All the argument are forwarded to the forecaster ``fit()`` method.

        """
        return super(ForecasterAnomalyDetector, self).fit(series, *args, **kwargs)
    
    def _fit(self, series, *args, **kwargs):

        if len(series.data_labels()) > 1:
            raise NotImplementedError('Multivariate time series are not yet supported')

        # Fit the forecaster
        self.forecaster.fit(series, *args, **kwargs)
        
        # Evaluate the forecaster for one step ahead and get the forecasting errors
        forecasting_errors = []
        for data_label in series.data_labels():
            
            for i, _ in enumerate(series):
                
                forecaster_window = self.forecaster.data['window']
                
                if i <=  forecaster_window:    
                    continue
                
                actual, predicted = self.__get_actual_and_predicted(series, i, data_label, forecaster_window)
                
                forecasting_errors.append(actual-predicted)

        # Store the forecasting errors internally in the model
        self.data['forecasting_errors'] = forecasting_errors

        # TODO: test that the errors actually follow a normal (gaussian) distribution

        # Fit the Gaussian
        gaussian = _Gaussian.from_data(forecasting_errors)

        self.data['gaussian_mu'] = gaussian.mu
        self.data['gaussian_sigma'] = gaussian.sigma


    def inspect(self, plot=True):

        print('Gaussian Mu: {}'.format(self.data['gaussian_mu']))
        print('Gausian Sigma: {}'.format(self.data['gaussian_sigma']))
        print('Predictive model error (avg): {}'.format(sum(self.data['forecasting_errors'])/len(self.data['forecasting_errors'])))
        print('Predictive model error (min): {}'.format(min(self.data['forecasting_errors'])))
        print('Predictive model error (max): {}'.format(max(self.data['forecasting_errors'])))

        if plot:        
            import matplotlib.pyplot as plt
            import numpy as np
            import scipy.stats as stats
    
            # Plot the histogram
            plt.hist(self.data['forecasting_errors'], bins=25, density=True, alpha=0.6, color='b')
              
            # Plot the PDF
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            p = stats.norm.pdf(x, self.data['gaussian_mu'], self.data['gaussian_sigma'])
              
            plt.plot(x, p, 'k', linewidth=2)
            title = "Mu: {:.6f}   Sigma: {:.6f}".format(self.data['gaussian_mu'], self.data['gaussian_sigma'])
            plt.title(title)
              
            plt.show()

    #def _apply(self, series, inplace=False, details=False, logs=False, stdevs=None, rescale_function=None,
    #           emphasize=0, stdevs_range=None, anomaly_index_type='inv', adherence_index=False):

    def apply(self, series, stdevs_range=None, stdevs=None, emphasize=0, details=False,
              rescale_function=None, adherence_index=False, _anomaly_index_type='inv'):
        """Apply the anomaly detection model on a series.
        
        Args:
            stdevs_range(tuple): the range in which the anomaly index is computed. Defaults to (0, max).
                                 Below it means no anomaly (0), and above always anomaly (1). In the middle
                                 it follows the error Gaussian distribution rescaled between 0 and 1. Usually,
                                 a good choice is (3,6).
            
            stdevs(float): the threshold to consider a data point as anomalous (1) or not (0). Same as setting
                           ``stdevs_range=(x,x)`` where x is the threshold. If set, overrides the stdevs_range
                           argument. It basically discretize the anomaly index to 0/1 values.
                           
            emphasize(int): by how emphasizing the anomalies. It applies an exponential transformation to the index,
                            so that the range is compressed and "bigger" anomalies stand out more with respect to
                            "smaller" ones that are compressed in the lower part of the anomaly index range.
            
            details(bool): if set, adds details to the time series as the predicted value, the error etc.
            
            rescale_function(function): a custom anomaly index rescaling function. From [0,1] to [0,1].
            
            adherence_index(bool): adds an "adherence" index to the time series, to be intended as how much
                                   each data point is compatible with the model prediction. It is basically
                                   the error Gaussian value of each error value.             
        """

        #if inplace:
        #    raise Exception('Anomaly detection cannot be run inplace')

        if len(series.data_labels()) > 1:
            raise NotImplementedError('Multivariate time series are not yet supported')
        
        result_series = series.__class__()

        gaussian = _Gaussian(self.data['gaussian_mu'], self.data['gaussian_sigma'])

        for data_label in series.data_labels():
            
            # Compute support stuff          
            forecasting_errors = self.data['forecasting_errors']
            max_abs_forecasting_error = max(abs(max(forecasting_errors)),abs(min(forecasting_errors)))
            max_abs_forecasting_error_gaussian_value = gaussian(max_abs_forecasting_error)

            if stdevs_range:
                
                if stdevs_range[0] == max:
                    range_start_gaussian_value = max_abs_forecasting_error_gaussian_value
                else:
                    range_start_gaussian_value = gaussian((gaussian.sigma * stdevs_range[0]) + gaussian.mu)

                if stdevs_range[1] == max:
                    range_end_gaussian_value = max_abs_forecasting_error_gaussian_value
                else:
                    range_end_gaussian_value = gaussian(gaussian.sigma * stdevs_range[1])
            else:
                range_start_gaussian_value = None
                range_end_gaussian_value = None

            if False:
                logger.debug('max_forecasting_error={}'.format(max(forecasting_errors)))
                logger.debug('max_forecasting_error_gaussian_value={}'.format(gaussian(max(forecasting_errors))))
    
                logger.debug('min_forecasting_error={}'.format(min(forecasting_errors)))
                logger.debug('min_forecasting_error_gaussian_value={}'.format(gaussian(min(forecasting_errors))))
    
                logger.debug('max_abs_forecasting_error={}'.format(max_abs_forecasting_error))
                logger.debug('max_abs_forecasting_error_gaussian_value={}'.format(max_abs_forecasting_error_gaussian_value))
                
                logger.debug('stdevs: 1={} ({})'.format(gaussian.sigma *1, gaussian(gaussian.sigma *1)))
                logger.debug('stdevs: 2={} ({})'.format(gaussian.sigma *2, gaussian(gaussian.sigma *2)))
                logger.debug('stdevs: 3={} ({})'.format(gaussian.sigma *3, gaussian(gaussian.sigma *3)))
              
                logger.debug('range_start_gaussian_value={}'.format(range_start_gaussian_value))
                logger.debug('range_end_gaussian_value={}'.format(range_end_gaussian_value))

            for i, item in enumerate(series):
                forecaster_window = self.forecaster.data['window']
                if i <=  forecaster_window:    
                    continue

                # New item
                item = deepcopy(item)

                # Compute the anomaly index or threshold
                actual, predicted = self.__get_actual_and_predicted(series, i, data_label, forecaster_window)

                #if logs:
                #    logger.info('{}: {} vs {}'.format(series[i].dt, actual, predicted))

                forecasting_error = actual-predicted
                absolute_forecasting_error = abs(forecasting_error)
                
                
                # Compute the model adherence probability index
                if adherence_index:
                    item.data_indexes['adherence'] = gaussian(forecasting_error)
                   
                # Are threshold/cutoff stdevs defined?
                if stdevs or 'stdevs' in self.data:

                    anomaly_index = 0
                    
                    # TODO: performance hit, move me outside the loop
                    if forecasting_error > 0:
                        error_threshold = gaussian.mu + (gaussian.sigma * stdevs if stdevs else self.data['stdevs'])
                        if forecasting_error > error_threshold:
                            anomaly_index = 1
                    else:
                        error_threshold = gaussian.mu - (gaussian.sigma * stdevs if stdevs else self.data['stdevs'])
                        if forecasting_error < error_threshold:
                            anomaly_index = 1

                    #if logs and anomaly_index == 1:
                    #    logger.info('Detected anomaly for item @ {} ({}) with AE="{:.3f}..."'.format(item.t, item.dt, absolute_forecasting_error))

                else:
                    
                    # Compute a continuous anomaly index
                    
                    if stdevs_range and stdevs_range != [0, max]:
                        # The range in which the anomaly index is rescaled. Defaults to 0, max.
                        # Below no the start it means anomaly (0), above always anomaly (1).
                        # In the middle it follows the error Gaussian distribution and it is  
                        # rescaled between 0 and 1.
                        
                        # Get this forecasting error
                        forecasting_error_gaussian_value = gaussian(forecasting_error)

                        # Check boundaries and if not compute 
                        if forecasting_error_gaussian_value>=range_start_gaussian_value:
                            anomaly_index = 0
                        elif forecasting_error_gaussian_value<=range_end_gaussian_value:
                            anomaly_index = 1
                        else:
                            total_gaussian_value = range_start_gaussian_value - range_end_gaussian_value
                            portion_gaussian_value = forecasting_error_gaussian_value - range_end_gaussian_value
                            ratio_gaussian_value = portion_gaussian_value / total_gaussian_value
                            
                            if _anomaly_index_type=='inv':
                                anomaly_index = 1 - (ratio_gaussian_value)
                            elif _anomaly_index_type=='log':
                                anomaly_index = log(ratio_gaussian_value)/log(range_end_gaussian_value)
                            else:
                                raise ValueError('Unknown anomaly index type "{}"'.format(_anomaly_index_type))

                    else:
                        if _anomaly_index_type=='inv':
                            anomaly_index = 1 - gaussian(forecasting_error)
                        elif _anomaly_index_type=='log':
                            anomaly_index = log(gaussian(forecasting_error))/log(max_abs_forecasting_error_gaussian_value)                           
                        else:
                            raise ValueError('Unknown anomaly index type "{}"'.format(_anomaly_index_type))
                    
                    if rescale_function:
                        anomaly_index = rescale_function(anomaly_index)

                    if emphasize:
                        anomaly_index = ((2**(anomaly_index*emphasize))-1)/(((2**emphasize))-1)
                    
                # Set the anomaly index
                item.data_indexes['anomaly'] = anomaly_index

                # Add details?
                if details:
                    if isinstance(details, list):
                        if 'abs_error' in details:
                            item.data['{}_abs_error'.format(data_label)] = absolute_forecasting_error
                        if 'error' in details:
                            item.data['{}_error'.format(data_label)] = forecasting_error
                        if 'predicted' in details:
                            item.data['{}_predicted'.format(data_label)] = predicted
                    else:
                        #item.data['{}_abs_error'.format(data_label)] = absolute_forecasting_error
                        item.data['{}_error'.format(data_label)] = forecasting_error
                        item.data['{}_predicted'.format(data_label)] = predicted
                        item.data['gaussian({}_error)'.format(data_label)] = gaussian(forecasting_error)

                result_series.append(item)
        
        return result_series 


#==================================
# PAvg Forecaster Anomaly detector
#==================================

class PeriodicAverageAnomalyDetector(ForecasterAnomalyDetector):
    """A series anomaly detection model based on a periodic average forecaster. The concept is simple: for each element of the series
    where the anomaly detection model is applied, the forecaster is asked to make a prediction. If the actual value is too "far" 
    from the prediction, then that is marked as an anomaly. The "far" concept is expressed in terms of error standard deviations.
    
    Args:
        path (str): a path from which to load a saved model. Will override all other init settings.
        forecaster_class(Forecaster): the forecaster to be used for the anomaly detection model, if not already set by extending the model class.
    """
    
    forecaster_class = PeriodicAverageForecaster
