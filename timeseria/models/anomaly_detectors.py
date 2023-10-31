# -*- coding: utf-8 -*-
"""Anomaly detection models."""

from copy import deepcopy
from ..utilities import _Gaussian
from .forecasters import Forecaster, PeriodicAverageForecaster
from .reconstructors import Reconstructor, PeriodicAverageReconstructor
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



#===================================
#  Utility functions
#===================================

def _inspect_error_distribution_based_anomaly_detection_model(model, plot=False):
    print('Gaussian Mu: {}'.format(model.data['gaussian_mu']))
    print('Gausian Sigma: {}'.format(model.data['gaussian_sigma']))
    print('Predictive model error (avg): {}'.format(sum(model.data['prediction_errors'])/len(model.data['prediction_errors'])))
    print('Predictive model error (min): {}'.format(min(model.data['prediction_errors'])))
    print('Predictive model error (max): {}'.format(max(model.data['prediction_errors'])))

    if plot:        
        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as stats

        # Plot the histogram
        plt.hist(model.data['prediction_errors'], bins=25, density=True, alpha=0.6, color='b')
          
        # Plot the PDF
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, model.data['gaussian_mu'], model.data['gaussian_sigma'])
          
        plt.plot(x, p, 'k', linewidth=2)
        title = "Mu: {:.6f}   Sigma: {:.6f}".format(model.data['gaussian_mu'], model.data['gaussian_sigma'])
        plt.title(title)
          
        plt.show()


def _get_actual_and_predicted_from_predictive_model(predictive_model, series, i, data_label, window):

        # Call model predict logic and compare with the actual data
        actual = series[i].data[data_label]
        if isinstance(predictive_model, Reconstructor):    
            prediction = predictive_model.predict(series, from_i=i,to_i=i)
        elif isinstance(predictive_model, Forecaster):
            prediction = predictive_model.predict(series, steps=1, forecast_start = i-1) # TOOD: why th "-1"?
        else:
            raise TypeError('Don\'t know how to handle predictive model of type "{}"'.format(predictive_model.__class__.__name__))

        #except TypeError as e:
        #    
        #    if '_predict() got an unexpected keyword argument \'forecast_start\'' in str(e):
        #        # Otherwise, create on the fly a slice of the series for the window.
        #        # so this is just a minor overhead. # TODO: series.view() ? Using an adapted _TimeSeriesView?
        #        # If items are Slots or Points, there are only linked, not copied,
        #        window_series = series[i-window:i]
        #        prediction = self.reconstructor.predict(window_series, steps=1)
        #    else:
        #        raise e

        # Handle list of dicts or dict of lists (of wich we have only one value here)
        #{'value': [0.2019341593004146, 0.29462641146884005]}
        
        if isinstance(prediction, list):
            predicted = prediction[0][data_label]        
        elif isinstance(prediction, dict):
            predicted = prediction[data_label][0]
        else:
            raise TypeError('Don\'t know how to handle a prediction with of type "{}"'.format(prediction.__class__.__name__))

        return (actual, predicted)


def _fit_predictive_model_based_anomaly_detector(anomaly_detector, series, *args, **kwargs):

        if len(series.data_labels()) > 1:
            raise NotImplementedError('Multivariate time series are not yet supported')

        # Detect the predictive model being used
        predictive_model = None
        try:
            # Reconstructor-based?
            predictive_model = anomaly_detector.reconstructor
        except AttributeError:
            pass
        try:
            # Forecaster-based?
            predictive_model = anomaly_detector.forecaster
        except AttributeError:
            pass
        if not predictive_model:
            raise Exception('Unknown predictive model')

        # Fit the predictive model
        predictive_model.fit(series, *args, **kwargs)
        
        # Evaluate the predictive for one step ahead and get the forecasting errors
        prediction_errors = []
        for data_label in series.data_labels():
            
            for i, _ in enumerate(series):
                
                # Before the window
                if i <=  predictive_model.window:    
                    continue
                
                # After the window
                if i >  len(series)-predictive_model.window-1:    
                    break 
                
                # Predict & append the error
                actual, predicted = _get_actual_and_predicted_from_predictive_model(predictive_model, series, i, data_label, predictive_model.window)
                prediction_errors.append(actual-predicted)

        # Store the forecasting errors internally in the model
        anomaly_detector.data['prediction_errors'] = prediction_errors

        # TODO: test that the errors actually follow a normal (gaussian) distribution

        # Fit the Gaussian
        gaussian = _Gaussian.from_data(prediction_errors)

        anomaly_detector.data['gaussian_mu'] = gaussian.mu
        anomaly_detector.data['gaussian_sigma'] = gaussian.sigma


def _apply_predictive_model_based_anomaly_detector(anomaly_detector, series, stdevs_range=None, stdevs=None, emphasize=0, details=False,
                                                   rescale_function=None, adherence_index=False, _anomaly_index_type='inv'):

    #if inplace:
    #    raise Exception('Anomaly detection cannot be run inplace')

    if len(series.data_labels()) > 1:
        raise NotImplementedError('Multivariate time series are not yet supported')

    # Detect the predictive model being used
    predictive_model = None
    try:
        # Reconstructor-based?
        predictive_model = anomaly_detector.reconstructor
    except AttributeError:
        pass
    try:
        # Forecaster-based?
        predictive_model = anomaly_detector.forecaster
    except AttributeError:
        pass
    if not predictive_model:
        raise Exception('Unknown predictive model')

    result_series = series.__class__()

    gaussian = _Gaussian(anomaly_detector.data['gaussian_mu'], anomaly_detector.data['gaussian_sigma'])

    for data_label in series.data_labels():
        
        # Compute support stuff          
        prediction_errors = anomaly_detector.data['prediction_errors']
        max_abs_prediction_error = max(abs(max(prediction_errors)),abs(min(prediction_errors)))
        max_abs_prediction_error_gaussian_value = gaussian(max_abs_prediction_error)

        if stdevs_range:
            
            if stdevs_range[0] == max:
                range_start_gaussian_value = max_abs_prediction_error_gaussian_value
            else:
                range_start_gaussian_value = gaussian((gaussian.sigma * stdevs_range[0]) + gaussian.mu)

            if stdevs_range[1] == max:
                range_end_gaussian_value = max_abs_prediction_error_gaussian_value
            else:
                range_end_gaussian_value = gaussian(gaussian.sigma * stdevs_range[1])
        else:
            range_start_gaussian_value = None
            range_end_gaussian_value = None

        if False:
            logger.debug('max_prediction_error={}'.format(max(prediction_errors)))
            logger.debug('max_prediction_error_gaussian_value={}'.format(gaussian(max(prediction_errors))))

            logger.debug('min_prediction_error={}'.format(min(prediction_errors)))
            logger.debug('min_prediction_error_gaussian_value={}'.format(gaussian(min(prediction_errors))))

            logger.debug('max_abs_prediction_error={}'.format(max_abs_prediction_error))
            logger.debug('max_abs_prediction_error_gaussian_value={}'.format(max_abs_prediction_error_gaussian_value))
            
            logger.debug('stdevs: 1={} ({})'.format(gaussian.sigma *1, gaussian(gaussian.sigma *1)))
            logger.debug('stdevs: 2={} ({})'.format(gaussian.sigma *2, gaussian(gaussian.sigma *2)))
            logger.debug('stdevs: 3={} ({})'.format(gaussian.sigma *3, gaussian(gaussian.sigma *3)))
          
            logger.debug('range_start_gaussian_value={}'.format(range_start_gaussian_value))
            logger.debug('range_end_gaussian_value={}'.format(range_end_gaussian_value))

        for i, item in enumerate(series):
            forecaster_window = predictive_model.window

            # Before the window 
            if i <=  predictive_model.window:    
                continue
            
            # After the window
            if i >  len(series)-predictive_model.window-1:    
                break

            # New item
            item = deepcopy(item)

            # Compute the anomaly index or threshold
            actual, predicted = _get_actual_and_predicted_from_predictive_model(predictive_model, series, i, data_label, forecaster_window)

            #if logs:
            #    logger.info('{}: {} vs {}'.format(series[i].dt, actual, predicted))

            prediction_error = actual-predicted
            absolute_prediction_error = abs(prediction_error)
            
            # Compute the model adherence probability index
            if adherence_index:
                item.data_indexes['adherence'] = gaussian(prediction_error)
               
            # Are threshold/cutoff stdevs defined?
            if stdevs or 'stdevs' in anomaly_detector.data:

                anomaly_index = 0
                
                # TODO: performance hit, move me outside the loop
                if prediction_error > 0:
                    error_threshold = gaussian.mu + (gaussian.sigma * stdevs if stdevs else anomaly_detector.data['stdevs'])
                    if prediction_error > error_threshold:
                        anomaly_index = 1
                else:
                    error_threshold = gaussian.mu - (gaussian.sigma * stdevs if stdevs else anomaly_detector.data['stdevs'])
                    if prediction_error < error_threshold:
                        anomaly_index = 1

                #if logs and anomaly_index == 1:
                #    logger.info('Detected anomaly for item @ {} ({}) with AE="{:.3f}..."'.format(item.t, item.dt, absolute_prediction_error))

            else:
                
                # Compute a continuous anomaly index
                
                if stdevs_range and stdevs_range != [0, max]:
                    # The range in which the anomaly index is rescaled. Defaults to 0, max.
                    # Below no the start it means anomaly (0), above always anomaly (1).
                    # In the middle it follows the error Gaussian distribution and it is  
                    # rescaled between 0 and 1.
                    
                    # Get this forecasting error
                    prediction_error_gaussian_value = gaussian(prediction_error)

                    # Check boundaries and if not compute 
                    if prediction_error_gaussian_value>=range_start_gaussian_value:
                        anomaly_index = 0
                    elif prediction_error_gaussian_value<=range_end_gaussian_value:
                        anomaly_index = 1
                    else:
                        total_gaussian_value = range_start_gaussian_value - range_end_gaussian_value
                        portion_gaussian_value = prediction_error_gaussian_value - range_end_gaussian_value
                        ratio_gaussian_value = portion_gaussian_value / total_gaussian_value
                        
                        if _anomaly_index_type=='inv':
                            anomaly_index = 1 - (ratio_gaussian_value)
                        elif _anomaly_index_type=='log':
                            anomaly_index = log(ratio_gaussian_value)/log(range_end_gaussian_value)
                        else:
                            raise ValueError('Unknown anomaly index type "{}"'.format(_anomaly_index_type))

                else:
                    if _anomaly_index_type=='inv':
                        anomaly_index = 1 - gaussian(prediction_error)
                    elif _anomaly_index_type=='log':
                        anomaly_index = log(gaussian(prediction_error))/log(max_abs_prediction_error_gaussian_value)                           
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
                        item.data['{}_abs_error'.format(data_label)] = absolute_prediction_error
                    if 'error' in details:
                        item.data['{}_error'.format(data_label)] = prediction_error
                    if 'predicted' in details:
                        item.data['{}_predicted'.format(data_label)] = predicted
                else:
                    #item.data['{}_abs_error'.format(data_label)] = absolute_prediction_error
                    item.data['{}_error'.format(data_label)] = prediction_error
                    item.data['{}_predicted'.format(data_label)] = predicted
                    item.data['gaussian({}_error)'.format(data_label)] = gaussian(prediction_error)

            result_series.append(item)
    
    return result_series 


#===================================
#  Generic Anomaly Detector
#===================================

class AnomalyDetector(Model):
    """A generic anomaly detection model.
 
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


#===================================
#  Generic LIVE Anomaly Detector
#===================================

class LiveAnomalyDetector(AnomalyDetector):
    """An generic live anomaly detection model.
 
    Args:
        path (str): a path from which to load a saved model. Will override all other init settings.
    """
    pass


#===================================
#  Predictive Anomaly Detector
#===================================

class PredictiveAnomalyDetector(AnomalyDetector):
    """A series anomaly detection model based on a reconstructor. The concept is simple: for each element of the series where
    the anomaly detection model is applied, the reconstructor is asked to make a prediction. If the actual value is too "far"
    from the prediction, then it is marked as anomalous. The "far" concept is expressed in terms of error standard deviations.
    
    Args:
        path (str): a path from which to load a saved model. Will override all other init settings.
        reconstructor_class(Reconstructor): the reconstructor to be used for the anomaly detection model, if not already set.
    """

    @property
    def reconstructor_class(self):
        try:
            return self._reconstructor_class
        except AttributeError:  
            raise NotImplementedError('No reconstructor set for this model')

    def __init__(self, path=None, reconstructor_class=None, *args, **kwargs):
        
        # Handle the reconstructor_class
        try:
            self.reconstructor_class
        except NotImplementedError:
            if not reconstructor_class:
                raise ValueError('The reconstructor_class is not set in the model nor given in the init')
        else:
            if reconstructor_class:
                raise ValueError('The reconstructor_class was given in the init but it is already set in the model')
            else:
                self._reconstructor_class = reconstructor_class            
        
        # Call parent init
        super(PredictiveAnomalyDetector, self).__init__(path=path)

        # Load the reconstructor as nested model if we have loaded the model
        if self.fitted:
            # Note: the reconstructor_id is the nested folder where the reconstructor is saved
            reconstructor_dir = path+'/'+self.data['reconstructor_id']
            self.reconstructor = self.reconstructor_class(reconstructor_dir)
        else:
            # Initialize the reconstructor              
            self.reconstructor = self.reconstructor_class(*args, **kwargs)
            
        # Finally, set the id of the reconstructor in the data
        self.data['reconstructor_id'] = self.reconstructor.data['id']
            
    def save(self, path):

        # Save the anomaly detection model
        super(PredictiveAnomalyDetector, self).save(path)

        # ..and save the reconstructor as nested model
        self.reconstructor.save(path+'/'+str(self.reconstructor.id))

    def _fit(self, series, *args, **kwargs):
        _fit_predictive_model_based_anomaly_detector(self, series, *args, **kwargs)
        
    def inspect(self, plot=True):
        '''Inspect the model and plot the error distribution'''        
        _inspect_error_distribution_based_anomaly_detection_model(self, plot=plot)

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
        return _apply_predictive_model_based_anomaly_detector(self, series, stdevs_range=stdevs_range, stdevs=stdevs, emphasize=emphasize, details=details,
                                                              rescale_function=rescale_function, adherence_index=adherence_index, _anomaly_index_type=_anomaly_index_type)




#===================================
#  Predictive LIVE Anomaly Detector
#===================================

class PredictiveLiveAnomalyDetector(LiveAnomalyDetector):
    """A series anomaly detection model based on a forecaster. The concept is simple: for each element of the series where
    the anomaly detection model is applied, the forecaster is asked to make a prediction. If the actual value is too "far"
    from the prediction, then it is marked as anomalous. The "far" concept is expressed in terms of error standard deviations.
    
    Args:
        path (str): a path from which to load a saved model. Will override all other init settings.
        forecaster_class(Forecaster): the forecaster to be used for the anomaly detection model, if not already set.
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
                raise ValueError('The forecaster_class was given in the init but it is already set in the model')
            else:
                self._forecaster_class = forecaster_class            
        
        # Call parent init
        super(PredictiveLiveAnomalyDetector, self).__init__(path=path)

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
        super(PredictiveLiveAnomalyDetector, self).save(path)

        # ..and save the forecaster as nested model
        self.forecaster.save(path+'/'+str(self.forecaster.id))

    def fit(self, series, *args, **kwargs):
        """Fit the anomaly detection model on a series.
        
        All the argument are forwarded to the forecaster ``fit()`` method.

        """
        return super(PredictiveLiveAnomalyDetector, self).fit(series, *args, **kwargs)
    
    def _fit(self, series, *args, **kwargs):
        _fit_predictive_model_based_anomaly_detector(self, series, *args, **kwargs)
        
    def inspect(self, plot=True):
        '''Inspect the model and plot the error distribution'''        
        _inspect_error_distribution_based_anomaly_detection_model(self, plot=plot)

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
        return _apply_predictive_model_based_anomaly_detector(self, series, stdevs_range=stdevs_range, stdevs=stdevs, emphasize=emphasize, details=details,
                                                              rescale_function=rescale_function, adherence_index=adherence_index, _anomaly_index_type=_anomaly_index_type)



#===================================
#        Periodic Average
#   Predictive Anomaly Detector
#===================================

class PeriodicAverageAnomalyDetector(PredictiveAnomalyDetector):
    """An anomaly detection model based on a periodic average reconstructor.
    
    Args:
        path (str): a path from which to load a saved model. Will override all other init settings.
        forecaster_class(Forecaster): the forecaster to be used for the anomaly detection model, if not already set by extending the model class.
    """
    
    reconstructor_class = PeriodicAverageReconstructor


#===================================
#        Periodic Average
# Predictive LIVE Anomaly Detector
#===================================

class PeriodicAverageLiveAnomalyDetector(PredictiveLiveAnomalyDetector):
    """An anomaly detection model based on a periodic average forecaster.
    
    Args:
        path (str): a path from which to load a saved model. Will override all other init settings.
        forecaster_class(Forecaster): the forecaster to be used for the anomaly detection model, if not already set by extending the model class.
    """
    
    forecaster_class = PeriodicAverageForecaster


