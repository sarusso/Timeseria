# -*- coding: utf-8 -*-
"""Anomaly detection models."""

from copy import deepcopy
from ..utilities import _Gaussian, rescale
from .forecasters import Forecaster, PeriodicAverageForecaster
from .reconstructors import Reconstructor, PeriodicAverageReconstructor
from .base import Model
from math import log10
from fitter import Fitter, get_common_distributions, get_distributions
from ..utilities import DistributionFunction


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

def _compute_distribution_approximation_errors(distribution_function, prediction_errors, bins=30, details=False):

    distribution_function_max_value = distribution_function(distribution_function.mu)

    # Support vars
    max_prediction_error = max(prediction_errors)
    min_prediction_error = min(prediction_errors)
    error_span = max_prediction_error - min_prediction_error

    # Create the bins
    error_step = error_span / bins
    prediction_error_bins = [[] for _ in range(bins)]
    for i, prediction_error in enumerate(prediction_errors):
        for j in range(bins-1):
            if ( (min_prediction_error + (error_step*j))  <= prediction_error <= (min_prediction_error + (error_step*(j+1))) ):
                prediction_error_bins[j].append(prediction_error)

    real_distribution_values = {} # x -> y
    max_len = 0
    for i, prediction_error_bin in enumerate(prediction_error_bins):
        if prediction_error_bin:
            real_distribution_values[min_prediction_error + (error_step*i) + (error_step/2)] = len(prediction_error_bin)
            if len(prediction_error_bin) > max_len:
                max_len = len(prediction_error_bin)

    # Normalize
    for key in real_distribution_values:
        real_distribution_values[key] = real_distribution_values[key]/max_len

    # Compute the approximation errors
    approximation_errors = []
    binned_distribution_values = []
    binned_real_distribution_values = []
    for key in real_distribution_values:
        #logger.debug('{}: \t{}\t vs\t {}'.format(key, distribution(key),real_distribution_values[key]))
        binned_distribution_values.append(distribution_function(key))
        binned_real_distribution_values.append(real_distribution_values[key])
        approximation_errors.append(abs(distribution_function(key)-real_distribution_values[key]))

    if details:
        return (approximation_errors, binned_distribution_values, binned_real_distribution_values)
    else:
        return approximation_errors

def _inspect_error_distribution_based_anomaly_detection_model(model, plot=False):
    print('Predictive model error (avg): {}'.format(sum(model.data['prediction_errors'])/len(model.data['prediction_errors'])))
    print('Predictive model error (min): {}'.format(min(model.data['prediction_errors'])))
    print('Predictive model error (max): {}'.format(max(model.data['prediction_errors'])))

    print('Error distribution: {}'.format(model.data['error_distribution']))
    print('Error distribution params: {}'.format(model.data['error_distribution_params']))
    print('Error distribution stats: {}'.format(model.data['error_distribution_stats']))

    if plot:
        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as stats

        x_min = min(model.data['prediction_errors'])
        x_max = max(model.data['prediction_errors'])

        # Plot the error distribution function
        distribution_function = DistributionFunction(model.data['error_distribution'],
                                                     model.data['error_distribution_params'])
        plt = distribution_function.plot(show=False, x_min=x_min, x_max=x_max)

        # Add the histogram to the plot
        plt.hist(model.data['prediction_errors'], bins=30, density=True, alpha=0.6, color='b')

        # Override title
        plt.title('Error distribution: {}'.format(model.data['error_distribution']))

        # Show the plot
        plt.show()

        #p = stats.norm.pdf(x, model.data['gaussian_mu'], model.data['gaussian_sigma'])
        #plt.plot(x, p, 'k', linewidth=2)
        #title = "Mu: {:.6f}   Sigma: {:.6f}".format(model.data['gaussian_mu'], model.data['gaussian_sigma'])
        #plt.title(title)
        #plt.show()

        # Plot both together
        #plt.plot(binned_distribution_values, color='black')
        #plt.plot(binned_real_distribution_values, color='blue')
        #plt.title('Error distribution function (black) vs real error distribution (blue)')
        #plt.show()


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

        distribution = kwargs.pop('distribution', None)
        if not distribution:
            distributions = kwargs.pop('distributions', get_common_distributions() +['gennorm'])
            #distributions = kwargs.pop('distributions', get_distributions())
        else:
            distributions = [distribution]

        summary = kwargs.pop('summary', False)

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

        # Fit the distributions and select the best one
        fitter_logger = logging.getLogger('fitter')
        fitter_logger.setLevel(level=logging.CRITICAL)

        fitter = Fitter(prediction_errors, distributions=distributions)
        fitter.fit(progress=False)

        if summary:
            print(fitter.summary())

        best_error_distribution = list(fitter.get_best().keys())[0]
        best_error_distribution_stats = fitter.summary(plot=False).transpose().to_dict()[best_error_distribution]
        error_distribution_params = fitter.get_best()[best_error_distribution]

        if best_error_distribution_stats['ks_pvalue'] < 0.05:

            logger.warning('The error distribution ({}) ks p-value is low ({}). '.format(best_error_distribution, best_error_distribution_stats['ks_pvalue']) + 
                           'Expect issues. In case of math domain errors, try using lower index boundaries.')

        if not (-0.01 <= error_distribution_params['loc'] <= 0.01):
            logger.warning('The error distribution is not centered in (almost) zero, but in {}. Expect issues.'.format(error_distribution_params['loc']))

        anomaly_detector.data['error_distribution'] = best_error_distribution
        anomaly_detector.data['error_distribution_params'] = error_distribution_params
        anomaly_detector.data['error_distribution_stats'] = best_error_distribution_stats

        from statistics import stdev
        anomaly_detector.data['stdev'] = stdev(prediction_errors)


def _apply_predictive_model_based_anomaly_detector(anomaly_detector, series, index_range=None, stdevs=None, emphasize=0,
                                                   details=False, rescale_function=None, add_adherence_index=False):

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

    distribution_function = DistributionFunction(anomaly_detector.data['error_distribution'],
                                                    anomaly_detector.data['error_distribution_params'])

    sigma = anomaly_detector.data['stdev'] 
    error_distribution_center = 0

    for data_label in series.data_labels():

        # Compute support stuff
        prediction_errors = anomaly_detector.data['prediction_errors']
        abs_prediction_errors = [abs(prediction_error) for prediction_error in prediction_errors]

        if index_range:

            for i in range(2):
                if isinstance(index_range[i], str):
                    if index_range[i] == 'max':
                        index_range[i] = max(abs_prediction_errors)

                    elif index_range[i] == 'avg':
                        index_range[i] = sum(abs_prediction_errors)/len(abs_prediction_errors)

                    elif index_range[i].endswith('_sigma'):
                        index_range[i] = float(index_range[i].replace('_sigma',''))*sigma

                    elif index_range[i].endswith('sig'):
                        index_range[i] = float(index_range[i].replace('sig',''))*sigma

            x_start = index_range[0]
            x_end = index_range[1]

            # Compute Normal (N) probabilities for start/end
            adherence_x_start = distribution_function.adherence(x_start)
            adherence_x_end = distribution_function.adherence(x_end)

        else:
            if not stdevs:
                raise ValueError('Got no index_range nor stdevs')


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

            # Compute the model adherence probability index
            if add_adherence_index:
                item.data_indexes['adherence'] = distribution_function(prediction_error)

            # Are threshold/cutoff stdevs defined?
            if stdevs or 'stdevs' in anomaly_detector.data:

                anomaly_index = 0

                # TODO: performance hit, move me outside the loop
                if prediction_error > 0:
                    error_threshold = error_distribution_center + (sigma * stdevs if stdevs else anomaly_detector.data['stdevs'])
                    if prediction_error > error_threshold:
                        anomaly_index = 1
                else:
                    error_threshold = error_distribution_center - (sigma * stdevs if stdevs else anomaly_detector.data['stdevs'])
                    if prediction_error < error_threshold:
                        anomaly_index = 1

                #if logs and anomaly_index == 1:
                #    logger.info('Detected anomaly for item @ {} ({}) with AE="{:.3f}..."'.format(item.t, item.dt, abs_prediction_error))

            else:

                # Compute the (continuous) anomaly index in the given range (which defaults to 0, max)
                # Below the start it means anomaly (0), above always anomaly (1). In the middle it 
                # follows the error distribution, and it is rescaled between 0 and 1.

                x = abs(prediction_error)

                # Adherence (0-1)
                adherence_x = distribution_function.adherence(x)

                if x <= x_start:
                    anomaly_index = 0
                elif x >= x_end:
                    anomaly_index = 1
                else:
                    try:
                        anomaly_index = (log10(adherence_x) - log10(adherence_x_start)) / (log10(adherence_x_end) - log10(adherence_x_start))
                    except ValueError as e:
                        if str(e) == 'math domain error':
                            raise ValueError('Got a math domain error. This is likely due to an error distribution function badly approximating the real error distribution. Try changing it, or using lower index boundaries.') from None
                        else:
                            raise
                        #raise ValueError('Got ')
                        #raise ValueError('adherence_x: {}, adherence_x_start: {}, adherence_x_end: {}'.format(adherence_x, adherence_x_start, adherence_x_end))

                # Do we have to apply any other rescaling?
                if rescale_function:
                    anomaly_index = rescale_function(anomaly_index)

                # Do we have to emphasize?
                if emphasize:
                    anomaly_index = ((2**(anomaly_index*emphasize))-1)/(((2**emphasize))-1)

            # Set the anomaly index
            item.data_indexes['anomaly'] = anomaly_index

            # Add details?
            if details:
                if isinstance(details, list):
                    if 'abs_error' in details:
                        item.data['{}_abs_error'.format(data_label)] = abs(prediction_error)
                    if 'error' in details:
                        item.data['{}_error'.format(data_label)] = prediction_error
                    if 'predicted' in details:
                        item.data['{}_predicted'.format(data_label)] = predicted
                else:
                    #item.data['{}_abs_error'.format(data_label)] = abs_prediction_error
                    item.data['{}_error'.format(data_label)] = prediction_error
                    item.data['{}_predicted'.format(data_label)] = predicted
                    item.data['distribution_function({}_error)'.format(data_label)] = distribution_function(prediction_error)

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

    def apply(self, series, index_range=['avg','max'], stdevs=None, emphasize=0,
              rescale_function=None, add_adherence_index=False, details=False):

        """Apply the anomaly detection model on a series.
        
        Args:
            index_range(tuple): the range from which the anomaly index is computed, in terms of prediction
                                error. Below the lower bound no anomaly will be assumed, and above always
                                anomalous (1). In the middle it follows the error Gaussian distribution.
                                Defaults to ['avg', 'max'] as it assumes to work in unsupervised mode.
                                Other supported values are 'x_sigma' where x is a sigma multiplier, or any
                                numerical value. A good choice for semi-spervised mode is ['max', '5_sigma'].
            
            stdevs(float): the threshold to consider a data point as anomalous (1) or not (0). Same as setting
                           ``index_range=('x_sigma','x_sigma')`` where x is the threshold. If set, overrides the
                           index_range argument. It basically discretize the anomaly index to 0/1 values.
                           
            emphasize(int): by how emphasizing the anomalies. It applies an exponential transformation to the index,
                            so that the range is compressed and "bigger" anomalies stand out more with respect to
                            "smaller" ones, that are compressed in the lower part of the anomaly index range.

            rescale_function(function): a custom anomaly index rescaling function. From [0,1] to [0,1].

            add_adherence_index(bool): adds an "adherence" index to the time series, to be intended as how much
                                       each data point is compatible with the model prediction. It is basically
                                       the error Gaussian value of each error value.

            details(bool): if set, adds details to the time series as the predicted value, the error etc.
        """
        return _apply_predictive_model_based_anomaly_detector(self, series, index_range=index_range, stdevs=stdevs, emphasize=emphasize,
                                                              rescale_function=rescale_function, add_adherence_index=add_adherence_index,
                                                              details=details)




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

    def apply(self, series, index_range=['avg','max'], stdevs=None, emphasize=0,
              rescale_function=None, add_adherence_index=False, details=False):

        """Apply the anomaly detection model on a series.
        
        Args:
            index_range(tuple): the range from which the anomaly index is computed, in terms of prediction
                                error. Below the lower bound no anomaly will be assumed, and above always
                                anomalous (1). In the middle it follows the error Gaussian distribution.
                                Defaults to ['avg', 'max'] as it assumes to work in unsupervised mode.
                                Other supported values are 'x_sigma' where x is a sigma multiplier, or any
                                numerical value. A good choice for semi-spervised mode is ['max', '5_sigma'].
    
            stdevs(float): the threshold to consider a data point as anomalous (1) or not (0). Same as setting
                           ``index_range=('x_sigma','x_sigma')`` where x is the threshold. If set, overrides the
                           index_range argument. It basically discretize the anomaly index to 0/1 values.
                           
            emphasize(int): by how emphasizing the anomalies. It applies an exponential transformation to the index,
                            so that the range is compressed and "bigger" anomalies stand out more with respect to
                            "smaller" ones, that are compressed in the lower part of the anomaly index range.

            rescale_function(function): a custom anomaly index rescaling function. From [0,1] to [0,1].

            add_adherence_index(bool): adds an "adherence" index to the time series, to be intended as how much
                                       each data point is compatible with the model prediction. It is basically
                                       the error Gaussian value of each error value.

            details(bool): if set, adds details to the time series as the predicted value, the error etc.
        """
        return _apply_predictive_model_based_anomaly_detector(self, series, index_range=index_range, stdevs=stdevs, emphasize=emphasize,
                                                              rescale_function=rescale_function, add_adherence_index=add_adherence_index,
                                                              details=details)


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


