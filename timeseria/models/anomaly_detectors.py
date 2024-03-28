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

import fitter as fitter_library



# Setup logging
import logging
logger = logging.getLogger(__name__)

fitter_library.fitter.logger = logging.getLogger('fitter')
fitter_library.fitter.logger.setLevel(level=logging.CRITICAL)



# Suppress TensorFlow warnings as default behavior
try:
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except (ImportError,AttributeError):
    pass


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
#  Model-based Anomaly Detector
#===================================

class ModelBasedAnomalyDetector(AnomalyDetector):
    """An anomaly detection model based on another model (either a forecaster or a reconstructor).
    For each element of the series where the anomaly detection model is applied, the model is asked to make a prediction.
    The predicted and actual values are then compared, and accordingly to the model error distribution, an anomaly index
    in the range 0-1 is computed.

    Args:
        path (str): a path from which to load a saved model. Will override all other init settings.
        model_class(Forecaster,Reconstructor): the model to be used for anomaly detection, if not already set.
    """

    @property
    def model_class(self):
        try:
            return self._model_class
        except AttributeError:
            raise NotImplementedError('No model class set for this anomaly detector')

    def __init__(self, path=None, model_class=None, *args, **kwargs):

        # Handle the model_class
        try:
            self.model_class
        except NotImplementedError:
            if not model_class:
                raise ValueError('The model_class is not set in the anomaly detector nor given in the init')
        else:
            if model_class:
                raise ValueError('The model_class was given in the init but it is already set in the anomaly detector')
            else:
                self._model_class = model_class

        # Call parent init
        super(ModelBasedAnomalyDetector, self).__init__(path=path)

        # Load the model as nested model if we have loaded the model
        if self.fitted:
            # Note: the model_id is the nested folder where the model is saved
            model_dir = path+'/'+self.data['model_id']
            self.model= self.model_class(model_dir)
        else:
            # Initialize the predictive model
            self.model = self.model_class(*args, **kwargs)

        # Finally, set the id of the model in the data
        self.data['model_id'] = self.model.data['id']

    def save(self, path):

        # Save the anomaly detection model
        super(ModelBasedAnomalyDetector, self).save(path)

        # ..and save the inner model as nested model
        self.model.save(path+'/'+str(self.model.id))


    def _get_actual_and_predicted(self, series, i, data_label, window):

            # Call model predict logic and compare with the actual data
            actual = series[i].data[data_label]
            if isinstance(self.model, Reconstructor):
                prediction = self.model.predict(series, from_i=i,to_i=i)
            elif isinstance(self.model, Forecaster):
                prediction = self.model.predict(series, steps=1, forecast_start = i-1) # TODO: why the "-1"?
            else:
                raise TypeError('Don\'t know how to handle predictive model of type "{}"'.format(self.model.__class__.__name__))

            # Handle list of dicts or dict of lists (of wich we have only one value here)
            #{'value': [0.2019341593004146, 0.29462641146884005]}

            if isinstance(prediction, list):
                predicted = prediction[0][data_label]
            elif isinstance(prediction, dict):
                predicted = prediction[data_label][0]
            else:
                raise TypeError('Don\'t know how to handle a prediction with of type "{}"'.format(prediction.__class__.__name__))

            return (actual, predicted)


    def _fit(self, series, *args, **kwargs):

        if len(series.data_labels()) > 1:
            raise NotImplementedError('Multivariate time series are not yet supported')

        error_distribution = kwargs.pop('error_distribution', None)
        if not error_distribution:
            error_distributions = kwargs.pop('error_distributions', fitter_library.fitter.get_common_distributions() +['gennorm'])
            #distributions = kwargs.pop('distributions', fitter_library.fitter.get_distributions())
        else:
            error_distributions = [error_distribution]

        summary = kwargs.pop('summary', False)

        # Fit the predictive model
        self.model.fit(series, *args, **kwargs)

        # Evaluate the predictive for one step ahead and get the forecasting errors
        prediction_errors = []
        for data_label in series.data_labels():

            for i, _ in enumerate(series):

                # Before the window
                if i <=  self.model.window:
                    continue

                # After the window (if using a reconstructor)
                if isinstance(self.model, Reconstructor):
                    if i > len(series)-self.model.window-1:
                        break

                # Predict & append the error
                actual, predicted = self._get_actual_and_predicted(series, i, data_label, self.model.window)
                prediction_errors.append(actual-predicted)

        # Store the forecasting errors internally in the model
        self.data['prediction_errors'] = prediction_errors

        # Fit the distributions and select the best one
        fitter = fitter_library.fitter.Fitter(prediction_errors, distributions=error_distributions)
        fitter.fit(progress=False)

        if summary:
            # Warning: the summary() function will also generate a plot
            print(fitter.summary())

        best_error_distribution = list(fitter.get_best().keys())[0]
        best_error_distribution_stats = fitter.summary(plot=False).transpose().to_dict()[best_error_distribution]
        error_distribution_params = fitter.get_best()[best_error_distribution]

        if best_error_distribution_stats['ks_pvalue'] < 0.05:

            logger.warning('The error distribution ({}) ks p-value is low ({}). '.format(best_error_distribution, best_error_distribution_stats['ks_pvalue']) +
                           'Expect issues. In case of math domain errors, try using lower index boundaries.')

        if not (-0.01 <= error_distribution_params['loc'] <= 0.01):
            logger.warning('The error distribution is not centered in (almost) zero, but in {}. Expect issues.'.format(error_distribution_params['loc']))

        self.data['error_distribution'] = best_error_distribution
        self.data['error_distribution_params'] = error_distribution_params
        self.data['error_distribution_stats'] = best_error_distribution_stats

        from statistics import stdev
        self.data['stdev'] = stdev(prediction_errors)

    def inspect(self, plot=True):
        '''Inspect the model and plot the error distribution'''

        abs_prediction_errors = [abs(prediction_error) for prediction_error in self.data['prediction_errors']]

        print('Predictive model avg error (abs): {}'.format(sum(abs_prediction_errors)/len(abs_prediction_errors)))
        print('Predictive model min error (abs): {}'.format(min(abs_prediction_errors)))
        print('Predictive model max error (abs): {}'.format(max(abs_prediction_errors)))

        print('Error distribution: {}'.format(self.data['error_distribution']))
        print('Error distribution params: {}'.format(self.data['error_distribution_params']))
        print('Error distribution stats: {}'.format(self.data['error_distribution_stats']))

        if plot:

            x_min = min(self.data['prediction_errors'])
            x_max = max(self.data['prediction_errors'])

            # Instantiate the errro distribution function
            distribution_function = DistributionFunction(self.data['error_distribution'],
                                                         self.data['error_distribution_params'])

            # Get the error distribution function plot
            plt = distribution_function.plot(show=False, x_min=x_min, x_max=x_max)

            # Add the histogram to the plot
            plt.hist(self.data['prediction_errors'], bins=100, density=True, alpha=1, color='steelblue')

            # Override title
            #plt.title('Error distribution: {}'.format(self.data['error_distribution']))

            # Show the plot
            plt.show()


    def apply(self, series, index_range=['avg_err','max_err'], index_type='log', threshold=None, details=False):

        """Apply the anomaly detection model on a series.

        Args:
            index_range(tuple): the range from which the anomaly index is computed, in terms of prediction
                                error. Below the lower bound no anomaly will be assumed, and above always
                                anomalous (1). In the middle it follows the error distribution.
                                Defaults to ['avg_err', 'max_err'] as it assumes to work in unsupervised mode.
                                Other supported values are 'x_sigma' where x is a sigma multiplier, or any
                                numerical value. A good choice for semi-spervised mode is ['max_err', '5_sigma'].

            index_type(str, callable): if to use a logarithmic anomaly index ("log", the default value) which compresses
                                       the index range so that bigger anomalies stand out more than smaller ones, or if to
                                       use a linear one ("lin"). Can also support a custom anomaly index as a callable,
                                       in which case the form must be ``f(x, y, x_start, x_end, y_start, y_end)`` where x
                                       is the model error, y its value on the distribution curve, and x_start/x_end together
                                       with y_start/y_end the respective x and y range start and end values, based on the
                                       range set by the ``index_range`` argument.

            threshold(float): a threshold to make the anomaly index categorical (0-1) instead of continuous.

            details(bool, list): if to add details to the time series as the predicted value, the error and the
                                 corresponding error distribution function (dist) value. If set to True, it adds
                                 all of them, if instead using a list only selected details can be added: "pred"
                                 for the predicted values, "err" for the error, and "dist" for the error distribution.
        """

        if len(series.data_labels()) > 1:
            raise NotImplementedError('Multivariate time series are not yet supported')

        # Support vars
        result_series = series.__class__()
        sigma = self.data['stdev']

        # Initialize the error distribution function
        error_distribution_function = DistributionFunction(self.data['error_distribution'],
                                                           self.data['error_distribution_params'])

        for data_label in series.data_labels():

            # Set anomaly index boundaries
            prediction_errors = self.data['prediction_errors']
            abs_prediction_errors = [abs(prediction_error) for prediction_error in prediction_errors]
            index_range = index_range[:]

            for i in range(2):
                if isinstance(index_range[i], str):
                    if index_range[i] == 'max_err':
                        index_range[i] = max(abs_prediction_errors)
                    elif index_range[i] == 'avg_err':
                        index_range[i] = sum(abs_prediction_errors)/len(abs_prediction_errors)
                    elif index_range[i].endswith('_sigma'):
                        index_range[i] = float(index_range[i].replace('_sigma',''))*sigma
                    elif index_range[i].endswith('sig'):
                        index_range[i] = float(index_range[i].replace('sig',''))*sigma
                    else:
                        raise ValueError('Unknwon index start or end value "{}"'.format(index_range[i]))

            x_start = index_range[0]
            x_end = index_range[1]

            # Compute error distribution function values for index start/end
            y_start = error_distribution_function(x_start)
            y_end = error_distribution_function(x_end)

            for i, item in enumerate(series):
                model_window = self.model.window

                # Before the window
                if i <=  self.model.window:
                    continue

                # After the window (if using a reconstructor)
                if isinstance(self.model, Reconstructor):
                    if i >  len(series)-self.model.window-1:
                        break

                # Duplicate this sereis item
                item = deepcopy(item)

                # Compute the prediction error index
                actual, predicted = self._get_actual_and_predicted(series, i, data_label, model_window)
                prediction_error = abs(actual-predicted)

                # Compute the anomaly index in the given range (which defaults to 0, max_err)
                # Below the start it means anomaly (0), above always anomaly (1). In the middle it
                # follows the error distribution, and it is rescaled between 0 and 1.

                x = prediction_error
                y = error_distribution_function(x)

                if x <= x_start:
                    anomaly_index = 0
                elif x >= x_end:
                    anomaly_index = 1
                else:
                    if index_type=='lin':
                        try:
                            anomaly_index = 1 - ((y-y_end)/(y_start-y_end))
                        except ValueError as e:
                            if str(e) == 'math domain error':
                                raise ValueError('Got a math domain error. This is likely due to an error distribution function badly approximating the real error distribution. Try changing it, or using lower index boundaries.') from None
                            else:
                                raise
                    elif  index_type=='log':
                        try:
                            anomaly_index = (log10(y) - log10(y_start)) / (log10(y_end) - log10(y_start))
                        except ValueError as e:
                            if str(e) == 'math domain error':
                                raise ValueError('Got a math domain error. This is likely due to an error distribution function badly approximating the real error distribution. Try changing it, or using lower index boundaries.') from None
                            else:
                                raise
                    else:
                        if callable(index_type):
                            anomaly_index = index_type(x, y, x_start, x_end, y_start, y_end)
                        else:
                            raise ValueError('Unknown index type "{}"'.format(index_type))

                if threshold is not None:
                    if anomaly_index < threshold:
                        anomaly_index = 0
                    else:
                        anomaly_index = 1

                # Set the anomaly index
                item.data_indexes['anomaly'] = anomaly_index

                # Add details?
                if details:
                    if isinstance(details, list):
                        if 'pred' in details:
                            item.data['{}_pred'.format(data_label)] = predicted
                        if 'err' in details:
                            item.data['{}_err'.format(data_label)] = prediction_error
                        if 'dist' in details:
                            item.data['dist({}_err)'.format(data_label)] = error_distribution_function(prediction_error)
                    else:
                        item.data['{}_pred'.format(data_label)] = predicted
                        item.data['{}_err'.format(data_label)] = prediction_error
                        item.data['dist({}_err)'.format(data_label)] = error_distribution_function(prediction_error)

                result_series.append(item)

        return result_series


    @property
    def error_distribution_function(self):
        distribution_function = DistributionFunction(self.data['error_distribution'],
                                                     self.data['error_distribution_params'])
        return distribution_function

#===================================
#   Periodic Average Forecaster
#   Predictive Anomaly Detector
#===================================

class PeriodicAverageAnomalyDetector(ModelBasedAnomalyDetector):
    """An anomaly detection model based on a periodic average forecaster.

    Args:
        path (str): a path from which to load a saved model. Will override all other init settings.
    """

    model_class = PeriodicAverageForecaster



#===================================
#  Periodic Average Reconstructor
#   Predictive Anomaly Detector
#===================================

class PeriodicAverageReconstructorAnomalyDetector(ModelBasedAnomalyDetector):
    """An anomaly detection model based on a periodic average reconstructor.

    Args:
        path (str): a path from which to load a saved model. Will override all other init settings.
    """

    model_class = PeriodicAverageReconstructor


