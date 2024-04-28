# -*- coding: utf-8 -*-
"""Anomaly detection models."""

from copy import deepcopy
from ..utilities import _Gaussian, rescale
from .forecasters import Forecaster, PeriodicAverageForecaster, LSTMForecaster
from .reconstructors import Reconstructor, PeriodicAverageReconstructor
from .base import Model
from math import log10
from fitter import Fitter, get_common_distributions, get_distributions
from ..utilities import DistributionFunction
from statistics import stdev
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
    """A generic anomaly detection model."""

    def predict(self, series, *args, **kwargs):
        """Disabled. Anomaly detectors can be used only with the ``apply()`` method."""
        raise NotImplementedError('Anomaly detectors can be used only with the apply() method') from None

    def evaluate(self, series, *args, **kwargs):
        """Disabled. Anomaly detectors cannot be evaluated yet."""
        raise NotImplementedError('Anomaly detectors cannot be evaluated yet.') from None

    def cross_validate(self, series, *args, **kwargs):
        """Disabled. Anomaly detectors cannot be evaluated yet."""
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
        model_class(Forecaster,Reconstructor): the model to be used for anomaly detection, if not already set.
    """

    @property
    def model_class(self):
        try:
            return self._model_class
        except AttributeError:
            raise NotImplementedError('No model class set for this anomaly detector')

    def __init__(self, model_class=None, *args, **kwargs):

        # Handle the model_class
        try:
            self.model_class
        except NotImplementedError:
            if not model_class:
                raise ValueError('The model_class is not set in the anomaly detector nor given in the init') from None
            else:
                self._model_class = model_class
        else:
            if model_class:
                raise ValueError('The model_class was given in the init but it is already set in the anomaly detector')

        self.predictive_model_args = args
        self.predictive_model_kwargs = kwargs

        # Call parent init
        super(ModelBasedAnomalyDetector, self).__init__()
 
    @classmethod
    def load(cls, path):

        # Load the anomaly detection model
        model = super().load(path)

        # ..and load the inner model(s) as nested model
        if model.data['with_context']:
            model.models = {}
            for data_label in model.data['data_labels']:
                model.models[data_label] = model.model_class.load('{}/{}_model'.format(path, data_label))
        else:
            model.model= model.model_class.load('{}/model'.format(path))

        return model

    def save(self, path):

        # Save the anomaly detection model
        super(ModelBasedAnomalyDetector, self).save(path)

        # ..and save the inner model(s) as nested model
        if self.data['with_context']:
            for data_label in self.models:
                self.models[data_label].save('{}/{}_model'.format(path, data_label))
        else:
            self.model.save('{}/model'.format(path))


    def _get_actual_and_predicted(self, series, i, data_label, with_context):

            # Call model predict logic and compare with the actual data
            actual = series[i].data[data_label]
            if issubclass(self.model_class, Reconstructor):
                prediction = self.model.predict(series, from_i=i,to_i=i)
            elif issubclass(self.model_class, Forecaster):
                if with_context:
                    prediction = self.models[data_label].predict(series, steps=1, from_i=i, context_data=series[i].data)
                else:
                    # TODO: in case of forecasters without partial predictions, this is a performance hit for multivariate
                    # time series as the same predict is called in the exact same way for each data label.
                    prediction = self.model.predict(series, steps=1, from_i=i)
            else:
                raise TypeError('Don\'t know how to handle predictive model class "{}"'.format(self.model_class.__name__))

            # Handle list of dicts or dict of lists (of which we have only one value here)
            #{'value': [0.2019341593004146, 0.29462641146884005]}

            if isinstance(prediction, list):
                predicted = prediction[0][data_label]
            elif isinstance(prediction, dict):
                if isinstance(prediction[data_label], list):
                    predicted = prediction[data_label][0]
                else:
                    predicted = prediction[data_label]
            else:
                raise TypeError('Don\'t know how to handle a prediction with of type "{}"'.format(prediction.__class__.__name__))
            #logger.debug('{:f}\tvs\t{}:\tdiff={}'.format(actual, predicted, actual-predicted))
            return (actual, predicted)


    @AnomalyDetector.fit_function
    def fit(self, series, with_context=False, error_distribution='auto', verbose=False, summary=False, **kwargs):

        # Handle the error distribution(s)
        if error_distribution == 'auto':
            error_distributions = kwargs.pop('error_distributions', fitter_library.fitter.get_common_distributions() +['gennorm'])
        else:
            error_distributions = [error_distribution]

        # Fit the predictive model(s)
        if with_context:
            if not len(series.data_labels) > 1:
                raise ValueError('Anomaly detection with partial predictions on univariate series does not make sense')

            # Fit separate models, one for each data label
            self.models = {}
            self.data['model_ids'] = {}
            for data_label in series.data_labels:
                if verbose:
                    print('Fitting for "{}":'.format(data_label))
                logger.debug('Fitting for "%s"...', data_label)

                # Initialize the internal model for this label
                self.models[data_label] = self.model_class(*self.predictive_model_args, **self.predictive_model_kwargs)

                # Set the id of the internal model in the data
                self.data['model_ids'][data_label] = self.models[data_label].data['id']

                # Fit it
                self.models[data_label].fit(series, **kwargs, target=data_label, with_context=True, verbose=verbose)

                # Set the model window if not already done:
                if 'model_window' not in self.data:
                    self.data['model_window'] = self.models[data_label].window

        else:
            # Initialize the internal model
            self.model = self.model_class(*self.predictive_model_args, **self.predictive_model_kwargs)

            # Set the id of the internal model in the data
            self.data['model_id'] = self.model.data['id']

            # Fit it
            self.model.fit(series, **kwargs, verbose=verbose)

            # Set the model window
            self.data['model_window'] = self.model.window

        if verbose:
            print('Predictive model(s) fitted, now evaluating')
        logger.info('Predictive model(s) fitted, now evaluating...')

        # Store if to use context or not
        self.data['with_context'] = with_context

        # Initialize internal dictionaries
        self.data['prediction_errors'] = {}
        self.data['stdevs'] = {} 
        self.data['error_distributions'] = {}
        self.data['error_distributions_params'] = {}
        self.data['error_distributions_stats'] = {}

        # Evaluate the predictive for one step ahead and get the forecasting errors
        prediction_errors = {}
        progress_step = len(series)/10

        for data_label in series.data_labels:

            prediction_errors[data_label] = []

            if verbose:
                print('Computing actual vs predicted for "{}": '.format(data_label), end='')
            logger.info('Computing actual vs predicted for "{}"...'.format(data_label))

            for i, _ in enumerate(series):
                if verbose:
                    if int(i%progress_step) == 0:
                        print('.', end='')

                # Before the window
                if i <=  self.data['model_window']:
                    continue

                # After the window (if using a reconstructor)
                if issubclass(self.model_class, Reconstructor):
                    if i > len(series)-self.data['model_window']-1:
                        break

                # Predict & append the error
                #logger.debug('Predicting and computing the difference (i=%s)', i)
                actual, predicted = self._get_actual_and_predicted(series, i, data_label, with_context)
                prediction_errors[data_label].append(actual-predicted)

            # Store the forecasting errors internally in the model
            self.data['prediction_errors'] = prediction_errors

            if verbose:
                print('')

        if verbose:
            print('Model(s) evaluated, now computing the error distribution(s)')
        logger.info('Model(s) evaluated, now computing the error distribution(s)...')

        for data_label in series.data_labels:
            #if verbose:
            #    print('Selecting error distribution for "{}"'.format(data_label))
            #logger.debug('Selecting error distribution for "%s"', data_label))

            # Fit the distributions and select the best one
            fitter = fitter_library.fitter.Fitter(prediction_errors[data_label], distributions=error_distributions)
            fitter.fit(progress=False)

            if summary:
                # Warning: the summary() function will also generate a plot
                print(fitter.summary())

            best_error_distribution = list(fitter.get_best().keys())[0]
            best_error_distribution_stats = fitter.summary(plot=False).transpose().to_dict()[best_error_distribution]
            error_distribution_params = fitter.get_best()[best_error_distribution]

            if best_error_distribution_stats['ks_pvalue'] < 0.05:

                logger.warning('The error distribution for "{}" ({}) ks p-value is low ({}). '.format(data_label, best_error_distribution, best_error_distribution_stats['ks_pvalue']) +
                               'Expect issues. In case of math domain errors, try using lower index boundaries.')

            if not (-0.01 <= error_distribution_params['loc'] <= 0.01):
                logger.warning('The error distribution for "{}" is not centered in (almost) zero, but in {}. Expect issues.'.format(data_label, error_distribution_params['loc']))

            self.data['error_distributions'][data_label] = best_error_distribution
            self.data['error_distributions_params'][data_label] = error_distribution_params
            self.data['error_distributions_stats'][data_label] = best_error_distribution_stats

            self.data['stdevs'][data_label] = stdev(prediction_errors[data_label])
        logger.info('Anomaly detector fitted')

    def inspect(self, plot=True):
        '''Inspect the model and plot the error distribution'''

        for data_label in self.data['error_distributions']:

            abs_prediction_errors = [abs(prediction_error) for prediction_error in self.data['prediction_errors'][data_label]]

            print('\nDetails for: "{}"'.format(data_label))

            print('Predictive model avg error (abs): {}'.format(sum(abs_prediction_errors)/len(abs_prediction_errors)))
            print('Predictive model min error (abs): {}'.format(min(abs_prediction_errors)))
            print('Predictive model max error (abs): {}'.format(max(abs_prediction_errors)))

            print('Error distribution: {}'.format(self.data['error_distributions'][data_label]))
            print('Error distribution params: {}'.format(self.data['error_distributions_params'][data_label]))
            print('Error distribution stats: {}'.format(self.data['error_distributions_stats'][data_label]))

            if plot:

                x_min = min(self.data['prediction_errors'][data_label])
                x_max = max(self.data['prediction_errors'][data_label])

                # Instantiate the errro distribution function
                distribution_function = DistributionFunction(self.data['error_distributions'][data_label],
                                                             self.data['error_distributions_params'][data_label])

                # Get the error distribution function plot
                plt = distribution_function.plot(show=False, x_min=x_min, x_max=x_max)

                # Add the histogram to the plot
                plt.hist(self.data['prediction_errors'][data_label], bins=100, density=True, alpha=1, color='steelblue')

                # Override title
                #plt.title('Error distribution: {}'.format(self.data['error_distribution']))

                # Show the plot
                plt.show()

    @Model.apply_function
    def apply(self, series, index_range=['avg_err','max_err'], index_type='log', threshold=None, multivariate_index_strategy='max', details=False, verbose=False):

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

            multivariate_index_strategy(str, callable): the strategy to use when computing the overall anomaly index for multivariate
                                                        time series items. Possible choices are "max" to use the maximum one, "avg"
                                                        for the mean and "min" for the minimum; or a callable taking as input the
                                                        list of the anomaly indexes for each data label. Defaults to "max".


            details(bool, list): if to add details to the time series as the predicted value, the error and the
                                 corresponding error distribution function (dist) value. If set to True, it adds
                                 all of them, if instead using a list only selected details can be added: "pred"
                                 for the predicted values, "err" for the error, and "dist" for the error distribution.
        """


        # Initialize the result time series
        result_series = series.__class__()

        # Initialize the error distribution function
        error_distribution_functions = {}
        for data_label in self.data['error_distributions']:
            error_distribution_functions[data_label] = DistributionFunction(self.data['error_distributions'][data_label],
                                                                            self.data['error_distributions_params'][data_label])

        # Set anomaly index boundaries
        x_starts = {}
        x_ends = {}
        y_starts = {}
        y_ends = {}
        for data_label in self.data['error_distributions']:

            abs_prediction_errors = [abs(prediction_error) for prediction_error in self.data['prediction_errors'][data_label]]
            index_range = index_range[:]

            for i in range(2):
                if isinstance(index_range[i], str):
                    if index_range[i] == 'max_err':
                        index_range[i] = max(abs_prediction_errors)
                    elif index_range[i] == 'avg_err':
                        index_range[i] = sum(abs_prediction_errors)/len(abs_prediction_errors)
                    elif index_range[i].endswith('_sigma'):
                        index_range[i] = float(index_range[i].replace('_sigma',''))*self.data['stdevs'][data_label]
                    elif index_range[i].endswith('sig'):
                        index_range[i] = float(index_range[i].replace('sig',''))*self.data['stdevs'][data_label]
                    else:
                        raise ValueError('Unknwon index start or end value "{}"'.format(index_range[i]))

            x_starts[data_label] = index_range[0]
            x_ends[data_label] = index_range[1]

            # Compute error distribution function values for index start/end
            y_starts[data_label] = error_distribution_functions[data_label](x_starts[data_label])
            y_ends[data_label] = error_distribution_functions[data_label](x_ends[data_label])

        # Start processing
        progress_step = len(series)/10
        if verbose:
            print('Applying the anomaly detector: ', end='')
        for i, item in enumerate(series):

            if verbose:
                if int(i%progress_step) == 0:
                    print('.', end='')

            # Before the window
            if i <=  self.data['model_window']:
                continue

            # After the window (if using a reconstructor)
            if issubclass(self.model_class, Reconstructor):
                if i >  len(series)-self.data['model_window']-1:
                    break

            item_anomaly_indexes = [] 

            for data_label in series.data_labels:

                # Shortcuts
                error_distribution_function = error_distribution_functions[data_label]
                x_start = x_starts[data_label]
                x_end = x_ends[data_label]
                y_start = y_starts[data_label]
                y_end = y_ends[data_label]

                # Duplicate this series item
                item = deepcopy(item)

                # Compute the prediction error index
                actual, predicted = self._get_actual_and_predicted(series, i, data_label, self.data['with_context'])
                prediction_error = abs(actual-predicted)
                #print(actual, predicted)

                # Compute the anomaly index in the given range (which defaults to 0, max_err)
                # Below the start it means anomaly (0), above always anomaly (1). In the middle it
                # follows the error distribution, and it is rescaled between 0 and 1.

                x = prediction_error
                y = error_distribution_function(x)
                #print('---------')
                #print(x,x_start,x_end)
                #print(y,y_start,y_end)
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
                    elif index_type=='log':
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

                # Add this anomaly index for this data label to the list of the item anomaly indexes
                item_anomaly_indexes.append(anomaly_index)

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

            # Set anomaly index & append
            if len(item_anomaly_indexes) == 1:
                item.data_indexes['anomaly'] = item_anomaly_indexes[0]
            else:
                if multivariate_index_strategy == 'min':
                    item.data_indexes['anomaly'] = min(item_anomaly_indexes)
                elif multivariate_index_strategy == 'max':
                    item.data_indexes['anomaly'] = max(item_anomaly_indexes)
                elif multivariate_index_strategy == 'avg':
                    item.data_indexes['anomaly'] = sum(item_anomaly_indexes)/len(item_anomaly_indexes)
                else:
                    item.data_indexes['anomaly'] = multivariate_index_strategy(item_anomaly_indexes)

            # Append
            result_series.append(item)

        if verbose:
            print('')

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
    """An anomaly detection model based on a periodic average forecaster."""

    model_class = PeriodicAverageForecaster



#===================================
#  Periodic Average Reconstructor
#   Predictive Anomaly Detector
#===================================

class PeriodicAverageReconstructorAnomalyDetector(ModelBasedAnomalyDetector):
    """An anomaly detection model based on a periodic average reconstructor."""

    model_class = PeriodicAverageReconstructor


#===================================
#  LSTM Anomaly Detector
#===================================

class LSTMAnomalyDetector(ModelBasedAnomalyDetector):
    """An anomaly detection model based on a LSTM neural network."""

    model_class = LSTMForecaster

