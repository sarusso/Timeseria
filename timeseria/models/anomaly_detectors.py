# -*- coding: utf-8 -*-
"""Anomaly detection models."""

from copy import deepcopy
from .forecasters import Forecaster, PeriodicAverageForecaster, LSTMForecaster, LinearRegressionForecaster
from .reconstructors import Reconstructor, PeriodicAverageReconstructor
from .base import Model
from math import log10
from fitter import Fitter, get_common_distributions, get_distributions
from ..utils import DistributionFunction
from statistics import stdev
import fitter as fitter_library
from ..datastructures import TimeSeries, DataTimePoint
from ..exceptions import ConsistencyException

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
        """Disabled. Anomaly detectors cannot be evaluated."""
        raise NotImplementedError('Anomaly detectors cannot be evaluated.') from None

    def cross_validate(self, series, *args, **kwargs):
        """Disabled. Anomaly detectors cannot be evaluated."""
        raise NotImplementedError('Anomaly detectors cannot be evaluated.') from None

    @staticmethod
    def mark_events(timeseries, index_treshold=1.0, min_persistence=2, max_gap=2, replace_index=False, inplace=False):
        """Mark ensembles of anomalous data points as single anomalous events, with some tolerance.

        Args:
            index_treshold(float): the anomaly index above which to consider a data point anomalous.
            min_persistence(int): the minimum persistence of an event, in terms of consecutive data points.
            max_gap(int): the maximum gap within a signle event of data points below the index_treshold.
            replace_index(bool): if to replace the existent ``anomaly`` index instead of adding a new ``anomaly_event`` one.

        Returns:
            TimeSeries: the time series with the events marked.
        """

        if inplace:
            event_timeseries = timeseries
        else:
            event_timeseries = timeseries.duplicate()
        event_start = None

        for i, item in enumerate(event_timeseries):

            # Set no event by default
            item.data_indexes['anomaly_event'] = 0

            # Detect new (potential) event start
            if item.data_indexes['anomaly'] >= index_treshold:
                if event_start is None:
                    logger.debug('Starting to suspect event @ element #%s',i)
                    event_start = i

            # Detect event end
            if event_start is not None:
                if item.data_indexes['anomaly'] < index_treshold:
                    logger.debug('Evaluating element #%s to trigger an event end', i)

                    # Are we still within the max gap? If so, don't close the (potential) event yet
                    from_i = (i - max_gap) if (i - max_gap) >= 0 else 0
                    to_i = i
                    event_ended = True
                    logger.debug('  checking elements from #%s to #%s', from_i, to_i)
                    for j in range(from_i, to_i+1):
                        if event_timeseries[j].data_indexes['anomaly'] >= index_treshold:
                            event_ended = False
                    if event_ended:
                        logger.debug('  event was ended @ element #%s', i - max_gap - 1)
                        event_persistence =  (i - max_gap) - event_start
                        if event_persistence < min_persistence:
                            logger.debug('  discarding as below minimum persistence')
                        else:
                            # Mark it
                            for j in range(event_start, i - max_gap):
                                event_timeseries[j].data_indexes['anomaly_event'] = 1

                        # Reset the event_start support var
                        event_start = None
                else:
                    logger.debug('Element #%s is part of the event', i)

        # Handle series last event, if any
        if event_start is not None:

            # When was the last event over?
            event_end = len(event_timeseries)
            for i in range(len(event_timeseries)-1, event_start, -1):
                if  event_timeseries[i].data_indexes['anomaly'] >= index_treshold:
                    event_end = i
                    break
            logger.debug('  event was ended @ element #%s', event_end)
            event_persistence =  event_end - event_start
            if event_persistence < min_persistence:
                logger.debug('  discarding as below minimum persistence')
            else:
                # Mark it
                for j in range(event_start, event_end+1):
                    event_timeseries[j].data_indexes['anomaly_event'] = 1

        if replace_index:
            for item in event_timeseries:
                item.data_indexes['anomaly'] = item.data_indexes.pop('anomaly_event') 

        return  event_timeseries


#===================================
#  Model-based Anomaly Detector
#===================================

class ModelBasedAnomalyDetector(AnomalyDetector):
    """An anomaly detection model based on another model. See the ``model_class`` property for which model is being used.

    In short, for each element of the series where the anomaly detector is applied, the model is asked to make a prediction.
    The predicted and actual values are then compared, and accordingly to the model error distribution, an anomaly index
    is computed. The anomaly index is proportional to the likelihood of each element to be anomalous, and ranges form 0 to 1.

    More in detail, in the fit phase the model set in the ``model_class`` is first fitted and then evaluated using the error
    metric set in the ``error_metric`` argument, which defaults to PE (Percentage Error). All of the error values recorded in
    the evaluation are then used to build the error distribution (one for each data label of the series), which defaults to a
    generalized normal distribution. If the p-value is low (below 0.05), a warning is issued.

    When the anomaly detector is then applied, the anomaly index is computed based on two boundaries of such error distribution,
    which are set in the ``index_bounds`` argument of the ``apply()`` method. Error values below the lower bound will map to an
    anomaly index of zero (no anomaly suspected at all), between the lower and the upper bound will map to anomaly indexes between
    zero and one (some degree of anomaly is suspected) and above the upper bound will map to an anomaly index of one (certainly
    an anomaly).

    How to set such boundaries depends on many factors, the most important of which being whether the anomaly detector is used in
    usupervised or semi-supervised mode. In unsupervised mode, a reasonable choice would be to set the lower bound to the average
    error, and the upper bound to the maximum error. In semi-supervised mode (which is way more powerful since a notion of normality
    is implicitly given in the fit data), a good choice would be to set the lower bound to the maximum error (which still belongs to
    the "normal" behavior) and the upper bound to a very high error (e.g. a billion times the probability of an observation to be
    adherent with the model). More details about how to set the boundaries are provided in the ``apply()`` method documentation.

    Args:
        model(Forecaster, Reconstructor): the model instance to be used for the anomaly detection.
        models(dict): the model instances by data label, if using different ones, to be used for the anomaly detection.
        model_class(Forecaster type, Reconstructor type): the model class to be used for anomaly detection, if not already set.
        index_window(int): the (rolling) window length to be used when computing the anomaly index. Defaults to 1.
    """

    @property
    def model_class(self):
        try:
            return self._model_class
        except AttributeError:
            raise NotImplementedError('No model class set for this anomaly detector')

    def __init__(self, model=None, models=None, model_class=None, index_window=1, *args, **kwargs):

        # Check model-related arguments
        try:
            self.model_class
        except NotImplementedError:
            # Model class is not set in the anomaly detector
            if sum(arg is not None for arg in [model, models, model_class]) > 1:
                raise ValueError('Please set only one between the model, models, and model_class arguments') from None
            if model_class:
                self._model_class = model_class
            elif model:
                self._model_class = model.__class__
            elif models:
                for i, data_label in enumerate(models):
                    if i == 0:
                        self._model_class = models[data_label].__class__
                    else:
                        if not isinstance(models[data_label], self.model_class):
                            raise TypeError('Inconsistent model classes: "{}" vs "{}". Please provide models all of the same class.'.format(self.model_class.__name__, models[data_label].__class__.__name__))
            else:
                raise ValueError('Please provide one of model_class, model or models arguments') from None
        else:
            # Model class is already set in the anomaly detector
            if model_class:
                raise ValueError('This model has a model class already set, redefining it via the model_class argument is not supported')
            if sum(arg is not None for arg in [model, models]) >1 :
                raise ValueError('Please set only one between the model and models arguments')
            if model:
                if not isinstance(model, model_class):
                    raise TypeError('This anomaly detector is designed to work with models of class "{}" but a model of class "{}" was provided'.format(self.model_class, model.__class__.__name__))
            if models:
                for data_label in models:
                    if not isinstance(models[data_label], model_class):
                        raise TypeError('This anomaly detector is designed to work with models of class "{}" but a model of class "{}" was provided'.format(self.model_class, model.__class__.__name__))
 
        # Set models in both cases
        self.model = model
        self.models = models

        # Set other arguments
        self.predictive_model_args = args
        self.predictive_model_kwargs = kwargs
        self.index_window = index_window

        # Call parent init
        super(ModelBasedAnomalyDetector, self).__init__()

    @classmethod
    def load(cls, path):

        # Load the anomaly detection model
        anomaly_detector = super().load(path)

        # ..and load the inner model(s) as nested model
        if anomaly_detector.data['with_context']:
            anomaly_detector.models = {}
            for data_label in anomaly_detector.data['data_labels']:
                anomaly_detector.models[data_label] = anomaly_detector.model_class.load('{}/{}_model'.format(path, data_label))
        else:
            anomaly_detector.model = anomaly_detector.model_class.load('{}/model'.format(path))

        return anomaly_detector

    def save(self, path):

        try:
            self._model_class
        except:
            pass
        else:
            raise NotImplementedError('Saving generic model-based anomaly detectors is not supported. Please create a custom class setting the model_class attribute.')

        # Save the anomaly detection model
        super(ModelBasedAnomalyDetector, self).save(path)

        # ..and save the inner model(s) as nested model
        if self.data['with_context']:
            for data_label in self.models:
                self.models[data_label].save('{}/{}_model'.format(path, data_label))
        else:
            self.model.save('{}/model'.format(path))

    def _get_actual_value(self, series, i, data_label):
        actual = series[i].data[data_label]
        return actual

    def _get_predicted_value(self, series, i, data_label, with_context):

            # Call model predict logic and compare with the actual data
            if issubclass(self.model_class, Reconstructor):

                # Reconstructors
                prediction = self.model.predict(series, from_i=i, to_i=i)
                if isinstance(prediction, list):
                    predicted = prediction[0][data_label]
                elif isinstance(prediction, dict):
                    if isinstance(prediction[data_label], list):
                        predicted = prediction[data_label][0]
                    else:
                        predicted = prediction[data_label]
                else:
                    raise TypeError('Don\'t know how to handle a prediction with of type "{}"'.format(prediction.__class__.__name__))

            elif issubclass(self.model_class, Forecaster):

                # Forecasters
                try:
                    # Try performing a bulk-optimized predict call
                    if with_context:
                        predicted = self.models[data_label]._get_predicted_value_bulk(series, i, data_label, with_context=True)
                    else:
                        if self.models:
                            predicted = self.models[data_label]._get_predicted_value_bulk(series, i, data_label, with_context=True)
                        else:
                            predicted = self.model._get_predicted_value_bulk(series, i, data_label, with_context=True)
                except (AttributeError, NotImplementedError):
                    # Perform a standard predict call
                    if with_context:
                        predicted = self.models[data_label]._get_predicted_value(series, i, data_label, with_context=True)
                    else:
                        if self.models:
                            predicted = self.models[data_label]._get_predicted_value(series, i, data_label, with_context=with_context)
                        else:
                            predicted = self.model._get_predicted_value(series, i, data_label, with_context=with_context)
            else:
                raise TypeError('Don\'t know how to handle predictive model class "{}"'.format(self.model_class.__name__))

            # TODO: unify the above (e.g. add a _get_predicted_value to the reconstructors)
            return predicted

    @AnomalyDetector.fit_method
    def fit(self, series, with_context=False, error_metric='PE', error_distribution='gennorm', store_errors=True, verbose=False, summary=False, **kwargs):
        """Fit the anomaly detection model on a series.

        Args:
            series(TimeSeries): the series on which to fit the model.
            with_context(bool): if to use context for multivariate time series or not. Defaults to ``False``.
            error_metric(str): the error metric to be used for evaluating the model and to build the error distribution for the anomaly index.
                               Supported values are: ``E``, ``AE``, ``PE`` and ``APE``. Defaults to ``PE``.
            error_distribution(str): if to use a specific error distribution or find it automatically (``error_distribution='auto'``).
                                     Defaults to ``gennorm``, a generalized normal distribution.
            store_errors(float): if to store the prediction errors (together with actual and predicted values) internally for further analysis. Access
                                 them with ``model.data['prediction_errors']``, ``model.data['actual_values']`` and ``model.data['predicted_values']``.
            verbose(bool): if to print the fit progress (one dot = 10% done).
            summary(bool): if to display a summary on the error distribution fitting or selection.
        """

        # Handle the error distribution(s)
        if error_distribution == 'auto':
            error_distributions = kwargs.pop('error_distributions', fitter_library.fitter.get_common_distributions() +['gennorm'])
        else:
            error_distributions = [error_distribution]

        # Handle predictive models
        if with_context:

            # With context, use separate models, one for each data label
            if not len(series.data_labels()) > 1:
                raise ValueError('Anomaly detection with context on univariate series does not make sense')
            if self.model:
                raise ValueError('Cannot fit with context with a single model, please provide a model for each data label')

            if not self.models:
                self.models = {}
            self.data['model_ids'] = {}

            for data_label in series.data_labels():
                if self.models and data_label in self.models:
                    if verbose:
                        print('Predictive model for {} already fitted, not re-fitting.'.format(data_label))
                    logger.debug('Predictive model for %s already fitted, not re-fitting.', data_label)
                else:
                    if verbose:
                        print('Fitting for "{}":'.format(data_label))
                    logger.debug('Fitting for "%s"...', data_label)
                    self.models[data_label] = self.model_class(*self.predictive_model_args, **self.predictive_model_kwargs)
                    self.models[data_label].fit(series, **kwargs, target=data_label, with_context=True, verbose=verbose)

        else:

            # Without context, use a single model, unless otherwise set with the "models" argument
            if self.model or self.models:
                if verbose:
                    print('Predictive model already fitted, not re-fitting.')
                logger.debug('Predictive model for already fitted, not re-fitting.')
            else:
                if verbose:
                    print('Fitting model'.format(data_label))
                logger.debug('Fitting model...')
                self.model = self.model_class(*self.predictive_model_args, **self.predictive_model_kwargs)
                self.model.fit(series, **kwargs, verbose=verbose)

        # Set the id of the internal model and the window in the data
        if self.model:
            self.data['model_id'] = self.model.data['id']
            self.data['model_window'] = self.model.window
        elif self.models:
            for data_label in self.models:
                self.data['model_ids'][data_label] = self.models[data_label].data['id']
                if 'model_window' not in self.data:
                    self.data['model_window'] = self.models[data_label].window
        else:
            raise ConsistencyException('No model nor models set?')

        if verbose:
            print('Predictive model(s) fitted, now evaluating')
        logger.info('Predictive model(s) fitted, now evaluating...')

        # Store if to use context or not
        self.data['with_context'] = with_context

        # Check and store error metric
        if error_metric not in ['E', 'AE', 'PE', 'APE', 'SLE']:
            raise ValueError('Unknown error metric "{}"'.format(error_metric))
        self.data['error_metric'] = error_metric

        # Initialize internal dictionaries
        if store_errors:
            self.data['actual_values'] = {}
            self.data['predicted_values'] = {}
            self.data['prediction_errors'] = {}
        self.data['stdevs'] = {} 
        self.data['error_distributions'] = {}
        self.data['error_distributions_params'] = {}
        self.data['error_distributions_stats'] = {}

        # Evaluate the predictive for one step ahead and get the forecasting errors
        actual_values = {}
        predicted_values = {}
        prediction_errors = {}
        cumulative_prediction_errors = {}

        progress_step = len(series)/10

        for data_label in series.data_labels():

            actual_values[data_label] = []
            predicted_values[data_label] = []
            prediction_errors[data_label] = []
            cumulative_prediction_errors[data_label] = []

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
                actual = self._get_actual_value(series, i, data_label)
                predicted = self._get_predicted_value(series, i, data_label, with_context)

                if store_errors:
                    actual_values[data_label].append(actual)
                    predicted_values[data_label].append(predicted)

                if error_metric == 'E':
                    prediction_error = actual-predicted
                elif error_metric == 'AE':
                    prediction_error = abs(actual-predicted)
                elif error_metric == 'PE':
                    prediction_error = (actual-predicted)/actual
                elif error_metric == 'APE':
                    prediction_error = abs((actual-predicted)/actual)
                else:
                    raise ValueError('Unknown error metric "{}"'.format(self.data['error_metric']))

                prediction_errors[data_label].append(prediction_error)
                if error_metric in ['AE', 'APE']:
                    prediction_errors[data_label].append(-prediction_error)

                # Handle the index window if any
                if (self.index_window > 1) and (i >= self.data['model_window'] + self.index_window):
                    if error_metric in ['AE', 'APE']:
                        raise NotImplementedError('Error metrics bases on absolute values are not yet supported when using a window for the anomaly index')
                        cumulative_error = 0
                        for j, item in enumerate(prediction_errors[data_label][-self.index_window*2:]):
                            if j % 2 == 0:
                                cumulative_error += item
                        cumulative_prediction_errors[data_label].append(cumulative_error) # Likely a bi-modal distribution
                        #cumulative_prediction_errors[data_label].append(-cumulative_error) # Likely a gamma distribution
                    else:
                        cumulative_error = sum(prediction_errors[data_label][-self.index_window:])
                        cumulative_prediction_errors[data_label].append(cumulative_error)

            if verbose:
                print('')

        if self.index_window > 1:
            prediction_errors = cumulative_prediction_errors

        if verbose:
            print('Model(s) evaluated, now computing the error distribution(s)')
        logger.info('Model(s) evaluated, now computing the error distribution(s)...')

        for data_label in series.data_labels():

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
                logger.warning('The error distribution for "{}" ({}) p-value is low: {}.'.format(data_label, best_error_distribution, best_error_distribution_stats['ks_pvalue']))

            self.data['error_distributions'][data_label] = best_error_distribution
            self.data['error_distributions_params'][data_label] = error_distribution_params
            self.data['error_distributions_stats'][data_label] = best_error_distribution_stats

            self.data['stdevs'][data_label] = stdev(prediction_errors[data_label])

        if store_errors:
            self.data['actual_values'] = actual_values
            self.data['predicted_values'] = predicted_values
            self.data['prediction_errors'] = prediction_errors

        self.data['index_window'] = self.index_window

        logger.info('Anomaly detector fitted')

    def inspect(self, plot=True, plot_x_min='auto', plot_x_max='auto', series=False):
        '''Inspect the model by printing a summary and plotting the error distribution and/or the fit series.

            Args:
                plot(bool): if to plot the error distribution. Defaults to True.
                plot_x_min(float or str): the minimum value of the x axis. Defaults to 'auto'.
                plot_x_max(float or str): the maximum value of the x axis. Defaults to 'auto'.
                series(bool): if to plot the fit series. 
        '''

        for data_label in self.data['error_distributions']:

            abs_prediction_errors = [abs(prediction_error) for prediction_error in self.data['prediction_errors'][data_label]]

            print('\nDetails for: "{}"'.format(data_label))

            if self.data['index_window']  >1:
                print('Predictive model avg error (abs, {} items): {}'.format(self.data['index_window'], sum(abs_prediction_errors)/len(abs_prediction_errors)))
                print('Predictive model min error (abs, {} items): {}'.format(self.data['index_window'], min(abs_prediction_errors), self.data['index_window']))
                print('Predictive model max error (abs, {} items): {}'.format(self.data['index_window'], max(abs_prediction_errors), self.data['index_window']))
            else:
                print('Predictive model avg error (abs): {}'.format(sum(abs_prediction_errors)/len(abs_prediction_errors)))
                print('Predictive model min error (abs): {}'.format(min(abs_prediction_errors)))
                print('Predictive model max error (abs): {}'.format(max(abs_prediction_errors)))

            print('Error distribution: {}'.format(self.data['error_distributions'][data_label]))
            print('Error distribution params: {}'.format(self.data['error_distributions_params'][data_label]))
            print('Error distribution stats: {}'.format(self.data['error_distributions_stats'][data_label]))

            if plot:

                if plot_x_min == 'auto':
                    this_plot_x_min = min(self.data['prediction_errors'][data_label])
                else:
                    this_plot_x_min = plot_x_min
                if plot_x_max == 'auto':
                    this_plot_x_max = max(self.data['prediction_errors'][data_label])
                else:
                    this_plot_x_max = plot_x_max

                # Instantiate the error distribution function
                distribution_function = DistributionFunction(self.data['error_distributions'][data_label],
                                                             self.data['error_distributions_params'][data_label])

                # Get the error distribution function plot
                plt = distribution_function.plot(show=False, x_min=this_plot_x_min, x_max=this_plot_x_max)

                # Add the histogram to the plot
                plt.hist(self.data['prediction_errors'][data_label], bins=100, density=True, alpha=1, color='steelblue')

                # Override title
                #plt.title('Error distribution: {}'.format(self.data['error_distribution']))

                # Show the plot
                plt.show()

            if series:
                if not 'prediction_errors' in self.data:
                    raise ValueError('Cannot inspect fit series if store_errors was not set.')
                fit_series = TimeSeries()
                first_data_label = list(self.data['actual_values'].keys())[0]
                for i in range(len(self.data['actual_values'][first_data_label])):
                    data = {}
                    for data_label in self.data['actual_values']:
                        data[data_label] = self.data['actual_values'][data_label][i]
                        data[data_label+'_pred'] = self.data['predicted_values'][data_label][i]
                        data[data_label+'_err'] = self.data['prediction_errors'][data_label][i]
                    fit_series.append(DataTimePoint(t=i, data=data))
                fit_series.plot()

    def preprocess(self, series, inplace=False, verbose=False):
        """Pre-process a time series for this anomaly detector so that multiple apply() calls are much faster.

            Args:
                series(TimeSeries): the series to pre-process.
                inplace(bool): if to pre-process in-place.
                verbose(bool): if to print the progress (one dot = 10% done).

            Returns:
                TimeSeries: the pre-processed series.
        """
        if not inplace:
            series = series.duplicate()

        # Start pre-processing
        series.predictions = {}
        progress_step = len(series)/10
        if verbose:
            print('Pre-computing model predictions: ', end='')

        for i in range(series):

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

            ith_predictions = {} 
            for data_label in series.data_labels():
                ith_predictions[data_label] = self._get_predicted_value(series, i, data_label, self.data['with_context'])

            series.predictions[i] = ith_predictions

        series.preprocessed_by = self.id
        return series

    @Model.apply_method
    def apply(self, series, index_bounds=['avg_err','max_err'], index_type='log', multivariate_index_strategy='max',
              data_loss_threshold=1.0, details=False, verbose=False):

        """Apply the anomaly detection model on a series.

        Args:
            series(TimeSeries): the series on which to apply the anomaly detector.
            index_bounds(tuple): the lower an upper bounds, in terms of the prediction error, from which the anomaly index is
                                 computed. Below the lower limit no anomaly will be assumed (``anomlay_index=0``), while errors
                                 above the upper will always be considered as anomalous (``anomlay_index=1``). In the middle, the
                                 anomaly index will range between 0 and 1 following the model error distribution. Defaults values
                                 are ``('avg_err', 'max_err')``, which are reasonable values for unsupervised anomaly detection.
                                 Other supported values are: ``x_sigma``, where x is a standard deviation multiplier;
                                 ``adherence/x``, where x is a divider for the model adherence probability; and any other
                                 numerical value in terms of prediction error. Can be also set as a tuple of dictionaries, to
                                 set the bounds on a per-data label basis.
            index_type(str, callable): if to use a logarithmic anomaly index ("log", the default value) which compresses
                                       the index range so that bigger anomalies stand out more than smaller ones, or if to
                                       use a linear one ("lin"). Can also support a custom anomaly index as a callable,
                                       in which case the form must be ``f(x, y, x_start, x_end, y_start, y_end)`` where x
                                       is the model error, y its value on the distribution curve, and x_start/x_end together
                                       with y_start/y_end the respective x and y range start and end values, based on the
                                       range set by the ``index_bounds`` argument.
            multivariate_index_strategy(str, callable): the strategy to use when computing the overall anomaly index for multivariate
                                                        time series items. Possible choices are "max" to use the maximum one, "avg"
                                                        for the mean and "min" for the minimum; or a callable taking as input the
                                                        list of the anomaly indexes for each data label. Defaults to "max".
            data_loss_treshold(float): if the data loss is equal or greater than this threshold value, then the anomaly detection is not applied.
            details(bool, list): if to add details to the time series as the predicted value, the error and the
                                 model adherence probability. If set to ``True``, it adds all of them, if instead using
                                 a list only selected details can be added: ``pred`` for the predicted values, ``err`` for
                                 the errors, and ``adh`` for the model adherence probability.
            verbose(bool): if to print the apply progress (one dot = 10% done).


        Returns:
            TimeSeries: the series with the anomaly detection results.
        """
        # Initialize the result time series
        result_series = series.__class__()

        # Initialize the error distribution function
        error_distribution_functions = {}
        for data_label in self.data['error_distributions']:
            error_distribution_functions[data_label] = DistributionFunction(self.data['error_distributions'][data_label],
                                                                            self.data['error_distributions_params'][data_label])

        # Check error metric
        if self.data['error_metric'] in ['E', 'AE', 'PE', 'APE']:
            pass
        elif callable(self.data['error_metric']):
            # If the model is loaded, it will never get here
            pass
        else:
            raise ValueError('Unknown error metric "{}"'.format(self.data['error']))

        # Set anomaly index boundaries
        x_starts = {}
        x_ends = {}
        y_starts = {}
        y_ends = {}
        log_10_y_starts = {}
        log_10_y_ends = {}
        y_maxes = {}
        rolling_prediction_errors = {}

        for data_label in series.data_labels():
            abs_prediction_errors = [abs(prediction_error) for prediction_error in self.data['prediction_errors'][data_label]]
            y_maxes[data_label] = error_distribution_functions[data_label](self.data['error_distributions_params'][data_label]['loc'])
            this_index_bounds = index_bounds[:]

            if isinstance(this_index_bounds[0], dict):
                if not isinstance(this_index_bounds[1], dict):
                    raise ValueError('Cannot mix label-specific boudaries and not')
                this_index_bounds[0] = this_index_bounds[0][data_label]
                this_index_bounds[1] = this_index_bounds[1][data_label]
            else:
                for i in range(2):
                    if isinstance(this_index_bounds[i], str):
                        if this_index_bounds[i] == 'max_err':
                            this_index_bounds[i] = max(abs_prediction_errors)
                        elif this_index_bounds[i] == 'avg_err':
                            this_index_bounds[i] = sum(abs_prediction_errors)/len(abs_prediction_errors)
                        elif this_index_bounds[i].endswith('_sigma'):
                            this_index_bounds[i] = float(this_index_bounds[i].replace('_sigma',''))*self.data['stdevs'][data_label]
                        elif this_index_bounds[i].endswith('sig'):
                            this_index_bounds[i] = float(this_index_bounds[i].replace('sig',''))*self.data['stdevs'][data_label]
                        elif this_index_bounds[i].startswith('adherence/'):
                            factor = float(this_index_bounds[i].split('/')[1])
                            this_index_bounds[i] = error_distribution_functions[data_label].find_x(y_maxes[data_label]/factor)
                        else:
                            raise ValueError('Unknown index start or end value "{}"'.format(this_index_bounds[i]))

            x_starts[data_label] = this_index_bounds[0]
            x_ends[data_label] = this_index_bounds[1]

            # Compute error distribution function values for index start/end
            y_starts[data_label] = error_distribution_functions[data_label](x_starts[data_label])
            y_ends[data_label] = error_distribution_functions[data_label](x_ends[data_label])

            if index_type == 'log':
                try:
                    log_10_y_starts[data_label] = log10(y_starts[data_label] )
                except ValueError as e:
                    if str(e) == 'math domain error':
                        raise ValueError('Got a math domain error in computing the anomaly index start boundary for label "{}". This is likely due to extreme values and/or an error distribution badly approximating the real error distribution. Try changing the anomaly index boundaries or find a better error distribution.'.format(data_label)) from None
                    else:
                        raise
                try:
                    log_10_y_ends[data_label] = log10(y_ends[data_label] )
                except ValueError as e:
                    if str(e) == 'math domain error':
                        raise ValueError('Got a math domain error in computing the anomaly index end boundary for label "{}". This is likely due to extreme values and/or an error distribution badly approximating the real error distribution. Try changing the anomaly index boundaries or find a better error distribution.'.format(data_label)) from None
                    else:
                        raise

            # Prepare for the rolling anomaly index
            rolling_prediction_errors[data_label] = []

        # Is this series pre-processed?
        try:
            preprocessed = True if series.preprocessed_by == self.id else False
        except AttributeError:
            preprocessed = False

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

            # Check if we are allowed to compute the anomaly index given the data loss threshold, or if we have to abort
            abort=False
            if series[i].data_loss is not None and series[i].data_loss >= data_loss_threshold:
                abort=True
            else:
                for j in range(self.data['model_window']):
                    if series[i-j-1].data_loss is not None and series[i-j-1].data_loss >= data_loss_threshold:
                        abort=True
                        break
                if not abort:
                    if issubclass(self.model_class, Reconstructor):
                        if series[i-j-1].data_loss is not None and series[i+j+1].data_loss >= data_loss_threshold:
                            abort=True
                            break
            if abort:
                # Just append the item as-is and continue
                result_series.append(item)
                continue

            # Start computing the anomaly index
            item_anomaly_indexes = []

            for data_label in series.data_labels():

                # Shortcuts
                error_distribution_function = error_distribution_functions[data_label]
                x_start = x_starts[data_label]
                x_end = x_ends[data_label]
                y_start = y_starts[data_label]
                y_end = y_ends[data_label]
                if index_type == 'log':
                    log_10_y_start = log_10_y_starts[data_label]
                    log_10_y_end = log_10_y_ends[data_label]
                y_max = y_maxes[data_label]

                # Duplicate this series item
                item = deepcopy(item)

                # Compute the prediction error index
                if preprocessed:
                    actual = self._get_actual_value(series, i, data_label)
                    predicted = series.predictions[i][data_label]
                else:
                    actual = self._get_actual_value(series, i, data_label)
                    predicted = self._get_predicted_value(series, i, data_label, self.data['with_context'])

                if self.data['error_metric'] == 'E':
                    prediction_error = actual-predicted
                elif self.data['error_metric'] == 'AE':
                    prediction_error = abs(actual-predicted)
                elif self.data['error_metric'] == 'PE':
                    prediction_error = (actual-predicted)/actual
                elif self.data['error_metric'] == 'APE':
                    prediction_error = abs((actual-predicted)/actual)
                else:
                    raise ValueError('Unknown error type "{}"'.format(self.data['error_metric']))

                if self.data['error_metric']  in ['AE', 'APE']:
                    prediction_error = abs(prediction_error)

                # Handle the rolling window for the index if any
                if self.data['index_window'] > 1:
                    punctual_prediction_error = prediction_error
                    rolling_prediction_errors[data_label].append(prediction_error)
                    if (i >= self.data['model_window'] + self.data['index_window'] ):

                        # Compute the cumulative prediction errors on the rolling errors
                        prediction_error = sum(rolling_prediction_errors[data_label])

                        # Move the window
                        rolling_prediction_errors[data_label] = rolling_prediction_errors[data_label][1:]

                    else:
                        continue

                # Reverse values on the left side of the error distribution to simplify the math below
                distribution_loc= self.data['error_distributions_params'][data_label]['loc']
                if prediction_error < distribution_loc:
                    prediction_error =  distribution_loc + (distribution_loc - prediction_error)

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
                    if index_type == 'lin':
                        anomaly_index = 1 - ((y-y_end)/(y_start-y_end))
                    elif index_type == 'log':
                        anomaly_index = (log10(y) - log_10_y_start) / (log_10_y_end - log_10_y_start)
                    else:
                        if callable(index_type):
                            anomaly_index = index_type(x, y, x_start, x_end, y_start, y_end)
                        else:
                            raise ValueError('Unknown index type "{}"'.format(index_type))

                # Add this anomaly index for this data label to the list of the item anomaly indexes
                item_anomaly_indexes.append(anomaly_index)

                # Add details?
                if details:
                    if self.data['index_window']>1:
                        if isinstance(details, list):
                            if 'pred' in details:
                                item.data['{}_pred'.format(data_label)] = predicted
                            if 'err' in details:
                                item.data['{}_err_cum'.format(data_label)] = prediction_error
                            if 'adh' in details:
                                item.data['{}_adh_cum)'.format(data_label)] = y/y_max
                        elif isinstance(details, bool):
                            item.data['{}_pred'.format(data_label)] = predicted
                            item.data['{}_err_cum'.format(data_label)] = prediction_error
                            item.data['{}_err'.format(data_label)] = punctual_prediction_error
                            item.data['{}_adh_cum'.format(data_label)] = y/y_max
                        else:
                            raise TypeError('The "details" argument accepts only True/False or a list containing what details to add, as strings.')
                    else:
                        if isinstance(details, list):
                            if 'pred' in details:
                                item.data['{}_pred'.format(data_label)] = predicted
                            if 'err' in details:
                                item.data['{}_err'.format(data_label)] = prediction_error
                            if 'adh' in details:
                                item.data['{}_adh)'.format(data_label)] = y/y_max
                        elif isinstance(details, bool):
                            item.data['{}_pred'.format(data_label)] = predicted
                            item.data['{}_err'.format(data_label)] = prediction_error
                            item.data['{}_adh'.format(data_label)] = y/y_max
                        else:
                            raise TypeError('The "details" argument accepts only True/False or a list containing what details to add, as strings.')

            # Set anomaly index & append
            if item_anomaly_indexes:
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
    """An anomaly detection model based on a LSTM neural network forecaster."""

    model_class = LSTMForecaster

#===================================
# Linear Regression Anomaly Detector
#===================================

class LinearRegressionAnomalyDetector(ModelBasedAnomalyDetector):
    """An anomaly detection model based on a linear regression forecaster."""

    model_class = LinearRegressionForecaster

