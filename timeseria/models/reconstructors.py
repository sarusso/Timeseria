# -*- coding: utf-8 -*-
"""Data reconstructions models."""

import copy
import statistics
from ..utilities import get_periodicity, get_periodicity_index, set_from_t_and_to_t, item_is_in_range, mean_absolute_percentage_error
from ..time import dt_from_s
from sklearn.metrics import mean_absolute_error, mean_squared_error
from ..units import TimeUnit
from pandas import DataFrame
from math import sqrt
from ..datastructures import TimeSeries
from .base import Model, SeriesModel, _ProphetModel

# Setup logging
import logging
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings as default behavior
try:
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except (ImportError,AttributeError):
    pass



#=====================================
#  Generic Reconstructor
#=====================================

class Reconstructor(Model):
    """A generic reconstruction model.
    
    Args:
        path (str): a path from which to load a saved model. Will override all other init settings.
    """
    pass



#=====================================
#  Series Reconstructor
#=====================================

class SeriesReconstructor(Reconstructor, SeriesModel):
    """A reconstructor specifically designed to work with series data.
    
    Args:
        path (str): a path from which to load a saved model. Will override all other init settings.
    """

    def predict(self, series, *args, **kwargs):
        """Disabled. Reconstructors can be used only with the ``apply()`` method."""
        raise NotImplementedError('Reconstructors can be used only with the apply() method') from None
 
    
    def _predict(self, series, *args, **kwargs):
        raise NotImplementedError('Reconstructors can be used only with the apply() method.') from None


    def _apply(self, series, remove_data_loss=False, data_loss_threshold=1, inplace=False):

        logger.debug('Using data_loss_threshold="%s"', data_loss_threshold)

        # TODO: understand if we want the apply from/to behavior. For now it is disabled
        # (add from_t=None, to_t=None, from_dt=None, to_dt=None in the function call above)
        # from_t, to_t = set_from_t_and_to_t(from_dt, to_dt, from_t, to_t)
        # Maybe also add a timeseries.mark=[from_dt, to_dt]
         
        from_t = None
        to_t   = None
        
        if not inplace:
            series = series.duplicate()

        if len(series.data_labels()) > 1:
            raise NotImplementedError('Multivariate time series are not yet supported')

        for key in series.data_labels():
            
            gap_started = None
            
            for i, item in enumerate(series):
                
                # Skip if before from_t/dt of after to_t/dt
                if from_t is not None and series[i].t < from_t:
                    continue
                try:
                    # Handle slots
                    if to_t is not None and series[i].end.t > to_t:
                        break
                except AttributeError:
                    # Handle points
                    if to_t is not None and series[i].t > to_t:
                        break                

                if item.data_loss is not None and item.data_loss >= data_loss_threshold:
                    # This is the beginning of an area we want to reconstruct according to the data_loss_threshold
                    if gap_started is None:
                        gap_started = i
                else:
                    
                    if gap_started is not None:
                    
                        # Reconstruct for this gap
                        self._reconstruct(series, from_index=gap_started, to_index=i, key=key)
                        gap_started = None
                    
                    item.data_indexes['data_reconstructed'] = 0
                    
                if remove_data_loss:
                    item.data_indexes.pop('data_loss', None)
            
            # Reconstruct the last gap as well if left "open"
            if gap_started is not None:
                self._reconstruct(series, from_index=gap_started, to_index=i+1, key=key)

        if not inplace:
            return series
        else:
            return None


    def evaluate(self, series, steps='auto', limit=None, data_loss_threshold=1, metrics=['RMSE', 'MAE'], details=False, from_t=None, to_t=None, from_dt=None, to_dt=None):
        """Evaluate the reconstructor on a series.

        Args:
            steps (int, list): a single value or a list of values for how many steps (intended as missing data points or slots) 
                               to reconstruct in the evaluation. Default to automatic detection based on the model.
            limit(int): set a limit for the time series elements to use for the evaluation.
            data_loss_threshold(float): the data loss threshold required for the reconstructir to kick-in.
            metrics(list): the error metrics to use for the evaluation.
                Supported values are:
                ``RMSE`` (Root Mean Square Error), 
                ``MAE``  (Mean Absolute Error), and 
                ``MAPE``  (Mean Absolute percentage Error).
            details(bool): if to add intermediate steps details to the evaluation results.
            from_t(float): evaluation starting epoch timestamp.
            to_t(float): evaluation ending epoch timestamp
            from_dt(datetime): evaluation starting datetime.
            to_dt(datetime) : evaluation ending datetime.
        """
        return super(Reconstructor, self).evaluate(series, steps, limit, data_loss_threshold, metrics, details, from_t, to_t, from_dt, to_dt)


    def _evaluate(self, series, steps='auto', limit=None, data_loss_threshold=1, metrics=['RMSE', 'MAE'], details=False, from_t=None, to_t=None, from_dt=None, to_dt=None):

        # Set evaluation_score steps if we have to
        if steps == 'auto':
            # TODO: move the "auto" concept as a function that can be overwritten by child classes
            try:
                steps = [1, self.data['periodicity']]
            except KeyError:
                steps = [1, 2, 3]
        elif isinstance(steps, list):
            pass
        else:
            steps = list(range(1, steps+1))
            
        if isinstance(series, TimeSeries):
            timeseries = series
         
            # Support vars
            evaluation_score = {}
            from_t, to_t = set_from_t_and_to_t(from_dt, to_dt, from_t, to_t)
            warned = False
            
            # Log
            logger.info('Will evaluate model for %s steps with metrics %s', steps, metrics)
            
            # Find areas where to evaluate the model
            for key in timeseries.data_labels():
                 
                for steps_round in steps:
                    
                    # Support vars
                    real_values = []
                    reconstructed_values = []
                    processed_samples = 0
    
                    # Here we will have steps=1, steps=2 .. steps=n          
                    logger.debug('Evaluating model for %s steps', steps_round)
                    
                    for i in range(len(timeseries)):
    
                        # Skip if needed
                        try:
                            if not item_is_in_range(timeseries[i], from_t, to_t):
                                continue
                        except StopIteration:
                            break                  
                    
                        # Skip the first and the last ones, otherwise reconstruct the ones in the middle
                        if (i == 0) or (i >= len(timeseries)-steps_round):
                            continue
    
                        # Is this a "good area" where to test or do we have to stop?
                        stop = False
                        if timeseries[i-1].data_loss is not None and timeseries[i-1].data_loss >= data_loss_threshold:
                            stop = True
                        for j in range(steps_round):
                            if timeseries[i+j].data_loss is not None and timeseries[i+j].data_loss >= data_loss_threshold:
                                stop = True
                                break
                        if timeseries[i+steps_round].data_loss is not None and timeseries[i+steps_round].data_loss >= data_loss_threshold:
                            stop = True
                        if stop:
                            continue
                                
                        # Set prev and next
                        prev_value = timeseries[i-1].data[key]
                        next_value = timeseries[i+steps_round].data[key]
                        
                        # Compute average value
                        average_value = (prev_value+next_value)/2
                        
                        # Data to be reconstructed
                        timeseries_to_reconstruct = timeseries.__class__()
                        
                        # Append prev
                        #timeseries_to_reconstruct.append(copy.deepcopy(timeseries[i-1]))
                        
                        # Append in the middle and store real values
                        for j in range(steps_round):
                            item = copy.deepcopy(timeseries[i+j])
                            # Set the data_loss to one so the item will be reconstructed
                            item.data_indexes['data_loss'] = 1
                            item.data[key] = average_value
                            timeseries_to_reconstruct.append(item)
                            
                            real_values.append(timeseries[i+j].data[key])
                  
                        # Append next
                        #timeseries_to_reconstruct.append(copy.deepcopy(timeseries[i+steps_round]))
                        
                        # Do we have a 1-point only timeseries? If so, manually set the resolution
                        # as otherwise it would be not defined. # TODO: does it make sense?
                        if len(timeseries_to_reconstruct) == 1:
                            timeseries_to_reconstruct._resolution = timeseries.resolution
    
                        # Apply model inplace
                        self._apply(timeseries_to_reconstruct, inplace=True)
                        processed_samples += 1
    
                        # Store reconstructed values
                        for j in range(steps_round):
                            reconstructed_values.append(timeseries_to_reconstruct[j].data[key])
                        
                        # Break if we have to
                        if limit is not None and processed_samples >= limit:
                            break
                        
                        # Warn if no limit given and we are over
                        if not limit and not warned and i > 10000:
                            logger.warning('No limit set in the evaluation with a quite long time series, this could take some time.')
                            warned=True
                            
                    if limit and processed_samples < limit:
                        logger.warning('The evaluation limit is set to "{}" but I have only "{}" samples for "{}" steps'.format(limit, processed_samples, steps_round))
    
                    if not reconstructed_values:
                        raise Exception('Could not evaluate model, maybe not enough data?')
    
                    # Compute RMSE and ME, and add to the evaluation_score
                    if 'RMSE' in metrics:
                        evaluation_score['RMSE_{}_steps'.format(steps_round)] = sqrt(mean_squared_error(real_values, reconstructed_values))
                    if 'MAE' in metrics:
                        evaluation_score['MAE_{}_steps'.format(steps_round)] = mean_absolute_error(real_values, reconstructed_values)
                    if 'MAPE' in metrics:
                        evaluation_score['MAPE_{}_steps'.format(steps_round)] = mean_absolute_percentage_error(real_values, reconstructed_values)
    
            # Compute overall RMSE
            if 'RMSE' in metrics:
                sum_rmse = 0
                count = 0
                for key in evaluation_score:
                    if key.startswith('RMSE_'):
                        sum_rmse += evaluation_score[key]
                        count += 1
                evaluation_score['RMSE'] = sum_rmse/count
    
            # Compute overall MAE
            if 'MAE' in metrics:
                sum_me = 0
                count = 0
                for key in evaluation_score:
                    if key.startswith('MAE_'):
                        sum_me += evaluation_score[key]
                        count += 1
                evaluation_score['MAE'] = sum_me/count
    
            # Compute overall MAPE
            if 'MAPE' in metrics:
                sum_me = 0
                count = 0
                for key in evaluation_score:
                    if key.startswith('MAPE_'):
                        sum_me += evaluation_score[key]
                        count += 1
                evaluation_score['MAPE'] = sum_me/count
            
            if not details:
                simple_evaluation_score = {}
                if 'RMSE' in metrics:
                    simple_evaluation_score['RMSE'] = evaluation_score['RMSE']
                if 'MAE' in metrics:
                    simple_evaluation_score['MAE'] = evaluation_score['MAE']
                if 'MAPE' in metrics:
                    simple_evaluation_score['MAPE'] = evaluation_score['MAPE']
                evaluation_score = simple_evaluation_score
                
            return evaluation_score
        
        else:
            raise NotImplementedError('Evaluating this model on data types other than TimeSeries is not yet implemented')

    def _reconstruct(self, *args, **krargs):
        raise NotImplementedError('Reconstruction for this model is not yet implemented')



#=====================================
# Linear Interpolation Reconstructor
#=====================================

class LinearInterpolationReconstructor(SeriesReconstructor):
    """A series reconstruction model based on linear interpolation.
    
    The main difference between an intepolator and a reconstructor is that interpolators are used in the transformations, *before*
    resampling or aggregating and thus must support variable-resolution time series, while reconstructors are applied *after* resampling
    or aggregating, when data has been made already uniform.
    
    In general, in Timeseria a reconstructor modifies (by reconstructing) data which is already present but that cannot be trusted, either
    because it was created with a high data loss from a transformation or because of other domain-specific factors, while an interpolator
    creates the missing samples of the underlying signal which are required for the transformations to work.
    
    Interpolators can be then seen as special case of data reconstruction models, that on one hand implement a simpler logic, but that
    on the other must provide support for time-based math in order or to be able to work on variable-resolution time series.
    
    This reconstructor wraps a linear interpolator in order to perform the data reconstruction, and can be useful for setting a baseline
    when evaluating other, more sophisticated, data reconstruction models.
    """

    def _reconstruct(self, series, key, from_index, to_index):
        logger.debug('Reconstructing between "{}" and "{}" (excluded)'.format(from_index, to_index))
        
        try:
            self.interpolator_initialize
        except AttributeError:
            from ..interpolators import LinearInterpolator
            self.interpolator = LinearInterpolator(series)
                
        for i in range(from_index, to_index):

            logger.debug('Processing point=%s', series[i])
            reconstructed_data = self.interpolator.evaluate(at=series[i].t, prev_i=from_index-1, next_i=to_index)

            logger.debug('Reconstructed data=%s', reconstructed_data)
            series[i]._data = reconstructed_data



#=====================================
#  Periodic Average Reconstructor
#=====================================

class PeriodicAverageReconstructor(SeriesReconstructor):
    """A series reconstruction model based on periodic averages.
    
    Args:
        path (str): a path from which to load a saved model. Will override all other init settings.
    """

    def fit(self, series, data_loss_threshold=0.5, periodicity='auto', dst_affected=False,  offset_method='average', from_t=None, to_t=None, from_dt=None, to_dt=None):
        # This is a fit wrapper only to allow correct documentation
        
        # TODO: periodicity, dst_affected, offset_method -> move them in the init?
        """
        Fit the reconstructor on a series.
 
        Args:
            data_loss_threshold(float): the threshold of the data loss to discard an element from the fit.
            periodicity(int): the periodicty of the time series. If set to ``auto`` then it will be automatically detected using a FFT.
            dst_affected(bool): if the model should take into account DST effects.
            offset_method(str): how to offset the reconstructed data in order to align it to the missing data gaps. Valuse are ``avergae``
                                to use the average gap value, or ``extrmes`` to use its extremes.
            from_t(float): fit starting epoch timestamp.
            to_t(float): fit ending epoch timestamp
            from_dt(datetime): fit starting datetime.
            to_dt(datetime) : fit ending datetime.
        """
        return super(PeriodicAverageReconstructor, self).fit(series, data_loss_threshold, periodicity, dst_affected, offset_method, from_t, to_t, from_dt, to_dt)


    def _fit(self, series, data_loss_threshold=0.5, periodicity='auto', dst_affected=False, offset_method='average', from_t=None, to_t=None, from_dt=None, to_dt=None):

        if  isinstance(series, TimeSeries):
            timeseries = series

            if not offset_method in ['average', 'extremes']:
                raise Exception('Unknown offset method "{}"'.format(offset_method))
            self.offset_method = offset_method
        
            if len(timeseries.data_labels()) > 1:
                raise NotImplementedError('Multivariate time series are not yet supported')
    
            from_t, to_t = set_from_t_and_to_t(from_dt, to_dt, from_t, to_t)
            
            # Set or detect periodicity
            if periodicity == 'auto':
                periodicity =  get_periodicity(timeseries)
                try:
                    if isinstance(timeseries.resolution, TimeUnit):
                        logger.info('Detected periodicity: %sx %s', periodicity, timeseries.resolution)
                    else:
                        logger.info('Detected periodicity: %sx %ss', periodicity, timeseries.resolution)
                except AttributeError:
                    logger.info('Detected periodicity: %sx %ss', periodicity, timeseries.resolution)
                    
            self.data['periodicity']  = periodicity
            self.data['dst_affected'] = dst_affected 
                    
            for key in timeseries.data_labels():
                sums   = {}
                totals = {}
                processed = 0
                for item in timeseries:
                    
                    # Skip if needed
                    try:
                        if not item_is_in_range(item, from_t, to_t):
                            continue
                    except StopIteration:
                        break
                    
                    # Process. Note: we do fit on data losses = None!
                    if item.data_loss is None or item.data_loss < data_loss_threshold:
                        periodicity_index = get_periodicity_index(item, timeseries.resolution, periodicity, dst_affected=dst_affected)
                        if not periodicity_index in sums:
                            sums[periodicity_index] = item.data[key]
                            totals[periodicity_index] = 1
                        else:
                            sums[periodicity_index] += item.data[key]
                            totals[periodicity_index] +=1
                    processed += 1
    
            averages={}
            for periodicity_index in sums:
                averages[periodicity_index] = sums[periodicity_index]/totals[periodicity_index]
            self.data['averages'] = averages
            
            logger.debug('Processed "%s" items', processed)
        
        else:
            raise NotImplementedError('Fitting this model on data types other than TimeSeries is not yet implemented')


    def _reconstruct(self, series, key, from_index, to_index):
        
        if isinstance (series, TimeSeries):
            
            timeseries = series
                
            logger.debug('Reconstructing between "{}" and "{}"'.format(from_index, to_index-1))
    
            # Compute offset (old approach)
            if self.offset_method == 'average':
                diffs=0
                for j in range(from_index, to_index):
                    real_value = timeseries[j].data[key]
                    periodicity_index = get_periodicity_index(timeseries[j], timeseries.resolution, self.data['periodicity'], dst_affected=self.data['dst_affected'])
                    reconstructed_value = self.data['averages'][periodicity_index]
                    diffs += (real_value - reconstructed_value)
                offset = diffs/(to_index-from_index)
            
            elif self.offset_method == 'extremes':
                # Compute offset (new approach)
                diffs=0
                try:
                    for j in [from_index-1, to_index+1]:
                        real_value = timeseries[j].data[key]
                        periodicity_index = get_periodicity_index(timeseries[j], timeseries.resolution, self.data['periodicity'], dst_affected=self.data['dst_affected'])
                        reconstructed_value = self.data['averages'][periodicity_index]
                        diffs += (real_value - reconstructed_value)
                    offset = diffs/2
                except IndexError:
                    offset=0
            else:
                raise Exception('Unknown offset method "{}"'.format(self.offset_method))
    
            # Actually reconstruct
            for j in range(from_index, to_index):
                item_to_reconstruct = timeseries[j]
                periodicity_index = get_periodicity_index(item_to_reconstruct, timeseries.resolution, self.data['periodicity'], dst_affected=self.data['dst_affected'])
                item_to_reconstruct.data[key] = self.data['averages'][periodicity_index] + offset
                item_to_reconstruct.data_indexes['data_reconstructed'] = 1

        else:
            raise NotImplementedError('Reconstructing with this model on data types other than TimeSeries is not yet implemented')

   
    def _plot_averages(self, timeseries, **kwargs):   
        averages_timeseries = copy.deepcopy(timeseries)
        for item in averages_timeseries:
            value = self.data['averages'][get_periodicity_index(item, averages_timeseries.resolution, self.data['periodicity'], dst_affected=self.data['dst_affected'])]
            if not value:
                value = 0
            item.data['periodic_average'] = value 
        averages_timeseries.plot(**kwargs)



#=====================================
#  Prophet Reconstructor
#=====================================

class ProphetReconstructor(SeriesReconstructor, _ProphetModel):
    """A reconstructor based on Prophet. Prophet (from Facebook) implements a procedure for forecasting time series data based
    on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. 
    
    Args:
        path (str): a path from which to load a saved model. Will override all other init settings.
    """

    def _fit(self, series, from_t=None, to_t=None, from_dt=None, to_dt=None):

        if isinstance(series, TimeSeries):
            
            timeseries = series
            
            from prophet import Prophet
    
            from_t, to_t = set_from_t_and_to_t(from_dt, to_dt, from_t, to_t)
    
            if len(timeseries.data_labels()) > 1:
                raise NotImplementedError('Multivariate time series are not yet supported')
    
            data = self._from_timeseria_to_prophet(timeseries, from_t, to_t)
    
            # Instantiate the Prophet model
            self.prophet_model = Prophet()
            
            # Fit tjhe Prophet model
            self.prophet_model.fit(data)
        else:
            raise NotImplementedError('Fitting this model on data types other than TimeSeries is not yet implemented')
            

    def _reconstruct(self, series, key, from_index, to_index):
        
        if isinstance(series, TimeSeries):
            
            timeseries = series
            
            logger.debug('Reconstructing between "{}" and "{}"'.format(from_index, to_index-1))
        
            # Get and prepare data to reconstruct
            items_to_reconstruct = []
            for j in range(from_index, to_index):
                items_to_reconstruct.append(timeseries[j])
            data_to_reconstruct = [self._remove_timezone(dt_from_s(item.t)) for item in items_to_reconstruct]
            dataframe_to_reconstruct = DataFrame(data_to_reconstruct, columns = ['ds'])
    
            # Apply Prophet fit
            forecast = self.prophet_model.predict(dataframe_to_reconstruct)
            #forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
    
            # Ok, replace the values with the reconsturcted ones
            for i, j in enumerate(range(from_index, to_index)):
                #logger.debug('Reconstructing item #{} with reconstucted item #{}'.format(j,i))
                item_to_reconstruct = timeseries[j]
                item_to_reconstruct.data[key] = forecast['yhat'][i]
                item_to_reconstruct.data_indexes['data_reconstructed'] = 1

        else:
            raise NotImplementedError('Reconstructing with this model on data types other than TimeSeries is not yet implemented')
            
