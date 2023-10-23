# -*- coding: utf-8 -*-
"""Data reconstructions models."""

import copy
import statistics
from ..utilities import detect_periodicity, _get_periodicity_index, _set_from_t_and_to_t, _item_is_in_range, mean_absolute_percentage_error
from ..time import dt_from_s
from sklearn.metrics import mean_absolute_error, mean_squared_error
from ..units import TimeUnit
from pandas import DataFrame
from math import sqrt
from ..datastructures import TimeSeries
from .base import Model, _ProphetModel
from datetime import datetime
from ..exceptions import NotEnoughDataError

# Setup logging
import logging
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings as default behavior
try:
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except (ImportError,AttributeError):
    pass

verbose_debug=False

#=====================================
#  Generic Reconstructor
#=====================================

class Reconstructor(Model):
    """A generic reconstruction model. This class of models work on reconstructing missing data,
    or in other words to fill gaps. Gaps need a "next" element after their end to be defined, which
    can bring much more information to the model with respect to a forecasting task.
    
    Args:
        path (str): a path from which to load a saved model. Will override all other init settings.
    """
    
    def predict(self, series, from_i, to_i, *args, **kwargs):
        "Predict the reconstructed value(s) from/to the given indexes (included)."
 
        # Check series # TODO: this check here is redundant, as already performed in the parent predict(),
        # but otherwise the following len check might fail.
        if not isinstance(series, TimeSeries):
            raise NotImplementedError('Models work only with TimeSeries data for now (got "{}")'.format(series.__class__.__name__))

        # Check if we have enough data to predict, if there is a window.
        # Note: for reconstructors, widows are assumed to be symmetric
        # Also note: considerationson the data loss are don in the apply, not here.
        try:
            if from_i - self.window < 0:
                raise ValueError('There is not enough data before the gap to reconstruct for the required window (from_i={}, to_i={}, window_size={}, series_len={})'.format(from_i, to_i, self.window, len(series)))
            if to_i + self.window > len(series) - 1:
                raise ValueError('There is not enough data after the gap to reconstruct for the required window (from_i={}, to_i={}, window_size={}, series_len={})'.format(from_i, to_i, self.window, len(series)))
        except KeyError:
            pass
        
        # Call parent predict
        return super(Reconstructor, self).predict(series, from_i, to_i, *args, **kwargs)

    def _apply(self, series, remove_data_loss=False, data_loss_threshold=1, inplace=False):

        logger.debug('Using data_loss_threshold="%s"', data_loss_threshold)

        # TODO: understand if we want the apply from/to behavior. For now it is disabled
        # (add from_t=None, to_t=None, from_dt=None, to_dt=None in the function call above)
        # from_t, to_t = _set_from_t_and_to_t(from_dt, to_dt, from_t, to_t)
        # Maybe also add a series.mark=[from_dt, to_dt]
         
        from_t = None
        to_t   = None
        
        if not inplace:
            series = series.duplicate()

        gap_start_i = None
        
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
                if gap_start_i is None:
                    logger.debug('Detected gap start: #%s', i)
                    gap_start_i = i
            else:
                
                if gap_start_i is not None:
                    gap_end_i = i-1
                    logger.debug('Detected gap to reconstruct from #%s to #%s (included)', gap_start_i,gap_end_i)
                    
                    self._reconstruct(series, from_i=gap_start_i, to_i=gap_end_i, inplace=True)

                    # Reset the gap
                    gap_start_i = None
                
                item.data_indexes['data_reconstructed'] = 0
                
            if remove_data_loss:
                item.data_indexes.pop('data_loss', None)
        
        # Try to reconstruct the last gap as well if left "open", but will likely raise and error due to no window
        if gap_start_i is not None:
            logger.debug('Detected last gap: from #%s to %s', gap_start_i, i)
            self._reconstruct(series, from_i=gap_start_i, to_i=i, inplace=True)

        if not inplace:
            return series
        else:
            return None

    def _reconstruct(self, series, from_i, to_i, inplace=False):
        logger.debug('Called _reconstruct from #%s to #%s', from_i, to_i)
        predicted_data = self.predict(series, from_i=from_i,to_i=to_i)
        logger.debug('Got predicted data: %s', predicted_data)
        if not inplace:
            reconstructed_data=[]
        
        for j in range(from_i, to_i+1):
            
            # Get the predicted value of each data label for
            # this step and compose the new item data
            if verbose_debug:
                logger.debug('Processing index #%s and relative index #%s', j, j-from_i)
            
            item_data = {}
            if isinstance(predicted_data, dict):
                # Dict of lists
                for data_label in series.data_labels():
                    item_data[data_label] = predicted_data[data_label][j-from_i]
            elif isinstance(predicted_data, list):
                # List of dicts
                item_data = predicted_data[j-from_i]
            else:
                raise TypeError('Don\'t know how to handle data of type "{}" returned by the reconstructor prediction'.format(predicted_data.__class__.__name__)) 

            if inplace:
                # Just change the data
                series[j]._data = item_data
                series[j].data_indexes['data_reconstructed'] = 1
                
            else:
                # Create an entire new item
                reconstructed_item = series.items_type.__class__(t=series[j].t,
                                                                 data=item_data,
                                                                 data_indexes=copy.deepcopy(series[j].data_indexes))
                reconstructed_item.data_indexes['data_reconstructed'] = 1
                reconstructed_data.append(reconstructed_item)
        
        if not inplace:
            return reconstructed_data


    def evaluate(self, series, steps='auto', limit=None, data_loss_threshold=1, metrics=['RMSE', 'MAE'], details=False, start=None, end=None, **kwargs):
        # This method is overwritten just to change the docstring
        # TODO: Maybe completely overwrite this kind of methods and get the automatic sanification with decorators,
        # leaving this approach only for the _fit and _predict but maybe even there it would be better to completely
        # overwrite and use decorators instead. Consider also fit_logic() or similar. p.s. @model_fit sounds nice.
        """Evaluate the reconstructor on a series.

        Args:
            steps (int, list): a single value or a list of values for how many steps (intended as missing data points or slots) 
                               to reconstruct in the evaluation. Default to automatic detection based on the model.
            limit(int): set a limit for the time data elements to use for the evaluation.
            data_loss_threshold(float): the data_loss index threshold required for the reconstructor to kick-in.
            metrics(list): the error metrics to use for the evaluation.
                Supported values are:
                ``RMSE`` (Root Mean Square Error), 
                ``MAE``  (Mean Absolute Error), and 
                ``MAPE``  (Mean Absolute percentage Error).
            details(bool): if to add intermediate steps details to the evaluation results.
            start(float, datetime): evaluation start (epoch timestamp or datetime).
            end(float, datetim): evaluation end (epoch timestamp or datetime).
        """
        return super(Reconstructor, self).evaluate(series, steps, limit, data_loss_threshold, metrics, details, start, end, **kwargs)

    def _evaluate(self, series, steps='auto', limit=None, data_loss_threshold=1, metrics=['RMSE', 'MAE'], details=False, start=None, end=None, **kwargs):

        if len(series.data_labels()) > 1:
            raise NotImplementedError('Evaluating on multivariate time sries is not yet implemented')
        
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

        # Handle start/end
        from_t = kwargs.get('from_t', None)
        to_t = kwargs.get('to_t', None)
        from_dt = kwargs.get('from_dt', None)
        to_dt = kwargs.get('to_dt', None)        
        if from_t or to_t or from_dt or to_dt:
            logger.warning('The from_t, to_t, from_dt and to_d arguments are deprecated, please use start and end instead')
        from_t, to_t = _set_from_t_and_to_t(from_dt, to_dt, from_t, to_t)

        if start is not None:       
            if isinstance(start, datetime):
                from_dt = start
            else:
                try:
                    from_t = float(start)
                except:
                    raise ValueError('Cannot use "{}" as start value, not a datetime nor an epoch timestamp'.format(start))
        if end is not None:       
            if isinstance(end, datetime):
                to_dt = end
            else:
                try:
                    to_t = float(end)
                except:
                    raise ValueError('Cannot use "{}" as end value, not a datetime nor an epoch timestamp'.format(end))

        # Support vars
        evaluation_score = {}
        warned = False
        
        # Log
        logger.info('Will evaluate model for %s steps with metrics %s', steps, metrics)


        # Find areas where to evaluate the model
        for data_label in series.data_labels():
             
            for steps_round in steps:
                
                # Support vars
                overall_actual_values = []
                overall_predicted_values = []
                processed_samples = 0

                # Here we will have steps=1, steps=2 .. steps=n          
                logger.debug('Evaluating model for %s steps', steps_round)
                
                for i in range(len(series)):

                    # Skip if needed
                    # TODO: Use series wiews/slices
                    try:
                        if not _item_is_in_range(series[i], from_t, to_t):
                            continue
                    except StopIteration:
                        break

                    # Break if we have to
                    if limit is not None and processed_samples >= limit:
                        break
                    
                    # Warn if no limit given and we are over
                    if not limit and not warned and i > 10000:
                        logger.warning('No limit set in the evaluation with a quite long time series, this could take some time.')
                        warned=True
                    
                    # Now start to check if this a "good area" where to evaluate.
                    stop = False
                    
                    # First, if there is a window check that there is enough window data and that it is good data
                    try:
                        #  i=0 -> 0-1 = -1-th element is the first required (not existent)
                        #  i=1 -> 1-1 = 0-th element is the last required 
                        #  i=2 -> 2-1 = 2-nd element is the last required
                        if i - self.window < 0:
                            stop = True
                        else:
                            for j in range(i - self.window, i):
                                if series[j].data_loss is not None and series[j].data_loss >= data_loss_threshold:
                                    stop = True
                                    break
                    except KeyError:
                        # Model has no window
                        pass
                    else:
                        if stop:
                            continue
                    try:
                        #logger.debug('{} - {} - {} -  {} - {} '.format(i, steps_round, self.window, i + steps_round + self.window, len(series)))
                        #  len=6:  i=0 -> 0+1+1 = 2-th element is the last required 
                        #  len=6:  i=3 -> 3+1+1 = 5-th element is the last required 
                        #  len=6:  i=4 -> 4+1+1 = 6-th element is the last required (not existent)
                        if (i + steps_round + self.window) >= len(series):
                            stop = True
                        else:
                            for j in range(i + steps_round, i + steps_round + self.window):
                                if series[j].data_loss is not None and series[j].data_loss >= data_loss_threshold:
                                    stop = True
                                    break
                    except KeyError:
                        # Model has no window
                        pass
                    else:
                        if stop:
                            continue

                    # Then, check that the gap data is good data:
                    for j in range(i, i + steps_round):
                        if series[j].data_loss is not None and series[j].data_loss >= data_loss_threshold:
                            stop = True
                            break
                    if stop:
                        continue

                    # Ok, now call the predict and compare with real data
                    predicted_values = self.predict(series, i, i+steps_round-1)
                    real_values = []
                    
                    for j in range(i, i + steps_round):
                        real_values.append(series[j].data[data_label])
                        
                    # Add predicted and real values to the overall data to be later used for the RMS etc.
                    overall_predicted_values += predicted_values[data_label]
                    overall_actual_values += real_values
                    processed_samples+=1

                if limit and processed_samples < limit:
                    logger.warning('The evaluation limit is set to "{}" but I got only "{}" samples for "{}" steps'.format(limit, processed_samples, steps_round))
                
                if not overall_predicted_values:
                    raise NotEnoughDataError('Could not evaluate the model at all for steps={} as not good evaluation data was found'.format(steps))

                # Compute RMSE and ME, and add to the evaluation_score
                if 'RMSE' in metrics:
                    evaluation_score['RMSE_{}_steps'.format(steps_round)] = sqrt(mean_squared_error(overall_predicted_values, overall_actual_values))
                if 'MAE' in metrics:
                    evaluation_score['MAE_{}_steps'.format(steps_round)] = mean_absolute_error(overall_predicted_values, overall_actual_values)
                if 'MAPE' in metrics:
                    evaluation_score['MAPE_{}_steps'.format(steps_round)] = mean_absolute_percentage_error(overall_predicted_values, overall_actual_values)

        # Compute overall RMSE
        if 'RMSE' in metrics:
            sum_rmse = 0
            count = 0
            for data_label in evaluation_score:
                if data_label.startswith('RMSE_'):
                    sum_rmse += evaluation_score[data_label]
                    count += 1
            evaluation_score['RMSE'] = sum_rmse/count

        # Compute overall MAE
        if 'MAE' in metrics:
            sum_me = 0
            count = 0
            for data_label in evaluation_score:
                if data_label.startswith('MAE_'):
                    sum_me += evaluation_score[data_label]
                    count += 1
            evaluation_score['MAE'] = sum_me/count

        # Compute overall MAPE
        if 'MAPE' in metrics:
            sum_me = 0
            count = 0
            for data_label in evaluation_score:
                if data_label.startswith('MAPE_'):
                    sum_me += evaluation_score[data_label]
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


#=====================================
# Linear Interpolation Reconstructor
#=====================================

class LinearInterpolationReconstructor(Reconstructor):
    """A reconstruction model based on linear interpolation.
    
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
    
    window = 1

    def _predict_to_refactor(self, series, data_label, from_index, to_index):
        pass
    
    def _predict(self, series, from_i, to_i):

        logger.debug('Linear Interpolator reconstructing from #%s to #%s (included)', from_i, to_i)
        
        try:
            self.interpolator_initialize
        except AttributeError:
            from ..interpolators import LinearInterpolator
            self.interpolator = LinearInterpolator(series)
        
        predicted_data = []
        for i in range(from_i, to_i+1):

            logger.debug('Processing point=%s', series[i])
            step_predicted_data = self.interpolator.evaluate(at=series[i].t, prev_i=from_i-1, next_i=to_i+1)

            logger.debug('Predicted data=%s', step_predicted_data)
            predicted_data.append(step_predicted_data)

        return predicted_data


#=====================================
#  Periodic Average Reconstructor
#=====================================

class PeriodicAverageReconstructor(Reconstructor):
    """A series reconstruction model based on periodic averages.
    
    Args:
        path (str): a path from which to load a saved model. Will override all other init settings.
    """

    def fit(self, data, data_loss_threshold=0.5, periodicity='auto', dst_affected=False,  offset_method='average', start=None, end=None, **kwargs):
        # This is a fit wrapper only to allow correct documentation
        
        # TODO: periodicity, dst_affected, offset_method -> move them in the init?
        """
        Fit the reconstructor on some data.
 
        Args:
            data_loss_threshold(float): the threshold of the data_loss index for discarding an element from the fit.
            periodicity(int): the periodicty of the time series. If set to ``auto`` then it will be automatically detected using a FFT.
            dst_affected(bool): if the model should take into account DST effects.
            offset_method(str): how to offset the reconstructed data in order to align it to the missing data gaps. Valuse are ``avergae``
                                to use the average gap value, or ``extrmes`` to use its extremes.
            start(float, datetime): fit start (epoch timestamp or datetime).
            end(float, datetim): fit end (epoch timestamp or datetime).
        """
        return super(PeriodicAverageReconstructor, self).fit(data, data_loss_threshold, periodicity, dst_affected, offset_method, start, end, **kwargs)

    def _fit(self, series, data_loss_threshold=0.5, periodicity='auto', dst_affected=False, offset_method='average', start=None, end=None, **kwargs):

        if not offset_method in ['average', 'extremes']:
            raise Exception('Unknown offset method "{}"'.format(offset_method))
        self.offset_method = offset_method
    
        if len(series.data_labels()) > 1:
            raise NotImplementedError('Multivariate time series are not yet supported')

        # This reconstructor has always a one-point window (before and after gaps)
        self.window = 1

        # Handle start/end
        from_t = kwargs.get('from_t', None)
        to_t = kwargs.get('to_t', None)
        from_dt = kwargs.get('from_dt', None)
        to_dt = kwargs.get('to_dt', None)        
        if from_t or to_t or from_dt or to_dt:
            logger.warning('The from_t, to_t, from_dt and to_d arguments are deprecated, please use the slice() operation instead or the square brackets notation.')
        from_t, to_t = _set_from_t_and_to_t(from_dt, to_dt, from_t, to_t)

        if start is not None:       
            if isinstance(start, datetime):
                from_dt = start
            else:
                try:
                    from_t = float(start)
                except:
                    raise ValueError('Cannot use "{}" as start value, not a datetime nor an epoch timestamp'.format(start))
        if end is not None:       
            if isinstance(end, datetime):
                to_dt = end
            else:
                try:
                    to_t = float(end)
                except:
                    raise ValueError('Cannot use "{}" as end value, not a datetime nor an epoch timestamp'.format(end))
                
        # Set or detect periodicity
        if periodicity == 'auto':
            periodicity =  detect_periodicity(series)
            try:
                if isinstance(series.resolution, TimeUnit):
                    logger.info('Detected periodicity: %sx %s', periodicity, series.resolution)
                else:
                    logger.info('Detected periodicity: %sx %ss', periodicity, series.resolution)
            except AttributeError:
                logger.info('Detected periodicity: %sx %ss', periodicity, series.resolution)
                
        self.data['periodicity']  = periodicity
        self.data['dst_affected'] = dst_affected 
                
        for data_label in series.data_labels():
            sums   = {}
            totals = {}
            processed = 0
            for item in series:
                
                # Skip if needed
                try:
                    if not _item_is_in_range(item, from_t, to_t):
                        continue
                except StopIteration:
                    break
                
                # Process. Note: we do fit on data losses = None!
                if item.data_loss is None or item.data_loss < data_loss_threshold:
                    periodicity_index = _get_periodicity_index(item, series.resolution, periodicity, dst_affected=dst_affected)
                    if not periodicity_index in sums:
                        sums[periodicity_index] = item.data[data_label]
                        totals[periodicity_index] = 1
                    else:
                        sums[periodicity_index] += item.data[data_label]
                        totals[periodicity_index] +=1
                processed += 1

        averages={}
        for periodicity_index in sums:
            averages[periodicity_index] = sums[periodicity_index]/totals[periodicity_index]
        self.data['averages'] = averages
        
        logger.debug('Processed "%s" items', processed)


    def _predict(self, series, from_i, to_i):
        
        if verbose_debug:
            logger.debug('Periodic Average Reconstructor predicting from_i=%s to_i=%s (included)', from_i, to_i)
        predictions={}
        
        # TODO: support multivariate
        #for data_label in series.data_labels():
        data_label = series.data_labels()[0]
            
        # Compute offset using the 1-point window. This basically compares the differences between
        # the values predicted by the model and the real ones *in the window* to compute the offset.
        diffs=0
        for j in [from_i-1, to_i+1]:
            real_value = series[j].data[data_label]
            periodicity_index = _get_periodicity_index(series[j], series.resolution, self.data['periodicity'], dst_affected=self.data['dst_affected'])
            reconstructed_value = self.data['averages'][periodicity_index]
            diffs += (real_value - reconstructed_value)
        offset = diffs/2

        # Predict the reconstructed value. The right is included, hence the "+1" in the range.
        predictions[data_label] = []
        for j in range(from_i, to_i+1):
            item_to_reconstruct = series[j]
            periodicity_index = _get_periodicity_index(item_to_reconstruct, series.resolution, self.data['periodicity'], dst_affected=self.data['dst_affected'])
            predictions[data_label].append(self.data['averages'][periodicity_index] + offset)
        
        return predictions

    def _plot_averages(self, series, **kwargs):   
        averages_series = copy.deepcopy(series)
        for item in averages_series:
            value = self.data['averages'][_get_periodicity_index(item, averages_series.resolution, self.data['periodicity'], dst_affected=self.data['dst_affected'])]
            if not value:
                value = 0
            item.data['periodic_average'] = value 
        averages_series.plot(**kwargs)


#=====================================
#  Prophet Reconstructor
#=====================================

class ProphetReconstructor(Reconstructor, _ProphetModel):
    """A series reconstruction model based on Prophet. Prophet (from Facebook) implements a procedure for forecasting time series data based
    on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. 
    
    Args:
        path (str): a path from which to load a saved model. Will override all other init settings.
    """

    window = 0

    def _fit(self, series, start=None, end=None, **kwargs):

        if len(series.data_labels()) > 1:
            raise NotImplementedError('Multivariate time series are not supported by this reconstructor')

        from prophet import Prophet

        # Handle start/end
        from_t = kwargs.get('from_t', None)
        to_t = kwargs.get('to_t', None)
        from_dt = kwargs.get('from_dt', None)
        to_dt = kwargs.get('to_dt', None)        
        if from_t or to_t or from_dt or to_dt:
            logger.warning('The from_t, to_t, from_dt and to_d arguments are deprecated, please use the slice() operation instead or the square brackets notation.')
        from_t, to_t = _set_from_t_and_to_t(from_dt, to_dt, from_t, to_t)

        if start is not None:       
            if isinstance(start, datetime):
                from_dt = start
            else:
                try:
                    from_t = float(start)
                except:
                    raise ValueError('Cannot use "{}" as start value, not a datetime nor an epoch timestamp'.format(start))
        if end is not None:       
            if isinstance(end, datetime):
                to_dt = end
            else:
                try:
                    to_t = float(end)
                except:
                    raise ValueError('Cannot use "{}" as end value, not a datetime nor an epoch timestamp'.format(end))

        data = self._from_timeseria_to_prophet(series, from_t, to_t)

        # Instantiate the Prophet model
        self.prophet_model = Prophet()
        
        # Fit tjhe Prophet model
        self.prophet_model.fit(data)

    def _predict(self, series, from_i, to_i):
        
        if verbose_debug:
            logger.debug('Periodic Average Reconstructor predicting from_i=%s to_i=%s (included)', from_i, to_i)
        
        if len(series.data_labels()) > 1:
            raise NotImplementedError('Multivariate time series are not supported by this reconstructor')

        data_label = series.data_labels()[0]
        
        # Get and prepare data to reconstruct
        # TODO: check the following..
        items_to_reconstruct = []
        for j in range(from_i, to_i+1):
            items_to_reconstruct.append(series[j])
        data_to_reconstruct = [self._remove_timezone(dt_from_s(item.t)) for item in items_to_reconstruct]
        dataframe_to_reconstruct = DataFrame(data_to_reconstruct, columns = ['ds'])

        # Apply Prophet fit
        forecast = self.prophet_model.predict(dataframe_to_reconstruct)
        #forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

        predicted_data = {data_label: []}
        for j in range(0, to_i-from_i+1):
            predicted_data[data_label].append(forecast['yhat'][j])

        return predicted_data

