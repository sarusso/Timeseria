import os
import json
import uuid
import copy
import statistics
from ..datastructures import DataTimeSlotSeries, DataTimeSlot, TimePoint, DataTimePointSeries, DataTimePoint, Slot, Point
from ..exceptions import NotFittedError, NonContiguityError, InputException
from ..utilities import get_periodicity, is_numerical, set_from_t_and_to_t, item_is_in_range
from ..utilities import check_timeseries, check_resolution, check_data_keys
from ..time import now_t, dt_from_s, s_from_dt
from datetime import timedelta, datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
from ..units import Unit, TimeUnit
from pandas import DataFrame
from numpy import array
from math import sqrt
from copy import deepcopy
from collections import OrderedDict
import shutil

# Keras and sklearn
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras import optimizers
from keras.models import load_model as load_keras_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Base models and utilities
from .base import TimeSeriesParametricModel, ProphetModel, ARIMAModel, KerasModel
from .base import get_periodicity_index, mean_absolute_percentage_error
from .forecasters import PeriodicAverageForecaster, LSTMForecaster

# Setup logging
import logging
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings as default behavior
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


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


    def __get_actual_and_predicted(self, timeseries, i, key, forecaster_window):

        # Call model predict logic and compare with the actual data
        actual    = timeseries[i].data[key]
        predicted = self.forecaster.predict(timeseries, n=1, forecast_start = i-1)[0][key]
        
        return (actual, predicted)


    def _fit(self, timeseries, *args, stdevs=3, **kwargs):

        if len(timeseries.data_keys()) > 1:
            raise NotImplementedError('Multivariate time series are not yet supported')
        
        # Initialize the forecaster              
        self.forecaster = self.forecaster_class()
        
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
        self.AE_threshold = stdev*stdevs


    def _apply(self, timeseries, inplace=False, details=False, logs=False):
        
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
                if AE > self.AE_threshold:
                    if logs:
                        logger.info('Detected anomaly for item starting @ {} ({}) with AE="{:.3f}..."'.format(item.t, item.dt, AE))
                    item.anomaly = 1
                    if details:
                        item.data['AE_{}'.format(key)] = AE
                        item.data['predicted_{}'.format(key)] = predicted
                else:
                    item.anomaly = 0
                    if details:
                        item.data['AE_{}'.format(key)] = AE
                        item.data['predicted_{}'.format(key)] = predicted

                result_timeseries.append(item)
        
        return result_timeseries 



#=============================
# P. Average Anomaly detector
#=============================

class PeriodicAverageAnomalyDetector(ForecasterAnomalyDetector):

    forecaster_class = PeriodicAverageForecaster



#=============================
# LSTM Anomaly detector
#=============================


#class LSTMAnomalyDetector(ForecasterAnomalyDetector):
#
#    forecaster_class = LSTMForecaster














