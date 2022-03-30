# -*- coding: utf-8 -*-
"""Data storages, as the CSV file and SQLite database."""

import os
import datetime
from .utilities import detect_encoding, sanitize_string, is_list_of_integers, to_float
from .units import TimeUnit
from .datastructures import DataTimePoint, DataTimePointSeries, TimePointSeries, TimePoint, DataTimeSlot, DataTimeSlotSeries
from .time import dt_from_str, dt_from_s, s_from_dt, timezonize, now_dt
from .exceptions import NoDataException, FloatConversionError, ConsistencyException


# Setup logging
import logging
logger = logging.getLogger(__name__)

HARD_DEBUG = False

POSSIBLE_TIMESTAMP_labelS = ['timestamp', 'epoch']
NO_DATA_PLACEHOLDERS = ['na', 'nan', 'null', 'nd', 'undef']
DEFAULT_SLOT_DATA_LOSS = None


#======================
#  Base Storage class
#======================
class Storage(object):
    """The base storage class. Can be implemented to store one or more time serieses. If storing more than
    one, then the id of which serises to load must be provided in the ``get()`` and ``put()`` methods."""
    
    def get(self, *args, **kwargs):
        raise NotImplementedError()
    
    def put(self, *args, **kwargs):
        raise NotImplementedError()


#======================
#  CSV File Storage
#======================

class CSVFileStorage(Storage):
    """A CSV file storage. Supports creating both point and slot time series.
    
    The file encoding, the series type and the tiemstamp columns are all auto-detect with an heuristic approach
    by defaaul, and asked to be set by ony if the euristic fails. In particular, wether to create point or solot
    series is based on the sampling interval automatically detected: if this is above 24 hours, then slots are used.
    
    The header with the column labels is optional, and if not present the column numbers are used as labels.
    Comments in the CSV file are supported Comments in the CSV file only as full-line coments, where the line starts
    with one of the charatchers listed as comment characheter (``#`` and ``;`` by default).
    
    The ``timestamp_format``, ``date_format`` and ``time_format`` arguments can be set using Python strptime format codes
    (https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes), but the ``timestamp_format``
    also supports two special formats:
    
        * ``epoch``: the epoch timestamp, intended as the number of seconds from the 1st January 1970 UTC with decimals for sub-second precision.
        * ``iso8601``: the ISO 8601 timestamp format (see https://www.w3.org/TR/NOTE-datetime)
    
    Data can be simply comma-separated (or using a custom separator) as well as single or double quoted, check out the RFC4180
    (https://datatracker.ietf.org/doc/html/rfc4180) for more details on the CSV file format specification.
    
    Args:
        filename: the file name (including its path).  
        timestamp_label: the column label to be used as timestamp. Either use this or the time_label and/or date_label parameters.
        timestamp_format: the timestamp column format.
        time_label: the column label to be used as the time part of the timestamp.
        time_format: the time column format.
        date_label: the column label to be used as the date part of the timestamp.
        date_format: the date column format. 
        tz: the timezone on which to create the time series on.
        data_labels: the column labels to be used as data by default. Excpected as a list of 
                     strings or list of integers, in which case are treated as column numbers.
        data_type: the data type (``list`` or ``dict``), set automatically by default.
        series_type: the default type of the series, if ``points`` or ``slots``. Automatically set by default.
        sort: if to sort the data file before creating the series.
        separator: the separator for the records (fields), ``,`` by default.
        newline: the newline charachter, ``\\n`` by default.
        comment_chars: the charachters used to mark a comment line. Defaulted to ``#`` and ``;``. 
        encoding: the encoding of the file, automatically detected by default.
        skip_errors: if to skip errors or raise an exception.
        silence_errors: if to completely silence errors when skipping them or not.     
    """
    
    def __init__(self, filename, timestamp_label='auto', timestamp_format='auto',
                 time_label=None, time_format=None, date_label=None, date_format=None,
                 tz='UTC', data_labels='all', data_type='auto', series_type='auto', sort=False,
                 separator=',', newline='\n', comment_chars = ['#', ';'],  encoding='auto',
                 skip_errors=False, silence_errors=False):

        # Parameters sanity checks and adjustments
        if timestamp_label is None and time_label is None and date_label is None and not data_labels:
            raise ValueError('No timestamp column, time column, date column or data columns provided, cannot get anything from this CSV file')

        if timestamp_label != 'auto' and (time_label or date_label):
            raise ValueError('Got both timestamp column and time/date columns, choose which approach to use.')

        if timestamp_label is not None and not isinstance(timestamp_label, int) and not isinstance(timestamp_label, str):
            raise ValueError('timestamp_label argument must be a string (or integer for coulmn number) (got "{}")'.format(timestamp_label.__class__.__name__))

        if time_label is not None:
            if not isinstance(time_label, int) and not isinstance(time_label, str):
                raise ValueError('time_label argument must be a string (or integer for coulmn number) (got "{}")'.format(time_label.__class__.__name__))
            if not time_format:
                raise ValueError('If giving a time_label, a time_format is required as well')
            
            # Disble the timestamp label if a time (or date) label is set
            timestamp_label = None
            timestamp_format = None

        if date_label is not None:
            if not isinstance(date_label, int) and not isinstance(date_label, str):
                raise ValueError('date_label argument must be a string (or integer for coulmn number) (got "{}")'.format(date_label.__class__.__name__))
            if not date_format:
                raise ValueError('If giving a date_label, a date_format is required as well')
 
            # Disble the timestamp label if a date (or time) label is set
            timestamp_label = None
            timestamp_format = None           

        if data_labels != 'all' and not isinstance(data_labels, list):
            raise ValueError('data_labels argument must be a list (got "{}")'.format(data_labels.__class__.__name__))

        # File & encoding
        self.filename = filename
        if encoding == 'auto':
            self.encoding = None
        else:
            self.encoding = encoding

        # Set time parameters
        self.timestamp_label = timestamp_label
        self.timestamp_format = timestamp_format
        
        self.date_label = date_label
        self.date_format = date_format        
        
        self.time_label = time_label
        self.time_format = time_format
        
        # Timezone
        self.tz = tz   
        
        # Data TODO: allow for columns mapping here? i.e data_labels = {1:'flow',2:'temperature'}
        self.data_labels = data_labels
        
        # Separator, newline and comment chars
        self.separator = separator
        self.newline = newline
        self.comment_chars = comment_chars

        # Other
        self.skip_errors = skip_errors
        self.silence_errors = silence_errors
        self.data_type = data_type
        self.sort = sort
        
        # Set series type (that will be forced)
        self.series_type = series_type


    def get(self, limit=None, as_tz=None, as_points=None, as_slots=None, data_labels='all', data_label=None):
        """Get the data out from the CSV file.

        Args:
            limit: a row number limit.  
            as_tz: force a specific timezone.
            as_points: force generating points.
            as_slots: force generating slots.
            data_labels: get only specific data labels.
            data_label: get only a specific data label.
        """
        
        
        # TODO: add from_dt / to_dt /from_t / to_t. Cfr series.filter()

        # Sanity checks
        if as_points and as_slots:
            raise ValueError('Got both as_points and as_slots, set only one or none')

        # Use the data label if given
        if data_label is not None:
            data_labels=[data_label]

        # Line counter
        line_number=0
        
        # Init column indexes and labels
        column_indexes = None
        column_labels = None
        
        timestamp_label_index = None
        time_label_index = None
        date_label_index = None
        
        data_label_indexes = None
        data_label_names = None
        
        # Data type
        data_type = None

        # Detect encoding if not set
        if not self.encoding:
            self.encoding = detect_encoding(self.filename, streaming=False)

        items = []

        # TODO: evaluate rU vs newline='\n'
        with open(self.filename, 'r', encoding=self.encoding) as csv_file:

            while True:
                
                # TODO: replace me using a custom line separator
                line = csv_file.readline()
                logger.debug('Processing line #%s: "%s"', line_number, sanitize_string(line,NO_DATA_PLACEHOLDERS))

                # Do we have to stop? Note: empty lines still have the "\n" char.
                if not line:
                    break
                
                # Is this line an empty line?
                #if not line.strip():
                #    continue 
                                
                # Is this line a comment? 
                if line[0] in self.comment_chars:
                    logger.debug('Skipping line "%s" as marked as comment line', line)
                    continue

                # Is this line something else? 
                if not self.separator in line:
                    logger.debug('Skipping line "%s" as marked no value separator found', line)
                    continue

                # Get values from this line:
                line_items = line.split(self.separator)

                # Set column keys if not already done
                if not column_indexes:
                    
                    # Is this line carrying non-numerical or non timestamp data?
                    not_converted = 0
                    for value in line_items:
                        try:
                            float(sanitize_string(value,NO_DATA_PLACEHOLDERS))
                        except:
                            try:
                                dt_from_str(sanitize_string(value,NO_DATA_PLACEHOLDERS))
                            except:
                                not_converted += 1
          
                    # If it is, use it as labels (and continue with the next line)
                    if not_converted: # == len(line_items):
                        column_indexes = [i for i in range(len(line_items)) if sanitize_string(line_items[i],NO_DATA_PLACEHOLDERS)]
                        column_labels = [sanitize_string(label,NO_DATA_PLACEHOLDERS) for label in line_items if sanitize_string(label,NO_DATA_PLACEHOLDERS)]
                        logger.debug('Set column indexes = "%s"  and column labels = "%s"', column_indexes, column_labels)
                        continue
                    
                    # Otherwise, just use integer keys (and go on)
                    else:
                        column_indexes = [i for i in range(len(line_items)) if sanitize_string(line_items[i],NO_DATA_PLACEHOLDERS)]
                        logger.debug('Set column indexes = "%s"  and column labels = "%s"', column_indexes, column_labels)

                    # Note: above we only used labels and indexes that carry content after being sanitize_stringd.
                    # TODO: what if a file starts with a line missing last value and we do or not do set data columns from the outside?
                    #       .. external data columns should have precedenze, but a mapping is then required more than just a list.

                #====================
                #  Time part
                #====================

                # Set timestamp column index if not already done
                if self.timestamp_label is not None and timestamp_label_index is None:
                    if self.timestamp_label=='auto':
                        # TODO: improve this auto-detect: try different column names and formats.
                        #       Maybe, fix position as the first element for the timestamp.
                        if column_labels:
                            for posssible_timestamp_label_name in POSSIBLE_TIMESTAMP_labelS:
                                if posssible_timestamp_label_name in column_labels:
                                    timestamp_label_index = column_labels.index(posssible_timestamp_label_name)
                                    self.timestamp_label = posssible_timestamp_label_name
                                    break                                
                            if timestamp_label_index is None:
                                #raise Exception('Cannot auto-detect timestamp column') 
                                timestamp_label_index = 0
                        else:
                            timestamp_label_index = 0
                        # TODO: Try to convert to timestamp here?
                    else:
                        if isinstance(self.timestamp_label, int):
                            timestamp_label_index = self.timestamp_label
                        else:
                            if self.timestamp_label in column_labels:
                                timestamp_label_index = column_labels.index(self.timestamp_label)
                            else:
                                if self.timestamp_label is not None:
                                    raise Exception('Cannot find requested timestamp column "{}" in labels (got "{}")'.format(self.timestamp_label, column_labels))                             
                                elif self.timestamp_label is not None:
                                    raise Exception('Cannot find requested time column "{}" in labels (got "{}")'.format(self.timestamp_label, column_labels))                             
                                else:
                                    pass
                    logger.debug('Set time column index = "%s"', timestamp_label_index)
                    
                # Set time column index if required and not already done
                if self.time_label is not None and time_label_index is None:
                    if isinstance(self.time_label, int):
                        time_label_index = self.time_label
                    else:
                        if self.time_label in column_labels:
                            time_label_index = column_labels.index(self.time_label)
                        else:
                            raise Exception('Cannot find requested time column "{}" in labels (got "{}")'.format(self.time_label, column_labels))                             
                    logger.debug('Set date column index = "%s"', time_label_index)                    
                    
                # Set date column index if required and not already done
                if self.date_label is not None and date_label_index is None:
                    if isinstance(self.date_label, int):
                        date_label_index = self.date_label
                    else:
                        if self.date_label in column_labels:
                            date_label_index = column_labels.index(self.date_label)
                        else:
                            raise Exception('Cannot find requested time column "{}" in labels (got "{}")'.format(self.time_label, column_labels))                             
                    logger.debug('Set date column index = "%s"', date_label_index)

                # If all the three of timestamp_label_index, time_label_index and date_label_index are None, there is something wrong:
                if timestamp_label_index is None and time_label_index is None and date_label_index is None:
                    raise ConsistencyException('Could not set timestamp_label_index nor time_label_index nor date_label_index, somehting wrong happened.')

                # Ok,now get the timestamp as string
                if timestamp_label_index is not None:
                    # Just use the Timestamp column
                    timestamp = sanitize_string(line_items[timestamp_label_index],NO_DATA_PLACEHOLDERS)

                else:
                    # Time part
                    time_part=None
                    if time_label_index is not None:
                        time_part = sanitize_string(line_items[time_label_index],NO_DATA_PLACEHOLDERS)
    
                    # Date part
                    date_part=None
                    if date_label_index is not None:
                        date_part = sanitize_string(line_items[date_label_index],NO_DATA_PLACEHOLDERS)
                                       
                    # Assemble timestamp
                    if time_part is not None and date_part is not None:
                        timestamp = date_part + '\t' + time_part
                    elif date_part is None:
                        timestamp = time_part
                    elif time_part is None:
                        timestamp = date_part
                    else:
                        raise ConsistencyException('Got both time_part and date_part as None, somehting wrong happened.')
                      
                # Convert timestamp to epoch seconds
                logger.debug('Will process timestamp "%s"', timestamp)
                try:
                    
                    # Both time and date labels
                    if self.time_label is not None and self.date_label is not None:
                        dt = datetime.datetime.strptime(timestamp, self.date_format + '\t' + self.time_format)
                        t = s_from_dt(dt)
                    
                    # Only date label
                    elif self.date_label is not None:
                        dt = datetime.datetime.strptime(timestamp, self.date_format)
                        t = s_from_dt(dt)                        

                    # Only time label (TODO: does this make sense?)
                    elif self.time_label is not None:
                        dt = datetime.datetime.strptime(timestamp, self.time_format)
                        t = s_from_dt(dt)    
                    
                    # Use the timestamp label and format
                    else:
                        
                        # Autodetect?
                        if self.timestamp_format == 'auto':
                            logger.debug('Auto-detecting timestamp format')
    
                            # Is this an epoch?
                            try:
                                t = float(timestamp)
                                logger.debug('Auto-detected timestamp format: epoch')
                                self.timestamp_format = 'epoch'
                            except:
                                try:
                                    # is this an iso8601?
                                    t = s_from_dt(dt_from_str(timestamp))
                                    logger.debug('Auto-detected timestamp format: iso8601')
                                    self.timestamp_format = 'iso8601'
                                except:
                                    raise Exception('Cannot auto-detect timestamp format (got "{}")'.format(timestamp)) from None
                        
                        # Epoch format?
                        elif self.timestamp_format == 'epoch':
                            t = float(timestamp)
    
                        # ISO8601 fromat?
                        elif self.timestamp_format == 'iso8601':
                            t = s_from_dt(dt_from_str(timestamp))
                        
                        # Custom format?
                        else:
                            dt = datetime.datetime.strptime(timestamp, self.timestamp_format)
                            t = s_from_dt(dt)    

                except Exception as e:
                    if self.skip_errors:
                        logger.debug('Cannot parse timestamp "%s" (%s) in line #%s, skipping the line.', timestamp, e, line_number)
                        continue
                    else:
                        raise ValueError('Cannot parse timestamp "{}" ({}) in line #%{}, aborting. Set "skip_errors=True" to drop them instead.'.format(timestamp, e, line_number)) from None

                #====================
                #  Data part
                #====================
                
                if data_label_indexes is None:
                    
                    # If the requested data labels are all o the (default) ones, use the internal value
                    if data_labels == 'all':
                        data_labels = self.data_labels
                    
                    # Do we have to select only some data columns?
                    if data_labels != 'all':
                        
                        if is_list_of_integers(data_labels):
                            # Case where we have been given a list of integers
                            data_label_indexes = data_labels
                            self.data_type = list

                        else:
                            # Case where we have been given a list of labels
                            if not column_labels:
                                raise Exception('You asked for data column labels but there are no labels in this CSV file')
                            data_label_indexes = []
                            for data_label in data_labels:
                                if data_label not in column_labels:
                                    raise Exception('Cannot find data column "{}" in CSV columns "{}"'.format(data_label, column_labels))
                                # What is the position of this requested data column in the CSV columns?
                                data_label_indexes.append(column_labels.index(data_label))

                    else:
                        # Case where we have to automatically set data column indexes to include all data columns
                        
                        # Initialize the data_label_indexes equal to the column_indexes
                        data_label_indexes = column_indexes
                        
                        # Remove timestamp, time and date indexes from the data_label_indexes
                        if timestamp_label_index is not None:
                            data_label_indexes.remove(timestamp_label_index)
                            
                        if time_label_index is not None:
                            data_label_indexes.remove(time_label_index)
                        
                        if date_label_index is not None:
                            data_label_indexes.remove(date_label_index)

                    # Ok, now based on the data_label_indexes fill the data_label_names if we have column labels
                    if column_labels:
                        data_label_names=[]
                        for data_label_index in data_label_indexes:
                            data_label_names.append(column_labels[data_label_index])
                            
                    logger.debug('Set data_label_indexes="%s" and data_label_names="%s"', data_label_indexes, data_label_names)

                # Set data type
                if not data_type:
                    if data_label_names and len(data_label_names) > 1 and self.data_type == float:
                        raise Exception('Requested data format as float but got more than 1 value')
                    if self.data_type=='auto':
                        if not data_label_names:
                            data_type = list
                        else:
                            data_type = dict
                    else:
                        data_type = self.data_type

                # Set data
                try:
                    
                    if data_type == float:
                        data = to_float(line_items[data_label_indexes[0]],NO_DATA_PLACEHOLDERS)
                    elif data_type == list:
                        data = [to_float(line_items[index],NO_DATA_PLACEHOLDERS) for index in data_label_indexes]
                    elif data_type == dict:
                        data = {column_labels[index]: to_float(line_items[index],NO_DATA_PLACEHOLDERS,column_labels[index]) for index in data_label_indexes}
                
                # TODO: here we drop the entire line. Instead, should we use a "None" and allow "Nones" in the DataPoints data? 
                except FloatConversionError as e:
                    if self.skip_errors:
                        logger.debug('Cannot convert value "%s" in line #%s to float, skipping the line.', e, line_number)
                        continue
                    else:
                        raise Exception('Cannot convert value "{}" in line #{} to float, aborting. Set "skip_errors=True" to drop them instead.'.format(e, line_number)) from None

                except IndexError as e:
                    if self.skip_errors:
                        logger.debug('Cannot parse in line #%s as some values are missing, skipping the line.', line_number)
                        continue
                    else:
                        raise Exception('Cannot parse line #{} as some values are missing, aborting. Set "skip_errors=True" to drop them instead.'.format(line_number)) from None

                logger.debug('Set data to "%s"', data)

                # Append to the series
                items.append([t, data])

                # Lastly, increase line_counter and check for limit
                line_number+=1
                if limit and line_number >= limit:
                    break

        # Were we able to read something?
        if not items:
            raise NoDataException('Cannot read any data!')

        # Sort if required
        if self.sort:
            from operator import itemgetter
            items = sorted(items, key=itemgetter(0))

        # Support var
        autodetect_series_type = False

        # Set series type
        if as_points:
            # Were we requested to generate points?
            series_type = 'points'
        elif as_slots:
            # Were we requested to generate slots?
            series_type = 'slots'
        else:
            # Do we have a series type set?     
            if self.series_type != 'auto':
                # Use the object one
                series_type = self.series_type
            else:
                # Go in auto mode
                series_type = None
            
        # Otherwise, auto-detect
        if not series_type or series_type=='slots':
    
            autodetect_series_type = True
    
            # Detect sampling interval to create right item type (points or slots)
            from .utilities import detect_sampling_interval
            
            # Add first ten elements max and then detect sampling interval.
            # Use a try-execpt as this can lead to errors due to duplicates,
            sample_timepointseries = TimePointSeries()
            for item in items:
                try:
                    sample_timepointseries.append(TimePoint(t=item[0]))
                except:
                    pass
                if len(sample_timepointseries)>10:
                    break
            detected_sampling_interval = detect_sampling_interval(sample_timepointseries)
            
            # Years
            if detected_sampling_interval in [86400*365, 886400*366]:
                detected_series_type = DataTimeSlot
                detected_unit = TimeUnit('1Y') 
                   
            # Months
            elif detected_sampling_interval in [86400*31, 86400*30, 86400*28]:
                detected_series_type = DataTimeSlot
                detected_unit = TimeUnit('1M')
            
            # Days
            elif detected_sampling_interval in [3600*24, 3600*23, 3600*25]:
                detected_series_type = DataTimeSlot
                detected_unit = TimeUnit('1D')
            
            # Weeks still to be implemented in the unit
            #elif detected_sampling_interval in [3600*24*7, (3600*24*7)-3600, (3600*24*7)+3600]:
            #    detected_series_type = DataTimeSlot
            #    detected_unit = TimeUnit('1D')
            
            # Else, use points with no unit if we were not using slots
            else:
                if series_type!='slots':
                    detected_series_type = DataTimePoint
                    detected_unit = None
                else:
                    # TODO: "detected_series_type" is not a nice name here, the
                    # code in the following should use series_type if forced..
                    detected_series_type = DataTimeSlot

                    # Can we auto-detect unit? TODO: can we standardize this? Check also in the entire codebase..
                    if detected_sampling_interval == 3600:
                        detected_unit = TimeUnit('1h')                    
                    elif detected_sampling_interval == 1800:
                        detected_unit = TimeUnit('30m')
                    elif detected_sampling_interval == 900:
                        detected_unit = TimeUnit('15m')
                    elif detected_sampling_interval == 600:
                        detected_unit = TimeUnit('10m')                                             
                    elif detected_sampling_interval == 300:
                        detected_unit = TimeUnit('5m') 
                    elif detected_sampling_interval == 60:
                        detected_unit = TimeUnit('1m') 
                    else:
                        detected_unit = TimeUnit('{}s'.format(detected_sampling_interval)) 
                        
        # Do we have to force a specific type?
        if not autodetect_series_type:
            if series_type == 'points':
                series_type = DataTimePoint
                unit = None 
            elif series_type == 'slots':
                series_type = DataTimeSlot
                unit = TimeUnit('{}s'.format(detected_sampling_interval))
            else:
                raise ValueError('Unknown value "{}" for type. Accepted types are "points" or "slots".'.format(self.series_type))
        else:
            # Log the type and unit we detected 
            if detected_series_type == DataTimeSlot:
                if series_type:
                    logger.info('Assuming {} time unit and creating Slots.'.format(detected_unit))                
                else:
                    logger.info('Assuming {} time unit and creating Slots. Use series_type=\'points\' if you want Points instead.'.format(detected_unit))
            #else:
            #    logger.info('Assuming {} sampling interval and creating {}.'.format(detected_sampling_interval, series_type.__class__.__name__))
            
            # and use it.
            series_type = detected_series_type
            unit = detected_unit

        # Set and timezonize the timezone. In this way the it will be just a pointer.
        if as_tz:
            tz = timezonize(as_tz)
        else:
            tz = timezonize(self.tz)

        # Create point or slot series
        if series_type == DataTimePoint:
            
            timeseries = DataTimePointSeries()
            
            for item in items:
                try:
                    # Handle data_indexes (including the data_loss)
                    data_indexes = None
                    if isinstance (item[1], dict):      
                        data_indexes = {}
                        for key in item[1]:
                            if key.startswith('__'):
                                data_indexes[key] = item[1][key]
                        # Remove data_indexes from item data 
                        for index in data_indexes:       
                            item[1].pop(index)
                                
                    # Create DataTimePoint, set data and data_indexes
                    data_time_point = DataTimePoint(t=item[0], data=item[1], data_indexes=data_indexes, tz=tz)

                    # Append
                    timeseries.append(data_time_point)

                except Exception as e:
                    if self.skip_errors:
                        if not self.silence_errors:
                            logger.error(e)
                    else:
                        raise e from None
        else:
            
            timeseries = DataTimeSlotSeries()
            
            for i, item in enumerate(items):
                try:
                    # Handle data_indexes (including the data_loss)
                    data_indexes = None
                    if isinstance (item[1], dict):      
                        data_indexes = {}
                        for key in item[1]:
                            if key.startswith('__'):
                                data_indexes[key] = item[1][key]
                        # Remove data_indexes from item data 
                        for index in data_indexes:       
                            item[1].pop(index)
                    
                    if DEFAULT_SLOT_DATA_LOSS is not None and 'data_loss' not in data_indexes:
                        data_indexes['data_loss'] = DEFAULT_SLOT_DATA_LOSS
                                                
                    # Create DataTimeSlot, set data and data_indexes
                    data_time_slot = DataTimeSlot(t=item[0], unit=unit, data=item[1], data_indexes=data_indexes, tz=tz)
                    
                    # Append
                    timeseries.append(data_time_slot)
                    
                except ValueError as e:
                    # The only ValueError that could (should) arise here is a "Not in succession" error.
                    missing_timestamps = []
                    prev_dt = dt_from_s(items[i-1][0], tz=tz)
                    while True:                        
                        dt = prev_dt + unit
                        # Note: the equal here is just to prevent endless loops, the check shoudl actally be just an equal     
                        if s_from_dt(dt) >= item[0]:
                            # We are arrived, append all the missing items and then the item we originally tried to and break
                            for j, missing_timestamp in enumerate(missing_timestamps):
                                # Set data by interpolation
                                interpolated_data = {data_label: (((items[i][1][data_label]-items[i-1][1][data_label])/(len(missing_timestamps)+1)) * (j+1)) + items[i-1][1][data_label]  for data_label in  items[-1][1] }
                                timeseries.append(DataTimeSlot(t=missing_timestamp, unit=unit, data=interpolated_data, data_loss=1, tz=tz))
                            timeseries.append(DataTimeSlot(t=item[0], unit=unit, data=item[1], data_loss=DEFAULT_SLOT_DATA_LOSS, tz=tz))
                            break
                        
                        else:
                            missing_timestamps.append(s_from_dt(dt))
                        prev_dt = dt

        return timeseries


    def put(self, timeseries, overwrite=False):
        
        if os.path.isfile(self.filename) and not overwrite:
            raise Exception('File already exists. use overwrite=True to overwrite.')

        # Set data_indexes here once for all or thre will be a slowndowd afterwards
        data_indexes = timeseries._all_data_indexes()

        with open(self.filename, 'w') as csv_file:
       
            # 0) Dump CSV-storage specific metadata & chck type     
            if isinstance(timeseries, DataTimePointSeries):
                csv_file.write('# Generated by Timeseria CSVFileStorage at {}\n'.format(now_dt()))
                csv_file.write('# {}\n'.format(timeseries))
                csv_file.write('# Extra parameters: {{"type": "points", "tz": "{}", "resolution": "{}"}}\n'.format(timeseries.tz, timeseries.resolution))
 
            elif isinstance(timeseries, DataTimeSlotSeries):                
                csv_file.write('# Generated by Timeseria CSVFileStorage at {}\n'.format(now_dt()))
                csv_file.write('# {}\n'.format(timeseries))
                csv_file.write('# Extra parameters: {{"type": "slots", "tz": "{}", "resolution": "{}"}}\n'.format(timeseries.tz, timeseries.resolution))

            else:
                raise TypeError('Can store only DataTimePointSeries or DataTimeSlotSeries')


            # 1) Dump headers
            data_labels_part = ','.join([str(key) for key in timeseries.data_labels()])
            data_indexes_part = ','.join(['__'+index for index in data_indexes])
            if data_indexes_part:
                csv_file.write('epoch,{},{}\n'.format(data_labels_part,data_indexes_part))
            else:
                csv_file.write('epoch,{}\n'.format(data_labels_part))
            

            # 2) Dump data (and data_indexes)
            for item in timeseries:
                data_part = ','.join([str(item.data[key]) for key in timeseries.data_labels()])
                data_indexes_part = ','.join([str(getattr(item, index)) for index in data_indexes])
                if data_indexes_part:
                    csv_file.write('{},{},{}\n'.format(item.t, data_part, data_indexes_part))
                else:
                    csv_file.write('{},{}\n'.format(item.t, data_part))
                    







