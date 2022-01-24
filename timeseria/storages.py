# -*- coding: utf-8 -*-
"""Data storages."""

import os
import datetime
from .utilities import detect_encoding, sanitize_string, is_list_of_integers, to_float
from .units import TimeUnit
from .datastructures import DataTimePoint, DataTimePointSeries, TimePointSeries, TimePoint, DataTimeSlot, DataTimeSlotSeries
from .time import dt_from_str, dt_from_s, s_from_dt, timezonize, now_dt
from .exceptions import NoDataException, FloatConversionError


# Setup logging
import logging
logger = logging.getLogger(__name__)

HARD_DEBUG = False

POSSIBLE_TIMESTAMP_COLUMNS = ['timestamp', 'epoch']
NO_DATA_PLACEHOLDERS = ['na', 'nan', 'null', 'nd', 'undef']


#======================
#  CSV File Storage
#======================

class CSVFileStorage(object):
    """A CSV file data storage."""
    
    def __init__(self, filename, timestamp_column='auto', timestamp_format='auto',
                 time_column=None, time_format=None, date_column=None, date_format=None,
                 tz='UTC', data_columns='all', data_type='auto', series_type='auto', sort=False,
                 separator=',', newline='\n', encoding='auto', skip_errors=False):
        """Ref to https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes
        for timestamp_format, date_format and time_format"""


        # Handle setting a time column
        if time_column or date_column:
            if timestamp_column == 'auto':
                timestamp_column = None

        # Sanity checks
        if timestamp_column is None and time_column is None and date_column is None and not data_columns:
            raise ValueError('No timestamp column, time column, date column or data columns provided, cannot get anything from this CSV file')

        if timestamp_column and (time_column or date_column):
            raise ValueError('Got both timestamp column and time/date columns, choose which approach to use.')

        if timestamp_column is not None and not isinstance(timestamp_column, int) and not isinstance(timestamp_column, str):
            raise ValueError('timestamp_column argument must be either integer or string (got "{}")'.format(timestamp_column.__class__.__name__))

        if time_column is not None and not isinstance(time_column, int) and not isinstance(time_column, str):
            raise ValueError('time_column argument must be either integer or string (got "{}")'.format(time_column.__class__.__name__))

        if date_column is not None and not isinstance(date_column, int) and not isinstance(date_column, str):
            raise ValueError('date_column argument must be either integer or string (got "{}")'.format(date_column.__class__.__name__))

        if data_columns != 'all' and not isinstance(data_columns, list):
            raise ValueError('data_columns argument must be a list (got "{}")'.format(data_columns.__class__.__name__))

        # File & encoding
        self.filename = filename
        if encoding == 'auto':
            self.encoding = None
        else:
            self.encoding = encoding

        # Time
        if timestamp_column:
            # We use "time" also for the timestamp. TODO: unroll this?
            self.time_column = timestamp_column
            self.time_format = timestamp_format  
        else:
            self.time_column = time_column
            self.time_format = time_format
        self.date_column = date_column
        self.date_format = date_format 
        
        # Time zone
        self.tz = tz   
        
        # Data
        self.data_columns = data_columns
        
        # Separator and newline chars
        self.separator = separator
        self.newline = newline

        # Other
        self.skip_errors = skip_errors
        self.data_type = data_type
        self.sort = sort
        
        # Set series type (that will be forced)
        self.series_type = series_type


    def get(self, limit=None, as_tz=None, as_series_type=None):

        # Line counter
        line_number=0
        
        # Init column indexes and labels
        column_indexes = None
        column_labels = None
        
        time_column_index = None
        date_column_index = None
        
        data_column_indexes = None
        data_column_labels = None

        # Detect encoding if not set
        if not self.encoding:
            self.encoding = detect_encoding(self.filename, streaming=False)

        items = []

        # TODO: evaluate rU vs newline='\n'
        with open(self.filename, 'r', encoding=self.encoding) as csv_file:

            while True:
                
                # Increase line_counter and check for limit
                line_number+=1
                if limit and line_number > limit:
                    break
                
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
                comment_chars = ['#', ';'] 
                if line[0] in comment_chars:
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

                # Set time column index if not already done
                if time_column_index is None:
                    if self.time_column=='auto':
                        # TODO: improve this auto-detect: try different column names and formats.
                        #       Maybe, fix position as the first element for the timestamp.
                        if column_labels:
                            for posssible_time_column_name in POSSIBLE_TIMESTAMP_COLUMNS:
                                if posssible_time_column_name in column_labels:
                                    time_column_index = column_labels.index(posssible_time_column_name)
                                    self.time_column = posssible_time_column_name
                                    break                                
                            if time_column_index is None:
                                #raise Exception('Cannot auto-detect timestamp column') 
                                time_column_index = 0
                        else:
                            time_column_index = 0
                        # TODO: Try to convert to timestamp here?
                    else:
                        if isinstance(self.time_column, int):
                            time_column_index = self.time_column
                        else:
                            if self.time_column in column_labels:
                                time_column_index = column_labels.index(self.time_column)
                            else:
                                raise Exception('Cannot find requested time column "{}" in labels (got "{}")'.format(self.time_column, column_labels))                             
                    logger.debug('Set time column index = "%s"', time_column_index)

                # Set date column index if required and not already done
                if self.date_column is not None and date_column_index is None:
                    if isinstance(self.date_column, int):
                        date_column_index = self.date_column
                    else:
                        if self.date_column in column_labels:
                            date_column_index = column_labels.index(self.date_column)
                        else:
                            raise Exception('Cannot find requested time column "{}" in labels (got "{}")'.format(self.time_column, column_labels))                             
                    logger.debug('Set date column index = "%s"', date_column_index)


                # Time part
                time_part=''
                if time_column_index is not None:
                    time_part = sanitize_string(line_items[time_column_index],NO_DATA_PLACEHOLDERS)


                # Date part
                date_part=''
                if date_column_index is not None:
                    date_part = sanitize_string(line_items[date_column_index],NO_DATA_PLACEHOLDERS)
                                   
                # Assemble timestamp
                if self.date_column is not None:
                    timestamp = date_part + '\t' + time_part
                else:
                    timestamp = time_part
                  
                # Convert timestamp to epoch seconds
                logger.debug('Will process timestamp "%s"', timestamp)
                try:
                    
                    if self.time_format == 'auto':
                        logger.debug('Auto-detecting timestamp format')

                        # Is this an epoch?
                        try:
                            t = float(timestamp)
                            logger.debug('Auto-detected timestamp format: epoch')
                            self.time_format = 'epoch'
                        except:
                            try:
                                # is this an iso8601?
                                t = s_from_dt(dt_from_str(timestamp))
                                logger.debug('Auto-detected timestamp format: iso8601')
                                self.time_format = 'iso8601'
                            except:
                                raise Exception('Cannot auto-detect timestamp format (got "{}")'.format(timestamp)) from None
                    
                    elif self.time_format == 'epoch':
                        t = float(timestamp)

                    elif self.time_format == 'iso8601':
                        t = s_from_dt(dt_from_str(timestamp))
                    
                    else:
                        if self.date_column is not None:
                            dt = datetime.datetime.strptime(timestamp, self.date_format + '\t' + self.time_format)
                            t = s_from_dt(dt)
                        else:
                            dt = datetime.datetime.strptime(timestamp, self.time_format)
                            t = s_from_dt(dt)
    
                        if not dt.tzinfo:
                            # TODO: we should not do this check here, but above.
                            # Also maybe is not necessary at all if s_from_dt accepts timezone-unaware datetimes.
                            import pytz
                            dt = pytz.timezone('UTC').localize(dt)
                            #t = s_from_dt(dt)
                except Exception as e:
                    if self.skip_errors:
                        logger.debug('Cannot parse timestamp "%s" (%s) in line #%s, skipping the line.', timestamp, e, line_number)
                        continue
                    else:
                        raise Exception('Cannot parse timestamp "{}" ({}) in line #%{}, aborting. Set "skip_errors=True" to drop them instead.'.format(timestamp, e, line_number)) from None

                #====================
                #  Data part
                #====================
                
                if data_column_indexes is None:
                    
                    # Do we have to select only some data columns?
                    if self.data_columns != 'all':
                        
                        if is_list_of_integers(self.data_columns):
                            # Case where we have been given a list of integers
                            data_column_indexes =  self.data_columns
                            self.data_type = list

                        else:
                            # Case where we have been given a list of labels
                            if not column_labels:
                                raise Exception('You asked for data column labels but there are no labels in this CSV file')
                            data_column_indexes = []
                            for data_column in self.data_columns:
                                if data_column not in column_labels:
                                    raise Exception('Cannot find data column "{}" in CSV columns "{}"'.format(data_column, column_labels))
                                # What is the position of this requested data column in the CSV columns?
                                data_column_indexes.append(column_labels.index(data_column))

                    else:
                        # Case where we have to automatically set data column indexes to include all data columns
                        
                        # Initialize the data_column_indexes equal to the column_indexes
                        data_column_indexes = column_indexes
                        
                        # Remove time and date indexes from the data_column_indexes
                        if time_column_index is not None:
                            data_column_indexes.remove(time_column_index)
                        
                        if date_column_index is not None:
                            data_column_indexes.remove(date_column_index)

                    # Ok, now based on the data_column_indexes fill the data_column_labels if we have column labels
                    if column_labels:
                        data_column_labels=[]
                        for data_column_index in data_column_indexes:
                            data_column_labels.append(column_labels[data_column_index])
                            
                    logger.debug('Set data_column_indexes="%s" and data_column_labels="%s"', data_column_indexes, data_column_labels)


                # Set data
                try:
                    if data_column_labels and len(data_column_labels) > 1 and self.data_type == float:
                        raise Exception('Requested data format as float but got more than 1 value')
                    if self.data_type == float:
                        data = to_float(line_items[data_column_indexes[0]],NO_DATA_PLACEHOLDERS)
                    elif self.data_type == list or not data_column_labels:
                        data = [to_float(line_items[index],NO_DATA_PLACEHOLDERS) for index in data_column_indexes]
                    else:
                        data = {column_labels[index]: to_float(line_items[index],NO_DATA_PLACEHOLDERS,column_labels[index]) for index in data_column_indexes}
               
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
        if not as_series_type and self.series_type != 'auto':
            # Use the object one
            series_type = self.series_type
        elif as_series_type != 'auto':
            # Use this one
            series_type = as_series_type
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
                    data_loss = None
                    indexes = None
                    if isinstance (item[1], dict):
                        # Handle indexes and special data loss case
                        indexes = {}
                        for key in item[1]:
                            if key.startswith('__'):
                                if key=='__data_loss':
                                    data_loss = item[1][key]
                                else:
                                    indexes[key] = item[1][key]
                                    
                        # Remove indexes and data loss from item data
                        for index in indexes:       
                            item[1].pop(index)
                        item[1].pop('__data_loss',None)   
                                
                    # Create DataTimePoint, set data loss and set indexes
                    data_time_point = DataTimePoint(t=item[0], data=item[1], data_loss=data_loss, tz=tz)
                    if indexes:
                        for index in indexes: 
                            # Set index. The [2:] removes the two trailing underscores
                            setattr(data_time_point, index[2:], indexes[index])
                            
                    # Append
                    timeseries.append(data_time_point)

                except Exception as e:
                    if self.skip_errors:
                        logger.error(e)
                    else:
                        raise e from None
        else:
            
            timeseries = DataTimeSlotSeries()
            
            for i, item in enumerate(items):
                try:
                    data_loss = 0
                    if isinstance (item[1], dict):
                        # Handle indexes and special data loss case
                        indexes = {}
                        for key in item[1]:
                            if key.startswith('__'):
                                if key=='__data_loss':
                                    data_loss = item[1][key]
                                else:
                                    indexes[key] = item[1][key]
                                    
                        # Remove indexes and data loss from item data
                        for index in indexes:       
                            item[1].pop(index)
                        item[1].pop('__data_loss',None)     
                            
                    # Create DataTimeSlot, set data loss and set indexes
                    data_time_slot = DataTimeSlot(t=item[0], unit=unit, data=item[1], data_loss=data_loss, tz=tz)
                    if indexes:
                        for index in indexes: 
                            # Set index. The [2:] removes the two trailing underscores
                            setattr(data_time_slot, index[2:], indexes[index])
                            
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
                                interpolated_data = {data_key: (((items[i][1][data_key]-items[i-1][1][data_key])/(len(missing_timestamps)+1)) * (j+1)) + items[i-1][1][data_key]  for data_key in  items[-1][1] }
                                timeseries.append(DataTimeSlot(t=missing_timestamp, unit=unit, data=interpolated_data, data_loss=1, tz=tz))
                            timeseries.append(DataTimeSlot(t=item[0], unit=unit, data=item[1], data_loss=0, tz=tz))
                            break
                        
                        else:
                            missing_timestamps.append(s_from_dt(dt))
                        prev_dt = dt

        return timeseries


    def put(self, timeseries, overwrite=False):
        
        if os.path.isfile(self.filename) and not overwrite:
            raise Exception('File already exists. use overwrite=True to overwrite.')

        # Set indexes here once for all or thre will be a slowndowd afterwards
        indexes = timeseries.indexes

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
            data_keys_part = ','.join([str(key) for key in timeseries.data_keys()])
            indexes_part = ','.join(['__'+index for index in indexes])
            if indexes_part:
                csv_file.write('epoch,{},{}\n'.format(data_keys_part,indexes_part))
            else:
                csv_file.write('epoch,{}\n'.format(data_keys_part))
            

            # 2) Dump data (and indexes)
            for item in timeseries:
                data_part = ','.join([str(item.data[key]) for key in timeseries.data_keys()])
                indexes_part = ','.join([str(getattr(item, index)) for index in indexes])
                if indexes_part:
                    csv_file.write('{},{},{}\n'.format(item.t, data_part, indexes_part))
                else:
                    csv_file.write('{},{}\n'.format(item.t, data_part))
                    







