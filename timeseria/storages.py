import csv
import re
from .common import detect_encoding
from .datastructures import DataTimePoint, DataTimePointSerie, DataPointSerie, TimePointSerie, TimePoint, DataPoint
from .time import s_from_dt, dt_from_str
import datetime
from collections import OrderedDict

# Setup logging
import logging
logger = logging.getLogger(__name__)

HARD_DEBUG = False

POSSIBLE_TIMESTAMP_COLUMNS = ['timestamp', 'epoch']
NO_DATA_PLACEHOLDERS = ['na', 'nan', 'null', 'nd', 'undef']

class FloatConversionError(Exception):
    pass

def sanitize(value):
    value = re.sub('\s+',' ',value).strip()
    if value.startswith('\'') or value.startswith('"'):
        value = value[1:]
    if value.endswith('\'') or value.endswith('"'):
        value = value[:-1]
    value = value.strip()
    if value.lower().replace('.','') in NO_DATA_PLACEHOLDERS:
        return None
    return value

def is_list_of_integers(list):
    for item in list:
        if not isinstance(item, int):
            return False
    else:
        return True

def to_float(string):
    sanitized_string = sanitize(string)
    if sanitized_string:
        sanitized_string = sanitized_string.replace(',','.')
    try:
        return float(sanitized_string)
    except (ValueError, TypeError):
        raise FloatConversionError(sanitized_string)

#======================
#  CSV File Storage
#======================

class CSVFileStorage(object):
    
    def __init__(self, filename_with_path, encoding = 'auto', time_column = 'auto', time_format = 'auto',
                 date_column = None, date_format = None, data_columns = 'all', value_separator=',', line_separator='\n',
                 skip_errors=False, data_format='auto'):
        ''' Ref to https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes
        for date_format and time_format'''
        
        # File & encoding
        self.filename_with_path = filename_with_path
        if encoding == 'auto':
            self.encoding = None
        else:
            self.encoding = encoding

        # Time
        self.time_column = time_column
        self.time_format = time_format
        self.date_column = date_column
        self.date_format = date_format    
        
        # Data
        self.data_columns = data_columns
        
        # Value and line separators
        self.value_separator = value_separator
        self.line_separator = line_separator

        # Other
        self.skip_errors = skip_errors
        self.data_format = data_format

        # Check
        if self.time_column is None and self.date_column is None and not self.data_columns:
            raise ValueError('No time column, date column or data columns provided, cannot get anyything fromt this CSV file')
        
        if self.time_column is not None and not isinstance(self.time_column, int) and not isinstance(self.time_column, str):
            raise ValueError('time_column argument must be either integer os string (got "{}")'.format(self.time_column.__class__.__name__))

        if self.date_column is not None and not isinstance(self.date_column, int) and not isinstance(self.date_column, str):
            raise ValueError('date_column argument must be either integer os string (got "{}")'.format(self.date_column.__class__.__name__))

        if self.data_columns != 'all' and not isinstance(self.data_columns, list):
            raise ValueError('data_columns argument must be a list (got "{}")'.format(self.data_columns.__class__.__name__))


    def get(self, limit=None):

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
            self.encofing = detect_encoding(self.filename_with_path, streaming=False)

        dataTimePointSerie = DataTimePointSerie()

        # TODO: evaluate rU vs newline='\n'
        with open(self.filename_with_path, 'r', encoding=self.encoding) as csv_file:

            while True:
                
                # Increase line_counter and check for limit
                line_number+=1
                if limit and line_number > limit:
                    break
                
                # TODO: replace me using a custom line separator
                line = csv_file.readline()
                logger.debug('Processing line #%s: "%s"', line_number, sanitize(line))

                # Do we have to stop? Note: empty lines still have the "\n" char.
                if not line:
                    break
                                
                # Is this line a comment? 
                comment_chars = ['#', ';'] 
                if line[0] in comment_chars:
                    logger.debug('Skipping line "%s" as marked as comment line', line)
                    continue

                # Is this line something else? 
                if not self.value_separator in line:
                    logger.debug('Skipping line "%s" as marked no value separator found', line)
                    continue

                # Get values from this line:
                line_items = line.split(self.value_separator)

                # Set column keys if not already done
                if not column_indexes:
                    
                    # Is this line carrying non-numerical or non timestamp data?
                    not_converted = 0
                    for value in line_items:
                        try:
                            float(sanitize(value))
                        except:
                            try:
                                dt_from_str(sanitize(value))
                            except:
                                not_converted += 1
          
                    # If it is, use it as labels (and continue with the next line)
                    if not_converted == len(line_items):
                        column_indexes = [i for i in range(len(line_items)) if sanitize(line_items[i])]
                        column_labels = [sanitize(label) for label in line_items if sanitize(label)]
                        logger.debug('Set column indexes = "%s"  and column labels = "%s"', column_indexes, column_labels)
                        continue
                    
                    # Otherwise, just use inteher keys (and go on)
                    else:
                        column_indexes = [i for i in range(len(line_items)) if sanitize(line_items[i])]
                        logger.debug('Set column indexes = "%s"  and column labels = "%s"', column_indexes, column_labels)

                    # Note: above we only used labels and indexes that carry content after being sanitized.
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
                                raise Exception('Cannot auto-detect timestamp column') 
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
                    time_part = sanitize(line_items[time_column_index])


                # Date part
                date_part=''
                if date_column_index is not None:
                    date_part = sanitize(line_items[date_column_index])
                                   
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
                        logger.debug('Cannot parse timestamp "%s" in line #%s, skipping the line.', e, line_number)
                        continue
                    else:
                        raise Exception('Cannot parse timestamp "{}" in line #%{}, aborting. Set "skip_errors=True" to drop them instead.'.format(e, line_number)) from None

                #====================
                #  Data part
                #====================
                
                if data_column_indexes is None:
                    
                    # Do we have to select only some data columns?
                    if self.data_columns != 'all':
                        
                        if is_list_of_integers(self.data_columns):
                            # Case where we have been given a list of integers
                            data_column_indexes =  self.data_columns
                            self.data_format = list

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
                    if data_column_labels:
                        if len(data_column_labels) >1 and self.data_format == float:
                            raise Exception('Requested data format as float but got more than 1 value')
                        if self.data_format == float:
                            data = to_float(line_items[data_column_indexes[0]])
                        elif self.data_format == list:
                            data = [to_float(line_items[index]) for index in data_column_indexes]
                        else:
                            data = {column_labels[index]: to_float(line_items[index]) for index in data_column_indexes}
                    else:
                        # Default here is to set a float if there is only one value.
                        # TODO: are we sure we want this?
                        if len (data_column_indexes) >1 or self.data_format == list:
                            data = [to_float(line_items[index]) for index in data_column_indexes]
                        else:
                            data = to_float(line_items[data_column_indexes[0]])
           
                # TODO: here we drop the entire line. Instead, should we use a "None" and allow "Nones" in the DataPoints data? 
                except FloatConversionError as e:
                    if self.skip_errors:
                        logger.debug('Cannot convert value "%s" in line #%s to float, skipping the line.', e, line_number)
                        continue
                    else:
                        raise Exception('Cannot convert value "{}" in line #%{} to float, aborting. Set "skip_errors=True" to drop them instead.'.format(e, line_number)) from None

                except IndexError as e:
                    if self.skip_errors:
                        logger.debug('Cannot parse in line #%s as some values are missing, skipping the line.', line_number)
                        continue
                    else:
                        raise Exception('Cannot parse line #%{} as some values are missing, aborting. Set "skip_errors=True" to drop them instead.'.format(line_number)) from None

                logger.debug('Set data to "%s"', data)

                # Append to the series
                dataTimePointSerie.append(DataTimePoint(t=t, data=data))
                    
        return dataTimePointSerie









