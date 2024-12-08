# -*- coding: utf-8 -*-
"""Data storages, as the CSV file storage."""

import os
import pytz
import datetime
from .utils import detect_encoding, _sanitize_string, _is_list_of_integers, _to_float
from .units import TimeUnit
from .datastructures import TimePoint, DataTimePoint, DataTimeSlot, TimeSeries
from propertime.utils import dt_from_str, dt_from_s, s_from_dt, timezonize, now_dt
from .exceptions import NoDataException, FloatConversionError, ConsistencyException

# Setup logging
import logging
logger = logging.getLogger(__name__)

POSSIBLE_TIMESTAMP_labelS = ['timestamp', 'epoch']
NO_DATA_PLACEHOLDERS = ['na', 'nan', 'null', 'nd', 'undef']
DEFAULT_SLOT_DATA_LOSS = None


#======================
#  Base Storage class
#======================

class Storage(object):
    """The base storage class. Can be implemented to store one or more series. If storing more than one,
    then the id of which series to load or store must be provided in the ``get()`` and ``put()`` methods."""

    def get(self, id=None, start=None, end=None, *args, **kwargs):
        raise NotImplementedError()

    def put(self, series, id=None, *args, **kwargs):
        raise NotImplementedError()


#======================
#  CSV File Storage
#======================

class CSVFileStorage(Storage):
    """A CSV file storage. Supports both point and slot time series.

    The file encoding, the series type and the timestamp columns are all auto-detect with an heuristic approach
    by default, and asked to be set manually only if the heuristics fails. In particular, whether to create point or slot
    series is based on the sampling interval automatically detected: if this is above 24 hours, then slots are used.

    The header with the column labels is optional, and if not present the column numbers are used as labels.
    Comments in the CSV file are supported Comments in the CSV file only as full-line comments, where the line starts
    with one of the characters listed as comment character (``#`` and ``;`` by default).

    The ``timestamp_format``, ``date_format`` and ``time_format`` arguments can be set using Python strptime format codes
    (https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes), but the ``timestamp_format``
    also supports two special formats:

        * ``epoch``: the epoch timestamp, intended as the number of seconds from the 1st January 1970 UTC with decimals for sub-second precision.
        * ``iso8601``: the ISO 8601 timestamp format (see https://www.w3.org/TR/NOTE-datetime)

    Data can be simply comma-separated (or using a custom separator) as well as single or double quoted, check out the RFC4180
    (https://datatracker.ietf.org/doc/html/rfc4180) for more details on the CSV file format specification.

    Args:
        filename: the file name (including its path).
        timestamp_column: the column label to be used as timestamp. Either use this or the time_column and/or date_column parameters.
        timestamp_format: the timestamp column format.
        time_column: the column label to be used as the time part of the timestamp.
        time_format: the time column format.
        date_column: the column label to be used as the date part of the timestamp.
        date_format: the date column format.
        tz: the timezone on which to create the time series on. If the timestamps in the file are naive, then they are assumed on
            such timezone. If they are instead offset-aware, including UTC, then they are just moved on the given timezone.
        data_labels: the column labels to be used as data by default. Excpected as a list of
                     strings or list of integers, in which case are treated as column numbers.
        data_type: the data type (``list`` or ``dict``), set automatically by default.
        series_type: the default type of the series, if ``points`` or ``slots``. Set automatically by default.
        sort: if to sort the data before creating the series.
        separator: the separator for the records (fields), ``,`` by default.
        newline: the newline character, ``\\n`` by default.
        comment_chars: the characters used to mark a comment line. Defaulted to ``#`` and ``;``.
        encoding: the encoding of the file, set automatically by default.
        skip_errors: if to skip errors or raise an exception.
        silence_errors: if to completely silence errors when skipping them or not.
    """

    def __init__(self, filename, timestamp_column='auto', timestamp_format='auto',
                 time_column=None, time_format=None, date_column=None, date_format=None,
                 tz='UTC', data_labels='all', data_type='auto', series_type='auto', sort=False,
                 separator=',', newline='\n', comment_chars = ['#', ';'],  encoding='auto',
                 skip_errors=False, silence_errors=False):

        # Parameters sanity checks and adjustments
        if timestamp_column is None and time_column is None and date_column is None and not data_labels:
            raise ValueError('No timestamp column, time column, date column or data columns provided, cannot get anything from this CSV file')

        if timestamp_column != 'auto' and (time_column or date_column):
            raise ValueError('Got both timestamp column and time/date columns, choose which approach to use.')

        if timestamp_column is not None and not isinstance(timestamp_column, int) and not isinstance(timestamp_column, str):
            raise ValueError('timestamp_column argument must be a string (or integer for coulmn number) (got "{}")'.format(timestamp_column.__class__.__name__))

        if time_column is not None:
            if not isinstance(time_column, int) and not isinstance(time_column, str):
                raise ValueError('time_column argument must be a string (or integer for coulmn number) (got "{}")'.format(time_column.__class__.__name__))
            if not time_format:
                raise ValueError('If giving a time_column, a time_format is required as well')

            # Disble the timestamp label if a time (or date) label is set
            timestamp_column = None
            timestamp_format = None

        if date_column is not None:
            if not isinstance(date_column, int) and not isinstance(date_column, str):
                raise ValueError('date_column argument must be a string (or integer for coulmn number) (got "{}")'.format(date_column.__class__.__name__))
            if not date_format:
                raise ValueError('If giving a date_column, a date_format is required as well')

            # Disble the timestamp label if a date (or time) label is set
            timestamp_column = None
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
        self.timestamp_column = timestamp_column
        self.timestamp_format = timestamp_format

        self.date_column = date_column
        self.date_format = date_format

        self.time_column = time_column
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


    def get(self, id=None, start=None, end=None, limit=None, filter_data_labels=[],
            force_tz=None, force_points=False, force_slots=False, force_slot_unit=None):
        """Load the time series from the CSV file.

        Args:
            id: Not implemented for this storage.
            start: Not implemented for this storage.
            end: Not implemented for this storage.
            limit: a row number limit.
            filter_data_labels: get only specific data labels.
            force_tz: force a specific timezone.
            force_points: force generating points.
            force_slots: force generating slots.
            force_slot_unit: set the unit of the slots.

        Returns:
            TimeSeries: the time series loaded from the CSV file.
        """

        if id:
            raise NotImplementedError('This storage does not support multiple time series, so the id argument cannot be used')

        if start or end:
            raise NotImplementedError('This storage does not support loading only a portion of the time time series, so the start and end arguments cannot be used')

        # Sanity checks
        if force_points and force_slots:
            raise ValueError('Got both as_points and as_slots, set only one or none')

        # Line counter
        line_number=0

        # Init column indexes and labels
        column_indexes = None
        column_labels = None

        timestamp_column_index = None
        time_column_index = None
        date_column_index = None

        data_label_indexes = None
        data_label_names = None

        # Data type
        data_type = None

        # Detect encoding if not set
        if not self.encoding:
            self.encoding = detect_encoding(self.filename, streaming=False)

        # Set and timezonize the timezone. Using this in the points or slots will be just a pointer
        if force_tz:
            tz = timezonize(force_tz)
        else:
            tz = timezonize(self.tz)

        # Loop over all CSV rows
        # TODO: evaluate rU vs newline='\n'
        items = []
        with open(self.filename, 'r', encoding=self.encoding) as csv_file:

            while True:

                # TODO: replace me using a custom line separator
                line = csv_file.readline()
                logger.debug('Processing line #%s: "%s"', line_number, _sanitize_string(line,NO_DATA_PLACEHOLDERS))

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
                            float(_sanitize_string(value,NO_DATA_PLACEHOLDERS))
                        except:
                            try:
                                dt_from_str(_sanitize_string(value,NO_DATA_PLACEHOLDERS))
                            except:
                                not_converted += 1

                    # If it is, use it as labels (and continue with the next line)
                    if not_converted: # == len(line_items):
                        column_indexes = [i for i in range(len(line_items)) if _sanitize_string(line_items[i],NO_DATA_PLACEHOLDERS)]
                        column_labels = [_sanitize_string(label,NO_DATA_PLACEHOLDERS) for label in line_items if _sanitize_string(label,NO_DATA_PLACEHOLDERS)]
                        logger.debug('Set column indexes = "%s"  and column labels = "%s"', column_indexes, column_labels)
                        continue

                    # Otherwise, just use integer keys (and go on)
                    else:
                        column_indexes = [i for i in range(len(line_items)) if _sanitize_string(line_items[i],NO_DATA_PLACEHOLDERS)]
                        logger.debug('Set column indexes = "%s"  and column labels = "%s"', column_indexes, column_labels)

                    # Note: above we only used labels and indexes that carry content after being sanitize_stringd.
                    # TODO: what if a file starts with a line missing last value and we do or not do set data columns from the outside?
                    #       .. external data columns should have precedenze, but a mapping is then required more than just a list.

                #====================
                #  Time part
                #====================

                # Set timestamp column index if not already done
                if self.timestamp_column is not None and timestamp_column_index is None:
                    if self.timestamp_column=='auto':
                        # TODO: improve this auto-detect: try different column names and formats.
                        #       Maybe, fix position as the first element for the timestamp.
                        if column_labels:
                            for posssible_timestamp_column_name in POSSIBLE_TIMESTAMP_labelS:
                                if posssible_timestamp_column_name in column_labels:
                                    timestamp_column_index = column_labels.index(posssible_timestamp_column_name)
                                    self.timestamp_column = posssible_timestamp_column_name
                                    break
                            if timestamp_column_index is None:
                                #raise Exception('Cannot auto-detect timestamp column')
                                timestamp_column_index = 0
                        else:
                            timestamp_column_index = 0
                        # TODO: Try to convert to timestamp here?
                    else:
                        if isinstance(self.timestamp_column, int):
                            timestamp_column_index = self.timestamp_column
                        else:
                            if self.timestamp_column in column_labels:
                                timestamp_column_index = column_labels.index(self.timestamp_column)
                            else:
                                if self.timestamp_column is not None:
                                    raise Exception('Cannot find requested timestamp column "{}" in labels (got "{}")'.format(self.timestamp_column, column_labels))
                                elif self.timestamp_column is not None:
                                    raise Exception('Cannot find requested time column "{}" in labels (got "{}")'.format(self.timestamp_column, column_labels))
                                else:
                                    pass
                    logger.debug('Set time column index = "%s"', timestamp_column_index)

                # Set time column index if required and not already done
                if self.time_column is not None and time_column_index is None:
                    if isinstance(self.time_column, int):
                        time_column_index = self.time_column
                    else:
                        if self.time_column in column_labels:
                            time_column_index = column_labels.index(self.time_column)
                        else:
                            raise Exception('Cannot find requested time column "{}" in labels (got "{}")'.format(self.time_column, column_labels))
                    logger.debug('Set date column index = "%s"', time_column_index)

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

                # If all the three of timestamp_column_index, time_column_index and date_column_index are None, there is something wrong:
                if timestamp_column_index is None and time_column_index is None and date_column_index is None:
                    raise ConsistencyException('Could not set timestamp_column_index nor time_column_index nor date_column_index, somehting wrong happened.')

                # Ok,now get the timestamp as string
                if timestamp_column_index is not None:
                    # Just use the Timestamp column
                    timestamp = _sanitize_string(line_items[timestamp_column_index],NO_DATA_PLACEHOLDERS)

                else:
                    # Time part
                    time_part=None
                    if time_column_index is not None:
                        time_part = _sanitize_string(line_items[time_column_index],NO_DATA_PLACEHOLDERS)

                    # Date part
                    date_part=None
                    if date_column_index is not None:
                        date_part = _sanitize_string(line_items[date_column_index],NO_DATA_PLACEHOLDERS)

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

                    t = None
                    dt = None

                    # Both time and date labels
                    if self.time_column is not None and self.date_column is not None:
                        dt = datetime.datetime.strptime(timestamp, self.date_format + '\t' + self.time_format)

                    # Only date label
                    elif self.date_column is not None:
                        dt = datetime.datetime.strptime(timestamp, self.date_format)

                    # Only time label (TODO: does this make sense?)
                    elif self.time_column is not None:
                        dt = datetime.datetime.strptime(timestamp, self.time_format)

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
                                    dt = dt_from_str(timestamp)
                                    logger.debug('Auto-detected timestamp format: iso8601')
                                    self.timestamp_format = 'iso8601'
                                except:
                                    raise Exception('Cannot auto-detect timestamp format (got "{}")'.format(timestamp)) from None

                        # Epoch format?
                        elif self.timestamp_format == 'epoch':
                            t = float(timestamp)

                        # ISO8601 format?
                        elif self.timestamp_format == 'iso8601':
                            dt = dt_from_str(timestamp)

                        # Custom format?
                        else:
                            dt = datetime.datetime.strptime(timestamp, self.timestamp_format)


                    # Convert to t
                    if t is None:
                        if dt.tzinfo is None:
                            dt = tz.localize(dt)
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
                    if not filter_data_labels:
                        data_labels = self.data_labels
                    else:
                        data_labels = filter_data_labels

                    # Do we have to select only some data columns?
                    if data_labels != 'all':

                        if _is_list_of_integers(data_labels):
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
                        if timestamp_column_index is not None:
                            data_label_indexes.remove(timestamp_column_index)

                        if time_column_index is not None:
                            data_label_indexes.remove(time_column_index)

                        if date_column_index is not None:
                            data_label_indexes.remove(date_column_index)

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
                        data = _to_float(line_items[data_label_indexes[0]],NO_DATA_PLACEHOLDERS)
                    elif data_type == list:
                        data = [_to_float(line_items[index],NO_DATA_PLACEHOLDERS) for index in data_label_indexes]
                    elif data_type == dict:
                        data = {column_labels[index]: _to_float(line_items[index],NO_DATA_PLACEHOLDERS,column_labels[index]) for index in data_label_indexes}

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
        if force_points:
            # Were we requested to generate points?
            series_type = 'points'
        elif force_slots:
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
            from .utils import detect_sampling_interval

            # Add first ten elements max and then detect sampling interval.
            # Use a try-execpt as this can lead to errors due to duplicates,
            sample_timepointseries = TimeSeries()
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
                auto_slot_unit = TimeUnit('1Y')

            # Months
            elif detected_sampling_interval in [86400*31, 86400*30, 86400*28]:
                detected_series_type = DataTimeSlot
                auto_slot_unit = TimeUnit('1M')

            # Days
            elif detected_sampling_interval in [3600*24, 3600*23, 3600*25]:
                detected_series_type = DataTimeSlot
                auto_slot_unit = TimeUnit('1D')

            # Weeks still to be implemented in the unit
            #elif detected_sampling_interval in [3600*24*7, (3600*24*7)-3600, (3600*24*7)+3600]:
            #    detected_series_type = DataTimeSlot
            #    auto_slot_unit = TimeUnit('1D')

            # Else, use points with no unit if we were not using slots
            else:
                if series_type!='slots':
                    detected_series_type = DataTimePoint
                    auto_slot_unit = None
                else:
                    # TODO: "detected_series_type" is not a nice name here, the
                    # code in the following should use series_type if forced..
                    detected_series_type = DataTimeSlot

                    # Can we auto-detect unit? TODO: can we standardize this? Check also in the entire codebase..
                    if detected_sampling_interval == 3600:
                        auto_slot_unit = TimeUnit('1h')
                    elif detected_sampling_interval == 1800:
                        auto_slot_unit = TimeUnit('30m')
                    elif detected_sampling_interval == 900:
                        auto_slot_unit = TimeUnit('15m')
                    elif detected_sampling_interval == 600:
                        auto_slot_unit = TimeUnit('10m')
                    elif detected_sampling_interval == 300:
                        auto_slot_unit = TimeUnit('5m')
                    elif detected_sampling_interval == 60:
                        auto_slot_unit = TimeUnit('1m')
                    else:
                        auto_slot_unit = TimeUnit('{}s'.format(detected_sampling_interval))

        # Do we have to force a specific type?
        if not autodetect_series_type:
            if series_type == 'points':
                series_type = DataTimePoint
                slot_unit = None
            elif series_type == 'slots':
                series_type = DataTimeSlot
                slot_unit = TimeUnit('{}s'.format(detected_sampling_interval))
            else:
                raise ValueError('Unknown value "{}" for type. Accepted types are "points" or "slots".'.format(self.series_type))
        else:
            # Log the type and unit we detected
            if detected_series_type == DataTimeSlot:
                if series_type:
                    logger.info('Assuming {} time unit and creating Slots.'.format(auto_slot_unit))
                else:
                    logger.info('Assuming {} time unit and creating Slots. Use series_type=\'points\' if you want Points instead.'.format(auto_slot_unit))
            #else:
            #    logger.info('Assuming {} sampling interval and creating {}.'.format(detected_sampling_interval, series_type.__class__.__name__))

            # and use it.
            series_type = detected_series_type
            slot_unit = auto_slot_unit

        # If we were explicitly given a slot unit, override
        if force_slot_unit:
            if not isinstance(force_slot_unit, TimeUnit):
                force_slot_unit = TimeUnit(force_slot_unit)
            slot_unit = force_slot_unit

        # Initialize the series
        series = TimeSeries()

        # Create point or slot series
        if series_type == DataTimePoint:

            for item in items:
                try:
                    # Handle data_indexes (including the data_loss)
                    data_indexes = None
                    if isinstance (item[1], dict):
                        data_indexes = {}
                        for key in item[1]:
                            if key.startswith('__'):
                                data_indexes[key[2:]] = item[1][key]
                        # Remove data_indexes from item data
                        for index in data_indexes:
                            item[1].pop('__'+index)

                    # Create DataTimePoint, set data and data_indexes
                    datatimepoint = DataTimePoint(t=item[0], data=item[1], data_indexes=data_indexes, tz=tz)

                    # Append
                    series.append(datatimepoint)

                except Exception as e:
                    if self.skip_errors:
                        if not self.silence_errors:
                            logger.error(e)
                    else:
                        raise e from None
        else:

            for i, item in enumerate(items):
                try:
                    # Handle data_indexes (including the data_loss)
                    data_indexes = None
                    if isinstance (item[1], dict):
                        data_indexes = {}
                        for key in item[1]:
                            if key.startswith('__'):
                                data_indexes[key[2:]] = item[1][key]
                        # Remove data_indexes from item data
                        for index in data_indexes:
                            item[1].pop('__'+index)

                    if DEFAULT_SLOT_DATA_LOSS is not None and 'data_loss' not in data_indexes:
                        data_indexes['data_loss'] = DEFAULT_SLOT_DATA_LOSS

                    # Create DataTimeSlot, set data and data_indexes
                    datatimeslot = DataTimeSlot(t=item[0], unit=slot_unit, data=item[1], data_indexes=data_indexes, tz=tz)

                    # Append
                    series.append(datatimeslot)

                except ValueError as e:
                    # The only ValueError that could (should) arise here is a "Not in succession" error.
                    missing_timestamps = []
                    prev_dt = dt_from_s(items[i-1][0], tz=tz)
                    while True:
                        dt = prev_dt + slot_unit
                        # Note: the equal here is just to prevent endless loops, the check shoudl actally be just an equal
                        if s_from_dt(dt) >= item[0]:
                            # We are arrived, append all the missing items and then the item we originally tried to and break
                            for j, missing_timestamp in enumerate(missing_timestamps):
                                # Set data by interpolation
                                interpolated_data = {data_label: (((items[i][1][data_label]-items[i-1][1][data_label])/(len(missing_timestamps)+1)) * (j+1)) + items[i-1][1][data_label]  for data_label in  items[-1][1] }
                                series.append(DataTimeSlot(t=missing_timestamp, unit=slot_unit, data=interpolated_data, data_loss=1, tz=tz))
                            series.append(DataTimeSlot(t=item[0], unit=slot_unit, data=item[1], data_loss=DEFAULT_SLOT_DATA_LOSS, tz=tz))
                            break

                        else:
                            missing_timestamps.append(s_from_dt(dt))
                        prev_dt = dt

        return series


    def put(self, series, id=None, overwrite=False):
        """Store the time series in the CSV file.

        Args:
            id: Not implemented for this storage.
            overwrite: if the destination file can be overwritten.
        """
        if id:
            raise NotImplementedError('This storage does not support multiple time series, so the id argument cannot be used')

        if os.path.isfile(self.filename) and not overwrite:
            raise Exception('File already exists. use overwrite=True to overwrite.')

        # Set data_indexes here once for all or there will be a slowdown afterwards
        data_indexes = series._all_data_indexes()

        with open(self.filename, 'w') as csv_file:

            # 0) Dump CSV-storage specific metadata & chck type
            if issubclass(series.item_type, DataTimePoint):
                csv_file.write('# Generated by Timeseria CSVFileStorage at {}\n'.format(now_dt()))
                csv_file.write('# {}\n'.format(series))
                csv_file.write('# Extra parameters: {{"type": "points", "tz": "{}", "resolution": "{}"}}\n'.format(series.tz, series.resolution))

            elif issubclass(series.item_type, DataTimeSlot):
                csv_file.write('# Generated by Timeseria CSVFileStorage at {}\n'.format(now_dt()))
                csv_file.write('# {}\n'.format(series))
                csv_file.write('# Extra parameters: {{"type": "slots", "tz": "{}", "resolution": "{}"}}\n'.format(series.tz, series.resolution))

            else:
                raise TypeError('Can store only time series of DataTimePoints or DataTimeSlots')

            # 1) Dump headers
            data_labels_part = ','.join([str(data_label) for data_label in series.data_labels()])
            data_indexes_part = ','.join(['__'+index for index in data_indexes])
            if data_indexes_part:
                csv_file.write('epoch,{},{}\n'.format(data_labels_part,data_indexes_part))
            else:
                csv_file.write('epoch,{}\n'.format(data_labels_part))

            # 2) Dump data (and data_indexes)
            for item in series:
                if isinstance (series[0].data, list) or isinstance (series[0].data, tuple):
                    data_part = ','.join([str(item.data[int(data_label)]) for data_label in series.data_labels()])
                else:
                    data_part = ','.join([str(item.data[data_label]) for data_label in series.data_labels()])
                data_indexes_part = ','.join([str(item.data_indexes[index]) for index in data_indexes])
                if data_indexes_part:
                    csv_file.write('{},{},{}\n'.format(item.t, data_part, data_indexes_part))
                else:
                    csv_file.write('{},{}\n'.format(item.t, data_part))




