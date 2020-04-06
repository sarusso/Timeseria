import csv
from .common import detect_encoding
from .datastructures import DataTimePoint, DataTimePointSerie, DataPointSerie, TimePointSerie, TimePoint, DataPoint
from .time import s_from_dt
import datetime

# Setup logging
import logging
logger = logging.getLogger(__name__)

HARD_DEBUG = False


#======================
#  CSV File Storage
#======================

class CSVFileStorage(object):
    
    def __init__(self, filename_with_path, encoding = None, time_column = None, time_format = None,
                 date_column = None, date_format = None, data_columns = None, separator=','):
        
        # Fine & encoding
        self.filename_with_path = filename_with_path
        if not encoding:
            self.encoding = detect_encoding(filename_with_path, streaming=False)
        else:
            self.encoding = encoding

        # Time
        self.time_column = time_column
        self.time_format = '' if time_format is None else time_format
        self.date_column = date_column
        self.date_format = '' if date_format is None else date_format    
        
        # Data
        self.data_columns = data_columns
        
        # Other
        self.separator = separator

        # Check
        if self.time_column is None and self.date_column is None and not self.data_columns:
            raise ValueError('No time column, date column or data columns provided, cannot get anyything fromt this CSV file')


    def get(self, *args, **kwargs):

        if args or kwargs:
            raise NotImplementedError('Sorry, the CSVFileStorage get does not accept any arguments.')

        if (self.time_column is None and self.date_column is None):
            serie_type = DataPointSerie
            serie = DataPointSerie()
        elif (not self.data_columns):
            serie_type = TimePointSerie
            serie = TimePointSerie()
        else:
            serie_type = DataTimePointSerie            
            serie = DataTimePointSerie()

        logger.debug('Set serie type to "%s"', serie_type.__class__.__name__)

        # TODO: evaluate rU vs newline='\n'
        with open(self.filename_with_path, 'r', encoding=self.encoding) as csv_file:
     
            csv_reader = csv.DictReader(csv_file, delimiter=self.separator)
            keys = None
            for i, row in enumerate(csv_reader):
                if HARD_DEBUG: logger.debug('Index:%s, row:"%s"',i,row)
                if not keys:
                    keys = list(row.keys())
                    if HARD_DEBUG: logger.debug('Setting keys "%s"', keys)

                # Handle timestamp if we have to
                if serie_type in [TimePointSerie, DataTimePointSerie]:
                
                    # Time part
                    time_part=''
                    if self.time_column:
                        if isinstance(self.time_column, int):
                            time_part = row[keys[self.time_column]]
                        else:
                            time_part = row[self.time_column]
                     
                    # Date part
                    date_part=''
                    if self.date_column:
                        if isinstance(self.date_column, int):
                            date_part = row[keys[self.date_column]]
                        else:
                            date_part = row[self.date_column]
                    
                    # Convert to timestamp
                    if self.date_column:
                        timestamp = date_part + '\t' + time_part
                    else:
                        timestamp = time_part
                        
                    
                    if HARD_DEBUG: logger.debug('Will process timestamp "%s"', timestamp)
                    if self.time_format == 'epoch':
                        t = float(timestamp)
                    else:
                        if self.date_column:
                            dt = datetime.datetime.strptime(timestamp, self.date_format + '\t' + self.time_format)
                            t = s_from_dt(dt)
                        else:
                            dt = datetime.datetime.strptime(timestamp, self.time_format)
                            t = s_from_dt(dt)
                    

                # Handle data if we have to
                if serie_type != TimePointSerie:
                    
                    # Handle data column(s)
                    if len(self.data_columns)>1:
                        raise NotImplementedError('Multivariate time series are not yet supported (Got data_columns="{}")'.format(self.data_columns))
                    if isinstance(self.data_columns[0], int):
                        data = row[keys[self.data_columns[0]]]
                    else:
                        data = row[self.data_columns[0]]

                    if isinstance(self.data_columns[0], int):
                        data = row[keys[self.data_columns[0]]]
                    else:
                        data = row[self.data_columns[0]]     
                
                    # Try to convert to float
                    try:
                        data = float(data)
                    except:
                        pass
                    
                
                # Append the right datastructure to the serie
                if serie_type == TimePointSerie:
                    serie.append(TimePoint(t=t))
                    
                elif serie_type == DataPointSerie:
                    serie.append(DataPoint(i=i, data=data))
                    
                elif serie_type == DataTimePointSerie:
                    serie.append(DataTimePoint(t=t, data=data))
                    
                else:
                    raise Exception('Consistency Error')
     
        return serie









