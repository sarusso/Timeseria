import unittest
import os
from ..datastructures import Point, TimePoint, DataPoint, DataTimePoint
from ..datastructures import Serie, TimePointSerie, DataPointSerie, DataTimePointSerie
from ..storages import CSVFileStorage


# Setup logging
import logging
logging.basicConfig(level=os.environ.get('LOGLEVEL') if os.environ.get('LOGLEVEL') else 'CRITICAL')

TEST_DATA_PATH = '/'.join(os.path.realpath(__file__).split('/')[0:-1]) + '/test_data/'


class TestCSVFileStorage(unittest.TestCase):

    def test_CSVFileStorage(self):

        # Only data
        storage = CSVFileStorage('{}/shampoo_sales.csv'.format(TEST_DATA_PATH),
                                 data_columns = ['Sales'])
        
        dataPointSerie = storage.get()
        self.assertTrue(isinstance(dataPointSerie, DataPointSerie))       
        #dataTimePointSerie.plot()


        # Only time
        storage = CSVFileStorage('{}/shampoo_sales.csv'.format(TEST_DATA_PATH),
                                 date_column  = 'Month',
                                 date_format  = '%y-%m')
        
        timePointSerie = storage.get()
        self.assertTrue(isinstance(timePointSerie, TimePointSerie))       
        #timePointSerie.plot()


        # Data and time (using time column instead of date)
        storage = CSVFileStorage('{}/shampoo_sales.csv'.format(TEST_DATA_PATH),
                                 time_column = 'Month',
                                 time_format = '%y-%m',
                                 data_columns = ['Sales'])
         
        dataTimePointSerie = storage.get()
        self.assertTrue(isinstance(dataTimePointSerie, DataTimePointSerie))        
        #dataTimePointSerie.plot()

        with self.assertRaises(NotImplementedError):
            storage.get(from_t=1246453)


        # Time in epoch format
        storage = CSVFileStorage('{}/temp_short_10m.csv'.format(TEST_DATA_PATH),
                                 time_column = 'epoch',
                                 time_format = 'epoch',
                                 data_columns = ['temp'])
         
        dataTimePointSerie = storage.get()

        self.assertEqual(len(dataTimePointSerie),100)

