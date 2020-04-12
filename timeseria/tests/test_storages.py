import unittest
import os
from ..datastructures import Point, TimePoint, DataPoint, DataTimePoint
from ..datastructures import Serie, TimePointSerie, DataPointSerie, DataTimePointSerie
from ..storages import CSVFileStorage

# Setup logging
import logging
logging.basicConfig(level=os.environ.get('LOGLEVEL') if os.environ.get('LOGLEVEL') else 'CRITICAL')

# Set test data path
TEST_DATA_PATH = '/'.join(os.path.realpath(__file__).split('/')[0:-1]) + '/test_data/'


class TestCSVFileStorage(unittest.TestCase):

    def test_CSVFileStorage_basic(self):

        # Basic iso8601 with two columns, one for the timestamp and one for the value, no labels
        storage = CSVFileStorage(TEST_DATA_PATH + '/csv/single_value_no_labels.csv')
        dataTimePointSerie = storage.get()
        self.assertEqual(len(dataTimePointSerie), 6)
        self.assertEqual(dataTimePointSerie[0].t, 946684800)
        self.assertEqual(dataTimePointSerie[0].data, 1000)
 
        # Basic iso8601 multi values no labels
        storage = CSVFileStorage(TEST_DATA_PATH + '/csv/multi_values_no_labels.csv')
        dataTimePointSerie = storage.get()
        self.assertEqual(len(dataTimePointSerie), 50)
        self.assertEqual(dataTimePointSerie[0].data, [1000,10])
        self.assertEqual(dataTimePointSerie[-1].data, [1040,14])
 
 
        # Basic iso8601 multi values no labels and filtering
        storage = CSVFileStorage(TEST_DATA_PATH + '/csv/multi_values_with_labels.csv', data_columns=[2])
        dataTimePointSerie = storage.get()
        self.assertEqual(len(dataTimePointSerie), 50)
        self.assertEqual(dataTimePointSerie[0].data, [10])
        self.assertEqual(dataTimePointSerie[-1].data, [14])
 
 
        # Basic iso8601 multi values no labels and filtering and force format
        storage = CSVFileStorage(TEST_DATA_PATH + '/csv/multi_values_with_labels.csv', data_columns=[1], data_format=list)
        dataTimePointSerie = storage.get()
        self.assertEqual(len(dataTimePointSerie), 50)
        self.assertEqual(dataTimePointSerie[0].data, [1000])
        self.assertEqual(dataTimePointSerie[-1].data, [1040])


        # Basic iso8601 multi values with labels
        storage = CSVFileStorage(TEST_DATA_PATH + '/csv/multi_values_with_labels.csv')
        dataTimePointSerie = storage.get()
        self.assertEqual(len(dataTimePointSerie), 50)
        self.assertEqual(dataTimePointSerie[0].data, {'flow': 1000.0, 'temp': 10.0})
        self.assertEqual(dataTimePointSerie[-1].data, {'flow': 1040.0, 'temp': 14.0})


        # Basic iso8601 multi values with labels
        storage = CSVFileStorage(TEST_DATA_PATH + '/csv/multi_values_with_labels.csv', data_columns=['temp'])
        dataTimePointSerie = storage.get()
        self.assertEqual(len(dataTimePointSerie), 50)
        self.assertEqual(dataTimePointSerie[0].data, {'temp': 10.0})
        self.assertEqual(dataTimePointSerie[-1].data,{'temp': 14.0})


        # Basic iso8601 multi values with labels
        storage = CSVFileStorage(TEST_DATA_PATH + '/csv/multi_values_with_labels.csv', data_columns=[1])
        dataTimePointSerie = storage.get()
        self.assertEqual(len(dataTimePointSerie), 50)
        self.assertEqual(dataTimePointSerie[0].data, [1000.0])
        self.assertEqual(dataTimePointSerie[-1].data, [1040.0])



    def test_CSVFileStorage_errors(self):

        # Test wrong datatypes:
        with self.assertRaises(Exception):
            CSVFileStorage(TEST_DATA_PATH + '/csv/multi_values_with_labels.csv', data_columns = 0)
        with self.assertRaises(Exception):
            CSVFileStorage(TEST_DATA_PATH + '/csv/multi_values_with_labels.csv', data_columns = 'flow')
        with self.assertRaises(Exception):
            CSVFileStorage(TEST_DATA_PATH + '/csv/multi_values_with_labels.csv', time_column = [0])
        with self.assertRaises(Exception):
            CSVFileStorage(TEST_DATA_PATH + '/csv/multi_values_with_labels.csv', date_column = [0])


        # Test requiring a not existent data column:
        storage = CSVFileStorage(TEST_DATA_PATH + '/csv/multi_values_with_labels.csv', data_columns = ['temp', 'flow_NO'])
        with self.assertRaises(Exception):
            storage.get()



    def test_CSVFileStorage_dirty(self):

        # Basic iso8601 with two columns, one for the timestamp and one for the value, no labels, dirty
        storage = CSVFileStorage(TEST_DATA_PATH + '/csv/single_value_no_labels_dirty.csv')
        with self.assertRaises(Exception):
            dataTimePointSerie = storage.get()
   
        storage = CSVFileStorage(TEST_DATA_PATH + '/csv/single_value_no_labels_dirty.csv', skip_errors=True)
        dataTimePointSerie = storage.get()       
        self.assertEqual(len(dataTimePointSerie), 3)
        self.assertEqual(dataTimePointSerie[0].t, 946684800)
        self.assertEqual(dataTimePointSerie[0].data, 1000)


        # Basic iso8601 with two labels, dirty
        storage = CSVFileStorage(TEST_DATA_PATH + '/csv/multi_values_with_labels_dirty.csv', value_separator=';', skip_errors=True)
        dataTimePointSerie = storage.get()        
        self.assertEqual(len(dataTimePointSerie), 3)
        #self.assertEqual(dataTimePointSerie[0], DataTimePoint(t= 946688400, data={'flow': 1010.0, 'temp': 11.0}))
        #self.assertEqual(dataTimePointSerie[1], DataTimePoint(t= 946706400, data={'flow': 1100.0, 'temp': 10.7}))
        #self.assertEqual(dataTimePointSerie[2], DataTimePoint(t= 946713600, data={'flow': 1300.0, 'temp': 10.0}))


 
    def test_CSVFileStorage_datetimeformats(self):

        # Autodetect epoch
        storage = CSVFileStorage(TEST_DATA_PATH + '/csv/temp_long_10m.csv')
        dataTimePointSerie = storage.get()
        self.assertEqual(len(dataTimePointSerie), 19140)

        # Autodetect iso
        storage = CSVFileStorage(TEST_DATA_PATH + '/csv/multi_values_with_labels.csv')
        self.assertEqual(storage.time_format, 'auto')
        self.assertEqual(storage.time_column, 'auto')
        dataTimePointSerie = storage.get()
        self.assertEqual(storage.time_format, 'iso8601')
        self.assertEqual(storage.time_column, 'timestamp')
        self.assertEqual(len(dataTimePointSerie), 50)

        # Epoch timestamp format
        storage = CSVFileStorage(TEST_DATA_PATH + '/csv/temp_short_1h.csv',
                                  time_column = 'epoch',
                                  time_format = 'epoch',)
        dataTimePointSerie = storage.get()
        self.assertEqual(len(dataTimePointSerie), 100)
        self.assertEqual(dataTimePointSerie[0].t, 1546477200)
        self.assertEqual(dataTimePointSerie[-1].t, 1546833600)


        # Use only month and year as time column
        storage = CSVFileStorage(TEST_DATA_PATH + '/csv/shampoo_sales.csv',
                                 time_column = 'Month',
                                 time_format = '%y-%m')
        dataTimePointSerie = storage.get()        
        self.assertEqual(len(dataTimePointSerie), 36)
        self.assertEqual(dataTimePointSerie[0].t, 978307200)
        self.assertEqual(dataTimePointSerie[-1].t, 1070236800)


        # Separate time and date columns, custom format
        storage = CSVFileStorage(TEST_DATA_PATH + '/csv/format2.csv',
                                 date_column = 'Date',
                                 date_format = '%d/%m/%Y',
                                 time_column = 'Time',
                                 time_format = '%H:%M')
        dataTimePointSerie = storage.get()        
        self.assertEqual(len(dataTimePointSerie), 6)
        self.assertEqual(dataTimePointSerie[0].t, 1583280000)
        self.assertEqual(dataTimePointSerie[-1].t, 1583298000)

