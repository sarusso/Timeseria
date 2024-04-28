import unittest
import os
import tempfile
from ..datastructures import DataTimePoint, DataTimeSlot, TimeSeries
from ..storages import CSVFileStorage
from ..units import TimeUnit

# Setup logging
from .. import logger
logger.setup()

# Set test data path
TEST_DATA_PATH = '/'.join(os.path.realpath(__file__).split('/')[0:-1]) + '/test_data/'


class TestCSVFileStorage(unittest.TestCase):

    def test_CSVFileStorage_get_basic(self):

        # Basic iso8601 with two columns, one for the timestamp and one for the value, no labels
        storage = CSVFileStorage(TEST_DATA_PATH + '/csv/single_value_no_labels.csv')
        data_time_point_series = storage.get()
        self.assertEqual(len(data_time_point_series), 6)
        self.assertEqual(data_time_point_series[0].t, 946684800)
        self.assertEqual(data_time_point_series[0].data, [1000])

        # Basic iso8601 multi values no labels
        storage = CSVFileStorage(TEST_DATA_PATH + '/csv/multi_values_no_labels.csv')
        data_time_point_series = storage.get()
        self.assertEqual(len(data_time_point_series), 50)
        self.assertEqual(data_time_point_series[0].data, [1000,10])
        self.assertEqual(data_time_point_series[-1].data, [1040,14])

        # Basic iso8601 multi values no labels and filtering
        storage = CSVFileStorage(TEST_DATA_PATH + '/csv/multi_values_with_labels.csv', data_labels=[2])
        data_time_point_series = storage.get()
        self.assertEqual(len(data_time_point_series), 50)
        self.assertEqual(data_time_point_series[0].data, [10])
        self.assertEqual(data_time_point_series[-1].data, [14])

        # Basic iso8601 multi values no labels and filtering and force data type
        storage = CSVFileStorage(TEST_DATA_PATH + '/csv/multi_values_with_labels.csv', data_labels=[1], data_type=float)
        data_time_point_series = storage.get()
        self.assertEqual(len(data_time_point_series), 50)
        self.assertEqual(data_time_point_series[0].data, [1000])
        self.assertEqual(data_time_point_series[-1].data, [1040])

        # Basic iso8601 multi values with labels
        storage = CSVFileStorage(TEST_DATA_PATH + '/csv/multi_values_with_labels.csv')
        data_time_point_series = storage.get()
        self.assertEqual(len(data_time_point_series), 50)
        self.assertEqual(data_time_point_series[0].data, {'flow': 1000.0, 'temp': 10.0})
        self.assertEqual(data_time_point_series[-1].data, {'flow': 1040.0, 'temp': 14.0})

        # Test we don't have any extra data index as well
        self.assertEqual(data_time_point_series._all_data_indexes(), [])

        # Basic iso8601 multi values with labels
        storage = CSVFileStorage(TEST_DATA_PATH + '/csv/multi_values_with_labels.csv', data_labels=['temp'])
        data_time_point_series = storage.get()
        self.assertEqual(len(data_time_point_series), 50)
        self.assertEqual(data_time_point_series[0].data, {'temp': 10.0})
        self.assertEqual(data_time_point_series[-1].data,{'temp': 14.0})

        # Basic iso8601 multi values with labels
        storage = CSVFileStorage(TEST_DATA_PATH + '/csv/multi_values_with_labels.csv', data_labels=[1])
        data_time_point_series = storage.get()
        self.assertEqual(len(data_time_point_series), 50)
        self.assertEqual(data_time_point_series[0].data, [1000.0])
        self.assertEqual(data_time_point_series[-1].data, [1040.0])


    def test_CSVFileStorage_get_errors(self):

        # Test wrong datatypes:
        with self.assertRaises(Exception):
            CSVFileStorage(TEST_DATA_PATH + '/csv/multi_values_with_labels.csv', data_labels = 0)
        with self.assertRaises(Exception):
            CSVFileStorage(TEST_DATA_PATH + '/csv/multi_values_with_labels.csv', data_labels = 'flow')
        with self.assertRaises(Exception):
            CSVFileStorage(TEST_DATA_PATH + '/csv/multi_values_with_labels.csv', time_label = [0])
        with self.assertRaises(Exception):
            CSVFileStorage(TEST_DATA_PATH + '/csv/multi_values_with_labels.csv', date_label = [0])

        # Test requiring a not existent data column:
        storage = CSVFileStorage(TEST_DATA_PATH + '/csv/multi_values_with_labels.csv', data_labels = ['temp', 'flow_NO'])
        with self.assertRaises(Exception):
            storage.get()


    def test_CSVFileStorage_get_dirty(self):

        # Basic iso8601 with two columns, one for the timestamp and one for the value, no labels, dirty
        storage = CSVFileStorage(TEST_DATA_PATH + '/csv/single_value_no_labels_dirty.csv')
        with self.assertRaises(Exception):
            _ = storage.get()

        storage = CSVFileStorage(TEST_DATA_PATH + '/csv/single_value_no_labels_dirty.csv', skip_errors=True)
        data_time_point_series = storage.get()
        self.assertEqual(len(data_time_point_series), 3)
        self.assertEqual(data_time_point_series[0].t, 946684800)
        self.assertEqual(data_time_point_series[0].data, [1000])

        # Basic iso8601 with two labels, dirty
        storage = CSVFileStorage(TEST_DATA_PATH + '/csv/multi_values_with_labels_dirty.csv', separator=';', skip_errors=True)
        data_time_point_series = storage.get()
        self.assertEqual(len(data_time_point_series), 3)
        self.assertEqual(data_time_point_series[0], DataTimePoint(t= 946688400, data={'flow': 1010.0, 'temp': 11.0}))
        self.assertEqual(data_time_point_series[1], DataTimePoint(t= 946706400, data={'flow': 1100.0, 'temp': 10.7}))
        self.assertEqual(data_time_point_series[2], DataTimePoint(t= 946713600, data={'flow': 1300.0, 'temp': 10.0}))


    def test_CSVFileStorage_get_unordered(self):

        storage = CSVFileStorage(TEST_DATA_PATH + '/csv/multi_values_with_labels_unordered.csv', sort=True)
        data_time_point_series = storage.get()
        self.assertEqual(len(data_time_point_series), 5)
        self.assertEqual(data_time_point_series[0].t, 946684800)
        self.assertEqual(data_time_point_series[0].data, {'flow': 1000.0, 'temp': 10.0})
        self.assertEqual(data_time_point_series[-1].t, 946684800 + 60*60*4)
        self.assertEqual(data_time_point_series[-1].data, {'flow': 1040.0, 'temp': 14.0})


    def test_CSVFileStorage_get_timestamp_formats(self):

        # Autodetect epoch
        storage = CSVFileStorage(TEST_DATA_PATH + '/csv/temp_long_10m.csv')
        data_time_point_series = storage.get()
        self.assertEqual(len(data_time_point_series), 19140)

        # Autodetect iso
        storage = CSVFileStorage(TEST_DATA_PATH + '/csv/multi_values_with_labels.csv')
        self.assertEqual(storage.timestamp_format, 'auto')
        self.assertEqual(storage.timestamp_column, 'auto')
        data_time_point_series = storage.get()
        self.assertEqual(storage.timestamp_format, 'iso8601')
        self.assertEqual(storage.timestamp_column, 'timestamp')
        self.assertEqual(len(data_time_point_series), 50)

        # Epoch timestamp format
        storage = CSVFileStorage(TEST_DATA_PATH + '/csv/temp_short_1h.csv',
                                  timestamp_column = 'epoch',
                                  timestamp_format = 'epoch',)
        data_time_point_series = storage.get()
        self.assertEqual(len(data_time_point_series), 100)
        self.assertEqual(data_time_point_series[0].t, 1546477200)
        self.assertEqual(data_time_point_series[-1].t, 1546833600)

        # Use only month and year as date column
        storage = CSVFileStorage(TEST_DATA_PATH + '/csv/shampoo_sales.csv',
                                 date_column = 'Month',
                                 date_format = '%y-%m')
        data_time_point_series = storage.get()
        self.assertEqual(len(data_time_point_series), 36)
        self.assertEqual(data_time_point_series[0].t, 978307200)
        self.assertEqual(data_time_point_series[-1].t, 1070236800)

        # Separate time and date columns, custom format
        storage = CSVFileStorage(TEST_DATA_PATH + '/csv/format2.csv',
                                 date_column = 'Date',
                                 date_format = '%d/%m/%Y',
                                 time_column = 'Time',
                                 time_format = '%H:%M')
        data_time_point_series = storage.get()
        self.assertEqual(len(data_time_point_series), 6)
        self.assertEqual(data_time_point_series[0].t, 1583280000)
        self.assertEqual(data_time_point_series[-1].t, 1583298000)

        # Test only date column and without meaningful timestamp label (and force points)
        storage = CSVFileStorage(TEST_DATA_PATH + '/csv/only_date_no_meaningful_timestamp_label.csv',
                                 timestamp_format = '%Y-%m-%d', series_type='points')

        data_time_point_series = storage.get()
        self.assertEqual(len(data_time_point_series), 95)
        self.assertEqual(data_time_point_series[0].t, 1197244800)
        self.assertEqual(data_time_point_series[-1].t, 1205798400)

        # Test data with quotes
        storage = CSVFileStorage(TEST_DATA_PATH + '/csv/format5.csv')
        data_time_point_series = storage.get()
        self.assertEqual(len(data_time_point_series), 6)
        self.assertEqual(data_time_point_series[2].data['temp'], 23.34)
        self.assertEqual(data_time_point_series[3].data['humi'], 55)

        # Test getting only a specific data labels
        storage = CSVFileStorage(TEST_DATA_PATH + '/csv/format5.csv')
        data_time_point_series = storage.get(filter_data_labels=['temp'])
        self.assertEqual(len(data_time_point_series), 6)
        self.assertEqual(data_time_point_series[0].data_labels(), ['temp'])


    def test_CSVFileStorage_get_timezones(self):

        # Set the default timezone
        storage = CSVFileStorage(TEST_DATA_PATH + '/csv/multi_values_with_labels.csv', tz='Europe/Rome')
        timeseries = storage.get()
        self.assertEqual(timeseries[0].t, 946684800.0)
        self.assertEqual(str(timeseries[0].tz), 'Europe/Rome')

        # Force a a specific timezone
        timeseries = storage.get(force_tz='America/New_York')
        self.assertEqual(timeseries[0].t, 946684800.0)
        self.assertEqual(str(timeseries.tz), 'America/New_York')


    def test_CSVFileStorage_get_slots_reconstruction(self):

        # Let the storage to generate slots with interpolated data
        storage = CSVFileStorage(TEST_DATA_PATH + '/csv/only_date_no_meaningful_timestamp_label.csv',
                                 timestamp_format = '%Y-%m-%d')
        data_time_slot_series = storage.get()
        self.assertEqual(len(data_time_slot_series), 100)

        # Check first
        self.assertEqual(data_time_slot_series[0].t, 1197244800)

        # Test correct data loss set and reconstruction
        self.assertEqual(data_time_slot_series[52].data_loss, 1)
        self.assertAlmostEqual(data_time_slot_series[52].data['y'], 8.748874781508915)

        self.assertEqual(data_time_slot_series[80].data_loss, 1)
        self.assertAlmostEqual(data_time_slot_series[80].data['y'], 7.56004994651234)

        self.assertEqual(data_time_slot_series[82].data_loss, 1)
        self.assertAlmostEqual(data_time_slot_series[82].data['y'], 7.444587100634211)

        self.assertEqual(data_time_slot_series[84].data_loss, 1)
        self.assertAlmostEqual(data_time_slot_series[84].data['y'], 7.599538949266937)

        self.assertEqual(data_time_slot_series[85].data_loss, 1)
        self.assertAlmostEqual(data_time_slot_series[85].data['y'], 7.862140984826253)

        # Check last
        self.assertEqual(data_time_slot_series[-1].t, 1205798400)

        # Check all others have data_loss = None
        for i, item in enumerate(data_time_slot_series):
            if i not in [52, 80, 82, 84, 85]:
                self.assertEqual(item.data_loss, None, 'Failed for i={}'.format(i))
            else:
                self.assertEqual(item.data_loss, 1, 'Failed for i={}'.format(i))

    def test_CSVFileStorage_get_series_types(self):

        # Get data as slots
        storage = CSVFileStorage(TEST_DATA_PATH + '/csv/temp_short_1h.csv')
        data_time_point_series = storage.get(force_slots=True)
        self.assertEqual(len(data_time_point_series), 100)
        self.assertEqual(data_time_point_series[0].start.t, 1546477200)
        self.assertEqual(data_time_point_series[-1].start.t, 1546833600)


    def test_CSVFileStorage_get_no_data(self):
        storage = CSVFileStorage(TEST_DATA_PATH + '/csv/no_data.csv')
        from ..exceptions import NoDataException
        with self.assertRaises(NoDataException):
            storage.get()


    def test_CSVFileStorage_put(self):

        with tempfile.TemporaryDirectory() as temp_dir:

            # Test on points
            data_time_point_series = TimeSeries(DataTimePoint(t=60, data=[23.8,3], data_loss=0.1),
                                                DataTimePoint(t=120, data=[24.1,4], data_loss=0.2),
                                                DataTimePoint(t=240, data=[23.1,5], data_loss=0.3),
                                                DataTimePoint(t=300, data=[22.7,6], data_loss=0.4))
            data_time_point_series.change_tz('Europe/Rome')

            storage = CSVFileStorage('/{}/file_1.csv'.format(temp_dir))
            storage.put(data_time_point_series)

            with self.assertRaises(Exception):
                storage.put(data_time_point_series)

            storage.put(data_time_point_series, overwrite=True)

            # Test getting back the series from the storage
            series = storage.get()
            self.assertEqual(len(series),4)
            self.assertTrue(isinstance(series[0], DataTimePoint))

            # Test indexes correctly se by the get
            self.assertEqual(len(series[0].data_indexes),1)
            self.assertEqual(series[0].data_indexes['data_loss'], 0.1)
            self.assertEqual(series[3].data_indexes['data_loss'], 0.4)

            # Now test on slots
            data_time_point_series = TimeSeries(DataTimeSlot(t=60, unit=TimeUnit('1m'), data=[23.8,3], data_loss=0.1),
                                                DataTimeSlot(t=120, unit=TimeUnit('1m'), data=[24.1,4], data_loss=0.2),
                                                DataTimeSlot(t=180, unit=TimeUnit('1m'), data=[23.1,5], data_loss=0.3),
                                                DataTimeSlot(t=240, unit=TimeUnit('1m'), data=[22.7,6], data_loss=0.4))
            data_time_point_series.change_tz('Europe/Rome')

            storage = CSVFileStorage('/{}/file_2.csv'.format(temp_dir))
            storage.put(data_time_point_series)

            with self.assertRaises(Exception):
                storage.put(data_time_point_series)

            storage.put(data_time_point_series, overwrite=True)

            # Test getting back the series from the storage
            series = storage.get(force_slots=True)
            self.assertEqual(len(series),4)
            self.assertTrue(isinstance(series[0], DataTimeSlot))

            # Test indexes correctly se by the get
            self.assertEqual(len(series[0].data_indexes),1)
            self.assertEqual(series[0].data_indexes['data_loss'], 0.1)
            self.assertEqual(series[3].data_indexes['data_loss'], 0.4)


