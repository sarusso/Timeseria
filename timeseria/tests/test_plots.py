import unittest
from propertime.utils import dt
from ..datastructures import TimeSeries, DataTimePoint

# Setup logging
from .. import logger
logger.setup()

class TestPlots(unittest.TestCase):

    def test_basic(self):

        timeseries = TimeSeries(DataTimePoint(dt=dt(2015,10,24,0,0,0, tz='Europe/Rome'),
                                              data={'value_1':23.8, 'value_2':17.5},
                                              data_indexes={'index_1':0.2, 'index_2':0.5}),
                                DataTimePoint(dt=dt(2015,10,25,0,0,0, tz='Europe/Rome'),
                                              data={'value_1':22.1, 'value_2':16.2},
                                              data_indexes={'index_1':0.1, 'index_2':0.4}),
                                DataTimePoint(dt=dt(2015,10,26,0,0,0, tz='Europe/Rome'),
                                              data={'value_1':25.3, 'value_2':19.1},
                                              data_indexes={'index_1':0.3, 'index_2':0.6}))
        html_output = timeseries.plot(html=True)

        data_string = '''[[new Date(Date.UTC(2015, 9, 24, 0, 0, 0)),23.80,17.50,0.2000,0.5000],[new Date(Date.UTC(2015, 9, 25, 0, 0, 0)),22.10,16.20,0.1000,0.4000],[new Date(Date.UTC(2015, 9, 26, 0, 0, 0)),25.30,19.10,0.3000,0.6000]],{labels: ['Timestamp', 'value_1', 'value_2', 'index_1', 'index_2']'''
        self.assertTrue(data_string in html_output)

        # Check that no errors are triggered
        timeseries.plot(html=True, color='#c0c0c0')
        timeseries.plot(html=True, data_label_colors=['#c0c0c0'])
        timeseries.plot(html=True, data_label_colors=['#c0c0c0', '#ff0000'])
        timeseries.plot(html=True, data_label_colors={'value_1':'#c0c0c0', 'value_2': '#ff0000'})
        timeseries.plot(html=True, data_index_colors=['#c0c0c0'])
        timeseries.plot(html=True, data_index_colors=['#c0c0c0', '#ff0000'])
        timeseries.plot(html=True, data_index_colors={'index_1':'#c0c0c0', 'index_2': '#ff0000'})

        # Check missing or wrong keys
        with self.assertRaises(KeyError):
            timeseries.plot(html=True, data_label_colors={'value_1':'#c0c0c0'})
        with self.assertRaises(KeyError):
            timeseries.plot(html=True, data_index_colors={'index_1':'#c0c0c0'})
        with self.assertRaises(KeyError):
            timeseries.plot(html=True, data_index_colors={'index_1':'#c0c0c0', 'index_3': '#ff0000'})

