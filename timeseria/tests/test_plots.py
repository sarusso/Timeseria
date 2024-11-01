import unittest
from propertime.utils import dt
from ..datastructures import TimeSeries, DataTimePoint

# Setup logging
from .. import logger
logger.setup()


class TestPlots(unittest.TestCase):

    def test_basic(self):

        timeseries = TimeSeries(DataTimePoint(dt=dt(2015,10,24,0,0,0, tz='Europe/Rome'), data={'value':23.8}),
                                DataTimePoint(dt=dt(2015,10,25,0,0,0, tz='Europe/Rome'), data={'value':22.1}),
                                DataTimePoint(dt=dt(2015,10,26,0,0,0, tz='Europe/Rome'), data={'value':25.3}))

        html_output = timeseries.plot(html=True)

        # True output len: 6564
        self.assertTrue(len(html_output)>6000)
        self.assertTrue(len(html_output)<7000)
