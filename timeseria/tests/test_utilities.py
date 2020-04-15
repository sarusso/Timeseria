import unittest
import os

from ..utilities import detect_encoding

# Setup logging
import logging
logging.basicConfig(level=os.environ.get('LOGLEVEL') if os.environ.get('LOGLEVEL') else 'CRITICAL')

TEST_DATA_PATH = '/'.join(os.path.realpath(__file__).split('/')[0:-1]) + '/test_data/'


class TestUtilities(unittest.TestCase):

    def test_detect_encoding(self):
        
        encoding = detect_encoding('{}/csv/shampoo_sales.csv'.format(TEST_DATA_PATH), streaming=False)
        self.assertEqual(encoding, 'ascii')
        
     
        
