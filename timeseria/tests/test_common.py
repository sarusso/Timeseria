import unittest
import os
import keras
import tensorflow
from ..common import detect_encoding

# Setup logging
import logging
logging.basicConfig(level=os.environ.get('LOGLEVEL') if os.environ.get('LOGLEVEL') else 'CRITICAL')

TEST_DATA_PATH = '/'.join(os.path.realpath(__file__).split('/')[0:-1]) + '/test_data/'


class TestCommon(unittest.TestCase):

    def test_versions(self):
        self.assertEqual(keras.__version__, '2.1.3')
        self.assertEqual(tensorflow.__version__, '1.9.0')


    def test_detect_encoding(self):
        
        encoding = detect_encoding('{}/shampoo_sales.csv'.format(TEST_DATA_PATH), streaming=False)
        self.assertEqual(encoding, 'ascii')
        
     
        
