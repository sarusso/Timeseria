import unittest
import keras
import tensorflow

class TestCommon(unittest.TestCase):

    def test_versions(self):
        self.assertEqual(keras.__version__, '2.1.3')
        self.assertEqual(tensorflow.__version__, '1.9.0')
