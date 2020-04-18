import unittest
import os
from ..exceptions import InputException
from ..utilities import detect_encoding, compute_coverage
from ..datastructures import DataTimePointSerie, DataTimePoint
from ..time import dt, s_from_dt, TimeSpan

# Setup logging
import logging
logging.basicConfig(level=os.environ.get('LOGLEVEL') if os.environ.get('LOGLEVEL') else 'CRITICAL')

TEST_DATA_PATH = '/'.join(os.path.realpath(__file__).split('/')[0:-1]) + '/test_data/'


class TestDetectEncoding(unittest.TestCase):

    def test_detect_encoding(self):
        
        encoding = detect_encoding('{}/csv/shampoo_sales.csv'.format(TEST_DATA_PATH), streaming=False)
        self.assertEqual(encoding, 'ascii')
        
     

class TestComputeCoverage(unittest.TestCase):

    def setUp(self):       
        
        # All the following time series have point with validity=1m
        
        # Time series from 16:58:00 to 17:32:00 (Europe/Rome)
        self.dataTimePointSerie1 = DataTimePointSerie()
        start_t = 1436022000 - 120
        for i in range(35):
            dataTimePoint = DataTimePoint(t = start_t + (i*60),
                                          data = {'temperature': 154+i})
            self.dataTimePointSerie1.append(dataTimePoint)
        
        # Time series from 17:00:00 to 17:30:00 (Europe/Rome)
        self.dataTimePointSerie2 = DataTimePointSerie()
        start_t = 1436022000
        for i in range(34):
            dataTimePoint = DataTimePoint(t    = start_t + (i*60),
                                          data = {'temperature': 154+i})
            self.dataTimePointSerie2.append(dataTimePoint)

        # Time series from 17:00:00 to 17:20:00 (Europe/Rome)
        self.dataTimePointSerie3 = DataTimePointSerie()
        start_t = 1436022000 - 120
        for i in range(23):
            dataTimePoint = DataTimePoint(t    = start_t + (i*60),
                                          data = {'temperature': 154+i})
            self.dataTimePointSerie3.append(dataTimePoint) 
        
        # Time series from 17:10:00 to 17:30:00 (Europe/Rome)
        self.dataTimePointSerie4 = DataTimePointSerie()
        start_t = 1436022000 + 600
        for i in range(21):
            dataTimePoint = DataTimePoint(t    = start_t + (i*60),
                                          data = {'temperature': 154+i})
            self.dataTimePointSerie4.append(dataTimePoint)

        # Time series from 16:58:00 to 17:32:00 (Europe/Rome)
        self.dataTimePointSerie5 = DataTimePointSerie()
        start_t = 1436022000 - 120
        for i in range(35):
            if i > 10 and i <21:
                continue
            dataTimePoint = DataTimePoint(t    = start_t + (i*60),
                                          data = {'temperature': 154+i})
            self.dataTimePointSerie5.append(dataTimePoint)

        # The following time series has point with validity=15m

        # Time series from 2019,10,1,1,0,0 to 2019,10,1,6,0,0 (Europe/Rome)
        from_dt  = dt(2019,10,1,1,0,0, tzinfo='Europe/Rome')
        to_dt    = dt(2019,10,1,6,0,0, tzinfo='Europe/Rome')
        timeSpan = TimeSpan('15m') 
        self.dataTimePointSerie6 = DataTimePointSerie()
        slider_dt = from_dt
        count = 0
        while slider_dt < to_dt:
            if count not in [1, 6, 7, 8, 9, 10]:
                dataTimePoint = DataTimePoint(t    = s_from_dt(slider_dt),
                                              data = {'temperature': 154+count})
                self.dataTimePointSerie6.append(dataTimePoint)
            slider_dt = slider_dt + timeSpan
            count += 1
 

    def test_compute_coverage(self):
        
        from_t = 1436022000       # 2015-07-04 17:00:00+02:00
        to_t   = 1436022000+1800  # 2015-07-04 17:30:00+02:00
        validity = 60


        # A) Full coverage (coverage=1.0) and again, to test reproducibility
        coverage = compute_coverage(dataTimePointSerie  = self.dataTimePointSerie1,
                                    from_t = from_t, to_t = to_t, validity=validity)  
        self.assertEqual(coverage, 1.0)
        
        coverage = compute_coverage(dataTimePointSerie  = self.dataTimePointSerie1,
                                    from_t = from_t, to_t = to_t, validity=validity)  
        self.assertEqual(coverage, 1.0)       

  
        # B) Full coverage (coverage=1.0) witjout prev/next in the timeSeries 
        coverage = compute_coverage(dataTimePointSerie  = self.dataTimePointSerie1,
                                    from_t = from_t, to_t = to_t, validity=validity)  
        self.assertEqual(coverage, 1.0) 
        self.assertEqual(coverage, 1.0)  

 
        # C) Missing ten minutes over 30 at the end (coverage=0.683))
        coverage = compute_coverage(dataTimePointSerie  = self.dataTimePointSerie3,
                                    from_t = from_t, to_t = to_t, validity=validity)  
        # 20 minutes plus other 30 secs validity for the 20th point over 30 minutes
        self.assertEqual(coverage, ( ((20*60.0)+30.0) / (30*60.0)) ) 
 
 
        # D) Missing ten minutes over 30 at the beginning (coverage=0.683)
        coverage = compute_coverage(dataTimePointSerie  = self.dataTimePointSerie4,
                                    from_t = from_t, to_t = to_t, validity=validity)  
        # 20 minutes plus other 30 secs (previous) validity for the 10th point over 30 minutes
        self.assertEqual(coverage, ( ((20*60.0)+30.0) / (30*60.0)) ) 
 
 
        # E) Missing eleven minutes over 30 in the middle (coverage=0.66)
        coverage = compute_coverage(dataTimePointSerie  = self.dataTimePointSerie5,
                                    from_t = from_t, to_t = to_t, validity=validity) 
        # 20 minutes plus other 30 secs (previous) validity for the 10th point over 30 minutes
        self.assertAlmostEqual(coverage, (2.0/3.0))
  
 
        # F) Missing half slot before slot re-start
        from_t = s_from_dt(dt=dt(2019,10,1,3,30,0, tzinfo='Europe/Rome'))
        to_t   = s_from_dt(dt=dt(2019,10,1,3,45,0, tzinfo='Europe/Rome'))
        coverage = compute_coverage(dataTimePointSerie  = self.dataTimePointSerie6,
                                    from_t = from_t, to_t = to_t, validity=900)         
        self.assertAlmostEqual(coverage, (0.5))
 

    def tearDown(self):
        pass






     
