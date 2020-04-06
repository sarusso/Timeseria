import os
import chardet
from chardet.universaldetector import UniversalDetector
from .time import dt_from_s

# Setup logging
import logging
logger = logging.getLogger(__name__)

HARD_DEBUG = False


def detect_encoding(filename, streaming=False):
    
    if streaming:
        detector = UniversalDetector()
        with open(filename, 'rb') as file_pointer:      
            for i, line in enumerate(file_pointer.readlines()):
                if HARD_DEBUG: logger.debug('Itearation #%s: confidence=%s',i,detector.result['confidence'])
                detector.feed(line)
                if detector.done:  
                    if HARD_DEBUG: logger.debug('Detected encoding at line "%s"', i)
                    break
        detector.close() 
        chardet_results = detector.result

    else:
        with open(filename, 'rb') as file_pointer:
            chardet_results = chardet.detect(file_pointer.read())
             
    logger.debug('Detected encoding "%s" with "%s" confidence (streaming=%s)', chardet_results['encoding'],chardet_results['confidence'], streaming)
    encoding = chardet_results['encoding']
     
    return encoding






