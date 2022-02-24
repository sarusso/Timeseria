import os
import logging

LOGLEVEL = os.environ.get('TIMESERIA_LOGLEVEL') if os.environ.get('TIMESERIA_LOGLEVEL') else 'CRITICAL'

levels_mapping = { 50: 'CRITICAL',
                   40: 'ERROR',
                   30: 'WARNING',
                   20: 'INFO',
                   10: 'DEBUG',
                    0: 'NOTSET'}
               

def setup(level=LOGLEVEL, force=False):
    timeseria_logger = logging.getLogger('timeseria')
    #print('Setting log level to "{}"'.format(level))
    try:
        configured = False
        for handler in timeseria_logger.handlers:
            if handler.get_name() == 'timeseria_handler':
                configured = True
                if force:
                    handler.setLevel(level=level) # Set global timeseria logging level
                    timeseria_logger.setLevel(level=level) # Set global timeseria logging level
                else:
                    if levels_mapping[handler.level] != level.upper():
                        timeseria_logger.warning('You tried to setup the logger with level "{}" but it is already configured with level "{}". Use force=True to force reconfiguring it.'.format(level, levels_mapping[handler.level]))
    except IndexError:
        configured=False
    
    if not configured:
        timeseria_handler = logging.StreamHandler()
        timeseria_handler.set_name('timeseria_handler')
        timeseria_handler.setLevel(level=level) # Set timeseria default (and only) handler level 
        timeseria_handler.setFormatter(logging.Formatter('[%(levelname)s] %(name)s: %(message)s'))
        timeseria_logger.addHandler(timeseria_handler)
        timeseria_logger.setLevel(level=level) # Set global timeseria logging level
