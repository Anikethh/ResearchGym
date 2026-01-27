import os
import sys
import logging
from datetime import datetime,timedelta

def get_timestamp():
    # return datetime.now().strftime('%y%m%d-%H%M%S')
    return (datetime.now()+timedelta(days=1/3)).strftime('%y%m%d-%H%M%S')

def setup_logger(logger_name, phase, level=logging.INFO, screen=False, tofile=False):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    root = './log_dir/'
    time_now = get_timestamp()
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(time_now))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)
    return time_now
