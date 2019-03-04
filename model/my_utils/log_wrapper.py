import logging
from time import gmtime, strftime
import sys

def create_logger(name, silent=False, to_disk=False, log_file=None, prefix=None):
    """Logger wrapper
    by xiaodong liu, xiaodl@microsoft.com
    """
    # setup logger
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
    if not silent:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        log.addHandler(ch)
    if to_disk:
        prefix = prefix if prefix is not None else 'my_log'
        log_file = log_file if log_file is not None else strftime('/log/{}-%Y-%m-%d-%H-%M-%S.log'.format(prefix), gmtime())
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        log.addHandler(fh)
    return log
