import time
import logging

def timeit(func):
    def wrapped(*args, **kargs):
        start = time.time()
        logging.info(f'{func.__name__} called with: {args}, {kargs}')
        res = func(*args, **kargs)
        end = time.time()
        logging.info(f'{func.__name__}: {end-start}ms')
        return res
    return wrapped

        
