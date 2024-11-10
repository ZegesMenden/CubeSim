import time
from loguru import logger
import sys

def runOnce(fn: callable) -> callable:
    """runOnce is a decorator that ensures a function is only run once

    Args:
        fn (callable): the function to be wrapped

    Returns:
        callable: the wrapped function that will only run once
    """
    def wrapper(*args, **kwargs):
        if not wrapper.hasRun:
            wrapper.hasRun = True
            return fn(*args, **kwargs)
        else:
            return None
    wrapper.hasRun = False
    return wrapper

def timedWrapper(fn: callable):
    """Times and logs the execution of a function

    Args:
        fn (callable): function to be timed
    """
    def wrapper(self, *args, **kwargs):
        _t0 = time.time_ns()
        ret = fn(self, *args, **kwargs)
        _t1 = time.time_ns()
        tCompute = (_t1 - _t0) / 1e6
        logger.trace(f"Function {fn.__name__} executed in {round(tCompute, 2)} ms")
        return ret
    return wrapper

@runOnce
def initLogging(level: int = 0, backtrace: bool = False, diagnose: bool = False, logfile: bool = False, fname: str = "simulation.log") -> None:
    logger.remove()
    
    # create timing level below trace
    logger.level("TIMING", no=2, color="<blue>")

    logger.add(sys.stderr, level=level, backtrace=backtrace, diagnose=diagnose, colorize=True)

    if logfile:
        logger.add(fname, level=level, backtrace=backtrace, diagnose=diagnose, colorize=False)