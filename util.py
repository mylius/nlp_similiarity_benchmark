import os
from multiprocessing import Pool, Process, Manager
import numpy as np
import time

def measure_time(string, func, *args):
    starttime = time.time()
    endtime = time.time()
    return endtime-starttime