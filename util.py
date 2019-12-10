import os
from multiprocessing import Pool, Process, Manager
import numpy as np
import importlib
import time


def inheritors(klass):
    """Implementation based on https://stackoverflow.com/questions/5881873/python-find-all-classes-which-inherit-from-this-one"""
    subclasses = set()
    work = [klass]
    while work:
        parent = work.pop()
        for child in parent.__subclasses__():
            if child not in subclasses:
                init = child()
                subclasses.add(init)
                work.append(child)
    return list(subclasses)

def measure_time(string, func, *args):
    starttime = time.time()
    result = func(*args)
    endtime = time.time()
    print("{}: {}s".format(string,endtime-starttime))
    return result