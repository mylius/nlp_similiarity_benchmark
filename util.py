import os
from multiprocessing import Pool, Process, Manager
import numpy as np
import importlib


def multithread_shared_object(function, s_type, iterable, arguments=None, not_util=2):
    """Hand a shared object of s_type and an iterable to a function be processed in parallel."""
    manager = Manager()
    # assign shared resource
    if s_type == "list":
        shared = manager.list(range(len(iterable)))
    if s_type == "dict":
        shared = manager.dict()
    # if threads > 2 reserve the number specified in not_util, use the rest
    if len(os.sched_getaffinity(0)) > 2:
        cpus = len(os.sched_getaffinity(0))-not_util
    else:
        cpus = len(os.sched_getaffinity(0))
    processes = []
    # split iterable into parts
    split_iter = np.array_split(np.array(iterable), cpus)
    # create process, start and join them
    for i in range(cpus):
        if arguments != None:
            p = Process(target=function, args=(
                [split_iter[i], shared, arguments, i]))
        else:
            p = Process(target=function, args=([split_iter[i], shared, i]))
        processes.append(p)
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    return shared


def path_import(absolute_path):
    """implementation taken from https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly"""
    spec = importlib.util.spec_from_file_location(absolute_path, absolute_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
