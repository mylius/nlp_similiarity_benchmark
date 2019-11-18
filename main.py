import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from multiprocessing import Pool, Process, Manager
import os
import argparse
import algs
import importlib


class Dataset:
    def __init__(self, name):
        self.name = name
        self.data = [[]]

    def __str__(self):
        output = ""
        for i in range(len(self.data[0])):
            output += str(self.data[0][i])+"\n"
        return output

    def search(self, word):
        results = []
        for item in self.data:
            if word in item:
                results.append(item)
        return results

    def load_data(self, path):
        """Loads a list of strings."""
        file = open(path, "r")
        for line in file.readlines():
            self.data[0].append(line)


class Dataset_annot(Dataset):
    def __init__(self, name):
        self.scores = []
        self.ids = []
        self.name = name
        self.data = [[], []]
        self.norm_score = []
        self.vecs = {}

    def load_sick(self):
        """Loads the SICK dataset."""
        file = open("./data/SICK.txt", "r")
        i = 0
        for line in file.readlines():
            line = line.split("\t")
            self.data[0].append(line[1])
            self.data[1].append(line[2])
            self.scores.append(float(line[4]))
            self.ids.append(i)
            i += 1

    def load_sts(self):
        """Loads the STS-2017-en-en dataset."""
        file = open("./data/STS.input.track5.en-en.txt", "r")
        i = 0
        for line in file.readlines():
            line = line.split("\t")
            self.data[0].append(line[0])
            self.data[1].append(line[1])
            self.ids.append(i)
            i += 1
        file = open("./data/STS.gs.track5.en-en.txt", "r")
        for line in file.readlines():
            self.scores.append(float(line))
            i += 1

    def __str__(self):
        output = ""
        for i in range(len(self.data[0])):
            output += str(self.ids[i]) + " " + self.data[0][i] + \
                "\t " + self.data[1][i] + "\t " + self.scores[i] + "\n"
        return output

    def __getitem__(self, y):
        output = str(self.ids[y]) + " " + self.data[0][y] + \
            "\t " + self.data[1][y] + "\t " + self.scores[y] + "\n"
        return output

    def norm_scores(self):
        """Creates a list of normed scores."""
        self.norm_score = []
        self.norm_score = preprocessing.minmax_scale(self.scores)

    def __len__(self):
        return len(self.ids)

    def search(self, word):
        """Search for a word in the data set and returns a list of all entries containing the word."""
        results = []
        for i in range(len(self.data[0])):
            if word in self.data[0][i]:
                results.append(self.data[0][i])
            if word in self.data[1][i]:
                results.append(self.data[1][i])
        return results

    def run_alg(self, alg):
        """Runs a given algorithm and returns the difference to the ground truth."""
        results = []
        if not alg.trained:
            print("Training...")
            alg.train(self)
        if alg in self.vecs:
            data = self.vecs[alg]
        else:
            self.calc_vecs(alg)
        for i in range(len(self.data[0])):
            res = float(alg.compare(
                self.vecs[alg][0][i], self.vecs[alg][1][i]))
            results.append(res)
        return pearsonr(results, self.norm_score)

    def calc_vecs(self, alg):
        """Precalculates the vectors and stores them in memory."""
        if not alg.trained:
            print("Training...")
            alg.train(self)
        self.vecs[alg] = [[], []]
        print("Creating Vectors")
        self.vecs[alg][0] = multithread_shared_object(
            alg.create_vec, "list", self.data[0])
        self.vecs[alg][1] = multithread_shared_object(
            alg.create_vec, "list", self.data[1])


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


def inheritors(klass):
    subclasses = set()
    work = [klass]
    while work:
        parent = work.pop()
        for child in parent.__subclasses__():
            if child not in subclasses:
                subclasses.add(child)
                work.append(child)
    return subclasses


def benchmark(algorithms=[]):
    if algorithms:
        algorithms = path_import(args.path)
        algorithms = inheritors(algorithms.Algorithm)
    standard = inheritors(algs.Algorithm)
    db = Dataset_annot("sick")
    db.load_sick()
    db.norm_scores()
    print("Results for SICK dataset:")
    for i in range(len(standard)):
        print(standard[i].name + " correlation: " +
              str(db.run_alg(standard[i])))
    for i in range(len(algorithms)):
        print(algorithms[i].name + " correlation: " +
              str(db.run_alg(algorithms[i])))
    db2 = Dataset_annot("sts")
    db2.load_sts()
    db2.norm_scores()
    print("Results for STS dataset:")
    for i in range(len(standard)):
        print(standard[i].name + " correlation: " +
              str(db2.run_alg(standard[i])))
    for i in range(len(algorithms)):
        print(algorithms[i].name + " correlation: " +
              str(db2.run_alg(algorithms[i])))


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmarks Semantic Similiarty Benchmarks")
    parser.add_argument("path", metavar="path", type=str,nargs='?',
                        help="The path to a *.py with algorithms to benchmark. The Algorithms need to inherit from the Algorithm class.")
    args = parser.parse_args()
    benchmark(args.path)
