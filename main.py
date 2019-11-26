import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import argparse
import algs
import util


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
        f = open(path, "r")
        for line in f.readlines():
            self.data[0].append(line)


class Dataset_annot(Dataset):
    def __init__(self, name):
        self.name = name
        self.train_data = [[], []]
        self.test_data = [[], []]
        self.train_score = []
        self.test_score = []
        self.cosine = {}
        self.phrase_vecs = {}

    def load(self, path, data_rows, data, score_row, scores):
        i = 0
        with open(path, "r") as f:
            for line in f.readlines():
                line = line.split("\t")
                j = 0
                for row in data_rows:
                    data[j].append(line[row])
                    j += 1
                scores.append((line[score_row]))
                i += 1

    def load_sick(self):
        """Loads the SICK dataset."""
        self.load("./data/SICK_train.txt", [1, 2],
             self.train_data, 3, self.train_score)
        self.load("./data/SICK.txt",[1, 2], self.test_data, 4, self.test_score)

    """def load_sts(self):
        #Loads the STS dataset.
        self.load("./data/sts_trial/SICK_train.txt", [1, 2],
             self.train_data, 3, self.train_score)
        self.load("./data/SICK.txt",[1, 2], self.test_data, 4, self.test_score)"""

    def norm_scores(self):
        """Creates a list of normed scores."""
        self.train_norm_score = []
        self.test_norm_score = []
        self.train_norm_score = preprocessing.minmax_scale(self.train_score)
        self.test_norm_score = preprocessing.minmax_scale(self.test_score)


    def calc_vecs(self, alg):
        """Precalculates the vectors and stores them in memory."""
        if not alg.trained:
            alg.train(self.train_data)
        self.phrase_vecs[alg] = [[], []]
        print("Creating Vectors")
        for item in self.test_data[0]:
            self.phrase_vecs[alg][0].append(alg.create_vec(item))
        for item in self.test_data[1]:
            self.phrase_vecs[alg][1].append(alg.create_vec(item))


    def calc_cosine(self, alg):
        """Runs a given algorithm and returns the difference to the ground truth."""
        results = []
        if not alg.trained:
            print("Training...")
            alg.train(self.train_data)
        if alg in self.phrase_vecs:
            data = self.phrase_vecs[alg]
        else:
            self.calc_vecs(alg)
        for i in range(len(self.test_data[0])):
            res = float(alg.compare(
                self.phrase_vecs[alg][0][i], self.phrase_vecs[alg][1][i]))
            results.append(res)
        self.cosine[alg] = results


    def compare(self, function, alg):
        if alg not in self.cosine:
            self.calc_cosine(alg)
        return function(self.cosine[alg], self.test_norm_score)


def benchmark():
    standard = util.inheritors(algs.Algorithm)
    """db2 = Dataset_annot("sts")
    db2.load_sts()
    db2.norm_scores()
    print("Results for STS dataset:")
    for i in range(len(standard)):
        print(standard[i].name + " correlation: " +
              str(db2.compare(pearsonr, standard[i])))"""
    db = Dataset_annot("sick")
    db.load_sick()
    db.norm_scores()
    print("Results for SICK dataset:")
    for i in range(len(standard)):
        standard[i].train(db.train_data)
        print(standard[i].name + " correlation: " +
              str(db.compare(pearsonr, standard[i])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmarks Semantic Similiarty Benchmarks")
    parser.add_argument("path", metavar="path", type=str, nargs='?',
                        help="The path to a *.py with algorithms to benchmark. The Algorithms need to inherit from the Algorithm class.")
    args = parser.parse_args()
    benchmark()
