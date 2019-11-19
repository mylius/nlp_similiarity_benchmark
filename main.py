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
        self.cosine = {}

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
            output += str(self.ids[i]) + " " + str(self.data[0][i]) + \
                "\t " + str(self.data[1][i]) + "\t " + str(self.scores[i]) + "\n"
        return output

    def __getitem__(self, y):
        output = str(self.ids[y]) + " " + str(self.data[0][y]) + \
            "\t " + str(self.data[1][y]) + "\t " + str(self.scores[y]) + "\n"
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

    def calc_cosine(self, alg):
        """Runs a given algorithm and returns the difference to the ground truth."""
        results = []
        if not alg.trained:
            print("Training...")
            alg.train(self.data)
        if alg in self.vecs:
            data = self.vecs[alg]
        else:
            self.calc_vecs(alg)
        for i in range(len(self.data[0])):
            res = float(alg.compare(
                self.vecs[alg][0][i], self.vecs[alg][1][i]))
            results.append(res)
        self.cosine[alg] = results
    
    def compare(self, function, alg):
        if alg not in self.cosine:
            self.calc_cosine(alg)
        return function(self.cosine[alg],self.norm_score)

    def calc_vecs(self, alg):
        """Precalculates the vectors and stores them in memory."""
        if not alg.trained:
            alg.train(self.data)
        self.vecs[alg] = [[], []]
        print("Creating Vectors")
        self.vecs[alg][0] = util.multithread_shared_object(
            alg.create_vec, "list", self.data[0])
        self.vecs[alg][1] = util.multithread_shared_object(
            alg.create_vec, "list", self.data[1])

class Dataset_annot_tt(Dataset_annot):
    def __init__(self, name):
        super().__init(name)
        self.train_data = [[], []]
        self.test_data = [[], []]
        self.train_norm_score = []
        self.test_norm_score = []


    def load(self,path,data_rows, data,score_row, scores):
        file = open("./data/SICK.txt", "r")
        i = 0
        for line in file.readlines():
            line = line.split("\t")
            j=0
            for row in data_rows:
                data.append(line[row])
                j+=1
            scores.append((line[score_row]))
            i += 1
    
    def load_sick():
        """Loads the SICK dataset."""
        load("./data/SICK_train.txt",[1,2],self.train_data,4,self.train_norm_score)
        load("./data/SICK.txt"[1,2],self.test_data,4,self.test_norm_score)

    def calc_cosine(self, alg):
        """Runs a given algorithm and returns the difference to the ground truth."""
        results = []
        if not alg.trained:
            print("Training...")
            alg.train(self.train_data)
        if alg in self.vecs:
            data = self.vecs[alg]
        else:
            self.calc_vecs(alg)
        for i in range(len(self.train_data[0])):
            res = float(alg.compare(
                self.vecs[alg][0][i], self.vecs[alg][1][i]))
            results.append(res)
        self.cosine[alg] = results




def benchmark(algorithms=[]):
    if algorithms:
        algorithms = util.path_import(args.path)
        algorithms = util.inheritors(algorithms.Algorithm)
        length = lenght(algorithms)
    else:
        length = 0
    standard = util.inheritors(algs.Algorithm)
    db2 = Dataset_annot("sts")
    db2.load_sts()
    db2.norm_scores()
    print("Results for STS dataset:")
    for i in range(len(standard)):
        print(standard[i].name + " correlation: " +
              str(db2.compare(pearsonr,standard[i])))
    for i in range(length):
        print(algorithms[i].name + " correlation: " +
              str(db2.run_alg(pearsonr,algorithms[i])))
    db = Dataset_annot("sick")
    db.load_sick()
    db.norm_scores()
    print("Results for SICK dataset:")
    for i in range(len(standard)):
        standard[i].train(db.data)
        print(standard[i].name + " correlation: " +
              str(db.compare(pearsonr,standard[i])))
    for i in range(length):
        algorithms[i].train(db.data)
        print(algorithms[i].name + " correlation: " +
              str(db.compare(pearsonr,algorithms[i])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmarks Semantic Similiarty Benchmarks")
    parser.add_argument("path", metavar="path", type=str, nargs='?',
                        help="The path to a *.py with algorithms to benchmark. The Algorithms need to inherit from the Algorithm class.")
    args = parser.parse_args()
    benchmark(args.path)
