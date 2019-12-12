import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import argparse
import algs
import util
from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_similarity
import time


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
        with open(path, "r") as f:
            for line in f.readlines():
                self.data[0].append(line)


class Dataset_annot(Dataset):
    def __init__(self, name):
        self.name = name
        self.train_data = [[], []]
        self.test_data = [[], []]
        self.train_score = []
        self.test_score = []
        self.train_ids = []
        self.test_ids = []
        self.results = {}
        self.phrase_vecs = {}

    def load(self, path, data_rows, data, score_row, scores, id_row=None, ids=None):
        with open(path, "r") as f:
            next(f)
            for line in f.readlines():
                line = line.split("\t")
                j = 0
                for row in data_rows:
                    data[j].append(line[row])
                    j += 1
                scores.append((line[score_row]))
                if id_row != None:
                    ids.append(line[id_row])

    def load_sick(self):
        """Loads the SICK dataset."""
        self.load("./data/SICK_train.txt", [1, 2],
                  self.train_data, 3, self.train_score, 0, self.train_ids)
        self.load("./data/SICK.txt", [1, 2],
                  self.test_data, 3, self.test_score, 0, self.test_ids)
        self.sick = True

    def load_sts(self):
        # Loads the STS dataset.
        self.load("./data/sts_train/raw_gs.txt", [1, 2],
                  self.train_data, 0, self.train_score)
        self.load("./data/sts_test/raw_gs.txt", [1, 2],
                  self.test_data, 0, self.test_score)

    def norm_scores(self):
        """Creates lists of normed scores."""
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

    def calc_results(self, alg):
        """Runs a given algorithm and returns the difference to the ground truth."""
        results = []
        if not alg.trained:
            alg.train(self.train_data)
        if alg in self.phrase_vecs:
            data = self.phrase_vecs[alg]
        else:
            self.calc_vecs(alg)
        for i in range(len(self.test_data[0])):
            res = float(alg.compare(
                self.phrase_vecs[alg][0][i], self.phrase_vecs[alg][1][i]))
            results.append(res)
        self.results[alg] = results

    def compare(self, function, alg):
        if alg not in self.results:
            self.calc_results(alg)
        return function(self.results[alg], self.test_norm_score)

    def output_sick(self, function, alg):
        if self.sick:
            with open("./data/results_SICK_{}".format(alg.name), "w+") as data:
                output = "pair_ID \t entailment_judgment \t relatedness_score\n"
                for i in range(len(self.results[alg])):
                    output += "{} \t NA \t {}\n".format(
                        self.test_ids[i], self.results[alg][i]*4+1)
                data.write(output)


def benchmark(algorithms):
    db2 = Dataset_annot("sts")
    db2.load_sts()
    db2.norm_scores()
    print("Results for STS dataset:")
    for i in range(len(algorithms)):
        util.measure_time("Traintime",algorithms[i].train,db2.train_data)
        starttime = time.time()
        print("{} correlation: \n Pearson:{} \n Spearman: {}\n MSE: {}".format(
            algorithms[i].name, db2.compare(pearsonr, algorithms[i]),
            db2.compare(spearmanr, algorithms[i]),
            db2.compare(mean_squared_error, algorithms[i])))
        endtime = time.time()
        print("Runtime: {}s".format(endtime-starttime))

    db = Dataset_annot("sick")
    db.load_sick()
    db.norm_scores()
    print("Results for SICK dataset:")
    for i in range(len(algorithms)):
        util.measure_time("Traintime",algorithms[i].train,db.train_data)
        starttime = time.time()
        print("{} correlation: \n Pearson:{} \n Spearman: {}\n MSE: {}".format(
            algorithms[i].name, db.compare(pearsonr, algorithms[i]),
            db.compare(spearmanr, algorithms[i]),
            db.compare(mean_squared_error, algorithms[i])))
        endtime = time.time()
        print("Runtime: {}s".format(endtime-starttime))


def create_alg_list(in_list):
    alg_list = []
    Algorithms = OrderedDict()
    Algorithms["bow"] = algs.BagOfWords
    Algorithms["bow_s"] = algs.BagOfWords_stop
    Algorithms["bow_j"] = algs.BagOfWords_jaccard
    Algorithms["bow_j_s"] = algs.BagOfWords_jaccard_stop
    Algorithms["bow_l"] = algs.BagOfWords_lemma
    Algorithms["bow_ls"] = algs.BagOfWords_lemma_stop
    Algorithms["bow_j_l"] = algs.BagOfWords_jaccard_lemma
    Algorithms["bow_j_ls"] = algs.BagOfWords_jaccard_lemma_stop
    Algorithms["spacy_w2v"] = algs.spacy_sem_sim
    Algorithms["spacy_bert"] = algs.spacy_bert
    if in_list != None:
        in_list = in_list.split(",")
        for alg in in_list:
            if alg in Algorithms:
                alg_list.append(Algorithms[alg]())
    else:
        for alg in Algorithms:
            alg_list.append(Algorithms[alg]())
    return alg_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmarks Semantic Similiarty Benchmarks")
    parser.add_argument("algs", metavar="algs", type=str, nargs='?',
                        help="Choose which Algorithms to run buy passing arguments: bow - simple bag of words, bow_l - bag of words using lemmatisation, bow_ls - bag of words eliminating stopwords using lemmatisation and",)
    args = parser.parse_args()
    benchmark(create_alg_list(args.algs))
