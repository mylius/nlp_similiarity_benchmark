import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import argparse
import algs
import hashlib
import util
from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_similarity
import time
import json
from os import path
import pickle
from collections import defaultdict

class Dataset:
    def __init__(self, name):
        self.name = name
        self.data = []
        self.ids = []
        self.annots = None
        self.phrase_vecs = {}

    def __str__(self):
        output = ""
        for value in self.data:
            output += str(value)+"\n"
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
            for idx, line in enumerate(f.readlines()):
                self.data.append(line)
                self.ids.append(idx)
        data_str = str(self.data)
        data_str = data_str.encode("utf-8")
        self.hash = hashlib.md5(data_str).hexdigest()

    def calc_vecs(self, alg):
        """Precalculates the vectors and stores them in memory."""
        if not alg.trained:
            alg.train(self.data)
        self.phrase_vecs[alg] = []
        print("Creating Vectors")
        for item in self.data:
            self.phrase_vecs[alg].append(alg.create_vec(item))


    def run_annot(self):
        """Randomly selects 3 sentences from self.data and asks the user to evaluate which sentence is more simmilar to a given reference sentence.
        The results are stored in a 3d array:
            1st dim: reference sentence id
            2nd dim: id with the more simmilar sentence
            3rd dim: id with the less similar sentence
        If you wanted to compare two sentences you could compare:
        self.annot[x][y][z] to self.annot[x][z][y]"""
        run = True
        alg = algs.spacy_sem_sim(language="german")
        id_len = len(self.ids)
        if path.exists("./data/{}-scores.json".format(self.hash)):
            with open("./data/{}-scores.json".format(self.hash)) as f:
                self.refscores = np.array(json.load(f))
        else:
            self.calc_vecs(alg)
            self.refscores = np.zeros((id_len+1, id_len+1))
            for vec1 in self.phrase_vecs[alg]:
                for vec2 in self.phrase_vecs[alg]:
                    self.refscores[row][col] = alg.compare(
                        vec1,vec2)
            with open("./data/{}-scores.json".format(self.hash),"w+") as f:
               json.dump(self.refscores.tolist(),f, indent =2)
        if path.exists("./data/{}-annots.json".format(self.hash)):
            with open("./data/{}-annots.json".format(self.hash)) as f:
                self.annots = defaultdict(int,json.load(f))
        else:
            self.annots = defaultdict(int)
        while run:
            sentence_ids = np.random.choice(self.ids, 3)
            sentences = np.array(self.data)[sentence_ids]
            answer = ""
            while answer not in ["quit", "q", "s", "1", "2"]:
                print("Reference:\n {}\n\n".format(sentences[0]), "Sentence 1:\n {}\n\n".format(
                    sentences[1]), "Sentence 2:\n {}".format(sentences[2]))
                answer = input().lower()
                if answer == "quit" or answer == "q":
                    run = False
                elif answer == "1":
                    print("Sentence 1 is more similar to reference sentence.\n")
                    self.annots[str((sentence_ids[0],
                                sentence_ids[1], sentence_ids[2]))] += 1
                    if self.refscores[sentence_ids[0]][sentence_ids[1]] > self.refscores[sentence_ids[0]][sentence_ids[2]]:
                        print("True!")
                    else:
                        print("False!")
                elif answer == "2":
                    self.annots[str((sentence_ids[0],
                                sentence_ids[2], sentence_ids[1]))] += 1
                    print("Sentence 2 is more similar to reference sentence.\n")
                    if self.refscores[sentence_ids[0]][sentence_ids[2]] > self.refscores[sentence_ids[0]][sentence_ids[1]]:
                        print("True!")
                    else:
                        print("False!")
                else:
                    print("Unrecognized answer.")
        with open("./data/{}-annots.json".format(self.hash),"w+") as f:
               json.dump(self.annots,f, indent =2)
        


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
        self.sick = False
        self.sts = False

    def load(self, path, data_rows, data, score_row, scores, id_row=None, ids=None):
        with open(path, "r") as f:
            next(f)
            for line in f.readlines():
                line = line.split("\t")
                for idx, row in enumerate(data_rows):
                    data[idx].append(line[row])
                scores.append(float(line[score_row]))
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
        self.sts = True

    def norm_results(self, feature_range):
        """Creates lists of normed scores."""
        self.normed_results = {}
        for key in self.results:
            if key not in self.normed_results:
                self.normed_results[key] = preprocessing.minmax_scale(
                    self.results[key], feature_range)

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
        for vec1,vec2 in zip(self.phrase_vecs[alg][0],self.phrase_vecs[alg][1]):
            res = float(alg.compare(
                vec1, vec2))
            results.append(res)
        self.results[alg] = results

    def compare(self, function, alg):
        if alg not in self.results:
            self.calc_results(alg)
        if self.sick:
            self.norm_results((1, 5))
        elif self.sts:
            self.norm_results((0, 5))
        return function(self.normed_results[alg], self.test_score)

    def output_sick(self, alg):
        if self.sick:
            with open("./data/results_SICK_{}".format(alg.name), "w+") as data:
                output = "pair_ID \t entailment_judgment \t relatedness_score\n"
                for idx,res in zip(self.test_ids,self.results[alg]):
                    output += "{} \t NA \t {}\n".format(
                        idx, res*4+1)
                data.write(output)


def run_alg(alg, db):
    result = {}
    result["traintime"] = round(util.measure_time(
        "Traintime", alg.train, db.train_data),3)
    starttime = time.time()
    result["pearson"] = round(db.compare(pearsonr, alg)[0],3)
    result["spearman"] = round(db.compare(spearmanr, alg)[0],3)
    result["mre"] = round(db.compare(mean_squared_error, alg),3)
    endtime = time.time()
    result["runtime"] = round(endtime-starttime,3)
    result["alg"] = alg.name
    result["db"] = db.name

    return result


def benchmark(algorithms):
    db2 = Dataset_annot("sts")
    db2.load_sts()
    print("Results for STS dataset:")
    run_results = {}
    for alg in algorithms:
        run_results[alg.name +
                    db2.name] = run_alg(alg, db2)
    db = Dataset_annot("sick")
    db.load_sick()
    print("Results for SICK dataset:")
    for alg in algorithms:
        run_results[alg.name + db.name] = run_alg(alg, db)
        #db.output_sick(alg)
    output = []
    for res in run_results:
        output.append(run_results[res])
    with open("./data/results.json", "w+") as f:
        json.dump(output,f,indent=2)


def create_alg_list(in_list):
    alg_list = []
    Algorithms = OrderedDict()
    Algorithms["bow"] = algs.BagOfWords
    Algorithms["bow_s"] = algs.BagOfWords_stop
    Algorithms["bow_l"] = algs.BagOfWords_lemma
    Algorithms["bow_ls"] = algs.BagOfWords_lemma_stop
    Algorithms["bow_j"] = algs.BagOfWords_jaccard
    Algorithms["bow_j_s"] = algs.BagOfWords_jaccard_stop
    Algorithms["bow_j_l"] = algs.BagOfWords_jaccard_lemma
    Algorithms["bow_j_ls"] = algs.BagOfWords_jaccard_lemma_stop
    Algorithms["bow_l2"] = algs.BagOfWords_l2
    Algorithms["bow_l2_s"] = algs.BagOfWords_l2_stop
    Algorithms["bow_l2_l"] = algs.BagOfWords_l2_lemma
    Algorithms["bow_l2_ls"] = algs.BagOfWords_l2_lemma_stop
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
                        help="Choose which Algorithms to run by passing arguments: bow - simple bag of words, bow_l - bag of words using lemmatisation, bow_ls - bag of words eliminating stopwords using lemmatisation and",)
    args = parser.parse_args()
    if args.algs != None:
        benchmark(create_alg_list(args.algs))
    db = Dataset("nachrichten") 
    db.load_data("./data/nachrichten.txt")
    db.run_annot()