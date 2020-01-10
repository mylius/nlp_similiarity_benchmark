import numpy as np
import algs
import hashlib
import json
from os import path
from collections import defaultdict

class Dataset:
    """
    A class that stores a not yet annotated dataset of strings, which allows you to annotate the data.
    The data is annotated by comparing  two sentece to a third reference sentence.
    When annotating the annotations are compared to spacy's w2v implementation.
    On loading a dataset a hash is created. The hash is used to store the annotations and the results expected based on w2v.
    This allows seamless work with different datasets.

    Parameters
    ----------
    name : A name for the dataset.


    Example
    ----------
    db = Dataset_annot("test")
    db.load_sick("./data/data.txt")
    db.run_annot()
    """
    def __init__(self, name):
        self.name = name
        self.data = []
        self.annots = None
        self.phrase_vecs = {}

    def __str__(self):
        """Returns all strings in the dataset."""
        output = ""
        for value in self.data:
            output += str(value)+"\n"
        return output

    def search(self, word):
        """
        Returns all strings containing a specific word.
        
         Parameters
        ----------
        word : The word to be searched for.
        """
        results = []
        for item in self.data:
            if word in item:
                results.append(item)
        return results

    def load_data(self, path):
        """
        Loads a list of strings into self.data and creates a hash for it.
        
        Parameters
        ----------
        path : A string denoting the path to the file to be loaded.
        """
        with open(path, "r") as f:
            for line in f.readlines():
                self.data.append(line)
        data_str = str(self.data)
        data_str = data_str.encode("utf-8")
        self.hash = hashlib.md5(data_str).hexdigest()

    def calc_vecs(self, alg):
        """
        Precalculates the vectors and stores them in memory.
        
        Parameters
        ----------
        alg : An algorithm to be used to create the vectors. Training can't need a training dataset.
            BERT and w2v are good candidates since they come with pretrained models.
        """
        if not alg.trained:
            alg.train(self.data)
        self.phrase_vecs[alg] = []
        print("Creating Vectors")
        for item in self.data:
            self.phrase_vecs[alg].append(alg.create_vec(item))

    def calc_scores(self):
        """
        Calculates the similarity scores for all sentence pairs and stores them using the calculated hash as the filename.
        """
        self.calc_vecs(alg)
        self.refscores = np.zeros((id_len, id_len))
        for idx, vec1 in enumerate(self.phrase_vecs[alg]):
            for idy, vec2 in enumerate(self.phrase_vecs[alg]):
                self.refscores[idx][idy] = alg.compare(
                    vec1, vec2)
        with open("./data/{}-scores.json".format(self.hash), "w+") as f:
            json.dump(self.refscores.tolist(), f, indent=2)


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
        id_len = len(self.data)
        if path.exists("./data/{}-scores.json".format(self.hash)):
            with open("./data/{}-scores.json".format(self.hash)) as f:
                self.refscores = np.array(json.load(f))
        else:
            self.calc_scores()
        if path.exists("./data/{}-annots.json".format(self.hash)):
            with open("./data/{}-annots.json".format(self.hash)) as f:
                self.annots = defaultdict(int, json.load(f))
        else:
            self.annots = defaultdict(int)
        while run:
            sentence_ids = np.random.choice(id_len, 3)
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
                        print("Matching expectation!")
                    else:
                        print("Unexpected annotation!")
                elif answer == "2":
                    self.annots[str((sentence_ids[0],
                                     sentence_ids[2], sentence_ids[1]))] += 1
                    print("Sentence 2 is more similar to reference sentence.\n")
                    if self.refscores[sentence_ids[0]][sentence_ids[2]] > self.refscores[sentence_ids[0]][sentence_ids[1]]:
                        print("Matching expectation!")
                    else:
                        print("Unexpected annotation!")
                else:
                    print("Unrecognized answer.")
        with open("./data/{}-annots.json".format(self.hash), "w+") as f:
            json.dump(self.annots, f, indent=2)

if __name__ == "__main__":
    db = Dataset("nachrichten")
    db.load_data("./data/nachrichten.txt")
    db.run_annot()