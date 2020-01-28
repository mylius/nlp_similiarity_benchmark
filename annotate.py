import numpy as np
import algs
import hashlib
import json
from os import path
from collections import defaultdict
from scipy import special
from itertools import permutations


class Dataset:
    """
    A class that stores a not yet annotated dataset of strings, which allows you to annotate the data.
    The data is annotated by comparing  two sentece to a third reference sentence.
    When annotating the annotations are compared to spacy's w2v implementation.
    The program can be quit by entering "q" during annotation and a triple can be skipped by entering "s".
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
        self.annots = defaultdict(int)
        self.phrase_vecs = {}
        self.correctness = []
        # This is a threshold. If difference in similarity between two sentences is bigger than this and the user still disagress with the annotation the script will save the ids.
        self.rec_thresh = 0.12
        self.strong_disagreement = []

    def __str__(self):
        """Returns all strings in the dataset."""
        output = ""
        for value in self.data:
            output += str(value) + "\n"
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
            alg.train(self.data,[])
        self.phrase_vecs[alg] = []
        print("Creating Vectors")
        for item in self.data:
            self.phrase_vecs[alg].append(alg.create_vec(item))

    def calc_scores(self, alg):
        """
        Calculates the similarity scores for all sentence pairs and stores them using the calculated hash as the filename.
        """
        self.calc_vecs(alg)
        self.refscores = np.zeros((self.id_len, self.id_len))
        for idx, vec1 in enumerate(self.phrase_vecs[alg]):
            for idy, vec2 in enumerate(self.phrase_vecs[alg]):
                self.refscores[idx][idy] = alg.compare(vec1, vec2)
        with open("./data/{}-scores.json".format(self.hash), "w+") as f:
            json.dump(self.refscores.tolist(), f, indent=2)

    def run_annot(self):
        """Randomly selects 3 sentences from self.data and asks the user to evaluate which sentence is more simmilar to a given reference sentence.
        The results are stored in a 3d array:
            1st dim: reference sentence id
            2nd dim: id with the more simmilar sentence
            3rd dim: id with the less similar sentence
        If you wanted to compare two sentences you would just check which one is in self.annots."""
        run = True
        alg = algs.spacy_sem_sim(language="german")
        self.id_len = len(self.data)
        self.load_files()
        while run:
            # check if all combinations have been annotated:
            if len(self.annots) >= self.id_len * special.binom(self.id_len - 1, 2):
                print("All possible combinations have been annotated.")
                break
            sentence_ids = list(np.random.choice(self.id_len, 3, replace=False))
            # make sure the triplet is new.
            if self.annots != {}:
                while (
                    str((sentence_ids[0], sentence_ids[1], sentence_ids[2]))
                    in self.annots
                    or str((sentence_ids[0], sentence_ids[2], sentence_ids[1]))
                    in self.annots
                    or sentence_ids[0] == sentence_ids[1]
                    or sentence_ids[0] == sentence_ids[2]
                    or sentence_ids[1] == sentence_ids[2]
                ):
                    sentence_ids = list(np.random.choice(self.id_len, 3, replace=False))
            sentences = np.array(self.data)[sentence_ids]
            answer = ""
            while answer not in ["quit", "q", "s", "1", "2"]:
                print(
                    "Reference:\n {}\n\n".format(sentences[0]),
                    "Sentence 1:\n {}\n\n".format(sentences[1]),
                    "Sentence 2:\n {}".format(sentences[2]),
                )
                answer = input().lower()
                if answer == "quit" or answer == "q":
                    run = False
                elif answer == "1":
                    print("Sentence 1 is more similar to reference sentence.\n")
                    self.annots[
                        str((sentence_ids[0], sentence_ids[1], sentence_ids[2]))
                    ] += 1
                    self.save_correct(sentence_ids[0], sentence_ids[1], sentence_ids[2])
                elif answer == "2":
                    self.annots[
                        str((sentence_ids[0], sentence_ids[2], sentence_ids[1]))
                    ] += 1
                    print("Sentence 2 is more similar to reference sentence.\n")
                    self.save_correct(sentence_ids[0], sentence_ids[2], sentence_ids[1])
                else:
                    print("Unrecognized answer.")
        self.save_results()

    def save_results(self):
        """ Saves the current results of annotations, their correctness and strong disagreements. """
        with open("./data/{}-annots.json".format(self.hash), "w+") as f:
            json.dump(self.annots, f, indent=2)
        with open("./data/{}-correctness.json".format(self.hash), "w+") as f:
            json.dump(self.correctness, f, indent=2)
        with open("./data/{}-disagreement.json".format(self.hash), "w+") as f:
            json.dump(self.strong_disagreement, f, indent=2)

    def load_files(self):
        """ Saves the saved results of annotations, their correctness and strong disagreements. """
        if path.exists("./data/{}-scores.json".format(self.hash)):
            with open("./data/{}-scores.json".format(self.hash)) as f:
                self.refscores = np.array(json.load(f))
        else:
            self.calc_scores(alg)
        if path.exists("./data/{}-annots.json".format(self.hash)):
            with open("./data/{}-annots.json".format(self.hash)) as f:
                self.annots = defaultdict(int, json.load(f))
        if path.exists("./data/{}-correctness.json".format(self.hash)):
            with open("./data/{}-correctness.json".format(self.hash)) as f:
                self.correctness = json.load(f)
        if path.exists("./data/{}-disagreement.json".format(self.hash)):
            with open("./data/{}-disagreement.json".format(self.hash)) as f:
                self.strong_disagreement = json.load(f)
                print(self.strong_disagreement)

    def save_correct(self, ref_id, sent_id1, sent_id2):
        """
        Checks if the users annotations matches the algorithms and saves whether the annotation matched.
        If the algorithm is very confident and the user disagrees the triplet is stored.

        Parameters
        ----------
        ref_id : Id of the reference sentence.
        sent_id1 : Id of the sentence the user thought to be more like the reference sentence.
        sent_id2 : Id of the sentence the user thought to be less like the reference sentence.
        """
        if self.refscores[ref_id][sent_id1] > self.refscores[ref_id][sent_id2]:
            print("Matching expectation!")
            self.correctness.append(1)
        else:
            print("Unexpected annotation!")
            self.correctness.append(0)

        if (
            self.refscores[ref_id][sent_id2] - self.refscores[ref_id][sent_id1]
            > self.rec_thresh
        ):
            print("Registered strong disagreement with algorithm")
            self.strong_disagreement.append(str((ref_id, sent_id1, sent_id2)))

    def evaluate(self):
        """Prints the percentage of correct annotations and a list of pairings where the user disagreed eventhough the algorithm had a clear preference."""
        if len(self.correctness) > 0:
            print(
                "Correct percentage is: {}%".format(
                    round(sum(self.correctness) / len(self.correctness) * 100, 2)
                )
            )
            print(
                "Triplets were the algorithm was pretty certain, but the annotation disagreed."
            )
            print(self.strong_disagreement)
            # one possible way to create scores.





if __name__ == "__main__":
    db = Dataset("nachrichten")
    db.load_data("./data/nachrichten.txt")
    db.run_annot()
    db.evaluate()
