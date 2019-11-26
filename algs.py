import re
import spacy
import numpy as np
from sklearn import preprocessing
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from collections import OrderedDict
from collections import defaultdict
from collections import Counter

class Algorithm:

    def __init__(self, name, language="english",):
        self.trained = False
        self.name = name
        self.language = language

    def train(self, input):
        pass

    def compare(self, a, b):
        """Returns the cosine similarity between two matrives a,b.
        Interestingly scipys cosine function doesn't work on scipys sparse matrices, while sklearns does."""
        return cosine_similarity(a, b)


class BagOfWords(Algorithm):

    def __init__(self, name="BagOfWords simple language agnostic", disable=["ner"], language="english"):
        super().__init__(name, language)
        self.dictionary = {}
        self.weights = []
        self.disable = disable
        if self.language == "english":
            self.nlp = spacy.load("en_core_web_sm")
        if self.language == "german":
            self.nlp = spacy.load("de_core_news_sm")

    def train(self, input):
        """Creates a dictionary of occuring words for a given dataset."""
        print("Training " + self.name)
        data = []
        for sets in input:
            for item in sets:
                data += re.sub(r'\W+', ' ', item).split(" ")
        data, self.weights = np.unique(data, return_counts=True)
        index = 0
        for value in data:
            self.dictionary[value] = index
            index += 1
        self.weights = sparse.csr_matrix(
            preprocessing.minmax_scale(self.weights))
        self.trained = True

    def append_dic(self, data, dict, i):
        for token in data:
            dict.append(token.lemma_)

    def create_vec(self, input):
        """Returns a matrix denoting which words from the dictionary occure in a given line."""
        words = np.array(re.sub(r'\W+', ' ', input).split(" "))
        count = defaultdict(int)
        result = OrderedDict()
        for token in words:
            count[token] += 1
        for key, value in count.items():
            if key in self.dictionary:
                result[self.dictionary[key]] = value
        result = OrderedDict(sorted(result.items()))
        col = []
        row = []
        data = []
        for key, value in result.items():
            data.append(value)
            col.append(key)
            row.append(0)
        return sparse.csr_matrix((data, (row, col)), shape=(1, len(self.dictionary)))


class BagOfWords_lemma(BagOfWords):

    def __init__(self, name="BagOfWords Lemmatized", disable=["ner"], language="english",):
        super().__init__(name, language)

    def train(self, input, stop=True):
        """Creates a dictionary of occuring words for a given dataset."""
        print("Training " + self.name)
        data = ''
        for sets in input:
            for item in sets:
                data = data + item + " "
        doc = self.nlp(data, disable=self.disable)
        data = []
        for item in doc:
            if not item.is_stop:
                data.append(item.lemma_)
        data, self.weights = np.unique(data, return_counts=True)
        index = 0
        for value in data:
            self.dictionary[value] = index
            index += 1
        self.weights = sparse.csr_matrix(
            preprocessing.minmax_scale(self.weights))
        self.trained = True

    def append_dic(self, data, dict, i):
        for token in data:
            dict.append(token.lemma_)

    def create_vec(self, input):
        """Returns a matrix denoting which words from the dictionary occure in a given line."""
        words = self.nlp(str(input), disable=self.disable)
        count = defaultdict(int)
        result = OrderedDict()
        #Using counter proofed unsucessfull since it bypasses lemmatization
        for token in words:
            if not token.is_stop:
                count[token.lemma_] += 1
        for key, value in count.items():
            if key in self.dictionary:
                result[self.dictionary[key]] = value
        result = OrderedDict(sorted(result.items()))
        col = []
        row = []
        data = []
        for key, value in result.items():
            data.append(value)
            col.append(key)
            row.append(0)
        return sparse.csr_matrix((data, (row, col)), shape=(1, len(self.dictionary)))

    