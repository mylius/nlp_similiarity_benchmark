import re
import spacy
import numpy as np
from sklearn import preprocessing
from scipy import sparse
<<<<<<< HEAD
from collections import OrderedDict
from collections import defaultdict
from collections import Counter
import warnings
from sklearn.utils.validation import DataConversionWarning
warnings.filterwarnings("ignore",category=DataConversionWarning)
=======
from sklearn.metrics.pairwise import cosine_similarity
from collections import OrderedDict
from collections import defaultdict
from collections import Counter
>>>>>>> cba3ad4330647d447a5a020f799a7ade5e37b174


class Algorithm:

    def __init__(self, name, language="english",):
        self.trained = False
        self.name = name
        self.language = language

    def train(self, in_dataset):
        raise NotImplementedError("Train method not implemented")

    def create_vec(self, in_line):
        """Returns a matrix denoting which words from the dictionary occure in a given line."""
        raise NotImplementedError("Create_vec method not implemented")

    def compare(self, a, b, func):
        """Returns the cosine similarity between two matrives a,b.
        Interestingly scipys cosine function doesn't work on scipys sparse matrices, while sklearns does."""
        return cosine_similarity(a, b)


class BagOfWords(Algorithm):

    def __init__(self, name="BagOfWords simple language agnostic", disable=["ner"], language="english", stop=True):
        super().__init__(name, language)
        self.dictionary = {}
        self.weights = []
        self.disable = disable
        self.stop = stop
        if self.language == "english":
            self.nlp = spacy.load("en_core_web_sm")
        elif self.language == "german":
            self.nlp = spacy.load("de_core_news_sm")
        else:
            raise ValueError("Unsupported language")

    def train(self, in_dataset):
        """Creates a dictionary of occuring words for a given dataset."""
        print("Training  {}".format(self.name))
        data = []
        for sets in in_dataset:
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

    def create_vec(self, in_line):
        """Returns a matrix denoting which words from the dictionary occure in a given line."""
        words = np.array(re.sub(r'\W+', ' ', in_line).split(" "))
        count = defaultdict(int)
        for token in words:
            count[token] += 1
        col = []
        row = []
        data = []
        for key, value in count.items():
            if key in self.dictionary:
                data.append(value)
                col.append(self.dictionary[key])
                row.append(0)
        return sparse.csr_matrix((data, (row, col)), shape=(1, len(self.dictionary)))


class BagOfWords_lemma(BagOfWords):

    def __init__(self, name="BagOfWords Lemmatized", disable=["ner"], language="english",stop=True):
        super().__init__(name, language)

    def train(self, in_dataset, stop=True):
        """Creates a dictionary of occuring words for a given dataset."""
        print("Training {}".format(self.name))
        data = ''
        for sets in in_dataset:
            for item in sets:
                data = data + item + " "
        doc = self.nlp(data, disable=self.disable)
        data = []
        for item in doc:
            if not item.is_stop and self.stop == True:
                data.append(item.lemma_)
        data, self.weights = np.unique(data, return_counts=True)
        index = 0
        for value in data:
            self.dictionary[value] = index
            index += 1
        self.weights = sparse.csr_matrix(
            preprocessing.minmax_scale(self.weights))
        self.trained = True

    def create_vec(self, in_line):
        """Returns a matrix denoting which words from the dictionary occure in a given line."""
        words = self.nlp(str(in_line), disable=self.disable)
        count = defaultdict(int)
        # Using counter proofed unsucessfull since it bypasses lemmatization
        for token in words:
            if not token.is_stop and self.stop == True:
                count[token.lemma_] += 1
        col = []
        row = []
        data = []
        for key, value in count.items():
            if key in self.dictionary:
                data.append(value)
                col.append(self.dictionary[key])
                row.append(0)
        return sparse.csr_matrix((data, (row, col)), shape=(1, len(self.dictionary)))


class spacy_sem_sim(Algorithm):

    def __init__(self, name="spacy", language="english", model = "md"):
        super().__init__(name, language)
        print("Initializing spacy model")
        if self.language == "english":
            self.nlp = spacy.load("en_core_web_{}".format(model))
        elif self.language == "german":
            self.nlp = spacy.load("de_core_news_{}".format(model))
        else:
            raise ValueError("Unsupported language")

    def compare(self, a, b, func):
        """Returns the cosine similarity between two matrices a,b."""
        return a.similarity(b)

    def create_vec(self, in_line):
        """Returns a matrix denoting which words from the dictionary occure in a given line."""
        return self.nlp(in_line)
    
    def train(self,in_dataset):
        pass

class spacy_bert(Algorithm):

    def __init__(self, name="spacy", language="english", model = "lg"):
        super().__init__(name, language)
        print("Initializing spacy model")
        if self.language == "english":
            self.nlp = spacy.load("en_trf_bertbaseuncased_{}".format(model))
        elif self.language == "german":
            self.nlp = spacy.load("en_trf_bertbaseuncased_lg{}".format(model))
        else:
            raise ValueError("Unsupported language")
    
    def compare(self, a, b, func):
        """Returns the cosine similarity between two matrices a,b."""
        return a.similarity(b)

    def create_vec(self, in_line):
        """Returns a matrix denoting which words from the dictionary occure in a given line."""
        return self.nlp(in_line)
    
    def train(self,in_dataset):
        pass
