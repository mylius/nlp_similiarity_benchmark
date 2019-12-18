import util
from functools import lru_cache
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
import re
import spacy
import numpy as np
from sklearn import preprocessing
from scipy import sparse
from collections import defaultdict
from collections import Counter
import warnings
from sklearn.utils.validation import DataConversionWarning
warnings.filterwarnings("ignore", category=DataConversionWarning)


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

    def compare(self, a, b):
        """Returns the cosine similarity between two matrices a,b."""
        return cosine_similarity(a, b)


class BagOfWords(Algorithm):

    def __init__(self, name="BagOfWords regex", disable=["ner"], language="english"):
        super().__init__(name, language)
        self.dictionary = {}
        self.weights = []
        self.disable = disable
        self.stop = False
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
            if self.language=="german":
                self.stop_list = spacy.lang.de.stop_words.STOP_WORDS
            if self.language=="english":
                self.stop_list = spacy.lang.en.stop_words.STOP_WORDS
            if value not in self.stop_list or not self.stop:
                self.dictionary[value] = index
                index += 1
        self.weights = sparse.csr_matrix(
            preprocessing.minmax_scale(self.weights))
        self.trained = True

    def create_count(self, in_line):
        words = np.array(re.sub(r'\W+', ' ', in_line).split(" "))
        count = defaultdict(int)
        for token in words:
            if token not in self.stop_list and self.stop:
                count[token] += 1
            elif not self.stop:
                count[token] += 1
        return count

    def create_vec(self, in_line):
        """Returns a matrix denoting which words from the dictionary occure in a given line."""
        count = self.create_count(in_line)
        col = []
        row = []
        data = []
        for key, value in count.items():
            if key in self.dictionary:
                data.append(value)
                col.append(self.dictionary[key])
                row.append(0)
        return sparse.csr_matrix((data, (row, col)), shape=(1, len(self.dictionary)))


class BagOfWords_stop(BagOfWords):
    def __init__(self, name="BagOfWords regex eliminating stopwords", disable=["ner"], language="english"):
        super().__init__(name, language)
        self.stop = True


class BagOfWords_jaccard(BagOfWords):
    def __init__(self, name="BagOfWords regex jaccard distance", disable=["ner"], language="english"):
        super().__init__(name, language)

    def compare(self, a, b):
        """Returns the jaccard similiarity between two matrices a,b."""
        # couldn't get this to work directly with the csr_matrices.
        return jaccard_score(a.todense().T, b.todense().T, average="macro")


class BagOfWords_jaccard_stop(BagOfWords_jaccard):
    def __init__(self, name="BagOfWords regex jaccard distance eliminating stopwords", disable=["ner"], language="english"):
        super().__init__(name, language)
        self.stop = True


class BagOfWords_l2(BagOfWords):
    def __init__(self, name="BagOfWords regex l2 distance", disable=["ner"], language="english"):
        super().__init__(name, language)

    def compare(self, a, b):
        """Returns the l2 similiarity between two matrices a,b."""
        return 1/(1+sparse.linalg.norm(a-b))


class BagOfWords_l2_stop(BagOfWords_l2):
    def __init__(self, name="BagOfWords regex l2 distance eliminating stopwords", disable=["ner"], language="english"):
        super().__init__(name, language)
        self.stop = True


class BagOfWords_lemma(BagOfWords):

    def __init__(self, name="BagOfWords Lemmatized", disable=["ner"], language="english"):
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
            if not item.is_stop or not self.stop:
                data.append(item.lemma_)
        data, self.weights = np.unique(data, return_counts=True)
        index = 0
        for value in data:
            self.dictionary[value] = index
            index += 1
        self.weights = sparse.csr_matrix(
            preprocessing.minmax_scale(self.weights))
        self.trained = True

    def create_count(self, in_line):
        words = self.nlp(str(in_line), disable=self.disable)
        count = defaultdict(int)
        # Using counter proofed unsucessfull since it bypasses lemmatization
        for token in words:
            if not token.is_stop and self.stop:
                count[token.lemma_] += 1
            elif not self.stop:
                count[token.lemma_] += 1
        return count


class BagOfWords_lemma_stop(BagOfWords_lemma):

    def __init__(self, name="BagOfWords Lemmatized, Stopwords", disable=["ner"], language="english"):
        super().__init__(name, language)
        self.stop = True


class BagOfWords_jaccard_lemma(BagOfWords_lemma):
    def __init__(self, name="BagOfWords lemmatized jaccard distance", disable=["ner"], language="english"):
        super().__init__(name, language)

    def compare(self, a, b):
        """Returns the jaccard distance between two matrices a,b."""
        # couldn't get this to work directly with the csr_matrices.
        return jaccard_score(a.todense().T, b.todense().T, average="macro")


class BagOfWords_jaccard_lemma_stop(BagOfWords_jaccard_lemma):
    def __init__(self, name="BagOfWords lemmatized jaccard distance eliminating stopwords", disable=["ner"], language="english"):
        super().__init__(name, language)
        self.stop = True


class BagOfWords_l2_lemma(BagOfWords_lemma):
    def __init__(self, name="BagOfWords lemmatized l2 distance", disable=["ner"], language="english"):
        super().__init__(name, language)

    def compare(self, a, b):
        """Returns the l2 similiarity between two matrices a,b."""
        return 1/(1+sparse.linalg.norm(a-b))


class BagOfWords_l2_lemma_stop(BagOfWords_l2_lemma):
    def __init__(self, name="BagOfWords lemmatized l2 distance eliminating stopwords", disable=["ner"], language="english"):
        super().__init__(name, language)
        self.stop = True


class spacy_sem_sim(Algorithm):

    def __init__(self, name="Spacy W2V", language="english", model="md"):
        super().__init__(name, language)
        self.model = model

    def compare(self, a, b):
        """Returns the cosine similarity between two matrices a,b."""
        return a.similarity(b)

    def create_vec(self, in_line):
        """Returns a matrix denoting which words from the dictionary occure in a given line."""
        return self.nlp(in_line)

    def train(self, in_dataset):
        print("Initializing spacy model")
        if self.language == "english":
            self.nlp = spacy.load("en_core_web_{}".format(self.model))
        elif self.language == "german":
            self.nlp = spacy.load("de_core_news_{}".format(self.model))
        else:
            raise ValueError("Unsupported language")
        self.trained = True


class spacy_bert(Algorithm):

    def __init__(self, name="spacy Bert", language="english", model="lg"):
        super().__init__(name, language)
        self.model = model

    def compare(self, a, b):
        """Returns the cosine similarity between two matrices a,b."""
        return a.similarity(b)

    def create_vec(self, in_line):
        """Returns a matrix denoting which words from the dictionary occure in a given line."""
        return self.nlp(in_line)

    def train(self, in_dataset):
        print("Initializing spacy model")
        if self.language == "english":
            self.nlp = spacy.load(
                "en_trf_bertbaseuncased_lg")
        elif self.language == "german":
            self.nlp = spacy.load(
                "de_trf_bertbasecased_lg")
        else:
            raise ValueError("Unsupported language")
        self.trained = True
