import re
import spacy
import numpy as np
from sklearn import preprocessing
from scipy import sparse
from util import multithread_shared_object
from sklearn.metrics.pairwise import cosine_similarity


class Algorithm:

    def __init__(self, name, language="english",):
        self.trained = False
        self.name = name
        self.language = language

    def train(self):
        pass

    def compare(self, a, b):
        """Returns the cosine similarity between two matrives a,b.
        Interestingly scipys cosine function doesn't work on scipys sparse matrices, while sklearns does."""
        return cosine_similarity(a, b)


class BagOfWords(Algorithm):

    def __init__(self, name="BagOfWords simple language agnostic", disable=["ner"], language="english"):
        super().__init__(name, language)
        self.dict = []
        self.weights = []
        self.disable = disable

    def train(self, Dataset):
        """Creates a dictionary of occuring words for a given dataset."""
        print("Training " + self.name)
        data = []
        for sets in Dataset.data:
            for item in sets:
                data+=re.sub(r'\W+', ' ', item).split(" ")
        self.dict, self.weights = np.unique(data, return_counts=True)
        self.weights = sparse.csr_matrix(
            preprocessing.minmax_scale(self.weights))
        self.trained = True


    def remove_stopwords(self):
        if self.language == "german":
            stopwords = spacy.lang.de.stop_words.STOP_WORDS
        if self.language == "english":
            stopwords = spacy.lang.en.stop_words.STOP_WORDS
        self.dict = [
            token for token in self.dict if not token in stopwords]

    def append_dic(self, data, dict, i):
        for token in data:
            dict.append(token.lemma_)

    def create_vec(self, data, result, i):
        """Returns a matrix denoting which words from the dictionary occure in a given line."""
        for j in range(len(data)):
            #print("seg: "+ str(i) +" row: " + str(j))
            words = np.array(re.sub(r'\W+', ' ', data[j]).split(" "))
            count = np.zeros(len(self.dict))
            for token in words:
                #print((self.dict==bytes(token.lemma_, 'utf-8')).tostring().find('\x01'))
                count[np.where(np.array(self.dict) == token)] += 1
            # print(sparse.csr_matrix(count).shape)
            result[i*len(data)+j] = (sparse.csr_matrix(count))


class BagOfWords_lemma(BagOfWords):

    def __init__(self, name="BagOfWords Lemmatized", disable=["ner"], language="english",):
        super().__init__(name, language)
        if self.language == "english":
            self.nlp = spacy.load("en_core_web_sm")
        if self.language == "german":
            self.nlp = spacy.load("de_core_news_sm")


    def train(self, Dataset, stop=True):
        """Creates a dictionary of occuring words for a given dataset."""
        print("Training " + self.name)
        data = ''
        for sets in Dataset.data:
            for item in sets:
                data = data + item + " "
        doc = self.nlp(data, disable=self.disable)
        self.dict = multithread_shared_object(self.append_dic, "list", doc)
        if stop:
            self.remove_stopwords()
        self.dict, self.weights = np.unique(self.dict, return_counts=True)
        self.weights = sparse.csr_matrix(
            preprocessing.minmax_scale(self.weights))
        self.trained = True

    def append_dic(self, data, dict, i):
        for token in data:
            dict.append(token.lemma_)

    def create_vec(self, data, result, i):
        """Returns a matrix denoting which words from the dictionary occure in a given line."""
        for j in range(len(data)):
            words = self.nlp(str(data[j]))
            count = np.zeros(len(self.dict))
            for token in words:
                count[np.where(np.array(self.dict) == token.lemma_)] += 1
            result[i*len(data)+j] = (sparse.csr_matrix(count))
