from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
import numpy as np
from sklearn import preprocessing

# bow
import re  # also lstm
from scipy import sparse
from collections import defaultdict
from collections import Counter
import warnings
from sklearn.utils.validation import DataConversionWarning

# w2v & bert
import spacy

# wmd
import gensim.downloader as api
from gensim.models import Word2Vec
from gensim.similarities import WmdSimilarity
# d2v
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from scipy.spatial.distance import cosine as sci_cos
#SentenceTransformer
from sentence_transformers import SentenceTransformer

# LSTM
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Lambda
import keras.backend as K
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import itertools
import datetime
from time import time
import matplotlib.pyplot as plt
import gensim.downloader as api

warnings.filterwarnings("ignore", category=DataConversionWarning)


class Algorithm:
    """Implements an algorithm which calculates the similarity between two strings."""

    def __init__(
        self, name, language="english",
    ):
        self.trained = False
        self.name = name
        self.language = language

    def train(self, in_dataset, in_score):
        """
        Trains the algorithm on a given list of strings.

        Parameters
        ----------
        in_dataset : Takes a list containing two lists of strings.
        in_score : Takes a list of floating point values.
        """
        raise NotImplementedError("Train method not implemented")

    def create_vec(self, in_line):
        """Returns a vector that can be used to calculate the difference between strings, when given a string.

        Parameters
        ----------
        in_line : A string for which a vector should be calculated.
        """
        raise NotImplementedError("Create_vec method not implemented")

    def compare(self, a, b):
        """Returns the cosine similarity between two matrices a,b.

        Parameters
        ----------
        a : A matrix.
        b : Another matrix.
        """
        return cosine_similarity(a, b)


class BagOfWords(Algorithm):
    """
    Implements the BagOfWords algorithm.

    """

    def __init__(self, name="BagOfWords regex", language="english"):
        super().__init__(name, language)
        self.dictionary = {}
        self.weights = []
        self.stop = False
        if self.language == "english":
            self.nlp = spacy.load("en_core_web_sm")
        elif self.language == "german":
            self.nlp = spacy.load("de_core_news_sm")
        else:
            raise ValueError("Unsupported language")

    def train(self, in_dataset, in_score):
        """Creates a dictionary of occuring words for a given dataset."""
        print("Training  {}".format(self.name))
        data = []
        for sets in in_dataset:
            for item in sets:
                data += re.sub(r"\W+", " ", item).split(" ")
        data, self.weights = np.unique(data, return_counts=True)
        index = 0
        for value in data:
            if self.language == "german":
                self.stop_list = spacy.lang.de.stop_words.STOP_WORDS
            if self.language == "english":
                self.stop_list = spacy.lang.en.stop_words.STOP_WORDS
            if value not in self.stop_list or not self.stop:
                self.dictionary[value] = index
                index += 1
        self.weights = sparse.csr_matrix(preprocessing.minmax_scale(self.weights))
        self.trained = True

    def create_count(self, in_line):
        words = np.array(re.sub(r"\W+", " ", in_line).split(" "))
        count = defaultdict(int)
        for token in words:
            if token not in self.stop_list or not self.stop:
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
    def __init__(
        self, name="BagOfWords regex eliminating stopwords", language="english"
    ):
        super().__init__(name, language)
        self.stop = True


class BagOfWords_jaccard(BagOfWords):
    def __init__(self, name="BagOfWords regex jaccard distance", language="english"):
        super().__init__(name, language)

    def compare(self, a, b):
        """Returns the jaccard similiarity between two matrices a,b."""
        # couldn't get this to work directly with the csr_matrices.
        return jaccard_score(a.todense().T, b.todense().T, average="macro")


class BagOfWords_jaccard_stop(BagOfWords_jaccard):
    def __init__(
        self,
        name="BagOfWords regex jaccard distance eliminating stopwords",
        language="english",
    ):
        super().__init__(name, language)
        self.stop = True


class BagOfWords_l2(BagOfWords):
    def __init__(self, name="BagOfWords regex l2 distance", language="english"):
        super().__init__(name, language)

    def compare(self, a, b):
        """Returns the l2 similiarity between two matrices a,b."""
        return 1 / (1 + sparse.linalg.norm(a - b))


class BagOfWords_l2_stop(BagOfWords_l2):
    def __init__(
        self,
        name="BagOfWords regex l2 distance eliminating stopwords",
        language="english",
    ):
        super().__init__(name, language)
        self.stop = True


class BagOfWords_lemma(BagOfWords):
    def __init__(
        self, name="BagOfWords Lemmatized", disable=["ner"], language="english"
    ):
        super().__init__(name, language)
        self.disable = disable

    def train(self, in_dataset, in_score, stop=True):
        """Creates a dictionary of occuring words for a given dataset."""
        print("Training {}".format(self.name))
        data = ""
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
        self.weights = sparse.csr_matrix(preprocessing.minmax_scale(self.weights))
        self.trained = True

    def create_count(self, in_line):
        words = self.nlp(str(in_line), disable=self.disable)
        count = defaultdict(int)
        # Using counter proofed unsucessfull since it bypasses lemmatization
        for token in words:
            if not token.is_stop or not self.stop:
                count[token.lemma_] += 1
        return count


class BagOfWords_lemma_stop(BagOfWords_lemma):
    def __init__(
        self,
        name="BagOfWords Lemmatized, Stopwords",
        disable=["ner"],
        language="english",
    ):
        super().__init__(name, language)
        self.stop = True


class BagOfWords_jaccard_lemma(BagOfWords_lemma):
    def __init__(
        self,
        name="BagOfWords lemmatized jaccard distance",
        disable=["ner"],
        language="english",
    ):
        super().__init__(name, language)

    def compare(self, a, b):
        """Returns the jaccard distance between two matrices a,b."""
        # couldn't get this to work directly with the csr_matrices.
        return jaccard_score(a.todense().T, b.todense().T, average="macro")


class BagOfWords_jaccard_lemma_stop(BagOfWords_jaccard_lemma):
    def __init__(
        self,
        name="BagOfWords lemmatized jaccard distance eliminating stopwords",
        disable=["ner"],
        language="english",
    ):
        super().__init__(name, language)
        self.stop = True


class BagOfWords_l2_lemma(BagOfWords_lemma, BagOfWords_l2):
    def __init__(
        self,
        name="BagOfWords lemmatized l2 distance",
        disable=["ner"],
        language="english",
    ):
        super().__init__(name, language)

    def compare(self, a, b):
        """Returns the l2 similiarity between two matrices a,b."""
        return BagOfWords_l2.compare(self, a, b)


class BagOfWords_l2_lemma_stop(BagOfWords_l2_lemma):
    def __init__(
        self,
        name="BagOfWords lemmatized l2 distance eliminating stopwords",
        disable=["ner"],
        language="english",
    ):
        super().__init__(name, language)
        self.stop = True


class spacy_sem_sim(Algorithm):
    """
    Implements a word2vec model by using spacy's existing vectors.

    Parameters
    ----------
    name : A name for the algorithm.
    language : the language of the dataset to be analyzed. Either "german" or "english". 
    model : The model size: "sm" small, "md" medium, "lg" large. Not all sizes do exist for each language.
    """

    def __init__(self, name="Spacy W2V", language="english", model="md"):
        super().__init__(name, language)
        self.model = model

    def compare(self, a, b):
        """Returns the cosine similarity between two matrices a,b."""
        return a.similarity(b)

    def create_vec(self, in_line):
        """Returns a matrix denoting which words from the dictionary occure in a given line."""
        return self.nlp(in_line)

    def train(self, in_dataset, in_score):
        print("Initializing spacy model")
        if self.language == "english":
            self.nlp = spacy.load("en_core_web_{}".format(self.model))
        elif self.language == "german":
            self.nlp = spacy.load("de_core_news_{}".format(self.model))
        else:
            raise ValueError("Unsupported language")
        self.trained = True


class spacy_bert(Algorithm):
    """
    Implements a BERT model by using spacytransformer's existing vectors.

    Parameters
    ----------
    name : A name for the algorithm.
    language : the language of the dataset to be analyzed. Either "german" or "english". 
    """

    def __init__(self, name="spacy Bert", language="english"):
        super().__init__(name, language)

    def compare(self, a, b):
        """Returns the cosine similarity between two matrices a,b."""
        return a.similarity(b)

    def create_vec(self, in_line):
        """Returns a matrix denoting which words from the dictionary occure in a given line."""
        return self.nlp(in_line)

    def train(self, in_dataset, in_score):
        print("Initializing spacy model")
        if self.language == "english":
            self.nlp = spacy.load("en_trf_bertbaseuncased_lg")
        elif self.language == "german":
            self.nlp = spacy.load("de_trf_bertbasecased_lg")
        else:
            raise ValueError("Unsupported language")
        self.trained = True


class sent_transf(Algorithm):
    """
    Implements a RoBERT model by using SentenceTransformer's existing vectors.

    Parameters
    ----------
    name : A name for the algorithm.
    language : the language of the dataset to be analyzed. Either "german" or "english". 
    """

    def __init__(self, name="Sentence Transformer", language="english"):
        super().__init__(name, language)

    def create_vec(self, in_line):
        """Returns a matrix denoting which words from the dictionary occure in a given line."""
        return self.nlp.encode([in_line])

    def train(self, in_dataset, in_score):
        print("Initializing Sentence Transformer model")
        if self.language == "english":
            self.nlp = SentenceTransformer('stsb-roberta-base')
        else:
            raise ValueError("Unsupported language")
        self.trained = True

class gensim_wmd(Algorithm):
    """
    Implements a Word Mover Distance comparison by using gensims implementation of pyemds earth mover distance and the vectors from word2vec-google-news-300.

    Parameters
    ----------
    name : A name for the algorithm.
    language : The language of the dataset to be analyzed. Either "german" or "english". The language is irrelevant for this particular algorithm.
    """

    def __init__(self, name="gensim wmd", language="english"):
        super().__init__(name, language)

    def compare(self, a, b):
        """
        Returns the word mover distance between two lists a,b.

        Parameters
        ----------
        a : The first list of strings to be compared to the second list.
        a : The second list of strings to be compared to the first list.
        """
        index = WmdSimilarity([a], self.model)
        return index[b]

    def create_vec(self, in_line):
        """
        Returns a list of strings, generated by spliting a string with spaces being seperation points.

        Parameters
        ----------
        inline : The string to be split.
        """
        return in_line.split(" ")

    def train(self, in_dataset, in_score):
        """
        Trains the model on all vectors in the in_dataset.

        Parameters
        ----------
        in_dataset : The dataset to train on.
        """
        print("Initializing gensim model")
        data = []
        for column in in_dataset:
            for sentence in column:
                data.append(sentence.split(" "))
        self.model = Word2Vec(data, min_count=1)
        self.trained = True

class gensim_d2v(Algorithm):
    """
    Implements a Doc2Vec comparison by using gensims implementation and training it on the datasets.

    Parameters
    ----------
    name : A name for the algorithm.
    language : The language of the dataset to be analyzed. Either "german" or "english". The language is irrelevant for this particular algorithm.
    """

    def __init__(self, name="gensim d2v", language="english"):
        super().__init__(name, language)

    def compare(self, a, b):
        """
        Returns the cosine_similarity of two lists a,b.

        Parameters
        ----------
        a : The first list of strings to be compared to the second list.
        a : The second list of strings to be compared to the first list.
        """
        return 1-sci_cos(a,b)

    def create_vec(self, in_line):
        """
        Returns d2v vectors, generated using the trained model.

        Parameters
        ----------
        inline : The string to be split.
        """
        result= self.d2v.infer_vector(in_line.split())
        return result

    def train(self, in_dataset, in_score):
        """
        Trains the model on all sentences in the in_dataset.

        Parameters
        ----------
        in_dataset : The dataset to train on.
        in_score : The socres for the sentence pairs.
        """
        print("Initializing gensim model")
        data = []
        self.id=0
        for col in in_dataset:
            for sentence in col:
                data.append(TaggedDocument(sentence.split(),str(self.id)))
                self.id+=1
        self.d2v = Doc2Vec(vector_size=300, window=10, min_count=5, workers=11,alpha=0.025, min_alpha=0.025) # use fixed learning ratemodel.build_vocab(it)for epoch in range(10):
        self.d2v.build_vocab(data)
        eps = 10
        for item in range(eps):
            self.d2v.train(data,total_examples=self.d2v.corpus_count, epochs=eps, start_alpha=0.025)
        self.trained = True



class MALSTM(Algorithm):
    """Implements a Manhattan-LSTM which calculates the similarity between two strings.
    
    This is an adaptation of the network presented in https://medium.com/mlreview/implementing-malstm-on-kaggles-quora-question-pairs-competition-8b31b0b16a07
    Source: https://github.com/eliorc/Medium/blob/master/MaLSTM.ipynb"""

    def __init__(
        self, name="LSTM", language="english", epochs=1500,
    ):
        super().__init__(name, language)
        self.n_epoch = (
            epochs  # I recommend a minimum of 7 to get somewhat decent results.
        )
        self.vocabulary = dict()
        self.word2vec = None
        self.malstm = None
        if self.language == "german":
            self.nlp = spacy.load("de_core_news_sm")
            self.stop_list = spacy.lang.de.stop_words.STOP_WORDS
        if self.language == "english":
            self.nlp = spacy.load("en_core_web_sm")
            self.stop_list = spacy.lang.en.stop_words.STOP_WORDS

    def text_to_word_list(self, text):
        """ Pre process and convert texts to a list of words """
        text = str(text)
        text = text.lower()

        # Clean the text
        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"\-", " - ", text)
        text = re.sub(r"\=", " = ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" u s ", " american ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e - mail", "email", text)
        text = re.sub(r"j k", "jk", text)
        text = re.sub(r"\s{2,}", " ", text)

        text = text.split()

        return text

    def exponent_neg_manhattan_distance(self, left, right):
        """ Helper function for the similarity estimate of the LSTMs outputs"""
        return K.exp(-K.sum(K.abs(left - right), axis=1, keepdims=True))

    def train(self, in_dataset, in_score):
        """
        Trains the algorithm on a given list of strings.

        Parameters
        ----------
        in_dataset : Takes a list containing two lists of strings.
        in_score : Takes a list of floating point values.
        """

        for i, item in enumerate(in_score):
            in_score[i] = round(item,1)
        df = pd.DataFrame(
            list(zip(in_dataset[0], in_dataset[1], in_score)),
            columns=["string_1", "string_2", "score"],
        )
        vocabulary = dict()
        inverse_vocabulary = [
            "<unk>"
        ]  # '<unk>' will never be used, it is only a placeholder for the [0, 0, ....0] embedding
        string_cols = ["string_1", "string_2"]
        for index, row in df.iterrows():
            # Iterate through the text of both strings of the row
            for text in string_cols:

                q2n = []  # q2n -> string numbers representation
                for word in row[text].split():

                    # Check for unwanted words
                    if word in self.stop_list:
                        continue

                    if word not in vocabulary:
                        vocabulary[word] = len(inverse_vocabulary)
                        q2n.append(len(inverse_vocabulary))
                        inverse_vocabulary.append(word)
                    else:
                        q2n.append(vocabulary[word])

                # Replace strings as word to string as number representation
                df.set_value(index, text, q2n)
        embedding_dim = 1024
        if self.word2vec == None:
            self.word2vec = Word2Vec(vocabulary.keys(), min_count=1 , size = embedding_dim, workers = 10)
        # This will be the embedding matrix
        embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)
        embeddings[0] = 0  # So that the padding will be ignored
        for word, index in vocabulary.items():
            if word in self.word2vec.wv.vocab:
                embeddings[index] = self.word2vec.wv[word]

        # prep data

        max_seq_length = max(
            df.string_1.map(lambda x: len(x)).max(),
            df.string_2.map(lambda x: len(x)).max(),
        )


        X = df[string_cols]
        Y = df["score"]

        # Split to dicts
        X = {"left": X.string_1, "right": X.string_2}

        # Convert labels to their numpy representations
        Yn = Y.values

        # Zero padding
        for dataset, side in itertools.product([X], ["left", "right"]):
            dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)

        n_hidden = 75
        gradient_clipping_norm = 1.25
        batch_size = 15

        # The visible layer
        left_input = Input(shape=(max_seq_length,), dtype="int32")
        right_input = Input(shape=(max_seq_length,), dtype="int32")

        embedding_layer = Embedding(
            len(embeddings),
            embedding_dim,
            weights=[embeddings],
            input_length=max_seq_length,
            trainable=False,
        )

        # Embedded version of the inputs
        encoded_left = embedding_layer(left_input)
        encoded_right = embedding_layer(right_input)

        # Since this is a siamese network, both sides share the same LSTM
        shared_lstm = LSTM(n_hidden)

        left_output = shared_lstm(encoded_left)
        right_output = shared_lstm(encoded_right)

        # Calculates the distance as defined by the MaLSTM model
        malstm_distance = Lambda(
            function=lambda x: self.exponent_neg_manhattan_distance(x[0], x[1]),
            output_shape=lambda x: (x[0][0], 1),
        )([left_output, right_output])

        # Pack it all up into a model
        self.malstm = Model([left_input, right_input], [malstm_distance])

        # Adadelta optimizer, with gradient clipping by norm
        optimizer = Adadelta(clipnorm=gradient_clipping_norm)

        self.malstm.compile(
            loss="mean_squared_error", optimizer=optimizer, metrics=["accuracy"]
        )

        # Start training
        training_start_time = time()

        malstm_trained = self.malstm.fit(
            [X["left"], X["right"]],
            Y,
            batch_size=batch_size,
            nb_epoch=self.n_epoch,
            validation_split=0.15
        )
        self.malstm.save("./models/MaLSTM.h5")
        # self.malstm.save('./models/MaLSTM.h5')
        print(
            "Training time finished.\n{} epochs in {}".format(
                self.n_epoch, datetime.timedelta(seconds=time() - training_start_time)
            )
        )

        plt.plot(malstm_trained.history["acc"])
        plt.plot(malstm_trained.history["val_acc"])
        plt.title("Model Accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Validation"], loc="upper left")
        plt.show()

        # Plot loss
        plt.plot(malstm_trained.history["loss"])
        plt.plot(malstm_trained.history["val_loss"])
        plt.title("Model Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Validation"], loc="upper right")
        plt.show()

        self.trained = True

    def create_vec(self, in_line):
        """Returns a vector that can be used to calculate the difference between strings, when given a string.

        Parameters
        ----------
        in_line : A string for which a vector should be calculated.
        """
        raise NotImplementedError("Create_vec method not implemented")

    def compare(self, a, b):
        """Returns the cosine similarity between two matrices a,b.

        Parameters
        ----------
        a : A matrix.
        b : Another matrix.
        """
        return cosine_similarity(a, b)
