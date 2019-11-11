from sklearn import preprocessing
import numpy as np
import spacy
import re
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse


class Dataset:
    def __init__(self,name):
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
    def __init__(self,name):
        self.scores = []
        self.ids = []
        self.name = name
        self.data = [[],[]]
        self.norm_score = []
        self.vecs = {}  

    def load_sick(self):
        """Loads the Sick dataset."""
        file = open("./data/SICK.txt", "r") 
        i=0
        for line in file.readlines():
            line = line.split("\t")
            self.data[0].append(line[1])
            self.data[1].append(line[2])
            self.scores.append(float(line[4]))
            self.ids.append(i)
            i+=1

    def __str__(self):
        output = ""
        for i in range(len(self.data[0])):
            output += str(self.ids[i]) + " " + self.data[0][i] + "\t " + self.data[1][i]+ "\t " + self.scores[i] + "\n"
        return output

    def __getitem__(self, y):
        output = str(self.ids[y]) + " " + self.data[0][y] + "\t " + self.data[1][y]+ "\t " + self.scores[y] + "\n"
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

    def run_alg(self,alg):
        """Runs a given algorithm and returns the difference to the ground truth."""
        results= []
        if not alg.trained:
            print("Training...")
            alg.train(self)
        if self.vecs[alg]!= None:
            data=self.vecs[alg]
            comp = alg.compare
        else:
            comp = alg.compare_create
            data=self.data
        for i in range(len(self.data[0])):
            res = float(comp(data[0][i],data[1][i]))
            results.append(res)
            print(res)
        return results-self.norm_score

    def calc_vecs(self,alg):
        """Precalculates the vectors and stores them in memory."""
        if not alg.trained:
            print("Training...")
            alg.train(self)
        self.vecs[alg]=[[],[]]
        print("Creating Vectors")
        for value in self.data[0]:
            self.vecs[alg][0].append(alg.create_vec(value))
        for value in self.data[1]:
            self.vecs[alg][1].append(alg.create_vec(value))

    

class BagOfWords:
    def __init__(self, language = "english",disable=["ner"]):
        self.dict = []
        self.language = language
        self.disable = disable
        self.trained = False
        if self.language == "english":
            self.nlp = spacy.load("en_core_web_sm")
        if self.language == "german":
            self.nlp = spacy.load("de_core_news_sm")

    def train(self,Dataset,lemmatize=True, stop = True):
        """Creates a dictionary of occuring words for a given dataset."""
        data = ''
        for sets in Dataset.data:
            for item in sets:
                data = data + item + " "
        if lemmatize:
            doc = self.nlp(data, disable = self.disable)
            for token in doc:
                self.dict.append(token.lemma_)
        else:
            data=np.array(re.sub(r'\W+', ' ', data).split(" "))
        if stop:
            if self.language=="german":
                stopwords = spacy.lang.de.stop_words.STOP_WORDS
            if self.language=="english":
                stopwords = spacy.lang.en.stop_words.STOP_WORDS
            self.dict = [token for token in self.dict if not token in stopwords]
        self.dict = np.unique(self.dict, return_counts=True)
        self.trained = True
        print(self.dict[0])


    def create_vec(self,line):
        """Returns a matrix denoting which words from the dictionary occure in a given line."""
        count = np.zeros(len(self.dict[0]))
        words = self.nlp(line)
        for token in words:
            count[np.where(np.array(self.dict[0])==token.lemma_)]+=1
        return sparse.csr_matrix(count)
    
    def compare_create(self, a,b):
        """Returns the cosine similarity between two matrives a,b.
        Interestingly scipys cosine function doesn't work on scipys sparse matrices, while sklearns does."""
        return cosine_similarity(self.create_vec(a),self.create_vec(b))
    
    def compare(self, a,b):
        """Returns the cosine similarity between two matrives a,b.
        Interestingly scipys cosine function doesn't work on scipys sparse matrices, while sklearns does."""
        return cosine_similarity(a,b)
    


db = Dataset_annot("sick")
db.load_sick()

BoW = BagOfWords()

db.calc_vecs(BoW)
db.run_alg(BoW)