from sklearn import preprocessing
import numpy as np
import spacy
import re
from scipy.spatial.distance import cosine


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

    def load_sick(self, path):
        file = open(path, "r") 
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
        self.norm_score = []        
        self.norm_score = preprocessing.minmax_scale(self.scores)
        
    def __len__(self):
        return len(self.ids)

    def search(self, word):
        results = []
        for i in range(len(self.data[0])):
            if word in self.data[0][i]:
                results.append(self.data[0][i])
            if word in self.data[1][i]:
                results.append(self.data[1][i])
        return results

    def run_alg(self,alg):
        results= []
        print("Training...")
        alg.train(self)
        for i in range(len(self.data[0])):
            results.append(float(alg.compare(self.data[0][i],self.data[1][i])))
            print(float(alg.compare(self.data[0][i],self.data[1][i])))
        return results-self.norm_score
    

class BagOfWords:
    def __init__(self, language = "english",disable=["ner"]):
        self.dict = []
        self.language = language
        self.disable = disable
        if self.language == "english":
            self.nlp = spacy.load("en_core_web_sm")
        if self.language == "german":
            self.nlp = spacy.load("de_core_news_sm")

    def train(self,Dataset,lemmatize=True, stop = True):
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
        print(self.dict[0])



    def create_vec(self,line):
        count = np.zeros(len(self.dict[0]))
        words = self.nlp(line)
        for token in words:
            count[np.where(np.array(self.dict[0])==token.lemma_)]+=1
        return count
    
    def compare(self, a,b):
        return cosine(self.create_vec(a),self.create_vec(b))
