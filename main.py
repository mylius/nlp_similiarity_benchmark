from sklearn import preprocessing
import numpy as np
import spacy
import re
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from multiprocessing import Pool,Process,Manager
import os

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
        """Loads the SICK dataset."""
        file = open("./data/SICK.txt", "r") 
        i=0
        for line in file.readlines():
            line = line.split("\t")
            self.data[0].append(line[1])
            self.data[1].append(line[2])
            self.scores.append(float(line[4]))
            self.ids.append(i)
            i+=1

    def load_sts(self):
        """Loads the STS-2017-en-en dataset."""
        file = open("./data/STS.input.track5.en-en.txt", "r") 
        i=0
        for line in file.readlines():
            line = line.split("\t")
            self.data[0].append(line[0])
            self.data[1].append(line[1])
            self.ids.append(i)
            i+=1
        file = open("./data/STS.gs.track5.en-en.txt", "r") 
        for line in file.readlines():
            self.scores.append(float(line))
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
        if alg in self.vecs:
            data=self.vecs[alg]
        else:
            self.calc_vecs(alg)
        for i in range(len(self.data[0])):
            res = float(alg.compare(self.vecs[alg][0][i],self.vecs[alg][1][i]))
            results.append(res)
        return pearsonr(results,self.norm_score)

    def calc_vecs(self,alg):
        """Precalculates the vectors and stores them in memory."""
        if not alg.trained:
            print("Training...")
            alg.train(self)
        self.vecs[alg]=[[],[]]
        print("Creating Vectors")
        self.vecs[alg][0] = multithread_shared_object(alg.create_vec,"list",self.data[0])
        self.vecs[alg][1] = multithread_shared_object(alg.create_vec,"list",self.data[1])


class Algorithm:

    def __init__(self, name, language = "english",):
        self.trained = False
        self.name = name
        self.language = language

    def train(self):
        pass
        
    def compare(self, a,b):
        """Returns the cosine similarity between two matrives a,b.
        Interestingly scipys cosine function doesn't work on scipys sparse matrices, while sklearns does."""
        return cosine_similarity(a,b)


class BagOfWords(Algorithm):
      
    def __init__(self, name="BagOfWords", disable=["ner"], language = "english",):
        super().__init__(name, language)
        self.dict = []
        self.weights=[]
        self.disable = disable
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
            self.dict = multithread_shared_object(self.append_dic,"list",doc)
        else:
            data=np.array(re.sub(r'\W+', ' ', data).split(" "))
            self.dict,self.weights = np.unique(data, return_counts=True)
        if stop:
            if self.language=="german":
                stopwords = spacy.lang.de.stop_words.STOP_WORDS
            if self.language=="english":
                stopwords = spacy.lang.en.stop_words.STOP_WORDS
            self.dict = [token for token in self.dict if not token in stopwords]
        self.dict, self.weights = np.unique(self.dict, return_counts=True)
        self.weights = sparse.csr_matrix(preprocessing.minmax_scale(self.weights))
        self.trained = True
    
    def append_dic(self,data, dict,i):
        for token in data:
            dict.append(token.lemma_)

    def create_vec(self,data,result,i):
        """Returns a matrix denoting which words from the dictionary occure in a given line."""
        
        for j in range(len(data)):
            words = self.nlp(str(data[j]))
            count = np.zeros(len(self.dict))
            for token in words:
                count[np.where(np.array(self.dict)==token.lemma_)]+=1
            result[i*len(data)+j] = (sparse.csr_matrix(count))


def benchmark(algs):
    db = Dataset_annot("sick")
    db.load_sick()
    db.norm_scores()
    print("Results for SICK dataset:")
    for i in range(len(algs)):
        print(algs[i].name + " correlation: " + str(db.run_alg(algs[i])))
    db2 = Dataset_annot("sts")
    db2.load_sts()
    db2.norm_scores()
    print("Results for STS dataset:")
    for i in range(len(algs)):
        print(algs[i].name + " correlation: " + str(db2.run_alg(algs[i])))

def multithread_shared_object(function,s_type, iterable,arguments=None,not_util=2):
    """Hand a shared object of s_type and an iterable to a function be processed in parallel."""
    manager = Manager()
    #assign shared resource
    if s_type == "list":
        shared = manager.list(range(len(iterable)))
    if s_type == "dict":
        shared = manager.dict()
    #if threads > 2 reserve the number specified in not_util, use the rest
    if len(os.sched_getaffinity(0)) > 2:
        cpus = len(os.sched_getaffinity(0))-not_util
    else:
        cpus = len(os.sched_getaffinity(0))
    processes = []
    #split iterable into parts
    split_iter = np.array_split(np.array(iterable),cpus)
    #create process, start and join them
    for i in range(cpus):
        if arguments!=None:
            p=Process(target=function, args=([split_iter[i],shared,arguments,i]))
        else:
            p=Process(target=function, args=([split_iter[i],shared,i]))
        processes.append(p)
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    return shared


BoW = BagOfWords()
benchmark([BoW])