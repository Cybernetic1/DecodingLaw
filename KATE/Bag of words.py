# Feature extraction from text
# Method: bag of words 
# https://pythonprogramminglanguage.com
 
from sklearn.feature_extraction.text import CountVectorizer
import os
import re
from nltk.corpus import stopwords
import numpy as np
import pickle
from math import sqrt

def euclidean_distance(v):
    return sqrt(sum(x**2 for x in v))

fixed_seq_len = 32

pickle_off = open("word2vec-map.pickle", "rb")
word2vec_map = pickle.load(pickle_off)

categories = ["matrimonial-rights", "separation", "divorce", "after-divorce", "divorce-maintenance",
  "property-on-divorce", "types-of-marriages", "battered-wife-and-children", "Harmony-House", "divorce-mediation"]

cats = []              # list of list of list of words (categories[lines[words]])
suffix = ""            # to be added to sub-directory, not needed currently

for i, category in enumerate(categories):
    print("\nCategory: ", category)
    for j, filename in enumerate(os.listdir("../categories/" + category + suffix)):
        with open("../categories/" + category + suffix + "/" + filename) as f:
            catLines = []
            for line in f:
                catWords = []
                line = re.sub(r"[^\w-]", " ", line)             # strip punctuations except hyphen
                line = re.sub(u"[\u4e00-\u9fff]", " ", line)    # strip Chinese
                line = re.sub(r"\d", " ", line)                 # strip numbers
                line = re.sub(r"-+", "-", line)                 # reduce multiple --- to -
                for word in line.lower().split():
                    if word not in stopwords.words('english'):
                        catWords.append(word)
                if len(catWords) > 0:                       # skip empty lines
                    catLines.append(catWords)
        cats.append(catLines)

vec = np.empty(shape = (300,))
for i,category in enumerate(categories):
	for line in cats[i]:
		vecLine = np.empty(shape=(300,))
		count = 0
		for word in line:
			if count < 4:
				print(vecLine.shape)
				#print(word2vec_map[word].shape)
				try:
					vecLine = np.concatenate((vecLine,np.asarray(word2vec_map[word])))
				except:
				    pass	
			else:
			    break
		np.concatenate((vec,vecLine))	    	

sorted(vec, key=euclidean_distance)        


file = open("prepared-data/train.pickle",'wb+')
pickle.dump(vec, file)
