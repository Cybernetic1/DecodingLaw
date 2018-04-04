# -*- coding: utf-8 -*-
"""
For A2J Hackathon 2018
Purpose:  try to generate random data from downloaded case-law files
@author: Abeer Arora
"""
import numpy as np
import os           # for os.listdir
from nltk.corpus import stopwords
import re           # for removing punctuations
import pickle
import sys            # for sys.stdout.flush()
from collections import defaultdict # for default value of word-vector dictionary
from tkinter import *

root = Tk()
root.title("Similarity measure")

canvas = Canvas(root, width=402, height=10)
canvas.create_rectangle(0, 0, 402, 10, fill="black")
canvas.pack()

text1 = Text(root, height=10)
text1.insert(INSERT, "Hello.....")
text1.pack()

text2 = Text(root, height=10)
text2.insert(INSERT, "Goodbye.....")
text2.pack()

root.update_idletasks()
root.update()

path_to_glove = "wiki-news-300d-1M.vec" # change to your path and filename
GLOVE_SIZE = 300        # dimension of word vectors in GloVe file
num_classes = 10
times_steps = 32        # this number should be same as fixed_seq_len below

# 10 categories:
categories = ["matrimonial-rights", "separation", "divorce", "after-divorce", "divorce-maintenance",
  "property-on-divorce", "types-of-marriages", "battered-wife-and-children", "Harmony-House", "divorce-mediation"]

suffix = ""   # to be added to sub-directory, not needed currently

def sent_avg_vector(words, vec_map, wordList):
  """ calculates average vector of sentence and returns value"""
  Vec = np.zeros((300,), dtype="float32")
  numWords = 0

  for word in words:
    if word in wordList:
      numWords += 1
      Vec = np.add(Vec, vec_map[word])

  if numWords>0:
    Vec = np.divide(Vec, numWords)

  return Vec

def similarity(Vec1,Vec2):
  """ calculating cosine similarity between two sentence vectors """

  return np.dot(Vec1,Vec2)       


# =================== Read case examples from file ======================
"""
0. for each category:
1. read all cases in folder
2.    for each case generate N examples (consecutive word sequences of fixed length)
"""
labels = []
data = []
fixed_seq_len = times_steps   # For each case law, take N consecutive words from text
stuff = []

print("\n**** Reading data files into memory....")
count = 0
for filenames in os.listdir("laws-TXT/family-laws"):
    with open("laws-TXT/family-laws/"+filenames, encoding="utf-8") as fh:
      for line in fh:
        line = re.sub(r'[^\w\s-]',' ', line)
        for word in line.lower().split():
          if(word not in stopwords.words('english') and word[0] not in "0123456789-" and \
            re.search(u'[\u4e00-\u9fff]+', word) == None and \
          re.search(r'\d', word) == None):
            stuff.append(word)
            count += 1
            print(count, end='\r')
        if(count >= 80000):
          break

print("\n**** Making examples....")
for k in range(0, 1000):   # number of examples per file (default: 500)
      #print(k, end = ": ")
      # Randomly select a sequence of words (of fixed length) in stuff text
      rand_start = np.random.choice(range(0, len(stuff) - fixed_seq_len))
      word_list = " ".join(stuff[rand_start: rand_start + fixed_seq_len])
      #print(word_list)
      data.append(word_list)
      #labels += [i]     # set label for training
      print(k, end='\r')

# ================ Find unique words ================

print("\n**** Finding unique words....")
word_list = []        # to store the list of words appearing in case-text
index = 0
for sent in data:
  for word in sent.split():
    if word not in word_list:
      # make sure no Chinese chars or numerals inside word
      if re.search(u'[\u4e00-\u9fff]+', word) == None and \
          re.search(r'\d', word) == None:
        word_list.append(word)
        index += 1
        print(word, "                        ", end='\r')
        sys.stdout.flush()

print(len(word_list), " unique words found")

    # ============== Create word-to-vector dictionary ===========

print("\n**** Looking up word vectors....")
word2vec_map = {}
count_all_words = 0
entry_number = 0
glove_file = open(path_to_glove, "r", encoding="utf-8")
f2 = open("found-words.txt", "w")
try:
  for word_entry in glove_file:
    vals = word_entry.split()
    word = str(vals[0])
    entry_number += 1
    if word in word_list:
      print(count_all_words, word, file=f2)
      print(entry_number, count_all_words, word, "             ", end='\r')
      sys.stdout.flush()
      count_all_words += 1
      coefs = np.asarray(vals[1:], dtype='float32')
      coefs /= np.linalg.norm(coefs)
      word2vec_map[word] = coefs
    if count_all_words == len(word_list):
      print("*** found all words ***")
      break
# if it takes too long to look up the entire dictionary, we can break it short
except KeyboardInterrupt:
  pass
glove_file.close()
f2.close()

# set default value = zero vector, if word not found in dictionary
zero_vector = np.asarray([0.0] * GLOVE_SIZE, dtype='float32')
word2vec_map = defaultdict(lambda: zero_vector, word2vec_map)

print("Vocabulary size = ", len(word2vec_map))                  

    # up to this point, we have accessed the 10 categories

    
    # ******************** Abeer, modify here ***********************

    # here, we may want to read: laws-TXT/family-laws/* 
    # and find sentences that are close to the sentences in "stuff"
    # 'closeness' is defined by: cosine distance between averaged word-vectors
    
    # sub-task: for each word you need to look up their word-vectors
    #       this is done below, and you may need to move the code up
    #     also, looking up vectors is time-consuming, best done only once

    # after finding such sentences, append it to "data" similar to below:
  


# Abeer:  for now, just print out some results and exit here

# ======================= Abeer, ignore stuff below this line for the moment =======

# convert to 1-hot encoding for labels
for i in range(len(labels)):
  label = labels[i]
  one_hot_encoding = [0] * num_classes
  one_hot_encoding[label] = 1
  labels[i] = one_hot_encoding

num_examples = len(data)
print("\nData size = ", num_examples, " examples")


print("\n**** Finding similar sentences....")
for i, category in enumerate(categories):
  print("\nCategory: ", category)
  for j, filename in enumerate(os.listdir("scraped-data/" + category + suffix)):
    with open("scraped-data/" + category + suffix + "/" + filename) as f:
      for line in f:
        senWords = []
        line = re.sub(r'[^\w\s-]',' ',line) # remove punctuations except hyphen
        for word in line.lower().split(): # convert to lowercase
          # remove stop words, ignore numbers and dangling hyphens
          if (word not in stopwords.words('english') and
              word[0] not in "0123456789-"):
            senWords.append(word)
        if senWords:
            text1.delete(1.0, END)
            text1.insert(INSERT, category + " : ")
            text1.insert(INSERT, line)
            text1.pack()
            vec1 = sent_avg_vector(senWords, word2vec_map, word_list)
            #print(vec1)
            iter = 0    
            for sent in data:
              iter += 1
              #print(iter)
              try:
                #print(iter)
                vec2 = sent_avg_vector(sent.split(), word2vec_map, word_list)
                #print(vec2)
                sim = similarity(vec1, vec2) * 100
                # print(line + " is ", sim, "% similar to \n")
                # print(sent + "\n")
                text2.delete(1.0, END)
                text2.insert(INSERT, sent)
                text2.pack()
                canvas.create_rectangle(0, 0, 402, 10, fill="black")
                canvas.create_rectangle(1, 1, sim * 4, 9, fill="red")
                canvas.pack()
                root.update_idletasks()
                root.update()
              except KeyboardInterrupt:
                #except Exception as e:
                print("key exception....")
                sys.stdin.readline()
                pass


folderName = "prepared-data/"

pickling_on = open(folderName + "training-data-3.pickle", "wb+")
pickle.dump(data, pickling_on)
pickling_on.close()

pickling_on = open(folderName + "training-labels-3.pickle", "wb+")
pickle.dump(labels, pickling_on)
pickling_on.close()

pickling_on = open(folderName + "training-word-list-3.pickle", "wb+")
pickle.dump(word_list, pickling_on)
pickling_on.close()

pickling_on = open(folderName + "training-word2vec-map-3.pickle", "wb+")
pickle.dump(word2vec_map, pickling_on)
pickling_on.close()
