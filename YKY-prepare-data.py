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
from collections import defaultdict	# for default value of word-vector dictionary
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

# ====================== load pre-trained word2vec dictionary ======================

pickle_off = open("word2vec-map.pickle", "rb")
word2vec_map = pickle.load(pickle_off)

pickle_off = open("unique-words.pickle", "rb")
unique_words = pickle.load(pickle_off)           # unique words list for case-law as well as categories

# set default value = zero vector, if word not found in dictionary
zero_vector = np.asarray([0.0] * GLOVE_SIZE, dtype='float32')
word2vec_map = defaultdict(lambda: zero_vector, word2vec_map)

# ======================= Functions for calculating cosine similarity ==================

def sent_avg_vector(words):
  """ calculates average vector of sentence and returns value"""
  Vec = np.zeros((300,), dtype="float32")
  numWords = 0

  for word in words:
    if word in unique_words:
      numWords += 1
      Vec = np.add(Vec, word2vec_map[word])

  if numWords>0:
    Vec = np.divide(Vec, numWords)

  return Vec

def similarity(Vec1,Vec2):
  """ calculating cosine similarity between two sentence vectors """

  return np.dot(Vec1,Vec2)       

# ==================== extract all sentences from categories =======================

catLines = []           # array of array of words
suffix = ""   # to be added to sub-directory, not needed currently

for i, category in enumerate(categories):
    print("\nCategory: ", category)
    for j, filename in enumerate(os.listdir("categories/" + category + suffix)):
      with open("scraped-data/" + category + suffix + "/" + filename) as f:
        for line in f:
            catWords = []
            line = re.sub(r"[^\w-]", " ", line)				# strip punctuations except hyphen
            line = re.sub(u"[\u4e00-\u9fff]", " ", line)	# strip Chinese
            line = re.sub(r"\d", " ", line)					# strip numbers
            line = re.sub(r"-+", "-", line)					# reduce multiple --- to -
            for word in line.lower().split():
                if word not in stopwords.words('english'):
                    catWords.append(word)
        if len(catWords) > 0:                       # skip empty lines
            catLines.append(catWords)

# ===================== Scan case examples from file ============================
labels = []
data = []
fixed_seq_len = times_steps   # For each case law, take N consecutive words from text

print("\n**** Calculating sentence similarity....")
print(time.strftime("%Y-%m-%d %H:%M"))

for filenames in os.listdir("laws-TXT/family-laws"):
    count = 0
    with open("laws-TXT/family-laws/" + filenames, encoding="utf-8") as fh:
      for line in fh:
          senWords = []
          line = re.sub(r"[^\w-]", " ", line)				# strip punctuations except hyphen
          line = re.sub(u"[\u4e00-\u9fff]", " ", line)	# strip Chinese
          line = re.sub(r"\d", " ", line)					# strip numbers
          line = re.sub(r"-+", "-", line)					# reduce multiple --- to -
          for word in line.lower().split():
            if word not in stopwords.words('english'):
              senWords.append(word)
              count += 1

          # ====== for each line, find similarity against N categories
          for i, category in enumerate(categories):
              print("\nCategory: ", category)
              text1.delete(1.0, END)
              text1.insert(INSERT, category + " :\n")
              text1.insert(INSERT, senWords)

              vec1 = sent_avg_vector(senWords)
              #print(vec1)
              iter = 0
              for catWords in catLines:
                iter += 1
                #print(iter)
                try:
                    #print(iter)
                    vec2 = sent_avg_vector(catWords)
                    #print(vec2)
                    text2.delete(1.0, END)
                    text2.insert(INSERT, catWords)
                    canvas.create_rectangle(0, 0, 402, 10, fill="black")
                    canvas.create_rectangle(1, 1, sim * 4, 9, fill="red")
                    root.update_idletasks()
                    root.update()

                    #print(line + " is ")
                    #print(similarity(vec1, vec2) * 100)
                    #print(" % similar to \n" + sent + "\n" )
                except KeyboardInterrupt:
                    #except Exception as e:
                    print("key exception....")
                    sys.stdin.readline()
                    pass

exit(0)

for k in range(0, 5000):   # number of examples per file (default: 500)
      #print(k, end = ": ")
      # Randomly select a sequence of words (of fixed length) in stuff text
      rand_start = np.random.choice(range(0, len(stuff) - fixed_seq_len))
      word_list = " ".join(stuff[rand_start: rand_start + fixed_seq_len])
      #print(word_list)
      data.append(word_list)
      #labels += [i]     # set label for training


# convert to 1-hot encoding for labels
for i in range(len(labels)):
  label = labels[i]
  one_hot_encoding = [0] * num_classes
  one_hot_encoding[label] = 1
  labels[i] = one_hot_encoding

num_examples = len(data)
print("\nData size = ", num_examples, " examples")

# ========================== save prepared data to files ==============================

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
