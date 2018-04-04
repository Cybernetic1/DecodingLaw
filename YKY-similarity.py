# -*- coding: utf-8 -*-
"""
For A2J Hackathon 2018
Purpose:  try to generate random data from downloaded case-law files
@author: Abeer Arora
"""
import numpy as np
import math
import os           # for os.listdir
from nltk.corpus import stopwords
import re           # for removing punctuations
import pickle
import sys            # for sys.stdout.flush()
from collections import defaultdict # for default value of word-vector dictionary
import time
import tkinter as tk

path_to_glove = "wiki-news-300d-1M.vec" # change to your path and filename
GLOVE_SIZE = 300        # dimension of word vectors in GloVe file
num_classes = 10
times_steps = 32        # this number should be same as fixed_seq_len below

# 10 categories:
categories = ["matrimonial-rights", "separation", "divorce", "after-divorce", "divorce-maintenance",
  "property-on-divorce", "types-of-marriages", "battered-wife-and-children", "Harmony-House", "divorce-mediation"]

root = tk.Tk()
root.title("Similarity measure")

canvas = tk.Canvas(root, width=402, height=10).grid(row=0,column=0)
canvas.create_rectangle(0, 0, 402, 10, fill="black")
canvas.pack()

text0 = Text(root, height=10).grid(row=1,column=0)
text0.insert(INSERT, "Case-law sentences")
text0.pack()

labels = []
canvases = []
for i, cat in enumerate(categories):
	label = Label(root, text=cat).grid(row=i+1,column=0)
	labels.append(label)
	canvas = Canvas(root, width=402, height=10).grid(row=i+1,column=1)
	canvases.append(canvas)

root.update_idletasks()
root.update()
root.mainloop()

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
  v = math.sqrt(sum(vec1**2))
  vv = math.sqrt(sum(vec2**2))
  numerator = np.dot(Vec1,Vec2)
  denominator = v*vv
  return numerator/denominator        

# ==================== extract all sentences from categories =======================

cats = []              # list of list of list of words (categories[lines[words]])
suffix = ""            # to be added to sub-directory, not needed currently

for i, category in enumerate(categories):
    print("\nCategory: ", category)
    for j, filename in enumerate(os.listdir("categories/" + category + suffix)):
        with open("categories/" + category + suffix + "/" + filename) as f:
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

# ===================== Scan case examples from file ============================
labels = []
data = []
fixed_seq_len = times_steps   # For each case law, take N consecutive words from text

print("\n**** Calculating sentence similarity....")
print(time.strftime("%Y-%m-%d %H:%M"))

for filenames in os.listdir("laws-TXT/family-laws"):
    with open("laws-TXT/family-laws/" + filenames, encoding="utf-8") as fh:
        for line in fh:
            words1 = []
            line = re.sub(r"[^\w-]", " ", line)               # strip punctuations except hyphen
            line = re.sub(u"[\u4e00-\u9fff]", " ", line)      # strip Chinese
            line = re.sub(r"\d", " ", line)                   # strip numbers
            line = re.sub(r"-+", "-", line)                   # reduce multiple --- to -
            text1.delete(1.0, END)
            for word in line.lower().split():
                if word not in stopwords.words('english'):
                    words1.append(word)
                    text1.insert(INSERT, word + " ")
            if len(words1) == 0:                            # skip empty lines
                continue

            vec1 = sent_avg_vector(words1)
            #print(vec1)

            # ====== for each case-law line, find similarity against N categories
            for i, category in enumerate(categories):
              # print("\nCategory: ", category)
              text2.delete(1.0, END)
              text2.insert(1.0, category)

              catLines = cats[i]
              for catWords in catLines:
                  text3.delete(1.0, END)
                  for w in catWords:
                      text3.insert(INSERT, w + " ")
                  try:
                      vec2 = sent_avg_vector(catWords)
                      sim = similarity(vec1, vec2) * 100
                      canvas.create_rectangle(0, 0, 402, 10, fill="black")
                      canvas.create_rectangle(1, 1, sim * 4, 9, fill="red")
                      root.update_idletasks()
                      root.update()
                  except KeyboardInterrupt:
                      #except Exception as e:
                      print("press enter to continue....")
                      sys.stdin.readline()

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
