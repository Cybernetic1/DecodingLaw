# -*- coding: utf-8 -*-
"""
@author: YKY
"""
import numpy as np
import os						# for os.listdir
from nltk.corpus import stopwords
import re						# for removing punctuations
import pickle
import sys						# for sys.stdout.flush()

path_to_glove = "/data/wiki-news-300d-1M.vec"	# change to your path and filename
GLOVE_SIZE = 300				# dimension of word vectors in GloVe file
num_classes = 3
times_steps = 32				# this number should be same as fixed_seq_len below

""" These are the 3 classes of laws:
* nuisance
* dangerous driving
* OTHERWISE, or work injuries
"""
categories = ["nuisance", "dangerous-driving", "injuries"]
suffix = "-pure"	# "-pure" means to train from full-text instead of YELLOW stuff

# =================== Read case examples from file ======================
"""
0. for each category:
1. read all cases in YELLOW folder
2.	  for each case generate N examples (consecutive word sequences of fixed length)
"""
labels = []
data = []
fixed_seq_len = times_steps		# For each case law, take N consecutive words from text

print("\n**** Preparing training data....")
for i, category in enumerate(categories):
	print("\nCategory: ", category)
	for j, filename in enumerate(os.listdir("laws-TXT/" + category + "-pure")):
		yellow_stuff = []
		with open("laws-TXT/" + category + "-pure/" + filename) as f:
			for line in f:
				line = re.sub(r'[^\w\s-]',' ',line)	# remove punctuations except hyphen
				for word in line.lower().split():	# convert to lowercase
					if word not in stopwords.words('english'):	# remove stop words
						yellow_stuff.append(word)
		print("Case-law #", j, " word count = ", len(yellow_stuff))

		for k in range(0, 500):
			# Randomly select a sequence of words (of fixed length) in yellow text
			seqlens.append(fixed_seq_len)		# this is variable in the original code (with zero-padding), but now it's fixed because we don't use zero-padding
			rand_start = np.random.choice(range(0, len(yellow_stuff) - fixed_seq_len))
			data.append(" ".join(yellow_stuff[rand_start: rand_start + fixed_seq_len]))
			labels += [i]			# set label

# convert to 1-hot encoding for labels
for i in range(len(labels)):
	label = labels[i]
	one_hot_encoding = [0] * num_classes
	one_hot_encoding[label] = 1
	labels[i] = one_hot_encoding

num_examples = len(data)
print("\nData size = ", num_examples, " examples")

# ================ Find unique words ================

print("\n**** Finding unique words....")
word_list = []				# to store the list of words appearing in case-text
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
glove_file = open(path_to_glove, "r")
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
		if count_all_words == len(word_list) - 1:
			print("*** found all words ***")
			break
# if it takes too long to look up the entire dictionary, we can break it short
except KeyboardInterrupt:
	pass
glove_file.close()
f2.close()

print("Vocabulary size = ", len(word2vec_map))

pickling_on = open("training-data.pickle", "wb+")
pickle.dump(data, pickling_on)
pickling_on.close()

pickling_on = open("training-labels.pickle", "wb+")
pickle.dump(labels, pickling_on)
pickling_on.close()

pickling_on = open("training-word-list.pickle", "wb+")
pickle.dump(word_list, pickling_on)
pickling_on.close()

pickling_on = open("training-word2vec-map.pickle", "wb+")
pickle.dump(word2vec_map, pickling_on)
pickling_on.close()
