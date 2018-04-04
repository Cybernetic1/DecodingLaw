# -*- coding: utf-8 -*-
"""
* find unique words
* look up word-vectors for all unique words
* save word-vector dictionary

@author: YKY
"""
import numpy as np
import os           # for os.listdir
from nltk.corpus import stopwords
import re           # for removing punctuations
import pickle
import sys			# for sys.stdout.flush()
import time			# for timing events

path_to_glove = "wiki-news-300d-1M.vec" # change to your path and filename
GLOVE_SIZE = 300        # dimension of word vectors in GloVe file
num_classes = 10

# 10 categories:
categories = ["matrimonial-rights", "separation", "divorce", "after-divorce", "divorce-maintenance",
	"property-on-divorce", "types-of-marriages", "battered-wife-and-children", "Harmony-House", "divorce-mediation"]

suffix = ""   # to be added to sub-directory, not needed currently

# ================== Find unique words in pre-recorded category files ========================

unique_words = []
print("\n**** Finding unique words in pre-recorded category files....")
print(time.strftime("%Y-%m-%d %H:%M"))
unique_count = 0
total_count = 0
for filenames in os.listdir("categories/"):
	with open("categories/" + filenames, encoding="utf-8") as fh:
		for line in fh:
			line = re.sub(r"[^\w-]", " ", line)				# strip punctuations except hyphen
			line = re.sub(u"[\u4e00-\u9fff]", " ", line)	# strip Chinese
			line = re.sub(r"\d", " ", line)					# strip numbers
			line = re.sub(r"-+", "-", line)					# reduce multiple --- to -
			for word in line.lower().split():
				total_count += 1
				if word not in stopwords.words('english'):
					if word not in unique_words:
						unique_words.append(word)
						unique_count += 1
				if total_count % 1000 == 0:
					print(unique_count, "/", (total_count // 1000), "K : ", word, "                        ", end='\r')
			#if count >= 20000:
			#	break

print("\n**** unique words found == ", len(unique_words))

# =================== Find unique words in case-law files ========================

unique_words = []
print("\n**** Finding unique words in law-case files....")
print(time.strftime("%Y-%m-%d %H:%M"), " (total 3070 K words, ~9 minutes)")
unique_count = 0
total_count = 0
for filenames in os.listdir("laws-TXT/family-laws"):
	with open("laws-TXT/family-laws/" + filenames, encoding="utf-8") as fh:
		for line in fh:
			line = re.sub(r"[^\w-]", " ", line)				# strip punctuations except hyphen
			line = re.sub(u"[\u4e00-\u9fff]", " ", line)	# strip Chinese
			line = re.sub(r"\d", " ", line)					# strip numbers
			line = re.sub(r"-+", "-", line)					# reduce multiple --- to -
			for word in line.lower().split():
				total_count += 1
				if word not in stopwords.words('english'):
					if word not in unique_words:
						unique_words.append(word)
						unique_count += 1
				if total_count % 1000 == 0:
					print(unique_count, "/", (total_count // 1000), "K : ", word, "                        ", end='\r')
			#if count >= 20000:
			#	break

print("\n**** unique words found == ", len(unique_words))

# ====================== Create word-to-vector dictionary =========================

print("\n**** Looking up word vectors (Ctrl-C to end sooner)....")
print(time.strftime("%Y-%m-%d %H:%M"), " (total 1000000 words, ~7 minutes)")
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
		if word in unique_words:
			print(count_all_words, word, file=f2)
			print(entry_number, count_all_words, word, "             ", end='\r')
			sys.stdout.flush()
			count_all_words += 1
			coefs = np.asarray(vals[1:], dtype='float32')
			coefs /= np.linalg.norm(coefs)
			word2vec_map[word] = coefs
		if count_all_words == len(unique_words) - 1:
			print("*** found all words ***")
			break
# if it takes too long to look up the entire dictionary, we can break it short
except KeyboardInterrupt:
	pass
glove_file.close()
f2.close()

print("\n**** vocabulary size = ", len(word2vec_map))
print(time.strftime("%Y-%m-%d %H:%M"))
folderName = ""

pickling_on = open(folderName + "word2vec-map.pickle", "wb+")
pickle.dump(word2vec_map, pickling_on)
pickling_on.close()
print("SUCCESS")
