# -*- coding: utf-8 -*-
"""
0. for each category:
1. read all cases in folder
2.	  for each case generate n examples (select k random words)

@author: YKY
"""
import numpy as np
import os						# for os.listdir
from nltk.corpus import stopwords
import re						# for removing punctuations

# 10 categories:
categories = ["matrimonial-rights", "separation", "divorce", "after-divorce", "divorce-maintenance",
	"property-on-divorce", "types-of-marriages", "battered-wife-and-children", "Harmony-House", "divorce-mediation"]

suffix = ""		# to be added to sub-directory, not needed currently

# =================== Read case examples from file ======================

print("\n**** Reading pre-recorded message texts....")
for i, category in enumerate(categories):
	print("\nCategory: ", category)
	for j, filename in enumerate(os.listdir("scraped-data/" + category + suffix)):
		stuff = []
		with open("scraped-data/" + category + suffix + "/" + filename) as f:
			for line in f:
				line = re.sub(r'[^\w\s-]',' ',line)	# remove punctuations except hyphen
				for word in line.lower().split():	# convert to lowercase
					# remove stop words, ignore numbers and dangling hyphens
					if (word not in stopwords.words('english') and
							word[0] not in "0123456789-"):
						stuff.append(word)
		print("word count = ", len(stuff), '\n')

		for n in range(0, 3):			# number of examples per file (default: 500)
			print(n + 1, end = ": ")
			for k in range(0, 8):		# number of words per example (default: 10)
				# Randomly select a words (of fixed length) in stuff text
				pos = np.random.choice(range(0, len(stuff) - 1))
				print(stuff[pos], end = ' AND ')
			print("\b\b\b\b     ")			# erase the last AND
