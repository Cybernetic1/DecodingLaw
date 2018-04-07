# Feature extraction from text
# Method: bag of words 
# https://pythonprogramminglanguage.com
 
from sklearn.feature_extraction.text import CountVectorizer
import os
import re
from nltk.corpus import stopwords
import numpy as np
import pickle
fixed_seq_len = 32

categories = ["matrimonial-rights", "separation", "divorce", "after-divorce", "divorce-maintenance",
  "property-on-divorce", "types-of-marriages", "battered-wife-and-children", "Harmony-House", "divorce-mediation"]

data = []
print("\n**** Preparing training data....")
for i, category in enumerate(categories):
	print("\nCategory: ", category)
	for j, filename in enumerate(os.listdir("scraped-data/" + category)):
		stuff = []
		with open("scraped-data/" + category + "/" + filename) as f:
			for line in f:
				line = re.sub(r'[^\w\s-]',' ',line)	# remove punctuations except hyphen
				for word in line.lower().split():	# convert to lowercase
					if word not in stopwords.words('english'):	# remove stop words
						stuff.append(word)
		print("Case-law #", j, " word count = ", len(stuff))

		for k in range(0, 3):		# number of examples per file (default: 500)
			# Randomly select a sequence of words (of fixed length) in stuff text
			rand_start = np.random.choice(range(0, len(stuff) - fixed_seq_len))
			data.append(" ".join(stuff[rand_start: rand_start + fixed_seq_len]))

vectorizer = CountVectorizer()
a = vectorizer.fit_transform(data).todense()
b = vectorizer.vocabulary_
k = {"vector":a,"vocab":b}
file = open("prepared-data/train.pickle",'wb+')
pickle.dump(k, file)