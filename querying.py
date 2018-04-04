import os						# for os.listdir and os.environ
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'	# suppress Tensorflow warnings
import numpy as np
from nltk.corpus import stopwords
import re						# for removing punctuations
import sys						# for sys.stdin.readline()
from collections import defaultdict	# for default value of word-vector dictionary
import pickle
import h5py

# Keras deep learning library
import keras
# Load the trained model
model = keras.models.load_model('Keras-NN.h5')

path_to_glove = "wiki-news-300d-1M.vec"
GLOVE_SIZE = 300				# dimension of word vectors in GloVe file
batch_size = 512
num_classes = 10
times_steps = 32

# 10 categories:
categories = ["matrimonial-rights", "separation", "divorce", "after-divorce", "divorce-maintenance",
	"property-on-divorce", "types-of-marriages", "battered-wife-and-children", "Harmony-House", "divorce-mediation"]

# =================== Read prepared training data from file ======================

pickle_off = open("prepared-data/training-word-list-3.pickle", "rb")
word_list = pickle.load(pickle_off)

pickle_off = open("prepared-data/training-word2vec-map-3.pickle", "rb")
word2vec_map = pickle.load(pickle_off)

zero_vector = np.asarray([0.0] * GLOVE_SIZE, dtype='float32')
word2vec_map = defaultdict(lambda: zero_vector, word2vec_map)

# set default value = zero vector, if word not found in dictionary
zero_vector = np.asarray([0.0] * GLOVE_SIZE, dtype='float32')
word2vec_map = defaultdict(lambda: zero_vector, word2vec_map)

# =================== Process a single query =================== #
try:
        while True:
                print("----------------------------\n? ", end = '')
                sys.stdout.flush()
                query = sys.stdin.readline()
                query = re.sub(r'[^\w\s-]',' ', query)	# remove punctuations except hyphen
                query_words = []
                for word in query.lower().split():		# convert to lowercase
                        if word not in stopwords.words('english'):	# remove stop words
                                query_words.append(word)

                # ===== convert query to word-vectors
                query_vectors = []
                glove_file = open(path_to_glove, "r",encoding = "utf-8")
                count_all_words = 0
                entry_number = 0
                for word_entry in glove_file:
                        vals = word_entry.split()
                        word = str(vals[0])
                        entry_number += 1
                        if word in query_words:
                                count_all_words += 1
                                print(count_all_words, word, end = '\r')
                                coefs = np.asarray(vals[1:], dtype='float32')
                                coefs /= np.linalg.norm(coefs)
                                word2vec_map[word] = coefs
                        if count_all_words == len(word_list) - 1:
                                break
                        if entry_number > 50000:
                                # took too long to find the words
                                break

                # ===== make the query length to be (32) = times_steps size
                long_enough = False
                while not long_enough:
                        for word in query_words:
                                query_vectors.append(word2vec_map[word])
                                if len(query_vectors) == times_steps:
                                        long_enough = True
                                        break

        #=========================  prediction ==============================#
                prediction = model.predict(np.expand_dims(query_vectors, axis=0))
                #result = np.argmax(prediction)          #get the max column
                result = []
                for i in range(len(prediction)):
                        result.append(categories[np.argmax(prediction[i])])
                print("\n ⟹  category: ", result[0])
                
except KeyboardInterrupt:
    pass

