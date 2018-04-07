import os                       # for os.listdir and os.environ
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # suppress Tensorflow warnings
import numpy as np
from nltk.corpus import stopwords
import re                       # for removing punctuations
import sys                      # for sys.stdin.readline()
from collections import defaultdict # for default value of word-vector dictionary
import pickle
import h5py

import tensorflow as tf

# Keras deep learning library
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Activation
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import Callback
from keras import backend as K

# Load the trained model
#model = keras.models.load_weights('Keras-NN_weigts.h5')
path_to_glove = "wiki-news-300d-1M.vec"
GLOVE_SIZE = 300
batch_size = 512
num_classes = 10
times_steps = 8                        # time steps for RNN.  This number should be same as fixed_seq_len below
fixed_seq_len = times_steps

# 10 categories:
categories = ["matrimonial-rights", "separation", "divorce", "after-divorce", "divorce-maintenance",
    "property-on-divorce", "types-of-marriages", "battered-wife-and-children", "Harmony-House", "divorce-mediation"]

# =================== Read unique words list & word2vec map ======================

pickle_off = open("prepared-data/unique-words.pickle", "rb")
word_list = pickle.load(pickle_off)

pickle_off = open("prepared-data/word2vec-map.pickle", "rb")
word2vec_map = pickle.load(pickle_off)

# set default value = zero vector, if word not found in dictionary
zero_vector = np.asarray([0.0] * GLOVE_SIZE, dtype='float32')
word2vec_map = defaultdict(lambda: zero_vector, word2vec_map)

# =================== Try to use continuous-training mode ======================

def get_next_word_vec():
    for filenames in os.listdir("laws-TXT/family-laws"):
        with open("laws-TXT/family-laws/" + filenames, encoding="utf-8") as in_file:
            for line in in_file:
                line = re.sub(r"[^\w-]", " ", line)             # strip punctuations except hyphen
                line = re.sub(u"[\u4e00-\u9fff]", " ", line)    # strip Chinese
                line = re.sub(r"\d", " ", line)                 # strip numbers
                line = re.sub(r"-+", "-", line)                 # reduce multiple --- to -
                for word in line.lower().split():
                    yield word2vec_map[word]

# =================== Prepare batch data ============================

g = get_next_word_vec()

def get_sentence_batch(batch_size):
    data = []
    for _ in range(batch_size):
        vecs = []
        for _ in range(fixed_seq_len):
            vecs.append(next(g))
        data.append(vecs)
    print(len(data))
    print(len(data[0]))
    print(len(data[0][0]))
    return data

# =========== define objective function for unsupervised competitive learning ==========

    """
    y_true can be ignored
    choose k winners out of n outputs
    
    loss[i] = 1 - y[i]     if i is winner
            = y[i]         if i is loser
    """

def bingo_loss(y_true, y_pred):
    alpha = 0.9
    y2 = alpha * (1 - y_pred)
    values, indices = tf.nn.top_k(y_pred, k = 3)
    min_val = tf.reduce_min(values, axis = 1)
    min_vals = tf.reshape(tf.tile(min_val, [10]), [-1, 10])
    loss = tf.where(tf.greater(y_pred, min_vals), y2, alpha * y_pred)
    return loss

# ========= define input, output, and NN structure - need to modify =========
#define the optimizer here
opt = Adam(lr=0.0067, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

#how many epoch to train
nb_epochs = 100
x_train = np.array(get_sentence_batch(batch_size))
x_test = np.array(get_sentence_batch(batch_size))
# generate a 2D array
y_train = np.random.rand(batch_size, num_classes)
##print (type(x_train))
##print (len(x_train))
##print (len(x_train[0]))
##print (len(x_train[0][0]))
print('Build NN model ...')
model = Sequential() # this is the first layer of keras
model.add(LSTM(units=128, dropout=0.05, recurrent_dropout=0.35, return_sequences=True, input_shape=[times_steps, GLOVE_SIZE]))
model.add(LSTM(units=64, dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
model.add(Dense(units=num_classes, activation='sigmoid'))

# compile the model -- define loss function and optimizer
print("Compiling ...")
# loss type of cateforical crossentropy is good to classification model
model.compile(loss=bingo_loss, optimizer=opt, metrics=[])
model.summary()
model.fit(x_train, y_train , batch_size=batch_size, epochs=nb_epochs)

print("\n***** Saving model....")
model.save('Keras-NN-training2.h5')

# ==================== extract all sentences from categories =======================

cats = []              # list of list of words (categories[words])
suffix = ""            # to be added to sub-directory, not needed currently

for i, category in enumerate(categories):
    print("\nCategory: ", category)
    for j, filename in enumerate(os.listdir("categories/" + category + suffix)):
        with open("categories/" + category + suffix + "/" + filename) as f:
            catWords = []
            for line in f:
                line = re.sub(r"[^\w-]", " ", line)             # strip punctuations except hyphen
                line = re.sub(u"[\u4e00-\u9fff]", " ", line)    # strip Chinese
                line = re.sub(r"\d", " ", line)                 # strip numbers
                line = re.sub(r"-+", "-", line)                 # reduce multiple --- to -
                for word in line.lower().split():
                    if word not in stopwords.words('english'):
                        catWords.append(word)
            cats.append(catWords)

# ====================== find classifications of categories ===========================

print("\n***** Finding classification of categories...")

# for each case-law line, print output
for i, category in enumerate(categories):
    print("\nCategory = ", category)
    catWords = cats[i]
    for j in range(0, len(catWords) - fixed_seq_len, 4):
        vecs = []
        for word in catWords[j : j + fixed_seq_len]:
            vecs.append(word2vec_map[word])

        prediction = model.predict(np.expand_dims(vecs, axis=0))
        for k in prediction[0]:
            if k > 0.4:
                print("█", end='')
            else:
                print("·", end='')
        print()

exit(0)

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
                glove_file = open(path_to_glove, "r", encoding = "utf-8")
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
                print(prediction)
                print(len(prediction))
                #result = np.argmax(prediction)          #get the max column
                result = []
                for i in range(len(prediction)):
                        result.append(categories[np.argmax(prediction[i])])
                print("\n ⟹  category: ", result[0])

except KeyboardInterrupt:
    pass
