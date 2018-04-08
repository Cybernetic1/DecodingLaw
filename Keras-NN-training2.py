# -*- coding: utf-8 -*-
import os                       # for os.listdir and os.environ
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # suppress Tensorflow warnings
import numpy as np
from nltk.corpus import stopwords
import re                       # for removing punctuations
import sys                      # for sys.stdin.readline()
from collections import defaultdict # for default value of word-vector dictionary
import pickle
import h5py
from random import *

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
num_classes = 8
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
    # skip a random number of words
    skip = randint(1, 1000)
    print("skip words = ", skip, end='\t')
    for _ in range(0, skip):
        next(g)
    data = []
    for _ in range(batch_size):
        vecs = []
        for _ in range(fixed_seq_len):
            vecs.append(next(g))
        data.append(vecs)
    #print('shape = [', len(data), len(data[0]), len(data[0][0]), ']')
    return data

# =========== define objective function for unsupervised competitive learning ==========

    """
    y_true can be ignored
    choose k winners out of n outputs
    
    loss[i] = 1 - y[i]     if i is winner
            = y[i]         if i is loser
    """

def loss1(y_true, y_pred):
    alpha = 0.9
    y2 = alpha * (1 - y_pred)
    values, indices = tf.nn.top_k(y_pred, k = 3)
    min_val = tf.reduce_min(values, axis = 1)
    min_vals = tf.reshape(tf.tile(min_val, [num_classes]), [-1, num_classes])
    loss = tf.where(tf.greater(y_pred, min_vals), y2, alpha * y_pred)
    return loss

def loss2(y_true, y_pred):
    alpha = 0.1
    y2 = alpha * (1 - y_pred)
    values, indices = tf.nn.top_k(y_pred, k = 8)
    min_val = tf.reduce_min(values, axis = 1)
    min_vals = tf.reshape(tf.tile(min_val, [num_classes]), [-1, num_classes])
    loss = tf.where(tf.greater(y_pred, min_vals), y2, alpha * (1 - y_pred))
    return loss

def loss3(y_true, y_pred):
    alpha = 0.1
    random_indices = tf.random_uniform([num_classes], 0, num_classes, dtype = tf.int32, seed = 0)
    opponents = tf.gather(y_pred, random_indices)
    opponents2 = tf.reshape(opponents, [-1, num_classes])
    y2 = alpha * (1 - y_pred)
    loss = tf.where(tf.greater(y_pred, opponents2), y2, alpha * y_pred)
    return loss

def loss4(y_true, y_pred):
    alpha = 0.2
    opponents = tf.random_shuffle(y_pred)
    y2 = alpha * (1 - y_pred)
    loss = tf.where(tf.greater(y_pred, opponents), y2, alpha * y_pred)
    return loss

def loss5(y_true, y_pred):
    alpha = 0.1
    opponents = tf.random_shuffle(y_pred)
    diff = y_pred - opponents
    y2 = alpha * (1 - y_pred)
    loss = tf.where(tf.greater(y_pred, opponents), alpha * diff, -alpha * diff)
    return loss

def loss6(y_true, y_pred):
    alpha = 0.1
    k = 32
    abs_y = tf.abs(y_pred)
    energy = tf.reduce_sum(abs_y, axis=1)
    top_vals, indices = tf.nn.top_k(abs_y, k)
    winner_energy = tf.reduce_sum(top_vals, axis=-1)
    loser_energy = energy - winner_energy
    threshold = winner_energy / k - 0.001
    thresholds = tf.reshape(tf.tile(threshold, [num_classes]), [-1, num_classes])
    updates = tf.reshape(tf.tile(loser_energy, [num_classes]), [-1, num_classes])
    loss = tf.where(tf.greater(y_pred, thresholds), updates, y_pred - y_pred)
    return loss

def loss7(y_true, y_pred):
    alpha = 0
    k = 4
    shape = y_pred.shape.as_list()
    ab = tf.abs(y_pred)
    energy = tf.reduce_sum(ab, axis=-1)
    top, ind = tf.nn.top_k(ab, k)
    energy -= tf.reduce_sum(top, axis=-1)
    mask = tf.reduce_sum(tf.one_hot(ind, depth=shape[-1], on_value=1.0, off_value=0.0, dtype=tf.float32), -2)
    masked = y_pred * mask
    signs = masked / (tf.abs(masked) + 0.0001)
    energy_term = tf.expand_dims(energy, -1) * signs
    return masked + alpha*energy_term

def loss(y_true, y_pred):
    α = 0.1
    k = 3
    shape = y_pred.shape.as_list()
    ab = tf.abs(y_pred)
    energy = tf.reduce_sum(ab, axis=-1)
    top, indices = tf.nn.top_k(ab, k)
    loser_energy = energy - tf.reduce_sum(top, axis=-1)
    mask = tf.reduce_sum(tf.one_hot(indices, depth=shape[-1], on_value=1.0, off_value=0.0, dtype=tf.float32), -2)
    masked_y = y_pred * mask
    signs = masked_y / (tf.abs(masked_y) + 0.0001)
    energy_term = tf.expand_dims(loser_energy, -1) * signs
    return α * energy_term

# ========= define input, output, and NN structure - need to modify =========
#define the optimizer here
opt = Adam(lr=0.0067, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

print('Build NN model ...')
model = Sequential() # this is the first layer of keras
model.add(LSTM(units=128, dropout=0.05, recurrent_dropout=0.35, return_sequences=True, input_shape=[times_steps, GLOVE_SIZE]))
model.add(LSTM(units=64, dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
model.add(Dense(units=num_classes, activation='tanh'))

# compile the model -- define loss function and optimizer
print("Compiling ...")
# loss type of cateforical crossentropy is good to classification model
model.compile(loss=loss, optimizer=opt, metrics=[])
model.summary()

#how many epoch to train
nb_epochs = 30
for i in range(0, nb_epochs):
    print("Iteration: ", i, end=' ')
    x_train = np.array(get_sentence_batch(batch_size))
    x_test = np.array(get_sentence_batch(batch_size))
    # generate a 2D array
    y_train = np.random.rand(batch_size, num_classes)
    ##print (type(x_train))
    ##print (len(x_train))
    ##print (len(x_train[0]))
    ##print (len(x_train[0][0]))
    model.fit(x_train, y_train , batch_size=batch_size, epochs=10)

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

print("\n***** Finding classifications of categories...")

# for each case-law line, print output
for i, category in enumerate(categories):
    print("Category = ", category)
    catWords = cats[i]
    for j in range(0, len(catWords) - fixed_seq_len, 16):
        vecs = []
        for word in catWords[j : j + fixed_seq_len]:
            vecs.append(word2vec_map[word])

        prediction = model.predict(np.expand_dims(vecs, axis=0))
        for k in prediction[0]:
            if k > 0.0:
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
