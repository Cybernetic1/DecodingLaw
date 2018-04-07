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
times_steps = 32                        # time steps for RNN.  This number should be same as fixed_seq_len below
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

filenames = os.listdir("laws-TXT/family-laws")
in_file = open("laws-TXT/family-laws/" + filenames[0], encoding="utf-8")      # try the 1st file, for now

remainders = []

def get_next_word_vec():
    global remainders
    while True:
        if len(remainders) > 0:
            head = remainders[0]
            remainders = remainders[1:]
            if head not in stopwords.words('english'):
                return word2vec_map[head]
        else:
            try:
                line = next(in_file)
                line = re.sub(r"[^\w-]", " ", line)             # strip punctuations except hyphen
                line = re.sub(u"[\u4e00-\u9fff]", " ", line)    # strip Chinese
                line = re.sub(r"\d", " ", line)                 # strip numbers
                line = re.sub(r"-+", "-", line)                 # reduce multiple --- to -
                remainders = line.lower().split()
            except:
                print("***** reached EOF")
                exit(0)

# =================== Prepare batch data ============================

def get_sentence_batch(batch_size):
    data = [[get_next_word_vec()
                for _ in range(fixed_seq_len)]
                    for _ in range(batch_size)]
    return data

# =========== define objective function for unsupervised competitive learning ==========

    # y_true can be ignored
    """
    threshold = min(np.partition(y_pred, -3)[-3 :])    # last 3 elements would be biggest
    loss = np.zeros(10)
    for i in range(0,10):
        y = y_pred[i]
        if y > threshold:                              
            loss[i] = 1.0 - y                          # if it is the winner, ideal value = 1.0
        else:
            loss[i] = y                                # if it is loser, ideal value = 0.0
    """

def bingo_loss(y_true, y_pred):
    alpha = 0.1
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
nb_epochs = 40
x_train = np.array(get_sentence_batch(batch_size))
x_test = np.array(get_sentence_batch(batch_size))
print (x_train)
print (len(x_train))
print (len(x_train[0]))
print (len(x_train[0][0]))

print('Build NN model ...')
model = Sequential() # this is the first layer of keras
model.add(LSTM(units=128, dropout=0.05, recurrent_dropout=0.35, return_sequences=True, input_shape=[times_steps, GLOVE_SIZE]))
model.add(LSTM(units=64, dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
model.add(Dense(units=num_classes, activation='softmax'))

# compile the model -- define loss function and optimizer
print("Compiling ...")
# loss type of cateforical crossentropy is good to classification model
model.compile(loss=bingo_loss, optimizer=opt, metrics=['accuracy'])
model.summary()
model.fit(x_train, x_train, batch_size=batch_size, epochs=nb_epochs)

# testing
print("\nTesting ...")

score, accuracy = model.evaluate(x_test, x_test, batch_size=batch_size, verbose=1)
print("Test loss:  ", score)
print("Test accuracy:  ", accuracy)

# Save the model #
model.save('Keras-NN-training2.h5')
